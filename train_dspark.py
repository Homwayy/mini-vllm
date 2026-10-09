"""Trains the DSpark-style draft model (minivllm/spec/dspark.py) on the frozen GPT-2 target (E8c).

Data: random 1024-token windows of nanoGPT's FineWeb-Edu training split, as for EAGLE (train_draft.py). The
target gives its features at every position and its distribution of every next token. Every position a is an
anchor: row k of its block (k = 0 .. block_len - 1) drafts the token at a + k + 1 from the context before a,
the anchor and the earlier rows, with the Markov head's term for the token at a + k (the data's, as in teacher
forcing). Each row's loss, weighted by exp(-k / block_len) as in DSpark:
  0.1 x cross-entropy of the data token, if it is in the draft vocabulary;
  0.9 x the L1 distance between the draft's and the target's distributions over the full vocabulary (the
        target's mass outside the draft vocabulary counts fully);
  1.0 x binary cross-entropy of the confidence against 1 - L1 / 2, the acceptance probability at temperature 1.
The draft vocabulary is EAGLE's: the most frequent tokens of the training split.

Every --eval-every steps, fixed validation windows give per row k: the T=1 acceptance 1 - L1 / 2 and the
greedy acceptance (the draft's argmax is the target's), both given that the earlier tokens are the data's; the
mean confidence; and the expected tokens per step for each block length, assuming each row is accepted with its
acceptance given the earlier ones were.

Run from the repo root, e.g.:
    python train_dspark.py --out checkpoints/gpt2_dspark.pt --log results/e8c_train.json
or, with two draft blocks (E8c.2):
    python train_dspark.py --n-layer 2 --out checkpoints/gpt2_dspark2.pt --log results/e8c2_train.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.utils.checkpoint import checkpoint

from minivllm.attention import AttnMetadata
from minivllm.model_runner import MODELS
from minivllm.models.gpt2 import load_gpt2
from minivllm.spec.dspark import FEATURE_LAYERS, DSparkDraft, train_mask_mod
from train_draft import DATA, REAL_VOCAB, DenseCausal, windows

ALPHA_CE, ALPHA_TV, ALPHA_CONF = 0.1, 0.9, 1.0


@torch.no_grad()
def target_pass(target, backend, ids, vocab_ids):
    """The target over windows ids [B, T]: its features (a list of [B, T, d]), its probabilities of the draft
    vocabulary [B, T, V] fp32 and its mass outside it [B, T], and its argmax over the full vocabulary [B, T]."""
    B, T = ids.shape
    backend.batch = B
    pos = torch.arange(T, device=ids.device).repeat(B)
    f = target(ids.view(-1), pos, AttnMetadata(is_prefill=True, slot_mapping=None))
    feats = [c.view(B, T, -1) for c in target.captured] + [f.view(B, T, -1)]
    p_in, p_out, top = [], [], []
    for chunk in f.split(T):  # full-vocabulary logits, one window at a time
        logits = target.compute_logits(chunk).float()
        logits[:, REAL_VOCAB:] = float("-inf")
        p = (logits[:, vocab_ids] - torch.logsumexp(logits, -1, keepdim=True)).exp()
        p_in.append(p)
        p_out.append((1 - p.sum(-1)).clamp_min(0))
        top.append(logits.argmax(-1))
    return feats, torch.cat(p_in).view(B, T, -1), torch.cat(p_out).view(B, T), torch.cat(top).view(B, T)


def row_terms(draft, h, prev, p_in, p_out):
    """Rows h [..., d] with the token before each, prev: the draft's log-probabilities [..., V] and the L1
    distance to the target's distribution [...]."""
    logq = F.log_softmax(draft.logits(h).float() + draft.transition(prev).float(), -1)
    return logq, (logq.exp() - p_in).abs().sum(-1) + p_out


def rows_of(k, ids, feats_t):
    """Row k of every block whose drafted token a + k + 1 is in the window: anchors a = 0 .. T - k - 2."""
    T = ids.shape[1]
    p_in, p_out, top = feats_t
    return ids[:, k : T - 1], ids[:, k + 1 :], p_in[:, k : T - 1], p_out[:, k : T - 1], top[:, k : T - 1]


def losses(draft, target, backend, ids, block_mask, flex, draft_index):
    feats, *feats_t = target_pass(target, backend, ids, draft.vocab_ids)
    T, n = ids.shape[1], draft.block_len
    with torch.autocast("cuda", torch.bfloat16):
        h = draft.train_forward(feats, ids, block_mask, flex)

    def head(hk, prev, tgt, p_in, p_out):
        logq, l1 = row_terms(draft, hk, prev, p_in, p_out)
        ti = draft_index[tgt]
        inside = ti >= 0
        ce = -(logq.gather(-1, ti.clamp_min(0)[..., None]).squeeze(-1) * inside).sum() / inside.sum().clamp_min(1)
        bce = F.binary_cross_entropy(draft.confidence(hk, prev), (1 - 0.5 * l1).detach().clamp(0, 1))
        return torch.stack([ce, l1.mean(), bce])

    total, parts, norm = 0.0, [], 0.0
    for k in range(n):
        prev, tgt, p_in, p_out, _ = rows_of(k, ids, feats_t)
        terms = checkpoint(head, h[:, k, : T - k - 1], prev, tgt, p_in, p_out, use_reentrant=False)
        w = math.exp(-k / n)
        total = total + w * (ALPHA_CE * terms[0] + ALPHA_TV * terms[1] + ALPHA_CONF * terms[2])
        norm += w
        parts.append(terms.detach())
    return total / norm, torch.stack(parts)


@torch.no_grad()
def evaluate(draft, target, backend, windows, block_mask, flex):
    n = draft.block_len
    sums = np.zeros((n, 4))  # T=1 acceptance, greedy acceptance, confidence, its label
    for ids in windows:
        feats, *feats_t = target_pass(target, backend, ids, draft.vocab_ids)
        with torch.autocast("cuda", torch.bfloat16):
            h = draft.train_forward(feats, ids, block_mask, flex)
        T = ids.shape[1]
        for k in range(n):
            prev, _, p_in, p_out, top = rows_of(k, ids, feats_t)
            hk = h[:, k, : T - k - 1]
            logq, l1 = row_terms(draft, hk, prev, p_in, p_out)
            sums[k] += [
                (1 - 0.5 * l1).mean().item(),
                (draft.vocab_ids[logq.argmax(-1)] == top).float().mean().item(),
                draft.confidence(hk, prev).mean().item(),
                (1 - 0.5 * l1).clamp(0, 1).mean().item(),
            ]
    sums /= len(windows)
    out = {name: sums[:, col].tolist() for name, col in (("acc_t1", 0), ("acc_greedy", 1), ("confidence", 2), ("confidence_label", 3))}
    for name in ("acc_greedy", "acc_t1"):
        expected, run, per_len = 1.0, 1.0, []
        for a in out[name]:
            run *= a
            expected += run
            per_len.append(expected)
        out[f"tokens_per_step_{name.removeprefix('acc_')}"] = per_len  # for block lengths 1 .. block_len
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=MODELS["gpt2"]["path"])
    ap.add_argument("--draft-vocab", type=int, default=16384, help="the most frequent tokens of the training split")
    ap.add_argument("--block-len", type=int, default=8, help="rows per block; drafting can use fewer")
    ap.add_argument("--rank", type=int, default=64, help="rank of the Markov head")
    ap.add_argument("--n-layer", type=int, default=1, help="GPT-2 blocks in the draft model")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-windows", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--log", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"
    backend = DenseCausal()
    target = load_gpt2(args.target, backend, device)
    target.capture = FEATURE_LAYERS
    train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
    counts = np.bincount(train, minlength=target.cfg.vocab_size)[:REAL_VOCAB]
    vocab_ids = np.sort(np.argsort(-counts, kind="stable")[: args.draft_vocab])
    print(f"draft vocabulary: {args.draft_vocab} tokens, {counts[vocab_ids].sum() / counts.sum():.4f} of training tokens")
    draft_index = torch.full((target.cfg.vocab_size,), -1, dtype=torch.long, device=device)
    draft_index[torch.as_tensor(vocab_ids, device=device)] = torch.arange(len(vocab_ids), device=device)

    draft = DSparkDraft(target.cfg, None, vocab_ids, args.block_len, args.rank, n_layer=args.n_layer)
    draft.to(device)
    draft.attach(target)
    params = [p for p in draft.parameters() if p.requires_grad]
    print(f"draft parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    block_mask = create_block_mask(train_mask_mod(args.seq_len), None, None, args.block_len * args.seq_len,
                                   (args.block_len + 1) * args.seq_len, device=device)
    flex = torch.compile(flex_attention, dynamic=False)

    rng = np.random.default_rng(args.seed)
    val_windows = [windows(val, np.random.default_rng(1000 + i), 2, args.seq_len, device) for i in range(args.eval_windows // 2)]
    history, t0 = [], time.perf_counter()

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        r = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * r))

    def save(step):
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        torch.save({"draft": draft.state_dict(), "vocab_ids": vocab_ids.tolist(), "feature_layers": list(FEATURE_LAYERS),
                    "target": args.target, "args": vars(args), "step": step}, args.out)
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        with open(args.log, "w") as f:
            json.dump({"args": vars(args), "draft_vocab_train_share": float(counts[vocab_ids].sum() / counts.sum()),
                       "train_s": time.perf_counter() - t0, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                       "history": history}, f, indent=1)

    def run_eval(step):
        draft.eval()
        ev = evaluate(draft, target, backend, val_windows, block_mask, flex)
        draft.train()
        ev.update({"step": step, "tokens_seen": step * args.batch_size * args.seq_len, "elapsed_s": time.perf_counter() - t0})
        history.append(ev)
        print(f"eval step {step}: T=1 acceptance {', '.join(f'{a:.3f}' for a in ev['acc_t1'])}; greedy "
              f"{', '.join(f'{a:.3f}' for a in ev['acc_greedy'])}; tokens/step at 4 rows: T=1 {ev['tokens_per_step_t1'][3]:.2f}, "
              f"greedy {ev['tokens_per_step_greedy'][3]:.2f}; confidence {ev['confidence'][0]:.3f}..{ev['confidence'][-1]:.3f}", flush=True)

    draft.train()
    run_eval(0)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, parts = losses(draft, target, backend, windows(train, rng, args.batch_size, args.seq_len, device),
                             block_mask, flex, draft_index)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0:
            p = parts.cpu().numpy()
            print(f"step {step}: loss {loss.item():.4f}, ce/L1/bce rows 0 and {len(p) - 1}: "
                  f"{p[0, 0]:.3f}/{p[0, 1]:.3f}/{p[0, 2]:.3f} {p[-1, 0]:.3f}/{p[-1, 1]:.3f}/{p[-1, 2]:.3f}, "
                  f"lr {lr_at(step):.2e}, {time.perf_counter() - t0:.0f} s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            run_eval(step + 1)
            save(step + 1)
    print("wrote", args.out, "and", args.log)


if __name__ == "__main__":
    main()
