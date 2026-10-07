"""Trains the EAGLE-style draft model (minivllm/spec/eagle.py) on the frozen GPT-2 target (E8b).

Data: random 1024-token windows of nanoGPT's FineWeb-Edu training split. For each window the target gives
its features f_t and its next-token distributions; the draft sees (f_t, x_{t+1}) and is trained to
  - regress its output g_t onto f_{t+1} (smooth L1), and
  - match the target's distribution of x_{t+2}, restricted to the draft vocabulary (cross-entropy with
    the target's probabilities, weighted by --cls-weight).
Pass 0 takes the target's features (with uniform noise of +-noise, as in EAGLE); passes 1..passes-1 take the
draft's own predictions, as the later steps of a draft chain do (see EagleDraft.train_forward).

Every --eval-every steps, a fixed set of validation windows gives per pass (= draft position):
  greedy acceptance: the draft's argmax equals the target's;
  T=1 acceptance: sum over the draft vocabulary of min(p, q), the acceptance probability of a draft
    sampled from q at temperature 1;
  expected tokens per step for a chain of `passes` drafts, assuming each position is accepted with its
    acceptance given the earlier ones were.

Run from the repo root, e.g.:
    python train_draft.py --out checkpoints/gpt2_eagle.pt --log results/e8b_train.json
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from minivllm.attention import AttnMetadata
from minivllm.model_runner import MODELS
from minivllm.models.gpt2 import load_gpt2
from minivllm.spec.eagle import EagleDraft

DATA = r"A:\GitHub project\nanoGPT\data\fineweb_edu"
REAL_VOCAB = 50257  # ids from here up pad GPT-2's embedding to 50304


class DenseCausal:
    """Causal attention over `batch` equal-length sequences packed as [B * T, H, D]; for training only."""

    batch = 1

    def __call__(self, q, k, v, k_cache, v_cache, meta):
        BT, H, D = q.shape
        q, k, v = (t.view(self.batch, BT // self.batch, H, D).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return o.transpose(1, 2).reshape(BT, H, D)


@torch.no_grad()
def target_features(target, backend, ids):
    B, T = ids.shape
    backend.batch = B
    pos = torch.arange(T, device=ids.device).repeat(B)
    return target(ids.view(-1), pos, AttnMetadata(is_prefill=True, slot_mapping=None)).view(B, T, -1)


def losses(draft, target, backend, ids, passes, noise, cls_weight):
    f = target_features(target, backend, ids)
    with torch.no_grad():
        logp = F.log_softmax(draft.logits(f[:, 1:]).float(), -1)  # target's distribution of x_{s+2}, on the draft vocab
    with torch.autocast("cuda", torch.bfloat16):
        outs = draft.train_forward(f[:, :-1], ids[:, 1:], passes, noise)
    total, parts = 0.0, []
    for j, g in enumerate(outs):
        reg = F.smooth_l1_loss(g[:, j:], f[:, 1 + j :])
        cls = -(logp[:, j:].exp() * F.log_softmax(draft.logits(g[:, j:]).float(), -1)).sum(-1).mean()
        total = total + reg + cls_weight * cls
        parts.append((reg.item(), cls.item()))
    return total / passes, parts


@torch.no_grad()
def evaluate(draft, target, backend, windows, passes):
    sums = np.zeros((passes, 4))  # reg, cls, greedy acceptance, T=1 acceptance
    p_vocab, n = 0.0, 0
    for ids in windows:
        f = target_features(target, backend, ids)
        full = F.linear(f[:, 1:].to(target.lm_head_weight.dtype), target.lm_head_weight).float()
        full[..., REAL_VOCAB:] = float("-inf")
        p = full.softmax(-1)  # target's full distribution of x_{s+2}
        p_draft_vocab = p[..., draft.vocab_ids]
        logp = F.log_softmax(full[..., draft.vocab_ids], -1)
        with torch.autocast("cuda", torch.bfloat16):
            outs = draft.train_forward(f[:, :-1], ids[:, 1:], passes)
        for j, g in enumerate(outs):
            q = F.softmax(draft.logits(g[:, j:]).float(), -1)
            sums[j] += [
                F.smooth_l1_loss(g[:, j:].float(), f[:, 1 + j :]).item(),
                -(logp[:, j:].exp() * q.clamp_min(1e-30).log()).sum(-1).mean().item(),
                (draft.vocab_ids[q.argmax(-1)] == p[:, j:].argmax(-1)).float().mean().item(),
                torch.minimum(p_draft_vocab[:, j:], q).sum(-1).mean().item(),
            ]
        p_vocab += p_draft_vocab.sum(-1).mean().item()
        n += 1
    sums /= n
    out = {"draft_vocab_mass": p_vocab / n}
    for name, col in (("reg", 0), ("cls", 1), ("acc_greedy", 2), ("acc_t1", 3)):
        out[name] = sums[:, col].tolist()
    for name in ("acc_greedy", "acc_t1"):
        expected, run = 1.0, 1.0
        for a in out[name]:
            run *= a
            expected += run
        out[f"tokens_per_step_{name.removeprefix('acc_')}"] = expected
    return out


def windows(data, rng, n, T, device):
    starts = rng.integers(0, len(data) - T, n)
    return torch.from_numpy(np.stack([data[s : s + T] for s in starts]).astype(np.int64)).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=MODELS["gpt2"]["path"])
    ap.add_argument("--draft-vocab", type=int, default=16384, help="the most frequent tokens of the training split")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--cls-weight", type=float, default=1.0)
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
    train = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(DATA, "val.bin"), dtype=np.uint16, mode="r")
    counts = np.bincount(train, minlength=target.cfg.vocab_size)[:REAL_VOCAB]
    vocab_ids = np.sort(np.argsort(-counts, kind="stable")[: args.draft_vocab])
    print(f"draft vocabulary: {args.draft_vocab} tokens, {counts[vocab_ids].sum() / counts.sum():.4f} of training tokens")

    draft = EagleDraft(target.cfg, None, vocab_ids)
    draft.init_as_copy()
    draft.to(device)
    draft.attach(target)
    params = [p for p in draft.parameters() if p.requires_grad]
    print(f"draft parameters: {sum(p.numel() for p in params) / 1e6:.2f} M")
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    rng = np.random.default_rng(args.seed)
    val_windows = [windows(val, np.random.default_rng(1000 + i), 2, args.seq_len, device) for i in range(args.eval_windows // 2)]
    history, t0 = [], time.perf_counter()

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        r = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * r))

    def run_eval(step):
        draft.eval()
        ev = evaluate(draft, target, backend, val_windows, args.passes)
        draft.train()
        ev.update({"step": step, "tokens_seen": step * args.batch_size * args.seq_len, "elapsed_s": time.perf_counter() - t0})
        history.append(ev)
        print(f"eval step {step}: greedy acceptance {', '.join(f'{a:.3f}' for a in ev['acc_greedy'])}, "
              f"T=1 acceptance {', '.join(f'{a:.3f}' for a in ev['acc_t1'])}, tokens/step greedy {ev['tokens_per_step_greedy']:.2f} "
              f"T=1 {ev['tokens_per_step_t1']:.2f}", flush=True)

    draft.train()
    run_eval(0)
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss, parts = losses(draft, target, backend, windows(train, rng, args.batch_size, args.seq_len, device),
                             args.passes, args.noise, args.cls_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 100 == 0:
            print(f"step {step}: loss {loss.item():.4f}, reg/cls by pass "
                  f"{' '.join(f'{r:.3f}/{c:.3f}' for r, c in parts)}, lr {lr_at(step):.2e}, "
                  f"{time.perf_counter() - t0:.0f} s", flush=True)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            run_eval(step + 1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"draft": draft.state_dict(), "vocab_ids": vocab_ids.tolist(), "target": args.target, "args": vars(args)}, args.out)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "w") as f:
        json.dump({"args": vars(args), "draft_vocab_train_share": float(counts[vocab_ids].sum() / counts.sum()),
                   "train_s": time.perf_counter() - t0, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "history": history}, f, indent=1)
    print("wrote", args.out, "and", args.log)


if __name__ == "__main__":
    main()
