"""E0: our Qwen3 forward against transformers, on real text.

1. Packed prefill of several prompts: logits at every position, compared with an fp32 HF reference.
   HF's own bf16 model gives the error level to expect from bf16.
2. Greedy decoding: our batched decode vs HF's bf16 decode, token by token.
3. Teacher forcing: our per-step decode logits vs the fp32 reference on the same tokens.

Run from the repo root: python -m tests.test_qwen3_hf
"""

import json
import os
import random

import numpy as np
import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minivllm.attention import AttnMetadata
from minivllm.attention.gather import GatherAttention
from minivllm.models.qwen3 import load_qwen3
from minivllm.utils import cap_gpu_memory

CKPT = os.path.join("checkpoints", "Qwen3-0.6B")
VAL_BIN = r"A:\GitHub project\nanoGPT\data\fineweb_edu\val.bin"
PROMPT_LENS = [17, 120, 517, 1000]
NEW_TOKENS = 64
BLOCK_SIZE = 16
OUT = os.path.join("results", "e0_qwen3.json")


def real_text_prompts(tok, lengths):
    """Leading tokens of FineWeb-Edu val documents (GPT-2 ids decoded to text, re-tokenized for Qwen3)."""
    data = np.array(np.memmap(VAL_BIN, dtype=np.uint16, mode="r")[:2_000_000])
    gpt2 = tiktoken.get_encoding("gpt2")
    bounds = np.flatnonzero(data == gpt2.eot_token)
    prompts = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        ids = tok(gpt2.decode(data[a + 1 : b].tolist()))["input_ids"]
        if len(ids) >= lengths[len(prompts)]:
            prompts.append(ids[: lengths[len(prompts)]])
            if len(prompts) == len(lengths):
                return prompts
    raise RuntimeError("not enough long documents")


def logit_stats(x, ref):
    x, ref = x.float(), ref.float()
    err = (x - ref).abs()
    choice = x.argmax(-1, keepdim=True)
    # how far below the reference's best logit the token picked from x is (0 when argmaxes agree)
    regret = ref.max(-1).values - ref.gather(-1, choice).squeeze(-1)
    return {
        "max_abs_err": err.max().item(),
        "mean_abs_err": err.mean().item(),
        "argmax_agree": (choice.squeeze(-1) == ref.argmax(-1)).float().mean().item(),
        "max_fp32_regret": regret.max().item(),
    }


class PagedSeqs:
    """Hands out shuffled blocks so that each sequence's KV is scattered through the pool."""

    def __init__(self, lens, max_len, model, device):
        nblk = [(max_len_i + BLOCK_SIZE - 1) // BLOCK_SIZE for max_len_i in max_len]
        ids = list(range(sum(nblk)))
        random.Random(0).shuffle(ids)
        self.block_tables = torch.zeros(len(lens), max(nblk), dtype=torch.int32)
        for i, n in enumerate(nblk):
            self.block_tables[i, :n] = torch.tensor(ids[:n])
            ids = ids[n:]
        self.block_tables = self.block_tables.to(device)
        cfg = model.cfg
        shape = (sum(nblk), BLOCK_SIZE, cfg.num_key_value_heads, cfg.head_dim)
        for layer in model.layers:
            layer.self_attn.k_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)
            layer.self_attn.v_cache = torch.zeros(shape, dtype=torch.bfloat16, device=device)

    def slots(self, seq_idx, positions):
        return self.block_tables[seq_idx, positions // BLOCK_SIZE].long() * BLOCK_SIZE + positions % BLOCK_SIZE


@torch.inference_mode()
def run_ours(model, prompts, device):
    lens = [len(p) for p in prompts]
    seqs = PagedSeqs(lens, [n + NEW_TOKENS for n in lens], model, device)

    # packed prefill of all prompts
    input_ids = torch.tensor(sum(prompts, []), device=device)
    positions = torch.cat([torch.arange(n, device=device) for n in lens])
    seq_idx = torch.repeat_interleave(torch.arange(len(lens), device=device), torch.tensor(lens, device=device))
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    meta = AttnMetadata(is_prefill=True, slot_mapping=seqs.slots(seq_idx, positions), cu_seqlens=cu)
    hidden = model(input_ids, positions, meta)

    # batched greedy decode; step_logits[i] holds the logits that chose generated token i
    last = model.compute_logits(hidden[torch.tensor(cu[1:], device=device) - 1])
    step_logits, tokens = [last.float()], [last.argmax(-1)]
    pos = torch.tensor(lens, device=device)
    rows = torch.arange(len(lens), device=device)
    for _ in range(NEW_TOKENS - 1):
        meta = AttnMetadata(
            is_prefill=False, slot_mapping=seqs.slots(rows, pos), block_tables=seqs.block_tables,
            context_lens=(pos + 1).int(), max_context_len=int(pos.max()) + 1,
        )
        logits = model.compute_logits(model(tokens[-1], pos, meta)).float()
        step_logits.append(logits)
        tokens.append(logits.argmax(-1))
        pos = pos + 1
    return hidden, cu, torch.stack(tokens, 1).tolist(), torch.stack(step_logits, 1)


@torch.inference_mode()
def hf_greedy(hf, ids, n, device):
    out = hf(torch.tensor([ids], device=device), use_cache=True)
    toks, margins = [], []
    for _ in range(n):
        top2 = out.logits[0, -1].float().topk(2)
        toks.append(top2.indices[0].item())
        margins.append((top2.values[0] - top2.values[1]).item())
        out = hf(torch.tensor([[toks[-1]]], device=device), past_key_values=out.past_key_values, use_cache=True)
    return toks, margins


@torch.inference_mode()
def main():
    device = "cuda"
    torch.cuda.init()
    cap_gpu_memory()
    tok = AutoTokenizer.from_pretrained(CKPT)
    prompts = real_text_prompts(tok, PROMPT_LENS)
    model = load_qwen3(CKPT, GatherAttention(), device)
    hf16 = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).to(device).eval()
    hf32 = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32).to(device).eval()

    hidden, cu, our_tokens, our_step_logits = run_ours(model, prompts, device)
    results = []
    for i, p in enumerate(prompts):
        ids = torch.tensor([p], device=device)
        ref = hf32(ids).logits[0]
        r = {
            "prompt_len": len(p),
            "prefill_ours_vs_fp32": logit_stats(model.compute_logits(hidden[cu[i] : cu[i + 1]]), ref),
            "prefill_hf_bf16_vs_fp32": logit_stats(hf16(ids).logits[0], ref),
        }
        del ref

        # teacher forcing on our generated tokens: rows len(p)-1 .. end-1 predict the generated tokens
        full = torch.tensor([p + our_tokens[i][:-1]], device=device)
        ref = hf32(full).logits[0, len(p) - 1 :]
        r["decode_ours_vs_fp32"] = logit_stats(our_step_logits[i], ref)
        r["decode_hf_bf16_vs_fp32"] = logit_stats(hf16(full).logits[0, len(p) - 1 :], ref)
        del ref

        hf_tokens, margins = hf_greedy(hf16, p, NEW_TOKENS, device)
        div = next((j for j in range(NEW_TOKENS) if hf_tokens[j] != our_tokens[i][j]), None)
        r["greedy_first_divergence"] = div
        r["hf_top2_margin_at_divergence"] = None if div is None else margins[div]
        results.append(r)

        print(f"prompt {len(p):4d} tokens")
        for k in ("prefill_ours_vs_fp32", "prefill_hf_bf16_vs_fp32", "decode_ours_vs_fp32", "decode_hf_bf16_vs_fp32"):
            s = r[k]
            print(
                f"  {k:24s} max {s['max_abs_err']:.4f}  mean {s['mean_abs_err']:.5f}  "
                f"argmax agree {s['argmax_agree']:.4f}  max fp32 regret {s['max_fp32_regret']:.4f}"
            )
        if div is None:
            print(f"  greedy: all {NEW_TOKENS} tokens identical to HF bf16")
        else:
            print(f"  greedy: first divergence at token {div}, HF top-2 margin there {margins[div]:.4f}")

    print("sample:", repr(tok.decode(prompts[1][-20:] + our_tokens[1][:30])))
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"model": "Qwen3-0.6B", "block_size": BLOCK_SIZE, "new_tokens": NEW_TOKENS, "prompts": results}, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
