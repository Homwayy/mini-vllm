"""E0: our Qwen3 forward against transformers, on real text.

1. Packed prefill of several prompts: logits at every position, compared with an fp32 HF reference.
   HF's own bf16 model gives the error level to expect from bf16.
2. Greedy decoding: our batched decode vs HF's bf16 decode, token by token.
3. Teacher forcing: our per-step decode logits vs the fp32 reference on the same tokens.

Run from the repo root: python -m tests.test_qwen3_hf [--attn gather|flex|triton|triton-decode]
"""

import json
import os

import tiktoken
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from minivllm.models.qwen3 import load_qwen3
from minivllm.utils import cap_gpu_memory
from tests.common import attention_from_args, logit_stats, print_stats, run_ours, val_documents

CKPT = os.path.join("checkpoints", "Qwen3-0.6B")
PROMPT_LENS = [17, 120, 517, 1000]
NEW_TOKENS = 64
BLOCK_SIZE = 16


def real_text_prompts(tok, lengths):
    """Leading tokens of FineWeb-Edu val documents, decoded from GPT-2 ids and re-tokenized for Qwen3."""
    gpt2 = tiktoken.get_encoding("gpt2")
    prompts = []
    for doc in val_documents():
        ids = tok(gpt2.decode(doc))["input_ids"]
        if len(ids) >= lengths[len(prompts)]:
            prompts.append(ids[: lengths[len(prompts)]])
            if len(prompts) == len(lengths):
                return prompts
    raise RuntimeError("not enough long documents")


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
    attn_name, attn = attention_from_args(BLOCK_SIZE, 4096, len(PROMPT_LENS), device)
    tok = AutoTokenizer.from_pretrained(CKPT)
    prompts = real_text_prompts(tok, PROMPT_LENS)
    model = load_qwen3(CKPT, attn, device)
    hf16 = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).to(device).eval()
    hf32 = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.float32).to(device).eval()

    hidden, cu, our_tokens, our_step_logits = run_ours(model, prompts, NEW_TOKENS, BLOCK_SIZE, device)
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
            print_stats(k, r[k])
        if div is None:
            print(f"  greedy: all {NEW_TOKENS} tokens identical to HF bf16")
        else:
            print(f"  greedy: first divergence at token {div}, HF top-2 margin there {margins[div]:.4f}")

    print("sample:", repr(tok.decode(prompts[1][-20:] + our_tokens[1][:30])))
    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"e0_qwen3_{attn_name}.json")
    with open(out, "w") as f:
        json.dump({"model": "Qwen3-0.6B", "attention": attn_name, "block_size": BLOCK_SIZE, "new_tokens": NEW_TOKENS, "prompts": results}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
