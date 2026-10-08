"""E0: our GPT-2 port against nanoGPT, on FineWeb-Edu validation text.

nanoGPT's own model under bf16 autocast (as its generation runs) is the reference for exact agreement;
the same model without autocast gives the fp32 reference.

1. Packed prefill of several prompts: logits at every position.
2. Greedy decoding: our batched decode vs nanoGPT's GPT.generate (temperature 0, KV cache), token by token.
3. Teacher forcing: our per-step decode logits vs both references on the same tokens.
4. Each prompt alone: with the gather backend, which calls the same SDPA kernels as nanoGPT, prefill logits
   must equal nanoGPT's bit for bit and greedy tokens must be identical. Packed with other prompts, the
   matmuls have more rows and cuBLAS may pick different kernels, so small differences there are expected.
   Other backends are judged by their errors against the fp32 reference.

With --fused, our model runs its fused forward (minivllm/fused.py), and the results get the suffix _fused.
Its LayerNorm sums in another order than PyTorch's, so the bitwise checks of 4 are not expected to hold.

Run from the repo root: python -m tests.test_gpt2_nanogpt [--attn gather|flex|triton|triton-decode] [--fused]
"""

import json
import os
import sys

import torch

from minivllm.models.gpt2 import load_gpt2
from minivllm.utils import cap_gpu_memory
from tests.common import EOT, attention_from_args, logit_stats, print_stats, run_ours, val_documents

NANOGPT = r"A:\GitHub project\nanoGPT"
sys.path.insert(0, os.path.join(NANOGPT, "transformer"))
from model import GPT, GPTConfig  # noqa: E402

CKPT = os.path.join(NANOGPT, "out", "final", "ckpt.pt")
PROMPT_LENS = [17, 120, 517, 896]  # the longest prompt plus NEW_TOKENS fills the 1024 context
NEW_TOKENS = 128
BLOCK_SIZE = 16


def prompts_from_val(lengths):
    """<|endoftext|> followed by the first tokens of a validation document, as in training."""
    prompts = []
    for doc in val_documents():
        if len(doc) + 1 >= lengths[len(prompts)]:
            prompts.append([EOT] + doc[: lengths[len(prompts)] - 1])
            if len(prompts) == len(lengths):
                return prompts
    raise RuntimeError("not enough long documents")


def load_nanogpt(device):
    ckpt = torch.load(CKPT, map_location="cpu", mmap=True)
    model = GPT(GPTConfig(**dict(ckpt["model_args"], sdpa_backend="auto")))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.inference_mode()
def main():
    device = "cuda"
    torch.cuda.init()
    cap_gpu_memory()
    attn_name, attn, fused = attention_from_args(BLOCK_SIZE, 4096, len(PROMPT_LENS), device)
    prompts = prompts_from_val(PROMPT_LENS)
    model = load_gpt2(CKPT, attn, device)
    model.fused = fused
    nano = load_nanogpt(device)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)

    def nano_logits(ids, bf16):
        # passing a target makes nanoGPT return logits at every position
        x = torch.tensor([ids], device=device)
        if bf16:
            with autocast:
                return nano(x, target=torch.zeros_like(x))[0][0]
        return nano(x, target=torch.zeros_like(x))[0][0]

    hidden, cu, our_tokens, our_step_logits = run_ours(model, prompts, NEW_TOKENS, BLOCK_SIZE, device)
    results = []
    for i, p in enumerate(prompts):
        ours = model.compute_logits(hidden[cu[i] : cu[i + 1]])
        ref32, ref16 = nano_logits(p, False), nano_logits(p, True)
        r = {
            "prompt_len": len(p),
            "prefill_ours_vs_nano_bf16": logit_stats(ours, ref16),
            "prefill_ours_vs_fp32": logit_stats(ours, ref32),
            "prefill_nano_bf16_vs_fp32": logit_stats(ref16, ref32),
            "prefill_bitwise_equal_to_nano_bf16": torch.equal(ours, ref16),
        }

        # teacher forcing on our generated tokens: rows len(p)-1 .. end-1 predict the generated tokens
        full = p + our_tokens[i][:-1]
        ref32, ref16 = nano_logits(full, False)[len(p) - 1 :], nano_logits(full, True)[len(p) - 1 :]
        r["decode_ours_vs_nano_bf16"] = logit_stats(our_step_logits[i], ref16)
        r["decode_ours_vs_fp32"] = logit_stats(our_step_logits[i], ref32)
        r["decode_nano_bf16_vs_fp32"] = logit_stats(ref16, ref32)

        with autocast:
            out = nano.generate(torch.tensor([p], device=device), NEW_TOKENS, temperature=0, use_cache=True)
        nano_tokens = out[0, len(p) :].tolist()
        div = next((j for j in range(NEW_TOKENS) if nano_tokens[j] != our_tokens[i][j]), None)
        r["greedy_first_divergence"] = div
        if div is not None:
            # nanoGPT's top-2 margin at the divergence, from its bf16 logits on its own tokens
            top2 = nano_logits(p + nano_tokens[:div], True)[-1].float().topk(2).values
            r["nano_top2_margin_at_divergence"] = (top2[0] - top2[1]).item()
        r["generated_padding_ids"] = sum(t >= model.cfg.real_vocab_size for t in our_tokens[i])

        # run alone, every matmul has the same shape as in nanoGPT, so the results should match bit for bit
        h, _, alone_tokens, _ = run_ours(model, [p], NEW_TOKENS, BLOCK_SIZE, device)
        r["alone_prefill_bitwise_equal"] = torch.equal(model.compute_logits(h), nano_logits(p, True))
        r["alone_greedy_identical"] = alone_tokens[0] == nano_tokens
        results.append(r)

        print(f"prompt {len(p):4d} tokens   prefill bitwise equal to nanoGPT bf16: {r['prefill_bitwise_equal_to_nano_bf16']}")
        for k in ("prefill_ours_vs_nano_bf16", "prefill_ours_vs_fp32", "prefill_nano_bf16_vs_fp32",
                  "decode_ours_vs_nano_bf16", "decode_ours_vs_fp32", "decode_nano_bf16_vs_fp32"):
            print_stats(k, r[k])
        if div is None:
            print(f"  greedy (batched): all {NEW_TOKENS} tokens identical to nanoGPT generate")
        else:
            print(f"  greedy (batched): first divergence at token {div}, nanoGPT top-2 margin there {r['nano_top2_margin_at_divergence']:.4f}")
        print(f"  alone: prefill bitwise equal {r['alone_prefill_bitwise_equal']}, greedy identical {r['alone_greedy_identical']}")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"e0_gpt2_{attn_name}{'_fused' if fused else ''}.json")
    with open(out, "w") as f:
        json.dump({"model": "nanoGPT GPT-2 124M", "attention": attn_name, "fused": fused, "block_size": BLOCK_SIZE, "new_tokens": NEW_TOKENS, "prompts": results}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
