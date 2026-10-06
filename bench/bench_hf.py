"""HF transformers static batching on a fixed workload: the reference point for E1.

Requests are taken in workload order, --batch-size at a time. Prompts are left-padded, and every batch
generates until its longest request is done (min_new_tokens = max_new_tokens), so rows whose request has
already finished keep generating. Throughput counts only each request's own max_tokens. Sampling matches
the workload: temperature only, with Qwen3's default top-k and top-p switched off.

Batch 24 is the largest that fits W1 on a 12 GB card: the cache grows by torch.cat every step, and at batch
32 fragmentation leaves 3.4 GiB reserved but unusable before the longest batches finish.

Run from the repo root:
    python -m bench.bench_hf --batch-size 24 --out results/e1_hf_static_b24.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from bench.workloads import load_workload
from minivllm.utils import cap_gpu_memory

CKPT = os.path.join("checkpoints", "Qwen3-0.6B")
PAD_ID = 151643  # Qwen3's <|endoftext|>, also its pad token


@torch.inference_mode()
def run_batch(model, reqs, temperature):
    L = max(len(r["prompt_token_ids"]) for r in reqs)
    ids = torch.full((len(reqs), L), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(reqs), L), dtype=torch.long)
    for i, r in enumerate(reqs):
        p = r["prompt_token_ids"]
        ids[i, L - len(p) :] = torch.tensor(p)
        mask[i, L - len(p) :] = 1
    n = max(r["max_tokens"] for r in reqs)
    model.generate(
        input_ids=ids.cuda(), attention_mask=mask.cuda(), max_new_tokens=n, min_new_tokens=n,
        do_sample=True, temperature=temperature, top_k=0, top_p=1.0, pad_token_id=PAD_ID,
    )
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="W1")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N requests (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.init()
    cap_gpu_memory()
    w = load_workload(args.workload)
    reqs = w["requests"][: args.limit or None]
    temperature = w["sampling"]["temperature"]
    model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16).cuda().eval()
    batches = [reqs[i : i + args.batch_size] for i in range(0, len(reqs), args.batch_size)]
    useful = sum(r["max_tokens"] for r in reqs)
    prompt_tokens = sum(len(r["prompt_token_ids"]) for r in reqs)

    t = time.perf_counter()
    run_batch(model, reqs[:2], temperature)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t

    runs = []
    for r in range(args.runs):
        torch.manual_seed(args.seed + r)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        generated, batch_s = 0, []
        for b in batches:
            tb = time.perf_counter()
            generated += len(b) * run_batch(model, b, temperature)
            torch.cuda.synchronize()
            batch_s.append(time.perf_counter() - tb)
        elapsed = time.perf_counter() - t0
        s = {
            "elapsed_s": elapsed,
            "output_tokens": useful,
            "generated_tokens": generated,  # includes rows that kept generating after their request was done
            "output_throughput": useful / elapsed,
            "total_throughput": (prompt_tokens + useful) / elapsed,
            "batch_s": batch_s,
            "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        runs.append(s)
        print(
            f"run {r}: {elapsed:.1f} s, output {s['output_throughput']:.0f} tok/s, total {s['total_throughput']:.0f} tok/s, "
            f"useful share of generated tokens {useful / generated:.3f}, peak {s['peak_memory_gib']:.2f} GiB"
        )

    med = float(np.median([s["output_throughput"] for s in runs]))
    print(f"median output throughput over {len(runs)} runs: {med:.0f} tok/s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "engine": "transformers generate, static batching", "batch_size": args.batch_size,
            "attn_implementation": model.config._attn_implementation, "workload": w["name"],
            "num_requests": len(reqs), "temperature": temperature, "warmup_s": warmup_s,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "median_output_throughput": med, "runs": runs,
        }, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
