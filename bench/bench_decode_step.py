"""Decode step time inside the engine (E4): CUDA graphs on, Qwen3, every sequence with a 1024-token prompt.

For each batch size, that many requests are prefilled, a few decode steps run as warm-up, and then the
wall time of --steps decode steps (sampling included) is measured. The context grows from about 1030 to
1030 + --steps tokens during the measurement.

Run from the repo root, e.g.:
    python -m bench.bench_decode_step --attention triton --out results/e4_decode_step_triton.json
"""

import argparse
import json
import os
import time

import torch

from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams

PROMPT_LEN = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attention", default="flex")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 48])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    engine = LLMEngine(EngineConfig(attention=args.attention))
    engine.warmup()
    rows = []
    for B in args.batches:
        engine.reset()
        for i in range(B):
            prompt = [(13 * i + j) % 10000 for j in range(PROMPT_LEN)]
            engine.add_request(prompt, SamplingParams(temperature=0.6, max_tokens=args.steps + 10, ignore_eos=True))
        while engine.scheduler.waiting:
            engine.step()
        for _ in range(5):
            engine.step()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.steps):
            engine.step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) / args.steps * 1e3
        rows.append({"batch": B, "step_ms": ms, "tokens_per_s": B / ms * 1e3})
        print(f"{args.attention}: batch {B:3d}, decode step {ms:6.2f} ms, {B / ms * 1e3:7.0f} token/s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"attention": args.attention, "prompt_len": PROMPT_LEN, "steps": args.steps,
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
