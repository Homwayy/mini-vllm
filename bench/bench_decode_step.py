"""Decode step time inside the engine (E4, E5, E9): every sequence with a --prompt-len prompt (1024 tokens).

For each batch size, that many requests are prefilled, a few decode steps run as warm-up, and then the
wall time of --steps decode steps (sampling included) is measured. The context grows from about
--prompt-len + 6 to --prompt-len + 6 + --steps tokens during the measurement.

Run from the repo root, e.g.:
    python -m bench.bench_decode_step --attention triton --out results/e4_decode_step_triton.json
    python -m bench.bench_decode_step --model gpt2 --prompt-len 512 --batches 1 8 32 64 --out results/e9_decode_step_gpt2.json
"""

import argparse
import json
import os
import time

import torch

from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=EngineConfig().model)
    ap.add_argument("--prompt-len", type=int, default=1024)
    ap.add_argument("--attention", default=EngineConfig().attention)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 48])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max-num-seqs", type=int, default=256, help="largest decode graph; gather needs 64 to fit in memory")
    ap.add_argument("--no-cuda-graph", action="store_true", help="launch the decode kernels from Python every step")
    ap.add_argument("--no-fused-kernels", action="store_true", help="run each layer's elementwise ops one PyTorch kernel at a time")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = EngineConfig(model=args.model, attention=args.attention, max_num_seqs=args.max_num_seqs,
                       max_model_len=1024 if args.model == "gpt2" else EngineConfig().max_model_len,
                       cuda_graph=not args.no_cuda_graph, fused_kernels=not args.no_fused_kernels)
    engine = LLMEngine(cfg)
    engine.warmup()
    rows = []
    for B in args.batches:
        engine.reset()
        for i in range(B):
            prompt = [(13 * i + j) % 10000 for j in range(args.prompt_len)]
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
        print(f"{args.model} {args.attention}{'' if engine.runner.use_cuda_graph else ' (no graph)'}"
              f"{'' if cfg.fused_kernels else ' (not fused)'}: batch {B:3d}, "
              f"decode step {ms:6.2f} ms, {B / ms * 1e3:7.0f} token/s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"model": args.model, "attention": args.attention, "cuda_graph": not args.no_cuda_graph,
                   "fused_kernels": cfg.fused_kernels, "prompt_len": args.prompt_len, "steps": args.steps,
                   "max_num_seqs": args.max_num_seqs,
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
