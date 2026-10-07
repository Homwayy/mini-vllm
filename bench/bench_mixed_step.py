"""What a prompt costs the sequences that are decoding (E7): Qwen3, wall time of single steps.

Without chunked prefill, a prompt runs as its own prefill step (eager, no padding) and the decoding
sequences wait for it and then for their decode step. With chunked prefill, the prompt or a chunk of it
runs in the same step as the decode rows, padded to a token bucket and replayed as a CUDA graph. For a
prompt of n tokens and --batch decoding sequences with --context tokens each, this measures:
  decode: the decode step alone;
  eager prefill: the prompt as its own step;
  graph prefill: the prompt alone through the mixed-step graph, which separates launch overhead from padding;
  mixed: decode rows and prompt in one step.
The pool is zeroed first, so attention reads real memory but the sampled tokens mean nothing.

Run from the repo root, e.g.:
    python -m bench.bench_mixed_step --out results/e7_mixed_step.json
"""

import argparse
import json
import os
import statistics
import time

import torch

from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams, Sequence


def wall_ms(fn, iters):
    for _ in range(3):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()  # ends by copying the sampled tokens to the CPU
        times.append(time.perf_counter() - t)
    return 1e3 * statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2048, help="largest mixed-step bucket that is captured")
    ap.add_argument("--batch", type=int, default=43, help="decoding sequences (W2's mean at 8 requests/s)")
    ap.add_argument("--context", type=int, default=1000)
    ap.add_argument("--prompts", type=int, nargs="+", default=[64, 128, 256, 469, 512, 720, 981, 1024, 1100, 1500, 2005, 2048])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = EngineConfig(chunked_prefill=True, max_num_batched_tokens=args.budget)
    engine = LLMEngine(cfg)
    runner, kv = engine.runner, engine.kv
    engine.warmup()
    for a in runner.model.attn_layers:
        a.k_cache.zero_()
        a.v_cache.zero_()

    def new_seq(n, seed):
        s = Sequence([(seed * 31 + j) % 10000 for j in range(n)], SamplingParams(temperature=0.6, max_tokens=1), 0.0)
        kv.admit(s)
        return s

    dec = [new_seq(args.context, i) for i in range(args.batch)]
    for s in dec:
        s.num_computed_tokens = len(s) - 1
    decode_ms = wall_ms(lambda: runner.run(dec, is_prefill=False), args.iters)
    print(f"decode step, batch {args.batch}, context {args.context}: {decode_ms:.2f} ms", flush=True)

    rows = []
    for n in args.prompts:
        p = new_seq(n, 1000 + n)
        row = {"prompt": n, "eager_prefill_ms": wall_ms(lambda: runner.run([p], is_prefill=True), args.iters)}
        row["separate_ms"] = row["eager_prefill_ms"] + decode_ms
        if n in runner.mixed_buckets:  # exactly a bucket: no padding
            row["graph_prefill_ms"] = wall_ms(lambda: runner.run_chunked([(p, n)]), args.iters)
        if args.batch + n <= args.budget:
            row["bucket"] = next(b for b in runner.mixed_buckets if b >= args.batch + n)
            row["mixed_ms"] = wall_ms(lambda: runner.run_chunked([(s, 1) for s in dec] + [(p, n)]), args.iters)
        kv.release(p)
        rows.append(row)
        print(f"prompt {n:5d}: eager prefill {row['eager_prefill_ms']:6.2f} ms"
              + (f", graph prefill {row['graph_prefill_ms']:6.2f} ms" if "graph_prefill_ms" in row else "")
              + f" | prefill + decode {row['separate_ms']:6.2f} ms"
              + (f", mixed ({args.batch} + {n} -> {row['bucket']}) {row['mixed_ms']:6.2f} ms" if "mixed_ms" in row else ""),
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"budget": args.budget, "batch": args.batch, "context": args.context, "iters": args.iters,
                   "decode_ms": decode_ms, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "rows": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
