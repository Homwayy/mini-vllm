"""Offline throughput of our engine on a fixed workload (E1-E7).

The engine is warmed up once (flex is compiled for every batch-size bucket), then the workload runs --runs
times. All requests arrive at t = 0. Each run's summary and per-request timings go to a JSON file.

Run from the repo root, e.g.:
    python -m bench.bench_offline --kv contiguous --batching static --out results/e1_ours.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from bench.workloads import load_workload
from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams


def time_weighted_mean(values, durations):
    return float(np.average(values, weights=durations)) if len(values) else float("nan")


def summarize(seqs, steps, t0, engine):
    cfg = engine.cfg
    pool_slots = cfg.num_kv_blocks * cfg.block_size
    elapsed = max(s.finish_time for s in seqs) - t0
    out_tokens = sum(s.num_completion_tokens for s in seqs)
    prompt_tokens = sum(s.num_prompt_tokens for s in seqs)
    ttft = [s.first_token_time - s.arrival_time for s in seqs]
    tpot = [(s.finish_time - s.first_token_time) / (s.num_completion_tokens - 1) for s in seqs if s.num_completion_tokens > 1]
    # inter-token latency: every gap between two consecutive tokens of a sequence, so single stalls show up
    itl = np.concatenate([np.diff(s.token_times) for s in seqs])
    e2e = [s.finish_time - s.arrival_time for s in seqs]
    dec = [r for r in steps if r.num_decode]  # steps that decode, alone or mixed with prompt chunks
    dur = [r.end - r.start for r in dec]
    pre_time = sum(r.end - r.start for r in steps if r.is_prefill)
    cached = sum(s.num_cached_tokens for s in seqs)

    def pct(x, q):
        return float(np.percentile(x, q))

    return {
        "elapsed_s": elapsed,
        "prompt_tokens": prompt_tokens,
        "output_tokens": out_tokens,
        "output_throughput": out_tokens / elapsed,
        "total_throughput": (prompt_tokens + out_tokens) / elapsed,
        "ttft_p50_s": pct(ttft, 50), "ttft_p99_s": pct(ttft, 99),
        "tpot_p50_ms": 1e3 * pct(tpot, 50), "tpot_p99_ms": 1e3 * pct(tpot, 99),
        "itl_p50_ms": 1e3 * pct(itl, 50), "itl_p99_ms": 1e3 * pct(itl, 99), "itl_max_ms": 1e3 * float(itl.max()),
        "e2e_p50_s": pct(e2e, 50), "e2e_p99_s": pct(e2e, 99),
        "prefill_steps": sum(r.is_prefill for r in steps),
        "mixed_steps": sum(r.is_prefill and r.num_decode > 0 for r in steps),
        "prefill_tokens_computed": sum(r.num_tokens - r.num_decode for r in steps),
        # prompt tokens whose K/V came from the prefix cache, at each sequence's last admission
        "cached_prompt_tokens": cached,
        "prefix_hit_rate": cached / prompt_tokens,
        "decode_steps": len(dec),
        "prefill_time_share": pre_time / elapsed,
        "decode_step_ms_mean": 1e3 * float(np.mean(dur)),
        # the averages below are weighted by the duration of the steps that decode
        "decode_batch_mean": time_weighted_mean([r.num_decode for r in dec], dur),
        "decode_batch_max": max(r.num_decode for r in dec),
        "kv_utilization": time_weighted_mean([r.kv_used_tokens / r.kv_allocated_slots for r in dec], dur),
        "kv_allocated_share_of_pool": time_weighted_mean([r.kv_allocated_slots / pool_slots for r in dec], dur),
        "kv_tokens_share_of_pool": time_weighted_mean([r.kv_used_tokens / pool_slots for r in dec], dur),
        "preemptions": engine.scheduler.num_preemptions,
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
    }


def main():
    d = EngineConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--workload", default="W1")
    ap.add_argument("--attention", default=d.attention)
    ap.add_argument("--kv", default=d.kv)
    ap.add_argument("--admission", default=d.admission)
    ap.add_argument("--batching", default=d.batching)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--chunked-prefill", action="store_true", help="mix prompt chunks with decoding; budget is --max-num-batched-tokens")
    ap.add_argument("--num-kv-blocks", type=int, default=d.num_kv_blocks)
    ap.add_argument("--max-num-seqs", type=int, default=d.max_num_seqs)
    ap.add_argument("--max-num-batched-tokens", type=int, default=d.max_num_batched_tokens)
    ap.add_argument("--max-model-len", type=int, default=d.max_model_len)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--no-fused-kernels", action="store_true", help="run each layer's elementwise ops one PyTorch kernel at a time")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N requests (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = EngineConfig(
        model=args.model, attention=args.attention, kv=args.kv, admission=args.admission, batching=args.batching,
        prefix_caching=args.prefix_caching, chunked_prefill=args.chunked_prefill,
        num_kv_blocks=args.num_kv_blocks, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
        cuda_graph=not args.no_cuda_graph, fused_kernels=not args.no_fused_kernels,
    )
    w = load_workload(args.workload)
    reqs = w["requests"][: args.limit or None]
    temperature = w["sampling"]["temperature"]

    engine = LLMEngine(cfg)
    runner = engine.runner
    print(f"{cfg}\nKV pool {runner.kv_bytes / 2**30:.2f} GiB, {cfg.num_kv_blocks * cfg.block_size} slots")
    if cfg.cuda_graph:
        print(f"captured {len(runner.graphs)} decode graphs and {len(runner.mixed_graphs)} mixed-step graphs in "
              f"{runner.graph_capture_s:.1f} s, {runner.graph_memory_bytes / 2**20:.0f} MiB")
    t = time.perf_counter()
    engine.warmup(temperature)
    warmup_s = time.perf_counter() - t
    print(f"warmup {warmup_s:.1f} s")

    runs = []
    for r in range(args.runs):
        torch.manual_seed(args.seed + r)
        torch.cuda.reset_peak_memory_stats()
        engine.reset()
        t0 = time.perf_counter()
        seqs = [
            engine.add_request(q["prompt_token_ids"], SamplingParams(temperature=temperature, max_tokens=q["max_tokens"], ignore_eos=True), t0)
            for q in reqs
        ]
        while engine.has_unfinished():
            engine.step()
        s = summarize(seqs, engine.steps, t0, engine)
        s["requests"] = {
            "prompt_len": [x.num_prompt_tokens for x in seqs],
            "output_len": [x.num_completion_tokens for x in seqs],
            "first_token_s": [x.first_token_time - t0 for x in seqs],
            "finish_s": [x.finish_time - t0 for x in seqs],
            "preemptions": [x.num_preemptions for x in seqs],
            "cached_tokens": [x.num_cached_tokens for x in seqs],
        }
        runs.append(s)
        print(
            f"run {r}: {s['elapsed_s']:.1f} s, output {s['output_throughput']:.0f} tok/s, total {s['total_throughput']:.0f} tok/s, "
            f"decode batch mean {s['decode_batch_mean']:.1f} max {s['decode_batch_max']}, KV util {s['kv_utilization']:.3f}, "
            f"tokens/pool {s['kv_tokens_share_of_pool']:.3f}, preemptions {s['preemptions']}, peak {s['peak_memory_gib']:.2f} GiB, "
            f"prefix hit rate {s['prefix_hit_rate']:.3f}, TTFT p50 {s['ttft_p50_s']:.2f} s, ITL p99 {s['itl_p99_ms']:.1f} ms"
        )

    med = float(np.median([s["output_throughput"] for s in runs]))
    print(f"median output throughput over {len(runs)} runs: {med:.0f} tok/s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": cfg.__dict__, "workload": w["name"], "num_requests": len(reqs), "temperature": temperature,
            "warmup_s": warmup_s, "graph_capture_s": runner.graph_capture_s, "graph_memory_mib": runner.graph_memory_bytes / 2**20,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "median_output_throughput": med, "runs": runs,
        }, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
