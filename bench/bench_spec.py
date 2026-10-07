"""Speculative decoding (E8): GPT-2 on W4, offline, at several concurrency levels.

The engine is built once with max_num_seqs set to the highest level; each level then caps how many requests
run at once. All requests arrive at t = 0. Concurrency 1 runs only the first --requests-c1 requests, so it
takes about as long as the other levels. Every level is run --runs times and the run with the median output
throughput is kept.

Besides throughput and latency, each level reports how drafting went over its decode steps:
  tokens per step: tokens a sequence gains per decode step (accepted drafts + 1), averaged over all
    (sequence, step) pairs; without drafting it is 1;
  drafts per step: draft tokens checked per sequence and step;
  acceptance rate: accepted drafts / drafts checked.
It also reports how repetitive the outputs are, which bounds what copying can achieve:
  copyable share: output tokens equal to the token that followed the latest earlier occurrence of the
    4 tokens before them (in prompt or output).

Run from the repo root, e.g.:
    python -m bench.bench_spec --out results/e8a_off.json
    python -m bench.bench_spec --spec ngram --num-spec-tokens 4 --out results/e8a_ngram4.json
    python -m bench.bench_spec --spec eagle --num-spec-tokens 4 --out results/e8b_eagle4.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from bench.bench_offline import summarize
from bench.workloads import load_workload
from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams


def copyable_share(seqs, n=4):
    hits = total = 0
    for s in seqs:
        t, last = s.token_ids, {}
        for i in range(n, len(t)):
            key = tuple(t[i - n : i])
            if i >= s.num_prompt_tokens:
                total += 1
                hits += key in last and t[last[key]] == t[i]
            last[key] = i
    return hits / total


def run_once(engine, reqs, concurrency, temperature, seed):
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    engine.reset()
    engine.scheduler.max_num_seqs = concurrency
    t0 = time.perf_counter()
    seqs = [
        engine.add_request(q["prompt_token_ids"], SamplingParams(temperature=temperature, max_tokens=q["max_tokens"], ignore_eos=True), t0)
        for q in reqs
    ]
    while engine.has_unfinished():
        engine.step()
    s = summarize(seqs, engine.steps, t0, engine)
    dec = [r for r in engine.steps if r.num_decode]
    seq_steps = sum(r.num_decode for r in dec)
    drafted, accepted = sum(r.num_draft for r in dec), sum(r.num_accepted for r in dec)
    s.update({
        "tokens_per_step": (seq_steps + accepted) / seq_steps,
        "drafts_per_step": drafted / seq_steps,
        "acceptance_rate": accepted / drafted if drafted else float("nan"),
        "drafted_tokens": drafted,
        "accepted_tokens": accepted,
        "copyable_share": copyable_share(seqs),
    })
    return s


def main():
    d = EngineConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="W4")
    ap.add_argument("--spec", default=None, help="drafting method, e.g. ngram; off by default")
    ap.add_argument("--num-spec-tokens", type=int, default=d.num_spec_tokens)
    ap.add_argument("--draft-path", default=d.draft_path, help="the EAGLE draft model")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--requests-c1", type=int, default=32, help="requests run at concurrency 1")
    ap.add_argument("--temperature", type=float, default=None, help="overrides the workload's temperature")
    ap.add_argument("--num-kv-blocks", type=int, default=d.num_kv_blocks)
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = load_workload(args.workload)
    temperature = w["sampling"]["temperature"] if args.temperature is None else args.temperature
    cfg = EngineConfig(
        model=w.get("model", "gpt2"), max_model_len=1024, num_kv_blocks=args.num_kv_blocks,
        max_num_seqs=max(args.concurrency), spec=args.spec, num_spec_tokens=args.num_spec_tokens, draft_path=args.draft_path,
        cuda_graph=not args.no_cuda_graph,
    )
    engine = LLMEngine(cfg)
    runner = engine.runner
    print(f"{cfg}\ntemperature {temperature}")
    if cfg.cuda_graph:
        print(f"captured {len(runner.graphs)} decode, {len(runner.mixed_graphs)} verify-step and "
              f"{len(runner.draft_graphs) + len(runner.chain_graphs)} drafting graphs in "
              f"{runner.graph_capture_s:.1f} s, {runner.graph_memory_bytes / 2**20:.0f} MiB")
    t = time.perf_counter()
    engine.warmup(temperature)
    warmup_s = time.perf_counter() - t

    levels = []
    for c in args.concurrency:
        reqs = w["requests"][: args.requests_c1] if c == 1 else w["requests"]
        runs = [run_once(engine, reqs, c, temperature, args.seed + r) for r in range(args.runs)]
        tput = [s["output_throughput"] for s in runs]
        s = runs[int(np.argsort(tput)[len(tput) // 2])]
        s.update({"concurrency": c, "num_requests": len(reqs), "output_throughput_runs": tput})
        levels.append(s)
        print(f"concurrency {c:3d}: output {s['output_throughput']:7.0f} tok/s (runs {', '.join(f'{x:.0f}' for x in tput)}), "
              f"TPOT p50 {s['tpot_p50_ms']:.2f} ms, decode steps {s['decode_steps']}, step {s['decode_step_ms_mean']:.2f} ms, "
              f"tokens/step {s['tokens_per_step']:.2f}, drafts/step {s['drafts_per_step']:.2f}, "
              f"acceptance {s['acceptance_rate']:.3f}, copyable {s['copyable_share']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": cfg.__dict__, "workload": w["name"], "temperature": temperature, "requests_c1": args.requests_c1,
            "warmup_s": warmup_s, "graph_capture_s": runner.graph_capture_s, "graph_memory_mib": runner.graph_memory_bytes / 2**20,
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "levels": levels,
        }, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
