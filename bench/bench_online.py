"""Online latency (W2): W1's requests arriving as a Poisson process, at several rates.

Each request is added to the engine once its arrival time has passed, and the engine keeps stepping in
between. TTFT, TPOT and end-to-end latency are measured from each request's own arrival time, so they
include waiting for the step that was running when the request arrived. Every rate replays the same
requests in the same order; only the gaps between arrivals scale.

Run from the repo root, e.g.:
    python -m bench.bench_online --batching continuous --rates 1 2 4 8 --out results/e3_online_continuous.json
"""

import argparse
import json
import os
import time

import torch

from bench.bench_offline import summarize
from bench.workloads import load_workload
from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams


def run_at_rate(engine, reqs, arrivals, temperature):
    engine.reset()
    t0 = time.perf_counter()
    seqs, i = [], 0
    while i < len(reqs) or engine.has_unfinished():
        now = time.perf_counter() - t0
        while i < len(reqs) and arrivals[i] <= now:
            params = SamplingParams(temperature=temperature, max_tokens=reqs[i]["max_tokens"], ignore_eos=True)
            seqs.append(engine.add_request(reqs[i]["prompt_token_ids"], params, t0 + arrivals[i]))
            i += 1
        if engine.has_unfinished():
            engine.step()
        elif i < len(reqs):
            time.sleep(max(0.0, arrivals[i] - (time.perf_counter() - t0)))
    return seqs, t0


def main():
    d = EngineConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--workload", default="W1")
    ap.add_argument("--attention", default=d.attention)
    ap.add_argument("--kv", default=d.kv)
    ap.add_argument("--admission", default=d.admission)
    ap.add_argument("--batching", default=d.batching)
    ap.add_argument("--num-kv-blocks", type=int, default=d.num_kv_blocks)
    ap.add_argument("--max-num-seqs", type=int, default=d.max_num_seqs)
    ap.add_argument("--max-num-batched-tokens", type=int, default=d.max_num_batched_tokens)
    ap.add_argument("--max-model-len", type=int, default=d.max_model_len)
    ap.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4, 8], help="requests per second")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N requests (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = EngineConfig(
        model=args.model, attention=args.attention, kv=args.kv, admission=args.admission, batching=args.batching,
        num_kv_blocks=args.num_kv_blocks, max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
    )
    w = load_workload(args.workload)
    reqs = w["requests"][: args.limit or None]
    arrivals_rate1 = w["arrival_times_rate1"][: len(reqs)]
    temperature = w["sampling"]["temperature"]

    engine = LLMEngine(cfg)
    print(cfg)
    engine.warmup(temperature)

    results = []
    for rate in args.rates:
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        seqs, t0 = run_at_rate(engine, reqs, [a / rate for a in arrivals_rate1], temperature)
        s = summarize(seqs, engine.steps, t0, engine)
        s["rate"] = rate
        s["requests"] = {
            "prompt_len": [x.num_prompt_tokens for x in seqs],
            "output_len": [x.num_completion_tokens for x in seqs],
            "arrival_s": [x.arrival_time - t0 for x in seqs],
            "first_token_s": [x.first_token_time - t0 for x in seqs],
            "finish_s": [x.finish_time - t0 for x in seqs],
            "preemptions": [x.num_preemptions for x in seqs],
        }
        results.append(s)
        print(
            f"rate {rate:g}/s: {s['elapsed_s']:.1f} s, output {s['output_throughput']:.0f} tok/s, "
            f"TTFT p50 {s['ttft_p50_s']:.2f} s p99 {s['ttft_p99_s']:.2f} s, TPOT p50 {s['tpot_p50_ms']:.1f} ms p99 {s['tpot_p99_ms']:.1f} ms, "
            f"decode batch mean {s['decode_batch_mean']:.1f}, preemptions {s['preemptions']}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": cfg.__dict__, "workload": "W2 (W1 requests, Poisson arrivals)", "num_requests": len(reqs),
            "temperature": temperature, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rates": results,
        }, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
