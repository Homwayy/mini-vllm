"""Where the time of a decode step goes with speculative decoding (E8b), on W4 at several concurrency levels.

Runs every level once as bench_spec does, and splits the wall time of each decode step (engine.step) into
  target: the target model's step over each sequence's last token and its drafts, with the acceptance
    decision (runner.run_verify; without drafts, the plain decode step runner.run);
  draft: drafting the next tokens: EAGLE's engine._draft (the draft model catches up on the verified
    positions, then drafts), or n-gram's lookups (proposer.propose);
  other: the rest of the step (scheduling, appending tokens, KV bookkeeping).
target and draft each end with a copy to the CPU, so their GPU work falls inside their time.

It also reports how much the graphs pad: the tokens a decode step computes in the target model, and the
draft model's catch-up rows, against the bucket each is padded to.

--detail runs every level once more and synchronizes around each buffer fill and graph replay, to split
target and draft further (the synchronizations make that run slightly slower):
  fill: writing the step's inputs into the static buffers; forward: the target model's graph;
  catchup: the draft model's graph over the verified positions; chain_fill, chain: the inputs and the graph
  of the drafting chain; rest: everything else (logits, sampling and acceptance, gathers, copies to the CPU).
  After a verify step, EAGLE builds its inputs in one kernel instead of fills, which counts as draft.rest.

Run from the repo root, e.g.:
    python -m bench.bench_spec_breakdown --spec eagle --detail --out results/e8b_breakdown_greedy_eagle4.json
"""

import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from bench.bench_spec import run_once
from bench.workloads import load_workload
from minivllm.engine import EngineConfig, LLMEngine


class StepTimer:
    """Adds the time of wrapped calls to a part of the current step. A call inside a part of the same name
    counts once; a detail call is named after the part it runs in (target.fill or draft.fill)."""

    def __init__(self):
        self.parts = defaultdict(float)
        self.active = set()
        self.sync = False

    def call(self, part, fn, *args, **kwargs):
        if part.startswith("."):
            part = ("draft" if "draft" in self.active else "target") + part
        sub = "." in part
        if part in self.active or (sub and not self.sync):  # sub-parts are only timed when synchronized
            return fn(*args, **kwargs)
        if sub:
            torch.cuda.synchronize()
        self.active.add(part)
        t = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            if sub:
                torch.cuda.synchronize()
            self.parts[part] += time.perf_counter() - t
            self.active.discard(part)

    def wrap(self, obj, name, part):
        fn = getattr(obj, name)
        setattr(obj, name, lambda *a, **k: self.call(part, fn, *a, **k))


class TimedGraph:
    def __init__(self, timer, part, graph):
        self.timer, self.part, self.graph = timer, part, graph

    def replay(self):
        self.timer.call(self.part, self.graph.replay)


def bucket(sizes, n):
    return next(b for b in sizes if b >= n)


def instrument(engine, timer):
    """Wraps the engine's step and its parts. Returns per-kind accumulators for decode and prefill steps, and
    the list that collects (rows, bucket) of every catch-up."""
    runner = engine.runner
    acc = {"decode": defaultdict(float), "prefill": defaultdict(float)}
    catchups = []
    draft_next = runner.draft_next

    def recorded_draft_next(entries):
        # EAGLE, after a decode or verify step: a catch-up over the verified rows (DSpark has none)
        if runner.spec == "eagle" and runner.last[3] is None and runner.last_prefill is None:
            rows = sum(n for _, n in entries)
            # after a verify step, it runs over the verify step's padded rows (ModelRunner.draft_after_verify)
            padded = runner.last_verify or bucket(runner.mixed_buckets, rows)
            catchups.append((rows, padded))
        return draft_next(entries)

    runner.draft_next = recorded_draft_next
    timer.wrap(runner, "run", "target")
    timer.wrap(runner, "run_verify", "target")
    timer.wrap(engine, "_draft", "draft")
    if engine.proposer is not None:
        timer.wrap(engine.proposer, "propose", "draft")
    timer.wrap(runner, "fill_decode_buffers", ".fill")
    timer.wrap(runner, "fill_mixed_buffers", ".fill")
    timer.wrap(runner, "fill_chain_buffers", "draft.chain_fill")
    for graphs, part in ((runner.graphs, "target.forward"), (runner.mixed_graphs, "target.forward"),
                         (runner.prefill_graphs, "target.forward"), (runner.draft_graphs, "draft.catchup"),
                         (runner.draft_prefill_graphs, "draft.catchup")):
        for k, (g, out) in graphs.items():
            graphs[k] = (TimedGraph(timer, part, g), out)
    for k, g in runner.chain_graphs.items():
        runner.chain_graphs[k] = TimedGraph(timer, "draft.chain", g)
    step = engine.step

    def timed_step():
        timer.parts.clear()
        n = len(engine.steps)
        t = time.perf_counter()
        finished = step()
        total = time.perf_counter() - t
        if len(engine.steps) > n:
            a = acc["prefill" if engine.steps[-1].is_prefill else "decode"]
            a["steps"] += 1
            a["total"] += total
            for part, v in timer.parts.items():
                a[part] += v
        return finished

    engine.step = timed_step
    return acc, catchups


def per_step_ms(a):
    n = a.pop("steps", 0)
    out = {part: 1e3 * v / n for part, v in sorted(a.items())} if n else {}
    if n:
        out["other"] = out["total"] - sum(out.get(p, 0.0) for p in ("target", "draft"))
        for top in ("target", "draft"):
            subs = [p for p in out if p.startswith(top + ".")]
            if subs:
                out[top + ".rest"] = out[top] - sum(out[p] for p in subs)
    return n, out


def padding(pairs):
    """Mean real and padded sizes of (real, bucket) pairs, and the share of each bucket."""
    if not pairs:
        return {}
    real, padded = np.array(pairs).T
    sizes, counts = np.unique(padded, return_counts=True)
    return {"real_mean": float(real.mean()), "padded_mean": float(padded.mean()),
            "bucket_share": {int(b): float(n / len(padded)) for b, n in zip(sizes, counts)}}


def measure(engine, timer, acc, catchups, reqs, c, temperature, seed, sync):
    for a in acc.values():
        a.clear()
    catchups.clear()
    timer.sync = sync
    s = run_once(engine, reqs, c, temperature, seed)
    n_dec, dec = per_step_ms(acc["decode"])
    n_pre, pre = per_step_ms(acc["prefill"])
    runner = engine.runner
    # a step with drafts runs a mixed-step graph over all its tokens; one without, the decode graph
    verify = [(r.num_tokens, bucket(runner.mixed_buckets, r.num_tokens) if r.num_draft else bucket(runner.buckets, r.num_seqs))
              for r in engine.steps if r.num_decode]
    return {
        "output_throughput": s["output_throughput"], "elapsed_s": s["elapsed_s"],
        "tokens_per_step": s["tokens_per_step"], "drafts_per_step": s["drafts_per_step"],
        "acceptance_rate": s["acceptance_rate"], "decode_batch_mean": s["decode_batch_mean"],
        "decode_steps": n_dec, "decode_ms": dec, "prefill_steps": n_pre, "prefill_ms": pre,
        "target_tokens": padding(verify), "catchup_rows": padding(catchups),
    }


def main():
    d = EngineConfig()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="W4")
    ap.add_argument("--spec", default=None, help="drafting method: ngram, eagle or dspark; off by default")
    ap.add_argument("--num-spec-tokens", type=int, default=d.num_spec_tokens)
    ap.add_argument("--draft-path", default=d.draft_path)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--requests-c1", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=None, help="overrides the workload's temperature")
    ap.add_argument("--detail", action="store_true", help="also run each level with synchronized sub-parts")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-fused-kernels", action="store_true", help="run each layer's elementwise ops one PyTorch kernel at a time")
    ap.add_argument("--prefill-graph-tokens", type=int, default=d.prefill_graph_tokens, help="largest prefill step replayed as a CUDA graph; 0: all eager")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = load_workload(args.workload)
    temperature = w["sampling"]["temperature"] if args.temperature is None else args.temperature
    cfg = EngineConfig(
        model=w.get("model", "gpt2"), max_model_len=1024, max_num_seqs=max(args.concurrency), spec=args.spec,
        num_spec_tokens=args.num_spec_tokens, draft_path=args.draft_path, fused_kernels=not args.no_fused_kernels,
        prefill_graph_tokens=args.prefill_graph_tokens,
    )
    engine = LLMEngine(cfg)
    engine.warmup(temperature)
    timer = StepTimer()
    acc, catchups = instrument(engine, timer)
    print(f"spec {cfg.spec}, num_spec_tokens {cfg.num_spec_tokens}, temperature {temperature}")

    levels = []
    for c in args.concurrency:
        reqs = w["requests"][: args.requests_c1] if c == 1 else w["requests"]
        level = {"concurrency": c, "num_requests": len(reqs)}
        level.update(measure(engine, timer, acc, catchups, reqs, c, temperature, args.seed, sync=False))
        if args.detail:
            level["detail"] = measure(engine, timer, acc, catchups, reqs, c, temperature, args.seed, sync=True)
        levels.append(level)
        ms = level["decode_ms"]
        print(f"concurrency {c:3d}: output {level['output_throughput']:7.0f} tok/s, decode step {ms['total']:.2f} ms = "
              f"target {ms['target']:.2f} + draft {ms.get('draft', 0.0):.2f} + other {ms['other']:.2f}, "
              f"tokens/step {level['tokens_per_step']:.2f}", flush=True)
        for name in ("target_tokens", "catchup_rows"):
            pad = level[name]
            if pad:
                print(f"    {name}: {pad['real_mean']:.1f} padded to {pad['padded_mean']:.1f}, buckets "
                      + ", ".join(f"{b}: {share:.0%}" for b, share in pad["bucket_share"].items()), flush=True)
        if args.detail:
            dm = level["detail"]["decode_ms"]
            print("    detail: " + ", ".join(f"{p} {v:.3f}" for p, v in dm.items()), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"config": cfg.__dict__, "workload": w["name"], "temperature": temperature,
                   "requests_c1": args.requests_c1, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                   "levels": levels}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
