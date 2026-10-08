"""The kernels of one CUDA graph replay, by kind, with their GPU time and the replay's wall time (E9).

Graphs: the decode step of Qwen3 and of GPT-2 at batch 1 and 64 (prompts of --prompt-len tokens, run
through the engine first, so attention reads real contexts), and with EAGLE (GPT-2) the verify step over
8 tokens, the draft chain at batch 1 and the draft's catch-up over 8 tokens. Each graph is replayed in 5
blocks of --reps replays under CUDA events, and the median block gives the wall time (back-to-back blocks
on this laptop GPU vary by up to 7%); then --reps replays run under the profiler for the kernels.
The kind of a kernel is read from its name: matmul, attention, or other (elementwise ops, norms, copies,
casts, indexing, softmax and sampling).

Run from the repo root, e.g.:
    python -m bench.bench_kernels --out results/e9_kernels.json
    python -m bench.bench_kernels --no-fused-kernels --out results/e9_kernels_unfused.json
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams


def kind(name):
    n = name.lower()
    if any(s in n for s in ("gemm", "gemv", "cublas", "splitk", "cutlass")):
        return "matmul"
    if name.startswith(("_decode", "_prefill", "_merge")):
        return "attention"
    return "other"


@torch.inference_mode()
def measure(graph, reps):
    for _ in range(5):
        graph.replay()
    blocks = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        blocks.append(start.elapsed_time(end) / reps * 1e3)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            graph.replay()
        torch.cuda.synchronize()
    by_kind = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type == DeviceType.CUDA:
            by_kind[kind(e.name)][0] += 1
            by_kind[kind(e.name)][1] += e.device_time
    return {
        "wall_us": sorted(blocks)[len(blocks) // 2], "wall_us_blocks": blocks,
        "kernels": sum(c for c, _ in by_kind.values()) / reps,
        "gpu_us": sum(t for _, t in by_kind.values()) / reps,
        "by_kind": {k: {"kernels": c / reps, "gpu_us": t / reps} for k, (c, t) in sorted(by_kind.items())},
    }


def prefill(engine, batch, prompt_len):
    engine.reset()
    for i in range(batch):
        engine.add_request([(13 * i + j) % 10000 for j in range(prompt_len)],
                           SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True))
    while engine.scheduler.waiting:
        engine.step()
    for _ in range(4):
        engine.step()
    return list(engine.scheduler.running)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--no-fused-kernels", action="store_true", help="run each layer's elementwise ops one PyTorch kernel at a time")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    fused = not args.no_fused_kernels

    rows = []

    def record(model, graph_name, size, graph):
        r = {"model": model, "graph": graph_name, "size": size, **measure(graph, args.reps)}
        rows.append(r)
        print(f"{model} {graph_name} {size:3d}: {r['kernels']:5.0f} kernels, GPU {r['gpu_us']:7.1f} us, wall {r['wall_us']:7.1f} us  "
              + ", ".join(f"{k} {v['kernels']:.0f} / {v['gpu_us']:.0f} us" for k, v in r["by_kind"].items()), flush=True)

    for model in ("qwen3", "gpt2"):
        cfg = EngineConfig(model=model, max_num_seqs=64, max_model_len=1024 if model == "gpt2" else 2048,
                           spec="eagle" if model == "gpt2" else None, fused_kernels=fused)
        engine = LLMEngine(cfg)
        engine.warmup()
        runner = engine.runner
        for batch in (1, 64):
            seqs = prefill(engine, batch, args.prompt_len)
            # without drafts, a step is a plain decode step
            padded = next(b for b in runner.buckets if b >= len(seqs))
            runner.fill_decode_buffers(seqs, padded)
            record(model, "decode", padded, runner.graphs[padded][0])
        if cfg.spec == "eagle":
            seqs = prefill(engine, 1, args.prompt_len)
            seq = seqs[0]
            runner.fill_mixed_buffers([(seq, len(seq) - 1, [seq.token_ids[-1]] * 8)], 8)
            record(model, "verify", 8, runner.mixed_graphs[8][0])
            record(model, "draft catch-up", 8, runner.draft_graphs[8][0])
            runner.fill_chain_buffers([seq], 1)
            record(model, "draft chain", 1, runner.chain_graphs[1])
        del engine, runner
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"fused_kernels": fused, "prompt_len": args.prompt_len, "reps": args.reps,
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "rows": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
