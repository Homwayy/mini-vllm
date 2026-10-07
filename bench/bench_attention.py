"""Decode attention on its own (E4): time per layer and effective bandwidth of each backend.

Qwen3 shapes (16 query heads, 8 KV heads, head_dim 128, 16-token pages). Each sequence's blocks are
consecutive, as the engine's free list hands them out at the start of a run. Every measurement runs the
backend on 8 layers with separate KV pools (2 GiB in total) inside one CUDA graph, so K/V cannot be served
from L2 between repeats; the time per layer is the graph's time divided by 8. The backends are measured in
turns over several rounds, and the median round is reported. The gather backend pads every sequence to
max_model_len (2048), as it does inside the engine's CUDA graphs.

Run from the repo root: python -m bench.bench_attention --out results/e4_attention.json
"""

import argparse
import json
import os
import random
import statistics

import torch

from minivllm.attention import AttnMetadata
from minivllm.attention.flex import FlexAttention
from minivllm.attention.gather import GatherAttention
from minivllm.attention.triton_paged import PagedDecode

PAGE, NUM_BLOCKS, MAX_LEN = 16, 4096, 2048
HQ, HKV, D = 16, 8, 128
LAYERS = 8
KV_BYTES_PER_TOKEN = 2 * HKV * D * 2  # K and V of one layer, bf16


def make_batch(lens, device):
    width = MAX_LEN // PAGE
    tables, nxt = [], 0
    for n in lens:
        k = -(-n // PAGE)
        tables.append(list(range(nxt, nxt + k)) + [0] * (width - k))
        nxt += k
    assert nxt <= NUM_BLOCKS
    q = torch.randn(len(lens), HQ, D, device=device, dtype=torch.bfloat16)
    bt = torch.tensor(tables, dtype=torch.int32, device=device)
    ctx = torch.tensor(lens, dtype=torch.int32, device=device)
    return q, bt, ctx


def per_layer_ms(call, iters=10, repeats=5):
    """Captures call(layer) for all layers in one CUDA graph; returns the median time per layer in ms."""
    def step():
        for layer in range(LAYERS):
            call(layer)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / iters / LAYERS)
    return statistics.median(ts)


def read_bandwidth_gbs(device):
    x = torch.empty(1 << 28, dtype=torch.float32, device=device).normal_()  # 1 GiB
    ms = per_layer_ms(lambda layer: x.sum(), iters=5)
    return x.numel() * 4 / (ms * 1e-3) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    torch._dynamo.config.recompile_limit = 64
    pools = [(torch.randn(NUM_BLOCKS + 1, PAGE, HKV, D, device=device, dtype=torch.bfloat16),
              torch.randn(NUM_BLOCKS + 1, PAGE, HKV, D, device=device, dtype=torch.bfloat16)) for _ in range(LAYERS)]

    rng = random.Random(0)
    shapes = [(f"batch {b}, context 1024", [1024] * b) for b in (1, 2, 4, 8, 16, 32, 64)]
    shapes += [(f"batch {b}, context 100-2000", [rng.randint(100, 2000) for _ in range(b)]) for b in (1, 8, 24, 48)]
    shapes += [(f"batch {b}, context 512", [512] * b) for b in (96, 128)]

    flex = FlexAttention(PAGE, 4096, 256, device)
    gather = GatherAttention()
    triton_decode = PagedDecode()
    bandwidth = read_bandwidth_gbs(device)
    print(f"read bandwidth (sum of a 1 GiB tensor): {bandwidth:.0f} GB/s")

    rows = []
    for name, lens in shapes:
        q, bt, ctx = make_batch(lens, device)
        meta = AttnMetadata(is_prefill=False, slot_mapping=None, block_tables=bt, context_lens=ctx, max_context_len=MAX_LEN)
        calls = {
            "flex": lambda l: flex(q, None, None, pools[l][0], pools[l][1], meta),
            "triton": lambda l: triton_decode(q, pools[l][0], pools[l][1], bt, ctx),
            "gather": lambda l: gather(q, None, None, pools[l][0], pools[l][1], meta),
        }
        times = {k: [] for k in calls}
        for _ in range(args.rounds):
            for k, call in calls.items():
                times[k].append(per_layer_ms(call) * 1e3)
        kv_bytes = sum(lens) * KV_BYTES_PER_TOKEN
        row = {"shape": name, "batch": len(lens), "context_tokens": sum(lens), "kv_bytes_per_layer": kv_bytes}
        for k, ts in times.items():
            us = statistics.median(ts)
            row[k] = {"us_per_layer": us, "rounds_us": ts, "effective_gbs": kv_bytes / (us * 1e-6) / 1e9}
        rows.append(row)
        print(
            f"{name:28s} " + " | ".join(f"{k} {row[k]['us_per_layer']:7.1f} us ({row[k]['effective_gbs']:4.0f} GB/s)" for k in calls)
            + f" | flex / triton {row['flex']['us_per_layer'] / row['triton']['us_per_layer']:.2f}x",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"read_bandwidth_gbs": bandwidth, "heads": [HQ, HKV, D], "page": PAGE, "layers": LAYERS,
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "shapes": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
