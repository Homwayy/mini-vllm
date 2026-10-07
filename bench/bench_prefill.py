"""Prefill attention on its own: FlexAttention (new tokens only) against the Triton paged kernel.

Qwen3 shapes (16 query heads, 8 KV heads, head_dim 128, 16-token pages). As in bench_attention, several
layers run inside one CUDA graph and the graph's time is divided by the number of layers; each layer has
its own new tokens and reads its own window of one large K/V buffer, and there are enough layers that a
replay reads at least 256 MiB of K/V. The new tokens' K/V is written into the pool first, as in the
models, so both backends see the same K/V and their outputs are compared. Shapes with a cached prefix,
which flex cannot attend to, run with Triton only. The backends are measured in turns over several rounds,
and the median round is reported.

Run from the repo root: python -m bench.bench_prefill --out results/e4b_prefill.json
"""

import argparse
import json
import math
import os
import random
import statistics

import torch

from minivllm.attention import AttnMetadata, store_kv
from minivllm.attention.flex import FlexAttention
from minivllm.attention.triton_paged import TritonAttention

PAGE, POOL_BLOCKS = 16, 4096 + 1
HQ, HKV, D = 16, 8, 128
MIN_LAYERS, MAX_LAYERS, READ_PER_REPLAY = 8, 64, 256 << 20
KV_BYTES_PER_TOKEN = 2 * HKV * D * 2


def per_layer_ms(call, layers, iters=10, repeats=5):
    """Captures call(layer) for every layer in one CUDA graph; returns the median time per layer in ms."""
    def step():
        for layer in range(layers):
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
        ts.append(a.elapsed_time(b) / iters / layers)
    return statistics.median(ts)


def matmul_tflops(device, n=8192):
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    ms = per_layer_ms(lambda layer: torch.matmul(a, b), 2, iters=5)
    return 2 * n**3 / (ms * 1e-3) / 1e12


def make_shape(cached, new, device):
    total = [c + n for c, n in zip(cached, new)]
    npages = [-(-t // PAGE) for t in total]
    width, tables, nxt = max(npages), [], 0
    for n in npages:
        tables.append(list(range(nxt, nxt + n)) + [0] * (width - n))
        nxt += n
    assert nxt < POOL_BLOCKS  # fits in the engine's pool
    bt = torch.tensor(tables, dtype=torch.int32, device=device)
    # slots of the new tokens, which are the last ones of each sequence
    slots = torch.cat([bt[b].long()[pos // PAGE] * PAGE + pos % PAGE
                       for b, pos in enumerate(torch.arange(c, t, device=device) for c, t in zip(cached, total))])
    cu = [0]
    for n in new:
        cu.append(cu[-1] + n)
    # every new token attends to its sequence's cached tokens and to the new ones up to itself
    pairs = sum(n * c + n * (n + 1) // 2 for c, n in zip(cached, new))
    return dict(total=total, nblocks=nxt, bt=bt, slots=slots, cu=cu, pairs=pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    torch._dynamo.config.recompile_limit = 128

    rng = random.Random(0)
    w1 = []  # W1-like prefill step: prompts of 100-1024 tokens, up to the 16,384-token budget
    while sum(w1) < 16384 - 1024:
        w1.append(rng.randint(100, 1024))
    shapes = [
        ("1 x 128", [0], [128]),
        ("1 x 1024", [0], [1024]),
        ("4 x 1024", [0] * 4, [1024] * 4),
        ("16 x 1024", [0] * 16, [1024] * 16),
        (f"{len(w1)} x 100-1024", [0] * len(w1), w1),
        ("1 x 512 after 1536 cached", [1536], [512]),
        ("16 x 256 after 768 cached", [768] * 16, [256] * 16),
        ("64 x 16 after 1008 cached", [1008] * 64, [16] * 64),
        ("64 x 4 after 1020 cached", [1020] * 64, [4] * 64),
        ("8 x 4 after 1020 cached", [1020] * 8, [4] * 8),
    ]
    plans = []
    for name, cached, new in shapes:
        x = make_shape(cached, new, device)
        layers = min(MAX_LAYERS, max(MIN_LAYERS, math.ceil(READ_PER_REPLAY / (sum(x["total"]) * KV_BYTES_PER_TOKEN))))
        plans.append((name, cached, new, x, layers))
    total_blocks = max((layers - 1) * x["nblocks"] + POOL_BLOCKS for *_, x, layers in plans)
    k_buf = torch.randn(total_blocks, PAGE, HKV, D, device=device, dtype=torch.bfloat16)
    v_buf = torch.randn(total_blocks, PAGE, HKV, D, device=device, dtype=torch.bfloat16)

    flex = FlexAttention(PAGE, 16384, 256, device)
    triton_attn = TritonAttention(PAGE, 16384, 256, device)
    peak = matmul_tflops(device)
    print(f"bf16 matmul 8192^3: {peak:.1f} TFLOPS; K/V buffer {k_buf.numel() * 4 / 2**30:.2f} GiB")

    rows = []
    for name, cached, new, x, layers in plans:
        T = sum(new)

        def pool(layer, nblocks=x["nblocks"]):
            first = layer * nblocks
            return k_buf[first:first + POOL_BLOCKS], v_buf[first:first + POOL_BLOCKS]

        # per layer, strided q/k/v views of one projection output, with k/v stored into the layer's pool
        qkv = []
        for layer in range(layers):
            t = torch.randn(T, (HQ + 2 * HKV) * D, device=device, dtype=torch.bfloat16)
            q, k, v = t.split([HQ * D, HKV * D, HKV * D], dim=-1)
            q, k, v = q.view(T, HQ, D), k.view(T, HKV, D), v.view(T, HKV, D)
            store_kv(k, v, *pool(layer), x["slots"])
            qkv.append((q, k, v))
        # one metadata object per backend, since each caches its per-step state in it, as in the engine
        meta = {b: AttnMetadata(is_prefill=True, slot_mapping=x["slots"], cu_seqlens=x["cu"], block_tables=x["bt"],
                                context_lens=torch.tensor(x["total"], dtype=torch.int32, device=device))
                for b in ("flex", "triton")}
        calls = {"triton": lambda l: triton_attn(*qkv[l], *pool(l), meta["triton"])}
        if not any(cached):
            calls["flex"] = lambda l: flex(*qkv[l], *pool(l), meta["flex"])
        diff = None
        if "flex" in calls:
            diff = (calls["flex"](0).float() - calls["triton"](0).float()).abs().max().item()
        times = {k: [] for k in calls}
        for _ in range(args.rounds):
            for k, call in calls.items():
                times[k].append(per_layer_ms(call, layers) * 1e3)
        flops = 4 * x["pairs"] * HQ * D
        row = {"shape": name, "cached": cached, "new": new, "new_tokens": T, "layers": layers, "flops": flops,
               "kv_bytes_per_layer": sum(x["total"]) * KV_BYTES_PER_TOKEN, "max_abs_diff_flex_vs_triton": diff}
        for k, ts in times.items():
            us = statistics.median(ts)
            row[k] = {"us_per_layer": us, "rounds_us": ts, "tflops": flops / (us * 1e-6) / 1e12}
        rows.append(row)
        line = f"{name:28s} " + " | ".join(f"{k} {row[k]['us_per_layer']:8.1f} us ({row[k]['tflops']:4.1f} TFLOPS)" for k in calls)
        if "flex" in calls:
            line += f" | flex / triton {row['flex']['us_per_layer'] / row['triton']['us_per_layer']:.2f}x | max diff {diff:.4f}"
        print(line, flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"matmul_tflops": peak, "heads": [HQ, HKV, D], "page": PAGE, "gpu": torch.cuda.get_device_name(),
                   "torch": torch.__version__, "shapes": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
