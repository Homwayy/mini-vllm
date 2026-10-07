"""Decode attention on its own (E4): time per layer and effective bandwidth of each backend.

Qwen3 shapes (16 query heads, 8 KV heads, head_dim 128, 16-token pages). Each sequence's blocks are
consecutive, as the engine's free list hands them out at the start of a run. A measurement runs the
backend on several layers inside one CUDA graph and divides the graph's time by the number of layers.
Each layer reads its own pool: a window of one large buffer with as many blocks as the engine's pool, so
flex sees the same KV length as in the engine. The windows overlap but the blocks the layers read do not,
and each shape uses enough layers that one replay reads at least 256 MiB of K/V, so K/V cannot be served
from L2 (36 MiB on this GPU) between repeats. The backends are measured in turns over several rounds, and
the median round is reported.

Backends: flex and triton as in the engine; flex_partial, flex with every page listed as a partial block
(see FlexPartial); gather, which pads every sequence to max_model_len (2048) as in the engine's graphs.

Run from the repo root: python -m bench.bench_attention --out results/e4_attention.json
"""

import argparse
import json
import math
import os
import random
import statistics

import torch
from torch.nn.attention.flex_attention import BlockMask

from minivllm.attention import AttnMetadata
from minivllm.attention.flex import SPARSE_BLOCK, FlexAttention
from minivllm.attention.gather import GatherAttention
from minivllm.attention.triton_paged import PagedDecode

PAGE, NUM_BLOCKS, MAX_LEN = 16, 4096, 2048
POOL_BLOCKS = NUM_BLOCKS + 1  # the engine's pool, scratch block included
HQ, HKV, D = 16, 8, 128
MIN_LAYERS, READ_PER_REPLAY = 8, 256 << 20
BLOCK_BYTES = PAGE * HKV * D * 2  # one page of K (or of V), bf16
KV_BYTES_PER_TOKEN = 2 * HKV * D * 2  # K and V of one layer


class FlexPartial(FlexAttention):
    """Flex decode with every page listed as a partial block; the mask then also checks for the last page.

    When the mask has full blocks, flex's decoding kernel splits the KV range by the length of the whole
    pool (65,552 slots), so a sequence much shorter than the pool falls into a single split and one program
    per (sequence, KV head) reads all of it. With partial blocks only, it splits each sequence's own pages,
    but evaluates the mask on every page.
    """

    def __init__(self, *args):
        super().__init__(*args)
        self.last_page = torch.zeros_like(self.last_len)  # block id of each sequence's last page
        last_len, last_page, P = self.last_len, self.last_page, self.block_size

        def partial_mask(b, h, q_idx, kv_idx):
            return (kv_idx // P != last_page[b]) | (kv_idx % P < last_len[b])

        self.partial_mask = partial_mask

    def decode_block_mask(self, block_tables, context_lens, num_blocks):
        P = self.block_size
        B, width = block_tables.shape
        if (B, width) not in self.decode_masks:
            kv_num = torch.zeros(B, 1, 1, dtype=torch.int32, device=block_tables.device)
            kv_idx = torch.zeros(B, 1, 1, width, dtype=torch.int32, device=block_tables.device)
            block_mask = BlockMask.from_kv_blocks(
                kv_num, kv_idx, BLOCK_SIZE=(SPARSE_BLOCK, P), mask_mod=self.partial_mask,
                seq_lengths=(1, num_blocks * P), compute_q_blocks=False,
            )
            self.decode_masks[B, width] = (block_mask, kv_num, kv_idx)
        block_mask, kv_num, kv_idx = self.decode_masks[B, width]
        npages = (context_lens + P - 1) // P
        self.last_len[:B] = context_lens - (npages - 1) * P
        self.last_page[:B] = block_tables.gather(1, (npages - 1).long().unsqueeze(1)).squeeze(1)
        kv_num.view(B).copy_(npages)
        kv_idx.view(B, width).copy_(block_tables)
        return block_mask


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
    return q, bt, ctx, nxt


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


def read_bandwidth_gbs(device):
    x = torch.empty(1 << 28, dtype=torch.float32, device=device).normal_()  # 1 GiB
    ms = per_layer_ms(lambda layer: x.sum(), MIN_LAYERS, iters=5)
    return x.numel() * 4 / (ms * 1e-3) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(0)
    torch._dynamo.config.recompile_limit = 128

    rng = random.Random(0)
    shapes = [(f"batch {b}, context 1024", [1024] * b) for b in (1, 2, 4, 8, 16, 32, 48, 64)]
    shapes += [(f"batch {b}, context 100-2000", [rng.randint(100, 2000) for _ in range(b)]) for b in (1, 8, 24, 48)]
    shapes += [(f"batch {b}, context 512", [512] * b) for b in (96, 128)]
    batches = []
    for name, lens in shapes:
        q, bt, ctx, nblocks = make_batch(lens, device)
        layers = max(MIN_LAYERS, math.ceil(READ_PER_REPLAY / (2 * nblocks * BLOCK_BYTES)))
        batches.append((name, lens, q, bt, ctx, nblocks, layers))
    total_blocks = max((layers - 1) * nblocks + POOL_BLOCKS for *_, nblocks, layers in batches)
    k_buf = torch.randn(total_blocks, PAGE, HKV, D, device=device, dtype=torch.bfloat16)
    v_buf = torch.randn(total_blocks, PAGE, HKV, D, device=device, dtype=torch.bfloat16)

    flex = FlexAttention(PAGE, 4096, 256, device)
    flex_partial = FlexPartial(PAGE, 4096, 256, device)
    gather = GatherAttention()
    triton_decode = PagedDecode()
    bandwidth = read_bandwidth_gbs(device)
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    print(f"read bandwidth (sum of a 1 GiB tensor): {bandwidth:.0f} GB/s, L2 {l2 / 2**20:.0f} MiB, "
          f"K/V buffer {2 * total_blocks * BLOCK_BYTES / 2**30:.2f} GiB")

    rows = []
    for name, lens, q, bt, ctx, nblocks, layers in batches:
        def pool(layer, nblocks=nblocks):
            first = layer * nblocks
            return k_buf[first:first + POOL_BLOCKS], v_buf[first:first + POOL_BLOCKS]

        # one metadata object per backend, since each caches its block mask in it, as in the engine
        meta = {k: AttnMetadata(is_prefill=False, slot_mapping=None, block_tables=bt, context_lens=ctx, max_context_len=MAX_LEN)
                for k in ("flex", "flex_partial", "gather")}
        calls = {
            "flex": lambda l: flex(q, None, None, *pool(l), meta["flex"]),
            "flex_partial": lambda l: flex_partial(q, None, None, *pool(l), meta["flex_partial"]),
            "triton": lambda l: triton_decode(q, *pool(l), bt, ctx),
            "gather": lambda l: gather(q, None, None, *pool(l), meta["gather"]),
        }
        ref = calls["triton"](1).float()
        diffs = {k: (call(1).float() - ref).abs().max().item() for k, call in calls.items()}
        times = {k: [] for k in calls}
        for _ in range(args.rounds):
            for k, call in calls.items():
                times[k].append(per_layer_ms(call, layers) * 1e3)
        kv_bytes = sum(lens) * KV_BYTES_PER_TOKEN
        row = {"shape": name, "batch": len(lens), "context_tokens": sum(lens), "kv_bytes_per_layer": kv_bytes, "layers": layers}
        for k, ts in times.items():
            us = statistics.median(ts)
            row[k] = {"us_per_layer": us, "rounds_us": ts, "effective_gbs": kv_bytes / (us * 1e-6) / 1e9,
                      "max_abs_diff_vs_triton": diffs[k]}
        rows.append(row)
        print(
            f"{name:28s} " + " | ".join(f"{k} {row[k]['us_per_layer']:7.1f} us ({row[k]['effective_gbs']:4.0f} GB/s)" for k in calls)
            + f" | flex / triton {row['flex']['us_per_layer'] / row['triton']['us_per_layer']:.2f}x"
            + f" | max diff {max(diffs.values()):.4f}",
            flush=True,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"read_bandwidth_gbs": bandwidth, "l2_bytes": l2, "read_per_replay_min_bytes": READ_PER_REPLAY,
                   "heads": [HQ, HKV, D], "page": PAGE, "pool_blocks": POOL_BLOCKS,
                   "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "shapes": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
