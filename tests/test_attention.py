"""Attention backends against an fp32 per-sequence reference, on random data with varied shapes.

Covers both head layouts (Qwen3: 16 query / 8 KV heads of 128; GPT-2: 12 / 12 of 64), prefill with packed
sequences of edge-case lengths, prefill after a cached prefix (Triton only: the other backends attend to the
new tokens alone), and paged decode, all with shuffled block tables. Also counts how many graphs
torch.compile builds while the shapes keep changing.

Run from the repo root: python -m tests.test_attention
"""

import random

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from minivllm.attention import AttnMetadata, store_kv
from minivllm.attention.flex import FlexAttention
from minivllm.attention.gather import GatherAttention
from minivllm.attention.triton_paged import TritonAttention
from minivllm.model_runner import decode_buckets

BLOCK_SIZE = 16
NUM_BLOCKS = 8192
MAX_CTX = 2048
MAX_TOKENS = 8192
MAX_SEQS = 64
TOL = 2e-2
EDGE_LENS = [1, 2, 15, 16, 17, 127, 128, 129, 1000]


def ref_attention(q, k, v, causal):
    # q [Lq, Hq, D], k/v [Lk, Hkv, D] -> fp32 [Lq, Hq, D]; causal aligns the last query with the last key
    g = q.shape[1] // k.shape[1]
    q4 = q.float().transpose(0, 1).unsqueeze(0)
    k4 = k.float().repeat_interleave(g, 1).transpose(0, 1).unsqueeze(0)
    v4 = v.float().repeat_interleave(g, 1).transpose(0, 1).unsqueeze(0)
    mask = None
    if causal:
        Lq, Lk = q.shape[0], k.shape[0]
        mask = torch.arange(Lk, device=q.device)[None, :] <= torch.arange(Lk - Lq, Lk, device=q.device)[:, None]
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask)[0].transpose(0, 1)


def graph_count():
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def shuffled_tables(lengths, rng, device):
    """Block tables [B, max_blocks] that give each sequence randomly chosen pages of the pool."""
    npages = [(n + BLOCK_SIZE - 1) // BLOCK_SIZE for n in lengths]
    pages = rng.sample(range(NUM_BLOCKS), sum(npages))
    table = [[0] * (MAX_CTX // BLOCK_SIZE) for _ in lengths]
    for b, n in enumerate(npages):
        table[b][:n], pages = pages[:n], pages[n:]
    return torch.tensor(table, dtype=torch.int32, device=device)


def check_prefill(backends, Hq, Hkv, D, rng, rounds, device, cached):
    # the new tokens' K/V go into the pool first, as in the models; with `cached`, a prefix of each
    # sequence is already there
    k_cache = torch.randn(NUM_BLOCKS + 1, BLOCK_SIZE, Hkv, D, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    worst = {name: 0.0 for name in backends}
    for r in range(rounds):
        if r < len(EDGE_LENS):
            lens = [EDGE_LENS[r]] + rng.sample(EDGE_LENS, 3)
        else:
            lens = [rng.randint(1, 1024) for _ in range(rng.randint(1, 12))]
        prefix = [rng.choice([0, 1, 15, 16, 17, 100, 1000, rng.randint(0, 1024)]) if cached else 0 for _ in lens]
        total = [p + n for p, n in zip(prefix, lens)]
        T = sum(lens)
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        block_tables = shuffled_tables(total, rng, device)
        # strided views of one projection output, as the models produce them
        qkv = torch.randn(T, (Hq + 2 * Hkv) * D, device=device, dtype=torch.bfloat16)
        q, k, v = qkv.split([Hq * D, Hkv * D, Hkv * D], dim=-1)
        q, k, v = q.view(T, Hq, D), k.view(T, Hkv, D), v.view(T, Hkv, D)
        slots = torch.cat([
            block_tables[b].long()[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
            for b, pos in enumerate(torch.arange(p, t, device=device) for p, t in zip(prefix, total))
        ])
        store_kv(k, v, k_cache, v_cache, slots)
        ref = []
        for b in range(len(lens)):
            pos = torch.arange(total[b], device=device)
            ctx = block_tables[b].long()[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
            kb, vb = k_cache.view(-1, Hkv, D)[ctx], v_cache.view(-1, Hkv, D)[ctx]
            ref.append(ref_attention(q[cu[b] : cu[b + 1]], kb, vb, True))
        ref = torch.cat(ref)
        for name, attn in backends.items():
            meta = AttnMetadata(
                is_prefill=True, slot_mapping=slots, cu_seqlens=cu, block_tables=block_tables,
                context_lens=torch.tensor(total, dtype=torch.int32, device=device),
            )
            out = attn(q, k, v, k_cache, v_cache, meta)
            worst[name] = max(worst[name], (out.float() - ref).abs().max().item())
    return worst


def check_decode(backends, Hq, Hkv, D, rng, rounds, device):
    # the last block is the scratch block that padding rows use, as in the model runner
    k_cache = torch.randn(NUM_BLOCKS + 1, BLOCK_SIZE, Hkv, D, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    max_blocks = MAX_CTX // BLOCK_SIZE
    buckets = decode_buckets(MAX_SEQS)
    worst = {name: 0.0 for name in backends}
    for r in range(rounds):
        B = [1, 2, 3, 64][r] if r < 4 else rng.randint(1, MAX_SEQS)
        ctx = [rng.choice([1, 16, 17, MAX_CTX]) if rng.random() < 0.2 else rng.randint(1, MAX_CTX) for _ in range(B)]
        npages = [(n + BLOCK_SIZE - 1) // BLOCK_SIZE for n in ctx]
        pages = rng.sample(range(NUM_BLOCKS), sum(npages))
        padded = next(b for b in buckets if b >= B)
        table = [[0] * max_blocks for _ in range(padded)]
        for b, n in enumerate(npages):
            table[b][:n], pages = pages[:n], pages[n:]
        for b in range(B, padded):
            table[b][0] = NUM_BLOCKS
        block_tables = torch.tensor(table, dtype=torch.int32, device=device)
        context_lens = torch.tensor(ctx + [1] * (padded - B), dtype=torch.int32, device=device)
        q = torch.randn(padded, Hq, D, device=device, dtype=torch.bfloat16)
        ref = []
        for b in range(B):
            pos = torch.arange(ctx[b], device=device)
            slots = block_tables[b].long()[pos // BLOCK_SIZE] * BLOCK_SIZE + pos % BLOCK_SIZE
            kb = k_cache.view(-1, Hkv, D)[slots]
            vb = v_cache.view(-1, Hkv, D)[slots]
            ref.append(ref_attention(q[b : b + 1], kb, vb, False))
        ref = torch.cat(ref)
        for name, attn in backends.items():
            meta = AttnMetadata(
                is_prefill=False, slot_mapping=None, block_tables=block_tables, context_lens=context_lens,
                max_context_len=max(ctx),
            )
            out = attn(q, None, None, k_cache, v_cache, meta)[:B]
            worst[name] = max(worst[name], (out.float() - ref).abs().max().item())
    return worst


@torch.inference_mode()
def main():
    device = "cuda"
    torch.manual_seed(0)
    rng = random.Random(0)
    ok = True
    for Hq, Hkv, D in ((16, 8, 128), (12, 12, 64)):
        backends = {
            "gather": GatherAttention(),
            "flex": FlexAttention(BLOCK_SIZE, MAX_TOKENS, MAX_SEQS, device),
            "triton": TritonAttention(BLOCK_SIZE, MAX_TOKENS, MAX_SEQS, device),
        }
        g0 = graph_count()
        prefill = check_prefill(backends, Hq, Hkv, D, rng, 30, device, cached=False)
        g1 = graph_count()
        decode = check_decode(backends, Hq, Hkv, D, rng, 30, device)
        g2 = graph_count()
        prefix = check_prefill({"triton": backends["triton"]}, Hq, Hkv, D, rng, 30, device, cached=True)
        print(f"heads {Hq}/{Hkv}, head_dim {D}")
        for name in backends:
            line = f"  {name:6s} max abs err vs fp32: prefill {prefill[name]:.4f}  decode {decode[name]:.4f}"
            ok &= prefill[name] < TOL and decode[name] < TOL
            if name in prefix:
                line += f"  prefill after a cached prefix {prefix[name]:.4f}"
                ok &= prefix[name] < TOL
            print(line)
        print(f"  graphs compiled: prefill {g1 - g0} over 30 shapes, decode {g2 - g1} over 30 shapes")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
