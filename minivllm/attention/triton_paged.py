"""Paged attention in Triton.

Decode (one query token per sequence) splits each sequence's context into segments (FlashDecoding).
Stage 1 runs one program per (sequence, KV head, segment), reads K/V straight from the pool through the
block table, and writes the segment's normalized output and log-sum-exp. Stage 2 merges the segments.
The query heads that share a KV head are handled by one program, so each K/V tile is read once per group;
they are padded to 16 rows because tl.dot needs at least 16. Decode is bound by reading K/V, so the
padding costs compute but not time.

The grid only depends on the batch size and the maximum context length, so a decode step that uses these
kernels can be captured in a CUDA graph; segments past a sequence's end write an empty result.

Prefill handles any number of new tokens per sequence. The new tokens are the last ones of the sequence,
their K/V is already in the pool, and the tokens before them can be a cached prefix; each new token
attends causally to all of it. One program computes a tile of new tokens of one sequence for one KV head,
with the tile's tokens times the heads of the group as rows.
"""

import math

import torch
import triton
import triton.language as tl

from minivllm.attention import AttnMetadata
from minivllm.attention.flex import FlexAttention

LOG2E = 1.4426950408889634


@triton.jit
def _decode_segment_kernel(
    Q, K, V, block_tables, context_lens, out_part, lse_part, qk_scale,
    stride_qb, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_btb,
    stride_pb, stride_ph, stride_ps,
    stride_lb, stride_lh,
    GROUP: tl.constexpr, BLOCK_H: tl.constexpr, HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_N: tl.constexpr, SEGMENT: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    seg = tl.program_id(2)
    ctx = tl.load(context_lens + b)
    start = seg * SEGMENT
    end = tl.minimum(start + SEGMENT, ctx)

    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_n = tl.arange(0, BLOCK_N)
    heads = kvh * GROUP + offs_h
    h_mask = offs_h < GROUP

    q = tl.load(Q + b * stride_qb + heads[:, None] * stride_qh + offs_d[None, :], mask=h_mask[:, None], other=0.0)
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, HEAD_DIM], dtype=tl.float32)
    for n0 in range(start, end, BLOCK_N):
        tok = n0 + offs_n
        n_mask = tok < end
        blk = tl.load(block_tables + b * stride_btb + tok // PAGE, mask=n_mask, other=0).to(tl.int64)
        slot = (tok % PAGE)[:, None]
        k = tl.load(K + blk[:, None] * stride_kb + slot * stride_kt + kvh * stride_kh + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)
        v = tl.load(V + blk[:, None] * stride_vb + slot * stride_vt + kvh * stride_vh + offs_d[None, :],
                    mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)) * qk_scale  # scores in log2 units
        s = tl.where(n_mask[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # an empty segment stores zeros and lse = -inf, so it gets zero weight when merged
    out = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    lse = tl.where(l_i > 0, m_i + tl.log2(l_i), float("-inf"))
    tl.store(out_part + b * stride_pb + heads[:, None] * stride_ph + seg * stride_ps + offs_d[None, :], out,
             mask=h_mask[:, None])
    tl.store(lse_part + b * stride_lb + heads * stride_lh + seg, lse, mask=h_mask)


@triton.jit
def _merge_segments_kernel(
    out_part, lse_part, Out,
    stride_pb, stride_ph, stride_ps,
    stride_lb, stride_lh,
    stride_ob, stride_oh,
    NUM_SEGMENTS: tl.constexpr, SEGMENTS_POW2: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, SEGMENTS_POW2)
    offs_d = tl.arange(0, HEAD_DIM)
    s_mask = offs_s < NUM_SEGMENTS
    lse = tl.load(lse_part + b * stride_lb + h * stride_lh + offs_s, mask=s_mask, other=float("-inf"))
    w = tl.exp2(lse - tl.max(lse, 0))  # segment 0 always has tokens, so the max is finite
    part = tl.load(out_part + b * stride_pb + h * stride_ph + offs_s[:, None] * stride_ps + offs_d[None, :],
                   mask=s_mask[:, None], other=0.0)
    out = tl.sum(part * w[:, None], 0) / tl.sum(w, 0)
    tl.store(Out + b * stride_ob + h * stride_oh + offs_d, out.to(Out.dtype.element_ty))


@triton.jit
def _attend_block(
    acc, m_i, l_i, q, K, V, bt, n0, kv_end, pos, kvh, qk_scale,
    stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh,
    HEAD_DIM: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr, MASKED: tl.constexpr,
):
    """Online softmax over keys n0 .. n0 + BLOCK_N for a tile of query rows at positions `pos`.

    MASKED applies the causal mask and skips keys from kv_end on; without it every key must be visible.
    """
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    tok = n0 + offs_n
    if MASKED:
        n_mask = tok < kv_end
        blk = tl.load(bt + tok // PAGE, mask=n_mask, other=0).to(tl.int64)
    else:
        blk = tl.load(bt + tok // PAGE).to(tl.int64)
    slot = (tok % PAGE)[:, None]
    k_ptr = K + blk[:, None] * stride_kb + slot * stride_kt + kvh * stride_kh + offs_d[None, :]
    v_ptr = V + blk[:, None] * stride_vb + slot * stride_vt + kvh * stride_vh + offs_d[None, :]
    if MASKED:
        k = tl.load(k_ptr, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptr, mask=n_mask[:, None], other=0.0)
    else:
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)
    s = tl.dot(q, tl.trans(k)) * qk_scale  # scores in log2 units
    if MASKED:
        s = tl.where(tok[None, :] <= pos[:, None], s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return acc, m_new, l_i


@triton.jit
def _prefill_kernel(
    Q, K, V, Out, block_tables, cu_seqlens, seq_lens, qk_scale, num_seqs,
    stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_vb, stride_vt, stride_vh,
    stride_ot, stride_oh,
    stride_btb,
    GROUP: tl.constexpr, GROUP_PAD: tl.constexpr, HEAD_DIM: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    BLOCK_Q: tl.constexpr = BLOCK_M // GROUP_PAD  # new tokens per tile
    # the last tiles of a sequence see the most keys, so start from the end of the grid
    tile = tl.num_programs(0) - 1 - tl.program_id(0)
    kvh = tl.program_id(1)

    # sequence b owns the tiles from cu_seqlens[b] // BLOCK_Q + b on, so the grid only depends on T and B
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(cu_seqlens + mid) // BLOCK_Q + mid <= tile:
            lo = mid + 1
        else:
            hi = mid
    b = lo - 1
    q_start = tl.load(cu_seqlens + b)
    q_len = tl.load(cu_seqlens + b + 1) - q_start
    q0 = (tile - q_start // BLOCK_Q - b) * BLOCK_Q  # first new token of this tile, counted within the sequence
    if q0 >= q_len:
        return
    first_pos = tl.load(seq_lens + b) - q_len  # position of the sequence's first new token

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_idx = q0 + offs_m // GROUP_PAD
    head = kvh * GROUP + offs_m % GROUP_PAD
    row_mask = (q_idx < q_len) & (offs_m % GROUP_PAD < GROUP)
    pos = first_pos + q_idx
    rows = (q_start + q_idx).to(tl.int64)
    q = tl.load(Q + rows[:, None] * stride_qt + head[:, None] * stride_qh + offs_d[None, :],
                mask=row_mask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    bt = block_tables + b * stride_btb
    kv_end = first_pos + tl.minimum(q0 + BLOCK_Q, q_len)  # keys from here on are masked for every row
    full_end = (first_pos + q0) // BLOCK_N * BLOCK_N  # keys before this are visible to every row
    for n0 in range(0, full_end, BLOCK_N):
        acc, m_i, l_i = _attend_block(
            acc, m_i, l_i, q, K, V, bt, n0, kv_end, pos, kvh, qk_scale,
            stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh,
            HEAD_DIM, PAGE, BLOCK_N, False,
        )
    for n0 in range(full_end, kv_end, BLOCK_N):
        acc, m_i, l_i = _attend_block(
            acc, m_i, l_i, q, K, V, bt, n0, kv_end, pos, kvh, qk_scale,
            stride_kb, stride_kt, stride_kh, stride_vb, stride_vt, stride_vh,
            HEAD_DIM, PAGE, BLOCK_N, True,
        )
    out = acc / l_i[:, None]
    tl.store(Out + rows[:, None] * stride_ot + head[:, None] * stride_oh + offs_d[None, :],
             out.to(Out.dtype.element_ty), mask=row_mask[:, None])


def segment_tokens(batch_size, num_kv_heads, block_n, max_context):
    """Tokens per segment: about 128 programs for 1024-token contexts, so small batches still fill the GPU.

    Tuned on Qwen3 shapes (8 KV heads, head_dim 128) for batch 1-128: this rule was within 5% of the best
    segment length at every shape tested.
    """
    seg = triton.next_power_of_2(8 * batch_size * num_kv_heads)
    return min(max(seg, block_n), triton.next_power_of_2(max_context))


class PagedDecode:
    """Scratch buffers per batch size, so the decode step keeps fixed addresses inside a CUDA graph."""

    def __init__(self, block_n=32, num_warps=4, num_stages=2, segment=None):
        self.block_n, self.num_warps, self.num_stages = block_n, num_warps, num_stages
        self.segment = segment  # tokens per segment; None picks it from the batch size
        self.scratch = {}

    def __call__(self, q, k_cache, v_cache, block_tables, context_lens):
        B, Hq, D = q.shape
        _, page, Hkv, _ = k_cache.shape
        group = Hq // Hkv
        max_context = block_tables.shape[1] * page
        segment = self.segment or segment_tokens(B, Hkv, self.block_n, max_context)
        num_segments = triton.cdiv(max_context, segment)
        key = (B, Hq, D, num_segments)
        if key not in self.scratch:
            self.scratch[key] = (
                torch.empty(B, Hq, num_segments, D, dtype=torch.float32, device=q.device),
                torch.empty(B, Hq, num_segments, dtype=torch.float32, device=q.device),
                torch.empty(B, Hq, D, dtype=q.dtype, device=q.device),
            )
        out_part, lse_part, out = self.scratch[key]
        _decode_segment_kernel[(B, Hkv, num_segments)](
            q, k_cache, v_cache, block_tables, context_lens, out_part, lse_part, LOG2E / math.sqrt(D),
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            block_tables.stride(0),
            out_part.stride(0), out_part.stride(1), out_part.stride(2),
            lse_part.stride(0), lse_part.stride(1),
            GROUP=group, BLOCK_H=max(16, triton.next_power_of_2(group)), HEAD_DIM=D,
            PAGE=page, BLOCK_N=self.block_n, SEGMENT=segment,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        _merge_segments_kernel[(B, Hq)](
            out_part, lse_part, out,
            out_part.stride(0), out_part.stride(1), out_part.stride(2),
            lse_part.stride(0), lse_part.stride(1),
            out.stride(0), out.stride(1),
            NUM_SEGMENTS=num_segments, SEGMENTS_POW2=triton.next_power_of_2(num_segments), HEAD_DIM=D,
        )
        return out


def prefill_block_m(max_new_tokens, group_pad):
    """Rows per tile: one sequence's new tokens times the group's heads, between 16 and 128.

    Tuned on Qwen3 shapes: 128 rows for long prompts, while 4-16 new tokens per sequence (as when
    verifying draft tokens) are 35-60% faster with 16-32 rows than with 128.
    """
    return min(128, max(16, triton.next_power_of_2(max_new_tokens * group_pad)))


class PagedPrefill:
    def __init__(self, block_n=32, num_warps=4, num_stages=2, block_m=None):
        self.block_n, self.num_warps, self.num_stages = block_n, num_warps, num_stages
        self.block_m = block_m  # None picks it from the most new tokens of any sequence

    def __call__(self, q, k_cache, v_cache, block_tables, cu_seqlens, seq_lens, max_new_tokens):
        """q [T, Hq, D]: the new tokens of all sequences, packed; their K/V must already be in the pools.

        cu_seqlens [B + 1] int32 holds the prefix sums of the new-token counts, seq_lens [B] int32 each
        sequence's length including its new tokens.
        """
        T, Hq, D = q.shape
        _, page, Hkv, _ = k_cache.shape
        group = Hq // Hkv
        group_pad = triton.next_power_of_2(group)
        block_m = self.block_m or prefill_block_m(max_new_tokens, group_pad)
        B = seq_lens.shape[0]
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        _prefill_kernel[(T // (block_m // group_pad) + B, Hkv)](
            q, k_cache, v_cache, out, block_tables, cu_seqlens, seq_lens, LOG2E / math.sqrt(D), B,
            q.stride(0), q.stride(1),
            k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
            v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
            out.stride(0), out.stride(1),
            block_tables.stride(0),
            GROUP=group, GROUP_PAD=group_pad, HEAD_DIM=D,
            PAGE=page, BLOCK_M=block_m, BLOCK_N=self.block_n,
            num_warps=self.num_warps, num_stages=self.num_stages,
        )
        return out


class TritonAttention:
    """Prefill and decode with the Triton paged kernels.

    With flex_prefill, prefill uses FlexAttention over the new tokens instead (the setup measured in E4).
    """

    def __init__(self, block_size, max_num_batched_tokens, max_num_seqs, device, flex_prefill=False):
        self.flex = FlexAttention(block_size, max_num_batched_tokens, max_num_seqs, device) if flex_prefill else None
        self.prefill = PagedPrefill()
        self.decode = PagedDecode()

    def __call__(self, q, k, v, k_cache, v_cache, meta: AttnMetadata):
        if meta.is_prefill:
            if self.flex is not None:
                return self.flex(q, k, v, k_cache, v_cache, meta)
            if meta.backend_state is None:
                cu = meta.cu_seqlens
                max_new = max(b - a for a, b in zip(cu[:-1], cu[1:]))
                meta.backend_state = (torch.tensor(cu, dtype=torch.int32, device=q.device), max_new)
            cu, max_new = meta.backend_state
            return self.prefill(q, k_cache, v_cache, meta.block_tables, cu, meta.context_lens, max_new)
        return self.decode(q, k_cache, v_cache, meta.block_tables, meta.context_lens)
