"""Paged attention in Triton.

Decode (one query token per sequence) splits each sequence's context into segments (FlashDecoding).
Stage 1 runs one program per (sequence, KV head, segment), reads K/V straight from the pool through the
block table, and writes the segment's normalized output and log-sum-exp. Stage 2 merges the segments.
The query heads that share a KV head are handled by one program, so each K/V tile is read once per group;
they are padded to 16 rows because tl.dot needs at least 16. Decode is bound by reading K/V, so the
padding costs compute but not time.

The grid only depends on the batch size and the maximum context length, so a decode step that uses these
kernels can be captured in a CUDA graph; segments past a sequence's end write an empty result.
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


class TritonAttention:
    """Decode with the Triton paged kernel. Prefill still uses FlexAttention."""

    def __init__(self, block_size, max_num_batched_tokens, max_num_seqs, device):
        self.flex = FlexAttention(block_size, max_num_batched_tokens, max_num_seqs, device)
        self.decode = PagedDecode()

    def __call__(self, q, k, v, k_cache, v_cache, meta: AttnMetadata):
        if meta.is_prefill:
            return self.flex(q, k, v, k_cache, v_cache, meta)
        return self.decode(q, k_cache, v_cache, meta.block_tables, meta.context_lens)
