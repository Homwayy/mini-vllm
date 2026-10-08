"""Fused Triton kernels for the elementwise parts of a decoder layer (E9).

The models' reference forward runs these parts as chains of PyTorch ops, one small kernel each: a Qwen3
RMSNorm alone is 8 kernels, GPT-2's RoPE 10 for q and 10 for k. Each kernel here runs one chain in a single
pass over a token's row, with the reference's roundings: a value is rounded to bf16 wherever the reference
keeps a bf16 tensor. Only the order of the fp32 sums in the norms differs, so now and then a value lands on
the other side of a bf16 rounding boundary (tests/test_fused.py).

GPT-2 (fp32 residual stream, LayerNorm, RoPE in fp32):
  add_layer_norm: x += branch (bf16, promoted to fp32), then LayerNorm(x) in the requested dtype.
  rope_store: RoPE of q and k in fp32, rounded to bf16; k and v written into the KV pool.
Qwen3 (bf16 residual stream, RMSNorm, RoPE in bf16):
  add_rms_norm: x = bf16(x + branch), then weight * bf16(x * rsqrt(mean(x^2) + eps)).
  qk_norm_rope_store: RMSNorm of every q and k head, then RoPE; k and v written into the KV pool.
  silu_mul: bf16(silu(gate)) * up.

The residual stream x is updated in place, so a residual add costs no kernel of its own. One program
handles one token, all heads at once. k and v also come back as tensors only when the attention backend
reads them (need_kv); the Triton backend reads them from the pool.

Where the reference is a chain of separate ops, each op's result is rounded before the next, so those
kernels compile with enable_fp_fusion=False: otherwise LLVM turns bf16(a * c) - bf16(b * s) into one bf16
FMA, which skips a rounding. The reference LayerNorm is a single CUDA kernel that uses FMAs itself, so
add_layer_norm keeps them.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

NO_FMA = dict(enable_fp_fusion=False)


@triton.jit
def _bf16(x):
    # round an fp32 value to bf16 and back, where the reference stores a bf16 tensor
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _add_layer_norm_kernel(x_ptr, b_ptr, w_ptr, bias_ptr, y_ptr, n_cols, eps, HAS_BRANCH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * n_cols
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row + cols, mask=mask, other=0.0)
    if HAS_BRANCH:
        x += tl.load(b_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(x_ptr + row + cols, x, mask=mask)
    mean = tl.sum(x, 0) / n_cols
    d = tl.where(mask, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(d * d, 0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row + cols, (d * rstd * w + bias).to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rope_store_kernel(qkv_ptr, pos_ptr, cos_ptr, sin_ptr, slot_ptr, q_ptr, k_ptr, v_ptr, kc_ptr, vc_ptr, n_heads,
                       NEED_KV: tl.constexpr, H_PAD: tl.constexpr, HALF: tl.constexpr):
    # GPT-2: qkv row [3, H, D]; dimension i pairs with i + D/2; cos, sin [positions, D/2] fp32
    t = tl.program_id(0).to(tl.int64)
    D: tl.constexpr = 2 * HALF
    h = tl.arange(0, H_PAD)[:, None]
    i = tl.arange(0, HALF)[None, :]
    mask = h < n_heads
    head = h * D + i  # first halves of the heads in a [H, D] row
    width = n_heads * D
    row = qkv_ptr + t * 3 * width
    pos = tl.load(pos_ptr + t)
    c = tl.load(cos_ptr + pos * HALF + i)
    s = tl.load(sin_ptr + pos * HALF + i)

    x1 = tl.load(row + head, mask=mask).to(tl.float32)
    x2 = tl.load(row + head + HALF, mask=mask).to(tl.float32)
    tl.store(q_ptr + t * width + head, (x1 * c - x2 * s).to(tl.bfloat16), mask=mask)
    tl.store(q_ptr + t * width + head + HALF, (x1 * s + x2 * c).to(tl.bfloat16), mask=mask)

    x1 = tl.load(row + width + head, mask=mask).to(tl.float32)
    x2 = tl.load(row + width + head + HALF, mask=mask).to(tl.float32)
    k1 = (x1 * c - x2 * s).to(tl.bfloat16)
    k2 = (x1 * s + x2 * c).to(tl.bfloat16)
    v1 = tl.load(row + 2 * width + head, mask=mask)
    v2 = tl.load(row + 2 * width + head + HALF, mask=mask)
    slot = tl.load(slot_ptr + t) * width
    tl.store(kc_ptr + slot + head, k1, mask=mask)
    tl.store(kc_ptr + slot + head + HALF, k2, mask=mask)
    tl.store(vc_ptr + slot + head, v1, mask=mask)
    tl.store(vc_ptr + slot + head + HALF, v2, mask=mask)
    if NEED_KV:
        tl.store(k_ptr + t * width + head, k1, mask=mask)
        tl.store(k_ptr + t * width + head + HALF, k2, mask=mask)
        tl.store(v_ptr + t * width + head, v1, mask=mask)
        tl.store(v_ptr + t * width + head + HALF, v2, mask=mask)


@triton.jit
def _add_rms_norm_kernel(x_ptr, b_ptr, w_ptr, y_ptr, n_cols, eps, HAS_BRANCH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * n_cols
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BRANCH:
        x = (x + tl.load(b_ptr + row + cols, mask=mask, other=0.0).to(tl.float32)).to(tl.bfloat16)
        tl.store(x_ptr + row + cols, x, mask=mask)
        x = x.to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / n_cols + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(y_ptr + row + cols, (w * _bf16(x * rstd)).to(tl.bfloat16), mask=mask)


@triton.jit
def _rms_norm_rope(ptrs, mask, w_ptr, c1, c2, s1, s2, eps, HALF: tl.constexpr):
    # Qwen3: RMSNorm of each head (a row of ptrs), then RoPE in bf16 like the reference,
    # x * cos + cat(-x2, x1) * sin with each product rounded; returns both halves
    i = tl.arange(0, HALF)[None, :]
    x1 = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(ptrs + HALF, mask=mask, other=0.0).to(tl.float32)
    rstd = tl.rsqrt((tl.sum(x1 * x1, 1) + tl.sum(x2 * x2, 1)) / (2 * HALF) + eps)[:, None]
    n1 = _bf16(tl.load(w_ptr + i).to(tl.float32) * _bf16(x1 * rstd))
    n2 = _bf16(tl.load(w_ptr + HALF + i).to(tl.float32) * _bf16(x2 * rstd))
    o1 = (_bf16(n1 * c1) - _bf16(n2 * s1)).to(tl.bfloat16)
    o2 = (_bf16(n2 * c2) + _bf16(n1 * s2)).to(tl.bfloat16)
    return o1, o2


@triton.jit
def _qk_norm_rope_store_kernel(qkv_ptr, qw_ptr, kw_ptr, pos_ptr, cos_ptr, sin_ptr, slot_ptr, q_ptr, k_ptr, v_ptr,
                               kc_ptr, vc_ptr, n_heads, n_kv_heads, eps, NEED_KV: tl.constexpr, H_PAD: tl.constexpr,
                               HKV_PAD: tl.constexpr, HALF: tl.constexpr):
    # Qwen3: qkv row [H * D | Hkv * D | Hkv * D]; cos, sin [positions, D] bf16
    t = tl.program_id(0).to(tl.int64)
    D: tl.constexpr = 2 * HALF
    i = tl.arange(0, HALF)[None, :]
    row = qkv_ptr + t * (n_heads + 2 * n_kv_heads) * D
    pos = tl.load(pos_ptr + t)
    c1 = tl.load(cos_ptr + pos * D + i).to(tl.float32)
    c2 = tl.load(cos_ptr + pos * D + HALF + i).to(tl.float32)
    s1 = tl.load(sin_ptr + pos * D + i).to(tl.float32)
    s2 = tl.load(sin_ptr + pos * D + HALF + i).to(tl.float32)

    hq = tl.arange(0, H_PAD)[:, None]
    mq = hq < n_heads
    o1, o2 = _rms_norm_rope(row + hq * D + i, mq, qw_ptr, c1, c2, s1, s2, eps, HALF)
    q_row = q_ptr + t * n_heads * D + hq * D + i
    tl.store(q_row, o1, mask=mq)
    tl.store(q_row + HALF, o2, mask=mq)

    hk = tl.arange(0, HKV_PAD)[:, None]
    mk = hk < n_kv_heads
    head = hk * D + i
    k1, k2 = _rms_norm_rope(row + n_heads * D + head, mk, kw_ptr, c1, c2, s1, s2, eps, HALF)
    v_src = row + (n_heads + n_kv_heads) * D + head
    v1 = tl.load(v_src, mask=mk)
    v2 = tl.load(v_src + HALF, mask=mk)
    slot = tl.load(slot_ptr + t) * n_kv_heads * D
    tl.store(kc_ptr + slot + head, k1, mask=mk)
    tl.store(kc_ptr + slot + head + HALF, k2, mask=mk)
    tl.store(vc_ptr + slot + head, v1, mask=mk)
    tl.store(vc_ptr + slot + head + HALF, v2, mask=mk)
    if NEED_KV:
        kv_row = t * n_kv_heads * D + head
        tl.store(k_ptr + kv_row, k1, mask=mk)
        tl.store(k_ptr + kv_row + HALF, k2, mask=mk)
        tl.store(v_ptr + kv_row, v1, mask=mk)
        tl.store(v_ptr + kv_row + HALF, v2, mask=mk)


@triton.jit
def _silu_mul_kernel(x_ptr, y_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    g = tl.load(x_ptr + row * 2 * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(x_ptr + row * 2 * n_cols + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    # the exact exp and division, as PyTorch's silu computes x / (1 + exp(-x))
    s = _bf16(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)))
    tl.store(y_ptr + row * n_cols + cols, (s * u).to(tl.bfloat16), mask=mask)


def add_layer_norm(x, branch, ln, dtype):
    """GPT-2. x [T, N] fp32: x += branch (bf16, if not None), in place; returns ln(x) in dtype."""
    T, N = x.shape
    y = torch.empty(T, N, dtype=dtype, device=x.device)
    _add_layer_norm_kernel[(T,)](x, x if branch is None else branch, ln.weight, ln.bias, y, N, ln.eps,
                                 HAS_BRANCH=branch is not None, BLOCK=triton.next_power_of_2(N), num_warps=4)
    return y


def rope_store(qkv, n_heads, positions, cos, sin, k_cache, v_cache, slot_mapping, need_kv):
    """GPT-2. qkv [T, 3 * H * D] bf16; cos, sin [positions, D / 2] fp32 tables. Writes k and v into the pools;
    returns q [T, H, D] and, if need_kv, k and v (else None)."""
    T, D = qkv.shape[0], k_cache.shape[-1]
    q = torch.empty(T, n_heads, D, dtype=qkv.dtype, device=qkv.device)
    k, v = (torch.empty_like(q), torch.empty_like(q)) if need_kv else (None, None)
    # without need_kv, q stands in for the k and v outputs, which the kernel then never writes
    _rope_store_kernel[(T,)](qkv, positions, cos, sin, slot_mapping, q, k if need_kv else q, v if need_kv else q,
                             k_cache, v_cache, n_heads, NEED_KV=need_kv, H_PAD=triton.next_power_of_2(n_heads), HALF=D // 2,
                             num_warps=4, **NO_FMA)
    return q, k, v


def add_rms_norm(x, branch, norm):
    """Qwen3. x [T, N] bf16: x = bf16(x + branch) (if branch is not None), in place; returns norm(x)."""
    T, N = x.shape
    y = torch.empty_like(x)
    _add_rms_norm_kernel[(T,)](x, x if branch is None else branch, norm.weight, y, N, norm.eps,
                               HAS_BRANCH=branch is not None, BLOCK=triton.next_power_of_2(N), num_warps=4, **NO_FMA)
    return y


def qk_norm_rope_store(qkv, n_heads, n_kv_heads, q_norm, k_norm, positions, cos, sin, k_cache, v_cache,
                       slot_mapping, need_kv):
    """Qwen3. qkv [T, (H + 2 Hkv) * D] bf16; cos, sin [positions, D] bf16 tables. Writes k and v into the pools;
    returns q [T, H, D] and, if need_kv, k and v [T, Hkv, D] (else None)."""
    T, D = qkv.shape[0], k_cache.shape[-1]
    q = torch.empty(T, n_heads, D, dtype=qkv.dtype, device=qkv.device)
    k, v = None, None
    if need_kv:
        k = torch.empty(T, n_kv_heads, D, dtype=qkv.dtype, device=qkv.device)
        v = torch.empty_like(k)
    _qk_norm_rope_store_kernel[(T,)](
        qkv, q_norm.weight, k_norm.weight, positions, cos, sin, slot_mapping, q, q if k is None else k,
        q if v is None else v, k_cache, v_cache,
        n_heads, n_kv_heads, q_norm.eps, NEED_KV=need_kv, H_PAD=triton.next_power_of_2(n_heads),
        HKV_PAD=triton.next_power_of_2(n_kv_heads), HALF=D // 2, num_warps=4, **NO_FMA)
    return q, k, v


def silu_mul(gate_up):
    """Qwen3. gate_up [T, 2 I] bf16 -> bf16(silu(gate)) * up, [T, I]."""
    T, n = gate_up.shape[0], gate_up.shape[1] // 2
    y = torch.empty(T, n, dtype=gate_up.dtype, device=gate_up.device)
    block = 1024
    _silu_mul_kernel[(T, triton.cdiv(n, block))](gate_up, y, n, BLOCK=block, num_warps=4)
    return y
