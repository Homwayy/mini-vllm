"""Drawing one draft token per row in one kernel (E8c.1), for EAGLE's draft chain and DSpark's block.

A sampled row draws from q = softmax(logits / temperature) over the draft vocabulary and keeps q for the
verification; a greedy row takes the argmax and keeps nothing (verification reads q only for sampled rows).
DSpark adds its Markov head's term to the logits in the same pass.

One program per row reads the logits twice: once for the softmax's running max and sum, then to write q and
draw the token by the exponential race, as rejection.py does. Random numbers come from Triton's Philox, four
per call (most of the time of a sampled row went to Philox), keyed by a seed in GPU memory that the caller
redraws with PyTorch's generator before each drafting, so this works inside CUDA graphs and torch.manual_seed
fixes it.
"""

import torch
import triton
import triton.language as tl

# (columns per iteration, warps) for few rows and for many: a whole draft-vocabulary row at once is quickest
# for few rows, smaller tiles for many (a sweep at 1, 8 and 64 rows of 16,384 columns)
CONFIGS = {True: (16384, 16), False: (4096, 8)}
FEW_ROWS = 16


@triton.jit(do_not_specialize=["step"])
def _draft_sample_kernel(logits_ptr, logits_stride, bias_ptr, bias_stride, temp_ptr, seed_ptr, vocab_ptr, n_cols, step,
                         tok_ptr, tok_stride, prev_ptr, q_ptr, q_stride, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    base = logits_ptr + r.to(tl.int64) * logits_stride
    bbase = bias_ptr + r.to(tl.int64) * bias_stride
    t = tl.load(temp_ptr + r)
    if t > 0:
        m = tl.zeros([], tl.float32) - float("inf")  # running max and sum of exp(x / t)
        s = tl.zeros([], tl.float32)
        for c0 in range(0, n_cols, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)
            if HAS_BIAS:
                x += tl.load(bbase + cols, mask=mask, other=0.0).to(tl.float32)
            y = x / t
            m_new = tl.maximum(m, tl.max(y, 0))
            s = s * tl.exp(m - m_new) + tl.sum(tl.exp(y - m_new), 0)
            m = m_new
        seed = tl.load(seed_ptr)
        qbase = q_ptr + r.to(tl.int64) * q_stride
        best = tl.zeros([], tl.float32) + float("inf")
        pick = tl.zeros([], tl.int32)
        rbase = (step * tl.num_programs(0) + r) * (n_cols // 4 + BLOCK // 4)  # this row's Philox counters
        for c0 in range(0, n_cols, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)
            if HAS_BIAS:
                x += tl.load(bbase + cols, mask=mask, other=0.0).to(tl.float32)
            q = tl.exp(x / t - m) / s
            tl.store(qbase + cols, q, mask=mask)
            # the exponential race: the argmin of E / q, with E ~ Exponential(1), is a token drawn from q
            u0, u1, u2, u3 = tl.rand4x(seed, (rbase + c0 // 4 + tl.arange(0, BLOCK // 4)).to(tl.int32))
            e = -tl.log(1.0 - tl.reshape(tl.join(tl.join(u0, u1), tl.join(u2, u3)), [BLOCK]))
            key = tl.where(q > 0, e / q, float("inf"))
            bk = tl.min(key, 0)
            pick = tl.where(bk < best, c0 + tl.argmin(key, 0), pick)
            best = tl.minimum(best, bk)
    else:
        best = tl.zeros([], tl.float32) - float("inf")  # the argmax, the first one on ties
        pick = tl.zeros([], tl.int32)
        for c0 in range(0, n_cols, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            mask = cols < n_cols
            x = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)
            if HAS_BIAS:
                x += tl.load(bbase + cols, mask=mask, other=0.0).to(tl.float32)
            bx = tl.max(x, 0)
            pick = tl.where(bx > best, c0 + tl.argmax(x, 0), pick)
            best = tl.maximum(best, bx)
    tok = tl.load(vocab_ptr + pick)
    tl.store(tok_ptr + r * tok_stride, tok)
    tl.store(prev_ptr + r, tok)


def sample_drafts(logits, temp, seed, vocab_ids, tokens, prev, probs, step, bias=None):
    """Draws the step-th draft of each row of logits [R, V] (rows may be strided; columns contiguous), plus bias
    [R, V] if given, at temperature temp [R] (greedy if <= 0). Writes the token ids (vocab_ids of the picks) to
    tokens [R] and prev [R], and q to probs [R, V] for the sampled rows. seed: int64 [1] on the GPU; step must be
    below 8 and differ between the calls that share a seed."""
    n_rows, n_cols = logits.shape
    has_bias = bias is not None
    assert logits.stride(1) == 1 and probs.stride(1) == 1 and (not has_bias or bias.stride(1) == 1)
    block, warps = CONFIGS[n_rows <= FEW_ROWS]
    assert step < 8 and 8 * n_rows * (n_cols // 4 + block) < 2**31  # Philox counters stay in int32
    _draft_sample_kernel[(n_rows,)](
        logits, logits.stride(0), bias if has_bias else logits, bias.stride(0) if has_bias else 0, temp, seed, vocab_ids,
        n_cols, step, tokens, tokens.stride(0), prev, probs, probs.stride(0),
        HAS_BIAS=has_bias, BLOCK=block, num_warps=warps)
