"""Checking draft tokens against the target model (speculative sampling).

A draft token d is accepted with probability min(1, p(d) / q(d)), where p is the target distribution after
temperature, top-k and top-p and q the drafter's; after a rejection, the token comes from max(0, p - q)
renormalized. Each output token is then distributed exactly as without drafting. A deterministic drafter
such as n-gram puts all of q on d, so d is accepted with probability p(d) and a rejection samples from p
without d. Greedy rows accept d only if it is the argmax.

Two fused kernels decide a step, unless a sampled request uses top-k or top-p:
  - one program per row reads the row's logits once, for its log-normalizer at the row's temperature, its
    argmax and whether its draft is accepted (p(d) and q(d) are all that needs);
  - one program per sequence counts its accepted drafts and samples the token after them, from the one row
    where that token comes from: the first rejected row, or the last row if every draft was accepted.
So no row's full distribution is ever stored. Top-k and top-p need each row's sorted distribution, and take
the full path, which computes every row's p and q over the whole vocabulary.

Each step copies its results to the CPU once.
"""

import numpy as np
import torch
import triton
import triton.language as tl

from minivllm.sampler import Sampler
from minivllm.sequence import SamplingParams

MAX_ROWS = 16  # rows per sequence: drafts + 1
BLOCK, NUM_WARPS = 4096, 8  # columns per loop iteration; chosen by a sweep over 2048-8192 and 4-16 warps


@triton.jit(do_not_specialize=["seed"])
def _accept_kernel(logits_ptr, stride, n_cols, inv_t_ptr, nxt_ptr, drow_ptr, q_ptr, q_stride, inv_vocab_ptr, seed,
                   lse_ptr, amax_ptr, accept_ptr, HAS_Q: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = logits_ptr + row.to(tl.int64) * stride
    inv_t = tl.load(inv_t_ptr + row)  # 1 / temperature; 0 for a greedy row
    m = tl.zeros([], tl.float32) - float("inf")  # running max and sum of exp(logit * inv_t)
    s = tl.zeros([], tl.float32)
    best = tl.zeros([], tl.float32) - float("inf")  # running argmax of the logits, the first one on ties
    best_i = tl.zeros([], tl.int32)
    for start in range(0, n_cols, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32)
        bx = tl.max(x, 0)
        best_i = tl.where(bx > best, start + tl.argmax(x, 0), best_i)
        best = tl.maximum(best, bx)
        y = tl.where(mask, x * inv_t, -float("inf"))
        m_new = tl.maximum(m, tl.max(y, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(y - m_new), 0)
        m = m_new
    lse = m + tl.log(s)

    d = tl.load(nxt_ptr + row)  # the draft this row checks; -1 on a sequence's last row
    dc = tl.maximum(d, 0)
    p_d = tl.exp(tl.load(base + dc).to(tl.float32) * inv_t - lse)
    if HAS_Q:
        k = tl.maximum(tl.load(drow_ptr + row), 0)
        q_d = tl.load(q_ptr + k.to(tl.int64) * q_stride + tl.maximum(tl.load(inv_vocab_ptr + dc), 0))
    else:
        q_d = 1.0
    ok = tl.where(inv_t > 0, tl.rand(seed, row) * q_d < p_d, best_i == d)
    tl.store(lse_ptr + row, lse)
    tl.store(amax_ptr + row, best_i)
    tl.store(accept_ptr + row, ((d >= 0) & ok).to(tl.int8))


@triton.jit(do_not_specialize=["seed"])
def _sample_kernel(logits_ptr, stride, n_cols, inv_t_ptr, nxt_ptr, drow_ptr, q_ptr, q_stride, inv_vocab_ptr, seed,
                   lse_ptr, amax_ptr, accept_ptr, start_ptr, count_ptr, out_ptr,
                   HAS_Q: tl.constexpr, MAX_ROWS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    start = tl.load(start_ptr + i)
    j = tl.arange(0, MAX_ROWS)
    acc = tl.load(accept_ptr + start + j, mask=j < tl.load(count_ptr + i), other=0)
    a = tl.min(tl.where(acc == 0, j, MAX_ROWS), 0)  # leading accepted drafts; the last row never accepts
    r = start + a
    inv_t = tl.load(inv_t_ptr + r)
    tok = tl.load(amax_ptr + r)
    if inv_t > 0:
        # the exponential race: argmin over tokens of E / max(0, p - q), with E ~ Exponential(1), picks a
        # token with probability proportional to max(0, p - q)
        base = logits_ptr + r.to(tl.int64) * stride
        lse = tl.load(lse_ptr + r)
        d = tl.load(nxt_ptr + r)  # -1 on the last row, where q = 0
        k = tl.maximum(tl.load(drow_ptr + r), 0).to(tl.int64)
        best = tl.zeros([], tl.float32) + float("inf")
        best_i = tl.zeros([], tl.int32)
        for c0 in range(0, n_cols, BLOCK):
            cols = c0 + tl.arange(0, BLOCK)
            mask = cols < n_cols
            p = tl.exp(tl.load(base + cols, mask=mask, other=-float("inf")).to(tl.float32) * inv_t - lse)
            if HAS_Q:
                jq = tl.load(inv_vocab_ptr + cols, mask=mask, other=-1)
                q = tl.load(q_ptr + k * q_stride + jq, mask=mask & (jq >= 0) & (d >= 0), other=0.0)
            else:
                q = tl.where(cols == d, 1.0, 0.0)
            res = tl.maximum(p - q, 0.0)
            e = -tl.log(1.0 - tl.rand(seed, (r * n_cols + cols).to(tl.int32)))
            key = tl.where(res > 0, e / res, float("inf"))
            bk = tl.min(key, 0)
            best_i = tl.where(bk < best, c0 + tl.argmin(key, 0), best_i)
            best = tl.minimum(best, bk)
        tok = best_i
    tl.store(out_ptr + 2 * i, a)
    tl.store(out_ptr + 2 * i + 1, tok)


_inverse = {}  # id(draft vocabulary) -> (vocabulary, number of columns, column -> index in the vocabulary or -1)


def _inverse_vocab(draft_vocab, n_cols):
    hit = _inverse.get(id(draft_vocab))
    if hit is None or hit[0] is not draft_vocab or hit[1] != n_cols:
        inv = torch.full((n_cols,), -1, dtype=torch.int32, device=draft_vocab.device)
        inv[draft_vocab] = torch.arange(len(draft_vocab), dtype=torch.int32, device=draft_vocab.device)
        hit = _inverse[id(draft_vocab)] = (draft_vocab, n_cols, inv)
    return hit[2]


def verify_drafts(sampler: Sampler, logits, params: list[SamplingParams], drafts: list[list[int]],
                  draft_probs=None, draft_vocab=None, out=None):
    """Returns, per sequence, its accepted drafts followed by one token from the target model.

    logits [N, V] holds, for each sequence in order, the rows of its last token and of each of its drafts
    (len(drafts[i]) + 1 rows); the row of position j gives the distribution of the token after it.
    draft_probs [sum of len(drafts[i]), len(draft_vocab)]: the distribution each draft was sampled from,
    over the token ids draft_vocab; None for deterministic drafts.
    out: an optional int64 GPU tensor with a row per sequence, which receives the number of accepted drafts
    and the token after them, for work that continues on the GPU.
    """
    if any(p.temperature > 0 and (p.top_k > 0 or p.top_p < 1.0) for p in params):
        result = _verify_full(sampler, logits, params, drafts, draft_probs, draft_vocab)
        if out is not None:
            out[: len(result)].copy_(torch.tensor([[len(r) - 1, r[-1]] for r in result]))
        return result
    device = logits.device
    counts = np.array([len(d) + 1 for d in drafts], np.int64)
    assert counts.max() <= MAX_ROWS
    num_seqs, num_rows = len(drafts), int(counts.sum())
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    nxt = np.array([t for d in drafts for t in d + [-1]], np.int64)
    drow = np.full(num_rows, -1, np.int64)
    drow[nxt >= 0] = np.arange(int((nxt >= 0).sum()))  # each row's draft, as a row of draft_probs
    meta = torch.from_numpy(np.concatenate([nxt, drow, starts, counts])).to(device)
    nxt_t, drow_t = meta[:num_rows], meta[num_rows : 2 * num_rows]
    start_t, count_t = meta[2 * num_rows : 2 * num_rows + num_seqs], meta[2 * num_rows + num_seqs :]
    inv_t = torch.from_numpy(np.repeat([1.0 / p.temperature if p.temperature > 0 else 0.0 for p in params],
                                       counts).astype(np.float32)).to(device)
    n_cols = sampler.vocab_limit or logits.shape[1]
    has_q = draft_probs is not None
    q = draft_probs if has_q else inv_t  # a placeholder pointer without q
    inv_vocab = _inverse_vocab(draft_vocab, n_cols) if has_q else inv_t
    seeds = torch.randint(2**31 - 1, (2,)).tolist()
    lse = torch.empty(num_rows, dtype=torch.float32, device=device)
    amax = torch.empty(num_rows, dtype=torch.int32, device=device)
    accept = torch.empty(num_rows, dtype=torch.int8, device=device)
    out = torch.empty(num_seqs, 2, dtype=torch.int64, device=device) if out is None else out[:num_seqs]
    q_stride = q.stride(0) if has_q else 0
    _accept_kernel[(num_rows,)](logits, logits.stride(0), n_cols, inv_t, nxt_t, drow_t, q, q_stride, inv_vocab,
                                seeds[0], lse, amax, accept, HAS_Q=has_q, BLOCK=BLOCK, num_warps=NUM_WARPS)
    _sample_kernel[(num_seqs,)](logits, logits.stride(0), n_cols, inv_t, nxt_t, drow_t, q, q_stride, inv_vocab,
                                seeds[1], lse, amax, accept, start_t, count_t, out,
                                HAS_Q=has_q, MAX_ROWS=MAX_ROWS, BLOCK=BLOCK, num_warps=NUM_WARPS)
    return [d[:a] + [tok] for d, (a, tok) in zip(drafts, out.tolist())]


def _verify_full(sampler, logits, params, drafts, draft_probs, draft_vocab):
    """The same decision with every row's p and q built over the whole vocabulary (top-k and top-p)."""
    device = logits.device
    counts = [len(d) + 1 for d in drafts]
    row_params = [p for p, c in zip(params, counts) for _ in range(c)]
    # the draft each row checks; a sequence's last row has none
    nxt_list = [t for d in drafts for t in d + [-1]]
    nxt = torch.tensor(nxt_list, device=device)
    probs, greedy = sampler.probs(logits, row_params)
    if probs is None:  # all greedy
        accept, out = greedy == nxt, greedy
    else:
        has_draft = nxt >= 0
        d = nxt.clamp(min=0)
        # q on the full vocabulary, zero in the rows without a draft
        q = torch.zeros_like(probs)
        if draft_probs is None:
            q.scatter_(1, d[:, None], has_draft.float()[:, None])
        else:
            rows = torch.tensor([r for r, t in enumerate(nxt_list) if t >= 0], device=device)
            q[rows] = torch.zeros_like(probs[: len(rows)]).scatter_(1, draft_vocab.expand(len(rows), -1), draft_probs)
        p_d, q_d = probs.gather(1, d[:, None]).squeeze(1), q.gather(1, d[:, None]).squeeze(1)
        is_greedy = torch.tensor([p.temperature == 0 for p in row_params], device=device)
        accept = torch.where(is_greedy, greedy == nxt, torch.rand_like(p_d) * q_d < p_d) & has_draft
        # after a rejection, sample from max(0, p - q); a last row (q = 0) samples from p
        residual = (probs - q).clamp_min_(0)
        sampled = residual.div_(torch.empty_like(residual).exponential_()).argmax(dim=-1)
        out = torch.where(is_greedy, greedy, sampled)
    accept, out = torch.stack([accept.long(), out]).tolist()
    result, r = [], 0
    for d, c in zip(drafts, counts):
        a = 0
        while a < len(d) and accept[r + a]:
            a += 1
        result.append(d[:a] + [out[r + a]])
        r += c
    return result
