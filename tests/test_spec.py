"""Speculative decoding pieces on their own: n-gram proposals, and draft verification.

Verification must leave the output distribution unchanged: with drafts [2, 5] on fixed target logits, the
first output token must follow the target distribution of row 0, the second (given the first was 2) that of
row 1, and the third (given 2, 5) that of row 2, over 200k samples. The same holds for drafts sampled from
another distribution q (EAGLE's), given in the order of a draft vocabulary, and on a vocabulary that spans
several blocks of the fused kernels and is cut by a vocab limit. Greedy rows accept exactly the drafts that
match the argmax.

Drafting draws each draft with one kernel (sample_drafts): greedy rows take the first argmax, sampled rows keep
q = softmax(logits / t) and draw from it, independently across rows, steps and CUDA graph replays.

Run from the repo root: python -m tests.test_spec
"""

import math

import torch

from minivllm.sampler import Sampler
from minivllm.sequence import SamplingParams, Sequence
from minivllm.spec import draft_sampling, rejection
from minivllm.spec.draft_sampling import sample_drafts
from minivllm.spec.ngram import NgramProposer
from minivllm.spec.rejection import verify_drafts
from tests.test_sampler import expected

N = 200_000


def seq(tokens):
    return Sequence(tokens, SamplingParams(max_tokens=16), 0.0)


def test_ngram():
    p = NgramProposer(n_max=3, n_min=2)
    s = seq([1, 2, 3, 4, 5, 1, 2, 3])
    assert p.propose(s, 3) == [4, 5, 1]  # [1, 2, 3] occurred at 0
    s.token_ids.append(9)
    assert p.propose(s, 3) == []  # neither [2, 3, 9] nor [3, 9] occurred before
    s.token_ids += [4, 1, 2]
    assert p.propose(s, 3) == [3, 9, 4]  # [1, 2] occurred at 0 and 5; the later one is used
    assert p.propose(s, 1) == [3]
    # an occurrence may overlap the last n-gram; then fewer than k tokens follow it
    assert NgramProposer(n_max=3, n_min=2).propose(seq([7, 7, 7, 7]), 4) == [7]
    assert NgramProposer().propose(seq([5]), 4) == []
    print("test_ngram: ok")


def tolerance(n):
    return 4.5 * math.sqrt(0.25 / n)  # 4.5 standard errors of a frequency near 0.5


def test_distribution():
    torch.manual_seed(0)
    base = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0], device="cuda")
    rows = torch.stack([base.roll(2), base.roll(5), base])  # make drafts 2 and 5 the likeliest tokens
    drafts = [2, 5]
    cases = [
        SamplingParams(temperature=1.0),
        SamplingParams(temperature=0.7),
        SamplingParams(temperature=1.0, top_k=3),
        SamplingParams(temperature=1.0, top_p=0.8),
        SamplingParams(temperature=0.6, top_k=5, top_p=0.9),
    ]
    sampler = Sampler()
    ok = True
    for p in cases:
        # every other sequence is greedy, to check that mixed batches keep per-row parameters
        params = [p, SamplingParams(temperature=0.0)] * (N // 2)
        out = verify_drafts(sampler, rows.repeat(N, 1), params, [drafts] * N)
        sampled, greedy = out[0::2], out[1::2]
        errs = []
        for depth in range(3):
            # outputs whose first `depth` tokens were the accepted drafts
            sub = [o[depth] for o in sampled if o[:depth] == drafts[:depth]]
            freq = torch.bincount(torch.tensor(sub), minlength=8).float() / len(sub)
            err = (freq - expected(rows[depth].cpu(), p)).abs().max().item()
            errs.append(err)
            ok &= err < tolerance(len(sub))
        greedy_ok = all(o == [2, 5, 0] for o in greedy)
        print(f"{p}: max |freq - exact| at positions 1-3: {', '.join(f'{e:.4f}' for e in errs)}; "
              f"greedy rows {'ok' if greedy_ok else 'WRONG'}")
        ok &= greedy_ok
    assert ok


def test_distribution_sampled_drafts():
    # drafts sampled from q, a different distribution over a reordered draft vocabulary (as EAGLE's)
    torch.manual_seed(1)
    base = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0], device="cuda")
    rows = torch.stack([base.roll(2), base.roll(5), base])
    vocab = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], device="cuda")
    q = torch.softmax(torch.stack([base.flip(0), base.roll(1).flip(0)]) * 0.5, -1)  # [2 positions, 8], over vocab
    sampler = Sampler()
    ok = True
    for p in (SamplingParams(temperature=1.0), SamplingParams(temperature=0.7), SamplingParams(temperature=0.7, top_p=0.9)):
        d = vocab[torch.stack([torch.multinomial(q[0], N, replacement=True), torch.multinomial(q[1], N, replacement=True)], 1)]
        drafts = d.tolist()
        out = verify_drafts(sampler, rows.repeat(N, 1), [p] * N, drafts, q.repeat(N, 1), vocab)
        errs = []
        for depth in range(3):
            sub = [o[depth] for o, dr in zip(out, drafts) if o[:depth] == dr[:depth]]
            freq = torch.bincount(torch.tensor(sub), minlength=8).float() / len(sub)
            err = (freq - expected(rows[depth].cpu(), p)).abs().max().item()
            errs.append(err)
            ok &= err < tolerance(len(sub))
        print(f"sampled drafts, {p}: max |freq - exact| at positions 1-3: {', '.join(f'{e:.4f}' for e in errs)}")
    assert ok


def test_large_vocab():
    # two blocks of the fused kernels and a partial third, cut by a vocab limit: likely tokens on both sides of
    # a block edge and at the limit, and the largest logit past the limit, which must never come out.
    # Frequencies are checked for the likely tokens and for all the others together.
    torch.manual_seed(2)
    b = rejection.BLOCK
    V, limit = 2 * b + 1000, 2 * b + 600
    hot = [5, b - 1, b, b + 400, limit - 1]
    hot_t = torch.tensor(hot, device="cuda")

    def row(shift):
        x = torch.full((V,), -6.0, device="cuda")
        x[hot_t] = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0], device="cuda").roll(shift)
        x[limit + 100] = 10.0
        return x

    rows = torch.stack([row(1), row(3), row(0)])  # argmaxes b - 1, b + 400 and 5
    drafts = [b - 1, b + 400]
    vocab = torch.tensor(sorted(set(range(0, limit, 2)) | set(hot)), device="cuda")
    boost = torch.zeros(len(vocab), device="cuda")
    q = torch.stack([boost.index_put((torch.searchsorted(vocab, hot_t),), torch.tensor(w, device="cuda"))
                     for w in ([0.0, 3.0, 3.0, 1.0, 0.0], [3.0, 0.0, 1.0, 3.0, 0.0])])
    q = torch.softmax(q - 2.0 * (q == 0), -1)  # [2 positions, len(vocab)]
    sampler, chunk = Sampler(limit), 4000
    g = SamplingParams(temperature=0.0)
    full = rejection._verify_full
    rejection._verify_full = None  # every case here must take the fused kernels
    try:
        assert verify_drafts(sampler, rows, [g], [drafts]) == [drafts + [5]]
        assert verify_drafts(sampler, rows, [g], [[b - 1, b]]) == [[b - 1, b + 400]]
        ok = True
        for p, sampled in ((SamplingParams(temperature=1.0), False), (SamplingParams(temperature=0.7), False),
                           (SamplingParams(temperature=1.0), True), (SamplingParams(temperature=0.7), True)):
            outs, drs = [], []
            for _ in range(N // chunk):
                if sampled:
                    d = vocab[torch.stack([torch.multinomial(q[j], chunk, replacement=True) for j in range(2)], 1)].tolist()
                    outs += verify_drafts(sampler, rows.repeat(chunk, 1), [p] * chunk, d, q.repeat(chunk, 1), vocab)
                else:
                    d = [drafts] * chunk
                    outs += verify_drafts(sampler, rows.repeat(chunk, 1), [p] * chunk, d)
                drs += d
            ok &= all(t < limit for o in outs for t in o)
            errs = []
            for depth in range(3):
                sub = torch.tensor([o[depth] for o, dr in zip(outs, drs) if o[:depth] == dr[:depth]], device="cuda")
                freq = torch.cat([(sub[:, None] == hot_t).float().mean(0), (~torch.isin(sub, hot_t)).float().mean()[None]])
                e = torch.softmax(rows[depth, :limit] / p.temperature, -1)
                want = torch.cat([e[hot_t], 1 - e[hot_t].sum()[None]])
                errs.append((freq - want).abs().max().item())
                ok &= errs[-1] < tolerance(len(sub))
            print(f"large vocabulary, {'sampled' if sampled else 'fixed'} drafts, temperature {p.temperature}: "
                  f"max |freq - exact| at positions 1-3: {', '.join(f'{x:.4f}' for x in errs)}")
    finally:
        rejection._verify_full = full
    assert ok


def test_draft_sampling():
    # a vocabulary of two blocks of the kernel (with many rows) and part of a third, in the order of a draft
    # vocabulary; a single block with few rows
    torch.manual_seed(3)
    b = draft_sampling.CONFIGS[False][0]
    V = 2 * b + 1000
    vocab = torch.randperm(50_000, device="cuda")[:V]
    seed = torch.zeros(1, dtype=torch.int64, device="cuda")

    def run(x, temp, bias=None, step=0):
        tok = torch.zeros(x.shape[0], 2, dtype=torch.int64, device="cuda")  # a strided column, as in the chain
        prev = torch.zeros(x.shape[0], dtype=torch.int64, device="cuda")
        probs = torch.zeros(x.shape[0], 2, V, device="cuda")
        sample_drafts(x, temp, seed, vocab, tok[:, 1], prev, probs[:, 1], step, bias)
        assert torch.equal(tok[:, 1], prev)
        return tok[:, 1], probs[:, 1]

    for rows in (64, draft_sampling.FEW_ROWS):
        # greedy rows with many ties take the first argmax, as torch.argmax does
        x = torch.randint(-4, 5, (rows, V), device="cuda").bfloat16()
        bias = torch.randint(-2, 3, (rows, V), device="cuda").bfloat16()
        greedy = torch.zeros(rows, device="cuda")
        assert torch.equal(run(x, greedy)[0], vocab[x.float().argmax(-1)])
        assert torch.equal(run(x, greedy, bias)[0], vocab[(x.float() + bias.float()).argmax(-1)])
        # sampled rows keep q
        x = (3 * torch.randn(rows, V, device="cuda")).bfloat16()
        for t in (0.5, 1.0, 1.7):
            for bb in (None, bias):
                want = torch.softmax((x.float() + (0.0 if bb is None else bb.float())) / t, -1)
                assert (run(x, torch.full((rows,), t, device="cuda"), bb)[1] - want).abs().max().item() < 1e-6
    # frequencies of likely tokens on both sides of block edges and at the end, and of all others together;
    # every 8 calls share a seed, with steps 0-7. Odd rows are greedy.
    hot = torch.tensor([5, b - 1, b, b + 400, V - 1], device="cuda")
    x = torch.full((V,), -6.0, device="cuda")
    x[hot] = torch.tensor([3.0, 2.5, 2.0, 1.5, 1.0], device="cuda")
    shift = torch.zeros(V, device="cuda")
    shift[b] = 1.0  # puts token b ahead of b - 1
    ok = True
    for rows, t, bias in ((4000, 1.0, None), (4000, 0.7, shift), (draft_sampling.FEW_ROWS, 1.0, shift)):
        logits = x.bfloat16().expand(rows, V)
        bb = None if bias is None else bias.bfloat16().expand(rows, V)
        full = logits[0].float() + (0.0 if bb is None else bb[0].float())
        temp = torch.tensor([t, 0.0], device="cuda").repeat(rows // 2)
        sampled = []
        for i in range(2 * N // rows):
            if i % 8 == 0:
                seed.random_(2**31 - 1)
            tok, _ = run(logits, temp, bb, step=i % 8)
            sampled.append(tok[0::2])
            ok &= bool((tok[1::2] == vocab[full.argmax()]).all())
        sub = torch.cat(sampled)
        freq = torch.cat([(sub[:, None] == vocab[hot]).float().mean(0), (~torch.isin(sub, vocab[hot])).float().mean()[None]])
        e = torch.softmax(full / t, -1)
        err = (freq - torch.cat([e[hot], 1 - e[hot].sum()[None]])).abs().max().item()
        ok &= err < tolerance(len(sub))
        print(f"draft sampling, {rows} rows, temperature {t}{', with a bias' if bias is not None else ''}: max |freq - exact| {err:.4f}")
    assert ok
    # in a CUDA graph the seed is redrawn on each replay: draws differ between replays, torch.manual_seed fixes them
    x, temp = torch.zeros(256, V, device="cuda").bfloat16(), torch.ones(256, device="cuda")
    tok, prev, probs = (torch.zeros(256, dtype=torch.int64, device="cuda"), torch.zeros(256, dtype=torch.int64, device="cuda"),
                        torch.zeros(256, V, device="cuda"))

    def draft():
        seed.random_(2**31 - 1)
        sample_drafts(x, temp, seed, vocab, tok, prev, probs, 0)

    draft()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        draft()
    draws = []
    for s in (0, 0, 1):
        torch.manual_seed(s)
        graph.replay()
        draws.append(tok.clone())
        graph.replay()
        draws.append(tok.clone())
    assert torch.equal(draws[0], draws[2]) and torch.equal(draws[1], draws[3])
    assert (draws[0] != draws[1]).float().mean() > 0.99 and (draws[0] != draws[4]).float().mean() > 0.99
    print("test_draft_sampling: ok")


def test_greedy_prefix():
    sampler = Sampler()
    base = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0], device="cuda")
    rows = torch.stack([base.roll(2), base.roll(5), base])  # argmaxes 2, 5, 0
    g = SamplingParams(temperature=0.0)
    for drafts, want in (([2, 5], [2, 5, 0]), ([2, 4], [2, 5]), ([3, 5], [2]), ([2], [2, 5])):
        out = verify_drafts(sampler, rows[: len(drafts) + 1], [g], [drafts])
        assert out == [want], (drafts, out)
    # sequences with different numbers of drafts in one call
    logits = torch.cat([rows, rows[:1], rows[:2]])
    assert verify_drafts(sampler, logits, [g] * 3, [[2, 5], [], [3]]) == [[2, 5, 0], [2], [2]]
    print("test_greedy_prefix: ok")


if __name__ == "__main__":
    test_ngram()
    test_greedy_prefix()
    test_distribution()
    test_distribution_sampled_drafts()
    test_large_vocab()
    test_draft_sampling()
    print("PASS")
