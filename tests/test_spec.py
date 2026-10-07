"""Speculative decoding pieces on their own: n-gram proposals, and draft verification.

Verification must leave the output distribution unchanged: with drafts [2, 5] on fixed target logits, the
first output token must follow the target distribution of row 0, the second (given the first was 2) that of
row 1, and the third (given 2, 5) that of row 2, over 200k samples. The same holds for drafts sampled from
another distribution q (EAGLE's), given in the order of a draft vocabulary. Greedy rows accept exactly the
drafts that match the argmax.

Run from the repo root: python -m tests.test_spec
"""

import math

import torch

from minivllm.sampler import Sampler
from minivllm.sequence import SamplingParams, Sequence
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
    for p in (SamplingParams(temperature=1.0), SamplingParams(temperature=0.7, top_p=0.9)):
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
    print("PASS")
