"""Sampler: empirical token frequencies against the exact distribution, over 100k samples per case.

Run from the repo root: python -m tests.test_sampler
"""

import torch

from minivllm.sampler import Sampler
from minivllm.sequence import SamplingParams

N = 200_000
TOL = 0.005  # the standard error of each frequency is at most 0.0016


def expected(logits, p: SamplingParams):
    probs = torch.softmax(logits / p.temperature, dim=-1)
    order = probs.argsort(descending=True)
    dropped = torch.zeros_like(probs, dtype=torch.bool)
    if p.top_k > 0:
        dropped[order[p.top_k :]] = True
    sorted_probs = torch.softmax(torch.where(dropped, float("-inf"), logits / p.temperature)[order], dim=-1)
    dropped[order[sorted_probs.cumsum(0) - sorted_probs >= p.top_p]] = True
    out = torch.where(dropped, 0.0, probs)
    return out / out.sum()


def main():
    torch.manual_seed(0)
    logits = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -3.0], device="cuda")
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
        # every other row is greedy, to check that mixed batches keep per-row parameters
        params = [p, SamplingParams(temperature=0.0)] * (N // 2)
        tokens = sampler(logits.expand(N, -1).clone(), params)
        sampled, greedy = tokens[0::2], tokens[1::2]
        freq = torch.bincount(sampled, minlength=len(logits)).float() / sampled.numel()
        err = (freq - expected(logits.float(), p)).abs().max().item()
        greedy_ok = bool((greedy == logits.argmax()).all())
        print(f"{p}: max |freq - exact| {err:.4f}, greedy rows {'ok' if greedy_ok else 'WRONG'}")
        ok &= err < TOL and greedy_ok
    # padding ids above vocab_limit are never sampled
    tokens = Sampler(vocab_limit=6)(torch.zeros(N, 8, device="cuda"), [SamplingParams(temperature=1.0)] * N)
    print(f"vocab_limit=6: max sampled id {tokens.max().item()}")
    ok &= tokens.max().item() < 6
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
