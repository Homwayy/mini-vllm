"""Checking draft tokens against the target model (speculative sampling).

A draft token d is accepted with probability min(1, p(d) / q(d)), where p is the target distribution after
temperature, top-k and top-p and q the drafter's; after a rejection, the token comes from max(0, p - q)
renormalized. Each output token is then distributed exactly as without drafting. A deterministic drafter
such as n-gram puts all of q on d, so d is accepted with probability p(d) and a rejection samples from p
without d. Greedy rows accept d only if it is the argmax.

All rows are decided on the GPU and copied to the CPU once per step.
"""

import torch

from minivllm.sampler import Sampler
from minivllm.sequence import SamplingParams


def verify_drafts(sampler: Sampler, logits, params: list[SamplingParams], drafts: list[list[int]]):
    """Returns, per sequence, its accepted drafts followed by one token from the target model.

    logits [N, V] holds, for each sequence in order, the rows of its last token and of each of its drafts
    (len(drafts[i]) + 1 rows); the row of position j gives the distribution of the token after it.
    """
    device = logits.device
    counts = [len(d) + 1 for d in drafts]
    row_params = [p for p, c in zip(params, counts) for _ in range(c)]
    # the draft each row checks; a sequence's last row has none
    nxt = torch.tensor([t for d in drafts for t in d + [-1]], device=device)
    has_draft = nxt >= 0
    probs, greedy = sampler.probs(logits, row_params)
    if probs is None:  # all greedy
        accept, out = greedy == nxt, greedy
    else:
        d = nxt.clamp(min=0)
        p_d = probs.gather(1, d[:, None]).squeeze(1)
        is_greedy = torch.tensor([p.temperature == 0 for p in row_params], device=device)
        accept = torch.where(is_greedy, greedy == nxt, torch.rand_like(p_d) < p_d) & has_draft
        # after a rejection, sample from p without the draft; a last row samples from p
        probs.scatter_(1, d[:, None], torch.where(has_draft, 0.0, p_d)[:, None])
        sampled = probs.div_(torch.empty_like(probs).exponential_()).argmax(dim=-1)
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
