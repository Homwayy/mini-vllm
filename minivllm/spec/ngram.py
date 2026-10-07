"""Prompt lookup (n-gram) drafting.

If the last n tokens of a sequence occurred earlier in it (prompt or output), the tokens that followed that
occurrence are proposed. n goes from n_max down to n_min, and the most recent occurrence is used.

Each sequence keeps, for every n, a dict from n-gram to the start of its latest occurrence, extended as
tokens are appended, so a step costs a few dict lookups per sequence instead of a scan of the sequence.
"""

import weakref

from minivllm.sequence import Sequence


class NgramProposer:
    def __init__(self, n_max=4, n_min=2):
        self.ns = range(n_max, n_min - 1, -1)
        self.index = weakref.WeakKeyDictionary()  # sequence -> (dict per n, next n-gram end to index)

    def propose(self, seq: Sequence, k: int) -> list[int]:
        """Up to k draft tokens for seq; empty if its last n-gram has not occurred before."""
        tokens, L = seq.token_ids, len(seq)
        dicts, next_end = self.index.get(seq) or ({n: {} for n in self.ns}, 1)
        # occurrences ending before the last token are earlier ones (they may overlap the last n-gram)
        for end in range(next_end, L):
            for n in self.ns:
                if end >= n:
                    dicts[n][tuple(tokens[end - n : end])] = end - n
        self.index[seq] = (dicts, L)
        for n in self.ns:
            start = dicts[n].get(tuple(tokens[L - n :])) if L >= n else None
            if start is not None:
                return tokens[start + n : start + n + k]
        return []
