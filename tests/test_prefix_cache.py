"""PrefixCachingKV on its own (CPU only): hits, reference counts, LRU eviction, hash checks and the
reserve accounting, plus invariants after many random admit / grow / release operations.

Run from the repo root: python -m tests.test_prefix_cache
"""

import random

from minivllm.kv_manager import PrefixCachingKV, cdiv
from minivllm.sequence import SamplingParams, Sequence

P = 16


def seq(tokens, max_tokens=16):
    return Sequence(tokens, SamplingParams(max_tokens=max_tokens), 0.0)


def admit(kv, s):
    # as the scheduler does for a prefill: the step computes all the tokens that were not cached
    kv.admit(s)
    kv.computed(s, s.num_cached_tokens, len(s))


def grow(kv, s):
    # as the scheduler does for a decode step
    kv.append(s)
    kv.computed(s, len(s) - 1, len(s))


def check_invariants(kv, live):
    refs = [0] * kv.num_blocks
    for s in live:
        for b in s.block_table:
            refs[b] += 1
    assert refs == kv.ref, "reference counts do not match the block tables"
    assert set(kv.free_blocks) == {b for b in range(kv.num_blocks) if kv.ref[b] == 0}
    for h, b in kv.cached.items():
        assert kv.block_hash[b] == h and kv.block_tokens[b] is not None
    assert sum(h is not None for h in kv.block_hash) == len(kv.cached)
    if kv.admission == "reserve":
        future = sum(cdiv(s.max_len, P) - len(s.block_table) for s in live)
        assert kv.future == future and future <= len(kv.free_blocks)


def test_hits_and_release():
    kv = PrefixCachingKV(64, P, "reserve")
    prefix = list(range(1000, 1040))  # 2 full blocks and 8 tokens
    a = seq(prefix + [1, 2, 3])
    assert kv.cached_tokens(a) == 0
    admit(kv, a)
    assert a.num_cached_tokens == 0 and len(a.block_table) == 3
    b = seq(prefix + [4, 5])  # shares the 2 full blocks
    assert kv.cached_tokens(b) == 32
    admit(kv, b)
    assert b.block_table[:2] == a.block_table[:2] and b.block_table[2] != a.block_table[2]
    assert b.num_cached_tokens == 32 and kv.ref[a.block_table[0]] == 2
    # a prompt that is exactly 2 cached blocks: the block with its last token is computed again
    c = seq(prefix[:32])
    assert kv.cached_tokens(c) == 16
    admit(kv, c)
    check_invariants(kv, [a, b, c])
    shared = a.block_table[:2]
    for s in (a, b, c):
        kv.release(s)
    check_invariants(kv, [])
    # released blocks keep their K/V: a new sequence with the prefix still hits, taking them off the free list
    d = seq(prefix + [9])
    assert kv.cached_tokens(d) == 32
    admit(kv, d)
    assert d.block_table[:2] == shared
    check_invariants(kv, [d])


def test_lru_eviction():
    kv = PrefixCachingKV(8, P, "optimistic")
    a = seq(list(range(33)))  # 2 full blocks, cached once admitted
    admit(kv, a)
    kv.release(a)
    # the free list now holds a's blocks after the 5 never-used ones; taking 7 blocks evicts a's tail first
    big = seq(list(range(5000, 5000 + 7 * P)))
    admit(kv, big)
    assert kv.cached_tokens(seq(list(range(33)))) == 16  # first block still cached, second evicted
    kv.release(big)
    check_invariants(kv, [])


def test_collision_check():
    kv = PrefixCachingKV(16, P, "reserve")
    a = seq(list(range(40)))
    admit(kv, a)
    # pretend another prefix hashes to the same value as a's first block
    b = seq(list(range(100, 140)))
    h = kv._hashes(b, 1)[0]
    kv.cached[h] = a.block_table[0]
    assert kv.cached_tokens(b) == 0  # the stored tokens differ, so it is not a hit
    del kv.cached[h]


def test_reserve_counts_shared_blocks_once():
    # 64 requests of 1000 shared + 100 own tokens, 100 output tokens: each needs 76 blocks on its own,
    # 4864 in all; with sharing, 62 + 64 * 14 = 958, which fits a 1024-block pool
    rng = random.Random(0)
    prefix = [rng.randrange(10000) for _ in range(1000)]
    kv = PrefixCachingKV(1024, P, "reserve")
    live = []
    for _ in range(64):
        s = seq(prefix + [rng.randrange(10000) for _ in range(100)], max_tokens=100)
        assert kv.can_admit(s, len(live))
        admit(kv, s)
        live.append(s)
    assert sum(s.num_cached_tokens for s in live) == 63 * 992
    check_invariants(kv, live)
    # every sequence can grow to its full length without running out
    for _ in range(100):
        for s in live:
            s.token_ids.append(0)
            assert kv.can_append(s)
            grow(kv, s)
    check_invariants(kv, live)


def test_random_operations():
    rng = random.Random(1)
    prefixes = [[rng.randrange(50) for _ in range(rng.randint(1, 80))] for _ in range(6)]
    for admission in ("reserve", "optimistic", "headroom"):
        kv = PrefixCachingKV(96, P, admission)
        live, waiting = [], []
        for _ in range(3000):
            op = rng.random()
            if op < 0.35:
                tokens = list(rng.choice(prefixes)) + [rng.randrange(50) for _ in range(rng.randint(0, 40))]
                waiting.append(seq(tokens, max_tokens=rng.randint(1, 60)))
            if op < 0.6 and waiting and waiting[0].max_len <= 96 * P and kv.can_admit(waiting[0], len(live)):
                s = waiting.pop(0)
                admit(kv, s)
                assert len(s.block_table) == cdiv(len(s), P) and s.num_cached_tokens < len(s)
                live.append(s)
            elif op < 0.9 and live:
                s = rng.choice(live)
                if len(s) < s.max_len:
                    s.token_ids.append(rng.randrange(50))
                    if kv.can_append(s):
                        grow(kv, s)
                    else:  # out of blocks, as the scheduler would see it: preempt
                        assert admission != "reserve"
                        s.token_ids.pop()
                        kv.release(s)
                        live.remove(s)
                        waiting.insert(0, s)
            elif live:
                s = live.pop(rng.randrange(len(live)))
                kv.release(s)
            check_invariants(kv, live)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name}: ok")
    print("PASS")
