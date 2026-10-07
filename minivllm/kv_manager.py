"""KV pool allocation policies. Both hand out block tables over the same pool layout, so the model and the
attention backends do not change between them.

Before a sequence runs a step, its blocks must cover all of its tokens: a prefill writes K/V for every
token, and a decode step writes K/V for the newest one.
"""

from collections import OrderedDict, deque

from minivllm.sequence import Sequence


def cdiv(a, b):
    return -(-a // b)


class ContiguousKV:
    """A fixed contiguous region per sequence, sized for max_model_len, like a static KV cache."""

    def __init__(self, num_blocks, block_size, max_model_len):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.blocks_per_seq = cdiv(max_model_len, block_size)
        self.num_regions = num_blocks // self.blocks_per_seq
        self.free_regions = deque(range(self.num_regions))

    def cached_tokens(self, seq: Sequence):
        return 0

    def computed(self, seq: Sequence, start, end):
        """The step being scheduled computes seq's tokens start .. end - 1 (only the prefix cache uses this)."""

    def can_admit(self, seq: Sequence, num_running: int):
        return bool(self.free_regions)

    def admit(self, seq: Sequence):
        r = self.free_regions.popleft()
        seq.block_table = list(range(r * self.blocks_per_seq, (r + 1) * self.blocks_per_seq))

    def can_append(self, seq: Sequence):
        return True

    def append(self, seq: Sequence):
        pass

    def release(self, seq: Sequence):
        self.free_regions.append(seq.block_table[0] // self.blocks_per_seq)
        seq.block_table = []

    def allocated_slots(self):
        return (self.num_regions - len(self.free_regions)) * self.blocks_per_seq * self.block_size


class PagedKV:
    """Blocks are handed out one at a time as sequences grow.

    Admission policies:
      reserve: admit a sequence only if blocks for its prompt plus max_tokens are still unreserved, so
        running sequences never run out of blocks.
      optimistic: admit a sequence as soon as its current tokens fit; the scheduler preempts when the pool
        runs dry.
      headroom: optimistic, but after admitting, one free block must remain for every running sequence,
        so each can grow by a block before anything is preempted.
    """

    def __init__(self, num_blocks, block_size, admission):
        assert admission in ("reserve", "optimistic", "headroom")
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.admission = admission
        self.free_blocks = deque(range(num_blocks))
        self.reserved = 0  # blocks promised to running sequences (reserve only)

    def cached_tokens(self, seq: Sequence):
        return 0

    def computed(self, seq: Sequence, start, end):
        """The step being scheduled computes seq's tokens start .. end - 1 (only the prefix cache uses this)."""

    def can_admit(self, seq: Sequence, num_running: int):
        if self.admission == "reserve":
            return self.reserved + cdiv(seq.max_len, self.block_size) <= self.num_blocks
        need = cdiv(len(seq), self.block_size)
        if self.admission == "headroom":
            need += num_running + 1
        return need <= len(self.free_blocks)

    def admit(self, seq: Sequence):
        if self.admission == "reserve":
            self.reserved += cdiv(seq.max_len, self.block_size)
        self.append(seq)

    def can_append(self, seq: Sequence):
        return cdiv(len(seq), self.block_size) - len(seq.block_table) <= len(self.free_blocks)

    def append(self, seq: Sequence):
        for _ in range(cdiv(len(seq), self.block_size) - len(seq.block_table)):
            seq.block_table.append(self.free_blocks.popleft())

    def release(self, seq: Sequence):
        if self.admission == "reserve":
            self.reserved -= cdiv(seq.max_len, self.block_size)
        self.free_blocks.extend(seq.block_table)
        seq.block_table = []

    def allocated_slots(self):
        return (self.num_blocks - len(self.free_blocks)) * self.block_size


class PrefixCachingKV(PagedKV):
    """PagedKV that shares full blocks between sequences with the same prefix.

    A full block is identified by a hash of its tokens chained with the previous block's hash, so equal
    hashes mean equal prefixes; on a hit the tokens are compared too, so a hash collision cannot share
    wrong K/V. Blocks are reference counted. A block whose count drops to zero keeps its K/V and its hash
    until it is handed out again, least recently released first. Admission reuses the cached blocks at
    the start of a sequence, and only its other tokens are computed. At least the last token is always
    computed, since its logits give the next token.

    A block is entered in the cache when the step that computes it is scheduled (computed()). Every layer
    writes the step's K/V into the pool before attention reads it, so a sequence scheduled later in the
    same step can already share it.

    reserve counts each held block once, plus the blocks that running sequences may still allocate, so
    shared blocks are not reserved twice.
    """

    def __init__(self, num_blocks, block_size, admission):
        super().__init__(num_blocks, block_size, admission)
        self.free_blocks = OrderedDict.fromkeys(range(num_blocks))  # reference count 0, oldest first
        self.ref = [0] * num_blocks
        self.block_hash = [None] * num_blocks
        self.block_tokens = [None] * num_blocks
        self.cached = {}  # hash -> block
        self.future = 0  # reserve: blocks the running sequences may still allocate

    def _hashes(self, seq: Sequence, n):
        """Prefix hashes of seq's first n full blocks."""
        P = self.block_size
        h = seq.block_hashes
        while len(h) < n:
            i = len(h)
            h.append(hash((h[-1] if h else 0, tuple(seq.token_ids[i * P : (i + 1) * P]))))
        return h[:n]

    def _match(self, seq: Sequence):
        """Cached blocks holding seq's leading full blocks, leaving at least its last token to compute."""
        P = self.block_size
        blocks = []
        for i, h in enumerate(self._hashes(seq, (len(seq) - 1) // P)):
            b = self.cached.get(h)
            if b is None or self.block_tokens[b] != tuple(seq.token_ids[i * P : (i + 1) * P]):
                break
            blocks.append(b)
        return blocks

    def cached_tokens(self, seq: Sequence):
        return len(self._match(seq)) * self.block_size

    def can_admit(self, seq: Sequence, num_running: int):
        hits = self._match(seq)
        held = sum(self.ref[b] > 0 for b in hits)  # hits that cost no free block
        if self.admission == "reserve":
            return self.future + cdiv(seq.max_len, self.block_size) - held <= len(self.free_blocks)
        need = cdiv(len(seq), self.block_size) - held
        if self.admission == "headroom":
            need += num_running + 1
        return need <= len(self.free_blocks)

    def admit(self, seq: Sequence):
        hits = self._match(seq)
        for b in hits:
            if self.ref[b] == 0:
                del self.free_blocks[b]
            self.ref[b] += 1
        seq.block_table = list(hits)
        seq.num_cached_tokens = len(hits) * self.block_size
        self._allocate(seq)
        if self.admission == "reserve":
            self.future += cdiv(seq.max_len, self.block_size) - len(seq.block_table)

    def append(self, seq: Sequence):
        n = self._allocate(seq)
        if self.admission == "reserve":
            self.future -= n

    def computed(self, seq: Sequence, start, end):
        """The step being scheduled computes seq's tokens start .. end - 1: their full blocks can be shared."""
        self._register(seq, start // self.block_size, end // self.block_size)

    def release(self, seq: Sequence):
        if self.admission == "reserve":
            self.future -= cdiv(seq.max_len, self.block_size) - len(seq.block_table)
        # tail blocks first, so that a sequence's prefix stays cached the longest
        for b in reversed(seq.block_table):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free_blocks[b] = None
        seq.block_table = []

    def _allocate(self, seq: Sequence):
        """Gives seq new blocks until its block table covers all of its tokens; returns how many."""
        n = cdiv(len(seq), self.block_size) - len(seq.block_table)
        for _ in range(n):
            b, _ = self.free_blocks.popitem(last=False)
            if self.block_hash[b] is not None:  # evict the K/V cached in it
                del self.cached[self.block_hash[b]]
                self.block_hash[b] = self.block_tokens[b] = None
            self.ref[b] = 1
            seq.block_table.append(b)
        return n

    def _register(self, seq: Sequence, start, end):
        """Enters seq's full blocks start .. end - 1 into the cache, unless their content is cached already."""
        P = self.block_size
        hashes = self._hashes(seq, end)
        for i in range(start, end):
            b, h = seq.block_table[i], hashes[i]
            if self.block_hash[b] is None and h not in self.cached:
                self.cached[h] = b
                self.block_hash[b] = h
                self.block_tokens[b] = tuple(seq.token_ids[i * P : (i + 1) * P])
