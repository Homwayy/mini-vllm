"""KV pool allocation policies. Both hand out block tables over the same pool layout, so the model and the
attention backends do not change between them.

Before a sequence runs a step, its blocks must cover all of its tokens: a prefill writes K/V for every
token, and a decode step writes K/V for the newest one.
"""

from collections import deque

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

    def can_admit(self, seq: Sequence):
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

    reserve_full_length: admit a sequence only if blocks for its prompt plus max_tokens are still
    unreserved, so running sequences never run out of blocks. Otherwise a sequence is admitted as soon
    as its prompt fits, and the scheduler preempts when the pool runs dry.
    """

    def __init__(self, num_blocks, block_size, reserve_full_length):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.reserve_full_length = reserve_full_length
        self.free_blocks = deque(range(num_blocks))
        self.reserved = 0  # blocks promised to running sequences (reserve_full_length only)

    def can_admit(self, seq: Sequence):
        if self.reserve_full_length:
            return self.reserved + cdiv(seq.max_len, self.block_size) <= self.num_blocks
        return cdiv(len(seq), self.block_size) <= len(self.free_blocks)

    def admit(self, seq: Sequence):
        if self.reserve_full_length:
            self.reserved += cdiv(seq.max_len, self.block_size)
        self.append(seq)

    def can_append(self, seq: Sequence):
        return cdiv(len(seq), self.block_size) - len(seq.block_table) <= len(self.free_blocks)

    def append(self, seq: Sequence):
        for _ in range(cdiv(len(seq), self.block_size) - len(seq.block_table)):
            seq.block_table.append(self.free_blocks.popleft())

    def release(self, seq: Sequence):
        if self.reserve_full_length:
            self.reserved -= cdiv(seq.max_len, self.block_size)
        self.free_blocks.extend(seq.block_table)
        seq.block_table = []

    def allocated_slots(self):
        return (self.num_blocks - len(self.free_blocks)) * self.block_size
