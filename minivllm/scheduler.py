"""Decides which sequences run in each step. A step is either all prefill or all decode.

Static batching: a batch is filled from the waiting queue, then decodes until every sequence in it has
finished; only then is the next batch formed. Continuous batching: waiting sequences are admitted at every
step as soon as they fit, and new prefills take priority over decoding.
"""

from collections import deque

from minivllm.sequence import Sequence, Status


class Scheduler:
    def __init__(self, kv, max_num_seqs, max_num_batched_tokens, continuous):
        self.kv = kv
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.continuous = continuous
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []  # in admission order
        self.filling = True  # static batching: the current batch still accepts sequences
        self.num_preemptions = 0

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def has_unfinished(self):
        return bool(self.waiting or self.running)

    def schedule(self):
        """Returns (sequences, is_prefill)."""
        if self.continuous or self.filling:
            seqs = self._admit()
            if seqs:
                return seqs, True
            self.filling = False
        return self._decode(), False

    def _admit(self):
        seqs, tokens = [], 0
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if tokens + len(seq) > self.max_num_batched_tokens or not self.kv.can_admit(seq):
                break
            self.waiting.popleft()
            self.kv.admit(seq)
            seq.status = Status.RUNNING
            self.running.append(seq)
            seqs.append(seq)
            tokens += len(seq)
        return seqs

    def _decode(self):
        scheduled, i = [], 0
        while i < len(self.running):
            seq = self.running[i]
            if self.kv.can_append(seq):
                self.kv.append(seq)
                scheduled.append(seq)
                i += 1
                continue
            # out of blocks: preempt the most recently admitted sequence; it is recomputed later
            victim = self.running.pop()
            self._preempt(victim)
            if victim is seq:
                break
        return scheduled

    def _preempt(self, seq: Sequence):
        self.kv.release(seq)
        seq.status = Status.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence):
        self.kv.release(seq)
        seq.status = Status.FINISHED
        self.running.remove(seq)
        if not self.running:
            self.filling = True
