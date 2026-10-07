"""Decides which sequences run in each step.

Static batching: a batch is filled from the waiting queue, then decodes until every sequence in it has
finished; only then is the next batch formed. Continuous batching: waiting sequences are admitted at every
step as soon as they fit, and new prefills take priority over decoding. In both, a step is either all
prefill or all decode.

Chunked prefill (continuous only): every step has a budget of max_num_batched_tokens. Sequences that are
decoding go first, one token each, so a long prompt never holds them up; the rest of the budget continues
partly computed prompts in admission order, then admits new ones, the last of which may be cut short.
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
            new = len(seq) - self.kv.cached_tokens(seq)  # tokens this prefill computes
            if tokens + new > self.max_num_batched_tokens or not self.kv.can_admit(seq, len(self.running)):
                break
            self.waiting.popleft()
            self.kv.admit(seq)
            self.kv.computed(seq, seq.num_cached_tokens, len(seq))  # the prefill computes all other tokens
            seq.status = Status.RUNNING
            self.running.append(seq)
            seqs.append(seq)
            tokens += new
        return seqs

    def _decode(self):
        scheduled, i = [], 0
        while i < len(self.running):
            seq = self.running[i]
            if self.kv.can_append(seq):
                self.kv.append(seq)
                self.kv.computed(seq, len(seq) - 1, len(seq))
                scheduled.append(seq)
                i += 1
                continue
            # out of blocks: preempt the most recently admitted sequence; it is recomputed later
            victim = self.running.pop()
            self._preempt(victim)
            if victim is seq:
                break
        return scheduled

    def schedule_chunked(self):
        """Returns [(sequence, number of its next tokens to compute)] for one step."""
        plan = []

        def add(seq, n):
            # entered now, so that sequences scheduled after it in this step can share its new blocks
            self.kv.computed(seq, seq.num_computed_tokens, seq.num_computed_tokens + n)
            plan.append((seq, n))

        # sequences with one token left to compute: decoding, or about to finish their prompt
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            if len(seq) - seq.num_computed_tokens > 1:  # a prompt in progress; its blocks came with admission
                i += 1
                continue
            if self.kv.can_append(seq):
                self.kv.append(seq)
                add(seq, 1)
                i += 1
                continue
            victim = self.running.pop()  # out of blocks: preempt the most recently admitted sequence
            self._preempt(victim)
            if victim is seq:
                break
        budget = self.max_num_batched_tokens - len(plan)
        for seq in self.running:
            left = len(seq) - seq.num_computed_tokens
            if left > 1 and budget > 0:
                add(seq, min(left, budget))
                budget -= plan[-1][1]
        while self.waiting and budget > 0 and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            if not self.kv.can_admit(seq, len(self.running)):
                break
            self.waiting.popleft()
            self.kv.admit(seq)
            seq.status = Status.RUNNING
            seq.num_computed_tokens = seq.num_cached_tokens
            self.running.append(seq)
            add(seq, min(len(seq) - seq.num_computed_tokens, budget))
            budget -= plan[-1][1]
        return plan

    def _preempt(self, seq: Sequence):
        self.kv.release(seq)
        seq.num_computed_tokens = 0
        seq.draft_tokens, seq.draft_probs = [], None
        seq.status = Status.WAITING
        seq.num_preemptions += 1
        self.num_preemptions += 1
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence):
        self.kv.release(seq)
        seq.status = Status.FINISHED
        # the draft distributions are a view of their whole step's tensor, which they would keep alive
        seq.draft_tokens, seq.draft_probs = [], None
        self.running.remove(seq)
        if not self.running:
            self.filling = True
