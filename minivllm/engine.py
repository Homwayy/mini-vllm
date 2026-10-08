import os
import time
from dataclasses import dataclass

import torch

from minivllm.kv_manager import ContiguousKV, PagedKV, PrefixCachingKV, cdiv
from minivllm.model_runner import ModelRunner
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence, Status
from minivllm.spec.ngram import NgramProposer
from minivllm.utils import cap_gpu_memory


@dataclass
class EngineConfig:
    model: str = "qwen3"  # "qwen3" or "gpt2"
    model_path: str | None = None
    # "triton" (Triton paged prefill and decode), "triton-decode" (Triton decode, flex prefill), "flex" or "gather"
    attention: str = "triton"
    kv: str = "paged"  # "paged", or "contiguous": a fixed max_model_len region per sequence
    admission: str = "reserve"  # paged only: "reserve", "optimistic" or "headroom" (see PagedKV)
    batching: str = "continuous"  # "continuous" or "static"
    prefix_caching: bool = False  # share full blocks between sequences with the same prefix (paged, triton)
    # mix prompt chunks with decoding in steps of at most max_num_batched_tokens (paged, triton, continuous)
    chunked_prefill: bool = False
    cuda_graph: bool = True  # capture the decode step per batch-size bucket
    # speculative decoding: up to num_spec_tokens draft tokens per sequence and decode step, which the target
    # model checks all in that step (attention="triton", without prefix caching or chunked prefill). "ngram"
    # copies them from earlier in the sequence; "eagle" drafts with the model in draft_path (gpt2 only).
    spec: str | None = None
    num_spec_tokens: int = 4
    draft_path: str = os.path.join("checkpoints", "gpt2_eagle.pt")
    block_size: int = 16
    num_kv_blocks: int = 4096
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 16384
    max_model_len: int = 2048


@dataclass
class StepRecord:
    start: float
    end: float
    is_prefill: bool
    num_seqs: int
    num_tokens: int
    # tokens of the running sequences; with chunked prefill this includes prompt tokens not computed yet,
    # whose blocks were allocated at admission
    kv_used_tokens: int
    kv_allocated_slots: int
    num_decode: int  # sequences that computed one token and got their next token from it
    num_draft: int = 0  # speculative decoding: draft tokens checked in this step
    num_accepted: int = 0  # and accepted


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        # the other backends attend to the new tokens only, so they cannot use a cached prefix or earlier chunks
        if cfg.prefix_caching and (cfg.kv != "paged" or cfg.attention != "triton"):
            raise ValueError("prefix_caching needs kv='paged' and attention='triton'")
        if cfg.chunked_prefill and (cfg.kv != "paged" or cfg.attention != "triton" or cfg.batching != "continuous"):
            raise ValueError("chunked_prefill needs kv='paged', attention='triton' and batching='continuous'")
        if cfg.chunked_prefill and cfg.max_num_batched_tokens < cfg.max_num_seqs:
            raise ValueError("chunked_prefill needs max_num_batched_tokens >= max_num_seqs, so every decode fits")
        if cfg.spec not in (None, "ngram", "eagle"):
            raise ValueError(f"unknown spec method {cfg.spec}")
        if cfg.spec == "eagle" and cfg.model != "gpt2":
            raise ValueError("the EAGLE draft model is trained for model='gpt2'")
        # drafts are checked by the paged prefill kernel, which reads the context from the pool
        if cfg.spec and (cfg.attention != "triton" or cfg.prefix_caching or cfg.chunked_prefill):
            raise ValueError("spec needs attention='triton', and is not combined with prefix caching or chunked prefill")
        self.cfg = cfg
        torch.cuda.init()
        cap_gpu_memory()
        self.runner = ModelRunner(cfg)
        self.proposer = NgramProposer() if cfg.spec == "ngram" else None
        self.reset()

    def reset(self):
        cfg = self.cfg
        if cfg.kv == "contiguous":
            kv = ContiguousKV(cfg.num_kv_blocks, cfg.block_size, cfg.max_model_len)
        elif cfg.prefix_caching:
            kv = PrefixCachingKV(cfg.num_kv_blocks, cfg.block_size, cfg.admission)
        else:
            kv = PagedKV(cfg.num_kv_blocks, cfg.block_size, cfg.admission)
        self.kv = kv
        self.scheduler = Scheduler(kv, cfg.max_num_seqs, cfg.max_num_batched_tokens, cfg.batching == "continuous")
        self.steps: list[StepRecord] = []

    def add_request(self, prompt_token_ids, params: SamplingParams, arrival_time=None) -> Sequence:
        cfg = self.cfg
        n = len(prompt_token_ids)
        if n + params.max_tokens > cfg.max_model_len:
            raise ValueError(f"prompt ({n}) + max_tokens ({params.max_tokens}) exceeds max_model_len {cfg.max_model_len}")
        if n > cfg.max_num_batched_tokens and not cfg.chunked_prefill:
            raise ValueError(f"prompt ({n}) exceeds max_num_batched_tokens {cfg.max_num_batched_tokens}")
        if cdiv(n + params.max_tokens, cfg.block_size) > cfg.num_kv_blocks:
            raise ValueError("request does not fit in the KV pool")
        seq = Sequence(prompt_token_ids, params, time.perf_counter() if arrival_time is None else arrival_time)
        self.scheduler.add(seq)
        return seq

    def has_unfinished(self):
        return self.scheduler.has_unfinished()

    def step(self) -> list[Sequence]:
        """Runs one step; returns the sequences that finished in it."""
        if self.cfg.chunked_prefill:
            return self._step_chunked()
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []
        if self.cfg.spec and not is_prefill:
            return self._step_spec(seqs)
        start = time.perf_counter()
        tokens = self.runner.run(seqs, is_prefill)
        now = time.perf_counter()
        num_tokens = sum(len(s) - s.num_cached_tokens for s in seqs) if is_prefill else len(seqs)
        record = StepRecord(start, now, is_prefill, len(seqs), num_tokens, self._kv_used(), self.kv.allocated_slots(),
                            0 if is_prefill else len(seqs))
        for seq in seqs:
            seq.num_computed_tokens = len(seq)
        finished = self._append(seqs, tokens, now)
        if self.cfg.spec == "eagle":  # a prefill: the draft model's prefill, and the first drafts
            self._draft([(i, seq, len(seq) - 1) for i, seq in enumerate(seqs) if seq.status == Status.RUNNING])
            record.end = time.perf_counter()
        self.steps.append(record)
        return finished

    def _step_chunked(self):
        plan = self.scheduler.schedule_chunked()
        if not plan:
            return []
        start = time.perf_counter()
        tokens = self.runner.run_chunked(plan)
        now = time.perf_counter()
        record = StepRecord(start, now, any(n > 1 for _, n in plan), len(plan), sum(n for _, n in plan),
                            self._kv_used(), self.kv.allocated_slots(), sum(n == 1 for _, n in plan))
        for seq, n in plan:
            seq.num_computed_tokens += n
        # a sequence in the middle of its prompt gets no token
        done = [(seq, tok) for (seq, _), tok in zip(plan, tokens) if tok is not None]
        finished = self._append([s for s, _ in done], [t for _, t in done], now)
        self.steps.append(record)
        return finished

    def _step_spec(self, seqs):
        drafts = []
        for seq in seqs:
            k = self._max_drafts(seq)
            if self.proposer is not None:
                d = self.proposer.propose(seq, k) if k > 0 else []
            else:
                d = seq.draft_tokens[:k]  # EAGLE: drafted at the end of this sequence's previous step
            while d and not self.kv.can_append(seq, len(d)):  # drafts only take blocks that are free
                d.pop()
            self.kv.append(seq, len(d) + self._chain_room(seq, len(d)))
            drafts.append(d)
        probs = None
        if self.cfg.spec == "eagle" and any(seq.params.temperature > 0 for seq in seqs):
            probs = [seq.draft_probs for seq in seqs]
        start = time.perf_counter()
        tokens = self.runner.run_verify(seqs, drafts, probs)
        now = time.perf_counter()
        record = StepRecord(start, now, False, len(seqs), sum(len(d) + 1 for d in drafts), self._kv_used(),
                            self.kv.allocated_slots(), len(seqs), sum(map(len, drafts)), sum(len(t) - 1 for t in tokens))
        finished, verified = [], []
        for i, (seq, toks) in enumerate(zip(seqs, tokens)):
            for tok in toks:
                if self._append_token(seq, tok, now):
                    finished.append(seq)
                    break
            else:
                seq.num_computed_tokens = len(seq) - 1
                verified.append((i, seq, len(toks)))  # the rows of its last token and its accepted drafts
        if self.cfg.spec == "eagle":
            self._draft(verified, after_verify=True)
            record.end = time.perf_counter()
        else:
            for _, seq, _ in verified:
                self.kv.truncate(seq)
        self.steps.append(record)
        return finished

    def _max_drafts(self, seq):
        # never more tokens than max_tokens allows: the drafts plus one from the target model
        return min(self.cfg.num_spec_tokens, seq.params.max_tokens - seq.num_completion_tokens - 1)

    def _chain_room(self, seq, num_drafts):
        """EAGLE drafts right after the verify step, with the blocks seq has in it (ModelRunner.draft_after_verify);
        the chain writes up to num_spec_tokens - 1 positions past the drafts. Returns how many of those
        positions to allocate now: all that max_len and the free blocks allow."""
        if self.cfg.spec != "eagle":
            return 0
        extra = max(0, min(self.cfg.num_spec_tokens - 1, seq.max_len - len(seq) - num_drafts))
        while extra and not self.kv.can_append(seq, num_drafts + extra):
            extra -= 1
        return extra

    def _draft(self, entries, after_verify=False):
        """EAGLE: entries is [(row of the sequence in this step, sequence, number of its verified rows)]. Catches
        the draft model up on those rows and drafts the sequences' next tokens, with blocks reserved for them."""
        keep = []
        for i, seq, n in entries:
            k = self._max_drafts(seq)
            if after_verify:  # draft m reads the chain's K/V up to position len(seq) + m - 3, in the step's blocks
                k = min(k, len(seq.block_table) * self.cfg.block_size - len(seq) + 2)
            while k > 0 and not self.kv.can_append(seq, k):
                k -= 1
            self.kv.truncate(seq, k)
            seq.draft_tokens, seq.draft_probs = [], None
            if k > 0:  # without drafts, the block for the next token is left to the scheduler, which can preempt
                self.kv.append(seq, k)
                keep.append((i, n, seq, k))
        if keep:
            drafts, probs = self.runner.draft_next([(i, n) for i, n, _, _ in keep])
            for j, (_, _, seq, k) in enumerate(keep):
                seq.draft_tokens = drafts[j][:k]
                seq.draft_probs = None if probs is None else probs[j]

    def _kv_used(self):
        return sum(len(s) for s in self.scheduler.running)  # before this step's new tokens are appended

    def _append(self, seqs, tokens, now):
        return [seq for seq, tok in zip(seqs, tokens) if self._append_token(seq, tok, now)]

    def _append_token(self, seq, tok, now):
        """Appends a generated token; returns whether seq finished with it."""
        seq.token_ids.append(tok)
        seq.token_times.append(now)
        if seq.first_token_time is None:
            seq.first_token_time = now
        done = seq.num_completion_tokens >= seq.params.max_tokens
        if done or (not seq.params.ignore_eos and tok in self.runner.eos_token_ids):
            seq.finish_time = now
            self.scheduler.finish(seq)
            return True
        return False

    def generate(self, prompts, params) -> list[Sequence]:
        """Offline use: runs all prompts to completion. params is one SamplingParams or one per prompt."""
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        seqs = [self.add_request(p, sp) for p, sp in zip(prompts, params)]
        while self.has_unfinished():
            self.step()
        return seqs

    def warmup(self, temperature=0.6):
        """Compiles every decode batch-size bucket and the prefill path, then clears all state.

        Sequence i generates i + 1 tokens, so the decode batch shrinks by one every step and passes
        through every batch size the configuration can reach.
        """
        n = self.cfg.max_num_seqs
        prompts = [[(7 * i + j) % 1000 for j in range(16 + i % 48)] for i in range(n)]
        params = [SamplingParams(temperature=temperature, max_tokens=i + 1, ignore_eos=True) for i in range(n)]
        self.generate(prompts, params)
        torch.cuda.synchronize()
        self.reset()
