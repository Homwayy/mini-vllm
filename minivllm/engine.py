import time
from dataclasses import dataclass

import torch

from minivllm.kv_manager import ContiguousKV, PagedKV, PrefixCachingKV, cdiv
from minivllm.model_runner import ModelRunner
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence
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


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        # the other backends attend to the new tokens only, so they cannot use a cached prefix or earlier chunks
        if cfg.prefix_caching and (cfg.kv != "paged" or cfg.attention != "triton"):
            raise ValueError("prefix_caching needs kv='paged' and attention='triton'")
        if cfg.chunked_prefill and (cfg.kv != "paged" or cfg.attention != "triton" or cfg.batching != "continuous"):
            raise ValueError("chunked_prefill needs kv='paged', attention='triton' and batching='continuous'")
        if cfg.chunked_prefill and cfg.max_num_batched_tokens < cfg.max_num_seqs:
            raise ValueError("chunked_prefill needs max_num_batched_tokens >= max_num_seqs, so every decode fits")
        self.cfg = cfg
        torch.cuda.init()
        cap_gpu_memory()
        self.runner = ModelRunner(cfg)
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
        start = time.perf_counter()
        tokens = self.runner.run(seqs, is_prefill)
        now = time.perf_counter()
        num_tokens = sum(len(s) - s.num_cached_tokens for s in seqs) if is_prefill else len(seqs)
        record = StepRecord(start, now, is_prefill, len(seqs), num_tokens, self._kv_used(), self.kv.allocated_slots(),
                            0 if is_prefill else len(seqs))
        for seq in seqs:
            seq.num_computed_tokens = len(seq)
        finished = self._append(seqs, tokens, now)
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

    def _kv_used(self):
        return sum(len(s) for s in self.scheduler.running)  # before this step's new tokens are appended

    def _append(self, seqs, tokens, now):
        finished = []
        for seq, tok in zip(seqs, tokens):
            seq.token_ids.append(tok)
            seq.token_times.append(now)
            if seq.first_token_time is None:
                seq.first_token_time = now
            done = seq.num_completion_tokens >= seq.params.max_tokens
            if done or (not seq.params.ignore_eos and tok in self.runner.eos_token_ids):
                seq.finish_time = now
                self.scheduler.finish(seq)
                finished.append(seq)
        return finished

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
