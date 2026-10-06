import time
from dataclasses import dataclass

import torch

from minivllm.kv_manager import ContiguousKV, PagedKV, cdiv
from minivllm.model_runner import ModelRunner
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence
from minivllm.utils import cap_gpu_memory


@dataclass
class EngineConfig:
    model: str = "qwen3"  # "qwen3" or "gpt2"
    model_path: str | None = None
    attention: str = "flex"  # "flex" or "gather"
    kv: str = "paged"  # "paged", or "contiguous": a fixed max_model_len region per sequence
    admission: str = "reserve"  # paged only: "reserve" blocks for prompt + max_tokens, or "optimistic"
    batching: str = "continuous"  # "continuous" or "static"
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
    kv_used_tokens: int  # tokens whose K/V is in the pool after this step
    kv_allocated_slots: int


class LLMEngine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        torch.cuda.init()
        cap_gpu_memory()
        self.runner = ModelRunner(cfg)
        self.reset()

    def reset(self):
        cfg = self.cfg
        if cfg.kv == "contiguous":
            kv = ContiguousKV(cfg.num_kv_blocks, cfg.block_size, cfg.max_model_len)
        else:
            kv = PagedKV(cfg.num_kv_blocks, cfg.block_size, reserve_full_length=cfg.admission == "reserve")
        self.kv = kv
        self.scheduler = Scheduler(kv, cfg.max_num_seqs, cfg.max_num_batched_tokens, cfg.batching == "continuous")
        self.steps: list[StepRecord] = []

    def add_request(self, prompt_token_ids, params: SamplingParams, arrival_time=None) -> Sequence:
        cfg = self.cfg
        n = len(prompt_token_ids)
        if n + params.max_tokens > cfg.max_model_len:
            raise ValueError(f"prompt ({n}) + max_tokens ({params.max_tokens}) exceeds max_model_len {cfg.max_model_len}")
        if n > cfg.max_num_batched_tokens:
            raise ValueError(f"prompt ({n}) exceeds max_num_batched_tokens {cfg.max_num_batched_tokens}")
        if cdiv(n + params.max_tokens, cfg.block_size) > cfg.num_kv_blocks:
            raise ValueError("request does not fit in the KV pool")
        seq = Sequence(prompt_token_ids, params, time.perf_counter() if arrival_time is None else arrival_time)
        self.scheduler.add(seq)
        return seq

    def has_unfinished(self):
        return self.scheduler.has_unfinished()

    def step(self) -> list[Sequence]:
        """Runs one prefill or decode step; returns the sequences that finished in it."""
        seqs, is_prefill = self.scheduler.schedule()
        if not seqs:
            return []
        start = time.perf_counter()
        tokens = self.runner.run(seqs, is_prefill)
        now = time.perf_counter()
        num_tokens = sum(len(s) for s in seqs) if is_prefill else len(seqs)
        kv_used = sum(len(s) for s in self.scheduler.running)
        kv_allocated = self.kv.allocated_slots()
        finished = []
        for seq, tok in zip(seqs, tokens):
            seq.token_ids.append(tok)
            if seq.first_token_time is None:
                seq.first_token_time = now
            done = seq.num_completion_tokens >= seq.params.max_tokens
            if done or (not seq.params.ignore_eos and tok in self.runner.eos_token_ids):
                seq.finish_time = now
                self.scheduler.finish(seq)
                finished.append(seq)
        self.steps.append(StepRecord(start, now, is_prefill, len(seqs), num_tokens, kv_used, kv_allocated))
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
