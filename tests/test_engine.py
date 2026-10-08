"""The engine end to end under each KV, admission and batching policy, checked for self-consistency.

Different schedules put a sequence in different batches, and batch composition changes bf16 results
slightly, so outputs are not compared across policies. Instead every greedy completion is replayed through
a reference forward (gather backend, one prefill of prompt + completion), and each generated token must be
the reference argmax up to bf16 noise. A KV bookkeeping error (a wrong slot, block table, or recompute
after preemption) produces tokens far from the argmax.

Decode runs in CUDA graphs. Replaying a graph runs the same kernels on the same padded shapes as the eager
path, so for each policy the eager engine must produce exactly the same tokens. Prefill steps of up to
prefill_graph_tokens tokens are padded and run in CUDA graphs too; larger ones, such as the first, run eagerly.

Prefix caching runs on prompts built from a few shared prefixes: one of exactly 3 blocks (so some prompts
are entirely cached and their last block is computed again), one of 6 blocks and 4 tokens, and one shorter
than a block; some prompts are duplicates. A small pool adds preemption and eviction.

Chunked prefill runs with a budget of 64 tokens per step, so prompts of up to 300 tokens are split into
several chunks and mixed with decoding; mixed steps run in CUDA graphs too, so the eager comparison covers
them. A small pool preempts sequences in the middle of their prompts.

Speculative decoding (n-gram drafts) runs on prompts that repeat a segment, so drafts are found; the check
requires both accepted and rejected drafts. Verify steps run in CUDA graphs too. EAGLE and DSpark drafting run
if their trained draft models (checkpoints/gpt2_eagle.pt, checkpoints/gpt2_dspark.pt) exist; EAGLE's catch-up
and draft chain, and DSpark's block, are captured as well.

Run from the repo root: python -m tests.test_engine
"""

import os
import random

import torch

from minivllm.attention import AttnMetadata
from minivllm.attention.gather import GatherAttention
from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams
from tests.common import PagedSeqs

MAX_REGRET = 1.0  # bf16 noise moves logits by a few tenths; corrupted K/V gives regrets of several units
PREFIX_CASES = [
    dict(kv="paged", admission="reserve", batching="continuous", prefix_caching=True),
    dict(kv="paged", admission="optimistic", batching="continuous", num_kv_blocks=48, prefix_caching=True),
]
CHUNKED_CASES = [
    dict(kv="paged", admission="reserve", batching="continuous", chunked_prefill=True),
    dict(kv="paged", admission="optimistic", batching="continuous", num_kv_blocks=48, chunked_prefill=True),
]
CASES = [
    dict(kv="contiguous", batching="static"),
    dict(kv="paged", admission="reserve", batching="static"),
    dict(kv="paged", admission="reserve", batching="continuous"),
    dict(kv="paged", admission="optimistic", batching="static", num_kv_blocks=48),
    dict(kv="paged", admission="optimistic", batching="continuous", num_kv_blocks=48),
    dict(kv="paged", admission="headroom", batching="continuous", num_kv_blocks=48),
]


def make_requests(n, vocab, max_prompt, max_out, seed=0):
    rng = random.Random(seed)
    prompts = [[rng.randrange(vocab) for _ in range(rng.randint(5, max_prompt))] for _ in range(n)]
    params = [SamplingParams(temperature=0.0, max_tokens=rng.randint(5, max_out), ignore_eos=True) for _ in range(n)]
    return prompts, params


def shared_prefix_requests(n, vocab, max_out, seed=2):
    rng = random.Random(seed)
    prefixes = [[rng.randrange(vocab) for _ in range(k)] for k in (48, 100, 7)]
    prompts = [rng.choice(prefixes) + [rng.randrange(vocab) for _ in range(rng.choice([0, 0, 1, 16, rng.randint(2, 60)]))]
               for _ in range(n)]
    params = [SamplingParams(temperature=0.0, max_tokens=rng.randint(5, max_out), ignore_eos=True) for _ in range(n)]
    return prompts, params


def repetitive_requests(n, vocab, max_out, seed=4):
    rng = random.Random(seed)
    prompts = []
    for _ in range(n):
        segment = [rng.randrange(vocab) for _ in range(rng.randint(3, 12))]
        prompts.append(segment * rng.randint(2, 6) + [rng.randrange(vocab) for _ in range(rng.choice([0, 1, 5]))])
    params = [SamplingParams(temperature=0.0, max_tokens=rng.randint(5, max_out), ignore_eos=True) for _ in range(n)]
    return prompts, params


@torch.inference_mode()
def replay_regret(model, seqs):
    """Max regret of the generated tokens under a reference forward, and the share that are exact argmaxes."""
    saved = [(a.attn, a.k_cache, a.v_cache) for a in model.attn_layers]
    gather = GatherAttention()
    for a in model.attn_layers:
        a.attn = gather
    worst, exact, total = 0.0, 0, 0
    try:
        for seq in seqs:
            full = seq.token_ids[:-1]  # the last generated token is never fed back
            paged = PagedSeqs(model, [len(full)], 16, "cuda")
            pos = torch.arange(len(full), device="cuda")
            meta = AttnMetadata(is_prefill=True, slot_mapping=paged.slots(torch.zeros_like(pos), pos), cu_seqlens=[0, len(full)])
            hidden = model(torch.tensor(full, device="cuda"), pos, meta)[seq.num_prompt_tokens - 1 :]
            logits = model.compute_logits(hidden).float()
            chosen = torch.tensor(seq.completion_token_ids, device="cuda")
            regret = logits.max(-1).values - logits.gather(-1, chosen[:, None]).squeeze(-1)
            worst = max(worst, regret.max().item())
            exact += (regret == 0).sum().item()
            total += len(chosen)
    finally:
        for a, (attn, k, v) in zip(model.attn_layers, saved):
            a.attn, a.k_cache, a.v_cache = attn, k, v
    return worst, exact / total


def run_case(engine, prompts, params, label):
    engine.runner.use_cuda_graph = True
    seqs = engine.generate(prompts, params)
    regret, exact = replay_regret(engine.runner.model, seqs)
    lengths_ok = all(s.num_completion_tokens == p.max_tokens for s, p in zip(seqs, params))
    released = engine.kv.allocated_slots() == 0
    prefills = sum(r.is_prefill for r in engine.steps)
    max_batch = max(r.num_decode for r in engine.steps)
    ok = lengths_ok and released and regret <= MAX_REGRET
    cached = sum(s.num_cached_tokens for s in seqs)
    mixed = sum(r.is_prefill and r.num_decode > 0 for r in engine.steps)
    drafted, accepted = sum(r.num_draft for r in engine.steps), sum(r.num_accepted for r in engine.steps)
    if engine.cfg.prefix_caching:
        ok &= cached > 0
    if engine.cfg.chunked_prefill:
        ok &= mixed > 0
    if engine.cfg.spec:
        ok &= 0 < accepted < drafted
    print(
        f"{label:58s} regret max {regret:.3f}, exact argmax {exact:.3f}, preemptions {engine.scheduler.num_preemptions:3d}, "
        f"prefill steps {prefills:3d}, max decode batch {max_batch:2d}, lengths {'ok' if lengths_ok else 'WRONG'}, "
        f"KV {'released' if released else 'LEAKED'}" + (f", cached prompt tokens {cached}" if engine.cfg.prefix_caching else "")
        + (f", mixed steps {mixed}" if engine.cfg.chunked_prefill else "")
        + (f", drafts accepted {accepted} / {drafted}" if engine.cfg.spec else "")
    )
    graph_tokens = [s.token_ids for s in seqs]
    engine.reset()
    engine.runner.use_cuda_graph = False
    same = [s.token_ids for s in engine.generate(prompts, params)] == graph_tokens
    engine.runner.use_cuda_graph = True
    print(f"{'':58s} eager decode gives {'identical' if same else 'DIFFERENT'} tokens")
    return ok and same


def main():
    torch.manual_seed(0)
    ok = True
    cfg = EngineConfig(model="qwen3", num_kv_blocks=1024, max_num_seqs=16, max_num_batched_tokens=4096, max_model_len=512)
    engine = LLMEngine(cfg)
    prompts, params = make_requests(24, 10000, 300, 80)
    for case in CASES:
        engine.cfg = EngineConfig(**{**cfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, prompts, params, f"qwen3 {cfg.attention} {case}")
    # sampling path: lengths only
    engine.cfg = cfg
    engine.reset()
    sampled = [SamplingParams(temperature=0.6, top_p=0.9, max_tokens=p.max_tokens, ignore_eos=True) for p in params]
    seqs = engine.generate(prompts, sampled)
    sample_ok = all(s.num_completion_tokens == p.max_tokens for s, p in zip(seqs, sampled))
    print(f"qwen3 {cfg.attention} sampling (temperature 0.6, top_p 0.9): lengths {'ok' if sample_ok else 'WRONG'}")
    ok &= sample_ok
    shared, shared_params = shared_prefix_requests(24, 10000, 80)
    for case in PREFIX_CASES:
        engine.cfg = EngineConfig(**{**cfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, shared, shared_params, f"qwen3 {cfg.attention} {case}")
    del engine
    torch.cuda.empty_cache()

    chunked_cfg = EngineConfig(**{**cfg.__dict__, "chunked_prefill": True, "max_num_batched_tokens": 64})
    engine = LLMEngine(chunked_cfg)
    for case in CHUNKED_CASES:
        engine.cfg = EngineConfig(**{**chunked_cfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, prompts, params, f"qwen3 {cfg.attention} {case} budget 64")
    engine.cfg = EngineConfig(**{**chunked_cfg.__dict__, **PREFIX_CASES[0], "chunked_prefill": True})
    engine.reset()
    ok &= run_case(engine, shared, shared_params, f"qwen3 {cfg.attention} prefix caching + chunked, budget 64")
    del engine
    torch.cuda.empty_cache()

    for attention in ("gather", "flex"):
        engine = LLMEngine(EngineConfig(**{**cfg.__dict__, "attention": attention}))
        ok &= run_case(engine, prompts, params, f"qwen3 {attention} {{'kv': 'paged', 'batching': 'continuous'}}")
        engine.cfg = EngineConfig(**{**engine.cfg.__dict__, **CASES[4]})
        engine.reset()
        ok &= run_case(engine, prompts, params, f"qwen3 {attention} {CASES[4]}")
        del engine
        torch.cuda.empty_cache()

    gcfg = EngineConfig(model="gpt2", num_kv_blocks=1024, max_num_seqs=16, max_num_batched_tokens=4096, max_model_len=512)
    engine = LLMEngine(gcfg)
    prompts, params = make_requests(24, 50257, 300, 80, seed=1)
    for case in (CASES[2], CASES[4]):
        engine.cfg = EngineConfig(**{**gcfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, prompts, params, f"gpt2 {gcfg.attention} {case}")
    shared, shared_params = shared_prefix_requests(24, 50257, 80, seed=3)
    for case in PREFIX_CASES:
        engine.cfg = EngineConfig(**{**gcfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, shared, shared_params, f"gpt2 {gcfg.attention} {case}")
    del engine
    torch.cuda.empty_cache()
    engine = LLMEngine(EngineConfig(**{**gcfg.__dict__, "chunked_prefill": True, "max_num_batched_tokens": 64}))
    ok &= run_case(engine, prompts, params, f"gpt2 {gcfg.attention} {CHUNKED_CASES[0]} budget 64")
    del engine
    torch.cuda.empty_cache()
    engine = LLMEngine(EngineConfig(**{**gcfg.__dict__, "attention": "flex"}))
    ok &= run_case(engine, prompts, params, "gpt2 flex {'kv': 'paged', 'batching': 'continuous'}")
    del engine
    torch.cuda.empty_cache()

    spec_cfg = EngineConfig(**{**gcfg.__dict__, "spec": "ngram"})
    engine = LLMEngine(spec_cfg)
    rep_prompts, rep_params = repetitive_requests(24, 50257, 80)
    for case in (CASES[2], CASES[4], CASES[1]):
        engine.cfg = EngineConfig(**{**spec_cfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, rep_prompts, rep_params, f"gpt2 {gcfg.attention} {case} spec ngram")
    engine.cfg = spec_cfg
    engine.reset()
    sampled = [SamplingParams(temperature=0.6, top_p=0.9, max_tokens=p.max_tokens, ignore_eos=True) for p in rep_params]
    seqs = engine.generate(rep_prompts, sampled)
    sample_ok = all(s.num_completion_tokens == p.max_tokens for s, p in zip(seqs, sampled))
    accepted = sum(r.num_accepted for r in engine.steps)
    print(f"gpt2 spec ngram sampling (temperature 0.6, top_p 0.9): lengths {'ok' if sample_ok else 'WRONG'}, "
          f"drafts accepted {accepted} / {sum(r.num_draft for r in engine.steps)}")
    ok &= sample_ok and accepted > 0
    del engine
    torch.cuda.empty_cache()
    engine = LLMEngine(EngineConfig(**{**cfg.__dict__, "spec": "ngram"}))
    rep_prompts, rep_params = repetitive_requests(24, 10000, 80, seed=5)
    ok &= run_case(engine, rep_prompts, rep_params, f"qwen3 {cfg.attention} {{'kv': 'paged', 'batching': 'continuous'}} spec ngram")
    del engine
    torch.cuda.empty_cache()

    for spec in ("eagle", "dspark"):
        path = os.path.join("checkpoints", f"gpt2_{spec}.pt")
        if not os.path.exists(path):
            print(f"spec {spec}: skipped, no draft model at {path}")
            continue
        spec_cfg = EngineConfig(**{**gcfg.__dict__, "spec": spec})
        engine = LLMEngine(spec_cfg)
        prompts, params = make_requests(24, 50257, 300, 80, seed=1)
        for case in (CASES[2], CASES[4], CASES[1]):
            engine.cfg = EngineConfig(**{**spec_cfg.__dict__, **case})
            engine.reset()
            ok &= run_case(engine, prompts, params, f"gpt2 {gcfg.attention} {case} spec {spec}")
        engine.cfg = spec_cfg
        # top-p takes the full acceptance path, plain temperature the fused kernels
        for sp in (SamplingParams(temperature=0.8, top_p=0.95), SamplingParams(temperature=1.0)):
            engine.reset()
            sampled = [SamplingParams(temperature=sp.temperature, top_p=sp.top_p, max_tokens=p.max_tokens, ignore_eos=True)
                       for p in params]
            seqs = engine.generate(prompts, sampled)
            sample_ok = all(s.num_completion_tokens == p.max_tokens for s, p in zip(seqs, sampled))
            accepted = sum(r.num_accepted for r in engine.steps)
            dropped = all(s.draft_probs is None for s in seqs)  # finished sequences keep no draft distributions
            print(f"gpt2 spec {spec} sampling (temperature {sp.temperature}, top_p {sp.top_p}): lengths "
                  f"{'ok' if sample_ok else 'WRONG'}, drafts accepted {accepted} / {sum(r.num_draft for r in engine.steps)}, "
                  f"draft distributions {'dropped' if dropped else 'KEPT'}")
            ok &= sample_ok and accepted > 0 and dropped
        del engine
        torch.cuda.empty_cache()
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
