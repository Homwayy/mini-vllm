"""The engine end to end under each KV, admission and batching policy, checked for self-consistency.

Different schedules put a sequence in different batches, and batch composition changes bf16 results
slightly, so outputs are not compared across policies. Instead every greedy completion is replayed through
a reference forward (gather backend, one prefill of prompt + completion), and each generated token must be
the reference argmax up to bf16 noise. A KV bookkeeping error (a wrong slot, block table, or recompute
after preemption) produces tokens far from the argmax.

Run from the repo root: python -m tests.test_engine
"""

import random

import torch

from minivllm.attention import AttnMetadata
from minivllm.attention.gather import GatherAttention
from minivllm.engine import EngineConfig, LLMEngine
from minivllm.sequence import SamplingParams
from tests.common import PagedSeqs

MAX_REGRET = 1.0  # bf16 noise moves logits by a few tenths; corrupted K/V gives regrets of several units
CASES = [
    dict(kv="contiguous", batching="static"),
    dict(kv="paged", admission="reserve", batching="static"),
    dict(kv="paged", admission="reserve", batching="continuous"),
    dict(kv="paged", admission="optimistic", batching="static", num_kv_blocks=48),
    dict(kv="paged", admission="optimistic", batching="continuous", num_kv_blocks=48),
]


def make_requests(n, vocab, max_prompt, max_out, seed=0):
    rng = random.Random(seed)
    prompts = [[rng.randrange(vocab) for _ in range(rng.randint(5, max_prompt))] for _ in range(n)]
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
    seqs = engine.generate(prompts, params)
    regret, exact = replay_regret(engine.runner.model, seqs)
    lengths_ok = all(s.num_completion_tokens == p.max_tokens for s, p in zip(seqs, params))
    released = engine.kv.allocated_slots() == 0
    prefills = sum(r.is_prefill for r in engine.steps)
    max_batch = max(r.num_seqs for r in engine.steps if not r.is_prefill)
    ok = lengths_ok and released and regret <= MAX_REGRET
    print(
        f"{label:58s} regret max {regret:.3f}, exact argmax {exact:.3f}, preemptions {engine.scheduler.num_preemptions:3d}, "
        f"prefill steps {prefills:3d}, max decode batch {max_batch:2d}, lengths {'ok' if lengths_ok else 'WRONG'}, "
        f"KV {'released' if released else 'LEAKED'}"
    )
    return ok


def main():
    torch.manual_seed(0)
    ok = True
    cfg = EngineConfig(model="qwen3", num_kv_blocks=1024, max_num_seqs=16, max_num_batched_tokens=4096, max_model_len=512)
    engine = LLMEngine(cfg)
    prompts, params = make_requests(24, 10000, 300, 80)
    for case in CASES:
        engine.cfg = EngineConfig(**{**cfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, prompts, params, f"qwen3 flex {case}")
    engine.cfg = cfg
    engine.reset()
    flex = engine.runner.model.attn_layers[0].attn
    for a in engine.runner.model.attn_layers:
        a.attn = GatherAttention()
    ok &= run_case(engine, prompts, params, "qwen3 gather {'kv': 'paged', 'batching': 'continuous'}")
    for a in engine.runner.model.attn_layers:
        a.attn = flex

    # sampling path: lengths and token range only
    engine.reset()
    sampled = [SamplingParams(temperature=0.6, top_p=0.9, max_tokens=p.max_tokens, ignore_eos=True) for p in params]
    seqs = engine.generate(prompts, sampled)
    sample_ok = all(s.num_completion_tokens == p.max_tokens for s, p in zip(seqs, sampled))
    print(f"qwen3 flex sampling (temperature 0.6, top_p 0.9): lengths {'ok' if sample_ok else 'WRONG'}")
    ok &= sample_ok
    del engine, flex
    torch.cuda.empty_cache()

    gcfg = EngineConfig(model="gpt2", num_kv_blocks=1024, max_num_seqs=16, max_num_batched_tokens=4096, max_model_len=512)
    engine = LLMEngine(gcfg)
    prompts, params = make_requests(24, 50257, 300, 80, seed=1)
    for case in (CASES[2], CASES[4]):
        engine.cfg = EngineConfig(**{**gcfg.__dict__, **case})
        engine.reset()
        ok &= run_case(engine, prompts, params, f"gpt2 flex {case}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
