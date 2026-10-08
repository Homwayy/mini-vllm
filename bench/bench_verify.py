"""Cost of deciding one speculative step (E8b.1): verify_drafts alone, on random logits of GPT-2's shape.

Each sequence has 4 drafts. At temperature 1.0 they are sampled from an EAGLE-style q over the draft
vocabulary of the trained draft model; at temperature 0 they are checked greedily. Compares the fused kernels
(verify_drafts) with the full path (_verify_full), which builds every row's p and q over the whole
vocabulary and was the only path before E8b.1. Reports per call: the wall time (the call ends with a copy to
the CPU, so its GPU work is inside), the number of kernels and their GPU time. The wall time is measured in
--rounds rounds that alternate the two paths, each the median of --reps calls; the report gives the median
round and the range, since single rounds of these sub-millisecond calls vary by up to 0.1 ms.

Run from the repo root: python -m bench.bench_verify --out results/e8b1_verify.json
"""

import argparse
import json
import os
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

from minivllm.model_runner import MODELS
from minivllm.sampler import Sampler
from minivllm.sequence import SamplingParams
from minivllm.spec.rejection import _verify_full, verify_drafts

V = 50304  # GPT-2's LM head rows; the sampler's vocab limit is 50257
NUM_DRAFTS = 4


def inputs(batch, temperature, vocab):
    logits = (torch.randn(batch * (NUM_DRAFTS + 1), V, device="cuda") * 3).to(torch.bfloat16)
    params = [SamplingParams(temperature=temperature)] * batch
    q = torch.softmax(torch.randn(batch * NUM_DRAFTS, len(vocab), device="cuda") * 3, -1)
    drafts = vocab[torch.multinomial(q, 1).squeeze(1)].view(batch, NUM_DRAFTS).tolist()
    return (logits, params, drafts) + ((q, vocab) if temperature > 0 else (None, None))


def wall_ms(fn, reps):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1e3 * sorted(ts)[len(ts) // 2]


def kernels(fn, reps=20):
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    ev = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    return len(ev) / reps, sum(e.device_time for e in ev) / reps / 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(0)
    ckpt = torch.load(os.path.join("checkpoints", "gpt2_eagle.pt"), map_location="cpu")
    vocab = torch.tensor(ckpt["vocab_ids"], device="cuda")
    sampler = Sampler(MODELS["gpt2"]["vocab_limit"])
    rows = []
    for temperature in (0.0, 1.0):
        for batch in args.batch:
            x = inputs(batch, temperature, vocab)
            paths = {"full": _verify_full, "fused": verify_drafts}
            rounds = {name: [] for name in paths}
            for _ in range(args.rounds):
                for name, fn in paths.items():
                    rounds[name].append(wall_ms(lambda: fn(sampler, *x), args.reps))
            for name, fn in paths.items():
                ms = sorted(rounds[name])
                n, gpu_ms = kernels(lambda: fn(sampler, *x))
                rows.append({"temperature": temperature, "batch": batch, "rows": batch * (NUM_DRAFTS + 1), "path": name,
                             "ms_per_call": ms[len(ms) // 2], "ms_per_call_rounds": rounds[name], "kernels": n,
                             "gpu_ms": gpu_ms})
                print(f"temperature {temperature} batch {batch:2d} {name:5s}: {ms[len(ms) // 2]:.3f} ms per call "
                      f"(rounds {ms[0]:.3f}-{ms[-1]:.3f}), {n:.0f} kernels, GPU {gpu_ms:.3f} ms", flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"num_drafts": NUM_DRAFTS, "vocab": V, "draft_vocab": len(vocab), "gpu": torch.cuda.get_device_name(),
                   "torch": torch.__version__, "results": rows}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
