"""Fixed benchmark workloads, generated once with a fixed seed and saved under bench/workloads/.

W1  offline throughput: 256 requests with random-token prompts of 100-1024 tokens and 100-1024 output
    tokens, EOS ignored. Same setup as nano-vllm's benchmark (token ids in [0, 10000), temperature 0.6).
W2  online latency: W1's requests arriving as a Poisson process. Arrival times are stored for a rate of
    1 request/s; for rate r they are divided by r, so every rate replays the same requests in the same order.
W3  shared prefix: 64 requests sharing a 1000-token prefix, each followed by 50-200 tokens of its own,
    with 50-200 output tokens.

Run from the repo root: python -m bench.workloads
"""

import json
import os

import numpy as np

OUT_DIR = os.path.join("bench", "workloads")
SEED = 0
TOKEN_ID_HIGH = 10000
SAMPLING = {"temperature": 0.6, "ignore_eos": True}


def random_ids(rng, n):
    return rng.integers(0, TOKEN_ID_HIGH, int(n)).tolist()


def make_w1(rng):
    n = 256
    prompt_lens = rng.integers(100, 1025, n)
    output_lens = rng.integers(100, 1025, n)
    requests = [{"prompt_token_ids": random_ids(rng, p), "max_tokens": int(o)} for p, o in zip(prompt_lens, output_lens)]
    gaps = rng.exponential(1.0, n)
    gaps[0] = 0.0  # the first request arrives at t = 0
    return {
        "name": "W1",
        "description": "offline throughput; W2 replays these requests at arrival_times_rate1 / rate",
        "sampling": SAMPLING,
        "requests": requests,
        "arrival_times_rate1": np.cumsum(gaps).round(6).tolist(),
    }


def make_w3(rng):
    n, prefix_len = 64, 1000
    prefix = random_ids(rng, prefix_len)
    requests = [
        {"prompt_token_ids": prefix + random_ids(rng, rng.integers(50, 201)), "max_tokens": int(rng.integers(50, 201))}
        for _ in range(n)
    ]
    return {
        "name": "W3",
        "description": f"{n} requests sharing a {prefix_len}-token prefix",
        "shared_prefix_len": prefix_len,
        "sampling": SAMPLING,
        "requests": requests,
    }


def load_workload(name):
    with open(os.path.join(OUT_DIR, f"{name.lower()}.json")) as f:
        return json.load(f)


def summarize(w):
    p = [len(r["prompt_token_ids"]) for r in w["requests"]]
    o = [r["max_tokens"] for r in w["requests"]]
    return (
        f"{w['name']}: {len(p)} requests, prompt tokens {sum(p)} (min {min(p)}, max {max(p)}), "
        f"output tokens {sum(o)} (min {min(o)}, max {max(o)}), longest request {max(a + b for a, b in zip(p, o))}"
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    # an independent random stream per workload, so adding one later leaves the others unchanged
    for k, make in ((1, make_w1), (3, make_w3)):
        w = make(np.random.default_rng([SEED, k]))
        with open(os.path.join(OUT_DIR, f"w{k}.json"), "w") as f:
            json.dump(w, f, separators=(",", ":"))
        print(summarize(w))


if __name__ == "__main__":
    main()
