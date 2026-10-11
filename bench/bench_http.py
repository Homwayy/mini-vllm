"""Online latency through the HTTP server (E11): bench_online's workload and arrival times, sent to the server
as streamed /v1/completions requests and timed on the client.

Each request is sent at its arrival time with its prompt as token ids, max_tokens, the workload's temperature
and ignore_eos, and asks for continuous usage stats, so every event says how many tokens it carries. TTFT,
TPOT, ITL and end-to-end latency are measured as in bench_online, from the scheduled arrival time, but to when
the events reach the client. A rate of inf sends every request at t = 0.

With --launch the script starts the server (python -m minivllm.server --record-steps, plus --server-args),
waits for /health, and stops it at the end, so one command measures a whole configuration. After each rate
it takes the server's step records (GET /debug/steps) for the engine-side numbers: decode step time, batch.

Run from the repo root, e.g.:
    python -m bench.bench_http --launch --rates 1 2 4 8 inf --out results/e11_http_online.json
"""

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
import time

import aiohttp
import numpy as np

from bench.workloads import load_workload


async def wait_until(t):
    """asyncio's clock on Windows (GetTickCount64) ticks every 15.6 ms, and a sleep ends 5-16 ms off: sleep to
    30 ms before t, then keep yielding to the event loop until t."""
    if t - time.perf_counter() > 0.03:
        await asyncio.sleep(t - time.perf_counter() - 0.03)
    while time.perf_counter() < t:
        await asyncio.sleep(0)


async def one_request(session, url, req, payload, start):
    await wait_until(start)
    sent = time.perf_counter()
    times, count = [], 0
    async with session.post(url, json={**payload, "prompt": req["prompt_token_ids"], "max_tokens": req["max_tokens"]}) as resp:
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {await resp.text()}")
        async for line in resp.content:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            now = time.perf_counter()
            n = json.loads(data)["usage"]["completion_tokens"]
            times += [now] * (n - count)  # tokens that arrive together share a time, as in the engine's records
            count = n
    return {"arrival": start, "sent": sent, "token_times": times, "prompt_len": len(req["prompt_token_ids"])}


def step_stats(steps, elapsed):
    """The engine-side fields of bench_offline.summarize, from the server's step records."""
    dec = [r for r in steps if r[5]]  # steps that decode
    dur = [r[1] - r[0] for r in dec]
    return {
        "prefill_steps": sum(bool(r[2]) for r in steps), "decode_steps": len(dec),
        "prefill_time_share": sum(r[1] - r[0] for r in steps if r[2]) / elapsed,
        "decode_step_ms_mean": 1e3 * float(np.mean(dur)),
        "decode_batch_mean": float(np.average([r[5] for r in dec], weights=dur)),
    }


def summarize(results, t0):
    def pct(x, q):
        return float(np.percentile(x, q))

    finish = [r["token_times"][-1] for r in results]
    elapsed = max(finish) - t0
    out = sum(len(r["token_times"]) for r in results)
    prompt = sum(r["prompt_len"] for r in results)
    ttft = [r["token_times"][0] - r["arrival"] for r in results]
    tpot = [(r["token_times"][-1] - r["token_times"][0]) / (len(r["token_times"]) - 1) for r in results if len(r["token_times"]) > 1]
    itl = np.concatenate([np.diff(r["token_times"]) for r in results])
    e2e = [f - r["arrival"] for f, r in zip(finish, results)]
    late = [r["sent"] - r["arrival"] for r in results]
    return {
        "elapsed_s": elapsed, "prompt_tokens": prompt, "output_tokens": out,
        "output_throughput": out / elapsed, "total_throughput": (prompt + out) / elapsed,
        "ttft_p50_s": pct(ttft, 50), "ttft_p99_s": pct(ttft, 99),
        "tpot_p50_ms": 1e3 * pct(tpot, 50), "tpot_p99_ms": 1e3 * pct(tpot, 99),
        "itl_p50_ms": 1e3 * pct(itl, 50), "itl_p99_ms": 1e3 * pct(itl, 99), "itl_max_ms": 1e3 * float(itl.max()),
        "e2e_p50_s": pct(e2e, 50), "e2e_p99_s": pct(e2e, 99),
        # how late the client sent requests after their arrival times (its own event loop being busy)
        "send_late_p50_ms": 1e3 * pct(late, 50), "send_late_max_ms": 1e3 * max(late),
    }


async def run(args, reqs, arrivals_rate1, temperature, base):
    url = base + "/v1/completions"
    payload = {"temperature": temperature, "ignore_eos": True, "stream": True, "stream_options": {"continuous_usage_stats": True}}
    if args.model_name:
        payload["model"] = args.model_name
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None)) as session:
        # a few short requests first, so the server's request path has run once
        warm = [{"prompt_token_ids": r["prompt_token_ids"], "max_tokens": 8} for r in reqs[:8]]
        await asyncio.gather(*(one_request(session, url, r, payload, 0.0) for r in warm))

        async def take_steps():  # the server's step records since the last call, if it keeps them
            async with session.get(base + "/debug/steps") as resp:
                return (await resp.json())["steps"] if resp.status == 200 else None

        await take_steps()
        out = []
        for rate in args.rates:
            t0 = time.perf_counter() + 0.1
            results = await asyncio.gather(*(one_request(session, url, r, payload, t0 + a / rate) for r, a in zip(reqs, arrivals_rate1)))
            bad = [i for i, (r, q) in enumerate(zip(results, reqs)) if len(r["token_times"]) != q["max_tokens"]]
            if bad:
                raise RuntimeError(f"requests {bad[:5]} got the wrong number of tokens")
            s = summarize(results, t0)
            steps = await take_steps()
            if steps is not None:
                s.update(step_stats(steps, s["elapsed_s"]))
            s["rate"] = rate
            s["requests"] = {
                "prompt_len": [r["prompt_len"] for r in results], "output_len": [len(r["token_times"]) for r in results],
                "arrival_s": [r["arrival"] - t0 for r in results], "first_token_s": [r["token_times"][0] - t0 for r in results],
                "finish_s": [r["token_times"][-1] - t0 for r in results],
            }
            out.append(s)
            print(f"rate {rate:g}/s: {s['elapsed_s']:.1f} s, output {s['output_throughput']:.0f} tok/s, "
                  f"TTFT p50 {s['ttft_p50_s']:.3f} s p99 {s['ttft_p99_s']:.2f} s, TPOT p50 {s['tpot_p50_ms']:.1f} ms p99 {s['tpot_p99_ms']:.1f} ms, "
                  f"ITL p99 {s['itl_p99_ms']:.1f} ms max {s['itl_max_ms']:.0f} ms, sends late p50 {s['send_late_p50_ms']:.2f} ms "
                  f"max {s['send_late_max_ms']:.1f} ms"
                  + (f", decode step {s['decode_step_ms_mean']:.2f} ms, batch mean {s['decode_batch_mean']:.1f}" if "decode_steps" in s else ""), flush=True)
        return out


def wait_ready(base, proc, timeout):
    import urllib.request

    t = time.perf_counter()
    while time.perf_counter() - t < timeout:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"the server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(base + "/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.5)
    raise RuntimeError("the server did not become ready")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="W1")
    ap.add_argument("--rates", type=float, nargs="+", default=[1, 2, 4, 8], help="requests per second; inf sends all at t = 0")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N requests (smoke tests)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-name", default=None, help="the served model name to send (optional)")
    ap.add_argument("--launch", action="store_true", help="start the server here, and stop it at the end")
    ap.add_argument("--server-args", default="", help="more arguments for python -m minivllm.server, as one string")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    w = load_workload(args.workload)
    reqs = w["requests"][: args.limit or None]
    arrivals = w["arrival_times_rate1"][: len(reqs)]
    base = f"http://{args.host}:{args.port}"
    server_args = ["--host", args.host, "--port", str(args.port), "--warmup-temperature", str(w["sampling"]["temperature"]),
                   "--record-steps", *shlex.split(args.server_args)]
    proc = None
    if args.launch:
        proc = subprocess.Popen([sys.executable, "-m", "minivllm.server", *server_args])
    try:
        wait_ready(base, proc, timeout=600)
        results = asyncio.run(run(args, reqs, arrivals, w["sampling"]["temperature"], base))
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "server_args": server_args if args.launch else None, "workload": f"{w['name']} requests, Poisson arrivals, over HTTP",
            "num_requests": len(reqs), "temperature": w["sampling"]["temperature"], "rates": results,
        }, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
