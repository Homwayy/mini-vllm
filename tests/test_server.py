"""The HTTP server (minivllm/server.py) end to end, on Qwen3 with a small engine in this process.

- Greedy text: a request on its own gets exactly the text of the same prompt run on its own in
  LLMEngine.generate (both at batch 1, so the same kernels), streamed or not; the streamed deltas add up to
  the whole text, and the usage counts every token.
- 16 concurrent streamed requests (sampled, EOS ignored) each get max_tokens tokens, counted per event with
  continuous usage stats, and the KV pool is empty afterwards.
- A stop string cuts the text before its first occurrence (finish_reason "stop"), streamed or not, and the
  request is aborted in the engine. A client that disconnects mid-stream gets its request aborted too.
- Chat completions (Qwen3's template), /v1/models, /health, and requests the server must reject.
- The incremental detokenizer and the stop checker: random token ids, in random pieces, give the same text
  as decoding them at once (Qwen3 and GPT-2 tokenizers).

Run from the repo root: python -m tests.test_server
"""

import asyncio
import json
import random
import time

import aiohttp
import torch
from aiohttp.test_utils import TestServer

from minivllm.async_engine import AsyncEngine
from minivllm.engine import EngineConfig, LLMEngine
from minivllm.model_runner import MODELS
from minivllm.sequence import SamplingParams
from minivllm.server import make_app
from minivllm.tokenizer import IncrementalDetokenizer, StopChecker, Tokenizer

CFG = EngineConfig(model="qwen3", num_kv_blocks=512, max_num_seqs=16, max_num_batched_tokens=4096, max_model_len=1024)
ok = True


def check(cond, label):
    global ok
    ok &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {label}", flush=True)


def test_detokenizer():
    rng = random.Random(0)
    for name in ("qwen3", "gpt2"):
        tok = Tokenizer(name, MODELS[name]["path"])
        worst = 0
        for trial in range(200):
            # ids from the whole vocabulary: many are pieces of characters that span several tokens
            ids = [rng.randrange(tok.vocab_size) for _ in range(rng.randrange(1, 120))]
            detok, out, i = IncrementalDetokenizer(tok.decode), [], 0
            while i < len(ids):
                k = rng.randrange(1, 5)
                out.append(detok.add(ids[i : i + k]))
                i += k
            out.append(detok.finish())
            whole = tok.decode(ids)
            worst = max(worst, "".join(out) != whole)
        check(not worst, f"{name}: incremental detokenization of random ids, in pieces of 1-4, equals decoding them at once")
    text = "abc STOP def ST OP xyz"
    for stops, expect in ((["STOP"], "abc "), (["OP x"], "abc STOP def ST "), (["zz"], text), (["d", "c S"], "ab")):
        for _ in range(20):
            sc, out, i, hit = StopChecker(stops), [], 0, False
            while i < len(text) and not hit:
                k = rng.randrange(1, 4)
                piece, hit = sc.add(text[i : i + k], i + k >= len(text))
                out.append(piece)
                i += k
            if "".join(out) != expect or hit != (expect != text):
                break
        else:
            check(True, f"stop strings {stops} cut {text!r} to {expect!r}, in random pieces")
            continue
        check(False, f"stop strings {stops}: got {''.join(out)!r}, expected {expect!r}")


def reference_texts(prompts, max_tokens, tok):
    """Greedy completions, each prompt on its own, from an engine in this thread."""
    engine = LLMEngine(CFG)
    texts = []
    for p in prompts:
        seq = engine.generate([p], SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True))[0]
        texts.append(tok.decode(seq.completion_token_ids))
    del engine
    torch.cuda.empty_cache()
    return texts


async def stream(session, url, payload):
    """Returns the events of a streamed request (parsed JSON), up to [DONE]."""
    events = []
    async with session.post(url, json={**payload, "stream": True}) as resp:
        assert resp.status == 200, await resp.text()
        async for line in resp.content:
            if line.startswith(b"data: "):
                data = line[6:].strip()
                if data == b"[DONE]":
                    break
                events.append(json.loads(data))
    return events


async def wait_idle(aeng, timeout=10.0):
    t = time.perf_counter()
    while aeng.engine.has_unfinished() and time.perf_counter() - t < timeout:
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.05)
    return not aeng.engine.has_unfinished() and aeng.engine.kv.allocated_slots() == 0


async def test_server(aeng, tok, prompts, refs, max_tokens):
    server = TestServer(make_app(aeng, tok, "qwen3"))
    await server.start_server()
    url = lambda path: str(server.make_url(path))
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        async with s.get(url("/health")) as r:
            check(r.status == 200 and await r.text() == "ok", "GET /health")
        async with s.get(url("/v1/models")) as r:
            m = await r.json()
            check(m["data"][0]["id"] == "qwen3" and m["data"][0]["max_model_len"] == CFG.max_model_len, "GET /v1/models")

        # greedy, one request at a time: the same text as the engine on its own
        same, usage_ok, deltas_ok = True, True, True
        for p, ref in zip(prompts, refs):
            payload = {"model": "qwen3", "prompt": p, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True}
            async with s.post(url("/v1/completions"), json=payload) as r:
                body = await r.json()
            same &= body["choices"][0]["text"] == ref and body["choices"][0]["finish_reason"] == "length"
            usage_ok &= body["usage"] == {"prompt_tokens": len(p), "completion_tokens": max_tokens, "total_tokens": len(p) + max_tokens}
            events = await stream(s, url("/v1/completions"), {**payload, "stream_options": {"include_usage": True}})
            text = "".join(e["choices"][0]["text"] for e in events if e["choices"])
            deltas_ok &= text == ref and events[-1]["usage"]["completion_tokens"] == max_tokens
            deltas_ok &= [e["choices"][0]["finish_reason"] for e in events if e["choices"]][-1] == "length"
        check(same and usage_ok, f"greedy: {len(prompts)} completions alone give the engine's own text and usage")
        check(deltas_ok, "greedy, streamed: the deltas add up to the same text; the usage event counts max_tokens")

        # 16 concurrent streams, sampled: every token is counted
        rng = random.Random(1)
        lens = [rng.randrange(20, 60) for _ in range(16)]
        reqs = [{"prompt": [rng.randrange(10000) for _ in range(rng.randrange(30, 300))], "max_tokens": n, "temperature": 0.8,
                 "top_p": 0.95, "ignore_eos": True, "stream_options": {"continuous_usage_stats": True}} for n in lens]
        results = await asyncio.gather(*(stream(s, url("/v1/completions"), r) for r in reqs))
        counts = [[e["usage"]["completion_tokens"] for e in ev] for ev in results]
        check(all(c[-1] == n and c == sorted(c) and len(c) == n for c, n in zip(counts, lens)),
              "16 concurrent streams: each gets max_tokens tokens, one event per token")
        check(await wait_idle(aeng), "after them the KV pool is empty")

        # stop strings
        ref = refs[3]  # a text prompt
        stop = ref[len(ref) // 2 : len(ref) // 2 + 3]
        expect = ref[: ref.find(stop)]
        payload = {"prompt": prompts[3], "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True, "stop": [stop, "\x00never"]}
        async with s.post(url("/v1/completions"), json=payload) as r:
            body = await r.json()
        events = await stream(s, url("/v1/completions"), payload)
        streamed = "".join(e["choices"][0]["text"] for e in events)
        check(body["choices"][0]["text"] == expect and body["choices"][0]["finish_reason"] == "stop" and streamed == expect
              and events[-1]["choices"][0]["finish_reason"] == "stop", f"stop {stop!r}: text cut before it, streamed or not")
        check(await wait_idle(aeng), "a stopped request is aborted: the KV pool is empty")

        # a client that goes away
        before = aeng.engine.kv.allocated_slots()
        resp = await s.post(url("/v1/completions"), json={"prompt": prompts[1], "max_tokens": 900, "ignore_eos": True, "stream": True})
        n = 0
        async for line in resp.content:
            n += line.startswith(b"data: ")
            if n == 3:
                break
        busy = aeng.engine.has_unfinished()
        resp.close()  # drops the connection
        check(before == 0 and busy and await wait_idle(aeng), "a client disconnecting mid-stream: its request is aborted, the pool emptied")

        # chat
        chat = {"messages": [{"role": "user", "content": "Name three colors."}], "max_tokens": 16, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        async with s.post(url("/v1/chat/completions"), json=chat) as r:
            body = await r.json()
        msg = body["choices"][0]["message"]
        events = await stream(s, url("/v1/chat/completions"), chat)
        streamed = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        check(msg["role"] == "assistant" and msg["content"] and streamed == msg["content"] and events[0]["choices"][0]["delta"]["role"] == "assistant"
              and body["usage"]["completion_tokens"] <= 16 and events[-1]["choices"][0]["finish_reason"] == body["choices"][0]["finish_reason"],
              f"chat: streamed deltas equal the whole message ({msg['content']!r})")

        # requests to reject
        bad = [("/v1/completions", "not json", 400), ("/v1/completions", {"prompt": [1] * 1000, "max_tokens": 100}, 400),
               ("/v1/completions", {"prompt": "hi", "n": 2}, 400), ("/v1/completions", {"prompt": "hi", "model": "gpt-4"}, 404),
               ("/v1/completions", {"prompt": [10**7]}, 400), ("/v1/completions", {"prompt": []}, 400),
               ("/v1/completions", {"prompt": "hi", "temperature": 3}, 400), ("/v1/completions", {"prompt": "hi", "logprobs": 2}, 400),
               ("/v1/chat/completions", {"messages": [{"role": "robot", "content": "x"}]}, 400)]
        codes = []
        for path, payload, status in bad:
            kw = {"data": payload} if isinstance(payload, str) else {"json": payload}
            async with s.post(url(path), **kw) as r:
                codes.append((r.status, status, (await r.json())["error"]["message"]))
        check(all(a == b for a, b, _ in codes), "rejected with 400/404: " + "; ".join(m for _, _, m in codes))
        check(await wait_idle(aeng), "the KV pool is empty at the end")
    await server.close()


def main():
    test_detokenizer()
    tok = Tokenizer("qwen3", MODELS["qwen3"]["path"])
    rng = random.Random(0)
    max_tokens = 40
    prompts = [[rng.randrange(10000) for _ in range(rng.randrange(20, 200))] for _ in range(3)]
    prompts += [tok.encode("The capital of France is"), tok.encode("def fibonacci(n):")]
    refs = reference_texts(prompts, max_tokens, tok)
    aeng = AsyncEngine(CFG, warmup_temperature=None)
    asyncio.run(test_server(aeng, tok, prompts, refs, max_tokens))
    aeng.shutdown()
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
