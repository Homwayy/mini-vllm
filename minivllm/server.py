"""An OpenAI-compatible HTTP server for the engine (E11).

POST /v1/completions and /v1/chat/completions, streamed as server-sent events or returned whole; GET /v1/models
and /health. The engine runs in its own thread (minivllm/async_engine.py). The handlers run in an aiohttp
event loop: each decodes the tokens its request gets in a step (IncrementalDetokenizer) and writes one event.

- prompt: a string or a list of token ids (completions); messages with text content (chat; Qwen3 only, with
  its chat template; chat_template_kwargs are passed on, e.g. {"enable_thinking": false}).
- Sampling: max_tokens (or max_completion_tokens), temperature, top_p, and as in vLLM top_k and ignore_eos.
- stop: up to 4 strings. The output ends before the first one, and the request is aborted in the engine.
- stream_options: include_usage adds a last event with the token counts; continuous_usage_stats (as in vLLM)
  puts them in every event, so a client can tell how many tokens each event carries.
- n > 1, best_of, logprobs, echo, suffix, logit_bias, penalties and tools are rejected with 400.
A request whose client goes away is aborted, which frees its KV blocks.

Run from the repo root, e.g.:
    python -m minivllm.server --model qwen3 --port 8000
"""

import argparse
import json
import time
import uuid

from aiohttp import web

from minivllm.async_engine import AsyncEngine
from minivllm.engine import EngineConfig
from minivllm.model_runner import MODELS
from minivllm.sequence import SamplingParams
from minivllm.tokenizer import IncrementalDetokenizer, StopChecker, Tokenizer


class BadRequest(Exception):
    def __init__(self, message, param=None, status=400):
        super().__init__(message)
        self.param = param
        self.status = status


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def error_response(message, status=400, param=None):
    kind = {400: "BadRequestError", 404: "NotFoundError"}.get(status, "InternalServerError")
    return web.json_response({"error": {"message": message, "type": kind, "param": param, "code": status}}, status=status, dumps=dumps)


async def read_body(request):
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise BadRequest("the body is not valid JSON")
    if not isinstance(body, dict):
        raise BadRequest("the body must be a JSON object")
    name = body.get("model")
    if name is not None and name != request.app["model_name"]:
        raise BadRequest(f"the model `{name}` does not exist", param="model", status=404)
    if body.get("n", 1) not in (None, 1) or body.get("best_of", 1) not in (None, 1):
        raise BadRequest("only n = 1 is supported", param="n")
    for key in ("logprobs", "top_logprobs", "echo", "suffix", "logit_bias", "presence_penalty", "frequency_penalty", "tools", "functions"):
        if body.get(key):
            raise BadRequest(f"{key} is not supported", param=key)
    return body


def number(body, key, default, lo, hi, integer=False):
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not float(v).is_integer()) or not lo <= v <= hi:
        raise BadRequest(f"{key} must be {'an integer' if integer else 'a number'} in [{lo}, {hi}]", param=key)
    return int(v) if integer else float(v)


def sampling_params(body, default_max_tokens):
    """The request's SamplingParams and stop strings."""
    key = "max_completion_tokens" if body.get("max_completion_tokens") is not None else "max_tokens"
    max_tokens = number(body, key, default_max_tokens, 1, 1 << 30, integer=True)
    top_p = number(body, "top_p", 1.0, 0.0, 1.0)
    if top_p == 0:
        raise BadRequest("top_p must be in (0, 1]", param="top_p")
    params = SamplingParams(temperature=number(body, "temperature", 1.0, 0.0, 2.0), top_p=top_p,
                            top_k=max(0, number(body, "top_k", 0, -1, 1 << 30, integer=True)),  # -1 (vLLM) or 0: off
                            max_tokens=max_tokens, ignore_eos=bool(body.get("ignore_eos", False)))
    stop = body.get("stop")
    stop = [stop] if isinstance(stop, str) else stop or []
    if not isinstance(stop, list) or len(stop) > 4 or not all(isinstance(s, str) for s in stop):
        raise BadRequest("stop must be a string or a list of at most 4 strings", param="stop")
    return params, [s for s in stop if s]


def checked(request, ids, params):
    tokenizer, engine = request.app["tokenizer"], request.app["engine"]
    if not all(0 <= i < tokenizer.vocab_size for i in ids):
        raise BadRequest(f"token ids must be in [0, {tokenizer.vocab_size})", param="prompt")
    try:
        engine.check(len(ids), params)
    except ValueError as e:
        raise BadRequest(str(e))


async def completions(request):
    try:
        body = await read_body(request)
        prompt = body.get("prompt")
        if isinstance(prompt, list) and len(prompt) == 1 and isinstance(prompt[0], (str, list)):
            prompt = prompt[0]  # a batch of one
        if isinstance(prompt, str):
            ids = request.app["tokenizer"].encode(prompt)
        elif isinstance(prompt, list) and all(type(i) is int for i in prompt):
            ids = prompt
        else:
            raise BadRequest("prompt must be a string or a list of token ids", param="prompt")
        params, stop = sampling_params(body, 16)
        checked(request, ids, params)
    except BadRequest as e:
        return error_response(str(e), e.status, e.param)
    return await generate(request, body, ids, params, stop, chat=False)


async def chat_completions(request):
    tokenizer = request.app["tokenizer"]
    try:
        body = await read_body(request)
        if not tokenizer.has_chat_template:
            raise BadRequest("this model has no chat template; use /v1/completions")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise BadRequest("messages must be a non-empty list", param="messages")
        turns = []
        for m in messages:
            content = m.get("content") if isinstance(m, dict) else None
            if isinstance(content, list):  # content parts: only text
                if not all(isinstance(p, dict) and p.get("type") == "text" for p in content):
                    raise BadRequest("only text content is supported", param="messages")
                content = "".join(p.get("text", "") for p in content)
            if not isinstance(content, str) or m.get("role") not in ("system", "user", "assistant"):
                raise BadRequest("each message needs a role (system, user or assistant) and text content", param="messages")
            turns.append({"role": m["role"], "content": content})
        kwargs = body.get("chat_template_kwargs") or {}
        if not isinstance(kwargs, dict):
            raise BadRequest("chat_template_kwargs must be an object", param="chat_template_kwargs")
        ids = tokenizer.apply_chat_template(turns, **kwargs)
        params, stop = sampling_params(body, max(1, request.app["engine"].cfg.max_model_len - len(ids)))
        checked(request, ids, params)
    except BadRequest as e:
        return error_response(str(e), e.status, e.param)
    return await generate(request, body, ids, params, stop, chat=True)


async def generate(request, body, ids, params, stop, chat):
    app = request.app
    engine, tokenizer = app["engine"], app["tokenizer"]
    rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
    created = int(time.time())
    detok = IncrementalDetokenizer(tokenizer.decode)
    stops = StopChecker(stop) if stop else None
    n_out = 0

    def usage():
        return {"prompt_tokens": len(ids), "completion_tokens": n_out, "total_tokens": len(ids) + n_out}

    def text_of(tokens, finished):
        """The text to send for a step's tokens, and whether a stop string ended the output."""
        text = detok.add(tokens)
        if finished:
            text += detok.finish()
        return stops.add(text, finished) if stops is not None else (text, False)

    try:
        req = engine.add_request(ids, params)
    except RuntimeError as e:
        return error_response(str(e), 500)

    if not body.get("stream"):
        parts, reason = [], None
        try:
            async for tokens, finished, reason in req:
                n_out += len(tokens)
                text, hit = text_of(tokens, finished)
                parts.append(text)
                if hit:
                    reason = "stop"
                    break
        except RuntimeError as e:
            return error_response(str(e), 500)
        finally:
            engine.abort(req)  # a stop string, or the client went away; nothing if the request finished
        text = "".join(parts)
        choice = ({"index": 0, "message": {"role": "assistant", "content": text}, "logprobs": None, "finish_reason": reason} if chat
                  else {"index": 0, "text": text, "logprobs": None, "finish_reason": reason})
        return web.json_response({"id": rid, "object": "chat.completion" if chat else "text_completion", "created": created,
                                  "model": app["model_name"], "choices": [choice], "usage": usage()}, dumps=dumps)

    opts = body.get("stream_options") or {}
    include_usage, continuous = bool(opts.get("include_usage")), bool(opts.get("continuous_usage_stats"))
    head = f'{{"id":"{rid}","object":"{"chat.completion.chunk" if chat else "text_completion"}","created":{created},"model":{dumps(app["model_name"])},'

    def event(text, reason, role=False):
        if chat:
            delta = {"role": "assistant", "content": ""} if role else ({"content": text} if text or reason is None else {})
            choice = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": reason}
        else:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": reason}
        tail = f',"usage":{dumps(usage())}' if continuous else ""
        return f'data: {head}"choices":[{dumps(choice)}]{tail}}}\n\n'.encode()

    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    try:
        await resp.prepare(request)
        if chat:
            await resp.write(event("", None, role=True))
        async for tokens, finished, reason in req:
            n_out += len(tokens)
            text, hit = text_of(tokens, finished)
            if hit:
                finished, reason = True, "stop"
            if text or finished or continuous:  # with continuous usage stats, every step's tokens get an event
                await resp.write(event(text, reason if finished else None))
            if finished:
                break
        if include_usage:
            await resp.write(f'data: {head}"choices":[],"usage":{dumps(usage())}}}\n\n'.encode())
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
    except ConnectionError:  # the client went away
        pass
    except RuntimeError as e:  # the engine stopped in the middle of the stream
        await resp.write(f"data: {dumps({'error': {'message': str(e), 'type': 'InternalServerError', 'code': 500}})}\n\n".encode())
    finally:
        engine.abort(req)
    return resp


async def models(request):
    app = request.app
    return web.json_response({"object": "list", "data": [{
        "id": app["model_name"], "object": "model", "created": app["created"], "owned_by": "minivllm",
        "max_model_len": app["engine"].cfg.max_model_len}]}, dumps=dumps)


async def health(request):
    error = request.app["engine"].error
    return web.Response(text="ok") if error is None else web.Response(status=503, text=repr(error))


async def debug_steps(request):
    """The engine's step records since the last call, for benchmarks (--record-steps): [start, end (perf_counter
    seconds, comparable across processes on Windows), is_prefill, sequences, tokens, decoding sequences]."""
    steps = request.app["engine"].take_steps()
    return web.json_response({"steps": [[r.start, r.end, r.is_prefill, r.num_seqs, r.num_tokens, r.num_decode] for r in steps]})


def make_app(engine: AsyncEngine, tokenizer: Tokenizer, model_name: str) -> web.Application:
    app = web.Application(client_max_size=64 << 20)
    app["engine"], app["tokenizer"], app["model_name"], app["created"] = engine, tokenizer, model_name, int(time.time())
    app.router.add_post("/v1/completions", completions)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/health", health)
    if engine.record_steps:
        app.router.add_get("/debug/steps", debug_steps)
    return app


def main():
    d = EngineConfig()
    ap = argparse.ArgumentParser(description="OpenAI-compatible HTTP server for mini vLLM")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-model-name", default=None, help="the model name clients use (by default --model)")
    ap.add_argument("--model", default=d.model)
    ap.add_argument("--model-path", default=d.model_path)
    ap.add_argument("--attention", default=d.attention)
    ap.add_argument("--kv", default=d.kv)
    ap.add_argument("--admission", default=d.admission)
    ap.add_argument("--batching", default=d.batching)
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--chunked-prefill", action="store_true")
    ap.add_argument("--no-cuda-graph", action="store_true")
    ap.add_argument("--prefill-graph-tokens", type=int, default=d.prefill_graph_tokens)
    ap.add_argument("--spec", default=d.spec)
    ap.add_argument("--num-spec-tokens", type=int, default=d.num_spec_tokens)
    ap.add_argument("--draft-path", default=d.draft_path)
    ap.add_argument("--num-kv-blocks", type=int, default=d.num_kv_blocks)
    ap.add_argument("--max-num-seqs", type=int, default=d.max_num_seqs)
    ap.add_argument("--max-num-batched-tokens", type=int, default=d.max_num_batched_tokens)
    ap.add_argument("--max-model-len", type=int, default=d.max_model_len)
    ap.add_argument("--warmup-temperature", type=float, default=0.6, help="temperature of the warm-up requests; negative: no warm-up")
    ap.add_argument("--record-steps", action="store_true", help="keep every step's record for GET /debug/steps (benchmarks)")
    args = ap.parse_args()

    cfg = EngineConfig(
        model=args.model, model_path=args.model_path, attention=args.attention, kv=args.kv, admission=args.admission,
        batching=args.batching, prefix_caching=args.prefix_caching, chunked_prefill=args.chunked_prefill,
        cuda_graph=not args.no_cuda_graph, prefill_graph_tokens=args.prefill_graph_tokens, spec=args.spec,
        num_spec_tokens=args.num_spec_tokens, draft_path=args.draft_path, num_kv_blocks=args.num_kv_blocks,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens, max_model_len=args.max_model_len,
    )
    tokenizer = Tokenizer(cfg.model, cfg.model_path or MODELS[cfg.model]["path"])
    engine = AsyncEngine(cfg, args.warmup_temperature if args.warmup_temperature >= 0 else None, args.record_steps)
    name = args.served_model_name or cfg.model
    print(f"{cfg}\nserving {name} at http://{args.host}:{args.port}", flush=True)
    web.run_app(make_app(engine, tokenizer, name), host=args.host, port=args.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
