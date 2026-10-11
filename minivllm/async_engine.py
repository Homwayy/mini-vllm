"""Runs the engine in a background thread, for the asyncio HTTP server (minivllm/server.py).

LLMEngine is not thread-safe, so one thread owns it: the thread creates it (and captures the CUDA graphs),
then loops: take new requests and aborts from a queue, run one step, and hand every request its new tokens.
Each request waits on an asyncio queue of its own; all of a step's updates go to the event loop in one
call_soon_threadsafe. With nothing to run, the thread blocks on its queue.
"""

import asyncio
import collections
import queue
import threading
import time
import traceback

from minivllm.engine import EngineConfig, LLMEngine, check_request
from minivllm.sequence import Status


class Request:
    """A request's stream of updates, (new token ids, finished, finish reason): "length" when max_tokens ran
    out, "stop" at EOS, None until it finishes."""

    def __init__(self, prompt_token_ids, params, loop):
        self.prompt_token_ids = prompt_token_ids
        self.params = params
        self.loop = loop
        self.arrival_time = time.perf_counter()
        self.queue = asyncio.Queue()
        self.finished = False  # set by the event loop: the last update arrived, or the request was aborted
        self.seq = None  # engine thread only
        self.sent = 0  # completion tokens handed out so far (engine thread)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.finished:
            raise StopAsyncIteration
        item = await self.queue.get()
        if isinstance(item, BaseException):
            self.finished = True
            raise item
        self.finished = item[1]
        return item


def _deliver(updates):
    for req, item in updates:
        req.queue.put_nowait(item)


class AsyncEngine:
    def __init__(self, cfg: EngineConfig, warmup_temperature: float | None = 0.6, record_steps=False):
        """Starts the engine thread and waits until the engine is built and, unless warmup_temperature is
        None, warmed up (LLMEngine.warmup). With record_steps, every step's record is kept until take_steps()
        (for benchmarks); otherwise only the last 1024."""
        self.cfg = cfg
        self.engine = None
        self.error = None
        self.record_steps = record_steps
        self.inbox = queue.SimpleQueue()
        ready = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(warmup_temperature, ready), name="engine", daemon=True)
        self.thread.start()
        ready.wait()
        if self.error is not None:
            raise RuntimeError("the engine failed to start") from self.error

    def check(self, num_prompt_tokens, params):
        """Raises ValueError for a request the engine cannot run (it only reads the config)."""
        check_request(self.cfg, num_prompt_tokens, params)

    def add_request(self, prompt_token_ids, params) -> Request:
        """Called from the event loop; the request has passed check()."""
        if self.error is not None:
            raise RuntimeError(f"the engine stopped: {self.error!r}")
        req = Request(prompt_token_ids, params, asyncio.get_running_loop())
        self.inbox.put(("add", req))
        return req

    def abort(self, req: Request):
        """Stops a request that has not finished; its blocks are freed before the next step."""
        if not req.finished:
            req.finished = True
            self.inbox.put(("abort", req))

    def take_steps(self):
        """The step records since the last call (with record_steps)."""
        steps, self.engine.steps = self.engine.steps, []
        return list(steps)

    def shutdown(self):
        self.inbox.put(("stop", None))
        self.thread.join()

    def _run(self, warmup_temperature, ready):
        try:
            engine = LLMEngine(self.cfg)
            if warmup_temperature is not None:
                engine.warmup(warmup_temperature)
            # a server runs for long: keep the recent records only, unless a benchmark takes them
            engine.steps = [] if self.record_steps else collections.deque(maxlen=1024)
            self.engine = engine
        except BaseException as e:
            self.error = e
            return
        finally:
            ready.set()
        active: list[Request] = []
        try:
            while True:
                items = [] if engine.has_unfinished() else [self.inbox.get()]
                while True:
                    try:
                        items.append(self.inbox.get_nowait())
                    except queue.Empty:
                        break
                for kind, req in items:
                    if kind == "stop":
                        return
                    if kind == "add" and not req.finished:  # an abort can overtake its request
                        try:
                            req.seq = engine.add_request(req.prompt_token_ids, req.params, req.arrival_time)
                        except ValueError as e:  # the server checks requests first, so this should not happen
                            req.loop.call_soon_threadsafe(_deliver, [(req, e)])
                            continue
                        active.append(req)
                    elif kind == "abort" and req in active:
                        engine.abort_request(req.seq)
                        active.remove(req)
                if not engine.has_unfinished():
                    continue
                engine.step()
                updates, still = [], []
                for req in active:
                    seq = req.seq
                    n = seq.num_completion_tokens
                    done = seq.status == Status.FINISHED
                    if n > req.sent or done:
                        reason = ("length" if n >= seq.params.max_tokens else "stop") if done else None
                        updates.append((req, (seq.token_ids[seq.num_prompt_tokens + req.sent :], done, reason)))
                        req.sent = n
                    if not done:
                        still.append(req)
                active = still
                if updates:
                    updates[0][0].loop.call_soon_threadsafe(_deliver, updates)
        except BaseException as e:  # a failed step: every request in the engine or on its way gets the error
            self.error = e
            traceback.print_exc()
            while True:
                try:
                    kind, req = self.inbox.get_nowait()
                except queue.Empty:
                    break
                if kind == "add":
                    active.append(req)
            for req in active:
                req.loop.call_soon_threadsafe(_deliver, [(req, RuntimeError(f"the engine stopped: {e!r}"))])
