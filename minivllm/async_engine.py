"""Runs the engine for the asyncio HTTP server (minivllm/server.py): in a process of its own (the default), or in
a thread of the server's process (E11).

LLMEngine is not thread-safe, so one loop owns it (EngineLoop): between steps it takes new requests, aborts and
queries, then runs a step and reports every request's new tokens in one message. On the server's side,
EngineClient keeps each request's asyncio queue and routes those messages to them.

- ThreadEngine runs the loop in a thread. Messages to it go through a queue, its reports through
  call_soon_threadsafe. The engine's Python code (scheduling, preparing each step's inputs) and the event
  loop (decoding, JSON, sockets) then take turns on one GIL.
- ProcessEngine runs the loop in a child process, as vLLM's EngineCore does, and talks to it over two pipes; a
  reader thread hands its reports to the event loop. The engine no longer waits for the server's GIL; each
  step costs a pickled message instead.
"""

import asyncio
import collections
import itertools
import multiprocessing
import queue
import threading
import time
import traceback

from minivllm.engine import EngineConfig, LLMEngine, check_request
from minivllm.sequence import Status


class EngineLoop:
    """Owns an LLMEngine. Messages in: ("add", id, prompt ids, SamplingParams, arrival time), ("abort", id),
    ("query", id, "stats" or "steps"), ("stop",). Messages out: ("updates", [(id, new token ids, finished,
    finish reason)]) after each step, ("reply", query id, value), ("reject", id, error message)."""

    def __init__(self, cfg: EngineConfig, warmup_temperature, record_steps):
        engine = LLMEngine(cfg)
        if warmup_temperature is not None:
            engine.warmup(warmup_temperature)
        # a server runs for long: keep the recent step records only, unless a benchmark takes them
        engine.steps = [] if record_steps else collections.deque(maxlen=1024)
        self.engine = engine
        self.active = {}  # request id -> [sequence, completion tokens reported]

    def run(self, recv, poll, send):
        """recv() waits for the next message, poll() tells whether one is waiting, send() reports."""
        engine = self.engine
        while True:
            msgs = [] if engine.has_unfinished() else [recv()]
            while poll():
                msgs.append(recv())
            for msg in msgs:
                if msg[0] == "stop":
                    return
                reply = self.handle(msg)
                if reply is not None:
                    send(reply)
            if engine.has_unfinished():
                engine.step()
                updates = self.updates()
                if updates:
                    send(("updates", updates))

    def handle(self, msg):
        kind = msg[0]
        if kind == "add":
            _, rid, ids, params, arrival_time = msg
            try:
                self.active[rid] = [self.engine.add_request(ids, params, arrival_time), 0]
            except ValueError as e:  # the server checks requests first, so this should not happen
                return ("reject", rid, str(e))
        elif kind == "abort":
            entry = self.active.pop(msg[1], None)
            if entry is not None:
                self.engine.abort_request(entry[0])
        elif kind == "query":
            if msg[2] == "steps":
                steps, self.engine.steps = self.engine.steps, []
                value = [[r.start, r.end, r.is_prefill, r.num_seqs, r.num_tokens, r.num_decode] for r in steps]
            else:
                value = {"unfinished": self.engine.has_unfinished(), "allocated_slots": self.engine.kv.allocated_slots(),
                         "requests": len(self.active)}
            return ("reply", msg[1], value)
        return None

    def updates(self):
        """Each request's tokens since its last update; "length" when max_tokens ran out, "stop" at EOS."""
        out, done = [], []
        for rid, entry in self.active.items():
            seq, sent = entry
            n = seq.num_completion_tokens
            finished = seq.status == Status.FINISHED
            if n > sent or finished:
                reason = ("length" if n >= seq.params.max_tokens else "stop") if finished else None
                out.append((rid, seq.token_ids[seq.num_prompt_tokens + sent :], finished, reason))
                entry[1] = n
            if finished:
                done.append(rid)
        for rid in done:
            del self.active[rid]
        return out


class Request:
    """A request's stream of updates: (new token ids, finished, finish reason)."""

    def __init__(self, rid):
        self.id = rid
        self.queue = asyncio.Queue()
        self.finished = False  # the last update arrived, or the request was aborted

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


class EngineClient:
    """The server's side of the engine: sends it requests, aborts and queries from the event loop, and routes
    its messages (posted to the event loop from another thread) to the requests."""

    def __init__(self, cfg: EngineConfig, record_steps):
        self.cfg = cfg
        self.record_steps = record_steps
        self.error = None
        self.loop = None
        self.requests = {}
        self.queries = {}
        self.ids = itertools.count()

    def check(self, num_prompt_tokens, params):
        """Raises ValueError for a request the engine cannot run (it only reads the config)."""
        check_request(self.cfg, num_prompt_tokens, params)

    def add_request(self, prompt_token_ids, params) -> Request:
        """Called from the event loop, for a request that passed check()."""
        if self.error is not None:
            raise self.error
        self.loop = asyncio.get_running_loop()
        req = Request(next(self.ids))
        self.requests[req.id] = req
        self._send(("add", req.id, prompt_token_ids, params, time.perf_counter()))
        return req

    def abort(self, req: Request):
        """Stops a request that has not finished; its blocks are freed before the engine's next step."""
        if not req.finished:
            req.finished = True
            self.requests.pop(req.id, None)
            try:
                self._send(("abort", req.id))
            except OSError:  # the engine process is gone; the error reaches the requests separately
                pass

    async def query(self, what):
        """"stats": whether the engine has work, its allocated KV slots and requests; "steps": the step records
        since the last such query (with record_steps), as [start, end, is_prefill, sequences, tokens, decoding]."""
        if self.error is not None:
            raise self.error
        self.loop = asyncio.get_running_loop()
        qid = next(self.ids)
        self.queries[qid] = fut = self.loop.create_future()
        self._send(("query", qid, what))
        return await fut

    def _post(self, msg):
        """From the engine's thread or the reader thread: hands a message to the event loop."""
        if self.loop is None:  # nothing was sent yet, so it can only be an error
            self.error = RuntimeError(f"the engine stopped: {msg[1]}")
            return
        try:
            self.loop.call_soon_threadsafe(self._receive, msg)
        except RuntimeError:  # the event loop is closed
            pass

    def _receive(self, msg):
        kind = msg[0]
        if kind == "updates":
            for rid, tokens, finished, reason in msg[1]:
                req = self.requests.get(rid)
                if req is not None:  # not aborted
                    req.queue.put_nowait((tokens, finished, reason))
                    if finished:
                        del self.requests[rid]
        elif kind == "reply":
            fut = self.queries.pop(msg[1], None)
            if fut is not None and not fut.done():
                fut.set_result(msg[2])
        elif kind == "reject":
            req = self.requests.pop(msg[1], None)
            if req is not None:
                req.queue.put_nowait(ValueError(msg[2]))
        elif kind == "error":  # the engine stopped: every request and query gets the error
            self.error = RuntimeError(f"the engine stopped: {msg[1]}")
            for req in self.requests.values():
                req.queue.put_nowait(self.error)
            for fut in self.queries.values():
                if not fut.done():
                    fut.set_exception(self.error)
            self.requests.clear()
            self.queries.clear()

    def _send(self, msg):
        raise NotImplementedError

    def shutdown(self):
        raise NotImplementedError


class ThreadEngine(EngineClient):
    """The engine loop in a thread of this process (E11)."""

    def __init__(self, cfg: EngineConfig, warmup_temperature: float | None = 0.6, record_steps=False):
        """Starts the engine thread and waits until the engine is built and, unless warmup_temperature is None,
        warmed up (LLMEngine.warmup). With record_steps every step's record is kept for query("steps")."""
        super().__init__(cfg, record_steps)
        self.inbox = queue.SimpleQueue()
        ready, failed = threading.Event(), []

        def main():
            try:
                loop = EngineLoop(cfg, warmup_temperature, record_steps)
            except BaseException as e:
                failed.append(e)
                return
            finally:
                ready.set()
            try:
                loop.run(self.inbox.get, lambda: not self.inbox.empty(), self._post)
            except BaseException as e:
                traceback.print_exc()
                self._post(("error", repr(e)))

        self.thread = threading.Thread(target=main, name="engine", daemon=True)
        self.thread.start()
        ready.wait()
        if failed:
            raise RuntimeError("the engine failed to start") from failed[0]

    def _send(self, msg):
        self.inbox.put(msg)

    def shutdown(self):
        self.inbox.put(("stop",))
        self.thread.join()


def _engine_process(inbox, outbox, cfg, warmup_temperature, record_steps):
    """The child process of ProcessEngine."""
    try:
        loop = EngineLoop(cfg, warmup_temperature, record_steps)
    except BaseException as e:
        traceback.print_exc()
        outbox.send(("failed", repr(e)))
        return
    outbox.send(("ready", None))
    try:
        loop.run(inbox.recv, inbox.poll, outbox.send)
    except (EOFError, OSError):  # the server went away
        pass
    except BaseException as e:
        traceback.print_exc()
        try:
            outbox.send(("error", repr(e)))
        except OSError:
            pass


class ProcessEngine(EngineClient):
    """The engine loop in a child process of its own (spawned, so CUDA starts fresh there), over two one-way
    pipes; a reader thread posts the child's messages to the event loop."""

    def __init__(self, cfg: EngineConfig, warmup_temperature: float | None = 0.6, record_steps=False):
        """Starts the engine process and waits until the engine is built and, unless warmup_temperature is
        None, warmed up. With record_steps every step's record is kept for query("steps")."""
        super().__init__(cfg, record_steps)
        ctx = multiprocessing.get_context("spawn")
        child_in, self.to_engine = ctx.Pipe(duplex=False)
        self.from_engine, child_out = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(target=_engine_process, args=(child_in, child_out, cfg, warmup_temperature, record_steps),
                                name="minivllm-engine", daemon=True)
        self.proc.start()
        child_in.close()  # this process keeps only its own ends, so it sees EOF if the child dies
        child_out.close()
        try:
            msg = self.from_engine.recv()
        except EOFError:
            msg = ("failed", f"the engine process exited with code {self.proc.exitcode}")
        if msg[0] != "ready":
            self.proc.join()
            raise RuntimeError(f"the engine failed to start: {msg[1]}")
        self.closing = False
        self.reader = threading.Thread(target=self._read, name="engine-reader", daemon=True)
        self.reader.start()

    def _read(self):
        while True:
            try:
                msg = self.from_engine.recv()
            except (EOFError, OSError):
                if not self.closing:
                    self._post(("error", f"the engine process exited with code {self.proc.exitcode}"))
                return
            self._post(msg)

    def _send(self, msg):
        self.to_engine.send(msg)

    def shutdown(self):
        self.closing = True
        try:
            self.to_engine.send(("stop",))
        except OSError:
            pass
        self.proc.join(timeout=30)
        if self.proc.is_alive():
            self.proc.terminate()
        self.to_engine.close()
