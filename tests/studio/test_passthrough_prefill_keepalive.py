"""A passthrough relay must not go silent while llama-server prefills.

llama-server flushes response headers as soon as the request is queued and then
sends nothing at all until the first token. Node's undici -- the transport under
every JS agent `unsloth start` supports -- aborts a response after 300s with no
bytes and reports a bare `terminated`, and each retry lands on a different
llama-server slot whose KV cache shares no prefix, so the retries restart prefill
from zero and never converge.

Behavioural verifies the extracted `_aiter_llama_stream_items` emits keepalives
across a stalled read *without restarting that read* (restarting it is not an
option: httpcore closes the body stream on any streaming exception), that the
stall guard still fires, and that teardown leaves no pending read. Structural
verifies every call site translates the sentinel before it can be mistaken for
an upstream item, so a fifth passthrough surface cannot be added without one.
"""

from __future__ import annotations

import ast
import asyncio
import time
from pathlib import Path

import httpx
import pytest


SOURCE_PATH = Path(__file__).resolve().parents[2] / "studio" / "backend" / "routes" / "inference.py"
SRC = SOURCE_PATH.read_text(encoding = "utf-8")
_TREE = ast.parse(SRC)

# Short enough to keep the suite fast, long enough that a tick is unambiguous.
_TICK_S = 0.05

_WANTED = {"_LlamaStreamKeepalive", "_LLAMA_STREAM_KEEPALIVE", "_aiter_llama_stream_items"}


# ── Loading ──────────────────────────────────────────────────


def _load_pump():
    """The real pump, cut from disk and exec'd against stubs.

    Importing `routes.inference` would pull in the whole backend package; the
    pump only needs httpx, asyncio and two constants, so give it exactly those.
    Each call gets a fresh namespace so no test can leak state into another.
    """
    chunks = []
    for node in _TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            wanted = node.name in _WANTED
        elif isinstance(node, ast.Assign):
            wanted = any(t.id in _WANTED for t in node.targets if isinstance(t, ast.Name))
        else:
            wanted = False
        if not wanted:
            continue
        segment = ast.get_source_segment(SRC, node)
        assert segment is not None, f"could not cut {getattr(node, 'name', node)} from source"
        chunks.append(segment)
    assert len(chunks) == len(_WANTED), f"expected {len(_WANTED)} definitions, cut {len(chunks)}"

    namespace = {
        "asyncio": asyncio,
        "httpx": httpx,
        "time": time,
        "Optional": __import__("typing").Optional,
        "Union": __import__("typing").Union,
        "Callable": __import__("typing").Callable,
        "Request": object,
        "_DEFAULT_STREAM_STALL_TIMEOUT_S": 120.0,
        "_DEFAULT_FIRST_TOKEN_TIMEOUT_S": 1200.0,
        "_first_token_timeout_s": lambda: 1200.0,
        "_TEARDOWN_TASK_STOP_TIMEOUT_S": 5.0,
        # Records the socket-level ceiling without needing a real response.
        "_set_stream_response_read_timeout": lambda response, read_timeout_s = None: None,
        "_discard_task_outcome": lambda task: None,
    }
    exec("\n\n".join(chunks), namespace)
    return namespace


class _ScriptedIter:
    """Async iterator that yields, then blocks on a gate, then yields again.

    Counts `__anext__` entries so a test can prove the pump kept waiting on one
    read rather than starting a fresh one per keepalive tick.
    """

    def __init__(
        self,
        immediate = (),
        gate = None,
        after_gate = (),
    ):
        self._immediate = list(immediate)
        self._gate = gate
        self._after_gate = list(after_gate)
        self.anext_calls = 0
        self.cancelled = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.anext_calls += 1
        if self._immediate:
            return self._immediate.pop(0)
        if self._gate is not None:
            gate, self._gate = self._gate, None
            try:
                await gate.wait()
            except asyncio.CancelledError:
                self.cancelled += 1
                raise
        if self._after_gate:
            return self._after_gate.pop(0)
        raise StopAsyncIteration


# ── Behavioural ──────────────────────────────────────────────


def test_keepalives_flow_during_prefill_without_restarting_the_read():
    pump_ns = _load_pump()
    keepalive = pump_ns["_LLAMA_STREAM_KEEPALIVE"]

    async def _run():
        gate = asyncio.Event()
        upstream = _ScriptedIter(gate = gate, after_gate = ["data: first-token"])
        seen_keepalives = 0
        items = []
        stream = pump_ns["_aiter_llama_stream_items"](
            upstream,
            first_token_deadline = time.monotonic() + 30.0,
            keepalive_interval_s = _TICK_S,
        )
        try:
            async for got in stream:
                if got is keepalive:
                    seen_keepalives += 1
                    if seen_keepalives >= 3:
                        # Only now let the single in-flight read complete.
                        gate.set()
                    continue
                items.append(got)
                break
        finally:
            await stream.aclose()
        return seen_keepalives, items, upstream

    seen_keepalives, items, upstream = asyncio.run(_run())

    assert (
        seen_keepalives >= 3
    ), f"pump must keep emitting keepalives while upstream is silent; got {seen_keepalives}"
    assert items == [
        "data: first-token"
    ], f"the real item must still arrive after the keepalives; got {items}"
    assert upstream.anext_calls == 1, (
        "keepalive ticks must await the SAME __anext__, not restart the read "
        f"(httpcore closes the body on any streaming exception); got {upstream.anext_calls} calls"
    )


def test_keepalives_flow_during_a_post_first_token_stall_then_the_guard_fires():
    pump_ns = _load_pump()
    keepalive = pump_ns["_LLAMA_STREAM_KEEPALIVE"]

    async def _run():
        upstream = _ScriptedIter(immediate = ["data: a"], gate = asyncio.Event())
        seen_keepalives = 0
        items = []
        stream = pump_ns["_aiter_llama_stream_items"](
            upstream,
            first_token_deadline = time.monotonic() + 30.0,
            post_first_item_read_timeout_s = 0.4,
            keepalive_interval_s = _TICK_S,
        )
        try:
            with pytest.raises(httpx.ReadTimeout) as excinfo:
                async for got in stream:
                    if got is keepalive:
                        seen_keepalives += 1
                        continue
                    items.append(got)
        finally:
            await stream.aclose()
        return seen_keepalives, items, str(excinfo.value)

    seen_keepalives, items, message = asyncio.run(_run())

    assert items == ["data: a"], f"the first item must be relayed; got {items}"
    assert (
        seen_keepalives >= 3
    ), f"a stall after the first token must keepalive too; got {seen_keepalives}"
    assert (
        message == "The model stopped producing tokens mid-response."
    ), f"the stall guard must still fire, with its own message; got {message!r}"


def test_first_token_deadline_still_fires_and_no_sentinel_when_disabled():
    pump_ns = _load_pump()
    keepalive = pump_ns["_LLAMA_STREAM_KEEPALIVE"]

    async def _run():
        upstream = _ScriptedIter(gate = asyncio.Event())
        seen = []
        stream = pump_ns["_aiter_llama_stream_items"](
            upstream,
            first_token_deadline = time.monotonic() + 0.2,
            keepalive_interval_s = None,
        )
        try:
            with pytest.raises(httpx.ReadTimeout) as excinfo:
                async for got in stream:
                    seen.append(got)
        finally:
            await stream.aclose()
        return seen, str(excinfo.value)

    seen, message = asyncio.run(_run())

    assert (
        seen == []
    ), f"keepalive_interval_s=None must relay in silence, emitting no sentinel; got {seen}"
    assert (
        message == "The model did not produce a first token in time."
    ), f"the first-token deadline must still bound the wait; got {message!r}"
    assert keepalive is not None


def test_teardown_cancels_the_in_flight_read():
    pump_ns = _load_pump()
    keepalive = pump_ns["_LLAMA_STREAM_KEEPALIVE"]

    async def _run():
        upstream = _ScriptedIter(gate = asyncio.Event())
        stream = pump_ns["_aiter_llama_stream_items"](
            upstream,
            first_token_deadline = time.monotonic() + 30.0,
            keepalive_interval_s = _TICK_S,
        )
        async for got in stream:
            assert got is keepalive
            break
        # Abandoning mid-stall must not leave the read running: the response is
        # closed right after, and a live __anext__ would raise "already running".
        await stream.aclose()
        await asyncio.sleep(0)
        return upstream

    upstream = asyncio.run(_run())

    assert (
        upstream.cancelled == 1
    ), f"aclose() must cancel the pending read exactly once; got {upstream.cancelled}"


# ── Structural (AST) ─────────────────────────────────────────


def _passthrough_relay_loops():
    loops = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.AsyncFor):
            continue
        call = node.iter
        if not isinstance(call, ast.Call):
            continue
        if not (isinstance(call.func, ast.Name) and call.func.id == "_aiter_llama_stream_items"):
            continue
        loops.append((node, call))
    return loops


def test_every_relay_loop_translates_the_sentinel_first():
    loops = _passthrough_relay_loops()
    assert len(loops) == 4, (
        "expected the 4 passthrough surfaces (/v1/completions, /v1/responses, "
        f"/v1/messages, /v1/chat/completions); found {len(loops)}. A new surface "
        "must handle _LLAMA_STREAM_KEEPALIVE too."
    )

    for node, call in loops:
        line = node.lineno
        keywords = {kw.arg for kw in call.keywords}
        assert "keepalive_interval_s" in keywords, (
            f"relay loop at line {line} must pass keepalive_interval_s, or it "
            "relays in silence for the whole prefill"
        )

        first = node.body[0]
        assert isinstance(first, ast.If), (
            f"relay loop at line {line} must test the sentinel as its FIRST "
            f"statement; found {type(first).__name__}"
        )
        test = first.test
        assert (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.comparators[0], ast.Name)
            and test.comparators[0].id == "_LLAMA_STREAM_KEEPALIVE"
        ), (
            f"relay loop at line {line} must begin with `<item> is "
            "_LLAMA_STREAM_KEEPALIVE`, before any startswith()/truthiness filter "
            "could drop it or any bookkeeping could count it as an upstream item"
        )


def test_sentinel_is_an_object_not_a_string():
    # A string would be silently eaten by the call sites' startswith("data:")
    # filters, and could in principle collide with upstream bytes.
    pump_ns = _load_pump()
    keepalive = pump_ns["_LLAMA_STREAM_KEEPALIVE"]
    assert not isinstance(
        keepalive, (str, bytes)
    ), f"sentinel must not be a string; got {type(keepalive).__name__}"


# ── Client tolerance ─────────────────────────────────────────


@pytest.mark.parametrize("module_name", ["openai", "anthropic"])
def test_sdk_sse_readers_ignore_the_keepalive_comment(module_name):
    """The whole fix rests on `:` comments being invisible to SSE readers."""
    streaming = pytest.importorskip(f"{module_name}._streaming")
    decoder = streaming.SSEDecoder()
    # split("\n"), not splitlines(): the trailing blank line is the dispatch
    # point, so dropping it would skip the only place an event could surface.
    events = [
        event
        for line in ": keep-alive\n\n".split("\n")
        for event in [decoder.decode(line)]
        if event is not None
    ]
    assert events == [], f"{module_name} SSE reader must ignore a comment; got {events}"
