# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A passthrough relay must not go silent while llama-server prefills.

llama-server sends nothing until the first token and undici aborts a response
after 300s with no bytes. Behavioural checks that the pump keepalives across a
stalled read without restarting it, that the stall guard still fires, and that
teardown leaves no pending read; structural checks that no fifth passthrough
surface can be added without translating the sentinel.
"""

from __future__ import annotations

import ast
import math
import asyncio
import time
from pathlib import Path

import httpx
import pytest


SOURCE_PATH = Path(__file__).resolve().parents[2] / "studio" / "backend" / "routes" / "inference.py"
SRC = SOURCE_PATH.read_text(encoding = "utf-8")
_TREE = ast.parse(SRC)

_TICK_S = 0.05

_WANTED = {"_LlamaStreamKeepalive", "_LLAMA_STREAM_KEEPALIVE", "_aiter_llama_stream_items"}


def _load_pump():
    """The real pump, cut from disk and exec'd against stubs: importing
    `routes.inference` would pull in the whole backend package."""
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
        "_set_stream_response_read_timeout": lambda response, read_timeout_s = None: None,
        "_discard_task_outcome": lambda task: None,
    }
    exec("\n\n".join(chunks), namespace)
    return namespace


class _ScriptedIter:
    """Yields, blocks on a gate, yields again. Counts `__anext__` entries so a
    test can prove the pump waited on one read rather than restarting per tick."""

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
        # A live __anext__ here would raise "already running" on the close.
        await stream.aclose()
        await asyncio.sleep(0)
        return upstream

    upstream = asyncio.run(_run())

    assert (
        upstream.cancelled == 1
    ), f"aclose() must cancel the pending read exactly once; got {upstream.cancelled}"


def test_closing_the_pump_first_leaves_the_iterator_closable():
    """Closing the iterator while the pump still holds its read raises
    "already running"; closing the pump first must make it clean."""
    pump_ns = _load_pump()
    keepalive = pump_ns["_LLAMA_STREAM_KEEPALIVE"]

    async def _run(close_pump_first):
        gate = asyncio.Event()

        async def _upstream():
            try:
                await gate.wait()
                yield "data: never"
            finally:
                pass

        iterator = _upstream()
        stream = pump_ns["_aiter_llama_stream_items"](
            iterator,
            first_token_deadline = time.monotonic() + 30.0,
            keepalive_interval_s = _TICK_S,
        )
        async for got in stream:
            assert got is keepalive
            break  # parked at a keepalive yield, read in flight

        errors = []
        if close_pump_first:
            try:
                await stream.aclose()
            except Exception as exc:  # pragma: no cover - would be a real defect
                errors.append(f"pump: {type(exc).__name__}: {exc}")
        try:
            await iterator.aclose()
        except Exception as exc:
            errors.append(f"iterator: {type(exc).__name__}: {exc}")
        if not close_pump_first:
            await stream.aclose()
        return errors

    # Without the ordering, the iterator close lands on a running generator.
    unordered = asyncio.run(_run(close_pump_first = False))
    assert any("already running" in e for e in unordered), (
        "expected the unordered close to hit 'asynchronous generator is already "
        f"running'; got {unordered}. If this stops reproducing the ordering "
        "contract may no longer be load-bearing."
    )

    ordered = asyncio.run(_run(close_pump_first = True))
    assert ordered == [], f"closing the pump first must leave a clean close; got {ordered}"


def _load_env_accessors():
    wanted = {
        "_positive_float_env",
        "_finite_positive_float_env",
        "_openai_passthrough_stream_keepalive_interval",
        "_first_token_timeout_s",
        "_OPENAI_COMPAT_STREAM_KEEPALIVE_ENV",
        "_OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT_ENV",
        "_OPENAI_PASSTHROUGH_PENDING_RESPONSE_KEEPALIVE_S",
    }
    chunks = []
    for node in _TREE.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            ok = node.name in wanted
        elif isinstance(node, ast.Assign):
            ok = any(t.id in wanted for t in node.targets if isinstance(t, ast.Name))
        else:
            ok = False
        if ok:
            chunks.append(ast.get_source_segment(SRC, node))
    ns = {
        "os": __import__("os"),
        "math": math,
        "_DEFAULT_FIRST_TOKEN_TIMEOUT_S": 1200.0,
    }
    exec("\n\n".join(chunks), ns)
    return ns


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 5.0),  # unset: pace like the header wait already does
        ("", 5.0),  # blank: same as unset
        ("0", None),  # documented off switch
        ("-1", None),  # non-positive is also off
        ("2.5", 2.5),
        ("garbage", 5.0),  # unparseable falls back, never crashes a stream
    ],
)
def test_keepalive_interval_env(monkeypatch, raw, expected):
    ns = _load_env_accessors()
    name = ns["_OPENAI_COMPAT_STREAM_KEEPALIVE_ENV"]
    monkeypatch.delenv(name, raising = False)
    if raw is not None:
        monkeypatch.setenv(name, raw)
    assert ns["_openai_passthrough_stream_keepalive_interval"]() == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, 1200.0),
        ("", 1200.0),
        ("0", 1200.0),  # unlike the stall guard, 0 must NOT remove the bound
        ("-5", 1200.0),
        ("garbage", 1200.0),
        ("2400", 2400.0),
        # `inf` is positive, so a `value > 0` parser used to let it through and
        # the deadline stopped existing. `1e309` is the same value written by a
        # human who meant "very large".
        ("inf", 1200.0),
        ("Infinity", 1200.0),
        ("1e309", 1200.0),
        ("nan", 1200.0),
    ],
)
def test_first_token_timeout_env_never_unbounded(monkeypatch, raw, expected):
    ns = _load_env_accessors()
    name = ns["_OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT_ENV"]
    monkeypatch.delenv(name, raising = False)
    if raw is not None:
        monkeypatch.setenv(name, raw)
    got = ns["_first_token_timeout_s"]()
    assert got == expected
    assert isinstance(got, float) and got > 0, (
        "the first-token deadline is unconditional downstream, so this accessor "
        f"must never return None or a non-positive value; got {got!r}"
    )
    assert math.isfinite(got), (
        "a non-finite bound is the same as no bound: this value also builds the "
        f"non-streaming httpx.Timeout, whose positional form covers connect, "
        f"read, write and pool; got {got!r}"
    )


_PUMP_LOCAL = "items_iter"


def _pump_constructions():
    """`items_iter = _aiter_llama_stream_items(...)` assignments."""
    made = []
    for node in ast.walk(_TREE):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        if not (isinstance(call.func, ast.Name) and call.func.id == "_aiter_llama_stream_items"):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        made.append((node, call, targets))
    return made


def _passthrough_relay_loops():
    return [
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.AsyncFor)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == _PUMP_LOCAL
    ]


def test_pump_is_bound_not_inlined_so_teardown_can_close_it():
    """Inlining the pump in the `async for` leaves it unnameable, so teardown
    cannot aclose() it before the iterator it is still reading."""
    made = _pump_constructions()
    assert len(made) == 4, (
        "expected the 4 passthrough surfaces to each bind the pump to a local; "
        f"found {len(made)}"
    )
    for node, call, targets in made:
        assert targets == [
            _PUMP_LOCAL
        ], f"pump at line {node.lineno} must bind to {_PUMP_LOCAL!r}, got {targets}"
        keywords = {kw.arg for kw in call.keywords}
        assert "keepalive_interval_s" in keywords, (
            f"pump at line {node.lineno} must pass keepalive_interval_s, or it "
            "relays in silence for the whole prefill"
        )

    inlined = [
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.AsyncFor)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "_aiter_llama_stream_items"
    ]
    assert not inlined, (
        "the pump must not be inlined into an `async for`; teardown cannot "
        f"aclose() what it cannot name (lines {[n.lineno for n in inlined]})"
    )


def test_every_teardown_closes_the_pump_before_the_iterator():
    calls = [
        node
        for node in ast.walk(_TREE)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_aclose_stream_resources"
    ]
    with_iterator = [c for c in calls if any(kw.arg == "iterator" for kw in c.keywords)]
    assert (
        len(with_iterator) == 4
    ), f"expected 4 teardowns closing a relay iterator; found {len(with_iterator)}"
    for call in with_iterator:
        names = [kw.arg for kw in call.keywords]
        assert "items" in names, (
            f"teardown at line {call.lineno} closes `iterator` but not `items`: "
            "the pump would still be running on it"
        )
        assert names.index("items") < names.index("iterator"), (
            f"teardown at line {call.lineno} must pass `items` before `iterator` "
            "to match the documented close order"
        )

    helper = next(
        n
        for n in ast.walk(_TREE)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_aclose_stream_resources"
    )
    args = [a.arg for a in helper.args.kwonlyargs]
    assert "items" in args and args.index("items") < args.index(
        "iterator"
    ), f"_aclose_stream_resources must take `items` before `iterator`; got {args}"
    body_src = ast.get_source_segment(SRC, helper) or ""
    assert body_src.index("items.aclose()") < body_src.index(
        "iterator.aclose()"
    ), "_aclose_stream_resources must await items.aclose() before iterator.aclose()"


def test_every_relay_loop_translates_the_sentinel_first():
    loops = _passthrough_relay_loops()
    assert len(loops) == 4, (
        "expected the 4 passthrough surfaces (/v1/completions, /v1/responses, "
        f"/v1/messages, /v1/chat/completions); found {len(loops)}. A new surface "
        "must handle _LLAMA_STREAM_KEEPALIVE too."
    )

    for node in loops:
        line = node.lineno
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
    # A string would be eaten by the call sites' startswith("data:") filters.
    pump_ns = _load_pump()
    keepalive = pump_ns["_LLAMA_STREAM_KEEPALIVE"]
    assert not isinstance(
        keepalive, (str, bytes)
    ), f"sentinel must not be a string; got {type(keepalive).__name__}"


@pytest.mark.parametrize("module_name", ["openai", "anthropic"])
def test_sdk_sse_readers_ignore_the_keepalive_comment(module_name):
    """The whole fix rests on `:` comments being invisible to SSE readers."""
    streaming = pytest.importorskip(f"{module_name}._streaming")
    decoder = streaming.SSEDecoder()
    # split, not splitlines: the trailing blank line is the dispatch point.
    events = [
        event
        for line in ": keep-alive\n\n".split("\n")
        for event in [decoder.decode(line)]
        if event is not None
    ]
    assert events == [], f"{module_name} SSE reader must ignore a comment; got {events}"


def test_the_tick_is_a_whole_comment_frame_not_a_bare_line():
    """The tick must be a complete SSE comment frame, blank line included.

    A bare `: keep-alive\\n` looks tempting: it puts bytes on the wire without
    closing a block, which avoids an SDK decoder bug where a data-less block
    dispatches an empty event once an `id:` has been seen. But a bare line rides
    at the head of the NEXT frame, and a reader that classifies a frame by its
    first character -- comment if it starts with ":", data if it starts with
    "data:" -- then files the whole frame as a comment and DROPS THE CHUNK. Raw
    curl and Node undici readers both do exactly that, and both lose the token.

    So the frame form is deliberate: it costs an ignorable empty event on two
    Python decoders, and only when the upstream sends `id:`, which llama-server
    does not. Losing a token is much worse than emitting an ignorable one.
    """
    assert '_OPENAI_PASSTHROUGH_SSE_KEEPALIVE = ": keep-alive\\n\\n"' in SRC
    assert (
        "_OPENAI_PASSTHROUGH_SSE_KEEPALIVE_LINE" not in SRC
    ), "the bare-line form drops a chunk for frame-prefix readers"
    emitters = SRC.count("yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE")
    assert emitters >= 4, f"expected at least the 4 tick emitters, found {emitters}"

    # A frame-prefix reader, which is what curl and undici consumers do.
    def frame_prefix_reader(stream):
        text = []
        for frame in stream.split("\n\n"):
            if frame.startswith(":") or not frame.strip():
                continue
            if frame.startswith("data: "):
                text.append(frame[6:])
        return text

    tick_frame = "data: a\n\n: keep-alive\n\ndata: b\n\n"
    tick_line = "data: a\n\n: keep-alive\ndata: b\n\n"
    assert frame_prefix_reader(tick_frame) == ["a", "b"]
    assert frame_prefix_reader(tick_line) == [
        "a"
    ], "this is the data loss the frame form exists to avoid"
