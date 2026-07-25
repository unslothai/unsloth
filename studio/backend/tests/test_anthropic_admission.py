# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Admission-control wiring for the Anthropic /v1/messages endpoint.

The FIFO queue itself is unit-tested in test_llama_admission.py; here we exercise
how anthropic_messages reserves a slot, queues when the backend is saturated,
streams keep-alives while waiting, releases on completion, and maps rejects to
429/503. Slot occupancy is driven directly through the shared queue (keyed by the
backend base_url) so generation stays fast and no thread has to block.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from types import SimpleNamespace

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod
from routes.inference import anthropic_messages
from models.inference import AnthropicMessagesRequest
from core.inference.api_monitor import ApiMonitor
from core.inference.llama_admission import (
    ADMISSION_CONTROL_ENV,
    ADMISSION_KEEPALIVE_INTERVAL_ENV,
    ADMISSION_MAX_QUEUE_ENV,
    ADMISSION_QUEUE_PER_SLOT_ENV,
    ADMISSION_QUEUE_TIMEOUT_ENV,
    LlamaAdmissionConfig,
    get_llama_admission_queue,
    reset_llama_admission_queues,
)
from fastapi import HTTPException

_KEY = "http://llama.admission.test:9999"


@pytest.fixture(autouse = True)
def _isolate(monkeypatch):
    reset_llama_admission_queues()
    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 64))
    for name in (
        ADMISSION_CONTROL_ENV,
        ADMISSION_QUEUE_TIMEOUT_ENV,
        ADMISSION_KEEPALIVE_INTERVAL_ENV,
        ADMISSION_MAX_QUEUE_ENV,
        ADMISSION_QUEUE_PER_SLOT_ENV,
        # Legacy spellings resolve too, so clear both for isolation.
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_CONTROL",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_QUEUE_TIMEOUT",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_KEEPALIVE_INTERVAL",
        "UNSLOTH_OPENAI_COMPAT_ADMISSION_MAX_QUEUE",
    ):
        monkeypatch.delenv(name, raising = False)
    yield
    reset_llama_admission_queues()


class _Request:
    def __init__(self, disconnected = False):
        self.state = SimpleNamespace()
        self.url = SimpleNamespace(path = "/v1/messages")
        self.method = "POST"
        self._disconnected = disconnected

    async def is_disconnected(self):
        return self._disconnected


def _install_backend(
    monkeypatch,
    *,
    slots = 1,
    base_url = _KEY,
):
    def _gen_plain(**_kwargs):
        yield "ok"

    def _gen_tools(**_kwargs):
        yield {"type": "content", "text": "ok"}

    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        supports_tool_passthrough = False,
        model_identifier = "test-model",
        context_length = 2048,
        count_chat_tokens = lambda *a, **k: 2,
        generate_chat_completion = _gen_plain,
        generate_chat_completion_with_tools = _gen_tools,
        effective_parallel_slots = slots,
        base_url = base_url,
    )
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    return backend


def _payload(**fields) -> AnthropicMessagesRequest:
    base = {"max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}
    base.update(fields)
    return AnthropicMessagesRequest(**base)


def _snapshot(key = _KEY):
    return get_llama_admission_queue(key).snapshot()


def _occupy(key, capacity, n):
    """Hold ``n`` slots on the queue so the next reserve must wait; returns leases."""
    leases = []
    for _ in range(n):
        reservation = get_llama_admission_queue(key).reserve(
            capacity = capacity, config = LlamaAdmissionConfig()
        )
        lease = reservation.lease_nowait()
        assert lease is not None
        leases.append(lease)
    return leases


async def _consume(response):
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


# ── Non-streaming ─────────────────────────────────────────────


def test_non_streaming_completes_and_releases_slot(monkeypatch):
    _install_backend(monkeypatch, slots = 2)

    async def _run():
        response = await anthropic_messages(_payload(), request = _Request(), current_subject = "t")
        assert response.status_code == 200
        snap = _snapshot()
        assert snap.active == 0 and snap.queued == 0

    asyncio.run(_run())


def test_non_streaming_queue_full_returns_429(monkeypatch):
    monkeypatch.setenv(ADMISSION_MAX_QUEUE_ENV, "1")
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)  # slot busy
        # One waiter fills the max_queue=1; the next reserve rejects.
        get_llama_admission_queue(_KEY).reserve(
            capacity = 1, config = LlamaAdmissionConfig(max_queue = 1)
        )
        with pytest.raises(HTTPException) as exc:
            await anthropic_messages(_payload(), request = _Request(), current_subject = "t")
        assert exc.value.status_code == 429
        # rate_limit_error is what Anthropic SDKs back off on; overloaded_error is 529.
        assert exc.value.detail["error"]["type"] == "rate_limit_error"
        for lease in held:
            lease.release()

    asyncio.run(_run())


def test_non_streaming_times_out_returns_503(monkeypatch):
    monkeypatch.setenv(ADMISSION_QUEUE_TIMEOUT_ENV, "0.15")
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)  # never released -> waiter times out
        with pytest.raises(HTTPException) as exc:
            await anthropic_messages(_payload(), request = _Request(), current_subject = "t")
        assert exc.value.status_code == 503
        for lease in held:
            lease.release()

    asyncio.run(_run())


def test_non_streaming_queued_then_admitted(monkeypatch):
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)
        task = asyncio.create_task(
            anthropic_messages(_payload(), request = _Request(), current_subject = "t")
        )
        await asyncio.sleep(0.1)
        assert _snapshot().queued == 1  # waiting on the busy slot
        held[0].release()  # free it
        response = await asyncio.wait_for(task, timeout = 2)
        assert response.status_code == 200
        assert _snapshot().active == 0 and _snapshot().queued == 0

    asyncio.run(_run())


def test_capacity_enforced_from_effective_parallel_slots(monkeypatch):
    _install_backend(monkeypatch, slots = 3)

    async def _run():
        held = _occupy(_KEY, 3, 3)  # all 3 slots busy
        task = asyncio.create_task(
            anthropic_messages(_payload(), request = _Request(), current_subject = "t")
        )
        await asyncio.sleep(0.1)
        snap = _snapshot()
        assert snap.capacity == 3 and snap.active == 3 and snap.queued == 1
        for lease in held:
            lease.release()
        response = await asyncio.wait_for(task, timeout = 2)
        assert response.status_code == 200

    asyncio.run(_run())


def test_disabled_admission_bypasses_limit(monkeypatch):
    monkeypatch.setenv(ADMISSION_CONTROL_ENV, "off")
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)  # would block if admission were on
        response = await asyncio.wait_for(
            anthropic_messages(_payload(), request = _Request(), current_subject = "t"),
            timeout = 2,
        )
        assert response.status_code == 200
        for lease in held:
            lease.release()

    asyncio.run(_run())


# ── Streaming ─────────────────────────────────────────────────


def test_streaming_completes_and_releases_slot(monkeypatch):
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        response = await anthropic_messages(
            _payload(stream = True), request = _Request(), current_subject = "t"
        )
        blob = await _consume(response)
        assert "event: message_start" in blob
        assert "event: message_stop" in blob
        assert _snapshot().active == 0 and _snapshot().queued == 0

    asyncio.run(_run())


def test_streaming_emits_keepalives_while_queued_then_streams(monkeypatch):
    monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.05")
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)
        response = await anthropic_messages(
            _payload(stream = True), request = _Request(), current_subject = "t"
        )
        body = response.body_iterator
        # First chunk must be a keep-alive comment (slot still busy).
        first = await asyncio.wait_for(body.__anext__(), timeout = 2)
        first = first.decode() if isinstance(first, (bytes, bytearray)) else first
        assert first.startswith(":")  # SSE comment keep-alive
        held[0].release()  # free the slot -> real stream follows
        rest = await asyncio.wait_for(_drain(body), timeout = 2)
        assert "event: message_start" in rest
        assert _snapshot().active == 0 and _snapshot().queued == 0

    asyncio.run(_run())


def test_streaming_queue_full_returns_429(monkeypatch):
    monkeypatch.setenv(ADMISSION_MAX_QUEUE_ENV, "1")
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)
        get_llama_admission_queue(_KEY).reserve(
            capacity = 1, config = LlamaAdmissionConfig(max_queue = 1)
        )
        with pytest.raises(HTTPException) as exc:
            await anthropic_messages(_payload(stream = True), request = _Request(), current_subject = "t")
        assert exc.value.status_code == 429
        for lease in held:
            lease.release()

    asyncio.run(_run())


def test_streaming_disconnect_while_queued_frees_slot(monkeypatch):
    monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.05")
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)
        response = await anthropic_messages(
            _payload(stream = True), request = _Request(), current_subject = "t"
        )
        body = response.body_iterator
        await asyncio.wait_for(body.__anext__(), timeout = 2)  # one keep-alive
        assert _snapshot().queued == 1
        await body.aclose()  # client goes away mid-wait
        held[0].release()
        await asyncio.sleep(0.05)
        snap = _snapshot()
        assert snap.queued == 0 and snap.active == 0

    asyncio.run(_run())


# ── Shared queue + fairness + speed ───────────────────────────


def test_shares_queue_with_openai_by_base_url(monkeypatch):
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        # A generation admitted through the OpenAI helper occupies the same key,
        # so an Anthropic request must queue behind it.
        held = _occupy(_KEY, 1, 1)
        task = asyncio.create_task(
            anthropic_messages(_payload(), request = _Request(), current_subject = "t")
        )
        await asyncio.sleep(0.1)
        assert _snapshot().queued == 1
        held[0].release()
        assert (await asyncio.wait_for(task, timeout = 2)).status_code == 200

    asyncio.run(_run())


def test_fifo_fairness_across_many_waiters(monkeypatch):
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)
        order = []

        async def _one(i):
            resp = await anthropic_messages(_payload(), request = _Request(), current_subject = "t")
            order.append(i)
            return resp

        tasks = [asyncio.create_task(_one(i)) for i in range(8)]
        await asyncio.sleep(0.2)
        assert _snapshot().queued == 8
        held[0].release()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout = 5)
        assert order == list(range(8))  # granted in arrival order
        assert _snapshot().active == 0 and _snapshot().queued == 0

    asyncio.run(_run())


def test_uncontended_hot_path_is_fast(monkeypatch):
    _install_backend(monkeypatch, slots = 4)

    async def _run():
        start = time.perf_counter()
        for _ in range(50):
            resp = await anthropic_messages(_payload(), request = _Request(), current_subject = "t")
            assert resp.status_code == 200
        elapsed = time.perf_counter() - start
        # 50 admit+release round-trips with no contention must be well under a
        # second; admission overhead per call is sub-millisecond.
        assert elapsed < 1.0
        assert _snapshot().active == 0 and _snapshot().queued == 0

    asyncio.run(_run())


async def _drain(body):
    chunks = []
    async for chunk in body:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


def test_streaming_midstream_cancel_finalizes_the_monitor(monkeypatch):
    # A mid-stream disconnect is delivered as CancelledError so the monitored body
    # can finalize its entry. Closing the inner iterator with aclose() instead
    # delivers GeneratorExit, and the entry stays "running" for the process life.
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        response = await anthropic_messages(
            _payload(stream = True), request = _Request(), current_subject = "t"
        )
        body = response.body_iterator
        await asyncio.wait_for(body.__anext__(), timeout = 2)  # stream started
        assert inf_mod.api_monitor.active_count() == 1

        # Propagates back out, as the un-admitted path did; what matters is that
        # the monitored body saw it on the way through.
        with pytest.raises(asyncio.CancelledError):
            await body.athrow(asyncio.CancelledError())  # client vanished

        assert inf_mod.api_monitor.active_count() == 0
        assert _snapshot().active == 0 and _snapshot().queued == 0

    asyncio.run(_run())


def test_streaming_give_up_while_queued_finalizes_the_monitor(monkeypatch):
    # Cancelled before the body ever ran, so nothing downstream can close the
    # entry out; the wrapper has to do it.
    monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.05")
    _install_backend(monkeypatch, slots = 1)

    async def _run():
        held = _occupy(_KEY, 1, 1)
        response = await anthropic_messages(
            _payload(stream = True), request = _Request(), current_subject = "t"
        )
        body = response.body_iterator
        await asyncio.wait_for(body.__anext__(), timeout = 2)  # keep-alive, still queued
        assert inf_mod.api_monitor.active_count() == 1

        await body.aclose()  # give up while waiting

        assert inf_mod.api_monitor.active_count() == 0
        for lease in held:
            lease.release()
        assert _snapshot().active == 0 and _snapshot().queued == 0

    asyncio.run(_run())


def test_every_dispatch_site_goes_through_admission():
    """All six generation returns in anthropic_messages are admission-wrapped.

    The tool paths need a passthrough-capable backend and a tools payload to reach
    at runtime, so guard them structurally instead: a new dispatch site added
    without admission (or one reverted to _monitored_anthropic) fails here.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(inf_mod).replace("\t", "    "))
    handler = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "anthropic_messages"
    )
    # The wrappers themselves call _monitored_anthropic; only the dispatch sites count.
    nested = {
        node
        for node in ast.walk(handler)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("_admitted_anthropic")
    }
    inner = {id(n) for wrapper in nested for n in ast.walk(wrapper)}

    called = []
    for node in ast.walk(handler):
        if id(node) in inner or not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            called.append(node.func.id)

    assert called.count("_admitted_anthropic") == 6
    assert called.count("_monitored_anthropic") == 0


def test_queued_give_up_runs_the_response_pre_start_cleanup(monkeypatch):
    """A stream abandoned while queued must run the builder's eager cleanup.

    The passthrough enters a _TrackedCancel before returning its response and
    relies on the stream's finally to exit it. That finally never runs for a
    generator that never started, so the response carries a pre-start hook and
    the admission wrapper has to chain to it instead of replacing it.
    """
    monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.05")
    _install_backend(monkeypatch, slots = 1)
    ran = []

    async def _hook():
        ran.append(True)

    real = inf_mod._sse_streaming_response

    def _tagged(content, *, unstarted_cleanup = None):
        return real(content, unstarted_cleanup = _hook)

    monkeypatch.setattr(inf_mod, "_sse_streaming_response", _tagged)

    async def _run():
        held = _occupy(_KEY, 1, 1)
        response = await anthropic_messages(
            _payload(stream = True), request = _Request(), current_subject = "t"
        )
        body = response.body_iterator
        await asyncio.wait_for(body.__anext__(), timeout = 2)  # keep-alive, still queued
        await body.aclose()  # give up before the body ran

        assert ran == [True]
        for lease in held:
            lease.release()

    asyncio.run(_run())


def test_passthrough_stream_registers_a_pre_start_cleanup():
    # Structural guard: the tracker is entered eagerly, so the response must
    # carry the hook that exits it when the body never starts.
    import ast
    import inspect

    src = inspect.getsource(inf_mod._anthropic_passthrough_stream)
    tree = ast.parse(src.replace("\t", "    ").lstrip())
    returns = [n for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value is not None]
    call = next(
        n.value
        for n in returns
        if isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", "") == "_sse_streaming_response"
    )
    assert "unstarted_cleanup" in {kw.arg for kw in call.keywords}


def test_slot_is_released_even_if_closing_the_body_raises(monkeypatch):
    # A slot lost here never comes back: with no queue timeout the pool silently
    # shrinks and later callers wait forever, so the release must not sit behind
    # anything that can throw.
    _install_backend(monkeypatch, slots = 1)

    async def _boom(iterator, *, cancelled):
        raise RuntimeError("close failed")

    monkeypatch.setattr(inf_mod, "_close_openai_admitted_stream_iterator", _boom)

    async def _run():
        response = await anthropic_messages(
            _payload(stream = True), request = _Request(), current_subject = "t"
        )
        body = response.body_iterator
        await asyncio.wait_for(body.__anext__(), timeout = 2)  # stream started
        assert _snapshot().active == 1

        with pytest.raises(RuntimeError):
            await body.aclose()

        assert _snapshot().active == 0  # slot returned despite the failure
        # And the pool still serves the next caller.
        again = get_llama_admission_queue(_KEY).reserve(
            capacity = 1, config = LlamaAdmissionConfig()
        )
        lease = again.lease_nowait()
        assert lease is not None
        lease.release()

    asyncio.run(_run())
