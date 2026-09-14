# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The approval park must give its cached context back, through the real route.

The admission tests drive the lease directly, so they still pass if nothing ever wires
llama-server's erasure to it. This drives the ASGI chat route with a backend double that
gates a tool call on confirmation, which is the only place that wiring exists.
"""

import asyncio
import json
import threading

import pytest
from fastapi import FastAPI

from auth.authentication import get_current_subject
from core.inference import llama_admission
import routes.inference as inference_route
from .llama_backend_double import FakeLlamaCppBackend
from .asgi_stream_helpers import wait_for_frame


BASE_URL = "http://llama.test"
BUDGET = 35000
DECODE_SLOT = 2


@pytest.fixture(autouse = True)
def _fresh_queues():
    llama_admission.reset_llama_admission_queues()
    yield
    llama_admission.reset_llama_admission_queues()


class _ApprovalGatedBackend(FakeLlamaCppBackend):
    """Decodes one round, then blocks on an unanswered tool-approval prompt."""

    base_url = BASE_URL
    effective_parallel_slots = 4
    supports_tools = True
    context_length = BUDGET

    def __init__(self, erasure_succeeds = True):
        self.answered = threading.Event()
        self.erasure_succeeds = erasure_succeeds
        self.erased = []

    def release_idle_chat_slot(self, base_url, slot):
        self.erased.append((base_url, slot))
        return self.erasure_succeeds

    def generate_chat_completion_with_tools(self, **kwargs):
        kwargs["on_decode_slot"](BASE_URL, DECODE_SLOT)
        yield {
            "type": "tool_start",
            "name": "python",
            "awaiting_confirmation": True,
            "approval_id": "a1",
        }
        self.answered.wait(10)
        yield {"type": "tool_end", "name": "python", "result": "ok"}
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            "timings": {"prompt_n": 4, "predicted_n": 1},
            "finish_reason": "stop",
        }


def _queue():
    return llama_admission.get_llama_admission_queue(BASE_URL)


async def _until(
    predicate,
    what,
    timeout = 10.0,
):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}: {_queue().snapshot()}")


def _request_scope(app, body):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/chat/completions",
        "raw_path": b"/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
            (b"x-unsloth-events", b"1"),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app,
    }


def _install(monkeypatch, backend):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _one_tool(payload, **kwargs):
        return [
            {
                "type": "function",
                "function": {"name": "python", "description": "run python", "parameters": {}},
            }
        ]

    monkeypatch.setattr(inference_route, "_select_request_tools", _one_tool)

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app


def test_an_unanswered_approval_gives_its_context_to_a_queued_chat(monkeypatch):
    backend = _ApprovalGatedBackend()
    app = _install(monkeypatch, backend)

    async def _drive():
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "compute 17 * 23 " + "x " * 4000}],
                "stream": True,
                "confirm_tool_calls": True,
            }
        ).encode()
        done = asyncio.Event()

        async def receive():
            if not done.is_set():
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            if message.get("type") == "http.response.body":
                if message.get("body", b"").decode() == inference_route._SSE_DONE_CHUNK:
                    done.set()

        task = asyncio.create_task(app(_request_scope(app, body), receive, send))
        rival = None
        try:
            await _until(
                lambda: _queue().snapshot().active == 0 and _queue().snapshot().committed > 0,
                "the approval to park while still holding its context",
            )
            held = _queue().snapshot().committed
            assert not backend.erased, "nothing is waiting, so nothing should be erased"

            # Sized to fit only once the parked chat gives its context back.
            rival = _queue().reserve(
                capacity = 4,
                config = llama_admission.LlamaAdmissionConfig(),
                tokens = BUDGET - held + 1,
                budget = BUDGET,
            )
            assert rival.lease_nowait() is None, "the rival should be queued behind the approval"

            await _until(lambda: bool(backend.erased), "the parked chat's cache to be erased")
            assert backend.erased == [(BASE_URL, DECODE_SLOT)]
            await _until(lambda: rival.lease_nowait() is not None, "the rival to be admitted")

            # The approved chat may only resume once the rival hands the context back.
            backend.answered.set()
            await asyncio.sleep(0.1)
            assert not done.is_set(), "resuming must wait for the context, not just a slot"
            rival.lease_nowait().release()
            rival = None
            await wait_for_frame(done, task, what = "the [DONE] frame")
        finally:
            if rival is not None:
                rival.cancel()
            backend.answered.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(_drive())


def test_an_unacknowledged_erasure_keeps_the_context_reserved(monkeypatch):
    """llama-server did not confirm the erasure, so the cache may still be resident.

    Handing the tokens back anyway would admit another chat against room nothing has
    freed, which is the overcommit the whole accounting exists to prevent.
    """
    backend = _ApprovalGatedBackend(erasure_succeeds = False)
    app = _install(monkeypatch, backend)

    async def _drive():
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "compute 17 * 23 " + "x " * 4000}],
                "stream": True,
                "confirm_tool_calls": True,
            }
        ).encode()
        done = asyncio.Event()

        async def receive():
            if not done.is_set():
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            if message.get("type") == "http.response.body":
                if message.get("body", b"").decode() == inference_route._SSE_DONE_CHUNK:
                    done.set()

        task = asyncio.create_task(app(_request_scope(app, body), receive, send))
        rival = None
        try:
            await _until(
                lambda: _queue().snapshot().active == 0 and _queue().snapshot().committed > 0,
                "the approval to park",
            )
            held = _queue().snapshot().committed
            rival = _queue().reserve(
                capacity = 4,
                config = llama_admission.LlamaAdmissionConfig(),
                tokens = BUDGET - held + 1,
                budget = BUDGET,
            )
            assert rival.lease_nowait() is None
            await _until(lambda: bool(backend.erased), "the erasure to be attempted")
            await asyncio.sleep(0.3)
            assert _queue().snapshot().committed == held, "an unconfirmed erasure frees nothing"
            assert rival.lease_nowait() is None, "the rival must stay queued"
        finally:
            if rival is not None:
                rival.cancel()
            backend.answered.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(_drive())


class _RecostLease:
    def __init__(self):
        self.allow_yield = None

    def recost_waiting(
        self,
        tokens,
        *,
        cancel_event = None,
        allow_yield = True,
    ):
        self.allow_yield = allow_yield
        return False


class _RecostReservation:
    def __init__(self, lease):
        self._lease = lease

    def lease_nowait(self):
        return self._lease


class _NoIdleClearingBackend(FakeLlamaCppBackend):
    """Windows full GPU offload: --cache-ram 0, so an idle slot keeps its cells."""

    base_url = BASE_URL
    effective_parallel_slots = 4
    context_length = BUDGET
    idle_slot_clearing_active = False


@pytest.mark.parametrize(
    ("cache_is_empty", "expected"),
    [(False, False), (True, True)],
)
def test_a_recost_may_wait_for_room_once_the_cells_were_erased(cache_is_empty, expected):
    lease = _RecostLease()

    inference_route._openai_llama_admission_recost(
        _RecostReservation(lease),
        [{"role": "user", "content": "hi"}],
        request = None,
        llama_backend = _NoIdleClearingBackend(),
        output_tokens = 256,
        cache_is_empty = cache_is_empty,
    )

    assert lease.allow_yield is expected


class _GrowsAfterApprovalBackend(_ApprovalGatedBackend):
    """Runs the next round after the approval, which is where the grown cost is charged."""

    def generate_chat_completion_with_tools(self, **kwargs):
        kwargs["on_decode_slot"](BASE_URL, DECODE_SLOT)
        yield {
            "type": "tool_start",
            "name": "python",
            "awaiting_confirmation": True,
            "approval_id": "a1",
        }
        self.answered.wait(10)
        yield {"type": "tool_end", "name": "python", "result": "ok"}
        kwargs["on_conversation_grew"](
            [
                {"role": "user", "content": "compute 17 * 23 " + "x " * 4000},
                {"role": "tool", "content": "391 " + "y " * 4000},
            ]
        )
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            "timings": {"prompt_n": 4, "predicted_n": 1},
            "finish_reason": "stop",
        }


def test_the_round_after_a_reclaim_is_told_its_cells_are_gone(monkeypatch):
    """Erased cells make yielding honest, so the grown round may wait instead of decoding.

    Without it the recost declines wherever idle slots are not cleared and the larger
    prompt goes out anyway, over a budget the pool refused it.
    """
    backend = _GrowsAfterApprovalBackend()
    app = _install(monkeypatch, backend)
    recosts = []
    real_recost = inference_route._openai_llama_admission_recost
    monkeypatch.setattr(
        inference_route,
        "_openai_llama_admission_recost",
        lambda *a, **kw: recosts.append(kw.get("cache_is_empty")),
    )
    assert real_recost is not None

    async def _drive():
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": "compute 17 * 23 " + "x " * 4000}],
                "stream": True,
                "confirm_tool_calls": True,
            }
        ).encode()
        done = asyncio.Event()

        async def receive():
            if not done.is_set():
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            if message.get("type") == "http.response.body":
                if message.get("body", b"").decode() == inference_route._SSE_DONE_CHUNK:
                    done.set()

        task = asyncio.create_task(app(_request_scope(app, body), receive, send))
        rival = None
        try:
            await _until(
                lambda: _queue().snapshot().active == 0 and _queue().snapshot().committed > 0,
                "the approval to park while still holding its context",
            )
            held = _queue().snapshot().committed
            rival = _queue().reserve(
                capacity = 4,
                config = llama_admission.LlamaAdmissionConfig(),
                tokens = BUDGET - held + 1,
                budget = BUDGET,
            )
            assert rival.lease_nowait() is None
            await _until(lambda: bool(backend.erased), "the parked chat's cache to be erased")
            await _until(lambda: rival.lease_nowait() is not None, "the rival to be admitted")

            backend.answered.set()
            rival.lease_nowait().release()
            rival = None
            await wait_for_frame(done, task, what = "the [DONE] frame")
        finally:
            if rival is not None:
                rival.cancel()
            backend.answered.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(_drive())

    assert recosts == [True], f"the round after the erasure was told {recosts}"
