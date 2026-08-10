# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Ordering rules for the early admission release at ``data: [DONE]``.

Freeing the llama-server slot at the sentinel is only correct when two things hold, and on a
one-slot backend both are load-bearing:

1. The release happens *before* the sentinel reaches the ASGI ``send()``. Starlette's
   ``stream_response`` suspends the body iterator at its ``yield`` for the whole of
   ``await send(...)``, and uvicorn's ``send()`` awaits ``flow.drain()`` on a write-paused
   transport, so a client that stops reading parks the generator there indefinitely. Starlette
   never ``aclose()``s a body iterator either, so that generator's ``finally`` is left to GC.

2. The sentinel really means "llama-server is done with this request". Two other emitters end
   in the same bytes: ``_openai_stream_error_sse``, yielded from inside the still-suspended
   generator's ``except`` block, and the cancel path, which breaks the read loop while the sync
   generator is still parked on a yield inside ``_open_stream``'s httpx client.
"""

import asyncio
import json
import threading

import pytest
from fastapi import FastAPI

from auth.authentication import get_current_subject
from core.inference import llama_admission
import routes.inference as inference_route


@pytest.fixture(autouse = True)
def _fresh_queues():
    llama_admission.reset_llama_admission_queues()
    yield
    llama_admission.reset_llama_admission_queues()


def _active_slots() -> int:
    with llama_admission._QUEUES_LOCK:
        queues = list(llama_admission._QUEUES.values())
    return sum(queue.snapshot().active for queue in queues)


class _OneSlotBackend:
    """A loaded 1-parallel GGUF backend, the shape CI runs."""

    is_loaded = True
    model_identifier = "test/model.gguf"
    base_url = "http://llama.test"
    effective_parallel_slots = 1
    _is_audio = False
    is_vision = False
    supports_tools = False

    def __init__(self):
        self.closing = threading.Event()
        self.finish_close = threading.Event()
        self.closed = threading.Event()
        self.cancel_event = None

    def generate_chat_completion(self, **kwargs):
        raise NotImplementedError


class _CompletingBackend(_OneSlotBackend):
    def generate_chat_completion(self, **kwargs):
        yield "hi"
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "timings": {"prompt_n": 3, "predicted_n": 1},
            "finish_reason": "stop",
        }


class _FailsMidStreamBackend(_OneSlotBackend):
    """Still decoding when the route's own chunk handling blows up.

    ``gen`` stays parked on its ``yield`` until the stream's ``finally`` closes it, and only
    that close drops the httpx stream llama-server is writing to.
    """

    def generate_chat_completion(self, **kwargs):
        try:
            yield "a"
            yield "ab"
            yield "abc"
        except GeneratorExit:
            self.closing.set()
            # Stand in for the time llama-server needs to notice the drop and free its slot.
            self.finish_close.wait(10.0)
            self.closed.set()
            raise


class _CancelledMidStreamBackend(_OneSlotBackend):
    """Cancelled by the user halfway through, the Stop-button path."""

    def generate_chat_completion(
        self,
        cancel_event = None,
        **kwargs,
    ):
        self.cancel_event = cancel_event
        try:
            yield "a"
            cancel_event.set()
            yield "ab"
            yield "abc"
        except GeneratorExit:
            self.closed.set()
            raise


def _scope(app, body: bytes) -> dict:
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
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
        "app": app,
    }


def _build_app(monkeypatch, backend):
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: False)
    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app


def _request_body() -> bytes:
    return json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True}).encode()


def test_slot_is_free_before_the_done_frame_reaches_send(monkeypatch):
    """The release must not sit behind ``await send(...)``.

    uvicorn's ``send()`` awaits ``flow.drain()`` on a write-paused socket (h11_impl.py), so a
    client that stops reading parks the body iterator on its ``yield`` indefinitely. Anything
    after that ``yield`` is unreachable, and Starlette never ``aclose()``s the iterator, so the
    outer ``finally`` is left to GC.
    """
    backend = _CompletingBackend()
    app = _build_app(monkeypatch, backend)

    async def _drive():
        body = _request_body()
        frames = []
        slots_at_done = []
        finished = asyncio.Event()

        async def receive():
            if not frames:
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            frames.append(message)
            if message.get("type") != "http.response.body":
                return
            if message.get("body", b"").decode() == "data: [DONE]\n\n":
                # Sampled exactly where a stalled client would wedge.
                slots_at_done.append(_active_slots())
                finished.set()

        task = asyncio.create_task(app(_scope(app, body), receive, send))
        try:
            await asyncio.wait_for(finished.wait(), timeout = 20.0)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

        assert slots_at_done == [0], (
            "the slot was still held while the [DONE] frame was being written; "
            "a client that stops reading would pin it there indefinitely"
        )

    asyncio.run(_drive())


def test_error_sentinel_keeps_the_slot_until_the_generator_is_closed(monkeypatch):
    """``_openai_stream_error_sse`` ends in ``data: [DONE]`` but is not a finish.

    It is yielded from inside ``gguf_stream_chunks``'s ``except`` block, so the generator has
    not yet run its ``finally``: the worker is undrained and ``gen`` is still open with
    llama-server streaming into it. Freeing the slot there puts two callers on a one-slot
    backend.
    """
    backend = _FailsMidStreamBackend()
    app = _build_app(monkeypatch, backend)

    calls = {"n": 0}

    def _boom(monitor_id, text):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("chunk handling failed")

    monkeypatch.setattr(inference_route.api_monitor, "append_reply", _boom)

    async def _drive():
        body = _request_body()
        frames = []
        saw_error = asyncio.Event()

        async def receive():
            if not frames:
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            frames.append(message)
            if message.get("type") != "http.response.body":
                return
            chunk = message.get("body", b"").decode()
            # The error form: a payload line plus the sentinel, in one chunk.
            if chunk.endswith("data: [DONE]\n\n") and chunk != "data: [DONE]\n\n":
                saw_error.set()

        task = asyncio.create_task(app(_scope(app, body), receive, send))
        try:
            await asyncio.wait_for(saw_error.wait(), timeout = 20.0)
            # Wait until cleanup reaches gen.close(), so llama-server still holds the slot.
            for _ in range(500):
                if backend.closing.is_set():
                    break
                await asyncio.sleep(0.01)
            assert backend.closing.is_set(), "cleanup never reached gen.close()"
            assert _active_slots() == 1, (
                "slot handed out while the failed request still owned "
                "llama-server; the next request would exceed the configured "
                "parallelism"
            )
        finally:
            backend.finish_close.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(_drive())


def test_cancelled_stream_keeps_the_slot_until_the_generator_is_closed(monkeypatch):
    """A cancelled stream emits the plain sentinel with ``gen`` still open.

    ``cancel_event.is_set()`` breaks the read loop at the top, so the sync generator never
    reaches StopIteration and stays parked on a ``yield`` inside ``_open_stream``'s httpx
    client. ``stream_completed`` is set all the same, which also makes the ``finally`` skip
    ``gen.close()``, so ``data: [DONE]`` here does not mean llama-server is finished.
    """
    backend = _CancelledMidStreamBackend()
    app = _build_app(monkeypatch, backend)

    wedged = asyncio.Event()

    async def _hang(watcher, *args, **kwargs):
        watcher.cancel()
        await wedged.wait()

    monkeypatch.setattr(inference_route, "_stop_local_disconnect_cancel_watcher", _hang)

    async def _drive():
        body = _request_body()
        frames = []
        saw_done = asyncio.Event()

        async def receive():
            if not frames:
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()

        async def send(message):
            frames.append(message)
            if message.get("type") != "http.response.body":
                return
            if message.get("body", b"").decode() == "data: [DONE]\n\n":
                saw_done.set()

        task = asyncio.create_task(app(_scope(app, body), receive, send))
        try:
            await asyncio.wait_for(saw_done.wait(), timeout = 20.0)
            for _ in range(50):
                if _active_slots() == 0:
                    break
                await asyncio.sleep(0.01)
            assert backend.cancel_event is not None and backend.cancel_event.is_set()
            assert (
                not backend.closed.is_set()
            ), "test setup: the generator should still be open here"
            assert _active_slots() == 1, (
                "slot freed on a cancelled stream whose llama-server request is "
                "still open; the next request would exceed the configured "
                "parallelism"
            )
        finally:
            wedged.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(_drive())
