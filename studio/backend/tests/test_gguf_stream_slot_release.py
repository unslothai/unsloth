# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A finished GGUF chat stream must free its llama-server slot at [DONE].

llama-server has a fixed slot count and Unsloth gates concurrent decoding on it
with an admission lease. The lease used to be released only in the stream's
outer finally, which does not run until the ASGI response body is torn down, so
any wedged teardown pinned a slot long after llama-server had freed it and the
next chat request queued behind a generation that had already finished. The
queue has no default timeout, so the wait was unbounded.

The wedge here stands in for the real one: the frontend deliberately does not
cancel its reader after [DONE] (chat-api.ts), and uvicorn advertises ASGI
spec_version 2.3, so Starlette's OSError/ClientDisconnect path -- the only
disconnect detector _SameTaskStreamingResponse keeps -- cannot fire.
"""

import asyncio
import json

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


def test_sse_chunk_is_stream_done_only_matches_the_success_sentinel():
    done = inference_route._sse_chunk_is_stream_done
    assert done("data: [DONE]\n\n")
    # The error path packs a payload line and the sentinel into one chunk. Its
    # cleanup has not run yet, so it must not look like a finished stream.
    assert not done('data: {"error": {"message": "boom"}}\n\ndata: [DONE]\n\n')
    assert not done('data: {"choices": [{"delta": {"content": "data: [DONE]"}}]}\n\n')
    assert not done("")
    assert not done(None)


_ONE_SLOT = llama_admission.LlamaAdmissionConfig(max_queue = 4)


def _reserve_one_slot():
    """Take the single slot of a 1-parallel backend. Needs a running loop."""
    queue = llama_admission.get_llama_admission_queue("http://llama.test")
    reservation = queue.reserve(capacity = 1, config = _ONE_SLOT)
    return queue, reservation.lease_nowait()


def test_slot_is_freed_at_done_even_if_teardown_never_finishes():
    """Model the stream's shape: yield chunks, then wedge in the finally.

    Without the release at [DONE] the slot stays held for as long as the
    teardown is stuck, which is what starved the following request in CI.
    """
    wedged = asyncio.Event()

    async def _stream():
        try:
            yield 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\n'
            yield "data: [DONE]\n\n"
        finally:
            # Stand-in for a teardown that never completes.
            await wedged.wait()

    async def _admitted(held):
        iterator = _stream()
        try:
            async for chunk in iterator:
                yield chunk
                if held is not None and inference_route._sse_chunk_is_stream_done(chunk):
                    held.release()
        finally:
            if held is not None:
                held.release()

    async def _drive():
        queue, lease = _reserve_one_slot()
        assert lease is not None
        assert _active_slots() == 1

        seen = []
        saw_done = asyncio.Event()

        async def _consume():
            # Stands in for Starlette's stream_response: it keeps pulling after
            # the last chunk, so the generator resumes past [DONE] and only then
            # runs into the wedged teardown.
            async for chunk in _admitted(lease):
                seen.append(chunk)
                if inference_route._sse_chunk_is_stream_done(chunk):
                    saw_done.set()

        task = asyncio.create_task(_consume())
        try:
            await asyncio.wait_for(saw_done.wait(), timeout = 5.0)
            # Give the generator a turn to resume past the [DONE] yield and
            # reach the wedge, without letting the wedge end the test.
            for _ in range(50):
                if _active_slots() == 0:
                    break
                await asyncio.sleep(0.01)
            assert not task.done(), "teardown should still be wedged"
            assert _active_slots() == 0, (
                "slot still held after [DONE]; the next chat request would "
                "queue behind a generation that already finished"
            )
            # A second caller must be admitted right away.
            second = queue.reserve(capacity = 1, config = _ONE_SLOT).lease_nowait()
            assert second is not None, "next request was refused a free slot"
            second.release()
        finally:
            wedged.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)
        return seen

    seen = asyncio.run(_drive())
    assert seen[-1] == "data: [DONE]\n\n"


def test_release_is_idempotent_so_the_finally_stays_a_backstop():
    async def _drive():
        _queue, lease = _reserve_one_slot()
        assert _active_slots() == 1
        lease.release()
        lease.release()
        assert _active_slots() == 0

    asyncio.run(_drive())


def test_stopping_the_disconnect_watcher_cannot_hang():
    """The watcher stop runs in the stream's finally; it must be bounded."""

    async def _drive():
        started = asyncio.Event()

        release = asyncio.Event()

        async def _unstoppable():
            started.set()
            while not release.is_set():
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    # A watcher that swallows cancellation, as the real one does
                    # for every exception on its way out.
                    if release.is_set():
                        raise
                    continue

        watcher = asyncio.create_task(_unstoppable())
        await started.wait()
        # Would hang forever if the stop awaited the watcher outright.
        await asyncio.wait_for(
            inference_route._stop_local_disconnect_cancel_watcher(watcher, timeout_s = 0.2),
            timeout = 5.0,
        )
        assert not watcher.done(), "watcher should have been abandoned, not awaited"
        release.set()
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions = True)

    asyncio.run(_drive())


class _OneSlotGgufBackend:
    """A loaded 1-parallel GGUF backend, the shape CI runs."""

    is_loaded = True
    model_identifier = "test/model.gguf"
    base_url = "http://llama.test"
    effective_parallel_slots = 1
    _is_audio = False
    is_vision = False
    supports_tools = False

    def generate_chat_completion(self, **kwargs):
        yield "hi"
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            "timings": {"prompt_n": 3, "predicted_n": 1},
            "finish_reason": "stop",
        }


def test_real_stream_frees_the_slot_at_done_with_a_wedged_teardown(monkeypatch):
    """Drive the real ASGI route, wedged exactly where CI wedged.

    ``_stop_local_disconnect_cancel_watcher`` runs in ``gguf_stream_chunks``'s
    finally on the success path. Hanging it reproduces a response that has sent
    [DONE] but cannot finish, which is the state the CI job was left in.
    """
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _OneSlotGgufBackend())
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: False)

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"

    async def _drive():
        wedged = asyncio.Event()

        async def _hang(watcher, *args, **kwargs):
            watcher.cancel()
            await wedged.wait()

        monkeypatch.setattr(inference_route, "_stop_local_disconnect_cancel_watcher", _hang)

        body = json.dumps(
            {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        ).encode()
        scope = {
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

        sent_body = asyncio.Event()
        frames = []

        async def receive():
            if not frames:
                return {"type": "http.request", "body": body, "more_body": False}
            # Never disconnect: the browser keeps the socket open after [DONE].
            await asyncio.Event().wait()

        async def send(message):
            frames.append(message)
            if message.get("type") == "http.response.body":
                chunk = message.get("body", b"").decode()
                if inference_route._sse_chunk_is_stream_done(chunk):
                    sent_body.set()

        task = asyncio.create_task(app(scope, receive, send))
        try:
            await asyncio.wait_for(sent_body.wait(), timeout = 20.0)
            for _ in range(200):
                if _active_slots() == 0:
                    break
                await asyncio.sleep(0.01)
            assert not task.done(), "response should still be wedged in teardown"
            assert _active_slots() == 0, (
                "slot still held after [DONE] on the real route; the next chat "
                "request would queue behind a finished generation"
            )
            queue = llama_admission.get_llama_admission_queue("http://llama.test")
            second = queue.reserve(capacity = 1, config = _ONE_SLOT).lease_nowait()
            assert second is not None, "next request was refused a free slot"
            second.release()
        finally:
            wedged.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions = True)

    asyncio.run(_drive())
