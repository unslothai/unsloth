# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Legacy /v1/completions takes the same admission lease as the chat routes.

Without one the advertised concurrency cap (--parallel, and the
UNSLOTH_API_MAX_CONCURRENCY override on top of it) was bypassable: /v1/completions
proxied straight to llama-server and ran as many generations at once as clients
cared to open.
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest


_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_admission import (
    peek_llama_admission_snapshot,
    reset_llama_admission_queues,
)


_API_CAP_ENV = "UNSLOTH_API_MAX_CONCURRENCY"
_BASE_URL = "http://llama.test"


@pytest.fixture(autouse = True)
def _reset_queues(monkeypatch):
    monkeypatch.delenv(_API_CAP_ENV, raising = False)
    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


class _Request:
    """Minimal stand-in for the Starlette Request /v1/completions reads."""

    def __init__(self, body):
        self._body = body
        self.method = "POST"
        self.url = SimpleNamespace(path = "/v1/completions")

    async def is_disconnected(self):
        return False

    async def json(self):
        return self._body


def _install_backend(monkeypatch, handler, *, backend_slots):
    """Point the proxy at an in-process llama-server that ``handler`` answers."""
    import httpx

    import routes.inference as inf_mod

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient
    monkeypatch.setattr(
        inf_mod.httpx,
        "AsyncClient",
        lambda *a, **kw: real_async_client(transport = transport, timeout = kw.get("timeout", 600)),
    )
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_loaded = True,
            context_length = 4096,
            base_url = _BASE_URL,
            model_identifier = "org/M-GGUF",
            # More decode slots than the cap, so a passing test can only be the cap
            # doing the work.
            effective_parallel_slots = backend_slots,
        ),
    )
    monkeypatch.setattr(inf_mod, "_automatic_model_load_may_run", lambda: False)

    async def _no_auto_switch(request, current_subject, **_kwargs):
        return await request.json()

    monkeypatch.setattr(inf_mod, "_auto_switch_from_request_body", _no_auto_switch)
    return inf_mod


def _sse_response(events):
    import httpx
    async def _chunks():
        for event in events:
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return httpx.Response(
        200,
        content = _chunks(),
        headers = {"content-type": "text/event-stream"},
    )


def test_non_stream_completions_holds_an_admission_slot(monkeypatch):
    """The slot must be held while llama-server is generating, not after it returns."""
    pytest.importorskip("fastapi", reason = "inference stack not installed")
    pytest.importorskip("routes.inference", reason = "inference stack not installed")
    import httpx

    monkeypatch.setenv(_API_CAP_ENV, "1")
    seen = {}

    def handler(request):
        # Sampled mid-flight: exactly the window a second request would have to wait in.
        seen["snapshot"] = peek_llama_admission_snapshot(_BASE_URL)
        return httpx.Response(200, json = {"id": "cmpl-x", "choices": [{"text": "33"}]})

    inf_mod = _install_backend(monkeypatch, handler, backend_slots = 4)

    asyncio.run(
        inf_mod.openai_completions(
            _Request({"prompt": "hi", "model": "org/M-GGUF", "max_tokens": 8}), "tester"
        )
    )

    snapshot = seen["snapshot"]
    assert snapshot is not None, "/v1/completions reached llama-server without a lease"
    assert (snapshot.capacity, snapshot.active, snapshot.free) == (1, 1, 0)
    # And it gives the slot back, or one completion would pin the pool for good.
    after = peek_llama_admission_snapshot(_BASE_URL)
    assert (after.active, after.free) == (0, 1)


def test_a_second_streaming_completion_queues_under_the_api_cap(monkeypatch):
    """--api-max-concurrency 1 with --parallel 4: the second call waits for the slot.

    Pre-fix it was admitted straight away, which is the bypass: the cap held on
    /v1/chat/completions and not on /v1/completions.
    """
    pytest.importorskip("fastapi", reason = "inference stack not installed")
    pytest.importorskip("routes.inference", reason = "inference stack not installed")

    monkeypatch.setenv(_API_CAP_ENV, "1")

    def handler(request):
        return _sse_response([{"choices": [{"text": "33"}]}])

    inf_mod = _install_backend(monkeypatch, handler, backend_slots = 4)

    def _body():
        return {"prompt": "hi", "stream": True, "model": "org/M-GGUF", "max_tokens": 8}

    async def run():
        first = await inf_mod.openai_completions(_Request(_body()), "tester")
        first_iter = first.body_iterator
        first_chunk = await asyncio.wait_for(first_iter.__anext__(), timeout = 10)

        second = await inf_mod.openai_completions(_Request(_body()), "tester")
        second_iter = second.body_iterator
        queued_chunk = await asyncio.wait_for(second_iter.__anext__(), timeout = 10)

        # Drain the holder; its lease is released when its generator finishes.
        async for _ in first_iter:
            pass

        admitted = []
        async for chunk in second_iter:
            admitted.append(chunk)
        return first_chunk, queued_chunk, b"".join(admitted)

    first_chunk, queued_chunk, admitted = asyncio.run(run())

    assert b'"text"' in first_chunk, first_chunk
    # An SSE comment, not a generation: the second caller is queued, not decoding.
    assert queued_chunk == b": admission-wait\n\n", queued_chunk
    # And it is admitted once the slot frees, rather than timing out or hanging.
    assert b'"text"' in admitted, admitted
