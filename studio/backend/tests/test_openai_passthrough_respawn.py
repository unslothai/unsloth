# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Restart survival for the OpenAI /v1/chat/completions passthrough.

A crashed llama-server relaunches on a NEW ephemeral port. /v1/messages already
respawns and retries; this surface did not, so a harness on the OpenAI API kept
posting to the dead port and stayed broken until the user reloaded the model by
hand, while an Anthropic-API client on the same backend recovered itself.

Twin of test_anthropic_passthrough_respawn.py, same stubs and same cases.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod
from models.inference import ChatCompletionRequest, ChatMessage
from routes.inference import (
    _openai_passthrough_non_streaming_upstream,
    _openai_passthrough_stream_admitted,
    _passthrough_retry_url,
)

_DEAD = "http://127.0.0.1:57953"
_FRESH = "http://127.0.0.1:62933"


class _Backend:
    """Stub llama backend whose base_url moves to a new port once respawned."""

    def __init__(self, *, respawn_ok = True, mtp_handled = False, stays_dead = False):
        self.base_url = _DEAD
        self.context_length = 4096
        self.respawn_calls = 0
        self.mtp_calls = 0
        self._respawn_ok = respawn_ok
        self._mtp_handled = mtp_handled
        # Models a relaunch that reports success but is not actually serving.
        self._stays_dead = stays_dead

    def count_chat_tokens(self, *_args, **_kwargs):
        return 2

    def _request_reasoning_kwargs(self, *_args, **_kwargs):
        return None

    def _maybe_recover_from_mtp_crash(self, _exc):
        self.mtp_calls += 1
        return self._mtp_handled

    def _respawn_if_dead(self):
        self.respawn_calls += 1
        if not self._respawn_ok:
            return False
        if not self._stays_dead:
            self.base_url = _FRESH
        return True


class _Request:
    async def is_disconnected(self):
        return False


class _Lease:
    """Records that the slot came back. Repeat calls are fine: the real lease is
    idempotent, and the stream's nested handlers both release."""

    def __init__(self):
        self.released = False

    def release(self):
        self.released = True


class _Tracker:
    def __exit__(self, *_exc):
        return False


class _FakeNonStreamingClient:
    def __init__(self):
        self.urls = []

    async def aclose(self):
        pass

    async def post(self, url, **_kwargs):
        self.urls.append(url)
        if url.startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        return httpx.Response(
            200,
            json = {
                "id": "chatcmpl-1",
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )


def _install_stream_transport(monkeypatch, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url).startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        content = (
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content = content.encode(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _client(*_args, **kwargs):
        return real_client(transport = transport, timeout = kwargs.get("timeout", 600))

    monkeypatch.setattr(inf_mod.httpx, "AsyncClient", _client)


def _payload():
    return ChatCompletionRequest(
        model = "default",
        messages = [ChatMessage(role = "user", content = "hi")],
    )


async def _run_non_streaming(backend):
    return await _openai_passthrough_non_streaming_upstream(
        backend,
        _payload(),
        "test-model",
        request = _Request(),
        cancel_event = threading.Event(),
    )


async def _run_stream(backend, lease = None):
    response = await _openai_passthrough_stream_admitted(
        _Request(),
        threading.Event(),
        backend,
        _payload(),
        "test-model",
        "chatcmpl-local",
        admission_lease = lease or _Lease(),
        tracker = _Tracker(),
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


# ── Helper ────────────────────────────────────────────────────


def test_the_retry_url_is_shared_with_the_anthropic_surface():
    """Both passthroughs post to the same upstream route, so one helper serves both.
    A rename that leaves this surface behind is the bug being fixed."""
    backend = _Backend()

    url = asyncio.run(_passthrough_retry_url(backend, httpx.ConnectError("x")))

    assert url == f"{_FRESH}/v1/chat/completions"
    assert backend.respawn_calls == 1


# ── Non-streaming ─────────────────────────────────────────────


def test_non_streaming_retries_against_the_new_port(monkeypatch):
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend()

    response = asyncio.run(_run_non_streaming(backend))

    assert response.status_code == 200
    assert backend.respawn_calls == 1
    assert client.urls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]


def test_non_streaming_still_502s_when_the_server_stays_dead(monkeypatch):
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(respawn_ok = False)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run_non_streaming(backend))

    assert exc.value.status_code == 502
    assert client.urls == [f"{_DEAD}/v1/chat/completions"]  # no blind retry


def test_non_streaming_does_not_retry_an_mtp_crash(monkeypatch):
    # An MTP+tensor crash schedules its own reload; retrying would race it.
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(mtp_handled = True)

    with pytest.raises(HTTPException):
        asyncio.run(_run_non_streaming(backend))

    assert backend.respawn_calls == 0


def test_non_streaming_respawns_at_most_once(monkeypatch):
    """A relaunch that reports success but is not serving must end in a 502, not a
    loop that respawns the model on every attempt."""
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(stays_dead = True)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run_non_streaming(backend))

    assert exc.value.status_code == 502
    assert backend.respawn_calls == 1
    assert client.urls == [f"{_DEAD}/v1/chat/completions"] * 2


# ── Streaming ─────────────────────────────────────────────────


def test_streaming_retries_against_the_new_port(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend()

    blob = asyncio.run(_run_stream(backend))

    assert backend.respawn_calls == 1
    assert calls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]
    # The retried stream really produced the turn, not just a clean-looking stop.
    assert "hi" in blob
    assert "[DONE]" in blob


def test_streaming_still_502s_when_the_server_stays_dead(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(respawn_ok = False)
    lease = _Lease()

    with pytest.raises(HTTPException) as exc:
        asyncio.run(_run_stream(backend, lease))

    assert exc.value.status_code == 502
    assert calls == [f"{_DEAD}/v1/chat/completions"]  # no blind retry
    assert lease.released, "the failed dispatch kept its admission slot"


def test_streaming_does_not_retry_an_mtp_crash(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(mtp_handled = True)

    with pytest.raises(HTTPException):
        asyncio.run(_run_stream(backend))

    assert backend.respawn_calls == 0
    assert calls == [f"{_DEAD}/v1/chat/completions"]


def test_streaming_respawns_at_most_once(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(stays_dead = True)
    lease = _Lease()

    with pytest.raises(HTTPException):
        asyncio.run(_run_stream(backend, lease))

    assert backend.respawn_calls == 1
    assert calls == [f"{_DEAD}/v1/chat/completions"] * 2
    assert lease.released, "the slot was leaked across the respawn retry"


def test_a_backend_without_respawn_hooks_is_untouched(monkeypatch):
    """Remote and external backends have no llama-server to relaunch."""
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = SimpleNamespace(
        base_url = _DEAD,
        context_length = 4096,
        count_chat_tokens = lambda *_a, **_k: 2,
        _request_reasoning_kwargs = lambda *_a, **_k: None,
    )

    with pytest.raises(HTTPException):
        asyncio.run(_run_non_streaming(backend))

    assert client.urls == [f"{_DEAD}/v1/chat/completions"]
