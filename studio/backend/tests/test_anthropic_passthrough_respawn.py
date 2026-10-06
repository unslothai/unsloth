# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Restart survival for the Anthropic /v1/messages passthrough.

A crashed llama-server relaunches on a NEW ephemeral port. Before the retry the
passthrough kept posting to the dead port, so a Claude Code session stayed broken
until the next explicit load. These cover the respawn-and-retry on both the
streaming and non-streaming passthroughs.
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

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod
from routes.inference import (
    _anthropic_passthrough_non_streaming,
    _passthrough_retry_url,
    _anthropic_passthrough_stream,
)

_DEAD = "http://127.0.0.1:57953"
_FRESH = "http://127.0.0.1:62933"


class _Backend:
    """Stub llama backend whose base_url moves to a new port once respawned."""

    def __init__(
        self,
        *,
        respawn_ok = True,
        mtp_handled = False,
    ):
        self.base_url = _DEAD
        self.context_length = 4096
        self.respawn_calls = 0
        self.mtp_calls = 0
        self._respawn_ok = respawn_ok
        self._mtp_handled = mtp_handled

    def count_chat_tokens(self, *_args, **_kwargs):
        return 2

    def _maybe_recover_from_mtp_crash(self, _exc):
        self.mtp_calls += 1
        return self._mtp_handled

    def _respawn_if_dead(self):
        self.respawn_calls += 1
        if not self._respawn_ok:
            return False
        self.base_url = _FRESH
        return True


class _Request:
    async def is_disconnected(self):
        return False


class _FakeNonStreamingClient:
    def __init__(self):
        self.urls = []
        self.closed = False

    async def aclose(self):
        self.closed = True

    async def post(self, url, **_kwargs):
        self.urls.append(url)
        if url.startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        return httpx.Response(
            200,
            json = {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )


def _install_stream_transport(monkeypatch, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if str(request.url).startswith(_DEAD):
            raise httpx.ConnectError("connection refused")
        content = (
            f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\ndata: [DONE]\n\n"
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


async def _run_stream(backend):
    response = await _anthropic_passthrough_stream(
        _Request(),
        threading.Event(),
        backend,
        [{"role": "user", "content": "hi"}],
        [],
        0.7,
        0.95,
        20,
        16,
        "msg_1",
        "test-model",
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


async def _run_non_streaming(backend, **kwargs):
    return await _anthropic_passthrough_non_streaming(
        backend,
        [{"role": "user", "content": "hi"}],
        [],
        0.7,
        0.95,
        20,
        16,
        "msg_1",
        "test-model",
        **kwargs,
    )


# ── Helper ────────────────────────────────────────────────────


def test_retry_url_rebuilds_from_the_respawned_base_url():
    backend = _Backend()

    url = asyncio.run(_passthrough_retry_url(backend, httpx.ConnectError("x")))

    assert url == f"{_FRESH}/v1/chat/completions"
    assert backend.respawn_calls == 1


def test_retry_url_is_none_when_nothing_respawned():
    backend = _Backend(respawn_ok = False)

    url = asyncio.run(_passthrough_retry_url(backend, httpx.ConnectError("x")))

    assert url is None


def test_retry_url_defers_to_the_mtp_crash_recovery():
    # An MTP+tensor crash schedules its own reload; retrying would race it.
    backend = _Backend(mtp_handled = True)

    url = asyncio.run(_passthrough_retry_url(backend, httpx.ConnectError("x")))

    assert url is None
    assert backend.respawn_calls == 0


def test_retry_url_tolerates_a_backend_without_respawn_hooks():
    backend = SimpleNamespace(base_url = _DEAD)

    url = asyncio.run(_passthrough_retry_url(backend, httpx.ConnectError("x")))

    assert url is None


# ── Non-streaming ─────────────────────────────────────────────


def test_non_streaming_retries_against_the_new_port(monkeypatch):
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend()

    response = asyncio.run(_run_non_streaming(backend))

    assert response.status_code == 200
    assert backend.respawn_calls == 1
    assert client.urls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]


def test_non_streaming_raises_when_the_server_stays_dead(monkeypatch):
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(respawn_ok = False)

    with pytest.raises(httpx.ConnectError):
        asyncio.run(_run_non_streaming(backend))

    assert client.urls == [f"{_DEAD}/v1/chat/completions"]  # no blind retry


def test_non_streaming_does_not_retry_an_mtp_crash(monkeypatch):
    client = _FakeNonStreamingClient()
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
    backend = _Backend(mtp_handled = True)

    with pytest.raises(httpx.ConnectError):
        asyncio.run(_run_non_streaming(backend))

    assert backend.respawn_calls == 0


def test_non_streaming_cancel_after_respawn_retry_is_not_a_transport_error(monkeypatch):
    class CancelAfterRespawnClient:
        def __init__(self):
            self.urls = []
            self.retry_started = asyncio.Event()
            self.closed = asyncio.Event()

        async def post(self, url, **_kwargs):
            self.urls.append(url)
            if len(self.urls) == 1:
                raise httpx.ConnectError("connection refused")
            self.retry_started.set()
            await self.closed.wait()
            raise httpx.ReadError("client closed")

        async def aclose(self):
            self.closed.set()

    async def _run():
        client = CancelAfterRespawnClient()
        monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)
        backend = _Backend()
        cancel_event = threading.Event()
        task = asyncio.create_task(
            _run_non_streaming(
                backend,
                request = _Request(),
                cancel_event = cancel_event,
            )
        )
        await asyncio.wait_for(client.retry_started.wait(), 0.2)
        cancel_event.set()

        with pytest.raises(inf_mod._NonStreamingRequestCancelled):
            await asyncio.wait_for(task, 0.5)

        assert client.urls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]
        assert client.closed.is_set()

    asyncio.run(_run())


def test_non_streaming_cancel_wins_response_race(monkeypatch):
    async def _run():
        cancel_event = threading.Event()

        class RacingClient(_FakeNonStreamingClient):
            async def post(self, url, **_kwargs):
                self.urls.append(url)
                cancel_event.set()
                return httpx.Response(
                    200,
                    json = {
                        "choices": [{"message": {"content": "too late"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    },
                )

        client = RacingClient()
        monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)

        with pytest.raises(inf_mod._NonStreamingRequestCancelled):
            await _run_non_streaming(
                _Backend(),
                request = _Request(),
                cancel_event = cancel_event,
            )

        assert client.closed

    asyncio.run(_run())


# ── Streaming ─────────────────────────────────────────────────


def test_streaming_retries_against_the_new_port(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend()

    blob = asyncio.run(_run_stream(backend))

    assert backend.respawn_calls == 1
    assert calls == [f"{_DEAD}/v1/chat/completions", f"{_FRESH}/v1/chat/completions"]
    # The retried stream really produced the turn, not just a clean-looking stop.
    assert "event: message_start" in blob
    assert "event: message_stop" in blob
    assert "hi" in blob


def test_streaming_emits_an_error_event_when_the_server_stays_dead(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(respawn_ok = False)

    blob = asyncio.run(_run_stream(backend))

    assert calls == [f"{_DEAD}/v1/chat/completions"]  # no blind retry
    assert "event: error" in blob


def test_streaming_does_not_retry_an_mtp_crash(monkeypatch):
    calls = []
    _install_stream_transport(monkeypatch, calls)
    backend = _Backend(mtp_handled = True)

    blob = asyncio.run(_run_stream(backend))

    assert backend.respawn_calls == 0
    assert calls == [f"{_DEAD}/v1/chat/completions"]
    assert "event: error" in blob
