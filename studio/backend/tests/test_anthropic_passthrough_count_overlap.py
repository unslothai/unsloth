# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Streamed /v1/messages client-tool passthrough: the message_start prompt count overlaps the
upstream request instead of delaying it, and the SSE bytes keep their order."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

import httpx
import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod
from routes.inference import _anthropic_passthrough_stream

_UPSTREAM_SSE = (
    f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\n"
    f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    "data: [DONE]\n\n"
)


class _Backend:
    base_url = "http://127.0.0.1:59999"
    context_length = 4096

    def __init__(self, count):
        self._count = count

    def count_chat_tokens(self, *args, **kwargs):
        return self._count(*args, **kwargs)


class _Request:
    async def is_disconnected(self):
        return False


def _install_upstream(
    monkeypatch,
    *,
    on_request = None,
    status = 200,
):
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if on_request is not None:
            on_request()
        if status != 200:
            return httpx.Response(status, content = b'{"error": {"message": "boom"}}')
        return httpx.Response(
            200,
            content = _UPSTREAM_SSE.encode(),
            headers = {"content-type": "text/event-stream"},
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _client(*_args, **kwargs):
        return real_client(transport = transport, timeout = kwargs.get("timeout", 600))

    monkeypatch.setattr(inf_mod.httpx, "AsyncClient", _client)
    return bodies


async def _stream_text(backend, **kwargs):
    response = await _anthropic_passthrough_stream(
        _Request(),
        threading.Event(),
        backend,
        [{"role": "user", "content": "hello"}],
        [],
        0.7,
        0.95,
        20,
        16,
        "msg_1",
        "test-model",
        **kwargs,
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


def _events(text):
    out = []
    for block in text.split("\n\n"):
        lines = block.strip().splitlines()
        if len(lines) == 2 and lines[0].startswith("event: "):
            out.append((lines[0][7:], json.loads(lines[1][6:])))
    return out


def test_count_runs_while_the_upstream_request_is_in_flight(monkeypatch):
    sent = threading.Event()
    seen = {}

    def count(*_args, **_kwargs):
        # Overlap: the upstream request arrives while the count is still running.
        seen["overlapped"] = sent.wait(timeout = 5)
        return 1234

    bodies = _install_upstream(monkeypatch, on_request = sent.set)

    text = asyncio.run(_stream_text(_Backend(count)))

    assert len(bodies) == 1
    assert seen["overlapped"] is True
    events = _events(text)
    assert events[0][0] == "message_start"
    assert events[0][1]["message"]["usage"]["input_tokens"] == 1234
    assert [name for name, _ in events][-1] == "message_stop"


def test_message_start_leads_and_the_stream_is_unchanged(monkeypatch):
    _install_upstream(monkeypatch)

    text = asyncio.run(_stream_text(_Backend(lambda *_a, **_k: 77)))

    names = [name for name, _ in _events(text)]
    assert names == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert _events(text)[0][1]["message"]["usage"]["input_tokens"] == 77
    # Nothing, not even a keepalive comment, precedes message_start.
    assert text.startswith("event: message_start\n")


def test_count_failure_reports_zero_and_still_streams(monkeypatch):
    def count(*_args, **_kwargs):
        raise RuntimeError("tokenizer unavailable")

    bodies = _install_upstream(monkeypatch)

    text = asyncio.run(_stream_text(_Backend(count)))

    assert len(bodies) == 1
    events = _events(text)
    assert events[0][0] == "message_start"
    assert events[0][1]["message"]["usage"]["input_tokens"] == 0
    assert "content_block_delta" in [name for name, _ in events]


def test_upstream_error_follows_message_start(monkeypatch):
    _install_upstream(monkeypatch, status = 500)

    text = asyncio.run(_stream_text(_Backend(lambda *_a, **_k: 5)))

    names = [name for name, _ in _events(text)]
    assert names[0] == "message_start"
    assert names[-1] == "error"


def test_pre_dispatch_failure_still_follows_message_start(monkeypatch):
    async def _broken_cut(*_args, **_kwargs):
        raise RuntimeError("cut replay failed")

    monkeypatch.setattr(inf_mod, "_prepare_passthrough_cut", _broken_cut)
    monkeypatch.setattr(
        inf_mod,
        "get_llama_cpp_backend",
        lambda: type("_B", (), {"_maybe_recover_from_mtp_crash": lambda self, _e: None})(),
    )
    bodies = _install_upstream(monkeypatch)

    text = asyncio.run(_stream_text(_Backend(lambda *_a, **_k: 9)))

    assert bodies == []
    events = _events(text)
    assert [name for name, _ in events] == ["message_start", "error"]
    assert events[0][1]["message"]["usage"]["input_tokens"] == 9


@pytest.mark.parametrize("announce", [False, True])
def test_stream_response_announces_dispatch_only_on_request(announce):
    class _Upstream(inf_mod._LlamaUpstream):
        def __init__(self):
            self._respawn_spent = False

        def dispatch_stream(self, *_args, **_kwargs):
            async def _send():
                return "response"

            return asyncio.ensure_future(_send())

    async def _collect():
        items = []
        async for item in _Upstream().stream_response(
            None, {}, None, None, announce_dispatch = announce
        ):
            items.append(item)
        return items

    items = asyncio.run(_collect())

    if announce:
        assert items == [inf_mod._LLAMA_STREAM_DISPATCHED, "response"]
    else:
        assert items == ["response"]
