# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A non-streaming raw passthrough waits out a server park longer than its read timeout.

llama-server writes no notice on a non-streaming request, so a park longer than the first
token timeout ended in a 502 for an answer that was coming. The wait now extends only while
the server reports a park, and never past `_RAW_PARK_STALL_CAP_S`.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from fastapi import HTTPException

import routes.inference as inf_mod
from models.inference import ChatCompletionRequest, ChatMessage

_BODY = {
    "id": "chatcmpl-park",
    "object": "chat.completion",
    "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
}


class _Backend:
    def __init__(self, port: int, *, server: bool, parked: bool):
        self.base_url = f"http://127.0.0.1:{port}"
        self.model_identifier = "test-gguf"
        self.context_length = 4096
        self.server_may_park = server
        self._parked = parked
        self.probes = 0

    def _server_park_grace_any(self) -> bool:
        # A non-streaming body carries no notices, so it is always the aggregate reading.
        self.probes += 1
        return self._parked

    def count_chat_tokens(self, *_args, **_kwargs):
        return 2

    def _request_reasoning_kwargs(self, *_args, **_kwargs):
        return None

    def _maybe_recover_from_mtp_crash(self, _exc):
        return False

    def _respawn_if_dead(self):
        return False


class _Request:
    async def is_disconnected(self):
        return False


async def _upstream(park_s: float):
    """A fake llama-server that sits silent for `park_s`, then answers."""
    seen = {"closed_early": False}

    async def handle(reader, writer):
        data = b""
        while b"\r\n\r\n" not in data:
            data += await reader.read(65536)
        try:
            await asyncio.sleep(park_s)
        except asyncio.CancelledError:
            raise
        payload = json.dumps(_BODY).encode()
        try:
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        except Exception:
            seen["closed_early"] = True
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _run(
    *,
    server: bool,
    parked: bool,
    park_s: float = 3.0,
):
    async def go():
        upstream, port = await _upstream(park_s)
        backend = _Backend(port, server = server, parked = parked)
        started = time.monotonic()
        try:
            result = await inf_mod._openai_passthrough_non_streaming_upstream(
                backend,
                ChatCompletionRequest(
                    model = "default", messages = [ChatMessage(role = "user", content = "hi")]
                ),
                "test-model",
                request = _Request(),
                cancel_event = threading.Event(),
            )
        except HTTPException as exc:
            result = exc
        elapsed = time.monotonic() - started
        upstream.close()
        return result, backend, elapsed

    return asyncio.run(go())


@pytest.fixture(autouse = True)
def _one_second_timeout(monkeypatch):
    monkeypatch.setenv(inf_mod._OPENAI_COMPAT_FIRST_TOKEN_TIMEOUT_ENV, "1")


def test_a_park_longer_than_the_timeout_is_waited_out():
    result, backend, elapsed = _run(server = True, parked = True)
    assert not isinstance(result, HTTPException), result.detail
    assert result.status_code == 200
    assert json.loads(result.body)["id"] == "chatcmpl-park"
    assert backend.probes >= 2
    assert elapsed >= 2.5


# A park far longer than any bound below, so finishing early proves it was not waited out. The
# bound allows for the in-flight POST's teardown, which can take seconds on a Windows runner.
_LONG_PARK_S = 20.0
_TEARDOWN_ALLOWANCE_S = inf_mod._TEARDOWN_TASK_STOP_TIMEOUT_S + 1.0


def test_without_a_park_the_timeout_still_holds():
    result, backend, elapsed = _run(server = True, parked = False, park_s = _LONG_PARK_S)
    assert isinstance(result, HTTPException) and result.status_code == 502
    assert backend.probes == 1
    assert elapsed < 1.0 + _TEARDOWN_ALLOWANCE_S


def test_the_wait_is_bounded_by_the_park_cap(monkeypatch):
    monkeypatch.setattr(inf_mod, "_RAW_PARK_STALL_CAP_S", 1.0)
    result, _backend, elapsed = _run(server = True, parked = True, park_s = _LONG_PARK_S)
    assert isinstance(result, HTTPException) and result.status_code == 502
    assert 2.0 <= elapsed < 2.0 + _TEARDOWN_ALLOWANCE_S


def test_a_server_that_does_not_park_is_unchanged():
    result, backend, elapsed = _run(server = False, parked = True)
    assert isinstance(result, HTTPException) and result.status_code == 502
    assert backend.probes == 0
    assert elapsed < 2.5
