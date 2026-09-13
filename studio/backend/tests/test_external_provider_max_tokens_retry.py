# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An upstream that rejects `max_tokens` gets one retry with `max_completion_tokens`.

Azure-hosted gpt-5.x / o-series behind OpenAI-compatible proxies answer 400
``unsupported_parameter`` naming `max_tokens` and ask for `max_completion_tokens`
(#10787). Custom providers route to whatever upstream they front, so the rename is
detected from the error body, not guessed from the model name: the first response is
400, the retry resends with the renamed field, and the client sees only the successful
stream.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import (
    ExternalProviderClient,
    _rename_max_tokens_body,
    _unsupported_max_tokens_error,
)

_ERROR_BODY = json.dumps(
    {
        "error": {
            "message": "Unsupported parameter: 'max_tokens' is not supported with this model. "
            "Use 'max_completion_tokens' instead.",
            "type": "invalid_request_error",
            "param": "max_tokens",
            "code": "unsupported_parameter",
        }
    }
).encode()

# A 400 the retry must NOT react to: same shape, different parameter.
_OTHER_ERROR_BODY = json.dumps(
    {
        "error": {
            "message": "Unsupported parameter: 'temperature'.",
            "type": "invalid_request_error",
            "param": "temperature",
            "code": "unsupported_parameter",
        }
    }
).encode()

_SSE = (
    'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
).encode()

_JSON_COMPLETION = json.dumps(
    {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ]
    }
).encode()


class _RejectingHandler(BaseHTTPRequestHandler):
    """`reject_body` on the first POST, a normal response afterwards; streaming requests
    get SSE, non-streaming ones get a JSON completion."""

    protocol_version = "HTTP/1.1"
    reject_body = _ERROR_BODY

    def log_message(self, *args, **kwargs) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode())
        self.server.recorded.append(body)  # type: ignore[attr-defined]
        if len(self.server.recorded) <= 1:  # type: ignore[attr-defined]
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.reject_body)))
            self.end_headers()
            self.wfile.write(self.reject_body)
            return
        payload = _SSE if body.get("stream") else _JSON_COMPLETION
        self.send_response(200)
        self.send_header(
            "Content-Type", "text/event-stream" if body.get("stream") else "application/json"
        )
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _AlwaysRejectingHandler(_RejectingHandler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        self.server.recorded.append(json.loads(raw.decode()))  # type: ignore[attr-defined]
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(self.reject_body)))
        self.end_headers()
        self.wfile.write(self.reject_body)


class _OtherErrorHandler(_AlwaysRejectingHandler):
    reject_body = _OTHER_ERROR_BODY


class _Server:
    def __init__(self, handler) -> None:
        self._handler = handler

    def __enter__(self) -> "_Server":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._httpd.recorded = []  # type: ignore[attr-defined]
        self._thread = threading.Thread(target = self._httpd.serve_forever, daemon = True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout = 10)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    @property
    def bodies(self) -> list[dict]:
        return self._httpd.recorded  # type: ignore[attr-defined]


def _run(coro) -> None:
    """One loop per call, with a matching client, mirroring the sibling test files."""
    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    client = httpx.AsyncClient()

    async def wrapper() -> None:
        try:
            await coro()
        finally:
            await client.aclose()

    ep_mod._http_client = client
    try:
        loop.run_until_complete(wrapper())
    finally:
        ep_mod._http_client = previous
        loop.close()


def test_a_rejected_max_tokens_is_retried_as_max_completion_tokens():
    with _Server(_RejectingHandler) as server:
        client = ExternalProviderClient(
            provider_type = "custom",
            base_url = server.base_url,
            api_key = "",
        )
        chunks: list[str] = []

        async def go() -> None:
            async for chunk in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                max_tokens = 512,
            ):
                chunks.append(chunk)

        _run(go)

        assert len(server.bodies) == 2, server.bodies
        assert "max_tokens" in server.bodies[0]
        assert "max_completion_tokens" not in server.bodies[0]
        assert server.bodies[1]["max_completion_tokens"] == 512
        assert "max_tokens" not in server.bodies[1]
        # The client saw the successful stream, not the 400.
        assert any('"content":"ok"' in chunk for chunk in chunks)
        assert not any(chunk.startswith('data: {"error"') or "400" in chunk for chunk in chunks)


def test_an_unrelated_400_is_not_retried():
    with _Server(_OtherErrorHandler) as server:
        client = ExternalProviderClient(
            provider_type = "custom",
            base_url = server.base_url,
            api_key = "",
        )
        chunks: list[str] = []

        async def go() -> None:
            async for chunk in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                max_tokens = 512,
            ):
                chunks.append(chunk)

        _run(go)

        # One request, and the error reached the client.
        assert len(server.bodies) == 1
        assert any("temperature" in chunk for chunk in chunks), chunks


def test_non_streaming_chat_completion_retries_too():
    with _Server(_RejectingHandler) as server:
        client = ExternalProviderClient(
            provider_type = "custom",
            base_url = server.base_url,
            api_key = "",
        )

        async def go() -> None:
            await client.chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "gpt-5.5",
                max_tokens = 256,
            )

        _run(go)

        assert len(server.bodies) == 2
        assert server.bodies[1]["max_completion_tokens"] == 256


def test_unsupported_max_tokens_error_detection():
    assert _unsupported_max_tokens_error(400, _ERROR_BODY.decode())
    # Substring fallback for a body that is not the OpenAI JSON envelope.
    assert _unsupported_max_tokens_error(
        400, "HTTP 400: unsupported_parameter: use max_completion_tokens instead"
    )
    # A different unsupported parameter, a different status, or a 200 all read as no.
    assert not _unsupported_max_tokens_error(
        400, json.dumps({"error": {"code": "unsupported_parameter", "param": "temperature"}})
    )
    assert not _unsupported_max_tokens_error(429, _ERROR_BODY.decode())
    assert not _unsupported_max_tokens_error(400, "unrelated failure")


def test_rename_max_tokens_body_moves_only_that_field():
    body = {"model": "m", "max_tokens": 7, "temperature": 0.5}
    renamed = _rename_max_tokens_body(body)
    assert renamed == {"model": "m", "max_completion_tokens": 7, "temperature": 0.5}
    # The original body is untouched: the first attempt must still send max_tokens.
    assert body == {"model": "m", "max_tokens": 7, "temperature": 0.5}
