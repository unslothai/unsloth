# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression for #10787: custom gateways that reject max_tokens."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient, _rejects_max_tokens

_AZURE_ERROR = json.dumps(
    {
        "error": {
            "message": (
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead."
            ),
            "type": "invalid_request_error",
            "param": "max_tokens",
            "code": "unsupported_parameter",
        }
    }
)


@pytest.mark.parametrize(
    ("status", "text", "expected"),
    [
        (400, _AZURE_ERROR, True),
        (400, "Unsupported parameter: 'max_tokens'. Use 'max_completion_tokens'.", True),
        (500, _AZURE_ERROR, False),
        (
            400,
            json.dumps({"error": {"message": "max_tokens too large", "param": "max_tokens"}}),
            False,
        ),
        (400, json.dumps({"error": {"message": "bad temperature", "param": "temperature"}}), False),
    ],
)
def test_rejects_max_tokens(status, text, expected):
    assert _rejects_max_tokens(status, text) is expected


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs) -> None:
        return

    def _send(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.recorded.append(body)  # type: ignore[attr-defined]
        if "max_tokens" in body:
            self._send(400, self.server.error.encode(), "application/json")  # type: ignore[attr-defined]
        elif body.get("stream"):
            sse = 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n' "data: [DONE]\n\n"
            self._send(200, sse.encode(), "text/event-stream")
        else:
            reply = {"choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]}
            self._send(200, json.dumps(reply).encode(), "application/json")


@pytest.fixture
def gateway(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.recorded = []  # type: ignore[attr-defined]
    httpd.error = _AZURE_ERROR  # type: ignore[attr-defined]
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout = 10)


def _run(coro_fn):
    async def wrapper():
        async with httpx.AsyncClient(trust_env = False) as http:
            ep_mod._http_client = http
            return await coro_fn()

    previous = ep_mod._http_client
    try:
        return asyncio.run(wrapper())
    finally:
        ep_mod._http_client = previous


def _client(httpd) -> ExternalProviderClient:
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    return ExternalProviderClient(provider_type = "custom", base_url = base_url, api_key = "")


def _stream(client) -> str:
    async def go():
        out = []
        async for line in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "GPT5-Mitarbeitende",
            max_tokens = 128,
        ):
            out.append(line)
        return "\n".join(out)

    return _run(go)


def test_stream_retries_with_max_completion_tokens(gateway):
    out = _stream(_client(gateway))
    assert [b.get("max_tokens") for b in gateway.recorded] == [128, None]
    assert gateway.recorded[1]["max_completion_tokens"] == 128
    assert '"content":"ok"' in out and "unsupported_parameter" not in out


def test_stream_other_400_is_not_retried(gateway):
    gateway.error = json.dumps({"error": {"message": "bad request", "param": "messages"}})
    out = _stream(_client(gateway))
    assert len(gateway.recorded) == 1
    assert "bad request" in out


def test_non_stream_retries_with_max_completion_tokens(gateway):
    client = _client(gateway)
    reply = _run(
        lambda: client.chat_completion(
            messages = [{"role": "user", "content": "hi"}], model = "gpt-5", max_tokens = 64
        )
    )
    assert reply["choices"][0]["message"]["content"] == "ok"
    assert [b.get("max_completion_tokens") for b in gateway.recorded] == [None, 64]
