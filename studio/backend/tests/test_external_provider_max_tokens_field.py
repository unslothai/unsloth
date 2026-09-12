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
from core.inference.external_provider import (
    ExternalProviderClient,
    _chat_completions_max_tokens_field,
    _rewrite_body_max_tokens_to_completion,
    _should_retry_with_max_completion_tokens,
)


@pytest.mark.parametrize(
    ("provider_type", "base_url", "expected"),
    [
        ("openai", "https://api.openai.com/v1", "max_completion_tokens"),
        ("custom", "https://my-resource.openai.azure.com/openai/v1", "max_completion_tokens"),
        ("custom", "https://chat.kiconnect.nrw/api/v1", "max_tokens"),
        ("custom", "https://my-vllm-server.com/v1", "max_tokens"),
        ("custom", "http://127.0.0.1:8080/v1", "max_tokens"),
        ("vllm", "http://127.0.0.1:8000/v1", "max_tokens"),
    ],
)
def test_chat_completions_max_tokens_field(provider_type, base_url, expected):
    assert _chat_completions_max_tokens_field(provider_type, base_url) == expected


def test_should_retry_with_max_completion_tokens_matches_openai_error_shape():
    body = {"model": "gpt-5", "messages": [], "max_tokens": 128}
    err = {
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
    assert _should_retry_with_max_completion_tokens(400, json.dumps(err), body)


def test_rewrite_body_max_tokens_to_completion():
    assert _rewrite_body_max_tokens_to_completion({"max_tokens": 64, "model": "m"}) == {
        "model": "m",
        "max_completion_tokens": 64,
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    attempts: list[dict] = []

    def log_message(self, *args, **kwargs) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw.decode())
        self.server.recorded.append(body)  # type: ignore[attr-defined]

        if "max_tokens" in body and "max_completion_tokens" not in body:
            err = json.dumps(
                {
                    "error": {
                        "message": "Use 'max_completion_tokens' instead.",
                        "param": "max_tokens",
                        "code": "unsupported_parameter",
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        sse = (
            'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


def _run(coro) -> None:
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


def test_custom_gateway_retries_with_max_completion_tokens_on_the_wire():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.recorded = []  # type: ignore[attr-defined]
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    try:
        client = ExternalProviderClient(provider_type = "custom", base_url = base_url, api_key = "")

        async def go() -> None:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "GPT5",
                max_tokens = 128,
            ):
                pass

        _run(go)
        assert len(httpd.recorded) == 2
        assert httpd.recorded[0].get("max_tokens") == 128
        assert httpd.recorded[1].get("max_completion_tokens") == 128
        assert "max_tokens" not in httpd.recorded[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout = 10)
