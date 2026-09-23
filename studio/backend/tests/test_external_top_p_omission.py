# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from models.inference import ChatCompletionRequest


_SSE = (
    'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
).encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode() if length else "{}")
        self.server.recorded.append(body)  # type: ignore[attr-defined]
        if "temperature" in body and "top_p" in body:
            payload = json.dumps(
                {
                    "error": {
                        "message": "`temperature` and `top_p` cannot both be specified for this model."
                    }
                }
            ).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
        elif body.get("stream"):
            payload = _SSE
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        else:
            payload = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _Gateway:
    def __enter__(self) -> "_Gateway":
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.recorded = []  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=10)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    @property
    def body(self) -> dict:
        assert self._httpd.recorded, "nothing reached the gateway"  # type: ignore[attr-defined]
        return self._httpd.recorded[-1]  # type: ignore[attr-defined]


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


def _through_route(**fields) -> tuple[dict, str]:
    import routes.inference as ri
    from starlette.requests import Request

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    with _Gateway() as gateway:
        payload = ChatCompletionRequest(
            provider_type="custom",
            provider_base_url=gateway.base_url,
            provider_api_key="k",
            messages=[{"role": "user", "content": "hi"}],
            model="claude-sonnet-4-6",
            stream=True,
            **fields,
        )
        request = Request(
            {
                "type": "http",
                "http_version": "1.1",
                "method": "POST",
                "path": "/v1/chat/completions",
                "raw_path": b"/v1/chat/completions",
                "root_path": "",
                "scheme": "http",
                "query_string": b"",
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 12345),
                "server": ("127.0.0.1", 8000),
            },
            receive,
        )
        chunks: list[str] = []

        async def go() -> None:
            response = await ri._proxy_to_external_provider(payload, request)
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

        _run(go)
        return gateway.body, "".join(chunks)


def test_an_omitted_top_p_is_not_forwarded_and_the_gateway_accepts():
    body, stream = _through_route(temperature=0.7)
    assert "top_p" not in body
    assert body["temperature"] == 0.7
    assert "cannot both be specified" not in stream
    assert '"content":"ok"' in stream


@pytest.mark.parametrize("top_p", [0.0, 0.9, 1.0])
def test_an_explicit_top_p_still_reaches_the_gateway(top_p):
    body, stream = _through_route(temperature=0.7, top_p=top_p)
    assert body["top_p"] == top_p
    assert "cannot both be specified" in stream


def test_the_client_omits_top_p_when_given_none():
    with _Gateway() as gateway:
        client = ExternalProviderClient(
            provider_type="custom", base_url=gateway.base_url, api_key="k"
        )

        async def go() -> None:
            async for _ in client.stream_chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                model="claude-sonnet-4-6",
                temperature=0.7,
                top_p=None,
            ):
                pass

        _run(go)
        assert "top_p" not in gateway.body


def test_the_non_streaming_helper_omits_top_p_when_given_none():
    with _Gateway() as gateway:
        client = ExternalProviderClient(
            provider_type="custom", base_url=gateway.base_url, api_key="k"
        )

        async def go() -> None:
            response = await client.chat_completion(
                messages=[{"role": "user", "content": "ping"}],
                model="claude-sonnet-4-6",
                temperature=0.0,
                top_p=None,
                max_tokens=1,
            )
            assert response["choices"][0]["message"]["content"] == "ok"

        _run(go)
        assert "top_p" not in gateway.body


def test_the_connection_test_ping_omits_top_p():
    from routes.providers import _test_custom_provider_connectivity

    with _Gateway() as gateway:
        real = ExternalProviderClient(
            provider_type="custom", base_url=gateway.base_url, api_key="k"
        )

        class _NoModelsOrSpeech:
            async def list_models(self):
                raise RuntimeError("no /models")

            async def create_speech(self, **kwargs):
                raise RuntimeError("no /audio/speech")

            async def chat_completion(self, **kwargs):
                return await real.chat_completion(**kwargs)

        results = []

        async def go() -> None:
            results.append(
                await _test_custom_provider_connectivity(_NoModelsOrSpeech(), "claude-sonnet-4-6")
            )

        _run(go)
        assert "top_p" not in gateway.body
        assert results[0].success, results[0].message
