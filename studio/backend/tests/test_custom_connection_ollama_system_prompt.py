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
from models.inference import ChatCompletionRequest

_DATE_LINE = "The current date is 2026-08-15."
_SSE = (
    'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
).encode()


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

    def do_GET(self) -> None:
        if self.path == "/api/version" and self.server.is_ollama:
            self._send(200, b'{"version":"0.40.17"}', "application/json")
        else:
            self._send(404, b"404 page not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.server.recorded.append(json.loads(self.rfile.read(length)))
        self._send(200, _SSE, "text/event-stream")


def _messages_sent(is_ollama: bool, messages: list[dict]) -> list[dict]:
    import routes.inference as ri
    from starlette.requests import Request

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.is_ollama = is_ollama
    httpd.recorded = []
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    payload = ChatCompletionRequest(
        provider_type = "custom",
        provider_base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1",
        provider_api_key = "k",
        messages = messages,
        model = "pirate:latest",
        stream = True,
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

    async def go() -> None:
        client = httpx.AsyncClient()
        previous = ep_mod._http_client
        ep_mod._http_client = client
        try:
            response = await ri._proxy_to_external_provider(payload, request)
            async for _ in response.body_iterator:
                pass
        finally:
            ep_mod._http_client = previous
            await client.aclose()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(go())
    finally:
        loop.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout = 10)
    assert httpd.recorded, "nothing reached the server"
    return httpd.recorded[-1]["messages"]


@pytest.fixture(autouse = True)
def _date_on(monkeypatch):
    import routes.inference as ri
    monkeypatch.setattr(ri, "current_date_prompt_line", lambda **_kwargs: _DATE_LINE)
    monkeypatch.setattr(ep_mod, "_served_ollama", {}, raising = False)


def test_a_custom_connection_to_ollama_keeps_the_modelfile_system_prompt():
    user = [{"role": "user", "content": "What is the capital of France?"}]
    assert _messages_sent(True, user) == user


def test_a_custom_connection_to_ollama_still_dates_a_studio_system_prompt():
    sent = _messages_sent(
        True,
        [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}],
    )
    assert sent[0] == {"role": "system", "content": f"{_DATE_LINE}\n\nBe terse."}


def test_a_custom_connection_to_another_server_still_gets_the_date():
    sent = _messages_sent(False, [{"role": "user", "content": "hi"}])
    assert sent == [{"role": "system", "content": _DATE_LINE}, {"role": "user", "content": "hi"}]
