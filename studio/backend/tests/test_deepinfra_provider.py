# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from core.inference import external_provider as ep_mod
from core.inference.providers import list_available_providers
from models.inference import ChatCompletionRequest


_SSE = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}\n\n'
    "data: [DONE]\n\n"
).encode()
_MODELS = b'{"object":"list","data":[{"id":"meta-llama/Llama-3.3-70B-Instruct","object":"model"},{"id":"Qwen/Qwen2.5-72B-Instruct","object":"model"}]}'
_NOT_FOUND = b'{"error":{"code":"404","message":"Not Found"}}'


class _DeepInfraLikeServer(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list = []

    def log_message(self, *args, **kwargs) -> None:
        return

    def _reply(self, status: int, content_type: str, raw: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        self.seen.append((self.path, None))
        if self.path.endswith("/models"):
            self._reply(200, "application/json", _MODELS)
        else:
            self._reply(404, "application/json", _NOT_FOUND)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.seen.append((self.path, body))
        if self.path.endswith("/chat/completions"):
            self._reply(200, "text/event-stream", _SSE)
        else:
            self._reply(404, "application/json", _NOT_FOUND)


def _against_the_server(make_call):
    _DeepInfraLikeServer.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _DeepInfraLikeServer)
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    ep_mod._http_client = httpx.AsyncClient()
    try:
        result = loop.run_until_complete(
            make_call(f"http://127.0.0.1:{httpd.server_address[1]}/v1/openai")
        )
        loop.run_until_complete(ep_mod._http_client.aclose())
    finally:
        ep_mod._http_client = previous
        loop.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout = 10)
    return result, _DeepInfraLikeServer.seen


async def _chat_through_the_route(base_url: str, service_tier: str | None) -> str:
    import routes.inference as ri
    from starlette.requests import Request

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    payload = ChatCompletionRequest(
        provider_type = "deepinfra",
        provider_base_url = base_url,
        messages = [{"role": "user", "content": "hi"}],
        model = "meta-llama/Llama-3.3-70B-Instruct",
        stream = True,
        service_tier = service_tier,
        temperature = 0.6,
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
    response = await ri._proxy_to_external_provider(payload, request)
    streamed = []
    async for chunk in response.body_iterator:
        streamed.append(chunk if isinstance(chunk, str) else chunk.decode())
    return "".join(streamed)


async def _check_and_list_models(base_url: str):
    import routes.providers as providers_route
    from models.providers import ProviderModelsRequest, ProviderTestRequest

    check = await providers_route.test_provider(
        ProviderTestRequest(provider_type = "deepinfra", base_url = base_url),
        _current_subject = "unsloth",
        via_api_key = False,
    )
    models = await providers_route.list_provider_models(
        ProviderModelsRequest(provider_type = "deepinfra", base_url = base_url),
        _current_subject = "unsloth",
        via_api_key = False,
    )
    return check, [m.id for m in models]


def test_deepinfra_is_offered_on_the_connections_api():
    deepinfra = next(
        e for e in list_available_providers() if e["provider_type"] == "deepinfra"
    )
    assert deepinfra["base_url"] == "https://api.deepinfra.com/v1/openai"
    assert not deepinfra["hidden"]
    assert deepinfra["model_list_mode"] == "remote"
    assert deepinfra["supports_service_tier"] is True
    assert deepinfra["service_tier_options"] == ["default", "priority", "flex"]


def test_deepinfra_forwards_service_tier_flex_on_the_wire():
    streamed, seen = _against_the_server(
        lambda base_url: _chat_through_the_route(base_url, "flex")
    )
    [(path, body)] = seen
    assert path == "/v1/openai/chat/completions"
    assert body["model"] == "meta-llama/Llama-3.3-70B-Instruct"
    assert body["service_tier"] == "flex"
    assert "hello" in streamed


def test_deepinfra_forwards_service_tier_priority_on_the_wire():
    _, seen = _against_the_server(
        lambda base_url: _chat_through_the_route(base_url, "priority")
    )
    [(_, body)] = seen
    assert body["service_tier"] == "priority"


def test_deepinfra_default_tier_is_omitted_from_the_wire():
    _, seen = _against_the_server(
        lambda base_url: _chat_through_the_route(base_url, "default")
    )
    [(_, body)] = seen
    assert "service_tier" not in body


def test_service_tier_is_not_forwarded_to_other_providers():
    from core.inference.providers import get_provider_info

    for provider_type in ("openai", "anthropic", "custom", "qwen", "zai"):
        info = get_provider_info(provider_type)
        assert not info.get("supports_service_tier"), provider_type


def test_deepinfra_normalizes_legacy_tier_spellings():
    # "standard_tier" / "standard" / "standard_only" normalize to "default".
    for spelling in ("standard_tier", "standard", "standard_only", "Standard"):
        req = ChatCompletionRequest(
            provider_type = "deepinfra",
            messages = [{"role": "user", "content": "hi"}],
            model = "meta-llama/Llama-3.3-70B-Instruct",
            stream = True,
            service_tier = spelling,
        )
        assert req.service_tier == "default", spelling
    # An unknown tier is rejected.
    import pytest

    with pytest.raises(Exception):
        ChatCompletionRequest(
            provider_type = "deepinfra",
            messages = [{"role": "user", "content": "hi"}],
            model = "meta-llama/Llama-3.3-70B-Instruct",
            stream = True,
            service_tier = "bogus",
        )


def test_deepinfra_check_and_model_list_read_the_live_models_route():
    (check, model_ids), seen = _against_the_server(_check_and_list_models)
    assert check.success is True, check.message
    assert model_ids == [
        "meta-llama/Llama-3.3-70B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
    ]
    assert [path for path, _ in seen] == [
        "/v1/openai/models",
        "/v1/openai/models",
    ]
