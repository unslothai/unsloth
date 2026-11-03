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
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking it over"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"hello"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    '"usage":{"prompt_tokens":8,"completion_tokens":2,"total_tokens":10}}\n\n'
    "data: [DONE]\n\n"
).encode()
_MODELS = b'{"object":"list","data":[{"id":"glm-5.3","object":"model"},{"id":"glm-4.7","object":"model"}]}'
_NOT_FOUND = b'{"error":{"code":"404","message":"Not Found"}}'


class _ZaiLikeServer(BaseHTTPRequestHandler):
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
    _ZaiLikeServer.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _ZaiLikeServer)
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()
    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    ep_mod._http_client = httpx.AsyncClient()
    try:
        result = loop.run_until_complete(
            make_call(f"http://127.0.0.1:{httpd.server_address[1]}/api/paas/v4")
        )
        loop.run_until_complete(ep_mod._http_client.aclose())
    finally:
        ep_mod._http_client = previous
        loop.close()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout = 10)
    return result, _ZaiLikeServer.seen


async def _chat_through_the_route(base_url: str) -> str:
    import routes.inference as ri
    from starlette.requests import Request

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    payload = ChatCompletionRequest(
        provider_type = "zai",
        provider_base_url = base_url,
        messages = [{"role": "user", "content": "hi"}],
        model = "glm-5.3",
        stream = True,
        presence_penalty = 0.5,
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
        ProviderTestRequest(provider_type = "zai", base_url = base_url),
        _current_subject = "unsloth",
        via_api_key = False,
    )
    models = await providers_route.list_provider_models(
        ProviderModelsRequest(provider_type = "zai", base_url = base_url),
        _current_subject = "unsloth",
        via_api_key = False,
    )
    return check, [m.id for m in models]


def test_zai_is_offered_on_the_pay_as_you_go_api():
    zai = next(e for e in list_available_providers() if e["provider_type"] == "zai")
    assert zai["base_url"] == "https://api.z.ai/api/paas/v4"
    assert not zai["hidden"]
    assert zai["model_list_mode"] == "remote"
    for free_model in ("glm-4.7-flash", "glm-4.5-flash", "glm-4.6v-flash"):
        assert free_model in zai["default_models"]
    vision = {m: c["vision"] for m, c in zai["model_capabilities"].items()}
    assert vision["glm-5.3"] is False and vision["glm-4.5-flash"] is False
    assert vision["glm-5.3-flash"] is True and vision["glm-4.6v-flash"] is True


def test_zai_chat_reaches_the_api_and_streams_reasoning_and_usage_back():
    streamed, seen = _against_the_server(_chat_through_the_route)
    [(path, body)] = seen
    assert path == "/api/paas/v4/chat/completions"
    assert body["model"] == "glm-5.3"
    assert body["temperature"] == 0.6
    assert "presence_penalty" not in body
    chunks = [
        json.loads(line[len("data: ") :])
        for line in streamed.splitlines()
        if line.startswith("data: {")
    ]
    deltas = [c["choices"][0]["delta"] for c in chunks if c.get("choices")]
    assert "thinking it over" in "".join(d.get("reasoning_content") or "" for d in deltas)
    assert "hello" in "".join(d.get("content") or "" for d in deltas)
    assert any((c.get("usage") or {}).get("completion_tokens") == 2 for c in chunks)


def test_zai_check_and_model_list_read_the_live_models_route():
    (check, model_ids), seen = _against_the_server(_check_and_list_models)
    assert check.success is True, check.message
    assert model_ids == ["glm-5.3", "glm-4.7"]
    assert [path for path, _ in seen] == ["/api/paas/v4/models", "/api/paas/v4/models"]


def test_zai_models_get_their_window_and_modalities_from_the_catalog():
    from core.inference.provider_model_capabilities import trim_models_dev_catalog

    catalog = trim_models_dev_catalog(
        {
            "zai": {
                "models": {
                    "glm-4.7-flashx": {
                        "modalities": {"input": ["text"]},
                        "limit": {"context": 200000},
                    },
                    "glm-5v-turbo": {
                        "modalities": {"input": ["text", "image"]},
                        "limit": {"context": 200000},
                    },
                }
            }
        }
    )
    assert catalog["zai"]["glm-4.7-flashx"] == {"input": ["text"], "context": 200000}
    assert catalog["zai"]["glm-5v-turbo"]["input"] == ["text", "image"]


def test_zai_mcp_pictures_reach_only_its_vision_models(monkeypatch):
    import routes.providers as providers_route
    from core.inference.providers import get_provider_info
    from routes.inference import _external_takes_mcp_images

    info = get_provider_info("zai")
    catalog = {
        "fetched_at": 0,
        "providers": {
            "zai": {
                "glm-5v-turbo": {"input": ["text", "image"]},
                "glm-4.7-flashx": {"input": ["text"]},
            }
        },
    }
    monkeypatch.setattr(providers_route, "_model_catalog_cache", catalog)
    assert _external_takes_mcp_images("zai", True, "glm-4.6v-flash", info) is True
    assert _external_takes_mcp_images("zai", True, "glm-5.3", info) is False
    assert _external_takes_mcp_images("zai", True, "glm-5v-turbo", info) is True
    assert _external_takes_mcp_images("zai", True, "glm-4.7-flashx", info) is False
    assert _external_takes_mcp_images("zai", True, "glm-unlisted", info) is False
