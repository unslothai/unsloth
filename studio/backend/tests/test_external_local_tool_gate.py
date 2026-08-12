# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route-level tests for the self-hosted local tool loop gate.

Self-hosted OAI-compat connections (llama.cpp / vLLM / Ollama / custom)
advertise the registry's ``studio_tools`` capability, so a request carrying the
explicit tool signal enters Unsloth's local loop. These tests drive the real
``_proxy_to_external_provider`` gate and assert the loop is (or is not) entered:

- ``enabled_tools`` / ``mcp_enabled`` on a self-hosted provider -> loop.
- Client-supplied ``tools`` schemas stay on the pure passthrough.
- A zero ``max_tool_calls_per_message`` budget disables the loop.
- A non-streaming tool request is rejected (the loop streams each turn).
"""

import httpx
import pytest

from core.inference import external_provider as ep_mod


def _sse(body: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        content = body,
        headers = {"content-type": "text/event-stream"},
    )


def _mock_external_http_client(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )


def _v1_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import routes.inference as inference_route
    from auth.authentication import get_current_subject
    from utils.api_errors import install_api_error_handlers

    class _UnusedBackend:
        is_loaded = False

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _UnusedBackend())

    app = FastAPI()
    app.include_router(inference_route.router, prefix = "/v1")
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app), inference_route


def _self_hosted_tool_request(**overrides) -> dict:
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "provider_type": "llama_cpp",
        "external_model": "llama-3.2-3b",
        "stream": True,
        "enable_tools": True,
        "enabled_tools": ["web_search"],
    }
    body.update(overrides)
    return body


def test_self_hosted_enabled_tools_enter_local_loop(monkeypatch):
    client, inference_route = _v1_client(monkeypatch)
    captured = {}

    async def fake_local_loop(**kwargs):
        captured.update(kwargs)
        yield 'data: {"type":"content","text":"local-loop-marker"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(inference_route, "_stream_external_local_tool_loop", fake_local_loop)

    resp = client.post("/v1/chat/completions", json = _self_hosted_tool_request())
    assert resp.status_code == 200
    assert "local-loop-marker" in resp.text
    # The loop was entered with the resolved model and the allow-listed tool.
    assert captured["model"] == "llama-3.2-3b"
    tool_names = [t["function"]["name"] for t in captured["tools"]]
    assert tool_names == ["web_search"]


def test_client_supplied_tools_stay_on_passthrough(monkeypatch):
    client, inference_route = _v1_client(monkeypatch)
    _mock_external_http_client(monkeypatch)

    def _boom(**kwargs):
        raise AssertionError("local loop must not run for client-supplied tools")

    monkeypatch.setattr(inference_route, "_stream_external_local_tool_loop", _boom)

    resp = client.post(
        "/v1/chat/completions",
        json = _self_hosted_tool_request(
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "my_custom_tool",
                        "description": "caller-owned schema",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        ),
    )
    assert resp.status_code == 200
    assert "ok" in resp.text


def test_zero_tool_budget_disables_loop(monkeypatch):
    client, inference_route = _v1_client(monkeypatch)
    _mock_external_http_client(monkeypatch)

    def _boom(**kwargs):
        raise AssertionError("local loop must not run with a zero tool budget")

    monkeypatch.setattr(inference_route, "_stream_external_local_tool_loop", _boom)

    resp = client.post(
        "/v1/chat/completions",
        json = _self_hosted_tool_request(max_tool_calls_per_message = 0),
    )
    assert resp.status_code == 200
    assert "ok" in resp.text


def test_confirm_tool_calls_allowed_for_self_hosted_local_tools(monkeypatch):
    # confirm_tool_calls on a self-hosted studio-tools request must NOT hit the
    # "only supported for local streaming tools" guard: the external loop parks
    # on the same approval gate as the local GGUF / safetensors loops.
    client, inference_route = _v1_client(monkeypatch)
    captured = {}

    async def fake_local_loop(**kwargs):
        captured.update(kwargs)
        yield 'data: {"type":"content","text":"loop-marker"}\n\n'
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(inference_route, "_stream_external_local_tool_loop", fake_local_loop)

    resp = client.post(
        "/v1/chat/completions",
        json = _self_hosted_tool_request(
            confirm_tool_calls = True,
            permission_mode = "ask",
        ),
    )
    assert resp.status_code == 200
    assert "loop-marker" in resp.text
    assert captured["payload"].confirm_tool_calls is True


def test_confirm_tool_calls_still_rejected_for_hosted_provider(monkeypatch):
    # The upstream "local only" guard still holds for hosted providers: an
    # openai_codex / openai request with confirm_tool_calls stays rejected.
    client, _inference_route = _v1_client(monkeypatch)

    resp = client.post(
        "/v1/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "provider_type": "openai",
            "external_model": "gpt-4.1",
            "enable_tools": True,
            "enabled_tools": ["web_search"],
            "confirm_tool_calls": True,
        },
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["param"] == "confirm_tool_calls"
    assert "only supported for local streaming tools" in body["error"]["message"]


def test_non_streaming_self_hosted_tool_request_rejected(monkeypatch):
    client, _inference_route = _v1_client(monkeypatch)

    resp = client.post(
        "/v1/chat/completions",
        json = _self_hosted_tool_request(stream = False),
    )
    assert resp.status_code == 400
    body = resp.json()
    assert body["error"]["param"] == "enable_tools"
    assert "require stream=true" in body["error"]["message"]


def test_self_hosted_tool_request_without_signal_stays_passthrough(monkeypatch):
    # No enabled_tools / mcp_enabled: the loop must not be entered even though
    # the provider advertises studio_tools (the loop follows the explicit opt-in).
    client, inference_route = _v1_client(monkeypatch)
    _mock_external_http_client(monkeypatch)

    def _boom(**kwargs):
        raise AssertionError("local loop must not run without the tool signal")

    monkeypatch.setattr(inference_route, "_stream_external_local_tool_loop", _boom)

    resp = client.post(
        "/v1/chat/completions",
        json = {
            "messages": [{"role": "user", "content": "hi"}],
            "provider_type": "llama_cpp",
            "external_model": "llama-3.2-3b",
            "stream": True,
        },
    )
    assert resp.status_code == 200
    assert "ok" in resp.text
