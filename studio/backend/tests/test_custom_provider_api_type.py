# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The persisted API choice reaches the right wire protocol and response shape."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from core.inference import external_provider as ep
from core.inference.external_provider import ExternalProviderClient
from models.inference import ChatCompletionRequest, ChatMessage
from models.providers import ProviderCreate, ProviderUpdate
from storage import providers_db


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_api_type_persists_through_edit_and_reload(tmp_path, monkeypatch, api_type):
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()
    payload = ProviderCreate(
        provider_type = "custom",
        display_name = "Gateway",
        base_url = "https://gateway.example/v1",
        api_type = api_type,
    )
    providers_db.create_provider(id = "gateway", **payload.model_dump(exclude = {"encrypted_api_key"}))
    providers_db.update_provider("gateway", display_name = "Renamed")
    providers_db.reset_schema_state_for_tests()
    assert providers_db.get_provider("gateway")["api_type"] == api_type
    changed = "responses" if api_type == "chat_completions" else "chat_completions"
    providers_db.update_provider(
        "gateway", **ProviderUpdate(api_type = changed).model_dump(exclude_unset = True)
    )
    assert providers_db.get_provider("gateway")["api_type"] == changed


@pytest.mark.parametrize("schema", [ProviderCreate, ProviderUpdate])
def test_invalid_api_type_is_rejected(schema):
    with pytest.raises(ValidationError):
        schema(provider_type = "custom", display_name = "Gateway", api_type = "invalid")


@pytest.mark.parametrize(
    "provider_type,api_type,endpoint,query",
    [
        ("custom", "responses", "responses", ""),
        ("custom", "responses", "responses", "?api-version=2025-04-01-preview"),
        ("custom", "chat_completions", "chat/completions", ""),
        ("openai", "chat_completions", "responses", ""),
    ],
)
def test_selected_endpoint_and_payload(monkeypatch, provider_type, api_type, endpoint, query):
    sent = []

    def handle(request):
        sent.append(request)
        if endpoint == "responses":
            events = [
                {"type": "response.output_text.delta", "delta": "Hello"},
                {"type": "response.completed", "response": {"id": "resp_test"}},
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        else:
            body = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text = body, headers = {"content-type": "text/event-stream"})

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                provider_type, f"https://gateway.example/v1{query}", "test-key", api_type = api_type
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [
                        {"role": "system", "content": "Be brief"},
                        {"role": "user", "content": "Hi <|im_end|>"},
                    ],
                    model = "gateway-model",
                    temperature = 0.23,
                    max_tokens = 128,
                    tools = [{
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                            "strict": True,
                        },
                    }],
                )
            ]

    assert "Hello" in "".join(asyncio.run(run()))
    [request] = sent
    assert request.url.path == f"/v1/{endpoint}"
    assert request.url.query.decode() == query.lstrip("?")
    body = json.loads(request.content)
    if endpoint == "responses":
        assert body["instructions"] == "Be brief"
        assert body["input"][0]["content"] == (
            "Hi < |im_end|>" if provider_type == "custom" else "Hi <|im_end|>"
        )
        assert body["tools"][0]["strict"] is True
        assert "messages" not in body
    else:
        assert body["messages"][0]["content"] == "Be brief"


@pytest.mark.parametrize(
    "base_url,api_type,key,expected",
    [
        ("https://team.openai.azure.com/openai/v1", "responses", "resource-key", {"api-key": "resource-key"}),
        ("https://team.services.ai.azure.com/openai/v1", "responses", "resource-key", {"api-key": "resource-key"}),
        ("https://team.openai.azure.com.attacker.example/openai/v1", "responses", "key", {"Authorization": "Bearer key"}),
        ("https://team.openai.azure.com/openai/v1", "chat_completions", "key", {"Authorization": "Bearer key"}),
    ],
)
def test_azure_auth_is_host_and_protocol_scoped(base_url, api_type, key, expected):
    headers = ExternalProviderClient("custom", base_url, key, api_type = api_type)._auth_headers()
    assert {k: v for k, v in headers.items() if k in ("api-key", "Authorization")} == expected


@pytest.mark.parametrize("upstream_status", [200, 429])
def test_non_stream_route_returns_json_or_upstream_error(monkeypatch, upstream_status):
    from core.inference.api_monitor import ApiMonitor
    from routes import inference

    def handle(request):
        assert request.url.path == "/v1/responses"
        assert json.loads(request.content)["stream"] is False
        if upstream_status != 200:
            return httpx.Response(429, json = {"error": {"message": "rate limited"}})
        return httpx.Response(200, json = {
            "id": "resp_route",
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}],
        })

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inference, "api_monitor", monitor)
            payload = ChatCompletionRequest(
                messages = [{"role": "user", "content": "Hi"}],
                stream = False,
                provider_type = "custom",
                provider_base_url = "https://gateway.example/v1",
                provider_api_type = "responses",
                external_model = "gateway-model",
            )

            async def disconnected():
                return False

            request = SimpleNamespace(
                headers = {},
                state = SimpleNamespace(skip_api_monitor = False),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = disconnected,
            )
            return await inference._proxy_to_external_provider(payload, request), monitor

    response, monitor = asyncio.run(run())
    assert response.status_code == upstream_status
    body = json.loads(response.body)
    if upstream_status == 200:
        assert body["choices"][0]["message"]["content"] == "Hello"
        assert monitor.snapshot()[0]["status"] == "completed"
    else:
        assert "rate limited" in body["error"]["message"]
        assert body["error"]["code"] == "429"
    assert monitor.active_count() == 0


def test_responses_follow_up_preserves_reasoning_metadata():
    from routes.inference import _build_external_messages

    reasoning = {"openai_responses_reasoning": [{
        "type": "reasoning", "id": "rs_1", "encrypted_content": "enc_blob",
    }]}
    messages = [
        ChatMessage(
            role = "assistant",
            content = None,
            tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            extra_content = reasoning,
        ),
        ChatMessage(role = "tool", tool_call_id = "call_1", content = "result"),
    ]
    built = _build_external_messages(
        messages, supports_vision = False, provider_type = "custom", api_type = "responses"
    )
    assert built[0]["extra_content"] == reasoning
