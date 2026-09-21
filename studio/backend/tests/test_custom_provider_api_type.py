# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom endpoint selection survives storage and reaches the selected wire protocol."""

import asyncio
import json
import sqlite3

import httpx
import pytest
from pydantic import ValidationError

from core.inference import external_provider as ep
from core.inference.external_provider import ExternalProviderClient
from models.providers import ProviderCreate, ProviderUpdate
from storage import providers_db


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_saved_api_type_survives_edit_and_reload(tmp_path, monkeypatch, api_type):
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()
    payload = ProviderCreate(
        provider_type = "custom",
        display_name = "Gateway",
        base_url = "https://gateway.example/v1",
        api_type = api_type,
    )
    providers_db.create_provider(
        id = "gateway",
        **payload.model_dump(exclude = {"encrypted_api_key"}),
    )
    providers_db.update_provider("gateway", display_name = "Renamed")
    providers_db.reset_schema_state_for_tests()
    assert providers_db.get_provider("gateway")["api_type"] == api_type
    assert providers_db.list_providers()[0]["api_type"] == api_type
    changed = "responses" if api_type == "chat_completions" else "chat_completions"
    providers_db.update_provider(
        "gateway", **ProviderUpdate(api_type = changed).model_dump(exclude_unset = True)
    )
    assert providers_db.get_provider("gateway")["api_type"] == changed


def test_legacy_database_defaults_to_chat_completions(tmp_path, monkeypatch):
    db = tmp_path / "studio.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE llm_providers (id TEXT PRIMARY KEY, provider_type TEXT, display_name TEXT, base_url TEXT, is_enabled INTEGER, created_at TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT INTO llm_providers VALUES ('old', 'custom', 'Gateway', 'https://gateway.example/v1', 1, '', '')"
        )
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db)
    providers_db.reset_schema_state_for_tests()
    assert providers_db.get_provider("old")["api_type"] == "chat_completions"


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
        ("vllm", "responses", "chat/completions", ""),
        ("openai", "chat_completions", "responses", ""),
    ],
)
def test_endpoint_and_payload_translation(monkeypatch, provider_type, api_type, endpoint, query):
    requests = []

    def handle(request):
        requests.append(request)
        if endpoint == "responses":
            events = [
                {"type": "response.output_text.delta", "delta": "Hello"},
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_test",
                        "usage": {"input_tokens": 4, "output_tokens": 1},
                    },
                },
            ]
            content = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        else:
            content = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, text = content, headers = {"content-type": "text/event-stream"})

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
                    top_p = 0.61,
                    max_tokens = 128,
                    tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                )
            ]

    lines = asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url.path == f"/v1/{endpoint}"
    assert requests[0].url.query.decode() == query.lstrip("?")
    assert requests[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(requests[0].content)
    assert "Hello" in "".join(lines)
    if endpoint == "responses":
        if provider_type == "custom":
            assert body["temperature"] == 0.23
            assert body["top_p"] == 0.61
        else:
            assert "temperature" not in body
            assert "top_p" not in body
        assert "messages" not in body
        assert body["instructions"] == "Be brief"
        assert body["input"][0] == {
            "role": "user",
            "content": "Hi < |im_end|>" if provider_type == "custom" else "Hi <|im_end|>",
        }
        assert body["max_output_tokens"] == 128
        assert body["tools"][0]["name"] == "lookup"
        assert "prompt_cache_retention" not in body
        assert "context_management" not in body
    else:
        assert body["messages"][0]["content"] == "Be brief"
        assert body["max_tokens"] == 128


def test_responses_non_streaming_translation(monkeypatch):
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json = {
                "id": "resp_test",
                "object": "response",
                "created_at": 1_700_000_000,
                "model": "gateway-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "lookup",
                        "arguments": '{"q":"docs"}',
                    },
                ],
                "usage": {
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                    "input_tokens_details": {"cached_tokens": 1},
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://gateway.example/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Hi"}],
                    model = "gateway-model",
                    stream = False,
                )
            ]

    lines = asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    assert json.loads(requests[0].content)["stream"] is False
    assert len(lines) == 1
    completion = json.loads(lines[0])
    assert completion["object"] == "chat.completion"
    assert completion["choices"][0]["message"] == {
        "role": "assistant",
        "content": "Hello",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"q":"docs"}'},
            }
        ],
    }
    assert completion["choices"][0]["finish_reason"] == "tool_calls"
    assert completion["usage"] == {
        "prompt_tokens": 4,
        "completion_tokens": 2,
        "total_tokens": 6,
        "prompt_tokens_details": {"cached_tokens": 1},
    }


def test_responses_non_streaming_failure_is_not_reported_as_completion(monkeypatch):
    def handle(request):
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            200,
            json = {
                "id": "resp_failed",
                "status": "failed",
                "error": {"message": "provider unavailable"},
            },
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            client = ExternalProviderClient(
                "custom",
                "https://gateway.example/v1",
                "test-key",
                api_type = "responses",
            )
            return [
                line
                async for line in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "Hi"}],
                    model = "gateway-model",
                    stream = False,
                )
            ]

    lines = asyncio.run(run())
    assert len(lines) == 1
    assert json.loads(lines[0])["error"]["message"] == "provider unavailable"


@pytest.fixture()
def provider_api(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from auth import storage as auth_storage
    from auth.authentication import (
        authenticated_via_api_key,
        get_current_credential,
        get_current_subject,
    )
    from routes import providers
    from storage import credential_secrets

    db = tmp_path / "studio.db"
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: db)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: db)
    monkeypatch.setattr(auth_storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    providers_db.reset_schema_state_for_tests()
    credential_secrets._schema_ready.clear()
    app = FastAPI()
    app.include_router(providers.router, prefix = "/providers")
    app.dependency_overrides[get_current_subject] = lambda: "unsloth"
    app.dependency_overrides[get_current_credential] = lambda: ("unsloth", None)
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    with TestClient(app) as client:
        yield client


def test_api_create_update_and_saved_target_binding(provider_api):
    from routes.providers import _bind_saved_provider_target
    from models.providers import ProviderTestRequest

    created = provider_api.post(
        "/providers/",
        json = {
            "provider_type": "custom",
            "display_name": "Gateway",
            "base_url": "https://gateway.example/v1",
            "api_type": "responses",
        },
    )
    assert created.status_code == 201
    saved = created.json()
    assert saved["api_type"] == "responses"
    assert provider_api.get("/providers/").json()[0]["api_type"] == "responses"
    renamed = provider_api.put(f"/providers/{saved['id']}", json = {"display_name": "Renamed"})
    assert renamed.json()["api_type"] == "responses"
    bound = _bind_saved_provider_target(
        ProviderTestRequest(
            provider_id = saved["id"],
            provider_type = "openai",
            base_url = "https://other.example/v1",
            api_type = "chat_completions",
        )
    )
    assert bound.provider_type == "custom"
    assert bound.base_url == "https://gateway.example/v1"
    assert bound.api_type == "responses"
    changed = provider_api.put(f"/providers/{saved['id']}", json = {"api_type": "chat_completions"})
    assert changed.status_code == 200
    assert changed.json()["api_type"] == "chat_completions"


@pytest.mark.parametrize("models_status", [200, 404])
@pytest.mark.parametrize("status", [200, 500])
def test_responses_only_connectivity_probe(provider_api, monkeypatch, models_status, status):
    paths = []

    def handle(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(
                models_status,
                json = {"data": [{"id": "responses-only"}]} if models_status == 200 else None,
            )
        assert request.url.path == "/v1/responses"
        assert "input" in json.loads(request.content)
        if status == 500:
            return httpx.Response(500, json = {"error": {"message": "provider unavailable"}})
        return httpx.Response(
            200,
            text = 'data: {"type":"response.output_text.delta","delta":"Hello"}\n\ndata: {"type":"response.completed","response":{"id":"resp_probe"}}\n\n',
        )

    async def run():
        from routes.providers import test_provider
        from models.providers import ProviderTestRequest
        async with httpx.AsyncClient(transport = httpx.MockTransport(handle)) as transport:
            monkeypatch.setattr(ep, "_http_client", transport)
            return await test_provider(
                ProviderTestRequest(
                    provider_type = "custom",
                    base_url = "https://gateway.example/v1",
                    api_type = "responses",
                    model_id = "responses-only",
                ),
                _current_subject = "unsloth",
                via_api_key = False,
            )

    result = asyncio.run(run())
    assert result.success == (status == 200)
    assert paths == ["/v1/models", "/v1/responses"]
