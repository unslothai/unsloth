# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ollama advertises reasoning per model, not per connection.

``/api/tags`` reports a ``capabilities`` list per model that names ``thinking``
for the ones that can reason. ``/v1/models`` answers first on a populated host
and carries no such field, so the native catalog has to be consulted for its
capabilities even when the OpenAI-compatible listing already succeeded — that is
what lets the chat UI key its thinking controls on the selected model instead of
asking the user to flag the whole connection.
"""

import asyncio

import httpx
import pytest

from core.inference import external_provider
from core.inference.external_provider import ExternalProviderClient
from models.providers import ProviderModelsRequest
from routes import providers as providers_route

TAGS = {
    "models": [
        {"name": "qwen3:8b", "capabilities": ["completion", "tools", "thinking"]},
        {"name": "llama3.2:3b", "capabilities": ["completion", "tools"]},
        # Older Ollama builds report no capabilities at all.
        {"name": "vicuna:7b"},
    ]
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _stub_http(monkeypatch, *, v1_payload, tags_payload):
    """Answer /v1/models and /api/tags; None means "raise like an unreachable host"."""
    seen: list[str] = []

    async def _get(url, **_kwargs):
        seen.append(url)
        payload = tags_payload if url.endswith("/api/tags") else v1_payload
        if payload is None:
            raise httpx.ConnectError("unreachable")
        return _FakeResponse(payload)

    monkeypatch.setattr(external_provider._http_client, "get", _get)
    return seen


def _list_models(monkeypatch, *, v1_payload, tags_payload = TAGS):
    seen = _stub_http(monkeypatch, v1_payload = v1_payload, tags_payload = tags_payload)
    client = ExternalProviderClient(
        provider_type = "ollama",
        base_url = "http://127.0.0.1:11434/v1",
        api_key = None,
    )
    models = asyncio.new_event_loop().run_until_complete(client.list_models())
    return models, seen


V1_CATALOG = {
    "data": [
        {"id": "qwen3:8b", "owned_by": "library"},
        {"id": "llama3.2:3b", "owned_by": "library"},
        {"id": "vicuna:7b", "owned_by": "library"},
    ]
}


def test_the_native_catalog_carries_per_model_capabilities(monkeypatch):
    models, _seen = _list_models(monkeypatch, v1_payload = {"data": []})
    by_id = {m["id"]: m for m in models}
    assert by_id["qwen3:8b"]["capabilities"] == ["completion", "tools", "thinking"]
    assert by_id["llama3.2:3b"]["capabilities"] == ["completion", "tools"]
    # A row that says nothing must not read as "explicitly not thinking".
    assert "capabilities" not in by_id["vicuna:7b"]


def test_a_populated_v1_catalog_still_gets_capabilities(monkeypatch):
    models, seen = _list_models(monkeypatch, v1_payload = V1_CATALOG)
    assert any(url.endswith("/api/tags") for url in seen), seen
    by_id = {m["id"]: m for m in models}
    # /v1/models stays the id source, so its own fields survive.
    assert by_id["qwen3:8b"]["owned_by"] == "library"
    assert "thinking" in by_id["qwen3:8b"]["capabilities"]
    assert "thinking" not in by_id["llama3.2:3b"]["capabilities"]
    assert "capabilities" not in by_id["vicuna:7b"]


def test_an_unreachable_tags_endpoint_leaves_the_catalog_usable(monkeypatch):
    models, _seen = _list_models(monkeypatch, v1_payload = V1_CATALOG, tags_payload = None)
    assert [m["id"] for m in models] == ["qwen3:8b", "llama3.2:3b", "vicuna:7b"]
    assert all("capabilities" not in m for m in models)


def test_a_non_ollama_provider_never_queries_tags(monkeypatch):
    seen = _stub_http(monkeypatch, v1_payload = V1_CATALOG, tags_payload = TAGS)
    client = ExternalProviderClient(
        provider_type = "vllm",
        base_url = "http://127.0.0.1:8000/v1",
        api_key = None,
    )
    asyncio.new_event_loop().run_until_complete(client.list_models())
    assert not any(url.endswith("/api/tags") for url in seen), seen


@pytest.mark.parametrize(
    "raw, expected",
    [
        (["thinking"], ["thinking"]),
        ([], []),
        # Foreign shapes are dropped, never surfaced as a broken capability list.
        ("thinking", None),
        ({"thinking": True}, None),
        (["thinking", 3, "", None], ["thinking"]),
    ],
)
def test_the_route_only_forwards_string_capability_names(raw, expected):
    assert providers_route._model_capability_names({"capabilities": raw}) == expected
    assert providers_route._model_capability_names({}) is None


def test_the_models_route_surfaces_capabilities(monkeypatch):
    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def list_models(self):
            return [
                {"id": "qwen3:8b", "capabilities": ["thinking"]},
                {"id": "llama3.2:3b", "capabilities": []},
                {"id": "vicuna:7b"},
            ]

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: None
    )
    payload = ProviderModelsRequest(
        provider_type = "ollama",
        base_url = "http://127.0.0.1:11434/v1",
    )
    result = asyncio.new_event_loop().run_until_complete(
        providers_route.list_provider_models(payload, "tester", False)
    )
    assert [(m.id, m.capabilities) for m in result] == [
        ("qwen3:8b", ["thinking"]),
        ("llama3.2:3b", []),
        ("vicuna:7b", None),
    ]
