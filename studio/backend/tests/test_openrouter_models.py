# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.provider_model_capabilities import provider_model_capabilities
from core.inference.providers import get_provider_info
from models.providers import ProviderTestRequest
from routes import providers as providers_route

CATALOG = {
    "data": [
        {
            "id": "old/chat",
            "created": 100,
            "context_length": 8192,
            "architecture": {"output_modalities": ["text"]},
            "endpoints": ["/api/v1/chat/completions"],
        },
        {
            "id": "new/chat",
            "created": 300,
            "context_length": 131072,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "endpoints": ["/api/v1/chat/completions", "/api/v1/messages"],
            "reasoning": {"supported_efforts": ["low", "high"], "mandatory": False},
            "top_provider": {"max_completion_tokens": 32768},
            "supported_parameters": ["reasoning", "tools"],
        },
        {
            "id": "new/chat:batch",
            "created": 300,
            "architecture": {"output_modalities": ["text"]},
            "endpoints": [],
        },
        {
            "id": "some/image-gen",
            "created": 200,
            "architecture": {"output_modalities": ["image"]},
            "endpoints": ["/api/v1/chat/completions"],
        },
        {"id": "legacy/no-metadata", "created": 50},
    ]
}


@pytest.fixture
def catalog_server(monkeypatch):
    state = {"calls": 0, "fail": False}

    async def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        await asyncio.sleep(0.01)
        assert request.url.path == "/api/v1/models"
        if state["fail"]:
            return httpx.Response(503)
        return httpx.Response(200, json = CATALOG)

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )
    monkeypatch.setattr(ep_mod, "_openrouter_catalog", {})
    monkeypatch.setattr(ep_mod, "_openrouter_catalog_locks", {})
    return state


def _expire_catalog():
    for key, (_, models) in ep_mod._openrouter_catalog.items():
        ep_mod._openrouter_catalog[key] = (0.0, models)


def _client(api_key: str = "") -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "openrouter",
        base_url = "https://openrouter.ai/api/v1",
        api_key = api_key,
    )


def test_registry_uses_live_catalog_with_live_defaults():
    info = get_provider_info("openrouter")
    assert info.get("model_list_mode", "remote") == "remote"
    assert "openrouter/free" in info["default_models"]


def test_filters_non_chat_models_and_sorts_newest_first(catalog_server):
    models = asyncio.run(_client().list_models())
    assert [m["id"] for m in models] == ["new/chat", "old/chat", "legacy/no-metadata"]
    assert models[0]["context_length"] == 131072


def test_keeps_metadata_for_model_capabilities(catalog_server):
    models = asyncio.run(_client().list_models())
    caps = {c["id"]: c for c in provider_model_capabilities("openrouter", models)}
    assert caps["new/chat"]["input_modalities"] == ["text"]
    assert caps["new/chat"]["reasoning"]["supported_efforts"] == ["low", "high"]
    assert caps["new/chat"]["max_output_tokens"] == 32768
    assert caps["new/chat"]["supported_parameters"] == ["reasoning", "tools"]


def test_catalog_is_cached_across_clients(catalog_server):
    async def run():
        await asyncio.gather(*(_client().list_models() for _ in range(5)))
        await _client().list_models()

    asyncio.run(run())
    assert catalog_server["calls"] == 1


def test_catalog_cache_is_scoped_to_the_api_key(catalog_server):
    async def run():
        await _client("key-a").list_models()
        await _client("key-b").list_models()
        await _client("key-a").list_models()

    asyncio.run(run())
    assert catalog_server["calls"] == 2


def test_serves_stale_catalog_and_backs_off_when_refresh_fails(catalog_server):
    first = asyncio.run(_client().list_models())
    _expire_catalog()
    catalog_server["fail"] = True
    assert asyncio.run(_client().list_models()) == first
    assert asyncio.run(_client().list_models()) == first
    assert catalog_server["calls"] == 2


def test_a_slow_catalog_does_not_block_another_key(monkeypatch):
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("authorization") == "Bearer slow":
            await release.wait()
        return httpx.Response(200, json = CATALOG)

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )
    monkeypatch.setattr(ep_mod, "_openrouter_catalog", {})
    monkeypatch.setattr(ep_mod, "_openrouter_catalog_locks", {})

    async def run():
        slow = asyncio.create_task(_client("slow").list_models())
        await asyncio.sleep(0)
        fast = await asyncio.wait_for(_client("fast").list_models(), timeout = 1)
        release.set()
        await slow
        return fast

    assert [m["id"] for m in asyncio.run(run())][0] == "new/chat"


def test_raises_when_no_cache_and_fetch_fails(catalog_server):
    catalog_server["fail"] = True
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(_client().list_models())


def test_connection_test_does_not_pass_on_a_cached_catalog(catalog_server, monkeypatch):
    monkeypatch.setattr(providers_route, "validate_provider_base_url", lambda url: url)
    monkeypatch.setattr(providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "")
    request = ProviderTestRequest(provider_type = "openrouter")
    assert asyncio.run(providers_route.test_provider(request, "tester", False)).success
    catalog_server["fail"] = True
    result = asyncio.run(providers_route.test_provider(request, "tester", False))
    assert not result.success
