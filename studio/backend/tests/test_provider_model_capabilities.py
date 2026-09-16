# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from pathlib import Path

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.provider_model_capabilities import (
    openrouter_model_capabilities,
    provider_model_capabilities,
)
from models.providers import ProviderModelsRequest


def _load_providers_route():
    routes_dir = Path(__file__).resolve().parents[1] / "routes"
    previous = sys.modules.get("routes")
    package = types.ModuleType("routes")
    package.__path__ = [str(routes_dir)]
    sys.modules["routes"] = package
    try:
        for name, path in (
            ("routes.provider_credentials", routes_dir / "provider_credentials.py"),
            ("_capabilities_providers_route", routes_dir / "providers.py"),
        ):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("routes.provider_credentials", None)
        if previous is None:
            sys.modules.pop("routes", None)
        else:
            sys.modules["routes"] = previous


providers_route = _load_providers_route()


_RAW_GPT = {
    "id": "openai/gpt-5.5",
    "architecture": {"input_modalities": ["file", "image", "text"], "output_modalities": ["text"]},
    "reasoning": {
        "mandatory": False,
        "default_enabled": True,
        "supported_efforts": ["xhigh", "high", "medium", "low", "none", "turbo"],
        "default_effort": "medium",
    },
    "top_provider": {"context_length": 400000, "max_completion_tokens": 128000},
    "supported_parameters": ["reasoning", "tools", "Temperature"],
}

_RAW_R1 = {
    "id": "deepseek/deepseek-r1",
    "architecture": {"input_modalities": ["text"]},
    "reasoning": {"mandatory": True},
    "top_provider": {"max_completion_tokens": None},
}


def test_openrouter_entry_maps_efforts_modalities_and_cap():
    mapped = openrouter_model_capabilities(_RAW_GPT)
    assert mapped == {
        "id": "openai/gpt-5.5",
        "input_modalities": ["file", "image", "text"],
        "reasoning": {
            "supported_efforts": ["none", "low", "medium", "high", "xhigh"],
            "mandatory": False,
            "default_effort": "medium",
            "default_enabled": True,
        },
        "max_output_tokens": 128000,
        "supported_parameters": ["reasoning", "tools", "temperature"],
    }


def test_mandatory_only_reasoning_and_missing_fields():
    mapped = openrouter_model_capabilities(_RAW_R1)
    assert mapped["reasoning"] == {
        "supported_efforts": None,
        "mandatory": True,
        "default_effort": None,
        "default_enabled": None,
    }
    assert mapped["max_output_tokens"] is None
    assert mapped["supported_parameters"] is None
    plain = openrouter_model_capabilities({"id": "acme/plain"})
    assert plain["reasoning"] is None
    assert plain["input_modalities"] is None
    assert openrouter_model_capabilities({"id": ""}) is None
    assert openrouter_model_capabilities({"name": "no id"}) is None


def test_only_openrouter_is_mapped():
    assert provider_model_capabilities("openai", [_RAW_GPT]) == []
    assert [
        m["id"] for m in provider_model_capabilities("openrouter", [_RAW_GPT, "junk", _RAW_R1])
    ] == [
        "openai/gpt-5.5",
        "deepseek/deepseek-r1",
    ]


class _FakeClient:
    calls = 0
    raw: list[dict] = []
    fail = False

    def __init__(self, **kwargs):
        pass

    async def list_models(self):
        type(self).calls += 1
        if type(self).fail:
            raise RuntimeError("upstream down")
        return list(type(self).raw)

    async def close(self):
        pass


@pytest.fixture
def capability_route(monkeypatch):
    providers_route._model_capability_cache.clear()
    _FakeClient.calls = 0
    _FakeClient.raw = [_RAW_GPT, _RAW_R1]
    _FakeClient.fail = False
    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-test"
    )

    def call(provider_type: str):
        return asyncio.run(
            providers_route.list_provider_model_capabilities(
                ProviderModelsRequest(provider_type = provider_type),
                _current_subject = "tester",
                via_api_key = False,
            )
        )

    yield call
    providers_route._model_capability_cache.clear()


def test_route_maps_and_caches_the_openrouter_catalog(capability_route):
    first = capability_route("openrouter")
    assert [m["id"] for m in first] == ["openai/gpt-5.5", "deepseek/deepseek-r1"]
    assert first[0]["reasoning"]["supported_efforts"] == ["none", "low", "medium", "high", "xhigh"]
    _FakeClient.fail = True
    assert capability_route("openrouter") == first
    assert _FakeClient.calls == 1


def test_route_answers_empty_for_providers_without_a_capability_catalog(capability_route):
    assert capability_route("openai") == []
    assert _FakeClient.calls == 0


def test_route_surfaces_upstream_failure_when_nothing_is_cached(capability_route):
    _FakeClient.fail = True
    with pytest.raises(Exception) as excinfo:
        capability_route("openrouter")
    assert getattr(excinfo.value, "status_code", None) == 502


def _openrouter_body(model: str = "deepseek/deepseek-v4-pro", **kwargs) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        sse = 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
        return httpx.Response(200, content = sse, headers = {"content-type": "text/event-stream"})

    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = "openrouter",
        base_url = "https://openrouter.ai/api/v1",
        api_key = "sk-test",
    )

    async def run() -> None:
        try:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = model,
                **kwargs,
            ):
                pass
        finally:
            await mock_client.aclose()

    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    ep_mod._http_client = mock_client
    try:
        loop.run_until_complete(run())
    finally:
        ep_mod._http_client = previous
        loop.close()
    return captured["body"]


@pytest.mark.parametrize("effort", ["minimal", "low", "medium", "high", "xhigh", "max"])
def test_every_effort_level_reaches_openrouter(effort):
    assert _openrouter_body(reasoning_effort = effort, enable_thinking = True)["reasoning"] == {
        "effort": effort
    }


def test_none_effort_switches_reasoning_off():
    assert _openrouter_body(reasoning_effort = "none", enable_thinking = False)["reasoning"] == {
        "enabled": False
    }


def test_none_effort_on_a_mandatory_route_sends_no_reasoning_field():
    body = _openrouter_body(
        model = "deepseek/deepseek-r1", reasoning_effort = "none", enable_thinking = False
    )
    assert "reasoning" not in body


def test_a_bare_toggle_still_maps_to_enabled():
    assert _openrouter_body(enable_thinking = True)["reasoning"] == {"enabled": True}
    assert _openrouter_body(enable_thinking = False)["reasoning"] == {"enabled": False}
    assert "reasoning" not in _openrouter_body()


def test_null_efforts_accept_the_full_scale_while_omitted_efforts_remain_toggle_only():
    unrestricted = openrouter_model_capabilities(
        {"id": "acme/reasoning", "reasoning": {"supported_efforts": None}}
    )
    assert unrestricted["reasoning"]["supported_efforts"] == [
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    toggle = openrouter_model_capabilities({"id": "acme/reasoning", "reasoning": {}})
    assert toggle["reasoning"]["supported_efforts"] is None
