# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.provider_model_capabilities import (
    MODELS_DEV_MODEL_LIST_SOURCES,
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


_RAW_MODELS_DEV = {
    "openrouter": {
        "models": {
            "DeepSeek/DeepSeek-V4-Pro": {
                "reasoning": True,
                "reasoning_options": [
                    {"type": "toggle"},
                    {"type": "effort", "values": ["xhigh", "high", "turbo"]},
                ],
                "modalities": {"input": ["text"]},
            },
            "openai/gpt-4o": {"reasoning": False, "modalities": {"input": ["text", "image"]}},
        }
    },
    "google": {
        "models": {
            "gemini-9-flash": {
                "reasoning": True,
                "reasoning_options": [],
                "modalities": {"input": ["text", "image"]},
            }
        }
    },
    "unrelated": {"models": {"x": {"reasoning": True}}},
}


def test_models_dev_catalog_is_trimmed_and_remapped():
    from core.inference.provider_model_capabilities import trim_models_dev_catalog

    catalog = trim_models_dev_catalog(_RAW_MODELS_DEV)
    assert set(catalog) == {"openrouter", "gemini"}
    assert catalog["openrouter"]["deepseek/deepseek-v4-pro"] == {
        "reasoning": True,
        "toggle": True,
        "efforts": ["high", "xhigh"],
        "input": ["text"],
    }
    assert catalog["openrouter"]["openai/gpt-4o"] == {"input": ["text", "image"]}
    assert catalog["gemini"]["gemini-9-flash"] == {"reasoning": True, "input": ["text", "image"]}


# The frontend merges this over its bundled snapshot, so a junk value is worse than none.
_CONTEXT_CASES = [
    ({"context": 1_050_000, "output": 100_000}, 1_050_000),
    ({"context": 480}, 480),
    (None, None),
    ("not-a-dict", None),
    ({"output": 4096}, None),
    ({"context": None}, None),
    ({"context": 0}, None),
    ({"context": -1}, None),
    # bool is an int and True > 0: check bool explicitly or a flag emits context=True.
    ({"context": True}, None),
    ({"context": False}, None),
    ({"context": 1000.5}, None),
    ({"context": "1000"}, None),
]


@pytest.mark.parametrize(("limit", "expected"), _CONTEXT_CASES)
def test_the_published_context_window_survives_trimming(limit, expected):
    from core.inference.provider_model_capabilities import _trim_models_dev_model

    model = {"modalities": {"input": ["text"]}}
    if limit is not None:
        model["limit"] = limit
    entry = _trim_models_dev_model(model)
    assert entry.get("context") == expected


def test_the_context_window_reaches_the_served_catalog():
    from core.inference.provider_model_capabilities import trim_models_dev_catalog

    raw = {
        "openrouter": {
            "models": {
                "openai/gpt-4o": {
                    "modalities": {"input": ["text", "image"]},
                    "limit": {"context": 128_000, "input": 120_000, "output": 16_384},
                }
            }
        }
    }
    catalog = trim_models_dev_catalog(raw)
    # limit.input is the prompt share and limit.output the completion cap; neither is the window.
    assert catalog["openrouter"]["openai/gpt-4o"] == {
        "input": ["text", "image"],
        "context": 128_000,
    }


class _FakeCatalogClient:
    calls = 0
    fail = False

    async def get(
        self,
        url,
        timeout = None,
    ):
        type(self).calls += 1
        if type(self).fail:
            raise RuntimeError("models.dev down")

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return _RAW_MODELS_DEV

        return _Resp()


@pytest.fixture
def catalog_route(monkeypatch, tmp_path):
    from core.inference import external_provider as ep

    providers_route._model_catalog_cache = None
    _FakeCatalogClient.calls = 0
    _FakeCatalogClient.fail = False
    monkeypatch.setattr(ep, "_client", lambda: _FakeCatalogClient())
    monkeypatch.setattr(providers_route, "cache_root", lambda: tmp_path)

    def call():
        return asyncio.run(providers_route.get_model_catalog(_current_subject = "tester"))

    yield call
    providers_route._model_catalog_cache = None


def test_model_catalog_route_fetches_once_and_persists_to_disk(
    capability_route, catalog_route, tmp_path
):
    first = catalog_route()
    assert first["providers"]["openrouter"]["deepseek/deepseek-v4-pro"]["efforts"] == [
        "high",
        "xhigh",
    ]
    assert (tmp_path / "model_catalog.json").exists()
    assert catalog_route() == first
    assert _FakeCatalogClient.calls == 1


def test_model_catalog_route_serves_the_disk_copy_when_models_dev_is_down(catalog_route, tmp_path):
    first = catalog_route()
    providers_route._model_catalog_cache = None
    stale = dict(first, fetched_at = first["fetched_at"] - 10 * 24 * 3600)
    (tmp_path / "model_catalog.json").write_text(json.dumps(stale))
    _FakeCatalogClient.fail = True
    served = catalog_route()
    assert served["providers"] == first["providers"]
    assert served["fetched_at"] == stale["fetched_at"]


def test_model_catalog_route_answers_503_offline_with_nothing_cached(catalog_route):
    _FakeCatalogClient.fail = True
    with pytest.raises(Exception) as excinfo:
        catalog_route()
    assert getattr(excinfo.value, "status_code", None) == 503


def _chat(released, **extra):
    return {
        "release_date": released,
        "modalities": {"input": ["text"], "output": ["text"]},
        **extra,
    }


_RAW_MODEL_LISTS = {
    "openai": {
        "models": {
            "gpt-8": _chat("2026-01-10"),
            "gpt-9": _chat("2026-05-01"),
            "gpt-9-2026-05-01": _chat("2026-05-01"),
            "gpt-7": _chat("2025-03-01", status = "deprecated"),
            "gpt-9-image": {
                "release_date": "2026-06-01",
                "modalities": {"input": ["text"], "output": ["image"]},
            },
            "text-embedding-9": _chat("2026-06-02"),
        }
    },
    "huggingface": {
        "models": {
            "openai/fresh-a": _chat("2026-04-01"),
            "openai/fresh-b": _chat("2026-04-02"),
            "community/fine-tune": _chat("2026-04-03"),
        }
    },
}


def _registry_models(provider_type):
    rows = asyncio.run(providers_route.list_registry(include_hidden = True, current_subject = "tester"))
    return next(r for r in rows if r["provider_type"] == provider_type)["default_models"]


@pytest.fixture
def listed_catalog(catalog_route, monkeypatch):
    for source, provider in _RAW_MODEL_LISTS.items():
        monkeypatch.setitem(_RAW_MODELS_DEV, source, provider)
    return catalog_route


def test_the_registry_lists_the_catalog_chat_models_newest_first(listed_catalog):
    listed_catalog()
    assert _registry_models("openai") == ["gpt-9", "gpt-8"]
    assert _registry_models("huggingface") == ["openai/fresh-b", "openai/fresh-a"]


def test_the_registry_never_fetches_and_is_never_empty_before_a_catalog_exists(listed_catalog):
    rows = asyncio.run(providers_route.list_registry(current_subject = "tester"))
    listed = {row["provider_type"] for row in rows if row["default_models"]}
    assert listed == set(MODELS_DEV_MODEL_LIST_SOURCES)
    assert _FakeCatalogClient.calls == 0


def test_providers_that_list_their_own_models_take_none_from_the_catalog(
    listed_catalog, monkeypatch
):
    monkeypatch.setitem(_RAW_MODELS_DEV, "openrouter", _RAW_MODEL_LISTS["openai"])
    listed_catalog()
    for name in ("openai_codex", "openrouter", "qwencloud", "typesafe", "liquid"):
        assert _registry_models(name) == [], name


def test_a_catalog_cached_before_model_lists_existed_is_refetched(listed_catalog, tmp_path):
    (tmp_path / "model_catalog.json").write_text(
        json.dumps({"fetched_at": time.time(), "providers": {}})
    )
    assert _registry_models("openai")
    assert _FakeCatalogClient.calls == 0
    listed_catalog()
    assert _FakeCatalogClient.calls == 1
    assert _registry_models("openai") == ["gpt-9", "gpt-8"]


def test_the_last_listed_models_survive_a_models_dev_outage(listed_catalog, tmp_path):
    first = listed_catalog()
    providers_route._model_catalog_cache = None
    stale = dict(
        json.loads((tmp_path / "model_catalog.json").read_text()),
        fetched_at = first["fetched_at"] - 10 * 24 * 3600,
    )
    (tmp_path / "model_catalog.json").write_text(json.dumps(stale))
    _FakeCatalogClient.fail = True
    listed_catalog()
    assert _registry_models("openai") == ["gpt-9", "gpt-8"]


def _serve_live_list(monkeypatch, live):
    class _LiveListClient:
        def __init__(self, **_kwargs):
            pass

        async def list_models(self):
            return [{"id": model_id} for model_id in live]

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _LiveListClient)
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "hf-test"
    )


def test_a_capped_live_list_keeps_the_catalog_models(listed_catalog, monkeypatch):
    live = [f"openai/older-{n}" for n in range(20)] + ["openai/fresh-a", "openai/fresh-b"]
    _serve_live_list(monkeypatch, live)
    listed_catalog()
    payload = ProviderModelsRequest(provider_type = "huggingface")
    listed = asyncio.run(providers_route.list_provider_models(payload, "tester", False))
    ids = [model.id for model in listed]
    assert ids[:2] == ["openai/fresh-a", "openai/fresh-b"]
    assert ids[2:] == live[:13]


def test_a_capped_live_list_keeps_the_connection_saved_models(listed_catalog, monkeypatch):
    live = [f"openai/older-{n}" for n in range(20)] + ["openai/fresh-a", "openai/fresh-b"]
    info = providers_route.get_provider_info("huggingface")
    saved = {
        "id": "hf-1",
        "provider_type": "huggingface",
        "api_type": "chat_completions",
        "base_url": info["base_url"],
        "display_name": "HF",
        "is_enabled": True,
        "models": ["openai/older-19"],
    }
    _serve_live_list(monkeypatch, live)
    monkeypatch.setattr(providers_route.providers_db, "get_provider", lambda _id: saved)
    monkeypatch.setattr(providers_route.providers_db, "is_owner_shared_provider", lambda _id: False)
    monkeypatch.setattr(
        providers_route.credential_secrets, "provider_api_key_saved", lambda _id: True
    )
    listed_catalog()
    payload = ProviderModelsRequest(provider_type = "huggingface", provider_id = "hf-1")
    listed = asyncio.run(providers_route.list_provider_models(payload, "tester", False))
    ids = [model.id for model in listed]
    assert len(ids) == 15
    assert ids[:3] == ["openai/older-19", "openai/fresh-a", "openai/fresh-b"]
