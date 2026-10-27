# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""OpenAI's model denylist applies only to its own host; Azure and self-hosted ids stay unfiltered."""

import asyncio

from models.providers import ProviderModelsRequest
from routes import providers as providers_route


def _list_models(monkeypatch, base_url: str, ids: list[str]) -> list[str]:
    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def list_models(self):
            return [{"id": model_id} for model_id in ids]

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-test"
    )
    payload = ProviderModelsRequest(provider_type = "openai", base_url = base_url)
    result = asyncio.new_event_loop().run_until_complete(
        providers_route.list_provider_models(payload, "tester", False)
    )
    return [m.id for m in result]


LIVE = [
    "gpt-5.5",
    "gpt-5.5-image-analysis",
    "gpt-4o-audio-summariser",
    "internal-search-preview",
    "qwen3-instruct",
]


def test_the_openai_denylist_only_applies_to_the_openai_host(monkeypatch):
    assert _list_models(monkeypatch, "https://my-resource.openai.azure.com/openai/v1", LIVE) == LIVE
    assert _list_models(monkeypatch, "http://127.0.0.1:11434/v1", LIVE) == LIVE


def test_the_openai_denylist_still_applies_on_api_openai_com(monkeypatch):
    kept = _list_models(monkeypatch, "https://api.openai.com/v1", LIVE)
    assert kept == ["gpt-5.5"], kept
    dropped = _list_models(
        monkeypatch,
        "https://api.openai.com/v1",
        ["text-embedding-3-small", "gpt-audio", "o3-deep-research", "tts-1"],
    )
    assert dropped == [], dropped
