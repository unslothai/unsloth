# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The registry model-id filters describe one vendor's own catalog.

``PROVIDER_REGISTRY["openai"]["model_id_denylist"]`` encodes facts about
OpenAI's published models: which ids are embeddings or TTS, which are not on
``/v1/responses``. None of that holds for an Azure deployment or a self-hosted
OpenAI-compatible server, whose ids are names the operator chose. Applying the
denylist there empties the picker for a connection that works. Gemini already
scopes its allowlist to the native host; these pin the same rule for OpenAI.
"""

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
    # Deployment / local names that read as non-chat to the OpenAI denylist.
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
