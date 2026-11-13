# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model listings keep the provider's own name for display; the id stays what requests send."""

import asyncio

from models.providers import ProviderModelsRequest
from routes import providers as providers_route


def _list(monkeypatch, provider_type: str, entries: list[dict]):
    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def list_models(self):
            return entries

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "sk-test"
    )
    payload = ProviderModelsRequest(provider_type = provider_type)
    return asyncio.new_event_loop().run_until_complete(
        providers_route.list_provider_models(payload, "tester", False)
    )


def test_anthropic_names_are_kept_and_ids_untouched(monkeypatch):
    result = _list(
        monkeypatch,
        "anthropic",
        [
            {"id": "claude-opus-4-5", "display_name": "Claude Opus 4.5", "type": "model"},
            {"id": "claude-haiku-4-5", "display_name": "  "},
            {"id": "claude-sonnet-4-5"},
        ],
    )
    assert [(m.id, m.display_name) for m in result] == [
        ("claude-opus-4-5", "Claude Opus 4.5"),
        ("claude-haiku-4-5", "claude-haiku-4-5"),
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
    ]
