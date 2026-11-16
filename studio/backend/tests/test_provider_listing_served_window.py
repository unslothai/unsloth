# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A self-hosted listing carries its served window so the UI can default Max Tokens from it (#17444)."""

import asyncio

import pytest

from models.providers import ProviderModelsRequest
from routes import providers as providers_route

_VLLM = {"id": "Qwen/Qwen3-8B", "owned_by": "vllm", "max_model_len": 16384}
_LLAMA = {"id": "qwen3", "owned_by": "llamacpp", "meta": {"n_ctx": 32768, "n_ctx_train": 40960}}
_HOSTED = {"id": "x/y", "context_length": 131072}


def _listed(monkeypatch, provider_type, entries):
    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def list_models(self):
            return entries

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)
    monkeypatch.setattr(providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: "")
    monkeypatch.setattr(providers_route, "validate_provider_base_url", lambda url, *a, **k: url)
    payload = ProviderModelsRequest(provider_type = provider_type, base_url = "http://10.0.0.5:8000/v1")
    result = asyncio.new_event_loop().run_until_complete(
        providers_route.list_provider_models(payload, "owner", False)
    )
    return {m.id: (m.context_length, m.served_context_window) for m in result}


@pytest.mark.parametrize("provider_type", ["vllm", "llama_cpp", "custom"])
def test_a_self_hosted_listing_reports_the_served_window(monkeypatch, provider_type):
    assert _listed(monkeypatch, provider_type, [_VLLM, _LLAMA, _HOSTED]) == {
        "Qwen/Qwen3-8B": (None, 16384),
        "qwen3": (None, 32768),
        "x/y": (131072, None),
    }


def test_a_hosted_listing_reports_no_served_window(monkeypatch):
    assert _listed(monkeypatch, "openrouter", [_VLLM, _HOSTED]) == {
        "Qwen/Qwen3-8B": (None, None),
        "x/y": (131072, None),
    }
