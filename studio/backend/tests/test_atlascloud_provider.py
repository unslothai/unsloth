# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused coverage for the Atlas Cloud provider preset."""

import pytest

import core.inference.external_provider as external_provider
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import get_provider_info, list_available_providers


def test_atlascloud_registry_entry():
    info = get_provider_info("atlascloud")

    assert info is not None
    assert info["base_url"] == "https://api.atlascloud.ai/v1"
    assert info["default_models"] == ["deepseek-ai/deepseek-v4-pro"]
    assert info["model_list_mode"] == "curated"
    assert any(entry["provider_type"] == "atlascloud" for entry in list_available_providers())


@pytest.mark.asyncio
async def test_atlascloud_chat_uses_openai_compatible_endpoint(monkeypatch):
    captured: dict[str, object] = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}]}

    class HTTPClient:
        async def post(self, url, *, json, headers, timeout):
            captured.update(
                {
                    "url": url,
                    "json": json,
                    "headers": headers,
                    "timeout": timeout,
                }
            )
            return Response()

    monkeypatch.setattr(external_provider, "_http_client", HTTPClient())
    client = ExternalProviderClient(
        provider_type = "atlascloud",
        base_url = "https://api.atlascloud.ai/v1",
        api_key = "test-key",
    )

    response = await client.chat_completion(
        messages = [{"role": "user", "content": "ping"}],
        model = "deepseek-ai/deepseek-v4-pro",
        max_tokens = 8,
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert captured["url"] == "https://api.atlascloud.ai/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "deepseek-ai/deepseek-v4-pro"
