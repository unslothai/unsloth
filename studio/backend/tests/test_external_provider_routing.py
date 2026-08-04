# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trust-boundary tests for saved external-provider routing."""

import asyncio
from types import SimpleNamespace

from models.inference import ChatCompletionRequest
from routes import inference as inference_mod


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search locally.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def test_saved_provider_owns_routing_and_keeps_unrelated_hosted_tools(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        inference_mod.providers_db,
        "get_provider",
        lambda _provider_id: {
            "id": "saved-provider",
            "display_name": "Saved Ollama",
            "provider_type": "ollama",
            "base_url": "http://127.0.0.1:11434/v1",
            "is_enabled": 1,
            "studio_tool_execution": 1,
        },
    )

    async def select_tools(*_args, **_kwargs):
        return [TOOL_SCHEMA]

    monkeypatch.setattr(inference_mod, "_select_request_tools", select_tools)

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def stream_chat_completion(self, **_kwargs):
            yield "data: [DONE]"

        async def close(self):
            return None

    monkeypatch.setattr(inference_mod, "ExternalProviderClient", FakeClient)

    async def fake_tool_loop(**kwargs):
        captured["loop"] = kwargs
        yield "data: [DONE]"

    monkeypatch.setattr(
        inference_mod,
        "stream_external_chat_with_tools",
        fake_tool_loop,
    )

    class FakeTracker:
        @classmethod
        def for_payload(cls, *_args, **_kwargs):
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(inference_mod, "_TrackedCancel", FakeTracker)

    request = SimpleNamespace(
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "qwen3",
        messages = [{"role": "user", "content": "search"}],
        stream = True,
        provider_id = "saved-provider",
        # These conflicting fields must never retarget the saved opt-in.
        provider_type = "custom",
        provider_base_url = "https://attacker.invalid/v1",
        enable_tools = True,
        enabled_tools = ["web_search", "web_fetch", "image_generation"],
    )

    async def run():
        response = await inference_mod._proxy_to_external_provider(payload, request)
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(run())

    assert captured["client"] == {
        "provider_type": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "",
    }
    assert captured["loop"]["tools"] == [TOOL_SCHEMA]
    assert captured["loop"]["provider_enabled_tools"] == ["web_fetch", "image_generation"]
    assert any(
        "[DONE]" in (chunk.decode() if isinstance(chunk, bytes) else chunk) for chunk in chunks
    )
