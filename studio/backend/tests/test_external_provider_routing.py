# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trust-boundary tests for saved external-provider routing."""

import asyncio
from types import SimpleNamespace

import pytest

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
        parallel_tool_calls = False,
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
    assert captured["loop"]["parallel_tool_calls"] is False
    assert any(
        "[DONE]" in (chunk.decode() if isinstance(chunk, bytes) else chunk) for chunk in chunks
    )


def test_managed_studio_tools_reject_caller_defined_functions(monkeypatch):
    monitor_starts = []
    client_starts = []
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
            client_starts.append(kwargs)

    monkeypatch.setattr(inference_mod, "ExternalProviderClient", FakeClient)
    monkeypatch.setattr(
        inference_mod.api_monitor,
        "start",
        lambda **kwargs: monitor_starts.append(kwargs),
    )
    monkeypatch.setattr(inference_mod, "_request_used_api_key", lambda _request: False)
    request = SimpleNamespace(
        state = SimpleNamespace(skip_api_monitor = False),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "qwen3",
        messages = [{"role": "user", "content": "search"}],
        stream = True,
        provider_id = "saved-provider",
        enable_tools = True,
        enabled_tools = ["web_search"],
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "caller_function",
                    "description": "Handled by the API caller.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )

    with pytest.raises(inference_mod.HTTPException) as caught:
        asyncio.run(inference_mod._proxy_to_external_provider(payload, request))

    assert caught.value.status_code == 400
    assert caught.value.detail["error"]["param"] == "tools"
    assert monitor_starts == []
    assert client_starts == []


def test_hosted_only_tools_reject_confirmation_without_managed_loop(monkeypatch):
    monkeypatch.setattr(
        inference_mod.providers_db,
        "get_provider",
        lambda _provider_id: {
            "id": "saved-provider",
            "display_name": "Saved OpenAI",
            "provider_type": "openai",
            "base_url": "https://api.openai.com/v1",
            "is_enabled": 1,
            "studio_tool_execution": 1,
        },
    )

    async def select_tools(*_args, **_kwargs):
        return []

    monkeypatch.setattr(inference_mod, "_select_request_tools", select_tools)
    monkeypatch.setattr(
        inference_mod,
        "ExternalProviderClient",
        lambda **_kwargs: SimpleNamespace(),
    )
    request = SimpleNamespace(
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "gpt-5",
        messages = [{"role": "user", "content": "draw"}],
        stream = True,
        provider_id = "saved-provider",
        enabled_tools = ["image_generation"],
        confirm_tool_calls = True,
    )

    with pytest.raises(inference_mod.HTTPException) as caught:
        asyncio.run(inference_mod._proxy_to_external_provider(payload, request))

    assert caught.value.status_code == 400
    assert caught.value.detail["error"]["param"] == "confirm_tool_calls"


def test_empty_mcp_selection_falls_through_to_plain_provider_proxy(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        inference_mod.providers_db,
        "get_provider",
        lambda _provider_id: {
            "id": "saved-provider",
            "display_name": "Saved OpenAI-compatible provider",
            "provider_type": "custom",
            "base_url": "https://provider.example/v1",
            "is_enabled": 1,
            "studio_tool_execution": 1,
        },
    )

    async def select_tools(*_args, **_kwargs):
        return []

    monkeypatch.setattr(inference_mod, "_select_request_tools", select_tools)

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def stream_chat_completion(self, **kwargs):
            captured["provider_call"] = kwargs
            yield "data: [DONE]"

        async def close(self):
            return None

    monkeypatch.setattr(inference_mod, "ExternalProviderClient", FakeClient)
    request = SimpleNamespace(
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "compatible-model",
        messages = [{"role": "user", "content": "hello"}],
        stream = True,
        provider_id = "saved-provider",
        enable_tools = True,
        mcp_enabled = True,
        confirm_tool_calls = True,
    )

    async def run():
        response = await inference_mod._proxy_to_external_provider(payload, request)
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(run())

    assert captured["provider_call"]["tools"] is None
    assert captured["provider_call"]["enabled_tools"] is None
    assert any(
        "[DONE]" in (chunk.decode() if isinstance(chunk, bytes) else chunk) for chunk in chunks
    )


def test_cancelled_managed_loop_marks_monitor_cancelled_without_done(monkeypatch):
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
        def __init__(self, **_kwargs):
            pass

        async def stream_chat_completion(self, **_kwargs):
            yield "data: [DONE]"

        async def close(self):
            return None

    monkeypatch.setattr(inference_mod, "ExternalProviderClient", FakeClient)

    async def cancelled_loop(**kwargs):
        kwargs["cancel_event"].set()
        if False:
            yield ""

    monkeypatch.setattr(inference_mod, "stream_external_chat_with_tools", cancelled_loop)

    class FakeTracker:
        @classmethod
        def for_payload(cls, *_args, **_kwargs):
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(inference_mod, "_TrackedCancel", FakeTracker)
    monkeypatch.setattr(inference_mod, "_request_used_api_key", lambda _request: False)
    monitor_finishes = []
    monkeypatch.setattr(inference_mod.api_monitor, "start", lambda **_kwargs: "monitor-id")
    monkeypatch.setattr(
        inference_mod.api_monitor,
        "finish",
        lambda monitor_id, state = "completed": monitor_finishes.append((monitor_id, state)),
    )
    request = SimpleNamespace(
        state = SimpleNamespace(skip_api_monitor = False),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
    )
    payload = ChatCompletionRequest(
        model = "default",
        external_model = "qwen3",
        messages = [{"role": "user", "content": "search"}],
        stream = True,
        provider_id = "saved-provider",
        enable_tools = True,
        enabled_tools = ["web_search"],
    )

    async def run():
        response = await inference_mod._proxy_to_external_provider(payload, request)
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(run())

    assert chunks == []
    assert monitor_finishes == [("monitor-id", "cancelled")]
