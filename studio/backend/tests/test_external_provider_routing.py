# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Trust-boundary coverage for saved external-provider tool execution."""

import asyncio
from types import SimpleNamespace

import pytest

from models.inference import ChatCompletionRequest
from routes import inference as inference_mod


TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search locally.",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _configure(monkeypatch, selected):
    captured = {}
    monkeypatch.setattr(
        inference_mod.providers_db,
        "get_provider",
        lambda _provider_id: {
            "id": "saved",
            "display_name": "Saved provider",
            "provider_type": "custom",
            "base_url": "https://provider.example/v1",
            "is_enabled": 1,
            "studio_tool_execution": 1,
        },
    )

    async def select_tools(*_args, **_kwargs):
        return selected

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def stream_chat_completion(self, **kwargs):
            captured["plain"] = kwargs
            yield "data: [DONE]"

        async def close(self):
            pass

    async def managed_loop(**kwargs):
        captured["managed"] = kwargs
        yield "data: [DONE]"

    class Tracker:
        @classmethod
        def for_payload(cls, *_args, **kwargs):
            captured["tracker"] = kwargs
            return cls()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(inference_mod, "_select_request_tools", select_tools)
    monkeypatch.setattr(inference_mod, "ExternalProviderClient", Client)
    monkeypatch.setattr(inference_mod, "stream_external_chat_with_tools", managed_loop)
    monkeypatch.setattr(inference_mod, "_TrackedCancel", Tracker)
    return captured


def _payload(**kwargs):
    return ChatCompletionRequest(
        model = "default",
        external_model = "model",
        messages = [{"role": "user", "content": "hello"}],
        stream = kwargs.pop("stream", True),
        provider_id = "saved",
        provider_type = "ollama",
        provider_base_url = "https://attacker.invalid/v1",
        enable_tools = True,
        mcp_enabled = True,
        confirm_tool_calls = True,
        **kwargs,
    )


def _run(payload):
    request = SimpleNamespace(
        state = SimpleNamespace(skip_api_monitor = True),
        url = SimpleNamespace(path = "/v1/chat/completions"),
        method = "POST",
    )

    async def collect():
        response = await inference_mod._proxy_to_external_provider(payload, request)
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(collect())


@pytest.mark.parametrize(("selected", "path"), [([], "plain"), ([TOOL], "managed")])
def test_saved_opt_in_owns_routing_and_empty_selection_falls_through(monkeypatch, selected, path):
    captured = _configure(monkeypatch, selected)

    _run(_payload())

    assert captured["client"] == {
        "provider_type": "custom",
        "base_url": "https://provider.example/v1",
        "api_key": "",
    }
    assert path in captured
    if path == "managed":
        assert captured["managed"]["tools"] == [TOOL]
        assert captured["tracker"]["track_active_generation"] is False
        with pytest.raises(inference_mod.HTTPException):
            _run(_payload(stream = False))
