# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Kimi K3 over the direct Moonshot API (#8735): it reaches the picker, and requests
follow https://platform.kimi.ai/docs/guide/kimi-k3-quickstart while k2.x keep their shape."""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.providers import PROVIDER_REGISTRY


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    return [line async for line in agen]


def _capture(monkeypatch, responses):
    bodies: list = []
    replies = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200, content = next(replies), headers = {"content-type": "text/event-stream"}
        )

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )
    return bodies


_PLAIN = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


def _stream(model, **kwargs):
    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi", base_url = "http://kimi.example/v1", api_key = "sk-test"
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}], model = model, **kwargs
            )
        )
        await client.close()

    _drive(run())


def _list_kimi_models(monkeypatch, ids):
    from models.providers import ProviderModelsRequest
    from routes import providers as providers_route

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
    payload = ProviderModelsRequest(provider_type = "kimi", base_url = "https://api.moonshot.ai/v1")
    result = _drive(providers_route.list_provider_models(payload, "tester", False))
    return [m.id for m in result]


_MOONSHOT_LIVE_IDS = [
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.7-code-highspeed",
    "kimi-k2.6",
    "kimi-k3.1-pro",
]


def test_kimi_k3_reaches_the_picker():
    from core.inference.providers import provider_default_models
    listed = [*_MOONSHOT_LIVE_IDS, "kimi-k2-thinking", "moonshot-v1-8k"]
    assert provider_default_models("kimi", {"kimi": listed}) == _MOONSHOT_LIVE_IDS


def test_every_current_kimi_model_reaches_the_picker(monkeypatch):
    assert _list_kimi_models(monkeypatch, _MOONSHOT_LIVE_IDS) == _MOONSHOT_LIVE_IDS


def test_kimi_previews_and_legacy_models_stay_out_of_the_picker(monkeypatch):
    ids = [
        "kimi-k3-preview",
        "kimi-k2-0905-preview",
        "kimi-k2-turbo-preview",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
        "moonshot-v1-8k",
    ]
    assert _list_kimi_models(monkeypatch, ids) == []


@pytest.mark.parametrize(
    ("effort", "sent"),
    [("low", "low"), ("high", "high"), ("max", "max"), ("medium", "high"), ("xhigh", "max")],
)
@pytest.mark.parametrize("model", ["kimi-k3", "kimi-k3.1-pro"])
def test_kimi_k3_sends_its_documented_controls(monkeypatch, model, effort, sent):
    bodies = _capture(monkeypatch, [_PLAIN])
    _stream(
        model,
        temperature = 0.7,
        top_p = 0.9,
        presence_penalty = 0.0,
        max_tokens = 64,
        enable_thinking = False,
        reasoning_effort = effort,
    )
    body = bodies[0]
    assert body["reasoning_effort"] == sent
    assert body["max_completion_tokens"] == 64
    for field in ("thinking", "max_tokens", "presence_penalty", "temperature", "top_p"):
        assert field not in body, field


def test_kimi_k3_without_an_effort_leaves_the_server_default(monkeypatch):
    bodies = _capture(monkeypatch, [_PLAIN])
    _stream("kimi-k3", max_tokens = 64, reasoning_effort = "none")
    assert "reasoning_effort" not in bodies[0]
    assert "thinking" not in bodies[0]


def test_kimi_k2_6_request_is_unchanged(monkeypatch):
    bodies = _capture(monkeypatch, [_PLAIN])
    _stream(
        "kimi-k2.6",
        presence_penalty = 0.5,
        max_tokens = 64,
        enable_thinking = False,
        reasoning_effort = "high",
    )
    body = bodies[0]
    assert body["thinking"] == {"type": "disabled"}
    assert body["max_tokens"] == 64
    assert body["presence_penalty"] == 0.5
    assert "reasoning_effort" not in body
    assert "max_completion_tokens" not in body


_SEARCH_CALL = (
    b'data: {"choices":[{"delta":{"reasoning_content":"look it up"}}]}\n\n'
    b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"builtin_function",'
    b'"function":{"name":"$web_search","arguments":"{\\"q\\":1}"}}]},"finish_reason":"tool_calls"}]}\n\n'
    b"data: [DONE]\n\n"
)


def test_kimi_k3_web_search_keeps_reasoning_and_drops_thinking(monkeypatch):
    bodies = _capture(monkeypatch, [_SEARCH_CALL, _PLAIN])
    _stream("kimi-k3", max_tokens = 64, reasoning_effort = "low", enabled_tools = ["web_search"])
    first, followup = bodies
    for body in (first, followup):
        assert "thinking" not in body
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens"] == 64
    replayed = followup["messages"][1]
    assert replayed["role"] == "assistant"
    assert replayed["reasoning_content"] == "look it up"


def test_kimi_k2_6_web_search_still_turns_thinking_off(monkeypatch):
    bodies = _capture(monkeypatch, [_SEARCH_CALL, _PLAIN])
    _stream("kimi-k2.6", max_tokens = 64, enabled_tools = ["web_search"])
    first = bodies[0]
    assert first["thinking"] == {"type": "disabled"}
    assert first["max_tokens"] == 64
    assert "reasoning_effort" not in first


def test_kimi_k3_history_keeps_prior_reasoning():
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    calls = [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    history = [
        ChatMessage(role = "user", content = "hi"),
        ChatMessage(role = "assistant", content = "", reasoning_content = "plan", tool_calls = calls),
        ChatMessage(role = "tool", content = "42", tool_call_id = "call_1"),
        ChatMessage(role = "assistant", content = "answer", reasoning_content = "done"),
    ]
    k3 = _build_external_messages(history, True, provider_type = "kimi", model = "kimi-k3")
    assert [m.get("reasoning_content") for m in k3 if m["role"] == "assistant"] == ["plan", "done"]
    k26 = _build_external_messages(history, True, provider_type = "kimi", model = "kimi-k2.6")
    assert all("reasoning_content" not in m for m in k26)


@pytest.mark.parametrize(
    ("model", "keeps"), [("kimi-k3", True), ("kimi-k3.1-pro", True), ("kimi-k2.6", False)]
)
def test_studio_tool_loop_replays_kimi_k3_reasoning(model, keeps):
    from core.inference.external_tool_transport import OAICompatTransport
    client = ExternalProviderClient(
        provider_type = "kimi", base_url = "http://kimi.example/v1", api_key = "sk-test"
    )
    assert OAICompatTransport(client, model = model).preserves_reasoning is keeps
