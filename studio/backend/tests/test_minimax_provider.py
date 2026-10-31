# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep
from core.inference.providers import get_base_url, get_provider_info, list_available_providers


def _capture(monkeypatch, model, **kwargs) -> dict:
    captured: dict = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, headers = {"content-type": "text/event-stream"}, content = "data: [DONE]\n\n"
        )

    async def run():
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as http:
            monkeypatch.setattr(ep, "_client", lambda: http)
            client = ep.ExternalProviderClient("minimax", get_base_url("minimax"), "sk-test")
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = model,
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 256,
                **kwargs,
            ):
                pass

    asyncio.run(run())
    return captured


def test_minimax_listed():
    assert "minimax" in {p["provider_type"] for p in list_available_providers()}


def test_minimax_lists_chat_models_live():
    info = get_provider_info("minimax")
    assert info.get("model_list_mode") != "curated"
    chat = ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    retired = ["MiniMax-M2.7", "MiniMax-M2.5-highspeed", "MiniMax-M2"]
    listed = chat + retired + ["speech-2.8-hd", "MiniMax-Hailuo-2.3", "image-01"]
    assert [m for m in listed if info["model_id_allowlist"].match(m)] == chat
    assert info["default_models"] == chat


def test_minimax_body_shape(monkeypatch):
    cap = _capture(monkeypatch, "MiniMax-M3")
    body = cap["body"]
    assert cap["url"] == "https://api.minimax.io/v1/chat/completions"
    assert body["max_completion_tokens"] == 256
    assert "max_tokens" not in body
    assert body["reasoning_split"] is True
    for field in ("temperature", "top_p", "presence_penalty", "thinking"):
        assert field not in body


def test_minimax_m3_thinking_off(monkeypatch):
    cap = _capture(monkeypatch, "MiniMax-M3", enable_thinking = False)
    assert cap["body"]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize(("effort", "sent"), [("xhigh", "xhigh"), ("none", None)])
def test_minimax_m31_flash_effort_and_never_disabled(monkeypatch, effort, sent):
    cap = _capture(
        monkeypatch, "MiniMax-M3.1-Flash", enable_thinking = False, reasoning_effort = effort
    )
    assert cap["body"].get("reasoning_effort") == sent
    assert "thinking" not in cap["body"]


def test_minimax_history_keeps_reasoning():
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
    out = _build_external_messages(history, True, provider_type = "minimax", model = "MiniMax-M3")
    replayed = [m.get("reasoning_content") for m in out if m["role"] == "assistant"]
    assert replayed == ["plan", "done"]


def test_studio_tool_loop_replays_minimax_reasoning():
    from core.inference.external_tool_transport import OAICompatTransport
    client = ep.ExternalProviderClient("minimax", get_base_url("minimax"), "sk-test")
    assert OAICompatTransport(client, model = "MiniMax-M3").preserves_reasoning is True
