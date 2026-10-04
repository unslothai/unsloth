# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from models.inference import ChatMessage
from routes.inference import _build_external_messages
from core.inference.external_provider import ExternalProviderClient
from core.inference.external_tool_transport import OAICompatTransport


@pytest.mark.parametrize("content", ["answer", "", None, [], [{"type": "text", "text": "answer"}]])
@pytest.mark.parametrize("vision", [True, False])
@pytest.mark.parametrize("with_tool", [True, False])
def test_llama_history_keeps_reasoning_and_tool_calls(content, vision, with_tool):
    calls = [
        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    ]
    message = ChatMessage(
        role = "assistant",
        content = content,
        reasoning_content = "prior thought",
        tool_calls = calls if with_tool else None,
    )
    result = _build_external_messages([message], vision, provider_type = "llama_cpp")
    assert len(result) == 1
    assert result[0]["reasoning_content"] == "prior thought"
    if with_tool:
        assert result[0]["tool_calls"][0]["function"]["name"] == "lookup"


@pytest.mark.parametrize("content", ["", None, []])
@pytest.mark.parametrize("vision", [True, False])
def test_llama_reasoning_survives_dropped_server_tool_cards(content, vision):
    card = {
        "id": "srv_1",
        "type": "function",
        "function": {"name": "web_search", "arguments": '{"_server_tool": true}'},
    }
    message = ChatMessage(
        role = "assistant", content = content, reasoning_content = "prior thought", tool_calls = [card]
    )
    result = _build_external_messages([message], vision, provider_type = "llama_cpp")
    assert len(result) == 1
    assert result[0]["reasoning_content"] == "prior thought"
    assert "tool_calls" not in result[0]


@pytest.mark.parametrize(
    "provider", ["custom", "openai", "vllm", "ollama", "anthropic", "openrouter"]
)
def test_other_providers_keep_existing_reasoning_policy(provider):
    message = ChatMessage(role = "assistant", content = "answer", reasoning_content = "prior thought")
    result = _build_external_messages([message], False, provider_type = provider)
    assert "reasoning_content" not in result[0]


@pytest.mark.parametrize("provider", ["llama_cpp", "custom", "vllm"])
@pytest.mark.parametrize("value", [True, False, None])
def test_tool_transport_reasoning_policy(provider, value):
    client = ExternalProviderClient(
        provider_type = provider, base_url = "http://localhost:8080/v1", api_key = ""
    )
    transport = OAICompatTransport(client, model = "test", preserve_thinking = value)
    assert transport.preserves_reasoning is (provider == "llama_cpp" and value is True)
