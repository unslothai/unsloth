# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import threading

import httpx
import pytest

from models.inference import ChatMessage
from routes.inference import _build_external_messages
from core.inference import external_provider as ep_mod
from core.inference import studio_tool_loop as loop_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.external_tool_transport import OAICompatTransport
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools


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
    assert transport.preserves_reasoning is (provider in ("llama_cpp", "vllm"))


def _sse(delta = None, finish = None) -> bytes:
    choice = {"index": 0, "delta": delta or {}}
    if finish:
        choice["finish_reason"] = finish
    return f"data: {json.dumps({'choices': [choice]})}\n\n".encode()


@pytest.mark.parametrize("provider", ["llama_cpp", "vllm", "custom"])
@pytest.mark.parametrize("preserve", [True, False, None])
def test_tool_round_thinking_reaches_the_next_request(monkeypatch, provider, preserve):
    # The template renders this turn's earlier tool rounds inside <think> whatever preserve_thinking says.
    call = {
        "index": 0,
        "id": "c1",
        "type": "function",
        "function": {"name": "web_search", "arguments": '{"query":"x"}'},
    }
    turns = [
        _sse({"reasoning_content": "Search x first, then y."})
        + _sse({"tool_calls": [call]})
        + _sse(finish = "tool_calls"),
        _sse({"content": "done"}) + _sse(finish = "stop"),
    ]
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            content = turns[len(bodies) - 1] + b"data: [DONE]\n\n",
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )
    monkeypatch.setattr(loop_mod, "execute_tool", lambda name, arguments, **kwargs: "result x")
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    client = ExternalProviderClient(
        provider_type = provider, base_url = "http://self-hosted.example/v1", api_key = ""
    )
    transport = OAICompatTransport(client, model = "qwen3.8", preserve_thinking = preserve)
    tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
    }

    async def run():
        async for _ in stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "compare x and y"}],
                session_id = "s1",
                thread_id = "t1",
                tool_choice = None,
                supports_vision = False,
            ),
            policy = ToolLoopPolicy(
                tools = [tool],
                max_calls = 25,
                timeout = 300,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = threading.Event(),
        ):
            pass

    asyncio.run(run())
    assert len(bodies) == 2
    [replayed] = [m for m in bodies[1]["messages"] if m.get("role") == "assistant"]
    assert replayed["tool_calls"][0]["function"]["name"] == "web_search"
    expected = None if provider == "custom" else "Search x first, then y."
    assert replayed.get("reasoning_content") == expected
    if provider == "vllm":
        assert replayed["reasoning"] == expected
