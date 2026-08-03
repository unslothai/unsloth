# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for Studio-managed external-provider function calls."""

import asyncio
import copy
import json
import threading

import pytest

from core.inference import external_tool_loop as loop_mod


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}


def _chunk(*, delta=None, finish_reason=None, usage=None):
    payload = {
        "id": "chatcmpl-ollama",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
        payload["choices"] = []
    return "data: " + json.dumps(payload)


def _tool_round(arguments='{"query":"Cairo weather"}'):
    split = max(1, len(arguments) // 2)
    return [
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": arguments[:split],
                        },
                    }
                ]
            }
        ),
        _chunk(
            delta={
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": arguments[split:]},
                    }
                ]
            }
        ),
        _chunk(delta={}, finish_reason="tool_calls"),
        _chunk(usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}),
        "data: [DONE]",
    ]


class _FakeClient:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = []

    async def stream_chat_completion(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        for line in self.rounds.pop(0):
            yield line


async def _collect(client, **overrides):
    kwargs = {
        "client": client,
        "messages": [{"role": "user", "content": "What is the weather?"}],
        "model": "qwen3",
        "tools": [TOOL_SCHEMA],
        "request_kwargs": {},
    }
    kwargs.update(overrides)
    return [line async for line in loop_mod.stream_external_chat_with_tools(**kwargs)]


def test_ollama_tool_call_executes_and_continues(monkeypatch):
    executed = []

    def fake_execute(name, arguments, **kwargs):
        executed.append((name, arguments, kwargs))
        return "Cairo: 24 C and sunny"

    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)
    client = _FakeClient(
        [
            _tool_round(),
            [
                _chunk(delta={"content": "It is 24 C and sunny."}),
                _chunk(delta={}, finish_reason="stop"),
                "data: [DONE]",
            ],
        ]
    )

    output = asyncio.run(_collect(client))

    assert executed[0][0:2] == ("web_search", {"query": "Cairo weather"})
    assert len(client.calls) == 2
    assert client.calls[0]["tools"] == [TOOL_SCHEMA]
    continued_messages = client.calls[1]["messages"]
    assert continued_messages[-2]["role"] == "assistant"
    assert continued_messages[-2]["tool_calls"][0]["id"] == "call_weather"
    assert continued_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_weather",
        "name": "web_search",
        "content": "Cairo: 24 C and sunny",
    }
    assert any('"type": "tool_start"' in line for line in output)
    assert any('"type": "tool_end"' in line for line in output)
    assert any("It is 24 C and sunny." in line for line in output)
    assert sum(line == "data: [DONE]" for line in output) == 1
    assert not any('"tool_calls"' in line for line in output)
    assert not any('"usage"' in line and '"prompt_tokens": 12' in line for line in output)


def test_tool_failure_becomes_a_tool_result_and_does_not_break_stream(monkeypatch):
    def fake_execute(_name, _arguments, **_kwargs):
        raise RuntimeError("isolated failure")

    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)
    client = _FakeClient(
        [
            _tool_round(),
            [_chunk(delta={"content": "I could not run the search."}), "data: [DONE]"],
        ]
    )

    output = asyncio.run(_collect(client))

    tool_result = client.calls[1]["messages"][-1]
    assert tool_result["role"] == "tool"
    assert "Error: tool raised an exception: isolated failure" in tool_result["content"]
    assert any("I could not run the search." in line for line in output)


def test_budget_final_pass_cannot_loop_on_nonconforming_provider(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    client = _FakeClient([_tool_round(), _tool_round('{"query":"again"}')])

    output = asyncio.run(_collect(client, max_tool_iterations=1))

    assert len(client.calls) == 2
    assert client.calls[1]["tools"] is None
    assert client.calls[1]["tool_choice"] == "none"
    assert output.count("data: [DONE]") == 0


def test_provider_hosted_tools_survive_every_studio_round(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    client = _FakeClient(
        [
            _tool_round(),
            [_chunk(delta = {"content": "done"}), "data: [DONE]"],
        ]
    )

    asyncio.run(
        _collect(
            client,
            provider_enabled_tools = ["web_fetch", "image_generation"],
        )
    )

    assert [call["enabled_tools"] for call in client.calls] == [
        ["web_fetch", "image_generation"],
        ["web_fetch", "image_generation"],
    ]


def test_cancel_interrupts_a_silent_provider_stream():
    class BlockingClient:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.closed = asyncio.Event()

        async def stream_chat_completion(self, **_kwargs):
            self.entered.set()
            try:
                await self.release.wait()
                yield "data: [DONE]"
            finally:
                self.closed.set()

    async def run():
        client = BlockingClient()
        cancel = threading.Event()
        stream = loop_mod.stream_external_chat_with_tools(
            client = client,
            messages = [{"role": "user", "content": "wait"}],
            model = "silent-model",
            tools = [TOOL_SCHEMA],
            request_kwargs = {},
            cancel_event = cancel,
        )
        pending = asyncio.create_task(stream.__anext__())
        await asyncio.wait_for(client.entered.wait(), timeout = 0.5)
        cancel.set()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(pending, timeout = 0.5)
        await asyncio.wait_for(client.closed.wait(), timeout = 0.5)
        await stream.aclose()

    asyncio.run(run())

