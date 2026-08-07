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


def _chunk(
    *,
    delta = None,
    finish_reason = None,
    usage = None,
):
    payload = {
        "id": "chatcmpl-ollama",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage is not None:
        payload["usage"] = usage
        payload["choices"] = []
    return "data: " + json.dumps(payload)


def _tool_round(arguments = '{"query":"Cairo weather"}'):
    split = max(1, len(arguments) // 2)
    return [
        _chunk(
            delta = {
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
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": arguments[split:]},
                    }
                ]
            }
        ),
        _chunk(delta = {}, finish_reason = "tool_calls"),
        _chunk(usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}),
        "data: [DONE]",
    ]


def _named_tool_round(name: str, call_id: str, arguments: str) -> list[str]:
    return [
        _chunk(
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                ]
            }
        ),
        _chunk(delta = {}, finish_reason = "tool_calls"),
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
                _chunk(delta = {"content": "It is 24 C and sunny."}),
                _chunk(delta = {}, finish_reason = "stop"),
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
    usage_payloads = [
        json.loads(line[len("data:") :])
        for line in output
        if line.startswith("data:")
        and line[len("data:") :].strip() != "[DONE]"
        and json.loads(line[len("data:") :]).get("usage")
    ]
    assert [payload["usage"] for payload in usage_payloads] == [
        {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    ]


def test_tool_failure_becomes_a_tool_result_and_does_not_break_stream(monkeypatch):
    def fake_execute(_name, _arguments, **_kwargs):
        raise RuntimeError("isolated failure")

    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)
    client = _FakeClient(
        [
            _tool_round(),
            [_chunk(delta = {"content": "I could not run the search."}), "data: [DONE]"],
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

    output = asyncio.run(_collect(client, max_tool_iterations = 1))

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


def test_provider_tool_events_survive_managed_round(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    hosted_event = {
        "id": "chatcmpl-hosted",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "_toolEvent": {
            "type": "tool_end",
            "tool_name": "image_generation",
            "tool_call_id": "image_1",
            "image_b64": "encoded-image",
        },
    }
    first_round = [
        "data: " + json.dumps(hosted_event),
        *_tool_round(),
    ]
    client = _FakeClient([first_round, [_chunk(delta = {"content": "done"}), "data: [DONE]"]])

    output = asyncio.run(_collect(client))

    forwarded = [json.loads(line[len("data:") :]) for line in output if "_toolEvent" in line]
    assert len(forwarded) == 1
    assert forwarded[0]["_toolEvent"] == hosted_event["_toolEvent"]
    assert forwarded[0]["choices"] == []


def test_usage_is_aggregated_across_provider_rounds(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    client = _FakeClient(
        [
            _tool_round(),
            [
                _chunk(delta = {"content": "done"}),
                _chunk(
                    usage = {
                        "prompt_tokens": 7,
                        "completion_tokens": 4,
                        "total_tokens": 11,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                    }
                ),
                "data: [DONE]",
            ],
        ]
    )

    output = asyncio.run(_collect(client))

    usage_payloads = [
        json.loads(line[len("data:") :])
        for line in output
        if line.startswith("data:")
        and line[len("data:") :].strip() != "[DONE]"
        and json.loads(line[len("data:") :]).get("usage")
    ]
    assert len(usage_payloads) == 1
    usage = usage_payloads[0]["usage"]
    assert usage["prompt_tokens"] == 19
    assert usage["completion_tokens"] == 7
    assert usage["total_tokens"] == 26
    assert usage["completion_tokens_details"]["reasoning_tokens"] == 2


def test_gemini_thought_signature_survives_tool_continuation(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    first_round = _tool_round()
    first_payload = json.loads(first_round[0][len("data:") :])
    first_payload["choices"][0]["delta"]["tool_calls"][0]["extra_content"] = {
        "google": {"thought_signature": "SIG-ABC"}
    }
    first_round[0] = "data: " + json.dumps(first_payload)
    client = _FakeClient(
        [
            first_round,
            [_chunk(delta = {"content": "done"}), "data: [DONE]"],
        ]
    )

    asyncio.run(_collect(client))

    assistant = client.calls[1]["messages"][-2]
    assert assistant["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "SIG-ABC"}
    }


def test_anthropic_thinking_metadata_is_private_and_replayed(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    thinking_block = {
        "type": "thinking",
        "thinking": "I should search.",
        "signature": "signed-thinking",
    }
    tool_round = [
        _chunk(
            delta = {
                "content": "<think>I should search.",
                "extra_content": {"anthropic": {"thinking_display": True}},
            }
        ),
        _chunk(
            delta = {
                "content": "</think>",
                "extra_content": {"anthropic": {"thinking_display": True}},
            }
        ),
        _chunk(
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "toolu_search",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"Cairo weather"}',
                        },
                        "extra_content": {"anthropic": {"thinking_blocks": [thinking_block]}},
                    }
                ]
            }
        ),
        _chunk(delta = {}, finish_reason = "tool_calls"),
        "data: [DONE]",
    ]
    client = _FakeClient(
        [
            tool_round,
            [_chunk(delta = {"content": "done"}), "data: [DONE]"],
        ]
    )

    output = asyncio.run(_collect(client))

    assistant = client.calls[1]["messages"][-2]
    assert assistant["content"] == ""
    assert assistant["extra_content"]["anthropic"]["thinking_blocks"] == [thinking_block]
    assert any("<think>I should search." in line for line in output)
    assert not any("signed-thinking" in line for line in output)


def test_openai_reasoning_item_survives_tool_continuation(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    reasoning_item = {
        "type": "reasoning",
        "id": "rs_abc",
        "status": "completed",
        "summary": [{"type": "summary_text", "text": "Use the tool."}],
    }
    tool_round = [
        _chunk(
            delta = {
                "content": "<think>Use the tool.</think>",
                "extra_content": {"openai": {"reasoning_display": True}},
            }
        ),
        _chunk(
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_search",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"Cairo"}',
                        },
                        "extra_content": {"openai": {"reasoning_items": [reasoning_item]}},
                    }
                ]
            }
        ),
        _chunk(delta = {}, finish_reason = "tool_calls"),
        "data: [DONE]",
    ]
    client = _FakeClient([tool_round, [_chunk(delta = {"content": "done"}), "data: [DONE]"]])

    output = asyncio.run(_collect(client))

    assistant = client.calls[1]["messages"][-2]
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["extra_content"] == {
        "openai": {"reasoning_items": [reasoning_item]}
    }
    assert any("<think>Use the tool.</think>" in line for line in output)


def test_parallel_tool_calls_false_executes_only_the_first_call(monkeypatch):
    executed: list[str] = []

    def fake_execute(name, _arguments, **_kwargs):
        executed.append(name)
        return "ok"

    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)
    parallel_round = [
        _chunk(
            delta = {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_first",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"first"}',
                        },
                    },
                    {
                        "index": 1,
                        "id": "call_second",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"query":"second"}',
                        },
                    },
                ]
            }
        ),
        _chunk(delta = {}, finish_reason = "tool_calls"),
        "data: [DONE]",
    ]
    client = _FakeClient([parallel_round, [_chunk(delta = {"content": "done"}), "data: [DONE]"]])

    asyncio.run(_collect(client, parallel_tool_calls = False))

    assert executed == ["web_search"]
    assistant = client.calls[1]["messages"][-2]
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_first"]


def test_noop_round_does_not_consume_execution_budget(monkeypatch):
    executed: list[str] = []

    def fake_execute(name, _arguments, **_kwargs):
        executed.append(name)
        return "ok"

    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)
    client = _FakeClient(
        [
            _named_tool_round("web_search", "call_first", '{"query":"Cairo"}'),
            _named_tool_round("web_search", "call_duplicate", '{"query":"Cairo"}'),
            _named_tool_round(
                "web_search",
                "call_second",
                '{"query":"Alexandria"}',
            ),
            [_chunk(delta = {"content": "done"}), "data: [DONE]"],
        ]
    )

    asyncio.run(_collect(client, max_tool_iterations = 2))

    assert executed == ["web_search", "web_search"]
    assert len(client.calls) == 4
    assert client.calls[2]["tools"] == [TOOL_SCHEMA]
    assert client.calls[3]["tools"] is None


def test_denied_round_does_not_consume_execution_budget(monkeypatch):
    executed: list[str] = []
    approvals = iter(["deny", "allow"])

    def fake_execute(name, _arguments, **_kwargs):
        executed.append(name)
        return "ok"

    monkeypatch.setattr(loop_mod, "execute_tool", fake_execute)
    monkeypatch.setattr(loop_mod, "new_approval_id", lambda: "approval")
    monkeypatch.setattr(loop_mod, "begin_tool_decision", lambda *_args: object())
    monkeypatch.setattr(loop_mod, "wait_tool_decision", lambda *_args: next(approvals))
    client = _FakeClient(
        [
            _named_tool_round("web_search", "call_denied", '{"query":"Cairo"}'),
            _named_tool_round(
                "web_search",
                "call_allowed",
                '{"query":"Alexandria"}',
            ),
            [_chunk(delta = {"content": "done"}), "data: [DONE]"],
        ]
    )

    asyncio.run(
        _collect(
            client,
            max_tool_iterations = 1,
            confirm_tool_calls = True,
        )
    )

    assert executed == ["web_search"]
    assert len(client.calls) == 3
    assert client.calls[1]["tools"] == [TOOL_SCHEMA]
    assert client.calls[2]["tools"] is None


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
