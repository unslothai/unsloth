# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Critical contracts for Studio-managed external-provider tool rounds."""

import asyncio
import copy
import json

import pytest

from core.inference import external_tool_loop as loop_mod


TOOL = {
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
    finish = None,
    usage = None,
):
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    if usage is not None:
        payload["choices"] = []
        payload["usage"] = usage
    return "data: " + json.dumps(payload)


def _tool_round(
    name = "web_search",
    call_id = "call_1",
    arguments = '{"query":"Cairo"}',
    *,
    extra = None,
    usage = None,
):
    call = {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }
    if extra:
        call["extra_content"] = extra
    lines = [
        _chunk(delta = {"tool_calls": [call]}),
        _chunk(delta = {}, finish = "tool_calls"),
    ]
    if usage:
        lines.append(_chunk(usage = usage))
    return [*lines, "data: [DONE]"]


class _Client:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = []

    async def stream_chat_completion(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        current_round = self.rounds.pop(0)
        for line in current_round:
            yield line


async def _collect(client, **overrides):
    kwargs = {
        "client": client,
        "messages": [{"role": "user", "content": "Help"}],
        "model": "test-model",
        "tools": [TOOL],
        "request_kwargs": {},
    }
    kwargs.update(overrides)
    return [line async for line in loop_mod.stream_external_chat_with_tools(**kwargs)]


def _payloads(lines):
    return [
        json.loads(line[len("data:") :])
        for line in lines
        if line.startswith("data:") and line[len("data:") :].strip() != "[DONE]"
    ]


def test_executes_tool_continues_and_aggregates_usage(monkeypatch):
    executed = []
    monkeypatch.setattr(loop_mod, "wait_tool_decision", lambda *_args: "approve")
    monkeypatch.setattr(
        loop_mod,
        "execute_tool",
        lambda name, arguments, **_kwargs: executed.append((name, arguments)) or "sunny",
    )
    client = _Client(
        [
            _tool_round(usage = {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}),
            [
                _chunk(delta = {"content": "24 C"}, finish = "stop"),
                _chunk(usage = {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}),
                "data: [DONE]",
            ],
        ]
    )

    output = asyncio.run(_collect(client, confirm_tool_calls = True))

    assert executed == [("web_search", {"query": "Cairo"})]
    assert [message["role"] for message in client.calls[1]["messages"][-2:]] == [
        "assistant",
        "tool",
    ]
    assert sum('"type": "tool_start"' in line for line in output) == 1
    assert any('"awaiting_confirmation": false' in line for line in output)
    assert sum('"type": "tool_end"' in line for line in output) == 1
    usage = [payload["usage"] for payload in _payloads(output) if payload.get("usage")]
    assert usage == [{"prompt_tokens": 19, "completion_tokens": 7, "total_tokens": 26}]


@pytest.mark.parametrize("terminal", ["eof", "done", "length"])
def test_tool_round_requires_an_explicit_tool_finish(monkeypatch, terminal):
    executed = []
    monkeypatch.setattr(
        loop_mod,
        "execute_tool",
        lambda *args, **_kwargs: executed.append(args),
    )
    call = _chunk(
        delta = {
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_partial",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"query":"Cairo"}'},
                }
            ]
        },
        finish = "length" if terminal == "length" else None,
    )
    stream = [call] + (["data: [DONE]"] if terminal != "eof" else [])

    with pytest.raises(RuntimeError):
        asyncio.run(_collect(_Client([stream])))
    assert executed == []


def test_dropped_parallel_call_unlinks_openai_response_and_keeps_gemini_parts(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    native = {"executableCode": {"language": "PYTHON", "code": "print(2)"}}
    reasoning = [{"type": "reasoning.encrypted", "data": "opaque"}]
    event = {
        "choices": [],
        "_toolEvent": {
            "type": "tool_start",
            "arguments": {"google": {"native_part": {"parts": [native]}}},
        },
    }
    linked = {"openai": {"previous_response_id": "resp_parallel"}}
    calls = [
        {
            "index": index,
            "id": f"call_{index}",
            "type": "function",
            "function": {"name": "web_search", "arguments": json.dumps({"query": index})},
            "extra_content": copy.deepcopy(linked),
        }
        for index in range(2)
    ]
    client = _Client(
        [
            [
                "data: " + json.dumps(event),
                _chunk(
                    delta = {
                        "tool_calls": calls,
                        "reasoning_details": reasoning,
                        "reasoning_content": "think",
                    }
                ),
                _chunk(delta = {}, finish = "tool_calls"),
                "data: [DONE]",
            ],
            [_chunk(delta = {"content": "done"}), "data: [DONE]"],
        ]
    )

    asyncio.run(_collect(client, parallel_tool_calls = False))

    assert client.calls[0]["parallel_tool_calls"] is False
    assistant = client.calls[1]["messages"][-2]
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_0"]
    assert "extra_content" not in assistant["tool_calls"][0]
    assert assistant["extra_content"]["google"]["hosted_parts"] == [native]
    assert assistant["reasoning_details"] == reasoning
    assert assistant["reasoning_content"] == "think"


def test_usage_precedes_provider_error(monkeypatch):
    monkeypatch.setattr(loop_mod, "execute_tool", lambda *_args, **_kwargs: "ok")
    error = "data: " + json.dumps({"error": {"message": "upstream failed"}})
    client = _Client(
        [
            _tool_round(usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}),
            [error],
        ]
    )

    output = asyncio.run(_collect(client))

    usage_index = next(i for i, payload in enumerate(_payloads(output)) if payload.get("usage"))
    error_index = next(i for i, payload in enumerate(_payloads(output)) if payload.get("error"))
    assert usage_index < error_index


def test_knowledge_base_searches_are_capped(monkeypatch):
    executed = []
    monkeypatch.setattr(
        loop_mod,
        "execute_tool",
        lambda name, arguments, **_kwargs: executed.append((name, arguments)) or "result",
    )
    kb_tool = copy.deepcopy(TOOL)
    kb_tool["function"]["name"] = "search_knowledge_base"
    client = _Client(
        [
            _tool_round(
                "search_knowledge_base",
                f"call_{index}",
                json.dumps({"query": f"paraphrase {index}"}),
            )
            for index in range(loop_mod.RAG_MAX_SEARCHES_PER_TURN + 1)
        ]
        + [[_chunk(delta = {"content": "done"}), "data: [DONE]"]]
    )

    asyncio.run(_collect(client, tools = [kb_tool], rag_scope = "scope"))

    assert len(executed) == loop_mod.RAG_MAX_SEARCHES_PER_TURN
    assert client.calls[-1]["messages"][-1]["content"] == loop_mod.RAG_SEARCH_CAP_NUDGE
