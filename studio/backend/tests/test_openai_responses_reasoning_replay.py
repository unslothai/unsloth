# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reasoning items have to survive a local tool hop on /v1/responses.

OpenAI's function-calling guide is explicit for manually managed history: "any
reasoning items returned in model responses with tool calls must also be passed
back with tool call outputs". The Studio tool loop rebuilds the conversation
between turns by hand, so the provider has to hand the items out on the terminal
chunk and take them back on the next request. The openai_codex client already
does exactly this round trip for the same endpoint; this is the generic path
catching up.

Order is part of the contract, not a preference: a reasoning item with nothing
after it is a 400 ("Item 'rs_...' of type 'reasoning' was provided without its
required following item").
"""

import asyncio
import json

import httpx

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _sse(events) -> bytes:
    return (
        b"".join(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
        + b"data: [DONE]\n\n"
    )


_TOOL_CALL_STREAM = _sse(
    (
        {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "check the docs"}],
                "encrypted_content": "enc_blob",
                "status": "completed",
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "web_search",
                "arguments": "{}",
            },
        },
        {"type": "response.completed", "response": {"id": "resp_1"}},
    )
)


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _client(
    monkeypatch,
    captured: dict,
    body: bytes = _TOOL_CALL_STREAM,
):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )
    return ExternalProviderClient(
        provider_type = "openai",
        base_url = "https://api.openai.com/v1",
        api_key = "k",
    )


def _run(client, messages):
    async def go():
        lines = [
            line async for line in client.stream_chat_completion(messages = messages, model = "gpt-5.1")
        ]
        await client.close()
        return lines

    return _drive(go())


def _terminal_delta(lines):
    for line in reversed(lines):
        if not line.startswith("data: ") or line.strip() == "data: [DONE]":
            continue
        payload = json.loads(line[len("data: ") :])
        choice = (payload.get("choices") or [{}])[0]
        if choice.get("finish_reason"):
            return choice.get("delta") or {}
    raise AssertionError("no terminal chunk in the stream")


def test_a_tool_call_turn_hands_its_reasoning_items_to_the_caller(monkeypatch):
    """Without this the loop has nothing to replay.

    The SSE translation keeps only the summary prose (as ``<think>`` text) and
    drops the rs_ id, so the item can never be reconstructed downstream.
    """
    captured: dict = {}
    lines = _run(_client(monkeypatch, captured), [{"role": "user", "content": "search"}])

    items = _terminal_delta(lines)["extra_content"]["openai_responses_reasoning"]
    assert [item["id"] for item in items] == ["rs_1"]
    assert items[0]["encrypted_content"] == "enc_blob"
    assert items[0]["summary"] == [{"type": "summary_text", "text": "check the docs"}]


def test_a_plain_prose_turn_hands_back_nothing(monkeypatch):
    """Reasoning only has to round-trip across a tool call.

    A prose turn that shipped them would grow every following request body and
    buy nothing: the model is trained to produce its best answer without them.
    """
    captured: dict = {}
    stream = _sse(
        (
            {
                "type": "response.output_item.done",
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            },
            {"type": "response.completed", "response": {"id": "resp_1"}},
        )
    )
    lines = _run(_client(monkeypatch, captured, stream), [{"role": "user", "content": "hi"}])

    assert "extra_content" not in _terminal_delta(lines)


def test_the_follow_up_turn_replays_reasoning_before_its_function_call(monkeypatch):
    captured: dict = {}
    _run(
        _client(monkeypatch, captured),
        [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
                "extra_content": {
                    "openai_responses_reasoning": [
                        {
                            "type": "reasoning",
                            "id": "rs_1",
                            "summary": [{"type": "summary_text", "text": "check"}],
                            "encrypted_content": "enc_blob",
                            "status": "completed",
                        }
                    ]
                },
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        ],
    )

    items = captured["body"]["input"]
    assert [item.get("type") or item.get("role") for item in items] == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
    ]
    reasoning = items[1]
    assert reasoning["id"] == "rs_1"
    assert reasoning["encrypted_content"] == "enc_blob"
    # Responses rejects `status` on an input item, and the recorded copy carries
    # one, so the replay has to be re-sanitized rather than passed through.
    assert "status" not in reasoning


def test_assistant_text_still_precedes_the_function_call(monkeypatch):
    """The reasoning item leads, but the existing text/call order is unchanged."""
    captured: dict = {}
    _run(
        _client(monkeypatch, captured),
        [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "Let me look that up.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
                "extra_content": {
                    "openai_responses_reasoning": [
                        {"type": "reasoning", "id": "rs_1", "summary": []}
                    ]
                },
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        ],
    )

    assert [item.get("type") or item.get("role") for item in captured["body"]["input"]] == [
        "user",
        "reasoning",
        "assistant",
        "function_call",
        "function_call_output",
    ]


def test_reasoning_is_dropped_when_the_turn_had_only_server_builtins(monkeypatch):
    """A dropped builtin card leaves no following item.

    A trailing reasoning item is a hard 400, which is a worse outcome than the
    lost thought it was meant to preserve.
    """
    captured: dict = {}
    _run(
        _client(monkeypatch, captured),
        [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"_server_tool": True}),
                        },
                    }
                ],
                "extra_content": {
                    "openai_responses_reasoning": [
                        {"type": "reasoning", "id": "rs_1", "summary": []}
                    ]
                },
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        ],
    )

    assert not [item for item in captured["body"]["input"] if item.get("type") == "reasoning"]


def test_a_conversation_without_reasoning_is_unchanged(monkeypatch):
    """The pre-existing shape has to survive: most turns carry no extra_content."""
    captured: dict = {}
    _run(
        _client(monkeypatch, captured),
        [
            {"role": "user", "content": "search"},
            {
                "role": "assistant",
                "content": "Looking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result text"},
        ],
    )

    assert [item.get("type") or item.get("role") for item in captured["body"]["input"]] == [
        "user",
        "assistant",
        "function_call",
        "function_call_output",
    ]
