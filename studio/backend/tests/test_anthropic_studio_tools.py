# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native Anthropic calls through the real Studio transport and execution loop."""

import asyncio
import json
import threading

import httpx
import pytest

from core.inference import external_provider as ep
from core.inference import studio_tool_loop as loop
from core.inference.external_tool_transport import OAICompatTransport


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Run {name}",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    for name in ("python", "web_search")
]


def _block(index, block, *deltas):
    return [
        {"type": "content_block_start", "index": index, "content_block": block},
        *({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas),
        {"type": "content_block_stop", "index": index},
    ]


def _finish(reason):
    return [
        {"type": "message_delta", "delta": {"stop_reason": reason}, "usage": {"output_tokens": 10}},
        {"type": "message_stop"},
    ]


def _response(events):
    return httpx.Response(
        200,
        content = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode(),
        headers = {"content-type": "text/event-stream"},
    )


def _client(monkeypatch, handler):
    monkeypatch.setattr(
        ep, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )
    return ep.ExternalProviderClient(
        provider_type = "anthropic", base_url = "https://api.anthropic.com/v1", api_key = "test"
    )


def _request(client, **kwargs):
    return client.stream_chat_completion(
        messages = kwargs.pop("messages", [{"role": "user", "content": "Use the tools"}]),
        model = kwargs.pop("model", "claude-sonnet-4-6"),
        temperature = 0.7,
        top_p = 0.95,
        max_tokens = 4096,
        enable_prompt_caching = False,
        **kwargs,
    )


def _collect(client, **kwargs):
    async def run():
        try:
            return [line async for line in _request(client, **kwargs)]
        finally:
            await client.close()

    return asyncio.run(run())


@pytest.mark.parametrize(
    "choice,expected",
    [
        (None, {"type": "auto"}),
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ({"type": "function", "function": {"name": "python"}}, {"type": "tool", "name": "python"}),
        ("none", None),
    ],
)
def test_function_schemas_and_choice_reach_anthropic(monkeypatch, choice, expected):
    bodies = []
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    _collect(client, tools = TOOLS, tool_choice = choice, enabled_tools = ["web_fetch"])
    body = bodies[0]
    assert body.get("tool_choice") == expected
    if choice == "none":
        assert "tools" not in body
    else:
        assert body["tools"][:2] == [
            {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            }
            for tool in TOOLS
        ]
        assert any(tool["name"] == "web_fetch" for tool in body["tools"]) is not isinstance(
            choice, dict
        )


def test_manual_thinking_does_not_override_a_forced_tool(monkeypatch):
    bodies = []
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    _collect(
        client,
        tools = TOOLS,
        tool_choice = "required",
        model = "claude-haiku-4-5-20251001",
        enable_thinking = True,
    )
    assert "thinking" not in bodies[0]
    assert bodies[0]["tool_choice"] == {"type": "any"}


@pytest.mark.parametrize("max_calls", [2, 4])
@pytest.mark.parametrize("pending_fetch", [False, True])
@pytest.mark.parametrize("repeat_calls", [False, True])
def test_loop_executes_fragmented_calls_and_replays_signed_and_hosted_blocks(
    monkeypatch, max_calls, pending_fetch, repeat_calls
):
    thinking = {"type": "thinking", "thinking": "Choose tools.", "signature": "signed-thinking"}
    redacted = {"type": "redacted_thinking", "data": "opaque"}
    hosted = {
        "type": "server_tool_use",
        "id": "srv_1",
        "name": "web_fetch",
        "input": {"url": "https://example.com"},
    }
    hosted_result = {
        "type": "web_fetch_tool_result",
        "tool_use_id": "srv_1",
        "content": {"type": "web_fetch_tool_error", "error_code": "unavailable"},
    }
    events = [
        *_block(
            0,
            {"type": "thinking", "thinking": "", "signature": ""},
            {"type": "thinking_delta", "thinking": thinking["thinking"]},
            {"type": "signature_delta", "signature": "signed-"},
            {"type": "signature_delta", "signature": "thinking"},
        ),
        *_block(1, redacted),
        *_block(
            2,
            {**hosted, "input": {}},
            {"type": "input_json_delta", "partial_json": json.dumps(hosted["input"])},
        ),
        *([] if pending_fetch else _block(3, hosted_result)),
        *_block(
            4,
            {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"query":'},
            {"type": "input_json_delta", "partial_json": '"one"}'},
        ),
        *_block(
            5,
            {"type": "tool_use", "id": "toolu_b", "name": "web_search", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"query":"two"}'},
        ),
        *_finish("tool_use"),
    ]
    bodies, executed = [], []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1 or (repeat_calls and len(bodies) == 2 and max_calls > 2):
            return _response(events)
        return _response(
            [
                *_block(
                    0, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Finished."}
                ),
                *_finish("end_turn"),
            ]
        )

    def execute(name, arguments, **kwargs):
        executed.append((name, arguments))
        return f"result:{arguments['query']}"

    monkeypatch.setattr(loop, "execute_tool", execute)
    monkeypatch.setattr(loop, "build_rag_autoinject", lambda *a, **k: None)
    client = _client(monkeypatch, handler)

    async def run():
        try:
            return [
                line
                async for line in loop.stream_with_studio_tools(
                    OAICompatTransport(
                        client,
                        model = "claude-sonnet-4-6",
                        temperature = 0.7,
                        top_p = 0.95,
                        max_tokens = 4096,
                        enable_thinking = True,
                        enable_prompt_caching = False,
                        enabled_tools = ["web_fetch"],
                    ),
                    run = loop.ToolLoopRun(messages = [{"role": "user", "content": "Use both tools"}]),
                    policy = loop.ToolLoopPolicy(
                        tools = TOOLS,
                        max_calls = max_calls,
                        timeout = 30,
                        permission_mode = "off",
                        confirm_calls = False,
                        bypass_permissions = False,
                        rag_scope = None,
                        auto_heal = False,
                        nudge_tool_calls = False,
                    ),
                    cancel_event = threading.Event(),
                )
            ]
        finally:
            await client.close()

    lines = asyncio.run(run())
    assert executed == [("python", {"query": "one"}), ("web_search", {"query": "two"})]
    assert len(bodies) == (3 if repeat_calls and max_calls > 2 else 2)
    assistant, results = bodies[1]["messages"][1:3]
    assert assistant["role"] == "assistant"
    native_prefix = [thinking, redacted, hosted] + ([] if pending_fetch else [hosted_result])
    assert assistant["content"][: len(native_prefix)] == native_prefix
    assert assistant["content"][len(native_prefix) :] == [
        {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {"query": "one"}},
        {"type": "tool_use", "id": "toolu_b", "name": "web_search", "input": {"query": "two"}},
    ]
    assert results == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "toolu_a", "content": "result:one"},
            {"type": "tool_result", "tool_use_id": "toolu_b", "content": "result:two"},
        ],
    }
    assert any("Finished." in line for line in lines)
    assert not any('"error"' in line for line in lines)
    assert len(bodies[1]["messages"]) == 3
    if max_calls == 2:
        assert {"python", "web_search"} <= {tool["name"] for tool in bodies[1]["tools"]}
        assert bodies[1]["tool_choice"] == {"type": "none"}
    if pending_fetch:
        assert any(tool["name"] == "web_fetch" for tool in bodies[1]["tools"])
    if len(bodies) == 3:
        noop_assistant, noop_results = bodies[-1]["messages"][-2:]
        assert noop_assistant["role"] == "assistant"
        assert [block["type"] for block in noop_results["content"]] == [
            "tool_result",
            "tool_result",
        ]
        assert (
            len([block for block in noop_assistant["content"] if block["type"] == "tool_use"]) == 2
        )


def test_tool_result_images_survive_translation(monkeypatch):
    bodies = []
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    _collect(
        client,
        tools = TOOLS,
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "toolu_a",
                        "type": "function",
                        "function": {"name": "python", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "toolu_a",
                "content": [
                    {"type": "text", "text": "Image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGlj"}},
                ],
            },
        ],
    )
    assert bodies[0]["messages"][-1]["content"][0]["content"] == [
        {"type": "text", "text": "Image"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "cGlj"}},
    ]


@pytest.mark.parametrize(
    "ending, expected_calls",
    [("tool_use", 1), ("max_tokens", 0), ("refusal", 0), ("error", 0), ("eof", 0)],
)
def test_incomplete_turns_never_execute_client_calls(monkeypatch, ending, expected_calls):
    executed, bodies = [], []
    events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {}},
        }
    ]
    if ending == "error":
        events.append(
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
        )
    elif ending != "eof":
        events.extend([{"type": "content_block_stop", "index": 0}, *_finish(ending)])

    def handler(request):
        bodies.append(json.loads(request.content))
        return _response(events if len(bodies) == 1 else _finish("end_turn"))

    def execute(name, arguments, **kwargs):
        executed.append((name, arguments))
        return "done"

    monkeypatch.setattr(loop, "execute_tool", execute)
    monkeypatch.setattr(loop, "build_rag_autoinject", lambda *a, **k: None)
    client = _client(monkeypatch, handler)

    async def run():
        try:
            return [
                line
                async for line in loop.stream_with_studio_tools(
                    OAICompatTransport(
                        client,
                        model = "claude-sonnet-4-6",
                        temperature = 0.7,
                        top_p = 0.95,
                        max_tokens = 4096,
                    ),
                    run = loop.ToolLoopRun(messages = [{"role": "user", "content": "Use tool"}]),
                    policy = loop.ToolLoopPolicy(
                        tools = TOOLS,
                        max_calls = 1,
                        timeout = 30,
                        permission_mode = "off",
                        confirm_calls = False,
                        bypass_permissions = False,
                        rag_scope = None,
                        auto_heal = False,
                        nudge_tool_calls = False,
                    ),
                    cancel_event = threading.Event(),
                )
            ]
        finally:
            await client.close()

    asyncio.run(run())
    assert len(executed) == expected_calls
    assert len(bodies) == 1 + expected_calls


def test_native_compaction_delta_is_replayed_with_encrypted_content(monkeypatch):
    bodies = []
    summary = {"content": "Earlier conversation summary", "encrypted_content": "opaque-compaction"}
    events = [
        *_block(
            0, {"type": "compaction", "content": None}, {"type": "compaction_delta", **summary}
        ),
        *_finish("tool_use"),
    ]
    client = _client(
        monkeypatch,
        lambda request: (bodies.append(json.loads(request.content)) or _response(events)),
    )
    lines = _collect(client, tools = TOOLS)
    deltas = [
        json.loads(line[6:])["choices"][0]["delta"]
        for line in lines
        if line.startswith("data: {") and json.loads(line[6:]).get("choices")
    ]
    extra = next(delta["extra_content"] for delta in deltas if "extra_content" in delta)
    assert extra["anthropic"]["content"] == [{"type": "compaction", **summary}]
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    _collect(
        client, tools = TOOLS, messages = [{"role": "assistant", "content": "", "extra_content": extra}]
    )
    assert bodies[-1]["messages"][0]["content"] == [{"type": "compaction", **summary}]


def test_tool_follow_up_replays_native_compaction_once(monkeypatch):
    bodies = []
    summary = {"content": "Earlier conversation summary", "encrypted_content": "opaque-compaction"}
    first_turn = [
        *_block(
            0,
            {"type": "compaction", "content": None},
            {"type": "compaction_delta", "content": "Earlier conversation "},
            {
                "type": "compaction_delta",
                "content": "summary",
                "encrypted_content": summary["encrypted_content"],
            },
        ),
        *_block(
            1,
            {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"query":"current"}'},
        ),
        *_finish("tool_use"),
    ]

    def handler(request):
        bodies.append(json.loads(request.content))
        return _response(first_turn if len(bodies) == 1 else _finish("end_turn"))

    monkeypatch.setattr(loop, "execute_tool", lambda *_args, **_kwargs: "done")
    monkeypatch.setattr(loop, "build_rag_autoinject", lambda *a, **k: None)
    client = _client(monkeypatch, handler)

    async def run():
        try:
            return [
                line
                async for line in loop.stream_with_studio_tools(
                    OAICompatTransport(
                        client,
                        model = "claude-sonnet-4-6",
                        temperature = 0.7,
                        top_p = 0.95,
                        max_tokens = 4096,
                    ),
                    run = loop.ToolLoopRun(
                        messages = [
                            {"role": "user", "content": "old question"},
                            {"role": "assistant", "content": "old answer"},
                            {"role": "user", "content": "current question"},
                        ]
                    ),
                    policy = loop.ToolLoopPolicy(
                        tools = TOOLS,
                        max_calls = 1,
                        timeout = 30,
                        permission_mode = "off",
                        confirm_calls = False,
                        bypass_permissions = False,
                        rag_scope = None,
                        auto_heal = False,
                        nudge_tool_calls = False,
                    ),
                    cancel_event = threading.Event(),
                )
            ]
        finally:
            await client.close()

    asyncio.run(run())
    assistant_parts = [
        part
        for message in bodies[1]["messages"]
        if message["role"] == "assistant"
        for part in message["content"]
    ]
    assert [part for part in assistant_parts if part["type"] == "compaction"] == [
        {"type": "compaction", **summary}
    ]
    assert len([part for part in assistant_parts if part["type"] == "tool_use"]) == 1


def test_failed_native_compaction_keeps_history_for_tool_follow_up(monkeypatch):
    bodies = []
    first_turn = [
        *_block(
            0,
            {
                "type": "compaction",
                "content": None,
                "encrypted_content": "opaque-failed",
            },
        ),
        *_block(
            1,
            {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"query":"current"}'},
        ),
        *_finish("tool_use"),
    ]

    def handler(request):
        bodies.append(json.loads(request.content))
        return _response(first_turn if len(bodies) == 1 else _finish("end_turn"))

    monkeypatch.setattr(loop, "execute_tool", lambda *_args, **_kwargs: "done")
    monkeypatch.setattr(loop, "build_rag_autoinject", lambda *a, **k: None)
    client = _client(monkeypatch, handler)

    async def run():
        try:
            return [
                line
                async for line in loop.stream_with_studio_tools(
                    OAICompatTransport(
                        client,
                        model = "claude-sonnet-4-6",
                        temperature = 0.7,
                        top_p = 0.95,
                        max_tokens = 4096,
                    ),
                    run = loop.ToolLoopRun(
                        messages = [
                            {"role": "user", "content": "old question"},
                            {"role": "assistant", "content": "old answer"},
                            {"role": "user", "content": "current question"},
                        ]
                    ),
                    policy = loop.ToolLoopPolicy(
                        tools = TOOLS,
                        max_calls = 1,
                        timeout = 30,
                        permission_mode = "off",
                        confirm_calls = False,
                        bypass_permissions = False,
                        rag_scope = None,
                        auto_heal = False,
                        nudge_tool_calls = False,
                    ),
                    cancel_event = threading.Event(),
                )
            ]
        finally:
            await client.close()

    asyncio.run(run())
    follow_up = json.dumps(bodies[1]["messages"])
    assert "old question" in follow_up
    assert "old answer" in follow_up
    assert "current question" in follow_up
    assert "opaque-failed" not in follow_up


def test_truncated_hosted_json_still_reports_length(monkeypatch):
    events = [
        *_block(
            0,
            {"type": "server_tool_use", "id": "srv_1", "name": "web_fetch", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"url":"https://exa'},
        ),
        *_finish("max_tokens"),
    ]
    client = _client(monkeypatch, lambda request: _response(events))
    lines = _collect(client, tools = TOOLS, enabled_tools = ["web_fetch"])
    assert any('"finish_reason": "length"' in line for line in lines)
    assert lines[-1] == "data: [DONE]"


def test_detached_mcp_images_join_tool_result_only_on_followup(monkeypatch):
    bodies = []
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_a",
                    "type": "function",
                    "function": {"name": "python", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_a", "content": "result"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Tool image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,cGlj"}},
            ],
        },
    ]
    original = json.dumps(messages)
    initial = [{"role": "user", "content": "Use the tools"}]
    third_turn = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_b",
                    "type": "function",
                    "function": {"name": "python", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_b", "content": "second result"},
    ]

    async def run():
        transport = OAICompatTransport(
            client,
            model = "claude-sonnet-4-6",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 4096,
            enable_prompt_caching = False,
        )
        try:
            for history in (initial, initial + messages, initial + messages + third_turn):
                async for _ in transport.stream(
                    messages = history,
                    tools = TOOLS,
                    tool_choice = "auto",
                    cancel_event = threading.Event(),
                ):
                    pass
        finally:
            await client.close()

    asyncio.run(run())
    assert len(bodies[0]["messages"]) == 1
    assert len(bodies[1]["messages"]) == 3
    assert len(bodies[2]["messages"]) == 5
    assert bodies[2]["messages"][:3] == bodies[1]["messages"]
    result = bodies[1]["messages"][-1]["content"][0]
    assert result["type"] == "tool_result"
    assert result["content"][-1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "cGlj"},
    }
    assert "Tool image" in result["content"][0]["text"]
    assert json.dumps(messages) == original


@pytest.mark.parametrize("name", ["web_search", "web_fetch", "code_execution"])
def test_caller_names_take_precedence_over_automatic_hosted_tools(monkeypatch, name):
    bodies = []
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    tool = {
        "type": "function",
        "function": {"name": name, "parameters": {"type": "object", "properties": {}}},
    }
    _collect(client, tools = [tool], enabled_tools = [name])
    assert bodies[0]["tools"] == [{"name": name, "input_schema": tool["function"]["parameters"]}]


@pytest.mark.parametrize(
    "content", ["Rendered reply", None, [{"type": "text", "text": "Rendered reply"}], []]
)
@pytest.mark.parametrize("vision", [True, False])
def test_client_continuation_replays_native_anthropic_state(monkeypatch, content, vision):
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    native = [
        {"type": "thinking", "thinking": "Choose tools", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
        {
            "type": "server_tool_use",
            "id": "srv_1",
            "name": "web_fetch",
            "input": {"url": "https://example.com"},
        },
    ]
    calls = [
        {"id": "toolu_1", "type": "function", "function": {"name": "get_status", "arguments": "{}"}}
    ]
    messages = [
        ChatMessage(role = "user", content = "Fetch the page and get status"),
        ChatMessage(
            role = "assistant",
            content = content,
            tool_calls = calls,
            extra_content = {
                "anthropic": {"content": native, "unrelated": "omit"},
                "google": {"thought_signature": "foreign"},
                "openai_codex": {"items": []},
            },
        ),
        ChatMessage(role = "tool", tool_call_id = "toolu_1", content = "READY"),
    ]
    normalized = _build_external_messages(messages, vision, provider_type = "anthropic")
    assert normalized[1]["extra_content"] == {"anthropic": {"content": native}}
    bodies = []
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    _collect(
        client,
        messages = normalized,
        tools = [{"type": "function", "function": {"name": "get_status"}}],
        enabled_tools = [],
        tool_choice = "none",
    )
    assert bodies[0]["messages"][1]["content"] == native + [
        {"type": "tool_use", "id": "toolu_1", "name": "get_status", "input": {}}
    ]
    assert bodies[0]["messages"][2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "READY"}
    ]
    assert [tool["name"] for tool in bodies[0]["tools"]] == ["get_status", "web_fetch"]
    assert bodies[0]["tool_choice"] == {"type": "none"}


def test_anthropic_does_not_forward_foreign_message_metadata(monkeypatch):
    from models.inference import ChatMessage
    from routes.inference import _build_external_messages

    messages = _build_external_messages(
        [
            ChatMessage(role = "user", content = "Hello"),
            ChatMessage(
                role = "assistant",
                content = "Previous reply",
                extra_content = {
                    "google": {"thought_signature": "foreign"},
                    "openai_codex": {"items": []},
                },
            ),
            ChatMessage(role = "user", content = "Continue"),
        ],
        True,
        provider_type = "anthropic",
    )
    bodies = []
    client = _client(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content)) or _response(_finish("end_turn"))
        ),
    )
    _collect(client, messages = messages)
    assert bodies[0]["messages"][1] == {"role": "assistant", "content": "Previous reply"}


def test_whitespace_text_blocks_are_not_replayed(monkeypatch):
    # Anthropic 400s "text content blocks must contain non-whitespace text" on a replayed blank block.
    events = [
        *_block(0, {"type": "text", "text": ""}, {"type": "text_delta", "text": "\n\n"}),
        *_block(
            1,
            {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {}},
            {"type": "input_json_delta", "partial_json": '{"query":"one"}'},
        ),
        *_finish("tool_use"),
    ]
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return _response(events)
        return _response(
            [
                *_block(0, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Done."}),
                *_finish("end_turn"),
            ]
        )

    monkeypatch.setattr(loop, "execute_tool", lambda name, arguments, **kwargs: "ok")
    monkeypatch.setattr(loop, "build_rag_autoinject", lambda *a, **k: None)
    client = _client(monkeypatch, handler)

    async def run():
        try:
            return [
                line
                async for line in loop.stream_with_studio_tools(
                    OAICompatTransport(
                        client,
                        model = "claude-sonnet-4-6",
                        temperature = 0.7,
                        top_p = 0.95,
                        max_tokens = 4096,
                        enable_prompt_caching = False,
                    ),
                    run = loop.ToolLoopRun(messages = [{"role": "user", "content": "Run it"}]),
                    policy = loop.ToolLoopPolicy(
                        tools = TOOLS,
                        max_calls = 5,
                        timeout = 30,
                        permission_mode = "off",
                        confirm_calls = False,
                        bypass_permissions = False,
                        rag_scope = None,
                        auto_heal = False,
                        nudge_tool_calls = False,
                    ),
                    cancel_event = threading.Event(),
                )
            ]
        finally:
            await client.close()

    asyncio.run(run())
    assert len(bodies) == 2
    assistant = bodies[1]["messages"][1]
    assert assistant["content"] == [
        {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {"query": "one"}}
    ]


def test_client_calls_keep_their_position_in_the_replayed_turn(monkeypatch):
    client_call = {"type": "tool_use", "id": "toolu_a", "name": "python", "input": {"query": "one"}}
    hosted = {
        "type": "server_tool_use",
        "id": "srv_1",
        "name": "web_fetch",
        "input": {"url": "https://example.com"},
    }
    events = [
        *_block(
            0,
            {**client_call, "input": {}},
            {"type": "input_json_delta", "partial_json": '{"query":"one"}'},
        ),
        *_block(
            1,
            {**hosted, "input": {}},
            {"type": "input_json_delta", "partial_json": json.dumps(hosted["input"])},
        ),
        *_block(2, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Working."}),
        *_finish("tool_use"),
    ]
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return _response(events)
        return _response(
            [
                *_block(0, {"type": "text", "text": ""}, {"type": "text_delta", "text": "Done."}),
                *_finish("end_turn"),
            ]
        )

    monkeypatch.setattr(loop, "execute_tool", lambda name, arguments, **kwargs: "ok")
    monkeypatch.setattr(loop, "build_rag_autoinject", lambda *a, **k: None)
    client = _client(monkeypatch, handler)

    async def run():
        try:
            return [
                line
                async for line in loop.stream_with_studio_tools(
                    OAICompatTransport(
                        client,
                        model = "claude-sonnet-4-6",
                        temperature = 0.7,
                        top_p = 0.95,
                        max_tokens = 4096,
                        enable_prompt_caching = False,
                        enabled_tools = ["web_fetch"],
                    ),
                    run = loop.ToolLoopRun(messages = [{"role": "user", "content": "Run it"}]),
                    policy = loop.ToolLoopPolicy(
                        tools = TOOLS,
                        max_calls = 5,
                        timeout = 30,
                        permission_mode = "off",
                        confirm_calls = False,
                        bypass_permissions = False,
                        rag_scope = None,
                        auto_heal = False,
                        nudge_tool_calls = False,
                    ),
                    cancel_event = threading.Event(),
                )
            ]
        finally:
            await client.close()

    asyncio.run(run())
    assert len(bodies) == 2
    assert bodies[1]["messages"][1]["content"] == [
        client_call,
        hosted,
        {"type": "text", "text": "Working."},
    ]
