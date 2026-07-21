# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for Anthropic Messages API schemas and translation layer (no server/GPU)."""

import sys
import os
import json
import threading

import httpx
import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from models.inference import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicMessage,
    AnthropicTextBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicTool,
    AnthropicUsage,
    AnthropicResponseTextBlock,
    AnthropicResponseToolUseBlock,
)
from core.inference.anthropic_compat import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    build_anthropic_sse_event,
    AnthropicStreamEmitter,
    AnthropicPassthroughEmitter,
)
from core.inference.api_monitor import ApiMonitor
from routes.inference import (
    _build_tool_action_nudge,
    _normalize_anthropic_openai_images,
    _select_anthropic_server_tools,
    _anthropic_requested_studio_tools,
    _anthropic_passthrough_stream,
    _anthropic_tool_non_streaming,
    _monitor_anthropic_sse_line,
    anthropic_messages,
)
from state.tool_policy import reset_tool_policy, set_tool_policy
from fastapi import HTTPException
import asyncio
import base64 as _b64
from io import BytesIO as _BytesIO
from types import SimpleNamespace


def _emitter_client_text(events: list[str]) -> str:
    """Concatenate the text_delta payloads an SSE event list carries."""
    text = ""
    for line in events:
        for raw in line.split("\n"):
            raw = raw.strip()
            if not raw.startswith("data: "):
                continue
            data = json.loads(raw[len("data: ") :])
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                text += delta.get("text", "")
    return text


def test_anthropic_emitter_closes_reasoning_only_think_block():
    # A reasoning-only reply streams <think>X live then shrinks to bare X at EOF.
    # This emitter diffs cumulative snapshots and drops the shrink, so without a
    # closing pass the client text would end on an unclosed <think>. finish()
    # must balance it.
    emitter = AnthropicStreamEmitter()
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "<think>The capital"})
    events += emitter.feed(
        {"type": "content", "text": "<think>The capital of France is Paris."}
    )
    # The generator's final bare-text shrink (dropped by the cumulative diff).
    events += emitter.feed(
        {"type": "content", "text": "The capital of France is Paris."}
    )
    events += emitter.finish()

    assert (
        _emitter_client_text(events) == "<think>The capital of France is Paris.</think>"
    )


def test_anthropic_emitter_does_not_double_close_balanced_think():
    # A reasoning-then-answer reply already closes its own </think>; the balancer
    # must not append a second one.
    emitter = AnthropicStreamEmitter()
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "<think>Thinking."})
    events += emitter.feed(
        {"type": "content", "text": "<think>Thinking.</think>Answer."}
    )
    events += emitter.finish()

    assert _emitter_client_text(events) == "<think>Thinking.</think>Answer."


def test_streamed_anthropic_tool_use_records_api_monitor_reply(monkeypatch):
    import routes.inference as inf_mod

    monitor = ApiMonitor(max_entries = 3)
    monkeypatch.setattr(inf_mod, "api_monitor", monitor)
    monitor_id = monitor.start(
        endpoint = "/v1/messages",
        method = "POST",
        model = "m",
        prompt = "hi",
    )

    for payload in (
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "lookup",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"query":"weather"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
    ):
        _monitor_anthropic_sse_line(monitor_id, f"data: {json.dumps(payload)}")

    entry = monitor.get(monitor_id)
    assert entry is not None
    assert entry["reply"] == 'Tool call: lookup\nInput: {"query":"weather"}'


# =====================================================================
# Tool nudge tests
# =====================================================================


class TestToolActionNudge:
    def test_balanced_nudge_uses_expanded_web_and_code_tips(self):
        nudge = _build_tool_action_nudge(
            tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "python"}},
            ],
            model_name = "Llama-3.1-70B-Instruct",
        )

        assert nudge.startswith("The current date is ")
        assert "Tools are available when they materially improve" in nudge
        assert "prefer using tools rather than answering from memory" not in nudge
        assert (
            "fetch its full content by calling web_search with the url parameter"
            in nudge
        )
        assert "Use code execution for math" in nudge
        assert "render_html" not in nudge

    def test_balanced_nudge_preserves_compact_web_tip_and_canvas_gate(self):
        nudge = _build_tool_action_nudge(
            tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "render_html"}},
            ],
            model_name = "Llama-3.1-8B-Instruct",
        )

        assert "When using web_search, do not repeat the same search query." in nudge
        assert "fetch its full content" not in nudge
        assert "call render_html once" in nudge

    def test_balanced_nudge_empty_without_known_tool_categories(self):
        assert (
            _build_tool_action_nudge(tools = [], model_name = "Llama-3.1-8B-Instruct") == ""
        )


# =====================================================================
# Pydantic model tests
# =====================================================================


class TestAnthropicModels:
    def test_minimal_request(self):
        req = AnthropicMessagesRequest(
            messages = [{"role": "user", "content": "Hi"}],
        )
        assert req.max_tokens is None
        assert req.model == "default"
        assert req.stream is False

    def test_max_tokens_optional(self):
        req = AnthropicMessagesRequest(
            max_tokens = 100,
            messages = [{"role": "user", "content": "Hi"}],
        )
        assert req.max_tokens == 100

    def test_system_as_string(self):
        req = AnthropicMessagesRequest(
            max_tokens = 50,
            messages = [{"role": "user", "content": "Hi"}],
            system = "You are helpful.",
        )
        assert req.system == "You are helpful."

    def test_system_role_message_normalized_to_system_field(self):
        req = AnthropicMessagesRequest(
            max_tokens = 50,
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
        )
        assert req.system == "You are helpful."
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_system_role_message_merges_with_existing_system_field(self):
        req = AnthropicMessagesRequest(
            max_tokens = 50,
            system = "Base instructions.",
            messages = [
                {"role": "user", "content": "Hi"},
                {"role": "system", "content": "Additional instructions."},
                {"role": "assistant", "content": "Hello."},
            ],
        )
        assert req.system == "Base instructions.\n\nAdditional instructions."
        assert [msg.role for msg in req.messages] == ["user", "assistant"]

    def test_system_role_message_with_null_content_ignored(self):
        req = AnthropicMessagesRequest(
            max_tokens = 50,
            system = "Base.",
            messages = [
                {"role": "system", "content": None},
                {
                    "role": "system",
                    "content": [
                        None,
                        {"type": "text", "text": "Use short answers."},
                    ],
                },
                {"role": "user", "content": "Hi"},
            ],
        )
        assert req.system == "Base.\n\nUse short answers."
        assert "None" not in str(req.system)
        assert [msg.role for msg in req.messages] == ["user"]

    def test_tools_field_parses(self):
        req = AnthropicMessagesRequest(
            max_tokens = 100,
            messages = [{"role": "user", "content": "Hi"}],
            tools = [{"name": "web_search", "input_schema": {"type": "object"}}],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "web_search"

    def test_server_tool_field_parses(self):
        req = AnthropicMessagesRequest(
            max_tokens = 100,
            messages = [{"role": "user", "content": "Hi"}],
            tools = [{"type": "web_fetch_20250910", "name": "web_fetch"}],
        )
        assert len(req.tools) == 1
        assert req.tools[0].type == "web_fetch_20250910"
        assert req.tools[0].name == "web_fetch"
        assert req.tools[0].input_schema is None

    def test_extra_fields_accepted(self):
        req = AnthropicMessagesRequest(
            max_tokens = 100,
            messages = [{"role": "user", "content": "Hi"}],
            some_future_field = "hello",
        )
        assert req.max_tokens == 100

    def test_stream_defaults_false(self):
        req = AnthropicMessagesRequest(
            max_tokens = 100,
            messages = [{"role": "user", "content": "Hi"}],
        )
        assert req.stream is False

    def test_enable_tools_shorthand(self):
        req = AnthropicMessagesRequest(
            messages = [{"role": "user", "content": "Hi"}],
            enable_tools = True,
            enabled_tools = ["web_search", "python"],
            session_id = "my-session",
        )
        assert req.enable_tools is True
        assert req.enabled_tools == ["web_search", "python"]
        assert req.session_id == "my-session"

    def test_extension_fields_default_none(self):
        req = AnthropicMessagesRequest(
            messages = [{"role": "user", "content": "Hi"}],
        )
        assert req.enable_tools is None
        assert req.enabled_tools is None
        assert req.session_id is None

    def test_response_model_defaults(self):
        resp = AnthropicMessagesResponse()
        assert resp.type == "message"
        assert resp.role == "assistant"
        assert resp.id.startswith("msg_")
        assert resp.content == []
        assert resp.usage.input_tokens == 0


# =====================================================================
# Message translation tests
# =====================================================================


class TestAnthropicMessagesToOpenAI:
    def test_simple_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = anthropic_messages_to_openai(msgs)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_system_string_prepended(self):
        msgs = [{"role": "user", "content": "Hello"}]
        result = anthropic_messages_to_openai(msgs, system = "Be brief.")
        assert result[0] == {"role": "system", "content": "Be brief."}
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_top_level_system_request_translates_unchanged(self):
        req = AnthropicMessagesRequest(
            messages = [{"role": "user", "content": "Hello"}],
            system = "Be brief.",
        )
        result = anthropic_messages_to_openai(
            [m.model_dump() for m in req.messages],
            req.system,
        )
        assert result == [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hello"},
        ]

    def test_system_as_block_list(self):
        system = [
            {"type": "text", "text": "Be brief."},
            {"type": "text", "text": "Be accurate."},
        ]
        msgs = [{"role": "user", "content": "Hello"}]
        result = anthropic_messages_to_openai(msgs, system = system)
        assert result[0]["role"] == "system"
        assert "Be brief." in result[0]["content"]
        assert "Be accurate." in result[0]["content"]

    def test_multi_turn_conversation(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    def test_assistant_tool_use_maps_to_tool_calls(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "web_search",
                        "input": {"query": "test"},
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 1
        m = result[0]
        assert m["role"] == "assistant"
        assert m["content"] == "Let me search."
        assert len(m["tool_calls"]) == 1
        tc = m["tool_calls"][0]
        assert tc["id"] == "tu_1"
        assert tc["function"]["name"] == "web_search"
        assert json.loads(tc["function"]["arguments"]) == {"query": "test"}

    def test_tool_result_maps_to_tool_role(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": "Result text",
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "tu_1"
        assert result[0]["content"] == "Result text"

    def test_mixed_text_and_tool_use_blocks(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Thinking..."},
                    {
                        "type": "tool_use",
                        "id": "tu_1",
                        "name": "python",
                        "input": {"code": "1+1"},
                    },
                    {
                        "type": "tool_use",
                        "id": "tu_2",
                        "name": "terminal",
                        "input": {"command": "ls"},
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 1
        m = result[0]
        assert m["content"] == "Thinking..."
        assert len(m["tool_calls"]) == 2

    def test_tool_result_with_list_content(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [
                            {"type": "text", "text": "Line 1"},
                            {"type": "text", "text": "Line 2"},
                        ],
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        assert result[0]["content"] == "Line 1 Line 2"

    def test_image_base64_block_becomes_multimodal_part(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "AAAA",
                        },
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        parts = result[0]["content"]
        assert isinstance(parts, list)
        assert parts[0] == {"type": "text", "text": "What is this?"}
        assert parts[1]["type"] == "image_url"
        assert parts[1]["image_url"]["url"] == "data:image/jpeg;base64,AAAA"

    def test_image_url_block_forwarded_as_url(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe it"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://x/y.png"},
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        parts = result[0]["content"]
        assert parts[1] == {
            "type": "image_url",
            "image_url": {"url": "https://x/y.png"},
        }

    def test_image_only_user_message_emits_no_text_part(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "ZZ",
                        },
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        parts = result[0]["content"]
        assert len(parts) == 1
        assert parts[0]["type"] == "image_url"

    def test_image_default_media_type_when_missing(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "data": "BB"},
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        parts = result[0]["content"]
        assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_image_text_order_preserved(self):
        # [text1, image1, text2, image2] must not collapse to
        # [text1+text2, image1, image2].
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AA",
                        },
                    },
                    {"type": "text", "text": "after"},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": "https://x/y.png"},
                    },
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        parts = result[0]["content"]
        assert [p["type"] for p in parts] == ["text", "image_url", "text", "image_url"]
        assert parts[0]["text"] == "before"
        assert parts[2]["text"] == "after"
        assert parts[1]["image_url"]["url"] == "data:image/png;base64,AA"
        assert parts[3]["image_url"]["url"] == "https://x/y.png"

    def test_malformed_image_block_is_skipped(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "image", "source": {"type": "base64"}},
                    {"type": "image", "source": {"type": "url"}},
                ],
            }
        ]
        result = anthropic_messages_to_openai(msgs)
        # No image parts emitted; message falls back to plain text.
        assert result[0] == {"role": "user", "content": "Hi"}


# =====================================================================
# Tool translation tests
# =====================================================================


class TestAnthropicToolsToOpenAI:
    def test_single_tool(self):
        tools = [
            {
                "name": "web_search",
                "description": "Search",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]
        result = anthropic_tools_to_openai(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "web_search"
        assert result[0]["function"]["parameters"]["type"] == "object"

    def test_multiple_tools(self):
        tools = [
            {"name": "a", "description": "Tool A", "input_schema": {}},
            {"name": "b", "description": "Tool B", "input_schema": {}},
        ]
        result = anthropic_tools_to_openai(tools)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_empty_list(self):
        assert anthropic_tools_to_openai([]) == []

    def test_server_tools_are_not_converted_to_openai_functions(self):
        tools = [
            {"type": "web_fetch_20250910", "name": "web_fetch"},
            {"type": "web_search_20250305", "name": "web_search"},
        ]
        assert anthropic_tools_to_openai(tools) == []

    def test_server_tool_selection_merges_enabled_tools_extension(self):
        all_tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "python"}},
            {"type": "function", "function": {"name": "terminal"}},
        ]

        result = _select_anthropic_server_tools(
            all_tools,
            requested_studio_tools = {"web_search"},
            enabled_tools = ["python"],
        )

        assert [tool["function"]["name"] for tool in result] == ["web_search", "python"]

    def test_pydantic_model_input(self):
        tool = AnthropicTool(
            name = "test", description = "desc", input_schema = {"type": "object"}
        )
        result = anthropic_tools_to_openai([tool])
        assert result[0]["function"]["name"] == "test"


# =====================================================================
# SSE event helper tests
# =====================================================================


class TestBuildAnthropicSSEEvent:
    def test_basic_event(self):
        result = build_anthropic_sse_event("message_start", {"type": "message_start"})
        assert result.startswith("event: message_start\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_data_is_valid_json(self):
        result = build_anthropic_sse_event("test", {"key": "value"})
        data_line = result.split("\n")[1]
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload == {"key": "value"}


# =====================================================================
# Stream emitter tests
# =====================================================================


class TestAnthropicStreamEmitter:
    def test_start_emits_message_start_and_content_block_start(self):
        e = AnthropicStreamEmitter()
        events = e.start("msg_123", "test-model")
        assert len(events) == 2
        assert "message_start" in events[0]
        assert "content_block_start" in events[1]
        assert '"type": "text"' in events[1]

    def test_content_delta_emits_text_delta(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        events = e.feed({"type": "content", "text": "Hello"})
        assert len(events) == 1
        parsed = json.loads(events[0].split("data: ")[1])
        assert parsed["delta"]["type"] == "text_delta"
        assert parsed["delta"]["text"] == "Hello"

    def test_cumulative_content_diffs_correctly(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed({"type": "content", "text": "Hel"})
        events = e.feed({"type": "content", "text": "Hello"})
        parsed = json.loads(events[0].split("data: ")[1])
        assert parsed["delta"]["text"] == "lo"

    def test_empty_content_diff_no_event(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed({"type": "content", "text": "Hi"})
        events = e.feed({"type": "content", "text": "Hi"})
        assert events == []

    def test_tool_start_closes_text_opens_tool_block(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed({"type": "content", "text": "Thinking"})
        events = e.feed(
            {
                "type": "tool_start",
                "tool_name": "web_search",
                "tool_call_id": "tc_1",
                "arguments": {"query": "test"},
            }
        )
        # content_block_stop + content_block_start(tool_use) + content_block_delta(input_json)
        assert len(events) == 3
        assert "content_block_stop" in events[0]
        assert "tool_use" in events[1]
        assert "input_json_delta" in events[2]

    def test_duplicate_tool_start_merges_into_open_tool_block(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        first_events = e.feed(
            {
                "type": "tool_start",
                "tool_name": "render_html",
                "tool_call_id": "call_0",
                "arguments": {},
            }
        )
        second_events = e.feed(
            {
                "type": "tool_start",
                "tool_name": "render_html",
                "tool_call_id": "call_0",
                "arguments": {"code": "<!doctype html><html></html>"},
            }
        )

        first_payloads = [
            json.loads(event.split("data: ")[1]) for event in first_events
        ]
        second_payloads = [
            json.loads(event.split("data: ")[1]) for event in second_events
        ]

        tool_starts = [
            payload
            for payload in first_payloads + second_payloads
            if payload["type"] == "content_block_start"
            and payload["content_block"]["type"] == "tool_use"
        ]
        assert len(tool_starts) == 1
        assert tool_starts[0]["content_block"]["id"].startswith("toolu_")
        assert second_payloads == [
            {
                "type": "content_block_delta",
                "index": tool_starts[0]["index"],
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(
                        {"code": "<!doctype html><html></html>"}
                    ),
                },
            }
        ]

    def test_tool_end_closes_tool_opens_new_text_block(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        start_events = e.feed(
            {
                "type": "tool_start",
                "tool_name": "t",
                "tool_call_id": "tc_1",
                "arguments": {},
            }
        )
        start_payload = next(
            json.loads(event.split("data: ")[1])
            for event in start_events
            if "content_block_start" in event
        )
        tool_use_id = start_payload["content_block"]["id"]
        assert tool_use_id.startswith("toolu_")
        events = e.feed(
            {
                "type": "tool_end",
                "tool_name": "t",
                "tool_call_id": "tc_1",
                "result": "done",
            }
        )
        # content_block_stop (tool) + tool_result + content_block_start (new text)
        assert len(events) == 3
        assert "content_block_stop" in events[0]
        assert "tool_result" in events[1]
        parsed = json.loads(events[1].split("data: ")[1])
        assert parsed["content"] == "done"
        assert parsed["tool_use_id"] == tool_use_id
        assert "content_block_start" in events[2]
        assert '"type": "text"' in events[2]

    def test_finish_emits_stop_events(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        events = e.finish("end_turn")
        # content_block_stop + message_delta + message_stop
        assert len(events) == 3
        assert "content_block_stop" in events[0]
        assert "message_delta" in events[1]
        assert "end_turn" in events[1]
        assert "message_stop" in events[2]

    def test_metadata_captured_in_finish_usage(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed(
            {
                "type": "metadata",
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )
        events = e.finish("end_turn")
        delta_event = [ev for ev in events if "message_delta" in ev][0]
        parsed = json.loads(delta_event.split("data: ")[1])
        assert parsed["usage"]["output_tokens"] == 20

    def test_status_events_ignored(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        events = e.feed({"type": "status", "text": "Searching..."})
        assert events == []

    def test_no_tool_calls_simple_text_flow(self):
        e = AnthropicStreamEmitter()
        start_events = e.start("msg_1", "m")
        content_events = e.feed({"type": "content", "text": "Hello world"})
        meta_events = e.feed(
            {"type": "metadata", "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        )
        end_events = e.finish("end_turn")

        assert len(start_events) == 2
        assert len(content_events) == 1
        assert meta_events == []
        assert len(end_events) == 3

    def test_block_index_increments(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        assert e.block_index == 0
        e.feed(
            {
                "type": "tool_start",
                "tool_name": "t",
                "tool_call_id": "tc_1",
                "arguments": {},
            }
        )
        assert e.block_index == 1
        e.feed(
            {
                "type": "tool_end",
                "tool_name": "t",
                "tool_call_id": "tc_1",
                "result": "ok",
            }
        )
        assert e.block_index == 2

    def test_text_after_tool_resets_prev_text(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed({"type": "content", "text": "Before tool"})
        e.feed(
            {
                "type": "tool_start",
                "tool_name": "t",
                "tool_call_id": "tc_1",
                "arguments": {},
            }
        )
        e.feed(
            {
                "type": "tool_end",
                "tool_name": "t",
                "tool_call_id": "tc_1",
                "result": "ok",
            }
        )
        # After tool_end, prev_text should be reset
        events = e.feed({"type": "content", "text": "After tool"})
        parsed = json.loads(events[0].split("data: ")[1])
        assert parsed["delta"]["text"] == "After tool"


# =====================================================================
# Non-streaming tool response tests
# =====================================================================


class TestAnthropicToolNonStreaming:
    def test_duplicate_tool_start_replaces_provisional_tool_block(self):
        def _run_gen():
            yield {
                "type": "tool_start",
                "tool_name": "render_html",
                "tool_call_id": "call_0",
                "arguments": {},
            }
            yield {
                "type": "tool_start",
                "tool_name": "render_html",
                "tool_call_id": "call_0",
                "arguments": {"code": "<!doctype html><html></html>"},
            }
            yield {
                "type": "tool_end",
                "tool_name": "render_html",
                "tool_call_id": "call_0",
                "result": "Rendered HTML canvas.",
            }

        response = asyncio.run(_anthropic_tool_non_streaming(_run_gen, "msg_1", "m"))
        body = json.loads(response.body)
        tool_blocks = [
            block for block in body["content"] if block["type"] == "tool_use"
        ]

        assert len(tool_blocks) == 1
        assert tool_blocks[0]["type"] == "tool_use"
        assert tool_blocks[0]["id"].startswith("toolu_")
        assert tool_blocks[0]["name"] == "render_html"
        assert tool_blocks[0]["input"] == {"code": "<!doctype html><html></html>"}

    def test_display_strip_gates_on_declared_tools(self):
        # A final answer containing NAME[ARGS]{json} is gated on the declared tools: undeclared
        # ``foo`` markup is prose and survives, the declared web_search rehearsal strips.
        def _run_gen():
            yield {
                "type": "content",
                "text": 'Try foo[ARGS]{"x": 1} but not web_search[ARGS]{"q": "hi"} here.',
            }

        tools = [
            {"type": "function", "function": {"name": "web_search", "parameters": {}}}
        ]
        response = asyncio.run(
            _anthropic_tool_non_streaming(_run_gen, "msg_1", "m", openai_tools = tools)
        )
        body = json.loads(response.body)
        text = "".join(b["text"] for b in body["content"] if b["type"] == "text")
        assert 'foo[ARGS]{"x": 1}' in text  # inactive name preserved as prose
        assert "web_search[ARGS]" not in text  # active name stripped from display


# =====================================================================
# Pass-through emitter tests (client-side tool execution path)
# =====================================================================


class TestAnthropicPassthroughEmitter:
    def _parse(self, event_str):
        return json.loads(event_str.split("data: ")[1])

    def test_start_emits_message_start_only(self):
        e = AnthropicPassthroughEmitter()
        events = e.start("msg_1", "test-model")
        assert len(events) == 1
        assert "message_start" in events[0]
        parsed = self._parse(events[0])
        assert parsed["message"]["id"] == "msg_1"
        assert parsed["message"]["model"] == "test-model"

    def test_text_chunk_opens_text_block_and_emits_delta(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        events = e.feed_chunk(chunk)
        # content_block_start + content_block_delta
        assert len(events) == 2
        assert "content_block_start" in events[0]
        assert '"type": "text"' in events[0]
        delta = self._parse(events[1])
        assert delta["delta"]["type"] == "text_delta"
        assert delta["delta"]["text"] == "Hello"

    def test_sequential_text_chunks_single_block(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        events1 = e.feed_chunk({"choices": [{"delta": {"content": "Hello"}}]})
        events2 = e.feed_chunk({"choices": [{"delta": {"content": " world"}}]})
        # First chunk opens the block, second only emits delta
        assert len(events1) == 2
        assert len(events2) == 1
        assert self._parse(events2[0])["delta"]["text"] == " world"

    def test_tool_call_opens_tool_use_block(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        chunk = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "Bash", "arguments": ""},
                            }
                        ]
                    }
                }
            ]
        }
        events = e.feed_chunk(chunk)
        assert len(events) == 1
        parsed = self._parse(events[0])
        assert parsed["type"] == "content_block_start"
        assert parsed["content_block"]["type"] == "tool_use"
        assert parsed["content_block"]["id"].startswith("toolu_")
        assert parsed["content_block"]["name"] == "Bash"

    def test_tool_call_arguments_streamed_as_input_json_delta(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        # Open the tool call
        e.feed_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        # Stream argument fragments
        events1 = e.feed_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '{"cmd'}}
                            ]
                        }
                    }
                ]
            }
        )
        events2 = e.feed_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '": "ls"}'}}
                            ]
                        }
                    }
                ]
            }
        )
        parsed1 = self._parse(events1[0])
        parsed2 = self._parse(events2[0])
        assert parsed1["delta"]["type"] == "input_json_delta"
        assert parsed1["delta"]["partial_json"] == '{"cmd'
        assert parsed2["delta"]["partial_json"] == '": "ls"}'

    def test_text_then_tool_closes_text_block(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        e.feed_chunk({"choices": [{"delta": {"content": "Let me check."}}]})
        events = e.feed_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": ""},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        # Should close text block and open tool_use block
        assert "content_block_stop" in events[0]
        assert "content_block_start" in events[1]
        assert '"type": "tool_use"' in events[1]

    def test_finish_reason_tool_calls_sets_tool_use_stop(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        e.feed_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        e.feed_chunk({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        events = e.finish()
        delta_event = [ev for ev in events if "message_delta" in ev][0]
        parsed = self._parse(delta_event)
        assert parsed["delta"]["stop_reason"] == "tool_use"

    def test_finish_reason_stop_sets_end_turn(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        e.feed_chunk({"choices": [{"delta": {"content": "Hi"}}]})
        e.feed_chunk({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        events = e.finish()
        delta_event = [ev for ev in events if "message_delta" in ev][0]
        parsed = self._parse(delta_event)
        assert parsed["delta"]["stop_reason"] == "end_turn"

    def test_finish_reason_length_sets_max_tokens(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        e.feed_chunk({"choices": [{"delta": {"content": "Hi"}}]})
        e.feed_chunk({"choices": [{"delta": {}, "finish_reason": "length"}]})
        events = e.finish()
        delta_event = [ev for ev in events if "message_delta" in ev][0]
        parsed = self._parse(delta_event)
        assert parsed["delta"]["stop_reason"] == "max_tokens"

    def test_finish_closes_current_block(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        e.feed_chunk({"choices": [{"delta": {"content": "Hi"}}]})
        events = e.finish()
        assert "content_block_stop" in events[0]
        assert "message_delta" in events[1]
        assert "message_stop" in events[2]

    def test_usage_chunk_captured(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        e.feed_chunk({"choices": [{"delta": {"content": "Hi"}}]})
        e.feed_chunk(
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )
        events = e.finish()
        delta_event = [ev for ev in events if "message_delta" in ev][0]
        parsed = self._parse(delta_event)
        assert parsed["usage"]["output_tokens"] == 5

    def test_empty_chunk_returns_no_events(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        events = e.feed_chunk({"choices": []})
        assert events == []

    def test_no_blocks_at_all_still_produces_valid_finish(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        events = e.finish()
        # No content_block_stop because no block was opened
        assert not any("content_block_stop" in ev for ev in events)
        assert any("message_delta" in ev for ev in events)
        assert any("message_stop" in ev for ev in events)

    def test_multiple_tool_calls_distinct_blocks(self):
        e = AnthropicPassthroughEmitter()
        e.start("msg_1", "m")
        # First tool call
        e.feed_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "Bash", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        # Second tool call (different index)
        events = e.feed_chunk(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "id": "c2",
                                    "type": "function",
                                    "function": {"name": "Read", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            }
        )
        # Should close block 0, open block 1
        assert "content_block_stop" in events[0]
        assert "content_block_start" in events[1]
        parsed = self._parse(events[1])
        assert parsed["content_block"]["name"] == "Read"
        assert parsed["content_block"]["id"].startswith("toolu_")


class TestAnthropicPassthroughStreamAdapter:
    class _Request:
        async def is_disconnected(self):
            return False

    @staticmethod
    async def _collect(response):
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
        return chunks

    @staticmethod
    def _payloads(lines, event_name):
        prefix = f"event: {event_name}\n"
        return [
            json.loads(line.split("data: ", 1)[1].strip())
            for line in lines
            if line.startswith(prefix)
        ]

    def test_stream_requests_usage_for_final_message_delta(self, monkeypatch):
        import routes.inference as inf_mod

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            chunks = [
                {"choices": [{"delta": {"content": "hi"}}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 4,
                        "total_tokens": 6,
                    },
                },
            ]
            content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            content += "data: [DONE]\n\n"
            return httpx.Response(
                200,
                content = content.encode(),
                headers = {"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def _client(*args, **kwargs):
            return real_async_client(
                transport = transport,
                timeout = kwargs.get("timeout", 600),
            )

        monkeypatch.setattr(inf_mod.httpx, "AsyncClient", _client)
        backend = SimpleNamespace(
            base_url = "http://llama.test",
            context_length = 4096,
            count_chat_tokens = lambda *args, **kwargs: 2,
        )

        async def run():
            response = await _anthropic_passthrough_stream(
                self._Request(),
                threading.Event(),
                backend,
                [{"role": "user", "content": "hi"}],
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                0.7,
                0.95,
                20,
                16,
                "msg_1",
                "test-model",
            )
            return await self._collect(response)

        lines = asyncio.run(run())

        assert captured["body"]["stream_options"] == {"include_usage": True}
        message_delta = self._payloads(lines, "message_delta")[0]
        assert message_delta["usage"]["input_tokens"] == 2
        assert message_delta["usage"]["output_tokens"] == 4


# =====================================================================
# Vision guard + PNG normalization (/v1/messages)
# =====================================================================


def _jpeg_data_url() -> str:
    from PIL import Image

    img = Image.new("RGB", (2, 2), (255, 0, 0))
    buf = _BytesIO()
    img.save(buf, format = "JPEG")
    b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class TestNormalizeAnthropicOpenAIImages:
    def test_noop_when_no_images(self):
        msgs = [{"role": "user", "content": "hi"}]
        has_image = _normalize_anthropic_openai_images(msgs, is_vision = False)
        assert has_image is False
        assert msgs == [{"role": "user", "content": "hi"}]

    def test_returns_true_when_image_present(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _jpeg_data_url()}},
                ],
            }
        ]
        assert _normalize_anthropic_openai_images(msgs, is_vision = True) is True

    def test_rejects_image_when_model_not_vision(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _jpeg_data_url()},
                    },
                ],
            }
        ]
        with pytest.raises(HTTPException) as exc:
            _normalize_anthropic_openai_images(msgs, is_vision = False)
        assert exc.value.status_code == 400

    def test_reencodes_jpeg_data_url_to_png(self):
        original_url = _jpeg_data_url()
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "?"},
                    {"type": "image_url", "image_url": {"url": original_url}},
                ],
            }
        ]
        _normalize_anthropic_openai_images(msgs, is_vision = True)
        new_url = msgs[0]["content"][1]["image_url"]["url"]
        assert new_url.startswith("data:image/png;base64,")
        assert new_url != original_url

    def test_remote_url_left_unchanged(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://x.example/y.png"},
                    },
                ],
            }
        ]
        _normalize_anthropic_openai_images(msgs, is_vision = True)
        assert msgs[0]["content"][0]["image_url"]["url"] == "https://x.example/y.png"

    def test_bad_base64_raises_400(self):
        msgs = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,!!!not-b64!!!"},
                    },
                ],
            }
        ]
        with pytest.raises(HTTPException) as exc:
            _normalize_anthropic_openai_images(msgs, is_vision = True)
        assert exc.value.status_code == 400


# =====================================================================
# Unsloth-tool alias detection (/v1/messages tool routing)
# =====================================================================


class TestAnthropicRequestedStudioTools:
    def test_recognizes_server_tool_by_type(self):
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        assert _anthropic_requested_studio_tools(tools) == {"web_search"}

    def test_bare_name_without_type_is_not_treated_as_server_tool(self):
        # Anthropic dispatches server tools by `type`; bare-name matching
        # would let a malformed client tool (missing input_schema) silently
        # flip the request into server-execution mode.
        tools = [{"name": "python"}]
        assert _anthropic_requested_studio_tools(tools) == set()

    def test_client_tool_named_python_is_not_misclassified(self):
        # input_schema is the client-tool discriminator; its presence must
        # prevent the name from being treated as an Unsloth alias.
        tools = [
            {
                "name": "python",
                "description": "user's own python",
                "input_schema": {"type": "object"},
            }
        ]
        assert _anthropic_requested_studio_tools(tools) == set()

    def test_mixed_request_only_extracts_server_tools(self):
        tools = [
            {"type": "web_search_20250305", "name": "web_search"},
            {"name": "custom_tool", "input_schema": {"type": "object"}},
        ]
        assert _anthropic_requested_studio_tools(tools) == {"web_search"}

    def test_pydantic_model_input(self):
        tools = [
            AnthropicTool(type = "web_fetch_20250910", name = "web_fetch"),
            AnthropicTool(name = "x", input_schema = {"type": "object"}),
        ]
        assert _anthropic_requested_studio_tools(tools) == {"web_search"}

    def test_empty_and_none(self):
        assert _anthropic_requested_studio_tools(None) == set()
        assert _anthropic_requested_studio_tools([]) == set()


# =====================================================================
# Route-level tool routing (/v1/messages)
# =====================================================================


def _mock_backend(monkeypatch, **overrides):
    """Install a minimal stub backend on routes.inference.

    Generation methods record which path the route entered, then yield one
    content event so the route can complete normally.
    """
    import routes.inference as inf_mod

    calls = []

    def _gen_plain(**kwargs):
        calls.append(("plain", kwargs))
        yield "ok"

    def _gen_tools(**kwargs):
        calls.append(("tools", kwargs))
        yield {"type": "content", "text": "ok"}

    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        model_identifier = "test-model",
        context_length = 4096,
        count_chat_tokens = lambda *args, **kwargs: 2,
        generate_chat_completion = _gen_plain,
        generate_chat_completion_with_tools = _gen_tools,
        calls = calls,
    )
    backend.__dict__.update(overrides)
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    return backend


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _basic_payload(**fields) -> AnthropicMessagesRequest:
    base = {
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(fields)
    return AnthropicMessagesRequest(**base)


@pytest.fixture(autouse = True)
def _reset_policy():
    reset_tool_policy()
    yield
    reset_tool_policy()


class TestAnthropicMessagesToolRouting:
    class _Request:
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/messages")
        method = "POST"

        async def is_disconnected(self):
            return False

    @staticmethod
    def _consume_response(response):
        async def _consume():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return chunks

        return _drive(_consume())

    def test_plain_non_streaming_records_api_monitor_entry(self, monkeypatch):
        import routes.inference as inf_mod

        _mock_backend(monkeypatch, context_length = 2048)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        payload = _basic_payload()

        response = _drive(
            anthropic_messages(payload, request = self._Request(), current_subject = "t")
        )

        assert response.status_code == 200
        [entry] = monitor.snapshot()
        assert entry["endpoint"] == "/v1/messages"
        assert entry["status"] == "completed"
        assert entry["model"] == "test-model"
        assert entry["prompt_preview"] == "user: hi"
        assert entry["reply_preview"] == "ok"
        assert entry["context_length"] == 2048
        assert monitor.active_count() == 0

    def test_tool_use_non_streaming_records_api_monitor_reply(self, monkeypatch):
        import routes.inference as inf_mod

        def _gen_tools(**_kwargs):
            yield {
                "type": "tool_start",
                "tool_call_id": "call_1",
                "tool_name": "lookup",
                "arguments": {"query": "weather"},
            }

        _mock_backend(
            monkeypatch,
            context_length = 2048,
            generate_chat_completion_with_tools = _gen_tools,
        )
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        payload = _basic_payload(
            enable_tools = True,
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        response = _drive(
            anthropic_messages(payload, request = self._Request(), current_subject = "t")
        )

        assert response.status_code == 200
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply_preview"] == 'Tool call: lookup({"query": "weather"})'

    def test_plain_streaming_records_active_and_completed_monitor_entry(
        self, monkeypatch
    ):
        import routes.inference as inf_mod

        _mock_backend(monkeypatch, context_length = 2048)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        payload = _basic_payload(stream = True)

        response = _drive(
            anthropic_messages(payload, request = self._Request(), current_subject = "t")
        )

        assert monitor.active_count() == 1
        self._consume_response(response)
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply_preview"] == "ok"
        assert entry["prompt_tokens"] == 2
        assert entry["context_length"] == 2048
        assert monitor.active_count() == 0

    def test_plain_streaming_pre_response_cancel_finalizes_monitor(self, monkeypatch):
        import routes.inference as inf_mod

        async def _cancelled_before_response(*_args, **_kwargs):
            raise asyncio.CancelledError()

        _mock_backend(monkeypatch, context_length = 2048)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(
            inf_mod, "_anthropic_plain_stream", _cancelled_before_response
        )
        payload = _basic_payload(stream = True)

        with pytest.raises(asyncio.CancelledError):
            _drive(
                anthropic_messages(
                    payload, request = self._Request(), current_subject = "t"
                )
            )

        [entry] = monitor.snapshot()
        assert entry["status"] == "cancelled"
        assert monitor.active_count() == 0

    @staticmethod
    def _sse_blob(chunks):
        # StreamingResponse may hand back str or already-encoded bytes.
        return "".join(
            c.decode() if isinstance(c, (bytes, bytearray)) else c for c in chunks
        )

    def test_plain_streaming_unclassified_error_emits_error_event(self, monkeypatch):
        # An unclassified mid-stream failure must surface as an SSE `error` event
        # and stop, not a message_stop that masks a truncated turn as clean.
        def _gen_boom(**_kwargs):
            yield "partial"
            raise RuntimeError("llama-server crashed mid-decode")

        _mock_backend(monkeypatch, generate_chat_completion = _gen_boom)
        payload = _basic_payload(stream = True)

        response = _drive(
            anthropic_messages(payload, request = self._Request(), current_subject = "t")
        )
        blob = self._sse_blob(self._consume_response(response))

        assert "event: error" in blob
        assert '"type": "error"' in blob
        assert "event: message_stop" not in blob

    def test_tool_streaming_unclassified_error_emits_error_event(self, monkeypatch):
        # Same guarantee on the tool-calling stream path.
        def _gen_tools_boom(**_kwargs):
            yield {"type": "content", "text": "partial"}
            raise RuntimeError("llama-server crashed mid-decode")

        _mock_backend(monkeypatch, generate_chat_completion_with_tools = _gen_tools_boom)
        payload = _basic_payload(
            stream = True,
            enable_tools = True,
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        response = _drive(
            anthropic_messages(payload, request = self._Request(), current_subject = "t")
        )
        blob = self._sse_blob(self._consume_response(response))

        assert "event: error" in blob
        assert '"type": "error"' in blob
        assert "event: message_stop" not in blob

    def test_mixed_server_and_client_tools_rejected_with_400(self, monkeypatch):
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "custom", "input_schema": {"type": "object"}},
            ],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "Mixing Anthropic server tools" in exc.value.detail

    def test_mixed_rejected_when_client_tool_name_collides_with_server_alias(
        self, monkeypatch
    ):
        # Regression: a client tool sharing a name with a mapped server tool
        # (e.g. a custom "web_search") must still trigger the mixed-mode 400;
        # otherwise the post-name filter drops the client tool and silently
        # routes to server-only.
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [
                {"type": "web_search_20250305", "name": "web_search"},
                {"name": "web_search", "input_schema": {"type": "object"}},
            ],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "Mixing Anthropic server tools" in exc.value.detail

    def test_client_tool_missing_input_schema_rejected_with_400(self, monkeypatch):
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"name": "my_tool", "description": "oops, schema typo"}],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "input_schema" in exc.value.detail

    def test_client_tool_missing_name_rejected_with_400(self, monkeypatch):
        # Regression: AnthropicTool.name was relaxed to Optional for server
        # tools, so a client-tool payload with input_schema but no `name`
        # (typo) now parses but would be silently dropped by
        # anthropic_tools_to_openai, leaving tool calling disabled. Reject at
        # the boundary instead.
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"input_schema": {"type": "object"}}],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "name" in exc.value.detail

    def test_client_tool_empty_name_rejected_with_400(self, monkeypatch):
        # Same silent-disable class as missing-name: `name: ""` passes the
        # isinstance check but is dropped by anthropic_tools_to_openai's
        # `if not name` guard. Reject at the boundary so the typo shows.
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"name": "", "input_schema": {"type": "object"}}],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "name" in exc.value.detail

    def test_alias_named_client_tool_without_schema_rejected_with_400(
        self, monkeypatch
    ):
        # Regression: a typo'd client tool whose name collides with an Unsloth
        # alias (e.g. a custom "python" tool missing input_schema) must
        # surface a 400, not silently switch into Unsloth's built-in python
        # execution.
        _mock_backend(monkeypatch)
        payload = _basic_payload(tools = [{"name": "python"}])

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "input_schema" in exc.value.detail

    def test_unrecognized_server_tool_accepted_as_noop(self, monkeypatch):
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"type": "code_execution_20250825", "name": "code_execution"}],
        )

        _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert backend.calls[0][0] == "plain"

    def test_disable_tools_policy_overrides_server_tool_alias(self, monkeypatch):
        # CLI `unsloth run --disable-tools` sets policy=False. A request with
        # an Unsloth server-tool alias must NOT enter the agentic loop then.
        backend = _mock_backend(monkeypatch)
        set_tool_policy(False)
        payload = _basic_payload(
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert backend.calls[0][0] == "plain"

    def test_server_tool_alias_enters_tool_path_when_policy_unset(self, monkeypatch):
        # Mirror of the previous test for the default (None) policy. An omitted
        # permission_mode still runs here because web_search is a safe server tool
        # (only a selected terminal/python would require the missing gate).
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert backend.calls[0][0] == "tools"

    def test_confirm_tool_calls_rejected_for_server_tools(self, monkeypatch):
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(
            confirm_tool_calls = True,
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert (
            "confirm_tool_calls is not supported"
            in exc.value.detail["error"]["message"]
        )
        assert backend.calls == []

    def test_permission_mode_gating_for_server_tools(self, monkeypatch):
        # ask is a request for a per-call pause this channel cannot honor, so it is
        # always rejected, even for a safe-only server tool (web_search).
        safe_tools = [{"type": "web_search_20250305", "name": "web_search"}]
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(tools = safe_tools, permission_mode = "ask")
        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "no confirmation channel" in exc.value.detail["error"]["message"]
        assert backend.calls == []

        # auto only gates unsafe calls, so a safe-only selection runs (nothing to
        # gate), like the omitted default. Both keep existing callers working.
        for extra in ({"permission_mode": "auto"}, {}):
            backend = _mock_backend(monkeypatch)
            payload = _basic_payload(tools = safe_tools, **extra)
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
            assert backend.calls[0][0] == "tools"

        # But auto or an omitted mode that would run a local tool (terminal/python,
        # via a bare Anthropic tool type or enabled_tools) is rejected, since that
        # tool could need the gate this channel lacks.
        for local_payload in (
            _basic_payload(tools = [{"type": "terminal", "name": "terminal"}]),
            _basic_payload(
                tools = [{"type": "terminal", "name": "terminal"}], permission_mode = "auto"
            ),
            _basic_payload(
                tools = safe_tools, enable_tools = True, enabled_tools = ["python"]
            ),
        ):
            backend = _mock_backend(monkeypatch)
            with pytest.raises(HTTPException) as exc:
                _drive(
                    anthropic_messages(local_payload, request = None, current_subject = "t")
                )
            assert exc.value.status_code == 400
            assert "terminal" in exc.value.detail["error"]["message"]
            assert backend.calls == []

        # off, full, and a legacy confirm_tool_calls=False opt-out all run, even
        # with a local tool selected. The explicit opt-out wins over the mode
        # (mirrors _permission_mode_confirm and the GGUF path), so it runs even
        # under ask, which otherwise always rejects.
        for extra in (
            {"tools": safe_tools, "permission_mode": "off"},
            {"tools": safe_tools, "permission_mode": "full"},
            {
                "tools": safe_tools,
                "enabled_tools": ["python"],
                "confirm_tool_calls": False,
            },
            {
                "tools": safe_tools,
                "permission_mode": "ask",
                "confirm_tool_calls": False,
            },
            {
                "tools": [{"type": "terminal", "name": "terminal"}],
                "permission_mode": "ask",
                "confirm_tool_calls": False,
            },
        ):
            backend = _mock_backend(monkeypatch)
            payload = _basic_payload(**extra)
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
            assert backend.calls[0][0] == "tools"

    def test_render_html_gated_for_server_tools(self, monkeypatch):
        # render_html is no longer unconditionally safe: a networked canvas prompts
        # in auto and this channel cannot present that gate, so selecting it under
        # ask/auto/omitted rejects like terminal/python; off/full (and an explicit
        # confirm opt-out) run it.
        rh = {"enable_tools": True, "enabled_tools": ["render_html"]}
        for mode in ("ask", "auto", None):
            backend = _mock_backend(monkeypatch)
            fields = dict(rh)
            if mode is not None:
                fields["permission_mode"] = mode
            payload = _basic_payload(**fields)
            with pytest.raises(HTTPException) as exc:
                _drive(anthropic_messages(payload, request = None, current_subject = "t"))
            assert exc.value.status_code == 400
            assert "no confirmation channel" in exc.value.detail["error"]["message"]
            assert backend.calls == []
        for extra in (
            {"permission_mode": "off"},
            {"permission_mode": "full"},
            {"confirm_tool_calls": False},
        ):
            backend = _mock_backend(monkeypatch)
            payload = _basic_payload(**{**rh, **extra})
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
            assert backend.calls[0][0] == "tools"

    def test_permission_mode_rejected_before_auto_switch(self, monkeypatch):
        # The unsupported-mode rejection must run before _maybe_auto_switch_model,
        # so an invalid confirm-gated request never evicts the resident model
        # (mirrors the pre-switch malformed- and mixed-tool guards).
        import routes.inference as inf_mod

        switch_calls = []

        async def _rec_switch(*_args, **_kwargs):
            switch_calls.append(1)

        monkeypatch.setattr(inf_mod, "_maybe_auto_switch_model", _rec_switch)
        safe_tools = [{"type": "web_search_20250305", "name": "web_search"}]
        local_tools = [{"type": "terminal", "name": "terminal"}]

        # ask (any server tool), auto with a local tool, and an omitted mode
        # selecting a local tool are all rejected up front, before the switch runs.
        for payload in (
            _basic_payload(tools = safe_tools, permission_mode = "ask"),
            _basic_payload(tools = local_tools, permission_mode = "auto"),
            _basic_payload(tools = local_tools),
        ):
            switch_calls.clear()
            _mock_backend(monkeypatch)
            with pytest.raises(HTTPException) as exc:
                _drive(anthropic_messages(payload, request = None, current_subject = "t"))
            assert exc.value.status_code == 400
            assert switch_calls == [], "rejection must precede the auto-switch"

        # A supported request (off) still reaches the switch and runs the loop.
        switch_calls.clear()
        _mock_backend(monkeypatch)
        payload = _basic_payload(tools = safe_tools, permission_mode = "off")
        _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert switch_calls == [1]

    def test_per_request_enable_tools_false_blocks_server_tool_alias(self, monkeypatch):
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(
            enable_tools = False,
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert backend.calls[0][0] == "plain"


def test_resumed_session_thinking_and_null_content_do_not_400():
    # A resumed session replays assistant turns with `thinking` (and sometimes null)
    # content. Those must be accepted (thinking dropped by the converter), not 400ed.
    from pydantic import ValidationError

    req = AnthropicMessagesRequest(
        model = "x",
        max_tokens = 16,
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "secret reasoning",
                        "signature": "s",
                    },
                    {"type": "text", "text": "the answer"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                ],
            },
            {"role": "assistant", "content": None},  # tool-only turn serialized as null
        ],
    )
    # Known blocks still parse as their typed models; only the unknown one is loose.
    assert type(req.messages[1].content[0]).__name__ == "AnthropicUnknownBlock"
    assert type(req.messages[1].content[1]).__name__ == "AnthropicTextBlock"
    assert req.messages[2].content == ""  # null coerced

    openai = anthropic_messages_to_openai([m.model_dump() for m in req.messages])
    assistant = next(m for m in openai if m["role"] == "assistant" and m.get("content"))
    assert assistant["content"] == "the answer"
    assert "secret reasoning" not in json.dumps(openai)  # thinking never forwarded

    # A malformed KNOWN block still fails cleanly instead of being swallowed.
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest(
            model = "x",
            max_tokens = 16,
            messages = [
                {"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]}
            ],
        )


def test_user_null_content_rejected():
    # The null->"" leniency is assistant-only; a null user content must be rejected
    # at the boundary, not coerced into an empty prompt and forwarded to the model.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest(
            model = "x",
            max_tokens = 16,
            messages = [{"role": "user", "content": None}],
        )


def test_user_unknown_block_rejected_not_silently_dropped():
    # The converter skips user blocks it cannot translate, so a user turn whose only
    # block is unknown would validate yet forward no content. Reject at the boundary
    # to avoid that silent data loss (the assistant fallback is unaffected).
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest(
            model = "x",
            max_tokens = 16,
            messages = [
                {"role": "user", "content": [{"type": "document", "source": {}}]},
            ],
        )


def test_user_translatable_blocks_still_accepted():
    # text / image / tool_result are translatable, so a real user message built from
    # them must still pass; the unknown-block guard only trips on other types.
    req = AnthropicMessagesRequest(
        model = "x",
        max_tokens = 16,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "AA",
                        },
                    },
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                ],
            }
        ],
    )
    assert [type(b).__name__ for b in req.messages[0].content] == [
        "AnthropicTextBlock",
        "AnthropicImageBlock",
        "AnthropicToolResultBlock",
    ]

    openai = anthropic_messages_to_openai([m.model_dump() for m in req.messages])
    assert any(m["role"] == "tool" and m["tool_call_id"] == "t1" for m in openai)


def test_user_malformed_known_block_still_rejected():
    # The guard only allow-lists a user block's *type*; the union still validates its
    # shape, so a known-but-malformed block (tool_result without tool_use_id) fails.
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        AnthropicMessagesRequest(
            model = "x",
            max_tokens = 16,
            messages = [
                {"role": "user", "content": [{"type": "tool_result", "content": "x"}]},
            ],
        )


def test_user_content_block_non_string_type_rejected_cleanly():
    # A user block whose `type` is a non-string (unhashable list / dict, or a stray
    # int) must fail as a clean validation error, not raise TypeError from the
    # frozenset membership test and escape as a 500.
    from pydantic import ValidationError
    for bad_type in ([], {}, 5):
        with pytest.raises(ValidationError):
            AnthropicMessagesRequest(
                model = "x",
                max_tokens = 16,
                messages = [{"role": "user", "content": [{"type": bad_type}]}],
            )


def test_assistant_missing_content_key_still_rejected():
    # The null -> "" leniency is only for an EXPLICIT null. An assistant message that
    # omits content entirely stays malformed and must fail required-field validation.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AnthropicMessagesRequest(
            model = "x",
            max_tokens = 16,
            messages = [{"role": "assistant"}],
        )
    # An explicit null is still accepted and coerced (regression guard).
    req = AnthropicMessagesRequest(
        model = "x",
        max_tokens = 16,
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ],
    )
    assert req.messages[1].content == ""


def test_resumed_null_assistant_between_users_coalesced_on_messages_route(monkeypatch):
    # user -> assistant(null) -> user is now accepted: the null assistant turn coerces
    # to "" and is dropped. The route must then coalesce the two remaining user turns
    # so a strict GGUF chat template does not 400 on non-alternating roles.
    backend = _mock_backend(monkeypatch, context_length = 2048)

    class _Req:
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/messages")
        method = "POST"

        async def is_disconnected(self):
            return False

    payload = AnthropicMessagesRequest(
        model = "x",
        max_tokens = 16,
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "please continue"},
        ],
    )

    response = _drive(anthropic_messages(payload, request = _Req(), current_subject = "t"))
    assert response.status_code == 200

    [(_path, kwargs)] = backend.calls
    user_turns = [m for m in kwargs["messages"] if m.get("role") == "user"]
    assert len(user_turns) == 1  # the two user turns were merged, not left adjacent
    merged = user_turns[0]["content"]
    if isinstance(merged, list):
        merged = " ".join(p.get("text", "") for p in merged if isinstance(p, dict))
    assert "first question" in merged and "please continue" in merged


def test_disable_parallel_tool_use_forwards_heartbeats_while_dropping():
    """Heartbeats from a parallel-disabled, dropped tool call must still reach
    the client as SSE keepalives: the dropped call runs server-side and the
    stall keepalive never fires while the generator keeps producing events, so
    swallowing them recreates the silent window keepalives exist to prevent."""
    import threading as _threading

    from routes.inference import (
        _OPENAI_PASSTHROUGH_SSE_KEEPALIVE,
        _anthropic_tool_stream,
    )

    def run_gen():
        def gen():
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "call_0",
                "arguments": {},
            }
            yield {"type": "heartbeat"}
            yield {
                "type": "tool_end",
                "tool_name": "python",
                "tool_call_id": "call_0",
                "result": "r1",
            }
            # Second call: dropped by disable_parallel_tool_use, still executed
            # server-side (heartbeats + live output).
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "call_1",
                "arguments": {},
            }
            yield {"type": "heartbeat"}
            yield {
                "type": "tool_output",
                "tool_name": "python",
                "tool_call_id": "call_1",
                "text": "x",
            }
            yield {"type": "heartbeat"}
            yield {
                "type": "tool_end",
                "tool_name": "python",
                "tool_call_id": "call_1",
                "result": "r2",
            }
            yield {"type": "content", "text": "final answer"}

        return gen()

    async def _drive():
        async def _is_disconnected():
            return False

        request = SimpleNamespace(is_disconnected = _is_disconnected)
        resp = await _anthropic_tool_stream(
            request,
            _threading.Event(),
            run_gen,
            "msg_hb",
            "m",
            disable_parallel_tool_use = True,
        )
        return [chunk async for chunk in resp.body_iterator]

    chunks = asyncio.run(_drive())
    keepalives = [c for c in chunks if c == _OPENAI_PASSTHROUGH_SSE_KEEPALIVE]
    # One heartbeat inside the kept call, two inside the dropped window.
    assert len(keepalives) >= 3
    # The dropped call must not surface as a second tool_use block.
    tool_use_starts = [
        c for c in chunks if "content_block_start" in c and '"tool_use"' in c
    ]
    assert len(tool_use_starts) == 1


def test_dropped_tool_output_events_emit_rate_limited_keepalives(monkeypatch):
    """A chatty tool streaming tool_output/tool_args with no heartbeats keeps the
    generator busy (stall keepalive never fires); the Anthropic path can't
    translate those events and drops them. Dropping silently would let an idle
    proxy kill the stream, so the drop branch emits a rate-limited keepalive."""
    import threading as _threading

    import routes.inference as inf_mod
    from routes.inference import (
        _OPENAI_PASSTHROUGH_SSE_KEEPALIVE,
        _anthropic_tool_stream,
    )

    # Deterministic clock: only the drop-branch keepalive uses time.monotonic
    # here, so jumping past the stall window per call makes each dropped event
    # cross the rate-limit threshold. asyncio.wait uses the loop clock and
    # next(gen) returns promptly, so the outer stall keepalive never fires --
    # every keepalive here is from the drop branch.
    _real_time = inf_mod.time
    _tick = {"v": 0.0}

    def _fast_monotonic():
        _tick["v"] += 100.0
        return _tick["v"]

    fake_time = SimpleNamespace(
        monotonic = _fast_monotonic,
        sleep = _real_time.sleep,
        time = _real_time.time,
        perf_counter = _real_time.perf_counter,
    )
    monkeypatch.setattr(inf_mod, "time", fake_time)

    n_output = 4

    def run_gen():
        def gen():
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "call_0",
                "arguments": {},
            }
            # Chatty streamed stdout, no heartbeats.
            for i in range(n_output):
                yield {
                    "type": "tool_output",
                    "tool_name": "python",
                    "tool_call_id": "call_0",
                    "text": f"line {i}\n",
                }
            yield {
                "type": "tool_end",
                "tool_name": "python",
                "tool_call_id": "call_0",
                "result": "done",
            }
            yield {"type": "content", "text": "final answer"}

        return gen()

    async def _drive():
        async def _is_disconnected():
            return False

        request = SimpleNamespace(is_disconnected = _is_disconnected)
        resp = await _anthropic_tool_stream(
            request,
            _threading.Event(),
            run_gen,
            "msg_drop_ka",
            "m",
        )
        return [chunk async for chunk in resp.body_iterator]

    chunks = asyncio.run(_drive())
    keepalives = [c for c in chunks if c == _OPENAI_PASSTHROUGH_SSE_KEEPALIVE]
    assert len(keepalives) == n_output
    # Final answer still reaches the client (drop is transport-only).
    assert any("final answer" in c for c in chunks)


def test_parallel_disabled_dropped_call_output_emits_rate_limited_keepalives(
    monkeypatch,
):
    """Under disable_parallel_tool_use a chatty second call is dropped whole
    (drop_until_tool_end). Its tool_output/tool_args events must still emit
    rate-limited keepalives: the drop window can last minutes with no heartbeats
    and no stall keepalive, so swallowing them silently would let an idle proxy
    kill the stream. The keepalive branch runs before the drop skip."""
    import threading as _threading

    import routes.inference as inf_mod
    from routes.inference import (
        _OPENAI_PASSTHROUGH_SSE_KEEPALIVE,
        _anthropic_tool_stream,
    )

    # Deterministic clock: jumps past the stall window per call (see sibling test).
    _real_time = inf_mod.time
    _tick = {"v": 0.0}

    def _fast_monotonic():
        _tick["v"] += 100.0
        return _tick["v"]

    fake_time = SimpleNamespace(
        monotonic = _fast_monotonic,
        sleep = _real_time.sleep,
        time = _real_time.time,
        perf_counter = _real_time.perf_counter,
    )
    monkeypatch.setattr(inf_mod, "time", fake_time)

    n_output = 4

    def run_gen():
        def gen():
            # First (kept) call.
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "call_0",
                "arguments": {},
            }
            yield {
                "type": "tool_end",
                "tool_name": "python",
                "tool_call_id": "call_0",
                "result": "r1",
            }
            # Second call: dropped whole by disable_parallel_tool_use but still
            # executed server-side, streaming chatty stdout with no heartbeats.
            yield {
                "type": "tool_start",
                "tool_name": "python",
                "tool_call_id": "call_1",
                "arguments": {},
            }
            for i in range(n_output):
                yield {
                    "type": "tool_output",
                    "tool_name": "python",
                    "tool_call_id": "call_1",
                    "text": f"line {i}\n",
                }
            yield {
                "type": "tool_end",
                "tool_name": "python",
                "tool_call_id": "call_1",
                "result": "r2",
            }
            yield {"type": "content", "text": "final answer"}

        return gen()

    async def _drive():
        async def _is_disconnected():
            return False

        request = SimpleNamespace(is_disconnected = _is_disconnected)
        resp = await _anthropic_tool_stream(
            request,
            _threading.Event(),
            run_gen,
            "msg_drop_ka2",
            "m",
            disable_parallel_tool_use = True,
        )
        return [chunk async for chunk in resp.body_iterator]

    chunks = asyncio.run(_drive())
    keepalives = [c for c in chunks if c == _OPENAI_PASSTHROUGH_SSE_KEEPALIVE]
    assert len(keepalives) == n_output
    # The dropped call must not surface as a second tool_use block.
    tool_use_starts = [
        c for c in chunks if "content_block_start" in c and '"tool_use"' in c
    ]
    assert len(tool_use_starts) == 1
    assert any("final answer" in c for c in chunks)


def test_plain_stream_emits_keepalive_during_prompt_stall(monkeypatch):
    """No-tool Anthropic stream must emit SSE keepalives while a long prompt
    prefill blocks next(gen), matching the tool stream (finding 5). The old
    single unbounded to_thread(next, ...) could sit silent past a proxy idle cap."""
    import threading as _threading
    import time as _time

    from routes import inference as inf_mod
    from routes.inference import (
        _OPENAI_PASSTHROUGH_SSE_KEEPALIVE,
        _anthropic_plain_stream,
    )

    monkeypatch.setattr(inf_mod, "_LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S", 0.05)

    def run_gen():
        def gen():
            _time.sleep(0.24)  # stall past several shortened keepalive windows
            yield "hello world"

        return gen()

    async def _drive():
        async def _is_disconnected():
            return False

        request = SimpleNamespace(is_disconnected = _is_disconnected)
        resp = await _anthropic_plain_stream(
            request, _threading.Event(), run_gen, "msg_plain_ka", "m"
        )
        return [chunk async for chunk in resp.body_iterator]

    chunks = asyncio.run(_drive())
    keepalives = [c for c in chunks if c == _OPENAI_PASSTHROUGH_SSE_KEEPALIVE]
    assert len(keepalives) >= 2
    assert any("hello world" in c for c in chunks)


def test_plain_stream_closes_generator_on_disconnect():
    """On disconnect the no-tool teardown must drain any pending worker and close
    the generator (finding 6). The old finally only stopped the disconnect
    watcher, leaking the generator. A fake generator records close() so the
    teardown is asserted deterministically, not via GC."""
    import threading as _threading

    from routes.inference import _anthropic_plain_stream

    closed = _threading.Event()

    class _FakeGen:
        def __init__(self):
            self._items = iter(["tok0", "tok1", "tok2", "tok3"])

        def __next__(self):
            return next(self._items)

        def close(self):
            closed.set()

    def run_gen():
        return _FakeGen()

    state = {"disconnected": False}

    async def _drive():
        async def _is_disconnected():
            return state["disconnected"]

        request = SimpleNamespace(is_disconnected = _is_disconnected)
        resp = await _anthropic_plain_stream(
            request, _threading.Event(), run_gen, "msg_plain_close", "m"
        )
        out = []
        async for chunk in resp.body_iterator:
            out.append(chunk)
            if "tok0" in chunk:
                # Client drops after the first token; the next loop turn tears down.
                state["disconnected"] = True
        return out

    asyncio.run(_drive())
    assert closed.is_set()
