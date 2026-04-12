# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the Anthropic Messages API schemas and translation layer.
No running server or GPU required.
"""

import sys
import os
import json

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

    def test_tools_field_parses(self):
        req = AnthropicMessagesRequest(
            max_tokens = 100,
            messages = [{"role": "user", "content": "Hi"}],
            tools = [{"name": "web_search", "input_schema": {"type": "object"}}],
        )
        assert len(req.tools) == 1
        assert req.tools[0].name == "web_search"

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

    def test_tool_end_closes_tool_opens_new_text_block(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed(
            {
                "type": "tool_start",
                "tool_name": "t",
                "tool_call_id": "tc_1",
                "arguments": {},
            }
        )
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
        assert parsed["tool_use_id"] == "tc_1"
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
