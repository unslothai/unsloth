# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the Anthropic Messages API schemas and translation layer.
No running server or GPU required.
"""

import sys
import os
import json

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
from routes.inference import (
    _normalize_anthropic_openai_images,
    _select_anthropic_server_tools,
    _anthropic_requested_studio_tools,
    _anthropic_tool_non_streaming,
    anthropic_messages,
)
from state.tool_policy import reset_tool_policy, set_tool_policy
from fastapi import HTTPException
import asyncio
import base64 as _b64
from io import BytesIO as _BytesIO
from types import SimpleNamespace


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
        assert [p["type"] for p in parts] == [
            "text",
            "image_url",
            "text",
            "image_url",
        ]
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

        assert [tool["function"]["name"] for tool in result] == [
            "web_search",
            "python",
        ]

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
        assert tool_starts[0]["content_block"]["id"] == "call_0"
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
                "result": "Rendered HTML artifact.",
            }

        response = asyncio.run(_anthropic_tool_non_streaming(_run_gen, "msg_1", "m"))
        body = json.loads(response.body)
        tool_blocks = [
            block for block in body["content"] if block["type"] == "tool_use"
        ]

        assert tool_blocks == [
            {
                "type": "tool_use",
                "id": "call_0",
                "name": "render_html",
                "input": {"code": "<!doctype html><html></html>"},
            }
        ]


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
        assert parsed["content_block"]["id"] == "call_1"
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
        assert parsed["content_block"]["id"] == "c2"


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
# Studio-tool alias detection (/v1/messages tool routing)
# =====================================================================


class TestAnthropicRequestedStudioTools:
    def test_recognizes_server_tool_by_type(self):
        tools = [{"type": "web_search_20250305", "name": "web_search"}]
        assert _anthropic_requested_studio_tools(tools) == {"web_search"}

    def test_bare_name_without_type_is_not_treated_as_server_tool(self):
        # Anthropic dispatches server tools by `type`; bare-name matching
        # would let a malformed client tool (e.g. user forgot input_schema)
        # silently flip the request into server-execution mode.
        tools = [{"name": "python"}]
        assert _anthropic_requested_studio_tools(tools) == set()

    def test_client_tool_named_python_is_not_misclassified(self):
        # input_schema is the client-tool discriminator; presence of it
        # must prevent the name from being treated as a Studio alias.
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


class _PlainPathCalled(Exception):
    pass


class _ToolPathCalled(Exception):
    pass


def _mock_backend(monkeypatch, **overrides):
    """Install a minimal stub backend on routes.inference.

    Generation methods raise sentinel exceptions so the caller can assert
    which path the route entered.
    """
    import routes.inference as inf_mod

    def _gen_plain(**kwargs):
        raise _PlainPathCalled()

    def _gen_tools(**kwargs):
        raise _ToolPathCalled()

    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        model_identifier = "test-model",
        generate_chat_completion = _gen_plain,
        generate_chat_completion_with_tools = _gen_tools,
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
        # Regression: a client tool sharing a name with a mapped server
        # tool (e.g. user defines their own "web_search") must still
        # trigger the mixed-mode 400 — the post-name filter would
        # otherwise drop the client tool and silently route to server-only.
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
        # tools, so a client-tool payload that has input_schema but omits
        # `name` (e.g. typo) now parses successfully but would be silently
        # dropped by anthropic_tools_to_openai, leaving the request with
        # tool calling disabled. Reject at the boundary instead.
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
        # `if not name` guard. Reject at the boundary so the typo surfaces.
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
        # Regression: a typo'd client tool whose name happens to collide
        # with a Studio alias (e.g. user meant a custom "python" tool but
        # forgot input_schema) must surface a 400, not silently switch
        # the request into Studio's built-in python execution.
        _mock_backend(monkeypatch)
        payload = _basic_payload(tools = [{"name": "python"}])

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "input_schema" in exc.value.detail

    def test_unrecognized_server_tool_accepted_as_noop(self, monkeypatch):
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"type": "code_execution_20250825", "name": "code_execution"}],
        )

        with pytest.raises(_PlainPathCalled):
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))

    def test_disable_tools_policy_overrides_server_tool_alias(self, monkeypatch):
        # CLI `unsloth run --disable-tools` sets policy=False. A request
        # carrying a Studio server-tool alias must NOT enter the agentic
        # loop in that configuration.
        _mock_backend(monkeypatch)
        set_tool_policy(False)
        payload = _basic_payload(
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        with pytest.raises(_PlainPathCalled):
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))

    def test_server_tool_alias_enters_tool_path_when_policy_unset(self, monkeypatch):
        # Mirror of the previous test for the default (None) policy.
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        with pytest.raises(_ToolPathCalled):
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))

    def test_per_request_enable_tools_false_blocks_server_tool_alias(self, monkeypatch):
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            enable_tools = False,
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        with pytest.raises(_PlainPathCalled):
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
