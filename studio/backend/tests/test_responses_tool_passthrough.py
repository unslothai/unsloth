# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the OpenAI /v1/responses client-side function-calling pass-through.

Covers:
- ResponsesRequest accepts Responses-shape `tools`, `tool_choice`,
  `parallel_tool_calls`, and the `function_call` / `function_call_output`
  input items used for multi-turn tool loops.
- _translate_responses_tools_to_chat() converts the flat Responses tool
  shape to the nested Chat Completions shape, drops non-function built-in
  tools, and returns None for empty lists.
- _translate_responses_tool_choice_to_chat() passes string choices through
  and converts {type:function,name:X} to Chat Completions' nested shape.
- _normalise_responses_input() maps function_call_output items to
  role="tool" ChatMessages with tool_call_id, and function_call items to
  assistant messages with tool_calls.
- _chat_tool_calls_to_responses_output() preserves call_id and drops
  non-function tool calls.
- ResponsesOutputFunctionCall and ResponsesResponse round-trip tool-call
  outputs without losing fields.

No running server or GPU required.
"""

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import json

import pytest
from pydantic import ValidationError

from models.inference import (
    ChatMessage,
    ResponsesFunctionCallInputItem,
    ResponsesFunctionCallOutputInputItem,
    ResponsesFunctionTool,
    ResponsesInputMessage,
    ResponsesOutputFunctionCall,
    ResponsesOutputMessage,
    ResponsesOutputTextContent,
    ResponsesOutputTextPart,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUnknownContentPart,
    ResponsesUnknownInputItem,
    ResponsesUsage,
)
from routes.inference import (
    _chat_tool_calls_to_responses_output,
    _normalise_responses_input,
    _translate_responses_tool_choice_to_chat,
    _translate_responses_tools_to_chat,
)


# =====================================================================
# Request model — tools / tool_choice / parallel_tool_calls
# =====================================================================


class TestResponsesRequestTools:
    def test_flat_function_tool_accepted(self):
        req = ResponsesRequest(
            input = "hi",
            tools = [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    "strict": True,
                }
            ],
        )
        assert req.tools is not None
        assert req.tools[0]["name"] == "get_weather"
        assert req.tools[0]["type"] == "function"
        assert req.tools[0]["strict"] is True

    def test_tool_choice_string_values(self):
        for choice in ("auto", "required", "none"):
            req = ResponsesRequest(input = "hi", tool_choice = choice)
            assert req.tool_choice == choice

    def test_tool_choice_forcing_object(self):
        req = ResponsesRequest(
            input = "hi",
            tool_choice = {"type": "function", "name": "get_weather"},
        )
        assert req.tool_choice == {"type": "function", "name": "get_weather"}

    def test_parallel_tool_calls(self):
        req = ResponsesRequest(input = "hi", parallel_tool_calls = True)
        assert req.parallel_tool_calls is True

    def test_builtin_tool_type_passes_validation(self):
        """Non-function built-in tools (web_search, file_search, mcp, ...) must
        not raise at request validation so SDKs that default to them don't
        fail on Studio; they are filtered out during translation."""
        req = ResponsesRequest(
            input = "hi",
            tools = [{"type": "web_search_preview"}],
        )
        assert req.tools == [{"type": "web_search_preview"}]

    def test_function_tool_model_direct(self):
        tool = ResponsesFunctionTool(
            type = "function",
            name = "send_email",
            parameters = {"type": "object", "properties": {}},
        )
        assert tool.name == "send_email"
        assert tool.description is None

    def test_function_tool_rejects_other_type(self):
        with pytest.raises(ValidationError):
            ResponsesFunctionTool(type = "web_search", name = "x")


# =====================================================================
# Request model — function_call / function_call_output input items
# =====================================================================


class TestResponsesMultiTurnInput:
    def test_function_call_input_item(self):
        req = ResponsesRequest(
            input = [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "type": "function_call",
                    "id": "fc_abc",
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_abc",
                    "output": '{"temp": 12}',
                },
            ],
        )
        assert len(req.input) == 3
        assert isinstance(req.input[1], ResponsesFunctionCallInputItem)
        assert req.input[1].call_id == "call_abc"
        assert isinstance(req.input[2], ResponsesFunctionCallOutputInputItem)
        assert req.input[2].call_id == "call_abc"
        assert req.input[2].output == '{"temp": 12}'

    def test_function_call_output_missing_call_id_rejected(self):
        with pytest.raises(ValidationError):
            ResponsesFunctionCallOutputInputItem(
                type = "function_call_output", output = "x"
            )

    def test_function_call_output_accepts_content_array(self):
        item = ResponsesFunctionCallOutputInputItem(
            type = "function_call_output",
            call_id = "call_1",
            output = [{"type": "output_text", "text": "done"}],
        )
        assert isinstance(item.output, list)


# =====================================================================
# Translators — tools, tool_choice
# =====================================================================


class TestToolsTranslation:
    def test_flat_to_nested(self):
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Returns weather.",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ]
        out = _translate_responses_tools_to_chat(tools)
        assert out == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Returns weather.",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ]

    def test_builtin_tools_dropped(self):
        out = _translate_responses_tools_to_chat(
            [
                {"type": "web_search_preview"},
                {"type": "file_search"},
                {
                    "type": "function",
                    "name": "search",
                    "parameters": {"type": "object"},
                },
            ]
        )
        assert len(out) == 1
        assert out[0]["function"]["name"] == "search"

    def test_empty_returns_none(self):
        assert _translate_responses_tools_to_chat(None) is None
        assert _translate_responses_tools_to_chat([]) is None

    def test_only_builtin_tools_returns_none(self):
        assert (
            _translate_responses_tools_to_chat([{"type": "web_search_preview"}]) is None
        )

    def test_description_optional(self):
        out = _translate_responses_tools_to_chat(
            [
                {
                    "type": "function",
                    "name": "noop",
                    "parameters": {"type": "object"},
                }
            ]
        )
        assert "description" not in out[0]["function"]


class TestToolChoiceTranslation:
    def test_string_passthrough(self):
        for v in ("auto", "required", "none"):
            assert _translate_responses_tool_choice_to_chat(v) == v

    def test_none_passthrough(self):
        assert _translate_responses_tool_choice_to_chat(None) is None

    def test_forcing_object_converted(self):
        assert _translate_responses_tool_choice_to_chat(
            {"type": "function", "name": "get_weather"}
        ) == {"type": "function", "function": {"name": "get_weather"}}

    def test_already_chat_nested_shape_passes_through(self):
        """If a client happens to send the Chat Completions nested shape,
        we don't double-wrap it."""
        already_nested = {"type": "function", "function": {"name": "get_weather"}}
        assert (
            _translate_responses_tool_choice_to_chat(already_nested) == already_nested
        )

    def test_unknown_shape_passes_through(self):
        obj = {"type": "allowed_tools", "tools": [{"type": "function", "name": "x"}]}
        assert _translate_responses_tool_choice_to_chat(obj) == obj


# =====================================================================
# _normalise_responses_input — multi-turn tool mapping
# =====================================================================


class TestNormaliseResponsesInputWithTools:
    def test_function_call_output_maps_to_tool_role(self):
        payload = ResponsesRequest(
            input = [
                {"role": "user", "content": "Weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"temp": 20}',
                },
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert len(msgs) == 3
        assert msgs[0].role == "user"

        assert msgs[1].role == "assistant"
        assert msgs[1].tool_calls is not None
        assert msgs[1].tool_calls[0]["id"] == "call_1"
        assert msgs[1].tool_calls[0]["function"]["name"] == "get_weather"

        assert msgs[2].role == "tool"
        assert msgs[2].tool_call_id == "call_1"
        assert msgs[2].content == '{"temp": 20}'

    def test_instructions_plus_developer_message_are_merged(self):
        """Codex CLI sends `instructions` (system prompt) AND a developer
        message in `input`. Strict chat templates (harmony / gpt-oss, Qwen3,
        ...) raise "System message must be at the beginning" when two
        separate system-role messages appear, so we must emit exactly one
        merged system message at the top.
        """
        payload = ResponsesRequest(
            instructions = "Base instructions.",
            input = [
                {"role": "developer", "content": "Developer override."},
                {"role": "user", "content": "Hi"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        system_roles = [m for m in msgs if m.role == "system"]
        assert len(system_roles) == 1
        assert "Base instructions." in system_roles[0].content
        assert "Developer override." in system_roles[0].content
        # System must be the very first message for strict templates.
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_developer_message_after_user_is_still_hoisted(self):
        """Multi-turn conversations where a developer message appears after
        user turns must still produce a single leading system message, not
        a mid-conversation system that strict templates reject."""
        payload = ResponsesRequest(
            input = [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "developer", "content": "Updated rules."},
                {"role": "user", "content": "Continue"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "system"
        assert "Updated rules." in msgs[0].content
        for m in msgs[1:]:
            assert m.role != "system", "no trailing system message permitted"

    def test_no_system_output_when_no_system_input(self):
        payload = ResponsesRequest(input = "Hi")
        msgs = _normalise_responses_input(payload)
        assert all(m.role != "system" for m in msgs)

    def test_multiple_system_messages_in_input_are_merged(self):
        payload = ResponsesRequest(
            input = [
                {"role": "system", "content": "A"},
                {"role": "system", "content": "B"},
                {"role": "user", "content": "Hi"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert sum(1 for m in msgs if m.role == "system") == 1
        assert "A" in msgs[0].content and "B" in msgs[0].content

    def test_content_array_output_serialised_to_json_string(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "output_text", "text": "ok"}],
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "tool"
        # Content is serialised so llama-server sees a string.
        assert json.loads(msgs[0].content) == [{"type": "output_text", "text": "ok"}]


# =====================================================================
# Response mapping — tool_calls → function_call output items
# =====================================================================


class TestChatToolCallsToResponsesOutput:
    def test_basic_mapping(self):
        items = _chat_tool_calls_to_responses_output(
            [
                {
                    "id": "call_abc",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }
            ]
        )
        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["call_id"] == "call_abc"
        assert items[0]["name"] == "get_weather"
        assert items[0]["arguments"] == '{"city":"Paris"}'
        assert items[0]["status"] == "completed"
        assert items[0]["id"].startswith("fc_")

    def test_multiple_tool_calls_preserved(self):
        items = _chat_tool_calls_to_responses_output(
            [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "a", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "b", "arguments": "{}"},
                },
            ]
        )
        assert [it["call_id"] for it in items] == ["call_1", "call_2"]

    def test_non_function_tool_call_dropped(self):
        items = _chat_tool_calls_to_responses_output([{"id": "x", "type": "retrieval"}])
        assert items == []

    def test_missing_arguments_coerced_to_empty_string(self):
        items = _chat_tool_calls_to_responses_output(
            [{"id": "call_1", "type": "function", "function": {"name": "x"}}]
        )
        assert items[0]["arguments"] == ""


# =====================================================================
# Response model — ResponsesOutputFunctionCall / mixed output
# =====================================================================


class TestResponsesOutputFunctionCall:
    def test_direct_construction(self):
        fc = ResponsesOutputFunctionCall(
            call_id = "call_1",
            name = "get_weather",
            arguments = '{"city":"Paris"}',
        )
        d = fc.model_dump()
        assert d["type"] == "function_call"
        assert d["call_id"] == "call_1"
        assert d["status"] == "completed"
        assert d["id"].startswith("fc_")

    def test_response_with_tool_call_output(self):
        resp = ResponsesResponse(
            model = "test",
            output = [
                ResponsesOutputFunctionCall(
                    call_id = "call_1",
                    name = "get_weather",
                    arguments = "{}",
                )
            ],
            usage = ResponsesUsage(input_tokens = 1, output_tokens = 1, total_tokens = 2),
        )
        d = json.loads(resp.model_dump_json())
        assert d["output"][0]["type"] == "function_call"
        assert d["output"][0]["call_id"] == "call_1"

    def test_response_with_mixed_output(self):
        resp = ResponsesResponse(
            model = "test",
            output = [
                ResponsesOutputMessage(
                    content = [ResponsesOutputTextContent(text = "Calling...")],
                ),
                ResponsesOutputFunctionCall(
                    call_id = "call_1",
                    name = "get_weather",
                    arguments = '{"city":"Paris"}',
                ),
            ],
        )
        d = resp.model_dump()
        assert d["output"][0]["type"] == "message"
        assert d["output"][1]["type"] == "function_call"


# =====================================================================
# Regression: ChatMessage validator still accepts mapped tool messages
# =====================================================================


class TestCodexStyleRequestShapes:
    """Regression tests for the request shapes OpenAI Codex CLI sends."""

    def test_assistant_replay_output_text_accepted(self):
        """Codex replays prior assistant turns with `output_text` content.
        Before, this triggered a 422 on every turn after the first."""
        req = ResponsesRequest(
            input = [
                {"role": "user", "content": "Hi"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello!",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                {"role": "user", "content": "Continue"},
            ],
        )
        assert len(req.input) == 3
        parts = req.input[1].content
        assert isinstance(parts, list)
        assert isinstance(parts[0], ResponsesOutputTextPart)
        assert parts[0].text == "Hello!"

    def test_reasoning_item_accepted_as_unknown(self):
        """`reasoning` items replayed from prior o-series turns must not
        fail validation — Codex preserves them in multi-turn."""
        req = ResponsesRequest(
            input = [
                {"role": "user", "content": "Hi"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "encrypted_content": "opaque",
                },
                {"role": "assistant", "content": "Hello!"},
            ],
        )
        assert len(req.input) == 3
        assert isinstance(req.input[1], ResponsesUnknownInputItem)

    def test_unknown_content_part_type_accepted(self):
        """Unknown content-part types (e.g. future input_audio) validate as
        ResponsesUnknownContentPart so the whole request doesn't 422."""
        req = ResponsesRequest(
            input = [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "See:"},
                        {"type": "input_audio", "audio": {"data": "..."}},
                    ],
                }
            ],
        )
        parts = req.input[0].content
        assert isinstance(parts[1], ResponsesUnknownContentPart)
        assert parts[1].type == "input_audio"

    def test_codex_full_shape_roundtrip(self):
        """End-to-end: developer + user + assistant(output_text) +
        function_call + function_call_output + reasoning in one request."""
        payload = ResponsesRequest(
            instructions = "Base instructions.",
            input = [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Dev override."}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Weather?"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"temp":20}',
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "It's 20°C.",
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
                {"role": "user", "content": "And tomorrow?"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        # Single leading merged system; no mid-conversation system.
        assert msgs[0].role == "system"
        assert sum(1 for m in msgs if m.role == "system") == 1
        assert "Base instructions." in msgs[0].content
        assert "Dev override." in msgs[0].content

        roles = [m.role for m in msgs[1:]]
        # Reasoning item is dropped. Order: user, assistant(tool_calls),
        # tool, assistant(text), user.
        assert roles == ["user", "assistant", "tool", "assistant", "user"]
        assert msgs[2].tool_calls is not None
        assert msgs[3].role == "tool"
        assert msgs[3].tool_call_id == "call_1"
        assert msgs[4].content == "It's 20°C."

    def test_single_output_text_part_flattens_to_string(self):
        """ChatMessage assistant role prefers plain string content — tests
        confirm we don't forward a single-part array that would otherwise
        force legacy chat templates into multimodal handling."""
        payload = ResponsesRequest(
            input = [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "output_text", "text": "ok", "annotations": []}
                    ],
                },
                {"role": "user", "content": "next"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "assistant"
        assert msgs[0].content == "ok"


class TestTranslatedMessagesValidate:
    """Verify that the messages produced by _normalise_responses_input
    satisfy ChatMessage's role-shape validator so the downstream /v1/chat/
    completions pass-through does not reject them."""

    def test_round_trip_multi_turn(self):
        payload = ResponsesRequest(
            input = [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": '{"temp": 20}',
                },
                {"role": "user", "content": "Thanks!"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        for m in msgs:
            # Constructing a fresh ChatMessage from the dump round-trips the
            # role-shape validator — the key invariant for the passthrough.
            ChatMessage(**m.model_dump(exclude_none = True))
