# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the OpenAI /v1/responses client-side function-calling pass-through.

Covers:
- ResponsesRequest accepts Responses-shape `tools`, `tool_choice`,
  `parallel_tool_calls`, and `function_call` / `function_call_output`
  input items for multi-turn tool loops.
- _translate_responses_tools_to_chat(): flat Responses tool shape ->
  nested Chat Completions shape, drops non-function built-in tools,
  returns None for empty lists.
- _translate_responses_tool_choice_to_chat(): passes string choices
  through, converts {type:function,name:X} to the nested shape.
- _normalise_responses_input(): maps function_call_output items to
  role="tool" ChatMessages with tool_call_id, and function_call items to
  assistant messages with tool_calls.
- _chat_tool_calls_to_responses_output(): keeps call_id, drops
  non-function tool calls.
- ResponsesOutputFunctionCall / ResponsesResponse round-trip tool-call
  outputs without losing fields.

No running server or GPU required.
"""

import os
import sys
import asyncio
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import json

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from core.inference.api_monitor import ApiMonitor
from models.inference import (
    ChatMessage,
    ResponsesFunctionCallInputItem,
    ResponsesFunctionCallOutputInputItem,
    ResponsesFunctionTool,
    ResponsesInputMessage,
    ResponsesOutputFunctionCall,
    ResponsesOutputMessage,
    ResponsesOutputReasoning,
    ResponsesOutputTextContent,
    ResponsesOutputTextPart,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUnknownContentPart,
    ResponsesUnknownInputItem,
    ResponsesUsage,
)
from routes.inference import (
    _ResponsesReasoningExtractor,
    _SameTaskStreamingResponse,
    _build_chat_request,
    _chat_tool_calls_to_responses_output,
    _extract_responses_reasoning,
    _normalise_responses_input,
    _responses_tool_output_content,
    _responses_non_streaming,
    _responses_stream,
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
        """Non-function built-in tools (web_search, file_search, mcp, ...)
        must not raise at validation so SDKs that default to them don't
        fail on Studio; they're filtered out during translation."""
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
            ResponsesFunctionCallOutputInputItem(type = "function_call_output", output = "x")

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
        assert _translate_responses_tools_to_chat([{"type": "web_search_preview"}]) is None

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
        """A client sending the Chat Completions nested shape isn't
        double-wrapped."""
        already_nested = {"type": "function", "function": {"name": "get_weather"}}
        assert _translate_responses_tool_choice_to_chat(already_nested) == already_nested

    def test_unknown_shape_passes_through(self):
        obj = {"type": "allowed_tools", "tools": [{"type": "function", "name": "x"}]}
        assert _translate_responses_tool_choice_to_chat(obj) == obj


class TestBuildChatRequest:
    def test_parallel_tool_calls_false_is_preserved_for_passthrough_caps(self):
        payload = ResponsesRequest(
            input = "hi",
            tools = [
                {
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                }
            ],
            parallel_tool_calls = False,
        )
        messages = [ChatMessage(role = "user", content = "hi")]

        chat_req = _build_chat_request(payload, messages, stream = True)

        assert chat_req.parallel_tool_calls is False

    def test_chat_template_kwargs_enable_thinking_true_is_lifted(self):
        payload = ResponsesRequest(
            input = "hi",
            chat_template_kwargs = {"enable_thinking": True},
        )
        messages = [ChatMessage(role = "user", content = "hi")]

        chat_req = _build_chat_request(payload, messages, stream = False)

        assert chat_req.enable_thinking is True

    def test_chat_template_kwargs_enable_thinking_false_is_lifted(self):
        payload = ResponsesRequest(
            input = "hi",
            chat_template_kwargs = {"enable_thinking": False},
        )
        messages = [ChatMessage(role = "user", content = "hi")]

        chat_req = _build_chat_request(payload, messages, stream = False)

        assert chat_req.enable_thinking is False

    def test_reasoning_effort_high_enables_local_thinking(self):
        payload = ResponsesRequest(input = "hi", reasoning = {"effort": "high"})
        messages = [ChatMessage(role = "user", content = "hi")]

        chat_req = _build_chat_request(payload, messages, stream = False)

        assert chat_req.reasoning_effort == "high"
        assert chat_req.enable_thinking is True

    def test_reasoning_effort_none_disables_local_thinking(self):
        payload = ResponsesRequest(input = "hi", reasoning = {"effort": "none"})
        messages = [ChatMessage(role = "user", content = "hi")]

        chat_req = _build_chat_request(payload, messages, stream = False)

        assert chat_req.reasoning_effort == "none"
        assert chat_req.enable_thinking is False

    def test_explicit_enable_thinking_false_disables_reasoning_effort(self):
        payload = ResponsesRequest(
            input = "hi",
            reasoning = {"effort": "high"},
            chat_template_kwargs = {"enable_thinking": False},
        )
        messages = [ChatMessage(role = "user", content = "hi")]

        chat_req = _build_chat_request(payload, messages, stream = False)

        assert chat_req.reasoning_effort == "none"
        assert chat_req.enable_thinking is False


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
        message in `input`. Strict chat templates (harmony / gpt-oss,
        Qwen3, ...) raise "System message must be at the beginning" on two
        separate system-role messages, so we emit exactly one merged
        system message at the top.
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
        # System must be the first message for strict templates.
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_developer_message_after_user_is_still_hoisted(self):
        """A developer message appearing after user turns must still
        produce a single leading system message, not a mid-conversation
        system that strict templates reject."""
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

    def test_content_array_text_output_flattens_to_tool_text(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "input_text", "text": "ok"}],
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "tool"
        assert msgs[0].content == "ok"

    def test_content_array_image_output_becomes_multimodal_tool_content(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "see image"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAA",
                            "detail": "high",
                        },
                    ],
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "tool"
        assert msgs[0].tool_call_id == "call_1"
        assert msgs[0].model_dump(exclude_none = True)["content"] == [
            {"type": "text", "text": "see image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AAA",
                    "detail": "high",
                },
            },
        ]

        chat_req = _build_chat_request(payload, msgs, stream = False)
        assert chat_req.model_dump(exclude_none = True)["messages"][0]["content"] == [
            {"type": "text", "text": "see image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64,AAA",
                    "detail": "high",
                },
            },
        ]

    def test_content_array_image_output_allows_original_detail(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/screenshot.png",
                            "detail": "original",
                        },
                    ],
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].model_dump(exclude_none = True)["content"] == [
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://example.com/screenshot.png",
                    "detail": "original",
                },
            },
        ]

    def test_content_array_file_id_image_output_rejected_clearly(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "see image"},
                        {"type": "input_image", "file_id": "file_abc"},
                    ],
                }
            ],
        )
        with pytest.raises(HTTPException) as exc:
            _normalise_responses_input(payload)
        assert exc.value.status_code == 400
        assert "file_id" in str(exc.value.detail)

    def test_content_array_file_output_rejected_clearly(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "see file"},
                        {
                            "type": "input_file",
                            "file_data": "data:application/pdf;base64,AAA",
                            "filename": "report.pdf",
                        },
                    ],
                }
            ],
        )
        with pytest.raises(HTTPException) as exc:
            _normalise_responses_input(payload)
        assert exc.value.status_code == 400
        assert "input_file" in str(exc.value.detail)

    def test_content_array_malformed_image_output_rejected_clearly(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "input_image", "detail": "high"}],
                }
            ],
        )
        with pytest.raises(HTTPException) as exc:
            _normalise_responses_input(payload)
        assert exc.value.status_code == 400
        assert "image_url" in str(exc.value.detail)

    def test_empty_function_call_output_gets_no_output_sentinel(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "",
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "tool"
        assert msgs[0].tool_call_id == "call_1"
        assert msgs[0].content == "(no output)"
        ChatMessage(**msgs[0].model_dump(exclude_none = True))

    def test_whitespace_function_call_output_gets_no_output_sentinel(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "   \n\t",
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].content == "(no output)"

    def test_empty_content_array_output_gets_no_output_sentinel(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [],
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].content == "(no output)"

    def test_image_content_array_tool_output_is_serialised(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        }
                    ],
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "tool"
        assert json.loads(msgs[0].content)[0]["type"] == "image"

    def test_image_payload_outside_output_gets_no_output_sentinel(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "iVBORw0KGgo=",
                            },
                        }
                    ],
                }
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "tool"
        assert msgs[0].tool_call_id == "call_1"
        assert msgs[0].content == "(no output)"

    def test_tool_output_serializer_preserves_non_empty_text(self):
        assert _responses_tool_output_content("done") == "done"
        assert _responses_tool_output_content("  done  ") == "  done  "


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
# Non-streaming Responses adapter
# =====================================================================


class TestResponsesNonStreamingAdapter:
    class _Request:
        pass

    @staticmethod
    def _run_with_message(
        monkeypatch,
        message,
        payload = None,
        llama_backend = None,
    ):
        import routes.inference as inf_mod

        async def fake_chat_completions(chat_req, request):
            return JSONResponse(
                content = {
                    "model": "test-model",
                    "choices": [{"message": message}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                }
            )

        monkeypatch.setattr(inf_mod, "openai_chat_completions", fake_chat_completions)
        if llama_backend is not None:
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: llama_backend)
        payload = payload or ResponsesRequest(input = "hi")
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_non_streaming(
                payload, messages, TestResponsesNonStreamingAdapter._Request()
            )
            return json.loads(response.body.decode())

        return asyncio.run(run())

    def test_think_block_becomes_reasoning_item_before_message(self, monkeypatch):
        payload = ResponsesRequest(input = "hi", reasoning = {"effort": "high"})
        body = self._run_with_message(
            monkeypatch,
            {"content": "<think>plan</think>33"},
            payload = payload,
        )

        assert [item["type"] for item in body["output"]] == ["reasoning", "message"]
        assert body["output"][0]["content"] == [{"type": "reasoning_text", "text": "plan"}]
        assert body["output"][0]["summary"] == []
        assert body["output"][1]["content"][0]["text"] == "33"
        assert "<think>" not in body["output"][1]["content"][0]["text"]
        assert "</think>" not in body["output"][1]["content"][0]["text"]

    def test_unclosed_think_block_extracts_as_reasoning(self):
        reasoning, visible = _extract_responses_reasoning(
            "<think>partial plan",
            parse_think_markers = True,
        )

        assert reasoning == "partial plan"
        assert visible == ""

    def test_monitor_records_translated_visible_text(self, monkeypatch):
        import routes.inference as inf_mod
        import routes.inference as inf_mod

        async def fake_chat_completions(chat_req, request):
            assert request.state.skip_api_monitor is True
            return JSONResponse(
                content = {
                    "model": "test-model",
                    "choices": [{"message": {"content": "<think>plan</think>answer"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                }
            )

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "openai_chat_completions", fake_chat_completions)
        payload = ResponsesRequest(input = "hi", reasoning = {"effort": "high"})
        messages = [ChatMessage(role = "user", content = "hi")]
        request = SimpleNamespace(
            state = SimpleNamespace(),
            url = SimpleNamespace(path = "/v1/responses"),
            method = "POST",
        )

        async def run():
            response = await _responses_non_streaming(payload, messages, request)
            return json.loads(response.body.decode())

        body = asyncio.run(run())

        assert body["output"][0]["content"] == [{"type": "reasoning_text", "text": "plan"}]
        assert body["output"][1]["content"][0]["text"] == "answer"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply"] == "answer"
        assert entry["prompt_tokens"] == 2
        assert entry["completion_tokens"] == 3
        assert request.state.skip_api_monitor is False

    def test_monitor_records_tool_only_reply(self, monkeypatch):
        import routes.inference as inf_mod

        async def fake_chat_completions(chat_req, request):
            assert request.state.skip_api_monitor is True
            return JSONResponse(
                content = {
                    "model": "test-model",
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query":"weather"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3},
                }
            )

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "openai_chat_completions", fake_chat_completions)
        payload = ResponsesRequest(
            input = "hi",
            tools = [{"type": "function", "name": "lookup"}],
        )
        messages = [ChatMessage(role = "user", content = "hi")]
        request = SimpleNamespace(
            state = SimpleNamespace(),
            url = SimpleNamespace(path = "/v1/responses"),
            method = "POST",
        )

        async def run():
            response = await _responses_non_streaming(payload, messages, request)
            return json.loads(response.body.decode())

        body = asyncio.run(run())

        assert body["output"][0]["type"] == "function_call"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply"] == 'Tool call: lookup({"query":"weather"})'
        assert request.state.skip_api_monitor is False

    def test_cancelled_chat_completion_finalizes_monitor(self, monkeypatch):
        import routes.inference as inf_mod

        async def fake_chat_completions(chat_req, request):
            assert request.state.skip_api_monitor is True
            raise asyncio.CancelledError()

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "openai_chat_completions", fake_chat_completions)
        payload = ResponsesRequest(input = "hi")
        messages = [ChatMessage(role = "user", content = "hi")]
        request = SimpleNamespace(
            state = SimpleNamespace(),
            url = SimpleNamespace(path = "/v1/responses"),
            method = "POST",
        )

        async def run():
            with pytest.raises(asyncio.CancelledError):
                await _responses_non_streaming(payload, messages, request)

        asyncio.run(run())

        [entry] = monitor.snapshot()
        assert entry["status"] == "cancelled"
        assert monitor.active_count() == 0
        assert request.state.skip_api_monitor is False

    def test_literal_think_tags_remain_visible_without_reasoning_request(self, monkeypatch):
        body = self._run_with_message(monkeypatch, {"content": "show <think>x</think> tags"})

        assert [item["type"] for item in body["output"]] == ["message"]
        assert body["output"][0]["content"][0]["text"] == "show <think>x</think> tags"

    def test_non_reasoning_gguf_keeps_literal_think_tags_visible(self, monkeypatch):
        payload = ResponsesRequest(input = "hi", reasoning = {"effort": "high"})
        body = self._run_with_message(
            monkeypatch,
            {"content": "show <think>x</think> tags"},
            payload = payload,
            llama_backend = SimpleNamespace(
                is_loaded = True,
                reasoning_always_on = False,
                supports_reasoning = False,
            ),
        )

        assert [item["type"] for item in body["output"]] == ["message"]
        assert body["output"][0]["content"][0]["text"] == "show <think>x</think> tags"

    def test_reasoning_capable_gguf_parses_think_tags_by_default(self, monkeypatch):
        body = self._run_with_message(
            monkeypatch,
            {"content": "<think>plan</think>answer"},
            llama_backend = SimpleNamespace(
                is_loaded = True,
                reasoning_always_on = False,
                supports_reasoning = True,
            ),
        )

        assert [item["type"] for item in body["output"]] == ["reasoning", "message"]
        assert body["output"][0]["content"] == [{"type": "reasoning_text", "text": "plan"}]
        assert body["output"][1]["content"][0]["text"] == "answer"

    def test_reasoning_capable_gguf_sanitizes_think_tags_when_disabled(self, monkeypatch):
        payload = ResponsesRequest(input = "hi", reasoning = {"effort": "none"})
        body = self._run_with_message(
            monkeypatch,
            {"content": "<think>leaked</think>answer"},
            payload = payload,
            llama_backend = SimpleNamespace(
                is_loaded = True,
                reasoning_always_on = False,
                supports_reasoning = True,
            ),
        )

        assert [item["type"] for item in body["output"]] == ["reasoning", "message"]
        assert body["output"][0]["content"] == [{"type": "reasoning_text", "text": "leaked"}]
        assert body["output"][1]["content"][0]["text"] == "answer"

    def test_structured_reasoning_content_extracts_text_parts(self, monkeypatch):
        body = self._run_with_message(
            monkeypatch,
            {
                "content": "33",
                "reasoning_content": [
                    {"type": "reasoning_text", "text": "plan"},
                    {"type": "reasoning_text", "text": " next"},
                ],
            },
        )

        assert [item["type"] for item in body["output"]] == ["reasoning", "message"]
        assert body["output"][0]["content"] == [{"type": "reasoning_text", "text": "plan next"}]
        assert body["output"][1]["content"][0]["text"] == "33"

    def test_plain_content_remains_message_only(self, monkeypatch):
        body = self._run_with_message(monkeypatch, {"content": "33"})

        assert [item["type"] for item in body["output"]] == ["message"]
        assert body["output"][0]["content"][0]["text"] == "33"

    def test_reasoning_only_stays_out_of_visible_message_text(self, monkeypatch):
        payload = ResponsesRequest(input = "hi", reasoning = {"effort": "high"})
        body = self._run_with_message(
            monkeypatch,
            {"content": "<think>plan</think>"},
            payload = payload,
        )

        assert [item["type"] for item in body["output"]] == ["reasoning"]
        assert body["output"][0]["content"][0]["text"] == "plan"


# =====================================================================
# Streaming Responses adapter
# =====================================================================


class TestResponsesStreamAdapter:
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

    @staticmethod
    def _install_stream_mock(
        monkeypatch,
        chunks,
        *,
        supports_reasoning = True,
        reasoning_always_on = False,
    ):
        import routes.inference as inf_mod

        def handler(request: httpx.Request) -> httpx.Response:
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
        monkeypatch.setattr(
            inf_mod,
            "get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                context_length = 4096,
                base_url = "http://llama.test",
                supports_reasoning = supports_reasoning,
                reasoning_always_on = reasoning_always_on,
                _request_reasoning_kwargs = (
                    lambda enable_thinking = None, reasoning_effort = None, preserve_thinking = None: None
                ),
            ),
        )

    def test_stream_response_avoids_legacy_receive_watcher(self, monkeypatch):
        self._install_stream_mock(
            monkeypatch,
            [{"choices": [{"delta": {"content": "33"}}]}],
        )
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            assert isinstance(response, _SameTaskStreamingResponse)

            sent = []

            async def receive():
                raise AssertionError("Responses streams poll disconnects in the generator")

            async def send(message):
                sent.append(message)

            await response({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
            return sent

        sent = asyncio.run(run())

        assert sent[0]["type"] == "http.response.start"
        body = b"".join(message.get("body", b"") for message in sent).decode()
        assert "response.output_text.delta" in body
        assert '"delta":"33"' in body.replace(" ", "")

    def test_split_think_markers_stream_as_reasoning_and_visible_text(self, monkeypatch):
        chunks = [
            {"choices": [{"delta": {"content": "<thi"}}]},
            {"choices": [{"delta": {"content": "nk>pla"}}]},
            {"choices": [{"delta": {"content": "n</th"}}]},
            {"choices": [{"delta": {"content": "ink>33"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        payload = ResponsesRequest(input = "hi", stream = True, reasoning = {"effort": "high"})
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        reasoning_deltas = self._payloads(lines, "response.reasoning_text.delta")
        text_deltas = self._payloads(lines, "response.output_text.delta")
        assert "".join(event["delta"] for event in reasoning_deltas) == "plan"
        assert "".join(event["delta"] for event in text_deltas) == "33"
        completed = self._payloads(lines, "response.completed")[0]
        assert [item["type"] for item in completed["response"]["output"]] == [
            "reasoning",
            "message",
        ]
        assert completed["response"]["output"][0]["content"][0]["text"] == "plan"
        assert completed["response"]["output"][1]["content"][0]["text"] == "33"

    def test_usage_only_chunk_updates_monitor(self, monkeypatch):
        import routes.inference as inf_mod

        chunks = [
            {"choices": [{"delta": {"content": "33"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/responses",
            method = "POST",
            model = "m",
            prompt = "hi",
        )
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(
                payload,
                messages,
                self._Request(),
                monitor_id = monitor_id,
            )
            return await self._collect(response)

        asyncio.run(run())

        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply"] == "33"
        assert entry["prompt_tokens"] == 2
        assert entry["completion_tokens"] == 3
        assert entry["total_tokens"] == 5
        assert entry["context_length"] == 4096

    def test_function_call_chunk_updates_monitor_reply(self, monkeypatch):
        import routes.inference as inf_mod

        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":"weather"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ]
        self._install_stream_mock(monkeypatch, chunks)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/responses",
            method = "POST",
            model = "m",
            prompt = "hi",
        )
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(
                payload,
                messages,
                self._Request(),
                monitor_id = monitor_id,
            )
            return await self._collect(response)

        lines = asyncio.run(run())

        assert self._payloads(lines, "response.output_item.done")[-1]["item"]["name"] == "lookup"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply"] == 'Tool call: lookup({"query":"weather"})'

    def test_preheader_cancel_finalizes_monitor(self, monkeypatch):
        import routes.inference as inf_mod

        self._install_stream_mock(monkeypatch, [])
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/responses",
            method = "POST",
            model = "m",
            prompt = "hi",
        )

        async def fake_send(*_args, **_kwargs):
            return None

        monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(
                payload,
                messages,
                self._Request(),
                monitor_id = monitor_id,
            )
            return await self._collect(response)

        asyncio.run(run())

        [entry] = monitor.snapshot()
        assert entry["status"] == "cancelled"
        assert monitor.active_count() == 0

    def test_stream_task_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_send(*_args, **_kwargs):
                return httpx.Response(200, content = b"")

            async def fake_items(*_args, **_kwargs):
                yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
                await asyncio.sleep(3600)

            self._install_stream_mock(monkeypatch, [])
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
            monitor_id = monitor.start(
                endpoint = "/v1/responses",
                method = "POST",
                model = "m",
                prompt = "hi",
            )
            payload = ResponsesRequest(input = "hi", stream = True)
            messages = [ChatMessage(role = "user", content = "hi")]

            response = await _responses_stream(
                payload,
                messages,
                self._Request(),
                monitor_id = monitor_id,
            )
            iterator = response.body_iterator
            first = ""
            for _ in range(8):
                first = await anext(iterator)
                if "hello" in first:
                    break
            else:
                pytest.fail("stream did not emit text delta")

            pending = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert entry["reply"] == "hello"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_final_visible_text_updates_monitor(self, monkeypatch):
        import routes.inference as inf_mod

        class FakeExtractor:
            def __init__(self, **_kwargs):
                pass

            def feed(
                self,
                _content,
                _reasoning_content = None,
            ):
                return "", ""

            def finish(self):
                return "", "tail"

        self._install_stream_mock(monkeypatch, [{"choices": [{"delta": {"content": "<tai"}}]}])
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "_ResponsesReasoningExtractor", FakeExtractor)
        monitor_id = monitor.start(
            endpoint = "/v1/responses",
            method = "POST",
            model = "m",
            prompt = "hi",
        )
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(
                payload,
                messages,
                self._Request(),
                monitor_id = monitor_id,
            )
            return await self._collect(response)

        lines = asyncio.run(run())

        assert self._payloads(lines, "response.output_text.delta")[-1]["delta"] == "tail"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply"] == "tail"

    def test_reasoning_only_stream_does_not_update_visible_monitor_reply(self, monkeypatch):
        import routes.inference as inf_mod

        class FakeExtractor:
            def __init__(self, **_kwargs):
                pass

            def feed(
                self,
                _content,
                _reasoning_content = None,
            ):
                return "", ""

            def finish(self):
                return "plan", ""

        self._install_stream_mock(monkeypatch, [{"choices": [{"delta": {"content": "<think>"}}]}])
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "_ResponsesReasoningExtractor", FakeExtractor)
        monitor_id = monitor.start(
            endpoint = "/v1/responses",
            method = "POST",
            model = "m",
            prompt = "hi",
        )
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(
                payload,
                messages,
                self._Request(),
                monitor_id = monitor_id,
            )
            return await self._collect(response)

        lines = asyncio.run(run())

        assert self._payloads(lines, "response.output_text.delta") == []
        assert self._payloads(lines, "response.reasoning_text.delta")[-1]["delta"] == "plan"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply"] == ""

    def test_reasoning_capable_gguf_stream_parses_think_tags_by_default(self, monkeypatch):
        chunks = [
            {"choices": [{"delta": {"content": "<thi"}}]},
            {"choices": [{"delta": {"content": "nk>plan</think>answer"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        reasoning_deltas = self._payloads(lines, "response.reasoning_text.delta")
        text_deltas = self._payloads(lines, "response.output_text.delta")
        assert "".join(event["delta"] for event in reasoning_deltas) == "plan"
        assert "".join(event["delta"] for event in text_deltas) == "answer"
        completed = self._payloads(lines, "response.completed")[0]
        assert [item["type"] for item in completed["response"]["output"]] == [
            "reasoning",
            "message",
        ]
        assert completed["response"]["output"][0]["content"][0]["text"] == "plan"
        assert completed["response"]["output"][1]["content"][0]["text"] == "answer"

    def test_non_reasoning_gguf_stream_keeps_literal_think_tags_visible(self, monkeypatch):
        chunks = [
            {"choices": [{"delta": {"content": "show <thi"}}]},
            {"choices": [{"delta": {"content": "nk>x</think> tags"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks, supports_reasoning = False)
        payload = ResponsesRequest(input = "hi", stream = True, reasoning = {"effort": "high"})
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        reasoning_deltas = self._payloads(lines, "response.reasoning_text.delta")
        text_deltas = self._payloads(lines, "response.output_text.delta")
        assert reasoning_deltas == []
        assert "".join(event["delta"] for event in text_deltas) == "show <think>x</think> tags"
        completed = self._payloads(lines, "response.completed")[0]
        assert [item["type"] for item in completed["response"]["output"]] == ["message"]
        assert completed["response"]["output"][0]["content"][0]["text"] == (
            "show <think>x</think> tags"
        )

    def test_reasoning_only_stream_stays_out_of_visible_message_text(self, monkeypatch):
        chunks = [
            {"choices": [{"delta": {"content": "<think>plan</think>"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        payload = ResponsesRequest(input = "hi", stream = True, reasoning = {"effort": "high"})
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        reasoning_deltas = self._payloads(lines, "response.reasoning_text.delta")
        text_deltas = self._payloads(lines, "response.output_text.delta")
        assert "".join(event["delta"] for event in reasoning_deltas) == "plan"
        assert text_deltas == []
        completed = self._payloads(lines, "response.completed")[0]
        assert [item["type"] for item in completed["response"]["output"]] == ["reasoning"]
        assert completed["response"]["output"][0]["content"][0]["text"] == "plan"

    def test_unclosed_think_stream_stays_out_of_visible_message_text(self, monkeypatch):
        chunks = [
            {"choices": [{"delta": {"content": "<thi"}}]},
            {"choices": [{"delta": {"content": "nk>plan"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        payload = ResponsesRequest(input = "hi", stream = True, reasoning = {"effort": "high"})
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        reasoning_deltas = self._payloads(lines, "response.reasoning_text.delta")
        text_deltas = self._payloads(lines, "response.output_text.delta")
        assert "".join(event["delta"] for event in reasoning_deltas) == "plan"
        assert text_deltas == []
        completed = self._payloads(lines, "response.completed")[0]
        assert [item["type"] for item in completed["response"]["output"]] == ["reasoning"]
        assert completed["response"]["output"][0]["content"][0]["text"] == "plan"

    def test_structured_reasoning_content_streams_as_reasoning(self, monkeypatch):
        chunks = [
            {"choices": [{"delta": {"reasoning_content": "plan"}}]},
            {"choices": [{"delta": {"content": "33"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        reasoning_deltas = self._payloads(lines, "response.reasoning_text.delta")
        text_deltas = self._payloads(lines, "response.output_text.delta")
        assert "".join(event["delta"] for event in reasoning_deltas) == "plan"
        assert "".join(event["delta"] for event in text_deltas) == "33"
        completed = self._payloads(lines, "response.completed")[0]
        assert completed["response"]["output"][0]["type"] == "reasoning"
        assert completed["response"]["output"][1]["type"] == "message"

    def test_structured_reasoning_content_parts_stream_as_reasoning(self, monkeypatch):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "reasoning_content": {
                                "content": [
                                    {"type": "reasoning_text", "text": "plan"},
                                    {"type": "reasoning_text", "text": " next"},
                                ]
                            }
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": "33"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        reasoning_deltas = self._payloads(lines, "response.reasoning_text.delta")
        text_deltas = self._payloads(lines, "response.output_text.delta")
        assert "".join(event["delta"] for event in reasoning_deltas) == "plan next"
        assert "".join(event["delta"] for event in text_deltas) == "33"
        assert "reasoning_text" not in "".join(event["delta"] for event in reasoning_deltas)
        completed = self._payloads(lines, "response.completed")[0]
        assert completed["response"]["output"][0]["content"][0]["text"] == "plan next"
        assert completed["response"]["output"][1]["content"][0]["text"] == "33"

    def test_tool_first_stream_closes_items_in_output_index_order(self, monkeypatch):
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_0",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {"content": "done"}}]},
            {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
        ]
        self._install_stream_mock(monkeypatch, chunks)
        payload = ResponsesRequest(input = "hi", stream = True)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        done_events = self._payloads(lines, "response.output_item.done")
        assert [event["output_index"] for event in done_events] == [0, 1]
        assert [event["item"]["type"] for event in done_events] == ["function_call", "message"]
        completed = self._payloads(lines, "response.completed")[0]
        assert [item["type"] for item in completed["response"]["output"]] == [
            "function_call",
            "message",
        ]

    def test_requests_usage_and_caps_parallel_tool_calls(self, monkeypatch):
        import routes.inference as inf_mod

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            chunks = [
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_0",
                                        "type": "function",
                                        "function": {"name": "first", "arguments": "{}"},
                                    },
                                    {
                                        "index": 1,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {"name": "second", "arguments": "{}"},
                                    },
                                ]
                            }
                        }
                    ]
                },
                {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
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
        monkeypatch.setattr(
            inf_mod,
            "get_llama_cpp_backend",
            lambda: SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                context_length = 4096,
                base_url = "http://llama.test",
                # Non-reasoning template: the real backend returns None here.
                _request_reasoning_kwargs = (
                    lambda enable_thinking = None, reasoning_effort = None, preserve_thinking = None: None
                ),
            ),
        )

        payload = ResponsesRequest(
            input = "hi",
            stream = True,
            parallel_tool_calls = False,
            tools = [
                {
                    "type": "function",
                    "name": "first",
                    "parameters": {"type": "object"},
                },
                {
                    "type": "function",
                    "name": "second",
                    "parameters": {"type": "object"},
                },
            ],
        )
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(payload, messages, self._Request())
            return await self._collect(response)

        lines = asyncio.run(run())

        assert captured["body"]["stream_options"] == {"include_usage": True}
        joined = "".join(lines)
        assert "call_0" in joined
        assert "call_1" not in joined
        completed = self._payloads(lines, "response.completed")[0]
        assert completed["response"]["usage"] == {
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
        }


# =====================================================================
# Response model — ResponsesOutputFunctionCall / mixed output
# =====================================================================


class TestResponsesOutputFunctionCall:
    def test_reasoning_output_item_serialises_full_reasoning_content(self):
        item = ResponsesOutputReasoning(content = [{"type": "reasoning_text", "text": "plan"}])
        d = item.model_dump()
        assert d["type"] == "reasoning"
        assert d["id"].startswith("rs_")
        assert d["status"] == "completed"
        assert d["summary"] == []
        assert d["content"] == [{"type": "reasoning_text", "text": "plan"}]

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
        """Codex replays prior assistant turns with `output_text` content;
        this used to 422 on every turn after the first."""
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
        fail validation — Codex keeps them in multi-turn."""
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

    def test_emitted_reasoning_item_replay_is_dropped_for_local_chat(self):
        payload = ResponsesRequest(
            input = [
                {"role": "user", "content": "Hi"},
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "plan"}],
                },
                {"role": "assistant", "content": "33"},
                {"role": "user", "content": "Continue"},
            ],
        )

        msgs = _normalise_responses_input(payload)

        assert [m.role for m in msgs] == ["user", "assistant", "user"]
        assert all("plan" not in (m.content or "") for m in msgs if isinstance(m.content, str))

    def test_unknown_content_part_type_accepted(self):
        """Unknown content-part types (e.g. future input_audio) validate as
        ResponsesUnknownContentPart so the request doesn't 422."""
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
        # One leading merged system; no mid-conversation system.
        assert msgs[0].role == "system"
        assert sum(1 for m in msgs if m.role == "system") == 1
        assert "Base instructions." in msgs[0].content
        assert "Dev override." in msgs[0].content

        roles = [m.role for m in msgs[1:]]
        # Reasoning dropped. Order: user, assistant(tool_calls), tool,
        # assistant(text), user.
        assert roles == ["user", "assistant", "tool", "assistant", "user"]
        assert msgs[2].tool_calls is not None
        assert msgs[3].role == "tool"
        assert msgs[3].tool_call_id == "call_1"
        assert msgs[4].content == "It's 20°C."

    def test_single_output_text_part_flattens_to_string(self):
        """ChatMessage assistant role prefers plain string content — we
        don't forward a single-part array that would force legacy chat
        templates into multimodal handling."""
        payload = ResponsesRequest(
            input = [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                },
                {"role": "user", "content": "next"},
            ],
        )
        msgs = _normalise_responses_input(payload)
        assert msgs[0].role == "assistant"
        assert msgs[0].content == "ok"


class TestTranslatedMessagesValidate:
    """Messages from _normalise_responses_input satisfy ChatMessage's
    role-shape validator so the downstream /v1/chat/completions
    pass-through doesn't reject them."""

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
            # Building a fresh ChatMessage from the dump round-trips the
            # role-shape validator — the passthrough's key invariant.
            ChatMessage(**m.model_dump(exclude_none = True))

    def test_empty_tool_output_round_trips_through_chat_message_validator(self):
        payload = ResponsesRequest(
            input = [
                {
                    "type": "function_call_output",
                    "call_id": "call_empty",
                    "output": "",
                },
            ],
        )
        msgs = _normalise_responses_input(payload)
        for m in msgs:
            ChatMessage(**m.model_dump(exclude_none = True))


# reasoning_prefilled: enable_thinking templates prefill an unclosed <think>, so
# generation begins inside the block; the extractor must start in reasoning.
class TestReasoningPrefilledExtractor:
    def test_prefilled_single_feed_splits_lone_close(self):
        # T1: reasoning...</think>answer with a prefilled (unseen) open tag.
        reasoning, visible = _extract_responses_reasoning(
            "plan</think>answer",
            parse_think_markers = True,
            reasoning_prefilled = True,
        )
        assert reasoning == "plan"
        assert visible == "answer"

    def test_prefilled_never_closed_is_all_reasoning(self):
        # T2: truncated mid-thought (no </think>) -> all reasoning (GGUF parity).
        reasoning, visible = _extract_responses_reasoning(
            "still thinking with no close",
            parse_think_markers = True,
            reasoning_prefilled = True,
        )
        assert reasoning == "still thinking with no close"
        assert visible == ""

    def test_prefilled_close_split_across_feeds(self):
        # T3: </think> straddles two feed() calls; holdback resolves it.
        ex = _ResponsesReasoningExtractor(parse_think_markers = True, reasoning_prefilled = True)
        r1, v1 = ex.feed("plan</th")
        r2, v2 = ex.feed("ink>ans")
        fr, fv = ex.finish()
        assert (r1 + r2 + fr) == "plan"
        assert (v1 + v2 + fv) == "ans"

    def test_prefilled_close_split_one_char_per_feed(self):
        # T4: every char in its own feed still splits correctly.
        ex = _ResponsesReasoningExtractor(parse_think_markers = True, reasoning_prefilled = True)
        reasoning, visible = "", ""
        for ch in "plan</think>x":
            r, v = ex.feed(ch)
            reasoning += r
            visible += v
        fr, fv = ex.finish()
        assert (reasoning + fr) == "plan"
        assert (visible + fv) == "x"

    def test_prefilled_empty_generation(self):
        # T5: nothing generated.
        reasoning, visible = _extract_responses_reasoning(
            "",
            parse_think_markers = True,
            reasoning_prefilled = True,
        )
        assert reasoning == ""
        assert visible == ""

    def test_prefilled_whitespace_after_close_is_visible(self):
        # T6: Qwen commonly emits </think>\n\n before the answer.
        reasoning, visible = _extract_responses_reasoning(
            "plan</think>\n\nanswer",
            parse_think_markers = True,
            reasoning_prefilled = True,
        )
        assert reasoning == "plan"
        assert visible == "\n\nanswer"

    def test_prefilled_stray_open_tag_is_suppressed(self):
        # T7: a re-emitted literal <think> inside prefilled reasoning is dropped,
        # not leaked into the drawer (covers enable_thinking_effort full-tag output).
        reasoning, visible = _extract_responses_reasoning(
            "a<think>b</think>c",
            parse_think_markers = True,
            reasoning_prefilled = True,
        )
        assert reasoning == "ab"
        assert visible == "c"
        assert "<think>" not in reasoning

    def test_prefilled_close_at_start_empty_reasoning(self):
        # T8: model closed immediately (empty reasoning) then answered.
        reasoning, visible = _extract_responses_reasoning(
            "</think>hi",
            parse_think_markers = True,
            reasoning_prefilled = True,
        )
        assert reasoning == ""
        assert visible == "hi"

    def test_not_prefilled_lone_close_preserves_current_behavior(self):
        # T9: without prefilled, a lone close tag keeps the pre-fix behavior (parity guard).
        reasoning, visible = _extract_responses_reasoning(
            "reasoning</think>ans",
            parse_think_markers = True,
            reasoning_prefilled = False,
        )
        assert reasoning == ""
        assert visible == "reasoningans"

    def test_not_prefilled_full_pair_still_splits(self):
        # T10: normal explicit <think>..</think> (GGUF / Harmony) unchanged.
        reasoning, visible = _extract_responses_reasoning(
            "<think>r</think>v",
            parse_think_markers = True,
            reasoning_prefilled = False,
        )
        assert reasoning == "r"
        assert visible == "v"

    def test_prefilled_ignored_when_markers_not_parsed(self):
        # T11: a non-reasoning model passes text through even with reasoning_prefilled False.
        reasoning, visible = _extract_responses_reasoning(
            "just an answer",
            parse_think_markers = False,
            reasoning_prefilled = False,
        )
        assert reasoning == ""
        assert visible == "just an answer"


# =====================================================================
# Streaming passthrough healing — text-form calls promoted in order
# =====================================================================


class TestResponsesStreamHealing:
    """Route-level healing on the /v1/responses stream: text-form tool calls
    are promoted through the same per-call item state machinery as structured
    deltas, and healer events keep their order (text around a healed call must
    not move relative to the function_call item)."""

    _XML = '<tool_call>{"name":"lookup","arguments":{"q":"x"}}</tool_call>'
    _TOOL = {"type": "function", "name": "lookup", "parameters": {"type": "object"}}

    @staticmethod
    def _ordered_events(lines):
        events = []
        for line in lines:
            if not line.startswith("event: "):
                continue
            name, _, rest = line.partition("\n")
            payload = json.loads(rest.split("data: ", 1)[1].strip())
            events.append((name[len("event: ") :], payload))
        return events

    def _run_stream(self, monkeypatch, content, **payload_kwargs):
        TestResponsesStreamAdapter._install_stream_mock(
            monkeypatch, [{"choices": [{"delta": {"content": content}}]}]
        )
        payload = ResponsesRequest(input = "hi", stream = True, tools = [self._TOOL], **payload_kwargs)
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(
                payload, messages, TestResponsesStreamAdapter._Request()
            )
            return await TestResponsesStreamAdapter._collect(response)

        return self._ordered_events(asyncio.run(run()))

    def test_text_around_healed_call_keeps_order(self, monkeypatch):
        events = self._run_stream(monkeypatch, f"before {self._XML} after.")
        pos_before = pos_item = pos_after = None
        for i, (name, payload) in enumerate(events):
            if name == "response.output_text.delta":
                if "before" in payload["delta"] and pos_before is None:
                    pos_before = i
                if "after" in payload["delta"]:
                    pos_after = i
            if (
                name == "response.output_item.added"
                and payload["item"]["type"] == "function_call"
                and pos_item is None
            ):
                pos_item = i
                assert payload["item"]["name"] == "lookup"
        assert pos_before is not None and pos_item is not None and pos_after is not None
        assert pos_before < pos_item < pos_after

    def test_call_before_trailing_text_claims_lower_output_index(self, monkeypatch):
        events = self._run_stream(monkeypatch, f"{self._XML} done.")
        item_added = [
            (name, payload) for name, payload in events if name == "response.output_item.added"
        ]
        # The call came first in the model output, so its item is added first
        # and claims the lower output_index; the trailing text's message item
        # follows.
        assert [payload["item"]["type"] for _, payload in item_added] == [
            "function_call",
            "message",
        ]
        call_idx = item_added[0][1]["output_index"]
        msg_idx = item_added[1][1]["output_index"]
        assert call_idx < msg_idx
        text = "".join(
            payload["delta"] for name, payload in events if name == "response.output_text.delta"
        )
        assert "done." in text
        assert "<tool_call>" not in text

    def test_tool_choice_none_streams_raw_text(self, monkeypatch):
        events = self._run_stream(monkeypatch, self._XML, tool_choice = "none")
        assert not any(
            payload["item"]["type"] == "function_call"
            for name, payload in events
            if name == "response.output_item.added"
        )
        text = "".join(
            payload["delta"] for name, payload in events if name == "response.output_text.delta"
        )
        assert text == self._XML

    def test_healed_call_splits_message_items(self, monkeypatch):
        # Text on both sides of a healed call becomes TWO message items: the
        # healed function_call closes the first, trailing text opens a fresh
        # one with a later output index (native Responses stream shape).
        events = self._run_stream(monkeypatch, f"before {self._XML} after.")
        added = [
            (payload["output_index"], payload["item"]["type"], payload["item"].get("id"))
            for name, payload in events
            if name == "response.output_item.added"
        ]
        assert [item_type for _, item_type, _ in added] == [
            "message",
            "function_call",
            "message",
        ]
        assert [idx for idx, _, _ in added] == sorted(idx for idx, _, _ in added)
        assert added[0][2] != added[2][2]  # distinct message item ids
        # Text deltas attribute to their OWN message item.
        deltas = [
            (payload["item_id"], payload["delta"])
            for name, payload in events
            if name == "response.output_text.delta"
        ]
        assert [d for i, d in deltas if i == added[0][2]] == ["before "]
        assert [d for i, d in deltas if i == added[2][2]] == [" after."]
        # The completed snapshot lists all three items with per-item text.
        completed = [payload for name, payload in events if name == "response.completed"]
        output = completed[0]["response"]["output"]
        assert [item["type"] for item in output] == ["message", "function_call", "message"]
        assert output[0]["content"][0]["text"] == "before "
        assert output[2]["content"][0]["text"] == " after."

    def test_parallel_cap_drops_native_after_healed(self, monkeypatch):
        # parallel_tool_calls=false: a healed call consumed the single allowed
        # slot; a later native structured call (index 0, so it survives
        # _drop_parallel_tool_call_deltas) must not open a second
        # function_call item.
        TestResponsesStreamAdapter._install_stream_mock(
            monkeypatch,
            [
                {"choices": [{"delta": {"content": self._XML}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_up",
                                        "function": {"name": "lookup", "arguments": "{}"},
                                    }
                                ]
                            }
                        }
                    ]
                },
            ],
        )
        payload = ResponsesRequest(
            input = "hi",
            stream = True,
            tools = [self._TOOL],
            parallel_tool_calls = False,
        )
        messages = [ChatMessage(role = "user", content = "hi")]

        async def run():
            response = await _responses_stream(
                payload, messages, TestResponsesStreamAdapter._Request()
            )
            return await TestResponsesStreamAdapter._collect(response)

        events = self._ordered_events(asyncio.run(run()))
        calls = [
            payload
            for name, payload in events
            if name == "response.output_item.added" and payload["item"]["type"] == "function_call"
        ]
        assert len(calls) == 1
        assert calls[0]["item"]["name"] == "lookup"
