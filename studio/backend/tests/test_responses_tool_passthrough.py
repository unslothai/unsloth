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
    _build_chat_request,
    _chat_tool_calls_to_responses_output,
    _normalise_responses_input,
    _responses_tool_output_text,
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
        assert _responses_tool_output_text("done") == "done"
        assert _responses_tool_output_text("  done  ") == "  done  "


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
