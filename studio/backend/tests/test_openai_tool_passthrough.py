# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""
Tests for the OpenAI /v1/chat/completions client-side tool pass-through.

Covers:
- ChatCompletionRequest accepts standard OpenAI `tools` / `tool_choice` / `stop`.
- ChatMessage accepts role="tool" with `tool_call_id` and role="assistant"
  with `content: None` + `tool_calls`.
- ChatCompletionRequest carries unknown fields via `extra="allow"`.
- anthropic_tool_choice_to_openai() covers all four Anthropic shapes.
- _build_passthrough_payload() honors a caller-supplied tool_choice and
  defaults to "auto" when unset.
- _friendly_error() maps httpx transport errors to legible messages
  instead of bare 500s.

No running server or GPU required.
"""

import os
import sys

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.inference import (
    ChatCompletionRequest,
    ChatMessage,
)
from core.inference.anthropic_compat import (
    anthropic_tool_choice_to_openai,
)
from routes.inference import (
    _build_passthrough_payload,
    _format_timeout_label,
    _friendly_error,
)


# =====================================================================
# ChatMessage — tool role, tool_calls, optional content
# =====================================================================


class TestChatMessageToolRoles:
    def test_tool_role_with_tool_call_id(self):
        msg = ChatMessage(
            role = "tool",
            tool_call_id = "call_abc123",
            content = '{"temperature": 72}',
        )
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_abc123"
        assert msg.content == '{"temperature": 72}'

    def test_tool_role_with_name(self):
        msg = ChatMessage(
            role = "tool",
            tool_call_id = "call_abc123",
            name = "get_weather",
            content = '{"temperature": 72}',
        )
        assert msg.name == "get_weather"

    def test_assistant_with_tool_calls_no_content(self):
        msg = ChatMessage(
            role = "assistant",
            content = None,
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}',
                    },
                }
            ],
        )
        assert msg.role == "assistant"
        assert msg.content is None
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["function"]["name"] == "get_weather"

    def test_assistant_with_content_and_tool_calls(self):
        msg = ChatMessage(
            role = "assistant",
            content = "Let me check the weather.",
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        )
        assert msg.content == "Let me check the weather."
        assert msg.tool_calls[0]["id"] == "call_1"

    def test_plain_user_message_still_works(self):
        msg = ChatMessage(role = "user", content = "Hello")
        assert msg.role == "user"
        assert msg.tool_call_id is None
        assert msg.tool_calls is None
        assert msg.name is None

    def test_invalid_role_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role = "function", content = "x")

    def test_content_absent_on_assistant_tool_call_defaults_to_none(self):
        # Assistant messages that carry only tool_calls are the one
        # documented case where `content=None` is permitted.
        msg = ChatMessage(
            role = "assistant",
            tool_calls = [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        )
        assert msg.content is None

    def test_tool_role_missing_tool_call_id_left_for_request_validator(self):
        # Per-message: missing tool_call_id is now allowed at this layer.
        # ChatCompletionRequest's walkback fills it in from the prior
        # assistant tool_calls; see test_inference_model_validation.py for
        # the resolution coverage.
        msg = ChatMessage(role = "tool", content = '{"temperature": 72}')
        assert msg.tool_call_id is None
        assert msg.content == '{"temperature": 72}'

    def test_tool_role_empty_tool_call_id_left_for_request_validator(self):
        msg = ChatMessage(
            role = "tool",
            tool_call_id = "",
            content = '{"temperature": 72}',
        )
        # Empty-string is treated the same as missing by the walkback.
        assert msg.tool_call_id in (None, "")

    # ── Role-aware content requirements ────────────────────────────

    @pytest.mark.parametrize("role", ["user", "system"])
    def test_empty_string_content_allowed(self, role):
        msg = ChatMessage(role = role, content = "")
        assert msg.content == ""

    def test_user_missing_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role = "user")

    def test_user_empty_list_content_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(role = "user", content = [])

    def test_tool_empty_content_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "tool", tool_call_id = "call_1", content = "")
        assert "content" in str(exc_info.value)

    def test_assistant_without_content_or_tool_calls_tolerated(self):
        # Stop-button leaves an empty assistant turn; tolerate so replay round-trips.
        msg = ChatMessage(role = "assistant")
        assert msg.content is None
        assert msg.tool_calls is None

    def test_assistant_empty_string_content_normalised_to_none(self):
        msg = ChatMessage(role = "assistant", content = "")
        assert msg.content is None

    def test_assistant_empty_list_content_normalised_to_none(self):
        msg = ChatMessage(role = "assistant", content = [])
        assert msg.content is None

    # ── Role-constrained tool-call metadata ────────────────────────

    def test_tool_calls_on_user_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(
                role = "user",
                content = "Hi",
                tool_calls = [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            )
        assert "tool_calls" in str(exc_info.value)

    def test_tool_call_id_on_user_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "user", content = "Hi", tool_call_id = "call_1")
        assert "tool_call_id" in str(exc_info.value)

    def test_name_on_user_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "user", content = "Hi", name = "get_weather")
        assert "name" in str(exc_info.value)


# =====================================================================
# ChatCompletionRequest — standard OpenAI tool fields
# =====================================================================


class TestChatCompletionRequestToolFields:
    def _make(self, **kwargs):
        base = {"messages": [{"role": "user", "content": "Hi"}]}
        base.update(kwargs)
        return ChatCompletionRequest(**base)

    def test_tools_parses(self):
        req = self._make(
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Return the weather in a city",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )
        assert req.tools is not None
        assert len(req.tools) == 1
        assert req.tools[0]["function"]["name"] == "get_weather"

    def test_image_base64_allows_empty_user_text(self):
        req = ChatCompletionRequest(
            messages = [{"role": "user", "content": ""}],
            image_base64 = "aW1hZ2U=",
        )
        assert req.messages[0].content == ""
        assert req.image_base64 == "aW1hZ2U="

    def test_tool_choice_string_auto(self):
        assert self._make(tool_choice = "auto").tool_choice == "auto"

    def test_tool_choice_string_required(self):
        assert self._make(tool_choice = "required").tool_choice == "required"

    def test_tool_choice_string_none(self):
        assert self._make(tool_choice = "none").tool_choice == "none"

    def test_tool_choice_named_function(self):
        tc = {"type": "function", "function": {"name": "get_weather"}}
        assert self._make(tool_choice = tc).tool_choice == tc

    def test_stop_string(self):
        assert self._make(stop = "\nUser:").stop == "\nUser:"

    def test_stop_list(self):
        assert self._make(stop = ["\nUser:", "\nAssistant:"]).stop == [
            "\nUser:",
            "\nAssistant:",
        ]

    def test_tools_default_none(self):
        req = self._make()
        assert req.tools is None
        assert req.tool_choice is None
        assert req.stop is None

    def test_extra_fields_accepted(self):
        # `frequency_penalty`, `seed`, `response_format` are not yet
        # explicitly declared but must survive Pydantic parsing now that
        # extra="allow" is set.
        req = self._make(
            frequency_penalty = 0.5,
            seed = 42,
            response_format = {"type": "json_object"},
        )
        # Extras land in model_extra
        assert req.model_extra is not None
        assert req.model_extra.get("frequency_penalty") == 0.5
        assert req.model_extra.get("seed") == 42
        assert req.model_extra.get("response_format") == {"type": "json_object"}

    def test_unsloth_extensions_still_work(self):
        req = self._make(
            enable_tools = True,
            enabled_tools = ["web_search", "python"],
            session_id = "abc",
        )
        assert req.enable_tools is True
        assert req.enabled_tools == ["web_search", "python"]
        assert req.session_id == "abc"

    def test_stream_defaults_false_matching_openai_spec(self):
        # OpenAI's /v1/chat/completions spec defaults `stream` to false.
        # Studio previously defaulted to true, which broke naive curl
        # clients (and .NET / System.Text.Json SDKs per #5047) that omit
        # `stream` -- they expect a JSON blob, got SSE.
        # Pin the corrected default so it can't silently regress.
        req = self._make()
        assert req.stream is False

    def test_post_without_stream_field_decodes_to_stream_false_over_http(
        self, monkeypatch
    ):
        # Wire-level guard for the same default: a POST body that omits
        # `stream` entirely (the exact shape naive curl / .NET clients
        # send) must deserialise into stream=False *and* the response
        # must be `application/json`, never `text/event-stream`.
        # Mounts the real `routes.inference.router` so this catches
        # regressions in middleware/aliasing on the actual endpoint
        # (e.g. someone adding a request layer that injects stream=True
        # before pydantic builds the model). Backends are bypassed by
        # routing through `provider_type` and stubbing the external
        # provider proxy.
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        import routes.inference as inference_route
        from auth.authentication import get_current_subject

        captured = {}

        async def _fake_proxy(payload, request):
            captured["stream"] = payload.stream
            return JSONResponse({"choices": [], "object": "chat.completion"})

        monkeypatch.setattr(inference_route, "_proxy_to_external_provider", _fake_proxy)

        app = FastAPI()
        app.include_router(inference_route.router)
        app.dependency_overrides[get_current_subject] = lambda: "test-user"

        client = TestClient(app)
        resp = client.post(
            "/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "provider_type": "openai",
            },
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert "text/event-stream" not in resp.headers["content-type"]
        assert captured["stream"] is False

    def test_multiturn_tool_loop_messages(self):
        req = ChatCompletionRequest(
            messages = [
                {"role": "user", "content": "What's the weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": '{"temperature": 14, "unit": "celsius"}',
                },
            ],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        assert len(req.messages) == 3
        assert req.messages[1].role == "assistant"
        assert req.messages[1].content is None
        assert req.messages[1].tool_calls[0]["id"] == "call_1"
        assert req.messages[2].role == "tool"
        assert req.messages[2].tool_call_id == "call_1"


# =====================================================================
# anthropic_tool_choice_to_openai — pure translation helper
# =====================================================================


class TestAnthropicToolChoiceToOpenAI:
    def test_auto(self):
        assert anthropic_tool_choice_to_openai({"type": "auto"}) == "auto"

    def test_any_becomes_required(self):
        assert anthropic_tool_choice_to_openai({"type": "any"}) == "required"

    def test_none(self):
        assert anthropic_tool_choice_to_openai({"type": "none"}) == "none"

    def test_tool_named(self):
        result = anthropic_tool_choice_to_openai(
            {"type": "tool", "name": "get_weather"}
        )
        assert result == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_missing_name_returns_none(self):
        assert anthropic_tool_choice_to_openai({"type": "tool"}) is None

    def test_none_input_returns_none(self):
        assert anthropic_tool_choice_to_openai(None) is None

    def test_unrecognized_shape_returns_none(self):
        assert anthropic_tool_choice_to_openai({"type": "wibble"}) is None
        assert anthropic_tool_choice_to_openai("auto") is None
        assert anthropic_tool_choice_to_openai(42) is None


# =====================================================================
# _build_passthrough_payload — tool_choice propagation
# =====================================================================


class TestBuildPassthroughPayloadToolChoice:
    def _args(self):
        return dict(
            openai_messages = [{"role": "user", "content": "Hi"}],
            openai_tools = [
                {
                    "type": "function",
                    "function": {"name": "f", "parameters": {"type": "object"}},
                }
            ],
            temperature = 0.6,
            top_p = 0.95,
            top_k = 20,
            max_tokens = 128,
            stream = False,
        )

    def test_default_tool_choice_is_auto(self):
        body = _build_passthrough_payload(**self._args())
        assert body["tool_choice"] == "auto"

    def test_override_tool_choice_required(self):
        body = _build_passthrough_payload(**self._args(), tool_choice = "required")
        assert body["tool_choice"] == "required"

    def test_override_tool_choice_none(self):
        body = _build_passthrough_payload(**self._args(), tool_choice = "none")
        assert body["tool_choice"] == "none"

    def test_override_tool_choice_named_function(self):
        tc = {"type": "function", "function": {"name": "f"}}
        body = _build_passthrough_payload(**self._args(), tool_choice = tc)
        assert body["tool_choice"] == tc

    def test_stream_adds_include_usage(self):
        args = self._args()
        args["stream"] = True
        body = _build_passthrough_payload(**args)
        assert body.get("stream_options") == {"include_usage": True}

    def test_generation_wall_clock_cap_not_forwarded(self):
        body = _build_passthrough_payload(**self._args())
        assert "t_max_predict_ms" not in body

    def test_repetition_penalty_renamed(self):
        body = _build_passthrough_payload(**self._args(), repetition_penalty = 1.1)
        assert body.get("repeat_penalty") == 1.1
        assert "repetition_penalty" not in body


# =====================================================================
# _friendly_error — httpx transport failures
# =====================================================================


class TestFriendlyErrorHttpx:
    """The async pass-through helpers talk to llama-server via httpx."""

    def _req(self):
        return httpx.Request("POST", "http://127.0.0.1:65535/v1/chat/completions")

    def test_connect_error_mapped(self):
        exc = httpx.ConnectError("All connection attempts failed", request = self._req())
        assert "Lost connection" in _friendly_error(exc)

    def test_read_error_mapped(self):
        exc = httpx.ReadError("EOF", request = self._req())
        assert "Lost connection" in _friendly_error(exc)

    def test_remote_protocol_error_mapped(self):
        exc = httpx.RemoteProtocolError("peer closed", request = self._req())
        assert "Lost connection" in _friendly_error(exc)

    def test_read_timeout_mapped(self):
        exc = httpx.ReadTimeout("timed out", request = self._req())
        assert "first token within 10 minutes" in _friendly_error(exc)

    def test_timeout_label_singular_plural(self):
        assert _format_timeout_label(1.0) == "1 second"
        assert _format_timeout_label(1.5) == "1.5 seconds"
        assert _format_timeout_label(2.0) == "2 seconds"
        assert _format_timeout_label(60.0) == "1 minute"
        assert _format_timeout_label(120.0) == "2 minutes"
        assert _format_timeout_label(90.0) == "90 seconds"

    def test_non_httpx_unchanged(self):
        # Non-httpx exceptions still fall through to the existing substring
        # heuristics — a context-size message must still produce the
        # "Message too long" path.
        ctx_msg = (
            "request (4096 tokens) exceeds the available context size (2048 tokens)"
        )
        assert "Message too long" in _friendly_error(ValueError(ctx_msg))

    def test_generic_exception_returns_generic_message(self):
        assert (
            _friendly_error(RuntimeError("unrelated")) == "An internal error occurred"
        )


from routes.inference import (  # noqa: E402
    _drop_empty_assistant_sentinels,
    _openai_messages_for_gguf_chat,
    _openai_messages_for_passthrough,
)


class TestDropEmptyAssistantSentinels:
    def test_drops_empty_assistant_between_real_turns(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"},
        ]
        out = _drop_empty_assistant_sentinels(msgs)
        assert out == [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "again"},
        ]

    def test_drops_assistant_with_no_content_key(self):
        # exclude_none=True strips the content key entirely; filter must catch this.
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "user", "content": "ok"},
        ]
        out = _drop_empty_assistant_sentinels(msgs)
        assert out == [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "ok"},
        ]

    def test_preserves_assistant_with_text(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello back"},
        ]
        out = _drop_empty_assistant_sentinels(msgs)
        assert out == msgs

    def test_preserves_assistant_with_tool_calls_only(self):
        msgs = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"t": 72}',
            },
        ]
        out = _drop_empty_assistant_sentinels(msgs)
        assert out == msgs

    def test_preserves_user_and_system_with_empty_content(self):
        # Filter scoped to role="assistant" only.
        msgs = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
        ]
        out = _drop_empty_assistant_sentinels(msgs)
        assert out == msgs

    def test_openai_messages_for_passthrough_drops_sentinel(self):
        """End-to-end: Stop-sentinel must not reach the wire."""
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                ChatMessage(role = "user", content = "hi"),
                ChatMessage(role = "assistant", content = ""),
                ChatMessage(role = "user", content = "again"),
            ],
        )
        out = _openai_messages_for_passthrough(req)
        roles = [m["role"] for m in out]
        assert roles == ["user", "user"]
        for m in out:
            assert m.get("content"), m


class TestGgufVisionMessages:
    _PNG_B64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mNk"
        "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )

    def test_preserves_multiturn_image_parts_on_original_turns(self):
        req = ChatCompletionRequest(
            model = "default",
            image_base64 = self._PNG_B64,
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe image one"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._PNG_B64}",
                            },
                        },
                    ],
                },
                {"role": "assistant", "content": "first answer"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe image two"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._PNG_B64}",
                            },
                        },
                    ],
                },
            ],
        )

        messages, has_image = _openai_messages_for_gguf_chat(req, is_vision = True)

        assert has_image is True
        assert messages[0]["content"][0] == {
            "type": "text",
            "text": "describe image one",
        }
        assert messages[0]["content"][1]["type"] == "image_url"
        assert len(messages[0]["content"]) == 2
        assert messages[2]["content"][0] == {
            "type": "text",
            "text": "describe image two",
        }
        assert messages[2]["content"][1]["type"] == "image_url"
        assert len(messages[2]["content"]) == 2
        assert isinstance(messages[1]["content"], str)

        # Legacy top-level image_base64 must be ignored when any message-level
        # image already exists; otherwise turn 2 ends up with two image parts.
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                image_parts = [p for p in content if p.get("type") == "image_url"]
                assert len(image_parts) == 1, msg

    def test_legacy_image_base64_is_injected_when_messages_are_text_only(self):
        req = ChatCompletionRequest(
            model = "default",
            image_base64 = self._PNG_B64,
            messages = [{"role": "user", "content": "describe this image"}],
        )

        messages, has_image = _openai_messages_for_gguf_chat(req, is_vision = True)

        assert has_image is True
        assert messages[0]["content"][0] == {
            "type": "text",
            "text": "describe this image",
        }
        assert messages[0]["content"][1]["type"] == "image_url"
        assert messages[0]["content"][1]["image_url"]["url"].startswith(
            "data:image/png;base64,"
        )

    def test_rejects_image_parts_for_text_only_gguf(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._PNG_B64}",
                            },
                        },
                    ],
                },
            ],
        )

        with pytest.raises(HTTPException) as exc_info:
            _openai_messages_for_gguf_chat(req, is_vision = False)
        assert "does not support vision" in str(exc_info.value)
