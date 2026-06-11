# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the OpenAI /v1/chat/completions client-side tool pass-through.

Covers ChatMessage tool/assistant roles, ChatCompletionRequest tool fields and
extra="allow", anthropic_tool_choice_to_openai, _build_passthrough_payload
tool_choice propagation, and _friendly_error's httpx-to-"Lost connection"
mapping. No server or GPU required.
"""

import os
import sys
import base64
import asyncio
import json
from io import BytesIO
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from PIL import Image

from models.inference import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionChoice,
    CompletionMessage,
)
from core.inference.anthropic_compat import (
    anthropic_tool_choice_to_openai,
)
import routes.inference as route
from routes.inference import (
    _build_openai_passthrough_body,
    _build_passthrough_payload,
    _clamp_finish_reason,
    _effective_max_tokens,
    _extract_content_parts,
    _friendly_error,
    _openai_chat_completions_impl,
    _openai_stream_usage_chunk,
    _set_or_prepend_system_message,
    openai_chat_completions,
)
from state.tool_policy import reset_tool_policy


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
        # Assistant messages carrying only tool_calls are the one documented
        # case where `content=None` is permitted.
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
        # ChatCompletionRequest's walkback fills it from the prior assistant
        # tool_calls; see test_inference_model_validation.py for resolution
        # coverage.
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
        # Stop-button leaves an empty assistant turn; tolerate for replay.
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
        assert self._make(stop = ["\nUser:", "\nAssistant:"]).stop == ["\nUser:", "\nAssistant:"]

    def test_tools_default_none(self):
        req = self._make()
        assert req.tools is None
        assert req.tool_choice is None
        assert req.stop is None

    def test_extra_fields_accepted(self):
        # `frequency_penalty` and `response_format` are not yet explicitly
        # declared but must survive Pydantic parsing now that extra="allow" is
        # set. `seed` is declared and should land on the typed field instead.
        req = self._make(
            frequency_penalty = 0.5,
            seed = 42,
            response_format = {"type": "json_object"},
        )
        assert req.seed == 42
        # Extras land in model_extra
        assert req.model_extra is not None
        assert req.model_extra.get("frequency_penalty") == 0.5
        assert "seed" not in req.model_extra
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
        # OpenAI defaults `stream` to false. Studio used to default true,
        # breaking naive curl/.NET clients (#5047) that omit it. Pin the fix.
        req = self._make()
        assert req.stream is False

    def test_post_without_stream_field_decodes_to_stream_false_over_http(self, monkeypatch):
        # Wire-level guard: a POST body omitting `stream` must deserialise to
        # stream=False and return application/json, never text/event-stream.
        # Mounts the real router to catch middleware/aliasing regressions;
        # backends are bypassed via provider_type + a stubbed proxy.
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

    def _v1_client(
        self,
        monkeypatch,
        llama_backend,
        inference_backend = None,
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import routes.inference as inference_route
        from auth.authentication import get_current_subject
        from utils.api_errors import install_api_error_handlers

        monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: llama_backend)
        if inference_backend is not None:
            monkeypatch.setattr(inference_route, "get_inference_backend", lambda: inference_backend)

        app = FastAPI()
        app.include_router(inference_route.router, prefix = "/v1")
        install_api_error_handlers(app)
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
        return TestClient(app)

    def _assert_unsupported_param(self, response, param):
        assert response.status_code == 400
        body = response.json()
        assert body["error"]["param"] == param
        assert body["error"]["code"] == "unsupported_parameter"

    def _assert_unsupported_n(self, response):
        self._assert_unsupported_param(response, "n")

    def test_n_allows_openai_chat_completion_range(self):
        req = self._make(n = 128)
        assert req.n == 128
        with pytest.raises(ValidationError):
            self._make(n = 129)

    def test_n_rejected_for_external_provider_path(self, monkeypatch):
        class _UnusedBackend:
            is_loaded = False

        client = self._v1_client(monkeypatch, _UnusedBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "provider_type": "openai",
                "n": 2,
            },
        )
        self._assert_unsupported_n(resp)

    def test_logprobs_rejected_until_supported(self, monkeypatch):
        class _UnusedBackend:
            is_loaded = False

        client = self._v1_client(monkeypatch, _UnusedBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "provider_type": "openai",
                "logprobs": True,
            },
        )
        self._assert_unsupported_param(resp, "logprobs")

    def test_top_logprobs_rejected_until_supported(self, monkeypatch):
        class _UnusedBackend:
            is_loaded = False

        client = self._v1_client(monkeypatch, _UnusedBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "provider_type": "openai",
                "top_logprobs": 3,
            },
        )
        self._assert_unsupported_param(resp, "top_logprobs")

    def test_n_rejected_for_gguf_streaming_path(self, monkeypatch):
        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = False
            is_vision = False
            _is_audio = False

        client = self._v1_client(monkeypatch, _GGUFBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "n": 2,
            },
        )
        self._assert_unsupported_n(resp)

    def test_n_rejected_for_gguf_tools_passthrough_path(self, monkeypatch):
        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = True
            is_vision = False
            _is_audio = False

        client = self._v1_client(monkeypatch, _GGUFBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
                "n": 2,
            },
        )
        self._assert_unsupported_n(resp)

    def test_n_rejected_for_non_gguf_path(self, monkeypatch):
        class _NoGGUFBackend:
            is_loaded = False

        class _InferenceBackend:
            active_model_name = "test-model"
            models = {"test-model": {}}

        client = self._v1_client(monkeypatch, _NoGGUFBackend(), _InferenceBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "n": 2,
            },
        )
        self._assert_unsupported_n(resp)

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


def _png_data_url() -> str:
    img = Image.new("RGB", (2, 2), (0, 255, 0))
    buf = BytesIO()
    img.save(buf, format = "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class TestOpenAIPassthroughImageSafety:
    def test_rejects_too_many_content_part_images(self, monkeypatch):
        monkeypatch.setattr(route, "_OPENAI_CHAT_MAX_IMAGES", 1)
        data_url = _png_data_url()
        req = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            tools = [{"type": "function", "function": {"name": "noop"}}],
        )

        with pytest.raises(HTTPException) as exc:
            route._openai_messages_for_passthrough(req, is_vision = True)

        assert exc.value.status_code == 413

    def test_rejects_passthrough_image_when_model_is_text_only(self):
        req = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _png_data_url()}},
                    ],
                }
            ],
            tools = [{"type": "function", "function": {"name": "noop"}}],
        )

        with pytest.raises(HTTPException) as exc:
            route._openai_messages_for_passthrough(req, is_vision = False)

        assert exc.value.status_code == 400

    def test_top_level_image_uses_size_guard(self, monkeypatch):
        monkeypatch.setattr(route, "_OPENAI_CHAT_MAX_IMAGE_BYTES", 1)
        monkeypatch.setattr(route, "_OPENAI_CHAT_MAX_IMAGE_BASE64_CHARS", 10_000)
        req = ChatCompletionRequest(
            messages = [{"role": "user", "content": "see image"}],
            image_base64 = _png_data_url().split(",", 1)[1],
            tools = [{"type": "function", "function": {"name": "noop"}}],
        )

        with pytest.raises(HTTPException) as exc:
            route._openai_messages_for_passthrough(req, is_vision = True)

        assert exc.value.status_code == 413


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
        result = anthropic_tool_choice_to_openai({"type": "tool", "name": "get_weather"})
        assert result == {"type": "function", "function": {"name": "get_weather"}}

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

    def test_stream_omits_usage_options_when_client_did_not_request_them(self):
        args = self._args()
        args["stream"] = True
        body = _build_passthrough_payload(**args)
        assert "stream_options" not in body

    def test_stream_forwards_include_usage_when_client_requests_it(self):
        args = self._args()
        args["stream"] = True
        body = _build_passthrough_payload(
            **args,
            stream_options = {"include_usage": True},
        )
        assert body.get("stream_options") == {"include_usage": True}

    def test_stream_forwards_include_usage_false_when_client_requests_it(self):
        args = self._args()
        args["stream"] = True
        body = _build_passthrough_payload(
            **args,
            stream_options = {"include_usage": False},
        )
        assert body.get("stream_options") == {"include_usage": False}

    def test_repetition_penalty_renamed(self):
        body = _build_passthrough_payload(**self._args(), repetition_penalty = 1.1)
        assert body.get("repeat_penalty") == 1.1
        assert "repetition_penalty" not in body

    def test_passthrough_body_merges_system_and_developer_messages(self):
        payload = ChatCompletionRequest(
            model = "default",
            messages = [
                {"role": "system", "content": "original system"},
                {"role": "developer", "content": "developer rules"},
                {"role": "user", "content": "hi"},
            ],
            tools = self._args()["openai_tools"],
        )

        body = _build_openai_passthrough_body(payload, backend_ctx = 4096)

        assert body["messages"] == [
            {"role": "system", "content": "original system\n\ndeveloper rules"},
            {"role": "user", "content": "hi"},
        ]


# =====================================================================
# Passthrough reasoning kwargs — enable_thinking / reasoning_effort /
# preserve_thinking must reach llama-server via chat_template_kwargs,
# gated on template capabilities like the non-passthrough paths.
# =====================================================================


def _reasoning_backend(
    supports_reasoning = True,
    reasoning_style = "enable_thinking",
    reasoning_always_on = False,
    supports_preserve_thinking = False,
):
    """Bare LlamaCppBackend with just the reasoning capability flags set,
    so _build_openai_passthrough_body exercises the real
    _request_reasoning_kwargs gating."""
    from core.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._supports_reasoning = supports_reasoning
    backend._reasoning_style = reasoning_style
    backend._reasoning_always_on = reasoning_always_on
    backend._supports_preserve_thinking = supports_preserve_thinking
    return backend


class TestPassthroughReasoningKwargs:
    def _payload(self, **fields):
        return ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hi"}],
            **fields,
        )

    def test_enable_thinking_forwarded(self):
        body = _build_openai_passthrough_body(
            self._payload(enable_thinking = False),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(),
        )
        assert body["chat_template_kwargs"] == {"enable_thinking": False}

    def test_preserve_thinking_forwarded_when_template_supports_it(self):
        body = _build_openai_passthrough_body(
            self._payload(enable_thinking = True, preserve_thinking = True),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(supports_preserve_thinking = True),
        )
        assert body["chat_template_kwargs"] == {
            "enable_thinking": True,
            "preserve_thinking": True,
        }

    def test_preserve_thinking_dropped_when_template_lacks_it(self):
        body = _build_openai_passthrough_body(
            self._payload(preserve_thinking = True),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(supports_preserve_thinking = False),
        )
        assert "chat_template_kwargs" not in body

    def test_reasoning_effort_forwarded_for_effort_style_models(self):
        body = _build_openai_passthrough_body(
            self._payload(reasoning_effort = "high"),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(reasoning_style = "reasoning_effort"),
        )
        assert body["chat_template_kwargs"] == {"reasoning_effort": "high"}

    def test_enable_thinking_maps_to_effort_for_effort_style_models(self):
        body = _build_openai_passthrough_body(
            self._payload(enable_thinking = False),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(reasoning_style = "reasoning_effort"),
        )
        assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}

    def test_always_on_reasoning_skips_thinking_kwargs(self):
        body = _build_openai_passthrough_body(
            self._payload(enable_thinking = False),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(reasoning_always_on = True),
        )
        assert "chat_template_kwargs" not in body

    def test_no_reasoning_fields_omits_chat_template_kwargs(self):
        body = _build_openai_passthrough_body(
            self._payload(),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(supports_preserve_thinking = True),
        )
        assert "chat_template_kwargs" not in body


# =====================================================================
# OpenAI API compatibility helpers — verified spec edge cases
# =====================================================================


class TestOpenAICompatibilityHelpers:
    def test_max_completion_tokens_wins_over_deprecated_max_tokens(self):
        payload = SimpleNamespace(max_tokens = 128, max_completion_tokens = 64)
        assert _effective_max_tokens(payload) == 64

    @pytest.mark.parametrize(
        "finish_reason",
        ["stop", "length", "tool_calls", "content_filter", "function_call"],
    )
    def test_clamp_finish_reason_preserves_openai_finish_reasons(self, finish_reason):
        assert _clamp_finish_reason(finish_reason) == finish_reason

    def test_clamp_finish_reason_defaults_unknown_to_stop(self):
        assert _clamp_finish_reason(None) == "stop"
        assert _clamp_finish_reason("unexpected") == "stop"

    def test_non_streaming_completion_choice_accepts_tool_calls_finish_reason(self):
        choice = CompletionChoice(
            index = 0,
            message = CompletionMessage(content = ""),
            finish_reason = "tool_calls",
        )
        assert choice.finish_reason == "tool_calls"

    def test_stream_usage_chunk_requires_include_usage(self):
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        payload = SimpleNamespace(stream_options = None)
        assert (
            _openai_stream_usage_chunk(payload, "chatcmpl-test", 123, "model", usage, None) is None
        )

        payload.stream_options = {"include_usage": True}
        line = _openai_stream_usage_chunk(payload, "chatcmpl-test", 123, "model", usage, None)
        assert line is not None
        assert '"choices":[]' in line
        assert '"usage"' in line

    def test_stream_usage_chunk_coerces_nullable_counts(self):
        payload = SimpleNamespace(stream_options = {"include_usage": True})
        line = _openai_stream_usage_chunk(
            payload,
            "chatcmpl-test",
            123,
            "model",
            {"prompt_tokens": None, "completion_tokens": 7, "total_tokens": None},
            None,
        )

        assert line is not None
        parsed = json.loads(line.removeprefix("data: "))
        usage = parsed["usage"]
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 7
        assert usage["total_tokens"] == 7

    def test_developer_message_preserves_existing_system_prompt(self):
        payload = ChatCompletionRequest(
            messages = [
                {"role": "system", "content": "original system"},
                {"role": "developer", "content": "developer rules"},
                {"role": "user", "content": "hi"},
            ]
        )
        for message in payload.messages:
            if message.role == "developer":
                message.role = "system"

        system_prompt, chat_messages, image_b64 = _extract_content_parts(payload.messages)

        assert system_prompt == "original system\n\ndeveloper rules"
        assert chat_messages == [{"role": "user", "content": "hi"}]
        assert image_b64 == []


# =====================================================================
# _friendly_error — httpx transport failures
# =====================================================================


class TestFriendlyErrorHttpx:
    """When llama-server is down, httpx RequestError strings lack the
    "Lost connection to llama-server" substring the sync path keys off, so the
    old substring-only `_friendly_error` returned a useless generic message.
    These tests pin the new isinstance-based mapping.
    """

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
        assert "Lost connection" in _friendly_error(exc)

    def test_non_httpx_unchanged(self):
        # Non-httpx exceptions still fall through to the substring heuristics
        # — a context-size message must still produce "Message too long".
        ctx_msg = "request (4096 tokens) exceeds the available context size (2048 tokens)"
        assert "Message too long" in _friendly_error(ValueError(ctx_msg))

    def test_generic_exception_returns_generic_message(self):
        assert _friendly_error(RuntimeError("unrelated")) == "An internal error occurred"


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
        assert out == [{"role": "user", "content": "hi"}, {"role": "user", "content": "again"}]

    def test_drops_assistant_with_no_content_key(self):
        # exclude_none=True strips the content key entirely; filter must catch it.
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant"},
            {"role": "user", "content": "ok"},
        ]
        out = _drop_empty_assistant_sentinels(msgs)
        assert out == [{"role": "user", "content": "hi"}, {"role": "user", "content": "ok"}]

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
        assert messages[0]["content"][0] == {"type": "text", "text": "describe image one"}
        assert messages[0]["content"][1]["type"] == "image_url"
        assert len(messages[0]["content"]) == 2
        assert messages[2]["content"][0] == {"type": "text", "text": "describe image two"}
        assert messages[2]["content"][1]["type"] == "image_url"
        assert len(messages[2]["content"]) == 2
        assert isinstance(messages[1]["content"], str)

        # Legacy top-level image_base64 must be ignored when a message-level
        # image exists; otherwise turn 2 ends up with two image parts.
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
        assert messages[0]["content"][0] == {"type": "text", "text": "describe this image"}
        assert messages[0]["content"][1]["type"] == "image_url"
        assert messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")

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

    def test_tool_nudge_system_update_preserves_image_parts(self):
        messages = [
            {"role": "system", "content": "Base instructions."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self._PNG_B64}",
                        },
                    },
                ],
            },
        ]

        updated = _set_or_prepend_system_message(
            messages, "Base instructions.\n\nUse tools when appropriate."
        )

        assert updated[0] == {
            "role": "system",
            "content": "Base instructions.\n\nUse tools when appropriate.",
        }
        assert updated[1]["content"][1]["type"] == "image_url"
        assert messages[1]["content"][1]["type"] == "image_url"

    def test_tool_nudge_system_update_handles_none_messages(self):
        assert _set_or_prepend_system_message(None, "") == []
        assert _set_or_prepend_system_message(None, "Use tools.") == [
            {"role": "system", "content": "Use tools."}
        ]

    def test_tool_nudge_system_update_dedupes_non_leading_system(self):
        messages = [
            {"role": "user", "content": "earlier"},
            {"role": "system", "content": "Mid instructions."},
            {"role": "user", "content": "now"},
        ]

        updated = _set_or_prepend_system_message(messages, "Mid instructions.\n\nUse tools.")

        assert [m["role"] for m in updated] == ["system", "user", "user"]
        assert updated[0]["content"] == "Mid instructions.\n\nUse tools."


class TestGgufVisionToolRouting:
    class _Request:
        async def is_disconnected(self):
            return False

    @staticmethod
    def _drive(coro):
        return asyncio.run(coro)

    @staticmethod
    def _consume_response(response):
        async def _consume():
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)
            return chunks

        return TestGgufVisionToolRouting._drive(_consume())

    def test_image_request_with_enabled_tools_enters_gguf_tool_loop(self, monkeypatch):
        import routes.inference as inf_mod

        reset_tool_policy()
        captured = {}

        def _plain(**kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**kwargs):
            captured["kwargs"] = kwargs
            yield {"type": "content", "text": "done"}

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = True,
            supports_tools = True,
            model_identifier = "gemma-4-12b-it-GGUF",
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            enable_tools = True,
            enabled_tools = ["web_search"],
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (f"data:image/png;base64,{TestGgufVisionMessages._PNG_B64}"),
                            },
                        },
                    ],
                },
            ],
        )

        response = self._drive(_openai_chat_completions_impl(payload, self._Request()))
        self._consume_response(response)

        assert "kwargs" in captured
        assert captured["kwargs"]["tools"]
        tool_messages = captured["kwargs"]["messages"]
        assert tool_messages[0]["role"] == "system"
        assert tool_messages[1]["role"] == "user"
        assert tool_messages[1]["content"][1]["type"] == "image_url"

    def test_parallel_tool_calls_false_reaches_gguf_tool_loop(self, monkeypatch):
        import routes.inference as inf_mod

        reset_tool_policy()
        captured = {}

        def _plain(**kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**kwargs):
            captured["kwargs"] = kwargs
            yield {"type": "content", "text": "done"}

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            model_identifier = "test-gguf",
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            enable_tools = True,
            enabled_tools = ["web_search"],
            parallel_tool_calls = False,
            messages = [{"role": "user", "content": "search once"}],
        )

        response = self._drive(_openai_chat_completions_impl(payload, self._Request()))
        self._consume_response(response)

        assert captured["kwargs"]["disable_parallel_tool_use"] is True

    def test_standard_gguf_merges_system_and_developer_messages(self, monkeypatch):
        import routes.inference as inf_mod

        captured = {}

        def _generate(**kwargs):
            captured["messages"] = kwargs["messages"]
            yield "done"
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                "finish_reason": "stop",
            }

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = False,
            model_identifier = "test-gguf",
            generate_chat_completion = _generate,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            messages = [
                {"role": "system", "content": "original system"},
                {"role": "developer", "content": "developer rules"},
                {"role": "user", "content": "hi"},
            ],
        )

        self._drive(_openai_chat_completions_impl(payload, self._Request()))

        assert captured["messages"] == [
            {"role": "system", "content": "original system\n\ndeveloper rules"},
            {"role": "user", "content": "hi"},
        ]

    @pytest.mark.parametrize(
        ("seed", "expected"),
        [
            (41, [41, 42, 43]),
            (-1, [-1, -1, -1]),
        ],
    )
    def test_gguf_n_choices_vary_explicit_non_negative_seed(self, monkeypatch, seed, expected):
        import routes.inference as inf_mod

        seen_seeds = []

        def _generate(**kwargs):
            seen_seeds.append(kwargs.get("seed"))
            yield f"choice-{len(seen_seeds)}"
            yield {
                "type": "metadata",
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 7,
                    "total_tokens": 12,
                },
                "finish_reason": "stop",
            }

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = False,
            model_identifier = "test-gguf",
            generate_chat_completion = _generate,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hi"}],
            n = 3,
            seed = seed,
        )

        response = self._drive(_openai_chat_completions_impl(payload, self._Request()))
        body = json.loads(response.body)

        assert seen_seeds == expected
        assert [choice["index"] for choice in body["choices"]] == [0, 1, 2]
