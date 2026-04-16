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
- _friendly_error() maps httpx transport errors to a "Lost connection"
  message so passthrough failures are legible instead of bare 500s.
- _llama_auth_headers() returns a Bearer header only when an API key is set.
- _openai_messages_for_passthrough() splices legacy image_base64 into the
  last user message and skips the splice when one is already inline.
- openai_chat_completions() rejects role="tool" / tool_calls-only messages
  when the request does not take the passthrough path.
- anthropic_messages() rejects the enable_tools + tool_choice combination.
- _openai_passthrough_non_streaming() wraps httpx transport errors as 502
  and returns the upstream JSON body verbatim on success.
- _openai_passthrough_stream() relays data: lines verbatim, breaks on
  [DONE], and always emits [DONE] after an upstream error.

No running server or GPU required.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

# conftest.py adds the backend root to sys.path so these flat imports resolve.
from models.inference import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    ChatMessage,
)
from core.inference.anthropic_compat import (
    anthropic_tool_choice_to_openai,
)
from routes import inference as inference_module
from routes.inference import (
    _build_passthrough_payload,
    _friendly_error,
    _llama_auth_headers,
    _openai_messages_for_passthrough,
    _openai_passthrough_non_streaming,
    _openai_passthrough_stream,
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

    def test_content_absent_defaults_to_none(self):
        msg = ChatMessage(
            role = "assistant",
            tool_calls = [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        )
        assert msg.content is None

    def test_tool_role_missing_tool_call_id_rejected(self):
        # Per OpenAI spec, role="tool" messages must carry tool_call_id so
        # upstream backends can associate the result with its prior call.
        # Pin the boundary-level rejection so a malformed tool-result
        # message never reaches the passthrough path.
        with pytest.raises(ValidationError) as exc_info:
            ChatMessage(role = "tool", content = '{"temperature": 72}')
        assert "tool_call_id" in str(exc_info.value)

    def test_tool_role_empty_tool_call_id_rejected(self):
        with pytest.raises(ValidationError):
            ChatMessage(
                role = "tool",
                tool_call_id = "",
                content = '{"temperature": 72}',
            )


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
        # clients that omit `stream` (they expect a JSON blob, got SSE).
        # Pin the corrected default so it can't silently regress.
        req = self._make()
        assert req.stream is False

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

    def test_repetition_penalty_renamed(self):
        body = _build_passthrough_payload(**self._args(), repetition_penalty = 1.1)
        assert body.get("repeat_penalty") == 1.1
        assert "repetition_penalty" not in body


# =====================================================================
# _friendly_error — httpx transport failures
# =====================================================================


class TestFriendlyErrorHttpx:
    """The async pass-through helpers talk to llama-server via httpx.
    When the subprocess is down, httpx raises RequestError subclasses
    whose string form (``"All connection attempts failed"``, ``"[Errno 111]
    Connection refused"``, ...) does NOT contain the substring
    ``"Lost connection to llama-server"`` the sync path uses, so the
    previous substring-only `_friendly_error` returned a useless generic
    message. These tests pin the new isinstance-based mapping.
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


# =====================================================================
# _llama_auth_headers — Bearer header only when API key is set
# =====================================================================


class TestLlamaAuthHeaders:
    def test_returns_bearer_header_when_api_key_set(self):
        class _Backend:
            _api_key = "k_secret_123"
        assert _llama_auth_headers(_Backend()) == {
            "Authorization": "Bearer k_secret_123",
        }

    def test_returns_none_when_api_key_none(self):
        class _Backend:
            _api_key = None
        assert _llama_auth_headers(_Backend()) is None

    def test_returns_none_when_api_key_attribute_missing(self):
        class _Backend:
            pass
        assert _llama_auth_headers(_Backend()) is None

    def test_returns_none_when_api_key_empty_string(self):
        class _Backend:
            _api_key = ""
        assert _llama_auth_headers(_Backend()) is None


# =====================================================================
# _openai_messages_for_passthrough — legacy image_base64 splice
# =====================================================================


_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYA"
    "AjCB0C8AAAAASUVORK5CYII="
)


def _inline_image_part(b64 = _TINY_PNG_B64):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


class _PayloadWithImage:
    """Minimal stand-in for ChatCompletionRequest; only attributes read by
    _openai_messages_for_passthrough matter here."""

    def __init__(self, messages, image_base64 = None):
        self.messages = messages
        self.image_base64 = image_base64


class TestOpenAIMessagesForPassthrough:
    def test_prior_user_image_does_not_block_new_base64(self):
        payload = _PayloadWithImage(
            messages = [
                ChatMessage(
                    role = "user",
                    content = [{"type": "text", "text": "q1"}, _inline_image_part()],
                ),
                ChatMessage(role = "assistant", content = "a1"),
                ChatMessage(role = "user", content = "q2 no inline"),
            ],
            image_base64 = _TINY_PNG_B64,
        )
        out = _openai_messages_for_passthrough(payload)
        assert isinstance(out[-1]["content"], list)
        assert any(part.get("type") == "image_url" for part in out[-1]["content"])

    def test_last_user_with_inline_image_skips_splice(self):
        payload = _PayloadWithImage(
            messages = [
                ChatMessage(role = "user", content = "earlier"),
                ChatMessage(
                    role = "user",
                    content = [
                        {"type": "text", "text": "last"},
                        _inline_image_part(),
                    ],
                ),
            ],
            image_base64 = _TINY_PNG_B64,
        )
        out = _openai_messages_for_passthrough(payload)
        last_parts = out[-1]["content"]
        assert sum(1 for part in last_parts if part.get("type") == "image_url") == 1

    def test_no_user_messages_appends_trailing_user(self):
        payload = _PayloadWithImage(
            messages = [ChatMessage(role = "system", content = "sys")],
            image_base64 = _TINY_PNG_B64,
        )
        out = _openai_messages_for_passthrough(payload)
        assert out[-1]["role"] == "user"
        assert any(part.get("type") == "image_url" for part in out[-1]["content"])

    def test_only_last_of_multiple_user_turns_receives_splice(self):
        payload = _PayloadWithImage(
            messages = [
                ChatMessage(role = "user", content = "u1"),
                ChatMessage(role = "assistant", content = "a1"),
                ChatMessage(role = "user", content = "u2"),
                ChatMessage(role = "assistant", content = "a2"),
                ChatMessage(role = "user", content = "u3"),
            ],
            image_base64 = _TINY_PNG_B64,
        )
        out = _openai_messages_for_passthrough(payload)
        assert isinstance(out[0]["content"], str)
        assert isinstance(out[2]["content"], str)
        assert isinstance(out[4]["content"], list)


# =====================================================================
# openai_chat_completions — tool-shape guards on non-passthrough paths
# =====================================================================


class _FakeLlamaBackend:
    def __init__(self, is_loaded = True, supports_tools = True, is_vision = False):
        self.is_loaded = is_loaded
        self.supports_tools = supports_tools
        self.is_vision = is_vision
        self._is_audio = False
        self.model_identifier = "stub"
        self.base_url = "http://127.0.0.1:0"
        self._api_key = None


class _FakeInferenceBackend:
    active_model_name = "hf"
    models = {"hf": {}}


class _FakeFastAPIRequest:
    async def is_disconnected(self):
        return False


async def _marker_stream(*args, **kwargs):
    return ("passthrough_stream", None)


async def _marker_nonstream(*args, **kwargs):
    return ("passthrough_nonstream", None)


async def _call_openai_chat_completions(payload, llama):
    with (
        patch.object(inference_module, "get_llama_cpp_backend", return_value = llama),
        patch.object(
            inference_module, "get_inference_backend", return_value = _FakeInferenceBackend(),
        ),
        patch.object(inference_module, "_openai_passthrough_stream", new = _marker_stream),
        patch.object(
            inference_module, "_openai_passthrough_non_streaming", new = _marker_nonstream,
        ),
    ):
        return await inference_module.openai_chat_completions(
            payload, _FakeFastAPIRequest(), current_subject = "u",
        )


class TestOpenAIChatCompletionsToolGuards:
    """When the request does NOT take the tool-passthrough path, messages
    with role="tool" or assistant tool_calls-only must be rejected at the
    route boundary rather than producing an opaque upstream error."""

    def _tool_result_payload(self, **extra):
        return ChatCompletionRequest(
            messages = [
                {"role": "user", "content": "q"},
                {"role": "tool", "tool_call_id": "c1", "content": "r"},
            ],
            **extra,
        )

    def test_role_tool_on_non_gguf_rejected(self):
        payload = self._tool_result_payload()
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                _call_openai_chat_completions(
                    payload, _FakeLlamaBackend(is_loaded = False),
                )
            )
        assert exc_info.value.status_code == 400

    def test_role_tool_on_gguf_without_tool_support_rejected(self):
        payload = self._tool_result_payload()
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                _call_openai_chat_completions(
                    payload, _FakeLlamaBackend(supports_tools = False),
                )
            )
        assert exc_info.value.status_code == 400

    def test_role_tool_with_enable_tools_true_rejected(self):
        payload = self._tool_result_payload(enable_tools = True)
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_call_openai_chat_completions(payload, _FakeLlamaBackend()))
        assert exc_info.value.status_code == 400

    def test_assistant_tool_calls_only_rejected_when_no_passthrough(self):
        payload = ChatCompletionRequest(
            messages = [
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
            ],
        )
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                _call_openai_chat_completions(
                    payload, _FakeLlamaBackend(supports_tools = False),
                )
            )
        assert exc_info.value.status_code == 400

    def test_plain_user_message_does_not_trip_tool_guard(self):
        payload = ChatCompletionRequest(
            messages = [{"role": "user", "content": "plain"}],
        )
        try:
            asyncio.run(
                _call_openai_chat_completions(
                    payload, _FakeLlamaBackend(supports_tools = False),
                )
            )
        except HTTPException as exc:
            # Downstream paths may fail for unrelated reasons; only assert
            # the tool-shape guard did NOT fire.
            assert "role='tool'" not in (exc.detail or "")
            assert "tool_calls-only" not in (exc.detail or "")


# =====================================================================
# anthropic_messages — enable_tools + tool_choice conflict
# =====================================================================


class TestAnthropicEnableToolsToolChoiceConflict:
    """Server-side agentic loop (enable_tools=True) does not honor
    tool_choice. Reject the combination at the route boundary with 400
    so callers don't silently see their tool_choice dropped."""

    def _payload(self, *, enable_tools, tool_choice):
        return AnthropicMessagesRequest(
            model = "default",
            max_tokens = 64,
            messages = [{"role": "user", "content": "q"}],
            tool_choice = tool_choice,
            enable_tools = enable_tools,
        )

    def test_enable_tools_true_with_tool_choice_raises_400(self):
        payload = self._payload(enable_tools = True, tool_choice = {"type": "any"})
        with patch.object(
            inference_module,
            "get_llama_cpp_backend",
            return_value = _FakeLlamaBackend(),
        ):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    inference_module.anthropic_messages(
                        payload, _FakeFastAPIRequest(), current_subject = "u",
                    )
                )
        assert exc_info.value.status_code == 400
        assert "tool_choice" in exc_info.value.detail
        assert "enable_tools" in exc_info.value.detail

    def test_enable_tools_false_with_tool_choice_skips_guard(self):
        payload = self._payload(enable_tools = False, tool_choice = {"type": "any"})
        with patch.object(
            inference_module,
            "get_llama_cpp_backend",
            return_value = _FakeLlamaBackend(),
        ):
            try:
                asyncio.run(
                    inference_module.anthropic_messages(
                        payload, _FakeFastAPIRequest(), current_subject = "u",
                    )
                )
            except HTTPException as exc:
                assert not (
                    exc.status_code == 400
                    and "tool_choice is not honored" in (exc.detail or "")
                )
            except Exception:
                # Any other failure downstream is fine; we only pin that
                # the enable_tools+tool_choice guard does NOT fire.
                pass


# =====================================================================
# _openai_passthrough_non_streaming — verbatim body + httpx 502 mapping
# =====================================================================


class _FakeLlamaBase:
    base_url = "http://127.0.0.1:0"
    _api_key = None


def _openai_tools_payload(stream = False):
    return ChatCompletionRequest(
        messages = [{"role": "user", "content": "q"}],
        tools = [
            {
                "type": "function",
                "function": {"name": "f", "parameters": {"type": "object"}},
            }
        ],
        stream = stream,
    )


def _mock_async_client_post(status_code, *, json_body = None, text_body = ""):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            resp = MagicMock()
            resp.status_code = status_code
            resp.text = text_body
            resp.json = lambda: (json_body if json_body is not None else {})
            return resp

    return _Client


def _mock_async_client_raise(exc_factory):
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, *args, **kwargs):
            raise exc_factory()

    return _Client


class TestOpenAIPassthroughNonStreaming:
    def test_verbatim_json_body_returned_on_success(self):
        native = {
            "id": "chatcmpl-foo",
            "object": "chat.completion",
            "model": "qwen-native",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
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
                }
            ],
            "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        }
        client_cls = _mock_async_client_post(200, json_body = native)
        with patch.object(inference_module.httpx, "AsyncClient", client_cls):
            resp = asyncio.run(
                _openai_passthrough_non_streaming(_FakeLlamaBase(), _openai_tools_payload())
            )
        import json
        body = json.loads(resp.body.decode("utf-8"))
        assert body == native
        assert body["choices"][0]["finish_reason"] == "tool_calls"

    def test_preserves_native_id_and_model(self):
        native = {
            "id": "chatcmpl-native-xyz",
            "model": "llama-native",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        client_cls = _mock_async_client_post(200, json_body = native)
        with patch.object(inference_module.httpx, "AsyncClient", client_cls):
            resp = asyncio.run(
                _openai_passthrough_non_streaming(_FakeLlamaBase(), _openai_tools_payload())
            )
        import json
        body = json.loads(resp.body.decode("utf-8"))
        assert body["id"] == "chatcmpl-native-xyz"
        assert body["model"] == "llama-native"

    def test_httpx_connect_error_mapped_to_502(self):
        client_cls = _mock_async_client_raise(
            lambda: httpx.ConnectError(
                "refused", request = httpx.Request("POST", "http://x"),
            )
        )
        with patch.object(inference_module.httpx, "AsyncClient", client_cls):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    _openai_passthrough_non_streaming(
                        _FakeLlamaBase(), _openai_tools_payload(),
                    )
                )
        assert exc_info.value.status_code == 502
        assert "Lost connection" in exc_info.value.detail

    def test_httpx_read_error_mapped_to_502(self):
        client_cls = _mock_async_client_raise(
            lambda: httpx.ReadError("eof", request = httpx.Request("POST", "http://x")),
        )
        with patch.object(inference_module.httpx, "AsyncClient", client_cls):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(
                    _openai_passthrough_non_streaming(
                        _FakeLlamaBase(), _openai_tools_payload(),
                    )
                )
        assert exc_info.value.status_code == 502


# =====================================================================
# _openai_passthrough_stream — verbatim data: lines, [DONE] on error
# =====================================================================


class _FakeStreamResponse:
    """Stand-in for httpx.Response that yields a fixed list of SSE lines."""

    def __init__(self, lines, status_code = 200):
        self.status_code = status_code
        self._lines = list(lines)

    def aiter_lines(self):
        parent = self

        class _Iter:
            def __init__(self):
                self._idx = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self._idx >= len(parent._lines):
                    raise StopAsyncIteration
                v = parent._lines[self._idx]
                self._idx += 1
                return v

            async def aclose(self):
                pass

        return _Iter()

    async def aread(self):
        return b""

    async def aclose(self):
        pass


def _run_passthrough_stream(*, send_returns = None, send_raises = None):
    """Drive _openai_passthrough_stream with a fake httpx client and return
    the list of emitted SSE chunks (decoded to str)."""

    async def _send(req, stream):
        if send_raises is not None:
            raise send_raises()
        return send_returns

    with patch.object(inference_module.httpx, "AsyncClient") as mock_cls:
        instance = MagicMock()
        instance.build_request = MagicMock(return_value = MagicMock())
        instance.send = _send
        instance.aclose = AsyncMock()
        mock_cls.return_value = instance

        resp = asyncio.run(
            _openai_passthrough_stream(
                _FakeFastAPIRequest(),
                threading.Event(),
                _FakeLlamaBase(),
                _openai_tools_payload(stream = True),
            )
        )

        async def _collect():
            return [
                chunk if isinstance(chunk, str) else chunk.decode("utf-8")
                async for chunk in resp.body_iterator
            ]

        return asyncio.run(_collect())


class TestOpenAIPassthroughStreamVerbatim:
    def test_relays_data_lines_verbatim(self):
        chunks = _run_passthrough_stream(
            send_returns = _FakeStreamResponse([
                'data: {"id":"abc","choices":[{"delta":{"content":"hi"}}]}',
                "data: [DONE]",
            ]),
        )
        assert any('"id":"abc"' in c for c in chunks)
        assert any("data: [DONE]" in c for c in chunks)

    def test_ignores_blank_and_non_data_lines(self):
        chunks = _run_passthrough_stream(
            send_returns = _FakeStreamResponse([
                "",
                ": heartbeat",
                'data: {"x":1}',
                "data: [DONE]",
            ]),
        )
        for chunk in chunks:
            assert chunk.startswith("data: ") or chunk == ""
        assert any('"x":1' in c for c in chunks)

    def test_breaks_on_done(self):
        chunks = _run_passthrough_stream(
            send_returns = _FakeStreamResponse([
                'data: {"a":1}',
                "data: [DONE]",
                'data: {"should_not_appear":true}',
            ]),
        )
        assert not any("should_not_appear" in c for c in chunks)


class TestOpenAIPassthroughStreamErrorTermination:
    """On upstream failure (non-200 or transport exception), the stream
    must emit an SSE error chunk followed by `data: [DONE]` so clients
    that wait for [DONE] don't hang."""

    def test_done_emitted_after_non_200_error(self):
        class _Resp:
            status_code = 500

            async def aread(self):
                return b"server oops"

            async def aclose(self):
                pass

        chunks = _run_passthrough_stream(send_returns = _Resp())
        assert any('"error"' in c for c in chunks)
        assert any(c.strip() == "data: [DONE]" for c in chunks)

    def test_done_emitted_after_transport_exception(self):
        chunks = _run_passthrough_stream(
            send_raises = lambda: httpx.ConnectError(
                "boom", request = httpx.Request("POST", "http://x"),
            ),
        )
        assert any('"error"' in c for c in chunks)
        assert any(c.strip() == "data: [DONE]" for c in chunks)

    def test_done_comes_after_error_chunk(self):
        chunks = _run_passthrough_stream(
            send_raises = lambda: httpx.ReadError(
                "reset", request = httpx.Request("POST", "http://x"),
            ),
        )
        err_idx = next(i for i, c in enumerate(chunks) if '"error"' in c)
        done_idx = next(i for i, c in enumerate(chunks) if c.strip() == "data: [DONE]")
        assert done_idx > err_idx
