# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the OpenAI /v1/chat/completions client-side tool pass-through."""

import os
import re
import sys
import asyncio
import base64
import json
import threading
import time
from types import SimpleNamespace

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.inference import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionChoice,
    CompletionMessage,
    ResponsesRequest,
)
from core.inference.anthropic_compat import (
    anthropic_tool_choice_to_openai,
)
from core.inference.api_monitor import ApiMonitor
from core.inference.llama_admission import (
    ADMISSION_KEEPALIVE_INTERVAL_ENV,
    ADMISSION_MAX_QUEUE_ENV,
    ADMISSION_QUEUE_TIMEOUT_ENV,
    LlamaAdmissionCancelled,
    LlamaAdmissionConfig,
    get_llama_admission_queue,
    reset_llama_admission_queues,
)
from routes.inference import (
    _aclose_stream_resources,
    _build_chat_request,
    _build_openai_passthrough_body,
    _build_passthrough_payload,
    _clamp_finish_reason,
    _cmpl_stream_event_out,
    _coalesce_consecutive_user_turns,
    _drop_empty_assistant_sentinels,
    _effective_max_tokens,
    _effective_openai_max_tokens,
    _effective_openai_max_tokens_from_values,
    _extract_content_parts,
    _images_in_last_user_message,
    _friendly_error,
    _friendly_upstream_error,
    _merge_user_content,
    _monitor_openai_chunk,
    _monitor_openai_sse_event,
    _normalize_openai_passthrough_sse_line,
    _openai_compat_stream_stall_timeout,
    _openai_llama_admission_capacity,
    _openai_messages_for_gguf_chat,
    _openai_passthrough_sse_line_terminal_state,
    _openai_passthrough_upstream_headers,
    _openai_passthrough_non_streaming,
    _openai_passthrough_stream,
    _responses_stream,
    _openai_stream_error_sse,
    _openai_stream_usage_chunk,
    _openai_admission_wait_stream_chunks,
    _wait_for_openai_admission_non_streaming,
    _proxy_to_external_provider,
    _SameTaskStreamingResponse,
    _OPENAI_COMPAT_STREAM_STALL_TIMEOUT_ENV,
    _set_or_prepend_system_message,
    _strip_provider_synthetic_tool_history,
    openai_completions,
    openai_embeddings,
    openai_chat_completions,
)
from state.tool_policy import reset_tool_policy, set_tool_policy

# 1x1 PNGs that Pillow can actually decode, for the paths that reach the decoder.
_RED_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
_BLUE_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQAIicLsAAAAAElFTkSuQmCC"
)


@pytest.fixture(autouse = True)
def _reset_admission_queues():
    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


def test_aclose_stream_resources_attempts_remaining_closes_after_cancel():
    class Closeable:
        def __init__(self, *, cancel = False):
            self.cancel = cancel
            self.closed = False

        async def aclose(self):
            self.closed = True
            if self.cancel:
                raise asyncio.CancelledError()

    async def _run():
        iterator = Closeable(cancel = True)
        resp = Closeable()
        client = Closeable()

        with pytest.raises(asyncio.CancelledError):
            await _aclose_stream_resources(iterator = iterator, resp = resp, client = client)

        assert iterator.closed
        assert resp.closed
        assert client.closed

    asyncio.run(_run())


class TestFriendlyUpstreamError:
    def test_grammar_parse_failure_gets_actionable_message(self):
        raw = '{"error":{"code":400,"message":"Failed to initialize samplers: failed to parse grammar","type":"invalid_request_error"}}'
        msg = _friendly_upstream_error(raw)
        assert "failed to parse grammar" not in msg  # raw body is not surfaced verbatim
        assert "compile a grammar" in msg and "Update Unsloth" in msg

    def test_sampler_failure_without_a_grammar_keeps_its_own_text(self):
        # llama-server prefixes every sampler failure the same way, so a bad penalty
        # would otherwise be reported as an uncompilable schema.
        detail = "Failed to initialize samplers: penalty_repeat must be finite and greater than 0"
        msg = _friendly_upstream_error(detail)
        assert "compile a grammar" not in msg
        assert "penalty_repeat" in msg

    def test_message_does_not_blame_the_model(self):
        # The request's schemas decide the failure; swapping the GGUF changes nothing.
        msg = _friendly_upstream_error("failed to parse grammar")
        assert "schema" in msg
        assert "GGUF" not in msg and "quant and tool-schema" not in msg

    def test_unrelated_error_passes_through(self):
        assert _friendly_upstream_error("out of memory") == "llama-server error: out of memory"

    def test_openai_passthrough_error_rewrites_grammar_failure(self):
        # OpenAI-compatible agents (opencode/openclaw/hermes/pi via /v1/chat/completions)
        # get the same actionable message as the Anthropic passthrough, not the raw body.
        from routes.inference import _openai_passthrough_error

        exc = _openai_passthrough_error(
            400, '{"error":{"message":"Failed to initialize samplers: failed to parse grammar"}}'
        )
        assert "compile a grammar" in exc.detail
        # An unrelated upstream error still passes through verbatim.
        assert "llama-server error:" in _openai_passthrough_error(500, "disk full").detail

    def test_anthropic_upstream_error_rewords_context_overflow(self):
        from routes.inference import _anthropic_upstream_error

        raw = (
            '{"error":{"code":500,"message":"the request exceeds the available context '
            'size. try increasing the context size or enable context shift",'
            '"type":"server_error"}}'
        )
        msg = _anthropic_upstream_error(raw)
        assert "prompt is too long" in msg.lower()
        assert "llama-server error:" not in msg

    def test_anthropic_upstream_error_includes_token_counts_when_known(self):
        from routes.inference import _anthropic_upstream_error

        msg = _anthropic_upstream_error(
            "the request (214331 tokens) exceeds the available context size (131072 tokens)"
        )
        m = re.search(r"prompt is too long[^0-9]*(\d+)\s*tokens?\s*>\s*(\d+)", msg, re.I)
        assert m and m.groups() == ("214331", "131072")

    def test_anthropic_upstream_error_leaves_other_failures_alone(self):
        from routes.inference import _anthropic_upstream_error
        assert _anthropic_upstream_error("disk full") == "llama-server error: disk full"
        assert "compile a grammar" in _anthropic_upstream_error("failed to parse grammar")

    def test_anthropic_upstream_error_keeps_kv_starvation_out_of_the_overflow_rewrite(self):
        """llama-server says "Context size has been exceeded" when concurrent
        generations drain the shared KV cache. Nothing about the request was too
        long, so rewording it as an oversized prompt sends the client compacting a
        valid conversation instead of retrying."""
        from routes.inference import _anthropic_upstream_error, _classify_llama_generation_error
        from core.inference.stream_errors import KV_STARVATION_MESSAGE

        for raw in (
            "Context size has been exceeded.",
            '{"error":{"message":"Context size has been exceeded.","code":500}}',
        ):
            assert _anthropic_upstream_error(raw) == KV_STARVATION_MESSAGE
            assert "too long" not in _anthropic_upstream_error(raw).lower()
            # None keeps the upstream status: a 400 would blame the caller's request.
            assert _classify_llama_generation_error(Exception(raw)) is None

    def test_in_band_sse_recovers_counts_from_the_chunk_the_message_came_from(self):
        """`_monitor_openai_error_message` returns only the message string. When that
        string carries no numbers but the chunk around it does, the totals -- and the
        window `oversize_advice` needs -- have to come from the payload."""
        import json as _json

        from core.inference import context_refusal
        from routes.inference import (
            _anthropic_upstream_stream_error_event,
            _json_dumps_safe,
            _monitor_openai_error_message,
        )

        chunk = {
            "error": {
                "code": 400,
                "message": "the request exceeds the available context size",
                "type": "exceed_context_size_error",
                "n_prompt_tokens": 70494,
                "n_ctx": 67584,
            }
        }
        message = _monitor_openai_error_message(chunk)
        source = _json_dumps_safe(chunk.get("error", chunk))

        def body(text, counts_source):
            event = _anthropic_upstream_stream_error_event(text, counts_source = counts_source)
            return _json.loads(event.split("data: ", 1)[1].strip().splitlines()[0])["error"]

        assert "70494" not in body(message, None)["message"]
        assert (
            "Prompt is too long: 70494 tokens > 67584 maximum" in body(message, source)["message"]
        )

        try:
            context_refusal.record_fit(
                {
                    "fits": False,
                    "context_length": 67584,
                    "irreducible_tokens": 70000,
                    "latest_turn_tokens": 500,
                }
            )
            # The recovered window is what lets the fit be consulted at all.
            assert "shortening the conversation will not help" in body(message, source)["message"]
        finally:
            context_refusal.clear()

        # An unrelated error must not have the payload spliced into its text.
        assert body("disk full", source)["message"] == "llama-server error: disk full"

    def test_counts_come_from_the_structured_fields_when_the_message_has_none(self):
        """An exceed_context_size_error body carries n_prompt_tokens/n_ctx even when its
        message spells out no numbers. These paths are handed the whole body, so the
        totals are right there; dropping them sends the client a count-less refusal."""
        from routes.inference import _anthropic_upstream_error, _oversize_counts

        body = (
            '{"error":{"code":400,"message":"the request exceeds the available context '
            'size","type":"exceed_context_size_error","n_prompt_tokens":70494,'
            '"n_ctx":67584}}'
        )
        assert _oversize_counts(body) == (70494, 67584)
        assert "Prompt is too long: 70494 tokens > 67584 maximum" in _anthropic_upstream_error(body)

    def test_the_remedy_comes_from_the_fit_not_a_flat_shorten_the_conversation(self):
        """When the latest turn or the irreducible floor is what does not fit, compacting
        the history cannot make it fit, and a client told to compact just retries."""
        from core.inference import context_refusal
        from routes.inference import _anthropic_upstream_error

        body = "the request (214331 tokens) exceeds the available context size (131072 tokens)"
        try:
            context_refusal.record_fit(
                {
                    "fits": False,
                    "context_length": 131072,
                    "irreducible_tokens": 140000,
                    "latest_turn_tokens": 500,
                }
            )
            msg = _anthropic_upstream_error(body)
            # The head Anthropic clients key on survives.
            assert msg.startswith("Prompt is too long: 214331 tokens > 131072 maximum.")
            assert "shortening the conversation will not help" in msg

            context_refusal.record_fit(
                {
                    "fits": False,
                    "context_length": 131072,
                    "irreducible_tokens": 140000,
                    "latest_turn_tokens": 139000,
                    "latest_turn_role": "tool",
                    "latest_turn_counted": True,
                }
            )
            msg = _anthropic_upstream_error(body)
            assert "A tool returned more than this context window can hold" in msg
            assert "shortening the conversation will not help" in msg
        finally:
            context_refusal.clear()

        # With no recorded fit it stays the generic wording.
        assert "Try increasing the Context Length" in _anthropic_upstream_error(body)

    def test_in_band_sse_error_gets_the_same_wording_as_the_non_200_branch(self):
        """A 200 stream that later emits data: {"error": ...} used to be wrapped in a
        plain RuntimeError, and _friendly_error flattens the count-less oversize
        refusal to "An internal error occurred" -- a 400/invalid_request_error with
        nothing the client can read or compact on."""
        import json as _json

        from routes.inference import _anthropic_upstream_stream_error_event

        def body(text):
            event = _anthropic_upstream_stream_error_event(text)
            return _json.loads(event.split("data: ", 1)[1].strip().splitlines()[0])["error"]

        no_counts = body(
            "the request exceeds the available context size. try increasing the context size"
        )
        assert "internal error" not in no_counts["message"].lower()
        assert no_counts["message"].startswith("Prompt is too long")
        assert no_counts["type"] == "invalid_request_error"

        counted = body(
            "the request (214331 tokens) exceeds the available context size (131072 tokens)"
        )
        assert "Prompt is too long: 214331 tokens > 131072 maximum" in counted["message"]
        assert counted["type"] == "invalid_request_error"

        # Starvation keeps the retry advice and must not become a 400.
        starved = body("Context size has been exceeded.")
        assert "shared pool of context" in starved["message"]
        assert "too long" not in starved["message"].lower()
        assert starved["type"] == "api_error"

        unrelated = body("disk full")
        assert unrelated["message"] == "llama-server error: disk full"
        assert unrelated["type"] == "api_error"

    def test_a_genuine_oversize_body_is_still_classified_as_an_overflow(self):
        from routes.inference import _classify_llama_generation_error
        assert (
            _classify_llama_generation_error(
                Exception(
                    "the request (214331 tokens) exceeds the available context size (131072 tokens)"
                )
            )
            is True
        )


# =====================================================================

# ChatMessage - tool role, tool_calls, optional content
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

    def test_tool_empty_content_accepted(self):
        # Empty tool output (mkdir, git add, ...) is routine in agentic loops;
        # OpenAI and llama-server both accept it, so Unsloth must not 400.
        msg = ChatMessage(role = "tool", tool_call_id = "call_1", content = "")
        assert msg.content == ""

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

# ChatCompletionRequest - standard OpenAI tool fields
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
        """Declared fields bind and are typed; unknown ones still survive
        extra="allow". Declaring response_format is what turns away a value of
        the wrong shape -- a bare string names no schema to constrain to. What
        is inside the object is the serving path's business, not the schema's."""
        from pydantic import ValidationError

        req = self._make(
            seed = 42,
            frequency_penalty = 0.5,
            response_format = {"type": "json_object"},
            unknown = "kept",
        )
        assert (req.seed, req.frequency_penalty) == (42, 0.5)
        assert req.response_format == {"type": "json_object"}
        assert not {"seed", "frequency_penalty", "response_format"} & set(req.model_extra or {})
        assert (req.model_extra or {}).get("unknown") == "kept"
        with pytest.raises(ValidationError):
            self._make(response_format = "json_object")

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
        # OpenAI defaults `stream` to false. Unsloth used to default true,
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

        async def _fake_proxy(payload, request, current_subject):
            assert current_subject == "test-user"
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

    def test_confirm_tool_calls_rejected_for_provider_tools(self, monkeypatch):
        class _UnusedBackend:
            is_loaded = False

        client = self._v1_client(monkeypatch, _UnusedBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "provider_type": "openai",
                "external_model": "gpt-4.1",
                "enable_tools": True,
                "enabled_tools": ["web_search"],
                "confirm_tool_calls": True,
            },
        )

        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["param"] == "confirm_tool_calls"
        assert "only supported for local streaming tools" in body["error"]["message"]

    def test_confirm_tool_calls_allowed_for_codex_studio_tools(self, monkeypatch):
        from routes import inference as inference_route

        class _UnusedBackend:
            is_loaded = False

        client = self._v1_client(monkeypatch, _UnusedBackend())

        async def fake_proxy(*_args, **_kwargs):
            return {"ok": True}

        monkeypatch.setattr(inference_route, "_proxy_to_external_provider", fake_proxy)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "search docs"}],
                "provider_type": "openai_codex",
                "external_model": "gpt-5.6-sol",
                "enable_tools": True,
                "enabled_tools": ["web_search"],
                "confirm_tool_calls": True,
                "permission_mode": "ask",
                "stream": True,
            },
        )

        assert resp.status_code == 200
        assert resp.json() == {"ok": True}

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
            context_length = 4096

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
        import routes.inference as inference_route

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = True
            is_vision = False
            _is_audio = False
            context_length = 4096

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
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
        [entry] = monitor.snapshot()
        assert entry["status"] == "error"
        assert "n > 1 is not supported" in entry["error"]
        assert monitor.active_count() == 0

    def test_a_no_op_response_format_keeps_a_gguf_request_off_the_passthrough(self):
        """The passthrough exists to reach llama-server's grammar engine, and
        ``{"type": "text"}`` names the default rather than a grammar: routing it
        there would withdraw what the ordinary path serves, n > 1 among them."""
        from routes.inference import _takes_tool_passthrough

        class _GGUFBackend:
            supports_tools = False

        backend = _GGUFBackend()
        assert _takes_tool_passthrough(self._make(response_format = {"type": "json_object"}), backend)
        # The field takes any object, and anything but the exact default -- an
        # unknown type, or a text format carrying members this build does not know
        # -- is a contract, so it keeps going where one can be answered or rejected.
        assert _takes_tool_passthrough(self._make(response_format = {"type": "later"}), backend)
        assert _takes_tool_passthrough(
            self._make(response_format = {"type": "text", "strict": True}), backend
        )
        assert not _takes_tool_passthrough(self._make(response_format = {"type": "text"}), backend)

    @pytest.mark.parametrize("enabled_tools", [None, []])
    def test_the_gguf_tool_loop_refuses_a_contract_it_cannot_forward(
        self, monkeypatch, enabled_tools
    ):
        """Unsloth's loop runs its own turns and never forwards response_format, so
        serving the request would answer with text that violates the contract while
        the client has no way to tell. A selection that resolves to no tool routes to
        the ordinary generator, which forwards it no more than the loop does, and the
        refusal lands after the monitor row opens, so it must close it."""
        import routes.inference as inference_route

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = True
            is_vision = False
            _is_audio = False
            context_length = 4096

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        client = self._v1_client(monkeypatch, _GGUFBackend())
        body = {
            "messages": [{"role": "user", "content": "hi"}],
            "enable_tools": True,
            "response_format": {"type": "json_object"},
        }
        if enabled_tools is not None:
            body["enabled_tools"] = enabled_tools
        resp = client.post("/v1/chat/completions", json = body)
        self._assert_unsupported_param(resp, "response_format")
        assert monitor.active_count() == 0

    def test_client_tools_rejected_when_gguf_template_has_no_tool_support(self, monkeypatch):
        import routes.inference as inference_route

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = False
            is_vision = False
            _is_audio = False
            context_length = 4096

            def generate_chat_completion(self, **_kwargs):
                raise AssertionError("client tools must not fall through to the standard GGUF path")

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
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
            },
        )

        self._assert_unsupported_param(resp, "tools")
        assert "does not advertise tools" in resp.json()["error"]["message"]
        [entry] = monitor.snapshot()
        assert entry["status"] == "error"
        assert "does not advertise tools" in entry["error"]
        assert monitor.active_count() == 0

    def test_a_multi_image_message_closes_the_monitor_row(self, monkeypatch):
        """The single-image refusal lands after the monitor row opens, so it must
        close it: a row left running keeps Studio reporting the backend as
        generating long after the request has been answered with a 400."""
        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        class _VisionBackend:
            active_model_name = "vision-sf"
            models = {
                "vision-sf": {
                    "is_vision": True,
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }

            def resize_image(self, image):
                return image

            def generate_chat_response(self, **kwargs):
                yield "ok"

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _VisionBackend())
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "compare these"}, image, image],
                    }
                ]
            },
        )

        assert resp.status_code == 400
        assert "one image per message" in resp.text
        [entry] = monitor.snapshot()
        assert entry["status"] == "error"
        assert "one image per message" in entry["error"]
        assert monitor.active_count() == 0

    def _standard_vision_client(
        self,
        monkeypatch,
        monitor,
        is_vision = True,
    ):
        """A loaded safetensors backend on the standard path, recording generation."""
        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        calls = []

        class _Backend:
            active_model_name = "vision-sf"
            models = {
                "vision-sf": {
                    "is_vision": is_vision,
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }
            generated = calls

            def resize_image(self, image):
                return image

            def generate_chat_response(self, **kwargs):
                calls.append(kwargs)
                yield "ok"

            def reset_generation_state(self, cancel_event = None):
                pass

        backend = _Backend()
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        return self._v1_client(monkeypatch, _LlamaOff(), backend), backend

    def test_several_undecodable_images_on_one_message_are_still_refused(self, monkeypatch):
        """The count is of image PARTS, not of images extraction could decode.

        Only a data URL becomes base64, so several remote image URLs decode to
        nothing; gating on a decoded image would let exactly those be flattened
        away unannounced, which is the silent drop this guard exists to stop."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        remote = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
        empty = {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}

        for parts in ([remote, remote], [empty, empty], [empty, remote]):
            resp = client.post(
                "/v1/chat/completions",
                json = {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "compare these"}, *parts],
                        }
                    ]
                },
            )
            assert resp.status_code == 400, parts
            assert "one image per message" in resp.text

        assert backend.generated == []
        assert monitor.active_count() == 0
        assert all(e["status"] == "error" for e in monitor.snapshot())

    @pytest.mark.parametrize(
        "legacy,expected",
        [(_BLUE_PNG_B64, 400), (_RED_PNG_B64, 200)],
        ids = ["distinct_legacy_image_is_a_second_image", "echoed_legacy_image_is_the_same_one"],
    )
    def test_a_distinct_legacy_image_beside_a_message_image(self, legacy, expected, monkeypatch):
        """`extracted_image_b64 or image_base64` always picks the message image, so a
        DIFFERENT top-level image is a second one the caller never hears about.
        Studio's echo byte-matches the part, so it must not be counted twice."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{_RED_PNG_B64}"},
                            },
                        ],
                    }
                ],
                "image_base64": legacy,
            },
        )

        assert resp.status_code == expected
        assert monitor.active_count() == 0
        if expected == 400:
            assert "one image per message" in resp.text
            assert backend.generated == []
        else:
            assert len(backend.generated) == 1

    @pytest.mark.parametrize(
        "bad_part",
        [
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        ],
        ids = ["remote_url", "payloadless_data_url"],
    )
    def test_an_unreadable_part_plus_a_distinct_legacy_image_is_two_images(
        self, bad_part, monkeypatch
    ):
        """The count is structural, so a part the extractor cannot read still counts.
        With a distinct top-level image beside it the caller supplied two, and the
        part is the one that gets dropped."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "x"}, bad_part]}
                ],
                "image_base64": _RED_PNG_B64,
            },
        )

        assert resp.status_code == 400
        assert "one image per message" in resp.text
        assert backend.generated == []
        assert monitor.active_count() == 0

    def test_an_unreadable_part_alone_is_still_answered(self, monkeypatch):
        """Control: one remote image and no top-level image is not a multi-image
        call, and clients relying on that text answer must keep it."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "x"},
                            {"type": "image_url", "image_url": {"url": "https://e.com/a.png"}},
                        ],
                    }
                ]
            },
        )

        assert resp.status_code == 200
        assert len(backend.generated) == 1
        assert backend.generated[0]["image"] is None
        assert monitor.active_count() == 0

    @pytest.mark.parametrize("n_images,expected", [(2, 400), (1, 400), (0, 200)])
    def test_a_tts_model_refuses_an_image_instead_of_speaking_over_it(
        self, n_images, expected, monkeypatch
    ):
        """The TTS auto-route returns before the standard image path and speaks the
        newest user text, so an attached image is discarded. It is also why the
        text-only rejection never sees such a request."""
        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        spoken = []

        class _TtsBackend:
            active_model_name = "tts"
            models = {
                "tts": {
                    "is_vision": False,
                    "is_audio": True,
                    "audio_type": "csm",
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }

            def resize_image(self, image):
                return image

            def generate_audio_response(self, **kwargs):
                spoken.append(kwargs)
                import numpy as np
                return np.zeros(16000, dtype = "float32"), 16000

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _TtsBackend())
        image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_RED_PNG_B64}"}}
        content = [{"type": "text", "text": "say hi"}] + [image] * n_images

        resp = client.post(
            "/v1/chat/completions",
            json = {"messages": [{"role": "user", "content": content}]},
        )

        assert resp.status_code == expected
        assert monitor.active_count() == 0
        if expected == 400:
            assert "does not read images" in resp.text
            assert spoken == []
        else:
            assert len(spoken) == 1

    @pytest.mark.parametrize(
        "legacy_is_newest,expected",
        [(False, 400), (True, 200)],
        ids = ["older_user_image_is_not_the_echo", "newest_user_image_is_the_echo"],
    )
    def test_only_the_newest_user_image_counts_as_the_echo(
        self, legacy_is_newest, expected, monkeypatch
    ):
        """findLatestUserImageBase64 returns the first image on the newest user turn,
        so that one value is all the field could carry. Matching anywhere let a
        client resend an older image while the newest turn supplied its own."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        older = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_RED_PNG_B64}"}}
        newest = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_BLUE_PNG_B64}"},
        }

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "first"}, older]},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": [{"type": "text", "text": "second"}, newest]},
                ],
                "image_base64": _BLUE_PNG_B64 if legacy_is_newest else _RED_PNG_B64,
            },
        )

        assert resp.status_code == expected
        assert monitor.active_count() == 0
        if expected == 400:
            assert "one image per message" in resp.text
            assert backend.generated == []
        else:
            assert len(backend.generated) == 1

    def test_a_comma_in_a_remote_url_is_not_an_echo(self, monkeypatch):
        """Only a data: URL carries an inline payload, the way _extract_content_parts
        reads it. Splitting every URL on its first comma let a remote URL that merely
        contains one pose as an echo of the top-level image, hiding a second image."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "x"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"https://h.example/img,{_RED_PNG_B64}"},
                            },
                        ],
                    }
                ],
                "image_base64": _RED_PNG_B64,
            },
        )

        assert resp.status_code == 400
        assert "one image per message" in resp.text
        assert backend.generated == []
        assert monitor.active_count() == 0

    def test_a_payloadless_newest_image_does_not_end_the_echo_scan(self, monkeypatch):
        """findLatestUserImageBase64 walks on past a falsy read, so an empty newest
        part is not the value the field carries. Stopping on it refused Studio's
        echo of a valid earlier image as a second attachment."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        older = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_RED_PNG_B64}"}}
        empty = {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "first"}, older]},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": [{"type": "text", "text": "second"}, empty]},
                ],
                "image_base64": _RED_PNG_B64,
            },
        )

        assert resp.status_code == 200
        assert len(backend.generated) == 1
        assert monitor.active_count() == 0

    @pytest.mark.parametrize(
        "n_images,expected",
        [(1, 400), (0, 200)],
        ids = ["image_on_a_gguf_speech_model_is_refused", "no_image_still_speaks"],
    )
    def test_a_gguf_speech_model_refuses_an_image(self, n_images, expected, monkeypatch):
        """The GGUF audio branch returns before every image check, exactly as the
        safetensors audio branch does, and speaks the newest user TEXT. An attached
        image went unread and unmentioned."""
        import routes.inference as inference_route

        spoken = []

        class _LlamaTts:
            is_loaded = True
            model_identifier = "tts.gguf"
            supports_tools = False
            is_vision = False
            _is_audio = True
            _audio_type = "snac"
            context_length = 4096

            def generate_audio_response(self, **kwargs):
                spoken.append(kwargs)
                import numpy as np
                return np.zeros(16000, dtype = "float32"), 16000

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        client = self._v1_client(monkeypatch, _LlamaTts(), None)
        image = {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_RED_PNG_B64}"}}

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "say hi"}] + [image] * n_images,
                    }
                ]
            },
        )

        assert resp.status_code == expected
        assert monitor.active_count() == 0
        if expected == 400:
            assert "does not read images" in resp.text
            assert spoken == []
        else:
            assert len(spoken) == 1

    def test_a_legacy_image_alone_is_still_served(self, monkeypatch):
        """With nothing extracted from the messages the top-level image IS the one
        that gets used, so it must not be refused as a second image."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "what is this?"}],
                "image_base64": _RED_PNG_B64,
            },
        )

        assert resp.status_code == 200
        assert len(backend.generated) == 1
        assert backend.generated[0]["image"] is not None
        assert monitor.active_count() == 0

    def test_a_single_undecodable_image_is_left_alone(self, monkeypatch):
        """One remote image is not a multi-image call. Clients that pass a remote
        URL today get a text answer, and this guard must not turn that into a 400."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/a.png"},
                            },
                        ],
                    }
                ]
            },
        )

        assert resp.status_code == 200
        assert len(backend.generated) == 1
        assert backend.generated[0]["image"] is None
        assert monitor.active_count() == 0

    def test_a_text_only_model_closes_the_monitor_row(self, monkeypatch):
        """Same defect as the multi-image refusal, one branch up: this rejection
        also lands after the monitor row opens and must finalize it."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor, is_vision = False)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                        ],
                    }
                ]
            },
        )

        assert resp.status_code == 400
        assert "text-only" in resp.text
        assert backend.generated == []
        [entry] = monitor.snapshot()
        assert entry["status"] == "error"
        assert "text-only" in entry["error"]
        assert monitor.active_count() == 0

    @pytest.mark.parametrize("n_images,expected", [(2, 400), (1, 400), (0, 200)])
    def test_an_image_with_audio_input_is_refused_not_dropped(
        self, n_images, expected, monkeypatch
    ):
        """The audio-input branch returns before the standard image path and forwards
        only the flattened messages plus the audio, so an attached image is
        discarded. Any count: one is dropped here as silently as two."""
        import numpy as np

        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        calls = []

        class _OmniBackend:
            active_model_name = "omni-sf"
            models = {
                "omni-sf": {
                    "is_vision": True,
                    "has_audio_input": True,
                    "audio_type": "audio_vlm",
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }
            generated = calls

            def resize_image(self, image):
                return image

            def generate_audio_input_response(self, **kwargs):
                calls.append(kwargs)
                yield "answered from the audio"

            def generate_chat_response(self, **kwargs):
                raise AssertionError("audio input should take the audio branch")

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        monkeypatch.setattr(
            inference_route,
            "_decode_audio_base64",
            lambda *a, **k: np.zeros(16000, dtype = "float32"),
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _OmniBackend())
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        content = [{"type": "text", "text": "what is this?"}] + [image] * n_images

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": content}],
                "audio_base64": base64.b64encode(b"RIFF....WAVEfmt ").decode(),
            },
        )

        assert resp.status_code == expected
        assert monitor.active_count() == 0
        if expected == 400:
            assert "audio or an image in one message" in resp.text
            assert calls == []
            [entry] = monitor.snapshot()
            assert entry["status"] == "error"
            assert "audio or an image in one message" in entry["error"]
        else:
            assert len(calls) == 1

    def test_a_legacy_image_field_with_audio_is_refused(self, monkeypatch):
        """A client attaching its image through the top-level image_base64
        extension, with no image parts anywhere, means that image for THIS
        request. The audio branch cannot forward it, so refuse rather than drop."""
        import numpy as np

        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        calls = []

        class _OmniBackend:
            active_model_name = "omni-sf"
            models = {
                "omni-sf": {
                    "is_vision": True,
                    "has_audio_input": True,
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }

            def resize_image(self, image):
                return image

            def generate_audio_input_response(self, **kwargs):
                calls.append(kwargs)
                yield "answered from the audio"

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        monkeypatch.setattr(
            inference_route,
            "_decode_audio_base64",
            lambda *a, **k: np.zeros(16000, dtype = "float32"),
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _OmniBackend())

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "what is this?"}],
                "image_base64": "AAAA",
                "audio_base64": base64.b64encode(b"RIFF....WAVEfmt ").decode(),
            },
        )

        assert resp.status_code == 400
        assert "audio or an image in one message" in resp.text
        assert calls == []
        assert monitor.active_count() == 0

    def test_a_derived_legacy_image_field_does_not_block_a_voice_follow_up(self, monkeypatch):
        """Studio fills image_base64 from anywhere in the thread, so it is not a
        per-turn signal. When an earlier turn carries the image the parts decide,
        and the voice follow-up must still be answered."""
        import numpy as np

        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        calls = []

        class _OmniBackend:
            active_model_name = "omni-sf"
            models = {
                "omni-sf": {
                    "is_vision": True,
                    "has_audio_input": True,
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }

            def resize_image(self, image):
                return image

            def generate_audio_input_response(self, **kwargs):
                calls.append(kwargs)
                yield "answered from the audio"

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        monkeypatch.setattr(
            inference_route,
            "_decode_audio_base64",
            lambda *a, **k: np.zeros(16000, dtype = "float32"),
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _OmniBackend())
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "what is this?"}, image],
                    },
                    {"role": "assistant", "content": "a square"},
                    {"role": "user", "content": "and now by voice"},
                ],
                "image_base64": "AAAA",
                "audio_base64": base64.b64encode(b"RIFF....WAVEfmt ").decode(),
            },
        )

        assert resp.status_code == 200
        assert len(calls) == 1
        assert monitor.active_count() == 0

    def test_a_new_legacy_image_is_refused_even_with_image_bearing_history(self, monkeypatch):
        """A direct client can attach a NEW image through the legacy field while the
        newest turn stays text-only. Studio's own field is copied out of the thread
        and byte-matches a part; one matching nothing came with this request."""
        import numpy as np

        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        calls = []

        class _OmniBackend:
            active_model_name = "omni-sf"
            models = {
                "omni-sf": {
                    "is_vision": True,
                    "has_audio_input": True,
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }

            def resize_image(self, image):
                return image

            def generate_audio_input_response(self, **kwargs):
                calls.append(kwargs)
                yield "answered from the audio"

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        monkeypatch.setattr(
            inference_route,
            "_decode_audio_base64",
            lambda *a, **k: np.zeros(16000, dtype = "float32"),
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _OmniBackend())
        old = {"type": "image_url", "image_url": {"url": "data:image/png;base64,T0xE"}}

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "this?"}, old]},
                    {"role": "assistant", "content": "a square"},
                    {"role": "user", "content": "and this one?"},
                ],
                "image_base64": "TkVX",
                "audio_base64": base64.b64encode(b"RIFF....WAVEfmt ").decode(),
            },
        )

        assert resp.status_code == 400
        assert "audio or an image in one message" in resp.text
        assert calls == []
        assert monitor.active_count() == 0

    def test_an_assistant_image_cannot_hide_a_top_level_attachment(self, monkeypatch):
        """findLatestUserImageBase64 derives the legacy field only from USER turns,
        so an image on an assistant turn is not something it could be an echo of.
        Matching any role let an explicitly attached image be discarded here."""
        import numpy as np

        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        calls = []

        class _OmniBackend:
            active_model_name = "omni-sf"
            models = {
                "omni-sf": {
                    "is_vision": True,
                    "has_audio_input": True,
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }

            def resize_image(self, image):
                return image

            def generate_audio_input_response(self, **kwargs):
                calls.append(kwargs)
                yield "answered from the audio"

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        monkeypatch.setattr(
            inference_route,
            "_decode_audio_base64",
            lambda *a, **k: np.zeros(16000, dtype = "float32"),
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _OmniBackend())
        generated = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{_BLUE_PNG_B64}"},
        }

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {"role": "user", "content": "draw one"},
                    {"role": "assistant", "content": [generated]},
                    {"role": "user", "content": "describe it"},
                ],
                "image_base64": _BLUE_PNG_B64,
                "audio_base64": base64.b64encode(b"RIFF....WAVEfmt ").decode(),
            },
        )

        assert resp.status_code == 400
        assert "audio or an image in one message" in resp.text
        assert calls == []
        assert monitor.active_count() == 0

    def test_an_image_on_an_earlier_turn_still_allows_a_voice_follow_up(self, monkeypatch):
        """The count is scoped to the newest user turn, so asking by voice about a
        picture attached on an earlier turn keeps working."""
        import numpy as np

        import routes.inference as inference_route

        class _LlamaOff:
            is_loaded = False
            supports_tools = False
            is_vision = False
            context_length = None

        calls = []

        class _OmniBackend:
            active_model_name = "omni-sf"
            models = {
                "omni-sf": {
                    "is_vision": True,
                    "has_audio_input": True,
                    "chat_template_info": {"template": "chatml"},
                    "context_length": 4096,
                }
            }

            def resize_image(self, image):
                return image

            def generate_audio_input_response(self, **kwargs):
                calls.append(kwargs)
                yield "answered from the audio"

            def reset_generation_state(self, cancel_event = None):
                pass

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": False},
        )
        monkeypatch.setattr(
            inference_route,
            "_decode_audio_base64",
            lambda *a, **k: np.zeros(16000, dtype = "float32"),
        )
        client = self._v1_client(monkeypatch, _LlamaOff(), _OmniBackend())
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}

        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "what is this?"}, image],
                    },
                    {"role": "assistant", "content": "a square"},
                    {"role": "user", "content": "and now by voice"},
                ],
                "audio_base64": base64.b64encode(b"RIFF....WAVEfmt ").decode(),
            },
        )

        assert resp.status_code == 200
        assert len(calls) == 1
        assert monitor.active_count() == 0

    def test_a_failed_image_decode_closes_the_monitor_row(self, monkeypatch):
        """A single image whose payload is not an image fails inside the decode,
        after the row opens. That rejection must finalize the row too."""
        monitor = ApiMonitor(max_entries = 3)
        client, backend = self._standard_vision_client(monkeypatch, monitor)
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,AAAA"},
                            },
                        ],
                    }
                ]
            },
        )

        assert resp.status_code == 400
        assert "Failed to decode image" in resp.text
        assert backend.generated == []
        [entry] = monitor.snapshot()
        assert entry["status"] == "error"
        assert "Failed to decode image" in entry["error"]
        assert monitor.active_count() == 0

    def test_client_tools_use_passthrough_capability_when_tool_loop_is_disabled(self, monkeypatch):
        import routes.inference as inference_route

        captured = {}

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = False
            supports_tool_passthrough = True
            is_vision = False
            _is_audio = False
            context_length = 4096
            base_url = "http://llama.passthrough-capability.test"
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None

            def generate_chat_completion(self, **_kwargs):
                raise AssertionError("client tools must use passthrough")

            def generate_chat_completion_with_tools(self, **_kwargs):
                raise AssertionError("Unsloth tool loop must stay disabled")

        async def fake_passthrough(llama_backend, payload, model_name, **kwargs):
            captured["body"] = inference_route._build_openai_passthrough_body(
                payload,
                backend_ctx = llama_backend.context_length,
                llama_backend = llama_backend,
            )
            inference_route.api_monitor.finish(kwargs.get("monitor_id"))
            return inference_route.JSONResponse({"ok": True, "model": model_name})

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_openai_passthrough_non_streaming",
            fake_passthrough,
        )
        client = self._v1_client(monkeypatch, _GGUFBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "use client tool"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert captured["body"]["tools"][0]["function"]["name"] == "lookup"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert monitor.active_count() == 0

    def test_permission_mode_does_not_reject_client_tool_passthrough(self, monkeypatch):
        # A non-streaming client-tool passthrough (client tools, no Unsloth tool
        # loop) that also carries permission_mode "ask"/"auto" must reach the
        # provider passthrough, not the confirm-without-stream guard: the
        # validator leaves confirm_tool_calls unset for passthrough, and a bare
        # permission_mode only gates Unsloth's own local tool loop. An explicit
        # confirm_tool_calls=True still forces the local-confirm rejection.
        # The pre-switch guard only runs when an automatic load may run, so force
        # that predicate on to exercise it against a resident passthrough backend.
        import routes.inference as inference_route

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = False
            supports_tool_passthrough = True
            is_vision = False
            _is_audio = False
            context_length = 4096
            base_url = "http://llama.permission-passthrough.test"
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None

            def generate_chat_completion(self, **_kwargs):
                raise AssertionError("client tools must use passthrough")

            def generate_chat_completion_with_tools(self, **_kwargs):
                raise AssertionError("Unsloth tool loop must stay disabled")

        async def fake_passthrough(llama_backend, payload, model_name, **kwargs):
            inference_route.api_monitor.finish(kwargs.get("monitor_id"))
            return inference_route.JSONResponse({"ok": True, "model": model_name})

        client_tools = [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ]

        def _setup(policy = None):
            reset_tool_policy()
            if policy is not None:
                set_tool_policy(policy)
            monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: True)
            monkeypatch.setattr(inference_route, "api_monitor", ApiMonitor(max_entries = 3))
            monkeypatch.setattr(
                inference_route, "_openai_passthrough_non_streaming", fake_passthrough
            )
            return self._v1_client(monkeypatch, _GGUFBackend())

        # A process --enable-tools policy must not turn a client-tool passthrough
        # into an Unsloth local loop, so a policy of None or True both keep the
        # passthrough (the guard mirrors _explicit_studio_tool_loop_requested).
        for policy in (None, True):
            for mode in ("ask", "auto"):
                client = _setup(policy)
                resp = client.post(
                    "/v1/chat/completions",
                    json = {
                        "messages": [{"role": "user", "content": "use client tool"}],
                        "tools": client_tools,
                        "permission_mode": mode,
                        "stream": False,
                    },
                )
                assert resp.status_code == 200, resp.text
                assert resp.json()["ok"] is True

        # A JSON-schema response_format is guided-decoding passthrough, not a local
        # tool loop, so a --enable-tools policy must not 400 a non-streaming ask/auto
        # structured-output request under the confirm guard.
        for mode in ("ask", "auto"):
            client = _setup(True)
            resp = client.post(
                "/v1/chat/completions",
                json = {
                    "messages": [{"role": "user", "content": "give me json"}],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "s", "schema": {"type": "object"}},
                    },
                    "permission_mode": mode,
                    "stream": False,
                },
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["ok"] is True

        # An explicit confirm_tool_calls=True with client tools and no stream is
        # still a confirm-without-stream request and must be rejected up front.
        client = _setup()
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "use client tool"}],
                "tools": client_tools,
                "confirm_tool_calls": True,
                "stream": False,
            },
        )
        assert resp.status_code == 400
        assert "requires stream=true" in resp.json()["error"]["message"]

    def test_permission_mode_policy_forced_local_loop_rejected_before_switch(self, monkeypatch):
        # A process --enable-tools policy forces Unsloth's own tool loop on even
        # when the request omits enable_tools and carries no client tools. A
        # non-streaming ask/auto request is then confirm-gated with no stream to
        # prompt on, so it must 400 at the pre-switch guard -- before
        # _maybe_auto_switch_model runs -- rather than evicting the resident model
        # and 400ing only at the per-backend check.
        import routes.inference as inference_route

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = True
            supports_tool_passthrough = True
            is_vision = False
            _is_audio = False
            context_length = 4096
            base_url = "http://llama.policy-forced.test"
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None

        switch_calls = []

        async def _no_switch(*_args, **_kwargs):
            switch_calls.append(1)

        def _setup():
            reset_tool_policy()
            set_tool_policy(True)
            monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: True)
            monkeypatch.setattr(inference_route, "api_monitor", ApiMonitor(max_entries = 3))
            monkeypatch.setattr(inference_route, "_maybe_auto_switch_model", _no_switch)
            return self._v1_client(monkeypatch, _GGUFBackend())

        try:
            for mode in ("ask", "auto"):
                switch_calls.clear()
                client = _setup()
                resp = client.post(
                    "/v1/chat/completions",
                    json = {
                        "messages": [{"role": "user", "content": "hi"}],
                        "permission_mode": mode,
                        "stream": False,
                    },
                )
                assert resp.status_code == 400, resp.text
                assert "requires stream=true" in resp.json()["error"]["message"]
                assert switch_calls == [], "guard must reject before the auto-switch"
        finally:
            reset_tool_policy()

    def test_enable_tools_on_non_tool_backend_keeps_client_tools_on_passthrough(self, monkeypatch):
        # DiffusionGemma forces supports_tools off while passthrough stays
        # available (#6851): enable_tools=True must not steal client tools
        # from the passthrough into an Unsloth tool loop that cannot run.
        import routes.inference as inference_route

        captured = {}

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = False
            supports_tool_passthrough = True
            is_vision = False
            _is_audio = False
            context_length = 4096
            base_url = "http://llama.passthrough-capability.test"
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None

            def generate_chat_completion(self, **_kwargs):
                raise AssertionError("client tools must use passthrough")

            def generate_chat_completion_with_tools(self, **_kwargs):
                raise AssertionError("Unsloth tool loop cannot run on a non-tool backend")

        async def fake_passthrough(llama_backend, payload, model_name, **kwargs):
            captured["body"] = inference_route._build_openai_passthrough_body(
                payload,
                backend_ctx = llama_backend.context_length,
                llama_backend = llama_backend,
            )
            inference_route.api_monitor.finish(kwargs.get("monitor_id"))
            return inference_route.JSONResponse({"ok": True, "model": model_name})

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        monkeypatch.setattr(
            inference_route,
            "_openai_passthrough_non_streaming",
            fake_passthrough,
        )
        client = self._v1_client(monkeypatch, _GGUFBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "use client tool"}],
                "enable_tools": True,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            },
        )

        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert captured["body"]["tools"][0]["function"]["name"] == "lookup"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert monitor.active_count() == 0

    def test_tool_choice_none_allows_tool_catalog_without_tool_template(self, monkeypatch):
        import routes.inference as inference_route

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = False
            is_vision = False
            _is_audio = False
            context_length = 4096

            def generate_chat_completion(self, **kwargs):
                assert kwargs["max_tokens"] is None
                yield "plain response"

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
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
                "tool_choice": "none",
            },
        )

        assert resp.status_code == 200
        assert resp.json()["choices"][0]["message"]["content"] == "plain response"
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply"] == "plain response"
        assert monitor.active_count() == 0

    def test_tool_call_history_rejected_when_gguf_template_has_no_tool_support(self, monkeypatch):
        import routes.inference as inference_route

        class _GGUFBackend:
            is_loaded = True
            model_identifier = "test-gguf"
            supports_tools = False
            is_vision = False
            _is_audio = False
            context_length = 4096

            def generate_chat_completion(self, **_kwargs):
                raise AssertionError(
                    "tool-call history must not fall through to the standard GGUF path"
                )

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        client = self._v1_client(monkeypatch, _GGUFBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [
                    {"role": "user", "content": "use a tool"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
                ],
            },
        )

        self._assert_unsupported_param(resp, "messages")
        assert "does not advertise tools" in resp.json()["error"]["message"]
        [entry] = monitor.snapshot()
        assert entry["status"] == "error"
        assert "does not advertise tools" in entry["error"]
        assert monitor.active_count() == 0

    @pytest.mark.parametrize(
        "param, model_entry, extra_body, unreachable_pre_switch",
        [
            ("n", {"is_audio": True, "audio_type": "tts"}, {"n": 2}, False),
            ("n", {"has_audio_input": True}, {"n": 2, "audio_base64": "AAAA"}, False),
            ("n", {}, {"n": 2, "enable_tools": True}, False),
            ("n", {}, {"n": 2, "stream": True}, True),
            ("response_format", {}, {"response_format": {"type": "json_object"}}, False),
            ("response_format", {}, {"response_format": {"type": "later"}}, False),
        ],
    )
    def test_the_non_gguf_paths_refuse_what_they_cannot_serve(
        self, monkeypatch, param, model_entry, extra_body, unreachable_pre_switch
    ):
        """Plain non-GGUF text now serves extra choices, so only the paths that
        answer something other than a sample still refuse n: one waveform, one
        transcript of the one recording, a loop that runs turns rather than
        samples, and one SSE stream, which carries a single choice whether or not
        the pre-switch check (reached only when a load may) ran. And no non-GGUF
        backend can constrain decoding at all."""
        import routes.inference as inference_route

        class _NoGGUFBackend:
            is_loaded = False
            supports_tools = False

        class _InferenceBackend:
            active_model_name = "test-model"
            models = {"test-model": model_entry}

        if unreachable_pre_switch:
            monkeypatch.setattr(inference_route, "_automatic_model_load_may_run", lambda: False)
        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda *a, **k: {"supports_tools": True},
        )
        client = self._v1_client(monkeypatch, _NoGGUFBackend(), _InferenceBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {"messages": [{"role": "user", "content": "hi"}], **extra_body},
        )
        self._assert_unsupported_param(resp, param)

    # Same two rejection sites, and same reason to pin one, as the GGUF case below.
    @pytest.mark.parametrize("validate_before_switch", [False, True])
    def test_confirm_tool_calls_requires_streaming_for_safetensors_tools(
        self, monkeypatch, validate_before_switch
    ):
        import routes.inference as inference_route

        class _NoGGUFBackend:
            is_loaded = False
            supports_tools = False

        class _InferenceBackend:
            active_model_name = "test-model"
            models = {"test-model": {"chat_template_info": {"template": "chatml"}}}

            def generate_chat_completion_with_tools(self, **kwargs):
                raise AssertionError("tool loop should be rejected before starting")

            def generate_chat_completion(self, **kwargs):
                raise AssertionError("plain path should not be used")

        monkeypatch.setattr(
            inference_route,
            "_detect_safetensors_features",
            lambda backend, chat_template, tools = None: {"supports_tools": True},
        )
        monkeypatch.setattr(
            inference_route, "_should_validate_before_switch", lambda: validate_before_switch
        )
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inference_route, "api_monitor", monitor)
        client = self._v1_client(monkeypatch, _NoGGUFBackend(), _InferenceBackend())
        resp = client.post(
            "/v1/chat/completions",
            json = {
                "messages": [{"role": "user", "content": "hi"}],
                "enable_tools": True,
                "enabled_tools": ["web_search"],
                "confirm_tool_calls": True,
                "stream": False,
            },
        )

        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["param"] == "confirm_tool_calls"
        assert "requires stream=true" in body["error"]["message"]
        if validate_before_switch:
            # Early site rejects before the monitor row is opened.
            assert monitor.snapshot() == []
        else:
            [entry] = monitor.snapshot()
            assert entry["status"] == "error"
            assert "confirm_tool_calls requires stream=true" in entry["error"]
        assert monitor.active_count() == 0

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

# anthropic_tool_choice_to_openai - pure translation helper
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

# _build_passthrough_payload - tool_choice propagation
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

    def test_llama_incompatible_tool_constraints_are_omitted(self):
        args = self._args()
        schema = args["openai_tools"][0]["function"]["parameters"]
        schema["properties"] = {
            "declarationKey": {"type": "string", "pattern": r"\S"},
            "exactKey": {"type": "string", "pattern": r"^[A-Z]+$"},
            "nested": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string", "pattern": "token"},
                        {"type": "string", "pattern": "^fixed$"},
                    ],
                    "default": {"pattern": "annotation data"},
                },
            },
            "largeScript": {"type": "string", "minLength": 1, "maxLength": 65536},
            # Each keyword's highest compilable bound survives; the first one that reaches
            # llama.cpp's rule budget does not.
            "keptScript": {"type": "string", "maxLength": 1999, "minLength": 1999},
            "budgetScript": {"type": "string", "maxLength": 2000},
            "budgetFloor": {"type": "string", "minLength": 2000},
            "keptList": {"type": "array", "items": {"type": "string"}, "maxItems": 1997},
            "budgetList": {"type": "array", "items": {"type": "string"}, "maxItems": 1998},
            "keptFloorList": {"type": "array", "items": {"type": "string"}, "minItems": 2000},
            "budgetFloorList": {"type": "array", "items": {"type": "string"}, "minItems": 2001},
            # An integral bound can decode to float, and llama.cpp reads one either way.
            "floatList": {"type": "array", "items": {"type": "string"}, "minItems": 2001.0},
            # draft-07 tuple form, which llama.cpp visits member by member.
            "tupleList": {
                "type": "array",
                "items": [{"type": "string", "maxLength": 2000}],
            },
            # Unsatisfiable pairs, small enough to pass every limit above: they reach the
            # parser as a descending repetition, which exhausts memory.
            "impossibleList": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 2,
            },
            "impossibleScript": {"type": "string", "minLength": 5, "maxLength": 2},
            "referenced": {"$ref": "#/$defs/Bounded"},
        }
        # llama.cpp resolves the reference, so its target has to be filtered too.
        schema["$defs"] = {"Bounded": {"type": "string", "maxLength": 2000}}

        body = _build_passthrough_payload(**args)
        forwarded = body["tools"][0]["function"]["parameters"]["properties"]

        assert forwarded["declarationKey"] == {"type": "string"}
        assert forwarded["exactKey"]["pattern"] == r"^[A-Z]+$"
        nested = forwarded["nested"]["items"]
        assert nested["anyOf"][0] == {"type": "string"}
        assert nested["anyOf"][1]["pattern"] == "^fixed$"
        assert nested["default"] == {"pattern": "annotation data"}
        assert forwarded["largeScript"] == {"type": "string", "minLength": 1}
        assert forwarded["keptScript"] == {"type": "string", "maxLength": 1999, "minLength": 1999}
        assert forwarded["budgetScript"] == {"type": "string"}
        assert forwarded["budgetFloor"] == {"type": "string"}
        assert forwarded["keptList"]["maxItems"] == 1997
        assert forwarded["budgetList"] == {"type": "array", "items": {"type": "string"}}
        assert forwarded["keptFloorList"]["minItems"] == 2000
        assert forwarded["budgetFloorList"] == {"type": "array", "items": {"type": "string"}}
        assert forwarded["floatList"] == {"type": "array", "items": {"type": "string"}}
        assert forwarded["tupleList"]["items"] == [{"type": "string"}]
        assert schema["properties"]["budgetScript"]["maxLength"] == 2000
        assert schema["properties"]["budgetList"]["maxItems"] == 1998
        assert forwarded["impossibleList"] == {"type": "array", "items": {"type": "string"}}
        assert forwarded["impossibleScript"] == {"type": "string"}
        assert schema["properties"]["tupleList"]["items"][0]["maxLength"] == 2000
        assert schema["properties"]["impossibleList"]["minItems"] == 5
        assert body["tools"][0]["function"]["parameters"]["$defs"] == {
            "Bounded": {"type": "string"}
        }
        assert schema["$defs"]["Bounded"]["maxLength"] == 2000
        assert schema["properties"]["declarationKey"]["pattern"] == r"\S"
        assert schema["properties"]["nested"]["items"]["anyOf"][0]["pattern"] == "token"
        assert schema["properties"]["largeScript"]["maxLength"] == 65536

    def test_repetition_limits_match_the_measured_grammar_budget(self):
        # First bound llama-server refuses, measured against llama.cpp b10639 and b10679 by
        # posting each schema to a live server. maxItems costs N+2 rules, not N+1, so 1998 is
        # already over budget even though the other three keywords reach 2000.
        from routes.inference import _JSON_SCHEMA_REPETITION_LIMITS
        first_rejected = {"maxItems": 1998, "maxLength": 2000, "minItems": 2001, "minLength": 2000}
        assert _JSON_SCHEMA_REPETITION_LIMITS == {
            keyword: value - 1 for keyword, value in first_rejected.items()
        }

    def test_response_format_schema_drops_incompatible_constraints(self):
        # Guided decoding reaches the same grammar engine tool schemas do.
        rf = {
            "type": "json_schema",
            "json_schema": {
                "name": "report",
                "schema": {
                    "type": "object",
                    "properties": {"summary": {"type": "string", "maxLength": 2000}},
                },
            },
        }
        body = _build_passthrough_payload(**self._args(), response_format = rf)
        forwarded = body["response_format"]["json_schema"]["schema"]["properties"]["summary"]
        assert forwarded == {"type": "string"}
        assert body["response_format"]["type"] == "json_schema"
        assert body["response_format"]["json_schema"]["name"] == "report"
        assert rf["json_schema"]["schema"]["properties"]["summary"]["maxLength"] == 2000

    def test_response_format_json_object_schema_is_filtered_too(self):
        # An array bound only reaches the grammar when the items schema does too.
        rf = {
            "type": "json_object",
            "schema": {"type": "array", "items": {"type": "string"}, "maxItems": 1999},
        }
        body = _build_passthrough_payload(**self._args(), response_format = rf)
        assert body["response_format"] == {
            "type": "json_object",
            "schema": {"type": "array", "items": {"type": "string"}},
        }

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

    def test_an_explicit_seed_disables_slot_prompt_cache_reuse(self):
        seeded = _build_passthrough_payload(**self._args(), seed = 3407)
        randomized = _build_passthrough_payload(**self._args(), seed = -1)
        ordinary = _build_passthrough_payload(**self._args())

        assert seeded["seed"] == 3407
        assert seeded["cache_prompt"] is False
        assert randomized["seed"] == -1
        assert "cache_prompt" not in randomized
        assert "cache_prompt" not in ordinary

    def test_response_format_without_tools_omits_tool_fields(self):
        args = self._args()
        args["openai_tools"] = None

        body = _build_passthrough_payload(
            **args,
            response_format = {"type": "json_object"},
        )

        assert body["response_format"] == {"type": "json_object"}
        assert "tools" not in body
        assert "tool_choice" not in body

    def test_repetition_penalty_renamed(self):
        body = _build_passthrough_payload(**self._args(), repetition_penalty = 1.1)
        assert body.get("repeat_penalty") == 1.1
        assert "repetition_penalty" not in body

    def test_omitted_passthrough_max_tokens_uses_backend_context(self):
        args = self._args()
        args["max_tokens"] = None

        body = _build_passthrough_payload(**args, backend_ctx = 4096)

        assert body["max_tokens"] == 4096

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


class TestOpenAIPassthroughSSETerminalState:
    def test_done_sentinel(self):
        assert _openai_passthrough_sse_line_terminal_state("data: [DONE]") == "done"

    def test_finish_reason_with_space(self):
        line = 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        assert _openai_passthrough_sse_line_terminal_state(line) == "finish"

    def test_finish_reason_without_space(self):
        line = 'data:{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}'
        assert _openai_passthrough_sse_line_terminal_state(line) == "finish"

    def test_usage_chunk(self):
        line = 'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}'
        assert _openai_passthrough_sse_line_terminal_state(line) == "usage"

    def test_error_chunk(self):
        line = 'data: {"error":{"message":"boom"}}'
        assert _openai_passthrough_sse_line_terminal_state(line) == "error"

    def test_cap_parallel_tool_calls_accepts_no_space_after_data_colon(self):
        line = (
            'data:{"choices":[{"delta":{"tool_calls":['
            '{"index":0,"function":{"name":"a"}},'
            '{"index":1,"function":{"name":"b"}}]}}]}'
        )

        capped = _normalize_openai_passthrough_sse_line(line, cap_parallel_tool_calls = True)

        data = json.loads(capped[len("data:") :].lstrip())
        assert data["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "function": {"name": "a"}}
        ]

    def test_plain_content_line_is_returned_identically(self):
        # The relay dispatches terminal classification on `out_line is raw_line`,
        # so the no-mutation path must return the identical string object.
        line = 'data: {"choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":null}]}'
        assert _normalize_openai_passthrough_sse_line(line) is line
        assert _normalize_openai_passthrough_sse_line(line, cap_parallel_tool_calls = True) is line

    def test_reasoning_key_inside_content_text_keeps_line_identical(self):
        # Fast-path substring gate fires, but the parse finds nothing to change:
        # the original object must come back so the relay stays byte-identical.
        line = (
            'data: {"choices":[{"index":0,"delta":{"content":'
            '"mentions \\"reasoning_content\\" in text"},"finish_reason":null}]}'
        )
        assert _normalize_openai_passthrough_sse_line(line) is line

    def test_reasoning_only_delta_gets_empty_content(self):
        line = (
            'data: {"choices":[{"index":0,'
            '"delta":{"reasoning_content":"thinking"},'
            '"finish_reason":null}]}'
        )

        normalized = _normalize_openai_passthrough_sse_line(line)

        data = json.loads(normalized[len("data:") :].lstrip())
        delta = data["choices"][0]["delta"]
        assert delta["reasoning_content"] == "thinking"
        assert delta["content"] == ""

    def test_reasoning_normalization_preserves_done_sentinel(self):
        assert _normalize_openai_passthrough_sse_line("data: [DONE]") == "data: [DONE]"


# =====================================================================

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

    def test_reasoning_effort_none_forwarded_for_effort_style_models(self):
        body = _build_openai_passthrough_body(
            self._payload(enable_thinking = False, reasoning_effort = "none"),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(reasoning_style = "reasoning_effort"),
        )
        assert body["chat_template_kwargs"] == {"reasoning_effort": "none"}

    def test_reasoning_effort_minimal_maps_to_low_for_effort_style_models(self):
        body = _build_openai_passthrough_body(
            self._payload(enable_thinking = True, reasoning_effort = "minimal"),
            backend_ctx = 4096,
            llama_backend = _reasoning_backend(reasoning_style = "reasoning_effort"),
        )
        assert body["chat_template_kwargs"] == {"reasoning_effort": "low"}

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

# OpenAI API compatibility helpers - verified spec edge cases
# =====================================================================


class TestOpenAICompatibilityHelpers:
    def test_max_completion_tokens_wins_over_deprecated_max_tokens(self):
        payload = SimpleNamespace(max_tokens = 128, max_completion_tokens = 64)
        assert _effective_max_tokens(payload) == 64

    def test_openai_compat_max_tokens_returns_none_when_omitted(self):
        payload = SimpleNamespace(max_tokens = None, max_completion_tokens = None)
        assert _effective_openai_max_tokens(payload) is None

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (SimpleNamespace(max_tokens = 8192, max_completion_tokens = None), 8192),
            (SimpleNamespace(max_tokens = 8192, max_completion_tokens = 256), 256),
        ],
    )
    def test_openai_compat_explicit_values_pass_through(self, payload, expected):
        assert _effective_openai_max_tokens(payload) == expected

    @pytest.mark.parametrize(
        ("payload", "param"),
        [
            (SimpleNamespace(max_tokens = "128", max_completion_tokens = None), "max_tokens"),
            (SimpleNamespace(max_tokens = True, max_completion_tokens = None), "max_tokens"),
            (SimpleNamespace(max_tokens = 12.5, max_completion_tokens = None), "max_tokens"),
            (
                SimpleNamespace(max_tokens = None, max_completion_tokens = "128"),
                "max_completion_tokens",
            ),
        ],
    )
    def test_openai_compat_max_tokens_rejects_non_integer_explicit_values(self, payload, param):
        with pytest.raises(HTTPException) as exc:
            _effective_openai_max_tokens(payload)

        assert exc.value.status_code == 400
        assert exc.value.detail["error"]["param"] == param
        assert exc.value.detail["error"]["code"] == "invalid_type"

    def test_openai_compat_max_tokens_zero_is_valid_and_negative_rejected(self):
        # Legacy completions spec: max_tokens has minimum 0, so 0 must pass
        # through; only negatives are invalid_value.
        assert _effective_openai_max_tokens_from_values(0) == 0

        with pytest.raises(HTTPException) as exc:
            _effective_openai_max_tokens_from_values(-1)

        assert exc.value.status_code == 400
        assert exc.value.detail["error"]["code"] == "invalid_value"
        assert exc.value.detail["error"]["param"] == "max_tokens"

    def test_chat_reasoning_chunk_carries_empty_content(self):
        from routes.inference import _chat_reasoning_chunk

        line = _chat_reasoning_chunk("chatcmpl-test", 123, "gguf", "thinking...")
        chunk = json.loads(line[len("data: ") :])
        delta = chunk["choices"][0]["delta"]

        assert delta["reasoning_content"] == "thinking..."
        assert delta["content"] == ""

    def test_passthrough_upstream_headers_include_backend_auth(self):
        headers = _openai_passthrough_upstream_headers(
            llama_backend = SimpleNamespace(_auth_headers = {"Authorization": "Bearer secret"}),
        )

        assert headers["Authorization"] == "Bearer secret"
        assert headers["Connection"] == "close"

    def test_openai_admission_capacity_prefers_backend_effective_slots(self):
        request = SimpleNamespace(
            app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
        )
        backend = SimpleNamespace(effective_parallel_slots = 3)

        assert _openai_llama_admission_capacity(request, backend) == 3

    @pytest.mark.parametrize("backend_value", [None, 0, -1, "not-an-int"])
    def test_openai_admission_capacity_falls_back_to_app_state(self, backend_value):
        request = SimpleNamespace(
            app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 2))
        )
        backend = SimpleNamespace(effective_parallel_slots = backend_value)

        assert _openai_llama_admission_capacity(request, backend) == 2

    def test_openai_admission_capacity_falls_back_to_one_without_request(self):
        assert _openai_llama_admission_capacity(None, SimpleNamespace()) == 1

    def test_openai_admission_non_streaming_exits_invalidated_waiter(self):
        async def _run():
            queue = get_llama_admission_queue("http://llama.invalidated.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None
            reservation = queue.reserve(capacity = 1, config = LlamaAdmissionConfig())
            assert reservation._waiter is not None

            reservation._waiter.future.cancel()

            with pytest.raises(LlamaAdmissionCancelled):
                await asyncio.wait_for(
                    _wait_for_openai_admission_non_streaming(
                        reservation,
                        LlamaAdmissionConfig(),
                        request = None,
                        cancel_event = None,
                    ),
                    timeout = 0.1,
                )

            blocker.release()
            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0

        asyncio.run(_run())

    def test_openai_admission_stream_exits_invalidated_waiter(self):
        async def _run():
            queue = get_llama_admission_queue("http://llama.invalidated.stream.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None
            reservation = queue.reserve(capacity = 1, config = LlamaAdmissionConfig())
            assert reservation._waiter is not None

            reservation._waiter.future.cancel()

            chunks = _openai_admission_wait_stream_chunks(
                reservation,
                LlamaAdmissionConfig(),
                request = None,
                cancel_event = None,
            )
            with pytest.raises(LlamaAdmissionCancelled):
                await asyncio.wait_for(chunks.__anext__(), timeout = 0.1)

            blocker.release()
            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0

        asyncio.run(_run())

    def test_openai_compat_stream_stall_timeout_uses_default(self, monkeypatch):
        monkeypatch.delenv(_OPENAI_COMPAT_STREAM_STALL_TIMEOUT_ENV, raising = False)
        assert _openai_compat_stream_stall_timeout() == 120.0

    def test_openai_compat_stream_stall_timeout_uses_env_override(self, monkeypatch):
        monkeypatch.setenv(_OPENAI_COMPAT_STREAM_STALL_TIMEOUT_ENV, "4.5")
        assert _openai_compat_stream_stall_timeout() == 4.5

    @pytest.mark.parametrize("raw_value", ["", "not-a-float"])
    def test_openai_compat_stream_stall_timeout_invalid_env_uses_default(
        self, monkeypatch, raw_value
    ):
        monkeypatch.setenv(_OPENAI_COMPAT_STREAM_STALL_TIMEOUT_ENV, raw_value)
        assert _openai_compat_stream_stall_timeout() == 120.0

    @pytest.mark.parametrize("raw_value", ["0", "-1"])
    def test_openai_compat_stream_stall_timeout_non_positive_env_disables(
        self, monkeypatch, raw_value
    ):
        monkeypatch.setenv(_OPENAI_COMPAT_STREAM_STALL_TIMEOUT_ENV, raw_value)
        assert _openai_compat_stream_stall_timeout() is None

    def test_openai_stream_error_sse_closes_with_done(self):
        error = {"error": {"message": "boom"}}
        assert _openai_stream_error_sse(error) == (
            'data: {"error": {"message": "boom"}}\n\ndata: [DONE]\n\n'
        )

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

    def test_completion_stream_monitor_reads_usage_before_client_strip(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/completions",
            method = "POST",
            model = "m",
            prompt = "hi",
            context_length = 100,
        )
        event = (
            b'data: {"id":"chatcmpl-test","choices":[{"text":"done","finish_reason":"stop"}],'
            b'"usage":{"prompt_tokens":4,"completion_tokens":6,"total_tokens":10}}\n'
        )

        _monitor_openai_sse_event(monitor_id, event, context_length = 100)
        out = _cmpl_stream_event_out(event, include_usage = False)

        assert out is not None
        assert b'"usage"' not in out
        [entry] = monitor.snapshot()
        assert entry["reply"] == "done"
        assert entry["prompt_tokens"] == 4
        assert entry["completion_tokens"] == 6
        assert entry["total_tokens"] == 10
        assert entry["context_usage"] == 0.1

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
        assert image_b64 is None

    def _turn(self, images: int):
        content = [{"type": "text", "text": "what is this?"}]
        for _ in range(images):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            )
        return {"role": "user", "content": content}

    def test_two_images_on_one_turn_are_counted(self):
        payload = ChatCompletionRequest(messages = [self._turn(2)])
        assert _images_in_last_user_message(payload.messages) == 2

    def test_one_image_per_turn_is_not_a_multi_image_call(self):
        payload = ChatCompletionRequest(
            messages = [
                self._turn(1),
                {"role": "assistant", "content": "the first one"},
                self._turn(1),
            ]
        )
        assert _images_in_last_user_message(payload.messages) == 1

    def test_earlier_multi_image_turn_does_not_block_a_later_one(self):
        payload = ChatCompletionRequest(
            messages = [
                self._turn(2),
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "and now just text"},
            ]
        )
        assert _images_in_last_user_message(payload.messages) == 0

    def test_undecodable_image_urls_still_count(self):
        """Structural count, not a decoded one: remote and empty data URLs are
        image parts the caller attached, whether or not extraction can read them."""
        payload = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "compare these"},
                        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
                    ],
                }
            ]
        )
        assert _images_in_last_user_message(payload.messages) == 2

    def test_a_trailing_assistant_turn_does_not_change_which_turn_is_counted(self):
        payload = ChatCompletionRequest(
            messages = [self._turn(2), {"role": "assistant", "content": "partial"}]
        )
        assert _images_in_last_user_message(payload.messages) == 2

    def test_latest_image_wins_over_earlier_turns(self):
        def turn(letter):
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"what is {letter}?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{letter}"},
                    },
                ],
            }

        payload = ChatCompletionRequest(
            messages = [
                turn("AAAA"),
                {"role": "assistant", "content": "the first one"},
                turn("BBBB"),
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "BBBB"

    def test_single_image_still_returned(self):
        payload = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,AAAA"},
                        },
                    ],
                }
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "AAAA"

    def test_payloadless_later_image_keeps_earlier_one(self):
        def turn(url):
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }

        for empty_url in ("data:image/png;base64,", "data:image/png;base64"):
            payload = ChatCompletionRequest(
                messages = [
                    turn("data:image/png;base64,AAAA"),
                    {"role": "assistant", "content": "a cat"},
                    turn(empty_url),
                ]
            )

            _, _, image_b64 = _extract_content_parts(payload.messages)

            assert image_b64 == "AAAA", empty_url

    def test_user_attachment_outranks_assistant_generated_image(self):
        payload = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,USERPHOTO"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "here is a cartoon of it"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,GENERATED"},
                        },
                    ],
                },
                {"role": "user", "content": "what colour was the shirt?"},
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "USERPHOTO"

    def test_assistant_only_image_is_still_returned(self):
        payload = ChatCompletionRequest(
            messages = [
                {"role": "user", "content": "show me something"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "here you go"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,GENERATED"},
                        },
                    ],
                },
                {"role": "user", "content": "describe it"},
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "GENERATED"

    def test_first_image_wins_within_one_message(self):
        # The composer allows multi-select, and findLatestUserImageBase64
        # (chat-adapter.ts) names the first of them, so this must agree.
        payload = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "compare these"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,LEFT"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,RIGHT"},
                        },
                    ],
                }
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "LEFT"

    def test_later_turn_still_wins_over_a_multi_image_turn(self):
        # Per-message first, per-thread latest: the two rules compose.
        payload = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "compare these"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,LEFT"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,RIGHT"},
                        },
                    ],
                },
                {"role": "assistant", "content": "two cats"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "and this?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,LATER"},
                        },
                    ],
                },
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "LATER"

    def test_payloadless_part_does_not_claim_the_message_slot(self):
        # An empty data URL is not an image, so the first real one still wins.
        payload = ChatCompletionRequest(
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "compare these"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,"},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,RIGHT"},
                        },
                    ],
                }
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "RIGHT"

    def test_earlier_real_image_survives_a_payloadless_opening_turn(self):
        # The old helper latched "" here and suppressed every later image, so
        # the request reached the model with none.
        def turn(url):
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }

        payload = ChatCompletionRequest(
            messages = [
                turn("data:image/png;base64,"),
                {"role": "assistant", "content": "I cannot see an image"},
                turn("data:image/png;base64,REAL"),
            ]
        )

        _, _, image_b64 = _extract_content_parts(payload.messages)

        assert image_b64 == "REAL"


# =====================================================================

# _friendly_error - httpx transport failures
# =====================================================================


class TestFriendlyErrorHttpx:
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
        assert "first token within 20 minutes" in _friendly_error(exc)

    def test_non_httpx_unchanged(self):
        # Non-httpx exceptions still fall through to the substring heuristics
        # - a context-size message must still produce "Message too long".
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

    def test_local_message_builders_preserve_assistant_reasoning(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "hello back",
                    "reasoning_content": "Answer the greeting briefly.",
                },
                {"role": "user", "content": "why is quantum mechanics random?"},
            ],
        )

        assert req.messages[1].reasoning_content == "Answer the greeting briefly."
        passthrough = _openai_messages_for_passthrough(req)
        gguf, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
        _, standard_local, _ = _extract_content_parts(req.messages)
        assert passthrough[1]["reasoning_content"] == "Answer the greeting briefly."
        assert gguf[1]["reasoning_content"] == "Answer the greeting briefly."
        assert standard_local[1]["reasoning_content"] == "Answer the greeting briefly."

    def test_standard_local_builder_preserves_reasoning_only_assistant_turn(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                {"role": "user", "content": "What is the answer?"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "The answer is forty-two.",
                },
                {"role": "user", "content": "Explain why."},
            ],
        )

        _, messages, _ = _extract_content_parts(req.messages)

        assert messages == [
            {"role": "user", "content": "What is the answer?"},
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "The answer is forty-two.",
            },
            {"role": "user", "content": "Explain why."},
        ]

    def test_reasoning_only_assistant_turn_is_not_treated_as_a_stop_sentinel(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "The response stopped before its final answer.",
                },
                {"role": "user", "content": "continue"},
            ],
        )

        passthrough = _openai_messages_for_passthrough(req)
        gguf, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
        assert [message["role"] for message in passthrough] == ["user", "assistant", "user"]
        assert [message["role"] for message in gguf] == ["user", "assistant", "user"]
        assert passthrough[1]["content"] == ""
        assert gguf[1]["content"] == ""

    def test_reasoning_only_dict_gains_llama_cpp_content_key(self):
        messages = _drop_empty_assistant_sentinels(
            [{"role": "assistant", "reasoning_content": "A completed answer."}]
        )
        assert messages == [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "A completed answer.",
            }
        ]

    def test_chained_synthetic_tool_fragments_collapse_into_visible_answer(self):
        def _synthetic_call(call_id: str) -> dict:
            return {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"_server_tool": true}',
                },
            }

        messages = [
            {"role": "user", "content": "Find it."},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Searching first."}],
                "reasoning_content": "first trace",
                "tool_calls": [_synthetic_call("call-1")],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "first result"},
            {
                "role": "assistant",
                "reasoning_content": "second trace",
                "tool_calls": [_synthetic_call("call-2")],
            },
            {"role": "tool", "tool_call_id": "call-2", "content": "second result"},
            {
                "role": "assistant",
                "content": "Here is the answer.",
                "reasoning_content": "final trace",
            },
            {"role": "user", "content": "Thanks."},
        ]

        out = _strip_provider_synthetic_tool_history(messages)

        assert [message["role"] for message in out] == ["user", "assistant", "user"]
        assert out[1]["content"] == [
            {"type": "text", "text": "Searching first."},
            {"type": "text", "text": "Here is the answer."},
        ]
        assert out[1]["reasoning_content"] == "first trace\n\nsecond trace\n\nfinal trace"

    def test_synthetic_reasoning_fragment_before_user_stays_valid(self):
        messages = [
            {"role": "user", "content": "Find it."},
            {
                "role": "assistant",
                "reasoning_content": "I should search.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": '{"_server_tool": true}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "user", "content": "Try another query."},
        ]

        out = _strip_provider_synthetic_tool_history(messages)

        assert [message["role"] for message in out] == ["user", "assistant", "user"]
        assert out[1] == {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should search.",
        }


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
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/chat/completions")
        method = "POST"

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

    @staticmethod
    def _sse_payloads(chunks):
        payloads = []
        for chunk in chunks:
            if isinstance(chunk, bytes):
                chunk = chunk.decode()
            for line in str(chunk).splitlines():
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    continue
                try:
                    payloads.append(json.loads(data))
                except json.JSONDecodeError:
                    pass
        return payloads

    def _run_gguf_case(
        self,
        monkeypatch,
        *,
        generate = None,
        tool_generate = None,
        payload_kwargs = None,
        backend_kwargs = None,
        request = None,
    ):
        import routes.inference as inf_mod

        reset_tool_policy()

        def _plain(**_kwargs):
            raise AssertionError("plain GGUF path should not be used")

        backend_data = {
            "is_loaded": True,
            "is_vision": False,
            "supports_tools": tool_generate is not None,
            "supports_reasoning": True,
            "reasoning_always_on": True,
            "_is_audio": False,
            "model_identifier": "test-gguf",
            "context_length": 4096,
            "generate_chat_completion": generate or _plain,
        }
        if tool_generate is not None:
            backend_data["generate_chat_completion_with_tools"] = tool_generate
        if backend_kwargs:
            backend_data.update(backend_kwargs)
        backend = SimpleNamespace(**backend_data)

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        request_data = {
            "model": "default",
            "messages": [{"role": "user", "content": "hi"}],
        }
        if payload_kwargs:
            request_data.update(payload_kwargs)
        payload = ChatCompletionRequest(**request_data)
        response = self._drive(
            openai_chat_completions(
                payload,
                request = request or self._Request(),
                current_subject = "test",
            )
        )
        result = SimpleNamespace(response = response, monitor = monitor, backend = backend)
        if request_data.get("stream"):
            result.chunks = self._consume_response(response)
            result.payloads = self._sse_payloads(result.chunks)
        else:
            result.body = json.loads(response.body)
        return result

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
            context_length = 4096,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            enable_tools = True,
            enabled_tools = ["web_search"],
            stream = True,
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

        response = self._drive(
            openai_chat_completions(payload, request = self._Request(), current_subject = "test")
        )
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
            context_length = 4096,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            enable_tools = True,
            enabled_tools = ["web_search"],
            parallel_tool_calls = False,
            stream = True,
            messages = [{"role": "user", "content": "search once"}],
        )

        response = self._drive(
            openai_chat_completions(payload, request = self._Request(), current_subject = "test")
        )
        self._consume_response(response)

        assert captured["kwargs"]["disable_parallel_tool_use"] is True

    def test_enabled_tools_web_search_only_enters_gguf_tool_loop(self, monkeypatch):
        """#9730: enable_tools + enabled_tools: ["web_search"] on a local GGUF
        must enter Studio's loop with that catalog, not answer from memory."""
        import routes.inference as inf_mod

        reset_tool_policy()
        captured = {}

        def _plain(**kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**kwargs):
            captured["kwargs"] = kwargs
            yield {"type": "content", "text": "The current version of the Linux kernel is 6.10."}

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            model_identifier = "Qwen3.5-2B-MTP-GGUF",
            context_length = 4096,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            messages = [
                {
                    "role": "user",
                    "content": (
                        "Search the web for the current version of the Linux kernel, "
                        "then answer in one sentence."
                    ),
                }
            ],
            enable_tools = True,
            enabled_tools = ["web_search"],
            permission_mode = "off",
            tool_choice = {"type": "function", "function": {"name": "web_search"}},
            max_tokens = 512,
            temperature = 0.0,
            stream = True,
        )

        response = self._drive(
            openai_chat_completions(payload, request = self._Request(), current_subject = "test")
        )
        self._consume_response(response)

        assert "kwargs" in captured
        assert [t["function"]["name"] for t in captured["kwargs"]["tools"]] == ["web_search"]
        assert captured["kwargs"]["tool_choice"] == {
            "type": "function",
            "function": {"name": "web_search"},
        }
        assert captured["kwargs"]["permission_mode"] == "off"

    def test_api_server_tool_loop_keeps_the_current_date(self, monkeypatch):
        import routes.inference as inf_mod

        captured = {}

        def _tools(**kwargs):
            captured.update(kwargs)
            yield {"type": "content", "text": "done"}

        class ApiRequest(self._Request):
            headers = {"authorization": "Bearer sk-unsloth-test"}
            state = SimpleNamespace(skip_api_monitor = True)

        monkeypatch.setattr(
            inf_mod,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-08-15.",
        )
        monkeypatch.setattr(inf_mod, "_request_is_internal_workflow", lambda _request: False)
        self._run_gguf_case(
            monkeypatch,
            tool_generate = _tools,
            payload_kwargs = {
                "enable_tools": True,
                "enabled_tools": ["web_search"],
                "permission_mode": "off",
                "stream": True,
            },
            request = ApiRequest(),
        )

        system_messages = [
            message for message in captured["messages"] if message["role"] == "system"
        ]
        assert len(system_messages) == 1
        assert system_messages[0]["content"].startswith("The current date is 2026-08-15.\n\n")

    def test_forced_tool_choice_must_be_in_the_selected_gguf_catalog(self, monkeypatch):
        import routes.inference as inf_mod

        reset_tool_policy()

        def _tools(**_kwargs):
            raise AssertionError("invalid forced choice must not start generation")

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            model_identifier = "test-gguf",
            context_length = 4096,
            generate_chat_completion_with_tools = _tools,
        )
        monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        payload = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "run python"}],
            enable_tools = True,
            enabled_tools = ["web_search"],
            permission_mode = "off",
            tool_choice = {"type": "function", "function": {"name": "python"}},
            stream = True,
        )

        with pytest.raises(HTTPException) as exc:
            self._drive(
                openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )
            )

        assert exc.value.status_code == 400
        assert exc.value.detail["error"]["param"] == "tool_choice"

    # Two sites reject this shape and `_should_validate_before_switch()` picks
    # which; the early one runs before the api_monitor row is opened. It reads
    # process-wide state, so unpinned this asserted whichever site the xdist
    # worker happened to leave reachable. Pinned, both sites are covered.
    @pytest.mark.parametrize("validate_before_switch", [False, True])
    def test_confirm_tool_calls_requires_streaming_for_gguf_tools(
        self, monkeypatch, validate_before_switch
    ):
        import routes.inference as inf_mod

        def _plain(**kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**kwargs):
            raise AssertionError("tool loop should be rejected before starting")

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            model_identifier = "test-gguf",
            context_length = 4096,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(
            inf_mod, "_should_validate_before_switch", lambda: validate_before_switch
        )
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)

        payload = ChatCompletionRequest(
            model = "default",
            enable_tools = True,
            enabled_tools = ["web_search"],
            confirm_tool_calls = True,
            stream = False,
            messages = [{"role": "user", "content": "search once"}],
        )

        with pytest.raises(HTTPException) as exc:
            self._drive(
                openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )
            )
        assert exc.value.status_code == 400
        assert "requires stream=true" in exc.value.detail["error"]["message"]
        if validate_before_switch:
            # Early site rejects before the monitor row is opened, so there is no
            # row to fail; opening one here would be a behaviour change, not a fix.
            assert monitor.snapshot() == []
        else:
            [entry] = monitor.snapshot()
            assert entry["status"] == "error"
            assert "confirm_tool_calls requires stream=true" in entry["error"]
        # Neither site may leave a row running.
        assert monitor.active_count() == 0

    def test_standard_gguf_stream_splits_reasoning_content(self, monkeypatch):
        def _generate(**_kwargs):
            yield "<thi"
            yield "<think>plan"
            yield "<think>plan</think>vis"
            yield "<think>plan</think>visible"
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "finish_reason": "stop",
            }

        result = self._run_gguf_case(
            monkeypatch,
            generate = _generate,
            payload_kwargs = {"stream": True},
        )
        deltas = [p["choices"][0].get("delta", {}) for p in result.payloads if p.get("choices")]

        assert "".join(d.get("reasoning_content", "") for d in deltas) == "plan"
        assert "".join(d.get("content", "") for d in deltas) == "visible"
        assert all("<think>" not in d.get("content", "") for d in deltas)
        assert all("content" in d for d in deltas if "reasoning_content" in d)
        [entry] = result.monitor.snapshot()
        assert entry["reply"] == "visible"

    def test_standard_gguf_stream_queued_request_sends_keepalive_before_generation(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            class Request(self._Request):
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))

            def _generate(**_kwargs):
                raise AssertionError("standard GGUF generation must not start while queued")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = False,
                supports_reasoning = True,
                reasoning_always_on = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.standard.test",
                effective_parallel_slots = 1,
                generate_chat_completion = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.01")
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

            queue = get_llama_admission_queue("http://llama.standard.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                stream = True,
            )
            response = await openai_chat_completions(
                payload,
                request = Request(),
                current_subject = "test",
            )
            iterator = response.body_iterator
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                assert chunk == ": admission-wait\n\n"
                snapshot = queue.snapshot()
                assert snapshot.active == 1
                assert snapshot.queued == 1
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_standard_gguf_stream_close_after_first_chunk_cleans_tracker(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            cancel_id = "standard-stream-close-cleanup"

            def _generate(**_kwargs):
                yield "visible"

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = False,
                supports_reasoning = True,
                reasoning_always_on = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.standard.test",
                effective_parallel_slots = 1,
                generate_chat_completion = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                stream = True,
                cancel_id = cancel_id,
            )
            response = await openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
            iterator = response.body_iterator
            assert cancel_id in inf_mod._CANCEL_REGISTRY
            await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
            aclose = getattr(iterator, "aclose", None)
            assert aclose is not None
            await aclose()

            assert cancel_id not in inf_mod._CANCEL_REGISTRY
            assert get_llama_admission_queue("http://llama.standard.test").snapshot().active == 0

        asyncio.run(_run())

    def test_standard_gguf_stream_task_cancel_after_first_chunk_finalizes_monitor(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            started = threading.Event()
            released = threading.Event()

            def _generate(**kwargs):
                cancel_event = kwargs["cancel_event"]
                started.set()
                while not cancel_event.is_set():
                    time.sleep(0.005)
                released.set()
                yield from ()

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = False,
                supports_reasoning = True,
                reasoning_always_on = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.standard.test",
                effective_parallel_slots = 1,
                generate_chat_completion = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                stream = True,
            )
            response = await openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
            iterator = response.body_iterator
            assert await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
            pending = asyncio.create_task(iterator.__anext__())
            assert await asyncio.to_thread(started.wait, 1.0)

            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout = 1.0)

            assert released.is_set()
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0
            assert get_llama_admission_queue("http://llama.standard.test").snapshot().active == 0

        asyncio.run(_run())

    def test_gguf_tool_stream_queued_request_sends_keepalive_before_generation(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request(self._Request):
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))

            async def fake_select_tools(*_args, **_kwargs):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]

            def _generate(**_kwargs):
                raise AssertionError("GGUF tool loop must not start while queued")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = True,
                supports_reasoning = True,
                reasoning_always_on = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.tool.test",
                effective_parallel_slots = 1,
                generate_chat_completion = lambda **_kwargs: "unused",
                generate_chat_completion_with_tools = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.01")
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(inf_mod, "_select_request_tools", fake_select_tools)

            queue = get_llama_admission_queue("http://llama.tool.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                enable_tools = True,
                stream = True,
            )
            response = await openai_chat_completions(
                payload,
                request = Request(),
                current_subject = "test",
            )
            iterator = response.body_iterator
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                assert chunk == ": admission-wait\n\n"
                snapshot = queue.snapshot()
                assert snapshot.active == 1
                assert snapshot.queued == 1
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_gguf_gated_tool_start_gets_a_separate_prompt_flush(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_select_tools(*_args, **_kwargs):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "python",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]

            stopped = threading.Event()

            def _generate(**kwargs):
                cancel_event = kwargs["cancel_event"]
                yield {"type": "status", "text": "Waiting for approval: Python"}
                yield {
                    "type": "tool_start",
                    "tool_name": "python",
                    "tool_call_id": "call_0",
                    "arguments": {"code": "print(1)"},
                    "awaiting_confirmation": True,
                    "approval_id": "approval_0",
                }
                while not cancel_event.is_set():
                    time.sleep(0.005)
                stopped.set()

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = True,
                supports_reasoning = True,
                reasoning_always_on = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.tool.test",
                effective_parallel_slots = 1,
                generate_chat_completion = lambda **_kwargs: "unused",
                generate_chat_completion_with_tools = _generate,
            )
            monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(inf_mod, "_select_request_tools", fake_select_tools)
            monkeypatch.setattr(inf_mod, "TOOL_APPROVAL_FLUSH_DELAY_S", 0.01)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "run python"}],
                enable_tools = True,
                stream = True,
            )
            response = await openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
            iterator = response.body_iterator
            try:
                for _ in range(8):
                    chunk = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                    if '"tool_start"' in chunk:
                        break
                else:
                    pytest.fail("gated tool_start was not emitted")

                next_turn = [False]
                asyncio.get_running_loop().call_soon(next_turn.__setitem__, 0, True)
                flush = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                assert flush == ": keep-alive\n\n"
                assert next_turn[0] is True
            finally:
                await iterator.aclose()

            assert stopped.is_set()

        asyncio.run(_run())

    def test_gguf_tool_stream_task_cancel_after_first_chunk_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_select_tools(*_args, **_kwargs):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]

            started = threading.Event()
            released = threading.Event()

            def _tools(**kwargs):
                cancel_event = kwargs["cancel_event"]
                started.set()
                while not cancel_event.is_set():
                    time.sleep(0.005)
                released.set()
                yield from ()

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = True,
                supports_reasoning = True,
                reasoning_always_on = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.tool.test",
                effective_parallel_slots = 1,
                generate_chat_completion = lambda **_kwargs: "unused",
                generate_chat_completion_with_tools = _tools,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(inf_mod, "_select_request_tools", fake_select_tools)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                enable_tools = True,
                stream = True,
            )
            response = await openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
            iterator = response.body_iterator
            assert await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
            pending = asyncio.create_task(iterator.__anext__())
            assert await asyncio.to_thread(started.wait, 1.0)

            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout = 1.0)

            assert released.is_set()
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0
            assert get_llama_admission_queue("http://llama.tool.test").snapshot().active == 0

        asyncio.run(_run())

    def test_global_enable_tools_does_not_preempt_response_format_passthrough(self, monkeypatch):
        import routes.inference as inf_mod

        reset_tool_policy()
        set_tool_policy(True)
        captured = {}

        def _plain(**_kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**_kwargs):
            raise AssertionError("Unsloth tool loop should not steal response_format")

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            _is_audio = False,
            model_identifier = "test-gguf",
            context_length = 4096,
            base_url = "http://llama.policy.test",
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )

        async def fake_passthrough(llama_backend, payload, model_name, **_kwargs):
            captured["body"] = inf_mod._build_openai_passthrough_body(
                payload,
                backend_ctx = llama_backend.context_length,
                llama_backend = llama_backend,
            )
            return inf_mod.JSONResponse({"ok": True, "model": model_name})

        try:
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_non_streaming",
                fake_passthrough,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "json"}],
                response_format = {"type": "json_object"},
            )
            response = self._drive(
                openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )
            )

            assert json.loads(response.body)["ok"] is True
            assert captured["body"]["response_format"] == {"type": "json_object"}
            assert "tools" not in captured["body"]
            assert "tool_choice" not in captured["body"]
        finally:
            reset_tool_policy()

    def test_global_enable_tools_does_not_replace_client_tools_passthrough(self, monkeypatch):
        import routes.inference as inf_mod

        reset_tool_policy()
        set_tool_policy(True)
        captured = {}
        client_tools = [
            {
                "type": "function",
                "function": {
                    "name": "client_lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        def _plain(**_kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**_kwargs):
            raise AssertionError("Unsloth tool loop should not replace client tools")

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            _is_audio = False,
            model_identifier = "test-gguf",
            context_length = 4096,
            base_url = "http://llama.policy.test",
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )

        async def fake_passthrough(llama_backend, payload, model_name, **_kwargs):
            captured["body"] = inf_mod._build_openai_passthrough_body(
                payload,
                backend_ctx = llama_backend.context_length,
                llama_backend = llama_backend,
            )
            return inf_mod.JSONResponse({"ok": True, "model": model_name})

        try:
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_non_streaming",
                fake_passthrough,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "use client tool"}],
                tools = client_tools,
            )
            response = self._drive(
                openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )
            )

            assert json.loads(response.body)["ok"] is True
            assert captured["body"]["tools"] == client_tools
            assert captured["body"]["tool_choice"] == "auto"
        finally:
            reset_tool_policy()

    def test_global_enable_tools_honors_client_tool_choice_none(self, monkeypatch):
        import routes.inference as inf_mod

        reset_tool_policy()
        set_tool_policy(True)
        client_tools = [
            {
                "type": "function",
                "function": {
                    "name": "client_lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        def _plain(**kwargs):
            assert kwargs["max_tokens"] is None
            yield "plain response"

        def _tools(**_kwargs):
            raise AssertionError("tool_choice='none' must not start Unsloth's tool loop")

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            _is_audio = False,
            model_identifier = "test-gguf",
            context_length = 4096,
            base_url = "http://llama.policy.test",
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )

        try:
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "do not use tools"}],
                tools = client_tools,
                tool_choice = "none",
            )
            response = self._drive(
                openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )
            )

            assert json.loads(response.body)["choices"][0]["message"]["content"] == "plain response"
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["reply"] == "plain response"
            assert monitor.active_count() == 0
        finally:
            reset_tool_policy()

    def test_enabled_tools_without_enable_tools_keeps_response_format_passthrough(
        self, monkeypatch
    ):
        import routes.inference as inf_mod

        reset_tool_policy()
        captured = {}

        def _plain(**_kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**_kwargs):
            raise AssertionError("enabled_tools alone must not start Unsloth's tool loop")

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            _is_audio = False,
            model_identifier = "test-gguf",
            context_length = 4096,
            base_url = "http://llama.enabled-tools.test",
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )

        async def fake_passthrough(llama_backend, payload, model_name, **_kwargs):
            captured["body"] = inf_mod._build_openai_passthrough_body(
                payload,
                backend_ctx = llama_backend.context_length,
                llama_backend = llama_backend,
            )
            return inf_mod.JSONResponse({"ok": True, "model": model_name})

        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(inf_mod, "_openai_passthrough_non_streaming", fake_passthrough)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)

        payload = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "json"}],
            enabled_tools = ["web_search"],
            response_format = {"type": "json_object"},
        )
        response = self._drive(
            openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
        )

        assert json.loads(response.body)["ok"] is True
        assert captured["body"]["response_format"] == {"type": "json_object"}

    def test_enabled_tools_without_enable_tools_keeps_client_tools_passthrough(self, monkeypatch):
        import routes.inference as inf_mod

        reset_tool_policy()
        captured = {}
        client_tools = [
            {
                "type": "function",
                "function": {
                    "name": "client_lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        def _plain(**_kwargs):
            raise AssertionError("plain GGUF path should not be used")

        def _tools(**_kwargs):
            raise AssertionError("enabled_tools alone must not start Unsloth's tool loop")

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = True,
            _is_audio = False,
            model_identifier = "test-gguf",
            context_length = 4096,
            base_url = "http://llama.enabled-tools.test",
            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            generate_chat_completion = _plain,
            generate_chat_completion_with_tools = _tools,
        )

        async def fake_passthrough(llama_backend, payload, model_name, **_kwargs):
            captured["body"] = inf_mod._build_openai_passthrough_body(
                payload,
                backend_ctx = llama_backend.context_length,
                llama_backend = llama_backend,
            )
            return inf_mod.JSONResponse({"ok": True, "model": model_name})

        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(inf_mod, "_openai_passthrough_non_streaming", fake_passthrough)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)

        payload = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "use client tool"}],
            enabled_tools = ["web_search"],
            tools = client_tools,
        )
        response = self._drive(
            openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
        )

        assert json.loads(response.body)["ok"] is True
        assert captured["body"]["tools"] == client_tools
        assert captured["body"]["tool_choice"] == "auto"

    def test_reasoning_capable_gguf_stream_splits_reasoning_by_default(self, monkeypatch):
        def _generate(**_kwargs):
            yield "<think>plan</think>visible"
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "finish_reason": "stop",
            }

        result = self._run_gguf_case(
            monkeypatch,
            generate = _generate,
            payload_kwargs = {"stream": True},
            backend_kwargs = {"reasoning_always_on": False},
        )
        deltas = [p["choices"][0].get("delta", {}) for p in result.payloads if p.get("choices")]

        assert "".join(d.get("reasoning_content", "") for d in deltas) == "plan"
        assert "".join(d.get("content", "") for d in deltas) == "visible"
        [entry] = result.monitor.snapshot()
        assert entry["reply"] == "visible"

    def test_reasoning_capable_gguf_stream_sanitizes_think_tags_when_disabled(self, monkeypatch):
        def _generate(**_kwargs):
            yield "<think>leaked</think>visible"
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "finish_reason": "stop",
            }

        result = self._run_gguf_case(
            monkeypatch,
            generate = _generate,
            payload_kwargs = {"stream": True, "enable_thinking": False},
            backend_kwargs = {"reasoning_always_on": False},
        )
        deltas = [p["choices"][0].get("delta", {}) for p in result.payloads if p.get("choices")]

        assert "".join(d.get("reasoning_content", "") for d in deltas) == "leaked"
        assert "".join(d.get("content", "") for d in deltas) == "visible"
        assert all("<think>" not in d.get("content", "") for d in deltas)
        [entry] = result.monitor.snapshot()
        assert entry["reply"] == "visible"

    def test_gguf_tool_stream_splits_reasoning_and_strips_gemma_tool_marker(self, monkeypatch):
        def _tools(**_kwargs):
            yield {
                "type": "content",
                "text": '<think>plan</think>visible <|tool_call>call:terminal{command:"ls"}<tool_call|>',
            }
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "finish_reason": "stop",
            }

        result = self._run_gguf_case(
            monkeypatch,
            tool_generate = _tools,
            payload_kwargs = {
                "stream": True,
                "enable_tools": True,
                "enabled_tools": ["terminal"],
                "messages": [{"role": "user", "content": "list files"}],
            },
        )
        deltas = [p["choices"][0].get("delta", {}) for p in result.payloads if p.get("choices")]

        assert "".join(d.get("reasoning_content", "") for d in deltas) == "plan"
        combined_content = "".join(d.get("content", "") for d in deltas)
        assert combined_content == "visible "
        assert "<|tool_call>" not in combined_content
        [entry] = result.monitor.snapshot()
        assert entry["reply"] == "visible "

    def test_gguf_tool_stream_flushes_held_text_before_status_reset(self, monkeypatch):
        def _tools(**_kwargs):
            yield {"type": "content", "text": "answer <"}
            yield {"type": "status", "text": ""}
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "finish_reason": "stop",
            }

        result = self._run_gguf_case(
            monkeypatch,
            tool_generate = _tools,
            payload_kwargs = {
                "stream": True,
                "enable_tools": True,
                "enabled_tools": ["terminal"],
                "messages": [{"role": "user", "content": "say literal"}],
            },
        )
        deltas = [p["choices"][0].get("delta", {}) for p in result.payloads if p.get("choices")]

        combined_content = "".join(d.get("content", "") for d in deltas)
        assert combined_content == "answer <"
        [entry] = result.monitor.snapshot()
        assert entry["reply"] == "answer <"

    def test_non_streaming_gguf_splits_reasoning_content(self, monkeypatch):
        def _generate(**_kwargs):
            yield "<think>plan</think>visible"
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                "finish_reason": "stop",
            }

        result = self._run_gguf_case(monkeypatch, generate = _generate)
        body = result.body
        message = body["choices"][0]["message"]

        assert message["content"] == "visible"
        assert message["reasoning_content"] == "plan"
        [entry] = result.monitor.snapshot()
        assert entry["reply"] == "visible"

    def test_standard_gguf_non_streaming_admission_timeout_before_generation(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request(self._Request):
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))

            def _generate(**_kwargs):
                raise AssertionError("standard GGUF generation must not start while queued")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = False,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.standard.test",
                effective_parallel_slots = 1,
                generate_chat_completion = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setenv(ADMISSION_QUEUE_TIMEOUT_ENV, "0.01")
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

            queue = get_llama_admission_queue("http://llama.standard.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
            )
            try:
                with pytest.raises(HTTPException) as exc:
                    await openai_chat_completions(
                        payload,
                        request = Request(),
                        current_subject = "test",
                    )
                assert exc.value.status_code == 503
            finally:
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0

        asyncio.run(_run())

    def test_standard_gguf_non_streaming_cancel_id_stops_queued_request_before_generation(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            class Request(self._Request):
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))

            def _generate(**_kwargs):
                raise AssertionError("standard GGUF generation must not start after cancel_id")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = False,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.standard.test",
                effective_parallel_slots = 1,
                generate_chat_completion = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

            queue = get_llama_admission_queue("http://llama.standard.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            cancel_id = "standard-nonstream-admission-cancel"
            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                cancel_id = cancel_id,
            )
            task = asyncio.create_task(
                openai_chat_completions(
                    payload,
                    request = Request(),
                    current_subject = "test",
                )
            )
            try:
                for _ in range(50):
                    if cancel_id in inf_mod._CANCEL_REGISTRY:
                        break
                    await asyncio.sleep(0.01)
                assert cancel_id in inf_mod._CANCEL_REGISTRY
                assert inf_mod._cancel_by_cancel_id_or_stash(cancel_id) == 1
                with pytest.raises(HTTPException) as exc:
                    await asyncio.wait_for(task, timeout = 0.5)
                assert exc.value.status_code == 499
            finally:
                if not task.done():
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                blocker.release()

            assert cancel_id not in inf_mod._CANCEL_REGISTRY
            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_standard_gguf_non_streaming_admission_task_cancel_cleans_tracker_and_slot(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            cancel_id = "standard-nonstream-task-cancel"

            async def fake_wait(*_args, **_kwargs):
                raise asyncio.CancelledError()

            def _generate(**_kwargs):
                raise AssertionError("standard GGUF generation must not start after task cancel")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = False,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.standard.test",
                effective_parallel_slots = 1,
                generate_chat_completion = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(
                inf_mod,
                "_wait_for_openai_admission_non_streaming",
                fake_wait,
            )

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                cancel_id = cancel_id,
            )
            with pytest.raises(asyncio.CancelledError):
                await openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )

            assert cancel_id not in inf_mod._CANCEL_REGISTRY
            assert get_llama_admission_queue("http://llama.standard.test").snapshot().active == 0

        asyncio.run(_run())

    def test_gguf_tool_non_streaming_admission_timeout_before_generation(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request(self._Request):
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))

            async def fake_select_tools(*_args, **_kwargs):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]

            def _generate(**_kwargs):
                raise AssertionError("GGUF tool loop must not start while queued")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.tool.test",
                effective_parallel_slots = 1,
                generate_chat_completion = lambda **_kwargs: "unused",
                generate_chat_completion_with_tools = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setenv(ADMISSION_QUEUE_TIMEOUT_ENV, "0.01")
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(inf_mod, "_select_request_tools", fake_select_tools)

            queue = get_llama_admission_queue("http://llama.tool.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                enable_tools = True,
            )
            try:
                with pytest.raises(HTTPException) as exc:
                    await openai_chat_completions(
                        payload,
                        request = Request(),
                        current_subject = "test",
                    )
                assert exc.value.status_code == 503
            finally:
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0

        asyncio.run(_run())

    def test_gguf_tool_non_streaming_cancel_drains_worker_before_releasing_slot(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_select_tools(*_args, **_kwargs):
                return [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ]

            started = threading.Event()
            released = threading.Event()

            def _tools(**kwargs):
                cancel_event = kwargs["cancel_event"]
                started.set()
                while not cancel_event.is_set():
                    time.sleep(0.005)
                released.set()
                yield from ()

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = True,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                base_url = "http://llama.tool.test",
                effective_parallel_slots = 1,
                generate_chat_completion = lambda **_kwargs: "unused",
                generate_chat_completion_with_tools = _tools,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(inf_mod, "_select_request_tools", fake_select_tools)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
                enable_tools = True,
            )
            task = asyncio.create_task(
                openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )
            )
            # Generous budgets. What this test asserts is that cancelling the
            # request drains the worker, and none of the numbers below are part
            # of that: they only bound how long to wait before calling it hung.
            # A one-second bound on a THREAD START is a bound on the scheduler,
            # not on this code, and it went red once on a runner busy with the
            # rest of the backend suite. Failing here still takes seconds, and
            # the assertion is unchanged.
            assert await asyncio.to_thread(started.wait, self._DRAIN_BUDGET_S)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout = self._DRAIN_BUDGET_S)

            # Waited on, not sampled. Cancelling the task unblocks the awaiting
            # coroutine; it does not join the worker, which is off polling
            # cancel_event every 5ms and only then sets this. Reading it the
            # instant the await returns is a race that happens to be won on an
            # idle box, and it is the drain itself that matters, not whether it
            # had already finished by the time we looked.
            assert await asyncio.to_thread(released.wait, self._DRAIN_BUDGET_S)
            assert get_llama_admission_queue("http://llama.tool.test").snapshot().active == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_non_streaming_gguf_n_records_all_monitor_replies(self, monkeypatch):
        import routes.inference as inf_mod

        calls = {"count": 0}

        def _generate(**kwargs):
            calls["count"] += 1
            callback = kwargs.get("perf_callback")
            if callback is not None:
                callback(
                    {
                        "prompt_ms": 100.0,
                        "prompt_per_second": 30.0,
                        "predicted_ms": 20.0,
                        "predicted_per_second": 50.0,
                    }
                )
            text = f"reply {calls['count']}"
            yield text
            yield {
                "type": "metadata",
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": calls["count"],
                    "total_tokens": 3 + calls["count"],
                },
            }

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = False,
            _is_audio = False,
            model_identifier = "test-gguf",
            context_length = 4096,
            generate_chat_completion = _generate,
        )
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            n = 2,
            messages = [{"role": "user", "content": "two please"}],
        )

        response = self._drive(
            openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
        )
        body = json.loads(response.body)

        assert [c["message"]["content"] for c in body["choices"]] == ["reply 1", "reply 2"]
        [entry] = monitor.snapshot()
        assert entry["reply"] == "Choice 1:\nreply 1\n\nChoice 2:\nreply 2"
        assert entry["completion_tokens"] == 3
        assert monitor.active_count() == 0

        assert entry["prompt_tok_per_sec"] is None
        assert entry["tok_per_sec"] is None
        assert entry["decode_ms"] is None

    def test_disabled_monitor_does_not_install_gguf_perf_callback(self, monkeypatch):
        import routes.inference as inf_mod

        captured = {}

        def _generate(**kwargs):
            captured.update(kwargs)
            yield "reply"
            yield {
                "type": "metadata",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = False,
            _is_audio = False,
            model_identifier = "test-gguf",
            context_length = 4096,
            generate_chat_completion = _generate,
        )
        monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3, enabled = False))
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        response = self._drive(
            openai_chat_completions(
                ChatCompletionRequest(
                    model = "default",
                    messages = [{"role": "user", "content": "hi"}],
                ),
                request = self._Request(),
                current_subject = "test",
            )
        )

        assert json.loads(response.body)["choices"][0]["message"]["content"] == "reply"
        assert captured["perf_callback"] is None

    # Wall-clock bound for the cancel drain. Only ever hit when something is
    # genuinely stuck, so it is sized for a loaded runner rather than for the
    # ~5ms this takes when it works.
    _DRAIN_BUDGET_S = 30.0

    def test_non_streaming_gguf_cancel_drains_worker(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            started = threading.Event()
            released = threading.Event()

            def _generate(**kwargs):
                cancel_event = kwargs["cancel_event"]
                started.set()
                while not cancel_event.is_set():
                    time.sleep(0.005)
                released.set()
                yield from ()

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                supports_tools = False,
                _is_audio = False,
                model_identifier = "test-gguf",
                context_length = 4096,
                generate_chat_completion = _generate,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [{"role": "user", "content": "hi"}],
            )
            task = asyncio.create_task(
                openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )
            )
            assert await asyncio.to_thread(started.wait, 1.0)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout = 1.0)

            assert released.is_set()
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def _drive_standard_gguf(self, monkeypatch, date_line: str) -> list[dict]:
        """Run one non-tool GGUF completion with the current-date setting pinned."""
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
            context_length = 4096,
            generate_chat_completion = _generate,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
        # Pinned, not left to the host's stored setting, so the assertion is the same everywhere.
        monkeypatch.setattr(inf_mod, "current_date_prompt_line", lambda **_kwargs: date_line)

        payload = ChatCompletionRequest(
            model = "default",
            messages = [
                {"role": "system", "content": "original system"},
                {"role": "developer", "content": "developer rules"},
                {"role": "user", "content": "hi"},
            ],
        )

        self._drive(
            openai_chat_completions(payload, request = self._Request(), current_subject = "test")
        )
        return captured["messages"]

    def test_standard_gguf_merges_system_and_developer_messages(self, monkeypatch):
        assert self._drive_standard_gguf(monkeypatch, "") == [
            {"role": "system", "content": "original system\n\ndeveloper rules"},
            {"role": "user", "content": "hi"},
        ]

    def test_standard_gguf_prefixes_the_current_date(self, monkeypatch):
        # Covers the wiring, not just the helper: a tool-less GGUF chat must carry the date.
        assert self._drive_standard_gguf(monkeypatch, "The current date is 2026-08-15.") == [
            {
                "role": "system",
                "content": "The current date is 2026-08-15.\n\noriginal system\n\ndeveloper rules",
            },
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
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)

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
            context_length = 4096,
            generate_chat_completion = _generate,
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        payload = ChatCompletionRequest(
            model = "default",
            messages = [{"role": "user", "content": "hi"}],
            n = 3,
            seed = seed,
        )

        response = self._drive(
            openai_chat_completions(payload, request = self._Request(), current_subject = "test")
        )
        body = json.loads(response.body)

        assert seen_seeds == expected
        assert [choice["index"] for choice in body["choices"]] == [0, 1, 2]
        assert body["usage"]["prompt_tokens"] == 5
        assert body["usage"]["completion_tokens"] == 21
        [entry] = monitor.snapshot()
        assert entry["prompt_tokens"] == 5
        assert entry["completion_tokens"] == 21
        assert entry["total_tokens"] == 26


class TestApiMonitorProviderAndCompletionStreams:
    class _Request:
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/chat/completions")
        method = "POST"

        async def is_disconnected(self):
            return False

    async def _run_passthrough_stream(
        self,
        monkeypatch,
        lines,
        stream_options = None,
    ):
        import routes.inference as inf_mod

        class Request:
            async def is_disconnected(self):
                return False

        upstream_bodies = []

        async def fake_send(_client, built_request, *_args, **_kwargs):
            upstream_bodies.append(json.loads(built_request.content))
            return httpx.Response(200, content = b"")

        async def fake_items(*_args, **_kwargs):
            for line in lines:
                yield line

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
        monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        payload = ChatCompletionRequest(
            model = "default",
            messages = [ChatMessage(role = "user", content = "hi")],
            stream = True,
            stream_options = stream_options,
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        response = await _openai_passthrough_stream(
            Request(),
            threading.Event(),
            SimpleNamespace(
                base_url = "http://llama.test",
                context_length = 4096,
                _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            ),
            payload,
            "gguf",
            "chatcmpl-test",
            monitor_id = monitor_id,
        )
        chunks = [chunk async for chunk in response.body_iterator]
        return SimpleNamespace(
            chunks = chunks,
            body = "".join(chunks),
            monitor = monitor,
            upstream_bodies = upstream_bodies,
        )

    @staticmethod
    def _captured_tool_call_frames():
        def _frame(delta, finish_reason = None):
            return "data: " + json.dumps(
                {
                    "id": "chatcmpl-captured",
                    "choices": [
                        {
                            "index": 0,
                            "delta": delta,
                            "finish_reason": finish_reason,
                        }
                    ],
                },
                separators = (",", ":"),
            )

        argument_fragments = [
            "{",
            '"command":"',
            "echo",
            " HER",
            "MES",
            "_UNS",
            "LO",
            "TH",
            "_OK",
            '"',
            "}",
        ]
        frames = [_frame({"role": "assistant", "content": None})]
        frames.append(
            _frame(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call-captured",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": argument_fragments[0],
                            },
                        }
                    ]
                }
            )
        )
        frames.extend(
            _frame(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "function": {"arguments": fragment},
                        }
                    ]
                }
            )
            for fragment in argument_fragments[1:]
        )
        frames.extend([_frame({}, "tool_calls"), "data: [DONE]"])
        assert len(frames) == 14
        return frames

    def test_passthrough_stream_preheader_dispatched_with_timeout(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()

            async def fake_send(*_args, **_kwargs):
                await gate.wait()
                return httpx.Response(200, content = b"")

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )

            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )
            assert isinstance(response, _SameTaskStreamingResponse)

            gate.set()
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
            assert "data: [DONE]\n\n" in "".join(chunks)

        asyncio.run(_run())

    def test_passthrough_stream_forwards_backend_auth_headers(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            captured_headers = {}

            async def fake_send(_client, req, *_args, **_kwargs):
                captured_headers.update(dict(req.headers))
                return httpx.Response(200, content = b"")

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )
            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    _auth_headers = {"Authorization": "Bearer secret"},
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "chatcmpl-test",
                "chatcmpl-test",
                monitor_id = monitor_id,
            )
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]

            assert "data: [DONE]\n\n" in "".join(chunks)
            assert captured_headers["authorization"] == "Bearer secret"
            assert captured_headers["connection"] == "close"

        asyncio.run(_run())

    def test_passthrough_stream_keepalive_while_upstream_headers_are_pending(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()

            async def fake_send(*_args, **_kwargs):
                await gate.wait()
                return httpx.Response(200, content = b"")

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(
                inf_mod,
                "_OPENAI_PASSTHROUGH_PENDING_RESPONSE_KEEPALIVE_S",
                0.01,
            )

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )

            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )

            first = await asyncio.wait_for(response.body_iterator.__anext__(), timeout = 5.0)
            assert first == ": keep-alive\n\n"

            gate.set()
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
            body = "".join(chunks)
            assert "data: [DONE]\n\n" in body

        asyncio.run(_run())

    def test_passthrough_stream_preheader_non_200_in_window(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_send(*_args, **_kwargs):
                return httpx.Response(400, content = b'{"error":"bad"}')

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )
            with pytest.raises(HTTPException) as exc:
                await _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                )
            assert exc.value.status_code == 400

        asyncio.run(_run())

    def test_passthrough_stream_preheader_request_error_in_window(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_send(*_args, **_kwargs):
                raise httpx.ConnectError("connectivity issue")

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )
            with pytest.raises(HTTPException) as exc:
                await _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                )
            assert exc.value.status_code == 502

        asyncio.run(_run())

    def test_passthrough_stream_preheader_delayed_non_200_returns_sse_error(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()

            async def fake_send(*_args, **_kwargs):
                await gate.wait()
                return httpx.Response(400, content = b'{"error":"bad"}')

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )
            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )
            assert isinstance(response, _SameTaskStreamingResponse)
            gate.set()
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
            body = "".join(chunks)
            assert "data:" in body
            assert '"error"' in body
            assert "data: [DONE]" in body
            [entry] = monitor.snapshot()
            assert entry["status"] == "error"
            assert "bad" in entry["error"]

        asyncio.run(_run())

    def test_passthrough_stream_preheader_delayed_context_error_keeps_error_envelope(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()
            ctx_msg = "request (4096 tokens) exceeds the available context size (2048 tokens)"

            async def fake_send(*_args, **_kwargs):
                await gate.wait()
                return httpx.Response(400, content = ctx_msg.encode("utf-8"))

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )
            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 2048,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )
            assert isinstance(response, _SameTaskStreamingResponse)

            gate.set()
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
            body = "".join(chunks)
            events = [
                line.removeprefix("data: ")
                for line in body.splitlines()
                if line.startswith("data: ")
            ]
            assert events[-1] == "[DONE]"
            payload = json.loads(events[0])
            assert payload["error"]["code"] == "context_length_exceeded"
            assert payload["error"]["param"] == "messages"
            assert isinstance(payload["error"], dict)

        asyncio.run(_run())

    def test_passthrough_stream_preheader_delayed_context_error_retries_truncation(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()
            calls = []
            err_body = json.dumps(
                {
                    "error": {
                        "message": "request (10000 tokens) exceeds the available context size (2048 tokens)",
                        "n_prompt_tokens": 10000,
                        "n_ctx": 2048,
                    }
                }
            ).encode("utf-8")

            async def fake_send(_client, req, *_args, **_kwargs):
                calls.append(json.loads(req.content.decode("utf-8")))
                if len(calls) == 1:
                    await gate.wait()
                    return httpx.Response(400, content = err_body)
                return httpx.Response(200, content = b"")

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            messages = [
                ChatMessage(role = "system", content = "system"),
                *[
                    ChatMessage(role = "user", content = f"turn {idx} " + ("x" * 1000))
                    for idx in range(8)
                ],
            ]
            payload = ChatCompletionRequest(
                model = "default",
                messages = messages,
                stream = True,
                context_overflow = "truncate_middle",
            )
            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 2048,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )
            assert isinstance(response, _SameTaskStreamingResponse)

            gate.set()
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
            assert "data: [DONE]\n\n" in "".join(chunks)
            assert len(calls) == 2
            assert len(calls[1]["messages"]) < len(calls[0]["messages"])
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"

        asyncio.run(_run())

    def test_passthrough_stream_preheader_immediate_context_retry_adopts_delayed_response(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()
            calls = []
            err_body = json.dumps(
                {
                    "error": {
                        "message": "request (10000 tokens) exceeds the available context size (2048 tokens)",
                        "n_prompt_tokens": 10000,
                        "n_ctx": 2048,
                    }
                }
            ).encode("utf-8")
            ok_lines = [
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,'
                '"model":"gguf","choices":[{"index":0,"delta":{"content":"OK"},'
                '"finish_reason":null}]}',
                'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,'
                '"model":"gguf","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]

            async def fake_send(_client, req, *_args, **_kwargs):
                calls.append(json.loads(req.content.decode("utf-8")))
                if len(calls) == 1:
                    return httpx.Response(400, content = err_body)
                await gate.wait()
                return httpx.Response(200, content = b"")

            async def fake_items(*_args, **_kwargs):
                for line in ok_lines:
                    yield line

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)

            messages = [
                ChatMessage(role = "system", content = "system"),
                *[
                    ChatMessage(role = "user", content = f"turn {idx} " + ("x" * 1000))
                    for idx in range(8)
                ],
            ]
            payload = ChatCompletionRequest(
                model = "default",
                messages = messages,
                stream = True,
                context_overflow = "truncate_middle",
            )
            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 2048,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )
            assert isinstance(response, _SameTaskStreamingResponse)

            gate.set()
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
            body = "".join(chunks)

            assert "OK" in body
            assert "context_length_exceeded" not in body
            assert len(calls) == 2
            assert len(calls[1]["messages"]) < len(calls[0]["messages"])
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"

        asyncio.run(_run())

    def test_passthrough_stream_preheader_delayed_request_error_cleans_up(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()
            cancel_id = "delayed-request-error-cancel"

            async def fake_send(*_args, **_kwargs):
                await gate.wait()
                raise httpx.ConnectError("delayed connectivity issue")

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                cancel_id = cancel_id,
            )
            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )
            assert isinstance(response, _SameTaskStreamingResponse)
            assert cancel_id in inf_mod._CANCEL_REGISTRY

            gate.set()
            chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in response.body_iterator
            ]
            body = "".join(chunks)
            assert "data:" in body
            assert '"error"' in body
            [entry] = monitor.snapshot()
            assert entry["status"] == "error"
            assert "Lost connection" in entry["error"]
            assert cancel_id not in inf_mod._CANCEL_REGISTRY

        asyncio.run(_run())

    def test_passthrough_stream_preheader_cancel_cleans_pending_send(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            entered = asyncio.Event()
            cancelled = asyncio.Event()
            cancel_id = "preheader-cancel-cleanup"

            async def fake_send(*_args, **_kwargs):
                entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                cancel_id = cancel_id,
            )
            task = asyncio.create_task(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                )
            )
            await asyncio.wait_for(entered.wait(), timeout = 5.0)
            assert cancel_id in inf_mod._CANCEL_REGISTRY

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(cancelled.wait(), timeout = 5.0)
            assert cancel_id not in inf_mod._CANCEL_REGISTRY

        asyncio.run(_run())

    def test_passthrough_stream_unstarted_cleanup_closes_completed_send_response(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            gate = asyncio.Event()
            returned = asyncio.Event()
            cancel_id = "unstarted-completed-send-cleanup"

            class Stream(httpx.AsyncByteStream):
                async def __aiter__(self):
                    if False:
                        yield b""

            stream = Stream()
            upstream_response = httpx.Response(200, stream = stream)

            async def fake_send(*_args, **_kwargs):
                await gate.wait()
                returned.set()
                return upstream_response

            class Request:
                async def is_disconnected(self):
                    return False

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                cancel_id = cancel_id,
            )
            response = await asyncio.wait_for(
                _openai_passthrough_stream(
                    Request(),
                    threading.Event(),
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "chatcmpl-test",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                ),
                timeout = 5.0,
            )
            assert isinstance(response, _SameTaskStreamingResponse)
            assert cancel_id in inf_mod._CANCEL_REGISTRY

            gate.set()
            await asyncio.wait_for(returned.wait(), timeout = 5.0)
            await asyncio.sleep(0)
            await response._unstarted_cleanup()
            assert upstream_response.is_closed
            assert cancel_id not in inf_mod._CANCEL_REGISTRY

        asyncio.run(_run())

    def test_external_non_streaming_json_updates_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class DummyExternalClient:
                def __init__(self, **_kwargs):
                    pass

                async def stream_chat_completion(self, **kwargs):
                    assert kwargs["stream"] is False
                    yield json.dumps(
                        {
                            "choices": [{"message": {"content": "provider [DONE] reply"}}],
                            "usage": {
                                "prompt_tokens": 3,
                                "completion_tokens": 4,
                                "total_tokens": 7,
                            },
                        }
                    )

                async def close(self):
                    pass

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "ExternalProviderClient", DummyExternalClient)
            payload = ChatCompletionRequest(
                model = "default",
                external_model = "gpt-test",
                provider_type = "openai",
                provider_base_url = "https://api.openai.com/v1",
                messages = [ChatMessage(role = "user", content = "hi")],
            )

            response = await _proxy_to_external_provider(payload, self._Request())
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            assert chunks[-1] == "data: [DONE]\n\n"
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["reply"] == "provider [DONE] reply"
            assert entry["prompt_tokens"] == 3
            assert entry["completion_tokens"] == 4
            assert entry["total_tokens"] == 7

        asyncio.run(_run())

    def test_external_stream_clean_eof_flushes_pending_tool_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            lines = [
                'data: {"choices":[{"index":0,"delta":{"tool_calls":['
                '{"index":0,"id":"call-eof","function":{"name":"lookup",'
                '"arguments":"{\\"query\\":"}}]}}]}',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":['
                '{"index":0,"function":{"arguments":"\\"weather\\"}"}}]}}]}',
            ]

            class DummyExternalClient:
                def __init__(self, **_kwargs):
                    pass

                async def stream_chat_completion(self, **_kwargs):
                    for line in lines:
                        yield line

                async def close(self):
                    pass

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "ExternalProviderClient", DummyExternalClient)
            payload = ChatCompletionRequest(
                model = "default",
                external_model = "gpt-test",
                provider_type = "openai",
                provider_base_url = "https://api.openai.com/v1",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )

            response = await _proxy_to_external_provider(payload, self._Request())
            chunks = [chunk async for chunk in response.body_iterator]

            assert chunks == [*(f"{line}\n\n" for line in lines), "data: [DONE]\n\n"]
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["reply"] == 'Tool call: lookup({"query":"weather"})'

        asyncio.run(_run())

    def test_external_stream_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class DummyExternalClient:
                def __init__(self, **_kwargs):
                    pass

                async def stream_chat_completion(self, **_kwargs):
                    yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
                    await asyncio.sleep(3600)

                async def close(self):
                    pass

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "ExternalProviderClient", DummyExternalClient)
            payload = ChatCompletionRequest(
                model = "default",
                external_model = "gpt-test",
                provider_type = "openai",
                provider_base_url = "https://api.openai.com/v1",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )

            response = await _proxy_to_external_provider(payload, self._Request())
            iterator = response.body_iterator
            first = await anext(iterator)
            assert "hello" in first

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

    def test_completions_preheader_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/completions")
                method = "POST"

                async def json(self):
                    return {"prompt": "hi", "stream": True}

                async def is_disconnected(self):
                    return False

            async def fake_send(*_args, **_kwargs):
                return None

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)

            response = await openai_completions(Request(), current_subject = "test")
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            assert chunks == []
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_completions_stream_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/completions")
                method = "POST"

                async def json(self):
                    return {"prompt": "hi", "stream": True}

                async def is_disconnected(self):
                    return False

            async def fake_send(*_args, **_kwargs):
                return httpx.Response(200, content = b"")

            async def fake_items(*_args, **_kwargs):
                yield b'data: {"choices":[{"text":"hello"}]}\n\n'
                await asyncio.sleep(3600)

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)

            response = await openai_completions(Request(), current_subject = "test")
            iterator = response.body_iterator
            first = await anext(iterator)
            assert b"hello" in first

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

    def test_completions_stream_requests_usage_only_for_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/completions")
                method = "POST"

                async def json(self):
                    return {"prompt": "hi", "stream": True}

                async def is_disconnected(self):
                    return False

            upstream_bodies = []

            async def fake_send(_client, built_request, *_args, **_kwargs):
                upstream_bodies.append(json.loads(built_request.content))
                return httpx.Response(200, content = b"")

            async def fake_items(*_args, **_kwargs):
                yield (
                    b'data: {"choices":[{"text":"ok","finish_reason":"stop"}]}\n\n'
                    b'data: {"choices":[],"usage":{"prompt_tokens":3,'
                    b'"completion_tokens":2,"total_tokens":5}}\n\n'
                    b"data: [DONE]\n\n"
                )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)

            response = await openai_completions(Request(), current_subject = "test")
            body = b"".join([chunk async for chunk in response.body_iterator])

            assert upstream_bodies[0]["stream_options"]["include_usage"] is True
            assert b'"usage"' not in body
            [entry] = monitor.snapshot()
            assert entry["prompt_tokens"] == 3
            assert entry["completion_tokens"] == 2
            assert entry["total_tokens"] == 5

        asyncio.run(_run())

    def test_completions_non_streaming_post_error_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/completions")
                method = "POST"

                async def json(self):
                    return {"prompt": "hi", "stream": False}

                async def is_disconnected(self):
                    return False

            class FailingAsyncClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return False

                async def aclose(self):
                    return None

                async def post(self, *_args, **_kwargs):
                    raise httpx.ConnectError("llama down")

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            # Per-request client so a forced swap can close it mid-call; the pooled one is shared.
            monkeypatch.setattr(
                inf_mod,
                "_cancelable_nonstreaming_client",
                lambda: FailingAsyncClient(),
            )
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )

            with pytest.raises(httpx.ConnectError):
                await openai_completions(Request(), current_subject = "test")

            [entry] = monitor.snapshot()
            assert entry["status"] == "error"
            assert "Lost connection to the model server" in entry["error"]
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_completions_omitted_max_tokens_falls_back_to_context(self, monkeypatch):
        # With no env knobs set, an omitted max_tokens must forward the
        # backend's context length, exactly as on main.
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/completions")
                method = "POST"

                async def json(self):
                    return {"prompt": "hi", "stream": False}

                async def is_disconnected(self):
                    return False

            captured = []

            class CapturingClient:
                async def aclose(self):
                    return None

                async def post(self, _url, *, json, **_kwargs):
                    captured.append(dict(json))
                    return httpx.Response(
                        200,
                        json = {
                            "id": "cmpl-test",
                            "choices": [{"text": "ok"}],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod, "_cancelable_nonstreaming_client", lambda: CapturingClient()
            )
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )

            await openai_completions(Request(), current_subject = "test")

            assert captured[0]["max_tokens"] == 4096
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_completions_forwards_spec_valid_zero_max_tokens(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/completions")
                method = "POST"

                async def json(self):
                    return {"prompt": "hi", "stream": False, "max_tokens": 0}

                async def is_disconnected(self):
                    return False

            captured = []

            class CapturingClient:
                async def aclose(self):
                    return None

                async def post(self, _url, *, json, **_kwargs):
                    captured.append(dict(json))
                    return httpx.Response(
                        200,
                        json = {
                            "id": "cmpl-test",
                            "choices": [{"text": "", "finish_reason": "length"}],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 0,
                                "total_tokens": 1,
                            },
                        },
                    )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod, "_cancelable_nonstreaming_client", lambda: CapturingClient()
            )
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )

            await openai_completions(Request(), current_subject = "test")

            assert captured[0]["max_tokens"] == 0
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_completions_rejects_non_integer_max_tokens_before_forwarding(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/completions")
                method = "POST"

                async def json(self):
                    return {"prompt": "hi", "stream": False, "max_tokens": "128"}

            class UnusedClient:
                async def post(self, *_args, **_kwargs):
                    raise AssertionError("invalid max_tokens must not reach llama-server")

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "nonstreaming_client", lambda: UnusedClient())
            monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: UnusedClient())
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )

            with pytest.raises(HTTPException) as exc:
                await openai_completions(Request(), current_subject = "test")

            assert exc.value.status_code == 400
            assert exc.value.detail["error"]["param"] == "max_tokens"
            assert exc.value.detail["error"]["code"] == "invalid_type"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_monitor_openai_chunk_records_all_choice_replies(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )

        _monitor_openai_chunk(
            monitor_id,
            {
                "choices": [
                    {"text": "first"},
                    {"text": "second"},
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 5,
                    "total_tokens": 7,
                },
            },
            4096,
        )

        entry = monitor.get(monitor_id)
        assert entry["reply"] == "Choice 1:\nfirst\n\nChoice 2:\nsecond"
        assert entry["prompt_tokens"] == 2
        assert entry["completion_tokens"] == 5
        assert entry["context_length"] == 4096

    def test_monitor_openai_chunk_records_tool_call_reply(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )

        _monitor_openai_chunk(
            monitor_id,
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":"weather"}',
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            4096,
        )

        entry = monitor.get(monitor_id)
        assert entry["reply"] == 'Tool call: lookup({"query":"weather"})'

    def test_streamed_tool_monitor_reassembles_captured_frames_without_changing_sse(
        self, monkeypatch
    ):
        async def _run():
            frames = self._captured_tool_call_frames()
            result = await self._run_passthrough_stream(monkeypatch, frames)

            assert result.body == "".join(f"{frame}\n\n" for frame in frames)
            [entry] = result.monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["stop_reason"] == "tool_calls"
            assert entry["reply"] == ('Tool call: terminal({"command":"echo HERMES_UNSLOTH_OK"})')
            assert entry["ttft_ms"] is not None

        asyncio.run(_run())

    def test_streamed_tool_monitor_joins_fragmented_name_and_arguments(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        for tool_call in [
            {
                "index": 0,
                "id": "call-fragmented",
                "function": {"name": "ter", "arguments": '{"com'},
            },
            {"index": 0, "function": {"name": "min", "arguments": 'mand":'}},
            {"index": 0, "function": {"name": "al", "arguments": '"echo"}'}},
        ]:
            _monitor_openai_chunk(
                monitor_id,
                {"choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}}]},
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == 'Tool call: terminal({"command":"echo"})'

    def test_streamed_tool_monitor_replaces_cumulative_names(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        for name, arguments in [("web", None), ("web_search", "{}")]:
            _monitor_openai_chunk(
                monitor_id,
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"name": name, "arguments": arguments},
                                    }
                                ]
                            },
                        }
                    ]
                },
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == "Tool call: web_search({})"

    def test_streamed_tool_monitor_serializes_structured_arguments(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        _monitor_openai_chunk(
            monitor_id,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "lookup",
                                        "arguments": {"query": "weather", "active": True},
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            streaming = True,
        )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == (
            'Tool call: lookup({"query": "weather", "active": true})'
        )

    def test_streamed_legacy_function_monitor_joins_name_and_arguments(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        for function_call in [
            {"name": "look", "arguments": '{"query":'},
            {"name": "up", "arguments": '"weather"}'},
        ]:
            _monitor_openai_chunk(
                monitor_id,
                {"choices": [{"index": 0, "delta": {"function_call": function_call}}]},
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == 'Tool call: lookup({"query":"weather"})'

    def test_streamed_tool_monitor_keeps_interleaved_indexes_separate(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        deltas = [
            {"index": 0, "id": "call-a", "function": {"name": "alpha", "arguments": '{"a":'}},
            {"index": 1, "id": "call-b", "function": {"name": "beta", "arguments": '{"b":'}},
            {"index": 0, "function": {"arguments": "1}"}},
            {"index": 1, "function": {"arguments": "2}"}},
        ]
        for tool_call in deltas:
            _monitor_openai_chunk(
                monitor_id,
                {"choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}}]},
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == (
            'Tool call: alpha({"a":1})\nTool call: beta({"b":2})'
        )

    def test_streamed_tool_monitor_routes_bare_fragments_to_latest_call(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        deltas = [
            {"index": 0, "function": {"name": "alpha", "arguments": '{"a":'}},
            {"index": 1, "function": {"name": "beta", "arguments": '{"b":'}},
            {"function": {"arguments": "2}"}},
            {"index": 0, "function": {"arguments": "1"}},
            {"function": {"arguments": "}"}},
        ]
        for tool_call in deltas:
            _monitor_openai_chunk(
                monitor_id,
                {"choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}}]},
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == (
            'Tool call: alpha({"a":1})\nTool call: beta({"b":2})'
        )

    def test_streamed_tool_monitor_keeps_choice_indexes_separate(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        chunks = [
            (0, {"index": 0, "id": "shared-id", "function": {"name": "alpha", "arguments": "{"}}),
            (1, {"index": 0, "id": "shared-id", "function": {"name": "beta", "arguments": "{"}}),
            (0, {"index": 0, "function": {"arguments": '"a":1}'}}),
            (1, {"index": 0, "function": {"arguments": '"b":2}'}}),
        ]
        for choice_index, tool_call in chunks:
            _monitor_openai_chunk(
                monitor_id,
                {
                    "choices": [
                        {
                            "index": choice_index,
                            "delta": {"tool_calls": [tool_call]},
                        }
                    ]
                },
                streaming = True,
            )
        for choice_index in (0, 1):
            _monitor_openai_chunk(
                monitor_id,
                {
                    "choices": [
                        {
                            "index": choice_index,
                            "delta": {},
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                streaming = True,
            )

        assert monitor.get(monitor_id)["reply"] == (
            'Tool call: alpha({"a":1})\nTool call: beta({"b":2})'
        )

    def test_streamed_tool_monitor_uses_ids_when_indexes_are_reused_or_omitted(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        deltas = [
            {"index": 0, "id": "call-a", "function": {"name": "alpha", "arguments": '{"a":'}},
            {"index": 0, "id": "call-b", "function": {"name": "beta", "arguments": '{"b":'}},
            {"index": 0, "function": {"arguments": "2}"}},
            {"id": "call-a", "function": {"arguments": "1}"}},
        ]
        for tool_call in deltas:
            _monitor_openai_chunk(
                monitor_id,
                {"choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}}]},
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == (
            'Tool call: alpha({"a":1})\nTool call: beta({"b":2})'
        )

    def test_streamed_tool_monitor_separates_duplicate_ids_by_explicit_index(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        for tool_call in [
            {
                "index": 0,
                "id": "duplicate",
                "function": {"name": "alpha", "arguments": '{"a":1}'},
            },
            {
                "index": 1,
                "id": "duplicate",
                "function": {"name": "beta", "arguments": '{"b":2}'},
            },
        ]:
            _monitor_openai_chunk(
                monitor_id,
                {"choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}}]},
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == (
            'Tool call: alpha({"a":1})\nTool call: beta({"b":2})'
        )

    @pytest.mark.parametrize("terminal", ["cancelled", "error", "eof"])
    def test_streamed_tool_monitor_terminal_cleanup_is_request_local(self, monkeypatch, terminal):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        first_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "first",
        )
        _monitor_openai_chunk(
            first_id,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-first",
                                    "function": {"name": "terminal", "arguments": "{"},
                                }
                            ]
                        },
                    }
                ]
            },
            streaming = True,
        )
        if terminal == "error":
            monitor.fail(first_id, "upstream failed")
        else:
            monitor.finish(first_id, "cancelled" if terminal == "cancelled" else "completed")

        second_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "second",
        )
        _monitor_openai_chunk(
            second_id,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '{"b":2}'}}]
                        },
                    }
                ]
            },
            streaming = True,
        )
        _monitor_openai_chunk(
            second_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            streaming = True,
        )

        assert monitor.get(first_id)["reply"] == "Tool call: terminal({)"
        assert monitor.get(second_id)["reply"] == 'Tool call: <unknown>({"b":2})'

    def test_streamed_non_tool_monitor_content_is_unchanged(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        for content in ["hel", "lo"]:
            _monitor_openai_chunk(
                monitor_id,
                {"choices": [{"index": 0, "delta": {"content": content}}]},
                streaming = True,
            )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == "hello"

    @pytest.mark.parametrize(
        ("tool_first", "expected"),
        [
            (False, "Checking:\nTool call: lookup({})"),
            (True, "Tool call: lookup({})\nDone."),
        ],
    )
    def test_streamed_tool_monitor_separates_mixed_text_segments(
        self, monkeypatch, tool_first, expected
    ):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        text = "Done." if tool_first else "Checking:"
        text_chunk = {"choices": [{"index": 0, "delta": {"content": text}}]}
        tool_chunk = {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-mixed",
                                "function": {"name": "lookup", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        }
        finish_chunk = {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
        if tool_first:
            for chunk in (tool_chunk, finish_chunk, text_chunk):
                _monitor_openai_chunk(monitor_id, chunk, streaming = True)
        else:
            for chunk in (text_chunk, tool_chunk, finish_chunk):
                _monitor_openai_chunk(monitor_id, chunk, streaming = True)

        assert monitor.get(monitor_id)["reply"] == expected

    def test_streamed_tool_monitor_keeps_final_text_after_pending_tool(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        _monitor_openai_chunk(
            monitor_id,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-final-text",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
                            ]
                        },
                    }
                ]
            },
            streaming = True,
        )
        _monitor_openai_chunk(
            monitor_id,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Done."},
                    }
                ]
            },
            streaming = True,
        )
        _monitor_openai_chunk(
            monitor_id,
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == "Tool call: lookup({})\nDone."

    def test_streamed_tool_monitor_flushes_scalar_call_before_terminal_content(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        _monitor_openai_chunk(
            monitor_id,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "lookup", "arguments": "1"},
                                }
                            ]
                        },
                    }
                ]
            },
            streaming = True,
        )
        _monitor_openai_chunk(
            monitor_id,
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Done."},
                        "finish_reason": "stop",
                    }
                ]
            },
            streaming = True,
        )

        assert monitor.get(monitor_id)["reply"] == "Tool call: lookup(1)\nDone."

    def test_streamed_tool_monitor_keeps_incomplete_call_across_content(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":',
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {"content": "Checking."}}]},
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{"index": 0, "function": {"arguments": '"weather"}'}}]
                        },
                    }
                ]
            },
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        ]
        for chunk in chunks:
            _monitor_openai_chunk(monitor_id, chunk, streaming = True)

        assert monitor.get(monitor_id)["reply"] == (
            'Checking.\nTool call: lookup({"query":"weather"})'
        )

    @pytest.mark.parametrize(
        "arguments",
        ["1", json.dumps({"query": "x" * 600})],
    )
    def test_streamed_tool_monitor_flushes_on_tool_event_boundary(self, monkeypatch, arguments):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        chunks = [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"name": "lookup", "arguments": arguments},
                                }
                            ]
                        },
                    }
                ]
            },
            {
                "choices": [{"index": 0, "delta": {}}],
                "_toolEvent": {"type": "tool_start"},
            },
            {"choices": [{"index": 0, "delta": {"content": "Done."}}]},
        ]
        for chunk in chunks:
            _monitor_openai_chunk(monitor_id, chunk, streaming = True)

        reply = monitor.get(monitor_id)["reply"]
        assert reply.startswith("Tool call: lookup(")
        assert reply.endswith("\nDone.")

    def test_streamed_tool_monitor_bounds_pending_preview_state(self, monkeypatch):
        import routes.inference as inf_mod

        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        monitor_id = monitor.start(
            endpoint = "/v1/chat/completions",
            method = "POST",
            model = "gguf",
            prompt = "hi",
        )
        for index in range(70):
            _monitor_openai_chunk(
                monitor_id,
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": index,
                                        "id": ("i" * 5000) + str(index),
                                        "function": {
                                            "name": "n" * 1000,
                                            "arguments": "a" * 5000,
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
                streaming = True,
            )

        [entry] = monitor._entries
        assert len(entry.openai_stream_tool_calls) <= 64
        assert all(
            call.call_id is None or len(call.call_id) <= 501
            for call in entry.openai_stream_tool_calls
        )
        assert all(len(call.name) <= 501 for call in entry.openai_stream_tool_calls)
        assert all(len(call.arguments) <= 501 for call in entry.openai_stream_tool_calls)
        assert (
            sum(len(call.name) + len(call.arguments) for call in entry.openai_stream_tool_calls)
            <= 12000
        )
        monitor.finish(monitor_id, "cancelled")
        assert entry.openai_stream_tool_calls == []
        assert entry.openai_stream_last_tool_indexes == {}

    def test_embeddings_request_is_counted_active_and_completed(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/embeddings")
                method = "POST"

                async def json(self):
                    return {"input": ["alpha", "beta"], "model": "embed"}

                async def is_disconnected(self):
                    return False

            class FakeAsyncClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return False

                async def aclose(self):
                    return None

                async def post(self, *_args, **_kwargs):
                    assert monitor.active_count() == 1
                    return httpx.Response(
                        200,
                        json = {
                            "data": [{"embedding": [0.1]}],
                            "usage": {"prompt_tokens": 4, "total_tokens": 4},
                        },
                    )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            # Per-request client so a forced swap can close it mid-call; the pooled one is shared.
            monkeypatch.setattr(
                inf_mod,
                "_cancelable_nonstreaming_client",
                lambda: FakeAsyncClient(),
            )
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    base_url = "http://llama.test",
                    context_length = 4096,
                    model_identifier = "gguf",
                ),
            )

            response = await openai_embeddings(Request(), current_subject = "test")

            assert response.status_code == 200
            [entry] = monitor.snapshot()
            assert entry["endpoint"] == "/v1/embeddings"
            assert entry["status"] == "completed"
            assert entry["prompt_preview"] == "alpha\nbeta"
            assert entry["prompt_tokens"] == 4
            assert entry["total_tokens"] == 4
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stream_task_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                async def is_disconnected(self):
                    return False

            async def fake_send(*_args, **_kwargs):
                return httpx.Response(200, content = b"")

            async def fake_items(*_args, **_kwargs):
                yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
                await asyncio.sleep(3600)

            cancel_id = "passthrough-stream-delete-cancel"

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                cancel_id = cancel_id,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    _auth_headers = {"Authorization": "Bearer secret"},
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                "chatcmpl-test",
                monitor_id = monitor_id,
            )
            assert isinstance(response, _SameTaskStreamingResponse)
            iterator = response.body_iterator
            first = await anext(iterator)
            assert "hello" in first
            assert cancel_id in inf_mod._CANCEL_REGISTRY

            pending = asyncio.create_task(anext(iterator))
            await asyncio.sleep(0)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending

            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert entry["reply"] == "hello"
            assert monitor.active_count() == 0
            assert cancel_id not in inf_mod._CANCEL_REGISTRY

        asyncio.run(_run())

    def test_passthrough_stream_immediate_task_cancel_releases_admission_and_tracker(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            async def fake_cancel_check(*_args, **_kwargs):
                raise asyncio.CancelledError()

            cancel_id = "passthrough-stream-immediate-task-cancel"
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "_raise_if_openai_admission_cancelled",
                fake_cancel_check,
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                cancel_id = cancel_id,
            )
            backend = SimpleNamespace(
                base_url = "http://llama.test",
                context_length = 4096,
                effective_parallel_slots = 1,
                _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            )
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )

            with pytest.raises(asyncio.CancelledError):
                await _openai_passthrough_stream(
                    self._Request(),
                    threading.Event(),
                    backend,
                    payload,
                    "gguf",
                    "chatcmpl-test",
                    monitor_id = monitor_id,
                )

            assert cancel_id not in inf_mod._CANCEL_REGISTRY
            assert get_llama_admission_queue("http://llama.test").snapshot().active == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stream_queued_cancel_before_inner_first_chunk_runs_cleanup(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            class Request(self._Request):
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))

            body_holder = {}
            cleanup_called = threading.Event()

            async def fake_admitted(*_args, admission_lease, tracker, **_kwargs):
                async def cleanup():
                    admission_lease.release()
                    tracker.__exit__(None, None, None)
                    cleanup_called.set()

                class BlockingBody:
                    def __init__(self):
                        self.started = threading.Event()
                        self.closed = False

                    def __aiter__(self):
                        return self

                    async def __anext__(self):
                        self.started.set()
                        await asyncio.sleep(3600)
                        raise StopAsyncIteration

                    async def aclose(self):
                        self.closed = True
                        await cleanup()

                body = BlockingBody()
                body_holder["body"] = body
                return _SameTaskStreamingResponse(
                    body,
                    media_type = "text/event-stream",
                    unstarted_cleanup = cleanup,
                )

            monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.01")
            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_stream_admitted",
                fake_admitted,
            )

            queue = get_llama_admission_queue("http://llama.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            cancel_id = "queued-inner-unstarted-cleanup"
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                cancel_id = cancel_id,
            )
            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    effective_parallel_slots = 1,
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                "chatcmpl-test",
            )
            iterator = response.body_iterator
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                assert chunk == ": admission-wait\n\n"
                assert cancel_id in inf_mod._CANCEL_REGISTRY

                blocker.release()
                # The lease is announced before handover, so drain that marker first.
                assert (
                    await asyncio.wait_for(iterator.__anext__(), timeout = 1.0)
                    == ": admission-done\n\n"
                )
                pending = asyncio.create_task(iterator.__anext__())
                for _ in range(100):
                    if "body" in body_holder:
                        break
                    await asyncio.sleep(0.01)
                body = body_holder["body"]
                assert await asyncio.to_thread(body.started.wait, 1.0)

                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pending, timeout = 1.0)
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                blocker.release()

            assert body_holder["body"].closed
            assert cleanup_called.is_set()
            assert cancel_id not in inf_mod._CANCEL_REGISTRY
            assert queue.snapshot().active == 0

        asyncio.run(_run())

    def test_passthrough_stream_queued_cancel_after_inner_first_chunk_finalizes_monitor(
        self, monkeypatch
    ):
        async def _run():
            import routes.inference as inf_mod

            class Request(self._Request):
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))

            async def fake_admitted(
                *_args,
                monitor_id = None,
                admission_lease,
                tracker,
                **_kwargs,
            ):
                async def cleanup():
                    admission_lease.release()
                    tracker.__exit__(None, None, None)

                async def body():
                    try:
                        yield 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                        await asyncio.sleep(3600)
                    except asyncio.CancelledError:
                        inf_mod.api_monitor.finish(monitor_id, "cancelled")
                        raise
                    finally:
                        await cleanup()

                return _SameTaskStreamingResponse(
                    body(),
                    media_type = "text/event-stream",
                    unstarted_cleanup = cleanup,
                )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.01")
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_stream_admitted",
                fake_admitted,
            )
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )

            queue = get_llama_admission_queue("http://llama.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            cancel_id = "queued-inner-cancel-monitor"
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                cancel_id = cancel_id,
            )
            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    effective_parallel_slots = 1,
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                "chatcmpl-test",
                monitor_id = monitor_id,
            )
            iterator = response.body_iterator
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                assert chunk == ": admission-wait\n\n"

                blocker.release()
                # The lease is announced before handover, so the payload is the next chunk.
                assert (
                    await asyncio.wait_for(iterator.__anext__(), timeout = 1.0)
                    == ": admission-done\n\n"
                )
                first = await asyncio.wait_for(iterator.__anext__(), timeout = 1.0)
                assert "hello" in first

                pending = asyncio.create_task(iterator.__anext__())
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(pending, timeout = 1.0)
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                blocker.release()

            assert cancel_id not in inf_mod._CANCEL_REGISTRY
            assert queue.snapshot().active == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stream_synthesizes_missing_finish_reason(self, monkeypatch):
        async def _run():
            result = await self._run_passthrough_stream(
                monkeypatch,
                [
                    (
                        'data: {"id":"upstream","created":123,"model":"gguf",'
                        '"choices":[{"index":0,"delta":{"content":"hello"}}]}'
                    ),
                    "data: [DONE]",
                ],
            )
            body = result.body

            assert '"finish_reason":"stop"' in body.replace(" ", "")
            assert "data: [DONE]" in body
            assert result.monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stream_synthesizes_tool_call_finish_reason(self, monkeypatch):
        async def _run():
            result = await self._run_passthrough_stream(
                monkeypatch,
                [
                    (
                        'data: {"id":"upstream","created":123,"model":"gguf",'
                        '"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,'
                        '"id":"call_1","type":"function","function":{"name":"lookup",'
                        '"arguments":"{}"}}]}}]}'
                    ),
                    "data: [DONE]",
                ],
            )
            compact = result.body.replace(" ", "")

            assert '"finish_reason":"tool_calls"' in compact
            assert '"finish_reason":"stop"' not in compact
            assert "data: [DONE]" in result.body
            assert result.monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stream_error_done_skips_synthetic_finish_reason(self, monkeypatch):
        async def _run():
            result = await self._run_passthrough_stream(
                monkeypatch,
                [
                    'data: {"error":{"message":"boom","type":"server_error"}}',
                    "data: [DONE]",
                ],
            )
            compact = result.body.replace(" ", "")

            assert '"error":{"message":"boom","type":"server_error"}' in compact
            assert '"finish_reason"' not in compact
            assert "data: [DONE]" in result.body
            [entry] = result.monitor.snapshot()
            assert entry["status"] == "error"
            assert entry["error"] == "boom"
            assert result.monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stream_error_eof_skips_synthetic_finish_reason(self, monkeypatch):
        async def _run():
            result = await self._run_passthrough_stream(
                monkeypatch,
                ['data: {"error":{"message":"boom","type":"server_error"}}'],
            )
            compact = result.body.replace(" ", "")

            assert '"error":{"message":"boom","type":"server_error"}' in compact
            assert '"finish_reason"' not in compact
            assert "data: [DONE]" not in result.body
            [entry] = result.monitor.snapshot()
            assert entry["status"] == "error"
            assert entry["error"] == "boom"
            assert result.monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_usage_done_are_separate_sse_events(self, monkeypatch):
        async def _run():
            result = await self._run_passthrough_stream(
                monkeypatch,
                [
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                    'data: {"id":"chatcmpl-test","object":"chat.completion.chunk","created":1,"model":"m","choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
                ],
                stream_options = {"include_usage": True},
            )

            assert (
                '"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2' in result.body
            )
            assert "data: [DONE]" in result.body
            assert "}\n\ndata: [DONE]\n\n" in result.body
            assert "}\ndata: [DONE]\n\n" not in result.body

        asyncio.run(_run())

    def test_passthrough_requests_usage_for_monitor_without_exposing_it(self, monkeypatch):
        async def _run():
            result = await self._run_passthrough_stream(
                monkeypatch,
                [
                    'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}',
                    'data: {"id":"chatcmpl-test","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
                    'data: {"id":"chatcmpl-test","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
                ],
            )

            assert result.upstream_bodies[0]["stream_options"]["include_usage"] is True
            assert '"usage"' not in result.body
            [entry] = result.monitor.snapshot()
            assert entry["prompt_tokens"] == 3
            assert entry["completion_tokens"] == 2
            assert entry["total_tokens"] == 5

        asyncio.run(_run())

    def test_passthrough_stream_queued_request_sends_keepalive_before_upstream(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
                url = SimpleNamespace(path = "/v1/chat/completions")

                async def is_disconnected(self):
                    return False

            async def fail_admitted(*_args, **_kwargs):
                raise AssertionError("upstream must not start while request is queued")

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.01")
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_openai_passthrough_stream_admitted", fail_admitted)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )

            queue = get_llama_admission_queue("http://llama.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
            )
            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    effective_parallel_slots = 1,
                    context_length = 4096,
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                "chatcmpl-test",
                monitor_id = monitor_id,
            )
            iterator = response.body_iterator
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                assert chunk == ": admission-wait\n\n"
                snapshot = queue.snapshot()
                assert snapshot.active == 1
                assert snapshot.queued == 1
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_admission_timeout_before_upstream(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
                url = SimpleNamespace(path = "/v1/chat/completions")

                async def is_disconnected(self):
                    return False

            async def fail_upstream(*_args, **_kwargs):
                raise AssertionError("upstream must not start while request is queued")

            monkeypatch.setenv(ADMISSION_QUEUE_TIMEOUT_ENV, "0.01")
            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_non_streaming_upstream",
                fail_upstream,
            )

            queue = get_llama_admission_queue("http://llama.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
            )
            try:
                with pytest.raises(HTTPException) as exc:
                    await _openai_passthrough_non_streaming(
                        SimpleNamespace(
                            base_url = "http://llama.test",
                            effective_parallel_slots = 1,
                            context_length = 4096,
                            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                        ),
                        payload,
                        "gguf",
                        request = Request(),
                        cancel_event = threading.Event(),
                    )
                assert exc.value.status_code == 503
            finally:
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_admission_queue_full_before_upstream(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
                url = SimpleNamespace(path = "/v1/chat/completions")

                async def is_disconnected(self):
                    return False

            async def fail_upstream(*_args, **_kwargs):
                raise AssertionError("upstream must not start when admission queue is full")

            monkeypatch.setenv(ADMISSION_MAX_QUEUE_ENV, "1")
            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_non_streaming_upstream",
                fail_upstream,
            )

            queue = get_llama_admission_queue("http://llama.test")
            blocker = queue.reserve(
                capacity = 1,
                config = LlamaAdmissionConfig(max_queue = 1),
            ).lease_nowait()
            queued = queue.reserve(capacity = 1, config = LlamaAdmissionConfig(max_queue = 1))
            assert blocker is not None
            assert queued.lease_nowait() is None

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
            )
            try:
                with pytest.raises(HTTPException) as exc:
                    await _openai_passthrough_non_streaming(
                        SimpleNamespace(
                            base_url = "http://llama.test",
                            effective_parallel_slots = 1,
                            context_length = 4096,
                            _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                        ),
                        payload,
                        "gguf",
                        request = Request(),
                        cancel_event = threading.Event(),
                    )
                assert exc.value.status_code == 429
            finally:
                queued.cancel()
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_immediate_cancel_stops_before_upstream(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fail_upstream(*_args, **_kwargs):
                raise AssertionError("upstream must not start after client cancellation")

            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_non_streaming_upstream",
                fail_upstream,
            )

            cancel_event = threading.Event()
            cancel_event.set()
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
            )

            with pytest.raises(HTTPException) as exc:
                await _openai_passthrough_non_streaming(
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        effective_parallel_slots = 1,
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "gguf",
                    monitor_id = monitor_id,
                    cancel_event = cancel_event,
                )

            assert exc.value.status_code == 499
            assert get_llama_admission_queue("http://llama.test").snapshot().active == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_admission_task_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_wait(*_args, **_kwargs):
                raise asyncio.CancelledError()

            async def fail_upstream(*_args, **_kwargs):
                raise AssertionError("upstream must not start after admission task cancel")

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "_wait_for_openai_admission_non_streaming",
                fake_wait,
            )
            monkeypatch.setattr(
                inf_mod,
                "_openai_passthrough_non_streaming_upstream",
                fail_upstream,
            )
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
            )

            with pytest.raises(asyncio.CancelledError):
                await _openai_passthrough_non_streaming(
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        effective_parallel_slots = 1,
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "gguf",
                    monitor_id = monitor_id,
                    cancel_event = threading.Event(),
                )

            assert get_llama_admission_queue("http://llama.test").snapshot().active == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class CancellingAsyncClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return False

                async def post(self, *_args, **_kwargs):
                    raise asyncio.CancelledError()

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "nonstreaming_client",
                lambda: CancellingAsyncClient(),
            )
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            with pytest.raises(asyncio.CancelledError):
                await _openai_passthrough_non_streaming(
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "gguf",
                    monitor_id = monitor_id,
                )

            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_cancel_closes_blocked_upstream_post(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class HangingCancelableClient:
                def __init__(self):
                    self.started = asyncio.Event()
                    self.closed = asyncio.Event()

                async def post(self, *_args, **_kwargs):
                    self.started.set()
                    await self.closed.wait()
                    raise httpx.ReadError("client closed")

                async def aclose(self):
                    self.closed.set()

            class Request:
                async def is_disconnected(self):
                    return False

            client = HangingCancelableClient()
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "_cancelable_nonstreaming_client",
                lambda: client,
            )
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            cancel_event = threading.Event()
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            task = asyncio.create_task(
                _openai_passthrough_non_streaming(
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "gguf",
                    monitor_id = monitor_id,
                    request = Request(),
                    cancel_event = cancel_event,
                )
            )
            await asyncio.wait_for(client.started.wait(), 0.2)
            cancel_event.set()

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 0.5)

            assert client.closed.is_set()
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_route_registers_cancel_id(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class HangingCancelableClient:
                def __init__(self):
                    self.started = asyncio.Event()
                    self.closed = asyncio.Event()

                async def post(self, *_args, **_kwargs):
                    self.started.set()
                    await self.closed.wait()
                    raise httpx.ReadError("client closed")

                async def aclose(self):
                    self.closed.set()

            class Request:
                state = SimpleNamespace()
                url = SimpleNamespace(path = "/v1/chat/completions")
                method = "POST"

                async def is_disconnected(self):
                    return False

            cancel_id = "passthrough-nonstream-cancel-id"
            client = HangingCancelableClient()
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: client)

            def _plain(**_kwargs):
                raise AssertionError("plain GGUF path should not be used")

            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    is_vision = False,
                    supports_tools = True,
                    _is_audio = False,
                    model_identifier = "test-gguf",
                    context_length = 4096,
                    base_url = "http://llama.test",
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    generate_chat_completion = _plain,
                ),
            )

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                cancel_id = cancel_id,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            task = asyncio.create_task(
                openai_chat_completions(
                    payload,
                    request = Request(),
                    current_subject = "test",
                )
            )
            await asyncio.wait_for(client.started.wait(), 5.0)
            assert cancel_id in inf_mod._CANCEL_REGISTRY
            assert inf_mod._cancel_by_cancel_id_or_stash(cancel_id) == 1

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5.0)

            assert client.closed.is_set()
            assert cancel_id not in inf_mod._CANCEL_REGISTRY
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_disconnect_closes_blocked_upstream_post(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class HangingCancelableClient:
                def __init__(self):
                    self.started = asyncio.Event()
                    self.closed = asyncio.Event()

                async def post(self, *_args, **_kwargs):
                    self.started.set()
                    await self.closed.wait()
                    raise httpx.ReadError("client closed")

                async def aclose(self):
                    self.closed.set()

            class Request:
                def __init__(self):
                    self.disconnected = False

                async def is_disconnected(self):
                    return self.disconnected

            client = HangingCancelableClient()
            request = Request()
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "_cancelable_nonstreaming_client",
                lambda: client,
            )
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            cancel_event = threading.Event()
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            task = asyncio.create_task(
                _openai_passthrough_non_streaming(
                    SimpleNamespace(
                        base_url = "http://llama.test",
                        context_length = 4096,
                        _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                    ),
                    payload,
                    "gguf",
                    monitor_id = monitor_id,
                    request = request,
                    cancel_event = cancel_event,
                )
            )
            await asyncio.wait_for(client.started.wait(), 0.2)
            request.disconnected = True

            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 0.5)

            assert client.closed.is_set()
            assert cancel_event.is_set()
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_non_streaming_forwards_backend_auth_headers(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            captured = {}

            class FakeNonStreamingClient:
                async def post(self, *_args, **kwargs):
                    captured["headers"] = kwargs.get("headers")
                    return httpx.Response(
                        200,
                        json = {
                            "id": "chatcmpl-test",
                            "object": "chat.completion",
                            "created": 123,
                            "model": "gguf",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": "OK"},
                                    "finish_reason": "stop",
                                }
                            ],
                        },
                    )

            monitor = ApiMonitor(max_entries = 3)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "nonstreaming_client",
                lambda: FakeNonStreamingClient(),
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            response = await _openai_passthrough_non_streaming(
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    _auth_headers = {"Authorization": "Bearer secret"},
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                monitor_id = monitor_id,
            )

            assert json.loads(response.body)["choices"][0]["message"]["content"] == "OK"
            assert captured["headers"]["Authorization"] == "Bearer secret"
            assert captured["headers"]["Connection"] == "close"

        asyncio.run(_run())

    def test_passthrough_non_streaming_forces_upstream_stream_false(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            captured = {}

            class FakeNonStreamingClient:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return False

                async def post(self, *_args, **kwargs):
                    captured["json"] = kwargs.get("json")
                    return httpx.Response(
                        200,
                        json = {
                            "id": "chatcmpl-test",
                            "object": "chat.completion",
                            "created": 123,
                            "model": "gguf",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": "OK"},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 1,
                                "completion_tokens": 1,
                                "total_tokens": 2,
                            },
                        },
                    )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "nonstreaming_client",
                lambda: FakeNonStreamingClient(),
            )
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                stream_options = {"include_usage": True},
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            await _openai_passthrough_non_streaming(
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                monitor_id = monitor_id,
            )

            assert captured["json"]["stream"] is False
            assert "stream_options" not in captured["json"]
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"

        asyncio.run(_run())

    def test_passthrough_clean_eof_finalizes_monitor(self, monkeypatch):
        async def _run():
            result = await self._run_passthrough_stream(
                monkeypatch,
                ['data: {"choices":[{"delta":{"content":"hello"}}]}'],
            )
            chunks = result.chunks

            assert chunks[0] == 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            compact = "".join(chunks).replace(" ", "")
            assert '"finish_reason":"stop"' in compact
            assert chunks[-1] == "data: [DONE]\n\n"
            [entry] = result.monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["reply"] == "hello"
            assert result.monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_finish_without_done_closes_stream_early(self, monkeypatch):
        # Some llama-server builds emit the finish chunk and then hold the HTTP
        # stream open without sending [DONE]; the terminal classifier must end
        # the client stream promptly instead of hanging on the open socket.
        async def _run():
            import routes.inference as inf_mod

            class Request:
                async def is_disconnected(self):
                    return False

            async def fake_send(*_args, **_kwargs):
                return httpx.Response(200, content = b"")

            async def fake_items(
                *_args,
                post_first_item_read_timeout_s = None,
                **_kwargs,
            ):
                yield 'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}'
                yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
                timeout = post_first_item_read_timeout_s()
                await asyncio.sleep(timeout)
                raise httpx.ReadTimeout("upstream never closed")

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                "chatcmpl-test",
                monitor_id = monitor_id,
            )

            async def _consume():
                return [chunk async for chunk in response.body_iterator]

            chunks = await asyncio.wait_for(_consume(), timeout = 2)
            body = "".join(chunks)

            assert '"finish_reason":"stop"' in body.replace(" ", "")
            assert body.endswith("data: [DONE]\n\n")
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stall_after_finish_closes_cleanly(self, monkeypatch):
        # include_usage keeps the stream open past the finish chunk waiting for
        # the usage chunk; if that never arrives, the post-terminal grace path
        # must close with a clean [DONE], not an in-band error.
        async def _run():
            import routes.inference as inf_mod

            class Request:
                async def is_disconnected(self):
                    return False

            async def fake_send(*_args, **_kwargs):
                return httpx.Response(200, content = b"")

            async def fake_items(*_args, **_kwargs):
                yield 'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}'
                yield 'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
                raise httpx.ReadTimeout("usage chunk never arrived")

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                stream_options = {"include_usage": True},
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                "chatcmpl-test",
                monitor_id = monitor_id,
            )
            chunks = [chunk async for chunk in response.body_iterator]
            body = "".join(chunks)

            assert '"type":"api_error"' not in body.replace(" ", "")
            assert body.endswith("data: [DONE]\n\n")
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_passthrough_stream_stall_after_data_emits_error(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class Request:
                async def is_disconnected(self):
                    return False

            async def fake_send(*_args, **_kwargs):
                return httpx.Response(200, content = b"")

            async def fake_items(*_args, **_kwargs):
                yield 'data: {"choices":[{"delta":{"content":"hello"}}]}'
                raise httpx.ReadTimeout("upstream went silent")

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fake_send)
            monkeypatch.setattr(inf_mod, "_aiter_llama_stream_items", fake_items)
            monitor_id = monitor.start(
                endpoint = "/v1/chat/completions",
                method = "POST",
                model = "gguf",
                prompt = "hi",
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                stream = True,
                tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            )

            response = await _openai_passthrough_stream(
                Request(),
                threading.Event(),
                SimpleNamespace(
                    base_url = "http://llama.test",
                    context_length = 4096,
                    _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
                ),
                payload,
                "gguf",
                "chatcmpl-test",
                monitor_id = monitor_id,
            )
            chunks = [chunk async for chunk in response.body_iterator]
            body = "".join(chunks)

            assert 'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n' in body
            assert '"finish_reason"' not in body.replace(" ", "")
            assert '"type":"api_error"' in body.replace(" ", "")
            assert "still processing the prompt" in body
            assert body.endswith("data: [DONE]\n\n")
            [entry] = monitor.snapshot()
            assert entry["status"] == "error"
            assert "still processing the prompt" in entry["error"]
            assert entry["reply"] == "hello"
            assert monitor.active_count() == 0

        asyncio.run(_run())


class TestApiMonitorSafetensorsUsage:
    class _Request:
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/chat/completions")
        method = "POST"

    def test_non_streaming_safetensors_keeps_event_loop_responsive(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            generation_started = threading.Event()
            generation_finished = threading.Event()
            generation_threads = []

            class DummyBackend:
                active_model_name = "safe-model"
                models = {"safe-model": {"context_length": 2048}}

                def generate_chat_response(self, **_kwargs):
                    generation_threads.append(threading.current_thread())
                    generation_started.set()
                    try:
                        time.sleep(0.08)
                        yield "safe reply"
                    finally:
                        generation_finished.set()

                def reset_generation_state(self, caller_cancel_event = None):
                    pass

            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = False,
                    supports_tools = False,
                    is_vision = False,
                    context_length = None,
                ),
            )
            monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: DummyBackend())
            monkeypatch.setattr(
                inf_mod,
                "_detect_safetensors_features",
                lambda *_args, **_kwargs: {"supports_tools": False},
            )
            monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
            )
            heartbeat_ticks = 0

            async def _heartbeat():
                nonlocal heartbeat_ticks
                while not generation_finished.is_set():
                    await asyncio.sleep(0.005)
                    if generation_started.is_set() and not generation_finished.is_set():
                        heartbeat_ticks += 1

            heartbeat = asyncio.create_task(_heartbeat())
            response = await openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
            await heartbeat

            assert response.status_code == 200
            assert heartbeat_ticks > 0
            assert len(generation_threads) == 1
            assert generation_threads[0] is not threading.current_thread()

        asyncio.run(_run())

    def test_non_streaming_safetensors_records_usage(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class DummyBackend:
                active_model_name = "safe-model"
                models = {"safe-model": {"context_length": 2048}}

                def generate_chat_response(self, *, stats_holder, **_kwargs):
                    stats_holder["stats"] = {
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 5,
                            "total_tokens": 13,
                        }
                    }
                    yield "safe reply"

                def reset_generation_state(self, caller_cancel_event = None):
                    pass

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = False,
                    supports_tools = False,
                    is_vision = False,
                    context_length = None,
                ),
            )
            monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: DummyBackend())
            monkeypatch.setattr(
                inf_mod,
                "_detect_safetensors_features",
                lambda *_args, **_kwargs: {"supports_tools": False},
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
            )

            response = await openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
            body = json.loads(response.body)

            assert body["choices"][0]["message"]["content"] == "safe reply"
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["reply"] == "safe reply"
            assert entry["prompt_tokens"] == 8
            assert entry["completion_tokens"] == 5
            assert entry["total_tokens"] == 13
            assert entry["context_length"] == 2048

        asyncio.run(_run())

    def test_non_streaming_safetensors_tool_cancel_records_cancelled(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            reset_tool_policy()

            class DummyBackend:
                active_model_name = "safe-model"
                models = {"safe-model": {"context_length": 2048}}

                def generate_chat_response(self, **_kwargs):
                    raise AssertionError("plain safetensors path should not be used")

                def generate_chat_completion_with_tools(
                    self, *, cancel_event, stats_holder, **_kwargs
                ):
                    stats_holder["stats"] = {
                        "usage": {
                            "prompt_tokens": 8,
                            "completion_tokens": 5,
                            "total_tokens": 13,
                        }
                    }
                    yield {"type": "content", "text": "partial"}
                    cancel_event.set()
                    yield {"type": "content", "text": "ignored"}

                def reset_generation_state(self, caller_cancel_event = None):
                    pass

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = False,
                    supports_tools = False,
                    is_vision = False,
                    context_length = None,
                ),
            )
            monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: DummyBackend())
            monkeypatch.setattr(
                inf_mod,
                "_detect_safetensors_features",
                lambda *_args, **_kwargs: {"supports_tools": True},
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                enable_tools = True,
                enabled_tools = ["web_search"],
                cancel_id = "safe-cancel",
            )

            response = await openai_chat_completions(
                payload,
                request = self._Request(),
                current_subject = "test",
            )
            body = json.loads(response.body)

            assert body["choices"][0]["message"]["content"] == "partial"
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert entry["reply"] == "partial"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_non_streaming_safetensors_tool_task_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            reset_tool_policy()
            reset_called = False

            class DummyBackend:
                active_model_name = "safe-model"
                models = {"safe-model": {"context_length": 2048}}

                def generate_chat_response(self, **_kwargs):
                    raise AssertionError("plain safetensors path should not be used")

                def generate_chat_completion_with_tools(self, **_kwargs):
                    yield {"type": "content", "text": "unused"}

                def reset_generation_state(self, caller_cancel_event = None):
                    nonlocal reset_called
                    reset_called = True

            async def fake_to_thread(
                func = None,
                *_args,
                **_kwargs,
            ):
                # Only the generation hop should cancel; resolution runs before the row opens.
                if getattr(func, "__name__", "") == "resolve_local_gguf":
                    return None
                # Resolving what is already serving is pre-row work too, offloaded for the
                # same reason: _loaded_satisfies reaches the singleton, whose build detects.
                if func in (inf_mod.get_inference_backend, inf_mod._loaded_satisfies):
                    return func(*_args, **_kwargs)
                raise asyncio.CancelledError()

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod.asyncio, "to_thread", fake_to_thread)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = False,
                    supports_tools = False,
                    is_vision = False,
                    context_length = None,
                ),
            )
            monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: DummyBackend())
            monkeypatch.setattr(
                inf_mod,
                "_detect_safetensors_features",
                lambda *_args, **_kwargs: {"supports_tools": True},
            )
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "hi")],
                enable_tools = True,
                enabled_tools = ["web_search"],
                cancel_id = "safe-cancel",
            )

            with pytest.raises(asyncio.CancelledError):
                await openai_chat_completions(
                    payload,
                    request = self._Request(),
                    current_subject = "test",
                )

            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0
            assert reset_called is True

        asyncio.run(_run())


class TestApiMonitorAudioInput:
    def _patch_audio_backend(self, monkeypatch, chunks):
        import routes.inference as inf_mod

        class DummyAudioBackend:
            active_model_name = "audio-model"
            models = {
                "audio-model": {
                    "has_audio_input": True,
                    "audio_type": "audio-input",
                }
            }

            def generate_audio_input_response(self, **_kwargs):
                yield from chunks

        monkeypatch.setattr(
            inf_mod,
            "get_llama_cpp_backend",
            lambda: SimpleNamespace(is_loaded = False),
        )
        monkeypatch.setattr(
            inf_mod,
            "get_inference_backend",
            lambda: DummyAudioBackend(),
        )
        monkeypatch.setattr(
            inf_mod,
            "_decode_audio_base64",
            lambda _payload: object(),
        )
        return inf_mod

    def test_audio_input_non_streaming_records_active_monitor(self, monkeypatch):
        async def _run():
            inf_mod = self._patch_audio_backend(monkeypatch, ["hello", " world"])
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "describe this audio")],
                audio_base64 = "ZmFrZQ==",
            )
            request = SimpleNamespace(
                state = SimpleNamespace(),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
            )

            response = await openai_chat_completions(
                payload,
                request = request,
                current_subject = "test",
            )
            body = json.loads(response.body)

            assert body["choices"][0]["message"]["content"] == "hello world"
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["reply"] == "hello world"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_audio_input_non_streaming_keeps_event_loop_responsive(self, monkeypatch):
        async def _run():
            generation_started = threading.Event()
            generation_finished = threading.Event()
            generation_threads = []

            def _chunks():
                generation_threads.append(threading.current_thread())
                generation_started.set()
                try:
                    time.sleep(0.08)
                    yield "hello"
                finally:
                    generation_finished.set()

            inf_mod = self._patch_audio_backend(monkeypatch, _chunks())
            monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))
            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "describe this audio")],
                audio_base64 = "ZmFrZQ==",
            )
            request = SimpleNamespace(
                state = SimpleNamespace(),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
            )
            heartbeat_ticks = 0

            async def _heartbeat():
                nonlocal heartbeat_ticks
                while not generation_finished.is_set():
                    await asyncio.sleep(0.005)
                    if generation_started.is_set() and not generation_finished.is_set():
                        heartbeat_ticks += 1

            heartbeat = asyncio.create_task(_heartbeat())
            response = await openai_chat_completions(
                payload,
                request = request,
                current_subject = "test",
            )
            await heartbeat

            assert response.status_code == 200
            assert heartbeat_ticks > 0
            assert len(generation_threads) == 1
            assert generation_threads[0] is not threading.current_thread()

        asyncio.run(_run())

    def test_audio_input_streaming_records_monitor_reply(self, monkeypatch):
        async def _run():
            inf_mod = self._patch_audio_backend(monkeypatch, ["hello", " world"])
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)

            async def is_disconnected():
                return False

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "describe this audio")],
                audio_base64 = "ZmFrZQ==",
                stream = True,
            )
            request = SimpleNamespace(
                state = SimpleNamespace(),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
                is_disconnected = is_disconnected,
            )

            response = await openai_chat_completions(
                payload,
                request = request,
                current_subject = "test",
            )
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)

            assert chunks[-1] == "data: [DONE]\n\n"
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["reply"] == "hello world"
            assert monitor.active_count() == 0

            def failing_chunks():
                yield "partial"
                raise RuntimeError("generation failed")

            self._patch_audio_backend(monkeypatch, failing_chunks())
            error_monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", error_monitor)
            error_response = await openai_chat_completions(
                payload,
                request = request,
                current_subject = "test",
            )
            error_chunks = [
                chunk.decode() if isinstance(chunk, bytes) else chunk
                async for chunk in error_response.body_iterator
            ]

            assert '"type": "server_error"' in error_chunks[-1]
            assert error_chunks[-1].endswith("data: [DONE]\n\n")
            [error_entry] = error_monitor.snapshot()
            assert error_entry["status"] == "error"
            assert error_monitor.active_count() == 0

        asyncio.run(_run())

    def test_non_gguf_tts_auto_route_records_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class DummyTtsBackend:
                active_model_name = "tts-model"
                models = {
                    "tts-model": {
                        "is_audio": True,
                        "audio_type": "snac",
                    }
                }

            async def fake_generate_audio(
                _payload,
                _request,
                current_subject = None,
            ):
                return inf_mod.JSONResponse(
                    content = {
                        "choices": [
                            {
                                "message": {
                                    "content": "[Generated audio]",
                                }
                            }
                        ]
                    }
                )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(is_loaded = False),
            )
            monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: DummyTtsBackend())
            monkeypatch.setattr(inf_mod, "generate_audio", fake_generate_audio)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "say hello")],
            )
            request = SimpleNamespace(
                state = SimpleNamespace(),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
            )

            response = await inf_mod.openai_chat_completions(
                payload,
                request = request,
                current_subject = "test",
            )

            assert json.loads(response.body)["choices"][0]["message"]["content"] == (
                "[Generated audio]"
            )
            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["model"] == "tts-model"
            assert entry["reply"] == "[Generated audio]"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_non_gguf_tts_cancel_finalizes_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            class DummyTtsBackend:
                active_model_name = "tts-model"
                models = {
                    "tts-model": {
                        "is_audio": True,
                        "audio_type": "snac",
                    }
                }

            async def fake_generate_audio(
                _payload,
                _request,
                current_subject = None,
            ):
                raise asyncio.CancelledError()

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(is_loaded = False),
            )
            monkeypatch.setattr(inf_mod, "get_inference_backend", lambda: DummyTtsBackend())
            monkeypatch.setattr(inf_mod, "generate_audio", fake_generate_audio)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "say hello")],
            )
            request = SimpleNamespace(
                state = SimpleNamespace(),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
            )

            with pytest.raises(asyncio.CancelledError):
                await inf_mod.openai_chat_completions(
                    payload,
                    request = request,
                    current_subject = "test",
                )

            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert entry["model"] == "tts-model"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_gguf_tts_auto_route_records_monitor(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fake_generate_audio(
                _payload,
                _request,
                current_subject = None,
            ):
                return inf_mod.JSONResponse(
                    content = {
                        "choices": [
                            {
                                "message": {
                                    "content": "[Generated audio]",
                                }
                            }
                        ]
                    }
                )

            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(
                inf_mod,
                "get_llama_cpp_backend",
                lambda: SimpleNamespace(
                    is_loaded = True,
                    _is_audio = True,
                    model_identifier = "gguf-tts",
                    context_length = 2048,
                ),
            )
            monkeypatch.setattr(inf_mod, "generate_audio", fake_generate_audio)

            payload = ChatCompletionRequest(
                model = "default",
                messages = [ChatMessage(role = "user", content = "say hello")],
            )
            request = SimpleNamespace(
                state = SimpleNamespace(),
                url = SimpleNamespace(path = "/v1/chat/completions"),
                method = "POST",
            )

            await inf_mod.openai_chat_completions(
                payload,
                request = request,
                current_subject = "test",
            )

            [entry] = monitor.snapshot()
            assert entry["status"] == "completed"
            assert entry["model"] == "gguf-tts"
            assert entry["context_length"] == 2048
            assert entry["reply"] == "[Generated audio]"
            assert monitor.active_count() == 0

            # This branch answers speech and returns before the routing that decides
            # a decoding contract, so it refuses one itself rather than ignoring it.
            payload.response_format = {"type": "json_object"}
            with pytest.raises(HTTPException) as excinfo:
                await inf_mod.openai_chat_completions(
                    payload, request = request, current_subject = "test"
                )
            assert excinfo.value.status_code == 400
            assert excinfo.value.detail["error"]["param"] == "response_format"
            # `{"type": "text"}` constrains nothing, so speech is still served.
            payload.response_format = {"type": "text"}
            await inf_mod.openai_chat_completions(payload, request = request, current_subject = "test")

        asyncio.run(_run())


# =====================================================================

# =====================================================================
# Responses API -> Chat Completions translation: chat_template_kwargs
# (e.g. {"enable_thinking": true}) sent via the Responses extra-body must
# reach the built ChatCompletionRequest's typed ``enable_thinking`` field,
# otherwise /v1/responses silently ignores reasoning control (issue #6198).
# =====================================================================


class TestResponsesChatTemplateKwargs:
    _messages = [ChatMessage(role = "user", content = "What is 100 - 67?")]

    class _Request:
        app = SimpleNamespace(state = SimpleNamespace(llama_parallel_slots = 1))
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/responses")
        method = "POST"

        async def is_disconnected(self):
            return False

    def test_enable_thinking_lifted_from_extra_body(self):
        payload = ResponsesRequest(
            model = "qwen-local",
            input = "What is 100 - 67?",
            chat_template_kwargs = {"enable_thinking": True},
        )
        chat_req = _build_chat_request(payload, self._messages, stream = False)
        assert chat_req.enable_thinking is True

    def test_enable_thinking_false_lifted_from_extra_body(self):
        payload = ResponsesRequest(
            model = "qwen-local",
            input = "hi",
            chat_template_kwargs = {"enable_thinking": False},
        )
        chat_req = _build_chat_request(payload, self._messages, stream = True)
        assert chat_req.enable_thinking is False

    def test_no_chat_template_kwargs_leaves_enable_thinking_unset(self):
        payload = ResponsesRequest(model = "qwen-local", input = "hi")
        chat_req = _build_chat_request(payload, self._messages, stream = False)
        assert chat_req.enable_thinking is None

    def test_chat_template_kwargs_without_enable_thinking_is_ignored(self):
        payload = ResponsesRequest(
            model = "qwen-local",
            input = "hi",
            chat_template_kwargs = {"some_other_flag": True},
        )
        chat_req = _build_chat_request(payload, self._messages, stream = False)
        assert chat_req.enable_thinking is None

    def test_responses_stream_queued_request_sends_keepalive_before_upstream(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fail_send(*_args, **_kwargs):
                raise AssertionError("responses upstream must not start while queued")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                base_url = "http://llama.responses.test",
                context_length = 4096,
                effective_parallel_slots = 1,
                _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setenv(ADMISSION_KEEPALIVE_INTERVAL_ENV, "0.01")
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fail_send)

            queue = get_llama_admission_queue("http://llama.responses.test")
            blocker = queue.reserve(capacity = 1, config = LlamaAdmissionConfig()).lease_nowait()
            assert blocker is not None
            monitor_id = monitor.start(
                endpoint = "/v1/responses",
                method = "POST",
                model = "qwen-local",
                prompt = "hi",
            )
            payload = ResponsesRequest(model = "qwen-local", input = "hi", stream = True)

            response = await _responses_stream(
                payload,
                [ChatMessage(role = "user", content = "hi")],
                self._Request(),
                monitor_id,
            )
            iterator = response.body_iterator
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
                assert chunk == ": admission-wait\n\n"
                snapshot = queue.snapshot()
                assert snapshot.active == 1
                assert snapshot.queued == 1
            finally:
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                blocker.release()

            snapshot = queue.snapshot()
            assert snapshot.active == 0
            assert snapshot.queued == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())

    def test_responses_stream_cancel_after_created_finalizes_monitor_and_slot(self, monkeypatch):
        async def _run():
            import routes.inference as inf_mod

            async def fail_send(*_args, **_kwargs):
                raise AssertionError("responses upstream must not start after created cancel")

            backend = SimpleNamespace(
                is_loaded = True,
                is_vision = False,
                base_url = "http://llama.responses.test",
                context_length = 4096,
                effective_parallel_slots = 1,
                _request_reasoning_kwargs = lambda *_args, **_kwargs: None,
            )
            monitor = ApiMonitor(max_entries = 3)
            monkeypatch.setattr(inf_mod, "api_monitor", monitor)
            monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
            monkeypatch.setattr(inf_mod, "_send_stream_with_preheader_cancel", fail_send)
            monitor_id = monitor.start(
                endpoint = "/v1/responses",
                method = "POST",
                model = "qwen-local",
                prompt = "hi",
            )
            payload = ResponsesRequest(model = "qwen-local", input = "hi", stream = True)

            response = await _responses_stream(
                payload,
                [ChatMessage(role = "user", content = "hi")],
                self._Request(),
                monitor_id,
            )
            iterator = response.body_iterator
            first = await asyncio.wait_for(iterator.__anext__(), timeout = 0.2)
            assert "event: response.created" in first

            with pytest.raises(asyncio.CancelledError):
                await iterator.athrow(asyncio.CancelledError())

            assert get_llama_admission_queue("http://llama.responses.test").snapshot().active == 0
            [entry] = monitor.snapshot()
            assert entry["status"] == "cancelled"
            assert monitor.active_count() == 0

        asyncio.run(_run())


# =====================================================================

# =====================================================================
# GGUF chat-template role alternation: coalesce orphaned user turns left
# behind when an empty assistant turn is dropped, so strict templates
# (Gemma 3, ...) do not 400 on a role-parity break.
# =====================================================================


class TestMergeUserContent:
    def test_strings_join_with_blank_line(self):
        assert _merge_user_content("hi", "again") == "hi\n\nagain"

    def test_empty_sides_passthrough(self):
        assert _merge_user_content("", "again") == "again"
        assert _merge_user_content("hi", "") == "hi"

    def test_multimodal_parts_concatenate(self):
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        out = _merge_user_content([{"type": "text", "text": "look"}, img], "and this?")
        assert out == [
            {"type": "text", "text": "look"},
            img,
            {"type": "text", "text": "and this?"},
        ]


class TestCoalesceConsecutiveUserTurns:
    def test_merges_two_string_user_turns(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        assert _coalesce_consecutive_user_turns(msgs) == [
            {"role": "user", "content": "hi\n\nagain"},
        ]

    def test_merges_three_consecutive_user_turns(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        assert _coalesce_consecutive_user_turns(msgs) == [
            {"role": "user", "content": "a\n\nb\n\nc"},
        ]

    def test_alternating_history_is_unchanged(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        assert _coalesce_consecutive_user_turns(msgs) == msgs

    def test_assistant_and_tool_turns_untouched(self):
        msgs = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        ]
        assert _coalesce_consecutive_user_turns(msgs) == msgs

    def test_multimodal_parts_survive_merge(self):
        img = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "look"}, img]},
            {"role": "user", "content": "and this?"},
        ]
        out = _coalesce_consecutive_user_turns(msgs)
        assert len(out) == 1
        assert out[0]["content"] == [
            {"type": "text", "text": "look"},
            img,
            {"type": "text", "text": "and this?"},
        ]

    def test_does_not_mutate_input(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        _coalesce_consecutive_user_turns(msgs)
        assert msgs[0]["content"] == "hi"


class TestGgufChatHistoryAlternation:
    def test_empty_assistant_turn_dropped_then_users_coalesced(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                ChatMessage(role = "user", content = "hi"),
                ChatMessage(role = "assistant", content = ""),
                ChatMessage(role = "user", content = "again"),
            ],
        )
        out, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
        roles = [m["role"] for m in out]
        assert roles == ["user"]
        assert out[0]["content"] == "hi\n\nagain"

    def test_bare_stop_sentinel_also_coalesced(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                ChatMessage(role = "user", content = "hi"),
                ChatMessage(role = "assistant"),
                ChatMessage(role = "user", content = "again"),
            ],
        )
        out, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
        roles = [m["role"] for m in out]
        assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)), roles
        assert roles == ["user"]

    def test_system_prompt_preserved(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                ChatMessage(role = "system", content = "be brief"),
                ChatMessage(role = "user", content = "hi"),
                ChatMessage(role = "assistant", content = ""),
                ChatMessage(role = "user", content = "again"),
            ],
        )
        out, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
        assert [m["role"] for m in out] == ["system", "user"]
        assert out[1]["content"] == "hi\n\nagain"

    def test_normal_history_unchanged(self):
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                ChatMessage(role = "user", content = "hi"),
                ChatMessage(role = "assistant", content = "hello"),
                ChatMessage(role = "user", content = "again"),
            ],
        )
        out, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
        assert [m["role"] for m in out] == ["user", "assistant", "user"]

    def test_tool_path_rebuild_stays_alternating(self):
        # Tool path rebuilds via _set_or_prepend_system_message over the coalesced
        # history, so it stays alternating too.
        req = ChatCompletionRequest(
            model = "default",
            messages = [
                ChatMessage(role = "user", content = "hi"),
                ChatMessage(role = "assistant", content = ""),
                ChatMessage(role = "user", content = "again"),
            ],
        )
        normalized, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
        rebuilt = _set_or_prepend_system_message(normalized, "You have access to tools.")
        roles = [m["role"] for m in rebuilt]
        assert roles == ["system", "user"]
        assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1)), roles


# ── Per-choice seeds on the GGUF drain ──────────────────────────────


def test_every_gguf_choice_gets_a_seed_of_its_own():
    """llama-server holds the seed as a uint32 and draws at random for exactly
    one value, LLAMA_DEFAULT_SEED (0xFFFFFFFF). Only -1 converts to it, so every
    other negative is an ordinary fixed seed there: exempting all of them from
    the offset sent n identical requests and returned n copies of one run."""
    from routes.inference import _choice_seed

    sent = 0xFFFFFFFF
    for seed in (-2, -3, 0, 5, 2**32 - 2):
        served = [_choice_seed(seed, i, negative_is_random = True) for i in range(3)]
        as_read = [v & 0xFFFFFFFF for v in served]
        assert len(set(as_read)) == 3, (seed, served)
        # A shifted seed that landed on the sentinel would sample at random where
        # the caller asked for a fixed run.
        assert sent not in as_read, (seed, served)

    # -1 is the sentinel itself: offsetting it would make every choice after the
    # first deterministic, which is the opposite of what was asked for.
    assert [_choice_seed(-1, i, negative_is_random = True) for i in range(3)] == [-1, -1, -1]

    # MLX maps every seed onto its key domain, so nothing is exempt there.
    assert [_choice_seed(-2, i) for i in range(3)] == [-2, -1, 0]
    assert [_choice_seed(5, i) for i in range(3)] == [5, 6, 7]

    # Choice 0 is always the caller's own seed, on both drains.
    assert _choice_seed(-2, 0, negative_is_random = True) == -2
    assert _choice_seed(None, 2, negative_is_random = True) is None


def test_the_two_seed_helpers_agree_on_which_seeds_are_random():
    """``-1`` is not the only request seed that reaches LLAMA_DEFAULT_SEED: the seed is a
    uint32 there, so ``-1``, ``4294967295`` and ``2**64-1`` are all the sentinel and the
    schemas accept all three. Both helpers must agree, or choice 0 keeps the caller's random
    seed while choice 1 is offset into a fixed one, half reproducible and half uncached."""
    from core.inference.llama_cpp import _LLAMA_RANDOM_SEED, _apply_seeded_llama_request
    from routes.inference import _choice_seed

    for seed in (-1, 0xFFFFFFFF, 2**64 - 1):
        served = [_choice_seed(seed, i, negative_is_random = True) for i in range(3)]
        assert all((v & 0xFFFFFFFF) == _LLAMA_RANDOM_SEED for v in served), (seed, served)

        for value in served:
            payload: dict = {}
            _apply_seeded_llama_request(payload, value)
            assert "cache_prompt" not in payload, (seed, value, payload)

    for seed in (0, 5, 4294967294):
        served = [_choice_seed(seed, i, negative_is_random = True) for i in range(3)]
        assert all((v & 0xFFFFFFFF) != _LLAMA_RANDOM_SEED for v in served), (seed, served)
        for value in served:
            payload = {}
            _apply_seeded_llama_request(payload, value)
            assert payload["cache_prompt"] is False, (seed, value)
