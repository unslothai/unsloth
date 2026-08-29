# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for Anthropic Messages API schemas and translation layer (no server/GPU)."""

import sys
import os
import json
import threading
import time

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
    anthropic_schema_client_tool_kind,
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
    _anthropic_plain_non_streaming,
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


def _emitter_client_thinking(events):
    thinking = ""
    for line in events:
        for raw in line.split("\n"):
            raw = raw.strip()
            if not raw.startswith("data: "):
                continue
            data = json.loads(raw[len("data: ") :])
            delta = data.get("delta", {})
            if delta.get("type") == "thinking_delta":
                thinking += delta.get("thinking", "")
    return thinking


def test_anthropic_emitter_reasoning_only_becomes_thinking_block():
    # Anthropic asks the GGUF generator not to promote reasoning into a duplicate
    # visible fallback; the balanced <think> markup becomes one typed thinking
    # block with no literal tags leaking to the client.
    emitter = AnthropicStreamEmitter()
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "<think>The capital"})
    events += emitter.feed({"type": "content", "text": "<think>The capital of France is Paris."})
    events += emitter.feed(
        {"type": "content", "text": "<think>The capital of France is Paris.</think>"}
    )
    events += emitter.finish()

    assert _emitter_client_thinking(events) == "The capital of France is Paris."
    assert _emitter_client_text(events) == ""
    thinking_start = next(e for e in events if '"type": "thinking"' in e)
    # signature is part of the Anthropic thinking-block shape; strict stream
    # decoders reject the block start without it.
    assert '"signature": ""' in thinking_start


def test_anthropic_emitter_splits_think_from_answer():
    # A reasoning-then-answer reply: the trace streams as a thinking block, the
    # answer as a separate text block, tags consumed by the splitter.
    emitter = AnthropicStreamEmitter()
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "<think>Thinking."})
    events += emitter.feed({"type": "content", "text": "<think>Thinking.</think>Answer."})
    events += emitter.finish()

    assert _emitter_client_thinking(events) == "Thinking."
    assert _emitter_client_text(events) == "Answer."


def test_anthropic_emitter_parse_think_off_keeps_literal_tags():
    # A non-reasoning model quoting <think> markup (e.g. an XML example) must
    # deliver it verbatim as text, not have it consumed into a thinking block.
    emitter = AnthropicStreamEmitter(parse_think = False)
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "Use <think>like this</think> tags."})
    events += emitter.finish()

    assert _emitter_client_thinking(events) == ""
    assert _emitter_client_text(events) == "Use <think>like this</think> tags."


def test_think_parsing_expected_gates_on_capability_and_request():
    from routes.inference import _think_parsing_expected

    class _Backend:
        def __init__(
            self,
            supports = True,
            always_on = False,
            default = True,
        ):
            self.supports_reasoning = supports
            self.reasoning_always_on = always_on
            self.reasoning_default = default

    # Non-reasoning model never parses; always-on always does.
    assert _think_parsing_expected(_Backend(supports = False), _basic_payload()) is False
    assert (
        _think_parsing_expected(_Backend(supports = False, always_on = True), _basic_payload()) is True
    )
    # Request-level off wins on a switchable model.
    assert _think_parsing_expected(_Backend(), _basic_payload(enable_thinking = False)) is False
    assert _think_parsing_expected(_Backend(), _basic_payload(reasoning_effort = "none")) is False
    # Unspecified follows the template default; explicit on parses.
    assert _think_parsing_expected(_Backend(default = False), _basic_payload()) is False
    assert _think_parsing_expected(_Backend(), _basic_payload()) is True
    assert _think_parsing_expected(_Backend(), _basic_payload(enable_thinking = True)) is True

    # Effort-dial templates (gpt-oss) map enable_thinking=False to a
    # low-but-thinking effort; the gate must follow the RESOLVED kwargs and
    # keep parsing on, disabling only for a genuine "none".
    class _EffortBackend(_Backend):
        def _request_reasoning_kwargs(self, enable_thinking, reasoning_effort, preserve_thinking):
            if reasoning_effort == "none":
                return {"reasoning_effort": "none"}
            return {"reasoning_effort": "low" if enable_thinking is False else "high"}

    assert _think_parsing_expected(_EffortBackend(), _basic_payload(enable_thinking = False)) is True
    assert (
        _think_parsing_expected(_EffortBackend(), _basic_payload(reasoning_effort = "none")) is False
    )


def test_anthropic_reasoning_args_maps_effort_only_to_enable_thinking():
    # Effort-only requests must drive enable_thinking-style templates the same
    # way /v1/responses does: "none" is off, a named level is on. Generation
    # and the parsing gate read these same args, so they cannot disagree.
    from routes.inference import _anthropic_reasoning_args

    assert (
        _anthropic_reasoning_args(_basic_payload(reasoning_effort = "none"))["enable_thinking"]
        is False
    )
    assert (
        _anthropic_reasoning_args(_basic_payload(reasoning_effort = "high"))["enable_thinking"]
        is True
    )
    assert _anthropic_reasoning_args(_basic_payload())["enable_thinking"] is None
    # An explicit boolean always wins; a contradictory effort is removed so
    # sampling and model-specific template resolution see the same request.
    assert _anthropic_reasoning_args(
        _basic_payload(enable_thinking = True, reasoning_effort = "none")
    ) == {
        "enable_thinking": True,
        "reasoning_effort": None,
        "preserve_thinking": None,
    }
    assert _anthropic_reasoning_args(
        _basic_payload(enable_thinking = False, reasoning_effort = "high")
    ) == {
        "enable_thinking": False,
        "reasoning_effort": None,
        "preserve_thinking": None,
    }


# thinking x reasoning_effort, every combination. The request model documents
# "[x-unsloth] reasoning controls ... win over `thinking` when both are
# present", so reasoning_effort must be resolved BEFORE the native block: an
# effort decides whenever it is sent, and `thinking` only speaks when neither
# x-unsloth control did. Pinned as a full cross product so the precedence
# cannot silently flip back -- reading resolved_enable_thinking() first made
# `thinking: enabled` + `reasoning_effort: "none"` still think on Qwen3.
_THINKING_EFFORT_MATRIX = [
    # (thinking, reasoning_effort, expected enable_thinking)
    (None, None, None),
    (None, "none", False),
    (None, "low", True),
    (None, "high", True),
    ("enabled", None, True),
    ("enabled", "none", False),
    ("enabled", "low", True),
    ("enabled", "high", True),
    ("disabled", None, False),
    ("disabled", "none", False),
    ("disabled", "low", True),
    ("disabled", "high", True),
]


@pytest.mark.parametrize("thinking_type, effort, expected", _THINKING_EFFORT_MATRIX)
def test_reasoning_effort_outranks_native_thinking(thinking_type, effort, expected):
    from routes.inference import _anthropic_reasoning_args

    fields = {}
    if thinking_type is not None:
        fields["thinking"] = {"type": thinking_type}
    if effort is not None:
        fields["reasoning_effort"] = effort
    args = _anthropic_reasoning_args(_basic_payload(**fields))

    assert args["enable_thinking"] is expected
    # The raw effort still reaches effort-dial templates untouched.
    assert args["reasoning_effort"] == effort
    # Precedence resolution must not touch preserve_thinking (three-valued:
    # None keeps llama-server on the load-time --chat-template-kwargs).
    assert args["preserve_thinking"] is None


@pytest.mark.parametrize("thinking_type, effort, expected", _THINKING_EFFORT_MATRIX)
def test_thinking_effort_matrix_reaches_enable_thinking_templates(thinking_type, effort, expected):
    """The matrix as the chat_template_kwargs a Qwen3-style template receives.

    Plain ``enable_thinking`` templates have no effort dial, so
    _request_reasoning_kwargs reads the boolean only -- if the effort loses to
    `thinking` upstream it is dropped here with nothing downstream to recover
    it, and a request that asked for no reasoning gets reasoning anyway.
    """
    from routes.inference import _anthropic_reasoning_args, _reasoning_template_kwargs

    class _QwenStyleBackend:
        """Mirrors LlamaCppBackend._request_reasoning_kwargs for reasoning_style
        'enable_thinking': the boolean is the only dial, effort is ignored."""

        def _request_reasoning_kwargs(self, enable_thinking, reasoning_effort, preserve_thinking):
            kwargs = {}
            if enable_thinking is not None:
                kwargs["enable_thinking"] = enable_thinking
            if preserve_thinking is not None:
                kwargs["preserve_thinking"] = preserve_thinking
            return kwargs or None

    fields = {}
    if thinking_type is not None:
        fields["thinking"] = {"type": thinking_type}
    if effort is not None:
        fields["reasoning_effort"] = effort
    args = _anthropic_reasoning_args(_basic_payload(**fields))
    resolved = _reasoning_template_kwargs(
        _QwenStyleBackend(),
        args["enable_thinking"],
        args["reasoning_effort"],
        args["preserve_thinking"],
    )

    if expected is None:
        assert resolved is None
    else:
        assert resolved == {"enable_thinking": expected}


def test_x_unsloth_enable_thinking_still_outranks_effort_and_thinking():
    # Precedence WITHIN the x-unsloth group is unchanged: the explicit boolean
    # is the most specific control and beats both the effort dial and the
    # native block.
    from routes.inference import _anthropic_reasoning_args

    args = _anthropic_reasoning_args(
        _basic_payload(
            thinking = {"type": "disabled"},
            enable_thinking = True,
            reasoning_effort = "none",
        )
    )
    assert args["enable_thinking"] is True
    args = _anthropic_reasoning_args(
        _basic_payload(
            thinking = {"type": "enabled"},
            enable_thinking = False,
            reasoning_effort = "high",
        )
    )
    assert args["enable_thinking"] is False


def test_replayed_thinking_preserved_only_when_requested():
    from core.inference.anthropic_compat import anthropic_messages_to_openai

    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "prior trace", "signature": ""},
                {"type": "text", "text": "the answer"},
            ],
        },
    ]
    dropped = anthropic_messages_to_openai(messages)
    assert "reasoning_content" not in dropped[-1]
    kept = anthropic_messages_to_openai(messages, preserve_thinking = True)
    assert kept[-1]["reasoning_content"] == "prior trace"
    assert kept[-1]["content"] == "the answer"


def test_anthropic_emitter_only_leading_think_is_reasoning():
    # Genuine reasoning is only ever a single leading <think> block; a model
    # quoting the tag mid-answer (an XML example) keeps it as literal text,
    # even with parsing enabled.
    emitter = AnthropicStreamEmitter()
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "<think>why</think>Use "})
    events += emitter.feed(
        {"type": "content", "text": "<think>why</think>Use <think>like this</think> tags."}
    )
    events += emitter.finish()

    assert _emitter_client_thinking(events) == "why"
    assert _emitter_client_text(events) == "Use <think>like this</think> tags."


def test_anthropic_emitter_no_leading_think_keeps_all_tags_literal():
    emitter = AnthropicStreamEmitter()
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "Wrap it in <think>"})
    events += emitter.feed({"type": "content", "text": "Wrap it in <think>...</think> please."})
    events += emitter.finish()

    assert _emitter_client_thinking(events) == ""
    assert _emitter_client_text(events) == "Wrap it in <think>...</think> please."


def test_split_think_segments_only_parses_leading_block():
    from routes.inference import _split_think_segments
    assert _split_think_segments("<think>why</think>Use <think>x</think> tags.") == [
        ("thinking", "why"),
        ("text", "Use <think>x</think> tags."),
    ]
    assert _split_think_segments("Use <think>x</think> tags.") == [
        ("text", "Use <think>x</think> tags.")
    ]


def test_anthropic_emitter_keeps_embedded_close_tag_in_trace():
    # Genuine reasoning ABOUT the </think> syntax contains the literal tag; the
    # generator-recorded length keeps the whole trace in the thinking block and
    # the real closing marker never leaks into the answer.
    trace = "the tag </think> ends a block"
    prov = {"wrapped": 1, "wraps": [{"len": 0}]}
    emitter = AnthropicStreamEmitter(think_provenance = prov)
    events = emitter.start("msg_1", "m")
    prov["wraps"][0]["len"] = len("the tag </think>")
    events += emitter.feed({"type": "content", "text": "<think>the tag </think>"})
    prov["wraps"][0]["len"] = len(trace)
    events += emitter.feed({"type": "content", "text": f"<think>{trace}"})
    events += emitter.feed({"type": "content", "text": f"<think>{trace}</think>Answer."})
    events += emitter.finish()

    assert _emitter_client_thinking(events) == trace
    assert _emitter_client_text(events) == "Answer."


def test_split_think_segments_wrap_length_beats_embedded_tag():
    from routes.inference import _split_think_segments

    trace = "quote </think> inside"
    text = f"<think>{trace}</think>Visible."
    assert _split_think_segments(text, {"len": len(trace)}) == [
        ("thinking", trace),
        ("text", "Visible."),
    ]
    # Without provenance the first marker still closes (heuristic fallback).
    assert _split_think_segments(text) == [
        ("thinking", "quote "),
        ("text", " inside</think>Visible."),
    ]


def test_anthropic_emitter_holds_back_partial_think_tag():
    # A tag split across deltas must not leak fragments into the wrong block.
    emitter = AnthropicStreamEmitter()
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": "<th"})
    events += emitter.feed({"type": "content", "text": "<think>Deep"})
    events += emitter.feed({"type": "content", "text": "<think>Deep</th"})
    events += emitter.feed({"type": "content", "text": "<think>Deep</think>Out"})
    events += emitter.finish()

    assert _emitter_client_thinking(events) == "Deep"
    assert _emitter_client_text(events) == "Out"


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

        # the date rides on the system prompt now, not the nudge.
        assert "The current date is " not in nudge
        assert nudge.startswith("Tools are available when they materially improve")
        assert "prefer using tools rather than answering from memory" not in nudge
        assert "fetch its full content by calling web_search with the url parameter" in nudge
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
        assert _build_tool_action_nudge(tools = [], model_name = "Llama-3.1-8B-Instruct") == ""


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
        assert parts[1] == {"type": "image_url", "image_url": {"url": "https://x/y.png"}}

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

    @pytest.mark.parametrize(
        ("type_", "name", "kind"),
        [
            ("bash_20250124", "bash", "bash"),
            ("text_editor_20250728", "str_replace_based_edit_tool", "text_editor"),
            ("computer_20251124", "computer", "computer"),
            ("memory_20250818", "memory", "memory"),
        ],
    )
    def test_schema_client_tools_are_converted_to_openai_functions(self, type_, name, kind):
        tool = {"type": type_, "name": name}

        [result] = anthropic_tools_to_openai([tool])

        assert anthropic_schema_client_tool_kind(tool) == kind
        assert result["function"]["name"] == name
        assert result["function"]["parameters"]["type"] == "object"

    @pytest.mark.parametrize(
        ("type_", "supports_undo"),
        [
            ("text_editor_20241022", True),
            ("text_editor_20250124", True),
            ("text_editor_20250429", False),
            ("text_editor_20250728", False),
        ],
    )
    def test_text_editor_commands_follow_tool_version(self, type_, supports_undo):
        [result] = anthropic_tools_to_openai(
            [{"type": type_, "name": "str_replace_based_edit_tool"}]
        )

        commands = result["function"]["parameters"]["properties"]["command"]["enum"]
        assert ("undo_edit" in commands) is supports_undo

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
        tool = AnthropicTool(name = "test", description = "desc", input_schema = {"type": "object"})
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
    def test_start_emits_message_start_only(self):
        # Content blocks open lazily on first output, so a turn that begins
        # with thinking gets a thinking block first, not an empty text block.
        e = AnthropicStreamEmitter()
        events = e.start("msg_123", "test-model")
        assert len(events) == 1
        assert "message_start" in events[0]

    def test_content_delta_opens_text_block_and_emits_delta(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        events = e.feed({"type": "content", "text": "Hello"})
        assert len(events) == 2
        assert "content_block_start" in events[0]
        assert '"type": "text"' in events[0]
        parsed = json.loads(events[1].split("data: ")[1])
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

        first_payloads = [json.loads(event.split("data: ")[1]) for event in first_events]
        second_payloads = [json.loads(event.split("data: ")[1]) for event in second_events]

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
                    "partial_json": json.dumps({"code": "<!doctype html><html></html>"}),
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
        # content_block_stop (tool) + tool_result; the next text opens its own block.
        assert len(events) == 2
        assert "content_block_stop" in events[0]
        assert "tool_result" in events[1]
        parsed = json.loads(events[1].split("data: ")[1])
        assert parsed["content"] == "done"
        assert parsed["tool_use_id"] == tool_use_id

    def test_finish_emits_stop_events(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed({"type": "content", "text": "Hi"})
        events = e.finish("end_turn")
        # content_block_stop + message_delta + message_stop
        assert len(events) == 3
        assert "content_block_stop" in events[0]
        assert "message_delta" in events[1]
        assert "end_turn" in events[1]
        assert "message_stop" in events[2]

    def test_finish_without_content_skips_block_stop(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        events = e.finish("end_turn")
        assert len(events) == 2
        assert "message_delta" in events[0]
        assert "message_stop" in events[1]

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

        assert len(start_events) == 1
        assert len(content_events) == 2
        assert meta_events == []
        assert len(end_events) == 3

    def test_block_index_increments(self):
        e = AnthropicStreamEmitter()
        e.start("msg_1", "m")
        e.feed({"type": "content", "text": "Before"})
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
        e.feed({"type": "content", "text": "After"})
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
        # After tool_end, prev_text should be reset; the content opens a fresh
        # text block and diffs against an empty baseline.
        events = e.feed({"type": "content", "text": "After tool"})
        assert "content_block_start" in events[0]
        parsed = json.loads(events[1].split("data: ")[1])
        assert parsed["delta"]["text"] == "After tool"


# =====================================================================
# Non-streaming tool response tests
# =====================================================================


class TestAnthropicToolNonStreaming:
    @pytest.mark.parametrize(
        ("helper", "event"),
        [
            pytest.param(
                _anthropic_tool_non_streaming,
                {"type": "content", "text": "ok"},
                id = "tools",
            ),
            pytest.param(_anthropic_plain_non_streaming, "ok", id = "plain"),
        ],
    )
    def test_complete_response_build_keeps_event_loop_responsive(self, helper, event):
        loop_thread = threading.current_thread()
        generator_threads = []

        def _run_gen():
            generator_threads.append(threading.current_thread())
            time.sleep(0.08)
            yield event

        async def _run():
            task = asyncio.create_task(helper(_run_gen, "msg_1", "m"))
            await asyncio.sleep(0)
            heartbeat_ticks = 0
            while not task.done():
                heartbeat_ticks += 1
                await asyncio.sleep(0.005)
            return await task, heartbeat_ticks

        response, heartbeat_ticks = asyncio.run(_run())

        assert response.status_code == 200
        assert heartbeat_ticks > 0
        assert len(generator_threads) == 1
        assert generator_threads[0] is not loop_thread

    def test_tool_event_reduction_keeps_event_loop_responsive(self, monkeypatch):
        import routes.inference as inf_mod

        reduction_threads = []
        real_strip = inf_mod._strip_tool_xml_for_display

        def _slow_strip(*args, **kwargs):
            reduction_threads.append(threading.current_thread())
            time.sleep(0.08)
            return real_strip(*args, **kwargs)

        monkeypatch.setattr(inf_mod, "_strip_tool_xml_for_display", _slow_strip)

        def _run_gen():
            yield {"type": "content", "text": "ok"}

        async def _run():
            task = asyncio.create_task(_anthropic_tool_non_streaming(_run_gen, "msg_1", "m"))
            await asyncio.sleep(0)
            heartbeat_ticks = 0
            while not task.done():
                heartbeat_ticks += 1
                await asyncio.sleep(0.005)
            return await task, heartbeat_ticks

        response, heartbeat_ticks = asyncio.run(_run())

        assert response.status_code == 200
        assert heartbeat_ticks > 0
        assert len(reduction_threads) == 1
        assert reduction_threads[0] is not threading.current_thread()
        assert reduction_threads[0].daemon is True

    @pytest.mark.parametrize(
        ("helper", "event"),
        [
            pytest.param(
                _anthropic_tool_non_streaming,
                {"type": "content", "text": "ok"},
                id = "tools",
            ),
            pytest.param(_anthropic_plain_non_streaming, "ok", id = "plain"),
        ],
    )
    def test_cancellation_waits_for_generator_worker(self, helper, event):
        generator_started = threading.Event()
        generator_stopped = threading.Event()
        cancel_event = threading.Event()

        def _run_gen():
            generator_started.set()
            assert cancel_event.wait(1.0)
            generator_stopped.set()
            yield event

        async def _cancel_generation():
            task = asyncio.create_task(helper(_run_gen, "msg_1", "m", cancel_event = cancel_event))
            assert await asyncio.to_thread(generator_started.wait, 1.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancel_event.is_set()
            assert generator_stopped.is_set()

        asyncio.run(_cancel_generation())

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
        tool_blocks = [block for block in body["content"] if block["type"] == "tool_use"]

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

        tools = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]
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
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"cmd'}}]}}
                ]
            }
        )
        events2 = e.feed_chunk(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '": "ls"}'}}]}}
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

    @pytest.mark.parametrize(
        "reasoning_kwargs, expected",
        [
            ({}, None),
            ({"enable_thinking": True}, {"enable_thinking": True}),
            ({"enable_thinking": False}, {"enable_thinking": False}),
            (
                {"enable_thinking": True, "reasoning_effort": "high"},
                {"enable_thinking": True, "reasoning_effort": "high"},
            ),
        ],
    )
    def test_stream_forwards_reasoning_to_llama_server(
        self, monkeypatch, reasoning_kwargs, expected
    ):
        """Without this the reasoning request is dropped and the model stays in
        its load-time default -- thinking can never be switched on."""
        import routes.inference as inf_mod

        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content.decode())
            content = 'data: {"choices": [{"delta": {"content": "hi"}}]}\n\ndata: [DONE]\n\n'
            return httpx.Response(
                200,
                content = content.encode(),
                headers = {"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        real_async_client = httpx.AsyncClient

        def _client(*args, **kwargs):
            return real_async_client(transport = transport, timeout = kwargs.get("timeout", 600))

        monkeypatch.setattr(inf_mod.httpx, "AsyncClient", _client)
        # Echo the request kwargs back the way the real backend does, so the
        # test covers the route wiring rather than the backend's style logic.
        backend = SimpleNamespace(
            base_url = "http://llama.test",
            context_length = 4096,
            count_chat_tokens = lambda *args, **kwargs: 2,
            _request_reasoning_kwargs = lambda et, re_, pt: (
                {
                    k: v
                    for k, v in (
                        ("enable_thinking", et),
                        ("reasoning_effort", re_),
                        ("preserve_thinking", pt),
                    )
                    if v is not None
                }
                or None
            ),
        )

        async def run():
            response = await _anthropic_passthrough_stream(
                self._Request(),
                threading.Event(),
                backend,
                [{"role": "user", "content": "hi"}],
                [{"type": "function", "function": {"name": "lookup", "parameters": {}}}],
                0.7,
                0.95,
                20,
                16,
                "msg_1",
                "test-model",
                **reasoning_kwargs,
            )
            return await self._collect(response)

        asyncio.run(run())
        assert captured["body"].get("chat_template_kwargs") == expected


class TestReasoningContentReachesTheClient:
    """llama-server puts the thinking trace in `reasoning_content`, not `content`.
    Reading only `content` drops it and the model looks like it never thought --
    which is what every Claude Code turn hit, since tool turns always split."""

    def test_stream_emits_thinking_block(self):
        from core.inference.anthropic_compat import AnthropicPassthroughEmitter

        emitter = AnthropicPassthroughEmitter()
        emitter.start("msg_1", "test-model")
        out = emitter.feed_chunk({"choices": [{"delta": {"reasoning_content": "step one"}}]})
        out += emitter.feed_chunk({"choices": [{"delta": {"content": "the answer"}}]})
        blob = "".join(out)

        assert '"type": "thinking"' in blob
        assert "thinking_delta" in blob
        assert "step one" in blob
        # Thinking must open before the text block, the order Anthropic defines.
        assert blob.index("thinking_delta") < blob.index("text_delta")

    def test_stream_thinking_only_still_emits(self):
        """A reasoning-only reply must not come back as an empty message."""
        from core.inference.anthropic_compat import AnthropicPassthroughEmitter

        emitter = AnthropicPassthroughEmitter()
        emitter.start("msg_1", "test-model")
        blob = "".join(emitter.feed_chunk({"choices": [{"delta": {"reasoning_content": "only"}}]}))
        assert "thinking_delta" in blob and "only" in blob

    def test_stream_reasoning_reconstructed_as_text_when_thinking_off(self):
        """Thinking effectively off: llama-server may still shunt a literal
        <think> example into reasoning_content; the client asked for those
        bytes, so they come back as text with the tags restored."""
        from core.inference.anthropic_compat import AnthropicPassthroughEmitter

        emitter = AnthropicPassthroughEmitter(reasoning_as_thinking = False)
        emitter.start("msg_1", "test-model")
        out = emitter.feed_chunk({"choices": [{"delta": {"reasoning_content": "like this"}}]})
        out += emitter.feed_chunk({"choices": [{"delta": {"content": " is the syntax"}}]})
        out += emitter.finish()
        blob = "".join(out)

        assert "thinking_delta" not in blob
        text = "".join(
            json.loads(stripped[len("data: ") :])["delta"].get("text", "")
            for line in out
            for stripped in (part.strip() for part in line.split("\n"))
            if stripped.startswith("data: ") and '"text_delta"' in stripped
        )
        assert text == "<think>like this</think> is the syntax"

    def test_stream_reconstruction_closes_before_same_chunk_content(self):
        """One chunk can carry the final reasoning fragment AND content; the
        closing tag must land between them, not after."""
        from core.inference.anthropic_compat import AnthropicPassthroughEmitter

        emitter = AnthropicPassthroughEmitter(reasoning_as_thinking = False)
        emitter.start("msg_1", "test-model")
        out = emitter.feed_chunk({"choices": [{"delta": {"reasoning_content": "like this"}}]})
        out += emitter.feed_chunk(
            {"choices": [{"delta": {"reasoning_content": " too", "content": "Answer"}}]}
        )
        out += emitter.finish()

        text = "".join(
            json.loads(stripped[len("data: ") :])["delta"].get("text", "")
            for line in out
            for stripped in (part.strip() for part in line.split("\n"))
            if stripped.startswith("data: ") and '"text_delta"' in stripped
        )
        assert text == "<think>like this too</think>Answer"

    def test_non_streaming_builds_thinking_block(self):
        from models.inference import (
            AnthropicMessagesResponse,
            AnthropicResponseTextBlock,
            AnthropicResponseThinkingBlock,
        )

        resp = AnthropicMessagesResponse(
            model = "m",
            content = [
                AnthropicResponseThinkingBlock(thinking = "because 2+2"),
                AnthropicResponseTextBlock(text = "4"),
            ],
        )
        dumped = resp.model_dump()
        assert [b["type"] for b in dumped["content"]] == ["thinking", "text"]
        assert dumped["content"][0]["thinking"] == "because 2+2"
        # signature is part of Anthropic's shape; empty is fine, missing is not.
        assert "signature" in dumped["content"][0]


class TestAnthropicReasoningArgs:
    """`/v1/messages` must accept Anthropic's `thinking` block and the
    x-unsloth reasoning fields instead of silently swallowing them."""

    @staticmethod
    def _payload(**kwargs):
        from models.inference import AnthropicMessagesRequest
        return AnthropicMessagesRequest(
            model = "m",
            max_tokens = 16,
            messages = [{"role": "user", "content": "hi"}],
            **kwargs,
        )

    @pytest.mark.parametrize(
        "kwargs, expected",
        [
            ({}, None),
            ({"thinking": {"type": "enabled", "budget_tokens": 600}}, True),
            ({"thinking": {"type": "disabled"}}, False),
            # Anthropic's adaptive tiers: unknown types must mean "think", never 400.
            ({"thinking": {"type": "adaptive"}}, True),
            ({"thinking": {"type": "auto"}}, True),
            ({"enable_thinking": True}, True),
            ({"enable_thinking": False}, False),
            # x-unsloth field wins, mirroring enable_tools precedence.
            ({"thinking": {"type": "disabled"}, "enable_thinking": True}, True),
        ],
    )
    def test_resolved_enable_thinking(self, kwargs, expected):
        assert self._payload(**kwargs).resolved_enable_thinking() is expected

    def test_reasoning_args_reach_the_generators(self):
        from routes.inference import _anthropic_reasoning_args
        payload = self._payload(
            thinking = {"type": "enabled"},
            reasoning_effort = "high",
            preserve_thinking = True,
        )
        assert _anthropic_reasoning_args(payload) == {
            "enable_thinking": True,
            "reasoning_effort": "high",
            "preserve_thinking": True,
        }

    @pytest.mark.parametrize("thinking_type", ["adaptive", "auto", "high", "future_tier"])
    def test_unknown_thinking_type_never_400s(self, thinking_type):
        """A strict Literal here regressed real Claude Code traffic to a 400:
        `thinking.type: Input should be 'enabled' or 'disabled'`."""
        payload = self._payload(thinking = {"type": thinking_type})
        assert payload.resolved_enable_thinking() is True

    def test_budget_tokens_accepted_not_rejected(self):
        """Claude Code always sends budget_tokens; llama-server has no budget,
        so it must be ignored rather than 400'd."""
        payload = self._payload(thinking = {"type": "enabled", "budget_tokens": 4096})
        assert payload.thinking.budget_tokens == 4096
        assert payload.resolved_enable_thinking() is True

    def test_replayed_thinking_blocks_are_accepted(self):
        """Anthropic's tool-use protocol makes clients replay thinking blocks
        with tool results. Rejecting them 422s turn 2 of every thinking session."""
        from core.inference.anthropic_compat import anthropic_messages_to_openai
        from models.inference import AnthropicMessagesRequest

        payload = AnthropicMessagesRequest(
            model = "m",
            max_tokens = 16,
            messages = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "let me check", "signature": ""},
                        {"type": "redacted_thinking", "data": "opaque"},
                        {"type": "tool_use", "id": "toolu_1", "name": "ls", "input": {}},
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"}],
                },
            ],
        )
        converted = anthropic_messages_to_openai([m.model_dump() for m in payload.messages])
        assistant = next(m for m in converted if m["role"] == "assistant")
        # Thinking is dropped from the prompt; the tool call survives.
        assert "thinking" not in json.dumps(assistant)
        assert assistant["tool_calls"][0]["function"]["name"] == "ls"


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

    # Pinned off by default so prompt assertions do not depend on the host's stored setting;
    # the date's own behaviour on this route is covered in test_current_date_prompt_settings.
    monkeypatch.setattr(inf_mod, "current_date_prompt_line", lambda **_kwargs: "")

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


@pytest.fixture(autouse = True)
def _reset_admission_queues():
    # The admission queue is process-global; isolate the shared "llama-server" key
    # so one test's leftover reservation can't stall the next.
    from core.inference.llama_admission import reset_llama_admission_queues

    reset_llama_admission_queues()
    yield
    reset_llama_admission_queues()


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

    def test_plain_non_streaming_states_the_current_date(self, monkeypatch):
        # /v1/messages used to get the date from the tool nudge; it now rides the system turn,
        # so this route needs its own coverage or the date silently disappears from it.
        import routes.inference as inf_mod

        backend = _mock_backend(monkeypatch, context_length = 2048)
        monkeypatch.setattr(
            inf_mod,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-08-15.",
        )
        _drive(anthropic_messages(_basic_payload(), request = self._Request(), current_subject = "t"))

        [(_path, kwargs)] = backend.calls
        assert kwargs["messages"][0] == {
            "role": "system",
            "content": "The current date is 2026-08-15.",
        }

    def test_plain_non_streaming_records_api_monitor_entry(self, monkeypatch):
        import routes.inference as inf_mod

        _mock_backend(monkeypatch, context_length = 2048)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        payload = _basic_payload()

        response = _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))

        assert response.status_code == 200
        [entry] = monitor.snapshot()
        assert entry["endpoint"] == "/v1/messages"
        assert entry["status"] == "completed"
        assert entry["model"] == "test-model"
        assert entry["prompt_preview"] == "user: hi"
        assert entry["reply_preview"] == "ok"
        assert entry["context_length"] == 2048
        assert monitor.active_count() == 0

    @pytest.mark.parametrize("stream", [False, True])
    @pytest.mark.parametrize("with_tools", [False, True])
    def test_reasoning_only_output_is_not_duplicated(self, monkeypatch, stream, with_tools):
        reasoning = "The capital of France is Paris."

        def _gen_plain(**kwargs):
            assert kwargs["promote_reasoning_only"] is False
            # Mirror the real generator: record that the leading <think> was
            # wrapped from reasoning_content, not literal model text.
            prov = kwargs.get("reasoning_provenance")
            if prov is not None:
                prov["wrapped"] = prov.get("wrapped", 0) + 1
            yield f"<think>{reasoning}"
            yield f"<think>{reasoning}</think>"

        def _gen_tools(**kwargs):
            assert kwargs["promote_reasoning_only"] is False
            prov = kwargs.get("reasoning_provenance")
            if prov is not None:
                prov["wrapped"] = prov.get("wrapped", 0) + 1
            yield {"type": "content", "text": f"<think>{reasoning}"}
            yield {"type": "content", "text": f"<think>{reasoning}</think>"}

        _mock_backend(
            monkeypatch,
            generate_chat_completion = _gen_plain,
            generate_chat_completion_with_tools = _gen_tools,
        )
        payload_fields = {"stream": stream}
        if with_tools:
            payload_fields.update(
                {
                    "enable_tools": True,
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                }
            )
        payload = _basic_payload(**payload_fields)

        response = _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))
        if stream:
            body = self._sse_blob(self._consume_response(response))
            assert body.count(reasoning) == 1
            assert "<think>" not in body  # markup split into a typed thinking block
        else:
            body = json.loads(response.body)
            assert body["content"][0]["type"] == "thinking"
            assert body["content"][0]["thinking"] == reasoning

    @pytest.mark.parametrize("stream", [False, True])
    def test_literal_leading_think_without_provenance_stays_text(self, monkeypatch, stream):
        # The model answered with literal <think> markup (user asked for it) and
        # produced no genuine reasoning: the generator recorded no wrap, so the
        # markup must come back as text, not be consumed into a thinking block.
        literal = "<think>like this</think>"

        def _gen_plain(**kwargs):
            assert kwargs.get("reasoning_provenance") is not None
            yield literal

        _mock_backend(monkeypatch, generate_chat_completion = _gen_plain)
        payload = _basic_payload(stream = stream)

        response = _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))
        if stream:
            body = self._sse_blob(self._consume_response(response))
            assert '"thinking_delta"' not in body
            assert literal in body
        else:
            body = json.loads(response.body)
            assert body["content"][0]["type"] == "text"
            assert body["content"][0]["text"] == literal

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

        response = _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))

        assert response.status_code == 200
        [entry] = monitor.snapshot()
        assert entry["status"] == "completed"
        assert entry["reply_preview"] == 'Tool call: lookup({"query": "weather"})'

    def test_plain_streaming_records_active_and_completed_monitor_entry(self, monkeypatch):
        import routes.inference as inf_mod

        _mock_backend(monkeypatch, context_length = 2048)
        monitor = ApiMonitor(max_entries = 3)
        monkeypatch.setattr(inf_mod, "api_monitor", monitor)
        payload = _basic_payload(stream = True)

        response = _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))

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
        monkeypatch.setattr(inf_mod, "_anthropic_plain_stream", _cancelled_before_response)
        payload = _basic_payload(stream = True)

        with pytest.raises(asyncio.CancelledError):
            _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))

        [entry] = monitor.snapshot()
        assert entry["status"] == "cancelled"
        assert monitor.active_count() == 0

    @staticmethod
    def _sse_blob(chunks):
        # StreamingResponse may hand back str or already-encoded bytes.
        return "".join(c.decode() if isinstance(c, (bytes, bytearray)) else c for c in chunks)

    def test_plain_streaming_unclassified_error_emits_error_event(self, monkeypatch):
        # An unclassified mid-stream failure must surface as an SSE `error` event
        # and stop, not a message_stop that masks a truncated turn as clean.
        def _gen_boom(**_kwargs):
            yield "partial"
            raise RuntimeError("llama-server crashed mid-decode")

        _mock_backend(monkeypatch, generate_chat_completion = _gen_boom)
        payload = _basic_payload(stream = True)

        response = _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))
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

        response = _drive(anthropic_messages(payload, request = self._Request(), current_subject = "t"))
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

    def test_explicit_server_loop_and_client_tools_rejected_with_400(self, monkeypatch):
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            enable_tools = True,
            tools = [{"name": "Write", "input_schema": {"type": "object"}}],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "Mixing Anthropic server tools" in exc.value.detail

    def test_explicit_server_loop_and_schema_client_tools_rejected_with_400(self, monkeypatch):
        _mock_backend(monkeypatch)
        payload = _basic_payload(
            enable_tools = True,
            tools = [{"type": "bash_20250124", "name": "bash"}],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "Mixing Anthropic server tools" in exc.value.detail

    def test_process_tool_policy_does_not_steal_schema_client_tools(self, monkeypatch):
        import routes.inference as inf_mod
        from fastapi.responses import JSONResponse

        backend = _mock_backend(monkeypatch)
        captured = {}

        async def _passthrough(*args, **kwargs):
            captured["tools"] = args[2]
            return JSONResponse(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "test-model",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

        monkeypatch.setattr(inf_mod, "_anthropic_passthrough_non_streaming", _passthrough)
        set_tool_policy(True)
        payload = _basic_payload(tools = [{"type": "bash_20250124", "name": "bash"}])

        _drive(anthropic_messages(payload, request = None, current_subject = "t"))

        assert backend.calls == []
        assert captured["tools"][0]["function"]["name"] == "bash"

    @pytest.mark.parametrize("permission_mode", [None, "ask"])
    @pytest.mark.parametrize(
        ("tool_policy", "enable_tools"),
        [(True, None), (False, True)],
    )
    def test_process_tool_policy_does_not_steal_client_tools(
        self, monkeypatch, permission_mode, tool_policy, enable_tools
    ):
        """A server-wide tool default must not replace Claude Code's own tools."""
        import routes.inference as inf_mod
        from fastapi.responses import JSONResponse

        backend = _mock_backend(monkeypatch)
        captured = {}

        async def _passthrough(*args, **kwargs):
            captured["tools"] = args[2]
            return JSONResponse(
                {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "ok"}],
                    "model": "test-model",
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            )

        monkeypatch.setattr(inf_mod, "_anthropic_passthrough_non_streaming", _passthrough)
        set_tool_policy(tool_policy)
        fields = {
            "tools": [
                {
                    "name": "Write",
                    "description": "Write a file",
                    "input_schema": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                    },
                }
            ],
        }
        if enable_tools is not None:
            fields["enable_tools"] = enable_tools
        if permission_mode is not None:
            fields["permission_mode"] = permission_mode
        payload = _basic_payload(**fields)

        _drive(anthropic_messages(payload, request = None, current_subject = "t"))

        assert backend.calls == []
        assert captured["tools"][0]["function"]["name"] == "Write"

    def test_mixed_rejected_when_client_tool_name_collides_with_server_alias(self, monkeypatch):
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

    def test_schema_client_tool_missing_name_rejected_with_400(self, monkeypatch):
        _mock_backend(monkeypatch)
        payload = _basic_payload(tools = [{"type": "bash_20250124"}])

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

    def test_alias_named_client_tool_without_schema_rejected_with_400(self, monkeypatch):
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

    def test_api_server_tool_request_keeps_the_current_date(self, monkeypatch):
        import routes.inference as inf_mod

        backend = _mock_backend(monkeypatch)
        monkeypatch.setattr(
            inf_mod,
            "current_date_prompt_line",
            lambda **_kwargs: "The current date is 2026-08-15.",
        )
        monkeypatch.setattr(inf_mod, "_request_is_internal_workflow", lambda _request: False)

        class ApiRequest(self._Request):
            headers = {"authorization": "Bearer sk-unsloth-test"}
            state = SimpleNamespace(skip_api_monitor = True)

        payload = _basic_payload(
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )
        _drive(anthropic_messages(payload, request = ApiRequest(), current_subject = "t"))

        call_kind, kwargs = backend.calls[0]
        assert call_kind == "tools"
        assert kwargs["messages"][0]["content"].startswith("The current date is 2026-08-15.\n\n")

    def test_server_tool_choice_alias_uses_the_selected_studio_name(self, monkeypatch):
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"type": "web_fetch_20250910", "name": "web_fetch"}],
            tool_choice = {"type": "tool", "name": "web_fetch"},
        )

        _drive(anthropic_messages(payload, request = None, current_subject = "t"))

        call_kind, kwargs = backend.calls[0]
        assert call_kind == "tools"
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "web_search"}}
        assert [tool["function"]["name"] for tool in kwargs["tools"]] == ["web_search"]

    def test_server_tool_choice_must_be_in_the_selected_catalog(self, monkeypatch):
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
            tool_choice = {"type": "tool", "name": "python"},
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))

        assert exc.value.status_code == 400
        assert "python" in exc.value.detail["error"]["message"]
        assert backend.calls == []

    def test_confirm_tool_calls_rejected_for_server_tools(self, monkeypatch):
        backend = _mock_backend(monkeypatch)
        payload = _basic_payload(
            confirm_tool_calls = True,
            tools = [{"type": "web_search_20250305", "name": "web_search"}],
        )

        with pytest.raises(HTTPException) as exc:
            _drive(anthropic_messages(payload, request = None, current_subject = "t"))
        assert exc.value.status_code == 400
        assert "confirm_tool_calls is not supported" in exc.value.detail["error"]["message"]
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

        # Reading this conversation's own archive is as read-only as the other two, and
        # is_potentially_unsafe_tool_call says so, so selecting it must not trip the gate.
        # Adding the schema to ALL_TOOLS without adding the name here made the Anthropic
        # selector pick it and the pre-switch guard reject the whole request with the
        # terminal/python message, on auto and on the omitted default alike.
        for extra in ({"permission_mode": "auto"}, {}):
            backend = _mock_backend(monkeypatch)
            payload = _basic_payload(
                enable_tools = True, enabled_tools = ["search_conversation"], **extra
            )
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
            _basic_payload(tools = safe_tools, enable_tools = True, enabled_tools = ["python"]),
        ):
            backend = _mock_backend(monkeypatch)
            with pytest.raises(HTTPException) as exc:
                _drive(anthropic_messages(local_payload, request = None, current_subject = "t"))
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
            {"tools": safe_tools, "enabled_tools": ["python"], "confirm_tool_calls": False},
            {"tools": safe_tools, "permission_mode": "ask", "confirm_tool_calls": False},
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

    def test_the_process_tool_default_alone_is_not_a_server_tool_selection(self, monkeypatch):
        """`unsloth studio run` resolves the policy to on unless --disable-tools. Reading that
        as "this request selected server tools" rejected every plain Messages request on a
        default server, and routing on it ran the local tool loop with terminal/python and no
        way to confirm them. A default is not a selection, in either direction."""
        import routes.inference as inf_mod
        from fastapi.responses import JSONResponse

        backend = _mock_backend(monkeypatch)

        async def _passthrough(*args, **kwargs):
            return JSONResponse({"type": "message", "content": []})

        monkeypatch.setattr(inf_mod, "_anthropic_passthrough_non_streaming", _passthrough)
        set_tool_policy(True)

        _drive(anthropic_messages(_basic_payload(), request = None, current_subject = "t"))
        assert backend.calls, "a plain chat request must still be served"
        path, kwargs = backend.calls[0]
        assert path == "plain", (
            "a tool-free request took the server-tool loop on the process default alone, so "
            "the model can call terminal/python with no confirmation channel"
        )
        assert not kwargs.get("tools")

        # An explicit ask still routes to the loop, with the mode that permits it.
        backend = _mock_backend(monkeypatch)
        _drive(
            anthropic_messages(
                _basic_payload(enable_tools = True, permission_mode = "off"),
                request = None,
                current_subject = "t",
            )
        )
        assert backend.calls[0][0] == "tools"

        # mcp_enabled is an ask on the OpenAI routes, which wire MCP discovery. This one does
        # not, and the request model is extra="allow" so the key does arrive: honouring it
        # would answer an MCP-only request with ALL_TOOLS' terminal/python under an MCP name.
        reset_tool_policy()
        backend = _mock_backend(monkeypatch)
        _drive(
            anthropic_messages(
                _basic_payload(mcp_enabled = True, permission_mode = "off"),
                request = None,
                current_subject = "t",
            )
        )
        assert backend.calls[0][0] == "plain"
        assert not backend.calls[0][1].get("tools")

        # The same default must not stop gating a request that does ask for server tools.
        for fields in (
            {"enable_tools": True},
            {"enable_tools": True, "permission_mode": "ask"},
            {"tools": [{"type": "terminal", "name": "terminal"}]},
        ):
            with pytest.raises(HTTPException) as exc:
                _drive(
                    anthropic_messages(_basic_payload(**fields), request = None, current_subject = "t")
                )
            assert exc.value.status_code == 400
            assert "no confirmation channel" in exc.value.detail["error"]["message"]

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
                    {"type": "thinking", "thinking": "secret reasoning", "signature": "s"},
                    {"type": "text", "text": "the answer"},
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                ],
            },
            {"role": "assistant", "content": None},  # tool-only turn serialized as null
        ],
    )
    # Known blocks parse as their typed models; replayed thinking is typed too.
    assert type(req.messages[1].content[0]).__name__ == "AnthropicThinkingBlock"
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
            messages = [{"role": "assistant", "content": [{"type": "tool_use", "name": "f"}]}],
        )


def test_user_thinking_block_rejected_not_silently_dropped():
    # Thinking blocks are typed for assistant replay only; the converter drops
    # them from user content, so accepting one there would lose the user turn.
    from pydantic import ValidationError
    for btype in ("thinking", "redacted_thinking"):
        with pytest.raises(ValidationError):
            AnthropicMessagesRequest(
                model = "x",
                max_tokens = 16,
                messages = [{"role": "user", "content": [{"type": btype}]}],
            )


def test_think_markup_split_preserves_text_verbatim():
    from routes.inference import _think_markup_to_blocks

    # Reasoning-free output passes through untouched, whitespace included.
    [block] = _think_markup_to_blocks("  indented\n  lines\n")
    assert block.text == "  indented\n  lines\n"

    # With markup, the trace and the answer keep their own bytes.
    thinking, text = _think_markup_to_blocks("<think>why</think>\n\n  answer\n")
    assert thinking.thinking == "why"
    assert text.text == "\n\n  answer\n"

    # Whitespace-only segments are dropped rather than emitted as empty blocks.
    [only_thinking] = _think_markup_to_blocks("<think>why</think>\n\n")
    assert only_thinking.thinking == "why"


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
                        "source": {"type": "base64", "media_type": "image/png", "data": "AA"},
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
    tool_use_starts = [c for c in chunks if "content_block_start" in c and '"tool_use"' in c]
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


def test_parallel_disabled_dropped_call_output_emits_rate_limited_keepalives(monkeypatch):
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
    tool_use_starts = [c for c in chunks if "content_block_start" in c and '"tool_use"' in c]
    assert len(tool_use_starts) == 1
    assert any("final answer" in c for c in chunks)


def test_plain_stream_emits_keepalive_during_prompt_stall(monkeypatch):
    """No-tool Anthropic stream must emit SSE keepalives while a long prompt
    prefill blocks next(gen), matching the tool stream (finding 5). The old
    single unbounded to_thread(next, ...) could sit silent past a proxy idle cap."""
    import threading as _threading
    import time as _time

    from routes import inference as inf_mod
    from routes.inference import _OPENAI_PASSTHROUGH_SSE_KEEPALIVE, _anthropic_plain_stream

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


def test_display_strip_keeps_provenance_trace_intact():
    # Genuine reasoning that quotes </think> and then rehearses an enabled call: the
    # think-aware cleaner closes the block at the quoted tag, so without provenance it
    # strips the rehearsal out of the trace while wrap["len"] still measures the full
    # one -- the split then eats the real terminator and the whole answer.
    from routes.inference import _ReasoningSpanGuard, _split_think_segments

    trace = (
        'emit </think> then <tool_call>{"name": "get_weather", "arguments": {}}</tool_call> ends it'
    )
    raw = f"<think>{trace}</think>It is sunny."
    prov = {"wrapped": 1, "wraps": [{"len": len(trace)}]}

    clean = _ReasoningSpanGuard(prov).strip(
        raw, auto_heal_tool_calls = True, enabled_tool_names = {"get_weather"}
    )
    assert clean == raw

    emitter = AnthropicStreamEmitter(think_provenance = prov)
    events = emitter.start("msg_1", "m")
    events += emitter.feed({"type": "content", "text": clean})
    events += emitter.finish()
    assert _emitter_client_thinking(events) == trace
    assert _emitter_client_text(events) == "It is sunny."
    assert _split_think_segments(clean, prov["wraps"][0]) == [
        ("thinking", trace),
        ("text", "It is sunny."),
    ]


def test_display_strip_protects_the_wrap_of_each_tool_turn():
    # A tool loop's second synthesis turn opens its own leading <think> backed by the NEXT
    # wrap entry, so the guard must advance with the emitter's ledger; measuring turn 2
    # against wraps[0] would put the boundary mid-trace and strip the rehearsal again.
    from routes.inference import _ReasoningSpanGuard

    first = "short first turn"
    second = (
        'emit </think> then <tool_call>{"name": "get_weather", "arguments": {}}</tool_call> ends it'
    )
    prov = {"wrapped": 2, "wraps": [{"len": len(first)}, {"len": len(second)}]}
    guard = _ReasoningSpanGuard(prov)
    names = {"get_weather"}

    turn1 = f"<think>{first}</think>Looking it up."
    assert guard.strip(turn1, auto_heal_tool_calls = True, enabled_tool_names = names) == turn1
    guard.tool_end()

    turn2 = f"<think>{second}</think>It is sunny."
    assert guard.strip(turn2, auto_heal_tool_calls = True, enabled_tool_names = names) == turn2


def test_display_strip_still_cleans_tool_xml_after_the_trace():
    # The protection covers only the recorded span; a leaked call in the answer still goes.
    from routes.inference import _ReasoningSpanGuard, _strip_tool_xml_for_display

    prov = {"wrapped": 1, "wraps": [{"len": len("plain reasoning")}]}
    raw = '<think>plain reasoning</think>Done <tool_call>{"name": "get_weather", "arguments": {}}</tool_call>ok'
    clean = _ReasoningSpanGuard(prov).strip(
        raw, auto_heal_tool_calls = True, enabled_tool_names = {"get_weather"}
    )
    assert clean == "<think>plain reasoning</think>Done ok"
    # No provenance -> byte-identical to the plain display strip.
    assert _ReasoningSpanGuard(None).strip(
        raw, auto_heal_tool_calls = True, enabled_tool_names = {"get_weather"}
    ) == _strip_tool_xml_for_display(
        raw, auto_heal_tool_calls = True, enabled_tool_names = {"get_weather"}
    )


def test_display_strip_leaves_unwrapped_leading_tag_to_the_cleaner():
    # wrapped == 0: the model typed the tag itself, so behaviour must not change.
    from routes.inference import _ReasoningSpanGuard, _strip_tool_xml_for_display
    raw = '<think>literal</think>Done <tool_call>{"name": "get_weather", "arguments": {}}</tool_call>ok'
    for prov in ({"wrapped": 0, "wraps": []}, None):
        assert _ReasoningSpanGuard(prov).strip(
            raw, auto_heal_tool_calls = True, enabled_tool_names = {"get_weather"}
        ) == _strip_tool_xml_for_display(
            raw, auto_heal_tool_calls = True, enabled_tool_names = {"get_weather"}
        )


# ─────────────────────────────────────────────────────────────────────────────
# SSE grammar conformance
# ─────────────────────────────────────────────────────────────────────────────


def _parse_anthropic_sse(lines):
    """Turn emitter output into (event_name, data) pairs, asserting well-formedness."""
    events = []
    blob = "".join(lines)
    assert "\r" not in blob, "carriage return in SSE: strict parsers split events on \\n\\n"
    for chunk in blob.split("\n\n"):
        if not chunk.strip():
            continue
        name = data = None
        for line in chunk.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        assert name is not None and data is not None, f"malformed SSE chunk {chunk!r}"
        events.append((name, data))
    return events


def assert_anthropic_stream_conformant(lines):
    """Assert the Anthropic Messages streaming grammar and return the blocks.

    A real SDK accumulates deltas into the block its index names, so an index that
    skips, repeats or arrives with no open block corrupts the message (or raises).
    This is the guard for that: message_start first, every block opened before it is
    written to and closed exactly once, indices gapless from 0, message_delta with
    stop_reason and usage before a final message_stop.
    """
    events = _parse_anthropic_sse(lines)
    assert events, "no events emitted"
    assert events[0][0] == "message_start", f"first event was {events[0][0]}"
    assert events[-1][0] == "message_stop", f"last event was {events[-1][0]}"

    blocks = {}
    open_index = None
    next_index = 0
    closed = []
    saw_message_delta = False

    for name, data in events[1:]:
        assert data.get("type") == name, f"event {name} carries data.type {data.get('type')}"
        if name in ("ping", "tool_result"):
            # tool_result is Unsloth's own event for a server-executed tool; real
            # SDKs ignore it, and it must never claim a content block index.
            assert "index" not in data, "tool_result must not consume a block index"
            continue
        if name == "content_block_start":
            assert not saw_message_delta, "content_block_start after message_delta"
            assert open_index is None, f"block {open_index} still open"
            assert (
                data["index"] == next_index
            ), f"index {data['index']} out of sequence, expected {next_index}"
            block = data["content_block"]
            if block["type"] == "thinking":
                assert block.get("thinking") == ""
                assert "signature" in block, "thinking block start needs a signature field"
            open_index = data["index"]
            next_index += 1
            blocks[open_index] = {"type": block["type"], "text": ""}
        elif name == "content_block_delta":
            assert open_index is not None, "delta with no open block"
            assert data["index"] == open_index, f"delta index {data['index']} != {open_index}"
            delta = data["delta"]
            expected = {
                "text_delta": "text",
                "thinking_delta": "thinking",
                "signature_delta": "thinking",
                "input_json_delta": "tool_use",
            }[delta["type"]]
            assert (
                blocks[open_index]["type"] == expected
            ), f"{delta['type']} inside a {blocks[open_index]['type']} block"
            blocks[open_index]["text"] += delta.get("text") or delta.get("thinking") or ""
        elif name == "content_block_stop":
            assert open_index is not None, "content_block_stop with no open block"
            assert data["index"] == open_index
            assert open_index not in closed, f"block {open_index} closed twice"
            closed.append(open_index)
            open_index = None
        elif name == "message_delta":
            assert not saw_message_delta, "two message_delta events"
            assert open_index is None, f"message_delta while block {open_index} is open"
            assert "stop_reason" in data["delta"] and "usage" in data
            saw_message_delta = True
        elif name == "message_stop":
            assert saw_message_delta, "message_stop without message_delta"
        else:
            raise AssertionError(f"unexpected event {name}")

    assert open_index is None, f"stream ended with block {open_index} open"
    assert saw_message_delta, "no message_delta"
    assert sorted(closed) == list(range(len(closed))), f"indices not gapless from 0: {closed}"
    return [(blocks[i]["type"], blocks[i]["text"]) for i in sorted(blocks)]


def _drive_emitter(
    chunks,
    provenance = None,
    parse_think = True,
    extra = (),
):
    """Feed cumulative content deltas (plus any extra events) through the emitter."""
    emitter = AnthropicStreamEmitter(parse_think = parse_think, think_provenance = provenance)
    lines = list(emitter.start("msg_1", "test-model", input_tokens = 3))
    cumulative = ""
    for chunk in chunks:
        cumulative += chunk
        lines.extend(emitter.feed({"type": "content", "text": cumulative}))
    for event in extra:
        lines.extend(emitter.feed(event))
    lines.extend(emitter.finish())
    return lines


class TestAnthropicStreamGrammar:
    """The index state machine an Anthropic SDK decodes against.

    Blocks open lazily now, so every index comes from _alloc_block_index rather
    than an eager bump in start(). A skipped or reused index is a user-visible
    break: the official SDK accumulates deltas by index.
    """

    @pytest.mark.parametrize(
        "text, provenance",
        [
            ("plain answer", None),
            ("<think>trace</think>answer", {"wrapped": 1, "wraps": [{"len": 5}]}),
            ("<think>unclosed trace", {"wrapped": 1, "wraps": [{"len": 14}]}),
            ("<think></think>answer", {"wrapped": 1, "wraps": [{"len": 0}]}),
            ("<think>a </think> b</think>answer", {"wrapped": 1, "wraps": [{"len": 11}]}),
            ("answer then <think>quoted</think>", {"wrapped": 0, "wraps": []}),
            ("<think>思考\U0001f9e0</think>答案", {"wrapped": 1, "wraps": [{"len": 3}]}),
        ],
    )
    @pytest.mark.parametrize("split", ["whole", "chars", "halves"])
    @pytest.mark.parametrize("parse_think", [True, False])
    def test_grammar_holds_for_every_shape(self, text, provenance, split, parse_think):
        chunks = {
            "whole": [text],
            "chars": list(text),
            "halves": [text[: len(text) // 2], text[len(text) // 2 :]],
        }[split]
        assert_anthropic_stream_conformant(
            _drive_emitter(
                [c for c in chunks if c], json.loads(json.dumps(provenance)), parse_think
            )
        )

    def test_a_reply_with_no_content_is_still_a_valid_stream(self):
        """start() no longer opens a block, so an empty generation emits none."""
        blocks = assert_anthropic_stream_conformant(_drive_emitter([]))
        assert blocks == []

    @pytest.mark.parametrize(
        "extra_first",
        [False, True],
    )
    def test_tool_blocks_keep_the_index_sequence(self, extra_first):
        """A tool call must take the next index, and text after it another one --
        whether or not text preceded it."""
        tool = [
            {
                "type": "tool_start",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "SF"}',
            },
            {
                "type": "tool_end",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "result": "sunny",
            },
        ]
        chunks = ["checking"] if extra_first else []
        blocks = assert_anthropic_stream_conformant(_drive_emitter(chunks, extra = tool))
        assert [b[0] for b in blocks] == (["text", "tool_use"] if extra_first else ["tool_use"])

    def test_thinking_then_tool_then_text_keeps_thinking_first(self):
        """Anthropic orders thinking ahead of the tool call it justifies."""
        provenance = {"wrapped": 1, "wraps": [{"len": len("need the weather")}]}
        emitter = AnthropicStreamEmitter(parse_think = True, think_provenance = provenance)
        lines = list(emitter.start("msg_1", "test-model"))
        lines.extend(emitter.feed({"type": "content", "text": "<think>need the weather</think>"}))
        lines.extend(
            emitter.feed(
                {
                    "type": "tool_start",
                    "tool_call_id": "call_1",
                    "name": "get_weather",
                    "arguments": "{}",
                }
            )
        )
        lines.extend(
            emitter.feed(
                {
                    "type": "tool_end",
                    "tool_call_id": "call_1",
                    "name": "get_weather",
                    "result": "sunny",
                }
            )
        )
        lines.extend(emitter.feed({"type": "content", "text": "It is sunny"}))
        lines.extend(emitter.finish())
        blocks = assert_anthropic_stream_conformant(lines)
        assert blocks == [
            ("thinking", "need the weather"),
            ("tool_use", ""),
            ("text", "It is sunny"),
        ]


class TestThinkTagSplitAcrossDeltas:
    """A <think> tag can be cut at any offset by tokenisation, so the emitter
    holds a trailing partial tag back. Sweeping every split position is the only
    way to know the hold-back never leaks markup or eats prose."""

    TEXT = "<think>abc</think>def"

    @pytest.mark.parametrize("cut", range(len(TEXT) + 1))
    def test_two_way_split_at_every_offset(self, cut):
        provenance = {"wrapped": 1, "wraps": [{"len": 3}]}
        blocks = assert_anthropic_stream_conformant(
            _drive_emitter([c for c in (self.TEXT[:cut], self.TEXT[cut:]) if c], provenance)
        )
        assert blocks == [("thinking", "abc"), ("text", "def")]

    @pytest.mark.parametrize("first", range(len(TEXT) + 1))
    def test_three_way_split_at_every_offset(self, first):
        for second in range(first, len(self.TEXT) + 1):
            chunks = [
                c for c in (self.TEXT[:first], self.TEXT[first:second], self.TEXT[second:]) if c
            ]
            blocks = assert_anthropic_stream_conformant(
                _drive_emitter(chunks, {"wrapped": 1, "wraps": [{"len": 3}]})
            )
            assert blocks == [
                ("thinking", "abc"),
                ("text", "def"),
            ], f"split at {first},{second} produced {blocks}"

    @pytest.mark.parametrize("cut", range(1, len("<think>")))
    def test_a_partial_tag_at_end_of_stream_is_literal_text(self, cut):
        """The model stopped mid-way through something that looked like a tag;
        those bytes are output, not markup, and must not be swallowed."""
        held = "<think>"[:cut]
        blocks = assert_anthropic_stream_conformant(
            _drive_emitter(["answer " + held], {"wrapped": 0, "wraps": []})
        )
        assert blocks == [("text", "answer " + held)]


class TestWhitespaceOnlyThinkingIsNotABlock:
    """Qwen3-style templates render "<think>\\n\\n</think>" on every reply when
    thinking is off, and llama-server parses that into reasoning_content. Opening
    a thinking block for it hangs an empty thought off ordinary answers, and the
    non-streaming reducer already drops it -- so the two paths disagreed."""

    @pytest.mark.parametrize("trace", ["", " ", "\n\n", "  \n \t "])
    def test_streaming_drops_it(self, trace):
        text = f"<think>{trace}</think>The answer"
        provenance = {"wrapped": 1, "wraps": [{"len": len(trace)}]}
        blocks = assert_anthropic_stream_conformant(_drive_emitter([text], provenance))
        assert blocks == [("text", "The answer")]

    @pytest.mark.parametrize("trace", ["", " ", "\n\n", "  \n \t "])
    def test_streaming_and_non_streaming_agree(self, trace):
        from routes.inference import _anthropic_plain_response_from_events

        text = f"<think>{trace}</think>The answer"
        provenance = {"wrapped": 1, "wraps": [{"len": len(trace)}]}
        streamed = assert_anthropic_stream_conformant(
            _drive_emitter([text], json.loads(json.dumps(provenance)))
        )
        response = _anthropic_plain_response_from_events(
            iter([text]),
            "msg_1",
            "test-model",
            parse_think = True,
            think_provenance = json.loads(json.dumps(provenance)),
        )
        reduced = [
            (b["type"], b.get("thinking") or b.get("text", ""))
            for b in json.loads(response.body)["content"]
        ]
        assert streamed == reduced

    def test_a_trace_that_merely_starts_with_whitespace_is_kept_verbatim(self):
        """Only an entirely blank trace is dropped; real reasoning keeps its
        leading newlines so the rendered thought matches the model's output."""
        trace = "\n\nWorking through it"
        provenance = {"wrapped": 1, "wraps": [{"len": len(trace)}]}
        blocks = assert_anthropic_stream_conformant(
            _drive_emitter(list(f"<think>{trace}</think>Answer"), provenance)
        )
        assert blocks == [("thinking", trace), ("text", "Answer")]


class TestReasoningSurvivesTheWholeChain:
    """End to end over a fake llama-server stream: the generator folds
    reasoning_content into <think> markup and records provenance, and the
    emitter splits it back into typed blocks. The two halves are only correct
    together, so they are exercised together against the bytes llama-server
    actually sends."""

    @staticmethod
    def _backend(monkeypatch, chunks):
        import contextlib
        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48851
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = True
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False

        @contextlib.contextmanager
        def fake_stream(
            _client,
            _url,
            payload,
            _cancel,
            headers = None,
            first_token_deadline = None,
        ):
            yield type("R", (), {"status_code": 200, "chunks": chunks})()

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream)
        monkeypatch.setattr(
            backend,
            "_iter_text_cancellable",
            lambda response, _c, first_token_deadline = None: iter(response.chunks),
        )
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *a, **k: False)
        return backend

    @staticmethod
    def _sse(delta):
        return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"

    def _run(self, monkeypatch, deltas):
        backend = self._backend(monkeypatch, [self._sse(d) for d in deltas] + ["data: [DONE]\n"])
        provenance = {"wrapped": 0}
        emitter = AnthropicStreamEmitter(parse_think = True, think_provenance = provenance)
        lines = list(emitter.start("msg_1", "test-model"))
        for cumulative in backend.generate_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            reasoning_provenance = provenance,
        ):
            if isinstance(cumulative, str):
                lines.extend(emitter.feed({"type": "content", "text": cumulative}))
        lines.extend(emitter.finish())
        return assert_anthropic_stream_conformant(lines)

    def test_reasoning_then_answer_becomes_a_thinking_block_and_a_text_block(self, monkeypatch):
        blocks = self._run(
            monkeypatch,
            [
                {"reasoning_content": "First I "},
                {"reasoning_content": "check the map."},
                {"content": "It is "},
                {"content": "north."},
            ],
        )
        assert blocks == [("thinking", "First I check the map."), ("text", "It is north.")]

    def test_a_reasoning_only_reply_is_not_empty(self, monkeypatch):
        blocks = self._run(monkeypatch, [{"reasoning_content": "just thinking"}])
        assert blocks and blocks[0][0] in ("thinking", "text")
        assert "just thinking" in blocks[0][1]

    def test_a_literal_think_tag_in_content_stays_text(self, monkeypatch):
        """No reasoning_content, so the generator wraps nothing and provenance
        stays at zero -- the model merely quoted the tag."""
        blocks = self._run(
            monkeypatch,
            [
                {"content": "<think>"},
                {"content": " is the tag"},
            ],
        )
        assert blocks == [("text", "<think> is the tag")]

    def test_a_close_tag_quoted_inside_the_trace_does_not_end_it(self, monkeypatch):
        blocks = self._run(
            monkeypatch,
            [
                {"reasoning_content": "the closer is </think>"},
                {"reasoning_content": " and it is quoted"},
                {"content": "done"},
            ],
        )
        assert blocks == [
            ("thinking", "the closer is </think> and it is quoted"),
            ("text", "done"),
        ]

    def test_an_old_llama_server_that_never_sends_reasoning_content_is_unchanged(self, monkeypatch):
        """Pre-reasoning_content builds only ever populate `content`."""
        blocks = self._run(monkeypatch, [{"content": "plain "}, {"content": "answer"}])
        assert blocks == [("text", "plain answer")]

    def test_a_blank_reasoning_trace_does_not_open_a_thinking_block(self, monkeypatch):
        """Qwen3 with thinking off renders "<think>\\n\\n</think>" and llama-server
        reports it as reasoning_content, on every single reply."""
        blocks = self._run(
            monkeypatch,
            [
                {"reasoning_content": "\n\n"},
                {"content": "The answer"},
            ],
        )
        assert blocks == [("text", "The answer")]


class TestReasoningProvenanceIsBoundToItsSynthesisTurn:
    """A tool loop's non-streaming reducer runs only after generation finished,
    so ``think_provenance`` is already at its FINAL aggregate. Attributing wraps
    by block order there let an early turn's literal ``<think>`` (Qwen3 with
    thinking off re-emits the closed empty block into `content`, llama.cpp
    common/chat-peg-parser.cpp discards whitespace-only reasoning so no
    reasoning_content is reported) consume a LATER turn's genuine wrap: the
    literal block was rendered as thinking and the real trace was delivered as
    raw tagged text. The streamed emitter reads the ledger live and never can,
    so the two paths must agree turn for turn.
    """

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    @staticmethod
    def _sse(delta):
        return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"

    @classmethod
    def _tool_call_sse(cls, call_id, city):
        return cls._sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": city}),
                        },
                    }
                ]
            }
        )

    @classmethod
    def _backend(cls, monkeypatch, streams):
        import contextlib

        from core.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend.__new__(LlamaCppBackend)
        backend._process = object()
        backend._healthy = True
        backend._port = 48853
        backend._api_key = None
        backend._effective_context_length = 4096
        backend._supports_reasoning = True
        backend._reasoning_always_on = False
        backend._reasoning_style = "enable_thinking"
        backend._supports_preserve_thinking = False
        _pending = list(streams)

        @contextlib.contextmanager
        def fake_stream(
            _client,
            _url,
            _payload,
            _cancel,
            headers = None,
            **_kw,
        ):
            yield type("R", (), {"status_code": 200, "chunks": _pending.pop(0)})()

        monkeypatch.setattr(backend, "_stream_with_retry", fake_stream)
        monkeypatch.setattr(
            backend,
            "_iter_text_cancellable",
            lambda response, _c, first_token_deadline = None: iter(response.chunks),
        )
        monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *a, **k: False)
        monkeypatch.setattr(
            "core.inference.tools.execute_tool", lambda name, arguments, **k: "sunny"
        )
        return backend

    def _generator(self, monkeypatch, streams, provenance):
        backend = self._backend(monkeypatch, streams)
        return backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "weather?"}],
            tools = self.TOOLS,
            max_tool_iterations = len(streams),
            reasoning_provenance = provenance,
        )

    def _streamed(self, monkeypatch, streams):
        provenance = {"wrapped": 0}
        emitter = AnthropicStreamEmitter(parse_think = True, think_provenance = provenance)
        lines = list(emitter.start("msg_1", "test-model"))
        for event in self._generator(monkeypatch, streams, provenance):
            lines.extend(emitter.feed(event))
        lines.extend(emitter.finish())
        return assert_anthropic_stream_conformant(lines)

    def _reduced(self, monkeypatch, streams):
        from routes.inference import (
            _anthropic_tool_response_from_events,
            _collect_anthropic_events,
        )

        provenance = {"wrapped": 0}
        gen = self._generator(monkeypatch, streams, provenance)
        # Exactly what _anthropic_tool_non_streaming does: drain first, reduce after.
        events = _collect_anthropic_events(lambda: gen, provenance)
        response = _anthropic_tool_response_from_events(
            events,
            "msg_1",
            "test-model",
            openai_tools = self.TOOLS,
            parse_think = True,
            think_provenance = provenance,
        )
        blocks = []
        for block in json.loads(response.body)["content"]:
            if block["type"] == "thinking":
                blocks.append(("thinking", block["thinking"]))
            elif block["type"] == "tool_use":
                blocks.append(("tool_use", block["name"]))
            else:
                blocks.append(("text", block["text"]))
        return blocks

    @staticmethod
    def _norm(blocks):
        """The stream conformance helper accumulates a tool_use block's
        input_json_delta, the reducer reports its name; compare the rest."""
        return [(t, "" if t == "tool_use" else v) for t, v in blocks]

    def _literal_then_genuine(self):
        return [
            # Turn 1: the model re-emitted the template's closed think block, so
            # llama-server reports no reasoning_content and the tags are content.
            [
                self._sse({"content": "<think>\n\n</think>\n\nLet me check."}),
                self._tool_call_sse("call_1", "SF"),
                "data: [DONE]\n",
            ],
            # Turn 2: genuine reasoning, reported as reasoning_content.
            [
                self._sse({"reasoning_content": "The tool said sunny, so"}),
                self._sse({"content": "It is sunny in SF."}),
                "data: [DONE]\n",
            ],
        ]

    def test_a_literal_leading_tag_does_not_steal_a_later_turns_wrap(self, monkeypatch):
        blocks = self._reduced(monkeypatch, self._literal_then_genuine())
        assert blocks == [
            ("text", "<think>\n\n</think>\n\nLet me check."),
            ("tool_use", "get_weather"),
            ("thinking", "The tool said sunny, so"),
            ("text", "It is sunny in SF."),
        ]

    def test_streaming_and_non_streaming_agree_on_a_mixed_tool_loop(self, monkeypatch):
        streamed = self._streamed(monkeypatch, self._literal_then_genuine())
        reduced = self._reduced(monkeypatch, self._literal_then_genuine())
        assert self._norm(streamed) == self._norm(reduced)

    def test_a_literal_tag_between_two_genuine_turns_keeps_both_traces(self, monkeypatch):
        """Three turns: wrap, literal, wrap. Block-order attribution handed the
        literal middle turn the SECOND wrap and left the third turn's real trace
        as tagged text."""
        streams = [
            [
                self._sse({"reasoning_content": "first I check"}),
                self._tool_call_sse("call_1", "SF"),
                "data: [DONE]\n",
            ],
            [
                self._sse({"content": "<think>\n\n</think>\n\nOne more."}),
                self._tool_call_sse("call_2", "LA"),
                "data: [DONE]\n",
            ],
            [
                self._sse({"reasoning_content": "both are sunny"}),
                self._sse({"content": "Both cities are sunny."}),
                "data: [DONE]\n",
            ],
        ]
        expected = [
            ("thinking", "first I check"),
            ("tool_use", "get_weather"),
            ("text", "<think>\n\n</think>\n\nOne more."),
            ("tool_use", "get_weather"),
            ("thinking", "both are sunny"),
            ("text", "Both cities are sunny."),
        ]
        assert self._reduced(monkeypatch, streams) == expected
        assert self._norm(self._streamed(monkeypatch, streams)) == self._norm(expected)

    def test_two_genuine_turns_still_map_in_order(self, monkeypatch):
        """The ordinary case must be untouched: each turn keeps its own trace."""
        streams = [
            [
                self._sse({"reasoning_content": "look it up"}),
                self._tool_call_sse("call_1", "SF"),
                "data: [DONE]\n",
            ],
            [
                self._sse({"reasoning_content": "report it"}),
                self._sse({"content": "Sunny."}),
                "data: [DONE]\n",
            ],
        ]
        expected = [
            ("thinking", "look it up"),
            ("tool_use", "get_weather"),
            ("thinking", "report it"),
            ("text", "Sunny."),
        ]
        assert self._reduced(monkeypatch, streams) == expected
        assert self._norm(self._streamed(monkeypatch, streams)) == self._norm(expected)


class TestALiteralTagAfterBlankSpaceIsNotDuplicated:
    """A reply that opens with a blank line and then quotes ``<think>`` -- the
    shape llama-server returns for a thinking model under
    ``--reasoning-format none``, where nothing is split into reasoning_content.
    The leading run was emitted, then emitted AGAIN as part of the literal-text
    fallback, so the client saw it twice."""

    @pytest.mark.parametrize("lead", ["\n", "\n\n", "  ", " \n\t"])
    @pytest.mark.parametrize("split", ["one", "chars"])
    def test_the_leading_run_is_delivered_exactly_once(self, lead, split):
        text = f"{lead}<think> is the tag you asked about."
        chunks = [text] if split == "one" else list(text)
        blocks = assert_anthropic_stream_conformant(
            _drive_emitter(chunks, {"wrapped": 0, "wraps": []})
        )
        assert blocks == [("text", text)]

    def test_a_genuine_trace_after_a_blank_line_still_parses(self):
        """Only the literal branch was doubling; the provenance-backed one
        already dropped the consumed run."""
        provenance = {"wrapped": 1, "wraps": [{"len": 5}]}
        blocks = assert_anthropic_stream_conformant(
            _drive_emitter(["\n\n<think>trace</think>Answer"], provenance)
        )
        assert blocks == [("text", "\n\n"), ("thinking", "trace"), ("text", "Answer")]


class TestPreserveThinkingHonoursTheBackendDefault:
    """An omitted `preserve_thinking` must follow the LOADED template's default.

    llama-server merges chat_template_kwargs per key, so a request that omits
    preserve_thinking leaves the launch-time --chat-template-kwargs value active
    (llama.cpp common/chat.cpp `extra_context`). Coercing the omission to False
    on the conversion side stripped the reasoning_content that same template was
    still being told to render: on a preserve-by-default family the replayed
    thinking silently vanished from the prompt. Counting shares the resolver so
    the total keeps describing the prompt generation actually builds.
    """

    TRACE = "The user prefers metric units."

    MESSAGES = [
        {"role": "user", "content": "How far is the moon?"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": TRACE, "signature": ""},
                {"type": "text", "text": "About 384,400 km."},
            ],
        },
        {"role": "user", "content": "And the sun?"},
    ]

    @staticmethod
    def _payload(**fields):
        base = {"max_tokens": 16, "messages": TestPreserveThinkingHonoursTheBackendDefault.MESSAGES}
        base.update(fields)
        return AnthropicMessagesRequest(**base)

    @staticmethod
    def _backend(default):
        from types import SimpleNamespace
        return SimpleNamespace(
            supports_preserve_thinking = True,
            preserve_thinking_default = default,
        )

    @pytest.mark.parametrize(
        "default, override, expected",
        [
            # The regression: omitted on a preserve-by-default model must keep
            # the block, not drop it.
            (True, None, True),
            # Omitted on an off-by-default model still drops (unchanged).
            (False, None, False),
            # Explicit values win over the default in both directions and are
            # unchanged by the resolver.
            (True, True, True),
            (True, False, False),
            (False, True, True),
            (False, False, False),
        ],
    )
    def test_resolver_is_three_valued(self, default, override, expected):
        from routes.inference import _anthropic_preserve_thinking
        resolved = _anthropic_preserve_thinking(
            self._backend(default), self._payload(preserve_thinking = override)
        )
        assert resolved is expected

    def test_none_and_explicit_false_are_distinguishable(self):
        """`bool()` made these identical; on a default-true backend they must
        now resolve to opposite values."""
        from routes.inference import _anthropic_preserve_thinking

        backend = self._backend(True)
        assert _anthropic_preserve_thinking(backend, self._payload()) is True
        assert (
            _anthropic_preserve_thinking(backend, self._payload(preserve_thinking = False)) is False
        )

    def test_a_backend_without_the_attribute_stays_off(self):
        """Test doubles and non-reasoning backends lack the property."""
        from types import SimpleNamespace

        from routes.inference import _anthropic_preserve_thinking

        assert _anthropic_preserve_thinking(SimpleNamespace(), self._payload()) is False

    @pytest.mark.parametrize("default", [True, False])
    @pytest.mark.parametrize("override", [None, True, False])
    def test_conversion_and_counting_cannot_drift(self, default, override):
        """Both routes resolve through the one helper, so the prompt the count
        prices is the prompt generation renders."""
        from routes.inference import _anthropic_preserve_thinking

        payload = self._payload(preserve_thinking = override)
        backend = self._backend(default)
        resolved = _anthropic_preserve_thinking(backend, payload)

        converted = anthropic_messages_to_openai(
            [m.model_dump() for m in payload.messages],
            payload.system,
            preserve_thinking = resolved,
        )
        counted = anthropic_messages_to_openai(
            [m.model_dump() for m in payload.messages],
            payload.system,
            preserve_thinking = _anthropic_preserve_thinking(backend, payload),
        )
        assert converted == counted

        assistant = next(m for m in converted if m["role"] == "assistant")
        expected_kept = override if override is not None else default
        assert ("reasoning_content" in assistant) is expected_kept
        if expected_kept:
            assert assistant["reasoning_content"] == self.TRACE
        # The answer text is never affected by the preserve decision.
        assert assistant["content"] == "About 384,400 km."

    @pytest.mark.parametrize(
        "default, override, expect_block",
        [(True, None, True), (False, None, False), (True, False, False), (False, True, True)],
    )
    def test_the_route_hands_the_block_to_the_generator(
        self, monkeypatch, default, override, expect_block
    ):
        """End to end over /v1/messages and /v1/messages/count_tokens against a
        deterministic backend double: what reaches the generator is what the
        counter prices."""
        import asyncio
        from types import SimpleNamespace

        import routes.inference as inf_mod
        from routes.inference import anthropic_count_tokens, anthropic_messages

        seen = {}

        def _gen(**kwargs):
            seen["messages"] = kwargs["messages"]
            seen["preserve_thinking"] = kwargs.get("preserve_thinking")
            yield "ok"

        def _count(
            messages,
            system,
            tools,
            strict = False,
            chat_template_kwargs = None,
        ):
            seen["count_messages"] = messages
            return sum(len(str(m.get("reasoning_content") or "")) for m in messages) + 1

        backend = SimpleNamespace(
            is_loaded = True,
            is_vision = False,
            supports_tools = False,
            supports_tool_passthrough = False,
            model_identifier = "unsloth/Qwen3.8-8B-GGUF",
            context_length = 4096,
            supports_reasoning = True,
            reasoning_always_on = False,
            reasoning_default = True,
            supports_preserve_thinking = True,
            preserve_thinking_default = default,
            count_chat_tokens = _count,
            generate_chat_completion = _gen,
            generate_chat_completion_with_tools = _gen,
            effective_parallel_slots = 4,
            base_url = "http://llama.preserve.test:9999",
            _request_reasoning_kwargs = lambda et, re_, pt: (
                {
                    k: v
                    for k, v in (("enable_thinking", et), ("preserve_thinking", pt))
                    if v is not None
                }
                or None
            ),
        )
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)

        class _Request:
            def __init__(self):
                self.state = SimpleNamespace()
                self.url = SimpleNamespace(path = "/v1/messages")
                self.method = "POST"

            async def is_disconnected(self):
                return False

        payload = self._payload(preserve_thinking = override)
        asyncio.run(anthropic_messages(payload, request = _Request(), current_subject = "t"))
        gen_messages = seen["messages"]
        assert any(m.get("reasoning_content") for m in gen_messages) is expect_block
        # An omitted override must stay None on the wire so llama-server keeps
        # falling back to the launch-time kwarg rather than being pinned here.
        assert seen["preserve_thinking"] == override

        asyncio.run(anthropic_count_tokens(payload, request = _Request(), current_subject = "t"))
        assert seen["count_messages"] == gen_messages
