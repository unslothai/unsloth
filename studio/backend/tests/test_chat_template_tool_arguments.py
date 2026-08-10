# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""apply_chat_template_for_generation must coerce assistant tool_call arguments
from the OpenAI JSON-string form to a dict before rendering. Strict tool
templates (e.g. mlx-community Qwen3.5 checkpoints) iterate arguments.items() and
raise "Can only get item pairs from a mapping." on the string form when a prior
tool call is re-rendered on the next turn (MLX + transformers paths).

It must likewise split parallel tool calls for templates that render only one
call per message (Llama 3.x).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.chat_template_helpers import (  # noqa: E402
    _normalize_tool_call_arguments,
    _split_parallel_tool_calls,
    apply_chat_template_for_generation,
)


def _conv(arguments):
    return [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "c1",
                    "function": {"name": "web_search", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "name": "web_search", "content": "21C sunny"},
    ]


class _StrictTemplateTokenizer:
    """Mimics a strict Qwen tool template: rejects string tool_call arguments."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize = False,
        add_generation_prompt = True,
        **kw,
    ):
        for msg in messages:
            for call in msg.get("tool_calls", []) or []:
                args = call.get("function", {}).get("arguments")
                if isinstance(args, str):
                    raise TypeError("Can only get item pairs from a mapping.")
        return "RENDERED"


def test_string_arguments_are_parsed_to_dict():
    out = _normalize_tool_call_arguments(_conv('{"query": "sweden"}'))
    args = out[1]["tool_calls"][0]["function"]["arguments"]
    assert args == {"query": "sweden"}


def test_dict_arguments_untouched_and_no_copy():
    conv = _conv({"query": "sweden"})
    assert _normalize_tool_call_arguments(conv) is conv


def test_non_json_string_left_as_is():
    out = _normalize_tool_call_arguments(_conv("not json"))
    assert out[1]["tool_calls"][0]["function"]["arguments"] == "not json"


def test_render_succeeds_on_strict_template_with_string_arguments():
    # Regression: strict template + string args used to raise.
    result = apply_chat_template_for_generation(_StrictTemplateTokenizer(), _conv('{"query": "x"}'))
    assert result == "RENDERED"


class _RecordingTokenizer:
    """Lenient template: renders whatever arguments it is given (string or dict)."""

    def __init__(self):
        self.seen_arguments = None

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize = False,
        add_generation_prompt = True,
        **kw,
    ):
        for msg in messages:
            for call in msg.get("tool_calls", []) or []:
                self.seen_arguments = call.get("function", {}).get("arguments")
        return "RENDERED"


def test_lenient_template_receives_original_string_untouched():
    # Lenient template must see the exact original string, not a coerced dict.
    tok = _RecordingTokenizer()
    apply_chat_template_for_generation(tok, _conv('{"query": "x"}'))
    assert tok.seen_arguments == '{"query": "x"}'


def test_messages_without_tool_calls_pass_through_unchanged():
    conv = [{"role": "user", "content": "hi"}]
    assert _normalize_tool_call_arguments(conv) is conv


class _RaiseExceptionTemplateTokenizer:
    """Mimics the bundled gemma-4.jinja: rejects string tool_call arguments via
    ``raise_exception(...)``, which surfaces as a Jinja error, NOT a TypeError."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize = False,
        add_generation_prompt = True,
        **kw,
    ):
        for msg in messages:
            for call in msg.get("tool_calls", []) or []:
                args = call.get("function", {}).get("arguments")
                if isinstance(args, str):
                    raise ValueError(
                        "chat_template: tool_calls[].function.arguments must be a "
                        "JSON object (mapping), not a string."
                    )
        return "RENDERED"


def test_render_succeeds_on_raise_exception_template_with_string_arguments():
    # Regression: gemma-4.jinja rejects string args via a non-TypeError; retry must still coerce.
    result = apply_chat_template_for_generation(
        _RaiseExceptionTemplateTokenizer(), _conv('{"query": "x"}')
    )
    assert result == "RENDERED"


def test_unrelated_template_error_still_propagates_with_dict_args():
    # Failure unrelated to string args (dict args, nothing to coerce) must propagate.
    class _AlwaysRaises:
        def apply_chat_template(self, messages, **kw):
            raise ValueError("template is broken")

    with pytest.raises(ValueError, match = "broken"):
        apply_chat_template_for_generation(_AlwaysRaises(), _conv({"query": "x"}))


def _parallel_conv(
    *,
    ids = ("c1", "c2"),
    results_have_ids = True,
    content = "sure",
):
    a, b = ids
    return [
        {"role": "user", "content": "search then render"},
        {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "type": "function",
                    "id": a,
                    "function": {"name": "web_search", "arguments": {"query": "x"}},
                },
                {
                    "type": "function",
                    "id": b,
                    "function": {"name": "render_html", "arguments": {"html": "<canvas>"}},
                },
            ],
        },
        {
            "role": "tool",
            "name": "web_search",
            **({"tool_call_id": a} if results_have_ids else {}),
            "content": "no text",
        },
        {
            "role": "tool",
            "name": "render_html",
            **({"tool_call_id": b} if results_have_ids else {}),
            "content": "ok",
        },
    ]


class _SingleToolCallTokenizer:
    """Mimics the Llama 3.x template: rejects >1 call per message."""

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize = False,
        add_generation_prompt = True,
        **kw,
    ):
        for msg in messages:
            if len(msg.get("tool_calls") or ()) > 1:
                raise ValueError("This model only supports single tool-calls at once!")
        return "RENDERED"


def test_parallel_calls_split_into_sequential_single_call_turns():
    out = _split_parallel_tool_calls(_parallel_conv())
    assert [(m["role"], m.get("name")) for m in out] == [
        ("user", None),
        ("assistant", None),
        ("tool", "web_search"),
        ("assistant", None),
        ("tool", "render_html"),
    ]
    assert [len(m["tool_calls"]) for m in out if m.get("tool_calls")] == [1, 1]
    assert out[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert out[3]["tool_calls"][0]["function"]["name"] == "render_html"


def test_split_pairs_results_by_tool_call_id_not_position():
    conv = _parallel_conv()
    conv[2], conv[3] = conv[3], conv[2]  # results arrive out of order
    out = _split_parallel_tool_calls(conv)
    assert out[1]["tool_calls"][0]["id"] == "c1" and out[2]["tool_call_id"] == "c1"
    assert out[3]["tool_calls"][0]["id"] == "c2" and out[4]["tool_call_id"] == "c2"


def test_split_falls_back_to_order_when_results_have_no_ids():
    out = _split_parallel_tool_calls(_parallel_conv(results_have_ids = False))
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant", "tool"]
    assert out[2]["name"] == "web_search" and out[4]["name"] == "render_html"


def test_split_keeps_content_on_first_piece_only():
    out = _split_parallel_tool_calls(_parallel_conv(content = "sure"))
    assert out[1]["content"] == "sure"
    assert out[3]["content"] == ""


def test_split_keeps_unmatched_results_after_the_split():
    conv = _parallel_conv()
    del conv[3]  # second call never returned a result
    out = _split_parallel_tool_calls(conv)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]


def test_split_leaves_later_turns_intact():
    conv = _parallel_conv() + [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
    ]
    out = _split_parallel_tool_calls(conv)
    assert [m["role"] for m in out[-2:]] == ["assistant", "user"]
    assert out[-2]["content"] == "done"


def test_single_call_and_plain_conversations_pass_through_unchanged():
    conv = _conv({"query": "x"})
    assert _split_parallel_tool_calls(conv) is conv
    plain = [{"role": "user", "content": "hi"}]
    assert _split_parallel_tool_calls(plain) is plain


def test_render_succeeds_on_single_call_template_with_parallel_calls():
    # Regression: two calls in one turn used to break every later render.
    result = apply_chat_template_for_generation(_SingleToolCallTokenizer(), _parallel_conv())
    assert result == "RENDERED"


def test_string_arguments_and_parallel_calls_are_repaired_together():
    conv = _parallel_conv()
    for call in conv[1]["tool_calls"]:
        call["function"]["arguments"] = json.dumps(call["function"]["arguments"])

    class _StrictAndSingleCall(_SingleToolCallTokenizer):
        def apply_chat_template(self, messages, **kw):
            for msg in messages:
                for call in msg.get("tool_calls", []) or []:
                    if isinstance(call.get("function", {}).get("arguments"), str):
                        raise TypeError("Can only get item pairs from a mapping.")
            return super().apply_chat_template(messages, **kw)

    assert apply_chat_template_for_generation(_StrictAndSingleCall(), conv) == "RENDERED"


def test_lenient_template_never_sees_a_split_conversation():
    seen = {}

    class _Lenient:
        def apply_chat_template(self, messages, **kw):
            seen["n"] = len(messages)
            return "RENDERED"

    apply_chat_template_for_generation(_Lenient(), _parallel_conv())
    assert seen["n"] == 4  # unsplit
