# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""apply_chat_template_for_generation must coerce assistant tool_call arguments
from the OpenAI JSON-string form to a dict before rendering. Strict tool
templates (e.g. mlx-community Qwen3.5 checkpoints) iterate arguments.items() and
raise "Can only get item pairs from a mapping." on the string form when a prior
tool call is re-rendered on the next turn (MLX + transformers paths).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.chat_template_helpers import (  # noqa: E402
    _normalize_tool_call_arguments,
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
    # The regression: strict template + OpenAI string args used to raise.
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
    # A template that renders string args must see the exact original string
    # (the dict-coercion fallback must not run for it). Guards against emitting a
    # Python dict repr instead of the OpenAI JSON string.
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
    # The regression: gemma-4.jinja rejects string args via raise_exception (a
    # non-TypeError), so the retry must still coerce to a dict and re-render instead
    # of letting the Jinja error propagate and fail the tool turn.
    result = apply_chat_template_for_generation(
        _RaiseExceptionTemplateTokenizer(), _conv('{"query": "x"}')
    )
    assert result == "RENDERED"


def test_unrelated_template_error_still_propagates_with_dict_args():
    # A template failure unrelated to string arguments (dict args -> nothing to
    # normalize) must still propagate; the broadened catch only retries when there
    # is actually a string arg to coerce.
    class _AlwaysRaises:
        def apply_chat_template(self, messages, **kw):
            raise ValueError("template is broken")

    with pytest.raises(ValueError, match = "broken"):
        apply_chat_template_for_generation(_AlwaysRaises(), _conv({"query": "x"}))
