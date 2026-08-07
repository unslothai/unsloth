# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Continuing a truncated answer resumes the trailing assistant turn.

With ``continue_final_message`` the prompt ends inside the partial response, so the
model emits the next token of the same sentence. Without it the same conversation
renders a fresh assistant turn, which is why a cut-off response used to restart.

The flag is self-limiting: only a plain-text trailing assistant turn can be resumed,
so anything else renders normally instead of raising.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.chat_template_helpers import (  # noqa: E402
    apply_chat_template_for_generation,
    trailing_assistant_text,
)

_PARTIAL = "The three steps are: first, preheat the"


def _conv(partial = _PARTIAL):
    return [
        {"role": "user", "content": "Explain the recipe."},
        {"role": "assistant", "content": partial},
    ]


class _ChatMLTokenizer:
    """Minimal ChatML renderer supporting both boundary kwargs."""

    chat_template = "chatml"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize = False,
        add_generation_prompt = True,
        continue_final_message = False,
        **kw,
    ):
        if add_generation_prompt and continue_final_message:
            raise ValueError(
                "Cannot set both add_generation_prompt and continue_final_message."
            )
        out = []
        for index, message in enumerate(messages):
            body = message.get("content") or ""
            last = index == len(messages) - 1
            if continue_final_message and last:
                out.append(f"<|im_start|>{message['role']}\n{body}")
            else:
                out.append(f"<|im_start|>{message['role']}\n{body}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n")
        return "".join(out)


class _LegacyTokenizer(_ChatMLTokenizer):
    """A tokenizer predating ``continue_final_message`` (transformers < 4.44)."""

    def apply_chat_template(self, messages, *, tokenize = False, **kw):
        if "continue_final_message" in kw:
            raise TypeError(
                "apply_chat_template() got an unexpected keyword argument "
                "'continue_final_message'"
            )
        return super().apply_chat_template(messages, tokenize = tokenize, **kw)


def test_continuation_prompt_ends_inside_the_partial_answer():
    prompt = apply_chat_template_for_generation(
        _ChatMLTokenizer(), _conv(), continue_final_message = True
    )
    assert prompt.endswith(_PARTIAL)
    # No end-of-turn marker and no second assistant header after the partial:
    # the next token generated continues that sentence.
    assert prompt.count("<|im_start|>assistant") == 1
    assert "<|im_end|>" not in prompt.split("<|im_start|>assistant")[-1]


def test_without_the_flag_the_same_conversation_starts_a_new_turn():
    prompt = apply_chat_template_for_generation(_ChatMLTokenizer(), _conv())
    assert prompt.endswith("<|im_start|>assistant\n")
    assert prompt.count("<|im_start|>assistant") == 2


def test_legacy_tokenizer_falls_back_to_a_manual_splice():
    """A tokenizer that rejects the kwarg still continues, byte-identically."""
    legacy = apply_chat_template_for_generation(
        _LegacyTokenizer(), _conv(), continue_final_message = True
    )
    native = apply_chat_template_for_generation(
        _ChatMLTokenizer(), _conv(), continue_final_message = True
    )
    assert legacy == native


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([{"role": "user", "content": "hi"}], id = "user_final"),
        pytest.param(
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "looking it up",
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "c1",
                            "function": {"name": "web_search", "arguments": {}},
                        }
                    ],
                },
            ],
            id = "tool_calls",
        ),
        pytest.param(
            [
                {"role": "user", "content": "hi"},
                {"role": "tool", "name": "web_search", "content": "21C"},
            ],
            id = "tool_result_final",
        ),
    ],
)
def test_non_continuable_histories_render_a_normal_turn(messages):
    prompt = apply_chat_template_for_generation(
        _ChatMLTokenizer(), messages, continue_final_message = True
    )
    assert prompt.endswith("<|im_start|>assistant\n")


def test_trailing_assistant_text_joins_text_parts_and_rejects_the_rest():
    assert trailing_assistant_text(_conv()) == _PARTIAL
    assert (
        trailing_assistant_text(
            [{"role": "assistant", "content": [
                {"type": "text", "text": "ab"},
                {"type": "text", "text": "cd"},
            ]}]
        )
        == "abcd"
    )
    # No resume point inside an image part.
    assert trailing_assistant_text(
        [{"role": "assistant", "content": [{"type": "image_url", "image_url": {}}]}]
    ) is None
    assert trailing_assistant_text([]) is None
