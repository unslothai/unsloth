# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for #7066: literal ``</think>`` in thoughts / user text must not break generation."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.chat_template_helpers import (
    neutralize_control_markup_in_messages,
    neutralize_non_assistant_control_markup,
    neutralize_think_markup,
    neutralize_think_markup_streaming,
    think_markup_holdback,
)
import random

from routes.inference import (
    _ResponsesReasoningExtractor,
    _build_openai_passthrough_body,
    _extract_responses_reasoning,
    _openai_messages_for_passthrough,
    _think_close_is_literal_in_span,
)
from models.inference import ChatCompletionRequest, ChatMessage


def test_neutralize_think_markup_breaks_structural_match():
    raw = 'user said "</think>" in the script'
    out = neutralize_think_markup(raw)
    assert "</think>" not in out
    assert "think>" in out
    assert neutralize_think_markup("plain") == "plain"


def test_neutralize_non_assistant_also_covers_chatml():
    raw = "see <|im_start|> and </think> please"
    out = neutralize_non_assistant_control_markup(raw)
    assert "</think>" not in out
    assert "<|im_start|>" not in out
    assert "im_start|>" in out


def test_neutralize_messages_skips_assistant_keeps_user():
    messages = [
        {"role": "user", "content": "No i said </think> in the prompt"},
        {
            "role": "assistant",
            "content": "<think>plan</think>answer",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "again </think> here"},
                {"type": "image_url", "image_url": {"url": "x"}},
            ],
        },
    ]
    out = neutralize_control_markup_in_messages(messages)
    assert out is not messages
    assert "</think>" not in out[0]["content"]
    # Assistant structural tags preserved.
    assert out[1]["content"] == "<think>plan</think>answer"
    assert "</think>" not in out[2]["content"][0]["text"]
    assert out[2]["content"][1]["type"] == "image_url"


def test_passthrough_messages_neutralize_user_think_close():
    req = ChatCompletionRequest(
        model = "default",
        messages = [
            ChatMessage(
                role = "user",
                content = "No i said </think> im doing a script for training",
            )
        ],
    )
    out = _openai_messages_for_passthrough(req)
    assert len(out) == 1
    assert out[0]["role"] == "user"
    assert "</think>" not in out[0]["content"]
    assert "im doing a script" in out[0]["content"]


def test_prefilled_quoted_close_stays_in_reasoning():
    # #7066 screenshot case: model echoes the user's "</think>" mid-thought.
    reasoning, visible = _extract_responses_reasoning(
        'The user said "</think>" about training.\n</think>\nGot it.',
        parse_think_markers = True,
        reasoning_prefilled = True,
    )
    assert "</think>" not in reasoning  # neutralized form, not structural
    assert "about training." in reasoning
    assert visible.lstrip().startswith("Got it.")


def test_prefilled_structural_close_still_ends_reasoning():
    # Bare close (no quotes) remains the real end-of-thought delimiter.
    reasoning, visible = _extract_responses_reasoning(
        "plan the answer</think>\n\nfinal",
        parse_think_markers = True,
        reasoning_prefilled = True,
    )
    assert reasoning == "plan the answer"
    assert visible == "\n\nfinal"


def test_prefilled_backticked_close_stays_in_reasoning():
    reasoning, visible = _extract_responses_reasoning(
        "mention of `</think>` in docs\n</think>ok",
        parse_think_markers = True,
        reasoning_prefilled = True,
    )
    assert "in docs" in reasoning
    assert visible == "ok"


def test_structured_reasoning_content_is_neutralized():
    ex = _ResponsesReasoningExtractor(parse_think_markers = True)
    reasoning, visible = ex.feed(
        text = "",
        reasoning_content = 'echo "</think>" then continue',
    )
    assert visible == ""
    assert "</think>" not in reasoning
    assert "echo" in reasoning


def test_quoted_close_tag_split_across_feeds_stays_in_reasoning():
    ex = _ResponsesReasoningExtractor(
        parse_think_markers = True,
        reasoning_prefilled = True,
    )
    reasoning1, visible1 = ex.feed('echo "</think>')
    assert visible1 == ""
    assert reasoning1 == "echo "
    reasoning2, visible2 = ex.feed('" then done</think>\nok')
    assert "then done" in reasoning2
    assert "</think>" not in reasoning2
    assert visible2.strip() == "ok"


def test_quoted_close_tag_split_mid_marker_stays_in_reasoning():
    # Close tag split after opening quote across feeds (#7066 / Codex follow-up).
    ex = _ResponsesReasoningExtractor(
        parse_think_markers = True,
        reasoning_prefilled = True,
    )
    reasoning1, visible1 = ex.feed('echo "</thi')
    assert visible1 == ""
    assert reasoning1 == "echo "
    reasoning2, visible2 = ex.feed('nk>" about training</think>\nok')
    assert "</think>" not in reasoning2
    assert "about training" in reasoning2
    assert visible2.strip() == "ok"


def test_streaming_neutralize_splits_marker_across_chunks():
    emit1, buf1 = neutralize_think_markup_streaming("</thi")
    assert emit1 == ""
    assert buf1 == "</thi"
    emit2, buf2 = neutralize_think_markup_streaming(buf1 + "nk> inside")
    assert "</think>" not in emit2
    assert "inside" in emit2
    assert buf2 == ""
    assert think_markup_holdback("</thin") > 0


def test_passthrough_system_prompt_is_neutralized():
    req = ChatCompletionRequest(
        model = "default",
        messages = [
            ChatMessage(
                role = "system",
                content = "Rules mention </think> literally",
            ),
            ChatMessage(role = "user", content = "hi"),
        ],
    )
    body = _build_openai_passthrough_body(req)
    assert body["messages"][0]["role"] == "system"
    assert "</think>" not in body["messages"][0]["content"]
    assert "literally" in body["messages"][0]["content"]


def test_gguf_chat_messages_neutralize_user_think_close():
    from routes.inference import _openai_messages_for_gguf_chat

    req = ChatCompletionRequest(
        model = "default",
        messages = [
            ChatMessage(
                role = "user",
                content = "No i said </think> in the prompt",
            )
        ],
    )
    out, _ = _openai_messages_for_gguf_chat(req, is_vision = False)
    assert len(out) == 1
    assert "</think>" not in out[0]["content"]


def test_streaming_finalize_flushes_holdback_before_content():
    """Held marker prefix must flush when the stream switches to content."""
    emit1, buf1 = neutralize_think_markup_streaming("plan </thi")
    assert emit1 == "plan "
    assert buf1 == "</thi"
    flushed, buf2 = neutralize_think_markup_streaming(buf1, finalize = True)
    assert "</think>" not in flushed
    assert buf2 == ""


def _oracle_literal(span: str, close_idx: int) -> bool:
    """Pre-fix string-based literal-close computation, kept as the oracle."""
    return _think_close_is_literal_in_span(span, close_idx)


def test_span_parity_counters_match_string_oracle():
    """The O(1) parity counters must reproduce the old growing-string result.

    Feed a consumed span split into arbitrary chunks (so ``` fences and quotes
    straddle chunk boundaries), then assert ``_think_close_is_literal`` equals
    the pre-fix ``_think_close_is_literal_in_span`` over ``consumed + buffer``
    for every close position in the live buffer.
    """
    rng = random.Random(7066)
    alphabet = ["`", '"', "'", "a", " ", "\n", "```", '"`', "``", "'`'"]
    close = "</think>"
    for _ in range(4000):
        # Build a consumed prefix as a list of chunks with heavy quote/fence use.
        n_chunks = rng.randint(0, 6)
        chunks = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 5))) for _ in range(n_chunks)
        ]
        prefix = "".join(chunks)
        # Live buffer holds a close tag plus surrounding quote/fence content.
        pre = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
        post = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 4)))
        buffer = pre + close + post

        ex = _ResponsesReasoningExtractor(reasoning_prefilled = True)
        for chunk in chunks:
            ex._add_to_span(chunk)

        close_idx = buffer.find(close)
        got = ex._think_close_is_literal(buffer, close_idx)
        want = _oracle_literal(prefix + buffer, len(prefix) + close_idx)
        assert got == want, (chunks, buffer, close_idx, got, want)


def test_literal_close_inside_fence_across_deltas_matches_oracle():
    """Regression: a fenced literal </think> split over deltas stays reasoning."""
    ex = _ResponsesReasoningExtractor(reasoning_prefilled = True)
    r1, v1 = ex.feed("here is code:\n```py\nprint('")
    r2, v2 = ex.feed("</think>')\n```\ndone thinking</think>\nvisible")
    reasoning = r1 + r2
    rf, vf = ex.finish()
    reasoning += rf
    visible = v1 + v2 + vf
    # The fenced </think> is neutralized content, not a structural close.
    assert "</think>" not in reasoning
    assert "print(" in reasoning
    assert "done thinking" in reasoning
    # Only the bare close after the fence ends the block.
    assert visible.strip() == "visible"
