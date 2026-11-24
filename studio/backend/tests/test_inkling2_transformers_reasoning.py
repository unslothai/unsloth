# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inkling 2 message protocol on the transformers path; pure string tests, no model."""

import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.chat_template_helpers import (  # noqa: E402
    InklingMessageNormalizer,
    detect_reasoning_channel_markers_from_template,
    make_reasoning_normalizer,
)
from core.inference.native_tool_tokens import reasoning_control_tokens  # noqa: E402

# The assistant branch of the released Inkling 2 chat template, trimmed to what detection reads.
INKLING_TEMPLATE = (
    '{%- set role_token = {"user": "<|message_user|>", "assistant": "<|message_model|>"} -%}'
    "{%- for message in messages -%}"
    '{%- if message.role == "assistant" and message.reasoning_content -%}'
    '{{- "<|message_model|><|content_thinking|>" ~ message.reasoning_content ~ "<|end_message|>" -}}'
    "{%- endif -%}"
    '{{- role_token[message.role] ~ "<|content_text|>" ~ message.content ~ "<|end_message|>" -}}'
    "{%- endfor -%}"
    '{%- if add_generation_prompt -%}{{- "<|message_model|>" -}}{%- endif -%}'
)
MARKERS = ("<|content_thinking|>", "<|end_message|>")


def _stream(normalizer, chunks):
    return "".join(normalizer.feed(chunk) for chunk in chunks) + normalizer.finish()


def _token_chunks(text):
    """One chunk per special token or word, the way the streamer hands them over."""
    out, i = [], 0
    while i < len(text):
        if text.startswith("<|", i):
            j = text.index("|>", i) + 2
        else:
            j = i + 1
            while j < len(text) and not text.startswith("<|", j) and text[j - 1] != " ":
                j += 1
        out.append(text[i:j])
        i = j
    return out


def test_template_selects_the_inkling_markers():
    assert detect_reasoning_channel_markers_from_template(INKLING_TEMPLATE) == MARKERS
    assert set(MARKERS) <= reasoning_control_tokens(MARKERS)
    assert {"<|message_model|>", "<|content_text|>"} <= reasoning_control_tokens(MARKERS)
    assert isinstance(make_reasoning_normalizer(MARKERS), InklingMessageNormalizer)


def test_thought_then_answer():
    raw = (
        "<|content_thinking|>The user wants two colors.<|end_message|>"
        "<|message_model|><|content_text|>Red\nBlue<|end_message|>"
    )
    out = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    assert out == "<think>The user wants two colors.</think>Red\nBlue"


def test_answer_without_a_thought():
    raw = "<|content_text|>Hello there, how are you?<|end_message|>"
    out = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    assert out == "Hello there, how are you?"


def test_tool_call_keeps_its_closer_for_the_parser():
    raw = (
        "<|content_thinking|>Search it.<|end_message|><|message_model|>"
        'web_search<|content_invoke_tool_json|>{"name":"web_search","args":{"query":"x"}}<|end_message|>'
    )
    out = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    assert out == (
        "<think>Search it.</think>"
        '<|content_invoke_tool_json|>{"name":"web_search","args":{"query":"x"}}<|end_message|>'
    )


def test_a_second_thought_leaves_no_markup():
    raw = (
        "<|content_thinking|>one<|end_message|><|message_model|><|content_text|>Answer.<|end_message|>"
        "<|message_model|><|content_thinking|>two<|end_message|>"
    )
    out = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    assert "<|" not in out
    assert out.startswith("<think>one</think>Answer.")


def test_a_closer_split_across_chunks_is_held():
    normalizer = make_reasoning_normalizer(MARKERS)
    out = normalizer.feed("<|content_text|>Answer.<|end_")
    assert out == "Answer."
    out += normalizer.feed("message|>") + normalizer.finish()
    assert out == "Answer."


def test_text_that_only_looks_like_a_marker_start_is_flushed():
    out = _stream(make_reasoning_normalizer(MARKERS), ["a < b and c <|"])
    assert out == "a < b and c <|"


def test_headers_drop_the_role_opener_and_text_opener():
    raw = "<|content_thinking|>t<|end_message|><|message_model|><|content_text|>Red\nBlue<|end_message|>"
    out = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    assert out == "<think>t</think>Red\nBlue"


def test_tool_call_name_prefix_is_dropped():
    raw = (
        "<|content_thinking|>Run it.<|end_message|><|message_model|>python<|content_invoke_tool_json|>"
        '{"name":"python","args":{"code":"print(37*41)"}}<|end_message|>'
    )
    out = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    assert out == (
        '<think>Run it.</think><|content_invoke_tool_json|>{"name":"python","args":{"code":"print(37*41)"}}'
        "<|end_message|>"
    )


def test_first_message_tool_call_without_a_role_opener():
    raw = 'python<|content_invoke_tool_json|>{"name":"python","args":{}}<|end_message|>'
    out = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    assert out == '<|content_invoke_tool_json|>{"name":"python","args":{}}<|end_message|>'


def test_every_chunking_gives_the_same_reply():
    raw = (
        "<|content_thinking|>a b<|end_message|><|message_model|><|content_text|>x y\\nz<|end_message|>"
        "<|message_model|>python<|content_invoke_tool_json|>{}<|end_message|>"
    )
    expected = _stream(make_reasoning_normalizer(MARKERS), [raw])
    for size in range(1, 12):
        chunks = [raw[i : i + size] for i in range(0, len(raw), size)]
        assert _stream(make_reasoning_normalizer(MARKERS), chunks) == expected, size


def test_header_without_a_content_opener_is_flushed_as_text():
    out = _stream(make_reasoning_normalizer(MARKERS), ["plain words with no protocol at all"])
    assert out == "plain words with no protocol at all"
