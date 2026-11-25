# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inkling 2 message protocol on the transformers path; pure string tests, no model."""

import json
import sys
from types import SimpleNamespace
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.chat_template_helpers import (  # noqa: E402
    InklingMessageNormalizer,
    detect_reasoning_channel_markers_from_template,
    make_reasoning_normalizer,
    prompt_opens_reasoning_channel,
    replays_reasoning_content,
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
    # The orchestrator has no tokenizer: the forwarded template decides, and only Inkling's does.
    assert replays_reasoning_content(None, {"chat_template_info": {"template": INKLING_TEMPLATE}})
    assert not replays_reasoning_content(None, {"chat_template_info": {"template": "<think>"}})
    # An override renders, so the cached native Inkling template must not turn the split on.
    native = {"native_chat_template": INKLING_TEMPLATE}
    override = SimpleNamespace(chat_template = "{{ messages[0].content }}<think>")
    assert not replays_reasoning_content(override, native)
    assert not replays_reasoning_content(None, native, template = "{{ messages }}")
    assert replays_reasoning_content(SimpleNamespace(chat_template = INKLING_TEMPLATE), native)


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


def test_answer_ending_in_the_tool_name_keeps_it_after_healing():
    from core.inference.passthrough_healing import heal_openai_message_events

    raw = (
        "<|content_text|>I will check that with python<|end_message|><|message_model|>python"
        '<|content_invoke_tool_json|>{"name":"python","args":{"code":"1+1"}}<|end_message|>'
    )
    text = _stream(make_reasoning_normalizer(MARKERS), _token_chunks(raw))
    events = heal_openai_message_events({"role": "assistant", "content": text}, {"python"})
    assert events[0] == ("text", "I will check that with python")
    assert events[1][0] == "tool_call" and events[1][1]["function"]["name"] == "python"
    # Raw streams still lose the echoed name at a message boundary.
    for prefix in ("", "Red<|end_message|>", "Hi<|end_message|><|message_model|>"):
        raw_call = (
            prefix + 'python<|content_invoke_tool_json|>{"name":"python","args":{}}<|end_message|>'
        )
        events = heal_openai_message_events({"role": "assistant", "content": raw_call}, {"python"})
        assert [kind for kind, _ in events] == (["text"] if prefix else []) + ["tool_call"]
        assert "python" not in "".join(value for kind, value in events if kind == "text")


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


def test_a_prompt_that_opens_the_thought_keeps_thought_and_answer():
    prompt = (
        "<|message_user|><|content_text|>hi<|end_message|><|message_model|><|content_thinking|>"
    )
    in_reasoning = prompt_opens_reasoning_channel(prompt, MARKERS)
    assert in_reasoning
    raw = "Short thought.<|end_message|><|message_model|><|content_text|>42<|end_message|>"
    out = _stream(make_reasoning_normalizer(MARKERS, in_reasoning = in_reasoning), _token_chunks(raw))
    assert out == "<think>Short thought.</think>42"


def test_a_continued_answer_keeps_its_text():
    normalizer = make_reasoning_normalizer(MARKERS, continued = True)
    assert normalizer.feed(" forecast.") == " forecast."
    rest = "<|end_message|><|message_model|>python<|content_invoke_tool_json|>{}<|end_message|>"
    out = _stream(normalizer, _token_chunks(rest))
    assert out == "<|content_invoke_tool_json|>{}<|end_message|>"


def test_quoted_control_text_in_tool_arguments_is_kept():
    code = "print('<|end_message|>', \"<|message_model|>\", '<|end_message|>')"
    call = json.dumps({"name": "python", "args": {"code": code}}, separators = (",", ":"))
    raw = "<|content_thinking|>t<|end_message|><|message_model|>python<|content_invoke_tool_json|>"
    raw += call + "<|end_message|>"
    expected = "<think>t</think><|content_invoke_tool_json|>" + call + "<|end_message|>"
    for size in range(1, 12):
        chunks = [raw[i : i + size] for i in range(0, len(raw), size)]
        assert _stream(make_reasoning_normalizer(MARKERS), chunks) == expected, size
