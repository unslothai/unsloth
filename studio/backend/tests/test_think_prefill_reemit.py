# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for local reasoning-stream helpers.

Reasoning templates (Qwen3.6-style) end the generation prompt with an open
``<think>\\n`` so the model starts reasoning immediately. skip_prompt
streaming drops that opening tag, so the safetensors/MLX paths must re-emit
it for the frontend's <think> parser to render a thinking block.
"""

import ast
import json
import os
import sys

import pytest
from pathlib import Path

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.tool_call_parser import parse_tool_calls_from_text
from core.inference.chat_template_helpers import (
    ReasoningChannelNormalizer,
    detect_reasoning_channel_markers,
    detect_reasoning_channel_markers_from_model_info,
    detect_think_prefill,
    normalize_reasoning_snapshots,
    prompt_opens_reasoning_channel,
    render_with_native_template_fallback,
    trailing_assistant_text,
)


QWEN_PROMPT = "<|im_start|>user\nHi!<|im_end|>\n<|im_start|>assistant\n"


def test_open_think_prefill_reemitted():
    """Qwen3.6-style enable_thinking=True prompt tail: <think>\\n."""
    assert detect_think_prefill(QWEN_PROMPT + "<think>\n") == "<think>\n"


def test_bare_open_think_prefill_reemitted():
    """Prefill without trailing newline still detected."""
    assert detect_think_prefill(QWEN_PROMPT + "<think>") == "<think>"


def test_closed_think_prefill_not_reemitted():
    """enable_thinking=False prefills a closed, empty think block."""
    assert detect_think_prefill(QWEN_PROMPT + "<think>\n\n</think>\n\n") == ""


def test_prompt_without_think_untouched():
    """Non-reasoning templates produce no prefix."""
    assert detect_think_prefill(QWEN_PROMPT) == ""


def test_historical_think_blocks_ignored():
    """A closed think block in a prior assistant turn (preserve_thinking)
    must not trigger re-emission when the generation tail is plain."""
    prompt = (
        "<|im_start|>user\nHi!<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nprior reasoning\n</think>\n\nHello!<|im_end|>\n"
        "<|im_start|>user\nAgain?<|im_end|>\n<|im_start|>assistant\n"
    )
    assert detect_think_prefill(prompt) == ""


def test_historical_blocks_plus_open_prefill():
    """Prior closed blocks plus a fresh open prefill: only the tail matters."""
    prompt = (
        "<|im_start|>assistant\n<think>\nprior\n</think>\n\nHello!<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    assert detect_think_prefill(prompt) == "<think>\n"


def test_content_after_open_tag_not_reemitted():
    """If non-whitespace follows the tag it is not a plain prefill."""
    assert detect_think_prefill(QWEN_PROMPT + "<think>\npartial reasoning") == ""


def test_empty_and_none_prompts():
    assert detect_think_prefill("") == ""
    assert detect_think_prefill(None) == ""


def test_guard_suppresses_when_close_tag_is_special():
    """If </think> is a special token, skip_special_tokens strips the model's
    close tag, so re-emitting the open would leave an unclosed block. Guard off."""
    specials = ["<|im_end|>", "<think>", "</think>"]
    assert detect_think_prefill(QWEN_PROMPT + "<think>\n", specials) == ""


def test_guard_emits_when_think_not_special():
    specials = ["<|im_end|>", "<|endoftext|>"]
    assert detect_think_prefill(QWEN_PROMPT + "<think>\n", specials) == "<think>\n"


def test_guard_default_and_empty_keep_emitting():
    assert detect_think_prefill(QWEN_PROMPT + "<think>\n", None) == "<think>\n"
    assert detect_think_prefill(QWEN_PROMPT + "<think>\n", []) == "<think>\n"


def test_gemma_channel_detection_uses_active_template_not_token_metadata():
    class TemplateTokenizer:
        chat_template = {"default": "...<|channel>thought\\n{{ eoc_token }}"}

    class NamedTemplateTokenizer:
        chat_template = {
            "default": "plain assistant template",
            "tool_use": "...<|channel>thought\\n{{ eoc_token }}",
        }

    class TokenMetadataOnly:
        chat_template = None
        soc_token = "<|channel>"
        eoc_token = "<channel|>"

    class NamedTemplateProcessor:
        chat_template = {
            "default": "plain processor default",
            "tool_use": "<|channel>thought\nprocessor tool template<channel|>",
        }
        tokenizer = TokenMetadataOnly()

        def apply_chat_template(self, *_args, **_kwargs):
            raise NotImplementedError

    expected = ("<|channel>thought", "<channel|>")
    assert detect_reasoning_channel_markers(TemplateTokenizer()) == expected
    assert detect_reasoning_channel_markers(NamedTemplateTokenizer()) is None
    assert (
        detect_reasoning_channel_markers(
            NamedTemplateTokenizer(), tools = [{"function": {"name": "web_search"}}]
        )
        == expected
    )
    assert detect_reasoning_channel_markers(NamedTemplateTokenizer(), tools = []) is None
    assert (
        detect_reasoning_channel_markers(
            NamedTemplateProcessor(), tools = [{"function": {"name": "web_search"}}]
        )
        is None
    )
    assert detect_reasoning_channel_markers(TokenMetadataOnly()) is None


def test_gemma_channel_detection_tries_no_argument_getter_fallback():
    class FallbackTokenizer:
        chat_template = "plain fallback template"

        def get_chat_template(self, **kwargs):
            if kwargs:
                raise ValueError("tools are not supported")
            return "...<|channel>thought\n<channel|>"

    assert detect_reasoning_channel_markers(
        FallbackTokenizer(), tools = [{"function": {"name": "web_search"}}]
    ) == ("<|channel>thought", "<channel|>")


def test_native_template_fallback_returns_selected_reasoning_metadata():
    from types import SimpleNamespace

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]

    def render(tokenizer, msgs, *, tools, **_kw):
        body = "".join(message["content"] for message in msgs)
        suffix = "|TOOLS" if tools else ""
        return body + suffix if tokenizer.chat_template == "NATIVE <|channel>thought\n" else body

    result = render_with_native_template_fallback(
        formatted_prompt = "hi",
        tokenizer = SimpleNamespace(chat_template = "OVERRIDE"),
        model_info = {
            "native_chat_template": "NATIVE <|channel>thought\n",
            "tokenizer": SimpleNamespace(chat_template = "OVERRIDE"),
        },
        active_model_name = "gemma-test",
        messages = messages,
        tools = tools,
        apply_fn = render,
        return_metadata = True,
    )

    assert result.prompt == "hi|TOOLS"
    assert result.reasoning_channel_markers == ("<|channel>thought", "<channel|>")


def test_cached_native_template_metadata_recovers_reasoning_markers_without_tools():
    from types import SimpleNamespace

    model_info = {"chat_template_info": {"template": "native <|channel>thought\n<channel|>"}}

    assert detect_reasoning_channel_markers_from_model_info(
        SimpleNamespace(chat_template = "override has no native markers"),
        model_info,
        tools = None,
    ) == ("<|channel>thought", "<channel|>")
    result = render_with_native_template_fallback(
        formatted_prompt = "prompt from override",
        tokenizer = SimpleNamespace(chat_template = "override has no native markers"),
        model_info = model_info,
        active_model_name = "gemma-test",
        messages = [{"role": "user", "content": "hi"}],
        tools = None,
        return_metadata = True,
    )
    assert result.prompt == "prompt from override"
    assert result.reasoning_channel_markers == ("<|channel>thought", "<channel|>")


def test_cached_native_markers_do_not_describe_live_tool_template():
    from types import SimpleNamespace

    tools = [{"type": "function", "function": {"name": "web_search"}}]

    class LiveTokenizer:
        chat_template = "live tool template without native markers"

    def render(_tokenizer, _messages, *, tools, **_kwargs):
        return "prompt with tools" if tools else "prompt without tools"

    result = render_with_native_template_fallback(
        formatted_prompt = "prompt with tools",
        tokenizer = LiveTokenizer(),
        model_info = {
            "chat_template_info": {"template": "native <|channel>thought\n<channel|>"},
            "tokenizer": SimpleNamespace(),
        },
        active_model_name = "gemma-test",
        messages = [{"role": "user", "content": "hi"}],
        tools = tools,
        apply_fn = render,
        return_metadata = True,
    )

    assert result.prompt == "prompt with tools"
    assert result.reasoning_channel_markers is None


def test_gemma_channel_normalization_is_prefix_monotonic_and_preserves_tools():
    parser = ReasoningChannelNormalizer("<|channel>thought", "<channel|>")
    output = ""
    snapshots = []
    for chunk in (
        "<|chan",
        "nel>thought",
        "\nReason",
        "<chan",
        "nel|><|tool_call>web_search<tool_call|>",
    ):
        delta = parser.feed(chunk)
        if delta:
            output += delta
            snapshots.append(output)

    assert snapshots == [
        "<think>",
        "<think>Reason",
        "<think>Reason</think><|tool_call>web_search<tool_call|>",
    ]
    assert snapshots[1].startswith(snapshots[0])
    compact = ReasoningChannelNormalizer("<|channel>thought", "<channel|>")
    assert compact.feed("<|channel>thought<channel|>answer") + compact.finish() == (
        "<think></think>answer"
    )


# --- Muse Glimmer: recipient-addressed assistant channels ---

_MUSE_TEMPLATE = (
    "{%- if message.get('reasoning_content') -%}"
    "{{- '<|start|>assistant to=self<|message|>' + message['reasoning_content'] + '<|eom|>' -}}"
    "{%- endif -%}{{- '<|start|>assistant' -}}"
)
_MUSE_MARKERS = ("self", "user")


def _muse_normalizer():
    from core.inference.chat_template_helpers import make_reasoning_normalizer
    return make_reasoning_normalizer(_MUSE_MARKERS)


def test_muse_glimmer_channel_detected_from_its_template():
    class TemplateTokenizer:
        chat_template = _MUSE_TEMPLATE

    class PlainTokenizer:
        chat_template = "plain assistant template"

    assert detect_reasoning_channel_markers(TemplateTokenizer()) == _MUSE_MARKERS
    assert detect_reasoning_channel_markers(PlainTokenizer()) is None


def test_muse_glimmer_marker_pair_selects_the_recipient_normalizer():
    from core.inference.chat_template_helpers import (
        RecipientChannelNormalizer,
        ReasoningChannelNormalizer,
        make_reasoning_normalizer,
    )
    assert isinstance(make_reasoning_normalizer(_MUSE_MARKERS), RecipientChannelNormalizer)
    assert isinstance(
        make_reasoning_normalizer(("<|channel>thought", "<channel|>")),
        ReasoningChannelNormalizer,
    )


def test_recipient_protocol_ignores_a_prompt_derived_open_channel():
    """A recipient name is not a marker, so the marker-pair rule does not apply to it.

    ``prompt_opens_reasoning_channel`` looks for its opener at the prompt tail, and here
    that opener is the recipient name "self", which any prompt may end on by chance.
    Generation resumes at "<|start|>assistant", so this protocol always starts between
    blocks and the model writes its own header.
    """
    from core.inference.chat_template_helpers import make_reasoning_normalizer

    assert prompt_opens_reasoning_channel("tell me about self", _MUSE_MARKERS)

    parser = make_reasoning_normalizer(_MUSE_MARKERS, in_reasoning = True)
    assert parser.feed("to=user<|message|>plain answer<|eot|>") == "plain answer"

    opened = make_reasoning_normalizer(("<|channel>thought", "<channel|>"), in_reasoning = True)
    assert opened.feed("still reasoning<channel|>answer") == "<think>still reasoning</think>answer"


def test_muse_glimmer_reply_header_is_consumed_across_chunk_boundaries():
    """Generation resumes after the prompt's trailing "<|start|>assistant", so
    the first header arrives without that prefix. Every header here is split
    across chunks, which is what the streamer sees token by token."""
    parser = _muse_normalizer()
    output = ""
    for chunk in (
        " to=self<|mess",
        "age|>Two plus two.",
        "<|eom|><|start|>assis",
        "tant to=user<|message|>",
        "4",
    ):
        output += parser.feed(chunk)
    output += parser.finish()

    assert output == "<think>Two plus two.</think>4"


def test_muse_glimmer_direct_reply_without_reasoning_is_normalized():
    """The reply header is consumed whether or not reasoning preceded it."""
    parser = _muse_normalizer()
    output = parser.feed(" to=user<|message|>4") + parser.finish()

    assert output == "4"


def test_muse_glimmer_reasoning_after_a_reply_still_becomes_a_think_block():
    """A reply closes with <|eom|> like any other block, so the turn can carry
    on afterwards; treating the reply as the end would leak the rest verbatim."""
    parser = _muse_normalizer()
    output = parser.feed(
        " to=user<|message|>Partly.<|eom|>"
        "<|start|>assistant to=self<|message|>Reconsider.<|eom|>"
        "<|start|>assistant to=user<|message|>Actually four."
    )
    output += parser.finish()

    assert output == "Partly.<think>Reconsider.</think>Actually four."


def test_muse_glimmer_call_grammar_allows_attributes_beside_the_name():
    """The checkpoint's own response_template matches `name` among other attributes;
    a stricter reading drops parameters or fails to see the call at all."""
    parser = _muse_normalizer()
    output = parser.feed(
        "to=web_search<|message|><atem:function_calls>\n"
        '<atem:invoke type="function" name="web_search">\n'
        '<atem:parameter type="string" name="query">FIFA</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
    )
    output += parser.finish()

    assert output == (
        '<tool_call>{"name": "web_search", "arguments": {"query": "FIFA"}}</tool_call>'
    )


def test_muse_glimmer_text_the_model_wrote_around_a_call_is_kept():
    """Only the call syntax and its envelope are framing; prose beside them is the
    answer, and rewriting the call must not quietly delete it."""
    parser = _muse_normalizer()
    output = parser.feed(
        "to=web_search<|message|>Looking it up.<atem:function_calls>"
        '<atem:invoke name="s"><atem:parameter name="q">v</atem:parameter></atem:invoke>'
        "</atem:function_calls>One moment.<|eom|>"
    )
    output += parser.finish()

    assert output == (
        "Looking it up."
        '<tool_call>{"name": "s", "arguments": {"q": "v"}}</tool_call>'
        "One moment."
    )


@pytest.mark.parametrize(
    ("written", "parsed"),
    [
        ("plain words", "plain words"),
        ('{"lang": "en"}', {"lang": "en"}),
        ("[1, 2]", [1, 2]),
        ("1", 1),
        ("true", True),
    ],
)
def test_muse_glimmer_parameter_values_follow_the_grammars_json_parser(written, parsed):
    """The grammar parses values as JSON and keeps whatever will not parse."""
    parser = _muse_normalizer()
    output = parser.feed(
        "to=t<|message|><atem:function_calls>"
        '<atem:invoke name="t"><atem:parameter name="v">'
        + written
        + "</atem:parameter></atem:invoke></atem:function_calls><|eom|>"
    )
    output += parser.finish()
    call = parse_tool_calls_from_text(output)[0]

    assert json.loads(call["function"]["arguments"])["v"] == parsed


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("to=user<|message|>Done.<|eom|><|eot|>", "Done."),
        ("to=user<|message|>Done.<|eot|>", "Done."),
        ("to=self<|message|>Thinking.<|eot|>", "<think>Thinking.</think>"),
        ("Hello<|eot|>", "Hello"),
        ("stray text to=user<|message|>Hi.<|eot|>", "stray text Hi."),
    ],
)
def test_muse_glimmer_turn_marker_is_framing_wherever_it_lands(raw, expected):
    """<|eot|> ends the turn, so it terminates whichever block is open and is consumed
    between blocks. Whitespace beside a block is framing but a space inside stray text
    is content, and which one a space is cannot depend on where the stream split."""
    parser = _muse_normalizer()
    assert parser.feed(raw) + parser.finish() == expected

    per_char = _muse_normalizer()
    assert "".join(per_char.feed(c) for c in raw) + per_char.finish() == expected


_CLOSED_CALL = '<atem:invoke name="t"><atem:parameter name="q">v</atem:parameter></atem:invoke>'
_OPEN_CALL = '<atem:invoke name="t">half'
_CANONICAL_CALL = '<tool_call>{"name": "t", "arguments": {"q": "v"}}</tool_call>'


@pytest.mark.parametrize(
    ("body", "cancelled", "completed"),
    [
        ("before" + _CLOSED_CALL, "before", "before" + _CANONICAL_CALL),
        (_CLOSED_CALL + "after", "after", _CANONICAL_CALL + "after"),
        ("before" + _OPEN_CALL, "before", "before"),
        (_CLOSED_CALL + "mid" + _OPEN_CALL, "mid", _CANONICAL_CALL + "mid"),
        # Where the budget usually runs out: partway through a tag.
        ('<atem:invoke name="web', "", ""),
        ("</atem:function_call", "", ""),
        # Markup this parser does not own is shown rather than risk deleting an answer.
        ("<atem:unknown>kept</atem:unknown>",) * 3,
        ("<atem:invoker is prose",) * 3,
        ('<atem:parameter name="q">kept',) * 3,
        ("just words",) * 3,
    ],
)
def test_muse_glimmer_text_beside_a_call_survives_wherever_the_stream_stops(
    body, cancelled, completed
):
    """Holding a tool block back holds the prose beside it back too. What the model
    wrote is content whether or not a call ever closed, and cancelling must not take
    it down with the call it refuses to promote; only unfinished markup is dropped."""
    parser = _muse_normalizer()
    assert parser.feed("to=t<|message|>" + body) + parser.drain() == cancelled

    finished = _muse_normalizer()
    assert finished.feed("to=t<|message|>" + body) + finished.finish() == completed


def test_muse_glimmer_parameter_text_reaches_the_tool_exactly_as_written():
    """The grammar captures a value verbatim; trimming it rewrites the argument."""
    parser = _muse_normalizer()
    output = parser.feed(
        'to=t<|message|><atem:invoke name="t"><atem:parameter name="q">'
        "\nfirst\nsecond\n</atem:parameter></atem:invoke><|eom|>"
    )
    output += parser.finish()
    call = parse_tool_calls_from_text(output)[0]

    assert json.loads(call["function"]["arguments"])["q"] == "\nfirst\nsecond\n"


def test_muse_glimmer_bare_repeated_invokes_are_calls_without_an_envelope():
    """The grammar makes <atem:invoke> the call and repeats it; the surrounding
    <atem:function_calls> is only what the prompt happens to teach. It also fixes
    where attributes may sit: before `name` on a call, either side of it on a
    parameter."""
    parser = _muse_normalizer()
    output = parser.feed(
        "to=t<|message|>"
        '<atem:invoke name="a"><atem:parameter name="q">1</atem:parameter></atem:invoke>'
        '<atem:invoke type="function" name="b"><atem:parameter name="q" type="integer">2'
        "</atem:parameter></atem:invoke><|eom|>"
    )
    output += parser.finish()
    calls = parse_tool_calls_from_text(output)

    assert [call["function"]["name"] for call in calls] == ["a", "b"]
    assert [json.loads(call["function"]["arguments"])["q"] for call in calls] == [1, 2]


def test_muse_glimmer_tool_addressed_block_after_a_reply_keeps_its_markup():
    """A tool block holding no call at all is markup this parser does not own, so it
    is handed on whole rather than reshaped into something downstream might run."""
    tool_block = (
        '<|start|>assistant to=web_search<|message|>{"q": 1}<|eom|>'
        "<|start|>assistant to=user<|message|>Done."
    )
    parser = _muse_normalizer()
    output = parser.feed(" to=user<|message|>Checking.<|eom|>" + tool_block)
    output += parser.finish()

    assert output == "Checking." + tool_block


def test_muse_glimmer_repeated_reasoning_blocks_each_become_a_think_block():
    parser = _muse_normalizer()
    output = parser.feed(
        "to=self<|message|>First.<|eom|>"
        "<|start|>assistant to=self<|message|>Second.<|eom|>"
        "<|start|>assistant to=user<|message|>Done."
    )
    output += parser.finish()

    assert output == "<think>First.</think><think>Second.</think>Done."


@pytest.mark.parametrize("gap", [" ", "\n", "\n\n"])
def test_muse_glimmer_whitespace_between_blocks_does_not_split_the_reasoning(gap):
    """Whitespace separating two blocks is framing. Emitted, it lands between the
    think blocks, and the UI merges only adjacent reasoning, so one reasoning pass
    renders as two thinking sections."""
    raw = (
        f"{gap}to=self<|message|>First.<|eom|>"
        f"{gap}<|start|>assistant to=self<|message|>Second.<|eom|>"
        f"{gap}<|start|>assistant to=user<|message|>Done."
    )
    expected = "<think>First.</think><think>Second.</think>Done."

    parser = _muse_normalizer()
    assert parser.feed(raw) + parser.finish() == expected

    per_char = _muse_normalizer()
    streamed = "".join(per_char.feed(char) for char in raw) + per_char.finish()
    assert streamed == expected


def test_muse_glimmer_tool_call_becomes_a_canonical_tool_call():
    """The native call syntax matches no downstream parser, so an untranslated
    block is streamed to the user as prose instead of being executed."""
    parser = _muse_normalizer()
    output = parser.feed(
        "to=self<|message|>Need a search.<|eom|>"
        "<|start|>assistant to=web_search<|message|><atem:function_calls>\n"
        '<atem:invoke name="web_search">\n'
        '<atem:parameter name="query">FIFA 2026 winner</atem:parameter>\n'
        '<atem:parameter name="filters">{"lang": "en"}</atem:parameter>\n'
        "</atem:invoke>\n"
        "</atem:function_calls><|eom|>"
        "<|start|>assistant to=user<|message|>Checking."
    )
    output += parser.finish()

    assert output == (
        "<think>Need a search.</think>"
        '<tool_call>{"name": "web_search", "arguments": '
        '{"query": "FIFA 2026 winner", "filters": {"lang": "en"}}}</tool_call>'
        "Checking."
    )
    assert parse_tool_calls_from_text(output)[0]["function"]["name"] == "web_search"


def test_muse_glimmer_tool_call_survives_arriving_one_character_at_a_time():
    raw = (
        "to=web_search<|message|><atem:function_calls>\n"
        '<atem:invoke name="web_search">\n'
        '<atem:parameter name="query">a "quoted" phrase\nover two lines</atem:parameter>\n'
        "</atem:invoke>\n</atem:function_calls><|eom|>"
    )
    parser = _muse_normalizer()
    output = "".join(parser.feed(char) for char in raw) + parser.finish()

    assert output == (
        '<tool_call>{"name": "web_search", "arguments": '
        '{"query": "a \\"quoted\\" phrase\\nover two lines"}}</tool_call>'
    )


def test_muse_glimmer_parallel_tool_calls_each_become_a_call():
    parser = _muse_normalizer()
    output = parser.feed(
        "to=web_search<|message|><atem:function_calls>\n"
        '<atem:invoke name="web_search">\n'
        '<atem:parameter name="query">first</atem:parameter>\n</atem:invoke>\n'
        '<atem:invoke name="web_search">\n'
        '<atem:parameter name="query">second</atem:parameter>\n</atem:invoke>\n'
        "</atem:function_calls><|eom|>"
    )
    output += parser.finish()

    queries = [
        json.loads(call["function"]["arguments"])["query"]
        for call in parse_tool_calls_from_text(output)
    ]
    assert queries == ["first", "second"]


def test_muse_glimmer_turn_marker_is_consumed_wherever_it_lands():
    """<|eot|> ends the turn, so it terminates whichever block is open."""
    for raw, expected in (
        ("to=user<|message|>Done.<|eom|><|eot|>", "Done."),
        ("to=user<|message|>Done.<|eot|>", "Done."),
        ("to=self<|message|>Thinking.<|eot|>", "<think>Thinking.</think>"),
    ):
        parser = _muse_normalizer()
        assert parser.feed(raw) + parser.finish() == expected

        per_char = _muse_normalizer()
        streamed = "".join(per_char.feed(char) for char in raw) + per_char.finish()
        assert streamed == expected, raw


def test_muse_glimmer_cut_short_tool_call_leaks_no_markup():
    """A call the token budget truncated has no arguments worth executing, and its
    header is protocol framing, so neither belongs in what the user reads."""
    parser = _muse_normalizer()
    output = parser.feed(
        "to=self<|message|>Need a search.<|eom|>"
        "<|start|>assistant to=web_search<|message|><atem:function_calls>\n"
        '<atem:invoke name="web_search">\n<atem:parameter name="query">FIFA'
    )
    output += parser.finish()

    assert output == "<think>Need a search.</think>"


def test_muse_glimmer_call_closed_inside_a_cut_short_block_survives_finish():
    """The block never got its terminator, but this call did close, so the answer the
    user waited for is not thrown away with the half-written one after it."""
    raw = (
        "to=web_search<|message|><atem:function_calls>\n"
        '<atem:invoke name="web_search">\n'
        '<atem:parameter name="query">first</atem:parameter>\n</atem:invoke>\n'
        '<atem:invoke name="web_search">\n<atem:parameter name="query">sec'
    )
    parser = _muse_normalizer()
    output = parser.feed(raw) + parser.finish()

    assert output == (
        '<tool_call>{"name": "web_search", "arguments": {"query": "first"}}</tool_call>'
    )


def test_muse_glimmer_cancelling_promotes_no_call():
    """drain() is the cancellation path. A call still held there is one the block never
    terminated, and promoting it would let cancelling a turn be the thing that starts a
    tool running. A block that did terminate settled during feed(), before there was
    anything to cancel, so the two cases are asserted together."""
    held = (
        "to=web_search<|message|><atem:function_calls>\n"
        '<atem:invoke name="web_search">\n'
        '<atem:parameter name="query">first</atem:parameter>\n</atem:invoke>\n'
    )
    call = '<tool_call>{"name": "web_search", "arguments": {"query": "first"}}</tool_call>'

    cancelled = _muse_normalizer()
    assert cancelled.feed(held) == ""  # nothing settles while the block is open
    assert cancelled.drain() == ""

    terminated = _muse_normalizer()
    assert terminated.feed(held + "</atem:function_calls><|eom|>") == call
    assert terminated.drain() == ""


def test_muse_glimmer_tool_header_split_across_chunks_is_not_swallowed():
    parser = _muse_normalizer()
    output = ""
    for chunk in (
        "to=self<|message|>Think.<|eom|><|start|>assistant to=web_",
        'search<|message|><atem:function_calls><atem:invoke name="s">',
        '<atem:parameter name="q">x</atem:parameter></atem:invoke>',
        "</atem:function_calls><|eom|>",
    ):
        output += parser.feed(chunk)
    output += parser.finish()

    assert output == (
        "<think>Think.</think>" '<tool_call>{"name": "s", "arguments": {"q": "x"}}</tool_call>'
    )


def test_muse_glimmer_unterminated_reasoning_block_is_closed_at_finish():
    parser = _muse_normalizer()
    output = parser.feed("to=self<|message|>Cut short.") + parser.finish()

    assert output == "<think>Cut short.</think>"


def test_muse_glimmer_stream_ending_inside_a_header_keeps_the_text():
    parser = _muse_normalizer()
    output = parser.feed("to=self<|message|>Done.<|eom|><|start|>assis")
    output += parser.finish()

    assert output == "<think>Done.</think><|start|>assis"


def test_gemma_pair_protocol_is_unchanged_by_the_recipient_normalizer():
    parser = ReasoningChannelNormalizer("<|channel>thought", "<channel|>")
    output = parser.feed("<|channel>thought\nReason<channel|>answer") + parser.finish()

    assert output == "<think>Reason</think>answer"


def test_every_production_site_builds_the_normalizer_through_the_factory():
    """Markers are recipient names now, so a site still calling the marker-pair
    parser directly would treat the literal "self"/"user" as markup."""
    root = Path(__file__).resolve().parent.parent / "core" / "inference"

    def constructions(tree):
        return {
            id(node): node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.endswith("ChannelNormalizer")
        }

    built_in_factory = set()
    built_anywhere = {}
    for name in ("chat_template_helpers.py", "inference.py", "mlx_inference.py"):
        tree = ast.parse((root / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "make_reasoning_normalizer":
                built_in_factory |= set(constructions(node))
        built_anywhere.update(
            {key: f"{name}:{node.lineno}" for key, node in constructions(tree).items()}
        )

    assert built_in_factory, "the factory itself must construct a normalizer"
    outside = sorted(
        location for key, location in built_anywhere.items() if key not in built_in_factory
    )
    assert outside == [], outside


def test_muse_glimmer_snapshot_stream_normalizes_both_channels():
    """The non-streaming path consumes cumulative snapshots rather than deltas."""
    from core.inference.chat_template_helpers import normalize_reasoning_snapshots

    def normalized(raw):
        snapshots = [raw[:index] for index in range(1, len(raw) + 1)]
        emitted = list(normalize_reasoning_snapshots(iter(snapshots), markers = _MUSE_MARKERS))
        assert all(
            later.startswith(earlier) for earlier, later in zip(emitted, emitted[1:])
        ), emitted
        return emitted[-1]

    turn = "to=self<|message|>Reason<|eom|><|start|>assistant to=user<|message|>Reply"
    assert normalized(turn) == "<think>Reason</think>Reply"
    # The transformers streamer keeps special tokens and Transformers emits the EOS
    # token before its stopping check fires, so a completed answer really does end
    # with the turn marker. Consume it rather than showing it.
    assert normalized(turn + "<|eot|>") == "<think>Reason</think>Reply"


def test_streamer_builds_the_recipient_parser_from_muse_markers():
    from core.inference.chat_template_helpers import RecipientChannelNormalizer
    from core.inference.inference import ReasoningTextIteratorStreamer

    class _Tokenizer:
        def decode(self, *_args, **_kwargs):
            return ""

    streamer = ReasoningTextIteratorStreamer(_Tokenizer(), markers = _MUSE_MARKERS)

    assert isinstance(streamer._normalizer, RecipientChannelNormalizer)
    assert streamer._normalizer.feed("to=self<|message|>x<|eom|>") == "<think>x</think>"


GEMMA_MARKERS = ("<|channel>thought", "<channel|>")
GEMMA_TOOL_TAIL = '<|tool_response>response:web_search{value:<|"|>18C<|"|>}<tool_response|>'
GEMMA_POST_TOOL_PROMPT = "<|turn>model\n" + GEMMA_TOOL_TAIL + "<|channel>thought\n"


def test_prompt_opens_reasoning_channel_tracks_generation_prompt_state():
    """Open only when nothing but whitespace follows the prompt's last opener."""
    assert prompt_opens_reasoning_channel(GEMMA_POST_TOOL_PROMPT, GEMMA_MARKERS)
    # History closes its own channel before the post-tool opener.
    assert prompt_opens_reasoning_channel(
        "<|turn>model\n<|channel>thought\nearlier\n<channel|>" + GEMMA_POST_TOOL_PROMPT,
        GEMMA_MARKERS,
    )
    # Ordinary turn: the model emits the opener itself.
    assert not prompt_opens_reasoning_channel(
        "<|turn>user\nhi<turn|>\n<|turn>model\n", GEMMA_MARKERS
    )
    # Thinking disabled: no opener at all.
    assert not prompt_opens_reasoning_channel("<|turn>model\n" + GEMMA_TOOL_TAIL, GEMMA_MARKERS)
    assert not prompt_opens_reasoning_channel(
        "<|turn>model\n<|channel>thought\nearlier\n<channel|>done<turn|>\n", GEMMA_MARKERS
    )
    assert not prompt_opens_reasoning_channel(GEMMA_POST_TOOL_PROMPT, None)
    assert not prompt_opens_reasoning_channel(None, GEMMA_MARKERS)
    assert not prompt_opens_reasoning_channel("", GEMMA_MARKERS)


def test_unclosed_opener_in_history_cannot_forge_the_channel_state():
    """An assistant turn keeps channel markup, so only the prompt tail may decide.

    Neutralization strips these markers from user / system / tool turns but leaves
    an assistant turn's own markup intact, so a template that renders assistant
    content verbatim can carry a client-supplied unclosed opener into the prompt.
    (Gemma's own template also strips it, but a marker-aware override need not.)
    Trusting the last opener anywhere would start the parser inside reasoning on an
    ordinary turn and hide the whole answer in a think block.
    """
    forged = "<|turn>model\nok <|channel>thought<turn|>\n<|turn>user\nagain?<turn|>\n<|turn>model\n"
    assert not prompt_opens_reasoning_channel(forged, GEMMA_MARKERS)

    parser = ReasoningChannelNormalizer(
        *GEMMA_MARKERS, in_reasoning = prompt_opens_reasoning_channel(forged, GEMMA_MARKERS)
    )
    assert parser.feed("Plain answer.") + parser.finish() == "Plain answer."


def test_continued_turn_never_reads_channel_state_from_its_tail():
    """A continued turn ends inside the client's assistant text, not template markup.

    ``continue_final_message`` splices the caller's partial onto the prompt, so a
    partial ending on the opener would otherwise look exactly like a template that
    opened the channel, and capture the entire continuation as reasoning.
    """
    for tail in ("<|channel>thought", "<|channel>thought\n", "<|channel>thought   "):
        spliced = "<|turn>model\nLet me think. " + tail
        assert not prompt_opens_reasoning_channel(spliced, GEMMA_MARKERS, True)
        parser = ReasoningChannelNormalizer(
            *GEMMA_MARKERS,
            in_reasoning = prompt_opens_reasoning_channel(spliced, GEMMA_MARKERS, True),
        )
        assert parser.feed(" The answer is 18C.") + parser.finish() == " The answer is 18C."

    assert prompt_opens_reasoning_channel(GEMMA_POST_TOOL_PROMPT, GEMMA_MARKERS, False)


def test_tool_loop_pass_of_a_continued_turn_still_reads_the_prompt():
    """The request flag outlives the continuation; the render it describes does not.

    A continued turn that calls a tool keeps ``continue_final_message`` set for the
    next pass, but that pass renders an ordinary post-tool generation prompt. Suppressing
    detection on the request flag alone would put the post-tool reasoning back in the
    visible answer, so the effective signal is whether a trailing assistant turn was
    actually resumed.
    """
    resumed = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "partial"}]
    post_tool = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": "partial",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "18C"},
    ]
    effective = lambda msgs: bool(trailing_assistant_text(msgs))

    assert effective(resumed) is True
    # The tool result is the trailing turn now, so nothing was resumed.
    assert effective(post_tool) is False
    assert prompt_opens_reasoning_channel(
        GEMMA_POST_TOOL_PROMPT, GEMMA_MARKERS, effective(post_tool)
    )


def test_prompt_opened_channel_normalizes_post_tool_reasoning():
    """Post-tool generation emits only the closing marker; it is still reasoning."""
    parser = ReasoningChannelNormalizer(*GEMMA_MARKERS, in_reasoning = True)
    output = ""
    snapshots = []
    # Split across chunks, as a token stream splits it.
    for chunk in ("The search ", "returned 18C.", "<chan", "nel|>", "It is 18C in Paris."):
        delta = parser.feed(chunk)
        if delta:
            output += delta
            snapshots.append(output)
    output += parser.finish()

    assert output == "<think>The search returned 18C.</think>It is 18C in Paris."
    assert "<channel|>" not in output
    assert all(later.startswith(earlier) for earlier, later in zip(snapshots, snapshots[1:]))

    # Streaming X from a prompt-opened channel must match generating the opener plus X:
    # a streamed leading newline is content, unlike the protocol newline after an opener.
    for streamed, expected in (
        ("reasoned<channel|>answer", "<think>reasoned</think>answer"),
        ("\nreasoned<channel|>answer", "<think>\nreasoned</think>answer"),
    ):
        generated = ReasoningChannelNormalizer(*GEMMA_MARKERS)
        prefilled = ReasoningChannelNormalizer(*GEMMA_MARKERS, in_reasoning = True)
        assert (
            generated.feed("<|channel>thought\n" + streamed) + generated.finish()
            == prefilled.feed(streamed) + prefilled.finish()
            == expected
        )


def test_prompt_opened_channel_without_generated_text_emits_no_think_block():
    """A cancelled or empty post-tool turn must not emit an orphan </think>."""
    empty = ReasoningChannelNormalizer(*GEMMA_MARKERS, in_reasoning = True)
    assert empty.feed("") == ""
    assert empty.finish() == ""

    cancelled = ReasoningChannelNormalizer(*GEMMA_MARKERS, in_reasoning = True)
    # Hold back a partial closing marker, so drain() has real buffered text.
    assert cancelled.feed("partial reasoning<chan") == "<think>partial reasoning"
    assert cancelled.drain() == "<chan"


def test_normalize_reasoning_snapshots_derives_state_from_prompt():
    def _stream(pieces):
        cumulative = ""
        for piece in pieces:
            cumulative += piece
            yield cumulative

    post_tool = list(
        normalize_reasoning_snapshots(
            _stream(["reasoning", "<channel|>", "answer"]),
            markers = GEMMA_MARKERS,
            prompt = GEMMA_POST_TOOL_PROMPT,
        )
    )
    assert post_tool[-1] == "<think>reasoning</think>answer"

    # Without a prompt-opened channel the model supplies both markers as before.
    first_turn = list(
        normalize_reasoning_snapshots(
            _stream(["<|channel>thought\n", "reasoning", "<channel|>", "answer"]),
            markers = GEMMA_MARKERS,
            prompt = "<|turn>model\n",
        )
    )
    assert first_turn[-1] == "<think>reasoning</think>answer"
