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
import os
import sys
from pathlib import Path

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.chat_template_helpers import (
    ReasoningChannelNormalizer,
    detect_reasoning_channel_markers,
    detect_reasoning_channel_markers_from_model_info,
    detect_think_prefill,
    render_with_native_template_fallback,
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

    assert output == " <think>Two plus two.</think>4"


def test_muse_glimmer_direct_reply_without_reasoning_is_normalized():
    """The reply header is consumed whether or not reasoning preceded it."""
    parser = _muse_normalizer()
    output = parser.feed(" to=user<|message|>4") + parser.finish()

    assert output == " 4"


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

    assert output == " Partly.<think>Reconsider.</think>Actually four."


def test_muse_glimmer_tool_addressed_block_after_a_reply_keeps_its_markup():
    """Normalization still ends at a tool call: the tool-call parser needs it."""
    tool_block = (
        '<|start|>assistant to=web_search<|message|>{"q": 1}<|eom|>'
        "<|start|>assistant to=user<|message|>Done."
    )
    parser = _muse_normalizer()
    output = parser.feed(" to=user<|message|>Checking.<|eom|>" + tool_block)
    output += parser.finish()

    assert output == " Checking." + tool_block


def test_muse_glimmer_repeated_reasoning_blocks_each_become_a_think_block():
    parser = _muse_normalizer()
    output = parser.feed(
        "to=self<|message|>First.<|eom|>"
        "<|start|>assistant to=self<|message|>Second.<|eom|>"
        "<|start|>assistant to=user<|message|>Done."
    )
    output += parser.finish()

    assert output == "<think>First.</think><think>Second.</think>Done."


def test_muse_glimmer_tool_call_header_survives_for_the_tool_parser():
    """A block addressed to a tool is content the tool-call parser must see."""
    parser = _muse_normalizer()
    output = parser.feed(
        "to=self<|message|>Need a search.<|eom|>"
        "<|start|>assistant to=web_search<|message|><atem:function_calls>"
    )
    output += parser.finish()

    assert output == (
        "<think>Need a search.</think>"
        "<|start|>assistant to=web_search<|message|><atem:function_calls>"
    )


def test_muse_glimmer_tool_header_split_across_chunks_is_not_swallowed():
    parser = _muse_normalizer()
    output = ""
    for chunk in (
        "to=self<|message|>Think.<|eom|><|start|>assistant to=web_",
        "search<|message|>x",
    ):
        output += parser.feed(chunk)
    output += parser.finish()

    assert output == "<think>Think.</think><|start|>assistant to=web_search<|message|>x"


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
    # Generation stops on <|eot|>, so it only reaches here when a caller replays
    # a whole turn; leave it to the control-markup neutralizer like other markup.
    assert normalized(turn + "<|eot|>") == "<think>Reason</think>Reply<|eot|>"


def test_streamer_builds_the_recipient_parser_from_muse_markers():
    from core.inference.chat_template_helpers import RecipientChannelNormalizer
    from core.inference.inference import ReasoningTextIteratorStreamer

    class _Tokenizer:
        def decode(self, *_args, **_kwargs):
            return ""

    streamer = ReasoningTextIteratorStreamer(_Tokenizer(), markers = _MUSE_MARKERS)

    assert isinstance(streamer._normalizer, RecipientChannelNormalizer)
    assert streamer._normalizer.feed("to=self<|message|>x<|eom|>") == "<think>x</think>"
