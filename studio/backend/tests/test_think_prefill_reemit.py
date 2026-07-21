# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Unit tests for local reasoning-stream helpers.

Reasoning templates (Qwen3.6-style) end the generation prompt with an open
``<think>\\n`` so the model starts reasoning immediately. skip_prompt
streaming drops that opening tag, so the safetensors/MLX paths must re-emit
it for the frontend's <think> parser to render a thinking block.
"""

import os
import sys

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
        return (
            body + suffix
            if tokenizer.chat_template == "NATIVE <|channel>thought\n"
            else body
        )

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

    model_info = {
        "chat_template_info": {"template": "native <|channel>thought\n<channel|>"}
    }

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
