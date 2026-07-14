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
    detect_think_prefill,
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


def test_gemma_channel_detection_uses_template_or_named_token_metadata():
    class TemplateTokenizer:
        chat_template = {"default": "...<|channel>thought\\n{{ eoc_token }}"}

    class LookalikeTokenizer:
        chat_template = "...<|channel>thoughtful\n...<channel|>..."

    class ChannelTokenizer:
        soc_token = "<|channel>"
        eoc_token = "<channel|>"
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|channel>": 100, "<channel|>": 101}.get(token, 0)

        def convert_ids_to_tokens(self, token_id):
            return {100: "<|channel>", 101: "<channel|>"}.get(token_id, "<unk>")

    class MarkerVocabTokenizer:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token):
            return {"<|channel>": 100, "<channel|>": 101}.get(token, 0)

        def convert_ids_to_tokens(self, token_id):
            return {100: "<|channel>", 101: "<channel|>"}.get(token_id, "<unk>")

    class VocabOnlyTokenizer:
        def get_vocab(self):
            return {"<|channel>": 100, "<channel|>": 101}

    class SplitVocabTokenizer:
        unk_token_id = 0

        def __init__(self, tokens):
            self._tokens = tokens

        def convert_tokens_to_ids(self, token):
            return self._tokens.get(token, 0)

        def convert_ids_to_tokens(self, token_id):
            return {value: key for key, value in self._tokens.items()}.get(token_id, "<unk>")

    class Processor:
        tokenizer = ChannelTokenizer()

    class SplitProcessor:
        soc_token = "<|channel>"
        eoc_token = "<channel|>"
        tokenizer = MarkerVocabTokenizer()

    class BadSplitProcessor:
        soc_token = "<|channel>"
        eoc_token = "<channel|>"
        tokenizer = SplitVocabTokenizer({"<channel|>": 101})

        def convert_tokens_to_ids(self, token):
            return {"<|channel>": 100}.get(token, 0)

        def convert_ids_to_tokens(self, token_id):
            return {100: "<|channel>"}.get(token_id, "<unk>")

    expected = ("<|channel>thought\n", "<channel|>")
    assert detect_reasoning_channel_markers(TemplateTokenizer()) == expected
    assert detect_reasoning_channel_markers(Processor()) == expected
    assert detect_reasoning_channel_markers(SplitProcessor()) == expected
    assert detect_reasoning_channel_markers(BadSplitProcessor()) is None
    assert detect_reasoning_channel_markers(LookalikeTokenizer()) is None
    assert detect_reasoning_channel_markers(VocabOnlyTokenizer()) is None
    assert detect_reasoning_channel_markers(object()) is None


def test_gemma_channel_normalization_is_prefix_monotonic_and_preserves_tools():
    parser = ReasoningChannelNormalizer("<|channel>thought\n", "<channel|>")
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
        "<think>Reason",
        "<think>Reason</think><|tool_call>web_search<tool_call|>",
    ]
    assert snapshots[1].startswith(snapshots[0])
