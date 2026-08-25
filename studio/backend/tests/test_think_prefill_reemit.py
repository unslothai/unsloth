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
