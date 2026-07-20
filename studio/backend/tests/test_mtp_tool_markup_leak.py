# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the MTP GGUF tool-call garbage bug (issue #7084).

On an MTP GGUF model, speculative decoding on a quantized target
(ggml-org/llama.cpp#25618) emits byte-fallback garbage that llama-server forwards as
U+FFFD plus an orphaned ``</tool_call>`` whose opener was drained/``�``-mangled;
``strip_tool_markup`` had no arm for a bare orphan close, so the reporter saw
``8��� </binary data> </tool_call>`` in chat.

The scrub is centralized at the tool-call chokepoints, so these tests pin only those:

  1. ``strip_tool_markup`` (finalized answer): scrubs U+FFFD / control chars and removes a
     trailing orphan-close run, while keeping well-formed stripping and mid-prose literals.
  2. ``strip_tool_markup_streaming`` (streaming display): the same scrub at the streaming
     entry, so the live display agrees with the finalized answer.
  3. ``sanitize_control_chars``: drops garbage but keeps ``\t \n \r`` / ESC.
  4. ``ToolLoopController.record_result``: scrubs a tool result before the model/card.
"""

import pytest

from core.inference.safetensors_agentic import strip_tool_markup_streaming
from core.inference.tool_call_parser import sanitize_control_chars, strip_tool_markup
from core.inference.tool_loop_controller import ToolCallDecision, ToolLoopController


# -- sanitize_control_chars ------------------------------------------


def test_sanitize_drops_replacement_and_control_chars():
    assert sanitize_control_chars("8��� ok") == "8 ok"
    assert sanitize_control_chars("a\x00b\x7fc\x9fd") == "abcd"


def test_sanitize_keeps_tab_newline_cr_and_esc():
    # ESC (\x1b) is preserved so terminal ANSI in a tool result survives.
    assert sanitize_control_chars("a\tb\nc\r\n\x1b[0m") == "a\tb\nc\r\n\x1b[0m"


def test_sanitize_noop_on_clean_text():
    s = "Perfectly normal answer with a supplementary-plane char \U00020000 and kanji 美味しい."
    assert sanitize_control_chars(s) == s


# -- strip_tool_markup (finalized answer): the reporter's exact garbage --


def test_reporter_garbage_is_cleaned():
    # Exact string from issue #7084; before the fix both the U+FFFD and the orphan
    # </tool_call> leaked.
    out = strip_tool_markup("8��� </binary data> </tool_call>", final = True)
    assert "�" not in out
    assert "</tool_call>" not in out
    # </binary data> is model-hallucinated prose, not a Studio token, so it is left as-is.
    assert out == "8 </binary data>"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Here is the answer.</tool_call>", "Here is the answer."),
        ("x<tool_call|>", "x"),
        ("answer</function>\n</tool_call>", "answer"),  # nested leak run ending in </tool_call>
        ("Here is the answer.</tool_call>\n", "Here is the answer."),  # trailing newline
        ("8��� </tool_call>\n", "8"),  # reporter's garbage + trailing newline
    ],
)
def test_trailing_orphan_closes_are_stripped_at_final(text, expected):
    assert strip_tool_markup(text, final = True) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Done.</function>",  # lone close, no </tool_call> sentinel
        "value</parameter>",
        "The XML closing tag is </function>",  # a code/XML answer ending on a literal
        "In XML you write <parameter>x</parameter> inside a tag.",  # mid-prose literal
    ],
)
def test_trailing_literal_close_without_tool_call_survives(text):
    # A trailing </function> / </parameter> with no </tool_call> sentinel reads as a
    # code/XML literal, not a leak, so it survives (a real leak carries </tool_call>).
    assert strip_tool_markup(text, final = True) == text


def test_wellformed_call_still_stripped():
    text = (
        "Prefix <tool_call>\n<function=web_search>\n<parameter=query>\nx\n"
        "</parameter>\n</function>\n</tool_call> suffix"
    )
    out = strip_tool_markup(text, final = True)
    assert "<tool_call>" not in out and "</tool_call>" not in out
    assert out == "Prefix  suffix"


def test_streaming_pass_final_false_buffers_orphan_but_scrubs():
    # final=False keeps in-progress markup buffered (orphan-run arm is end-of-turn only)
    # but still scrubs U+FFFD.
    assert strip_tool_markup("Here is the answer.</tool_call>", final = False) == (
        "Here is the answer.</tool_call>"
    )
    assert "�" not in strip_tool_markup("hi � there", final = False)


# -- strip_tool_markup_streaming (streaming display) chokepoint -------


def test_streaming_strips_trailing_orphan_close():
    # The MTP byte-fallback U+FFFD is scrubbed at the streaming entry, so the display strip
    # sees a bare orphan close and removes it, matching strip_tool_markup(final=True).
    assert strip_tool_markup_streaming("answer�</tool_call>") == "answer"
    # A genuine complete call is still fully stripped (no under-strip regression).
    assert strip_tool_markup_streaming('<tool_call>{"name":"x","arguments":{}}</tool_call>') == ""
    # A lone </function> without a sentinel is kept (likely code/XML), not over-stripped.
    assert strip_tool_markup_streaming("see </function> here") == "see </function> here"


def test_streaming_entry_scrubs_control_chars_even_when_disabled():
    # The entry scrub runs before the auto-heal gate, so byte-fallback garbage is dropped
    # from streamed display even with stripping disabled.
    out = strip_tool_markup_streaming(
        "hi �\x00 there", auto_heal_tool_calls = False, tool_protocol_active = False
    )
    assert "�" not in out and "\x00" not in out


def test_gguf_streaming_closure_wires_scrub_and_orphan_strip():
    # The GGUF streaming stripper is a nested closure, so pin the fix by source: its entry must
    # scrub control chars and its final-segment block must drop trailing orphan closes, matching
    # the safetensors path and strip_tool_markup(final=True).
    import inspect

    from core.inference.llama_cpp import LlamaCppBackend

    src = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
    assert "_sanitize_control_chars(text)" in src
    assert "_strip_trailing_orphan_close_run(seg)" in src


# -- record_result scrubs the tool result on both boundaries ---------


def _decision(name = "web_search"):
    return ToolCallDecision(
        action = "execute",
        tool_name = name,
        arguments = {"query": "x"},
        tool_call_id = "call_0",
        key = f"{name}:{{}}",
    )


def test_record_result_scrubs_tool_result_for_model_and_display():
    ctrl = ToolLoopController(tools = [{"function": {"name": "web_search"}}])
    dirty = "Title: Page�� body\x00 text�"
    completion = ctrl.record_result(_decision(), dirty)
    # Fed back to the model:
    assert "�" not in completion.model_message()["content"]
    assert "\x00" not in completion.model_message()["content"]
    # Shown in the tool card:
    assert "�" not in completion.tool_end_payload()["result"]
    assert completion.result == "Title: Page body text"


def test_record_result_keeps_clean_result_intact():
    ctrl = ToolLoopController(tools = [{"function": {"name": "web_search"}}])
    clean = "Title: Florida ACA 2026\nSilver premium: $1,900/mo"
    completion = ctrl.record_result(_decision(), clean)
    assert completion.result == clean


# -- Kimi + DeepSeek end-of-turn closers belong to the orphan-close set --
#
# ``_ORPHAN_CLOSE_TOKENS`` / ``_ORPHAN_SENTINELS`` also list the Kimi
# ``<|tool_call_end|><|tool_calls_section_end|>`` and DeepSeek
# ``<tool_call_end><tool_calls_end>`` closers: back-to-back special tokens, never legit
# prose, so a run whose opener was drained/U+FFFD-mangled is scrubbed like ``</tool_call>``.


def _kimi_deepseek_tokens():
    from core.inference.tool_call_parser import (
        _DEEPSEEK_CALL_END,
        _DEEPSEEK_END,
        _KIMI_CALL_END,
        _KIMI_SECTION_END,
    )
    return _KIMI_CALL_END, _KIMI_SECTION_END, _DEEPSEEK_CALL_END, _DEEPSEEK_END


def test_kimi_and_deepseek_trailing_closers_stripped_at_final():
    kimi_end, kimi_section_end, ds_call_end, ds_end = _kimi_deepseek_tokens()
    assert strip_tool_markup("answer " + kimi_end + kimi_section_end, final = True) == "answer"
    assert strip_tool_markup("answer " + ds_call_end + ds_end, final = True) == "answer"


def test_kimi_and_deepseek_closers_stripped_in_streaming():
    kimi_end, kimi_section_end, ds_call_end, ds_end = _kimi_deepseek_tokens()
    assert strip_tool_markup_streaming("answer" + kimi_end + kimi_section_end) == "answer"
    assert strip_tool_markup_streaming("answer" + ds_call_end + ds_end) == "answer"


def test_kimi_closer_in_mid_prose_survives():
    # Only TRAILING orphans are stripped: a token embedded in prose (with real text after) is
    # not a trailing run, so a plain answer is never over-stripped.
    kimi_end, *_ = _kimi_deepseek_tokens()
    text = "the token " + kimi_end + " appears mid sentence"
    assert strip_tool_markup(text, final = True) == text
