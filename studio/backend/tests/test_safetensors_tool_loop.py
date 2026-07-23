# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the safetensors agentic tool loop.

Covers the ``tool_call_parser`` helpers and the cumulative-text state machine in
``run_safetensors_tool_loop``, run against fake single-turn generators (no model
load). Edge cases: plain answers, JSON and XML tool-call forms, truncated/unclosed
calls, tool-result feedback, bad-JSON heal, duplicate-call short-circuit,
``__IMAGES__`` sentinel stripping, executor errors, cancel, and the iteration cap.
"""

import json
import threading
from typing import cast

import pytest

from core.inference import safetensors_agentic
from core.inference.safetensors_agentic import (
    _coerce_arguments,
    _detect_render_html_tool_start,
    run_safetensors_tool_loop,
    strip_tool_markup_streaming,
)
from core.inference.tool_call_parser import (
    RAG_MAX_SEARCHES_PER_TURN,
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from state import tool_approvals
from state.tool_approvals import resolve_tool_decision
from utils.datasets import is_gpt_oss_model_name


# ────────────────────────────────────────────────────────────────────
# parse_tool_calls_from_text
# ────────────────────────────────────────────────────────────────────


class TestParser:
    def test_json_tool_call(self):
        text = '<tool_call>{"name":"web_search","arguments":{"query":"hello"}}</tool_call>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        tc = result[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"
        # Arguments must always be a JSON string.
        assert isinstance(tc["function"]["arguments"], str)
        assert "hello" in tc["function"]["arguments"]

    def test_json_tool_call_unclosed(self):
        # No </tool_call>; balanced-brace extractor must still close it.
        text = '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"

    def test_json_tool_call_unclosed_requires_healing(self):
        text = '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}'
        assert parse_tool_calls_from_text(text)[0]["function"]["name"] == "python"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []

    def test_gemma_native_tool_call(self):
        text = '<|tool_call>call:terminal{command:"ls -la",workdir:"."}<tool_call|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "terminal"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"command": "ls -la", "workdir": "."}

    def test_gemma_native_tool_call_template_quotes(self):
        text = '<|tool_call>call:web_search{query:<|"|>openai news<|"|>}<tool_call|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "openai news"}

    def test_gemma_native_tool_call_template_quotes_escape_backslashes(self):
        text = r'<|tool_call>call:ls{path:<|"|>C:\Users\wasim\repo<|"|>}<tool_call|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "ls"
        assert json.loads(result[0]["function"]["arguments"]) == {"path": r"C:\Users\wasim\repo"}

    def test_gemma_native_tool_call_hyphenated_argument_name(self):
        text = '<|tool_call>call:mcp__srv__create-issue{issue-title:"Bug report"}<tool_call|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "mcp__srv__create-issue"
        assert json.loads(result[0]["function"]["arguments"]) == {"issue-title": "Bug report"}

    def test_gemma_native_tool_call_keeps_braces_inside_string_value(self):
        text = '<|tool_call>call:terminal{command:"echo {foo:bar}"}<tool_call|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "terminal"
        assert json.loads(result[0]["function"]["arguments"]) == {"command": "echo {foo:bar}"}

    def test_gemma_native_tool_call_bare_string_values(self):
        text = "<|tool_call>call:get_weather{location:Tokyo,unit:celsius}<tool_call|>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {
            "location": "Tokyo",
            "unit": "celsius",
        }

    def test_xml_function_call(self):
        text = "<function=python><parameter=code>print('hi')</parameter></function>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"
        assert "print('hi')" in result[0]["function"]["arguments"]

    def test_xml_param_preserves_leading_indentation(self):
        import json

        # Only the wrapping newline is trimmed; code-argument indentation survives.
        text = (
            "<function=python><parameter=code>\n"
            "    indented = 1\n"
            "    more\n"
            "</parameter></function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {
            "code": "    indented = 1\n    more"
        }

    def test_xml_unclosed(self):
        # Closing tags omitted; parser must still extract the value.
        text = "<function=terminal><parameter=command>ls -la"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "terminal"
        assert "ls -la" in result[0]["function"]["arguments"]

    def test_xml_unclosed_requires_healing(self):
        text = "<function=terminal><parameter=command>ls -la"
        assert parse_tool_calls_from_text(text)[0]["function"]["name"] == "terminal"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []

    def test_code_with_embedded_xml(self):
        # A code parameter with a literal </parameter> must not truncate: the
        # parser uses end-of-body as the only boundary for single-param calls.
        text = (
            "<function=python><parameter=code>html = '<a></a>'\nprint('hi')</parameter></function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert "print('hi')" in result[0]["function"]["arguments"]

    def test_xml_param_preserves_leading_indentation(self):
        # Only the wrapping newline is trimmed, so code-argument indentation survives (str.strip() destroyed it).
        text = (
            "<function=python><parameter=code>\n"
            "    indented = 1\n"
            "    more\n"
            "</parameter></function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {
            "code": "    indented = 1\n    more"
        }

    def test_function_signal_inside_parameter_is_literal(self):
        text = (
            "<function=python>"
            "<parameter=code>print('<function=render_html>')</parameter>"
            "</function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"
        assert "<function=render_html>" in result[0]["function"]["arguments"]

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>'
            '<tool_call>{"name":"web_search","arguments":{"query":"b"}}</tool_call>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "web_search"
        assert result[1]["function"]["name"] == "web_search"

    def test_bad_json_does_not_raise(self):
        text = "<tool_call>{not valid json}</tool_call>"
        result = parse_tool_calls_from_text(text)
        # Bad JSON is dropped silently; caller can fall back to text.
        assert result == []

    def test_has_tool_signal(self):
        assert has_tool_signal("blah <tool_call> x")
        assert has_tool_signal("blah <|tool_call>call:terminal")
        assert has_tool_signal("hi <function=foo>...")
        assert has_tool_signal("ok [TOOL_CALLS]web_search{...")
        assert has_tool_signal("fine python[ARGS]{...")
        assert not has_tool_signal("hello world")

    def test_render_html_start_detector_uses_first_tool(self):
        assert _detect_render_html_tool_start("<function=render_html>")
        assert _detect_render_html_tool_start(
            '<tool_call>{"name":"render_html","arguments":{"code":"<html>"}'
        )
        assert not _detect_render_html_tool_start(
            "<function=python><parameter=code>'<function=render_html>'"
        )
        assert not _detect_render_html_tool_start(
            '<tool_call>{"name":"python","arguments":{"code":"<function=render_html>"}}'
        )

    def test_render_html_start_detector_covers_mistral_and_rehearsal_forms(self):
        # The provisional render-html card must fire for bracket-tag forms too, not only XML.
        assert _detect_render_html_tool_start('[TOOL_CALLS]render_html{"code":"<html>"}')
        assert _detect_render_html_tool_start('[TOOL_CALLS]render_html[ARGS]{"code":"x"}')
        assert _detect_render_html_tool_start(
            '[TOOL_CALLS] [{"name":"render_html","arguments":{}}]'
        )
        assert _detect_render_html_tool_start('render_html[ARGS]{"code":"<html>"}')
        # A different first tool (or a prose mention with no JSON body) must not fire.
        assert not _detect_render_html_tool_start('[TOOL_CALLS]web_search{"q":"x"}')
        assert not _detect_render_html_tool_start('web_search[ARGS]{"q":"x"}')
        assert not _detect_render_html_tool_start('python[ARGS]{"code":"render_html[ARGS]{}"}')
        assert not _detect_render_html_tool_start("use render_html[ARGS] to render")

    def test_render_html_start_detector_skips_think_block_rehearsal(self):
        # A render_html rehearsed inside think must not fire the card; the outside-think call decides.
        assert not _detect_render_html_tool_start(
            '<think>draft render_html[ARGS]{"code":"x"}</think>python[ARGS]{"code":"print(1)"}'
        )
        assert not _detect_render_html_tool_start(
            '[THINK]render_html[ARGS]{"code":"x"}[/THINK]web_search[ARGS]{"q":"y"}'
        )
        # A real render_html AFTER a rehearsed non-render_html inside think still fires.
        assert _detect_render_html_tool_start(
            '<think>web_search[ARGS]{"q":"x"}</think>render_html[ARGS]{"code":"<html>"}'
        )
        # A render_html rehearsed inside think with no real call after does not fire.
        assert not _detect_render_html_tool_start('<think>render_html[ARGS]{"code":"x"}</think>')

    def test_render_html_start_detector_reads_top_level_array_name(self):
        # Array form: the name is the object's top-level ``"name"``, not an argument key.
        assert not _detect_render_html_tool_start(
            '[TOOL_CALLS] [{"arguments":{"name":"render_html"},"name":"python"}]'
        )
        assert _detect_render_html_tool_start(
            '[TOOL_CALLS] [{"arguments":{"name":"python"},"name":"render_html"}]'
        )

    def test_strip_markup_closed(self):
        text = "before <tool_call>{}</tool_call> after"
        assert strip_tool_markup(text) == "before  after"
        text = 'before <|tool_call>call:terminal{command:"ls"}<tool_call|> after'
        assert strip_tool_markup(text) == "before  after"

    def test_strip_named_mistral_call_consumes_trailing_eos(self):
        # The named ``[TOOL_CALLS]name{json}`` shape must eat the optional
        # trailing ``</s>`` like the array shape, so the EOS marker is not left
        # behind as visible content.
        text = '[TOOL_CALLS]web_search{"query":"cats"}</s>'
        assert strip_tool_markup(text) == ""
        text = '[TOOL_CALLS]web_search{"query":"cats"}</s> and then'
        assert strip_tool_markup(text) == " and then"

    def test_strip_markup_unclosed_final(self):
        text = "before <tool_call>{partial"
        # final=True drops the trailing run.
        assert strip_tool_markup(text, final = True) == "before"
        # Without final=True the unclosed run is preserved.
        assert "partial" in strip_tool_markup(text)
        assert strip_tool_markup("before <|tool_call>call:terminal{", final = True) == "before"

    def test_streaming_strip_respects_disabled_healing(self):
        raw = 'before <tool_call>{"name":"web_search"'
        assert strip_tool_markup_streaming(raw, auto_heal_tool_calls = False) == raw
        assert strip_tool_markup_streaming(raw) == "before "

    def test_streaming_strip_respects_disabled_healing_without_tool_protocol(self):
        raw = 'before <tool_call>{"name":"web_search"'
        assert strip_tool_markup_streaming(raw, auto_heal_tool_calls = False) == raw
        assert (
            strip_tool_markup_streaming(
                raw,
                auto_heal_tool_calls = False,
                tool_protocol_active = True,
            )
            == "before "
        )

    # Mistral [TOOL_CALLS] bracket-tag.

    def test_mistral_bracket_basic(self):
        # Devstral / Mistral-Small fallback when bypassing native FC.
        text = '[TOOL_CALLS]web_search{"query":"weather"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        assert isinstance(result[0]["function"]["arguments"], str)
        assert "weather" in result[0]["function"]["arguments"]

    def test_rehearsal_inside_unclosed_think_is_ignored(self):
        """Rehearsal-shaped markup inside an unclosed <think> block must
        not be executed as a real tool call. Mid-stream the </think>
        tag has not arrived yet, so the strip regex has to accept
        end-of-string as a terminator. Regression for the Gemini
        high-severity flag on this PR."""
        text = (
            "<think>I should call web_search[ARGS]" '{"query":"weather"} next to find the answer.'
        )
        result = parse_tool_calls_from_text(text)
        # Inside an unclosed think block no calls are yielded.
        assert result == []

    def test_rehearsal_inside_unclosed_bracket_think_is_ignored(self):
        text = "[THINK]planning to use python[ARGS]" '{"code":"print(1)"} but not yet.'
        result = parse_tool_calls_from_text(text)
        assert result == []

    def test_rehearsal_after_closed_think_still_parsed(self):
        text = "<think>planning</think>" 'web_search[ARGS]{"code":"print(1)"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_rehearsal_inside_prefilled_think_is_ignored(self):
        """Reasoning models (Qwen3.5 enable_thinking) open <think> in the PROMPT,
        so generated content starts inside the thought and carries only a closing
        </think>. A call rehearsed in that leading thought must be skipped, while a
        real call after the close still fires."""
        text = (
            'planning web_search[ARGS]{"query":"draft"}</think>get_weather[ARGS]{"code":"print(1)"}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"

    def test_literal_close_think_in_leading_argument_not_prefill(self):
        """A </think> literal inside a real leading call's arguments must not be
        read as a prefilled-reasoning close (which would skip the call)."""
        text = 'web_search[ARGS]{"query":"what is </think>"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_stray_close_after_real_call_not_treated_as_prefill(self):
        """A real leading call followed by a stray </think> and no further call is
        a normal answer, not prefilled reasoning; the call must still fire (the
        virtual span only applies when a real call follows the close)."""
        text = 'Now web_search[ARGS]{"query":"x"}</think> answer'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_mistral_bracket_with_whitespace(self):
        # Optional whitespace (incl. newlines) between the name and the opening brace.
        text = '[TOOL_CALLS]python  \n  {"code":"print(1)"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"
        assert "print(1)" in result[0]["function"]["arguments"]

    def test_mistral_bracket_nested_json(self):
        # Brace-balance scan handles nested objects and braces inside string literals.
        text = "[TOOL_CALLS]web_search" '{"query":"a {nested} brace","opts":{"limit":5}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        import json as _json

        args = _json.loads(result[0]["function"]["arguments"])
        assert args["query"] == "a {nested} brace"
        assert args["opts"] == {"limit": 5}

    def test_mistral_bracket_with_prose(self):
        # Bracket-tag surrounded by prose is still recognised.
        text = (
            "Sure, I will look that up.\n"
            '[TOOL_CALLS]web_search{"query":"weather"}\n'
            "Calling now."
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_mistral_bracket_bad_json_dropped(self):
        text = "[TOOL_CALLS]web_search{not valid}"
        result = parse_tool_calls_from_text(text)
        # No usable tool call; callers fall back to text.
        assert result == []

    def test_mistral_bracket_object_with_array_value(self):
        # Args must be a JSON object; a dict wrapping an array value is accepted.
        text = '[TOOL_CALLS]web_search{"opts":[1,2,3]}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    # Rehearsal syntax name[ARGS]{json}.

    def test_rehearsal_basic(self):
        text = 'web_search[ARGS]{"code":"print(1)"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        assert "print(1)" in result[0]["function"]["arguments"]

    def test_rehearsal_with_prose(self):
        text = "I should call the web_search tool. Like this: " 'web_search[ARGS]{"code":"x = 1"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_rehearsal_bad_json_dropped(self):
        text = "python[ARGS]{not valid json}"
        result = parse_tool_calls_from_text(text)
        assert result == []

    def test_mistral_bracket_hyphenated_mcp_name(self):
        # Dashed MCP names must be captured whole, not truncated at the first dash.
        text = '[TOOL_CALLS]mcp__srv__list-issues{"q":"x"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "mcp__srv__list-issues"

    def test_rehearsal_hyphenated_mcp_name(self):
        text = 'mcp__srv__list-issues[ARGS]{"q":"x"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "mcp__srv__list-issues"

    def test_streaming_strip_removes_partial_bracket_marker(self):
        # A bracket tag streamed before its opening brace must strip on the final pass, not leak.
        assert strip_tool_markup("answer [TOOL_CALLS]web_search", final = True) == "answer"
        assert strip_tool_markup("text get_weather[ARGS]", final = True) == "text"
        # Non-final must keep the in-progress tag buffered (not yet stripped).
        partial = "answer [TOOL_CALLS]web_search"
        assert strip_tool_markup(partial, final = False) == partial

    def test_strip_removes_two_level_nested_bracket_call_keeps_prose(self):
        # Two-level-nested args must be removed whole; the balanced scan handles any depth.
        text = 'before [TOOL_CALLS]search{"f":{"g":{"h":1}}} after'
        assert strip_tool_markup(text, final = False) == "before  after"
        assert strip_tool_markup(text, final = True) == "before  after"

    def test_strip_removes_call_with_literal_think_in_argument(self):
        # A literal think block inside arguments strips with the call, not as a reasoning block.
        text = (
            '<tool_call>{"name":"write","arguments":'
            '{"text":"compare <think> and </think> tags"}}</tool_call>'
        )
        assert strip_tool_markup(text, final = True) == ""

    def test_strip_preserves_real_think_but_strips_call_with_literal_think(self):
        text = (
            "<think>planning</think> ok "
            '<tool_call>{"name":"w","arguments":{"t":"<think>x</think>"}}</tool_call> done'
        )
        out = strip_tool_markup(text, final = True)
        assert "<think>planning</think>" in out
        assert "<tool_call>" not in out and '"name"' not in out
        assert "ok" in out and "done" in out

    def test_prose_mentioning_args_marker_is_not_truncated(self):
        # ``foo[ARGS] to the template`` is prose; the catch-all must not delete the sentence.
        text = "Please pass foo[ARGS] to the template and continue reading."
        assert strip_tool_markup(text, final = True) == text

    def test_streaming_strip_handles_mistral_v11_call_id_args(self):
        # The streaming strip uses the regex patterns directly, so they must cover the v11
        # [CALL_ID]/[ARGS] metadata (aligned with the parser).
        raw = 'before [TOOL_CALLS]web_search[CALL_ID]abc123[ARGS]{"q":"x"} after'
        out = strip_tool_markup_streaming(raw)
        assert "[TOOL_CALLS]" not in out and "[CALL_ID]" not in out and "[ARGS]" not in out
        assert "before" in out and "after" in out

    # <think> pre-strip.

    def test_think_block_stripped_before_xml(self):
        # The think block is stripped before matching so the post-thinking call is recognised.
        text = (
            "<think>I will use web_search to find the weather.</think>"
            '<tool_call>{"name":"web_search","arguments":{"query":"sf"}}</tool_call>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_think_block_stripped_before_bracket_tag(self):
        text = (
            "<think>Let me search for that.</think>\n" '[TOOL_CALLS]web_search{"query":"weather"}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_uppercase_think_tag_stripped(self):
        # Some templates use [THINK]...[/THINK] instead of <think>.
        text = "[THINK]planning my next call[/THINK]" '[TOOL_CALLS]python{"code":"print(1)"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"

    def test_think_block_hides_inner_tool_call(self):
        # A call mentioned inside think is a rehearsal; the wrapper strip removes the inner markup.
        text = (
            "<think>I might call "
            '<tool_call>{"name":"web_search","arguments":{}}</tool_call> '
            "but I am not sure</think>\n"
            "Let me just answer directly."
        )
        result = parse_tool_calls_from_text(text)
        assert result == []

    def test_think_literal_inside_real_tool_argument_is_preserved(self):
        # A real call whose argument contains a literal think tag must not be corrupted.
        text = (
            '<tool_call>{"name":"write","arguments":'
            '{"text":"compare <think> and </think> tags"}}</tool_call>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"])["text"] == (
            "compare <think> and </think> tags"
        )

    def test_bracket_tag_argument_with_think_literal_is_preserved(self):
        text = '[TOOL_CALLS]search{"q":"explain [THINK] blocks"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"])["q"] == "explain [THINK] blocks"

    def test_real_call_after_think_with_rehearsal_inside(self):
        # A rehearsal inside <think> is skipped, but the real call after the close tag parses.
        text = '<think>plan: search[ARGS]{"q":"x"}</think>search[ARGS]{"q":"real"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"])["q"] == "real"

    # XML takes precedence over bracket-tag.

    def test_xml_wins_over_bracket(self):
        # When a model emits both forms in one message, the XML form is canonical and wins.
        text = (
            '<tool_call>{"name":"primary","arguments":{}}</tool_call>'
            '[TOOL_CALLS]secondary{"k":"v"}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "primary"

    # Strip patterns include bracket-tag and rehearsal.

    def test_strip_bracket_tag_closed(self):
        text = 'before [TOOL_CALLS]web_search{"q":"hi"} after'
        assert "[TOOL_CALLS]" not in strip_tool_markup(text)
        assert "before" in strip_tool_markup(text)
        assert "after" in strip_tool_markup(text)

    def test_strip_rehearsal_closed(self):
        text = 'prose web_search[ARGS]{"code":"x"} more prose'
        cleaned = strip_tool_markup(text)
        assert "[ARGS]" not in cleaned
        assert "prose" in cleaned
        assert "more prose" in cleaned

    def test_strip_bracket_tag_unclosed_final(self):
        text = 'before [TOOL_CALLS]web_search{"q":"part'
        # Final-mode strip drops the trailing unclosed run.
        cleaned = strip_tool_markup(text, final = True)
        assert "TOOL_CALLS" not in cleaned
        assert cleaned == "before"

    # Canonical Mistral array, v11 [CALL_ID], unified multi-call (PR review fixes).

    def test_mistral_canonical_array_is_parsed(self):
        # Canonical multi-call array: every call must parse (was dropped then deleted to EOS).
        text = '[TOOL_CALLS] [{"name":"a","arguments":{"x":1}},{"name":"b","arguments":{"y":2}}]'
        result = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in result] == ["a", "b"]
        assert json.loads(result[0]["function"]["arguments"]) == {"x": 1}
        assert json.loads(result[1]["function"]["arguments"]) == {"y": 2}

    def test_mistral_array_string_arguments_are_decoded(self):
        # OpenAI-spec arguments arrive as a JSON string; decode to an object.
        text = '[TOOL_CALLS] [{"name":"a","arguments":"{\\"x\\":1}"}]'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {"x": 1}

    def test_mistral_array_scalar_string_argument_not_double_encoded(self):
        # A bare scalar string argument in the Mistral array form must be kept
        # raw, exactly like the <tool_call> path, so the downstream argument
        # healer wraps ``weather`` into the single-string tool's key -- not
        # ``"weather"`` with literal quotes from a redundant json.dumps.
        array = parse_tool_calls_from_text(
            '[TOOL_CALLS][{"name":"web_search","arguments":"weather"}]'
        )
        xml = parse_tool_calls_from_text(
            '<tool_call>{"name":"web_search","arguments":"weather"}</tool_call>'
        )
        assert array[0]["function"]["arguments"] == xml[0]["function"]["arguments"] == "weather"
        healed = _coerce_arguments(
            array[0]["function"]["arguments"], heal = True, tool_name = "web_search"
        )
        assert healed == {"query": "weather"}

    def test_mistral_array_strip_keeps_trailing_prose(self):
        # The array form must be removed whole, not deleted to end-of-string.
        text = 'answer [TOOL_CALLS] [{"name":"a","arguments":{}}] tail'
        assert strip_tool_markup(text, final = True) == "answer  tail"

    def test_mistral_and_rehearsal_in_one_message_both_parse(self):
        # A Mistral call and a rehearsal call together: both must parse.
        text = '[TOOL_CALLS]a{"x":1} then b[ARGS]{"y":2}'
        result = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in result] == ["a", "b"]

    def test_mistral_v11_call_id_is_not_the_function_name(self):
        # v11 shape: the function name is ``name``, never the opaque call-id token.
        result = parse_tool_calls_from_text('[TOOL_CALLS]get_weather[CALL_ID]abc123[ARGS]{"q":"x"}')
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "x"}
        # v11 without a call-id parses the same name.
        r2 = parse_tool_calls_from_text('[TOOL_CALLS]get_weather[ARGS]{"q":"y"}')
        assert r2[0]["function"]["name"] == "get_weather"

    def test_strip_preserves_rehearsal_inside_think(self):
        # A rehearsal inside <think> is reasoning; strip keeps it verbatim.
        text = '<think>plan: search[ARGS]{"q":"x"}</think> A'
        out = strip_tool_markup(text, final = True)
        assert out == text
        assert "search[ARGS]" in out

    def test_streaming_strip_preserves_rehearsal_inside_think(self):
        # The streaming strip must also preserve a think rehearsal: a mid-stream strip shrinks
        # then regrows the cumulative text (corrupts append-by-length consumers). Matches GGUF.
        text = '<think>plan: search[ARGS]{"q":"x"}</think> A'
        assert strip_tool_markup_streaming(text) == text
        assert strip_tool_markup_streaming(text, tool_protocol_active = True) == text
        # An unclosed block during streaming is preserved too (the parser keeps it).
        partial = '<think>plan: search[ARGS]{"q":"x"}'
        assert strip_tool_markup_streaming(partial, tool_protocol_active = True) == partial

    def test_streaming_strip_still_removes_real_call_outside_think(self):
        # The think guard must not stop the streaming strip removing a call outside the block.
        text = '<think>reason</think> web_search[ARGS]{"q":"x"}'
        out = strip_tool_markup_streaming(text, tool_protocol_active = True)
        assert "web_search[ARGS]" not in out
        assert "<think>reason</think>" in out

    def test_strip_bracket_calls_is_linear(self):
        # Many complete bracket calls must strip in ~linear time (was O(n^2) per match).
        import time

        text = '[TOOL_CALLS]f{"a":1}' * 4000  # ~80KB, 4000 complete calls
        t0 = time.perf_counter()
        out = strip_tool_markup(text, final = True)
        elapsed = time.perf_counter() - t0
        assert "[TOOL_CALLS]" not in out
        assert elapsed < 1.0, f"strip took {elapsed * 1000:.0f}ms on 4000 bracket calls"

    def test_streaming_strip_handles_nested_mistral_json(self):
        # The non-greedy [TOOL_CALLS]name{...} pattern truncates nested JSON at the first }; the
        # balanced helper must remove the whole call so no trailing brace leaks to the streaming ...
        raw = 'ok [TOOL_CALLS]foo{"a":{"b":1}} tail'
        out = strip_tool_markup_streaming(raw)
        assert "[TOOL_CALLS]" not in out
        assert "}" not in out
        assert "ok " in out and "tail" in out

    def test_streaming_strip_handles_nested_wrapperless_gemma(self):
        # Same class of bug for the wrapper-less Gemma call:NAME{...} form with a
        # nested object argument.
        raw = "ok call:f{loc:{city:NYC},n:3} tail"
        out = strip_tool_markup_streaming(raw)
        assert "call:f" not in out
        assert "}" not in out
        assert "ok " in out and "tail" in out

    def test_streaming_strip_keeps_prose_after_function_xml_with_literal_marker(self):
        # A literal ``<function=...>`` in a value is data: the strip must close at the REAL
        # ``</function>`` and keep trailing prose (the open-ended regex ate to EOF).
        raw = (
            "pref <function=python><parameter=code>"
            'print("<function=x>")</parameter></function> tail'
        )
        assert strip_tool_markup_streaming(raw) == "pref  tail"
        # Streaming and final strip agree on the visible text (final also trims).
        assert strip_tool_markup_streaming(raw) == strip_tool_markup(raw, final = True)

    def test_streaming_strip_drops_leading_magistral_reasoning(self):
        # Magistral emits reasoning as a leading ``[THINK]...[/THINK]`` bracket block
        # (not the ``<think>`` the reasoning channel renders). The streaming display
        # strip must drop it so the raw chain-of-thought does not leak into the
        # safetensors content; GGUF routes it to reasoning_content natively.
        closed = "[THINK]Let me think. 2+2 is 4.[/THINK]The answer is 4."
        assert strip_tool_markup_streaming(closed) == "The answer is 4."
        assert strip_tool_markup_streaming(closed) == strip_tool_markup(closed, final = True)
        # Unclosed mid-stream reasoning is held from the marker on (nothing leaks, and
        # the cleaned text only grows as the answer streams in after ``[/THINK]``).
        assert strip_tool_markup_streaming("[THINK]still thinking") == ""
        assert strip_tool_markup_streaming("[THINK]r[/THINK]The") == "The"
        assert strip_tool_markup_streaming("[THINK]r[/THINK]The answer") == "The answer"
        # A non-leading ``[THINK]`` is ordinary prose and is left untouched.
        assert strip_tool_markup_streaming("hi [THINK] later") == "hi [THINK] later"


class TestParserMultiFormat:
    """Shared-parser coverage: every family's emission maps to the same OpenAI shape."""

    # Llama-3

    def test_llama3_python_tag_dot_call(self):
        # Llama-3 built-in tools: <|python_tag|>NAME.call(k="v", ...).
        import json

        text = '<|python_tag|>brave_search.call(query="weather in Tokyo")'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "brave_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "weather in Tokyo"}

    def test_llama3_python_tag_dot_call_multi_arg(self):
        import json

        text = "<|python_tag|>get_weather.call(" 'location="Tokyo", units="celsius", days=5)'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"location": "Tokyo", "units": "celsius", "days": 5}

    def test_llama3_python_tag_json_form(self):
        import json

        text = '<|python_tag|>{"name":"web_search","parameters":{"query":"hi","n":5}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "hi", "n": 5}

    def test_llama3_python_tag_json_form_with_eom(self):
        # Llama-3 emits ``<|eom_id|>`` after the JSON; must not break parsing.
        import json

        text = '<|python_tag|>{"name":"python","parameters":{"code":"print(2+2)"}}<|eom_id|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"code": "print(2+2)"}

    def test_llama3_strip_markup_final(self):
        text = '<|python_tag|>brave_search.call(query="x")'
        assert strip_tool_markup(text, final = True) == ""

    def test_llama3_python_tag_json_form_non_scalar_args_skipped(self):
        # Should NOT fabricate ``{"value": args}`` when the JSON form
        # has a non-dict / non-string ``arguments`` value.
        for bad in (
            '<|python_tag|>{"name":"foo","arguments":42}',
            '<|python_tag|>{"name":"foo","arguments":[1,2,3]}',
            '<|python_tag|>{"name":"foo","arguments":null}',
            '<|python_tag|>{"name":"foo","arguments":true}',
        ):
            assert parse_tool_calls_from_text(bad) == [], bad

    # ── Llama-3.2 bare JSON ``custom_tools`` ─────────────────────

    def test_llama3_2_bare_json_parameters(self):
        # Llama-3.2-Instruct emits bare JSON directly as content; no
        # <|python_tag|> prefix per its training template.
        import json

        text = '{"name":"web_search","parameters":{"query":"Tokyo weather"}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "Tokyo weather"}

    def test_llama3_2_bare_json_arguments_key(self):
        import json

        text = '{"name":"add","arguments":{"a":1,"b":2}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"a": 1, "b": 2}

    def test_llama3_2_bare_json_multi_call(self):
        # Llama-3 may chain calls with ``; `` per training template.
        text = '{"name":"a","parameters":{}}; {"name":"b","parameters":{}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_llama3_2_bare_json_with_eom_sentinel(self):
        text = '{"name":"x","parameters":{"y":1}}<|eom_id|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "x"

    def test_llama3_2_bare_json_leading_sentinel_skipped(self):
        # Sometimes prior <|eot_id|> leaks into the next turn.
        text = '<|eot_id|>{"name":"x","parameters":{}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "x"

    def test_llama3_2_bare_json_plain_prose_does_not_fire(self):
        # Defensive: must NOT fire on plain assistant prose.
        text = "Hello world, how are you today?"
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_embedded_in_prose_does_not_fire(self):
        # Defensive: JSON embedded in prose must NOT fire (parser is
        # strict about content STARTING with `{`).
        text = 'The tool result was: {"name":"foo"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_missing_name_does_not_fire(self):
        text = '{"result":"ok","data":[1,2,3]}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_missing_args_does_not_fire(self):
        text = '{"name":"x"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_args_not_dict_does_not_fire(self):
        text = '{"name":"x","parameters":42}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_string_parameters_does_not_fire(self):
        # Llama-3 spec: parameters must be a dict. Prose like
        # ``{"name":"foo","parameters":"a sentence"}`` must NOT trigger.
        text = '{"name":"foo","parameters":"this is a sentence"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_string_arguments_not_json_does_not_fire(self):
        # OpenAI ``arguments`` may be a JSON-string of a dict, but a
        # plain non-JSON string must not pass the guard.
        text = '{"name":"foo","arguments":"not json"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_string_arguments_json_dict_fires(self):
        # OpenAI shape: arguments is a JSON-encoded string of a dict.
        text = '{"name":"foo","arguments":"{\\"q\\":\\"x\\"}"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "foo"
        # arguments stays as the original JSON-string.
        assert result[0]["function"]["arguments"] == '{"q":"x"}'

    def test_llama3_2_bare_json_string_arguments_json_non_dict_does_not_fire(self):
        # JSON-string that parses to a list / scalar / null must NOT fire.
        for bad in (
            '{"name":"foo","arguments":"[1,2,3]"}',
            '{"name":"foo","arguments":"\\"plain\\""}',
            '{"name":"foo","arguments":"null"}',
            '{"name":"foo","arguments":"42"}',
        ):
            assert parse_tool_calls_from_text(bad) == [], bad

    # Mistral pre-v11

    def test_mistral_pre_v11_array(self):
        import json

        text = '[TOOL_CALLS] [{"name":"web_search","arguments":{"query":"hello"},"id":"abc"}]'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        # Mistral provides its own id; preserve it.
        assert result[0]["id"] == "abc"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "hello"}

    def test_mistral_array_parameters_key_alias(self):
        import json

        # Array object keyed on ``parameters`` (not ``arguments``) must keep its
        # payload, matching the JSON/XML paths and SGLang's base detector.
        text = '[TOOL_CALLS] [{"name":"get_weather","parameters":{"city":"Paris"}}]'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Paris"}

    def test_mistral_pre_v11_array_multi(self):
        text = (
            '[TOOL_CALLS] [{"name":"a","arguments":{"x":1},"id":"id1"},'
            '{"name":"b","arguments":{"y":2},"id":"id2"}]'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_mistral_pre_v11_unclosed_array(self):
        # Closing ``]`` truncated -- parser must heal off individual objects.
        text = '[TOOL_CALLS] [{"name":"web_search","arguments":{"q":"x"},"id":"id"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    # Mistral v11+

    def test_mistral_v11_single(self):
        # Magistral / Mistral Small 3.1: bare ``name{json}`` after trigger.
        import json

        text = '[TOOL_CALLS]add{"a":3.5,"b":4}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "add"
        assert json.loads(result[0]["function"]["arguments"]) == {"a": 3.5, "b": 4}

    def test_mistral_v11_parallel(self):
        # v11+ parallel: ``[TOOL_CALLS]a{...}[TOOL_CALLS]b{...}``.
        text = '[TOOL_CALLS]add{"a":1}[TOOL_CALLS]sub{"b":2}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "add"
        assert result[1]["function"]["name"] == "sub"

    def test_mistral_v11_with_args_marker(self):
        # Ministral / Mistral Large 3: ``[TOOL_CALLS]name[ARGS]{json}``.
        import json

        text = '[TOOL_CALLS]add[ARGS]{"a":1,"b":2}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "add"
        assert json.loads(result[0]["function"]["arguments"]) == {"a": 1, "b": 2}

    def test_mistral_strip_markup_v11(self):
        text = '[TOOL_CALLS]add{"a":1}'
        assert strip_tool_markup(text, final = True) == ""

    def test_mistral_call_id_form(self):
        # Mistral Small 3.2: ``[TOOL_CALLS]name[CALL_ID]<id>[ARGS]{json}``.
        # The ``[CALL_ID]`` segment must be skipped, not treated as a stop
        # (llama.cpp test-chat.cpp:4785 parses this to one call).
        import json

        text = '[TOOL_CALLS]special_function[CALL_ID]123456789[ARGS]{"arg1": 1}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "special_function"
        assert json.loads(result[0]["function"]["arguments"]) == {"arg1": 1}

    def test_mistral_call_id_form_parallel(self):
        text = (
            '[TOOL_CALLS]special_function[CALL_ID]000000001[ARGS]{"arg1": 1}'
            "[TOOL_CALLS]special_function_with_opt[CALL_ID]000000002"
            '[ARGS]{"arg1": 1, "arg2": 2}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "special_function"
        assert result[1]["function"]["name"] == "special_function_with_opt"

    def test_mistral_call_id_form_stripped(self):
        text = '[TOOL_CALLS]special_function[CALL_ID]123456789[ARGS]{"arg1": 1}'
        assert strip_tool_markup(text, final = True) == ""

    def test_mistral_think_reasoning_ignored(self):
        # Magistral wraps reasoning in ``[THINK]...[/THINK]``. A ``[TOOL_CALLS]``
        # inside the reasoning is chain-of-thought, not a real call; only the
        # call after ``[/THINK]`` counts (llama.cpp test-chat.cpp:2285).
        import json

        text = (
            '[THINK]Let me think about [TOOL_CALLS]fake[ARGS]{"x":1} '
            'and more[/THINK][TOOL_CALLS]real_fn[ARGS]{"y":2}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "real_fn"
        assert json.loads(result[0]["function"]["arguments"]) == {"y": 2}

    def test_mistral_think_reasoning_no_real_call(self):
        # Reasoning that merely mentions a tool call but does not emit one
        # after ``[/THINK]`` yields no calls.
        text = '[THINK]I might call [TOOL_CALLS]fake[ARGS]{"x":1}[/THINK]Done.'
        assert parse_tool_calls_from_text(text) == []

    def test_mistral_think_literal_in_argument_preserved(self):
        # A literal ``[THINK]`` inside a real tool argument (after the call)
        # must not be stripped or corrupt the parse.
        import json

        text = '[TOOL_CALLS]search[ARGS]{"q":"explain the [THINK] token"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "explain the [THINK] token"}

    # Gemma 4

    def test_gemma4_simple_call(self):
        import json

        text = (
            "<|tool_call>call:get_weather{"
            'location:<|"|>Tokyo<|"|>,units:<|"|>celsius<|"|>}<tool_call|>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"location": "Tokyo", "units": "celsius"}

    def test_gemma4_with_primitives(self):
        import json

        text = (
            "<|tool_call>call:set_pref{"
            "enabled:true,attempts:5,threshold:1.5,nickname:null}<tool_call|>"
        )
        result = parse_tool_calls_from_text(text)
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"enabled": True, "attempts": 5, "threshold": 1.5, "nickname": None}

    def test_gemma4_nested_args(self):
        # Gemma 4 nests dicts / lists with bare keys and ``<|"|>`` strings.
        import json

        text = (
            "<|tool_call>call:search{"
            'query:<|"|>foo<|"|>,filters:{site:<|"|>example.com<|"|>,recent:true},'
            'tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>'
        )
        result = parse_tool_calls_from_text(text)
        args = json.loads(result[0]["function"]["arguments"])
        assert args["query"] == "foo"
        assert args["filters"] == {"site": "example.com", "recent": True}
        assert args["tags"] == ["a", "b"]

    def test_gemma4_multi_call(self):
        text = "<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:2}<tool_call|>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_gemma4_unclosed_does_not_raise(self):
        # Truncated mid-stream; must not raise.
        text = '<|tool_call>call:foo{x:<|"|>bar<|"|>'
        result = parse_tool_calls_from_text(text)
        assert isinstance(result, list)

    def test_gemma4_strip_markup_final(self):
        text = "<|tool_call>call:foo{x:1}<tool_call|>"
        assert strip_tool_markup(text, final = True) == ""

    # ── Gemma 4 wrapper-less (skip_special_tokens stripped) ───────────

    def test_gemma4_bare_stripped_call(self):
        # skip_special_tokens removes <|tool_call>/<tool_call|> and <|"|>,
        # leaving a bare call:NAME{...} with an unquoted value.
        import json

        text = "call:web_search{query:weather in San Francisco right now}"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "weather in San Francisco right now"}

    def test_gemma4_bare_code_with_commas(self):
        # A code value with commas must not truncate at the first comma.
        import json

        text = (
            "call:web_search{code:def f(n):\n    a, b = 0, 1\n"
            "    for _ in range(2, n+1):\n        a, b = b, a + b\n"
            "    return b\n\nprint(f(30))}"
        )
        result = parse_tool_calls_from_text(text)
        assert result[0]["function"]["name"] == "web_search"
        code = json.loads(result[0]["function"]["arguments"])["code"]
        assert "a, b = 0, 1" in code and "print(f(30))" in code

    def test_gemma4_bare_quotes_normalized(self):
        # The same value quoted vs unquoted must parse identically so the
        # agentic loop can collapse a looping model's repeated calls.
        import json

        a = parse_tool_calls_from_text('call:web_search{query:"foo bar"}')
        b = parse_tool_calls_from_text("call:web_search{query:foo bar}")
        assert json.loads(a[0]["function"]["arguments"]) == {"query": "foo bar"}
        assert json.loads(a[0]["function"]["arguments"]) == json.loads(
            b[0]["function"]["arguments"]
        )

    def test_gemma4_bare_multi_arg(self):
        import json

        text = "call:web_search{query:pytorch latest, url:https://pytorch.org}"
        result = parse_tool_calls_from_text(text)
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "pytorch latest", "url": "https://pytorch.org"}

    def test_gemma4_bare_not_matched_in_prose(self):
        # A word ending in "call:" must not trigger a bare tool call.
        text = "I will recall:that the function{ } is helpful."
        result = parse_tool_calls_from_text(text)
        assert result == []

    def test_gemma4_bare_strip_markup_final(self):
        text = "Here you go: call:web_search{query:weather today}"
        assert "call:web_search" not in strip_tool_markup(text, final = True)

    # ── Cross-format sentinels ────────────────────────────────────

    def test_all_markers_in_tool_xml_signals(self):
        # Streaming buffer wakes up on every emission marker.
        from core.inference.tool_call_parser import TOOL_XML_SIGNALS
        for marker in (
            "<tool_call>",
            "<function=",
            "<|python_tag|>",
            "[TOOL_CALLS]",
            "<|tool_call>",
        ):
            assert marker in TOOL_XML_SIGNALS, f"streaming loop would not wake on {marker!r}"

    def test_has_tool_signal_for_all_formats(self):
        assert has_tool_signal('<|python_tag|>brave_search.call(q="x")')
        assert has_tool_signal('[TOOL_CALLS] [{"name":"x"}]')
        assert has_tool_signal('[TOOL_CALLS]add{"a":1}')
        assert has_tool_signal("<|tool_call>call:foo{}<tool_call|>")


# ────────────────────────────────────────────────────────────────────
# run_safetensors_tool_loop
# ────────────────────────────────────────────────────────────────────


def _fake_stream(chunks):
    """Build a single-turn generator that yields cumulative snapshots."""

    def _gen(_messages):
        acc = ""
        for c in chunks:
            acc += c
            yield acc

    return _gen


def _const_stream(text):
    """A single-turn generator that yields one cumulative snapshot."""

    def _gen(_messages):
        yield text

    return _gen


class FakeExecuteTool:
    """Stand-in for ``core.inference.tools.execute_tool``."""

    def __init__(self, results):
        # ``results`` is a list of strings or RuntimeError instances.
        self.results = list(results)
        self.calls: list[tuple[str, dict]] = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        thread_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        self.calls.append((name, arguments))
        result = self.results.pop(0) if self.results else "OK"
        if isinstance(result, Exception):
            raise result
        return result


def _collect_events(generator, max_events = 200):
    events = []
    for ev in generator:
        events.append(ev)
        if len(events) >= max_events:
            break
    return events


def _make_loop(
    *,
    turns,
    exec_results = None,
    **kwargs,
):
    """Build a configured loop with a multi-turn fake generator.

    ``turns`` is a list of chunk-lists; iteration N yields chunks from ``turns[N]``.
    """
    turn_iter = iter(turns)

    def _gen(_messages):
        try:
            chunks = next(turn_iter)
        except StopIteration:
            return
        acc = ""
        for c in chunks:
            acc += c
            yield acc

    exec_fn = FakeExecuteTool(exec_results or [])
    return run_safetensors_tool_loop(
        single_turn = _gen,
        messages = [{"role": "user", "content": "hi"}],
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "python"}},
            {"type": "function", "function": {"name": "terminal"}},
        ],
        execute_tool = exec_fn,
        **kwargs,
    ), exec_fn


class TestMarkerlessExecToolGuardLoop:
    """End-to-end guard for the two prompt-injection -> RCE findings: a bare (unwrapped)
    ``python``/``terminal`` call quoted in assistant prose must never reach ``execute_tool``,
    even with those tools enabled, while the trusted wrapped/marker forms still execute."""

    def test_bare_execution_call_in_prose_is_not_executed(self):
        # ``_make_loop`` enables web_search + python + terminal.
        prose = (
            'You could run call:terminal{command:"id"} or terminal[ARGS]{"command":"id"}, '
            'and even call:python{code:"import os; os.system(1)"}, but I will not.'
        )
        loop, exec_fn = _make_loop(turns = [[prose]])
        events = _collect_events(loop)
        # execute_tool was never invoked and no tool lifecycle event was emitted.
        assert exec_fn.calls == []
        assert not any(e.get("type") in ("tool_start", "tool_end") for e in events)
        # The bare call text stays visible to the user (parse/strip symmetry).
        contents = [e.get("text", "") for e in events if e.get("type") == "content"]
        final = contents[-1] if contents else ""
        assert "call:terminal{command:" in final
        assert 'terminal[ARGS]{"command":"id"}' in final
        assert "call:python{code:" in final

    def test_wrapped_gemma_execution_call_in_loop_still_executes(self):
        # A properly wrapped Gemma call is trusted -- the fix only blocks the markerless form.
        turns = [
            ['<|tool_call>call:terminal{command:<|"|>id<|"|>}<tool_call|>'],
            ["All done."],
        ]
        loop, exec_fn = _make_loop(turns = turns, exec_results = ["uid=0(root)"], max_tool_iterations = 3)
        events = _collect_events(loop)
        assert [name for name, _args in exec_fn.calls] == ["terminal"]
        assert any(e.get("type") == "tool_start" for e in events)

    def test_marker_rehearsal_execution_call_in_loop_still_executes(self):
        # The [TOOL_CALLS] marker makes the rehearsal trusted, so terminal still runs.
        turns = [['[TOOL_CALLS]terminal[ARGS]{"command":"id"}'], ["done"]]
        loop, exec_fn = _make_loop(turns = turns, exec_results = ["uid=0"], max_tool_iterations = 3)
        _collect_events(loop)
        assert [name for name, _args in exec_fn.calls] == ["terminal"]


class TestParserDeepSeek:
    """DeepSeek R1 / V3 / V3.1 coverage. Markers use full-width pipes
    (U+FF5C) and lower-one-eighth-block (U+2581). R1 wraps args in a
    Markdown ``` ```json ``` ``` fence; V3 / V3.1 emit bare JSON."""

    def test_r1_simple_call_with_code_fence(self):
        import json as _json

        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>function"
            "<｜tool▁sep｜>special_function\n"
            "```json\n"
            '{"arg1": 1}\n'
            "```"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "special_function"
        assert _json.loads(result[0]["function"]["arguments"]) == {"arg1": 1}

    def test_r1_short_form_outer_marker(self):
        # llama.cpp accepts ``<｜tool▁calls｜>`` as the short-form opener.
        import json as _json

        text = (
            "<｜tool▁calls｜>function"
            "<｜tool▁sep｜>get_time\n"
            "```json\n"
            '{"city": "Paris"}\n'
            "```"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_time"

    def test_v3_1_bare_json(self):
        # V3 / V3.1 omit the ``function`` prefix and the code fence.
        import json as _json

        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>get_time"
            "<｜tool▁sep｜>"
            '{"city": "Tokyo"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_time"
        assert _json.loads(result[0]["function"]["arguments"]) == {"city": "Tokyo"}

    def test_v3_1_multi_call_shares_envelope(self):
        # Parallel calls share one outer envelope; each inner call has
        # its own ``<｜tool▁call▁begin｜>...<｜tool▁call▁end｜>``.
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>get_time"
            "<｜tool▁sep｜>"
            '{"city": "Paris"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁call▁begin｜>get_weather"
            "<｜tool▁sep｜>"
            '{"city": "Paris"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_time"
        assert result[1]["function"]["name"] == "get_weather"

    def test_v3_1_with_reasoning(self):
        # Reasoning <think>...</think> precedes the tool block.
        text = (
            "<think>I'm thinking</think>\n"
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>get_time"
            "<｜tool▁sep｜>"
            '{"city": "Tokyo"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_time"

    def test_v3_1_strict_rejects_unclosed_envelope(self):
        # Envelope truncated mid-stream (no <｜tool▁calls▁end｜>): healed by
        # default, rejected with Auto-Heal off.
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>get_time"
            "<｜tool▁sep｜>"
            '{"city": "Tokyo"}'
        )
        assert len(parse_tool_calls_from_text(text)) == 1
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []

    def test_v3_1_multi_call_recovers_when_first_end_marker_missing(self):
        # First inner call omits its <｜tool▁call▁end｜>; the second must still be parsed.
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>get_time"
            "<｜tool▁sep｜>"
            '{"city": "Paris"}'
            "<｜tool▁call▁begin｜>get_weather"
            "<｜tool▁sep｜>"
            '{"city": "Paris"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        result = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in result] == ["get_time", "get_weather"]

    def test_v3_1_strict_recovers_after_missing_call_end(self):
        # Strict mode (Auto-Heal off): the FIRST inner call is missing its <｜tool▁call▁end｜>
        # terminator, so it is skipped -- but the parser must keep scanning and still return the ...
        text = (
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>get_weather"
            "<｜tool▁sep｜>"
            '{"city": "SF"}'
            "<｜tool▁call▁begin｜>get_time"
            "<｜tool▁sep｜>"
            '{"tz": "PST"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        # Auto-Heal keeps both; strict skips the truncated first, keeps the second.
        assert [c["function"]["name"] for c in parse_tool_calls_from_text(text)] == [
            "get_weather",
            "get_time",
        ]
        strict = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert [c["function"]["name"] for c in strict] == ["get_time"]

    def test_r1_strict_recovers_after_missing_close_fence(self):
        # R1 form.
        text = (
            "<｜tool▁calls▁begin｜>"
            "function<｜tool▁sep｜>get_weather\n```json\n"
            '{"city": "SF"}'
            "function<｜tool▁sep｜>get_time\n```json\n"
            '{"tz": "PST"}'
            "\n```<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
        )
        strict = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert [c["function"]["name"] for c in strict] == ["get_time"]

    def test_deepseek_strip_markup(self):
        text = (
            "before "
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>foo"
            "<｜tool▁sep｜>"
            "{}"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
            " after"
        )
        assert strip_tool_markup(text, final = True) == "before  after"

    def test_deepseek_signal_wakes_streaming(self):
        # The streaming buffer state machine must wake on the DeepSeek opener so the rest of the
        # section is drained instead of leaked.
        text = "<｜tool▁calls▁begin｜>..."
        assert has_tool_signal(text)

    def test_deepseek_short_opener_is_stripped(self):
        # The short ``<｜tool▁calls｜>`` opener is parsed, so its markup must also be stripped (the
        # strip patterns used to require ...calls_begin and left the short-opener markup leaking to ...
        text = (
            "before "
            "<｜tool▁calls｜>"
            "<｜tool▁call▁begin｜>foo"
            "<｜tool▁sep｜>"
            "{}"
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
            " after"
        )
        assert strip_tool_markup(text, final = True) == "before  after"


class TestParserGLM:
    """GLM 4.5 / 4.6 / 4.7 coverage. Marker collides with Qwen's
    ``<tool_call>`` but the body shape is XML kv pairs instead of JSON,
    so the dispatch order keeps both formats working."""

    def test_glm_simple_call(self):
        import json as _json

        text = (
            "<tool_call>web_search\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>weather Tokyo</arg_value>\n"
            "</tool_call>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = _json.loads(result[0]["function"]["arguments"])
        # Strings come through raw; the parser does not double-quote.
        assert args == {"query": "weather Tokyo"}

    def test_glm_mixed_types_decode_correctly(self):
        # Per the chat_template.jinja, strings are emitted raw and non-strings are JSON-encoded.
        import json as _json

        text = (
            "<tool_call>complex_function\n"
            "<arg_key>name</arg_key>\n<arg_value>John Doe</arg_value>\n"
            "<arg_key>age</arg_key>\n<arg_value>30</arg_value>\n"
            "<arg_key>active</arg_key>\n<arg_value>true</arg_value>\n"
            "<arg_key>score</arg_key>\n<arg_value>95.5</arg_value>\n"
            "</tool_call>"
        )
        result = parse_tool_calls_from_text(text)
        args = _json.loads(result[0]["function"]["arguments"])
        assert args == {"name": "John Doe", "age": 30, "active": True, "score": 95.5}

    def test_glm_multi_call_back_to_back(self):
        # GLM emits parallel calls as consecutive ``<tool_call>...
        # </tool_call>`` blocks with no outer envelope.
        text = (
            "<tool_call>a\n<arg_key>x</arg_key>\n<arg_value>1</arg_value>\n</tool_call>"
            "<tool_call>b\n<arg_key>y</arg_key>\n<arg_value>2</arg_value>\n</tool_call>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_glm_unclosed_tool_call_does_not_lose_value(self):
        # Truncated mid-stream (no </tool_call>) -- the parser must
        # still surface what it found rather than dropping the call.
        text = "<tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>partial</arg_value>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_glm_does_not_break_qwen_path(self):
        # Real Qwen emission must still be parsed by the Qwen branch,
        # not silently misrouted to GLM (the marker is shared).
        text = '<tool_call>{"name":"web_search","arguments":{"q":"x"}}</tool_call>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_glm_strip_markup(self):
        text = (
            "before "
            "<tool_call>a\n<arg_key>x</arg_key>\n<arg_value>1</arg_value>\n</tool_call>"
            " after"
        )
        assert strip_tool_markup(text, final = True) == "before  after"

    def test_glm_zero_arg_inline_call(self):
        # GLM 4.7 emits a no-argument call inline as ``<tool_call>name</tool_call>`` (name followed
        # straight by the close tag, no \n / <arg_key>).
        import json as _json

        text = "<tool_call>get_current_date</tool_call>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_current_date"
        assert _json.loads(result[0]["function"]["arguments"]) == {}

    def test_glm_zero_arg_call_in_parallel_batch(self):
        # A no-arg call alongside a normal one must not make either vanish.
        text = (
            "<tool_call>get_current_date</tool_call>"
            "<tool_call>get_weather\n<arg_key>city</arg_key>\n"
            "<arg_value>Tokyo</arg_value></tool_call>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_current_date"
        assert result[1]["function"]["name"] == "get_weather"

    def test_glm_string_value_whitespace_preserved(self):
        # The template emits string args verbatim, so significant leading / trailing whitespace
        # (code, diffs) must survive.
        import json as _json

        text = (
            "<tool_call>run\n<arg_key>code</arg_key>\n"
            "<arg_value>    indented code    </arg_value></tool_call>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = _json.loads(result[0]["function"]["arguments"])
        assert args == {"code": "    indented code    "}


class TestParserKimi:
    """Kimi K2 / Moonshot coverage. ASCII pipes only (NOT full-width).
    Name arrives as ``functions.NAME:IDX``; the parser strips the
    prefix and the index to recover the bare callable name while
    preserving the full id for round-trip rendering."""

    def test_kimi_simple_call(self):
        import json as _json

        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.special_function:0"
            "<|tool_call_argument_begin|>"
            '{"arg1": 1}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        # Bare name recovered; full id preserved verbatim.
        assert result[0]["function"]["name"] == "special_function"
        assert result[0]["id"] == "functions.special_function:0"
        assert _json.loads(result[0]["function"]["arguments"]) == {"arg1": 1}

    def test_outer_tool_call_with_embedded_kimi_marker_parses_outer(self):
        # A Qwen/Hermes <tool_call> whose argument contains literal Kimi markup (a user asking
        # about that syntax) must execute the OUTER call, not the embedded marker via the ...
        text = (
            '<tool_call>{"name":"web_search","arguments":{"query":'
            '"explain <|tool_call_begin|>functions.evil:0'
            '<|tool_call_argument_begin|>{}<|tool_call_end|>"}}'
            "</tool_call>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_genuine_kimi_call_without_envelope_still_parses(self):
        # Control: a real Kimi call with no leading <tool_call> envelope must
        # still go through the pre-pass.
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.web_search:0"
            '<|tool_call_argument_begin|>{"query":"x"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    def test_kimi_multi_call_with_index(self):
        # Multiple consecutive calls inside a single section, each
        # with its own monotonically incrementing ``:IDX``.
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.read_file:0"
            "<|tool_call_argument_begin|>"
            '{"path":"a"}'
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.web_search:1"
            "<|tool_call_argument_begin|>"
            '{"query":"x"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "read_file"
        assert result[0]["id"].endswith(":0")
        assert result[1]["function"]["name"] == "web_search"
        assert result[1]["id"].endswith(":1")

    def test_kimi_dotted_name_keeps_full_dotted_name(self):
        # A dotted Kimi id keeps its FULL name after stripping only the ``functions.`` prefix and
        # ``:idx`` suffix -- matching current vLLM ...
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>a.b.c:2"
            "<|tool_call_argument_begin|>"
            "{}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "a.b.c"

    def test_kimi_dotted_mcp_name_with_functions_prefix(self):
        # ``functions.mcp.server-list:0`` must resolve to ``mcp.server-list``
        # (only the ``functions.`` prefix and ``:idx`` are removed).
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.mcp.server-list:0"
            "<|tool_call_argument_begin|>"
            "{}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "mcp.server-list"

    def test_kimi_multi_call_recovers_when_first_end_marker_missing(self):
        # First call omits its <|tool_call_end|>; the second must still parse.
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.read_file:0"
            "<|tool_call_argument_begin|>"
            '{"path":"a"}'
            "<|tool_call_begin|>functions.web_search:1"
            "<|tool_call_argument_begin|>"
            '{"query":"x"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in result] == ["read_file", "web_search"]

    def test_kimi_handles_unclosed_section(self):
        # End marker missing -- the parser must still extract the call.
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.foo:0"
            "<|tool_call_argument_begin|>"
            '{"a":1}'
            "<|tool_call_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "foo"

    def test_kimi_strip_markup(self):
        text = (
            "before "
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.x:0"
            "<|tool_call_argument_begin|>"
            "{}"
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            " after"
        )
        assert strip_tool_markup(text, final = True) == "before  after"

    def test_kimi_signal_wakes_streaming(self):
        text = "<|tool_calls_section_begin|>..."
        assert has_tool_signal(text)

    def test_kimi_call_without_section_wrapper(self):
        # llama.cpp makes the ``<|tool_calls_section_begin|>`` wrapper optional -- Kimi K2 can emit
        # a bare ``<|tool_call_begin|>`` call.
        import json as _json

        text = (
            "<|tool_call_begin|>functions.execute_command:0"
            "<|tool_call_argument_begin|>"
            '{"cmd":"ls"}'
            "<|tool_call_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "execute_command"
        assert _json.loads(result[0]["function"]["arguments"]) == {"cmd": "ls"}

    def test_kimi_malformed_json_recovers_later_calls(self):
        # A call with malformed / truncated JSON must not drop the valid calls that follow it in
        # the same section (the bad call is skipped, the good one is recovered).
        import json as _json

        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.a:0"
            '<|tool_call_argument_begin|>{"city":"Beijing"'  # missing closing brace
            "<|tool_call_end|>"
            "<|tool_call_begin|>functions.b:1"
            '<|tool_call_argument_begin|>{"city":"Shanghai"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "b"
        assert _json.loads(result[0]["function"]["arguments"]) == {"city": "Shanghai"}


class TestParserCrossFormatRouting:
    """Ensure the per-format dispatch order doesn't misroute any
    family. Real emissions for each new family + every old family
    must still parse correctly when intermixed."""

    def test_dispatch_routes_each_family_correctly(self):
        cases = [
            (
                "Qwen",
                '<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>',
                "a",
            ),
            (
                "DeepSeek V3.1",
                "<｜tool▁calls▁begin｜>"
                "<｜tool▁call▁begin｜>get_time"
                "<｜tool▁sep｜>"
                '{"city":"Tokyo"}'
                "<｜tool▁call▁end｜>"
                "<｜tool▁calls▁end｜>",
                "get_time",
            ),
            (
                "GLM",
                "<tool_call>web_search\n"
                "<arg_key>q</arg_key>\n<arg_value>x</arg_value>\n"
                "</tool_call>",
                "web_search",
            ),
            (
                "Kimi",
                "<|tool_calls_section_begin|>"
                "<|tool_call_begin|>functions.add:0"
                "<|tool_call_argument_begin|>"
                '{"a":1}'
                "<|tool_call_end|>"
                "<|tool_calls_section_end|>",
                "add",
            ),
        ]
        for label, text, expected_name in cases:
            result = parse_tool_calls_from_text(text)
            assert len(result) == 1, f"{label}: parser missed the call"
            assert result[0]["function"]["name"] == expected_name, (
                f"{label}: got {result[0]['function']['name']!r}, " f"expected {expected_name!r}"
            )

    def test_all_new_markers_in_tool_xml_signals(self):
        # The safetensors / MLX streaming buffer must wake on every supported emission marker --
        # otherwise the BUFFERING state leaks tool content to the user before parse.
        from core.inference.tool_call_parser import TOOL_XML_SIGNALS
        for marker in (
            "<｜tool▁calls▁begin｜>",
            "<｜tool▁call▁begin｜>",
            "<|tool_calls_section_begin|>",
            "<|tool_call_begin|>",
        ):
            assert marker in TOOL_XML_SIGNALS, f"streaming loop would not wake on {marker!r}"


def test_active_tools_are_passed_to_single_turn_after_render_html_success():
    captured_tool_names: list[list[str]] = []
    exec_fn = FakeExecuteTool(["Rendered HTML canvas."])

    def fake_single_turn(_messages, *, active_tools = None):
        captured_tool_names.append(
            [
                (tool.get("function") or {}).get("name")
                for tool in (active_tools or [])
                if (tool.get("function") or {}).get("name")
            ]
        )
        if len(captured_tool_names) == 1:
            yield '<tool_call>{"name":"render_html","arguments":{"code":"<html>one</html>"}}</tool_call>'
        else:
            yield "Done."

    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = fake_single_turn,
            messages = [{"role": "user", "content": "make html"}],
            tools = [
                {"type": "function", "function": {"name": "render_html"}},
                {"type": "function", "function": {"name": "web_search"}},
            ],
            execute_tool = exec_fn,
            max_tool_iterations = 3,
        )
    )

    assert exec_fn.calls == [("render_html", {"code": "<html>one</html>"})]
    assert captured_tool_names == [["render_html", "web_search"], ["web_search"]]
    assert any(event.get("type") == "content" and event.get("text") == "Done." for event in events)


def test_spent_one_shot_rehearsal_repeat_is_detected_not_blank_continuation():
    # A spent one-shot (render_html) stays in the ORIGINAL tool list; detection is gated on
    # that list (matching the strip gate) so a re-emitted repeat is drained and routed to the
    # repeat no-op instead of stripped into a blank continuation.
    exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
    turns = iter(
        [
            [
                '<tool_call>{"name":"render_html","arguments":{"code":"<html>one</html>"}}</tool_call>'
            ],
            ['render_html[ARGS]{"code":"<html>two</html>"}'],  # spent one-shot rehearsal
            ["The chart is above."],
        ]
    )

    def gen(_messages, *, active_tools = None):
        try:
            chunks = next(turns)
        except StopIteration:
            return
        acc = ""
        for c in chunks:
            acc += c
            yield acc

    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = gen,
            messages = [{"role": "user", "content": "make a chart"}],
            tools = [
                {"type": "function", "function": {"name": "render_html"}},
                {"type": "function", "function": {"name": "web_search"}},
            ],
            execute_tool = exec_fn,
            max_tool_iterations = 5,
        )
    )
    contents = [e["text"] for e in events if e["type"] == "content"]
    # render_html ran exactly once; the repeat was a no-op, not a second execution.
    assert exec_fn.calls == [("render_html", {"code": "<html>one</html>"})], exec_fn.calls
    # The loop continued past the repeat to the real answer (not a blank continuation).
    assert any("The chart is above." in t for t in contents), contents
    # The raw rehearsal markup never leaked as visible content.
    assert not any("render_html[ARGS]" in t for t in contents), contents


def test_rehearsal_call_name_is_not_streamed_before_args():
    # A rehearsal whose name and [ARGS] arrive together must drain, not stream the bare name.
    loop, exec_fn = _make_loop(
        turns = [['web_search[ARGS]{"query":"cats"}'], ["Found."]],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("web_search" in t for t in contents), contents


def test_rehearsal_call_name_split_before_args_is_not_streamed():
    # Finding 5: name and [ARGS] in separate chunks -- the bare name is held until [ARGS] arrives.
    loop, exec_fn = _make_loop(
        turns = [["web_search", '[ARGS]{"query":"cats"}'], ["Found."]],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("web_search" in t for t in contents), contents


def test_plain_word_matching_no_tool_still_streams():
    # The prefix guard must not swallow prose: a non-tool bare word streams.
    loop, _exec = _make_loop(
        turns = [["weather", " is nice today."]],
        max_tool_iterations = 1,
    )
    events = _collect_events(loop)
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert "weather is nice today." in contents, contents


def test_rehearsal_name_after_prose_in_streaming_is_not_streamed():
    # After prose has streamed (STREAMING state), a split rehearsal name must still be held.
    loop, exec_fn = _make_loop(
        turns = [
            # _make_loop accumulates these deltas into cumulative snapshots.
            ["Let me think. ", "I will search ", "web_search", '[ARGS]{"query":"cats"}'],
            ["Found."],
        ],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("web_search" in t for t in contents), contents


def test_rehearsal_name_after_prose_same_chunk_in_streaming_is_not_streamed():
    # Prose then ``web_search[ARGS]{...}`` in one chunk: the boundary is pulled back over the name.
    loop, exec_fn = _make_loop(
        turns = [
            ["Sure. ", 'now web_search[ARGS]{"query":"cats"}'],
            ["Found."],
        ],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("web_search" in t for t in contents), contents


def test_initial_buffer_flush_holds_split_rehearsal_name():
    # First flush out of BUFFERING applies the same trailing-name hold as STREAMING.
    loop, exec_fn = _make_loop(
        turns = [["I will use web_search", '[ARGS]{"code":"print(1)"}'], ["done"]],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"code": "print(1)"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("web_search" in t for t in contents), contents


def test_think_rehearsal_streams_monotonically_and_keeps_reasoning():
    # A think rehearsal streams the same text the final strip keeps: cumulative content is
    # monotonically non-decreasing and ends with the markup intact.
    loop, exec_fn = _make_loop(
        turns = [["<think>plan ", 'search[ARGS]{"q":"x"}', "</think> visible"]],
        max_tool_iterations = 1,
    )
    events = _collect_events(loop)
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert exec_fn.calls == [], exec_fn.calls
    assert all(len(b) >= len(a) for a, b in zip(contents, contents[1:])), contents
    final = contents[-1] if contents else ""
    assert 'search[ARGS]{"q":"x"}' in final, contents
    assert "visible" in final, contents


def test_plain_answer_ending_with_tool_name_word_is_preserved():
    # End-of-stream flush: a plain answer ending on a tool-name word is prose, not dropped.
    loop, exec_fn = _make_loop(
        turns = [["I think ", "you should ", "web_search"]],
        max_tool_iterations = 1,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert any(t.rstrip().endswith("web_search") for t in contents), contents


def test_long_tool_name_split_rehearsal_is_not_capped_and_executes():
    # Finding 10/11: an MCP name longer than the buffer cap, split before [ARGS], is still
    # held (self-bounding prefix); no leak and the call executes.
    from core.inference.safetensors_agentic import _MAX_BUFFER_CHARS

    name = "mcp__github__create_pull_request"
    assert len(name) >= _MAX_BUFFER_CHARS, len(name)
    exec_fn = FakeExecuteTool(["RESULT"])
    _turns = iter([[name, name + '[ARGS]{"x":1}'], ["done"]])

    def st(_messages, active_tools = None):
        yield from next(_turns)

    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = st,
            messages = [{"role": "user", "content": "go"}],
            tools = [{"type": "function", "function": {"name": name}}],
            execute_tool = exec_fn,
            max_tool_iterations = 2,
        )
    )
    assert exec_fn.calls == [(name, {"x": 1})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any(name in t for t in contents), contents


def test_unrestricted_mode_split_rehearsal_name_is_not_streamed():
    # Finding 6: unrestricted mode treats any bare identifier as a possible rehearsal NAME.
    exec_fn = FakeExecuteTool(["RESULT"])
    _turns = iter([["web_search", 'web_search[ARGS]{"q":"x"}'], ["done"]])

    def st(_messages, active_tools = None):
        yield from next(_turns)

    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = st,
            messages = [{"role": "user", "content": "go"}],
            tools = [],  # unrestricted
            execute_tool = exec_fn,
            max_tool_iterations = 2,
        )
    )
    assert exec_fn.calls == [("web_search", {"q": "x"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("web_search" in t for t in contents), contents


def test_unrestricted_mode_split_after_bracket_is_not_streamed():
    # Unrestricted mode: a chunk split right after ``NAME[`` is still held (parity with the
    # restricted-mode startswith hold).
    exec_fn = FakeExecuteTool(["RESULT"])
    _turns = iter([["web_search[", 'web_search[ARGS]{"q":"x"}'], ["done"]])

    def st(_messages, active_tools = None):
        yield from next(_turns)

    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = st,
            messages = [{"role": "user", "content": "go"}],
            tools = [],  # unrestricted
            execute_tool = exec_fn,
            max_tool_iterations = 2,
        )
    )
    assert exec_fn.calls == [("web_search", {"q": "x"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("web_search[" in t for t in contents), contents


def test_unrestricted_mode_plain_prose_still_streams():
    # The unrestricted hold releases a held identifier once the rest of the sentence follows.
    def st(_messages, active_tools = None):
        for snap in ("Hello", "Hello there friend."):
            yield snap

    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = st,
            messages = [{"role": "user", "content": "hi"}],
            tools = [],
            execute_tool = FakeExecuteTool([]),
            max_tool_iterations = 1,
        )
    )
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert "Hello there friend." in contents, contents


def test_safety_net_honors_disabled_auto_heal_for_late_incomplete_call():
    # A late call caught by the safety net: an unclosed ``<tool_call>`` heals only with Auto-Heal on;
    # off, the safety net must not pass ``allow_incomplete=True`` and execute a truncated call.
    prose = "Sure, let me look that up for you right now. "
    incomplete = '<tool_call>{"name":"web_search","arguments":{"query":"weather in Sydney"}}'

    loop_off, exec_off = _make_loop(
        turns = [[prose, incomplete], ["Final answer."]],
        exec_results = ["RESULT"],
        auto_heal_tool_calls = False,
        max_tool_iterations = 3,
    )
    events_off = _collect_events(loop_off)
    assert exec_off.calls == [], "disabled Auto-Heal must not execute a healed incomplete call"
    assert not [e for e in events_off if e.get("type") == "tool_start"]

    loop_on, exec_on = _make_loop(
        turns = [[prose, incomplete], ["Final answer."]],
        exec_results = ["RESULT"],
        auto_heal_tool_calls = True,
        max_tool_iterations = 3,
    )
    _collect_events(loop_on)
    assert exec_on.calls == [("web_search", {"query": "weather in Sydney"})], exec_on.calls


def test_bare_json_tool_call_is_not_streamed_as_content():
    # Llama-3.2 ``custom_tools`` bare form ``{"name":..,"parameters":..}`` carries no
    # XML signal. The loop must BUFFER it until the object closes and execute it via
    # the safety net, never leaking the raw JSON to streaming clients as content.
    bare = '{"name":"web_search","parameters":{"query":"cats"}}'
    loop, exec_fn = _make_loop(
        turns = [[bare], ["Here are the results."]],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any('"name"' in t or "web_search" in t for t in contents), contents
    assert any("Here are the results." in t for t in contents)


def test_ordinary_json_with_name_key_is_shown_not_treated_as_tool_call():
    # Markerless JSON whose "name" is not an enabled tool (e.g. a person record
    # ``{"name":"Alice",...}``) must be shown as the answer, not misread as a call
    # to a disabled tool and dropped. _make_loop enables web_search/python/terminal.
    answer = '{"name":"Alice","parameters":{"age":30}}'
    loop, exec_fn = _make_loop(turns = [[answer]], max_tool_iterations = 1)
    events = _collect_events(loop)
    assert exec_fn.calls == [], exec_fn.calls
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert "Alice" in contents, contents


def test_bare_json_tool_call_split_across_chunks_is_not_streamed():
    # Same as above but the bare object arrives split mid-key, so the buffer is
    # held open across chunks before it balances.
    loop, exec_fn = _make_loop(
        turns = [
            ['{"name":"web_', 'search","parameters":{"query":"cats"}}'],
            ["Done."],
        ],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any('"name"' in t or "web_search" in t for t in contents), contents


def test_gemma_wrapperless_call_is_not_streamed_as_content():
    # Gemma 4 wrapper-less ``call:NAME{...}`` has no XML signal; the loop must hold
    # it (BUFFERING) and execute it, never streaming the raw call text.
    loop, exec_fn = _make_loop(
        turns = [["call:web_search{query:cats}"], ["Found."]],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("call:web_search" in t for t in contents), contents


def test_gemma_wrapperless_call_with_whitespace_is_suppressed_when_streamed():
    # Gemma may emit ``call : NAME{...}`` with whitespace around the colon, split across stream
    # chunks.
    loop, exec_fn = _make_loop(
        turns = [["call", " : ", "web_search", "{query:cats}"], ["Found."]],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("call" in t for t in contents), contents


def test_long_gemma_tool_name_is_not_streamed_as_content():
    # A tool name longer than the small buffer cap (OpenAI 64 chars, MCP longer)
    # must still be held: the ``call:NAME`` prefix keeps buffering until ``{``
    # instead of leaking ``call:longname`` as visible text.
    long_name = "mcp__github__list_repository_issues"  # 35 chars
    turns = iter([list('call:%s{repo:"octo/hello"}' % long_name), ["Done."]])

    def _gen(_messages):
        try:
            chunks = next(turns)
        except StopIteration:
            return
        acc = ""
        for c in chunks:
            acc += c
            yield acc

    exec_fn = FakeExecuteTool(["RESULT"])
    loop = run_safetensors_tool_loop(
        single_turn = _gen,
        messages = [{"role": "user", "content": "hi"}],
        tools = [{"type": "function", "function": {"name": long_name}}],
        execute_tool = exec_fn,
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [(long_name, {"repo": "octo/hello"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any("call:" in t for t in contents), contents


def test_leading_json_answer_is_not_dropped():
    # A leading ``{...}`` that is NOT a tool call must still surface as content:
    # the bare-JSON hold can only ever delay it to end-of-object, never drop it.
    obj = '{"answer": 42, "note": "done"}'
    loop, exec_fn = _make_loop(
        turns = [[obj]],
        exec_results = [],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == []
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert any('"answer"' in t for t in contents), contents


def _reprompt_loop(*, auto_heal_tool_calls):
    """Drive one restricted tool with an intent-only first turn to exercise the nudge; returns conversations and events."""
    captured: list[list] = []

    def fake_single_turn(messages, active_tools = None):
        captured.append(list(messages))
        if len(captured) == 1:
            yield "I'll search for that now."  # forward-looking intent, no call
        else:
            yield "Final answer."

    exec_fn = FakeExecuteTool([])
    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = fake_single_turn,
            messages = [{"role": "user", "content": "find X"}],
            tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}],
            execute_tool = exec_fn,
            auto_heal_tool_calls = auto_heal_tool_calls,
            # Unsloth always nudges (always-on for the Unsloth inference paths); the
            # API opts in per request. Model the Unsloth caller here.
            nudge_tool_calls = True,
            max_tool_iterations = 3,
        )
    )
    return captured, events


def test_reprompt_names_only_active_tools_not_hardcoded():
    # The plan-without-action nudge must name the tools actually enabled, never the
    # old hardcoded ``web_search``/``python`` (which a restricted set would reject).
    captured, _events = _reprompt_loop(auto_heal_tool_calls = True)
    assert len(captured) >= 2, "intent prose should have triggered a re-prompt turn"
    reprompt = captured[1][-1]
    assert reprompt["role"] == "user"
    assert "search_knowledge_base" in reprompt["content"]
    assert "web_search" not in reprompt["content"]
    assert "python" not in reprompt["content"]


def test_reprompt_suppressed_when_auto_heal_disabled():
    # With Auto-Heal off the safetensors nudge must stay silent for backend parity
    # with the GGUF loop, so only the single initial generation runs.
    captured, events = _reprompt_loop(auto_heal_tool_calls = False)
    assert len(captured) == 1, captured
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert any("search for that" in t for t in contents)


class TestLoopBasic:
    def test_plain_answer(self):
        # No tool XML; loop should yield content then status="".
        loop, _exec = _make_loop(
            turns = [["Hello", " world", "!"]],
            exec_results = [],
        )
        events = _collect_events(loop)
        contents = [e for e in events if e["type"] == "content"]
        statuses = [e for e in events if e["type"] == "status"]
        assert contents, "expected at least one content event"
        # Final cumulative content must contain the answer.
        final_text = contents[-1]["text"]
        assert "Hello world!" in final_text
        assert statuses and statuses[-1]["text"] == ""

    def test_single_tool_then_answer(self):
        loop, exec_fn = _make_loop(
            turns = [
                # Tool call only.
                [
                    '<tool_call>{"name":"web_search",',
                    '"arguments":{"query":"weather"}}',
                    "</tool_call>",
                ],
                # Final answer.
                ["The ", "weather is ", "sunny."],
            ],
            exec_results = ["Sunny and 22C"],
        )
        events = _collect_events(loop)
        kinds = [e["type"] for e in events]

        assert "tool_start" in kinds
        assert "tool_end" in kinds
        # Tool was called with the parsed arguments.
        assert exec_fn.calls == [("web_search", {"query": "weather"})]

        tool_start = next(e for e in events if e["type"] == "tool_start")
        assert tool_start["tool_name"] == "web_search"
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert tool_end["result"] == "Sunny and 22C"

        contents = [e for e in events if e["type"] == "content"]
        assert contents and "sunny" in contents[-1]["text"].lower()

    def test_function_xml_form(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["<function=python><parameter=code>print(1)</parameter></function>"],
                ["Result: 1"],
            ],
            exec_results = ["1\n"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("python", {"code": "print(1)"})]
        contents = [e for e in events if e["type"] == "content"]
        assert "Result: 1" in contents[-1]["text"]

    def test_llama3_python_tag_form(self):
        # The agentic loop must recognise Llama-3's <|python_tag|>
        # marker, drain the rest of the turn, and execute the call.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<|python_tag|>web_search.call(",
                    'query="weather in Tokyo"',
                    ")",
                ],
                ["The weather is sunny."],
            ],
            exec_results = ["Sunny, 22C"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "weather in Tokyo"})]
        contents = [e for e in events if e["type"] == "content"]
        assert "sunny" in contents[-1]["text"].lower()

    def test_llama3_bare_json_form_fires_tool(self):
        # Llama-3.1 / 3.2 emit a bare-JSON tool call
        # ``{"name":..,"parameters":..}`` with NO XML signal. The loop's
        # safety-net parse must still fire the tool instead of treating the
        # turn as "planned without calling tools" and re-prompting the model
        # into giving up. Regression for the has_tool_signal gate that
        # dropped these; GGUF's llama-server parses them natively.
        loop, exec_fn = _make_loop(
            turns = [
                ['{"name": "web_search", "parameters": {"query": "weather in SF"}}'],
                ["The weather is sunny."],
            ],
            exec_results = ["Sunny, 18C"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "weather in SF"})]
        contents = [e for e in events if e["type"] == "content"]
        assert "sunny" in contents[-1]["text"].lower()

    def test_mistral_pre_v11_form(self):
        # Pre-v11 Mistral emission: ``[TOOL_CALLS] [{...}]``.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '[TOOL_CALLS] [{"name":"web_search",',
                    '"arguments":{"query":"hi"},"id":"abc"}]',
                ],
                ["done"],
            ],
            exec_results = ["ok"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "hi"})]
        # Mistral-provided ids must propagate to tool_start events.
        tool_start = next(e for e in events if e["type"] == "tool_start")
        assert tool_start["tool_call_id"] == "abc"

    def test_mistral_v11_form(self):
        # v11+ Mistral emission: bare ``name{json}`` after the trigger.
        loop, exec_fn = _make_loop(
            turns = [
                ['[TOOL_CALLS]web_search{"query":"hi"}'],
                ["done"],
            ],
            exec_results = ["ok"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "hi"})]

    def test_gemma4_form(self):
        # Gemma 4 emission: ``<|tool_call>call:NAME{...}<tool_call|>``.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<|tool_call>call:web_search{",
                    'query:<|"|>weather<|"|>',
                    "}<tool_call|>",
                ],
                ["sunny"],
            ],
            exec_results = ["Sunny, 22C"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "weather"})]

    def test_deepseek_v3_1_form(self):
        # DeepSeek V3.1 emission inside the agentic loop -- the buffer state machine must wake on
        # ``<｜tool▁calls▁begin｜>`` and the parser must extract the V3.1 bare-JSON body.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<｜tool▁calls▁begin｜>",
                    "<｜tool▁call▁begin｜>web_search",
                    "<｜tool▁sep｜>",
                    '{"query":"Tokyo weather"}',
                    "<｜tool▁call▁end｜>",
                    "<｜tool▁calls▁end｜>",
                ],
                ["The weather is sunny."],
            ],
            exec_results = ["Sunny, 22C"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "Tokyo weather"})]
        contents = [e for e in events if e["type"] == "content"]
        assert contents and "sunny" in contents[-1]["text"].lower()

    def test_glm_form(self):
        # GLM 4.x emission: ``<tool_call>NAME\n<arg_key>...``.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<tool_call>web_search\n",
                    "<arg_key>query</arg_key>\n",
                    "<arg_value>Tokyo</arg_value>\n",
                    "</tool_call>",
                ],
                ["found"],
            ],
            exec_results = ["..."],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "Tokyo"})]

    def test_kimi_form(self):
        # Kimi K2 emission ``<|tool_calls_section_begin|>...``.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<|tool_calls_section_begin|>",
                    "<|tool_call_begin|>functions.web_search:0",
                    "<|tool_call_argument_begin|>",
                    '{"query":"Tokyo"}',
                    "<|tool_call_end|>",
                    "<|tool_calls_section_end|>",
                ],
                ["done"],
            ],
            exec_results = ["..."],
        )
        events = _collect_events(loop)
        # The bare name must reach execute_tool, even though the model
        # emitted ``functions.web_search:0`` as the formatted id.
        assert exec_fn.calls == [("web_search", {"query": "Tokyo"})]
        # tool_start carries the original full id so the conversation
        # roundtrip can replay it verbatim.
        tool_start = next(e for e in events if e["type"] == "tool_start")
        assert tool_start["tool_call_id"] == "functions.web_search:0"

    def test_render_html_emits_provisional_tool_start(self):
        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    "<function=render_html>",
                    "<parameter=code><!doctype html><html>",
                    "<body>Hi</body></html></parameter></function>",
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert len(tool_starts) == 2
        assert tool_starts[0]["tool_name"] == "render_html"
        assert tool_starts[0]["arguments"] == {}
        assert tool_starts[1]["tool_name"] == "render_html"
        assert "<!doctype html>" in tool_starts[1]["arguments"]["code"]
        assert exec_fn.calls[0][0] == "render_html"
        assert "<!doctype html>" in exec_fn.calls[0][1]["code"]

    def test_render_html_confirmation_gate_suppresses_early_provisional(self, monkeypatch):
        """When a human confirmation gate is active, render_html must not surface
        an early provisional tool_start: that card (keyed by tool_call_id, no
        approval) would show the tool 'running' before the user approves. The
        gated real tool_start is the first signal the UI receives instead."""
        monkeypatch.setattr(safetensors_agentic, "new_approval_id", lambda: "approval-rh")
        monkeypatch.setattr(safetensors_agentic, "begin_tool_decision", lambda *_a, **_k: object())
        monkeypatch.setattr(safetensors_agentic, "wait_tool_decision", lambda *_a, **_k: "allow")

        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    "<function=render_html>",
                    "<parameter=code><!doctype html><html>",
                    "<body>Hi</body></html></parameter></function>",
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
            confirm_tool_calls = True,
            session_id = "sess",
            max_tool_iterations = 3,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        # No early provisional (empty-args) card while confirmation is pending.
        assert [e for e in tool_starts if e.get("arguments") == {}] == []
        # The real, gated tool_start still surfaces with the full arguments.
        real = [e for e in tool_starts if e.get("arguments", {}).get("code")]
        assert len(real) == 1
        assert real[0].get("awaiting_confirmation") is True
        assert "<!doctype html>" in real[0]["arguments"]["code"]
        assert exec_fn.calls[0][0] == "render_html"

    def test_render_html_bypass_permissions_keeps_early_provisional(self, monkeypatch):
        """bypass_permissions wins over the confirm gate, so the early provisional
        card is preserved (no human approval is required)."""
        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    "<function=render_html>",
                    "<parameter=code><!doctype html><html>",
                    "<body>Hi</body></html></parameter></function>",
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
            confirm_tool_calls = True,
            bypass_permissions = True,
            session_id = "sess",
            max_tool_iterations = 3,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert len(tool_starts) == 2
        assert tool_starts[0]["arguments"] == {}
        assert "<!doctype html>" in tool_starts[1]["arguments"]["code"]

    def test_render_html_auto_mode_static_runs_without_prompt(self):
        """permission_mode="auto" ships confirm_tool_calls=true. render_html is no
        longer unconditionally safe (a networked canvas must ask), so its early
        provisional card is suppressed under the confirm gate; a static canvas is
        still classified safe and runs without an approval prompt."""
        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    "<function=render_html>",
                    "<parameter=code><!doctype html><html>",
                    "<body>Hi</body></html></parameter></function>",
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
            confirm_tool_calls = True,
            permission_mode = "auto",
            session_id = "sess",
            max_tool_iterations = 3,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        # No early provisional card under the auto confirm gate; just the real call.
        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "render_html"
        assert "<!doctype html>" in tool_starts[0]["arguments"]["code"]
        # A static canvas is classified safe, so it runs without an approval gate.
        assert tool_starts[0].get("awaiting_confirmation") in (False, None)

    def test_render_html_provisional_card_closed_on_generator_exception(self):
        """If the model generator raises mid-stream after a provisional render_html
        card was surfaced, the loop must close that card as errored before the
        exception propagates, so the UI never leaves a tool spinning forever."""
        exec_fn = FakeExecuteTool([])

        def _gen(_messages):
            acc = ""
            for chunk in ["<function=render_html>", "<parameter=code><!doctype html><html>"]:
                acc += chunk
                yield acc
            raise RuntimeError("model pipeline exploded")

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
        )

        collected: list[dict] = []
        raised = False
        try:
            for event in loop:
                collected.append(event)
        except RuntimeError as exc:
            raised = True
            assert "exploded" in str(exc)

        assert raised
        provisional = [
            e for e in collected if e["type"] == "tool_start" and e.get("arguments") == {}
        ]
        assert len(provisional) == 1
        # The provisional card is closed (as an error) before the exception
        # propagates, so it never dangles.
        closing = [
            e
            for e in collected
            if e["type"] == "tool_end" and e.get("tool_call_id") == provisional[0]["tool_call_id"]
        ]
        assert len(closing) == 1
        assert "Error" in (closing[0].get("result") or "")

    def test_python_tool_containing_render_html_signal_does_not_emit_provisional_start(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<function=python>",
                    "<parameter=code>print('<function=render_html>')",
                    "</parameter></function>",
                ],
                ["Done."],
            ],
            exec_results = ["ok"],
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "python"
        assert exec_fn.calls == [("python", {"code": "print('<function=render_html>')"})]

    def test_render_html_rehearsed_in_think_block_emits_no_provisional_start(self):
        # BUG B: a render_html rehearsed inside think before a real python call must not emit a
        # provisional render_html card; only the outside-think call fires.
        exec_fn = FakeExecuteTool(["ok"])
        turn_iter = iter(
            [
                [
                    '<think>draft render_html[ARGS]{"code":"x"}</think>',
                    'web_search[ARGS]{"code":"print(1)"}',
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "run code"}],
            tools = [
                {"type": "function", "function": {"name": "render_html"}},
                {"type": "function", "function": {"name": "web_search"}},
            ],
            execute_tool = exec_fn,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert [e["tool_name"] for e in tool_starts] == ["web_search"], tool_starts
        assert exec_fn.calls == [("web_search", {"code": "print(1)"})]

    def test_render_html_success_blocks_second_canvas_call(self):
        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    '<tool_call>{"name":"render_html",',
                    '"arguments":{"code":"<html>one</html>"}}',
                ],
                [
                    '<tool_call>{"name":"render_html",',
                    '"arguments":{"code":"<html>two</html>"}}',
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert exec_fn.calls == [("render_html", {"code": "<html>one</html>"})]
        assert [e["arguments"] for e in tool_starts] == [{}, {"code": "<html>one</html>"}]

    def test_truncated_unclosed_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [
                # No </tool_call>; balanced-brace parser still succeeds because
                # the JSON itself is balanced.
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}'],
                ["done"],
            ],
            exec_results = ["result"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]

    def test_bad_json_healed_to_query(self):
        # Non-JSON string arguments heal to {"query": ...} under auto_heal_tool_calls.
        loop, exec_fn = _make_loop(
            turns = [
                # ``arguments`` is a string _coerce_arguments can't parse, so heal runs.
                ['<tool_call>{"name":"web_search","arguments":"hello world"}</tool_call>'],
                ["ok"],
            ],
            exec_results = ["..."],
        )
        events = _collect_events(loop)
        assert exec_fn.calls and exec_fn.calls[0][0] == "web_search"
        assert exec_fn.calls[0][1] == {"query": "hello world"}


class TestLoopBehaviour:
    def test_duplicate_tool_call_internal_noop(self):
        captured_messages: list[list[dict]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["final"],
            ]
        )

        def fake_single_turn(messages):
            captured_messages.append([dict(message) for message in messages])
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-result-1"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 3,
            )
        )

        assert exec_fn.calls == [("web_search", {"query": "x"})]
        assert [e["tool_call_id"] for e in events if e["type"] == "tool_end"] == ["call_0"]
        assert not [
            e
            for e in events
            if e.get("tool_call_id") == "call_1" and e.get("type") in {"tool_start", "tool_end"}
        ]
        duplicate_nudges = [
            message
            for message in captured_messages[-1]
            if message.get("role") == "user"
            and "already completed successfully" in message.get("content", "")
        ]
        assert len(duplicate_nudges) == 1

    def test_same_turn_duplicate_does_not_drop_later_parallel_call(self):
        # Turn 1 runs search(x). Turn 2's batch is [search(x) duplicate, python]:
        # the duplicate is a no-op, but python after it must still run, and the
        # no-op nudge must land after python's result rather than splitting it.
        captured_messages: list[list[dict]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                    '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'
                ],
                ["final"],
            ]
        )

        def fake_single_turn(messages, active_tools = None):
            captured_messages.append([dict(m) for m in messages])
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-x", "py-result"])
        _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [
                    {"type": "function", "function": {"name": "web_search"}},
                    {"type": "function", "function": {"name": "python"}},
                ],
                execute_tool = exec_fn,
                max_tool_iterations = 4,
            )
        )

        # Turn-1 search and turn-2 python both ran; the turn-2 duplicate search did not.
        assert exec_fn.calls == [
            ("web_search", {"query": "x"}),
            ("python", {"code": "print(1)"}),
        ]

        conv = captured_messages[-1]
        turn2 = [m for m in conv if m.get("role") == "assistant" and m.get("tool_calls")][-1]
        assert [tc["function"]["name"] for tc in turn2["tool_calls"]] == ["python"]
        after = conv[conv.index(turn2) + 1 :]
        assert after[0]["role"] == "tool" and after[0]["content"] == "py-result"
        assert after[1]["role"] == "user"  # deferred duplicate nudge, after the result
        assert after[1]["content"].startswith(
            "One earlier request to call tool 'web_search' in this batch was not executed"
        )
        assert "previous tool request" not in after[1]["content"].lower()

    def test_duplicate_tool_call_internal_noop_allows_distinct_followup_tool(self):
        captured_messages: list[list[dict]] = []
        captured_tool_names: list[list[str]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'],
                ["final"],
            ]
        )

        def fake_single_turn(messages, active_tools = None):
            captured_messages.append([dict(message) for message in messages])
            captured_tool_names.append(
                [
                    tool["function"]["name"]
                    for tool in (active_tools or [])
                    if tool.get("function", {}).get("name")
                ]
            )
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-result-1", "python-result"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [
                    {"type": "function", "function": {"name": "web_search"}},
                    {"type": "function", "function": {"name": "python"}},
                ],
                execute_tool = exec_fn,
                max_tool_iterations = 4,
            )
        )

        assert exec_fn.calls == [
            ("web_search", {"query": "x"}),
            ("python", {"code": "print(1)"}),
        ]
        assert [e["tool_call_id"] for e in events if e["type"] == "tool_end"] == [
            "call_0",
            "call_2",
        ]
        assert not [
            e
            for e in events
            if e.get("tool_call_id") == "call_1" and e.get("type") in {"tool_start", "tool_end"}
        ]
        duplicate_nudges = [
            message
            for message in captured_messages[2]
            if message.get("role") == "user"
            and "already completed successfully" in message.get("content", "")
        ]
        assert len(duplicate_nudges) == 1
        assert captured_tool_names[2] == ["web_search", "python"]

    def test_duplicate_noop_does_not_consume_budget_at_small_cap(self):
        # A duplicate/disabled no-op turn is a correction turn and must NOT spend the
        # caller's tool budget, so with max_tool_iterations=2 the model can still make a
        # DISTINCT valid call after repeating one. Only turns that actually execute a
        # tool count -- matching the GGUF loop. (The budget used to be charged per
        # non-re-prompt iteration, so the duplicate burned the second slot and the third
        # turn was sent with no tools, dropping the ``python`` call.)
        captured_tool_names: list[list[str]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'],
                ["final"],
            ]
        )

        def fake_single_turn(messages, active_tools = None):
            captured_tool_names.append(
                [
                    tool["function"]["name"]
                    for tool in (active_tools or [])
                    if tool.get("function", {}).get("name")
                ]
            )
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-result", "python-result"])
        _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [
                    {"type": "function", "function": {"name": "web_search"}},
                    {"type": "function", "function": {"name": "python"}},
                ],
                execute_tool = exec_fn,
                max_tool_iterations = 2,
            )
        )

        # Both distinct tools execute; the repeated call in between did not cost a slot.
        assert exec_fn.calls == [
            ("web_search", {"query": "x"}),
            ("python", {"code": "print(1)"}),
        ]
        # The turn after the duplicate still offered tools (budget not yet spent).
        assert captured_tool_names[2] == ["web_search", "python"]

    def test_repeated_duplicate_noop_transitions_to_final_attempt(self):
        captured_tool_names: list[list[str]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["final from first result"],
            ]
        )

        def fake_single_turn(messages, active_tools = None):
            captured_tool_names.append(
                [
                    (tool.get("function") or {}).get("name")
                    for tool in (active_tools or [])
                    if (tool.get("function") or {}).get("name")
                ]
            )
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-result"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 10,
            )
        )

        assert exec_fn.calls == [("web_search", {"query": "x"})]
        assert [
            event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
        ] == ["call_0"]
        assert captured_tool_names[-1] == []
        assert any(
            event.get("type") == "content" and "final from first result" in event.get("text", "")
            for event in events
        )

    def test_kb_search_capped_per_turn(self):
        # Paraphrased KB searches differ by args (dup guard misses them); the
        # per-turn cap stops the runaway re-search loop.
        n = RAG_MAX_SEARCHES_PER_TURN
        queries = [f"paraphrase {i}" for i in range(n + 1)]
        turns = [
            [
                '<tool_call>{"name":"search_knowledge_base",'
                f'"arguments":{{"query":"{q}"}}}}</tool_call>'
            ]
            for q in queries
        ] + [["final answer"]]
        turn_iter = iter(turns)

        def _gen(_messages):
            try:
                chunks = next(turn_iter)
            except StopIteration:
                return
            acc = ""
            for c in chunks:
                acc += c
                yield acc

        exec_fn = FakeExecuteTool([f"chunk-{i}" for i in range(n)])
        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}],
            execute_tool = exec_fn,
        )
        events = _collect_events(loop)
        assert len(exec_fn.calls) == n
        assert all(c[0] == "search_knowledge_base" for c in exec_fn.calls)
        tool_end_events = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_end_events) == n + 1
        assert "do not search again" in tool_end_events[n]["result"].lower()

    def test_image_sentinel_stripped_from_model_feed(self):
        # The image sentinel is stripped before the next turn, but tool_end still
        # carries the raw result for the UI.
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"python","arguments":{"code":"plot()"}}</tool_call>'],
                ["see chart"],
            ],
            exec_results = ["chart\n__IMAGES__:/tmp/chart.png"],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert "__IMAGES__" in tool_end["result"]

    def test_image_sentinel_stripped_with_leading_marker(self):
        # Sentinel at start (no newline) must not leak to the model.
        from core.inference import safetensors_agentic as _sa

        captured: list[list[dict]] = []

        def fake_single_turn(messages, **_kw):
            captured.append([dict(m) for m in messages])
            if len(captured) == 1:
                yield '<tool_call>{"name":"python","arguments":{"code":"plot()"}}</tool_call>'
            else:
                yield "done"

        events = list(
            _sa.run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "plot please"}],
                tools = [{"function": {"name": "python"}}],
                execute_tool = lambda *_a, **_kw: "__IMAGES__:/tmp/x.png",
                cancel_event = threading.Event(),
                max_tool_iterations = 3,
                auto_heal_tool_calls = True,
            )
        )
        # The model's second turn must not see "__IMAGES__".
        assert len(captured) >= 2
        tool_msgs = [m for m in captured[1] if m.get("role") == "tool"]
        assert tool_msgs, "no tool message reached the model"
        for tm in tool_msgs:
            assert "__IMAGES__" not in tm["content"], f"sentinel leaked to model: {tm['content']!r}"

    def test_image_sentinel_stripped_with_multiple_markers(self):
        # Consecutive sentinels: cut at the first, nothing leaks.
        from core.inference import safetensors_agentic as _sa

        captured: list[list[dict]] = []

        def fake_single_turn(messages, **_kw):
            captured.append([dict(m) for m in messages])
            if len(captured) == 1:
                yield '<tool_call>{"name":"python","arguments":{"code":"plot()"}}</tool_call>'
            else:
                yield "done"

        multi = "panel\n__IMAGES__:/tmp/a.png\n__IMAGES__:/tmp/b.png"
        events = list(
            _sa.run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "plot please"}],
                tools = [{"function": {"name": "python"}}],
                execute_tool = lambda *_a, **_kw: multi,
                cancel_event = threading.Event(),
                max_tool_iterations = 3,
                auto_heal_tool_calls = True,
            )
        )
        tool_msgs = [m for m in captured[1] if m.get("role") == "tool"]
        assert tool_msgs
        for tm in tool_msgs:
            assert "__IMAGES__" not in tm["content"], f"second sentinel leaked: {tm['content']!r}"
            assert tm["content"] == "panel", f"expected payload-only 'panel', got {tm['content']!r}"

    def test_tool_execution_error_is_emitted_but_loop_continues(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["sorry, that failed"],
            ],
            exec_results = ["Error: network unreachable"],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert tool_end["result"].startswith("Error")
        # The loop must still emit a content event after the failure.
        contents = [e for e in events if e["type"] == "content"]
        assert contents

    def test_exception_in_executor_does_not_raise(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["recovered"],
            ],
            exec_results = [RuntimeError("boom")],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert "boom" in tool_end["result"]


class TestLoopRePrompt:
    """Plan-without-action re-prompt parity with GGUF: nudge instead of terminating, up to ``MAX_ACT_REPROMPTS`` extra slots. Unsloth always nudges, so these drive the loop with ``nudge_tool_calls=True``."""

    def test_reasoning_intent_does_not_reprompt_a_visible_answer(self):
        generations = 0

        def _gen(_messages, active_tools = None):
            nonlocal generations
            generations += 1
            yield (
                "<think>Let me prepare the requested summary carefully.</think>"
                "This is the final visible answer."
            )

        exec_fn = FakeExecuteTool([])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = _gen,
                messages = [{"role": "user", "content": "summarize this"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                nudge_tool_calls = True,
            )
        )

        assert generations == 1
        assert exec_fn.calls == []
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1].endswith("This is the final visible answer.")

    def test_prefilled_reasoning_intent_does_not_reprompt_a_visible_answer(self):
        generations = 0

        def _gen(_messages, active_tools = None):
            nonlocal generations
            generations += 1
            yield "Let me prepare the requested summary carefully.</think>This is the final visible answer."

        exec_fn = FakeExecuteTool([])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = _gen,
                messages = [{"role": "user", "content": "summarize this"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                nudge_tool_calls = True,
                reasoning_prefilled = True,
            )
        )

        assert generations == 1
        assert exec_fn.calls == []
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1].endswith("This is the final visible answer.")

    def test_prefilled_reasoning_with_reemitted_think_does_not_reprompt(self):
        generations = 0

        def _gen(_messages, active_tools = None):
            nonlocal generations
            generations += 1
            yield (
                "Let me prepare the requested summary carefully."
                "<think>more private planning</think>This is the final visible answer."
            )

        exec_fn = FakeExecuteTool([])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = _gen,
                messages = [{"role": "user", "content": "summarize this"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                nudge_tool_calls = True,
                reasoning_prefilled = True,
            )
        )

        assert generations == 1
        assert exec_fn.calls == []
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1].endswith("This is the final visible answer.")

    def test_prefilled_reasoning_with_later_think_does_not_reprompt(self):
        generations = 0

        def _gen(_messages, active_tools = None):
            nonlocal generations
            generations += 1
            yield (
                "private prefilled planning</think>"
                "<think>Let me prepare the requested summary carefully.</think>"
                "This is the final visible answer."
            )

        exec_fn = FakeExecuteTool([])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = _gen,
                messages = [{"role": "user", "content": "summarize this"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                nudge_tool_calls = True,
                reasoning_prefilled = True,
            )
        )

        assert generations == 1
        assert exec_fn.calls == []
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1].endswith("This is the final visible answer.")

    def test_reasoning_only_intent_still_reprompts_and_uses_a_tool(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["<think>Let me search for that.</think>"],
                ['<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'],
                ["Here is the answer."],
            ],
            exec_results = ["result"],
            nudge_tool_calls = True,
        )

        events = _collect_events(loop)

        assert exec_fn.calls == [("web_search", {"query": "cats"})]
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1] == "Here is the answer."

    def test_prefilled_no_close_reasoning_intent_still_reprompts(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["I need more context.<think>Let me search for that."],
                ['<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'],
                ["Here is the answer."],
            ],
            exec_results = ["result"],
            nudge_tool_calls = True,
            reasoning_prefilled = True,
        )

        events = _collect_events(loop)

        assert exec_fn.calls == [("web_search", {"query": "cats"})]
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1] == "Here is the answer."

    def test_prefilled_reasoning_prefix_is_kept_for_reasoning_only_reprompt(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["Let me search for that.</think><think>checking details</think>"],
                ['<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'],
                ["Here is the answer."],
            ],
            exec_results = ["result"],
            nudge_tool_calls = True,
            reasoning_prefilled = True,
        )

        events = _collect_events(loop)

        assert exec_fn.calls == [("web_search", {"query": "cats"})]
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1] == "Here is the answer."

    def test_reprompt_history_uses_visible_intent_text(self):
        captured: list[list[dict]] = []

        def _gen(messages, active_tools = None):
            captured.append([dict(message) for message in messages])
            if len(captured) == 1:
                yield "<think>private planning details</think>Let me search for that."
            elif len(captured) == 2:
                yield '<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'
            else:
                yield "Here is the answer."

        exec_fn = FakeExecuteTool(["result"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = _gen,
                messages = [{"role": "user", "content": "find cats"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                nudge_tool_calls = True,
            )
        )

        assert exec_fn.calls == [("web_search", {"query": "cats"})]
        assert captured[1][1] == {"role": "assistant", "content": "Let me search for that."}
        contents = [e["text"] for e in events if e["type"] == "content"]
        assert contents[-1] == "Here is the answer."

    def test_intent_signal_triggers_reprompt(self):
        # Turn 1: intent signal, no tool call.
        # Turn 2 (re-prompt): proper tool call -> executes.
        # Turn 3: final answer.
        loop, exec_fn = _make_loop(
            turns = [
                ["Let me search for that."],
                [
                    '<tool_call>{"name":"web_search","arguments":'
                    '{"query":"sky color"}}</tool_call>'
                ],
                ["The sky is blue."],
            ],
            exec_results = ["Blue (Rayleigh scattering)"],
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        # web_search must have been called once (after the re-prompt).
        assert exec_fn.calls == [("web_search", {"query": "sky color"})]
        contents = [e for e in events if e["type"] == "content"]
        assert contents and "blue" in contents[-1]["text"].lower()

    def test_intent_signal_without_tools_does_not_reprompt(self):
        # Same intent signal but no tools enabled -- must NOT re-prompt.
        loop, exec_fn = _make_loop(
            turns = [["Let me think about that for a moment."]],
            exec_results = [],
        )
        # _make_loop hard-codes three tools; rebuild without tools.
        from core.inference.safetensors_agentic import run_safetensors_tool_loop

        def _gen(_messages):
            yield "Let me think about that for a moment."

        exec_fn = FakeExecuteTool([])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = _gen,
                messages = [{"role": "user", "content": "hi"}],
                tools = [],
                execute_tool = exec_fn,
            )
        )
        assert exec_fn.calls == []
        contents = [e for e in events if e["type"] == "content"]
        assert contents and "think" in contents[-1]["text"].lower()

    def test_direct_answer_does_not_trigger_reprompt(self):
        # Plain answer with no intent words: do NOT re-prompt.
        loop, exec_fn = _make_loop(
            turns = [["4"]],
            exec_results = [],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        contents = [e for e in events if e["type"] == "content"]
        assert contents and contents[-1]["text"].strip() == "4"

    def test_max_reprompts_capped(self):
        # Model keeps stalling with intent -- after MAX_ACT_REPROMPTS re-prompts
        # the loop must give up rather than burn forever.
        turns = [["Let me search for that."]] * 6  # well over the cap
        loop, exec_fn = _make_loop(
            turns = turns,
            exec_results = [],
            nudge_tool_calls = True,
        )
        events = _collect_events(loop, max_events = 500)
        # No tool ever ran, but the loop terminated cleanly.
        assert exec_fn.calls == []
        statuses = [e for e in events if e["type"] == "status"]
        assert statuses and statuses[-1]["text"] == ""

    def test_short_intent_below_buffer_threshold_triggers_reprompt(self):
        # Short emission that never exits BUFFERING (< 32 chars + no
        # marker prefix). The unified buffer-end path must still
        # trigger the intent re-prompt, not silently terminate.
        loop, exec_fn = _make_loop(
            turns = [
                ["Let me check."],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["found"],
            ],
            exec_results = ["..."],
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]

    def test_reprompt_does_not_consume_tool_budget(self):
        # max_tool_iterations=1: one re-prompt, then one real tool call,
        # then the budget-exhausted final answer must still fire. If the
        # re-prompt ate the slot the tool call would never run.
        loop, exec_fn = _make_loop(
            turns = [
                # 1. Intent stall (re-prompt).
                ["Let me search for that."],
                # 2. Real tool call (uses the budget slot).
                ['<tool_call>{"name":"web_search","arguments":{"query":"weather"}}</tool_call>'],
                # 3. Budget exhausted -> nudged final answer.
                ["Final: it is sunny"],
            ],
            exec_results = ["sunny"],
            max_tool_iterations = 1,
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "weather"})]
        contents = [e for e in events if e["type"] == "content"]
        assert contents and "sunny" in contents[-1]["text"].lower()


class TestLoopCanonicalHealKey:
    """Per-tool canonical heal key (``code``/``command``/``query``), mirroring GGUF."""

    def test_python_bare_string_heals_to_code(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"python","arguments":"print(1)"}' "</tool_call>"],
                ["done"],
            ],
            exec_results = ["1\n"],
        )
        events = _collect_events(loop)
        # The bare string must heal to {"code": "print(1)"}, not
        # {"query": ...}, so the python sandbox actually executes it.
        assert exec_fn.calls == [("python", {"code": "print(1)"})]

    def test_terminal_bare_string_heals_to_command(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"terminal","arguments":"ls -la"}' "</tool_call>"],
                ["done"],
            ],
            exec_results = ["..."],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("terminal", {"command": "ls -la"})]

    def test_unknown_tool_bare_string_heals_to_query(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":"hello"}' "</tool_call>"],
                ["ok"],
            ],
            exec_results = ["..."],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "hello"})]


class TestGGUFSafetensorsHealingParity:
    """Pin GGUF vs safetensors/MLX loop parity so a regression on either side breaks CI."""

    def test_gguf_imports_shared_signal_markers(self):
        # The GGUF BUFFERING state machine must wake on every emission
        # marker the shared parser knows -- otherwise Llama-3 / Mistral
        # / Gemma 4 emissions slip past as plain prose when the
        # llama-server structured channel fails.
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
        assert "_SHARED_TOOL_XML_SIGNALS" in src, (
            "GGUF agentic loop must reuse the shared TOOL_XML_SIGNALS "
            "tuple so it wakes on all five emission formats"
        )

    def test_gguf_uses_shared_strip_helper(self):
        # The GGUF stream-cleanup function must delegate to the shared
        # strip_tool_markup so closed-pair markup is removed for every
        # emission family (Llama-3 <|python_tag|>, Mistral [TOOL_CALLS],
        # Gemma 4 <|tool_call>...<tool_call|>).
        import inspect

        from core.inference.llama_cpp import LlamaCppBackend

        src = inspect.getsource(LlamaCppBackend.generate_chat_completion_with_tools)
        assert (
            "_shared_strip_tool_markup" in src
        ), "GGUF stream cleanup must delegate to the shared strip_tool_markup helper"

    def test_gguf_uses_canonical_heal_keys(self):
        # GGUF and safetensors heal a bare-string ``arguments`` to the same
        # per-tool canonical key -- ``code`` for python, ``command`` for
        # terminal, ``query`` for everything else. The mapping is centralised in
        # the shared ToolLoopController (both backends route bare-string args
        # through ``coerce_tool_arguments``), so the two paths cannot drift.
        from core.inference.tool_loop_controller import (
            _CANONICAL_HEAL_ARG,
            coerce_tool_arguments,
        )

        assert _CANONICAL_HEAL_ARG["python"] == "code"
        assert _CANONICAL_HEAL_ARG["terminal"] == "command"
        assert coerce_tool_arguments("print(1)", heal = True, tool_name = "python").arguments == {
            "code": "print(1)"
        }
        assert coerce_tool_arguments("ls -la", heal = True, tool_name = "terminal").arguments == {
            "command": "ls -la"
        }
        assert coerce_tool_arguments("weather", heal = True, tool_name = "web_search").arguments == {
            "query": "weather"
        }

    def test_intent_regex_matches_same_phrases_as_gguf(self):
        # The intent re-prompt regex is now a single shared source of truth
        # (tool_call_parser.INTENT_SIGNAL) consumed by both the GGUF and the
        # safetensors/MLX loops, so behaviour is identical on Mac and Linux.
        # Both backends must resolve to that one shared helper.
        from core.inference.llama_cpp import (
            _is_short_intent_without_action as gguf_fn,
        )
        from core.inference.safetensors_agentic import (
            is_short_intent_without_action as sf_fn,
        )
        from core.inference.tool_call_parser import (
            INTENT_SIGNAL as shared_re,
            is_short_intent_without_action as shared_fn,
        )

        assert gguf_fn is shared_fn and sf_fn is shared_fn

        for phrase in (
            "I'll search for that",
            "I will look it up",
            "Let me check",
            "I am going to call the tool",
            "First, I will explore",
            "Here's my plan",
            "Now I need to call web_search",
        ):
            assert shared_re.search(phrase), f"missed {phrase!r}"
            assert shared_fn(phrase), f"helper missed {phrase!r}"

        for plain in (
            "4",
            "Hello!",
            "The sky is blue.",
            "I can help with that.",
            "I should mention",
            "Let's go.",
            # Negated intent is a refusal, not a plan: neither backend may
            # force a tool-call re-prompt on it.
            "I will not search the web for that.",
            "I'll never call that tool.",
        ):
            assert not shared_re.search(plain), f"wrongly fired on {plain!r}"
            assert not shared_fn(plain), f"helper wrongly fired on {plain!r}"

    def test_max_reprompts_equal_on_both_backends(self):
        # Both loops draw the cap from the shared constant, so they stay equal.
        from core.inference.llama_cpp import _MAX_REPROMPTS as gguf_cap
        from core.inference.safetensors_agentic import MAX_ACT_REPROMPTS as sf_cap
        from core.inference.tool_call_parser import MAX_ACT_REPROMPTS as shared_cap

        assert gguf_cap == sf_cap == shared_cap


class TestLoopControl:
    def test_cancel_event_breaks_loop(self):
        cancel = threading.Event()
        cancel.set()
        # With cancel set, the loop bails before invoking execute_tool.
        exec_fn = FakeExecuteTool([])
        events = list(
            run_safetensors_tool_loop(
                single_turn = _const_stream(
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ),
                messages = [{"role": "user", "content": "hi"}],
                tools = [],
                execute_tool = exec_fn,
                cancel_event = cancel,
            )
        )
        assert events == []
        assert exec_fn.calls == []

    def test_max_iterations_caps_loop(self):
        # The loop stops after max_tool_iterations even if the model keeps
        # asking for tools, then emits a final-attempt round.
        loop, exec_fn = _make_loop(
            turns = [
                # Tool call (executes once).
                ['<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>'],
                # Model gives a final answer when nudged.
                ["here is the final answer"],
            ],
            exec_results = ["result"],
            max_tool_iterations = 1,
        )
        events = _collect_events(loop)
        contents = [e for e in events if e["type"] == "content"]
        # Final content must contain the final answer.
        assert contents and "final answer" in contents[-1]["text"]


class TestStatusFormatting:
    def test_status_for_known_tools(self):
        # Call the private helper directly to verify status formatting.
        assert (
            safetensors_agentic._status_for_tool("web_search", {"query": "abc"}) == "Searching: abc"
        )
        assert (
            safetensors_agentic._status_for_tool("web_search", {"url": "https://www.example.com/x"})
            == "Reading: example.com"
        )
        assert safetensors_agentic._status_for_tool("python", {"code": "x = 1"}).startswith(
            "Running Python:"
        )
        assert safetensors_agentic._status_for_tool("terminal", {"command": "ls"}).startswith(
            "Running:"
        )
        assert safetensors_agentic._status_for_tool("unknown_tool", {}).startswith("Calling:")


class TestProseMentioningToolCall:
    def test_assistant_prose_with_literal_tool_call_text_survives(self):
        # Regression: prose that mentions a literal ``<tool_call>`` (no real call)
        # must surface in full, not be stripped past the marker.
        loop, exec_fn = _make_loop(
            turns = [
                # A real tool call so the loop advances a turn.
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                # Prose that mentions the literal text.
                ["the docs say <tool_call> means an LLM tool call wrapper"],
            ],
            exec_results = ["result"],
        )
        events = _collect_events(loop)
        contents = [e for e in events if e["type"] == "content"]
        assert contents, "expected at least one content event"
        final = contents[-1]["text"]
        assert (
            "LLM tool" in final
        ), f"prose mentioning <tool_call> should not be truncated; got {final!r}"

    def test_tool_result_with_tool_call_text_does_not_retrigger(self):
        # A literal ``<tool_call>`` in the tool result must not re-trigger: the
        # loop parses only model output, so exactly one call.
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["the docs mention <tool_call> wrappers"],
            ],
            exec_results = ["Page text: <tool_call> appears here in the docs"],
        )
        events = _collect_events(loop)
        assert len(exec_fn.calls) == 1


class TestChatTemplateHelper:
    """Cover the dependency-light helper used by InferenceBackend."""

    def setup_method(self):
        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
        )
        self.apply = apply_chat_template_for_generation

    class _Tok:
        def __init__(self, accepted):
            self.accepted = accepted
            self.call_count = 0
            self.last_kwargs = None

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize = False,
            add_generation_prompt = True,
            **kw,
        ):
            self.call_count += 1
            unknown = set(kw) - self.accepted
            if unknown:
                raise TypeError(f"unexpected kwargs: {sorted(unknown)}")
            self.last_kwargs = dict(kw)
            return "PROMPT"

    def test_richest_call_wins_when_template_supports_all(self):
        tok = self._Tok({"tools", "enable_thinking"})
        self.apply(tok, [], tools = [{}], enable_thinking = True)
        assert tok.call_count == 1
        assert tok.last_kwargs is not None
        assert "tools" in tok.last_kwargs
        assert "enable_thinking" in tok.last_kwargs

    def test_falls_back_when_template_rejects_reasoning_kwarg(self):
        tok = self._Tok({"tools"})
        self.apply(tok, [], tools = [{}], enable_thinking = True)
        assert tok.call_count >= 2
        assert tok.last_kwargs == {"tools": [{}]}

    def test_falls_back_to_bare_call(self):
        tok = self._Tok(set())
        self.apply(tok, [], tools = [{}], enable_thinking = True)
        assert tok.last_kwargs == {}

    def test_jinja_error_propagates(self):
        class Boom:
            def apply_chat_template(self, *a, **kw):
                raise ValueError("jinja: missing var")

        with pytest.raises(ValueError):
            self.apply(Boom(), [])

    def test_no_kwargs_single_call(self):
        tok = self._Tok(set())
        self.apply(tok, [])
        assert tok.call_count == 1


# ────────────────────────────────────────────────────────────────────
# Guardrails (allowlist, budget, streaming-leak, dedup, id offset,
# auto_heal=False, canonical healed-arg key)
# ────────────────────────────────────────────────────────────────────


class TestGuardrails:
    def test_disabled_tool_is_not_executed(self):
        captured_messages: list[list[dict]] = []

        def fake_single_turn(messages):
            captured_messages.append([dict(message) for message in messages])
            if len(captured_messages) == 1:
                yield '<tool_call>{"name":"terminal","arguments":{"command":"echo bypass"}}</tool_call>'
            else:
                yield "final"

        exec_fn = FakeExecuteTool([])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 2,
            )
        )

        assert exec_fn.calls == []
        assert not [event for event in events if event.get("type") in {"tool_start", "tool_end"}]
        disabled_nudges = [
            message
            for message in captured_messages[-1]
            if message.get("role") == "user" and "not enabled" in message.get("content", "")
        ]
        assert len(disabled_nudges) == 1

    def test_empty_tools_list_means_allow_all_in_core_loop(self):
        turns = iter(
            [
                ['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'],
                ["done"],
            ]
        )

        def fake_single_turn(_messages, active_tools = None):
            assert active_tools == []
            acc = ""
            for chunk in next(turns):
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["OK"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [],
                execute_tool = exec_fn,
                max_tool_iterations = 2,
            )
        )
        assert exec_fn.calls == [("python", {"code": "print(1)"})]
        assert any(event.get("type") == "tool_end" for event in events)

    def test_max_iterations_zero_executes_no_tools(self):
        loop, exec_fn = _make_loop(
            turns = [['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>']],
            exec_results = ["OK"],
            max_tool_iterations = 0,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        assert events and events[-1] == {"type": "status", "text": ""}

    def test_streaming_clips_before_tool_signal_no_leak(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "I will look this up. ",
                    "Some more prose that's long enough to leave the buffer. ",
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>',
                ],
                ["all done"],
            ],
            exec_results = ["weather: sunny"],
            max_tool_iterations = 2,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]
        for e in events:
            if e["type"] == "content":
                assert "<tool_call>" not in e["text"]
                assert "web_search" not in e["text"]

    def test_auto_heal_disabled_still_parses_valid_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["done"],
            ],
            exec_results = ["OK"],
            auto_heal_tool_calls = False,
            max_tool_iterations = 2,
        )
        _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]

    def test_confirm_tool_calls_close_after_prompt_cleans_slot(self, monkeypatch):
        approval_id = "approval-close-sf"
        monkeypatch.setattr(safetensors_agentic, "new_approval_id", lambda: approval_id)

        loop, exec_fn = _make_loop(
            turns = [['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>']],
            exec_results = ["OK"],
            confirm_tool_calls = True,
            session_id = "sess",
            max_tool_iterations = 1,
        )

        with tool_approvals._lock:
            tool_approvals._pending.clear()

        try:
            assert next(loop)["type"] == "status"
            start = next(loop)
            assert start["type"] == "tool_start"
            assert start["approval_id"] == approval_id
            with tool_approvals._lock:
                assert approval_id in tool_approvals._pending
        finally:
            loop.close()

        with tool_approvals._lock:
            assert approval_id not in tool_approvals._pending
        assert resolve_tool_decision(approval_id, "allow", session_id = "sess") is False
        assert exec_fn.calls == []

    def test_confirm_tool_calls_skips_rag_autoinject(self, monkeypatch):
        def fail_autoinject(*_args, **_kwargs):
            raise AssertionError("RAG autoinject must not run before approval")

        monkeypatch.setattr("core.inference.tools.build_rag_autoinject", fail_autoinject)
        loop, exec_fn = _make_loop(
            turns = [["plain answer"]],
            confirm_tool_calls = True,
            rag_scope = {"thread_id": "t1"},
        )
        events = _collect_events(loop)
        assert any(e.get("type") == "content" and e.get("text") == "plain answer" for e in events)
        assert exec_fn.calls == []

    def test_auto_mode_still_runs_rag_autoinject(self, monkeypatch):
        # "auto" sends confirm_tool_calls=true so unsafe calls gate, but the
        # safe search_knowledge_base retrieval never gates, so autoinject must
        # still run (unlike ask mode above).
        ran = {"called": False}

        def fake_autoinject(*_args, **_kwargs):
            ran["called"] = True
            return None

        monkeypatch.setattr("core.inference.tools.build_rag_autoinject", fake_autoinject)
        loop, _exec_fn = _make_loop(
            turns = [["plain answer"]],
            confirm_tool_calls = True,
            permission_mode = "auto",
            rag_scope = {"thread_id": "t1"},
        )
        _collect_events(loop)
        assert ran["called"] is True

    def test_auto_heal_disabled_preserves_xml_on_final_no_tools_pass(self):
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"literal"}}</tool_call>'],
            ]
        )

        def fake_single_turn(_messages, active_tools = None):
            acc = ""
            for chunk in next(turns):
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["OK"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "show literal"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 1,
                auto_heal_tool_calls = False,
            )
        )
        assert exec_fn.calls == [("web_search", {"query": "x"})]
        assert any(
            event.get("type") == "content" and "<tool_call>" in event.get("text", "")
            for event in events
        )

    def test_auto_heal_disabled_does_not_repair_unclosed_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}'],
            ],
            exec_results = ["OK"],
            auto_heal_tool_calls = False,
            max_tool_iterations = 1,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        assert any(
            event.get("type") == "content" and "<tool_call>" in event.get("text", "")
            for event in events
        )

    def test_auto_heal_enabled_strips_unparseable_xml_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [["<tool_call>{not valid json}</tool_call>"]],
            exec_results = ["OK"],
            auto_heal_tool_calls = True,
            max_tool_iterations = 1,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        assert not any(
            event.get("type") == "content" and "<tool_call>" in event.get("text", "")
            for event in events
        )

    def test_non_consecutive_duplicate_is_short_circuited(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"B"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'],
                ["final"],
            ],
            exec_results = ["res-A", "res-B"],
            max_tool_iterations = 4,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "A"}), ("web_search", {"query": "B"})]
        assert [
            event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
        ] == ["call_0", "call_1"]
        assert not [
            event
            for event in events
            if event.get("tool_call_id") == "call_2"
            and event.get("type") in {"tool_start", "tool_end"}
        ]

    def test_same_turn_duplicate_is_short_circuited(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'
                    '<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'
                ],
                ["final"],
            ],
            exec_results = ["res-A"],
            max_tool_iterations = 2,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "A"})]
        assert [
            event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
        ] == ["call_0"]
        assert not [
            event
            for event in events
            if event.get("tool_call_id") == "call_1"
            and event.get("type") in {"tool_start", "tool_end"}
        ]

    def test_same_turn_distinct_calls_are_capped(self):
        # >_MAX_TOOL_CALLS_PER_TURN DISTINCT calls in one turn must be capped so a runaway turn
        # cannot fan out into many executions (the GGUF path is held back by llama-server's lazy ...
        from core.inference.safetensors_agentic import _MAX_TOOL_CALLS_PER_TURN

        n = _MAX_TOOL_CALLS_PER_TURN + 4
        turn = "".join(
            '<tool_call>{"name":"web_search","arguments":{"query":"q%d"}}</tool_call>' % i
            for i in range(n)
        )
        loop, exec_fn = _make_loop(
            turns = [[turn], ["final"]],
            exec_results = ["r"] * n,
            max_tool_iterations = 2,
        )
        _collect_events(loop)
        assert len(exec_fn.calls) == _MAX_TOOL_CALLS_PER_TURN
        # The first N distinct queries executed, in document order.
        assert [a["query"] for _name, a in exec_fn.calls] == [
            "q%d" % i for i in range(_MAX_TOOL_CALLS_PER_TURN)
        ]

    def test_coerce_string_args_python_uses_code_key(self):
        assert _coerce_arguments("print(1)", heal = True, tool_name = "python") == {"code": "print(1)"}

    def test_coerce_string_args_terminal_uses_command_key(self):
        assert _coerce_arguments("ls -la", heal = True, tool_name = "terminal") == {"command": "ls -la"}

    def test_tool_call_ids_unique_across_loop_iterations(self):
        loop, _exec = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"B"}}</tool_call>'],
                ["done"],
            ],
            exec_results = ["A", "B"],
            max_tool_iterations = 3,
        )
        events = _collect_events(loop)
        ids = [e["tool_call_id"] for e in events if e["type"] == "tool_start"]
        assert len(ids) == 2 and ids[0] != ids[1]


# ────────────────────────────────────────────────────────────────────
# Shared gpt-oss name detector
# ────────────────────────────────────────────────────────────────────


class TestGptOssNameDetection:
    def test_substring_match(self):
        assert is_gpt_oss_model_name("unsloth/gpt-oss-20b") is True

    def test_negative_known_non_oss_model(self):
        assert is_gpt_oss_model_name("meta-llama/Llama-3.1-8B-Instruct") is False

    def test_empty_or_none_returns_false(self):
        assert is_gpt_oss_model_name("") is False
        assert is_gpt_oss_model_name(cast(str, None)) is False


# ────────────────────────────────────────────────────────────────────
# Plan-without-action re-prompt (GGUF loop parity)
# ────────────────────────────────────────────────────────────────────


class TestPlanWithoutActionReprompt:
    def test_short_intent_is_reprompted_and_tool_executes(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["I'll search the web for that."],
                ['<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'],
                ["Here is the final answer."],
            ],
            exec_results = ["result-1"],
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert [c[0] for c in exec_fn.calls] == ["web_search"]
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert any("Here is the final answer." in t for t in texts)

    def test_reprompt_fires_up_to_the_cap(self):
        # GGUF parity: a persistently stalling model is re-prompted up to
        # MAX_ACT_REPROMPTS times, then the last stall is surrendered as the
        # final answer and no further turn is generated.
        from core.inference.tool_call_parser import MAX_ACT_REPROMPTS

        stall = "Let me look into it first."
        turns = [["I'll search the web for that."]]
        turns += [[stall]] * MAX_ACT_REPROMPTS
        turns += [["SHOULD NOT APPEAR"]]

        generations = {"count": 0}
        turn_iter = iter(turns)

        def _gen(_messages):
            generations["count"] += 1
            try:
                chunks = next(turn_iter)
            except StopIteration:
                return
            acc = ""
            for c in chunks:
                acc += c
                yield acc

        exec_fn = FakeExecuteTool([])
        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            execute_tool = exec_fn,
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        # One initial turn plus exactly MAX_ACT_REPROMPTS re-prompted turns.
        assert generations["count"] == MAX_ACT_REPROMPTS + 1
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert any(stall in t for t in texts)
        assert not any("SHOULD NOT APPEAR" in t for t in texts)

    def test_long_prose_answer_is_not_reprompted(self):
        long_answer = "I'll keep explaining the details of the topic. " * 60
        loop, exec_fn = _make_loop(
            turns = [
                [long_answer],
                ["SHOULD NOT APPEAR"],
            ],
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert not any("SHOULD NOT APPEAR" in t for t in texts)

    def test_disabled_auto_heal_is_not_reprompted(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["I'll search the web for that."],
                ["SHOULD NOT APPEAR"],
            ],
            auto_heal_tool_calls = False,
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert any("I'll search the web for that." in t for t in texts)
        assert not any("SHOULD NOT APPEAR" in t for t in texts)

    def test_explicit_nudge_off_is_not_reprompted(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["I'll search the web for that."],
                ["SHOULD NOT APPEAR"],
            ],
            nudge_tool_calls = False,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert any("I'll search the web for that." in t for t in texts)
        assert not any("SHOULD NOT APPEAR" in t for t in texts)

    def test_omitted_nudge_flag_is_not_reprompted(self):
        # The retry is new on this loop: API callers who do not send the flag
        # must keep today's behavior. Unsloth opts in explicitly.
        loop, exec_fn = _make_loop(
            turns = [
                ["I'll search the web for that."],
                ["SHOULD NOT APPEAR"],
            ],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert any("I'll search the web for that." in t for t in texts)
        assert not any("SHOULD NOT APPEAR" in t for t in texts)

    def test_rag_autoinject_counts_as_executed_tool(self, monkeypatch):
        # Autoinject already ran a KB search outside the controller; a short
        # post-retrieval intent must not trigger a spurious re-prompt.
        import core.inference.tools as tools_mod

        def fake_autoinject(conversation, rag_scope):
            return {
                "events": [
                    {"type": "tool_start", "tool_name": "search_knowledge_base"},
                    {"type": "tool_end", "tool_name": "search_knowledge_base"},
                ],
                "messages": [{"role": "tool", "content": "kb result"}],
            }

        monkeypatch.setattr(tools_mod, "build_rag_autoinject", fake_autoinject)
        loop, exec_fn = _make_loop(
            turns = [
                ["I'll search the docs."],
                ["SHOULD NOT APPEAR"],
            ],
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        assert any(e.get("type") == "tool_start" for e in events)
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert any("I'll search the docs." in t for t in texts)
        assert not any("SHOULD NOT APPEAR" in t for t in texts)

    def test_no_reprompt_after_a_denied_tool_confirmation(self, monkeypatch):
        # An explicit user denial must not be answered with a nudge to call
        # the tool again (which would raise another confirmation prompt).
        monkeypatch.setattr(safetensors_agentic, "new_approval_id", lambda: "appr-1")
        monkeypatch.setattr(safetensors_agentic, "begin_tool_decision", lambda *_a, **_k: object())
        monkeypatch.setattr(safetensors_agentic, "wait_tool_decision", lambda *_a, **_k: "deny")
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'],
                ["I'll search again."],
                ["SHOULD NOT APPEAR"],
            ],
            confirm_tool_calls = True,
            session_id = "sess",
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert any("I'll search again." in t for t in texts)
        assert not any("SHOULD NOT APPEAR" in t for t in texts)

    def test_no_reprompt_after_a_tool_already_executed(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'],
                ["Now I'll refine the search."],
                ["SHOULD NOT APPEAR"],
            ],
            exec_results = ["result-1"],
            nudge_tool_calls = True,
        )
        events = _collect_events(loop)
        assert [c[0] for c in exec_fn.calls] == ["web_search"]
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert not any("SHOULD NOT APPEAR" in t for t in texts)


# Routes-level python_tag strip (multi-line; stop on next sentinel)
class TestRoutesPythonTagStrip:
    """``_TOOL_XML_RE`` must consume multi-line code, embedded JSON, and bare ``<`` (earlier ``[^\n<]*`` / ``[^\n]*`` revisions leaked tails); the streaming route-level strip is the regression-prone path."""

    def _strip(self, text: str) -> str:
        # Import inside the test so a routes-module import error does
        # not blow up the entire test file at collection time.
        from routes.inference import _strip_tool_xml
        return _strip_tool_xml(text)

    def test_single_line_python_tag_stripped(self):
        # Floor: the original 5620 single-line behaviour still works.
        text = '<|python_tag|>brave_search.call(query="weather")'
        assert self._strip(text) == ""

    def test_python_tag_with_less_than_in_code(self):
        # 5615 regression: literal ``<`` inside code must NOT terminate
        # the strip early.
        text = '<|python_tag|>python.call(code="if x < 10: pass")'
        assert self._strip(text) == ""

    def test_python_tag_multiline_code_stripped(self):
        # 5620 round-1 regression: multi-line code's second line leaked.
        text = '<|python_tag|>python.call(code="line1\nline2\nline3")'
        assert self._strip(text) == ""

    def test_python_tag_multiline_with_less_than(self):
        # Combined: multi-line code AND literal ``<`` in code.
        text = (
            '<|python_tag|>python.call(code="for i in range(10):\n'
            "    if i < 5:\n"
            '        print(i)")'
        )
        assert self._strip(text) == ""

    def test_python_tag_stops_at_eom_sentinel(self):
        # Strip stops at the next Llama-3 ``<|`` sentinel so any
        # trailing assistant content survives.
        text = '<|python_tag|>python.call(code="multi\nline")' "<|eom_id|>final answer text"
        assert self._strip(text) == "<|eom_id|>final answer text"

    def test_python_tag_stops_at_eot_sentinel(self):
        text = '<|python_tag|>brave_search.call(query="x")' "<|eot_id|>after"
        assert self._strip(text) == "<|eot_id|>after"

    def test_python_tag_json_form_multiline_stripped(self):
        # The JSON form of python_tag with newlines inside string args.
        text = '<|python_tag|>{"name":"python","parameters":{"code":"a = 1\nb = 2\nprint(a+b)"}}'
        assert self._strip(text) == ""

    def test_python_tag_with_eom_then_trailing_python_tag(self):
        # Two python_tag emissions back-to-back across a sentinel: both
        # should strip independently.
        text = (
            '<|python_tag|>brave_search.call(query="a")'
            "<|eom_id|>"
            '<|python_tag|>python.call(code="x=1")'
        )
        # ``<|eom_id|>`` between the two strips remains; both
        # python_tag blocks are fully consumed.
        assert self._strip(text) == "<|eom_id|>"


# Robustness fixes uncovered while validating against vLLM / sglang.
class TestParserRobustness:
    def test_tool_call_json_accepts_parameters_key(self):
        # Hermes wrapper around a Llama-3.2 bare-JSON object that uses
        # ``parameters`` instead of ``arguments``. The bare-JSON and
        # python_tag paths already accept both keys; this path now does
        # too. Was extracting name only and silently dropping the args.
        import json

        text = "<tool_call>\n" '{"name": "search", "parameters": {"q": "ramen"}}\n' "</tool_call>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "search"
        assert json.loads(result[0]["function"]["arguments"]) == {"q": "ramen"}

    def test_function_xml_attribute_form(self):
        # MiniCPM-5 / MiniMax-M2 attribute syntax:
        # ``<function name="..."><param name="...">v</param></function>``.
        import json

        text = '<function name="get_weather">' '<param name="city">Tokyo</param>' "</function>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Tokyo"}

    def test_function_xml_attribute_form_multi_param(self):
        import json

        text = (
            '<function name="get_weather">'
            '<param name="city">Tokyo</param>'
            '<param name="unit">celsius</param>'
            "</function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"city": "Tokyo", "unit": "celsius"}

    def test_function_xml_legacy_equals_form_still_works(self):
        # Regression guard: the old ``<function=name><parameter=k>v``
        # syntax must keep parsing after the regex broadening.
        import json

        text = "<function=get_weather><parameter=city>Tokyo</parameter></function>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Tokyo"}

    def test_function_attribute_form_has_tool_signal(self):
        # The standalone ``<function name="...">`` attribute form must flip
        # the streaming buffer; otherwise the end-of-turn safety-net parse in
        # the agentic loop is gated off and the real call is dropped.
        assert has_tool_signal('<function name="get_weather">') is True

    def test_function_attribute_form_strip_markup(self):
        # The attribute form must also be stripped from displayed text, like
        # the legacy ``<function=...>`` form.
        text = 'result <function name="g"><param name="c">X</param></function>'
        assert strip_tool_markup(text, final = True) == "result"

    def test_llama3_chat_template_round_trip(self):
        # Meta's official Llama-3.x chat template prefixes every
        # assistant turn with
        # ``<|start_header_id|>assistant<|end_header_id|>\n\n``. The
        # sentinel-strip in ``_parse_llama3_bare_json`` must reach past
        # the role label to the JSON body, else every round-tripped
        # tool call in history silently drops.
        import json

        text = (
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            '{"name": "get_weather", "parameters": {"city": "Tokyo"}}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Tokyo"}

    def test_llama3_round_trip_all_roles(self):
        # Same logic must work for every role the chat template inserts.
        import json
        for role in ("assistant", "user", "system", "tool", "ipython"):
            text = (
                f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                '{"name": "f", "parameters": {"x": 1}}'
            )
            result = parse_tool_calls_from_text(text)
            assert len(result) == 1, f"failed for role={role}"
            assert json.loads(result[0]["function"]["arguments"]) == {"x": 1}

    def test_llama3_round_trip_with_eot_prefix(self):
        # Prior assistant turn closes with ``<|eot_id|>``, then the
        # new header opens. Both sentinels + the role must be consumed.
        import json

        text = (
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            '{"name": "f", "parameters": {}}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "f"

    def test_function_xml_followed_by_prose(self):
        # Models routinely follow a tool call with explanatory prose.
        # Body must terminate at ``</function>`` even without a
        # ``</tool_call>`` wrapper, else trailing prose leaks into the
        # last parameter value.
        import json

        text = (
            "<function=get_weather>"
            "<parameter=city>Tokyo</parameter>"
            "</function>\n\nHere is what I found."
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Tokyo"}

    def test_function_attribute_xml_followed_by_prose(self):
        # Same expectation for the MiniCPM-5 attribute form.
        import json

        text = (
            '<function name="get_weather">'
            '<param name="city">Tokyo</param>'
            "</function>\n\nLet me know if you need anything else."
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert json.loads(result[0]["function"]["arguments"]) == {"city": "Tokyo"}


def test_render_with_native_template_returns_render_only_when_tools_emitted():
    # The native-template fallback re-renders with the model's repo template when an override drops
    # the tools schema.
    from types import SimpleNamespace

    from core.inference.chat_template_helpers import render_native_template

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    model_info = {
        "native_chat_template": "TPL",
        "tokenizer": SimpleNamespace(chat_template = "OVERRIDE"),
    }

    def emitting(tokenizer, msgs, *, tools, **_kw):
        body = "".join(m["content"] for m in msgs)
        return body + ("|TOOLS=" + ",".join(t["function"]["name"] for t in tools) if tools else "")

    def ignoring(tokenizer, msgs, *, tools, **_kw):
        return "".join(m["content"] for m in msgs)  # never reflects tools

    out = render_native_template(
        model_info = dict(model_info),
        active_model_name = "x",
        messages = messages,
        tools = tools,
        apply_fn = emitting,
    )
    assert out == "hi|TOOLS=web_search"
    # The native template must be restored on the live tokenizer after probing.
    assert model_info["tokenizer"].chat_template == "OVERRIDE"

    assert (
        render_native_template(
            model_info = dict(model_info),
            active_model_name = "x",
            messages = messages,
            tools = tools,
            apply_fn = ignoring,
        )
        is None
    )

    # No tokenizer and no processor -> return None instead of an AttributeError.
    no_tok = {"native_chat_template": "TPL"}
    assert (
        render_native_template(
            model_info = no_tok,
            active_model_name = "x",
            messages = messages,
            tools = tools,
            apply_fn = emitting,
        )
        is None
    )


def test_render_with_native_template_does_not_mutate_shared_tokenizer():
    # The shared tokenizer must never carry the temporary native template, even mid-render: this
    # runs outside the generation lock, so a concurrent request could otherwise render with the ...
    from types import SimpleNamespace

    from core.inference.chat_template_helpers import render_native_template

    shared = SimpleNamespace(chat_template = "OVERRIDE")
    seen = []

    def capture(tokenizer, msgs, *, tools, **_kw):
        seen.append((tokenizer is shared, shared.chat_template))
        body = "".join(m["content"] for m in msgs)
        return body + ("|T" if tools else "")

    model_info = {"native_chat_template": "TPL", "tokenizer": shared}
    render_native_template(
        model_info = model_info,
        active_model_name = "x",
        messages = [{"role": "user", "content": "hi"}],
        tools = [{"type": "function", "function": {"name": "web_search"}}],
        apply_fn = capture,
    )
    # Rendering happened on a copy, and the shared tokenizer stayed "OVERRIDE"
    # throughout (never the temporary "TPL").
    assert seen and all(not is_shared for is_shared, _ in seen)
    assert all(tpl == "OVERRIDE" for _, tpl in seen)
    assert shared.chat_template == "OVERRIDE"


def test_native_template_loads_from_base_model_for_lora(monkeypatch):
    # For a LoRA adapter the chat template lives on the base model; active_model_name
    # is the adapter id and may ship no template. The loader must read base_model.
    from types import SimpleNamespace

    import transformers

    from core.inference.chat_template_helpers import render_native_template

    captured = {}

    def fake_from_pretrained(name, *args, **kwargs):
        captured["source"] = name
        return SimpleNamespace(chat_template = "BASE_TPL")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", fake_from_pretrained)

    def emitting(tokenizer, msgs, *, tools, **_kw):
        body = "".join(m["content"] for m in msgs)
        return body + ("|T" if tools else "")

    model_info = {
        "base_model": "base/model-id",
        "tokenizer": SimpleNamespace(chat_template = "OVERRIDE"),
    }
    out = render_native_template(
        model_info = model_info,
        active_model_name = "adapter/path",
        messages = [{"role": "user", "content": "hi"}],
        tools = [{"type": "function", "function": {"name": "web_search"}}],
        apply_fn = emitting,
    )
    assert captured["source"] == "base/model-id"
    assert out == "hi|T"


def test_render_with_native_template_fallback_swaps_when_override_drops_tools():
    # The shared gate (used by the transformers and MLX backends): when the live render is
    # identical with and without tools, re-render with the native template and return it.
    from types import SimpleNamespace

    from core.inference.chat_template_helpers import render_with_native_template_fallback

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]

    # apply_fn that IGNORES tools -> live render drops the schema.
    def ignoring(tokenizer, msgs, *, tools, **_kw):
        return "".join(m["content"] for m in msgs)

    model_info = {
        "native_chat_template": "TPL",
        "tokenizer": SimpleNamespace(chat_template = "OVERRIDE"),
    }

    # Native render emits the tools, so the fallback swaps to it.
    def native_emits(tokenizer, msgs, *, tools, **_kw):
        body = "".join(m["content"] for m in msgs)
        return body + ("|TOOLS" if tools else "")

    out = render_with_native_template_fallback(
        formatted_prompt = ignoring(None, messages, tools = tools),
        tokenizer = SimpleNamespace(),
        model_info = dict(model_info),
        active_model_name = "x",
        messages = messages,
        tools = tools,
        apply_fn = lambda tok, msgs, *, tools, **kw: (
            native_emits(tok, msgs, tools = tools)
            if getattr(tok, "chat_template", None) == "TPL"
            else ignoring(tok, msgs, tools = tools)
        ),
    )
    assert out == "hi|TOOLS", out


def test_render_with_native_template_fallback_keeps_prompt_when_tools_emitted():
    # Live render already differs with vs without tools -> no fallback, returned
    # unchanged. Also a no-tools call is a passthrough.
    from types import SimpleNamespace

    from core.inference.chat_template_helpers import render_with_native_template_fallback

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]

    def emitting(tokenizer, msgs, *, tools, **_kw):
        body = "".join(m["content"] for m in msgs)
        return body + ("|T" if tools else "")

    kept = render_with_native_template_fallback(
        formatted_prompt = emitting(None, messages, tools = tools),
        tokenizer = SimpleNamespace(),
        model_info = {"native_chat_template": "TPL", "tokenizer": SimpleNamespace()},
        active_model_name = "x",
        messages = messages,
        tools = tools,
        apply_fn = emitting,
    )
    assert kept == "hi|T", kept

    # No tools -> passthrough (native template never consulted).
    passthrough = render_with_native_template_fallback(
        formatted_prompt = "hi",
        tokenizer = SimpleNamespace(),
        model_info = {},
        active_model_name = "x",
        messages = messages,
        tools = None,
        apply_fn = emitting,
    )
    assert passthrough == "hi"


def test_render_with_native_template_fallback_keeps_prompt_when_no_tools_probe_raises():
    # A template that REQUIRES tools can raise on the no-tools probe.
    from types import SimpleNamespace

    from core.inference.chat_template_helpers import render_with_native_template_fallback

    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]

    def raises_without_tools(tokenizer, msgs, *, tools, **_kw):
        if not tools:
            raise RuntimeError("template requires tools")
        return "".join(m["content"] for m in msgs) + "|T"

    out = render_with_native_template_fallback(
        formatted_prompt = "hi|T",
        tokenizer = SimpleNamespace(),
        model_info = {"native_chat_template": "TPL", "tokenizer": SimpleNamespace()},
        active_model_name = "x",
        messages = messages,
        tools = tools,
        apply_fn = raises_without_tools,
    )
    assert out == "hi|T", out


def test_truncated_bare_json_at_eof_is_not_leaked():
    # Stream ends mid bare-JSON object: the held fragment must be dropped at the
    # EOF resolver, not flushed as plain assistant content (GGUF parity).
    loop, _exec = _make_loop(
        turns = [['{"name":"web_search","parameters":{"query":"weather in S']],
        max_tool_iterations = 1,
    )
    events = _collect_events(loop)
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any('"name"' in t for t in contents), contents


def test_oversized_bare_json_call_is_not_leaked_and_executes():
    # A bare-JSON call whose arguments exceed _MAX_BARE_JSON_BUFFER must DRAIN
    # (suppress) rather than stream the raw JSON prefix, and still execute once
    # the full object is parsed by the safety net.
    from core.inference.safetensors_agentic import _MAX_BARE_JSON_BUFFER

    big = "A" * (_MAX_BARE_JSON_BUFFER + 5000)
    full = '{"name":"web_search","parameters":{"code":"' + big + '"}}'
    chunks = [full[i : i + 2000] for i in range(0, len(full), 2000)]
    loop, exec_fn = _make_loop(turns = [chunks, ["done"]], exec_results = ["OK"], max_tool_iterations = 2)
    events = _collect_events(loop)
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert not any(t.lstrip().startswith('{"name') for t in contents), contents[:1]
    assert exec_fn.calls and exec_fn.calls[0][0] == "web_search"
    assert len(exec_fn.calls[0][1].get("code", "")) > _MAX_BARE_JSON_BUFFER


def test_oversized_plain_json_answer_still_streams():
    # A giant plain JSON answer (no "name" key) is NOT a tool call and must still
    # stream -- the oversized DRAIN route is gated on a "name" key.
    from core.inference.safetensors_agentic import _MAX_BARE_JSON_BUFFER

    big = "A" * (_MAX_BARE_JSON_BUFFER + 5000)
    full = '{"result":"' + big + '"}'
    chunks = [full[i : i + 2000] for i in range(0, len(full), 2000)]
    loop, _exec = _make_loop(turns = [chunks], max_tool_iterations = 1)
    events = _collect_events(loop)
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert '"result"' in contents


def test_oversized_disabled_name_json_answer_still_streams():
    # A giant still-open JSON answer whose "name" is NOT an enabled tool must stream:
    # the oversized DRAIN branch was gated only on the presence of a "name" key, so a
    # large ordinary record ({"name":"Alice",...}) was drained instead of shown.
    from core.inference.safetensors_agentic import _MAX_BARE_JSON_BUFFER

    big = "A" * (_MAX_BARE_JSON_BUFFER + 5000)
    answer = '{"name":"Alice","parameters":{"bio":"' + big  # never closes
    chunks = [answer[i : i + 2000] for i in range(0, len(answer), 2000)]
    loop, exec_fn = _make_loop(turns = [chunks], max_tool_iterations = 1)
    events = _collect_events(loop)
    assert exec_fn.calls == [], exec_fn.calls
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert "Alice" in contents, contents[:80]


def test_truncated_disabled_name_json_is_shown_at_eof():
    # A truncated ordinary JSON answer whose name is not an enabled tool, held to EOF,
    # must be shown -- the EOF bare-JSON DRAIN branch was gated only on a "name" key.
    truncated = '{"name":"Alice","parameters":{"age":'
    loop, exec_fn = _make_loop(turns = [[truncated]], max_tool_iterations = 1)
    events = _collect_events(loop)
    assert exec_fn.calls == [], exec_fn.calls
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert "Alice" in contents, contents


def test_truncated_plain_json_with_nested_enabled_name_is_visible():
    # A truncated ordinary JSON answer with a NESTED ``"name"`` matching an enabled
    # tool ({"result":{"name":"web_search",...) must be shown, not suppressed: the
    # gate now extracts the TOP-LEVEL name only, so the nested field is just data.
    loop, exec_fn = _make_loop(
        turns = [['{"result":{"name":"web_search","age":']],
        max_tool_iterations = 1,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == []
    contents = "".join(e["text"] for e in events if e["type"] == "content")
    assert '"result"' in contents and "web_search" in contents, contents


def test_bare_json_call_not_replayed_in_next_turn_content():
    # After a complete bare-JSON call executes, the assistant content fed to the
    # next turn must not contain the raw call (next-turn contamination).
    captured: list[list[dict]] = []
    exec_fn = FakeExecuteTool(["RESULT"])

    def st(messages, active_tools = None):
        captured.append([dict(m) for m in messages])
        if len(captured) == 1:
            yield '{"name":"web_search","parameters":{"query":"cats"}}'
        else:
            yield "Found."

    _collect_events(
        run_safetensors_tool_loop(
            single_turn = st,
            messages = [{"role": "user", "content": "cats"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            execute_tool = exec_fn,
            max_tool_iterations = 3,
        )
    )
    assert len(captured) >= 2, captured
    asst = [m for m in captured[1] if m.get("role") == "assistant"]
    assert asst and not any('"name"' in (m.get("content") or "") for m in asst), asst


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_streaming_strip_keeps_bare_args_before_think_block():
    # F3: a bare ``foo[ARGS]`` before a think block is prose; EOS-anchored tail arms run only
    # on the last segment.
    text = "Please pass foo[ARGS] <think>pause</think> to the template."
    out = strip_tool_markup_streaming(text, tool_protocol_active = True)
    assert out == text


def test_streaming_strip_still_removes_complete_call_before_think_block():
    # A complete bracket call before a think block still strips in the non-last segment.
    text = 'go web_search[ARGS]{"q":"x"} <think>z</think> done'
    out = strip_tool_markup_streaming(text, tool_protocol_active = True)
    assert "web_search[ARGS]" not in out
    assert "<think>z</think>" in out
    assert "go" in out and "done" in out


def test_prose_args_marker_before_real_call_does_not_drain_the_prose():
    # F5: an inactive ``foo[ARGS]`` in prose is not a call boundary; the prose streams in
    # full and the later real call still executes.
    loop, exec_fn = _make_loop(
        turns = [
            ["Intro ", "foo[ARGS] syntax. ", 'web_search[ARGS]{"query":"cats"}'],
            ["Cats are great."],
        ],
        exec_results = ["RESULT"],
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [("web_search", {"query": "cats"})], exec_fn.calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    # The prose between the bogus marker and the real call must survive.
    assert any("foo[ARGS] syntax." in t for t in contents), contents
    # The real call markup is never shown as content.
    assert not any("web_search[ARGS]" in t for t in contents), contents


def test_inactive_name_args_with_body_is_not_parsed_into_disabled_noop():
    # BUG A: a prose answer with an inactive ``foo[ARGS]{...}`` is not drained into a
    # disabled no-op extra turn; the [ARGS] checks are name-gated.
    turns = [['foo[ARGS]{"x":1} is just syntax.']]
    turn_calls: list[int] = []

    def _gen(_messages):
        turn_calls.append(1)
        chunks = turns[len(turn_calls) - 1] if len(turn_calls) <= len(turns) else []
        acc = ""
        for chunk in chunks:
            acc += chunk
            yield acc

    exec_fn = FakeExecuteTool([])
    loop = run_safetensors_tool_loop(
        single_turn = _gen,
        messages = [{"role": "user", "content": "explain"}],
        tools = [{"type": "function", "function": {"name": "web_search"}}],
        execute_tool = exec_fn,
        max_tool_iterations = 3,
    )
    events = _collect_events(loop)
    assert exec_fn.calls == [], exec_fn.calls
    assert not any(e["type"] in ("tool_start", "tool_end") for e in events), events
    # Exactly one generation turn -- no disabled ``foo`` no-op re-prompt.
    assert len(turn_calls) == 1, turn_calls
    contents = [e["text"] for e in events if e["type"] == "content"]
    assert any("is just syntax." in t for t in contents), contents


class TestEnabledToolNameGate:
    """The safetensors loop passes the active tool names into parse/strip so the
    ambiguous bare-rehearsal ``NAME[ARGS]{json}`` is treated as a call only when NAME
    is an active tool (#5704). Without the gate an inactive ``foo[ARGS]{...}`` in prose
    was parsed into a disabled no-op call and stripped from the visible text."""

    def _names(self, calls):
        return [c["function"]["name"] for c in calls]

    def test_parse_inactive_rehearsal_does_not_swallow_active_call(self):
        text = 'foo[ARGS]{"a":1} web_search[ARGS]{"query":"cats"}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert self._names(calls) == ["web_search"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}

    def test_parse_inactive_rehearsal_alone_is_prose(self):
        assert (
            parse_tool_calls_from_text('foo[ARGS]{"a":1}', enabled_tool_names = {"web_search"}) == []
        )

    def test_streaming_strip_keeps_inactive_rehearsal(self):
        raw = 'answer foo[ARGS]{"x":1} tail'
        assert strip_tool_markup_streaming(raw, enabled_tool_names = {"web_search"}) == raw

    def test_streaming_strip_removes_active_rehearsal(self):
        raw = 'answer web_search[ARGS]{"q":1} tail'
        out = strip_tool_markup_streaming(raw, enabled_tool_names = {"web_search"})
        assert "web_search[ARGS]" not in out
        assert out == "answer  tail"

    def test_final_strip_keeps_inactive_rehearsal(self):
        text = 'foo[ARGS]{"x":1} is just syntax.'
        assert strip_tool_markup(text, final = True, enabled_tool_names = {"web_search"}) == text

    def test_gate_none_preserves_legacy_strip_and_parse(self):
        text = 'foo[ARGS]{"x":1} tail'
        assert self._names(parse_tool_calls_from_text(text)) == ["foo"]
        assert strip_tool_markup_streaming(text) == " tail"


def test_drain_truncated_enabled_name_json_preserved_when_auto_heal_disabled():
    # F3: with Auto-Heal OFF, a truncated ENABLED-name bare-JSON fragment that did
    # not parse must stay visible (disabled-Auto-Heal contract: malformed markup is
    # preserved), matching the XML strip in the same drain branch. With Auto-Heal ON
    # the same fragment is suppressed.
    trunc = '{"name":"web_search","parameters":{"query":"weather'
    off, exec_off = _make_loop(turns = [[trunc]], max_tool_iterations = 1, auto_heal_tool_calls = False)
    events_off = _collect_events(off)
    assert exec_off.calls == [], exec_off.calls
    contents_off = "".join(e["text"] for e in events_off if e["type"] == "content")
    assert "web_search" in contents_off, contents_off

    on, exec_on = _make_loop(turns = [[trunc]], max_tool_iterations = 1, auto_heal_tool_calls = True)
    events_on = _collect_events(on)
    assert exec_on.calls == [], exec_on.calls
    contents_on = "".join(e["text"] for e in events_on if e["type"] == "content")
    assert "web_search" not in contents_on, contents_on


def test_looks_like_enabled_bare_json_accepts_function_alias():
    # The safetensors buffering gate must recognise the "function" bare-JSON alias
    # the parser accepts, so a truncated/complete {"function":<enabled tool>} call is
    # buffered/healed instead of streaming as visible content.
    from core.inference.safetensors_agentic import _looks_like_enabled_bare_json

    enabled = {"web_search"}
    assert _looks_like_enabled_bare_json(
        '{"function":"web_search","parameters":{"q":"x"}}', enabled
    )
    # A non-tool "function" value is an ordinary JSON answer -> not gated.
    assert not _looks_like_enabled_bare_json('{"function":"Alice","parameters":{}}', enabled)


class TestFalseAlarmMarkerProse:
    def test_leading_marker_prose_streams_intact(self):
        # An answer that starts with a literal marker is a false alarm: the
        # drain finds no calls and the full prose must reach the client.
        text = "[TOOL_CALLS] is the Mistral tool marker. More prose after."
        loop, exec_fn = _make_loop(turns = [[text]])
        events = _collect_events(loop)
        assert exec_fn.calls == []
        texts = [e["text"] for e in events if e["type"] == "content"]
        assert texts and texts[-1] == text

    def test_chained_bare_json_calls_not_replayed_in_history(self):
        # Both chained calls execute; the kept content (next-turn assistant
        # history) must not contain the second call's raw JSON.
        chained = (
            '{"name":"web_search","parameters":{"q":"first"}};'
            '{"name":"get_weather","parameters":{"code":"x"}}'
        )
        convs = []
        turn_iter = iter([[chained], ["Final answer."]])

        def gen(messages, active_tools = None):
            convs.append([dict(m) for m in messages])
            try:
                chunks = next(turn_iter)
            except StopIteration:
                return
            acc = ""
            for c in chunks:
                acc += c
                yield acc

        exec_fn = FakeExecuteTool(["r1", "r2"])
        loop = run_safetensors_tool_loop(
            single_turn = gen,
            messages = [{"role": "user", "content": "hi"}],
            tools = [
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "get_weather"}},
            ],
            execute_tool = exec_fn,
        )
        _collect_events(loop)
        assert [c[0] for c in exec_fn.calls] == ["web_search", "get_weather"]
        assistant = next(m for m in convs[1] if m["role"] == "assistant")
        assert '"get_weather"' not in (assistant.get("content") or "")
