# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Strict-mode (Auto-Heal disabled) tool-call parsing.

With ``allow_incomplete=False`` the parser must accept a well-formed
``<function=...>...</function>`` call even when the model appends prose
after the closing tag -- matching the JSON-style ``<tool_call>...`` path,
which already tolerates trailing text -- while still rejecting genuinely
truncated calls that never close.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tool_call_parser import parse_tool_calls_from_text


def _only(text: str) -> dict:
    calls = parse_tool_calls_from_text(text, allow_incomplete = False)
    assert len(calls) == 1, f"expected exactly one call, got {len(calls)}: {calls!r}"
    fn = calls[0]["function"]
    return {"name": fn["name"], "arguments": json.loads(fn["arguments"])}


class TestFunctionStyleTrailingText:
    def test_closed_function_with_trailing_prose_is_accepted(self):
        text = (
            "<function=web_search><parameter=query>weather london</parameter></function>"
            " Let me check that for you."
        )
        call = _only(text)
        assert call == {"name": "web_search", "arguments": {"query": "weather london"}}

    def test_closed_function_with_trailing_whitespace_is_accepted(self):
        text = "<function=web_search><parameter=query>cats</parameter></function>   \n\n"
        call = _only(text)
        assert call == {"name": "web_search", "arguments": {"query": "cats"}}

    def test_closed_function_without_trailing_text_still_parses(self):
        text = "<function=web_search><parameter=query>cats</parameter></function>"
        call = _only(text)
        assert call == {"name": "web_search", "arguments": {"query": "cats"}}

    def test_multi_param_with_trailing_prose(self):
        text = (
            "<function=terminal><parameter=command>ls -la</parameter>"
            "<parameter=workdir>home</parameter></function> running it now"
        )
        call = _only(text)
        assert call == {
            "name": "terminal",
            "arguments": {"command": "ls -la", "workdir": "home"},
        }

    def test_code_value_containing_literal_close_tag_is_preserved(self):
        # The real closing </function> is the last one; the literal inside
        # the code argument must survive (rfind, not the first match).
        text = (
            "<function=python><parameter=code>"
            'print("</function>")'
            "</parameter></function> all done"
        )
        call = _only(text)
        assert call == {"name": "python", "arguments": {"code": 'print("</function>")'}}

    def test_closed_function_with_trailing_prose_heal_path(self):
        # Regression: the heal / finalize path (allow_incomplete=True) used to fold
        # </parameter></function> and the trailing prose into the argument and drop
        # the prose from visible content. It must now match the strict path -- keep a
        # clean argument and leave the trailing prose outside the call span.
        text = "<function=web_search><parameter=query>cats</parameter></function> trailing words"
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        fn = calls[0]["function"]
        assert fn["name"] == "web_search"
        assert json.loads(fn["arguments"]) == {"query": "cats"}
        # The trailing prose sits outside the removed span, so it stays visible.
        from core.tool_healing import (
            parse_tool_calls_from_text as _parse_with_spans,
        )

        _calls, spans = _parse_with_spans(text, allow_incomplete = True, with_spans = True)
        out = text
        for s, e in sorted(spans, reverse = True):
            out = out[:s] + out[e:]
        assert out == " trailing words"

    def test_incomplete_function_without_close_is_still_rejected(self):
        text = "<function=web_search><parameter=query>weather london"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []

    def test_param_without_close_tag_is_rejected_in_strict_mode(self):
        # Closing </function> present, but the single parameter never closes.
        text = "<function=web_search><parameter=query>weather london</function>"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []


class TestParityWithJsonStyle:
    def test_json_tool_call_with_trailing_prose_is_accepted(self):
        text = (
            '<tool_call>{"name":"web_search","arguments":{"query":"weather london"}}</tool_call>'
            " Let me check that for you."
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"

    def test_function_and_json_styles_agree_on_trailing_text(self):
        q = "weather london"
        func = parse_tool_calls_from_text(
            f"<function=web_search><parameter=query>{q}</parameter></function> trailing",
            allow_incomplete = False,
        )
        js = parse_tool_calls_from_text(
            f'<tool_call>{{"name":"web_search","arguments":{{"query":"{q}"}}}}</tool_call> trailing',
            allow_incomplete = False,
        )
        assert len(func) == len(js) == 1
        assert json.loads(func[0]["function"]["arguments"]) == {"query": q}
        assert json.loads(js[0]["function"]["arguments"]) == {"query": q}


class TestGemmaNativeStyle:
    def test_closed_native_call_with_trailing_prose_is_accepted(self):
        text = (
            '<|tool_call>call:terminal{command:"ls -la",workdir:"."}<tool_call|>' " running it now"
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "terminal"
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "command": "ls -la",
            "workdir": ".",
        }

    def test_unclosed_native_call_requires_healing(self):
        text = '<|tool_call>call:terminal{command:"ls"}'
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "terminal"

    def test_hyphenated_native_argument_name_is_accepted(self):
        text = '<|tool_call>call:mcp__srv__create-issue{issue-title:"Bug report"}<tool_call|>'
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "mcp__srv__create-issue"
        assert json.loads(calls[0]["function"]["arguments"]) == {"issue-title": "Bug report"}

    def test_native_template_quotes_preserve_windows_path(self):
        text = r'<|tool_call>call:ls{path:<|"|>C:\Users\wasim\repo<|"|>}<tool_call|>'
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {"path": r"C:\Users\wasim\repo"}

    def test_bare_unquoted_string_values_are_accepted(self):
        # Gemma can emit enum/string args unquoted; bare JSON scalars stay typed.
        text = (
            "<|tool_call>call:get_weather{location:Tokyo,unit:celsius,days:3,live:true}<tool_call|>"
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "location": "Tokyo",
            "unit": "celsius",
            "days": 3,
            "live": True,
        }


class TestHealingPathUnaffected:
    def test_auto_heal_still_repairs_unclosed_function(self):
        text = "<function=web_search><parameter=query>cats"
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"

    def test_closed_function_call_keeps_trailing_prose_out_of_arguments(self):
        # allow_incomplete exists for truncated output; a call that DID close
        # must parse identically to strict mode, leaving prose after
        # </function> out of the last parameter and out of the removal span.
        from core.tool_healing import parse_tool_calls_from_text as parse_with_spans

        text = "<function=web_search><parameter=query>cats</parameter></function> trailing"
        calls, spans = parse_with_spans(text, allow_incomplete = True, with_spans = True)
        (call,) = calls
        assert json.loads(call["function"]["arguments"]) == {"query": "cats"}
        (span,) = spans
        assert text[span[0] : span[1]] == (
            "<function=web_search><parameter=query>cats</parameter></function>"
        )
