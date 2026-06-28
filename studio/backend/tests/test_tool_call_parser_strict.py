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


class TestLlama3PythonTagStrict:
    def test_closed_dot_call_is_accepted(self):
        text = '<|python_tag|>get_weather.call(location="Tokyo")'
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert json.loads(calls[0]["function"]["arguments"]) == {"location": "Tokyo"}

    def test_truncated_dot_call_is_rejected(self):
        # No closing paren (depth > 0 at EOF): truncated, reject in strict mode.
        text = '<|python_tag|>get_weather.call(location="Tokyo"'
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
        # Auto-Heal still recovers it.
        assert len(parse_tool_calls_from_text(text, allow_incomplete = True)) == 1


class TestMistralArrayStrict:
    def test_closed_array_is_accepted(self):
        text = '[TOOL_CALLS] [{"name":"web_search","arguments":{"q":"x"}}]'
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"

    def test_unclosed_array_is_rejected(self):
        # Missing the closing ]; strict mode must not heal it.
        text = '[TOOL_CALLS] [{"name":"web_search","arguments":{"q":"x"}}'
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
        # Auto-Heal still recovers the object by hand.
        assert len(parse_tool_calls_from_text(text, allow_incomplete = True)) == 1


class TestHealingPathUnaffected:
    def test_auto_heal_still_repairs_unclosed_function(self):
        text = "<function=web_search><parameter=query>cats"
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"


class TestGlmStrict:
    def test_closed_glm_call_is_accepted(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"

    def test_unclosed_glm_call_is_rejected(self):
        # No </tool_call> close: truncated, reject with Auto-Heal off.
        text = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
        assert len(parse_tool_calls_from_text(text, allow_incomplete = True)) == 1


class TestKimiStrict:
    _SB = "<|tool_calls_section_begin|>"
    _KB = "<|tool_call_begin|>"
    _AB = "<|tool_call_argument_begin|>"
    _KE = "<|tool_call_end|>"
    _SE = "<|tool_calls_section_end|>"

    def test_full_kimi_call_is_accepted(self):
        text = self._SB + self._KB + "functions.x:0" + self._AB + '{"a":1}' + self._KE + self._SE
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "x"

    def test_kimi_call_without_call_end_is_rejected(self):
        # Section closed but the call lacks <|tool_call_end|>: reject in strict.
        text = self._SB + self._KB + "functions.x:0" + self._AB + '{"a":1}' + self._SE
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
        assert len(parse_tool_calls_from_text(text, allow_incomplete = True)) == 1

    def test_kimi_without_section_end_is_rejected(self):
        # No <|tool_calls_section_end|>: truncated section, reject in strict.
        text = self._SB + self._KB + "functions.x:0" + self._AB + '{"a":1}' + self._KE
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
        assert len(parse_tool_calls_from_text(text, allow_incomplete = True)) == 1


class TestParserLinearity:
    """The Llama-3 ``.call`` kwargs and Mistral-array healing paths must stay
    linear: both formerly ran a regex per offset and blew up (tens of seconds)
    on a long truncated body reachable from the agentic loop."""

    def test_llama3_unterminated_call_arg_is_linear(self):
        import time

        text = '<|python_tag|>upload.call(data="' + "A" * 200_000  # no closing quote/paren
        t0 = time.perf_counter()
        parse_tool_calls_from_text(text, allow_incomplete = True)
        assert time.perf_counter() - t0 < 2.0

    def test_llama3_huge_wordrun_call_arg_is_linear(self):
        import time

        text = "<|python_tag|>upload.call(" + "a" * 200_000  # giant word run, no '='
        t0 = time.perf_counter()
        parse_tool_calls_from_text(text, allow_incomplete = True)
        assert time.perf_counter() - t0 < 2.0

    def test_mistral_unclosed_array_open_braces_is_linear(self):
        import time

        text = "[TOOL_CALLS] [" + "{" * 200_000  # unclosed array, all open braces
        t0 = time.perf_counter()
        parse_tool_calls_from_text(text, allow_incomplete = True)
        assert time.perf_counter() - t0 < 2.0

    def test_llama3_call_kwargs_still_parse(self):
        text = '<|python_tag|>do.call(s="hi 😀", n=42, f=1.5, b=true, z=null)'
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "s": "hi 😀",
            "n": 42,
            "f": 1.5,
            "b": True,
            "z": None,
        }

    def test_mistral_unclosed_array_recovers_top_level_objects(self):
        text = (
            '[TOOL_CALLS] [{"name":"a","arguments":{"k":1}},'
            '{"name":"b","arguments":{"j":2}}'  # missing closing ]
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert [c["function"]["name"] for c in calls] == ["a", "b"]
