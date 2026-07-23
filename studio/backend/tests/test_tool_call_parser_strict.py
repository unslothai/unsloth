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
        # Regression: the heal path (allow_incomplete=True) must match the strict path --
        # keep a clean argument and leave trailing prose outside the call span.
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

    def test_attribute_form_literal_close_tag_is_preserved(self):
        # The attribute form <function name="..."> (MiniCPM-5 / MiniMax-M2) also ends at the
        # LAST </function>, so a literal close tag inside a code argument survives.
        text = (
            '<function name="python"><param name="code">'
            'print("</function>")'
            "</param></function> all done"
        )
        call = _only(text)
        assert call == {"name": "python", "arguments": {"code": 'print("</function>")'}}

    def test_closed_zero_param_attribute_call_is_accepted_in_strict_mode(self):
        # A closed call with no parameters is a valid zero-argument call; strict
        # mode must not treat the empty parameter list as a truncated call.
        assert _only('<function name="ping"></function>') == {"name": "ping", "arguments": {}}
        # A no-arg call that never closes is still rejected as truncated.
        assert parse_tool_calls_from_text('<function name="ping">', allow_incomplete = False) == []


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

    def test_closed_function_call_keeps_trailing_prose_out_of_arguments(self):
        # A call that DID close must parse identically to strict mode, leaving prose after
        # </function> out of the last parameter and the removal span.
        from core.tool_healing import parse_tool_calls_from_text as parse_with_spans

        text = "<function=web_search><parameter=query>cats</parameter></function> trailing"
        calls, spans = parse_with_spans(text, allow_incomplete = True, with_spans = True)
        (call,) = calls
        assert json.loads(call["function"]["arguments"]) == {"query": "cats"}
        (span,) = spans
        assert text[span[0] : span[1]] == (
            "<function=web_search><parameter=query>cats</parameter></function>"
        )

    def test_wrapperless_fallback_calls_carry_spans(self):
        # The wrapperless function-XML fallback must report spans too, so with_spans
        # consumers strip exactly the promoted markup (through </function> when closed).
        from core.tool_healing import parse_tool_calls_from_text as parse_with_spans

        closed = "before <function=web_search><parameter=query>cats</parameter></function> after"
        calls, spans = parse_with_spans(closed, allow_incomplete = True, with_spans = True)
        (call,) = calls
        assert json.loads(call["function"]["arguments"]) == {"query": "cats"}
        (span,) = spans
        assert closed[span[0] : span[1]] == (
            "<function=web_search><parameter=query>cats</parameter></function>"
        )

        healed = "x <function=web_search><parameter=query>dogs"
        calls, spans = parse_with_spans(healed, allow_incomplete = True, with_spans = True)
        (call,) = calls
        assert json.loads(call["function"]["arguments"]) == {"query": "dogs"}
        (span,) = spans
        assert healed[span[0] : span[1]] == "<function=web_search><parameter=query>dogs"


class TestEnabledToolNameGate:
    """``enabled_tool_names`` disambiguates the ambiguous bare-rehearsal
    ``NAME[ARGS]{json}`` form (#5704): NAME is a call only when it is an active tool,
    otherwise it is prose. ``None`` (the default) keeps the legacy unrestricted parse
    so existing callers are unaffected."""

    def _names(self, calls):
        return [c["function"]["name"] for c in calls]

    def test_inactive_rehearsal_before_active_call_does_not_swallow_it(self):
        # P1: an inactive ``foo[ARGS]{...}`` before a real call must not consume the real call.
        text = 'foo[ARGS]{"a":1} web_search[ARGS]{"query":"cats"}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert self._names(calls) == ["web_search"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}

    def test_inactive_rehearsal_alone_is_not_a_call(self):
        text = 'foo[ARGS]{"a":1}'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []

    def test_active_rehearsal_is_still_parsed(self):
        text = 'web_search[ARGS]{"query":"cats"}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert self._names(calls) == ["web_search"]

    def test_unrestricted_gate_none_preserves_legacy_behavior(self):
        # Without a gate every ``NAME[ARGS]{...}`` is parsed, as before the gate landed.
        text = 'foo[ARGS]{"a":1} web_search[ARGS]{"query":"cats"}'
        assert self._names(parse_tool_calls_from_text(text)) == ["foo", "web_search"]
        assert self._names(parse_tool_calls_from_text(text, enabled_tool_names = None)) == [
            "foo",
            "web_search",
        ]


class TestBracketCallSpans:
    """with_spans tiling for Mistral bracket calls: promoted markup strips
    exactly once, filtered calls' bytes stay visible, closers strip too."""

    def test_mixed_array_filtered_first_keeps_its_bytes_only(self):
        from core.inference.passthrough_healing import heal_openai_message_events

        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        content = (
            '[TOOL_CALLS][{"name":"bad","arguments":{"x":1}},'
            '{"name":"lookup","arguments":{"q":"cats"}}]'
        )
        events = heal_openai_message_events(
            {"role": "assistant", "content": content}, {"lookup"}, tools
        )
        kinds = [k for k, _v in events]
        assert kinds == ["text", "tool_call"]
        text = events[0][1]
        assert '"bad"' in text
        # The promoted call's markup must not survive in the text event.
        assert '"lookup"' not in text

    def test_mixed_array_filtered_second_stays_visible(self):
        from core.inference.passthrough_healing import heal_openai_message_events

        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        content = (
            '[TOOL_CALLS][{"name":"lookup","arguments":{"q":"cats"}},'
            '{"name":"bad","arguments":{"x":1}}]'
        )
        events = heal_openai_message_events(
            {"role": "assistant", "content": content}, {"lookup"}, tools
        )
        assert events[0][0] == "tool_call"
        trailing = "".join(v for k, v in events if k == "text")
        assert '"bad"' in trailing

    def test_v11_closer_inside_span(self):
        from core.tool_healing import parse_tool_calls_from_text as parse_with_spans

        text = '[TOOL_CALLS]web_search[ARGS]{"query":"cats"}[/TOOL_CALLS] after'
        calls, spans = parse_with_spans(text, allow_incomplete = True, with_spans = True)
        (call,) = calls
        assert call["function"]["name"] == "web_search"
        (span,) = spans
        assert text[span[0] : span[1]].endswith("[/TOOL_CALLS]")
        assert text[span[1] :] == " after"

    def test_fully_promoted_array_strips_whole_region(self):
        from core.inference.passthrough_healing import heal_openai_message_events

        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
        content = (
            '[TOOL_CALLS][{"name":"lookup","arguments":{"q":"a"}},'
            '{"name":"lookup","arguments":{"q":"b"}}] after'
        )
        events = heal_openai_message_events(
            {"role": "assistant", "content": content}, {"lookup"}, tools
        )
        assert [k for k, _v in events] == ["tool_call", "tool_call", "text"]
        assert events[2][1] == " after"


class TestMistralArrayHealing:
    """Draining the whole [TOOL_CALLS] array for the shapes the repo's own
    Mistral/Ollama templates emit."""

    def test_comma_less_multi_call_array_parses_all_calls(self):
        # ollama_template_mappers.py renders multi-call turns as [{...}{...}] with no
        # comma separator; a single json.loads of the body rejects it and dropped every
        # call. The element-by-element decode must recover all of them.
        text = '[TOOL_CALLS] [{"name":"a","arguments":{"x":1}}{"name":"b","arguments":{"y":2}}]'
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["a", "b"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
        assert json.loads(calls[1]["function"]["arguments"]) == {"y": 2}

    def test_comma_separated_and_single_arrays_still_parse(self):
        both = parse_tool_calls_from_text(
            '[TOOL_CALLS] [{"name":"a","arguments":{}},{"name":"b","arguments":{}}]'
        )
        assert [c["function"]["name"] for c in both] == ["a", "b"]
        one = parse_tool_calls_from_text('[TOOL_CALLS] [{"name":"a","arguments":{}}]')
        assert [c["function"]["name"] for c in one] == ["a"]

    def test_mistral_array_null_arguments_normalized_to_empty_object(self):
        # ``"arguments": null`` is a no-arg call; it must become {} (as the <tool_call>
        # path does), not the string "null" that auto-heal turns into {"query":"null"}.
        calls = parse_tool_calls_from_text('[TOOL_CALLS][{"name":"get_time","arguments":null}]')
        assert calls[0]["function"]["arguments"] == "{}"


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
    """Llama-3 ``.call`` kwargs and Mistral-array healing must stay linear (a regex-per-offset blew up on long truncated bodies)."""

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

    def test_gemma_wrapperless_deep_nesting_is_linear(self):
        # Wrapper-less Gemma ``call:f{a:{a:{...}}}`` deep nesting must parse in linear time (no quadratic re-scan).
        import time

        def nested(d):
            return "call:f{a:" + "{a:" * d + "x:1" + "}" * d + "}"

        def best_ms(depth):
            text = nested(depth)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                calls = parse_tool_calls_from_text(text)
                best = min(best, time.perf_counter() - t0)
            assert calls and json.loads(calls[0]["function"]["arguments"]), "nested args dropped"
            return best

        t200 = best_ms(200)
        t400 = best_ms(400)
        assert t400 < t200 * 3.0, (t200, t400)

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

    def test_llama3_call_scientific_notation_args_parse(self):
        # Scientific notation must decode as float (the old regex truncated 1e-3 -> 1).
        text = "<|python_tag|>calc.call(x=1e-3, y=-2E+4, z=0.5e2, n=42)"
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"x": 1e-3, "y": -2e4, "z": 50.0, "n": 42}
        assert isinstance(args["n"], int) and isinstance(args["x"], float)

    def test_mistral_unclosed_array_recovers_top_level_objects(self):
        text = (
            '[TOOL_CALLS] [{"name":"a","arguments":{"k":1}},'
            '{"name":"b","arguments":{"j":2}}'  # missing closing ]
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert [c["function"]["name"] for c in calls] == ["a", "b"]


class TestLlamaBuiltinChainAndNesting:
    """Llama-3 ``.call`` built-ins: ``; `` chaining and nested-tag isolation."""

    def test_semicolon_chained_builtin_calls_all_parse(self):
        # Only the first call is anchored to <|python_tag|>; the rest chain via ';'.
        text = "<|python_tag|>alpha.call(x=1); beta.call(y=2); gamma.call(z=3)"
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert [c["function"]["name"] for c in calls] == ["alpha", "beta", "gamma"]
        assert json.loads(calls[1]["function"]["arguments"]) == {"y": 2}

    def test_nested_python_tag_in_json_string_arg_is_not_a_call(self):
        # A code arg literally containing a <|python_tag|>...call(...) string: the real call is the
        # outer "python", not the nested "os" -- the scan stays anchored to the first tag.
        text = (
            '<|python_tag|>{"name":"python","parameters":'
            '{"code":"<|python_tag|>os.call(\'rm -rf /\')"}}'
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "python"
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == "<|python_tag|>os.call('rm -rf /')"

    def test_single_builtin_call_unchanged(self):
        text = '<|python_tag|>web_search.call(query="cats")'
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}


def test_glm_open_does_not_parse_spaced_prose_as_tool_name():
    # The GLM <tool_call>NAME opener must reject spaced literal prose (V10); only a
    # valid [\w.\-]+ name (followed by newline/<arg_key>/</tool_call>) is a call.
    assert parse_tool_calls_from_text("<tool_call>not a call</tool_call>") == []
    ok = parse_tool_calls_from_text(
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>NYC</arg_value>\n</tool_call>"
    )
    assert [c["function"]["name"] for c in ok] == ["get_weather"]


def test_deepseek_r1_missing_call_terminator_rejected_in_strict_mode():
    # R1 must reject a fenced call whose closing ``` + <｜tool▁call▁end｜> never
    # arrived when Auto-Heal is off, matching V3/V3.1 strictness (V6).
    text = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        "```json\n"
        '{"city":"NYC"}'
        "<｜tool▁calls▁end｜>"
    )
    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    assert len(parse_tool_calls_from_text(text, allow_incomplete = True)) == 1


def test_deepseek_r1_complete_call_accepted_in_strict_mode():
    # A fully-terminated R1 call (close fence + per-call end) is still accepted.
    text = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        "```json\n"
        '{"city":"NYC"}\n'
        "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    calls = parse_tool_calls_from_text(text, allow_incomplete = False)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather"


def test_strip_leading_bare_json_call_drops_complete_call():
    from core.inference.tool_call_parser import strip_leading_bare_json_call

    # A complete Llama-3.2 bare-JSON call is removed; trailing prose is kept.
    assert strip_leading_bare_json_call('{"name":"web_search","parameters":{"query":"cats"}}') == ""
    assert (
        strip_leading_bare_json_call('{"name":"python","parameters":{"code":"x"}} done') == "done"
    )


def test_strip_leading_bare_json_call_drops_truncated_call():
    from core.inference.tool_call_parser import strip_leading_bare_json_call

    # A truncated call (no closing brace) collapses to "" -- nothing recoverable.
    assert (
        strip_leading_bare_json_call('{"name":"web_search","parameters":{"query":"weather in S')
        == ""
    )


def test_strip_leading_bare_json_call_preserves_plain_json_and_prose():
    from core.inference.tool_call_parser import strip_leading_bare_json_call

    # No "name" key -> plain JSON answer, left untouched.
    assert (
        strip_leading_bare_json_call('{"result": 42, "ok": true}') == '{"result": 42, "ok": true}'
    )
    # Prose before the brace -> not a leading bare call, untouched.
    assert strip_leading_bare_json_call('here is {"name":"x"}') == 'here is {"name":"x"}'
    # Ordinary text untouched.
    assert strip_leading_bare_json_call("just a sentence.") == "just a sentence."


def test_glm_literal_close_tag_in_string_arg_not_truncated():
    import json

    from core.inference.tool_call_parser import parse_tool_calls_from_text

    # A GLM string argument may legitimately contain the literal close tag ``</tool_call>``.
    text = (
        "<tool_call>run_code\n"
        "<arg_key>code</arg_key>\n"
        '<arg_value>print("</tool_call>")</arg_value>\n'
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text, allow_incomplete = True)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["code"] == 'print("</tool_call>")', args


def test_glm_truncated_block_rejected_in_strict_mode_but_healed_otherwise():
    from core.inference.tool_call_parser import parse_tool_calls_from_text

    # No </tool_call> close: strict mode (Auto-Heal off) rejects the truncated
    # block; with Auto-Heal it keeps the partial call.
    text = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>NYC"
    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    healed = parse_tool_calls_from_text(text, allow_incomplete = True)
    assert len(healed) == 1 and healed[0]["function"]["name"] == "get_weather"


def test_truncated_wrapperless_gemma_call_is_stripped():
    from core.inference.tool_call_parser import strip_tool_markup

    # A wrapper-less Gemma ``call:NAME{...`` cut off mid-arguments (no closing
    # brace) must not leak the raw call into the visible stream.
    text = 'Sure!\ncall:web_search{"query": "weather in San Fr'
    stripped = strip_tool_markup(text, final = True)
    assert "call:web_search" not in stripped, repr(stripped)
    assert stripped.strip() == "Sure!"


def test_complete_wrapperless_gemma_call_keeps_trailing_prose():
    from core.inference.tool_call_parser import strip_tool_markup

    # The truncation pattern must run AFTER the closed form, so a complete call
    # followed by prose keeps the prose instead of eating to EOS.
    text = 'call:web_search{"query": "cats"} Here you go.'
    stripped = strip_tool_markup(text, final = True)
    assert "call:web_search" not in stripped
    assert stripped.strip() == "Here you go."


def test_bare_json_gated_on_enabled_tool_names():
    from core.inference.tool_call_parser import parse_tool_calls_from_text

    alice = '{"name":"Alice","parameters":{"age":30}}'
    real = '{"name":"web_search","parameters":{"query":"cats"}}'
    # With an enabled set, markerless JSON whose name is not a tool is NOT a call.
    assert parse_tool_calls_from_text(alice, enabled_tool_names = {"web_search"}) == []
    # A real call (enabled name) still parses.
    got = parse_tool_calls_from_text(real, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in got] == ["web_search"]
    # No enabled set (None) keeps the name-agnostic behaviour for direct callers.
    assert [c["function"]["name"] for c in parse_tool_calls_from_text(alice)] == ["Alice"]
    # Marker-based forms are NOT gated (an explicit signal is a real call attempt).
    xml = '<tool_call>{"name":"Alice","arguments":{}}</tool_call>'
    assert parse_tool_calls_from_text(xml, enabled_tool_names = {"web_search"})


def test_strip_leading_bare_json_call_gated_on_enabled_tool_names():
    from core.inference.tool_call_parser import strip_leading_bare_json_call

    alice = '{"name":"Alice","parameters":{"age":30}}'
    # Not an enabled tool -> ordinary JSON answer, kept verbatim.
    assert strip_leading_bare_json_call(alice, {"web_search"}) == alice
    # Enabled tool -> a real call, stripped (trailing prose kept).
    assert (
        strip_leading_bare_json_call(
            '{"name":"web_search","parameters":{"q":1}} hi', {"web_search"}
        )
        == "hi"
    )


def test_function_xml_strip_keeps_literal_close_tag_in_param_value():
    from core.inference.tool_call_parser import strip_tool_markup

    # The strip uses the LAST </function> (like the parser) so a literal </function> in a value doesn't
    # truncate it; separate calls still strip independently.
    text = '<function=python><parameter=code>print("</function>")</parameter></function> done'
    assert strip_tool_markup(text, final = True) == "done"
    two = (
        "a <function=f><parameter=x>1</parameter></function> mid "
        "<function=g><parameter=y>2</parameter></function> end"
    )
    assert strip_tool_markup(two, final = True) == "a  mid  end"


def test_function_xml_strip_keeps_trailing_text_after_literal_open_tag():
    from core.inference.tool_call_parser import parse_tool_calls_from_text, strip_tool_markup

    # A literal ``<function=x>`` opener inside a parameter value is data, not a call: the scan-based
    # strip keeps " done" (the old negative-lookahead regex ate the trailing prose).
    text = '<function=python><parameter=code>print("<function=x>")</parameter></function> done'
    assert parse_tool_calls_from_text(text)[0]["function"]["name"] == "python"
    assert strip_tool_markup(text, final = True) == "done"
    # Non-final (streaming) keeps an unclosed call buffered, does not eat prose early.
    open_text = 'pre <function=python><parameter=code>print("<function=x>")'
    assert strip_tool_markup(open_text, final = False) == open_text


def test_final_strip_removes_magistral_think_reasoning():
    from core.inference.tool_call_parser import strip_tool_markup

    # Magistral emits reasoning as ``[THINK]...[/THINK]`` (bracket form, not ``<think>``);
    # at end-of-turn it must be dropped so it doesn't leak into display / history.
    text = "[THINK]The user greeted me, I should say hi.[/THINK]Hello! How can I help?"
    assert strip_tool_markup(text, final = True) == "Hello! How can I help?"
    # A ``[TOOL_CALLS]`` living inside the reasoning goes with it.
    with_call = '[THINK]Maybe I should search.[/THINK][TOOL_CALLS]search{"q":"x"}'
    assert strip_tool_markup(with_call, final = True) == ""


def test_streaming_strip_keeps_magistral_think_buffered():
    from core.inference.tool_call_parser import strip_tool_markup

    # Mid-stream (final=False) the reasoning block is left intact; only the
    # end-of-turn pass removes it.
    text = "[THINK]still thinking"
    assert strip_tool_markup(text, final = False) == text


def test_final_strip_leaves_non_magistral_bracket_text_untouched():
    from core.inference.tool_call_parser import strip_tool_markup

    # Only a LEADING ``[THINK]`` block is reasoning; unrelated bracketed prose stays.
    text = "See [THINK about it] later"
    assert strip_tool_markup(text, final = True) == "See [THINK about it] later"


def test_strip_leading_bare_json_call_ignores_nested_name():
    from core.inference.tool_call_parser import strip_leading_bare_json_call

    # A nested ``"name"`` must NOT gate the strip (only a TOP-LEVEL enabled name is a call); the
    # ordinary JSON answer is kept verbatim, truncated or complete.
    nested_trunc = '{"result":{"name":"web_search","age":'
    nested_full = '{"result":{"name":"web_search","age":1}}'
    assert strip_leading_bare_json_call(nested_trunc, {"web_search"}) == nested_trunc
    assert strip_leading_bare_json_call(nested_full, {"web_search"}) == nested_full
    # A real top-level call (even with a top-level array before the name) still strips.
    assert (
        strip_leading_bare_json_call(
            '{"data":[1,2],"name":"web_search","parameters":{}}', {"web_search"}
        )
        == ""
    )


def test_mistral_single_object_call_is_stripped_for_display():
    from core.inference.tool_call_parser import (
        _strip_mistral_closed_calls,
        parse_tool_calls_from_text,
    )

    # The parser accepts the single-object [TOOL_CALLS]{...} shape, so the display
    # strip must remove it too (asymmetry would leak the raw object).
    text = '[TOOL_CALLS]{"name":"web_search","arguments":{"filters":{"date":"2024"}}} tail'
    assert [c["function"]["name"] for c in parse_tool_calls_from_text(text)] == ["web_search"]
    assert _strip_mistral_closed_calls(text) == " tail"
    # A literal [TOOL_CALLS] in prose (no following object) is left untouched.
    assert _strip_mistral_closed_calls("See the [TOOL_CALLS] docs") == "See the [TOOL_CALLS] docs"


def test_tool_call_parser_declares_future_annotations_for_py39_import():
    # F1: the parser is imported standalone on python >=3.9, where its PEP 604 ``X | None``
    # annotations need ``from __future__ import annotations``; guard that the import stays.
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "tool_call_parser.py"
    ).read_text()
    assert "from __future__ import annotations" in src


def test_glm_strip_treats_literal_close_tag_in_arg_value_as_data():
    # Core strip parity: a literal </tool_call> inside a GLM <arg_value> is argument data, so the whole call is stripped (no leaked tail).
    from core.inference.tool_call_parser import strip_tool_markup

    text = (
        "<tool_call>web_search\n<arg_key>query</arg_key>\n"
        "<arg_value>see </tool_call> tag</arg_value>\n</tool_call> tail"
    )
    assert strip_tool_markup(text, final = True) == "tail"
    calls = parse_tool_calls_from_text(text)
    assert [c["function"]["name"] for c in calls] == ["web_search"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "see </tool_call> tag"}


def test_bare_json_function_alias_parses_and_strips_symmetrically():
    # The bare-JSON parser accepts the "function" alias for the call name;
    # strip_leading_bare_json_call must recognise it too (parser/strip symmetry).
    from core.inference.tool_call_parser import (
        parse_tool_calls_from_text,
        strip_leading_bare_json_call,
        _top_level_bare_json_name,
    )

    enabled = {"web_search"}
    text = '{"function":"web_search","parameters":{"query":"cats"}}'
    calls = parse_tool_calls_from_text(text, enabled_tool_names = enabled)
    assert [c["function"]["name"] for c in calls] == ["web_search"]
    assert strip_leading_bare_json_call(text, enabled) == ""

    # "name" still takes precedence when both are present; nested aliases are data.
    assert _top_level_bare_json_name('{"function":"foo","name":"web_search"}') == "web_search"
    assert _top_level_bare_json_name('{"function":"web_search"}') == "web_search"
    assert _top_level_bare_json_name('{"result":{"function":"web_search"}}') is None
    # A non-enabled function-alias object is ordinary content and is preserved.
    assert (
        strip_leading_bare_json_call('{"function":"not_a_tool","parameters":{}}', enabled)
        == '{"function":"not_a_tool","parameters":{}}'
    )


class TestMistralOuterOverXmlLiteral:
    """Quoted tool XML inside a [TOOL_CALLS] call's arguments is data; the outer call executes. Reverse order keeps the XML."""

    def test_mistral_v11_arg_quoting_function_xml(self):
        text = (
            '[TOOL_CALLS]web_search[ARGS]{"query":"literal '
            '<function=evil><parameter=x>1</parameter></function>"}'
        )
        for strict in (True, False):
            calls = parse_tool_calls_from_text(text, allow_incomplete = not strict)
            assert [c["function"]["name"] for c in calls] == ["web_search"]
            assert "<function=evil>" in json.loads(calls[0]["function"]["arguments"])["query"]

    def test_mistral_array_arg_quoting_tool_call_json(self):
        text = (
            '[TOOL_CALLS][{"name":"web_search","arguments":{"query":'
            '"see <tool_call>{\\"name\\":\\"evil\\"}</tool_call>"}}]'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_xml_outer_keeps_winning_over_mistral_literal(self):
        text = (
            '<tool_call>{"name":"web_search","arguments":'
            '{"query":"docs say [TOOL_CALLS]evil[ARGS]{}"}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestHealerSignalAlignment:
    """The healer buffers only formats its shared parser can promote. Mistral's
    ``[TOOL_CALLS]`` is promotable (rescued), so it is a heal signal; the loop-only
    text-call markers (Llama ``<|python_tag|>``, bare ``[ARGS]``) are not, so they
    stream through instead of stalling as prose that never yields a call."""

    def test_heal_signals_subset_of_promotable_formats(self):
        from core.inference.passthrough_healing import _HEAL_SIGNALS
        assert set(_HEAL_SIGNALS) == {
            "<tool_call>",
            "<|tool_call>",
            "<function=",
            "[TOOL_CALLS]",
            "<|content_invoke_tool_json|>",
        }

    def test_stream_healer_does_not_hold_llama_python_tag_text(self):
        from core.inference.passthrough_healing import StreamToolCallHealer

        healer = StreamToolCallHealer(
            {"web_search"},
            [{"type": "function", "function": {"name": "web_search", "parameters": {}}}],
        )
        # Llama <|python_tag|> is not a healer-promotable format, so it streams through as text.
        events = list(healer.feed('<|python_tag|>web_search.call(query="cats")'))
        text_out = "".join(v for k, v in events if k == "text")
        assert "<|python_tag|>" in text_out  # streamed through, not buffered
        assert not list(healer.finalize()) or all(k == "text" for k, _v in healer.finalize())


class TestGemmaWrapperlessLiteralMarkers:
    """Wrapper-less Gemma calls whose ARGUMENTS mention Gemma's own markup.

    The tool_healing deferral must key on an actual wrapped opener
    (``<|tool_call>call:...``), not the wrapper literal anywhere in content:
    a query about the marker has nothing tool_healing can parse, and deferring
    it loses the call entirely (not executed AND stripped from display)."""

    def test_marker_literal_in_argument_still_parses(self):
        text = 'call:web_search{query:"what does <|tool_call> mean"}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "what does <|tool_call> mean"

    def test_real_wrapped_call_still_deferred_to_tool_healing(self):
        from core.inference.tool_call_parser import _parse_gemma_tool_calls

        # An actual wrapped opener present: the Gemma fallback must keep
        # deferring to the shared tool_healing parser that owns that form.
        text = '<|tool_call>call:web_search{query:<|"|>cats<|"|>}<tool_call|>'
        assert _parse_gemma_tool_calls(text, id_offset = 0) == []

    def test_single_quoted_brace_does_not_truncate_code(self):
        text = "call:python{code:print('}')}"
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"python"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == "print('}')"

    def test_single_quoted_brace_strip_span_covers_whole_call(self):
        from core.inference.tool_call_parser import strip_tool_markup

        text = "call:python{code:print('}')} Done."
        stripped = strip_tool_markup(text, final = True, enabled_tool_names = {"python"})
        assert "call:python" not in stripped
        assert "')}" not in stripped
        assert stripped.strip() == "Done."


class TestGlmEmbeddedClosePair:
    """A GLM value whose string literal embeds the full close-tag pair
    ``</arg_value></tool_call>`` (code documenting the GLM format) must not be
    truncated at the embedded pair: a structural close sits at balanced quote
    state, an embedded one is inside an open string literal."""

    def test_embedded_pair_inside_quoted_value_not_structural(self):
        text = (
            "<tool_call>python\n"
            "<arg_key>code</arg_key>\n"
            '<arg_value>print("</arg_value></tool_call>")\nx = 1</arg_value>\n'
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == 'print("</arg_value></tool_call>")\nx = 1'

    def test_strip_covers_the_full_call(self):
        from core.inference.tool_call_parser import strip_tool_markup

        text = (
            "<tool_call>python\n"
            "<arg_key>code</arg_key>\n"
            '<arg_value>print("</arg_value></tool_call>")\nx = 1</arg_value>\n'
            "</tool_call> Done."
        )
        stripped = strip_tool_markup(text, final = True)
        assert "arg_value" not in stripped
        assert stripped.strip() == "Done."

    def test_unbalanced_apostrophe_falls_back_to_first_candidate(self):
        # Prose-like value with an apostrophe: no candidate reaches balanced
        # quote state, so the first token-valid close wins (prior behavior).
        text = (
            "<tool_call>web_search\n"
            "<arg_key>query</arg_key>\n"
            "<arg_value>it's fine</arg_value>\n"
            "</tool_call>"
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "it's fine"


class TestPythonTagLiteralInsideMistralArgs:
    """A python_tag LITERAL inside a leading Mistral call's arguments is data; the outer call executes."""

    def test_mistral_arg_quoting_python_tag_call(self):
        text = (
            '[TOOL_CALLS] [{"name": "web_search", "arguments": '
            '{"query": "what is <|python_tag|>evil.call(x=1)"}}]'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "what is <|python_tag|>evil.call(x=1)"


class TestPythonTagOuterOverXmlLiteral:
    """A leading Llama-3 ``<|python_tag|>`` call owns the turn: tool XML/Mistral
    markup quoted in a ``.call(...)`` string argument (or in trailing prose) is
    data, so the outer call executes -- parity with the bare-JSON / Mistral /
    attribute-form leading-ownership rules. XML before the tag keeps normal order."""

    def test_call_arg_quoting_complete_function_xml(self):
        # A closed <function=...> in a .call() code arg must not beat the leading python_tag call.
        text = (
            '<|python_tag|>python.call(code="<function=render_html>'
            '<parameter=x>1</parameter></function>")'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["python"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == "<function=render_html><parameter=x>1</parameter></function>"

    def test_call_arg_quoting_bare_function_tag_in_query(self):
        # A query mentioning <function=...> must search, not execute a phantom tool.
        text = '<|python_tag|>web_search.call(query="how do I use <function=foo> in llama")'
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "how do I use <function=foo> in llama"

    def test_call_arg_quoting_tool_call_json(self):
        text = (
            "<|python_tag|>save_file.call(content="
            '"<tool_call>{\\"name\\": \\"delete\\", \\"arguments\\": {}}</tool_call>")'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["save_file"]

    def test_json_form_code_arg_quoting_function_xml(self):
        # JSON emission: a <function=...> in the code arg is data; the outer "python" call runs.
        text = (
            '<|python_tag|>{"name":"python","parameters":'
            '{"code":"<function=terminal>ls</function>"}}'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["python"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == "<function=terminal>ls</function>"

    def test_call_arg_quoting_mistral_trigger(self):
        text = '<|python_tag|>web_search.call(query="see [TOOL_CALLS]evil[ARGS]{}")'
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_leading_call_wins_over_trailing_xml(self):
        # A leading python_tag call owns the turn even when a real XML literal follows.
        text = (
            '<|python_tag|>web_search.call(query="cats") '
            "<function=evil><parameter=x>1</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_xml_before_python_tag_keeps_xml_order(self):
        # A foreign signal BEFORE the tag keeps normal document order (XML wins).
        text = (
            "<function=web_search><parameter=q>x</parameter></function> "
            '<|python_tag|>python.call(code="y")'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestBareJsonOuterOverXmlLiteral:
    """Quoted tool XML inside a leading bare-JSON call is data; XML before the JSON keeps normal order."""

    def test_bare_json_code_arg_quoting_function_xml(self):
        text = (
            '{"name": "python", "arguments": '
            '{"code": "run() # <function=terminal>ls</function>"}}'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"python"})
        assert [c["function"]["name"] for c in calls] == ["python"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == "run() # <function=terminal>ls</function>"

    def test_bare_json_outer_unrestricted_mode(self):
        text = '{"name": "python", "parameters": {"code": "<function=terminal>ls</function>"}}'
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["python"]

    def test_xml_before_json_keeps_xml_order(self):
        text = (
            "<function=web_search><parameter=query>cats</parameter></function>"
            ' {"name": "python", "arguments": {"code": "x"}}'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestMagistralThinkRehearsal:
    """A call rehearsed inside [THINK]...[/THINK] is reasoning; the real call after wins, and parse agrees with strip."""

    def test_function_xml_rehearsal_in_think_is_not_promoted(self):
        text = (
            '[THINK]I could emit <function=web_search>{"query":"x"}</function>'
            ' here[/THINK][TOOL_CALLS] [{"name":"terminal","arguments":{"cmd":"ls"}}]'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["terminal"]

    def test_hermes_rehearsal_in_think_is_not_promoted(self):
        text = (
            '[THINK]maybe <tool_call>{"name":"web_search","arguments":'
            '{"query":"x"}}</tool_call>[/THINK]'
            '[TOOL_CALLS] [{"name":"terminal","arguments":{"cmd":"ls"}}]'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["terminal"]

    def test_unclosed_think_parses_nothing(self):
        text = '[THINK]let me try <function=web_search>{"query":"x"}</function>'
        assert parse_tool_calls_from_text(text) == []


class TestGemmaUnquotedApostrophes:
    """Quotes open strings only at value-start context: an apostrophe inside
    an unquoted wrapper-less value (contractions, possessives) is prose, and
    treating it as an opener swallowed the closing brace and lost the call."""

    def test_contraction_in_unquoted_query_parses(self):
        text = "call:web_search{query:what's the weather}"
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "what's the weather"

    def test_contraction_does_not_swallow_next_key(self):
        text = "call:web_search{query:what's up, n:3}"
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "what's up"
        assert args["n"] == 3

    def test_contraction_strip_span_covers_whole_call(self):
        from core.inference.tool_call_parser import strip_tool_markup

        text = "call:web_search{query:what's the weather} Done."
        stripped = strip_tool_markup(text, final = True, enabled_tool_names = {"web_search"})
        assert "call:web_search" not in stripped
        assert stripped.strip() == "Done."

    def test_quoted_values_still_hide_delimiters(self):
        text = 'call:web_search{query:"weather, location: Boston", n:2}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "weather, location: Boston"
        assert args["n"] == 2


class TestGlmKeyWithoutValue:
    """A GLM <arg_key> with no <arg_value> tag: strict mode rejects the call
    (same contract as an unclosed value) instead of executing it with the
    argument silently dropped; Auto-Heal keeps the lenient skip."""

    def test_strict_rejects_key_without_value(self):
        text = "<tool_call>web_search\n<arg_key>query</arg_key>\n</tool_call>"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []

    def test_heal_keeps_the_lenient_skip(self):
        text = "<tool_call>web_search\n<arg_key>query</arg_key>\n</tool_call>"
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "web_search"
        assert json.loads(calls[0]["function"]["arguments"]) == {}


class TestDisabledBareJsonLiteralNotPromoted:
    """A leading non-enabled-name object is content: nothing inside promotes, and a call after it still parses."""

    def test_literal_inside_disabled_json_stays_data(self):
        text = (
            '{"name": "Alice", "note": "try <function=web_search>'
            '<parameter=query>x</parameter></function>"}'
        )
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []

    def test_python_tag_literal_inside_disabled_json_stays_data(self):
        text = '{"name": "Alice", "note": "<|python_tag|>web_search.call(query=1)"}'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []

    def test_real_call_after_disabled_json_still_parses(self):
        text = (
            '{"name": "Alice", "note": "<function=evil>x</function>"} '
            '<tool_call>{"name": "web_search", "arguments": {"query": "cats"}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestDeepSeekMarkerInsideLeadingEnvelopes:
    """A DeepSeek/Kimi marker quoted inside a leading bare-JSON or Mistral
    call's argument strings is data: the pre-pass must not promote the
    embedded no-arg literal and drop the real outer call."""

    def test_marker_inside_leading_json_call_stays_data(self):
        text = (
            '{"name": "web_search", "arguments": '
            '{"query": "what is <｜tool▁calls▁begin｜>...{}..."}}'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert "tool▁calls▁begin" in args["query"]

    def test_marker_inside_leading_mistral_call_stays_data(self):
        text = (
            '[TOOL_CALLS] [{"name": "web_search", "arguments": '
            '{"query": "docs on <｜tool▁calls▁begin｜> markers"}}]'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_standalone_deepseek_call_still_parses(self):
        text = (
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>web_search\n"
            '```json\n{"query": "cats"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestMistralLiteralInsideLeadingJson:
    """A [TOOL_CALLS] literal quoted inside a leading JSON object must not be promoted over it."""

    def test_outer_json_call_wins_over_mistral_literal(self):
        text = '{"name": "python", "arguments": {"code": "[TOOL_CALLS]web_search{}"}}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"python", "web_search"})
        assert [c["function"]["name"] for c in calls] == ["python"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["code"] == "[TOOL_CALLS]web_search{}"

    def test_disabled_outer_json_keeps_mistral_literal_as_data(self):
        text = '{"name": "Alice", "note": "[TOOL_CALLS]web_search{}"}'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []


class TestGemmaWrappedWhitespace:
    """Whitespace drift around ``call``/``:`` in wrapped Gemma calls must still parse (no fallback exists)."""

    def test_space_after_call_colon_parses(self):
        text = '<|tool_call>call: web_search{query:<|"|>cats<|"|>}<tool_call|>'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}

    def test_space_around_colon_parses(self):
        text = '<|tool_call>call : web_search{query:<|"|>cats<|"|>}<tool_call|>'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_strict_mode_still_requires_the_closing_tag(self):
        text = '<|tool_call>call: web_search{query:<|"|>cats<|"|>}'
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []


class TestDisabledJsonBeforeDeepSeekCall:
    """A disabled leading bare-JSON object whose strings mention a
    DeepSeek/Kimi marker is dropped and the tail parsed, so a REAL
    DeepSeek/Kimi call after the object still executes instead of the whole
    message skipping the pre-pass."""

    _DS = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>web_search\n"
        '```json\n{"query": "cats"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )

    def test_real_deepseek_call_after_disabled_json_parses(self):
        text = '{"name": "Alice", "note": "<｜tool▁calls▁begin｜>"} ' + self._DS
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}

    def test_disabled_json_with_marker_alone_stays_data(self):
        text = '{"name": "Alice", "note": "<｜tool▁calls▁begin｜>"}'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []


class TestGemmaDottedArgumentKeys:
    """Dotted Gemma keys (namespaced schemas) must survive key-quoting or the call is lost."""

    def test_dotted_key_parses(self):
        text = '<|tool_call>call:web_search{user.name:<|"|>bob<|"|>, query:<|"|>x<|"|>}<tool_call|>'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"user.name": "bob", "query": "x"}


class TestLeadingWrapperlessGemmaOverEmbeddedMarkers:
    """A leading wrapper-less Gemma call to an enabled tool owns the turn: a
    quoted foreign literal inside its argument (a query citing another tool
    syntax) is data, and tool_healing must not promote it before the Gemma
    fallback runs. Foreign markup leading keeps the normal order."""

    def test_leading_gemma_wins_over_quoted_xml_literal(self):
        text = (
            'call:web_search{query:"explain <tool_call>'
            '{"name":"evil","arguments":{}}</tool_call>"}'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_xml_leading_keeps_normal_order(self):
        text = (
            '<tool_call>{"name":"web_search","arguments":'
            '{"query":"call:evil{x:1} example"}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestLeadingMistralCallOwnsTheTurn:
    """A leading Mistral call wins in document order over literal XML in trailing prose."""

    def test_leading_mistral_wins_over_trailing_xml_literal(self):
        text = (
            '[TOOL_CALLS]web_search[ARGS]{"query":"cats"} '
            "Note: <function=evil><parameter=x>1</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_function_xml_leading_keeps_normal_order(self):
        text = (
            "<function=web_search><parameter=query>x</parameter></function> "
            "[TOOL_CALLS]evil[ARGS]{}"
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestGemmaDottedKeyAfterBareValue:
    def test_dotted_key_after_bare_value_is_a_boundary(self):
        text = "<|tool_call>call:web_search{query:foo,user.name:bob}<tool_call|>"
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"query": "foo", "user.name": "bob"}


class TestJsonAnswersAreDataForMarkerlessScans:
    """A whole-content JSON value is a structured answer: a quoted example of
    an enabled tool's syntax inside it must not execute the tool, and the
    display strip must not mutilate the answer."""

    def test_gemma_example_inside_json_answer_not_promoted(self):
        text = '{"answer":"Gemma syntax is call:web_search{query:hi}"}'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []

    def test_gemma_example_inside_json_answer_not_stripped(self):
        from core.inference.tool_call_parser import strip_tool_markup
        text = '{"answer":"Gemma syntax is call:web_search{query:hi}"}'
        assert strip_tool_markup(text, final = True, enabled_tool_names = {"web_search"}) == text

    def test_kimi_marker_inside_json_answer_not_promoted(self):
        text = (
            '{"answer":"<|tool_call_begin|>functions.web_search:0'
            '<|tool_call_argument_begin|>{}<|tool_call_end|>"}'
        )
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []


class TestGemmaNestedQuotedLeaves:
    def test_nested_object_and_array_values_are_unquoted(self):
        text = 'call:f{loc:{city:"New York"},items:["a","b"],n:3}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"f"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"loc": {"city": "New York"}, "items": ["a", "b"], "n": 3}


class TestEarliestEnvelopeWinsAcrossDeepSeekKimi:
    """The DeepSeek/Kimi pre-pass dispatches by earliest envelope opener: a
    leading real call wins over a trailing example of the sibling format in
    either direction (document order, like the other leading guards)."""

    _DS = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>evil\n"
        '```json\n{"x": 1}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )
    _KIMI = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"query": "cats"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )

    def test_leading_kimi_wins_over_trailing_deepseek_example(self):
        text = self._KIMI + " For reference: " + self._DS
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_leading_deepseek_wins_over_trailing_kimi_example(self):
        text = self._DS + " Kimi format: " + self._KIMI
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["evil"]


class TestNamelessLeadingJsonAnswerIsData:
    """A nameless leading JSON answer is an envelope: quoted markup stays data, and a call after it parses."""

    def test_xml_literal_inside_json_answer_stays_data(self):
        text = '{"answer": "use <function=web_search><parameter=query>x</parameter></function>"}'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []

    def test_real_call_after_json_answer_still_parses(self):
        text = (
            '{"answer": "docs"} <tool_call>{"name": "web_search", '
            '"arguments": {"query": "cats"}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestClosedCallPrecedesMarkerPrePass:
    """A closed non-DeepSeek/Kimi call that precedes the first DS/Kimi marker
    owns the turn: a trailing example (or an example quoted inside a wrapped
    Gemma argument) must not be promoted by the pre-pass."""

    _KIMI_EVIL = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.evil:0"
        '<|tool_call_argument_begin|>{"x": 1}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )

    def test_kimi_example_inside_wrapped_gemma_arg_stays_data(self):
        text = (
            '<|tool_call>call:web_search{query:<|"|>explain '
            + self._KIMI_EVIL
            + '<|"|>}<tool_call|>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_leading_xml_call_wins_over_trailing_kimi_example(self):
        text = (
            '<tool_call>{"name":"web_search","arguments":{"query":"cats"}}</tool_call>'
            " For reference: " + self._KIMI_EVIL
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_standalone_kimi_call_still_parses(self):
        calls = parse_tool_calls_from_text(self._KIMI_EVIL)
        assert [c["function"]["name"] for c in calls] == ["evil"]


class TestTruncatedWrapperlessGemmaStopsScan:
    def test_call_quoted_inside_truncated_arg_not_promoted(self):
        text = 'call:python{code:example("call:web_search{query:hi}") and then it cut'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"python", "web_search"}) == []


class TestGemmaQuotedNestedDelimiters:
    def test_comma_inside_quoted_nested_string_not_a_split(self):
        text = 'call:f{loc:{city:"New, York"},n:1}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"f"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"loc": {"city": "New, York"}, "n": 1}


class TestGemmaStringMarkerLiteralInArgs:
    def test_string_marker_literal_does_not_lose_the_call(self):
        text = "call:web_search{query:'what does <|\"|> mean in Gemma'}"
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == 'what does <|"|> mean in Gemma'


class TestGemmaMidValueQuotedPhrase:
    def test_quoted_phrase_mid_value_hides_delimiters(self):
        text = 'call:web_search{query:find "weather, location: Boston", limit:3}'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"query": 'find "weather, location: Boston"', "limit": 3}

    def test_apostrophes_still_prose_mid_value(self):
        text = "call:web_search{query:what's on at the museum, n:2}"
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"query": "what's on at the museum", "n": 2}


class TestGlmStrictRefusesInQuoteFallback:
    """A truncated GLM value whose only close candidates sit inside a string
    literal must reject in strict mode instead of executing truncated
    arguments; Auto-Heal keeps the lenient partial value."""

    _TRUNC = (
        '<tool_call>python\n<arg_key>code</arg_key>\n<arg_value>print("</arg_value></tool_call>")'
    )

    def test_strict_rejects_truncated_in_string_close(self):
        assert parse_tool_calls_from_text(self._TRUNC, allow_incomplete = False) == []

    def test_heal_keeps_partial_value(self):
        calls = parse_tool_calls_from_text(self._TRUNC, allow_incomplete = True)
        assert len(calls) == 1 and calls[0]["function"]["name"] == "python"


class TestGemmaGuardCoversPreambles:
    def test_preamble_then_gemma_call_quoting_xml_wins(self):
        text = (
            "Sure, searching now. call:web_search{query:"
            '"explain <tool_call>{"name":"evil","arguments":{}}</tool_call>"}'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestGlmStrictAcceptsApostrophes:
    def test_apostrophe_value_parses_in_strict_mode(self):
        text = (
            "<tool_call>web_search\n<arg_key>query</arg_key>\n"
            "<arg_value>what's the weather</arg_value>\n</tool_call>"
        )
        calls = parse_tool_calls_from_text(text, allow_incomplete = False)
        assert len(calls) == 1
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"query": "what's the weather"}


class TestDisabledGemmaCallLiteralsAreData:
    def test_literal_inside_disabled_call_not_promoted(self):
        text = 'call:foo{query:"<function=python><parameter=code>x</parameter></function>"}'
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"python", "web_search"}) == []

    def test_real_call_after_disabled_example_still_parses(self):
        text = (
            'call:foo{query:"<function=python><parameter=code>x</parameter></function>"}'
            " call:web_search{query:hi}"
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"python", "web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]


class TestLeadingJsonArrayAnswerIsData:
    def test_kimi_marker_inside_json_array_answer_not_promoted(self):
        text = (
            '[{"answer": "<|tool_call_begin|>functions.web_search:0'
            '<|tool_call_argument_begin|>{}<|tool_call_end|>"}]'
        )
        assert parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"}) == []


class TestLeadingBareJsonOwnsTurnOverTrailingXml:
    """Document order: a leading closed bare-JSON call owns the turn even when
    tool XML appears AFTER it (inside-or-after, mirroring the Mistral rule)."""

    def test_leading_call_wins_over_trailing_xml(self):
        text = (
            '{"name":"lookup","parameters":{"q":"first"}} Example: '
            '<tool_call>{"name":"delete_all","arguments":{}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"lookup", "delete_all"})
        assert [c["function"]["name"] for c in calls] == ["lookup"], calls
        assert json.loads(calls[0]["function"]["arguments"]) == {"q": "first"}

    def test_chained_leading_calls_win_over_trailing_xml(self):
        text = (
            '{"name":"lookup","parameters":{"q":"first"}};'
            '{"name":"lookup","parameters":{"q":"second"}} '
            '<tool_call>{"name":"delete_all","arguments":{}}</tool_call>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"lookup", "delete_all"})
        assert [c["function"]["name"] for c in calls] == ["lookup", "lookup"], calls

    def test_non_call_leading_object_defers_to_trailing_real_call(self):
        # Nameless answers and disabled-name objects take the decline path:
        # the object is dropped and the real trailing call still parses.
        for lead in ('{"answer": 42}', '{"name":"draft","parameters":{}}'):
            text = lead + ' <tool_call>{"name":"delete_all","arguments":{}}</tool_call>'
            calls = parse_tool_calls_from_text(text, enabled_tool_names = {"delete_all"})
            assert [c["function"]["name"] for c in calls] == ["delete_all"], (lead, calls)

    def test_leading_xml_call_still_wins_over_trailing_bare_json(self):
        text = (
            '<tool_call>{"name":"delete_all","arguments":{}}</tool_call> '
            'Example: {"name":"lookup","parameters":{"q":"x"}}'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"lookup", "delete_all"})
        assert [c["function"]["name"] for c in calls] == ["delete_all"], calls


class TestProseCloseTagAfterClosedFunctionCall:
    """A literal </function> in prose after a closed call is data: the call
    ends at its first close that is not parameter data, so arguments never
    swallow the prose between the real close and the literal."""

    def test_arguments_do_not_swallow_prose(self):
        text = (
            "<function=web_search><parameter=query>cats</parameter></function>"
            " Done. The tag </function> closes a call."
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"], calls
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}

    def test_literal_close_inside_open_parameter_stays_data(self):
        text = '<function=python><parameter=code>print("</function>")</parameter></function>'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"python"})
        assert [c["function"]["name"] for c in calls] == ["python"], calls
        assert json.loads(calls[0]["function"]["arguments"]) == {"code": 'print("</function>")'}

    def test_attribute_form_arguments_do_not_swallow_prose(self):
        # The <function name="..."> attribute form shares the first-balanced-close
        # rule: prose mentioning a literal close tag never folds into arguments.
        text = (
            '<function name="web_search"><parameter name="query">cats</parameter></function>'
            " Done. The tag </function> closes a call."
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"], calls
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "cats"}

    def test_attribute_form_literal_close_in_open_parameter_stays_data(self):
        text = '<function name="python"><parameter name="code">print("</function>")</parameter></function>'
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"python"})
        assert json.loads(calls[0]["function"]["arguments"]) == {"code": 'print("</function>")'}

    def test_attribute_form_two_calls_both_parse(self):
        text = (
            '<function name="web_search"><parameter name="query">cats</parameter></function>'
            '<function name="python"><parameter name="code">x=1</parameter></function>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "python"})
        assert [c["function"]["name"] for c in calls] == ["web_search", "python"], calls


class TestEnabledNameJsonAnswerIsContent:
    """A JSON answer whose top-level name matches an enabled tool but has no
    call shape is content: the parser rejects it, so the strip and the drain
    gate must keep it visible too."""

    def test_answer_survives_strip(self):
        from core.inference.tool_call_parser import strip_leading_bare_json_call
        ans = '{"name":"web_search","result":"no call"}'
        assert strip_leading_bare_json_call(ans, {"web_search"}) == ans

    def test_answer_does_not_route_to_draining(self):
        from core.inference.safetensors_agentic import _looks_like_enabled_bare_json
        assert not _looks_like_enabled_bare_json(
            '{"name":"web_search","result":"no call"}', {"web_search"}
        )

    def test_real_call_still_strips_and_drains(self):
        from core.inference.safetensors_agentic import _looks_like_enabled_bare_json
        from core.inference.tool_call_parser import strip_leading_bare_json_call

        real = '{"name":"web_search","parameters":{"q":"x"}}'
        assert strip_leading_bare_json_call(real, {"web_search"}) == ""
        assert _looks_like_enabled_bare_json(real, {"web_search"})

    def test_arguments_string_call_still_strips(self):
        from core.inference.tool_call_parser import strip_leading_bare_json_call
        call = '{"name":"web_search","arguments":"{\\"q\\":\\"x\\"}"} tail'
        assert strip_leading_bare_json_call(call, {"web_search"}) == "tail"


class TestAttributeFormLeadingContainment:
    """A leading attribute-form call owns the turn: markup quoted inside its
    parameter is data, not a call for the shared XML parser to promote."""

    def test_quoted_tool_call_inside_param_stays_data(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        text = (
            '<function name="web_search"><param name="query">find '
            '<tool_call>{"name":"delete","arguments":{}}</tool_call></param></function>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "delete"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        assert "delete" in json.loads(calls[0]["function"]["arguments"])["query"]

    def test_real_xml_call_before_attribute_form_keeps_order(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        text = (
            '<tool_call>{"name":"delete","arguments":{}}</tool_call> Example: '
            '<function name="web_search"><param name="q">x</param></function>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "delete"})
        assert calls[0]["function"]["name"] == "delete"


class TestParameterKeepsMultipleLiteralCloses:
    """A parameter that provably closes with its own tag keeps every literal
    function close inside it as data (regression: the first literal close was
    treated as ending the parameter, truncating the value)."""

    def test_two_literal_closes_in_one_parameter(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        text = (
            '<function name="web_search"><param name="query">'
            "a </function> b </function> c </param></function>"
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert json.loads(calls[0]["function"]["arguments"]) == {
            "query": "a </function> b </function> c"
        }

    def test_strip_removes_the_whole_call(self):
        from core.inference.tool_call_parser import strip_tool_markup
        text = (
            '<function name="web_search"><param name="query">'
            "a </function> b </function> c </param></function> after"
        )
        assert strip_tool_markup(text, final = True) == "after"

    def test_unclosed_parameter_still_heals_at_function_close(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text
        calls = parse_tool_calls_from_text(
            "<function=web_search><parameter=query>val</function>",
            enabled_tool_names = {"web_search"},
        )
        assert json.loads(calls[0]["function"]["arguments"]) == {"query": "val"}


class TestMistralPreambleOwnership:
    """A visible preface before the first Mistral call must not hand the turn
    to a later XML literal: the Mistral call is first in document order."""

    def test_v11_named_form_after_preface(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        text = (
            'pref [TOOL_CALLS]web_search[ARGS]{"query":"cats"} Note '
            "<function=evil><parameter=x>1</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_array_form_after_preface(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        text = (
            'pref [TOOL_CALLS][{"name":"web_search","arguments":{"query":"cats"}}] Note '
            "<function=evil><parameter=x>1</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]

    def test_xml_call_before_trigger_keeps_order(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        text = (
            "<function=evil><parameter=x>1</parameter></function> then "
            '[TOOL_CALLS][{"name":"web_search","arguments":{}}]'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "evil"})
        assert calls[0]["function"]["name"] == "evil"

    def test_prose_mention_without_call_shape_keeps_order(self):
        from core.inference.tool_call_parser import parse_tool_calls_from_text

        text = (
            "See [TOOL_CALLS] docs for details. "
            "<function=evil><parameter=x>1</parameter></function>"
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"evil"})
        assert [c["function"]["name"] for c in calls] == ["evil"]


class TestBareJsonStripRequiresTopLevelName:
    """The strip's shape gate requires the parser's TOP-LEVEL name in every
    mode: a JSON answer with only a nested name is content, even name-agnostic."""

    def test_nested_name_answer_survives_name_agnostic_strip(self):
        from core.inference.tool_call_parser import strip_leading_bare_json_call

        ans = '{"parameters":{},"result":{"name":"web_search"}}'
        assert strip_leading_bare_json_call(ans) == ans
        assert strip_leading_bare_json_call(ans, {"web_search"}) == ans

    def test_real_call_still_strips_name_agnostic(self):
        from core.inference.tool_call_parser import strip_leading_bare_json_call
        assert strip_leading_bare_json_call('{"name":"web_search","parameters":{"q":"x"}}') == ""


class TestGemmaAwareClosedBlockPrePass:
    """The closed JSON/function strip pre-pass must not delete across a complete
    Gemma span (a quoted <function=...> plus a later real </function>)."""

    def test_literal_function_in_gemma_arg_with_later_real_call(self):
        from core.tool_healing import strip_tool_call_markup
        text = (
            'before <|tool_call>call:python{code:<|"|>print("<function=x>")<|"|>}'
            "<tool_call|> <function=terminal><parameter=cmd>ls</parameter>"
            "</function> after"
        )
        assert strip_tool_call_markup(text, final = True) == "before   after"

    def test_literal_function_in_gemma_arg_with_prose_closer(self):
        from core.tool_healing import strip_tool_call_markup

        text = (
            'before <|tool_call>call:python{code:<|"|>print("<function=x>")<|"|>}'
            "<tool_call|> then use </function> to close. after"
        )
        out = strip_tool_call_markup(text, final = True)
        assert out.startswith("before")
        assert out.endswith("after")
        assert "call:python" not in out

    def test_gemma_opener_inside_json_arg_still_strips_block(self):
        from core.tool_healing import strip_tool_call_markup
        text = (
            '<tool_call>{"name":"t","arguments":{"code":"<|tool_call>call:x{"}}</tool_call> after'
        )
        assert strip_tool_call_markup(text, final = True) == "after"

    def test_gemma_opener_inside_function_param_still_strips_block(self):
        from core.tool_healing import strip_tool_call_markup
        text = (
            '<function=python><parameter=code>x = "<|tool_call>call:t{"</parameter>'
            "</function> after"
        )
        assert strip_tool_call_markup(text, final = True) == "after"
