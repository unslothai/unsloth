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

    def test_attribute_form_literal_close_tag_is_preserved(self):
        # The attribute form <function name="..."> (MiniCPM-5 / MiniMax-M2) must
        # also end at the LAST </function>, so a literal close tag inside a code
        # argument survives instead of truncating the call.
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

    def test_gemma_wrapperless_deep_nesting_is_linear(self):
        # Wrapper-less Gemma ``call:f{a:{a:{...}}}`` formerly pre-scanned each
        # subtree with a balanced-brace walk and then re-parsed it, so doubling
        # the nesting depth ~quadrupled the time (O(n^2)). The single-pass parser
        # is ~linear, so 2x depth should be well under 3x time.
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
        # The numeric kwarg regex matched only the mantissa, so scientific notation
        # was truncated to its leading digits (1e-3 -> 1) and the call executed with
        # the wrong value. Exponent and decimal forms must decode as float.
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
    """Llama-3 ``<|python_tag|>NAME.call(...)`` built-in form: ``; ``-chaining and
    nested-tag isolation."""

    def test_semicolon_chained_builtin_calls_all_parse(self):
        # Only the first call is anchored to <|python_tag|>; the rest chain via ';'.
        text = "<|python_tag|>alpha.call(x=1); beta.call(y=2); gamma.call(z=3)"
        calls = parse_tool_calls_from_text(text, allow_incomplete = True)
        assert [c["function"]["name"] for c in calls] == ["alpha", "beta", "gamma"]
        assert json.loads(calls[1]["function"]["arguments"]) == {"y": 2}

    def test_nested_python_tag_in_json_string_arg_is_not_a_call(self):
        # The custom JSON form carries a code arg that literally contains a
        # <|python_tag|>...call(...) string. The real call is the outer "python",
        # not the nested "os" -- the built-in scan must stay anchored to the first
        # tag and let the JSON parser win.
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

    # A GLM string argument may legitimately contain the literal close tag
    # ``</tool_call>`` (e.g. code that prints it). Pre-bounding the body at the
    # first ``</tool_call>`` truncated the value; walking against the full content
    # keeps it because each <arg_value> is delimited by its own </arg_value>.
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

    # The parser uses the LAST </function>; the strip must too, so a literal
    # </function> inside a parameter value does not truncate the strip and leak the
    # tail. Separate calls must still be stripped independently.
    text = '<function=python><parameter=code>print("</function>")</parameter></function> done'
    assert strip_tool_markup(text, final = True) == "done"
    two = (
        "a <function=f><parameter=x>1</parameter></function> mid "
        "<function=g><parameter=y>2</parameter></function> end"
    )
    assert strip_tool_markup(two, final = True) == "a  mid  end"


def test_function_xml_strip_keeps_trailing_text_after_literal_open_tag():
    from core.inference.tool_call_parser import parse_tool_calls_from_text, strip_tool_markup

    # A literal ``<function=x>`` OPENER inside a parameter value is data, not a new
    # call (the parser ignores it via _inside_open_parameter). The strip must do the
    # same: a regex negative-lookahead stopped at the nested opener and the
    # unclosed-tail arm then ate the trailing prose. Scan-based strip keeps " done".
    text = '<function=python><parameter=code>print("<function=x>")</parameter></function> done'
    assert parse_tool_calls_from_text(text)[0]["function"]["name"] == "python"
    assert strip_tool_markup(text, final = True) == "done"
    # Non-final (streaming) keeps an unclosed call buffered, does not eat prose early.
    open_text = 'pre <function=python><parameter=code>print("<function=x>")'
    assert strip_tool_markup(open_text, final = False) == open_text


def test_strip_leading_bare_json_call_ignores_nested_name():
    from core.inference.tool_call_parser import strip_leading_bare_json_call

    # A nested ``"name"`` equal to an enabled tool must NOT gate the strip: the object
    # is an ordinary JSON answer, not a call. Both truncated and complete forms are
    # kept verbatim. Only a TOP-LEVEL enabled name is treated as a real call.
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
    # F1: the parser is dependency-light (external llama-server wrappers import it
    # standalone) and the package targets python >=3.9. Its PEP 604 ``X | None``
    # return annotations would raise TypeError on a 3.9 import without
    # ``from __future__ import annotations``; guard that the import stays present.
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "core" / "inference" / "tool_call_parser.py").read_text()
    assert "from __future__ import annotations" in src
