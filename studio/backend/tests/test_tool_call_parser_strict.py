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

    def test_wrapperless_fallback_calls_carry_spans(self):
        # The wrapperless function-XML fallback must report spans too, so
        # with_spans consumers (passthrough healing) strip exactly the promoted
        # markup: through </function> when closed, to the scanned end when healed.
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
    """Llama-3 ``<|python_tag|>NAME.call(...)`` built-in form: ``; ``-chaining and
    nested-tag isolation."""

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

    # Magistral emits reasoning as ``[THINK]...[/THINK]`` (bracket form, not the
    # ``<think>`` the reasoning channel renders). At end-of-turn it must be dropped
    # so it does not leak as raw content into the display / conversation history.
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
    # F1: the parser is dependency-light (external llama-server wrappers import it
    # standalone) and the package targets python >=3.9. Its PEP 604 ``X | None``
    # return annotations would raise TypeError on a 3.9 import without
    # ``from __future__ import annotations``; guard that the import stays present.
    from pathlib import Path
    src = (
        Path(__file__).resolve().parent.parent / "core" / "inference" / "tool_call_parser.py"
    ).read_text()
    assert "from __future__ import annotations" in src


def test_bare_json_function_alias_parses_and_strips_symmetrically():
    # The markerless bare-JSON parser accepts the "function" alias for the call name
    # (obj.get("name") or obj.get("function")). strip_leading_bare_json_call must
    # recognise the same alias so an executed {"function":...} call is not left as
    # raw content (parser/strip symmetry).
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
    """A well-formed [TOOL_CALLS] call whose JSON arguments quote tool XML must
    execute the OUTER Mistral call; the literal is argument data. The reverse
    direction (XML outer, [TOOL_CALLS] literal in its arguments) keeps the XML."""

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
    """The passthrough healer buffers only formats its parser can promote; a
    Mistral or Llama text call must stream through instead of being held until
    finalization and flushed as prose."""

    def test_heal_signals_subset_of_promotable_formats(self):
        from core.inference.passthrough_healing import _HEAL_SIGNALS
        assert set(_HEAL_SIGNALS) == {"<tool_call>", "<|tool_call>", "<function="}

    def test_stream_healer_does_not_hold_mistral_text(self):
        from core.inference.passthrough_healing import StreamToolCallHealer

        healer = StreamToolCallHealer(
            {"web_search"},
            [{"type": "function", "function": {"name": "web_search", "parameters": {}}}],
        )
        events = list(healer.feed('[TOOL_CALLS]web_search[ARGS]{"query":"cats"}'))
        text_out = "".join(v for k, v in events if k == "text")
        assert "[TOOL_CALLS]" in text_out  # streamed through, not buffered
        assert not list(healer.finalize()) or all(k == "text" for k, _v in healer.finalize())


class TestPythonTagLiteralInsideMistralArgs:
    """A python_tag LITERAL (spelled-out text, not the stripped special token)
    inside a leading Mistral call's arguments is data: the outer [TOOL_CALLS]
    call must execute, not the rehearsed inner literal."""

    def test_mistral_arg_quoting_python_tag_call(self):
        text = (
            '[TOOL_CALLS] [{"name": "web_search", "arguments": '
            '{"query": "what is <|python_tag|>evil.call(x=1)"}}]'
        )
        calls = parse_tool_calls_from_text(text)
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args["query"] == "what is <|python_tag|>evil.call(x=1)"


class TestBareJsonOuterOverXmlLiteral:
    """A leading bare-JSON call whose string arguments quote tool XML must
    execute the OUTER call (sibling of the Mistral-outer guard); XML before
    the JSON keeps the normal order."""

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
    """A tool call rehearsed inside a leading [THINK]...[/THINK] block (any
    format) is reasoning, not a call: the real call AFTER the think block must
    win, and parse must agree with the display strip that drops the block."""

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


class TestDisabledBareJsonLiteralNotPromoted:
    """When the leading bare-JSON object is ordinary content (its name is not
    an enabled tool), the tool literals quoted inside its strings are data:
    nothing inside the object may be promoted, while a real call AFTER the
    object still parses."""

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


class TestMistralLiteralInsideLeadingJson:
    """The Mistral trigger is foreign to a JSON envelope: a [TOOL_CALLS]
    literal quoted inside the leading object's strings must not be promoted
    over the outer call by the earlier Mistral parser."""

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
    """Sampling drift puts whitespace around ``call``/``:`` in wrapped Gemma
    calls; rejecting those in tool_healing lost the call entirely (no
    fallback re-parses the wrapped form)."""

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


class TestGemmaDottedArgumentKeys:
    """Gemma emits dotted argument keys (user.name:...) for namespaced
    schemas; the key-quoting scanner must accept dots like the parser's
    key/name charset, or json.loads fails and the whole call is lost."""

    def test_dotted_key_parses(self):
        text = (
            '<|tool_call>call:web_search{user.name:<|"|>bob<|"|>, query:<|"|>x<|"|>}<tool_call|>'
        )
        calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"]
        args = json.loads(calls[0]["function"]["arguments"])
        assert args == {"user.name": "bob", "query": "x"}
