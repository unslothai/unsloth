# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-call parser edge cases for DeepSeek, GLM and Kimi: arg coercion, dotted names, truncation."""

import json

import pytest

from core.inference.tool_call_parser import (
    parse_tool_calls_from_text,
    strip_tool_markup,
)

from growth import assert_linear  # tests/_shared, on sys.path via tests/conftest.py


@pytest.mark.parametrize(
    "raw_val, expected_python",
    [
        ("42", 42),
        ("true", True),
        ("false", False),
        ("null", None),
        ("3.14", 3.14),
        ("-7", -7),
        ("1e3", 1000.0),
    ],
)
def test_glm_numeric_and_bool_literals_are_json_decoded(raw_val, expected_python):
    text = (
        "<tool_call>n\n"
        f"<arg_key>v</arg_key>\n"
        f"<arg_value>{raw_val}</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["v"] == expected_python


@pytest.mark.parametrize(
    "raw_val",
    [
        "hello world",
        "True",  # Python literal, not JSON
        "None",  # Python literal, not JSON
        "if x < 10: pass",
        "{not valid json",
        "[oops",
    ],
)
def test_glm_non_json_shapes_stay_raw(raw_val):
    text = (
        "<tool_call>n\n"
        f"<arg_key>v</arg_key>\n"
        f"<arg_value>{raw_val}</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["v"] == raw_val
    assert isinstance(args["v"], str)


def test_glm_json_object_arg_decoded():
    text = (
        "<tool_call>nest\n"
        "<arg_key>opts</arg_key>\n"
        '<arg_value>{"limit": 10}</arg_value>\n'
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["opts"] == {"limit": 10}


def test_glm_json_array_arg_decoded():
    text = (
        "<tool_call>nest\n"
        "<arg_key>ids</arg_key>\n"
        "<arg_value>[1, 2, 3]</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["ids"] == [1, 2, 3]


def test_glm_arg_value_with_literal_less_than():
    text = (
        "<tool_call>run\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>if x < 10: pass</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["code"] == "if x < 10: pass"


def test_glm_4_7_no_newlines_between_name_and_arg_key():
    """GLM 4.7 strips the ``\\n`` after the name (``{{- ... -}}`` in the
    template) so ``<arg_key>`` follows directly. Parser must accept both."""
    text = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>London</arg_value>"
        "<arg_key>units</arg_key><arg_value>celsius</arg_value>"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"city": "London", "units": "celsius"}


def test_glm_4_7_no_newlines_multi_call():
    """Back-to-back GLM 4.7 calls without intervening newlines."""
    text = (
        "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        "<tool_call>b<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "a"
    assert calls[1]["function"]["name"] == "b"


# Kimi K2 dotted name + bare counter (finding C in plan)


def test_kimi_dotted_namespace_keeps_full_dotted_name():
    # A dotted Kimi id keeps its FULL name; only the ``functions.`` prefix and ``:idx`` suffix drop (vLLM parity).
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.my.tool:0"
        "<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "my.tool"
    assert calls[0]["id"] == "functions.my.tool:0"


def test_kimi_two_sections_in_one_stream_both_parse():
    """Outer loop walks every ``<|tool_calls_section_begin|>...end|>``
    so vLLM / SGLang parity holds even on multi-section streams."""
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.a:0"
        '<|tool_call_argument_begin|>{"x":1}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
        " some prose between sections "
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.b:0"
        '<|tool_call_argument_begin|>{"y":2}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "a"
    assert calls[1]["function"]["name"] == "b"
    assert calls[0]["id"] == "functions.a:0"
    assert calls[1]["id"] == "functions.b:0"


def test_kimi_bare_counter_id_is_dropped():
    """Bare-digit id (``3``) is dropped (matches vLLM); SGLang infers
    name from schema, which we don't have at parse time."""
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>3"
        "<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(text)
    assert calls == []


def test_deepseek_v3_1_huge_truncated_body_is_linear():
    """Adversarial input: DeepSeek envelope with no JSON brace and a long
    body. A regex-based ``[^\\n<]+?`` name capture is O(N^2) here; the
    parser uses ``str.find`` on the sep marker so it stays linear."""
    build = lambda n: "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>fn<｜tool▁sep｜>" + "x" * n
    assert assert_linear(parse_tool_calls_from_text, build, "V3", 50_000) == []


def test_deepseek_r1_huge_fenceless_body_is_linear():
    """R1 detection used a greedy ``([^\\n]+)\\n```json`` regex that is O(N^2) on a
    fence-less body of repeated ``function<sep>`` tokens. The parser now scans with
    ``str.find``."""
    build = lambda n: "<｜tool▁calls▁begin｜>" + "function<｜tool▁sep｜>a" * n
    assert assert_linear(parse_tool_calls_from_text, build, "R1", 10_000) == []


def test_deepseek_r1_fenceless_body_with_one_distant_object_is_linear():
    """A fenceless DeepSeek R1 body with one distant object must parse in linear time, not quadratic."""
    build = lambda n: "<｜tool▁calls▁begin｜>" + "function<｜tool▁sep｜>a" * n + '{"a": 1}'
    assert_linear(parse_tool_calls_from_text, build, "R1 distant-object", 10_000)


def test_glm_unclosed_body_many_arg_keys_is_linear():
    """An unclosed GLM ``<tool_call>`` body runs to EOF; a lazy-group ``finditer``
    over many bare ``<arg_key>`` tokens was O(N^2). The parser now walks pairs with
    ``str.find``."""
    build = lambda n: "<tool_call>foo\n" + "<arg_key>k" * n
    assert_linear(parse_tool_calls_from_text, build, "GLM", 10_000)


def test_deepseek_r1_fenced_json_parses():
    """R1 wraps args in a ```json fence after ``function<sep>NAME``."""
    import json as _json

    text = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        "```json\n"
        '{"city":"NYC","unit":"c"}\n'
        "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _json.loads(calls[0]["function"]["arguments"]) == {"city": "NYC", "unit": "c"}


def test_deepseek_v3_1_truncated_arguments_drops_call_without_crash():
    text = (
        "<｜tool▁calls▁begin｜>" "<｜tool▁call▁begin｜>get_time" "<｜tool▁sep｜>" '{"city":"Tokyo"'
    )
    calls = parse_tool_calls_from_text(text)
    assert calls == []


def test_deepseek_v3_1_truncated_after_end_marker_still_yields_call():
    text = (
        "<｜tool▁calls▁begin｜>" "<｜tool▁call▁begin｜>get_time" "<｜tool▁sep｜>" '{"city":"Tokyo"}'
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_time"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            "before "
            "<｜tool▁calls▁begin｜>"
            "<｜tool▁call▁begin｜>get_time"
            '<｜tool▁sep｜>{"city":"Tokyo"}'
            "<｜tool▁call▁end｜>"
            "<｜tool▁calls▁end｜>"
            " after",
            id = "routes_layer_strip_removes_deepseek_envelope",
        ),
        pytest.param(
            "before "
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.web_search:0"
            '<|tool_call_argument_begin|>{"q":"x"}'
            "<|tool_call_end|>"
            "<|tool_calls_section_end|>"
            " after",
            id = "routes_layer_strip_removes_kimi_section",
        ),
        pytest.param(
            "before "
            "<tool_call>web_search\n"
            "<arg_key>q</arg_key>\n<arg_value>x</arg_value>\n"
            "</tool_call>"
            " after",
            id = "routes_layer_strip_removes_glm_block",
        ),
    ],
)
def test_routes_layer_strip_removes_tool_envelopes(text):
    from routes.inference import _strip_tool_xml as _routes_strip
    stripped = _routes_strip(text)
    assert stripped == "before  after"


def test_strip_tool_markup_handles_deepseek_envelope():
    text = (
        "before "
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        '<｜tool▁sep｜>{"city":"Tokyo"}'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
        " after"
    )
    stripped = strip_tool_markup(text, final = True)
    assert "before" in stripped and "after" in stripped
    assert "｜tool▁" not in stripped
    assert "get_time" not in stripped and "Tokyo" not in stripped


def test_strip_tool_markup_handles_kimi_section():
    text = (
        "before "
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"q":"x"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
        " after"
    )
    stripped = strip_tool_markup(text, final = True)
    assert "before" in stripped and "after" in stripped
    assert "tool_calls_section_begin" not in stripped


def test_glm_quoted_string_arg_keeps_its_quotes():
    text = (
        "<tool_call>web_search\n"
        "<arg_key>query</arg_key>\n"
        '<arg_value>"exact phrase"</arg_value>\n'
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["query"] == '"exact phrase"'


def test_glm_unclosed_arg_value_is_rejected_in_strict_mode():
    text = (
        "<tool_call>web_search\n"
        "<arg_key>query</arg_key>\n"
        "<arg_value>Tokyo weather"
        "</tool_call>"
    )
    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    healed = parse_tool_calls_from_text(text, allow_incomplete = True)
    assert len(healed) == 1
    args = json.loads(healed[0]["function"]["arguments"])
    assert "Tokyo weather" in args.get("query", "")


def test_deepseek_v3_missing_call_terminator_rejected_in_strict_mode():
    text = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        '<｜tool▁sep｜>{"city":"Tokyo"}'
        "<｜tool▁calls▁end｜>"
    )
    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    healed = parse_tool_calls_from_text(text, allow_incomplete = True)
    assert len(healed) == 1
    assert healed[0]["function"]["name"] == "get_time"


def test_deepseek_v3_with_call_terminator_parses_in_strict_mode():
    text = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        '<｜tool▁sep｜>{"city":"Tokyo"}'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )
    calls = parse_tool_calls_from_text(text, allow_incomplete = False)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_time"


def test_strip_tool_markup_removes_nested_wrapperless_gemma_call():
    text = "answer:\ncall:f{loc:{city:NYC},n:3} done"
    stripped = strip_tool_markup(text, final = True)
    assert "call:f" not in stripped
    assert "}" not in stripped
    assert "answer:" in stripped and "done" in stripped

    prose = "answer: call:f{loc:{city:NYC},n:3} done"
    assert strip_tool_markup(prose, final = True) == prose


def test_strip_tool_markup_non_final_removes_bare_kimi_call():
    text = (
        "before "
        "<|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"q":"x"}'
        "<|tool_call_end|>"
        " after"
    )
    stripped = strip_tool_markup(text, final = False)
    assert "tool_call_begin" not in stripped
    assert "tool_call_end" not in stripped
    assert "before" in stripped and "after" in stripped


def test_routes_layer_strip_removes_wrapperless_gemma_call():
    from routes.inference import _strip_tool_xml as _routes_strip

    text = 'before\ncall:web_search{query:"weather in Sydney"} after'
    stripped = _routes_strip(text)
    assert "call:web_search" not in stripped
    assert "before" in stripped and "after" in stripped

    prose = 'before call:web_search{query:"weather in Sydney"} after'
    assert _routes_strip(prose) == prose


def test_deepseek_envelope_end_inside_arg_string_is_not_a_truncation():
    content = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>web_search<｜tool▁sep｜>"
        '{"query":"what does <｜tool▁calls▁end｜> mean"}'
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    calls = parse_tool_calls_from_text(content)
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": "what does <｜tool▁calls▁end｜> mean"
    }


def test_glm_value_containing_literal_arg_value_close_is_preserved():
    content = (
        "<tool_call>run<arg_key>code</arg_key>"
        '<arg_value>print("</arg_value>")</arg_value></tool_call>'
    )
    calls = parse_tool_calls_from_text(content)
    assert len(calls) == 1, calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"code": 'print("</arg_value>")'}


def test_attribute_form_function_with_embedded_marker_runs_outer_call():
    content = (
        '<function name="respond"><parameter name="answer">'
        "The Kimi format is <|tool_call_begin|>functions.delete_all:0"
        "<|tool_call_argument_begin|>{}<|tool_call_end|>"
        "</parameter></function>"
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["respond"], calls


def test_wrapperless_gemma_call_gated_by_enabled_tools():
    # Without the <|tool_call> wrapper, call:NAME{...} is indistinguishable from prose.
    prose = "Here is an example of the syntax: call:foo{x:1}. That shows how tools work."
    assert parse_tool_calls_from_text(prose, enabled_tool_names = {"web_search"}) == []
    assert "call:foo{x:1}" in strip_tool_markup(
        prose, final = True, enabled_tool_names = {"web_search"}
    )
    real = "Answer.\ncall:web_search{query:hi}"
    calls = parse_tool_calls_from_text(real, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls
    assert "call:web_search" not in strip_tool_markup(
        real, final = True, enabled_tool_names = {"web_search"}
    )

    inline = "Answer. call:web_search{query:hi}"
    inline_calls = parse_tool_calls_from_text(inline, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in inline_calls] == ["web_search"], inline_calls
    assert strip_tool_markup(inline, final = True, enabled_tool_names = {"web_search"}) == inline


def test_kimi_section_end_inside_arg_string_is_not_a_truncation():
    content = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
        '{"q":"cats"}<|tool_call_end|>'
        "<|tool_call_begin|>functions.explain:1<|tool_call_argument_begin|>"
        '{"text":"the token <|tool_calls_section_end|> means end"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["search", "explain"], calls
    assert json.loads(calls[1]["function"]["arguments"]) == {
        "text": "the token <|tool_calls_section_end|> means end"
    }


def test_closed_envelope_before_deepseek_block_owns_turn():
    # Document order is the contract: a closed call before a DeepSeek/Kimi block owns the turn.
    deepseek = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>search_web\n"
        "```json\n"
        '{"query":"weather in Paris"}\n'
        "```"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    prose = (
        'A Qwen call looks like <tool_call>{"name":"example_tool","arguments":{}}</tool_call>.\n'
    )
    calls = parse_tool_calls_from_text(prose + deepseek)
    assert [c["function"]["name"] for c in calls] == ["example_tool"], calls

    kimi = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:0"
        '<|tool_call_argument_begin|>{"id":7}<|tool_call_end|><|tool_calls_section_end|>'
    )
    calls_k = parse_tool_calls_from_text("Example: <function=demo>{}</function> and now:\n" + kimi)
    assert [c["function"]["name"] for c in calls_k] == ["demo"], calls_k


def test_marker_inside_closed_outer_envelope_still_runs_outer_call():
    outer = (
        "<function=lookup><parameter=q>what does <｜tool▁calls▁begin｜> mean</parameter></function>"
    )
    calls = parse_tool_calls_from_text(outer)
    assert [c["function"]["name"] for c in calls] == ["lookup"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "q": "what does <｜tool▁calls▁begin｜> mean"
    }


def test_truncated_outer_envelope_with_embedded_marker_heals_outer_call():
    trunc = '<function=python><parameter=code>x = "<｜tool▁calls▁begin｜>sample"</parameter>'
    calls = parse_tool_calls_from_text(trunc)
    assert [c["function"]["name"] for c in calls] == ["python"], calls


def test_python_tag_call_with_embedded_marker_runs_outer_call():
    kimi = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.delete_all:0"
        "<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
    )
    deepseek = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>delete_all<｜tool▁sep｜>{}"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    for embedded in (kimi, deepseek):
        builtin = '<|python_tag|>web_search.call(query="explain ' + embedded + '")'
        calls = parse_tool_calls_from_text(builtin, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"], calls
        custom = (
            '<|python_tag|>{"name":"web_search","parameters":'
            '{"query":"explain ' + embedded + '"}}'
        )
        calls = parse_tool_calls_from_text(custom, enabled_tool_names = {"web_search"})
        assert [c["function"]["name"] for c in calls] == ["web_search"], calls

    prose = "The token <|python_tag|> is used. " + kimi
    calls = parse_tool_calls_from_text(prose)
    assert [c["function"]["name"] for c in calls] == ["delete_all"], calls


def test_gemma_wrapperless_quoted_value_with_comma_not_split():
    text = 'call:web_search{query:"weather, location: Boston", limit:3}'
    calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": "weather, location: Boston",
        "limit": 3,
    }


def test_literal_close_tag_in_xml_arg_before_marker_runs_outer_call():
    text = (
        '<function=python><parameter=code>x = "</function> '
        "<|tool_call_begin|>functions.delete_all:0<|tool_call_argument_begin|>{}"
        '<|tool_call_end|>"</parameter></function>'
    )
    calls = parse_tool_calls_from_text(text)
    assert [c["function"]["name"] for c in calls] == ["python"], calls


def test_literal_tool_call_close_in_qwen_json_before_marker_runs_outer_call():
    text = (
        '<tool_call>{"name":"search","arguments":{"query":"explain </tool_call> then '
        "<|tool_call_begin|>functions.delete_all:0<|tool_call_argument_begin|>{}"
        '<|tool_call_end|>"}}</tool_call>'
    )
    calls = parse_tool_calls_from_text(text)
    assert [c["function"]["name"] for c in calls] == ["search"], calls
    bb = (
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        '<tool_call>{"name":"b","arguments":{}}</tool_call>'
    )
    assert [c["function"]["name"] for c in parse_tool_calls_from_text(bb)] == ["a", "b"]


def test_r1_heal_keeps_later_call_when_first_omits_close_fence():
    text = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n"
        '{"city":"SF"}\n```'
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_time\n```json\n"
        '{"tz":"UTC"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )
    heal = [c["function"]["name"] for c in parse_tool_calls_from_text(text)]
    assert "get_time" in heal, heal
    strict = [
        c["function"]["name"] for c in parse_tool_calls_from_text(text, allow_incomplete = False)
    ]
    assert set(strict) <= set(heal), (strict, heal)


def test_wrapperless_gemma_nested_call_in_arg_is_not_a_second_call():
    text = 'call:web_search{query:"explain call:delete_all{target:files}"}'
    calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "delete_all"})
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": "explain call:delete_all{target:files}"
    }
    two = "call:web_search{query:hi}call:get_time{tz:UTC}"
    assert [
        c["function"]["name"]
        for c in parse_tool_calls_from_text(two, enabled_tool_names = {"web_search", "get_time"})
    ] == ["web_search", "get_time"]


def test_leading_bare_json_call_owns_quoted_gemma_snippet():
    text = (
        '{"name":"lookup","parameters":{"note":"use call:web_search{query:cats} for this"}}\n'
        "That is the call I would make."
    )
    calls = parse_tool_calls_from_text(text, enabled_tool_names = {"lookup", "web_search"})
    assert [c["function"]["name"] for c in calls] == ["lookup"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "note": "use call:web_search{query:cats} for this"
    }

    two = (
        '{"name":"lookup","parameters":{"note":"see call:web_search{query:cats}"}};'
        '{"name":"lookup","parameters":{"q":"second"}}'
    )
    calls_two = parse_tool_calls_from_text(two, enabled_tool_names = {"lookup", "web_search"})
    assert [c["function"]["name"] for c in calls_two] == ["lookup", "lookup"], calls_two


def test_leading_gemma_call_still_wins_over_trailing_json_example():
    text = 'call:web_search{query:cats} Example JSON: {"name":"demo_tool","parameters":{}}'
    calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "demo_tool"})
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls

    prose = "You can run call:web_search{query:cats} to search."
    calls_p = parse_tool_calls_from_text(prose, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in calls_p] == ["web_search"], calls_p


def test_leading_gemma_call_owns_quoted_mistral_trigger():
    text = 'call:web_search{query:"docs say [TOOL_CALLS]delete_all{}"}'
    calls = parse_tool_calls_from_text(text, enabled_tool_names = {"web_search", "delete_all"})
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": "docs say [TOOL_CALLS]delete_all{}"
    }

    real = '[TOOL_CALLS]delete_all{"x":1}'
    calls_m = parse_tool_calls_from_text(real, enabled_tool_names = {"web_search", "delete_all"})
    assert [c["function"]["name"] for c in calls_m] == ["delete_all"], calls_m

    mixed = (
        'Example: call:demo{note:"see [TOOL_CALLS]delete_all{}"}\n'
        '[TOOL_CALLS]web_search{"q":"real"}'
    )
    calls_d = parse_tool_calls_from_text(mixed, enabled_tool_names = {"web_search", "delete_all"})
    assert [c["function"]["name"] for c in calls_d] == ["web_search"], calls_d


def test_chained_bare_json_owns_kimi_marker_in_later_call():
    kimi = (
        "<|tool_call_begin|>functions.delete_all:0"
        "<|tool_call_argument_begin|>{}<|tool_call_end|>"
    )
    two = (
        '{"name":"lookup","parameters":{"q":"first"}};'
        '{"name":"lookup","parameters":{"note":"' + kimi + '"}}'
    )
    calls = parse_tool_calls_from_text(two, enabled_tool_names = {"lookup", "delete_all"})
    assert [c["function"]["name"] for c in calls] == ["lookup", "lookup"], calls

    real = "Let me check.\n<|tool_calls_section_begin|>" + kimi + "<|tool_calls_section_end|>"
    calls_k = parse_tool_calls_from_text(real, enabled_tool_names = {"lookup", "delete_all"})
    assert [c["function"]["name"] for c in calls_k] == ["delete_all"], calls_k

    mistral = '[TOOL_CALLS]lookup{"q":"first"} then example ' + kimi
    calls_m = parse_tool_calls_from_text(mistral, enabled_tool_names = {"lookup", "delete_all"})
    assert [c["function"]["name"] for c in calls_m] == ["lookup"], calls_m


def test_nested_gemma_values_keep_commas_and_parens():
    calls = parse_tool_calls_from_text(
        "call:web_search{opts:{code:print(1,2),lang:py}}", enabled_tool_names = {"web_search"}
    )
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "opts": {"code": "print(1,2)", "lang": "py"}
    }

    arr = parse_tool_calls_from_text(
        "call:web_search{opts:[1,2,{a:f(1,2)}]}", enabled_tool_names = {"web_search"}
    )
    assert json.loads(arr[0]["function"]["arguments"]) == {"opts": [1, 2, {"a": "f(1,2)"}]}

    prose_comma = parse_tool_calls_from_text(
        "call:web_search{opts:{note:hello, world}}", enabled_tool_names = {"web_search"}
    )
    assert json.loads(prose_comma[0]["function"]["arguments"]) == {"opts": {"note": "hello, world"}}

    quoted = parse_tool_calls_from_text(
        'call:web_search{opts:{q:say "a, b" now,n:3}}', enabled_tool_names = {"web_search"}
    )
    assert json.loads(quoted[0]["function"]["arguments"]) == {
        "opts": {"q": 'say "a, b" now', "n": 3}
    }

    nested_q = parse_tool_calls_from_text(
        'call:web_search{loc:{city:"New York"}}', enabled_tool_names = {"web_search"}
    )
    assert json.loads(nested_q[0]["function"]["arguments"]) == {"loc": {"city": "New York"}}
    multi = parse_tool_calls_from_text(
        "call:web_search{opts:{a:1,b:2},n:3}", enabled_tool_names = {"web_search"}
    )
    assert json.loads(multi[0]["function"]["arguments"]) == {"opts": {"a": 1, "b": 2}, "n": 3}
    trunc = parse_tool_calls_from_text(
        "call:web_search{opts:{code:print(1,2}}", enabled_tool_names = {"web_search"}
    )
    assert json.loads(trunc[0]["function"]["arguments"]) == {"opts": "{code:print(1,2}"}


def test_multi_gemma_calls_own_turn_over_signal_in_later_call():
    en = {"get_time", "web_search", "delete_all"}
    both = parse_tool_calls_from_text(
        'call:get_time{} call:web_search{query:"docs say [TOOL_CALLS]delete_all{}"}',
        enabled_tool_names = en,
    )
    assert [c["function"]["name"] for c in both] == ["get_time", "web_search"], both
    assert json.loads(both[1]["function"]["arguments"]) == {
        "query": "docs say [TOOL_CALLS]delete_all{}"
    }

    xml = parse_tool_calls_from_text(
        'call:get_time{} call:web_search{query:"see <tool_call>delete_all</tool_call>"}',
        enabled_tool_names = en,
    )
    assert [c["function"]["name"] for c in xml] == ["get_time", "web_search"], xml
    kimi = parse_tool_calls_from_text(
        'call:get_time{} call:web_search{query:"see <|tool_call_begin|>'
        'functions.delete_all:0<|tool_call_argument_begin|>{}<|tool_call_end|>"}',
        enabled_tool_names = en,
    )
    assert [c["function"]["name"] for c in kimi] == ["get_time", "web_search"], kimi

    prose = parse_tool_calls_from_text(
        "call:get_time{} Example: [TOOL_CALLS]delete_all{}", enabled_tool_names = en
    )
    assert [c["function"]["name"] for c in prose] == ["get_time"], prose


def test_multi_gemma_ownership_reverse_controls():
    en = {"get_time", "web_search", "delete_all"}
    mistral = parse_tool_calls_from_text(
        '[TOOL_CALLS][{"name":"delete_all","arguments":{}}] Example: call:web_search{query:cats}',
        enabled_tool_names = en,
    )
    assert [c["function"]["name"] for c in mistral] == ["delete_all"], mistral
    xml_first = parse_tool_calls_from_text(
        '<tool_call>{"name":"delete_all","arguments":{}}</tool_call> call:web_search{query:cats}',
        enabled_tool_names = en,
    )
    assert [c["function"]["name"] for c in xml_first] == ["delete_all"], xml_first
    agnostic = parse_tool_calls_from_text(
        'call:foo{} <tool_call>{"name":"delete_all","arguments":{}}</tool_call>'
    )
    assert [c["function"]["name"] for c in agnostic] == ["delete_all"], agnostic


def test_disabled_leading_bare_json_does_not_hide_later_marker_call():
    kimi = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"q":"cats"}<|tool_call_end|><|tool_calls_section_end|>'
    )
    calls = parse_tool_calls_from_text(
        '{"name":"draft","parameters":{}} ' + kimi, enabled_tool_names = {"web_search"}
    )
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": "cats"}

    deepseek = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>web_search\n"
        '```json\n{"q":"cats"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
    )
    calls_ds = parse_tool_calls_from_text(
        '{"name":"draft","parameters":{}} ' + deepseek, enabled_tool_names = {"web_search"}
    )
    assert [c["function"]["name"] for c in calls_ds] == ["web_search"], calls_ds


def test_disabled_leading_bare_json_ownership_controls():
    kimi_delete = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.delete_all:0"
        "<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>"
    )
    owns = parse_tool_calls_from_text(
        '{"name":"web_search","parameters":{"q":"first"}} ' + kimi_delete,
        enabled_tool_names = {"web_search", "delete_all"},
    )
    assert [c["function"]["name"] for c in owns] == ["web_search"], owns
    inside = parse_tool_calls_from_text(
        '{"name":"draft","parameters":{"note":"see <|tool_call_begin|>functions.delete_all:0'
        '<|tool_call_argument_begin|>{}<|tool_call_end|>"}}\nsome trailing prose',
        enabled_tool_names = {"web_search", "delete_all"},
    )
    assert inside == [], inside
    nameless = parse_tool_calls_from_text(
        '{"answer":42} ' + kimi_delete, enabled_tool_names = {"delete_all"}
    )
    assert [c["function"]["name"] for c in nameless] == ["delete_all"], nameless
    agnostic = parse_tool_calls_from_text('{"name":"draft","parameters":{}} ' + kimi_delete)
    assert [c["function"]["name"] for c in agnostic] == ["draft"], agnostic


def test_leading_json_answer_with_prose_keeps_quoted_gemma_snippet_as_data():
    obj = '{"summary":"use call:web_search{query:cats} to search"}\nHope that helps!'
    assert parse_tool_calls_from_text(obj, enabled_tool_names = {"web_search"}) == []
    arr = '["use call:web_search{query:cats} to search"]\nHope that helps!'
    assert parse_tool_calls_from_text(arr, enabled_tool_names = {"web_search"}) == []
    assert strip_tool_markup(obj, enabled_tool_names = {"web_search"}) == obj

    tail = '{"summary":"done"}\ncall:web_search{query:cats}'
    calls = parse_tool_calls_from_text(tail, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls

    not_json = "{not json} call:web_search{query:cats}"
    calls_nj = parse_tool_calls_from_text(not_json, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in calls_nj] == ["web_search"], calls_nj


def test_glm_heal_bounds_unclosed_value_at_tool_call_close():
    one = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>NYC</tool_call>"
    calls = parse_tool_calls_from_text(one, allow_incomplete = True)
    assert [c["function"]["name"] for c in calls] == ["get_weather"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "NYC"}

    two = one + "\nLet me check that for you."
    calls_two = parse_tool_calls_from_text(two, allow_incomplete = True)
    assert json.loads(calls_two[0]["function"]["arguments"]) == {"city": "NYC"}

    assert parse_tool_calls_from_text(one, allow_incomplete = False) == []

    eof = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>New York Ci"
    calls_eof = parse_tool_calls_from_text(eof, allow_incomplete = True)
    assert json.loads(calls_eof[0]["function"]["arguments"]) == {"city": "New York Ci"}
    lit = (
        "<tool_call>get_weather<arg_key>city</arg_key>"
        '<arg_value>print("</tool_call>")</arg_value></tool_call>'
    )
    calls_lit = parse_tool_calls_from_text(lit, allow_incomplete = True)
    assert json.loads(calls_lit[0]["function"]["arguments"]) == {"city": 'print("</tool_call>")'}


def test_prose_mentioning_ds_kimi_markers_survives_final_strip():
    from core.inference.tool_call_parser import strip_tool_markup

    for text in [
        "The Kimi marker <|tool_calls_section_begin|> starts a section.",
        "DeepSeek uses <｜tool▁calls▁begin｜> to open calls.",
        "See <|tool_call_begin|> in the docs.",
    ]:
        assert strip_tool_markup(text, final = True) == text

    truncated_kimi = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"q'
    )
    assert strip_tool_markup(truncated_kimi, final = True) == ""
    assert strip_tool_markup("prefix <|tool_calls_section_begin|>", final = True) == "prefix"
