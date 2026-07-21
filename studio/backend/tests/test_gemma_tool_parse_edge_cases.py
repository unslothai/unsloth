# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Gemma-native tool-call parsing edge cases: commas inside bare string values,
and markers inside another call's argument data staying data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tool_call_parser import (
    _gemma_parse_value,
    parse_tool_calls_from_text,
)
from core.tool_healing import strip_tool_call_markup


def _args(call: dict) -> dict:
    return json.loads(call["function"]["arguments"])


def test_bare_string_argument_with_comma_is_kept():
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:get_weather{location:New York, NY,unit:celsius}<tool_call|>"
    )
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "get_weather"
    assert _args(calls[0]) == {"location": "New York, NY", "unit": "celsius"}


def test_normal_multi_key_arguments_still_split():
    calls = parse_tool_calls_from_text(
        '<|tool_call>call:f{a:1,b:hello,c:"x,y"}<tool_call|>'
    )
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"a": 1, "b": "hello", "c": "x,y"}


def test_empty_bare_value_becomes_empty_string_not_dropped():
    # An empty bare value (``{query:}``) must serialise as ``""`` (``{"query":}`` is invalid JSON and dropped the call).
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:search{query:,unit:celsius}<tool_call|>"
    )
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"query": "", "unit": "celsius"}

    only = parse_tool_calls_from_text("<|tool_call>call:get{q:}<tool_call|>")
    assert len(only) == 1, only
    assert _args(only[0]) == {"q": ""}


def test_bare_value_with_timestamps_after_comma_is_kept():
    # A comma before digits-then-colon (timestamp/ratio) is value text, not a key.
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:remind{query:meet at 10:00, 11:00 tomorrow,priority:high}<tool_call|>"
    )
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {
        "query": "meet at 10:00, 11:00 tomorrow",
        "priority": "high",
    }


def test_wrapperless_bare_value_with_timestamps_after_comma_is_kept():
    # The wrapper-less Gemma form (no <|tool_call> markers) goes through the
    # _gemma_parse_stripped_body scanner and its _GEMMA_KEY_RE.
    calls = parse_tool_calls_from_text(
        "call:web_search{query:meet at 10:00, 11:00 tomorrow}"
    )
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "web_search"
    assert _args(calls[0]) == {"query": "meet at 10:00, 11:00 tomorrow"}


def test_marker_inside_json_argument_is_not_a_second_call():
    content = (
        '<tool_call>{"name":"python","arguments":{"code":'
        '"x = 1  # <|tool_call>call:terminal{command:ls}<tool_call|>"}}</tool_call>'
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["python"], calls


def test_two_separate_gemma_calls_both_parse():
    content = (
        "<|tool_call>call:a{x:1}<tool_call|> and <|tool_call>call:b{y:2}<tool_call|>"
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["a", "b"], calls
    assert _args(calls[0]) == {"x": 1}
    assert _args(calls[1]) == {"y": 2}


def test_mixed_format_calls_preserve_document_order():
    content = (
        "<|tool_call>call:create{path:a}<tool_call|> then "
        '<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>'
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["create", "read"], calls


def test_json_marker_inside_gemma_argument_is_not_a_second_call():
    content = (
        '<|tool_call>call:python{code:<|"|>'
        'print(<tool_call>{"name":"terminal","arguments":{"command":"ls"}}</tool_call>)'
        '<|"|>}<tool_call|>'
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["python"], calls


def test_nested_gemma_marker_in_unquoted_arg_does_not_run_inner_call():
    # An UNQUOTED Gemma value containing a literal marker: the marker is nested in the outer
    # candidate span, so it must not be promoted to a standalone `terminal` call (no tool call).
    content = "<|tool_call>call:python{code:<|tool_call>call:terminal{command:ls}<tool_call|>}<tool_call|>"
    calls = parse_tool_calls_from_text(content)
    assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_bare_string_array_argument_is_quoted():
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:label{labels:[bug,ui]}<tool_call|>"
    )
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"labels": ["bug", "ui"]}


def test_array_keeps_numbers_and_quoted_elements():
    calls = parse_tool_calls_from_text(
        '<|tool_call>call:f{nums:[1,2],tags:[<|"|>a,b<|"|>,c]}<tool_call|>'
    )
    assert _args(calls[0]) == {"nums": [1, 2], "tags": ["a,b", "c"]}


def test_array_of_objects_is_normalised():
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:batch{items:[{path:a,mode:r},{path:b,mode:w}]}<tool_call|>"
    )
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {
        "items": [{"path": "a", "mode": "r"}, {"path": "b", "mode": "w"}]
    }


def test_nested_array_elements_are_normalised():
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:grid{cells:[[a,b],[c,d]]}<tool_call|>"
    )
    assert _args(calls[0]) == {"cells": [["a", "b"], ["c", "d"]]}


def test_gemma_marker_inside_xml_parameter_is_not_a_second_call():
    content = (
        "<tool_call><function=python><parameter=code>"
        "x = 1  # <|tool_call>call:terminal{command:ls}<tool_call|>"
        "</parameter></function></tool_call>"
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["python"], calls
    assert "terminal" in _args(calls[0])["code"]


def test_json_marker_inside_xml_parameter_is_not_a_second_call():
    content = (
        "<tool_call><function=python><parameter=code>"
        'run(<tool_call>{"name":"terminal","arguments":{"command":"ls"}}</tool_call>)'
        "</parameter></function></tool_call>"
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["python"], calls


def test_unclosed_think_literal_inside_tool_argument_does_not_hide_later_call():
    # A literal <think> inside a completed call's arguments is argument data; both calls must parse.
    text = '[TOOL_CALLS]a{"x":"literal <think> marker"} b[ARGS]{"y":2}'
    calls = parse_tool_calls_from_text(text)
    assert [c["function"]["name"] for c in calls] == ["a", "b"], calls


def test_real_think_block_with_rehearsal_inside_still_skips_only_the_rehearsal():
    # A genuine reasoning block still hides its rehearsal while a real call after it parses.
    text = '<think>web_search[ARGS]{"q":"draft"}</think>real[ARGS]{"q":"go"}'
    calls = parse_tool_calls_from_text(text)
    assert [c["function"]["name"] for c in calls] == ["real"], calls


def test_wrapperless_nested_object_argument_is_parsed():
    # skip_special_tokens stream: wrapper and <|"|> markers stripped, so a nested object arrives bare.
    calls = parse_tool_calls_from_text("call:f{loc:{city:NYC},n:3}")
    assert len(calls) == 1
    assert _args(calls[0]) == {"loc": {"city": "NYC"}, "n": 3}


def test_wrapperless_array_argument_is_parsed():
    calls = parse_tool_calls_from_text("call:label{labels:[bug,ui],n:2}")
    assert len(calls) == 1
    assert _args(calls[0]) == {"labels": ["bug", "ui"], "n": 2}


def test_wrapperless_deeply_nested_object_and_array_are_preserved():
    # The single-pass parser must keep multi-level nesting (objects inside
    # objects, arrays inside arrays) intact, not flatten or drop it.
    calls = parse_tool_calls_from_text(
        "call:f{loc:{city:NYC,geo:{lat:1,lng:2}},tags:[a,b,[c,d]],n:3}"
    )
    assert len(calls) == 1
    assert _args(calls[0]) == {
        "loc": {"city": "NYC", "geo": {"lat": 1, "lng": 2}},
        "tags": ["a", "b", ["c", "d"]],
        "n": 3,
    }


def test_gemma_parse_array_advances_on_stray_brace():
    # Regression: a stray '}' / ']' / ',' where an array element is expected must
    # not stall _gemma_parse_value at the same index (it looped forever before).
    from core.inference.tool_call_parser import _gemma_parse_array

    items, end, closed = _gemma_parse_array("[a,}]", 0)
    assert end == 5 and closed is True  # consumed through the closing ']'
    assert items[0] == "a"


def test_gemma_close_marker_inside_quoted_arg_is_not_leaked_when_stripping():
    # Parse keeps the quoted close marker as data; strip removes the whole span.
    text = '<|tool_call>call:python{code:<|"|>print("<tool_call|>")<|"|>}<tool_call|>'
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"code": 'print("<tool_call|>")'}
    assert strip_tool_call_markup("before " + text + " after") == "before  after"
    assert (
        strip_tool_call_markup("before " + text + " after", final = True)
        == "before  after"
    )


def test_nested_xml_in_malformed_gemma_call_does_not_execute():
    # The failed Gemma candidate's span still covers its nested <function=>.
    text = (
        "<|tool_call>call:outer{code:<function=terminal><parameter=command>id"
        "</parameter></function></tool_call>, broken:{x}}<tool_call|>"
    )
    for allow_incomplete in (True, False):
        calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
        assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_unbalanced_gemma_call_with_xml_does_not_execute():
    # Unclosed braces cover to EOF, so the trailing <function=> is excluded.
    text = (
        "<|tool_call>call:outer{code:<function=terminal>"
        "<parameter=command>id</parameter></function>"
    )
    for allow_incomplete in (True, False):
        calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
        assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_standalone_function_xml_still_parses():
    text = "<function=terminal><parameter=command>id</parameter></function>"
    calls = parse_tool_calls_from_text(text)
    assert [c["function"]["name"] for c in calls] == ["terminal"], calls


def test_xml_between_braces_and_close_marker_does_not_execute():
    # Coverage runs to the close marker, so <function=> in the gap is data.
    text = (
        "<|tool_call>call:outer{broken:{x}}<function=terminal>"
        "<parameter=command>id</parameter></function><tool_call|>"
    )
    for allow_incomplete in (True, False):
        calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
        assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_balanced_inner_call_inside_unclosed_outer_does_not_execute():
    text = (
        "<|tool_call>call:outer{code:<|tool_call>call:terminal{command:id}<tool_call|>"
    )
    for allow_incomplete in (True, False):
        calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
        assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_strip_preserves_text_after_malformed_gemma_close():
    # Junk before the close is a malformed span: strip through it, keep the tail.
    text = "pre <|tool_call>call:t{a:1} note <tool_call|> post"
    assert strip_tool_call_markup(text) == "pre  post"
    assert strip_tool_call_markup(text, final = True) == "pre  post"


def test_malformed_closed_gemma_span_is_stripped():
    assert (
        strip_tool_call_markup('before <|tool_call>{"name":"x"}<tool_call|> after')
        == "before  after"
    )


def test_valid_call_after_missing_close_is_recovered():
    # A close-less call covers only its braces, so the later call is recovered.
    text = "<|tool_call>call:a{x:1} <|tool_call>call:b{y:2}<tool_call|>"
    names_inc = [
        c["function"]["name"]
        for c in parse_tool_calls_from_text(text, allow_incomplete = True)
    ]
    assert "b" in names_inc, names_inc
    names_strict = [
        c["function"]["name"]
        for c in parse_tool_calls_from_text(text, allow_incomplete = False)
    ]
    assert names_strict == ["b"], names_strict


def test_strip_non_final_keeps_incomplete_gemma_block():
    text = "before <|tool_call>call:t{"
    assert strip_tool_call_markup(text) == text
    assert strip_tool_call_markup(text, final = True) == "before"


def test_json_call_between_gemma_braces_and_close_does_not_execute():
    # A JSON call between the outer's braces and its close is covered data.
    text = (
        "<|tool_call>call:outer{broken:{x}}"
        '<tool_call>{"name":"terminal","arguments":{"command":"id"}}</tool_call>'
        "<tool_call|>"
    )
    for allow_incomplete in (True, False):
        calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
        assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_gemma_call_between_gemma_braces_and_close_does_not_execute():
    # Same escape with a Gemma-native inner marker.
    text = "<|tool_call>call:outer{broken:{x}}<|tool_call>call:terminal{command:id}<tool_call|><tool_call|>"
    for allow_incomplete in (True, False):
        calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
        assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_strip_final_keeps_text_after_closed_xml_with_inner_gemma_opener():
    # The to-EOF Gemma sweep must not eat visible text after </function>.
    text = 'before <function=python><parameter=code>print("<|tool_call>")</parameter></function> after'
    assert strip_tool_call_markup(text, final = True) == "before  after"
    assert strip_tool_call_markup(text) == "before  after"


def test_strip_final_keeps_text_after_closed_block_with_call_form_gemma_opener():
    # A call-form Gemma opener quoted in a closed block must not truncate it.
    xml = "<function=python><parameter=code><|tool_call>call:t{</parameter></function>"
    json_block = '<tool_call>{"name":"python","arguments":{"code":"<|tool_call>call:t{"}}</tool_call>'
    for block in (xml, json_block):
        text = "before " + block + " after"
        assert strip_tool_call_markup(text, final = True) == "before  after", block
        assert strip_tool_call_markup(text) == "before  after", block


def test_function_sibling_after_close_less_gemma_marker_is_recovered():
    # The close-less marker covers only its braces; the XML sibling is recovered.
    text = (
        "<|tool_call>call:bad{broken:{x}} "
        "<function=terminal><parameter=command>id</parameter></function>"
    )
    for allow_incomplete in (True, False):
        calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
        assert [c["function"]["name"] for c in calls] == ["terminal"], calls


def test_valid_call_after_close_less_marker_with_quoted_close_token_is_recovered():
    # A close token quoted in the later call must not extend the earlier
    # close-less marker's coverage over that call.
    gemma = '<|tool_call>call:a{x:1} <|tool_call>call:b{note:<|"|></tool_call><|"|>}<tool_call|>'
    names = [
        c["function"]["name"]
        for c in parse_tool_calls_from_text(gemma, allow_incomplete = False)
    ]
    assert names == ["b"], names
    json_text = (
        '<tool_call>{"name":"a","arguments":{}} '
        '<tool_call>{"name":"b","arguments":{"x":"</tool_call>"}}</tool_call>'
    )
    names_j = [
        c["function"]["name"]
        for c in parse_tool_calls_from_text(json_text, allow_incomplete = False)
    ]
    assert "b" in names_j, names_j


def test_gemma_parse_value_always_advances_on_stray_delimiter():
    # A stray delimiter (`,`, `}`, `]`) at the primitive position must still advance the
    # index by at least one, or a caller looping on it spins forever at 100% CPU (DoS).
    for delim in (",", "}", "]"):
        text = delim + "rest"
        value, nxt, _explicit = _gemma_parse_value(text, 0)
        assert nxt > 0, (delim, value, nxt)


def test_malformed_gemma_array_does_not_hang():
    # ``[},]`` puts a stray ``}`` at the primitive position inside a list body.
    # On the buggy parser this hangs the server; guard with a wall-clock timeout
    # so the regression fails loudly instead of blocking CI forever.
    import threading

    result: dict = {}

    def _run():
        result["calls"] = parse_tool_calls_from_text(
            "<|tool_call>call:f{a:[},]}<tool_call|>"
        )

    t = threading.Thread(target = _run, daemon = True)
    t.start()
    t.join(timeout = 10.0)
    assert not t.is_alive(), "parse_tool_calls_from_text hung on malformed array input"


def test_malformed_gemma_mapping_value_does_not_hang():
    # A stray ``}`` where a mapping value is expected must also terminate.
    import threading

    result: dict = {}

    def _run():
        result["calls"] = parse_tool_calls_from_text(
            "<|tool_call>call:f{a:}},b:1}<tool_call|>"
        )

    t = threading.Thread(target = _run, daemon = True)
    t.start()
    t.join(timeout = 10.0)
    assert (
        not t.is_alive()
    ), "parse_tool_calls_from_text hung on malformed mapping input"
