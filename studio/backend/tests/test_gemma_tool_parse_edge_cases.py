# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Edge cases in Gemma-native tool-call parsing.

Covers two failure modes:
  1. A bare (unquoted) string argument that contains a comma, e.g.
     ``location:New York, NY`` -- the comma must not be treated as the next
     key boundary, or the whole call is dropped.
  2. A tool-call marker that appears INSIDE another call's argument string is
     data, not a real call, so it must not be promoted to a second tool call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.tool_call_parser import parse_tool_calls_from_text


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
    calls = parse_tool_calls_from_text('<|tool_call>call:f{a:1,b:hello,c:"x,y"}<tool_call|>')
    assert len(calls) == 1, calls
    # Numbers stay numeric, bare strings get quoted, an explicit quoted comma
    # stays inside its value.
    assert _args(calls[0]) == {"a": 1, "b": "hello", "c": "x,y"}


def test_empty_bare_value_becomes_empty_string_not_dropped():
    # An empty bare value (``{query:}``) must serialise as ``""`` so json.loads
    # sees ``{"query":""}``; emitting bare ``{"query":}`` is invalid JSON and
    # silently dropped the whole call.
    calls = parse_tool_calls_from_text("<|tool_call>call:search{query:,unit:celsius}<tool_call|>")
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"query": "", "unit": "celsius"}

    only = parse_tool_calls_from_text("<|tool_call>call:get{q:}<tool_call|>")
    assert len(only) == 1, only
    assert _args(only[0]) == {"q": ""}


def test_bare_value_with_timestamps_after_comma_is_kept():
    # A comma followed by digits-then-colon (a timestamp/ratio) is value text,
    # not a new key, so the whole query must be preserved as one argument.
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:remind{query:meet at 10:00, 11:00 tomorrow,priority:high}<tool_call|>"
    )
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"query": "meet at 10:00, 11:00 tomorrow", "priority": "high"}


def test_marker_inside_json_argument_is_not_a_second_call():
    # A python call whose `code` argument contains a Gemma marker string. The
    # marker is data and must not execute as a second `terminal` call.
    content = (
        '<tool_call>{"name":"python","arguments":{"code":'
        '"x = 1  # <|tool_call>call:terminal{command:ls}<tool_call|>"}}</tool_call>'
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["python"], calls


def test_two_separate_gemma_calls_both_parse():
    content = "<|tool_call>call:a{x:1}<tool_call|> and <|tool_call>call:b{y:2}<tool_call|>"
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["a", "b"], calls
    assert _args(calls[0]) == {"x": 1}
    assert _args(calls[1]) == {"y": 2}


def test_mixed_format_calls_preserve_document_order():
    # A Gemma-native call precedes a JSON-format call in the text; tools execute
    # in returned order, so `create` must come before `read`.
    content = (
        "<|tool_call>call:create{path:a}<tool_call|> then "
        '<tool_call>{"name":"read","arguments":{"path":"a"}}</tool_call>'
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["create", "read"], calls


def test_json_marker_inside_gemma_argument_is_not_a_second_call():
    # The reverse of the JSON-outer case: a JSON-style marker inside a Gemma
    # call's quoted argument is code text, not a second `terminal` call.
    content = (
        '<|tool_call>call:python{code:<|"|>'
        'print(<tool_call>{"name":"terminal","arguments":{"command":"ls"}}</tool_call>)'
        '<|"|>}<tool_call|>'
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["python"], calls


def test_nested_gemma_marker_in_unquoted_arg_does_not_run_inner_call():
    # An UNQUOTED Gemma value containing a literal marker: the outer object fails
    # to normalize (the inner braces/marker break the JSON), but the inner marker
    # is nested in the outer candidate span, so it must not be promoted to a
    # standalone `terminal` call. The safe outcome is no executed tool call.
    content = "<|tool_call>call:python{code:<|tool_call>call:terminal{command:ls}<tool_call|>}<tool_call|>"
    calls = parse_tool_calls_from_text(content)
    assert "terminal" not in [c["function"]["name"] for c in calls], calls


def test_bare_string_array_argument_is_quoted():
    # Gemma may emit an array of bare strings without per-element quotes; they
    # must be quoted so the call is not dropped.
    calls = parse_tool_calls_from_text("<|tool_call>call:label{labels:[bug,ui]}<tool_call|>")
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"labels": ["bug", "ui"]}


def test_array_keeps_numbers_and_quoted_elements():
    calls = parse_tool_calls_from_text(
        '<|tool_call>call:f{nums:[1,2],tags:[<|"|>a,b<|"|>,c]}<tool_call|>'
    )
    assert _args(calls[0]) == {"nums": [1, 2], "tags": ["a,b", "c"]}


def test_array_of_objects_is_normalised():
    # Arrays of objects are a common tool-schema shape; their (unquoted) keys and
    # bare values must be normalised too, not left verbatim, or the call drops.
    calls = parse_tool_calls_from_text(
        "<|tool_call>call:batch{items:[{path:a,mode:r},{path:b,mode:w}]}<tool_call|>"
    )
    assert len(calls) == 1, calls
    assert _args(calls[0]) == {"items": [{"path": "a", "mode": "r"}, {"path": "b", "mode": "w"}]}


def test_nested_array_elements_are_normalised():
    calls = parse_tool_calls_from_text("<|tool_call>call:grid{cells:[[a,b],[c,d]]}<tool_call|>")
    assert _args(calls[0]) == {"cells": [["a", "b"], ["c", "d"]]}


def test_gemma_marker_inside_xml_parameter_is_not_a_second_call():
    # An XML-style <function=...> call whose <parameter=code> value contains a
    # Gemma marker: the marker is the parameter's data, not a separate terminal
    # call, so only the python call must be returned.
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


def test_wrapperless_nested_object_argument_is_parsed():
    # skip_special_tokens stream: the <|tool_call> wrapper and <|"|> string
    # markers were stripped, so a nested object arrives bare. It must parse as a
    # nested dict, not the literal string "{city:NYC}".
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
