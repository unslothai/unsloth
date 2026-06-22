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
        '<|tool_call>call:create{path:a}<tool_call|> then '
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
