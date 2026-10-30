# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A model can sample a call's arguments in near-JSON (single quotes, a trailing comma, True/None, unquoted
keys, raw newlines, an object encoded as a string). Healing reads the object it spells; anything ambiguous or
cut off is not guessed, and a malformed object is never passed whole as a string argument."""

from __future__ import annotations

import json

import pytest

from core.inference.tool_call_parser import parse_tool_calls_from_text
from core.inference.tool_loop_controller import UNPARSED_ARGUMENTS_KEY, coerce_tool_arguments

SEARCH_DOCS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                    "exact": {"type": "boolean"},
                },
                "required": ["query"],
            },
        },
    }
]
WANT = {"query": "rope scaling", "limit": 5, "exact": True}
ENABLED = {"web_search", "search_docs", "python"}


@pytest.mark.parametrize(
    "raw",
    [
        "{'query': 'rope scaling', 'limit': 5, 'exact': true}",
        '{"query": "rope scaling", "limit": 5, "exact": true,}',
        '{query: "rope scaling", limit: 5, exact: true}',
        '{"query": "rope scaling", "limit": 5, "exact": True}',
        json.dumps(json.dumps(WANT)),
        json.dumps(json.dumps(json.dumps(WANT))),
    ],
)
def test_near_json_arguments_heal_to_the_object(raw):
    coerced = coerce_tool_arguments(
        raw, heal = True, tool_name = "search_docs", tool_schemas = SEARCH_DOCS
    )
    assert coerced.arguments == WANT


@pytest.mark.parametrize(
    "raw",
    [
        '{"query": "rope scal',  # cut off
        '{"query": "the "best" rope"}',  # unescaped quote: ambiguous
        '{"query": "x", "mode": fast}',  # unknown bare word
        '{"query": NaN}',
    ],
)
def test_ambiguous_or_cut_off_arguments_are_not_turned_into_an_object(raw):
    coerced = coerce_tool_arguments(
        raw, heal = True, tool_name = "search_docs", tool_schemas = SEARCH_DOCS
    )
    assert coerced.arguments != WANT
    assert set(coerced.arguments) <= {"query", UNPARSED_ARGUMENTS_KEY}
    if "query" in coerced.arguments:
        assert coerced.arguments["query"] == raw  # unchanged fallback: the old single-string heal


def test_plain_string_argument_heal_is_unchanged():
    coerced = coerce_tool_arguments(
        "rope scaling", heal = True, tool_name = "search_docs", tool_schemas = SEARCH_DOCS
    )
    assert coerced.arguments == {"query": "rope scaling"}


def _parsed(text):
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)
    ]


CODE = "import json\nd = {'a': [1, 2]}\nprint(json.dumps(d))"


@pytest.mark.parametrize(
    "text, want",
    [
        (
            "<tool_call>\n{\"name\": \"search_docs\", \"arguments\": {'query': 'rope scaling', 'limit': 5, 'exact': true}}\n</tool_call>",
            WANT,
        ),
        (
            '<tool_call>\n{"name": "search_docs", "arguments": {"query": "rope scaling", "limit": 5, "exact": true},}\n</tool_call>',
            WANT,
        ),
        (
            '<tool_call>\n{name: "search_docs", arguments: {query: "rope scaling", limit: 5, exact: True}}\n</tool_call>',
            WANT,
        ),
        (
            '<tool_call>\n{"name": "python", "arguments": {"code": "'
            + CODE.replace("\n", "\n").replace('"', '\\"')
            + '"}}\n</tool_call>',
            {"code": CODE},
        ),
        (
            '<|python_tag|>{"name": "search_docs", "parameters": {"query": "rope scaling", "limit": 5, "exact": True}}',
            WANT,
        ),
        ("[TOOL_CALLS]search_docs[ARGS]{'query': 'rope scaling', 'limit': 5, 'exact': true}", WANT),
    ],
)
def test_wrapped_calls_with_near_json_parse(text, want):
    assert len(_parsed(text)) == 1
    assert _parsed(text)[0][1] == want


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "search_docs", "parameters": {\'query\': \'x\'}}',  # markerless bare JSON stays strict
        '<tool_call>\n{"name": "search_docs", "arguments": {\'query\': \'rope scal',  # truncated
        "Use {'name': 'search_docs'} in your prompt.",
    ],
)
def test_bare_truncated_and_prose_stay_text(text):
    assert _parsed(text) == []


def test_well_formed_call_is_identical():
    text = '<tool_call>\n{"name": "search_docs", "arguments": {"query": "rope scaling", "limit": 5, "exact": true}}\n</tool_call>'
    assert parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)[0]["function"][
        "arguments"
    ] == json.dumps(WANT)


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>\n{'name': 'search_docs', 'arguments': {'query': 'rope scaling'}}\n</tool_call>",
        '<tool_call>\n{"name": "search_docs", "arguments": {"query": "rope scaling",}}\n</tool_call>',
        "<|python_tag|>{'name': 'search_docs', 'parameters': {'query': 'rope scaling'}}",
        "[TOOL_CALLS]search_docs[ARGS]{'query': 'rope scaling'}",
    ],
)
def test_auto_heal_off_does_not_repair(text):
    # Auto-Heal off (allow_incomplete=False) is llama-server's strict path: near-JSON is not a call.
    assert len(_parsed(text)) == 1
    assert (
        parse_tool_calls_from_text(text, enabled_tool_names = ENABLED, allow_incomplete = False) == []
    )


def test_python_tag_unbalanced_braces_stay_linear():
    import time

    text = "<|python_tag|>" + "{" * 20000
    t0 = time.perf_counter()
    assert _parsed(text) == []
    # Brace-matching every ``{`` to the end was O(n^2): about 150 s here, against well under a second.
    assert time.perf_counter() - t0 < 10


_CODE = (
    '# }}\nexample = {name: "terminal", parameters: {command: "echo unintended"}}\n'
    '[TOOL_CALLS]terminal[ARGS]{command: "echo unintended"}'
)


@pytest.mark.parametrize(
    "text",
    [
        "<|python_tag|>{'name': 'python', 'parameters': {'code': '" + _CODE + "'}}",
        "[TOOL_CALLS][{'name': 'python', 'arguments': {'code': '" + _CODE + "'}}]",
        "[TOOL_CALLS]python[ARGS]{'code': '" + _CODE + "'}",
        "<tool_call>{'name':'python','arguments':{'code':'"
        + _CODE
        + '\n<tool_call>{name: "terminal", '
        'arguments: {command: "echo unintended"}}\'}}</tool_call>',
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>python<｜tool▁sep｜>{'code': '"
        + _CODE
        + "'}<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.python:0<|tool_call_argument_begin|>"
        "{'code': '" + _CODE + "'}<|tool_call_end|><|tool_calls_section_end|>",
    ],
)
def test_braces_in_a_single_quoted_argument_never_end_the_call(text):
    calls = parse_tool_calls_from_text(
        text, allow_incomplete = True, enabled_tool_names = ENABLED | {"terminal"}
    )
    assert [c["function"]["name"] for c in calls] == ["python"]
    assert json.loads(calls[0]["function"]["arguments"])["code"].startswith(_CODE)


@pytest.mark.parametrize(
    "text",
    [
        '[TOOL_CALLS]{name:"python", arguments:{code:\'[TOOL_CALLS]terminal[ARGS]'
        '{command:"echo unintended"}\'},unknown:bad}',
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>python<｜tool▁sep｜>{'code': 'function<｜tool▁sep｜>terminal"
        '\n```json\n{command: "echo unintended"}\n```<｜tool▁call▁end｜>\'}<｜tool▁call▁end｜><｜tool▁calls▁end｜>',
    ],
)
def test_near_json_nested_in_a_rejected_object_is_never_a_call(text):
    calls = parse_tool_calls_from_text(
        text, allow_incomplete = True, enabled_tool_names = ENABLED | {"terminal"}
    )
    assert "terminal" not in [c["function"]["name"] for c in calls]
