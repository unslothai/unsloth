# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Small Qwen models writing Python into a Hermes call often leave the code's own double quotes unescaped
(Qwen3-0.6B at temperature 1.2: about a quarter of its python calls). Strict JSON and brace matching both
lose that call; the ``</tool_call>`` closer still bounds it, so a single string argument is recovered.
Anything that could be a different call is left alone."""

from __future__ import annotations

import json

import pytest

from core.inference.tool_call_parser import parse_tool_calls_from_text

ENABLED = {"python", "web_search"}


def _calls(text, **kw):
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in parse_tool_calls_from_text(text, enabled_tool_names = ENABLED, **kw)
    ]


# Verbatim shape from a real Qwen3-0.6B Q4_K_M sample.
REAL = (
    '\n\n<tool_call>\n{"name": "python", "arguments": {"code": "import json\\njson.dumps({\'a\': {\'b\': [1, '
    "{'c': '}{'}]}, 'msg': \"it's \\\"quoted\\\"\"})\"}}\n</tool_call>"
)


def test_real_sample_is_recovered_verbatim():
    # The model's own ``\\"`` beside bare quotes is the code's escape, so the recovered code is valid Python.
    code = "import json\njson.dumps({'a': {'b': [1, {'c': '}{'}]}, 'msg': \"it's \\\"quoted\\\"\"})"
    assert _calls(REAL) == [("python", {"code": code})]
    compile(code, "<tool>", "exec")


def test_escaped_quotes_without_bare_quotes_are_json_escapes():
    text = '<tool_call>\n{"name": "python", "arguments": {"code": "print(\\"a\\")\nprint(\\"b\\")"}}\n</tool_call>'
    assert _calls(text) == [("python", {"code": 'print("a")\nprint("b")'})]


@pytest.mark.parametrize("key", ["arguments", "parameters"])
def test_simple_unescaped_quotes(key):
    text = '<tool_call>\n{"name": "python", "' + key + '": {"code": "print("hi")"}}\n</tool_call>'
    assert _calls(text) == [("python", {"code": 'print("hi")'})]


@pytest.mark.parametrize(
    "text",
    [
        # a second argument may have been swallowed: the value's end is not knowable
        '<tool_call>\n{"name": "python", "arguments": {"code": "print("hi")", "timeout": "5"}}\n</tool_call>',
        # the same with a single-quoted or unquoted key, and an outer key after the arguments object
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "rm -rf "build"", \'dry_run\': "true"}}\n</tool_call>',
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "rm -rf "build"", dry_run: "true"}}\n</tool_call>',
        '<tool_call>\n{"name": "terminal", "arguments": {"command": "echo "a"}, "parameters": {"command": "rm -rf ~"}}\n</tool_call>',
        # no closer: possibly cut off
        '<tool_call>\n{"name": "python", "arguments": {"code": "print("hi")"}}',
        # a rehearsal inside reasoning, and an example inside a code fence
        '<think>\n<tool_call>\n{"name": "python", "arguments": {"code": "print("hi")"}}\n</tool_call>\n</think>',
        '```\n<tool_call>\n{"name": "python", "arguments": {"code": "print("hi")"}}\n</tool_call>\n```',
    ],
)
def test_ambiguous_shapes_stay_text(text):
    assert _calls(text) == []


def test_strict_mode_does_not_heal():
    text = '<tool_call>\n{"name": "python", "arguments": {"code": "print("hi")"}}\n</tool_call>'
    assert _calls(text, allow_incomplete = False) == []


def test_well_formed_call_is_unchanged():
    text = '<tool_call>\n{"name": "python", "arguments": {"code": "print(\\"ok\\")"}}\n</tool_call>'
    assert parse_tool_calls_from_text(text, enabled_tool_names = ENABLED)[0]["function"][
        "arguments"
    ] == json.dumps({"code": 'print("ok")'})


def test_a_call_inside_an_unclosed_call_body_stays_data():
    text = '<|tool_call>call:python{code:x<tool_call>\n{"name": "python", "arguments": {"code": "print("hi")"}}\n</tool_call>'
    assert _calls(text) == []
