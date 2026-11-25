# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The display strip keeps a rehearsal quoted in code, as the canonical strip and the parser do."""

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference.tool_call_parser import strip_tool_markup  # noqa: E402

ENABLED = {"get_weather", "search", "web_search"}


@pytest.fixture(scope = "module")
def route_strip():
    routes_inference = pytest.importorskip("routes.inference")
    return routes_inference._strip_tool_xml_for_display


@pytest.mark.parametrize(
    "text",
    [
        '```\nget_weather[ARGS]{"city": "Paris"}\n```',
        '~~~\nget_weather[ARGS]{"city": "Paris"}\n~~~',
        "`get_weather[ARGS]{}`",
        'Call it as `search[ARGS]{"q": "x"}` when needed.',
    ],
)
def test_route_strip_keeps_a_rehearsal_quoted_in_code(route_strip, text):
    assert route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == text
    assert strip_tool_markup(text, final = True, enabled_tool_names = ENABLED) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ('Checking. get_weather[ARGS]{"city": "Paris"}', "Checking. "),
        ('Checking. get_weather[ARGS]{"city": "Par', "Checking. "),
        ("Checking. unknown_tool[ARGS]{}", "Checking. unknown_tool[ARGS]{}"),
        # Markup quoted in a kept example stays; a real call after the code block still goes.
        (
            'Try `other[ARGS]{"query":"<tool_call>"}` to search for the literal tag.',
            'Try `other[ARGS]{"query":"<tool_call>"}` to search for the literal tag.',
        ),
        (
            '```\nsearch[ARGS]{}\n```\n<tool_call>{"name":"search","arguments":{}}</tool_call>Done.',
            "```\nsearch[ARGS]{}\n```\nDone.",
        ),
        (
            '<tool_call>{"name":"web_search","arguments":{"q":"web_search[ARGS]{ syntax"}}</tool_call>Here.',
            "Here.",
        ),
    ],
)
def test_route_strip_still_removes_calls_outside_code(route_strip, text, expected):
    assert route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == expected


def test_route_strip_still_removes_a_real_call(route_strip):
    text = 'Answer <tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    assert route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == "Answer "


def test_route_strip_is_disabled_when_healing_is_off(route_strip):
    text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    assert route_strip(text, auto_heal_tool_calls = False, enabled_tool_names = ENABLED) == text
