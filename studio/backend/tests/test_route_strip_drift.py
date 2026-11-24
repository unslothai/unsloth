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
    ],
)
def test_route_strip_still_removes_an_enabled_rehearsal_outside_code(route_strip, text, expected):
    assert route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == expected


def test_route_strip_still_removes_a_real_call(route_strip):
    """Whatever else differs, the route strip must keep doing its actual job."""
    text = 'Here you go. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    stripped = route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED)

    assert "<tool_call>" not in stripped
    assert "Here you go." in stripped


def test_route_strip_is_disabled_when_healing_is_off(route_strip):
    text = '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>'

    assert route_strip(text, auto_heal_tool_calls = False, enabled_tool_names = ENABLED) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            'Try `other[ARGS]{"query":"<tool_call>"}` to search for the literal tag.',
            'Try `other[ARGS]{"query":"<tool_call>"}` to search for the literal tag.',
        ),
        (
            'Example: `web_search[ARGS]{"filters":{"date":{"after":"2025-01-01"}}}`. Answer.</tool_call>',
            'Example: `web_search[ARGS]{"filters":{"date":{"after":"2025-01-01"}}}`. Answer.',
        ),
        (
            'Try `search_docs[ARGS]{"query":"<tool_call>","filters":{"date":{"after":"2025-01-01"}}}`'
            " to find the reference.",
            'Try `search_docs[ARGS]{"query":"<tool_call>","filters":{"date":{"after":"2025-01-01"}}}`'
            " to find the reference.",
        ),
        (
            'The log stops at `search_docs[ARGS]{"query":"<tool_call>`. The JSON string is still open.'
            " Retry the request.",
            'The log stops at `search_docs[ARGS]{"query":"<tool_call>`. The JSON string is still open.'
            " Retry the request.",
        ),
        (
            'The log stops at `search_docs[ARGS]{"query":"x`. Retry.</tool_call>',
            'The log stops at `search_docs[ARGS]{"query":"x`. Retry.',
        ),
        # Protection ends with the code span, even when a brace closes in a later span.
        (
            "The call starts with `web_search[ARGS]{`.</tool_call> Close it with `}`.",
            "The call starts with `web_search[ARGS]{`. Close it with `}`.",
        ),
        (
            '<tool_call>{"name":"web_search","arguments":{"query":"Mistral web_search[ARGS]{ syntax"}}'
            "</tool_call>Here is the reference.",
            "Here is the reference.",
        ),
    ],
)
def test_route_strip_keeps_rehearsal_and_marker_rules_apart(route_strip, text, expected):
    assert route_strip(text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED) == expected
