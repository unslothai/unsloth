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
        '~~~\nget_weather[ARGS]{"city": "call: Paris [ARGS]"}\n~~~',
        "`get_weather[ARGS]{}`",
        'Call it as `search[ARGS]{"q": "x"}` when needed.',
        'Call it as `search[ARGS]{"q": "a } b"}` after this.',
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
        # An example the rehearsal arm over-matches (3-level JSON, unbalanced placeholder) ends at its body and span.
        (
            '```\nsearch[ARGS]{"a": {"b": {"c": 1}}}\n```\nOK <tool_call>{"name":"search","arguments":{}}</tool_call>',
            '```\nsearch[ARGS]{"a": {"b": {"c": 1}}}\n```\nOK ',
        ),
        (
            '```\nsearch[ARGS]{"a": {"b": {"c": 1}}} <|tool_call>call:search{}<tool_call|>\n```',
            '```\nsearch[ARGS]{"a": {"b": {"c": 1}}} \n```',
        ),
        # An example whose body is unbalanced in its span, or holds a call the parser runs, goes as before.
        ("`search[ARGS]{` and `}` done", "`` done"),
        (
            '```\nsearch[ARGS]{\n<tool_call>{"name":"search","arguments":{}}</tool_call>\n}\n```\nDone.',
            "```\n",
        ),
        (
            '~~~\nsearch[ARGS]{\n<|content_invoke_tool_json|>{"name":"search","args":{}}<|end_message|>\n}\n~~~',
            "~~~\n",
        ),
        # A fence closed after a quoted think block is not reopened by the segment split.
        ("```xml\n<think>x</think>\n```\nsearch[ARGS]{}", "```xml\n<think>x</think>\n```\n"),
        (
            "```xml\n<think>x</think>\n```\nUse `search[ARGS]{}` here.",
            "```xml\n<think>x</think>\n```\nUse `search[ARGS]{}` here.",
        ),
        # A backtick inside reasoning is masked, as the parser masks it.
        ("`<think>`example</think> `search[ARGS]{}`", "`<think>`example</think> ``"),
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


@pytest.mark.parametrize(
    "text",
    [
        "Use ``search[ARGS]{}`` here.",
        'Call it as `search[ARGS]{"q": "x"} when needed` ok.',
        "Try `a` then `search[ARGS]{} x` and `b`.",
        "Open `search[ARGS]{} never closed\nnext line",
        "Open `search[ARGS]{} never closed",
        "Use `Example: search[ARGS]{}` here.",
        '```\nsearch[ARGS]{"q":"a } b"}\n```\nDone.',
    ],
)
def test_streamed_inline_example_is_append_only(text):
    routes_inference = pytest.importorskip("routes.inference")
    strip = routes_inference._StreamingDisplayStrip(
        auto_heal_tool_calls = True, enabled_tool_names = ENABLED
    )
    shown = ""
    for i in range(1, len(text) + 1):
        out = strip.strip(routes_inference._hold_pending_mention(text[:i], True, ENABLED))
        assert out.startswith(shown)
        shown = out
    held = routes_inference._held_mention_display(strip, text, True, ENABLED)
    final = shown if held is None else held
    assert final.startswith(shown)
    assert final == routes_inference._strip_tool_xml_for_display(
        text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED
    )


@pytest.mark.parametrize("text", ["Use `python[ARGS]{}` here.", "Use `other[ARGS]{} x` here."])
def test_streamed_unpromotable_inline_example_is_never_held(text):
    routes_inference = pytest.importorskip("routes.inference")
    strip = routes_inference._StreamingDisplayStrip(
        auto_heal_tool_calls = True, enabled_tool_names = ENABLED
    )
    shown = ""
    for i in range(1, len(text) + 1):
        out = strip.strip(routes_inference._hold_pending_mention(text[:i], True, ENABLED))
        assert out.startswith(shown)
        shown = out
    assert shown == text


@pytest.mark.parametrize(
    "chunks",
    [
        ["Use `Example: ", "search[ARGS]{", "}` here."],
        ['```\nsearch[ARGS]{"q":"a } b', '"}\n```\nDone.'],
        ["Use `search", "[ARGS] syntax", "` here."],
        ["<think>Try `search[AR", "GS]{} for this", "` then answer.</think>Done."],
        ['python[ARGS]{"code":"Use `search', "[ARGS]{", '}` here."}'],
        ["Use `search", "[ARGS]{", "}` here."],
        ["Use `sea", "rch[AR", "GS]{", "}` and `searching` here."],
        ["<think>`note</think> Use `search", "[ARGS]{", "}` here."],
        ["The `<thinking>` tag. Use `search", "[ARGS]{", "}` here."],
    ],
)
def test_streamed_chunks_never_rewind(chunks):
    routes_inference = pytest.importorskip("routes.inference")
    strip = routes_inference._StreamingDisplayStrip(
        auto_heal_tool_calls = True, enabled_tool_names = ENABLED
    )
    shown = text = ""
    for chunk in chunks:
        text += chunk
        out = strip.strip(routes_inference._hold_pending_mention(text, True, ENABLED))
        assert out.startswith(shown)
        shown = out
    assert shown == routes_inference._strip_tool_xml_for_display(
        text, auto_heal_tool_calls = True, enabled_tool_names = ENABLED
    )
