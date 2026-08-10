# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bare rehearsal ``name[ARGS]{json}`` quoted in markdown code stays prose.

The rehearsal form has no sentinel of its own (unlike ``[TOOL_CALLS]`` or the
XML markup), so an answer that *documents* the syntax used to parse as a real
call and then vanish from the rendered message. A fenced block or an inline
span marks the text as an example, so it is neither promoted nor stripped --
the same contract an inactive tool name already gets.

Explicit markers keep their unconditional behaviour inside code: a ```json
block is still a real call for the templates that emit one.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tool_healing import parse_tool_calls_from_text, strip_tool_call_markup

ENABLED = {"terminal", "web_search"}
FENCE = "`" * 3


def _names(content, enabled_tool_names = ENABLED):
    calls = parse_tool_calls_from_text(content, enabled_tool_names = enabled_tool_names)
    return [call["function"]["name"] for call in calls]


QUOTED = {
    "fenced": f'Docs:\n{FENCE}\nterminal[ARGS]{{"command": "id"}}\n{FENCE}',
    "fenced_with_language": f'{FENCE}json\nterminal[ARGS]{{"command": "id"}}\n{FENCE}',
    "fenced_indented": f'  {FENCE}\n  terminal[ARGS]{{"command": "id"}}\n  {FENCE}',
    "tilde_fence": '~~~\nterminal[ARGS]{"command": "id"}\n~~~',
    "inline_span": 'Write `terminal[ARGS]{"command": "id"}` to run it.',
    # A span opened with N backticks closes on a run of N, so it is one span rather
    # than two empty pairs around live markup.
    "inline_double_backtick": 'Write ``terminal[ARGS]{"command": "id"}`` as docs.',
    # A lone backtick is valid content inside a doubled span, so only a run of two closes it.
    "inline_double_with_inner_backtick": (
        'Write ``terminal[ARGS]{"command": "id"} and `x` `` as docs.'
    ),
    "blockquoted_fence": f'> {FENCE}\n> terminal[ARGS]{{"command": "id"}}\n> {FENCE}',
    # A block still streaming has no closing fence yet; it must not execute in the
    # window before the fence arrives.
    "unclosed_fence": f'{FENCE}\nterminal[ARGS]{{"command": "id"}}',
    "crlf_fence": f'{FENCE}\r\nterminal[ARGS]{{"command": "id"}}\r\n{FENCE}',
}


# A real call must still run when the text around it only looks like a fence. These are the
# opposite failure: over-suppression silently drops a tool call the user asked for.
LIVE_AFTER = {
    # A backtick fence info string cannot contain backticks, so this opens an inline span,
    # not a fence running to EOF.
    "line_start_inline_span": f'{FENCE}example{FENCE} then terminal[ARGS]{{"command": "id"}}',
    "crlf_closed_fence": f'{FENCE}\r\ndocs\r\n{FENCE}\r\nterminal[ARGS]{{"command": "id"}}',
}


@pytest.mark.parametrize("case", sorted(LIVE_AFTER))
def test_call_after_fence_like_text_still_runs(case):
    assert _names(LIVE_AFTER[case]) == ["terminal"]


@pytest.mark.parametrize("case", sorted(QUOTED))
def test_quoted_rehearsal_is_not_a_call(case):
    assert _names(QUOTED[case]) == []


@pytest.mark.parametrize("case", sorted(QUOTED))
def test_quoted_rehearsal_stays_visible(case):
    """Parse and strip must agree: text that is not promoted is not removed."""
    content = QUOTED[case]
    assert strip_tool_call_markup(content, enabled_tool_names = ENABLED) == content


def test_unquoted_rehearsal_still_calls():
    content = 'Running now. terminal[ARGS]{"command": "id"}'
    assert _names(content) == ["terminal"]
    assert strip_tool_call_markup(content, enabled_tool_names = ENABLED) == "Running now. "


def test_rehearsal_after_a_closed_fence_still_calls():
    content = f'{FENCE}\nx = 1\n{FENCE}\nterminal[ARGS]{{"command": "id"}}'
    assert _names(content) == ["terminal"]


def test_indented_rehearsal_outside_code_still_calls():
    """Leading whitespace alone is not a code block."""
    assert _names('  terminal[ARGS]{"command": "id"}') == ["terminal"]


@pytest.mark.parametrize(
    "body",
    [
        '[TOOL_CALLS]web_search{"query": "x"}',
        '<tool_call>{"name": "web_search", "arguments": {"query": "x"}}</tool_call>',
    ],
)
def test_explicit_markers_in_a_fence_still_call(body):
    assert _names(f"{FENCE}json\n{body}\n{FENCE}") == ["web_search"]


def test_unrestricted_mode_also_skips_quoted_rehearsal():
    """The code gate does not depend on the enabled-name gate."""
    assert _names(QUOTED["fenced"], enabled_tool_names = None) == []


def test_unmatched_backtick_does_not_hide_a_later_call():
    """An inline span needs a closing run, so a stray backtick stays prose."""
    assert _names('cost is 5` then terminal[ARGS]{"command": "id"}') == ["terminal"]


def test_many_quoted_examples_stay_linear():
    """Ordered spans are bisected, not rescanned per candidate (264 KB / 8k examples)."""
    import time

    content = " ".join('`terminal[ARGS]{"command": "id"}`' for _ in range(8000))
    start = time.perf_counter()
    assert _names(content) == []
    assert strip_tool_call_markup(content, enabled_tool_names = ENABLED) == content
    assert time.perf_counter() - start < 5.0


def test_unmatched_backtick_runs_stay_linear():
    """Inline runs are enumerated by length; a backreference backtracked over 21 KB."""
    import time

    body = "".join("`" * n + " text " for n in range(1, 300))
    content = body + ' terminal[ARGS]{"command": "id"}'
    start = time.perf_counter()
    assert _names(content) == ["terminal"]
    assert time.perf_counter() - start < 2.0


def test_truncated_call_after_a_quoted_example_is_still_stripped():
    """The tail pattern runs to EOF, so a quoted opener must not shield a real call."""
    content = '`terminal[ARGS]{"command": "doc"}` then terminal[ARGS]{"command": '
    stripped = strip_tool_call_markup(content, final = True, enabled_tool_names = ENABLED)
    assert stripped == '`terminal[ARGS]{"command": "doc"}` then'


def test_quoted_example_alone_survives_the_final_pass():
    content = '`terminal[ARGS]{"command": "doc"}`'
    assert strip_tool_call_markup(content, final = True, enabled_tool_names = ENABLED) == content


def test_backtick_inside_arguments_does_not_hide_a_later_call():
    content = 'web_search[ARGS]{"query": "what is `ls`"}\nterminal[ARGS]{"command": "id"}'
    assert _names(content) == ["web_search", "terminal"]
