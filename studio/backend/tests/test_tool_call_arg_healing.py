# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Healing must not invent an argument the tool does not have.

`_CANONICAL_HEAL_ARG` was hand-kept and defaulted to "query" for anything absent, so it
went stale the moment a tool was added: `edit_file` landed with three required arguments
and no entry, and a call whose JSON was cut off mid-string was healed into `{"query": ...}`
and reported back as "'old_string' and 'new_string' must both be strings" -- a type error
blaming the model for a key it never sent.
"""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tool_loop_controller import (
    _looks_like_broken_json,
    UNPARSED_ARGUMENTS_KEY,
    _heal_arg_key,
    coerce_tool_arguments,
)
from core.inference.tools import execute_tool

_TRUNCATED = '{"path":"flappy-bird.html","old_string":"","new_string":"<!DOCTYPE html>'


@pytest.mark.parametrize(
    "tool_name, key",
    [
        ("python", "code"),
        ("terminal", "command"),
        ("render_html", "code"),
        # No REQUIRED argument at all -- a url-only call fetches without searching -- so
        # this one cannot be derived and is named explicitly.
        ("web_search", "query"),
        # Derived from the schema rather than hand-listed.
        ("search_knowledge_base", "query"),
        ("search_conversation", "query"),
    ],
)
def test_a_single_string_tool_still_heals(tool_name, key):
    coerced = coerce_tool_arguments("some text", heal = True, tool_name = tool_name)
    assert coerced.healed is True
    assert coerced.arguments == {key: "some text"}


@pytest.mark.parametrize("tool_name", ["edit_file", "mcp__server__tool", ""])
def test_a_tool_with_no_single_string_argument_is_not_healed(tool_name):
    coerced = coerce_tool_arguments(_TRUNCATED, heal = True, tool_name = tool_name)
    assert _heal_arg_key(tool_name) is None
    assert coerced.healed is False
    assert coerced.arguments == {UNPARSED_ARGUMENTS_KEY: _TRUNCATED}


def test_valid_json_is_never_healed():
    coerced = coerce_tool_arguments(
        '{"path":"a.py","edits":[{"old_string":"a","new_string":"b"}]}',
        heal = True,
        tool_name = "edit_file",
    )
    assert coerced.healed is False
    assert coerced.arguments["path"] == "a.py"


def test_the_model_is_told_the_arguments_were_cut_off():
    """Naming the real fault is what makes the retry the right one."""
    coerced = coerce_tool_arguments(_TRUNCATED, heal = True, tool_name = "edit_file")

    result = execute_tool("edit_file", coerced.arguments, session_id = "t")

    assert result.startswith("Error:")
    assert "edit_file" in result
    assert "cut off" in result
    assert "Nothing was run" in result
    # The old answer, which blamed the model for keys it never sent.
    assert "must both be strings" not in result


def test_unparseable_but_complete_arguments_are_not_called_truncated():
    coerced = coerce_tool_arguments("not json at all", heal = True, tool_name = "edit_file")

    result = execute_tool("edit_file", coerced.arguments, session_id = "t")

    assert "not valid JSON" in result
    assert "cut off" not in result


def test_a_healable_tool_still_reaches_the_tool_not_the_guard():
    """The guard must catch only the calls that could not be read, not every healed one.

    Retargeted: this once passed `_TRUNCATED` and asserted python healed it into `code`.
    That WAS the defect -- broken JSON became the program -- so the case is now covered by
    `test_broken_json_is_not_healed_even_for_a_single_string_tool`, and what belongs here
    is the bare string healing actually exists for.
    """
    coerced = coerce_tool_arguments("print('hi')", heal = True, tool_name = "python")

    assert UNPARSED_ARGUMENTS_KEY not in coerced.arguments
    assert coerced.arguments == {"code": "print('hi')"}


_TRUNCATED_PYTHON = '{"code":"html = open(\'game.html\',\'w\')\\nhtml.write(\'<!DOCTYPE'


def test_broken_json_is_not_healed_even_for_a_single_string_tool():
    """`python` has one `code` argument, which hid this defect rather than avoiding it.

    A truncated call arrived as `{"code":"html = ...`, healing wrapped the whole fragment
    as the PROGRAM, and the model then read its own file back as `{"code":"html = ...` and
    spent the rest of the turn convinced the sandbox had mangled its content.
    """
    coerced = coerce_tool_arguments(_TRUNCATED_PYTHON, heal = True, tool_name = "python")

    assert coerced.healed is False
    assert coerced.arguments == {UNPARSED_ARGUMENTS_KEY: _TRUNCATED_PYTHON}
    assert "code" not in coerced.arguments


@pytest.mark.parametrize(
    "raw",
    [
        # Fails at char 1, with the rest of the text still to go: not a cut-off call.
        "{not json at all",
        "{oops",
        # Complete JSON with something after it -- broken, but nothing was lost.
        '{"a": 1} trailing',
    ],
)
def test_text_that_merely_opens_with_a_brace_still_heals(raw):
    """Guarding on the opening bracket alone refused calls that were never truncated.

    `test_non_json_arguments_still_reach_the_tool_as_a_dict` covers the contract from the
    other side: a single-required-argument tool is handed the raw text rather than a blob.
    """
    assert _looks_like_broken_json(raw) is False

    coerced = coerce_tool_arguments(raw, heal = True, tool_name = "web_search")

    assert coerced.healed is True
    assert coerced.arguments == {"query": raw}


@pytest.mark.parametrize(
    "raw",
    [
        _TRUNCATED,                      # stops inside new_string
        _TRUNCATED_PYTHON,               # stops inside code
        '{"a": 1,',                      # stops after a comma
        '{"a": ',                        # stops before a value
        '[{"a":1},',                     # stops inside an array
    ],
)
def test_a_call_that_ran_out_of_input_is_never_healed(raw):
    assert _looks_like_broken_json(raw) is True

    coerced = coerce_tool_arguments(raw, heal = True, tool_name = "web_search")

    assert coerced.healed is False
    assert coerced.arguments == {UNPARSED_ARGUMENTS_KEY: raw}


def test_a_genuine_bare_string_still_heals():
    """The case healing exists for: one argument sent as a string instead of an object."""
    coerced = coerce_tool_arguments("print(1 + 1)", heal = True, tool_name = "python")

    assert coerced.healed is True
    assert coerced.arguments == {"code": "print(1 + 1)"}


def test_the_model_is_told_python_arguments_were_cut_off():
    coerced = coerce_tool_arguments(_TRUNCATED_PYTHON, heal = True, tool_name = "python")

    result = execute_tool("python", coerced.arguments, session_id = "t")

    assert "could not be read" in result
    assert "cut off" in result
    assert "Nothing was run" in result
