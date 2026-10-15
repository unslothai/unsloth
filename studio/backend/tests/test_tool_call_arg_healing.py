# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Healing must not invent an argument: a hand-kept key map defaulted to query and went stale."""

import json
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
        # No REQUIRED argument (url-only fetches), so it is named explicitly.
        ("web_search", "query"),
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
    assert "nothing ran" in result
    assert "must both be strings" not in result


def test_unparseable_but_complete_arguments_are_not_called_truncated():
    coerced = coerce_tool_arguments("not json at all", heal = True, tool_name = "edit_file")

    result = execute_tool("edit_file", coerced.arguments, session_id = "t")

    assert "not valid JSON" in result
    assert "cut off" not in result


def test_a_healable_tool_still_reaches_the_tool_not_the_guard():
    """The guard catches only unreadable calls; a bare string still heals into its one argument."""
    coerced = coerce_tool_arguments("print('hi')", heal = True, tool_name = "python")

    assert UNPARSED_ARGUMENTS_KEY not in coerced.arguments
    assert coerced.arguments == {"code": "print('hi')"}


_TRUNCATED_PYTHON = "{\"code\":\"html = open('game.html','w')\\nhtml.write('<!DOCTYPE"


def test_broken_json_is_not_healed_even_for_a_single_string_tool():
    """Truncated JSON must not heal into python's single code argument, which hid this defect."""
    coerced = coerce_tool_arguments(_TRUNCATED_PYTHON, heal = True, tool_name = "python")

    assert coerced.healed is False
    assert coerced.arguments == {UNPARSED_ARGUMENTS_KEY: _TRUNCATED_PYTHON}
    assert "code" not in coerced.arguments


@pytest.mark.parametrize(
    "raw",
    [
        "{not json at all",
        "{oops",
        # Complete JSON plus a tail: nothing lost; "Extra data" must be excluded by name.
        '{"a": 1} trailing',
        '{"a": 1} tail',
        "[1,2] rest",
    ],
)
def test_text_that_merely_opens_with_a_brace_still_heals(raw):
    """Guarding on the opening bracket alone refused calls that were never truncated."""
    assert _looks_like_broken_json(raw) is False

    coerced = coerce_tool_arguments(raw, heal = True, tool_name = "web_search")

    assert coerced.healed is True
    assert coerced.arguments == {"query": raw}


@pytest.mark.parametrize(
    "raw",
    [
        _TRUNCATED,
        _TRUNCATED_PYTHON,
        '{"a": 1,',
        '{"a": ',
        '[{"a":1},',
        # Cut inside a bare literal: these report at the token start, not end of input.
        '{"flag":tru',  # Expecting value
        '{"a":nul',
        '{"n":1e',  # Expecting ',' delimiter
        '{"n":12.',
    ],
)
def test_a_call_that_ran_out_of_input_is_never_healed(raw):
    assert _looks_like_broken_json(raw) is True

    coerced = coerce_tool_arguments(raw, heal = True, tool_name = "web_search")

    assert coerced.healed is False
    assert coerced.arguments == {UNPARSED_ARGUMENTS_KEY: raw}


def _decision_for(raw: str):
    from core.inference.tool_loop_controller import ToolCallDecision
    coerced = coerce_tool_arguments(raw, heal = True, tool_name = "edit_file")
    return ToolCallDecision(
        action = "execute",
        tool_name = "edit_file",
        arguments = coerced.arguments,
        tool_call_id = "call_0",
    )


def test_the_sentinel_never_reaches_the_tool_card():
    """The unparsed-arguments sentinel is internal plumbing; it once escaped into a tool card."""
    payload = _decision_for(_TRUNCATED).tool_start_payload()

    assert UNPARSED_ARGUMENTS_KEY not in json.dumps(payload)
    assert payload["arguments"] == {"raw": _TRUNCATED}


@pytest.mark.parametrize(
    "raw",
    [
        _TRUNCATED,
        _TRUNCATED_PYTHON,
        "not json at all",
        '{"a": 1,',
        '{"unterminated": "' + "x" * 4000,
    ],
)
def test_replayed_arguments_always_parse_as_json(raw):
    """llama-server parses replayed arguments while rendering, so one bad value fails the whole request."""
    tool_call = _decision_for(raw).as_assistant_tool_call()

    parsed = json.loads(tool_call["function"]["arguments"])
    assert isinstance(parsed, dict)


def test_a_replayed_unreadable_call_stays_small():
    """The fragment is the content that overflowed the window; resending it is backwards."""
    tool_call = _decision_for(
        '{"path":"x","edits":[{"new_string":"' + "y" * 8000
    ).as_assistant_tool_call()

    assert len(tool_call["function"]["arguments"]) < 200


def test_the_sentinel_never_reaches_the_model():
    """Replaying it taught the model a key that no tool declares."""
    tool_call = _decision_for(_TRUNCATED).as_assistant_tool_call()

    assert UNPARSED_ARGUMENTS_KEY not in json.dumps(tool_call)
    # `arguments` is a string the server PARSES; replaying the fragment caused a 500.
    assert "cut off" in tool_call["function"]["arguments"]


def test_a_readable_call_is_unaffected_at_both_boundaries():
    from core.inference.tool_loop_controller import ToolCallDecision

    decision = ToolCallDecision(
        action = "execute",
        tool_name = "edit_file",
        arguments = {"path": "a.py", "edits": []},
        tool_call_id = "call_0",
    )

    assert decision.unparsed_fragment is None
    assert decision.tool_start_payload()["arguments"] == {"path": "a.py", "edits": []}
    assert json.loads(decision.as_assistant_tool_call()["function"]["arguments"]) == {
        "path": "a.py",
        "edits": [],
    }


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
    assert "nothing ran" in result


_MCP_TOOL = {
    "type": "function",
    "function": {
        "name": "mcp__notes__search",
        "parameters": {
            "type": "object",
            "properties": {"phrase": {"type": "string"}},
            "required": ["phrase"],
        },
    },
}


def test_an_mcp_tool_with_one_string_argument_is_healed_from_the_request_schemas():
    """MCP tools are found at runtime, so healing keys must come from the request schemas, not ALL_TOOLS."""
    coerced = coerce_tool_arguments(
        "quarterly report",
        heal = True,
        tool_name = "mcp__notes__search",
        tool_schemas = [_MCP_TOOL],
    )

    assert coerced.healed is True
    assert coerced.arguments == {"phrase": "quarterly report"}


def test_the_request_schemas_are_not_cached_across_chats():
    """One chat's MCP server must not decide another chat's healing."""
    coerce_tool_arguments(
        "quarterly report",
        heal = True,
        tool_name = "mcp__notes__search",
        tool_schemas = [_MCP_TOOL],
    )

    coerced = coerce_tool_arguments("quarterly report", heal = True, tool_name = "mcp__notes__search")

    assert coerced.healed is False


def test_a_truncated_mcp_call_is_still_not_healed():
    """Knowing the key must not resurrect the defect the guard was added for."""
    coerced = coerce_tool_arguments(
        _TRUNCATED,
        heal = True,
        tool_name = "mcp__notes__search",
        tool_schemas = [_MCP_TOOL],
    )

    assert coerced.healed is False
    assert coerced.arguments == {UNPARSED_ARGUMENTS_KEY: _TRUNCATED}


def test_the_controller_hands_its_own_tools_to_the_healer():
    from core.inference.tool_loop_controller import ToolLoopController  # noqa: PLC0415

    controller = ToolLoopController(tools = [_MCP_TOOL])

    decision = controller.prepare_call(
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "mcp__notes__search", "arguments": "quarterly report"},
        }
    )

    assert decision.arguments == {"phrase": "quarterly report"}
