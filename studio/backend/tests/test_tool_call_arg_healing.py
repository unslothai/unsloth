# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Healing must not invent an argument the tool does not have.

`_CANONICAL_HEAL_ARG` was hand-kept and defaulted to "query" for anything absent, so it
went stale the moment a tool was added: `edit_file` landed with three required arguments
and no entry, and a call whose JSON was cut off mid-string was healed into `{"query": ...}`
and reported back as "'old_string' and 'new_string' must both be strings" -- a type error
blaming the model for a key it never sent.
"""

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
    assert "nothing ran" in result
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


_TRUNCATED_PYTHON = "{\"code\":\"html = open('game.html','w')\\nhtml.write('<!DOCTYPE"


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
        # Complete JSON with something after it -- broken, but nothing was lost. The
        # generalisation below keys on "the tail is one unfinished token", and `trailing`
        # is exactly that shape, so "Extra data" has to be excluded by name.
        '{"a": 1} trailing',
        '{"a": 1} tail',
        "[1,2] rest",
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
        _TRUNCATED,  # stops inside new_string
        _TRUNCATED_PYTHON,  # stops inside code
        '{"a": 1,',  # stops after a comma
        '{"a": ',  # stops before a value
        '[{"a":1},',  # stops inside an array
        # Cut inside a bare LITERAL rather than a string. These report at the token's
        # start, not at the end of input, so an end-of-input test alone misses them and
        # the fragment gets healed into an argument.
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
    """It is plumbing between the coercion and execute_tool, and it escaped into the UI.

    Reported from a live thread as a tool card reading
    `{"__unsloth_unparsed_arguments__":"{\\"path\\":\\"flappy-bird.html\\", ...`
    """
    payload = _decision_for(_TRUNCATED).tool_start_payload()

    assert UNPARSED_ARGUMENTS_KEY not in json.dumps(payload)
    # The model's own text is still shown: the user needs to see what was cut off.
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
    """The invariant that matters more than any of the wording below.

    llama-server parses this field while rendering the template, so a value that does not
    parse fails the WHOLE request, not just the one call. Replaying the fragment verbatim
    looked like the honest thing to do and produced a live 500:

        Failed to parse tool call arguments as JSON: [json.exception.parse_error.101]
        parse error at line 1, column 7201: missing closing quote

    A fragment is unparseable by definition, that being why it is here at all.
    """
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
    # This asserted the fragment was replayed verbatim, on the reasoning that `arguments`
    # is a string in this format so the fragment was the honest value. It is a string the
    # server PARSES, which the assertion did not consider, and the result was a live 500.
    # The detail the model needs is in the tool result; the replay only has to be readable.
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
    """MCP tools are discovered at runtime, so `ALL_TOOLS` cannot know them.

    Deriving the key from the static catalogue alone silently withdrew healing from
    every MCP tool: a bare string reached `execute_tool` as the unparsed sentinel and
    was answered as a call that could not be read, though nothing was wrong with it.
    """
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
