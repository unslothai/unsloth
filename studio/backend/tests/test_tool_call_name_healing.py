# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A sampled tool name that differs from an enabled tool only in case or ``-``/``_`` runs that tool. Never an
execution-class or MCP tool, never an ambiguous match, and only with auto-heal on."""

from __future__ import annotations

import json

import pytest

from core.inference.tool_loop_controller import ToolLoopController


def _tool(name: str, prop: str = "query") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {prop: {"type": "string"}}},
        },
    }


TOOLS = [
    _tool("web_search"),
    _tool("search_docs"),
    _tool("python", "code"),
    _tool("mcp__fs__read_file", "path"),
]


def _decide(
    name: str,
    tools = TOOLS,
    heal: bool = True,
):
    controller = ToolLoopController(tools = tools, auto_heal_tool_calls = heal)
    call = {"id": "c1", "function": {"name": name, "arguments": json.dumps({"query": "x"})}}
    return controller.prepare_call(call)


@pytest.mark.parametrize(
    "sampled, canonical",
    [
        ("WEB_SEARCH", "web_search"),
        ("Web_Search", "web_search"),
        ("web-search", "web_search"),
        ("Search-Docs", "search_docs"),
    ],
)
def test_near_spelling_runs_the_enabled_tool(sampled, canonical):
    decision = _decide(sampled)
    assert decision.action == "execute"
    assert decision.tool_name == canonical
    assert decision.provenance.get("healed") is True
    assert decision.as_assistant_tool_call()["function"]["name"] == canonical


@pytest.mark.parametrize(
    "sampled", ["PYTHON", "Python", "MCP__FS__READ_FILE", "mcp--fs--read-file"]
)
def test_execution_class_and_mcp_tools_need_their_exact_name(sampled):
    assert _decide(sampled).action == "disabled"


def test_ambiguous_spelling_is_not_guessed():
    tools = [_tool("web_search"), _tool("Web_Search")]
    assert _decide("WEB_SEARCH", tools).action == "disabled"


def test_unknown_name_and_heal_off_are_unchanged():
    assert _decide("search_the_web").action == "disabled"
    assert _decide("WEB_SEARCH", heal = False).action == "disabled"


def test_exact_name_is_untouched():
    decision = _decide("web_search")
    assert decision.action == "execute" and decision.tool_name == "web_search"
    assert not decision.provenance.get("healed")


@pytest.mark.parametrize("sampled", ["web_ſearch", "WEB_ſEARCH", "search_docѕ"])
def test_non_ascii_spelling_is_not_folded_onto_a_tool(sampled):
    assert _decide(sampled).action == "disabled"
