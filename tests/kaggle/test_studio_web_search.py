# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Checks the execute_tool log line from this request only, and never fails on empty search results."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _func(name: str) -> ast.FunctionDef:
    for cls in ast.walk(ast.parse(SRC)):
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f"no method named {name!r}")


def _body(name: str = "assert_web_search") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def test_the_assertion_exists_and_is_driven_from_the_run():
    assert _body()
    assert "self.assert_web_search()" in _body("execute")


def test_only_the_web_search_tool_is_offered():
    """With `enabled_tools` omitted the loop may reach for python instead, and
    a run that executed python would satisfy a looser log check while saying
    nothing about search."""
    body = _body()
    assert "enable_tools = True" in body
    assert 'enabled_tools = ["web_search"]' in body


def test_the_evidence_is_the_execution_line_and_not_a_selection_one():
    body = _body()
    assert 'marker = "execute_tool: name=web_search"' in body, (
        "the line must be the one execute_tool writes, because a line written "
        "where the tool is CHOSEN is emitted whether or not it then runs"
    )


def test_it_counts_only_what_this_request_wrote():
    """The payload drives several tool assertions against one server. A grep
    over the whole log would let an earlier assertion's tool call stand in for
    this one, which is a green tick for a search that never happened."""
    body = _body()
    assert "before = self.server_log.read_text" in body
    # Whitespace-insensitive: the formatter rewrites the slice spacing.
    assert "fresh=after[len(before):]" in "".join(body.split())
    assert "fresh.count(marker)" in body, "counted over the fresh slice, not the file"


def test_an_empty_result_set_is_reported_and_not_failed():
    """Deliberately narrow. ddgs runs with no API key, so a provider throttling
    a Kaggle egress IP is a fact about the day; failing on it would be a red
    nobody can act on. The execution is the claim; the results are context."""
    func = _func("assert_web_search")
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        appends = [
            n
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "append"
        ]
        if appends and ("results" in test or "reply" in test):
            raise AssertionError(
                f"failing on {test!r} makes this red whenever the search "
                f"provider is having a bad day"
            )


def test_the_failure_fires_when_nothing_executed():
    func = _func("assert_web_search")
    # Only verdict branches; the early break on a positive count is control flow.
    guarded = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.If)
        and "executions" in ast.unparse(node.test)
        and any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "append"
            for inner in ast.walk(node)
        )
    ]
    assert guarded, "nothing that decides the verdict depends on the tool having run"
    assert all(
        isinstance(n.test, ast.UnaryOp) and isinstance(n.test.op, ast.Not) for n in guarded
    ), "the failure must fire on ZERO executions"


def test_the_cpu_fallback_records_the_assertion_rather_than_omitting_it():
    assert '"web_search",' in _body("execute")


def test_the_search_tool_call_is_FORCED_rather_than_hoped_for():
    """The search call is forced by name, since a bare required let the model answer without searching."""
    body = _body()
    # By name: a bare "required" still let the model answer without searching.
    assert '"function": {"name": "web_search"}' in body
    assert 'tool_choice = "required"' not in body


def test_both_tool_selections_are_tried_before_the_verdict():
    """One attempt cannot tell a selection bug from a model that will not search, so both selections run."""
    body = _body()
    assert '("named", {"enabled_tools": ["web_search"]})' in body
    assert '("all_local_tools", {})' in body
    assert '"any_tool_executions"' in body, (
        "without a count of ANY tool execution, a loop that ran and chose "
        "something else is indistinguishable from a loop that never ran"
    )


def test_the_second_attempt_is_skipped_once_one_succeeds():
    """A passing first attempt must not spend a second inference on the same
    claim; the loop breaks on a positive count."""
    func = _func("assert_web_search")
    src = ast.get_source_segment(SRC, func) or ""
    assert 'if record["executions"]:' in src
    assert "break" in src


def test_the_verdict_counts_web_search_and_not_any_tool():
    """Mutation found this: summing `any_tool_executions` instead passes on the
    python tool being run, which is a different assertion in this same payload.
    The wider count is diagnostic context, never the rule."""
    func = _func("assert_web_search")
    verdict = next(
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Subscript) and ast.unparse(t) == "detail['executions']"
            for t in node.targets
        )
    )
    source = ast.unparse(verdict.value)
    assert "'executions'" in source
    assert (
        "any_tool_executions" not in source
    ), "the verdict counts any tool at all, so the python tool satisfies the web-search claim"
