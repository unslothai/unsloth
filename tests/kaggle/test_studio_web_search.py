# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The web_search tool must be executed, and the log is the only witness.

`assert_code_execution` reads its verdict off the filesystem; a web search
leaves nothing there. What it does leave is Studio's own
`execute_tool: name=...` line, emitted from INSIDE `execute_tool`, so it is
written by execution rather than by selection. A loop that hands the model the
schema and never runs the call produces no such line.

Two rules stop the check drifting into vacuity in opposite directions.

It must count only what THIS request wrote. The payload runs several
tool-driven assertions against one long-lived server, and a whole-file grep
would let `assert_code_execution`'s call satisfy this one.

And it must NOT fail on an empty result set. `_web_search` fans out through
ddgs with no API key, so a provider rate-limiting a Kaggle egress IP is a fact
about the day rather than a Studio defect, and failing on it would put a red in
front of every PR that no reader could act on.
"""

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
    # Whitespace-insensitive: the repo's formatter rewrites this slice to
    # `after[len(before) :]`, and a guard that matched the unformatted spelling
    # went red on a reformat rather than on a regression.
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
    # Only the branches that DECIDE the verdict. The loop also breaks early on
    # a positive count, which is a control-flow test rather than a rule, and
    # reading it as one made this guard red on a correct body.
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
    """Otherwise this measures a model's judgement, not Studio's plumbing.

    On kernel unsloth-probe-studio-full2-815a0c the 2B model answered "The
    current version of the Linux kernel is 6.10" straight from parametric
    knowledge, never emitted a call, and the assertion reported that Studio had
    offered web_search and not run it. Studio had done nothing wrong.

    `assert_tool_calling` already forces its own tool the same way. The claim
    here is that Studio EXECUTES the call, and the `execute_tool` log check
    still decides that, so forcing the call narrows the assertion onto the
    thing it is about rather than weakening it.
    """
    body = _body()
    # BY NAME, not the bare "required": that was tried on kernel
    # unsloth-probe-studio-r3-0b85d4 and the model still answered from
    # parametric knowledge with executions 0.
    assert '"function": {"name": "web_search"}' in body
    assert 'tool_choice = "required"' not in body


def test_both_tool_selections_are_tried_before_the_verdict():
    """One attempt cannot tell a selection bug from a model that will not
    search.

    `enabled_tools = ["web_search"]` is one name out of ALL_TOOLS, and
    `routes/inference.py` also reads a request naming only hosted-tool names as
    a provider-hosted ask. Omitting `enabled_tools` selects every local tool,
    which is a different path through the same loop. Reporting "the loop
    offered web_search and never ran it" off the first alone was a guess: no
    evidence in that run showed the tool had been offered at all.
    """
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
