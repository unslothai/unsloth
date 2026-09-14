# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Studio's local `python` tool must be EXECUTED, not merely offered.

The vacuity this file exists against is a specific one. `assert_tool_calling`
already proves the model can emit a tool call, and a weather tool is never run
by Studio at all -- the caller runs it. The local `python` tool is different:
Studio executes it in a per-session sandbox, and the failure worth catching is
a loop that hands the model the schema, never runs the call, and lets the model
narrate a plausible result. From the reply text that is indistinguishable from
success, because a language model will happily report the output of code that
never ran.

So the pass condition must come off the FILESYSTEM. A per-run token is written
by the executed code into `<studio home>/sandbox`, and only real execution puts
those bytes on this disk.

One configuration detail is load-bearing rather than incidental:
`routes/inference.py` rejects a local python/terminal tool with a 400 under
`permission_mode` `ask`, and under `auto` or the omitted default, because there
is no confirmation channel on this path. A payload that left it unset would
fail on configuration and read as a broken tool.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "studio_gpu" / "run_studio_gpu.py"
SRC = PAYLOAD.read_text(encoding = "utf-8")


def _func(name: str) -> ast.FunctionDef:
    """The METHOD of that name, not a module-level function.

    `run` is both: a module-level `subprocess` helper and the harness's own
    sequencer. Walking the whole tree finds the helper first, and every
    assertion about the sequence then reads a body that never mentions it.
    """
    tree = ast.parse(SRC)
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for node in cls.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
    raise AssertionError(f"no method named {name!r}")


def _body(name: str = "assert_code_execution") -> str:
    return ast.get_source_segment(SRC, _func(name)) or ""


def test_the_assertion_exists_and_is_driven_from_the_run():
    body = _body()
    assert body
    run = _body("execute")
    assert "self.assert_code_execution()" in run, "an assertion nothing calls is not coverage"


def test_the_python_tool_is_the_one_requested():
    """`enabled_tools` omitted means ALL local tools, which would let the model
    satisfy this with `terminal` and say nothing about `python`."""
    body = _body()
    assert "enable_tools = True" in body
    assert 'enabled_tools = ["python"]' in body


def test_the_permission_mode_is_one_the_local_python_tool_survives():
    """Not a style choice. `ask` is rejected outright for a local python tool,
    and so are `auto` and the omitted default, with a 400 -- there is no
    confirmation channel on this path. Either of those would fail the run on
    configuration while looking like a broken tool."""
    body = _body()
    assert 'permission_mode = "off"' in body
    assert 'permission_mode = "ask"' not in body
    assert 'permission_mode = "auto"' not in body


def test_the_token_is_minted_per_run_and_not_hardcoded():
    """A fixed token would be satisfied by a file an EARLIER run left in the
    sandbox, which is a green tick for code that did not run this time."""
    body = _body()
    assert "secrets_module.token_hex" in body
    call = next(
        node
        for node in ast.walk(_func("assert_code_execution"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "token_hex"
    )
    assert call is not None


def test_the_verdict_is_read_off_the_filesystem_and_not_off_the_reply():
    """The rule this whole file is about. The failure must be raised by the
    absence of a written file; a check on the reply text would pass on a model
    that narrated an execution that never happened.

    Asserted structurally rather than by message matching, because a message
    is satisfied by its own surrounding text.
    """
    func = _func("assert_code_execution")

    # Every `failures.append` that decides the verdict, and what it is guarded by.
    guarded_by_written = False
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
        if not appends:
            continue
        if "written" in test:
            guarded_by_written = True
            assert isinstance(node.test, ast.UnaryOp) and isinstance(
                node.test.op, ast.Not
            ), "the failure must fire when NO file carries the token"
    assert guarded_by_written, (
        "nothing in this assertion depends on a file being written, so it "
        "would pass on a reply that merely claimed the code ran"
    )

    # And the reply is recorded but never judged.
    for node in ast.walk(func):
        if isinstance(node, ast.If) and "reply" in ast.unparse(node.test):
            raise AssertionError(
                "the verdict branches on the model's prose, which is exactly "
                "what an unexecuted tool call can fake"
            )


def test_the_search_reads_content_rather_than_only_a_filename():
    """A small local model rewording the path is not a Studio defect. What
    cannot be faked is the token's BYTES landing on disk, so the content is
    what the search is entitled to rely on."""
    body = _body()
    assert "read_text" in body
    assert "token in body" in body


def test_it_looks_under_the_studio_home_sandbox_and_not_a_stray_root():
    """`sandbox_root()` puts the per-session working directories under the
    studio home precisely so `UNSLOTH_STUDIO_HOME` keeps them together. Reading
    a fixed `~/studio_sandbox` would search a directory this run never wrote
    to, and report a correct execution as a failure."""
    body = _body()
    assert 'self.studio_home / "sandbox"' in body


def test_the_cpu_fallback_records_the_assertion_rather_than_omitting_it():
    """A skipped assertion that leaves no record is indistinguishable from one
    that never existed, and the summary would be short by one line rather than
    red."""
    run = _body("execute")
    assert '"code_execution",' in run, "the non-GPU branch must record code_execution explicitly"
