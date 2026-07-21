# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for `unsloth studio stop` on Windows (PR #5940).

`stop` once used `os.kill(pid, 0)`, which raises WinError 87 on Windows before
reaching taskkill; the fix adds cross-platform `_pid_alive` (tasklist on Windows,
signal-0 elsewhere). AST + mock-only; no real processes, no Unsloth deps imported.
"""

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

_STUDIO_CMD_PY = (
    Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"
)
_SOURCE = _STUDIO_CMD_PY.read_text(encoding = "utf-8")


def _func_source(name: str) -> str:
    """Return the source of a top-level function `name` in studio.py."""
    tree = ast.parse(_SOURCE)
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ):
            return ast.get_source_segment(_SOURCE, node)
    raise AssertionError(f"function {name!r} not found in studio.py")


def _load_pid_alive(platform: str, fake_run = None):
    """Exec just `_pid_alive` with injectable sys/subprocess to drive the win32
    branch on any host without importing unsloth_cli."""
    src = _func_source("_pid_alive")
    fake_sys = types.SimpleNamespace(platform = platform)
    fake_sub = (
        types.SimpleNamespace(run = fake_run) if fake_run is not None else subprocess
    )
    ns = {"os": os, "sys": fake_sys, "subprocess": fake_sub}
    exec(src, ns)
    return ns["_pid_alive"]


# ── AST: stop() must not use the broken bare liveness probe ──────────────────


def test_stop_does_not_use_bare_oskill_liveness_probe():
    """stop() must not call os.kill(pid, 0) -- it crashes on Windows."""
    stop_src = _func_source("stop")
    tree = ast.parse(stop_src)
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        f = call.func
        is_os_kill = (
            isinstance(f, ast.Attribute)
            and f.attr == "kill"
            and isinstance(f.value, ast.Name)
            and f.value.id == "os"
        )
        if is_os_kill and len(call.args) == 2:
            sig = call.args[1]
            if isinstance(sig, ast.Constant) and sig.value == 0:
                raise AssertionError(
                    "stop() still uses os.kill(pid, 0); it raises WinError 87 on "
                    "Windows. Use the cross-platform _pid_alive() helper instead."
                )


def test_pid_alive_helper_is_defined_and_used_by_stop():
    assert "def _pid_alive(" in _SOURCE, "_pid_alive helper missing"
    assert "_pid_alive(pid)" in _func_source("stop"), "stop() must use _pid_alive"
    # The helper must special-case Windows via tasklist (os.kill(pid,0) is invalid there).
    helper = _func_source("_pid_alive")
    assert 'sys.platform == "win32"' in helper
    assert "tasklist" in helper


# ── Behavioral: the win32 tasklist branch ────────────────────────────────────


def _fake_tasklist(returns_pid: int | None, *, raises: bool = False):
    def _run(
        cmd,
        capture_output = False,
        text = False,
        timeout = None,
    ):
        assert cmd[0] == "tasklist"
        assert "/FI" in cmd  # filtered by PID
        if raises:
            raise OSError("boom")
        if returns_pid is None:
            stdout = "INFO: No tasks are running which match the specified criteria.\n"
        else:
            stdout = f'"python.exe","{returns_pid}","Console","1","12,345 K"\n'
        return types.SimpleNamespace(stdout = stdout, returncode = 0)

    return _run


def test_pid_alive_windows_true_when_tasklist_lists_pid():
    pid_alive = _load_pid_alive("win32", fake_run = _fake_tasklist(4242))
    assert pid_alive(4242) is True


def test_pid_alive_windows_false_when_tasklist_empty():
    pid_alive = _load_pid_alive("win32", fake_run = _fake_tasklist(None))
    assert pid_alive(4242) is False


def test_pid_alive_windows_assumes_alive_when_tasklist_errors():
    # Can't determine -> assume alive; taskkill is the source of truth.
    pid_alive = _load_pid_alive("win32", fake_run = _fake_tasklist(None, raises = True))
    assert pid_alive(4242) is True


# ── Behavioral: the POSIX signal-0 branch (skip on Windows runners) ───────────


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX os.kill(pid,0) branch")
def test_pid_alive_posix_true_for_self_false_for_dead():
    pid_alive = _load_pid_alive("linux")
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_000_000_000) is False
