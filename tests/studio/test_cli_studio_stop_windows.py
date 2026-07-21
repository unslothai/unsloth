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
import typer

_STUDIO_CMD_PY = Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"
_SOURCE = _STUDIO_CMD_PY.read_text(encoding = "utf-8")
_BACKEND_RUN_PY = Path(__file__).resolve().parents[2] / "studio" / "backend" / "run.py"
_BACKEND_RUN_SOURCE = _BACKEND_RUN_PY.read_text(encoding = "utf-8")


def _func_source(name: str, source: str = _SOURCE) -> str:
    """Return the source of a top-level function from the selected module text."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name!r} not found")


def _load_pid_alive(platform: str, fake_run = None):
    """Exec just `_pid_alive` with injectable sys/subprocess to drive the win32
    branch on any host without importing unsloth_cli."""
    src = _func_source("_pid_alive")
    fake_sys = types.SimpleNamespace(platform = platform)
    fake_sub = types.SimpleNamespace(run = fake_run) if fake_run is not None else subprocess
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


def _run_stop_windows(tmp_path, record: str, current_identity: str):
    pid_file = tmp_path / "studio.pid"
    pid_file.write_text(record)
    calls = []
    liveness = iter((True, False))

    def fake_run(command, check = False):
        calls.append(command)
        return types.SimpleNamespace(returncode = 0)

    ns = {
        "_PID_FILE": pid_file,
        "_pid_alive": lambda _pid: next(liveness),
        "_pid_start_identity": lambda _pid: current_identity,
        "os": os,
        "subprocess": types.SimpleNamespace(run = fake_run),
        "sys": types.SimpleNamespace(platform = "win32"),
        "time": types.SimpleNamespace(sleep = lambda _seconds: None),
        "typer": typer,
    }
    exec(_func_source("stop"), ns)
    with pytest.raises(typer.Exit) as exc:
        ns["stop"]()
    return exc.value.exit_code, calls, pid_file


def test_stop_windows_terminates_verified_process_tree(tmp_path):
    """A matching start identity permits `/T`, so llama-server is terminated too."""
    code, calls, pid_file = _run_stop_windows(tmp_path, "4242:123.5", "123.5")
    assert code == 0
    assert calls == [["taskkill", "/PID", "4242", "/T", "/F"]]
    assert not pid_file.exists()


def test_stop_windows_rejects_recycled_pid_before_taskkill(tmp_path):
    """A stale pidfile must never tree-kill the unrelated process now using its PID."""
    code, calls, pid_file = _run_stop_windows(tmp_path, "4242:123.5", "999.0")
    assert code == 0
    assert calls == []
    assert not pid_file.exists()


def test_stop_windows_legacy_pidfile_does_not_expand_to_process_tree(tmp_path):
    """Bare legacy records keep the old parent-only behavior until Studio restarts."""
    code, calls, pid_file = _run_stop_windows(tmp_path, "4242", "")
    assert code == 0
    assert calls == [["taskkill", "/PID", "4242", "/F"]]
    assert not pid_file.exists()


def test_backend_pidfile_records_process_start_identity(tmp_path):
    """New Studio processes persist enough identity to reject later PID reuse."""
    pid_file = tmp_path / "studio.pid"
    ns = {
        "_PID_FILE": pid_file,
        "_pid_start_identity": lambda _pid: "123.5",
        "os": types.SimpleNamespace(getpid = lambda: 4242),
    }
    exec(_func_source("_write_pid_file", _BACKEND_RUN_SOURCE), ns)
    ns["_write_pid_file"]()
    assert pid_file.read_text() == "4242:123.5"


# ── Behavioral: the POSIX signal-0 branch (skip on Windows runners) ───────────


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX os.kill(pid,0) branch")
def test_pid_alive_posix_true_for_self_false_for_dead():
    pid_alive = _load_pid_alive("linux")
    assert pid_alive(os.getpid()) is True
    assert pid_alive(2_000_000_000) is False
