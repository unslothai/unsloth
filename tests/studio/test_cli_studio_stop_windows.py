# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for `unsloth studio stop` on Windows (PR #5940).

`stop` once used `os.kill(pid, 0)`, which raises WinError 87 on Windows before
reaching taskkill; the fix adds cross-platform `_pid_alive` (tasklist on Windows,
signal-0 elsewhere). AST + mock-only, except the two code-page tests, which run a real
Python child in place of the Windows command they read. The second of those covers
`unsloth start`: #10173 is one defect across both helpers.
"""

import ast
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import typer

import unsloth_cli.commands.start as start_cli

_STUDIO_CMD_PY = Path(__file__).resolve().parents[2] / "unsloth_cli" / "commands" / "studio.py"
_SOURCE = _STUDIO_CMD_PY.read_text(encoding = "utf-8")


def _func_source(name: str) -> str:
    """Return the source of a top-level function `name` in studio.py."""
    tree = ast.parse(_SOURCE)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(_SOURCE, node)
    raise AssertionError(f"function {name!r} not found in studio.py")


def _load_pid_alive(platform: str, fake_run = None):
    """Exec just `_pid_alive` with injectable sys/subprocess, so the win32 branch runs
    on any host."""
    src = _func_source("_pid_alive")
    fake_sys = types.SimpleNamespace(platform = platform)
    fake_sub = types.SimpleNamespace(run = fake_run) if fake_run is not None else subprocess
    ns = {"os": os, "sys": fake_sys, "subprocess": fake_sub}
    exec(src, ns)
    return ns["_pid_alive"]


# ── AST: stop() must not use the broken bare liveness probe ──────────────────


# `stop` delegates signalling to `_signal_stop`, so guarding only `stop` would let os.kill(pid, 0) come back one
# function along and still pass.
@pytest.mark.parametrize("func", ["stop", "_signal_stop"])
def test_stop_does_not_use_bare_oskill_liveness_probe(func):
    """The signalling path must not call os.kill(pid, 0) -- WinError 87 on Windows."""
    stop_src = _func_source(func)
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
                    f"{func}() still uses os.kill(pid, 0); it raises WinError 87 "
                    "on Windows. Use the cross-platform _pid_alive() helper."
                )


def test_pid_alive_helper_is_defined_and_used_by_stop():
    assert "def _pid_alive(" in _SOURCE, "_pid_alive helper missing"
    assert "_pid_alive(pid)" in _func_source("stop"), "stop() must use _pid_alive"
    # The kill itself moved into _signal_stop; keep both ends of the path pinned.
    assert "def _signal_stop(" in _SOURCE, "_signal_stop helper missing"
    assert "taskkill" in _func_source("_signal_stop")
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
        **decode_kwargs,
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


# What a localized tasklist writes, which -X utf8 decodes as UTF-8 (#10173).
_LOCALIZED_TASKLIST = (
    "import sys\n"
    "sys.stdout.buffer.write('\\u4fe1\\u606f: \\u6ca1\\u6709\\u8fd0\\u884c\\u7684\\u4efb\\u52a1\\u5339\\u914d\\u6307\\u5b9a\\u6807\\u51c6\\u3002\\n'.encode('gbk'))\n"
)


def test_pid_alive_reads_a_localized_tasklist_notice(tmp_path):
    fake = tmp_path / "tasklist.py"
    fake.write_text(_LOCALIZED_TASKLIST, encoding = "utf-8")

    def run_fake_tasklist(command, *args, **kwargs):
        assert command[0] == "tasklist"
        kwargs.setdefault("encoding", "utf-8")  # what the launcher's -X utf8 does
        return subprocess.run([sys.executable, str(fake), *command[1:]], *args, **kwargs)

    pid_alive = _load_pid_alive("win32", fake_run = run_fake_tasklist)
    assert pid_alive(43210) is False


def test_a_profile_that_does_not_decode_fails_loudly(monkeypatch, tmp_path, capsys):
    fake = tmp_path / "cmd.py"
    fake.write_text(
        "import sys\nsys.stdout.buffer.write('C:\\\\Users\\\\\\u4e02\\u5f20\\u4e09\\n'.encode('gbk'))\n",
        encoding = "utf-8",
    )
    real_check_output = subprocess.check_output

    def run_fake_cmd(command, *args, **kwargs):
        if command[0] != "cmd.exe":
            return real_check_output(command, *args, **kwargs)
        kwargs.setdefault("encoding", "utf-8")  # what the launcher's -X utf8 does
        return real_check_output([sys.executable, str(fake)], *args, **kwargs)

    monkeypatch.delenv("USERPROFILE", raising = False)
    monkeypatch.setattr(start_cli.subprocess, "check_output", run_fake_cmd)
    with pytest.raises(typer.Exit):
        start_cli._wsl_windows_user_profile(sys.executable)
    assert "set USERPROFILE" in capsys.readouterr().err
