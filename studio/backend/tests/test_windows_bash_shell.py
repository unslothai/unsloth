# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The terminal tool must run bash on Windows, not cmd.

Models write bash for a shell tool, and every other platform runs bash. ``cmd /c``
executes only the first line of a multi-line command, leaves single quotes in
the argument, and does not understand bash quoting, so a correct script
half-executes and reports success. These run on every OS by faking the platform,
because studio-backend-ci is Linux-only.
"""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools


@pytest.fixture(autouse = True)
def _clear_bash_cache():
    # Bound before the test so a monkeypatched _windows_bash (a plain lambda,
    # with no cache_clear) does not break teardown.
    cached = tools._windows_bash
    cached.cache_clear()
    yield
    cached.cache_clear()


def test_posix_shell_is_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert tools._get_shell_cmd("echo hi") == ["bash", "-c", "echo hi"]


def test_windows_uses_bash_when_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert tools._get_shell_cmd("echo hi") == [
        r"C:\Program Files\Git\bin\bash.exe",
        "-c",
        "echo hi",
    ]


def test_windows_falls_back_to_cmd_without_bash(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert tools._get_shell_cmd("echo hi") == ["cmd", "/c", "echo hi"]


def test_prefers_git_for_windows_over_path(monkeypatch, tmp_path):
    git_bash = tmp_path / "Git" / "bin" / "bash.exe"
    git_bash.parent.mkdir(parents = True)
    git_bash.write_text("")
    monkeypatch.setattr(os, "environ", {"ProgramW6432": str(tmp_path)})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: r"C:\somewhere\else\bash.exe")
    assert tools._windows_bash() == str(git_bash)


@pytest.mark.parametrize(
    "wsl_path",
    [
        r"C:\Windows\System32\bash.exe",
        r"C:\Users\me\AppData\Local\Microsoft\WindowsApps\bash.exe",
        "C:/Windows/System32/bash.exe",
    ],
)
def test_wsl_launcher_is_rejected(monkeypatch, wsl_path):
    # WSL's bash runs in a different filesystem, so the sandbox workdir would not
    # apply. Falling back to cmd is worse but stays inside the sandbox.
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: wsl_path)
    assert tools._windows_bash() is None


def test_no_bash_anywhere_returns_none(monkeypatch):
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: None)
    assert tools._windows_bash() is None


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows shell behaviour")
def test_multiline_script_runs_every_line_on_windows():
    # The regression itself: under cmd /c only the first line ran, so the loop
    # body and the redirect were silently dropped.
    if tools._windows_bash() is None:
        pytest.skip("no native Win32 bash on this host")
    script = "\n".join(
        [
            "value=unsloth",
            "for i in 1 2 3; do",
            '  echo "line $i $value"',
            "done",
        ]
    )
    out = tools._bash_exec(script)
    for expected in ("line 1 unsloth", "line 2 unsloth", "line 3 unsloth"):
        assert expected in out, out


def test_paths_note_names_the_real_platform():
    # A note that only cites /mnt/data and /tmp/outputs reads as "you are on
    # Linux", and models then decline to launch Windows programs that do exist.
    note = tools._SANDBOX_PATHS_NOTE
    if sys.platform == "win32":
        assert "Windows" in note
        assert "/mnt/data" not in note
        assert "/tmp/outputs" not in note
    else:
        assert "/mnt/data" in note


def test_terminal_tool_description_carries_the_note():
    description = tools.TERMINAL_TOOL["function"]["description"]
    assert tools._SANDBOX_PATHS_NOTE in description


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows-only wording")
def test_paths_note_says_where_commands_run():
    # Without this, models decline to launch a window they believe the user
    # cannot see, and hand back manual instructions instead.
    note = tools._SANDBOX_PATHS_NOTE
    assert "user's own machine" in note
    assert "Start-Process" in note
