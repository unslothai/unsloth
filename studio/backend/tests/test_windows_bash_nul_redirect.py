# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows Terminal runs Git Bash, where cmd's `2>nul` makes a real file named nul (#16991)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from core.inference import tools

_rewrite = tools._bash_nul_to_dev_null


@pytest.mark.parametrize(
    "command, expected",
    [
        ("dir 2>nul", "dir 2>/dev/null"),
        ("echo hi > nul", "echo hi > /dev/null"),
        ("cmd >nul 2>&1", "cmd >/dev/null 2>&1"),
        ("cmd > NUL 2>&1 || echo no", "cmd > /dev/null 2>&1 || echo no"),
        ("cmd &>nul; ls", "cmd &>/dev/null; ls"),
        ("cmd >>nul", "cmd >>/dev/null"),
        ("cmd >|nul", "cmd >|/dev/null"),
        ("cmd >&nul", "cmd >&/dev/null"),
        ("read x <nul", "read x </dev/null"),
        ("(where git 2>nul)", "(where git 2>/dev/null)"),
        ('x="$(git rev-parse HEAD 2>nul)"', 'x="$(git rev-parse HEAD 2>/dev/null)"'),
        ('echo "v: $(python -V 2>nul) ok"', 'echo "v: $(python -V 2>/dev/null) ok"'),
        ('x="$(echo "$(ls 2>nul)")"', 'x="$(echo "$(ls 2>/dev/null)")"'),
        (
            "cd sub && python a.py 2>nul\nls 2>nul",
            "cd sub && python a.py 2>/dev/null\nls 2>/dev/null",
        ),
    ],
)
def test_cmd_style_nul_redirects_go_to_dev_null(command, expected):
    assert _rewrite(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "echo hi",
        "cat nul.txt",
        "ls > null",
        "ls > nul.txt",
        "ls > nullable",
        "ls 2>/dev/null",
        "grep '2>nul' run.bat",
        'git commit -m "send output >nul"',
        "echo \"$(echo '2>nul')\"",
        "echo \\>nul",
        "ls # 2>nul",
        "node -e 'x=>nul'",
        "cat <<< nul",
        "cat <<EOF > run.bat\n@echo off\ndir 2>nul\nEOF",
        "cat <<-'EOF' > run.bat\n\tdir 2>nul\n\tEOF",
    ],
)
def test_other_text_is_untouched(command):
    assert _rewrite(command) == command


def test_redirects_after_a_heredoc_are_still_rewritten():
    command = "cat <<EOF > run.bat\ndir 2>nul\nEOF\nrun.bat 2>nul"
    assert _rewrite(command) == "cat <<EOF > run.bat\ndir 2>nul\nEOF\nrun.bat 2>/dev/null"


# System32 bash.exe on Windows is the WSL launcher; use the Git Bash the Terminal runs.
_BASH = tools._windows_bash() if sys.platform == "win32" else shutil.which("bash")


@pytest.mark.skipif(_BASH is None, reason = "needs bash")
def test_rewritten_command_leaves_no_file_and_keeps_its_output(tmp_path):
    out = subprocess.run(
        [_BASH, "-c", _rewrite("echo hidden > nul 2>&1; ls missing 2>nul; echo shown")],
        cwd = tmp_path,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert out.stdout == "shown\n"
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize(
    "platform, profile, rewritten",
    [("win32", "bash", True), ("win32", "cmd_fallback", False), ("linux", "bash", False)],
)
def test_only_the_windows_bash_terminal_rewrites(monkeypatch, platform, profile, rewritten):
    seen = []
    monkeypatch.setattr(tools, "_bash_nul_to_dev_null", lambda c: seen.append(c) or c)
    monkeypatch.setattr(tools, "_terminal_profile", lambda *_a, **_k: profile)

    def stop(*_a, **_k):
        raise RuntimeError("stop")

    monkeypatch.setattr(tools, "_get_workdir", stop)
    monkeypatch.setattr(tools, "_references_studio_credential_here", lambda *_a, **_k: False)
    monkeypatch.setattr(tools, "_harden_parent_against_proc_env_leak", lambda: True)
    monkeypatch.setattr(sys, "platform", platform)
    out = tools._bash_exec("echo hi > nul", None, 5, "t", disable_sandbox = True)
    assert out == "Execution error: stop"
    assert seen == (["echo hi > nul"] if rewritten else [])


@pytest.mark.skipif(sys.platform != "win32", reason = "needs Git Bash on Windows")
def test_terminal_on_git_bash_leaves_no_nul_file():
    if tools._windows_bash() is None:
        pytest.skip(
            reason = "no trusted Git Bash: the Terminal runs cmd.exe, where > nul is the device"
        )
    session = "nul-redirect-16991"
    out = tools._bash_exec(
        "echo hidden > nul 2>&1; echo shown", None, 60, session, disable_sandbox = True
    )
    assert "shown" in out and "hidden" not in out
    workdir = tools._get_workdir(session)
    assert "nul" not in [name.lower() for name in os.listdir("\\\\?\\" + os.path.abspath(workdir))]
