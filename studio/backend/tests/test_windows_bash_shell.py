# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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


def _fake_trusted_root(monkeypatch, root):
    """Point the Program Files trust check at ``root``.

    _windows_program_roots goes through SHGetKnownFolderPath, absent off
    Windows. The roots are faked here rather than %ProgramFiles%, which the
    resolver deliberately does not read (a caller could relocate the boundary).
    """
    monkeypatch.setattr(tools, "_windows_program_roots", lambda: [str(root)])


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
    _fake_trusted_root(monkeypatch, tmp_path)
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: r"C:\somewhere\else\bash.exe")
    assert tools._windows_bash() == str(git_bash)


def test_untrusted_path_bash_is_rejected(monkeypatch, tmp_path):
    # A bash in a user-writable dir (Scoop, a per-user Git, a project checkout)
    # would run every sandboxed command, defeating the PATH/PATHEXT hardening in
    # _build_safe_env: the command passed the blocklist, then the shell itself is
    # attacker-controlled. cmd is worse to write for but stays trusted.
    scoop_bash = tmp_path / "scoop" / "shims" / "bash.exe"
    scoop_bash.parent.mkdir(parents = True)
    scoop_bash.write_text("")
    _fake_trusted_root(monkeypatch, tmp_path / "Program Files")
    monkeypatch.setattr(os, "environ", {"PATH": str(scoop_bash.parent)})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(scoop_bash))
    assert tools._windows_bash() is None


def test_trusted_path_bash_behind_an_untrusted_shim_is_found(monkeypatch, tmp_path):
    # shutil.which stops at the first hit, so a user shim early on PATH used to
    # decide the answer for a host that does have a trusted bash.
    shim = tmp_path / "shims" / "bash.exe"
    shim.parent.mkdir(parents = True)
    shim.write_text("")
    trusted_dir = tmp_path / "Program Files" / "Git" / "bin"
    trusted_dir.mkdir(parents = True)
    (trusted_dir / "bash.exe").write_text("")
    _fake_trusted_root(monkeypatch, tmp_path / "Program Files")
    monkeypatch.setattr(
        os, "environ", {"PATH": os.pathsep.join([str(shim.parent), str(trusted_dir)])}
    )
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(shim))
    assert tools._windows_bash() == str(trusted_dir / "bash.exe")


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
    # apply. Falling back to cmd is worse but stays inside the sandbox. The dir
    # is trusted here so the marker is the only thing that can reject these.
    monkeypatch.setattr(tools, "_is_trusted_windows_program_dir", lambda _dir: True)
    assert tools._is_trusted_windows_bash(wsl_path) is False


def test_windowsapps_under_program_files_is_still_rejected(monkeypatch, tmp_path):
    # A store package installs under Program Files, so the trust check alone
    # would accept the WSL shim there.
    store_bash = tmp_path / "WindowsApps" / "bash.exe"
    store_bash.parent.mkdir(parents = True)
    store_bash.write_text("")
    _fake_trusted_root(monkeypatch, tmp_path)
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: str(store_bash))
    assert tools._windows_bash() is None


def test_no_bash_anywhere_returns_none(monkeypatch):
    monkeypatch.setattr(tools, "_windows_program_roots", lambda: [])
    monkeypatch.setattr(os, "environ", {})
    monkeypatch.setattr(tools.shutil, "which", lambda _name: None)
    assert tools._windows_bash() is None


def test_blocklist_lexes_bash_syntax_when_the_shell_is_bash(monkeypatch):
    # The scan is keyed to the shell, not the OS: the non-posix lexer never
    # splits on `;`, so a blocked command after a control-flow keyword stayed in
    # argument position and `if true; then rm -rf x; fi` really ran under bash.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "rm" in tools._find_blocked_commands("if true; then rm -rf x; fi")
    assert "curl" in tools._find_blocked_commands("for i in 1; do curl http://x; done")


@pytest.mark.parametrize("command", ["'rm' -rf x", '"rm" -rf x', "echo hi; 'rm' -rf x"])
def test_blocklist_sees_through_quoting_when_the_shell_is_bash(monkeypatch, command):
    # The non-posix lexer keeps the quote marks and _token_basename strips only
    # meta-chars, so `'rm'` never matched `rm`. Harmless in front of cmd, where
    # it is a literal unknown program; under bash it really deletes.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "rm" in tools._find_blocked_commands(command)


def test_blocklist_still_catches_the_plain_form_under_cmd(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    assert "rm" in tools._find_blocked_commands("rm -rf x")


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


def test_shell_note_names_the_shell_that_will_run(monkeypatch):
    # Telling a model it has bash on a host that fell back to cmd reintroduces
    # the multi-line half-execution this note exists to prevent.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "The shell is bash" in tools._build_terminal_shell_note()
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    cmd_note = tools._build_terminal_shell_note()
    assert "cmd, not bash" in cmd_note
    assert "one command per call" in cmd_note


def test_posix_shell_note_is_empty(monkeypatch):
    # The POSIX descriptions are unchanged by this PR.
    monkeypatch.setattr(sys, "platform", "linux")
    assert tools._build_terminal_shell_note() == ""


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
def test_notes_never_recommend_a_blocked_program(monkeypatch, bash):
    # powershell/pwsh are in _BLOCKED_COMMANDS_WIN, so a sandboxed user who
    # follows the prompt gets a hard block instead of a command. `cmd /c start`
    # is not blocked, which makes naming it worse: it advertises the gap.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    text = (tools._build_sandbox_paths_note() + tools._build_terminal_shell_note()).lower()
    for program in tools._BLOCKED_COMMANDS_WIN:
        assert program not in text
    assert "start-process" not in text
    assert "/c start" not in text


def test_terminal_tool_description_is_the_two_notes_appended():
    # _TERMINAL_SHELL_NOTE is "" off Windows, so `note in description` proves
    # nothing about the shell half there. Compare the whole string instead, which
    # at least pins the composition, and cover the note's own content below.
    description = tools.TERMINAL_TOOL["function"]["description"]
    assert description == (
        "Execute a terminal command and return stdout/stderr."
        + tools._SANDBOX_PATHS_NOTE
        + tools._TERMINAL_SHELL_NOTE
    )


@pytest.mark.parametrize("bash", [r"C:\Program Files\Git\bin\bash.exe", None])
def test_windows_shell_note_is_never_empty(monkeypatch, bash):
    # The half the test above cannot reach on a Linux runner: whichever shell the
    # resolver lands on, the terminal description gains a sentence naming it.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
    note = tools._build_terminal_shell_note()
    assert note.startswith(" The shell is ")
    assert note not in tools._build_sandbox_paths_note()


def test_python_tool_description_omits_the_shell():
    # The shell note on the python description would point a model at
    # subprocess/os.system as a way around the terminal blocklist, and none of
    # it applies to the python sandbox anyway.
    description = tools.PYTHON_TOOL["function"]["description"]
    assert tools._SANDBOX_PATHS_NOTE in description
    assert "shell" not in description.lower()
    if sys.platform == "win32":
        assert tools._TERMINAL_SHELL_NOTE not in description


def test_notes_say_where_commands_run(monkeypatch):
    # Without this, models decline to launch a window they believe the user
    # cannot see, and hand back manual instructions instead.
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tools, "_windows_bash", lambda: r"C:\Program Files\Git\bin\bash.exe")
    assert "user's own machine" in tools._build_sandbox_paths_note()
    assert "opens a window on the user's desktop" in tools._build_terminal_shell_note()


# powershell/pwsh only join _BLOCKED_COMMANDS on win32, so these name `rm`
# from the common set to stay meaningful on the Linux CI runner.
@pytest.mark.parametrize(
    "command",
    [
        "cmd /c rm -rf x",
        "cmd //c rm -rf x",
        "cmd //k rm -rf x",
        'cmd //c start "" rm -rf x',
        'cmd //c start /b "" rm -rf x',
        'cmd //c start //min "" rm -rf x',
        r'cmd //c start /d C:/tmp "" rm -rf x',
        # cmd reads a quoted first argument as a title, and shlex drops the
        # quotes, so a named window must not hide the program behind it.
        'cmd //c start "Studio" rm -rf x',
        'cmd //c start //min "Title" rm -rf x',
    ],
)
def test_cmd_shellout_is_screened_through_mangled_switches(command):
    # Git Bash turns a lone /c into a path, so a model writes //c. That spelling
    # skipped the nested scan, making `cmd //c rm` reachable where `cmd /c rm`
    # was blocked, and `start` launches its argument too.
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd //c start "" bash -c "bash /c/x.sh"',
        'cmd //c start "Matrix" bash -c "bash /c/x.sh"',
        "cmd //c start wt",
        "cmd //c dir",
        "start notepad",
        # `start` outside a cmd payload is data, not a launcher.
        "echo start rm",
        "grep start rm file",
        "start rm",
        # ...including after a cmd payload has ended at a separator.
        "cmd /c dir && echo start rm",
        "cmd /c dir; grep start rm file",
    ],
)
def test_detached_windows_stay_launchable(command):
    # `start` is the only route to a window on the user's desktop, which the
    # terminal description promises, so screening must not blanket-block cmd.
    assert not tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    "command",
    ["cmd /c del notes.txt", "cmd //c del notes.txt", "cmd //c git clean -fd"],
)
def test_auto_approval_sees_through_mangled_switches(command):
    # del and git clean are not hard-blocked, so if the high-risk scan misses
    # the //c spelling auto mode approves a destructive payload unprompted.
    assert tools._terminal_is_high_risk(command)


@pytest.mark.parametrize(
    "command",
    [
        'cmd /c start "" rm -rf x',
        'cmd /c start "Studio" rm -rf x',
        "cmd /c rm -rf x",
    ],
)
def test_start_scan_survives_non_posix_lexing(monkeypatch, command):
    # A cmd-only Windows host lexes non-posix, where "" survives as the literal
    # two-character token '""'. Screening the title slot and the token after it
    # covers both lexers without special-casing either.
    monkeypatch.setattr(tools, "_shell_is_posix", lambda: False)
    assert tools._find_blocked_commands(command)


@pytest.mark.parametrize(
    ("command", "blocked"),
    [
        # cmd is the shell, so `cmd /c` wraps the whole command and a top-level
        # start really launches, with no cmd token in the string to key on.
        ("start rm -rf x", True),
        ('start "" rm -rf x', True),
        ('start "Studio" rm -rf x', True),
        ("dir && start rm -rf x", True),
        # ...but a start that is not at command position is still just data.
        ("echo start rm", False),
        ("grep start rm file", False),
        ("dir && echo start rm", False),
        ("start notepad", False),
    ],
)
def test_top_level_start_launches_when_cmd_is_the_shell(monkeypatch, command, blocked):
    monkeypatch.setattr(tools, "_shell_is_posix", lambda: False)
    assert bool(tools._find_blocked_commands(command)) is blocked
