# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""unsloth#12678: reading the login shell's environment must not touch its history.

Real shells, throwaway HOME; each case first shows the old command damaging it.
"""

from __future__ import annotations

import shutil
import sys

import pytest

from utils import desktop_shell_env as dse

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "no POSIX shell on Windows")

SENTINEL = "UNSLOTH_TEST_RC_LOADED"
ZSH = (
    "HISTFILE=$HOME/.zsh_history\nHISTSIZE=2000\nSAVEHIST=1000\n"
    "unsetopt APPEND_HISTORY\nprint -s plugin-entry\n"
)
BASH = "HISTSIZE=100000\nHISTFILESIZE=100000\nhistory -s plugin-entry\n"


def _history(extended: bool) -> bytes:
    lines = [f"echo history-{i}" for i in range(3000)]
    if extended:
        lines = [f": {1700000000 + i}:0;{line}" for i, line in enumerate(lines)]
    return ("\n".join(lines) + "\n").encode()


# name: (shell, rc file, rc body, extra files, extended history, history after the fix)
CASES = {
    "bash_history_s": ("bash", ".bash_profile", BASH, {}, False, "kept"),
    "bash_exit_trap": (
        "bash",
        ".bash_profile",
        BASH + "trap 'history -w ~/.bash_history' EXIT\n",
        {},
        False,
        "kept",
    ),
    # zsh execs a lone final command and skips its save; a trap turns that off.
    "zsh_exit_trap": ("zsh", ".zshrc", ZSH + "trap true EXIT\n", {}, False, "kept"),
    "zsh_share_history_extended": (
        "zsh",
        ".zshrc",
        ZSH + "setopt EXTENDED_HISTORY SHARE_HISTORY\ntrap true EXIT\n",
        {},
        True,
        "kept",
    ),
    "zsh_zlogout": (
        "zsh",
        ".zshrc",
        ZSH + "trap true EXIT\n",
        {".zlogout": "fc -W $HOME/.zsh_history\n"},
        False,
        "kept",
    ),
    # Readonly HISTFILE: RCS still stops the save; the failed unset must not abort.
    "zsh_readonly_histfile_errexit": (
        "zsh",
        ".zshrc",
        ZSH + "trap true EXIT\ntypeset -r HISTFILE SAVEHIST\nsetopt ERR_EXIT\n",
        {},
        False,
        "kept",
    ),
    # bash has no RCS: appends as on base, but the rewriting trap no longer runs.
    "bash_readonly_histfile_errexit": (
        "bash",
        ".bash_profile",
        BASH + "trap 'history -w ~/.bash_history' EXIT\nreadonly HISTFILE\nset -e\n",
        {},
        False,
        "appended",
    ),
}


def _home(tmp_path, monkeypatch, name, case):
    shell, rc, body, extra, extended, _ = case
    path = shutil.which(shell)
    if path is None:
        pytest.skip(
            reason = f"{shell} is not installed; this case needs the real shell (unsloth#12678)"
        )
    home = tmp_path / name
    home.mkdir()
    (home / rc).write_text(f"export {SENTINEL}=1\n{body}", encoding = "utf-8")
    for extra_name, text in extra.items():
        (home / extra_name).write_text(text, encoding = "utf-8")
    hist = home / (".zsh_history" if shell == "zsh" else ".bash_history")
    hist.write_bytes(_history(extended))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("ZDOTDIR", str(home))
    # Apple Terminal's session-history code, and the runner's own history settings.
    for var in (
        "HISTFILE",
        "ENV",
        "BASH_ENV",
        "TERM_PROGRAM",
        "TERM_SESSION_ID",
        "SHELL_SESSION_HISTORY",
        "HISTCONTROL",
        "HISTIGNORE",
    ):
        monkeypatch.delenv(var, raising = False)
    return path, hist


@pytest.mark.parametrize("name", sorted(CASES))
def test_reading_the_login_shell_leaves_its_history_alone(tmp_path, monkeypatch, name):
    case = CASES[name]
    extended, after = case[4], case[5]

    # Control: the command as it was must damage this fixture, or the test proves nothing.
    shell, hist = _home(tmp_path, monkeypatch, name + "_before", case)
    with monkeypatch.context() as patch:
        patch.setattr(dse, "probe_command", lambda _shell, command: command)
        assert dse.read_login_shell_env(shell = shell).get(SENTINEL) == "1"
    assert hist.read_bytes() != _history(extended)

    shell, hist = _home(tmp_path, monkeypatch, name + "_after", case)
    for _ in range(2):
        assert dse.read_login_shell_env(shell = shell).get(SENTINEL) == "1"
    data = hist.read_bytes()
    if after == "kept":
        assert data == _history(extended)
    else:
        assert data.startswith(_history(extended)) and len(data) > len(_history(extended))


@pytest.mark.parametrize(
    "shell",
    [
        "/bin/sh",
        "/bin/bash",
        "/usr/local/bin/zsh",
        "/opt/homebrew/bin/bash",
        "ksh",
        "zsh5",
        "bash-5.2",
    ],
)
def test_posix_shells_unset_histfile_first(shell):
    command = dse.probe_command(shell, "env -0 > x")
    assert command.endswith("(unset HISTFILE) 2>/dev/null && unset HISTFILE; exec env -0 > x")
    assert "unsetopt RCS" in command


@pytest.mark.parametrize(
    "shell", ["/usr/bin/fish", "/opt/homebrew/bin/nu", "/bin/tcsh", "/usr/bin/pwsh"]
)
def test_other_shells_keep_the_command(shell):
    assert dse.probe_command(shell, "env -0 > x") == "env -0 > x"


@pytest.mark.parametrize("shell", ["dash", "bash", "zsh"])
def test_readonly_histfile_still_reads_the_environment(tmp_path, monkeypatch, shell):
    # dash exits on `unset` of a readonly variable, even inside eval with `|| :`.
    path = shutil.which(shell)
    if path is None:
        pytest.skip(
            reason = f"{shell} is not installed; this case needs the real shell (unsloth#12678)"
        )
    rc = f"export {SENTINEL}=1\nHISTFILE=$HOME/.h\nreadonly HISTFILE\nset -e\n"
    for name in (".profile", ".bash_profile", ".zshrc"):
        (tmp_path / name).write_text(rc, encoding = "utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ZDOTDIR", str(tmp_path))
    for var in ("HISTFILE", "ENV", "BASH_ENV"):
        monkeypatch.delenv(var, raising = False)
    assert dse.read_login_shell_env(shell = path).get(SENTINEL) == "1"


def test_a_shell_rejecting_the_probe_falls_back(tmp_path):
    # Stands in for a non-POSIX shell missing from the list (Plan 9 rc, a renamed fish).
    shell = tmp_path / "notposix"
    shell.write_text(
        f'#!/bin/sh\ncase "$2" in if*) exit 2 ;; esac\n{SENTINEL}=1 exec /bin/sh -c "$2"\n',
        encoding = "utf-8",
    )
    shell.chmod(0o755)
    assert dse.read_login_shell_env(shell = str(shell)).get(SENTINEL) == "1"
