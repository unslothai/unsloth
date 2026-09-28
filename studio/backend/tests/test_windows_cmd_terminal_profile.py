# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The isolated Windows Terminal: cmd.exe inside MXC when Git Bash cannot start there.

Faked platform throughout, since studio-backend-ci is Linux-only; the native tests cover a real host.
"""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import mxc_policy, mxc_probe, os_sandbox, tools

BASH = r"C:\Program Files\Git\bin\bash.exe"


def _cap(available, reason = ""):
    return os_sandbox.SandboxCapability(backend = "mxc", available = available, reason = reason)


@pytest.fixture
def windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.delenv("UNSLOTH_MXC_TERMINAL_CMD", raising = False)
    calls = []

    def host(
        bash = BASH,
        bash_cap = None,
        cmd_cap = None,
        dacl = True,
    ):
        monkeypatch.setattr(tools, "_windows_bash", lambda: bash)
        monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: dacl)

        def snapshot(
            *,
            execution_kind = None,
            selected_executable = None,
            **_kw,
        ):
            calls.append(selected_executable)
            if selected_executable == BASH:
                return bash_cap
            return cmd_cap

        monkeypatch.setattr(os_sandbox, "capability_snapshot", snapshot)
        return calls

    return host


MSYS = mxc_probe.MSYS_NAMESPACE_REASON


@pytest.mark.parametrize(
    "bash, bash_cap, cmd_cap, expected",
    [
        (BASH, _cap(True), _cap(True), "bash"),
        (BASH, _cap(False, MSYS), _cap(True), "cmd_isolated"),
        (BASH, _cap(False, MSYS), _cap(False, "no"), "bash"),
        (BASH, _cap(False, "the live MXC probe did not complete cleanly"), _cap(True), "bash"),
        (None, None, _cap(True), "cmd_isolated"),
        (None, None, _cap(False, "no"), "cmd_fallback"),
    ],
)
def test_profile_matrix(windows, bash, bash_cap, cmd_cap, expected):
    windows(bash = bash, bash_cap = bash_cap, cmd_cap = cmd_cap)
    assert tools._terminal_profile() == expected


def test_cmd_is_only_probed_after_the_msys_verdict(windows):
    calls = windows(
        bash_cap = _cap(False, "the live MXC probe did not complete cleanly"), cmd_cap = _cap(True)
    )
    tools._terminal_profile()
    assert calls == [BASH]


def test_full_access_and_opt_out_keep_the_host_shell(windows, monkeypatch):
    calls = windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    assert tools._terminal_profile(disable_sandbox = True) == "bash"
    monkeypatch.setenv("UNSLOTH_MXC_TERMINAL_CMD", "0")
    assert tools._terminal_profile() == "bash"
    assert calls == []


def test_a_probe_error_keeps_the_host_shell(windows, monkeypatch):
    windows()

    def boom(**_kw):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(os_sandbox, "capability_snapshot", boom)
    assert tools._terminal_profile() == "bash"


def test_posix_is_always_bash(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert tools._terminal_profile() == "bash"


def _userland(monkeypatch, tmp_path):
    bin_dir = tmp_path / "Git" / "usr" / "bin"
    bin_dir.mkdir(parents = True)
    monkeypatch.setattr(tools, "_windows_bash_userland_dirs", lambda: [str(bin_dir)])
    monkeypatch.setattr(
        tools, "_resolve_trusted_windows_git", lambda: (str(tmp_path / "Git" / "cmd"), ".EXE")
    )
    return str(bin_dir)


def test_cmd_env_drops_bash_userland_and_disables_git_helpers(windows, monkeypatch, tmp_path):
    windows()
    userland = _userland(monkeypatch, tmp_path)
    env = tools._build_safe_env(str(tmp_path), shell = "cmd_isolated")
    path = env["PATH"].split(";") if ";" in env["PATH"] else env["PATH"].split(":")
    assert userland not in path
    assert str(tmp_path / "Git" / "cmd") in env["PATH"]
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert (env["GIT_CONFIG_KEY_0"], env["GIT_CONFIG_VALUE_0"]) == ("core.hooksPath", "NUL")
    assert env["GIT_PAGER"] == ""
    assert env["GIT_EDITOR"] == env["GIT_SEQUENCE_EDITOR"] == "unsloth-no-editor"
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def test_cmd_env_finds_git_installed_for_git_bash_only(windows, monkeypatch, tmp_path):
    windows()
    _userland(monkeypatch, tmp_path)
    (tmp_path / "Git" / "bin").mkdir()
    (tmp_path / "Git" / "cmd").mkdir()
    (tmp_path / "Git" / "cmd" / "git.exe").write_bytes(b"MZ")
    monkeypatch.setattr(tools, "_resolve_trusted_windows_git", lambda: ("", ""))
    monkeypatch.setattr(tools, "_windows_bash", lambda: str(tmp_path / "Git" / "bin" / "bash.exe"))
    monkeypatch.setattr(tools, "_is_trusted_windows_program_dir", lambda _path: True)
    env = tools._build_safe_env(str(tmp_path), shell = "cmd_isolated")
    assert os.path.realpath(tmp_path / "Git" / "cmd") in env["PATH"]
    assert str(tmp_path / "Git" / "cmd") not in tools._build_safe_env(str(tmp_path))["PATH"]


def test_default_env_is_unchanged(windows, monkeypatch, tmp_path):
    windows()
    userland = _userland(monkeypatch, tmp_path)
    env = tools._build_safe_env(str(tmp_path))
    assert userland in env["PATH"]
    assert not any(key.startswith("GIT_") for key in env)


def test_blocklist_still_catches_blocked_commands_under_cmd(windows):
    windows()  # bash on the host: the lexer must follow the explicit dialect, not the host shell
    for command in (
        "rm -rf x",
        "echo hi && rm -rf x",
        'cmd /c "rm -rf x"',
        "echo hi & dd if=a of=b",
    ):
        assert tools._find_blocked_commands(command, posix = False), command
    assert tools._find_blocked_commands("echo rm", posix = False) == set()


def test_description_swap_is_copy_on_write():
    original = tools.TERMINAL_TOOL["function"]["description"]
    listed = list(tools.ALL_TOOLS)
    swapped = tools.apply_terminal_profile_description(listed, "cmd_isolated")
    assert listed == list(tools.ALL_TOOLS)
    assert tools.TERMINAL_TOOL["function"]["description"] == original
    terminal = next(t for t in swapped if t["function"]["name"] == "terminal")
    assert "The shell is cmd, running isolated" in terminal["function"]["description"]
    assert terminal["function"]["parameters"] == tools.TERMINAL_TOOL["function"]["parameters"]
    others = [t for t in swapped if t["function"]["name"] != "terminal"]
    assert others == [t for t in listed if t["function"]["name"] != "terminal"]
    for profile in ("bash", "cmd_fallback"):
        assert tools.apply_terminal_profile_description(listed, profile) is listed


def _exec(monkeypatch, tmp_path, command):
    seen = {}
    monkeypatch.setattr(tools, "_get_workdir", lambda _sid: str(tmp_path))
    monkeypatch.setattr(tools, "_account_confinement", lambda: None)
    monkeypatch.setattr(tools.subprocess, "CREATE_NO_WINDOW", 0, raising = False)
    monkeypatch.setattr(tools, "_resolve_trusted_windows_git", lambda: ("", ""))
    monkeypatch.setattr(tools, "_windows_bash_userland_dirs", lambda: [])

    def prepare(plan, **_kw):
        seen["plan"] = plan
        raise os_sandbox.SandboxUnavailableError("stop here")

    monkeypatch.setattr(tools, "_prepare_tool_launch", prepare)
    result = tools._bash_exec(command, session_id = "s1")
    return seen.get("plan"), result


def test_cmd_isolated_runs_system_cmd_with_the_git_env(windows, monkeypatch, tmp_path):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    plan, _result = _exec(monkeypatch, tmp_path, 'git commit -m "a b"')
    assert plan.argv == (tools._windows_system_cmd(), "/c", 'git commit -m "a b"')
    assert plan.argv[0].lower().endswith("\\system32\\cmd.exe")
    assert plan.env["GIT_CONFIG_VALUE_0"] == "NUL"
    assert plan.execution_kind == "terminal"


def test_cmd_isolated_refuses_multiline(windows, monkeypatch, tmp_path):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    plan, result = _exec(monkeypatch, tmp_path, "echo one\necho two")
    assert plan is None
    assert result == tools._CMD_MULTILINE_REFUSED


def test_bash_profile_keeps_multiline_and_bash_argv(windows, monkeypatch, tmp_path):
    windows(bash_cap = _cap(True), cmd_cap = _cap(True))
    plan, _result = _exec(monkeypatch, tmp_path, "echo one\necho two")
    assert plan.argv == (BASH, "-c", "echo one\necho two")
    assert "GIT_CONFIG_COUNT" not in plan.env


def test_bash_hosts_keep_bash_outside_the_measured_dacl_tier(windows):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True), dacl = False)
    assert tools._terminal_profile() == "bash"


def test_cmd_isolated_strips_a_trailing_newline(windows, monkeypatch, tmp_path):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    plan, _result = _exec(monkeypatch, tmp_path, "git status\r\n")
    assert plan.argv[2] == "git status"


def test_cmd_isolated_is_never_replayed_on_the_host(windows, monkeypatch, tmp_path):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    plan, _result = _exec(monkeypatch, tmp_path, "dir")
    assert plan.requested_mode == "required"


@pytest.mark.parametrize(
    "command",
    [
        "echo a&rm -rf x",
        "dir&&rm -rf x",
        "echo x|rm -rf x",
        "r^m -rf x",
        '"r"m -rf x',
    ],
)
def test_cmd_isolated_screens_every_reading_of_the_command(windows, monkeypatch, tmp_path, command):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    plan, result = _exec(monkeypatch, tmp_path, command)
    assert plan is None
    assert result.startswith("Blocked command(s) for safety: rm"), result


def test_cmd_isolated_does_not_treat_single_quotes_as_quoting(windows, monkeypatch, tmp_path):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    monkeypatch.setattr(
        tools, "_BLOCKED_COMMANDS", tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN
    )
    plan, result = _exec(monkeypatch, tmp_path, "echo 'ok & rmdir /s /q x & echo done'")
    assert plan is None
    assert result.startswith("Blocked command(s) for safety: rmdir"), result


@pytest.mark.parametrize(
    "command",
    [
        "if 1==1 rmdir /s /q x",
        "IF exist x rmdir /s /q x",
        "if /i not a==b rmdir x",
        "if 1 equ 1 rmdir x",
        "if 1 == 1 rmdir x",
        'if "a"=="a" rmdir x',
    ],
)
def test_the_blocklist_reads_past_a_cmd_if_condition(windows, monkeypatch, command):
    windows()
    monkeypatch.setattr(
        tools, "_BLOCKED_COMMANDS", tools._BLOCKED_COMMANDS_COMMON | tools._BLOCKED_COMMANDS_WIN
    )
    found = tools._find_blocked_commands(command, posix = False) | tools._find_blocked_commands(
        command, posix = True
    )
    assert "rmdir" in found, command
    assert tools._find_blocked_commands("if 1==1 echo rmdir", posix = False) == set()


def test_a_tool_list_without_the_terminal_never_probes(windows):
    calls = windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    listed = [t for t in tools.ALL_TOOLS if t["function"]["name"] != "terminal"]
    assert tools.apply_terminal_profile_for_request(listed) is listed
    assert calls == []


def test_approval_reads_a_cmd_command_the_way_cmd_splits_it(windows, monkeypatch):
    windows(bash_cap = _cap(False, MSYS), cmd_cap = _cap(True))
    command = "echo 'hi & del victim.txt & echo bye'"
    assert tools._terminal_is_high_risk(command) is True
    monkeypatch.setattr(tools, "_terminal_profile", lambda *_a, **_k: "bash")
    assert tools._terminal_is_high_risk(command) is False  # bash keeps the whole thing one argument


def test_host_launches_keep_cmd_quoting(monkeypatch):
    monkeypatch.setattr(mxc_policy, "_system_cmd", lambda: r"C:\Windows\System32\cmd.exe")
    assert mxc_policy.host_spawn_args(["cmd", "/c", 'type "a b.txt"']) == (
        '"C:\\Windows\\System32\\cmd.exe" /d /s /c "type "a b.txt""'
    )
    assert mxc_policy.host_spawn_args(["bash", "-c", "ls"]) == ["bash", "-c", "ls"]
    multiline = ["cmd", "/c", "echo a\necho b"]
    assert mxc_policy.host_spawn_args(multiline) is multiline
