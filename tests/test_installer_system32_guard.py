# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the System32 working-directory guards: elevated PowerShell starts in System32, so
install.ps1 must leave C:\\Windows before spending minutes on PyTorch (it used to fail there only after the
download, then roll back), and the CLI guard must name the folder and how to get out of it."""

from __future__ import annotations

import ntpath
import os
import re
import shutil
import subprocess
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
CLI_INIT = REPO_ROOT / "unsloth_cli" / "__init__.py"


# ── install.ps1: relocate before doing any work ──


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def test_install_ps1_leaves_system_directory_before_installing():
    """The guard must fire before winget/Python/uv/venv/PyTorch, not after the download."""
    src = _install_ps1()
    guard_idx = src.index("$InSystemDir = Test-UnderSystemRoot $CurrentDir")
    for marker in (
        'step "winget" "available"',
        "uv venv $VenvDir --python",
        'step "setup" "running unsloth studio setup..."',
    ):
        assert guard_idx < src.index(marker), (
            f"the system-directory guard must run before {marker!r}; installing from System32 "
            "otherwise fails only after PyTorch has downloaded"
        )


def test_install_ps1_guard_covers_the_whole_windows_directory():
    """%SystemRoot%, not just System32: SysWOW64 and WinSxS are equally unusable."""
    src = _install_ps1()
    idx = src.index("$SystemRootDir = if ($env:SystemRoot)")
    block = src[idx : idx + 2500]
    assert '{ "C:\\Windows" }' in block, "SystemRoot must fall back to C:\\Windows"
    assert (
        "Get-Location -PSProvider FileSystem" in block
    ), "the guard must read the filesystem location so a caller parked on HKLM:\\ is still checked"
    assert "OrdinalIgnoreCase" in block, "Windows paths compare case-insensitively"


def test_install_ps1_guard_relocates_and_keeps_relative_llama_cpp_dir():
    src = _install_ps1()
    idx = src.index("$InSystemDir = Test-UnderSystemRoot $CurrentDir")
    block = src[idx : idx + 3000]
    assert "Set-Location -LiteralPath $candidate" in block, "the guard must relocate, not just warn"
    llama_idx = block.index("GetUnresolvedProviderPathFromPSPath($WithLlamaCppDir)")
    set_loc_idx = block.index("Set-Location -LiteralPath $candidate")
    assert (
        llama_idx < set_loc_idx
    ), "--with-llama-cpp-dir must be pinned to the original directory before Set-Location"
    assert "GetFullPath($WithLlamaCppDir)" not in block, (
        "GetFullPath resolves against [Environment]::CurrentDirectory, which Set-Location "
        "does not update; the PSPath resolver follows the PowerShell location"
    )


def test_install_ps1_guard_rejects_candidates_inside_the_windows_directory():
    """SYSTEM's profile is C:\\Windows\\System32\\config\\systemprofile: relocating there fixes nothing."""
    src = _install_ps1()
    idx = src.index("$SafeDirCandidates = @(")
    block = src[idx : idx + 700]
    assert (
        "Test-UnderSystemRoot" in block
    ), "candidate directories under %SystemRoot% must be filtered out"


def test_install_ps1_guard_failure_message_is_actionable():
    """With nowhere safe to go, say where the user is, why, and what to type."""
    src = _install_ps1()
    idx = src.index("[ERROR] Unsloth cannot be installed from $CurrentDir.")
    block = src[idx : src.index("\n        }\n", idx)]
    assert "Windows system folder" in block
    assert "Run as administrator" in block, "explain how the user got into System32"
    assert "irm https://unsloth.ai/install.ps1 | iex" in block, "give the exact re-run command"
    assert "Exit-InstallFailure" in block, "must go through the rollback-aware failure path"
    assert "$env:USERPROFILE" not in block, (
        "this branch is only reached once USERPROFILE was rejected as a candidate, so "
        "naming it would send the user back into the same tree"
    )
    assert "SYSTEM" in block, "name the account type that actually lands here"


def _extract_helper() -> str:
    """install.ps1's Test-UnderSystemRoot, verbatim, so the tests cannot drift from it."""
    src = _install_ps1()
    start = src.index("    function Test-UnderSystemRoot {")
    return src[start : src.index("\n    }\n", start) + len("\n    }\n")]


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (r"C:\Windows\System32", "True"),
        (r"c:\windows\system32", "True"),
        (r"C:\Windows\System32\drivers\etc", "True"),
        (r"C:\Windows\SysWOW64", "True"),
        (r"C:\Windows", "True"),
        (r"C:\Users\me", "False"),
        # Siblings sharing the prefix: rejecting these would abort an install with a
        # supported absolute UNSLOTH_STUDIO_HOME override.
        (r"C:\Windows2", "False"),
        (r"C:\WindowsApps\stuff", "False"),
        (r"C:\WindowsStudio", "False"),
        (r"C:\Windows.old\Users\me", "False"),
    ],
)
def test_install_ps1_under_system_root(path: str, expected: str):
    """The separator is injected because pwsh on Linux reports / for DirectorySeparatorChar."""
    script = (
        "$SystemRootDir = 'C:\\Windows'\n"
        "$SystemRootPrefix = 'C:\\Windows\\'\n"
        f"{_extract_helper()}"
        f"\"RESULT=$(Test-UnderSystemRoot '{path}')\"\n"
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert result.returncode == 0, result.stderr
    assert f"RESULT={expected}" in result.stdout, result.stdout


def test_install_ps1_containment_has_a_path_boundary():
    src = _install_ps1()
    assert (
        "$SystemRootPrefix = $SystemRootDir + [System.IO.Path]::DirectorySeparatorChar" in src
    ), "the prefix must carry a separator so siblings are not swallowed"
    helper = _extract_helper()
    assert "$SystemRootPrefix" in helper and "$SystemRootDir + '\\'" not in helper


def _extract_relocation_block() -> str:
    src = _install_ps1()
    start = src.index("    if ($InSystemDir) {")
    end = src.index("\n    # ── Check winget ──", start)
    return src[start:end]


_PS_STUBS = (
    "function Write-TauriLog { param([string]$Tag, [string]$Message) }\n"
    'function step { param($Label, $Value, $Color) Write-Host "STEP:$Label|$Value" }\n'
    'function substep { param($Message, $Color) Write-Host "SUBSTEP:$Message" }\n'
    'function Exit-InstallFailure { param($Message, $Code = 1) Write-Host "FAILED:$Message"; exit 42 }\n'
    # install.ps1 prints through its UTF-8 stdout sink; the harness only needs the text.
    "function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }\n"
    "$InSystemDir = $true\n"
)


def _run_relocation_block(
    tmp_path: Path,
    system_root: Path,
    current_dir: Path,
    home_env: Path,
    with_llama_cpp_dir: str = "",
    studio_home: Path | None = None,
) -> subprocess.CompletedProcess:
    """Run the relocation body on the host's own filesystem (no '\\' concatenation in it)."""
    script = (
        _PS_STUBS
        + f"$SystemRootDir = '{system_root}'\n"
        + f"$CurrentDir = '{current_dir}'\n"
        + f"$WithLlamaCppDir = '{with_llama_cpp_dir}'\n"
        + f"$StudioHome = '{studio_home or (home_env / '.unsloth' / 'studio')}'\n"
        + "$SystemRootPrefix = $SystemRootDir + [System.IO.Path]::DirectorySeparatorChar\n"
        + _extract_helper()
        + "Set-Location -LiteralPath $CurrentDir\n"
        + _extract_relocation_block()
        + '\nWrite-Host "CWD:$((Get-Location).ProviderPath)"\n'
        + 'Write-Host "LLAMA:$WithLlamaCppDir"\n'
    )
    # Inherit the real environment (pwsh needs PATH and SystemRoot) and repoint every
    # home-ish variable at the fixture. HOMEDRIVE/HOMEPATH too: PowerShell builds $HOME
    # from that pair on Windows, and a stale one would look like a safe directory.
    env = dict(os.environ)
    drive, tail = os.path.splitdrive(str(home_env))
    env.update(
        {
            "HOME": str(home_env),
            "USERPROFILE": str(home_env),
            "PUBLIC": str(home_env),
            "TEMP": str(home_env),
            "TMP": str(home_env),
            "HOMEDRIVE": drive,
            "HOMEPATH": tail,
        }
    )
    return subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 60,
        env = env,
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_relocation_block_moves_out_and_rebases_relative_llama_cpp_dir(tmp_path):
    system_root = tmp_path / "Windows"
    current_dir = system_root / "System32"
    current_dir.mkdir(parents = True)
    home = tmp_path / "home"
    home.mkdir()
    res = _run_relocation_block(
        tmp_path, system_root, current_dir, home, with_llama_cpp_dir = "llama.cpp"
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert f"CWD:{home}" in res.stdout, f"must relocate to the home directory; got {res.stdout!r}"
    assert "Windows system folder" in res.stdout, "the user must be told why the directory changed"
    assert (
        f"LLAMA:{current_dir / 'llama.cpp'}" in res.stdout
    ), "a relative --with-llama-cpp-dir must still resolve against the original directory"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_relocation_block_leaves_a_fully_qualified_llama_cpp_dir_alone(tmp_path):
    system_root = tmp_path / "Windows"
    current_dir = system_root / "System32"
    current_dir.mkdir(parents = True)
    home = tmp_path / "home"
    home.mkdir()
    absolute = tmp_path / "elsewhere" / "llama.cpp"
    res = _run_relocation_block(
        tmp_path, system_root, current_dir, home, with_llama_cpp_dir = str(absolute)
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert (
        f"LLAMA:{absolute}" in res.stdout
    ), "an already qualified path must pass through untouched"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_relocation_block_refuses_a_studio_home_under_the_system_root(tmp_path):
    """Relocating the CWD does not move $StudioHome, which SYSTEM resolves under
    System32\\config\\systemprofile: fail rather than install there."""
    system_root = tmp_path / "Windows"
    current_dir = system_root / "System32"
    current_dir.mkdir(parents = True)
    home = tmp_path / "home"
    home.mkdir()
    studio_home = system_root / "System32" / "config" / "systemprofile" / ".unsloth" / "studio"
    res = _run_relocation_block(tmp_path, system_root, current_dir, home, studio_home = studio_home)
    assert res.returncode == 42, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "would install into" in res.stdout
    assert "normal user account" in res.stdout
    assert "FAILED:" in res.stdout, "must route through Exit-InstallFailure for rollback"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_relocation_block_allows_a_studio_home_beside_the_system_root(tmp_path):
    """UNSLOTH_STUDIO_HOME=C:\\WindowsStudio is a supported absolute override, not a descendant."""
    system_root = tmp_path / "Windows"
    current_dir = system_root / "System32"
    current_dir.mkdir(parents = True)
    home = tmp_path / "home"
    home.mkdir()
    res = _run_relocation_block(
        tmp_path,
        system_root,
        current_dir,
        home,
        studio_home = tmp_path / "WindowsStudio",
    )
    assert res.returncode == 0, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "would install into" not in res.stdout, "a sibling of the Windows directory is fine"
    assert f"CWD:{home}" in res.stdout


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_relocation_block_fails_fast_when_every_candidate_is_a_system_directory(tmp_path):
    """A SYSTEM account's profile lives under System32, so nothing qualifies: fail before any install work."""
    system_root = tmp_path / "Windows"
    current_dir = system_root / "System32"
    current_dir.mkdir(parents = True)
    home = system_root / "System32" / "config" / "systemprofile"
    home.mkdir(parents = True)
    res = _run_relocation_block(tmp_path, system_root, current_dir, home)
    assert res.returncode == 42, f"stdout={res.stdout!r} stderr={res.stderr!r}"
    assert "cannot be installed from" in res.stdout
    assert "irm https://unsloth.ai/install.ps1 | iex" in res.stdout
    assert "FAILED:" in res.stdout, "must route through Exit-InstallFailure for rollback"


# ── unsloth_cli: the message the user actually reads ──


class _FakeExit(Exception):
    def __init__(self, code: int = 0):
        self.code = code


def _run_cli_guard(
    cwd: str,
    argv: list[str] | None = None,
    userprofile: str = r"C:\Users\me",
    public: str | None = None,
) -> tuple[str | None, int | None]:
    """Exec the CLI's win32 guard block with ntpath semantics; returns (message, exit code)."""
    src = CLI_INIT.read_text(encoding = "utf-8")
    start = src.index('    if (\n        _sys.platform == "win32"\n    ):')
    end = src.index("\n\n", src.index("raise typer.Exit(code = 1)", start))
    block = "\n".join(
        line[4:] if line.startswith("    ") else line for line in src[start:end].split("\n")
    )

    captured: dict[str, object] = {}

    def _secho(
        message,
        fg = None,
        err = False,
    ):
        captured["message"] = message

    environ = {"WINDIR": r"C:\Windows", "USERPROFILE": userprofile}
    if public is not None:
        environ["PUBLIC"] = public
    fake_typer = types.SimpleNamespace(secho = _secho, Exit = _FakeExit)
    # ntpath for the path semantics, with expanduser pinned: the real one reads the host's
    # HOME, and on Windows "~" is USERPROFILE, SYSTEM's included.
    fake_path = types.SimpleNamespace(
        normcase = ntpath.normcase,
        normpath = ntpath.normpath,
        join = ntpath.join,
        expanduser = lambda _p: userprofile,
    )
    fake_os = types.SimpleNamespace(
        path = fake_path,
        sep = "\\",
        getcwd = lambda: cwd,
        environ = environ,
    )
    fake_sys = types.SimpleNamespace(platform = "win32", argv = argv or ["unsloth", "studio", "setup"])

    namespace = {"_os": fake_os, "_sys": fake_sys, "typer": fake_typer}
    try:
        exec(compile(block, "cli_guard", "exec"), namespace)
    except _FakeExit as exit_signal:
        return captured.get("message"), exit_signal.code
    return captured.get("message"), None


@pytest.mark.parametrize(
    "cwd",
    [
        r"C:\Windows\System32",
        r"c:\windows\system32",
        r"C:\Windows\System32\config",
        r"C:\Windows\SysWOW64",
    ],
)
def test_cli_guard_refuses_system_directories(cwd: str):
    message, code = _run_cli_guard(cwd)
    assert code == 1, f"{cwd} must exit non-zero"
    assert message is not None


@pytest.mark.parametrize("cwd", [r"C:\Users\me", r"C:\Windows2\System32x", r"D:\work"])
def test_cli_guard_allows_normal_directories(cwd: str):
    message, code = _run_cli_guard(cwd)
    assert code is None, f"{cwd} must not be refused"
    assert message is None


def test_cli_guard_message_names_the_folder_and_the_fix():
    message, _ = _run_cli_guard(r"C:\Windows\System32", argv = ["unsloth", "studio", "setup"])
    assert message is not None
    assert r"C:\Windows\System32" in message, "the message must name the offending directory"
    assert "Run as administrator" in message, "explain how the user got here"
    assert r"cd 'C:\Users\me'" in message, "hand back a cd the user can paste"
    assert "cmd.exe" in message, "cmd needs cd /d, PowerShell does not"
    assert "unsloth studio setup" in message, "repeat the command being retried"


def _cd_line(message: str, shell: str) -> str:
    """The pasteable `cd ...` out of the recovery block, minus its `(shell)` label."""
    line = next(line for line in message.splitlines() if line.strip().endswith(f"({shell})"))
    return line.strip()[: -len(f"({shell})")].strip()


@pytest.mark.parametrize(
    "profile_name",
    [
        "me",
        "Jane Doe",  # space: `cd C:\Users\Jane Doe` binds 'Doe' as a second argument
        "O'Brien",  # apostrophe: single quotes must be escaped by doubling
    ],
)
@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_cli_guard_cd_line_actually_runs_in_powershell(profile_name: str, tmp_path):
    """The advertised recovery command must survive a paste: Set-Location takes one -Path,
    so an unquoted profile with a space fails."""
    profile = tmp_path / profile_name
    profile.mkdir()
    message, _ = _run_cli_guard(r"C:\Windows\System32", userprofile = str(profile))
    assert message is not None
    command = _cd_line(message, "PowerShell")
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", f"{command}; (Get-Location).Path"],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert result.returncode == 0, f"{command!r} failed: {result.stderr!r}"
    assert str(profile) in result.stdout, f"{command!r} landed in {result.stdout!r}"


@pytest.mark.parametrize("profile_name", ["me", "Jane Doe", "O'Brien"])
def test_cli_guard_cd_lines_are_quoted(profile_name: str):
    """Both shells get a quoted path: cmd's double quotes cannot be broken out of (" is not
    legal in a Windows path) and PowerShell's are verbatim, with '' escaping an apostrophe."""
    profile = "C:\\Users\\" + profile_name
    message, _ = _run_cli_guard(r"C:\Windows\System32", userprofile = profile)
    assert message is not None
    assert _cd_line(message, "PowerShell") == "cd '" + profile.replace("'", "''") + "'"
    assert _cd_line(message, "cmd.exe") == 'cd /d "' + profile + '"'


def test_cli_guard_skips_a_system_account_home():
    """SYSTEM's USERPROFILE is under System32, so the cd would land back in the guard."""
    message, code = _run_cli_guard(
        r"C:\Windows\System32",
        userprofile = r"C:\Windows\System32\config\systemprofile",
        public = r"C:\Users\Public",
    )
    assert code == 1
    assert message is not None
    assert "systemprofile" not in message, "must not advertise a home inside the Windows tree"
    assert _cd_line(message, "PowerShell") == "cd 'C:\\Users\\Public'"


def test_cli_guard_omits_the_cd_line_when_every_home_is_a_system_directory():
    message, code = _run_cli_guard(
        r"C:\Windows\System32",
        userprofile = r"C:\Windows\System32\config\systemprofile",
        public = r"C:\Windows\Temp",
    )
    assert code == 1
    assert message is not None
    assert "cd " not in message, "no pasteable path is better than one that fails again"
    assert r"any folder outside C:\Windows" in message


def test_cli_guard_message_repeats_the_actual_command():
    message, _ = _run_cli_guard(
        r"C:\Windows\System32", argv = ["unsloth", "train", "--model", "my model"]
    )
    assert message is not None
    assert (
        'unsloth train --model "my model"' in message
    ), "the retry line must reproduce the invoked command, re-quoting arguments with spaces"
