# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the System32 working-directory guards: install.ps1 must leave C:\\Windows before it
spends minutes on PyTorch (elevated PowerShell starts in System32, so `irm ... | iex` used to fail there with
"unsloth studio setup failed (exit code 1)" plus a rollback), and the CLI guard must say which folder and how
to get out of it."""

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
    guard_idx = src.index("$InSystemDir = $CurrentDir -and (")
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
    idx = src.index("$InSystemDir = $CurrentDir -and (")
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
        "StartsWith(" in block and "$SystemRootDir" in block
    ), "candidate directories under %SystemRoot% must be filtered out"


def test_install_ps1_guard_failure_message_is_actionable():
    """With nowhere safe to go, the error must say where the user is, why, and what to type."""
    src = _install_ps1()
    idx = src.index("[ERROR] Unsloth cannot be installed from $CurrentDir.")
    block = src[idx : idx + 1500]
    assert "Windows system folder" in block
    assert "Run as administrator" in block, "explain how the user got into System32"
    assert "cd `$env:USERPROFILE" in block, "give the exact cd command"
    assert "irm https://unsloth.ai/install.ps1 | iex" in block, "give the exact re-run command"
    assert "Exit-InstallFailure" in block, "must go through the rollback-aware failure path"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("current_dir", "expected"),
    [
        (r"C:\Windows\System32", "True"),
        (r"c:\windows\system32", "True"),
        (r"C:\Windows\System32\drivers\etc", "True"),
        (r"C:\Windows\SysWOW64", "True"),
        (r"C:\Windows", "True"),
        (r"C:\Users\me", "False"),
        (r"C:\Windows2", "False"),
        (r"C:\WindowsApps\stuff", "False"),
    ],
)
def test_install_ps1_system_dir_match(current_dir: str, expected: str):
    """Run the extracted match expression against Windows-shaped paths (works on any pwsh host)."""
    src = _install_ps1()
    match = re.search(
        r"\$InSystemDir = \$CurrentDir -and \(\n.*?\n    \)\n",
        src,
        flags = re.DOTALL,
    )
    assert match is not None, "install.ps1 $InSystemDir expression not found"
    script = (
        f"$SystemRootDir = 'C:\\Windows'\n"
        f"$CurrentDir = '{current_dir}'\n"
        f"{match.group(0)}"
        '"RESULT=$InSystemDir"\n'
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert result.returncode == 0, result.stderr
    assert f"RESULT={expected}" in result.stdout, result.stdout


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
    "$InSystemDir = $true\n"
)


def _run_relocation_block(
    tmp_path: Path,
    system_root: Path,
    current_dir: Path,
    home_env: Path,
    with_llama_cpp_dir: str = "",
) -> subprocess.CompletedProcess:
    """Run the relocation body on the host's own filesystem (no '\\' concatenation in it)."""
    script = (
        _PS_STUBS
        + f"$SystemRootDir = '{system_root}'\n"
        + f"$CurrentDir = '{current_dir}'\n"
        + f"$WithLlamaCppDir = '{with_llama_cpp_dir}'\n"
        + "Set-Location -LiteralPath $CurrentDir\n"
        + _extract_relocation_block()
        + '\nWrite-Host "CWD:$((Get-Location).ProviderPath)"\n'
        + 'Write-Host "LLAMA:$WithLlamaCppDir"\n'
    )
    # Inherit the real environment (pwsh needs PATH and SystemRoot on Windows) and point
    # every home-ish variable at the fixture. HOMEDRIVE/HOMEPATH too: that pair is where
    # PowerShell gets $HOME on Windows, and a stale one would look like a safe directory.
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
def test_relocation_block_fails_fast_when_every_candidate_is_a_system_directory(tmp_path):
    """No safe directory (SYSTEM account: profile lives under System32) must fail before any install work."""
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
    # ntpath for the path semantics, but expanduser has to be pinned: the real one reads
    # the host's HOME, and on Windows "~" is USERPROFILE, SYSTEM's included.
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
        "Jane Doe",  # a space: `cd C:\Users\Jane Doe` binds 'Doe' as a second argument
        "O'Brien",  # an apostrophe: single quotes must be escaped by doubling
    ],
)
@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_cli_guard_cd_line_actually_runs_in_powershell(profile_name: str, tmp_path):
    """The advertised recovery command must survive a paste, not just read well.

    Set-Location takes one -Path, so an unquoted profile with a space fails with
    "A positional parameter cannot be found that accepts argument 'Doe'".
    """
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
    """Both shells get a quoted path; " is not legal in a Windows path, so cmd's
    double quotes cannot be broken out of, and PowerShell's single quotes are
    verbatim (no $var expansion) with '' escaping an embedded apostrophe."""
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
