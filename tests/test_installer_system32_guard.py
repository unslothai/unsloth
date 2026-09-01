# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the System32 working-directory guards: elevated PowerShell starts in System32, so
install.ps1 must leave C:\\Windows before spending minutes on PyTorch (it used to fail there only after the
download, then roll back), and the CLI guard must name the folder and how to get out of it."""

from __future__ import annotations

import importlib.util
import ntpath
import os
import re
import shutil
import subprocess
import types
from unittest import mock
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
CLI_INIT = REPO_ROOT / "unsloth_cli" / "__init__.py"


def _load_guard_module():
    """Load the guard by path: importing the package would drag in typer and every command."""
    path = REPO_ROOT / "unsloth_cli" / "_system_dir_guard.py"
    spec = importlib.util.spec_from_file_location("unsloth_cli_system_dir_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_system_dir_guard = _load_guard_module()


def _install_ps1() -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def test_install_ps1_leaves_system_directory_before_installing():
    """The guard must fire before winget/Python/uv/venv/PyTorch, not after the download."""
    src = _install_ps1()
    guard_idx = src.index("$InSystemDir = Test-UnderSystemRoot $CurrentDir")
    for marker in (
        'step "winget" "available"',
        # No "uv " prefix: the installer invokes the resolved $script:UvExe, not the bare token.
        "venv $VenvDir --python",
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
        # Siblings sharing the prefix: rejecting these would abort an install with a supported absolute
        # UNSLOTH_STUDIO_HOME override.
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
    # read here as the containment check answering wrongly for this path.
    # run_pwsh, not subprocess.run:
    result = run_pwsh(
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
    # install.ps1 prints through its UTF-8 stdout sink;
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
    # Inherit the real environment (pwsh needs PATH and SystemRoot) and repoint every home-ish variable at the fixture.
    # HOMEDRIVE/HOMEPATH too: PowerShell builds $HOME from that pair on Windows, and a stale one would look like a safe
    # directory.
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
    # run_pwsh, not subprocess.run:
    return run_pwsh(
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


def _expand_windows_user(value: str, environ: dict[str, str]) -> str:
    """The real ntpath.expanduser, against the fake environment."""
    with mock.patch.dict(os.environ, environ, clear = True):
        return ntpath.expanduser(value)


def _expand_windows_vars(value: str, environ: dict[str, str]) -> str:
    """The real ntpath.expandvars, against the fake environment.

    Reimplementing it here would test the reimplementation: the guard is given
    the same function it gets in production, reading the environment the test
    describes rather than the host's.
    """
    with mock.patch.dict(os.environ, environ, clear = True):
        return ntpath.expandvars(value)


def _guard_outcome(
    cwd: str | None,
    argv: list[str] | None = None,
    userprofile: str = r"C:\Users\me",
    public: str | None = None,
    environ_extra: dict[str, str] | None = None,
    chdir_error: OSError | None = None,
    getcwd_error: OSError | None = None,
    makedirs_error: OSError | None = None,
    windows_dirs: tuple[str, ...] = (r"C:\Windows",),
    environ_out: dict[str, str] | None = None,
    drive_cwd: dict[str, str] | None = None,
    missing_homes: tuple[str, ...] = (),
    syspath: list[str] | None = None,
    real_paths: tuple[str, ...] = (),
    pass_syspath: bool = True,
    chdir_lands_in: str | None = None,
) -> tuple[str | None, str | None, list[str]]:
    """Run the guard with ntpath semantics; returns (message, colour, chdir calls)."""
    real_windows_dirs = {
        ntpath.normcase(ntpath.join(directory, "System32")) for directory in windows_dirs
    }
    environ = {"WINDIR": r"C:\Windows", "USERPROFILE": userprofile}
    if public is not None:
        environ["PUBLIC"] = public
    if environ_extra:
        environ.update(environ_extra)

    # ntpath for the path semantics, with expanduser pinned: the real one reads the host's HOME, and on Windows "~" is
    # USERPROFILE, SYSTEM's included.
    fake_path = types.SimpleNamespace(
        normcase = ntpath.normcase,
        normpath = ntpath.normpath,
        join = ntpath.join,
        isabs = ntpath.isabs,
        splitdrive = ntpath.splitdrive,
        expanduser = lambda _p: userprofile,
    )

    def _abspath(value):
        """Windows keeps a current directory per drive; only it can resolve these."""
        drive, tail = ntpath.splitdrive(value)
        base = (drive_cwd or {}).get(drive.upper())
        if base is None:
            raise OSError(f"no current directory for {drive}")
        return ntpath.join(base, tail)

    chdir_calls: list[str] = []
    # The guard re-reads the directory after moving, so the fake has to move too.
    current = {"cwd": cwd}

    def _chdir(target):
        chdir_calls.append(target)
        if chdir_error is not None:
            raise chdir_error
        # A junction, or a profile that is itself inside the Windows tree:
        current["cwd"] = (
            chdir_lands_in if chdir_lands_in is not None and target == _RELOCATED else target
        )

    def _getcwd():
        if getcwd_error is not None:
            raise getcwd_error
        return current["cwd"]

    def _makedirs(target, exist_ok = False):
        if makedirs_error is not None:
            raise makedirs_error

    message, colour, fatal = _system_dir_guard.check_working_directory(
        (argv or ["unsloth", "studio", "setup"])[1:],
        environ,
        "win32",
        getcwd = _getcwd,
        chdir = _chdir,
        pathmod = fake_path,
        sep = "\\",
        expanduser = lambda path: _expand_windows_user(path, environ),
        makedirs = _makedirs,
        # Only a folder that really holds System32 counts as a Windows directory.
        isdir = lambda path: ntpath.normcase(path) in real_windows_dirs,
        # sys.path entries are on disk only when the caller says so, the way the filesystem would answer.
        exists = lambda path: ntpath.normcase(path) in {ntpath.normcase(p) for p in real_paths},
        abspath = _abspath,
        home_isdir = lambda path: ntpath.normcase(path)
        not in {ntpath.normcase(home) for home in missing_homes},
        # Windows expansion, from the same environment the guard is reading.
        expandvars = lambda value: _expand_windows_vars(value, environ),
        # The real sys.path belongs to pytest, so the guard gets a copy to pin.
        # pass_syspath = False leaves it out, which is what the console script does: only then does the guard reach the
        # real list.
        **({"syspath": syspath if syspath is not None else []} if pass_syspath else {}),
    )
    if environ_out is not None:
        environ_out.clear()
        environ_out.update(environ)
    assert fatal == (colour == "red"), "only a red message may stop the command"
    return message, colour, chdir_calls


def _run_cli_guard(
    cwd: str,
    argv: list[str] | None = None,
    userprofile: str = r"C:\Users\me",
    public: str | None = None,
) -> tuple[str | None, int | None]:
    """(message, exit code) for a hand-typed command, which is never relocated."""
    message, colour, _ = _guard_outcome(cwd, argv, userprofile, public)
    return message, 1 if colour == "red" else None


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
        "Jane Doe",  # space: `cd C:\Users\Jane Doe` binds 'Doe' as a second argument apostrophe:
        "O'Brien",
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
    # run_pwsh, not subprocess.run:
    # command for it is exactly the wrong reading. See tests/_shared/unsloth_pwsh_runner.py.
    # run_pwsh, not subprocess.run:
    result = run_pwsh(
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


# Issue #8510:

# ── "Run Unsloth at login" (issue #8510): the desktop cannot choose its own cwd ── Windows registers login startup as
# an HKCU Run value, which carries no working directory, so Unsloth Desktop and every CLI child it spawns start in
# System32.
_RELOCATED = r"C:\Users\me\.unsloth"


@pytest.mark.parametrize(
    "argv",
    [
        ["unsloth", "studio", "--api-only", "-H", "127.0.0.1", "-p", "8888"],
        ["unsloth", "studio", "--api-only"],
        ["unsloth", "studio", "provision-desktop-auth"],
        ["unsloth", "studio", "desktop-capabilities", "--json"],
        # The command that upgrades a desktop too old to set the marker;
        ["unsloth", "studio", "update"],
        ["unsloth", "studio", "--help"],
    ],
)
def test_cli_guard_relocates_a_desktop_managed_command(argv: list[str]):
    message, colour, chdir_calls = _guard_outcome(r"C:\Windows\System32", argv = argv)
    assert colour == "yellow", f"{argv} must continue, not exit"
    assert chdir_calls == [_RELOCATED]
    assert message is not None and _RELOCATED in message


def test_cli_guard_lands_where_the_desktop_puts_its_children():
    """studio/src-tauri/src/process.rs pins ~/.unsloth; both halves must agree, or the
    cwd-relative ./models scan finds different folders depending on which half ran."""
    _, _, chdir_calls = _guard_outcome(
        r"C:\Windows\System32", argv = ["unsloth", "studio", "--api-only"]
    )
    assert chdir_calls == [_RELOCATED]


def test_cli_guard_relocates_when_the_desktop_marks_the_child():
    """Newer desktop builds set the marker; the argv rules above cover older ones.

    The marker exists for a command shape this CLI does not know yet, so it is
    tried here with one.
    """
    message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "desktop-handshake", "--json"],
        environ_extra = {"UNSLOTH_DESKTOP_MANAGED": "1"},
    )
    assert colour == "yellow"
    assert chdir_calls == [_RELOCATED]
    assert message is not None


@pytest.mark.parametrize(
    "rest",
    [
        ["run", "some-model"],
        ["run", "--model", r".\local.gguf", "--api-only"],
        ["update", "--local", r".\checkout"],
    ],
)
def test_cli_guard_marker_does_not_authorise_a_command_carrying_a_path(rest: list[str]):
    """The backend and everything below it inherit the marker, so it must not
    widen the set to commands that resolve a path the caller gave them."""
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", *rest],
        environ_extra = {"UNSLOTH_DESKTOP_MANAGED": "1"},
    )
    assert (colour, chdir_calls) == ("red", [])


@pytest.mark.parametrize(
    "argv",
    [
        ["unsloth", "train", "--dataset", "data.json"],
        ["unsloth", "export", "--output", "out"],
        ["unsloth", "studio", "setup"],
        ["unsloth", "start", "claude"],
        # must not be mistaken for the desktop's backend launch.
        # `studio run` declares its own --api-only and takes user paths, so it must not be mistaken for the desktop's
        ["unsloth", "studio", "run", "--model", "./local.gguf", "--api-only"],
        ["unsloth", "studio", "run", "--api-only"],
        # The top-level alias for `studio run`, same reasoning.
        ["unsloth", "run", "--api-only"],
    ],
)
def test_cli_guard_still_refuses_commands_that_take_relative_paths(argv: list[str]):
    """Relocating these would silently resolve the user's own relative paths elsewhere."""
    message, colour, chdir_calls = _guard_outcome(r"C:\Windows\System32", argv = argv)
    assert colour == "red"
    assert chdir_calls == [], "a user command must never be moved out from under its paths"
    assert message is not None


def test_cli_guard_marker_must_be_exactly_one():
    _, colour, _ = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "train"],
        environ_extra = {"UNSLOTH_DESKTOP_MANAGED": "yes"},
    )
    assert colour == "red", "only the value the desktop sets authorises a move"


def test_cli_guard_fails_closed_when_the_chdir_fails():
    message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        chdir_error = PermissionError("denied"),
    )
    assert colour == "red", "a failed move must not continue in the system folder"
    assert chdir_calls == [_RELOCATED]
    assert message is not None


def test_cli_guard_fails_closed_when_the_work_dir_cannot_be_created():
    """Unsloth has to write under the home anyway, and the Rust half stops here too."""
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        makedirs_error = PermissionError("denied"),
    )
    assert colour == "red"
    assert chdir_calls == []


def test_cli_guard_never_relocates_into_the_public_profile():
    """A SYSTEM or service profile must not send one account's caches, scans and
    outputs into a folder every other account on the machine can write."""
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        userprofile = r"C:\Windows\System32\config\systemprofile",
        public = r"C:\Users\Public",
    )
    assert colour == "red"
    assert chdir_calls == []


def test_cli_guard_still_suggests_the_public_profile_to_a_person():
    """It is a fine folder to type by hand, just not one to be moved into silently."""
    message, colour, _ = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "train"],
        userprofile = r"C:\Windows\System32\config\systemprofile",
        public = r"C:\Users\Public",
    )
    assert colour == "red"
    assert message is not None
    assert _cd_line(message, "PowerShell") == "cd 'C:\\Users\\Public'"


@pytest.mark.parametrize("home", [".", r"..\Users\me", r"C:Users\me"])
def test_cli_guard_rejects_a_home_that_is_not_rooted(home: str):
    """A relative home resolves against the folder being escaped."""
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        userprofile = home,
    )
    assert colour == "red"
    assert chdir_calls == []


def test_cli_guard_sees_through_an_extended_length_path():
    r"""\\?\C:\Windows\System32 is the same folder spelled the long way."""
    _, colour, _ = _guard_outcome(r"\\?\C:\Windows\System32", argv = ["unsloth", "train"])
    assert colour == "red"


@pytest.mark.parametrize(
    "argv",
    [
        # A relative repo path here is resolved against the working directory.
        ["unsloth", "studio", "update", "--local", "./repo"],
        ["unsloth", "studio", "desktop-capabilities", "--out", "./x"],
    ],
)
def test_cli_guard_only_relocates_the_exact_desktop_argv(argv: list[str]):
    _, colour, chdir_calls = _guard_outcome(r"C:\Windows\System32", argv = argv)
    assert colour == "red"
    assert chdir_calls == []


def test_cli_guard_marker_does_not_cover_path_taking_commands():
    """The marker is inherited by everything the backend spawns, so it authorises
    the studio commands the desktop runs and nothing else."""
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "train", "--dataset", ".\\data.json"],
        environ_extra = {"UNSLOTH_DESKTOP_MANAGED": "1"},
    )
    assert colour == "red"
    assert chdir_calls == []


def test_cli_guard_fails_closed_when_every_home_is_a_system_directory():
    message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        userprofile = r"C:\Windows\System32\config\systemprofile",
        public = r"C:\Windows\Temp",
    )
    assert colour == "red"
    assert chdir_calls == []
    assert message is not None
    # This one is read in the desktop's logs, not a terminal, so it must not tell
    assert "Run as administrator" not in message


def test_cli_guard_reports_a_deleted_working_directory_for_what_it_is():
    """getcwd() itself raises if the launch directory was removed; that used to be an
    uncaught traceback, and it must not be described as a Windows system folder."""
    message, colour, chdir_calls = _guard_outcome(
        None,
        argv = ["unsloth", "studio", "--api-only"],
        getcwd_error = FileNotFoundError("gone"),
    )
    assert colour == "red"
    assert chdir_calls == []
    assert message is not None
    assert "cannot determine its current folder" in message
    assert r"C:\Windows" not in message, "the user was never in a Windows folder"


def test_cli_guard_finds_the_real_windows_folder_past_a_shadowed_windir():
    """WINDIR can be shadowed from HKCU\\Environment; SystemRoot cannot."""
    _, colour, _ = _guard_outcome(
        r"D:\Windows\System32",
        argv = ["unsloth", "train"],
        environ_extra = {"SystemRoot": r"D:\Windows", "WINDIR": r"C:\Users\me"},
        windows_dirs = (r"D:\Windows",),
    )
    assert colour == "red", "the real Windows folder must still be caught"


def test_cli_guard_ignores_a_windir_that_holds_no_system32():
    """A WINDIR pointed at the user's own profile must not turn their project
    folders into system folders, which would block every command they run."""
    message, colour, _ = _guard_outcome(
        r"C:\Users\me\projects\llm",
        argv = ["unsloth", "train"],
        environ_extra = {"SystemRoot": r"C:\Windows", "WINDIR": r"C:\Users\me"},
        windows_dirs = (r"C:\Windows",),
    )
    assert (message, colour) == (None, None)


def test_cli_guard_keeps_working_when_nothing_looks_like_windows():
    """No candidate holds System32 (an unusual image, or a test): fall back to
    SystemRoot rather than trusting the settable value."""
    _, colour, _ = _guard_outcome(
        r"E:\Windows\System32",
        argv = ["unsloth", "train"],
        environ_extra = {"SystemRoot": r"E:\Windows", "WINDIR": r"C:\Users\me"},
        windows_dirs = (),
    )
    assert colour == "red"


def test_cli_guard_pins_a_relative_studio_home_before_moving():
    """Unsloth resolves the override with Path.resolve(), which anchors a relative
    value to the working directory, so moving first would retarget it."""
    environ_out: dict[str, str] = {}
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "update"],
        environ_extra = {"UNSLOTH_STUDIO_HOME": r".\custom"},
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert chdir_calls == [_RELOCATED]
    assert (
        environ_out["UNSLOTH_STUDIO_HOME"] == r"C:\Windows\System32\.\custom"
    ), "the override must keep naming the folder the caller meant"


@pytest.mark.parametrize("name", _system_dir_guard._RELATIVE_PATH_ENV)
def test_cli_guard_pins_every_relative_path_override(name: str):
    environ_out: dict[str, str] = {}
    _, colour, _ = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        environ_extra = {name: "cache"},
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert environ_out[name] == r"C:\Windows\System32\cache"


def test_cli_guard_does_not_recreate_a_profile_that_is_not_mounted():
    """A roaming profile that has not arrived yet has a writable parent, so
    makedirs would build an empty second one that shadows the real profile."""
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        missing_homes = (r"C:\Users\me",),
    )
    assert (colour, chdir_calls) == ("red", [])


def test_cli_guard_never_relocates_into_the_public_profile_from_userprofile():
    """The public profile is readable and writable by every account on the
    machine, so it is no place for one account's caches, scans and outputs."""
    public = r"C:\Users\Public"
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        userprofile = public,
        environ_extra = {"PUBLIC": public},
    )
    assert (colour, chdir_calls) == ("red", [])


@pytest.mark.parametrize("public", [r"C:\Users\Public", r"c:\users\public\\"])
def test_safe_user_dir_rejects_the_public_profile_from_either_candidate(public: str):
    """It is the folder that is shared, not the variable that named it, so `~`
    resolving there is refused exactly as USERPROFILE is."""
    environ = {"USERPROFILE": r"C:\Windows\System32\config\systemprofile", "PUBLIC": public}
    chosen = _system_dir_guard.safe_user_dir(
        environ,
        [r"C:\Windows"],
        pathmod = ntpath,
        sep = "\\",
        expanduser = lambda _p: r"C:\Users\Public",
    )
    assert chosen is None
    # A human can still be told about it, which is what allow_public is for.
    assert (
        _system_dir_guard.safe_user_dir(
            environ,
            [r"C:\Windows"],
            pathmod = ntpath,
            sep = "\\",
            expanduser = lambda _p: r"C:\Users\Public",
            allow_public = True,
        )
        == public
    )


def test_cli_guard_pins_the_local_checkout_update_reads():
    """`studio update` relocates, and commands/studio.py resolves a relative
    STUDIO_LOCAL_REPO against the working directory, so it has to be pinned."""
    environ_out: dict[str, str] = {}
    _, colour, _ = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "update"],
        environ_extra = {"STUDIO_LOCAL_REPO": r"..\src\unsloth"},
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert environ_out["STUDIO_LOCAL_REPO"] == r"C:\Windows\System32\..\src\unsloth"


@pytest.mark.parametrize(
    ("value", "qualified"),
    [
        (r"C:\cache", True),
        ("C:/cache", True),
        (r"\\server\share\cache", True),
        (r"\\?\C:\cache", True),
        (r"\\?\unc\server\share", True),
        # Rooted, but only to the drive of the current directory.
        (r"\cache", False),
        ("/cache", False),
        # Relative to the current directory on drive D.
        ("D:cache", False),
        ("cache", False),
        (r".\cache", False),
    ],
)
def test_only_a_value_naming_one_folder_counts_as_fully_qualified(value: str, qualified: bool):
    """isabs() answered True for a leading separator until Python 3.13 and False
    after it, so the folder a value names must not be decided by it."""
    assert _system_dir_guard._is_fully_qualified(value, ntpath) is qualified


def test_cli_guard_resolves_a_root_relative_override_through_the_os():
    r"""A single leading separator takes the drive of the current directory, so
    moving to a profile on another drive would move the folder with it."""
    environ_out: dict[str, str] = {}
    _, colour, _ = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        environ_extra = {"HF_HOME": r"\hf-cache"},
        environ_out = environ_out,
        drive_cwd = {"": "C:\\"},
    )
    assert colour == "yellow"
    assert environ_out["HF_HOME"] == r"C:\hf-cache"


def test_cli_guard_anchors_each_entry_of_a_path_list():
    """One relative entry decides what the whole allowlist authorises."""
    environ_out: dict[str, str] = {}
    _, colour, _ = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        environ_extra = {"UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH": r"trusted;D:\shared"},
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert environ_out["UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH"] == (
        r"C:\Windows\System32\trusted;D:\shared"
    )


@pytest.mark.parametrize(
    "profile",
    [r"\\?\UNC\server\profiles\me", r"\\?\unc\server\profiles\me"],
)
def test_cli_guard_accepts_an_extended_unc_profile_in_either_case(profile: str):
    r"""The object manager is case-insensitive, so reading \\?\unc\ as a relative
    name would reject a profile Windows itself resolves."""
    chosen = _system_dir_guard.safe_user_dir(
        {"USERPROFILE": profile},
        [r"C:\Windows"],
        pathmod = ntpath,
        sep = "\\",
        expanduser = lambda _p: profile,
    )
    assert chosen == profile


def test_cli_guard_resolves_a_drive_relative_override_through_the_os():
    """ "D:cache" means the current directory on drive D, which join() cannot know
    and which the move changes, so Windows itself has to resolve it first."""
    environ_out: dict[str, str] = {}
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        environ_extra = {"HF_HOME": "D:cache"},
        environ_out = environ_out,
        drive_cwd = {"D:": r"D:\work"},
    )
    assert (colour, chdir_calls) == ("yellow", [_RELOCATED])
    assert environ_out["HF_HOME"] == r"D:\work\cache"


def test_cli_guard_refuses_to_move_when_a_drive_relative_override_cannot_be_resolved():
    """Moving anyway would silently retarget it, which is what the pinning exists
    to prevent, so the guard fails closed instead."""
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        environ_extra = {"HF_HOME": "D:cache"},
        drive_cwd = {},
    )
    assert (colour, chdir_calls) == ("red", [])


def test_cli_guard_pins_every_storage_root_override_studio_reads():
    """storage_roots.py owns the Unsloth folders, and each of its overrides is a
    plain user-supplied path, so a relative one must be pinned before the move.
    Reading them from the module keeps the guard honest as roots are added."""
    storage_roots = REPO_ROOT / "studio" / "backend" / "utils" / "paths" / "storage_roots.py"
    source = storage_roots.read_text(encoding = "utf-8")
    overrides = set(re.findall(r'environ\.get\(\s*"(UNSLOTH_[A-Z_]*(?:HOME|PATH|DIR))"', source))
    assert overrides, "no storage root overrides found: has storage_roots.py moved?"
    missing = sorted(overrides - set(_system_dir_guard._RELATIVE_PATH_ENV))
    assert not missing, f"relative values of {missing} would be retargeted by the move"


@pytest.mark.parametrize("value", [r"C:\elsewhere\custom", r"\\server\share\c"])
def test_cli_guard_leaves_an_already_anchored_override_alone(value: str):
    """An absolute or UNC value does not depend on the working directory."""
    environ_out: dict[str, str] = {}
    _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "studio", "--api-only"],
        environ_extra = {"UNSLOTH_STUDIO_HOME": value},
        environ_out = environ_out,
    )
    assert environ_out["UNSLOTH_STUDIO_HOME"] == value


def test_cli_guard_does_not_touch_overrides_when_it_refuses():
    """No move, nothing to pin: the command runs where the caller left it."""
    environ_out: dict[str, str] = {}
    _, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        argv = ["unsloth", "train"],
        environ_extra = {"UNSLOTH_STUDIO_HOME": r".\custom"},
        environ_out = environ_out,
    )
    assert (colour, chdir_calls) == ("red", [])
    assert environ_out["UNSLOTH_STUDIO_HOME"] == r".\custom"


def test_cli_guard_does_nothing_off_windows():
    outcome = _system_dir_guard.check_working_directory(
        ["studio", "--api-only"],
        {"WINDIR": r"C:\Windows"},
        "linux",
        getcwd = lambda: r"C:\Windows\System32",
        chdir = lambda _target: pytest.fail("must not move on non-Windows platforms"),
    )
    assert outcome == (None, None, False)


def test_cli_guard_runs_before_the_command_modules_are_imported():
    """unsloth_cli.commands.studio resolves STUDIO_HOME at import time, so a chdir in
    the callback would come too late for a relative UNSLOTH_STUDIO_HOME."""
    source = CLI_INIT.read_text(encoding = "utf-8")
    guard_call = source.index("_check_working_directory(_sys.argv[1:]")
    first_command_import = source.index("from unsloth_cli.commands.")
    assert guard_call < first_command_import
    assert source.index("import typer") > guard_call


def test_cli_callback_exits_only_on_a_fatal_outcome():
    """The callback must key the exit on the guard's own fatal flag, not on the colour."""
    source = CLI_INIT.read_text(encoding = "utf-8")
    assert "_message, _colour, _fatal = _guard" in source
    assert "if _fatal:\n        raise typer.Exit(code = 1)" in source


def test_cli_guard_leaves_values_the_working_directory_does_not_resolve():
    r"""Two readers in the tree take these as something other than a path, so
    anchoring one would change its meaning instead of preserving it: MLX_HOSTFILE
    holds either a filename or the host list itself, and a bare on/off token is
    ignored by the pre-quant allowlist precisely so there is no "allow all" mode."""
    environ_out: dict[str, str] = {}
    values = {
        "MLX_HOSTFILE": '[{"ssh": "node0"}]',
        "UNSLOTH_ALLOW_LOCAL_PREQUANT_PATH": "1",
    }
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = dict(values),
        environ_out = environ_out,
    )
    assert (colour, chdir_calls) == ("yellow", [_RELOCATED])
    for name, value in values.items():
        assert environ_out[name] == value, f"{name} was rewritten to {environ_out[name]}"


def test_cli_guard_expands_a_cache_override_before_deciding():
    r"""huggingface_hub expands %LOCALAPPDATA% in HF_HOME and Unsloth's own
    hf_cache_settings does not, so leaving the value as written sends the two
    readers to different folders once the process moves. Expanding here settles
    it; a variable this machine does not set stays literal and is anchored."""
    environ_out: dict[str, str] = {}
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {
            "LOCALAPPDATA": r"C:\Users\me\AppData\Local",
            "HF_HOME": r"%LOCALAPPDATA%\hf",
            "HF_HUB_CACHE": r"%NOT_SET%\hub",
        },
        environ_out = environ_out,
    )
    assert (colour, chdir_calls) == ("yellow", [_RELOCATED])
    assert environ_out["HF_HOME"] == r"C:\Users\me\AppData\Local\hf"
    assert environ_out["HF_HUB_CACHE"] == r"C:\Windows\System32\%NOT_SET%\hub"


def test_cli_guard_refuses_a_path_attached_to_its_own_option():
    r"""`--frontend=.\dist` carries a path inside the option token, so a marked
    grandchild running it must be refused rather than moved: relocating would
    silently rebase that path under the profile."""
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", r"--frontend=.\dist"],
        environ_extra = {_system_dir_guard.DESKTOP_MANAGED_ENV: "1"},
    )
    assert (colour, chdir_calls) == ("red", [])


def test_cli_guard_refuses_a_root_relative_system_profile():
    r"""SYSTEM's profile spelled without a drive still names SYSTEM's profile, and
    it compares equal to no drive-qualified Windows root, so a rooted-only test
    would send the caller back into the folder it is leaving."""
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        userprofile = r"\Windows\System32\config\systemprofile",
    )
    assert (colour, chdir_calls) == ("red", [])


def test_cli_guard_anchors_relative_import_roots_before_it_moves():
    r"""PYTHONPATH entries reach sys.path as written and are resolved on every
    import, so a move would let whatever sits in ~/.unsloth shadow them."""
    environ_out: dict[str, str] = {}
    syspath = [
        r"C:\Python\Lib",
        "lib",
        "",
        r".\plugins",
        # setuptools registers this for an editable namespace install and its own path hook accepts it by exact string;
        "__editable__.unsloth-2026.8.15.finder.__path_hook__",
        # A relative archive: importable, so it moves with the process.
        "modules.zip",
    ]
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {"PYTHONPATH": r"lib;C:\shared\lib"},
        environ_out = environ_out,
        syspath = syspath,
        real_paths = ("lib", r".\plugins", "modules.zip"),
    )
    assert (colour, chdir_calls) == ("yellow", [_RELOCATED])
    assert environ_out["PYTHONPATH"] == r"C:\Windows\System32\lib;C:\shared\lib"
    assert syspath == [
        r"C:\Python\Lib",
        r"C:\Windows\System32\lib",
        r"C:\Windows\System32",
        # join, not normpath: the same spelling the environment pinning uses.
        r"C:\Windows\System32\.\plugins",
        # Names nothing on disk: setuptools' editable sentinel, which its own path hook accepts back by exact string.
        "__editable__.unsloth-2026.8.15.finder.__path_hook__",
        # An archive that is really there is anchored like any other root.
        r"C:\Windows\System32\modules.zip",
    ]


def test_cli_guard_writes_back_only_an_expansion_the_reader_agrees_with():
    r"""Every reader expands once, so the guard expands once and writes the result
    back only when expanding it again would change nothing. When one pass still
    names one folder the value is left alone; when it does not, the folder depends
    on where the process is standing, so the move is refused rather than taken
    with the value silently following it."""
    environ_out: dict[str, str] = {}
    _message, colour, _chdir_calls = _guard_outcome(
        # An empty entry means the working directory, which is about to change.
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {
            "LOCALAPPDATA": r"C:\Users\me\AppData\Local",
            "HF_HUB_CACHE": r"%LOCALAPPDATA%\hub",
            # One pass leaves another reference, but it already names a drive, so it means the same folder from
            "HF_ASSETS_CACHE": r"C:\cache\%UNSET%\assets",
        },
        environ_out = environ_out,
    )
    assert colour == "yellow"
    # One pass settles it, so it names one folder and is written back.
    assert environ_out["HF_HUB_CACHE"] == r"C:\Users\me\AppData\Local\hub"
    assert environ_out["HF_ASSETS_CACHE"] == r"C:\cache\%UNSET%\assets"


def test_cli_guard_refuses_to_move_under_an_expansion_that_stays_relative():
    r"""A nested reference, an escaped %%NAME%% and a self-reference all leave one
    pass holding another reference, and the reader resolves what it gets against
    the working directory. Moving would take the value with it, so each of these
    refuses instead, naming the setting that could not be preserved."""
    for name, value, extra in (
        ("HF_ASSETS_CACHE", r"%NESTED%\assets", {"NESTED": r"%USERPROFILE%\AppData\Local"}),
        ("XDG_CACHE_HOME", r"%%USERPROFILE%%\xdg", {}),
        ("HF_HOME", r"%HF_HOME%\cache", {}),
    ):
        environ_out: dict[str, str] = {}
        message, colour, chdir_calls = _guard_outcome(
            r"C:\Windows\System32",
            ["unsloth", "studio", "--api-only"],
            environ_extra = {name: value, **extra},
            environ_out = environ_out,
        )
        assert (colour, chdir_calls) == ("red", []), name
        assert "path settings" in message, name
        assert name in message, name
        # Nothing is left rewritten by the attempt.
        assert environ_out[name] == value


def test_cli_guard_refuses_a_value_that_would_not_fit_in_the_environment():
    r"""A scalar that was already near the 32767-character limit crosses it once
    it names its folder in full, and a variable Windows will not accept is a
    failure to report here rather than in the next process."""
    # The limit is lowered rather than the value grown: the harness sets the environment through os.environ, and Windows
    # will not store 32767 characters there.
    with mock.patch.object(_system_dir_guard, "_WINDOWS_ENV_VALUE_LIMIT", 64):
        message, colour, chdir_calls = _guard_outcome(
            r"C:\Windows\System32",
            ["unsloth", "studio", "--api-only"],
            environ_extra = {"HF_HOME": "x" * 50},
        )
    assert (colour, chdir_calls) == ("red", [])
    assert "path settings" in message
    assert "HF_HOME" in message


def test_cli_guard_refuses_a_list_that_would_not_fit_in_the_environment():
    r"""Windows caps a variable at 32767 characters, and a list of relative
    entries can cross that once each names its folder in full. Reporting it here
    names the setting; discovering it when the next process starts does not."""
    # Same reason as the scalar above, and the raw list fits where the anchored one does not.
    entries = ";".join(["entry"] * 3)
    with mock.patch.object(_system_dir_guard, "_WINDOWS_ENV_VALUE_LIMIT", 64):
        message, colour, chdir_calls = _guard_outcome(
            r"C:\Windows\System32",
            ["unsloth", "studio", "--api-only"],
            environ_extra = {"PYTHONPATH": entries},
        )
    assert (colour, chdir_calls) == ("red", [])
    assert "path settings" in message
    assert "PYTHONPATH" in message


def test_cli_guard_never_refuses_an_update_over_the_local_checkout_setting():
    r"""A stale STUDIO_LOCAL_REPO on a drive that is gone must not stop the one
    update form that relocates: the bare update drops the value before anything
    reads it, so refusing over it would defeat the fallback this guard exists
    for. Every other setting still refuses, because something does read those."""
    environ_out: dict[str, str] = {}
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "update"],
        # Z: has no current directory, so this one cannot be resolved at all.
        environ_extra = {"STUDIO_LOCAL_REPO": "Z:checkout"},
        environ_out = environ_out,
    )
    assert (colour, chdir_calls) == ("yellow", [_RELOCATED])
    # Left exactly as written, and the update proceeds.
    assert environ_out["STUDIO_LOCAL_REPO"] == "Z:checkout"

    # The same value under a name something reads still stops the move.
    message, colour, _chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "update"],
        environ_extra = {"HF_HOME": "Z:cache"},
    )
    assert colour == "red"
    assert "path settings" in message


def test_cli_guard_refuses_a_path_attached_to_a_short_option():
    r"""Click reads `-f.\dist` as a value the same way it reads
    `--frontend=.\dist`, so a marked invocation carrying one is refused rather
    than rebased under the new folder."""
    for argv in (
        ["unsloth", "studio", "-f.\\dist"],
        ["unsloth", "studio", "--frontend=.\\dist"],
        ["unsloth", "studio", "-f", ".\\dist"],
    ):
        message, colour, chdir_calls = _guard_outcome(
            r"C:\Windows\System32",
            argv,
            environ_extra = {"UNSLOTH_DESKTOP_MANAGED": "1"},
        )
        assert (colour, chdir_calls) == ("red", []), argv
        assert "cannot run from" in message
    # The bare marked form the desktop actually runs still relocates.
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "desktop-handshake"],
        environ_extra = {"UNSLOTH_DESKTOP_MANAGED": "1"},
    )
    assert (colour, chdir_calls) == ("yellow", [_RELOCATED])


def test_cli_guard_goes_back_when_the_move_lands_somewhere_still_refused():
    r"""chdir can succeed into a folder the guard still refuses, through a
    junction or a profile that is itself inside the Windows tree. The process
    has to end up where it started, or the values written for the move would
    name folders under a directory nobody chose."""
    environ_out: dict[str, str] = {}
    message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {"HF_HOME": "cache"},
        environ_out = environ_out,
        # A junction under the profile:
        chdir_lands_in = r"C:\Windows\System32\config\systemprofile",
    )
    assert colour == "red"
    assert "cannot run from" in message
    # Back where it started, and the override is exactly as the caller wrote it.
    assert chdir_calls[-1] == r"C:\Windows\System32"
    assert environ_out["HF_HOME"] == "cache"


def test_cli_guard_restores_the_real_sys_path_when_the_move_fails():
    """The console script passes no list, so the guard has to snapshot the real
    sys.path before pinning it, or a chdir that then fails leaves the process
    carrying import roots it never agreed to."""
    import sys

    before = list(sys.path)
    sys.path.insert(0, "lib")
    try:
        _message, colour, chdir_calls = _guard_outcome(
            r"C:\Windows\System32",
            ["unsloth", "studio", "--api-only"],
            chdir_error = PermissionError("denied"),
            real_paths = ("lib",),
            pass_syspath = False,
        )
        assert colour == "red"
        assert chdir_calls == [_RELOCATED]
        assert sys.path[0] == "lib"
    finally:
        sys.path[:] = before


def test_cli_guard_pins_the_model_paths_llama_server_reads_for_itself():
    r"""llama-server resolves LLAMA_ARG_MODEL, LLAMA_ARG_MMPROJ and the draft
    spellings against its own working directory, and Unsloth reads them back when
    it sizes a launch, so a relative one has to move with the process. The URL
    spelling names no local file and is left alone."""
    environ_out: dict[str, str] = {}
    _message, colour, _chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {
            "LLAMA_ARG_MODEL": r"models\qwen.gguf",
            "LLAMA_ARG_MMPROJ": r".\mmproj.gguf",
            "LLAMA_ARG_MODEL_DRAFT": "draft.gguf",
            "LLAMA_ARG_SPEC_DRAFT_MODEL": r"D:\drafts\small.gguf",
            "LLAMA_ARG_MMPROJ_URL": "https://example.invalid/proj.gguf",
        },
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert environ_out["LLAMA_ARG_MODEL"] == r"C:\Windows\System32\models\qwen.gguf"
    assert environ_out["LLAMA_ARG_MMPROJ"] == r"C:\Windows\System32\.\mmproj.gguf"
    assert environ_out["LLAMA_ARG_MODEL_DRAFT"] == r"C:\Windows\System32\draft.gguf"
    assert environ_out["LLAMA_ARG_SPEC_DRAFT_MODEL"] == r"D:\drafts\small.gguf"
    assert environ_out["LLAMA_ARG_MMPROJ_URL"] == "https://example.invalid/proj.gguf"


def test_cli_guard_leaves_a_bracketed_directory_alone_only_where_it_is_json():
    r"""A folder really called "[llama]" is legal, so the inline-JSON exemption is
    scoped to the one variable whose reader accepts JSON."""
    environ_out: dict[str, str] = {}
    _message, colour, _chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {
            "UNSLOTH_LLAMA_CPP_PATH": "[llama]",
            "MLX_HOSTFILE": "[llama]",
            "UNSLOTH_STUDIO_HOME": "%data%",
        },
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert environ_out["UNSLOTH_LLAMA_CPP_PATH"] == r"C:\Windows\System32\[llama]"
    assert environ_out["UNSLOTH_STUDIO_HOME"] == r"C:\Windows\System32\%data%"
    assert environ_out["MLX_HOSTFILE"] == "[llama]"


@pytest.mark.skipif(importlib.util.find_spec("typer") is None, reason = "typer is not installed")
def test_cli_guard_reads_the_invocation_not_the_hosts_argv():
    """A host that imports the app and calls it never touches sys.argv, so the
    callback has to classify the arguments Click was given: `studio --api-only`
    from a host whose own argv says `train --dataset .\\d.json`, and the reverse."""
    from typer.testing import CliRunner

    import unsloth_cli

    seen: list[list[str]] = []
    original = unsloth_cli._check_working_directory
    unsloth_cli._check_working_directory = lambda argv, environ, platform, **kw: (
        seen.append(list(argv)) or (None, None, False)
    )
    try:
        CliRunner().invoke(unsloth_cli.app, ["studio", "--help"])
        CliRunner().invoke(unsloth_cli.app, ["train", "--help"])
    finally:
        unsloth_cli._check_working_directory = original
    assert seen == [["studio", "--help"], ["train", "--help"]]


def test_cli_guard_pins_the_token_path_and_the_special_pythonpath_entries():
    r"""HF_TOKEN_PATH is where huggingface_hub reads the credential file, and
    PYTHONPATH has two spellings that follow the process: an empty component means
    the working directory, and `~` is never expanded there."""
    environ_out: dict[str, str] = {}
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {
            "HF_TOKEN_PATH": r"secrets\token",
            "PYTHONPATH": r";~\plugins;C:\shared\lib",
        },
        environ_out = environ_out,
    )
    assert (colour, chdir_calls) == ("yellow", [_RELOCATED])
    assert environ_out["HF_TOKEN_PATH"] == r"C:\Windows\System32\secrets\token"
    assert environ_out["PYTHONPATH"] == (
        # The empty component is the folder being left;
        r"C:\Windows\System32;C:\Windows\System32\~\plugins;C:\shared\lib"
    )


def test_cli_guard_says_which_setting_stopped_the_move():
    r"""A value that cannot be pinned is a different failure from a missing
    profile: an oversized PYTHONPATH (Windows caps a variable at 32767 chars) or
    a drive with no current directory of its own. Blaming the user folder for it
    sends the reader looking in the wrong place."""
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        # No current directory for drive D, so the OS cannot resolve this.
        environ_extra = {"HF_HOME": "D:cache"},
        drive_cwd = {},
    )
    assert (colour, chdir_calls) == ("red", [])
    assert _message is not None
    assert "path settings" in _message
    assert "user profile" not in _message


def test_cli_guard_expands_a_hyphenated_name_the_way_ntpath_does():
    r"""ntpath counts the hyphen as part of a $ name, so $CACHE-ROOT is one
    variable and not $CACHE followed by -ROOT. The desktop twin has to agree, or
    the two layers send the same install to two different folders."""
    environ_out: dict[str, str] = {}
    _message, colour, _chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {
            "CACHE": r"C:\wrong",
            "CACHE-ROOT": r"C:\right",
            "HF_HOME": r"$CACHE-ROOT\hf",
        },
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert environ_out["HF_HOME"] == r"C:\right\hf"


def test_cli_guard_puts_back_what_it_pinned_when_the_move_fails():
    r"""Inside a host process the environment and sys.path are the caller's, so a
    chdir that fails must not leave them rewritten as though it had moved."""
    environ_out: dict[str, str] = {}
    syspath = ["lib", r"C:\Python\Lib"]
    _message, colour, chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {"HF_HOME": "cache", "PYTHONPATH": "lib"},
        environ_out = environ_out,
        syspath = syspath,
        real_paths = ("lib",),
        chdir_error = PermissionError("access denied"),
    )
    assert (colour, chdir_calls) == ("red", [_RELOCATED])
    assert environ_out["HF_HOME"] == "cache"
    assert environ_out["PYTHONPATH"] == "lib"
    assert syspath == ["lib", r"C:\Python\Lib"]


def test_cli_guard_will_not_relocate_once_the_commands_are_imported():
    r"""A host that imports the app has already resolved STUDIO_HOME, so a move
    from the callback would leave that cached root in the folder being left."""
    outcome = _system_dir_guard.check_working_directory(
        ["studio", "--api-only"],
        {"WINDIR": r"C:\Windows", "USERPROFILE": r"C:\Users\me"},
        "win32",
        getcwd = lambda: r"C:\Windows\System32",
        chdir = lambda _target: pytest.fail("a library call must not move the host"),
        pathmod = ntpath,
        sep = "\\",
        expanduser = lambda _p: r"C:\Users\me",
        isdir = lambda path: ntpath.normcase(path) == ntpath.normcase(r"C:\Windows\System32"),
        relocate = False,
    )
    message, colour, fatal = outcome
    assert (colour, fatal) == ("red", True)
    assert message is not None


def test_cli_guard_treats_the_jaccl_device_file_like_the_host_file():
    r"""MLX_IBV_DEVICES is read exactly like MLX_HOSTFILE (`_json_rank_count_from_env`):
    either inline JSON or a filename, so it is pinned and the JSON left alone."""
    environ_out: dict[str, str] = {}
    _message, colour, _chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {"MLX_IBV_DEVICES": "devices.json"},
        environ_out = environ_out,
    )
    assert colour == "yellow"
    assert environ_out["MLX_IBV_DEVICES"] == r"C:\Windows\System32\devices.json"

    environ_out.clear()
    _message, colour, _chdir_calls = _guard_outcome(
        r"C:\Windows\System32",
        ["unsloth", "studio", "--api-only"],
        environ_extra = {"MLX_IBV_DEVICES": '[{"device": "mlx5_0"}]'},
        environ_out = environ_out,
    )
    assert environ_out["MLX_IBV_DEVICES"] == '[{"device": "mlx5_0"}]'
