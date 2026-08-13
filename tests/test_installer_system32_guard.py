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
from pathlib import Path

import pytest

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


# ── install.ps1: relocate before doing any work ──


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

    # ntpath for the path semantics, with expanduser pinned: the real one reads the host's
    # HOME, and on Windows "~" is USERPROFILE, SYSTEM's included.
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
        current["cwd"] = target

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
        expanduser = lambda _p: userprofile,
        makedirs = _makedirs,
        # Only a folder that really holds System32 counts as a Windows directory.
        isdir = lambda path: ntpath.normcase(path) in real_windows_dirs,
        abspath = _abspath,
        home_isdir = lambda path: ntpath.normcase(path)
        not in {ntpath.normcase(home) for home in missing_homes},
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


# ── "Run Unsloth at login" (issue #8510): the desktop cannot choose its own cwd ──
#
# Windows registers login startup as an HKCU Run value, which carries no working
# directory, so Unsloth Desktop and every CLI child it spawns start in System32.
# The commands it runs take no path from the user, so they move out of the folder
# instead of refusing and leaving the user with a tray icon and no server.

_RELOCATED = r"C:\Users\me\.unsloth"


@pytest.mark.parametrize(
    "argv",
    [
        ["unsloth", "studio", "--api-only", "-H", "127.0.0.1", "-p", "8888"],
        ["unsloth", "studio", "--api-only"],
        ["unsloth", "studio", "provision-desktop-auth"],
        ["unsloth", "studio", "desktop-capabilities", "--json"],
        # The command that upgrades a desktop too old to set the marker; without
        # it such a user gets a working backend and no way to update from the tray.
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
        # `studio run` declares its own --api-only and takes user paths, so it
        # must not be mistaken for the desktop's backend launch.
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
    """Studio has to write under the home anyway, and the Rust half stops here too."""
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
    # the reader to cd somewhere or claim they used "Run as administrator".
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
    """Studio resolves the override with Path.resolve(), which anchors a relative
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
    """storage_roots.py owns the Studio folders, and each of its overrides is a
    plain user-supplied path, so a relative one must be pinned before the move.
    Reading them from the module keeps the guard honest as roots are added."""
    storage_roots = REPO_ROOT / "studio" / "backend" / "utils" / "paths" / "storage_roots.py"
    source = storage_roots.read_text(encoding = "utf-8")
    overrides = set(re.findall(r'environ\.get\(\s*"(UNSLOTH_[A-Z_]*(?:HOME|PATH|DIR))"', source))
    assert overrides, "no storage root overrides found: has storage_roots.py moved?"
    missing = sorted(overrides - set(_system_dir_guard._RELATIVE_PATH_ENV))
    assert not missing, f"relative values of {missing} would be retargeted by the move"


@pytest.mark.parametrize("value", [r"C:\elsewhere\custom", r"~\custom", r"\\server\share\c"])
def test_cli_guard_leaves_an_already_anchored_override_alone(value: str):
    """An absolute, UNC or ~ value does not depend on the working directory."""
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
