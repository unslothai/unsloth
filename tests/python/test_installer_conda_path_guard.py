# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""No installer PATH write may demote an ACTIVE conda environment.

A prepend into the persistent User PATH outlives the activation it was made under, so from
the next shell on conda resolves binaries and DLLs out of our directory. That is #5871:
after installing Studio, `conda update --all --yes` leaves the Anaconda base unusable with
`ImportError: DLL load failed while importing _ctypes`.

One decision function per installer, with every persistent write routed through it, so a
new call site cannot reintroduce the defect. Asserted here: the decision, that both
PowerShell installers use it, and that the system-Python installer switch follows it. The
POSIX half is tests/sh/test_install_conda_path_guard.sh.
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
INSTALL_SH = REPO_ROOT / "install.sh"
SETUP_SH_POSIX = REPO_ROOT / "studio" / "setup.sh"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]

# install.ps1 nests its helpers inside Install-UnslothStudio, setup.ps1 defines them at top
# level, so the indent is part of the pattern rather than something to normalise away.
_PS_FILES = {
    "install.ps1": (INSTALL_PS1, "    "),
    "studio/setup.ps1": (SETUP_PS1, ""),
}


def _function(path: Path, indent: str, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"{indent}function {name} \{{.*?\n{indent}\}}\n", source, flags=re.DOTALL)
    assert match is not None, f"{path.name} no longer defines {name}"
    return match.group(0)


def _run(
    shell: str,
    script: str,
    env: dict[str, str] | None = None,
) -> str:
    base = {k: v for k, v in os.environ.items() if not k.startswith("CONDA_")}
    result = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        env={**base, **(env or {})},
        timeout=60,
    )
    return result.stdout.strip()


_NO_CONDA = {}
_ACTIVE_CONDA = {"CONDA_PREFIX": "E:\\anaconda\\install", "CONDA_DEFAULT_ENV": "base"}
_PREFIX_ONLY = {"CONDA_PREFIX": "E:\\anaconda\\install"}
# A hook that exports only CONDA_DEFAULT_ENV still leaves the caller inside conda's
# ordering, so it counts too.
_NAME_ONLY = {"CONDA_DEFAULT_ENV": "base"}
# Set but empty is not an activation: conda's own deactivate clears these by emptying them.
_EMPTY = {"CONDA_PREFIX": "", "CONDA_DEFAULT_ENV": ""}


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
@pytest.mark.parametrize(
    "requested,env,expected",
    [
        ("Prepend", _NO_CONDA, "Prepend"),
        ("Prepend", _ACTIVE_CONDA, "Append"),
        ("Prepend", _PREFIX_ONLY, "Append"),
        ("Prepend", _NAME_ONLY, "Append"),
        ("Prepend", _EMPTY, "Prepend"),
        # An append was already behind everything; conda changes nothing about it.
        ("Append", _ACTIVE_CONDA, "Append"),
        ("Append", _NO_CONDA, "Append"),
    ],
    ids=[
        "prepend-no-conda",
        "prepend-active-conda",
        "prepend-prefix-only",
        "prepend-name-only",
        "prepend-empty-vars",
        "append-active-conda",
        "append-no-conda",
    ],
)
def test_a_prepend_becomes_an_append_inside_an_active_conda_environment(
    shell: str, path_label: str, requested: str, env: dict[str, str], expected: str
):
    path, indent = _PS_FILES[path_label]
    script = (
        _function(path, indent, "Test-ActiveCondaEnvironment")
        + _function(path, indent, "Resolve-UserPathPosition")
        + f"\nWrite-Host ('POSITION=' + (Resolve-UserPathPosition -Position '{requested}'))\n"
    )
    out = _run(shell, script, env=env)
    assert f"POSITION={expected}" in out


@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
def test_every_persistent_path_write_routes_through_the_guard(path_label: str):
    """Add-ToUserPath is the only persistent PATH writer, and it must not be bypassable.

    Asserted on the function body rather than on a call site, because the call sites are
    the thing that keeps changing: install.ps1 prepends the uv destination, the shim
    directory and the shortcut shim directory, setup.ps1 five more, and a sixth added later
    must inherit the guard without anyone remembering it exists.
    """
    path, indent = _PS_FILES[path_label]
    body = _function(path, indent, "Add-ToUserPath")
    assert (
        "Resolve-UserPathPosition" in body
    ), f"{path_label}: Add-ToUserPath must resolve its position through the guard"
    # Ahead of the registry read, or the decision lands after the value it decides is used.
    assert body.index("Resolve-UserPathPosition") < body.index(
        "Registry"
    ), f"{path_label}: the guard has to run before the registry write is prepared"


@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
def test_a_downgraded_prepend_still_repositions_an_existing_front_entry(path_label: str):
    """Declining to ADD is not the same as declining to DEMOTE.

    A machine that ran the installer once outside conda has our directory at the FRONT of
    the User PATH. Rerunning inside conda downgrades the Prepend to an Append, but an
    unconditional "already present, nothing to do" early return leaves that front entry in
    place, so the run says "conda keeps priority" while #5871 stays armed.

    Structural, because Add-ToUserPath writes through [Microsoft.Win32.Registry], which does
    not exist off Windows.
    """
    path, indent = _PS_FILES[path_label]
    body = _function(path, indent, "Add-ToUserPath")
    match = re.search(r"\$alreadyPresent -and \$Position -eq 'Append'([^\)]*)\)", body)
    assert match is not None, f"{path_label}: no append early-return found in Add-ToUserPath"
    assert "-not $positionDowngraded" in match.group(1), (
        f"{path_label}: the append early-return must not fire for a prepend that conda "
        f"downgraded, or an existing front entry is never moved to the back"
    )
    # The flag has to be derived from the guard's own answer, not hardcoded.
    assert re.search(
        r"\$positionDowngraded\s*=\s*\(", body
    ), f"{path_label}: $positionDowngraded must be computed from Resolve-UserPathPosition"


@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
def test_the_two_installers_agree_on_the_guard(path_label: str):
    """Both copies decide the same way. They are separate scripts, not a shared module."""
    path, indent = _PS_FILES[path_label]
    body = _function(path, indent, "Resolve-UserPathPosition")
    assert "Test-ActiveCondaEnvironment" in body
    assert "'Append'" in body
    conda = _function(path, indent, "Test-ActiveCondaEnvironment")
    assert "CONDA_PREFIX" in conda
    assert "CONDA_DEFAULT_ENV" in conda


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "env,expected",
    [(_NO_CONDA, "PrependPath=1"), (_ACTIVE_CONDA, "AppendPath=1"), (_EMPTY, "PrependPath=1")],
    ids=["no-conda", "active-conda", "empty-vars"],
)
def test_the_system_python_install_follows_the_same_rule(
    shell: str, env: dict[str, str], expected: str
):
    """The reporter's own diagnosis was the system-wide Python, not our own directories."""
    script = (
        _function(INSTALL_PS1, "    ", "Test-ActiveCondaEnvironment")
        + _function(INSTALL_PS1, "    ", "Get-PythonInstallerPathSwitch")
        + "\nWrite-Host ('SWITCH=' + (Get-PythonInstallerPathSwitch))\n"
    )
    out = _run(shell, script, env=env)
    assert f"SWITCH={expected}" in out


def test_the_python_org_installer_arguments_use_the_switch_function():
    """A literal PrependPath=1 in the argument list would bypass the decision entirely."""
    source = INSTALL_PS1.read_text(encoding="utf-8")
    args = re.search(r"\$installArgs = @\(.*?\n        \)", source, flags=re.DOTALL)
    assert args is not None, "install.ps1 no longer builds $installArgs for python.org"
    block = args.group(0)
    assert "Get-PythonInstallerPathSwitch" in block
    assert (
        "PrependPath" not in block
    ), "the PATH switch has to come from Get-PythonInstallerPathSwitch, not be spelled here"


def test_winget_is_not_the_first_route_inside_an_active_conda_environment():
    """winget's Python package hard-codes PrependPath=1, so it cannot be asked politely."""
    source = INSTALL_PS1.read_text(encoding="utf-8")
    start = source.index('substep "installing Python ${PythonVersion}..."')
    conda_first = source.index("if (Test-ActiveCondaEnvironment)", start)
    winget = source.index("if ($script:WingetAvailable", start)
    assert conda_first < winget, "the conda-safe route has to be attempted before winget"
    # And the winget arm must now respect an interpreter the conda route already found.
    arm = source[winget : source.index("\n", winget)]
    assert "-not $DetectedPython" in arm, (
        "winget would otherwise install a second Python over the one just installed, with "
        "PrependPath=1, undoing the guard"
    )


@pytest.mark.parametrize(
    "path,funcs",
    [
        (INSTALL_SH, ("_persist_login_path_dir", "_persist_fish_path_dir")),
        (SETUP_SH_POSIX, ("_setup_persist_uv_path",)),
    ],
    ids=["install.sh", "studio/setup.sh"],
)
def test_the_posix_installers_carry_the_same_guard(path: Path, funcs: tuple[str, ...]):
    """Cross-platform parity, asserted here so the PowerShell and POSIX halves cannot drift.

    The behavioural cases are in tests/sh/test_install_conda_path_guard.sh, which runs the
    extracted shell functions; this only holds that the guard exists at all, so a rewrite of
    a PATH persistence block on the POSIX side cannot quietly drop it. Four writers in all:
    the registry in install.ps1 and setup.ps1, the rc files in install.sh and setup.sh.
    """
    source = path.read_text(encoding="utf-8")
    for func in funcs:
        body = re.search(rf"\n{func}\(\) \{{.*?\n\}}\n", source, flags=re.DOTALL)
        assert body is not None, f"{path.name} no longer defines {func}"
        assert "_unsloth_conda_env_active" in body.group(0), f"{func} must not prepend inside conda"


@pytest.mark.parametrize(
    "path", [INSTALL_SH, SETUP_SH_POSIX], ids=["install.sh", "studio/setup.sh"]
)
def test_both_halves_call_the_same_environment_an_active_conda(path: Path):
    """The PowerShell and POSIX halves must agree on WHICH variables mean "inside conda"."""
    posix = path.read_text(encoding="utf-8")
    body = re.search(r"\n_unsloth_conda_env_active\(\) \{.*?\n\}\n", posix, flags=re.DOTALL)
    assert body is not None, f"{path.name} no longer defines _unsloth_conda_env_active"
    posix_vars = set(re.findall(r"\bCONDA_[A-Z_]+", body.group(0)))

    windows = INSTALL_PS1.read_text(encoding="utf-8")
    probe = re.search(
        r"function Test-ActiveCondaEnvironment \{.*?\n    \}", windows, flags=re.DOTALL
    )
    assert probe is not None, "install.ps1 no longer defines Test-ActiveCondaEnvironment"
    windows_vars = set(re.findall(r"\bCONDA_[A-Z_]+", probe.group(0)))

    assert posix_vars == windows_vars, (
        f"{path.name} reads {sorted(posix_vars)} but install.ps1 reads " f"{sorted(windows_vars)}"
    )


# ── The LIVE session PATH, not only the registry ──
# The registry half above fixes the NEXT shell. `irm ... | iex` is the documented command,
# and environment changes made by `iex` stay in the caller's own process, so the PATH the
# installer leaves behind IS that prompt. Rebuilt as machine + user + previous, every
# activated-conda entry lands behind the User PATH, because none of it is in the registry at
# all: `python` and `conda update` then resolve the non-conda install first, in a session the
# installer has just told the user keeps conda's priority.


def _refresh_preamble(current_path: str) -> str:
    """Refresh-SessionPath with the two registry reads stubbed and $env:Path seeded."""
    body = (
        _function(INSTALL_PS1, "    ", "Test-ActiveCondaEnvironment")
        + _function(INSTALL_PS1, "    ", "Get-ActiveCondaPrefixes")
        + _function(INSTALL_PS1, "    ", "Test-PathUnderCondaPrefix")
        + _function(INSTALL_PS1, "    ", "Refresh-SessionPath")
    )
    return f"""
$ErrorActionPreference = "Stop"
{body}
# The suite itself runs inside a virtualenv, and Refresh-SessionPath puts $VIRTUAL_ENV
# first. That branch is not what these cases are about, and leaving it on would make the
# expected orderings depend on where the suite was launched from.
$env:VIRTUAL_ENV = ""
$env:Path = "{current_path}"
Refresh-SessionPath
Write-Host ("PATH=" + $env:Path)
"""


MACHINE = "C:\\Windows\\system32;C:\\Windows"
USER_PATH = "C:\\Users\\me\\.local\\bin;C:\\Users\\me\\AppData\\Local\\Programs\\Python\\Python312"
CONDA_ROOT = "E:\\anaconda\\install"
_CONDA_ENTRIES = (
    f"{CONDA_ROOT}\\envs\\ml;"
    f"{CONDA_ROOT}\\envs\\ml\\Library\\bin;"
    f"{CONDA_ROOT}\\envs\\ml\\Scripts;"
    f"{CONDA_ROOT}\\Scripts"
)


def _stub_registry(script: str) -> str:
    """Answer the two registry reads without touching the machine running the suite."""
    return script.replace(
        '$machine = [System.Environment]::GetEnvironmentVariable("Path", "Machine")',
        f'$machine = "{MACHINE}"',
    ).replace(
        '$user    = [System.Environment]::GetEnvironmentVariable("Path", "User")',
        f'$user    = "{USER_PATH}"',
    )


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_session_refresh_keeps_an_active_conda_ahead_of_the_user_path(shell: str):
    """The case #5871 is about, in the prompt the user is standing in.

    Without this the message that conda keeps priority is false for the rest of that
    session: the registry ordering is right and the live PATH is not.
    """
    script = _stub_registry(_refresh_preamble(f"{_CONDA_ENTRIES};{MACHINE}"))
    out = _run(
        shell,
        script,
        env={
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\ml",
            "CONDA_DEFAULT_ENV": "ml",
            "CONDA_EXE": f"{CONDA_ROOT}\\Scripts\\conda.exe",
        },
    )
    assert out.startswith("PATH="), out
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]

    conda_positions = [
        index for index, entry in enumerate(lowered) if entry.startswith(CONDA_ROOT.lower())
    ]
    assert conda_positions, f"the conda entries were dropped entirely: {entries}"
    user_first = min(
        index for index, entry in enumerate(lowered) if entry.startswith("c:\\users\\me")
    )
    assert (
        max(conda_positions) < user_first
    ), f"an activated conda environment was left behind the User PATH: {entries}"
    # Conda's own ordering inside the environment is preserved, not re-sorted.
    assert lowered[:4] == [
        f"{CONDA_ROOT}\\envs\\ml".lower(),
        f"{CONDA_ROOT}\\envs\\ml\\Library\\bin".lower(),
        f"{CONDA_ROOT}\\envs\\ml\\Scripts".lower(),
        f"{CONDA_ROOT}\\Scripts".lower(),
    ], entries
    # And nothing is duplicated by the move.
    assert len(lowered) == len(set(lowered)), entries


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_session_refresh_is_unchanged_outside_conda(shell: str):
    """No conda, no reordering. The refresh is on the hot path of the whole installer and
    must keep doing exactly what it did."""
    previous = "C:\\tools\\bin"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(shell, script, env=_NO_CONDA)
    entries = out[len("PATH=") :].split(";")
    assert entries == MACHINE.split(";") + USER_PATH.split(";") + [previous], entries


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_sibling_directory_is_not_dragged_forward_with_the_prefix(shell: str):
    """`C:\\conda-backup` is not inside `C:\\conda`. Matching on the raw string would move an
    unrelated directory to the front of PATH, which is the same class of defect this change
    exists to prevent, pointed the other way."""
    sibling = f"{CONDA_ROOT}-backup\\bin"
    script = _stub_registry(_refresh_preamble(f"{sibling};{CONDA_ROOT}\\Scripts"))
    out = _run(
        shell,
        script,
        env={
            "CONDA_PREFIX": CONDA_ROOT,
            "CONDA_DEFAULT_ENV": "base",
        },
    )
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]
    assert lowered[0] == f"{CONDA_ROOT}\\scripts".lower(), entries
    assert lowered.index(sibling.lower()) > lowered.index(
        USER_PATH.split(";")[0].lower()
    ), f"a sibling of the conda prefix was promoted with it: {entries}"


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_stacked_conda_activation_keeps_both_prefixes_in_front(shell: str):
    """`conda activate` inside another environment stacks them and records the outer one in
    CONDA_PREFIX_1. Its entries are on PATH exactly as the inner one's are."""
    outer = f"{CONDA_ROOT}\\envs\\outer\\Scripts"
    inner = f"{CONDA_ROOT}\\envs\\inner\\Scripts"
    script = _stub_registry(_refresh_preamble(f"{inner};{outer}"))
    out = _run(
        shell,
        script,
        env={
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\inner",
            "CONDA_PREFIX_1": f"{CONDA_ROOT}\\envs\\outer",
            "CONDA_DEFAULT_ENV": "inner",
        },
    )
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]
    assert lowered[:2] == [inner.lower(), outer.lower()], entries


def test_the_refresh_consults_the_conda_helper_at_all():
    """The three helpers are only worth anything if Refresh-SessionPath calls them, and a
    refresh that rebuilt PATH the old way would leave every case above untouched if they
    were driven directly instead."""
    body = _function(INSTALL_PS1, "    ", "Refresh-SessionPath")
    assert "Test-ActiveCondaEnvironment" in body
    assert "Get-ActiveCondaPrefixes" in body
    assert "Test-PathUnderCondaPrefix" in body
    # And the conda entries go in FRONT of the registry values, which is the whole point.
    assert body.index("$sources += $condaFront") < body.index(
        "$sources += @($machine"
    ), "the conda entries are appended after the User PATH, which changes nothing"


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_name_only_activation_keeps_the_session_it_already_had(shell: str):
    """A hook that exports CONDA_DEFAULT_ENV and nothing that names a directory.

    Test-ActiveCondaEnvironment accepts that shape on purpose, but nothing then identifies
    which PATH entries are conda's, so the refresh would rebuild PATH as Machine, User,
    previous and put the hook's entries behind the User PATH again. Guessing from the
    environment NAME would promote any directory containing it, so the caller's PATH is kept
    whole and in front instead.
    """
    previous = f"{CONDA_ROOT}\\envs\\ml\\Scripts;C:\\tools\\bin"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(shell, script, env={"CONDA_DEFAULT_ENV": "ml"})
    entries = out[len("PATH=") :].split(";")
    assert entries[: len(previous.split(";"))] == previous.split(";"), entries
    # The registry values are still there, just behind, so a directory registered by this
    # run is reachable in the same session without displacing the environment.
    assert USER_PATH.split(";")[0] in entries, entries
    assert MACHINE.split(";")[0] in entries, entries
    assert entries.index(USER_PATH.split(";")[0]) > entries.index("C:\\tools\\bin"), entries


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_prefix_activation_still_takes_the_precise_route(shell: str):
    """The whole-PATH fallback is for the case where nothing better can be known, so an
    activation that DOES name its prefix must keep the targeted behaviour: only the conda
    entries move, and an unrelated directory that happened to be on PATH does not."""
    previous = f"C:\\tools\\bin;{CONDA_ROOT}\\envs\\ml\\Scripts"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(
        shell,
        script,
        env={
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\ml",
            "CONDA_DEFAULT_ENV": "ml",
        },
    )
    entries = out[len("PATH=") :].split(";")
    assert entries[0] == f"{CONDA_ROOT}\\envs\\ml\\Scripts", entries
    assert entries.index("C:\\tools\\bin") > entries.index(USER_PATH.split(";")[0]), entries


# ── setup.ps1 refreshes the session too ──
# install.ps1 hands off to studio/setup.ps1, which calls Refresh-Environment after it
# registers CMake or Python. Direct execution shares the caller's PowerShell process, so a
# rebuild as machine + user + previous demotes the active conda environment for the rest of
# the setup AND after it returns, which is the same defect in the second half of the install.


def _setup_refresh_preamble(current_path: str) -> str:
    """Refresh-Environment with the registry reads stubbed and $env:Path seeded."""
    body = (
        _function(SETUP_PS1, "", "Test-ActiveCondaEnvironment")
        + _function(SETUP_PS1, "", "Get-ActiveCondaPrefixes")
        + _function(SETUP_PS1, "", "Test-PathUnderCondaPrefix")
        + _function(SETUP_PS1, "", "Refresh-Environment")
    )
    return f"""
$ErrorActionPreference = "Stop"
{body}
$env:VIRTUAL_ENV = ""
$env:Path = "{current_path}"
Refresh-Environment
Write-Host ("PATH=" + $env:Path)
"""


def _stub_setup_registry(script: str) -> str:
    return script.replace(
        "$machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')",
        f'$machinePath = "{MACHINE}"',
    ).replace(
        "$userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')",
        f'$userPath = "{USER_PATH}"',
    )


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_setup_refresh_keeps_an_active_conda_ahead_of_the_user_path(shell: str):
    script = _stub_setup_registry(_setup_refresh_preamble(f"{_CONDA_ENTRIES};{MACHINE}"))
    assert "GetEnvironmentVariable('Path'" not in script, "the registry stub stopped matching"
    out = _run(
        shell,
        script,
        env={
            "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\ml",
            "CONDA_DEFAULT_ENV": "ml",
            "CONDA_EXE": f"{CONDA_ROOT}\\Scripts\\conda.exe",
        },
    )
    assert out.startswith("PATH="), out
    entries = out[len("PATH=") :].split(";")
    lowered = [entry.rstrip("\\").lower() for entry in entries]
    conda_positions = [
        index for index, entry in enumerate(lowered) if entry.startswith(CONDA_ROOT.lower())
    ]
    assert conda_positions, f"the conda entries were dropped entirely: {entries}"
    user_first = min(
        index for index, entry in enumerate(lowered) if entry.startswith("c:\\users\\me")
    )
    assert (
        max(conda_positions) < user_first
    ), f"setup.ps1 left an activated conda environment behind the User PATH: {entries}"
    assert len(lowered) == len(set(lowered)), entries


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_setup_refresh_is_unchanged_outside_conda(shell: str):
    """Refresh-Environment runs on the hot path of the whole setup and must keep doing
    exactly what it did when no conda environment is active."""
    previous = "C:\\tools\\bin"
    script = _stub_setup_registry(_setup_refresh_preamble(previous))
    out = _run(shell, script, env=_NO_CONDA)
    entries = out[len("PATH=") :].split(";")
    assert entries == MACHINE.split(";") + USER_PATH.split(";") + [previous], entries


def test_the_setup_refresh_consults_the_conda_helper_at_all():
    """A static pair for the pwsh cases above, so a Windows-only regression is still caught
    on a runner without PowerShell."""
    body = _function(SETUP_PS1, "", "Refresh-Environment")
    assert "Test-ActiveCondaEnvironment" in body, body
    assert "Get-ActiveCondaPrefixes" in body, body
    assert "Test-PathUnderCondaPrefix" in body, body


# ── a prepend a previous run persisted has to be REPOSITIONED, not accepted ──
# The presence checks accept any spelling, so they never add a second line. That is right for
# the duplicate it prevents and wrong for the one it hides: a run from outside conda writes the
# prepend, and a later run from inside one then finds "the directory is already there" and
# leaves our entry ahead of the environment in every shell from then on.


def _shell_function(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\(\) \{{.*?^\}}\n", source, flags=re.DOTALL | re.M)
    assert match is not None, f"{path.name} no longer defines {name}"
    return match.group(0)


@pytest.mark.parametrize("path", [INSTALL_SH, SETUP_SH_POSIX], ids=["install.sh", "setup.sh"])
def test_the_rc_repointer_rewrites_only_the_line_it_wrote(path: Path, tmp_path: Path):
    rc = tmp_path / "rc"
    rc.write_text(
        "# Added by Unsloth installer\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        "# a comment\n"
        'export PATH="/opt/mine:$PATH"\n'
        "alias ll='ls -l'\n",
        encoding="utf-8",
    )
    script = (
        _shell_function(path, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(
        ["sh", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "status=0" in out.stdout, (out.stdout, out.stderr)
    assert rc.read_text(encoding="utf-8") == (
        "# Added by Unsloth installer\n"
        'export PATH="$PATH:$HOME/.local/bin"\n'
        "# a comment\n"
        # The user's own line is untouched: only an exact match on what we write is ours.
        'export PATH="/opt/mine:$PATH"\n'
        "alias ll='ls -l'\n"
    )
    # And no staging file is left in the user's home.
    assert [p.name for p in tmp_path.iterdir()] == ["rc"]


def test_the_rc_repointer_keeps_a_symlinked_rc_a_symlink(tmp_path: Path):
    """An rc file managed by chezmoi or home-manager is a symlink into a repo. Replacing it
    with a fresh file would detach it from the thing that manages it."""
    real = tmp_path / "dotfiles" / "bashrc"
    real.parent.mkdir()
    real.write_text(
        '# Added by Unsloth installer\nexport PATH="$HOME/.local/bin:$PATH"\n',
        encoding="utf-8",
    )
    link = tmp_path / ".bashrc"
    link.symlink_to(real)

    script = (
        _shell_function(INSTALL_SH, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{link}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
    )
    subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    assert link.is_symlink(), "the rc file was replaced instead of rewritten"
    assert real.read_text(encoding="utf-8") == (
        '# Added by Unsloth installer\nexport PATH="$PATH:$HOME/.local/bin"\n'
    )


@pytest.mark.parametrize(
    "path,arms",
    [
        (INSTALL_SH, ("_persist_login_path_dir", "_persist_fish_path_dir")),
        (SETUP_SH_POSIX, ("_setup_persist_uv_path",)),
    ],
    ids=["install.sh", "studio/setup.sh"],
)
def test_every_posix_writer_repositions_a_stale_prepend(path: Path, arms: tuple[str, ...]):
    """A guard that only applies to a FIRST run is not a guard: every one of these writers is
    reached again by the next `irm | iex`."""
    for name in arms:
        body = _shell_function(path, name)
        assert (
            "_unsloth_repoint_rc_line" in body
        ), f"{path.name}:{name} accepts a stale prepend as present instead of moving it"


def test_the_repoint_pass_runs_before_the_presence_guards():
    """The guard that skips the writers is satisfied by the very line that needs moving.

    A previous run wrote the prepend, the shell that launched this installer evaluated it,
    so the directory IS on the login PATH and `_path_has_dir` short-circuits the call. The
    repointing therefore has to happen ahead of that guard, and it has to add nothing when
    it gets there.
    """
    source = INSTALL_SH.read_text(encoding="utf-8")
    shim_guard = source.index('if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_LOCAL_BIN"')
    shim_repoint = source.index('"~/.local/bin" \'\\.local/bin\' "" repoint')
    assert shim_repoint < shim_guard, "the shim repoint is behind the guard that skips it"

    uv_guard = source.index('if ! _path_has_dir "$_UNSLOTH_LOGIN_PATH" "$_UNSLOTH_UV_BIN_DIR"')
    uv_repoint = source.index('"$_UNSLOTH_UV_BIN_DIR" "" "$_uv_prof" repoint')
    assert uv_repoint < uv_guard, "the uv repoint is behind the guard that skips it"

    setup = SETUP_SH_POSIX.read_text(encoding="utf-8")
    early_return = setup.index('_setup_path_has_dir "${_SETUP_LOGIN_PATH:-$PATH}" "$_supp_dir"')
    assert (
        "_unsloth_conda_env_active || return 0" in setup[early_return : early_return + 300]
    ), "studio/setup.sh still returns before it can reposition anything"


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("repoint", '# Added by Unsloth installer\nexport PATH="$PATH:/opt/unsloth/bin"\n'),
        ("", '# Added by Unsloth installer\nexport PATH="$PATH:/opt/unsloth/bin"\n'),
    ],
    ids=["repoint-only", "ordinary"],
)
def test_the_repoint_only_mode_adds_nothing(tmp_path: Path, mode: str, expected: str):
    """Repointing must not put a line in the rc file of someone whose PATH comes from
    somewhere else: the caller's guard already decided against adding one."""
    rc = tmp_path / "rc"
    rc.write_text(
        '# Added by Unsloth installer\nexport PATH="/opt/unsloth/bin:$PATH"\n',
        encoding="utf-8",
    )
    empty = tmp_path / "empty"
    empty.write_text("# nothing here\n", encoding="utf-8")
    source = INSTALL_SH.read_text(encoding="utf-8")
    body = (
        _shell_function(INSTALL_SH, "_unsloth_repoint_rc_line")
        + _shell_function(INSTALL_SH, "_unsloth_conda_env_active")
        + _shell_function(INSTALL_SH, "_persist_fish_path_dir")
        + re.search(r"^_PATH_LINE_RE=.*$", source, flags=re.M).group(0)
        + "\n"
        + _shell_function(INSTALL_SH, "_persist_login_path_dir")
    )
    script = f"""
CONDA_PREFIX=/opt/conda
step() {{ :; }}
substep() {{ :; }}
C_WARN=""
{body}
_persist_login_path_dir "/opt/unsloth/bin" "/opt/unsloth/bin" "/opt/unsloth/bin" \\
    "/opt/unsloth/bin" "{rc}" "{mode}"
_persist_login_path_dir "/opt/unsloth/bin" "/opt/unsloth/bin" "/opt/unsloth/bin" \\
    "/opt/unsloth/bin" "{empty}" "{mode}"
"""
    subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    # The stale prepend is repositioned in both modes.
    assert rc.read_text(encoding="utf-8") == expected
    # And only the ordinary mode adds a line where there was none.
    added = "/opt/unsloth/bin" in empty.read_text(encoding="utf-8")
    assert added is (mode != "repoint"), empty.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "path",
    [INSTALL_SH, SETUP_SH_POSIX],
    ids=["install.sh", "studio/setup.sh"],
)
def test_the_rc_repointer_never_exposes_a_truncated_rc_file(path: Path, tmp_path: Path):
    """`cat staged > rc` truncates the user's profile before it writes a byte back.

    An interrupt or an I/O error in between leaves a half-written rc file, and the failure
    branch then deleted the staged copy, which was the only complete one left. The next
    login sources the wreckage. A rename is atomic: the file is the old one or the new one
    and never neither, which is asserted here by making the rename itself fail.
    """
    rc = tmp_path / "rc"
    original = 'export PATH="$HOME/.local/bin:$PATH"\n# keep me\n'
    rc.write_text(original, encoding="utf-8")
    script = (
        # `mv` is what commits the rewrite, so a shell function that shadows it and fails is
        # exactly "the commit did not complete".
        "mv() { return 1; }\n"
        + _shell_function(path, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    assert "status=1" in out.stdout, (out.stdout, out.stderr)
    # The whole point: the user's profile is intact, not empty and not half of itself.
    assert rc.read_text(encoding="utf-8") == original
    assert [p.name for p in tmp_path.iterdir()] == ["rc"], "a staging file was left behind"


def test_the_rc_repointer_keeps_the_permission_bits_it_found(tmp_path: Path):
    """A 0600 rc file must not come back 0644 because the rename handed it the umask.

    The staged file is created as a COPY for that reason, so it carries the original's bits
    before a line of it is rewritten.
    """
    rc = tmp_path / "rc"
    rc.write_text(
        '# Added by Unsloth installer\nexport PATH="$HOME/.local/bin:$PATH"\n',
        encoding="utf-8",
    )
    rc.chmod(0o600)
    script = (
        _shell_function(INSTALL_SH, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
    )
    subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    assert rc.read_text(encoding="utf-8") == (
        '# Added by Unsloth installer\nexport PATH="$PATH:$HOME/.local/bin"\n'
    )
    assert stat.S_IMODE(rc.stat().st_mode) == 0o600, oct(rc.stat().st_mode)


@pytest.mark.parametrize(
    "path,indent",
    [(INSTALL_PS1, "    "), (SETUP_PS1, "")],
    ids=["install.ps1", "studio/setup.ps1"],
)
def test_every_stacked_conda_prefix_is_enumerated(path: Path, indent: str):
    """`conda activate --stack` records the outer environments as CONDA_PREFIX_1, _2, _3,
    _4 and on, and CONDA_SHLVL counts them.

    A fixed list ending at _3 dropped everything past the fourth environment, so
    `Refresh-SessionPath` left those prefixes out of the conda front and their entries
    sorted after Machine and User -- the inverse of the ordering the stack established.
    """
    body = _function(path, indent, "Get-ActiveCondaPrefixes")
    assert "CONDA_SHLVL" in body, body
    assert 'GetEnvironmentVariable("CONDA_PREFIX_$level")' in body, body
    # And no hard-coded tail is left to go stale again.
    for stale in ("CONDA_PREFIX_1", "CONDA_PREFIX_2", "CONDA_PREFIX_3"):
        assert f"$env:{stale}" not in body, (stale, body)
    # A bad or absent CONDA_SHLVL must not spin or skip the active environment.
    assert "$levels -gt 64" in body, body
    assert "$env:CONDA_PREFIX)" in body, body


@pytest.mark.parametrize(
    "path",
    [INSTALL_SH, SETUP_SH_POSIX],
    ids=["install.sh", "studio/setup.sh"],
)
def test_the_rc_repointer_rewrites_a_line_holding_a_backslash(path: Path, tmp_path: Path):
    """POSIX awk decodes backslash escapes in a `-v` assignment.

    The writers escape a backslash before building the line, so the value handed to awk was
    not the value in the file: the literal `grep -qxF` matched, awk matched nothing, and the
    helper renamed an unchanged file and reported success -- the installer said the stale
    prepend had been moved while conda was still in front of it.
    """
    rc = tmp_path / "rc"
    old_line = 'export PATH="/opt/od\\\\d/bin:$PATH"'
    new_line = 'export PATH="$PATH:/opt/od\\\\d/bin"'
    rc.write_text("# Added by Unsloth installer\n" + old_line + "\n# keep me\n", encoding="utf-8")
    script = (
        _shell_function(path, "_unsloth_repoint_rc_line")
        + "_unsloth_repoint_rc_line "
        + f"'{rc}' '{old_line}' '{new_line}'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    assert "status=0" in out.stdout, (out.stdout, out.stderr)
    assert rc.read_text(encoding="utf-8") == (
        "# Added by Unsloth installer\n" + new_line + "\n# keep me\n"
    ), rc.read_text(encoding="utf-8")


def test_the_uv_repoint_pass_moves_the_home_relative_spelling_too():
    """The default uv destination IS ~/.local/bin, and the shim block writes that line
    unexpanded as `$HOME/.local/bin`."""
    source = INSTALL_SH.read_text(encoding="utf-8")
    start = source.index("_uv_repoint_literal=")
    block = source[start : source.index("_persist_fish_path_dir", start)]
    assert "_uv_repoint_home_literal" in block, block
    assert "'$HOME'" in block, block
    # Both spellings inside the per-profile loop, not only the expanded one.
    assert block.count("_persist_login_path_dir") == 2, block
    # $HOME stays unexpanded; only the rest of the path is escaped.
    assert "${_UNSLOTH_UV_BIN_DIR#$HOME}" in block, block


def test_the_standalone_setup_repoints_the_home_relative_prepend_too():
    """`studio/setup.sh` runs on its own during an update, after install.sh wrote the shim line
    as `export PATH="$HOME/.local/bin:$PATH"`."""
    body = _shell_function(SETUP_SH_POSIX, "_setup_persist_uv_path")
    assert "_supp_export_home_prepend" in body, body
    assert "'$HOME'" in body, body
    assert "${_supp_dir#$HOME}" in body, body
    # In the repoint-only branch AND in the present-already branch, which are the two places
    # a stale prepend is reachable.
    assert body.count('_supp_export_home_prepend"') >= 2, body


@pytest.mark.parametrize(
    "path",
    [INSTALL_SH, SETUP_SH_POSIX],
    ids=["install.sh", "studio/setup.sh"],
)
def test_the_rc_repointer_leaves_a_line_the_user_wrote(path: Path, tmp_path: Path):
    """`export PATH="$HOME/.local/bin:$PATH"` is a line people write by hand all the time.

    Rewriting one moves every executable in that directory behind the rest of PATH for good,
    in every later shell, conda or not. Both installers write a `# Added by Unsloth ...`
    comment immediately above the line they add, so that marker is the ownership record and
    an identical line without it is not ours to touch.
    """
    rc = tmp_path / "rc"
    original = "# my own path\n" 'export PATH="$HOME/.local/bin:$PATH"\n' 'alias ll="ls -l"\n'
    rc.write_text(original, encoding="utf-8")
    script = (
        _shell_function(path, "_unsloth_repoint_rc_line")
        + f'_unsloth_repoint_rc_line "{rc}"'
        + " 'export PATH=\"$HOME/.local/bin:$PATH\"'"
        + " 'export PATH=\"$PATH:$HOME/.local/bin\"'\n"
        + 'echo "status=$?"\n'
    )
    out = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=60)
    # Not ours -> a refusal, and the caller's own branch prints the manual advice.
    assert "status=1" in out.stdout, (out.stdout, out.stderr)
    assert rc.read_text(encoding="utf-8") == original, rc.read_text(encoding="utf-8")
