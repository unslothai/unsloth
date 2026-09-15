# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""No installer PATH write may demote an ACTIVE conda environment.

A prepend into the persistent User PATH is an ordering that outlives the conda activation
it was made under: from the next shell on, our directory sits ahead of the conda
environment's own entries, so conda resolves binaries and DLLs out of ours. #5871 is that
shape end to end, reported on Windows 11 with Anaconda: after installing Studio, `conda
update --all --yes` leaves the base environment unusable with `ImportError: DLL load failed
while importing _ctypes`, and it recurs on every reinstall of Anaconda.

The guard is one decision function per installer, and every persistent write routes through
it, so a new call site cannot reintroduce the defect by passing -Position Prepend. Three
things are asserted: the decision itself, that both PowerShell installers route through it,
and that the choice of installer switch for a system Python follows the same rule.

install.sh and studio/setup.sh carry the same defect in their rc-file and fish-config
writes and take the same treatment; that half is tests/sh/test_install_conda_path_guard.sh.
"""

from __future__ import annotations

import os
import re
import shutil
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
    source = path.read_text(encoding = "utf-8")
    match = re.search(rf"{indent}function {name} \{{.*?\n{indent}\}}\n", source, flags = re.DOTALL)
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
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "strict",
        env = {**base, **(env or {})},
        timeout = 60,
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


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
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
    ids = [
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
    out = _run(shell, script, env = env)
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
    assert "Resolve-UserPathPosition" in body, (
        f"{path_label}: Add-ToUserPath must resolve its position through the guard"
    )
    # Ahead of the registry read, or the decision lands after the value it decides is used.
    assert body.index("Resolve-UserPathPosition") < body.index("Registry"), (
        f"{path_label}: the guard has to run before the registry write is prepared"
    )


@pytest.mark.parametrize("path_label", sorted(_PS_FILES))
def test_the_two_installers_agree_on_the_guard(path_label: str):
    """Both copies decide the same way. They are separate scripts, not a shared module.

    setup.ps1 runs in its own process after install.ps1 hands off, so it has its own
    Add-ToUserPath and its own copy of this decision. A fix applied to one and not the
    other leaves the CUDA bin, MSVC and llama.cpp prepends in setup.ps1 still demoting
    conda, which is the same reported breakage by another route.
    """
    path, indent = _PS_FILES[path_label]
    body = _function(path, indent, "Resolve-UserPathPosition")
    assert "Test-ActiveCondaEnvironment" in body
    assert "'Append'" in body
    conda = _function(path, indent, "Test-ActiveCondaEnvironment")
    assert "CONDA_PREFIX" in conda
    assert "CONDA_DEFAULT_ENV" in conda


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "env,expected",
    [(_NO_CONDA, "PrependPath=1"), (_ACTIVE_CONDA, "AppendPath=1"), (_EMPTY, "PrependPath=1")],
    ids = ["no-conda", "active-conda", "empty-vars"],
)
def test_the_system_python_install_follows_the_same_rule(
    shell: str, env: dict[str, str], expected: str
):
    """The reporter's own diagnosis was the system-wide Python, not our own directories.

    "The problem not from .venv unsloth_studio but from python (system wide) which
    installed by unsloth installer, superseed the anaconda python." PrependPath=1 is a
    documented python.org option that puts the new interpreter and its Scripts directory at
    the front of the User PATH; AppendPath=1 is the same registration at the back.
    """
    script = (
        _function(INSTALL_PS1, "    ", "Test-ActiveCondaEnvironment")
        + _function(INSTALL_PS1, "    ", "Get-PythonInstallerPathSwitch")
        + "\nWrite-Host ('SWITCH=' + (Get-PythonInstallerPathSwitch))\n"
    )
    out = _run(shell, script, env = env)
    assert f"SWITCH={expected}" in out


def test_the_python_org_installer_arguments_use_the_switch_function():
    """A literal PrependPath=1 in the argument list would bypass the decision entirely."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    args = re.search(r"\$installArgs = @\(.*?\n        \)", source, flags = re.DOTALL)
    assert args is not None, "install.ps1 no longer builds $installArgs for python.org"
    block = args.group(0)
    assert "Get-PythonInstallerPathSwitch" in block
    assert "PrependPath" not in block, (
        "the PATH switch has to come from Get-PythonInstallerPathSwitch, not be spelled here"
    )


def test_winget_is_not_the_first_route_inside_an_active_conda_environment():
    """winget's Python package hard-codes PrependPath=1, so it cannot be asked politely.

    The manifest for Python.Python.3.13 carries `InstallAllUsers=0 PrependPath=1` for user
    scope and `InstallAllUsers=1 PrependPath=1` for machine scope, so every winget install
    of Python takes PATH priority regardless of what we pass. Inside a conda environment
    the python.org route, whose switch we choose, is therefore tried first, and winget
    stays the fallback it has always been: a failed install is worse than a warned PATH.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
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
    ids = ["install.sh", "studio/setup.sh"],
)
def test_the_posix_installers_carry_the_same_guard(path: Path, funcs: tuple[str, ...]):
    """Cross-platform parity, asserted here so the PowerShell and POSIX halves cannot drift.

    The behavioural cases are in tests/sh/test_install_conda_path_guard.sh, which runs the
    extracted shell functions; this only holds that the guard exists at all, so a rewrite of
    a PATH persistence block on the POSIX side cannot quietly drop it. Four writers in all:
    the registry in install.ps1 and setup.ps1, the rc files in install.sh and setup.sh.
    """
    source = path.read_text(encoding = "utf-8")
    for func in funcs:
        body = re.search(rf"\n{func}\(\) \{{.*?\n\}}\n", source, flags = re.DOTALL)
        assert body is not None, f"{path.name} no longer defines {func}"
        assert "CONDA_PREFIX" in body.group(0), f"{func} must not prepend inside conda"
