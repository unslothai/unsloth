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

    The machine that matters here has run the installer once already, outside conda, so our
    directory is at the FRONT of the User PATH. Rerun it from an activated environment and
    Resolve-UserPathPosition turns the Prepend into an Append -- but an unconditional
    "already present and appending, nothing to do" early return never rewrites the registry,
    so the front entry survives and conda keeps resolving binaries and DLLs out of our
    directory. The run says "conda keeps priority" while #5871 stays armed.

    Structural rather than behavioural because Add-ToUserPath writes through
    [Microsoft.Win32.Registry], which does not exist off Windows, so the CI leg that could
    execute this is the one place it cannot be checked by running it.
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
    assert (
        "PrependPath" not in block
    ), "the PATH switch has to come from Get-PythonInstallerPathSwitch, not be spelled here"


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
        assert "_unsloth_conda_env_active" in body.group(0), f"{func} must not prepend inside conda"


@pytest.mark.parametrize(
    "path", [INSTALL_SH, SETUP_SH_POSIX], ids = ["install.sh", "studio/setup.sh"]
)
def test_both_halves_call_the_same_environment_an_active_conda(path: Path):
    """The PowerShell and POSIX halves must agree on WHICH variables mean "inside conda".

    `conda activate` exports CONDA_PREFIX and CONDA_DEFAULT_ENV, but a shell hook can
    export only the second and still leave the caller inside conda's PATH ordering.
    install.ps1 has always read both. The POSIX writers read only CONDA_PREFIX, which gave
    one user a prepend on Linux and an append on Windows for the same conda; asserting the
    two name sets are equal is what keeps that from coming back, in either direction.
    """
    posix = path.read_text(encoding = "utf-8")
    body = re.search(r"\n_unsloth_conda_env_active\(\) \{.*?\n\}\n", posix, flags = re.DOTALL)
    assert body is not None, f"{path.name} no longer defines _unsloth_conda_env_active"
    posix_vars = set(re.findall(r"\bCONDA_[A-Z_]+", body.group(0)))

    windows = INSTALL_PS1.read_text(encoding = "utf-8")
    probe = re.search(
        r"function Test-ActiveCondaEnvironment \{.*?\n    \}", windows, flags = re.DOTALL
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


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
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
        env = {
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


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_session_refresh_is_unchanged_outside_conda(shell: str):
    """No conda, no reordering. The refresh is on the hot path of the whole installer and
    must keep doing exactly what it did."""
    previous = "C:\\tools\\bin"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(shell, script, env = _NO_CONDA)
    entries = out[len("PATH=") :].split(";")
    assert entries == MACHINE.split(";") + USER_PATH.split(";") + [previous], entries


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
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
        env = {
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


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
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
        env = {
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
<<<<<<< Updated upstream
    assert body.index("$sources += $condaFront") < body.index(
        "$sources += @($machine"
    ), "the conda entries are appended after the User PATH, which changes nothing"
=======
    assert body.index("$sources += $condaFront") < body.index("$sources += @($machine"), (
        "the conda entries are appended after the User PATH, which changes nothing"
    )


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_name_only_activation_keeps_the_session_it_already_had(shell: str):
    """A hook that exports CONDA_DEFAULT_ENV and nothing that names a directory.

    `Test-ActiveCondaEnvironment` accepts that shape deliberately: the caller IS inside
    conda's ordering. But there is then nothing to test a PATH entry against, so the prefix
    branch produced an empty front and the refresh rebuilt PATH as Machine, User, previous
    -- putting the hook's entries behind the User PATH again, on the exact activation shape
    the guard above says it recognises.

    Guessing from the environment NAME would match any directory that happens to contain
    it, which is the promote-a-stranger defect the boundary check exists to prevent. The
    caller's existing PATH is kept whole and in front instead: it already holds conda's
    ordering, whatever that is.
    """
    previous = f"{CONDA_ROOT}\\envs\\ml\\Scripts;C:\\tools\\bin"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(shell, script, env = {"CONDA_DEFAULT_ENV": "ml"})
    entries = out[len("PATH="):].split(";")
    assert entries[: len(previous.split(";"))] == previous.split(";"), entries
    # The registry values are still there, just behind, so a directory registered by this
    # run is reachable in the same session without displacing the environment.
    assert USER_PATH.split(";")[0] in entries, entries
    assert MACHINE.split(";")[0] in entries, entries
    assert entries.index(USER_PATH.split(";")[0]) > entries.index("C:\\tools\\bin"), entries


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_prefix_activation_still_takes_the_precise_route(shell: str):
    """The whole-PATH fallback is for the case where nothing better can be known, so an
    activation that DOES name its prefix must keep the targeted behaviour: only the conda
    entries move, and an unrelated directory that happened to be on PATH does not."""
    previous = f"C:\\tools\\bin;{CONDA_ROOT}\\envs\\ml\\Scripts"
    script = _stub_registry(_refresh_preamble(previous))
    out = _run(shell, script, env = {
        "CONDA_PREFIX": f"{CONDA_ROOT}\\envs\\ml",
        "CONDA_DEFAULT_ENV": "ml",
    })
    entries = out[len("PATH="):].split(";")
    assert entries[0] == f"{CONDA_ROOT}\\envs\\ml\\Scripts", entries
    assert entries.index("C:\\tools\\bin") > entries.index(USER_PATH.split(";")[0]), entries
>>>>>>> Stashed changes
