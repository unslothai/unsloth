# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows on ARM has to end up on an x64 interpreter, or stop.

pyarrow (reached through `datasets`) and hf-transfer publish no `win_arm64` wheel, at any
version, so a native ARM64 CPython source-builds both and dies on CMake or Rust minutes
into the install. Measured on a real windows-11-arm runner with native ARM64 CPython
3.12.10:

    ERROR: Could not find a version that satisfies the requirement pyarrow (from versions: none)
    *** CMake configuration failed
    ERROR: Failed building wheel for pyarrow
    ERROR: Could not find a version that satisfies the requirement hf-transfer (from versions: none)

The same probe on x64 Windows resolved both, so it is the interpreter architecture and not
the package set. install.ps1 already swaps a freshly selected ARM64 interpreter for x64.
The two holes these tests close are what it does when that swap CANNOT be made, and the
fact that the swap only ever looks at a FRESH selection, so an environment migrated from an
older install is reused on whatever interpreter it was built with.

Provenance: #8495 (Snapdragon X2 Elite, Windows and Linux), #10875 (ARM64 desktop installer
fails on pyarrow while the CLI succeeds).
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
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]


def _function(name: str) -> str:
    """The named function's source, verbatim from install.ps1.

    Extracted rather than restated so these cases track the installer instead of a copy of
    it, and indented-brace delimited rather than marker delimited so rewording a comment
    cannot silently change what is under test.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    match = re.search(rf"    function {name} \{{.*?\n    \}}\n", source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 no longer defines {name}"
    return match.group(0)


def _run(
    shell: str,
    script: str,
    env: dict[str, str] | None = None,
) -> str:
    # run_pwsh, not subprocess.run: both shells write UTF-8 while `text = True` alone
    # decodes with the runner's locale codec. See tests/_shared/unsloth_pwsh_runner.py.
    result = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        errors = "strict",
        env = {**os.environ, **(env or {})},
        timeout = 60,
    )
    return result.stdout.strip()


# The shared preamble: the two host helpers the functions under test call, stubbed so the
# decision is driven by the case rather than by the machine running the suite. Neither stub
# stands in for logic under test here: Get-HostMachineArch is exercised against real
# hardware on the windows-11-arm leg, and Get-PythonPlatformTag asks an interpreter for
# sysconfig.get_platform(), which needs a Windows interpreter to answer win-arm64.
def _preamble(host_arch: str, venv_tag: str, x64_available: bool) -> str:
    # Verbatim from install.ps1, not a stub: both the fresh-selection path and the venv
    # re-check call it, so a change to what the opt-out accepts has to move these cases too.
    opt_out = _function("Test-Arm64PythonOptOut")
    x64 = (
        '@{ Version = "3.13"; Path = "C:\\x64\\python.exe"; Arch = "x86_64" }'
        if x64_available
        else "$null"
    )
    return f"""
$ErrorActionPreference = "Stop"
$script:StudioVtOk = $false
$script:StudioStdoutRedirected = $false
function Write-StudioLine {{
    param([string]$Message = "", [string]$ForegroundColor)
    Write-Host $Message
}}
function step {{ param([string]$Label, [string]$Value, [string]$Color = "Green") Write-Host "STEP ${{Label}}: $Value" }}
function substep {{ param([string]$Message, [string]$Color = "DarkGray") Write-Host "SUBSTEP $Message" }}
function Get-HostMachineArch {{ return "{host_arch}" }}
function Get-PythonPlatformTag {{ param([string]$Exe) return "{venv_tag}" }}
function Install-X64Python {{ Write-Host "INSTALL-X64-CALLED"; return {x64} }}
{opt_out}
"""


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_arm64_host_takes_the_x64_interpreter_when_one_can_be_installed(shell: str):
    """The path that already worked: ARM64 host, x64 obtainable, x64 is what is returned."""
    script = (
        _preamble("arm64", "win-arm64", x64_available = True)
        + f"""
{_function("Resolve-WindowsOnArmX64Python")}
$selected = @{{ Version = "3.13"; Path = "C:\\arm\\python.exe"; Arch = "arm64" }}
$chosen = Resolve-WindowsOnArmX64Python -SelectedPython $selected
Write-Host ("CHOSEN=" + $(if ($chosen) {{ $chosen.Arch }} else {{ "none" }}))
Write-Host ("PATH=" + $(if ($chosen) {{ $chosen.Path }} else {{ "none" }}))
"""
    )
    out = _run(shell, script)
    assert "CHOSEN=x86_64" in out
    assert "PATH=C:\\x64\\python.exe" in out
    assert "using x64 Python" in out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_no_x64_interpreter_is_a_stop_not_a_warning(shell: str):
    """$null is the answer when x64 cannot be had, and the advice names the real fix.

    This is the half of #10875 that survived re-checking. The installer used to print
    [WARN] and carry on, which turns a diagnosable "install x64 Python" into a pyarrow
    CMake failure several minutes later, in a log the reporter has to scroll to find.
    """
    script = (
        _preamble("arm64", "win-arm64", x64_available = False)
        + f"""
{_function("Resolve-WindowsOnArmX64Python")}
$selected = @{{ Version = "3.13"; Path = "C:\\arm\\python.exe"; Arch = "arm64" }}
$chosen = Resolve-WindowsOnArmX64Python -SelectedPython $selected
Write-Host ("CHOSEN=" + $(if ($chosen) {{ $chosen.Arch }} else {{ "none" }}))
"""
    )
    out = _run(shell, script)
    assert "CHOSEN=none" in out, "an ARM64 interpreter must not be handed back as usable"
    assert "[ERROR]" in out
    assert "[WARN]" not in out
    assert "python.org/downloads/windows" in out
    assert "UNSLOTH_ALLOW_ARM64_PYTHON" in out, "a fail-closed path has to name its opt-out"


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_opt_out_keeps_the_old_warn_and_continue(shell: str, value: str):
    """Someone who has built both wheels locally can still ask for the ARM64 interpreter."""
    script = (
        _preamble("arm64", "win-arm64", x64_available = False)
        + f"""
{_function("Resolve-WindowsOnArmX64Python")}
$selected = @{{ Version = "3.13"; Path = "C:\\arm\\python.exe"; Arch = "arm64" }}
$chosen = Resolve-WindowsOnArmX64Python -SelectedPython $selected
Write-Host ("CHOSEN=" + $(if ($chosen) {{ $chosen.Arch }} else {{ "none" }}))
"""
    )
    out = _run(shell, script, env = {"UNSLOTH_ALLOW_ARM64_PYTHON": value})
    assert "CHOSEN=arm64" in out
    assert "[WARN]" in out
    assert "[ERROR]" not in out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_opt_out_is_read_before_the_x64_install_is_attempted(shell: str, value: str):
    """The opt-out has to opt out on a machine where the x64 install would SUCCEED.

    Read only after Install-X64Python fails, the variable does nothing at all on any host
    that can reach python.org, which is most of them: an interpreter the user did not ask
    for is downloaded, installed and then used. Someone who sets this has pyarrow and
    hf-transfer built for ARM64 already.
    """
    script = (
        _preamble("arm64", "win-arm64", x64_available = True)
        + f"""
{_function("Resolve-WindowsOnArmX64Python")}
$selected = @{{ Version = "3.13"; Path = "C:\\arm\\python.exe"; Arch = "arm64" }}
$chosen = Resolve-WindowsOnArmX64Python -SelectedPython $selected
Write-Host ("CHOSEN=" + $(if ($chosen) {{ $chosen.Arch }} else {{ "none" }}))
"""
    )
    out = _run(shell, script, env = {"UNSLOTH_ALLOW_ARM64_PYTHON": value})
    assert "CHOSEN=arm64" in out
    assert (
        "INSTALL-X64-CALLED" not in out
    ), "an x64 Python was installed for someone who asked to keep the ARM64 one"
    assert "[WARN]" in out
    assert "[ERROR]" not in out
    assert "Unset UNSLOTH_ALLOW_ARM64_PYTHON" in out, "the warning has to say how to undo it"


def test_the_python_org_route_is_not_attempted_twice_inside_conda():
    """Inside conda, python.org is tried FIRST, so the later winget fallback must not
    repeat it.

    On an offline machine the second attempt is the same failing download again, and it is
    announced as "falling back to python.org" for a fallback that already ran.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    start = source.index('substep "installing Python ${PythonVersion}..."')
    end = source.index("Python installation failed", start)
    block = source[start:end]
    assert (
        block.count("Install-PythonFromPythonOrg") == 2
    ), "the conda-first call and the winget fallback, and nothing else"
    guard = source.index('substep "winget could not install Python', start)
    condition = source.rindex("if (-not $DetectedPython", start, guard)
    assert (
        "-not $pythonOrgTried" in source[condition : source.index("\n", condition)]
    ), "the winget fallback would otherwise run python.org a second time"


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "host_arch,venv_tag,selected_arch,venv_exists,expected",
    [
        # The reported shape: ARM64 host, x64 interpreter chosen for this run, environment
        # migrated from an older install and therefore still native ARM64.
        ("arm64", "win-arm64", "x86_64", True, True),
        # Already x64: nothing to rebuild.
        ("arm64", "win-amd64", "x86_64", True, False),
        # Not Windows on ARM at all.
        ("x86_64", "win-amd64", "x86_64", True, False),
        # ARM64 host where the chosen interpreter is ARM64 too. Nothing better is in hand,
        # so this must NOT fire: it is also the shape a host served by native ARM64 wheels
        # has, and rebuilding its environment would be a regression rather than a fix.
        ("arm64", "win-arm64", "arm64", True, False),
        # No environment to reuse: the fresh-venv branch handles this one.
        ("arm64", "win-arm64", "x86_64", False, False),
        # An interpreter that cannot be launched answers nothing; the managed-Python error
        # further down install.ps1 is what should report that, not a silent rebuild.
        ("arm64", "", "x86_64", True, False),
    ],
)
def test_a_reused_environment_is_re_checked_for_its_interpreter_architecture(
    tmp_path: Path,
    shell: str,
    host_arch: str,
    venv_tag: str,
    selected_arch: str,
    venv_exists: bool,
    expected: bool,
):
    """The swap only covers a fresh selection, so reuse has to be re-checked separately.

    #8495: "the x64 swap only covers a fresh interpreter selection, so any reuse path on
    Windows ARM64 still lands on native ARM64 CPython". install.ps1 reaches its
    `step "venv" "using migrated environment"` line with a venv built by whichever
    installer created it, which on these machines predates the swap entirely.
    """
    venv_python = tmp_path / "python.exe"
    if venv_exists:
        venv_python.write_text("", encoding = "utf-8")
    selected = f'@{{ Version = "3.13"; Path = "C:\\p\\python.exe"; Arch = "{selected_arch}" }}'
    script = (
        _preamble(host_arch, venv_tag, x64_available = True)
        + f"""
{_function("Test-StudioVenvArchMismatch")}
$selected = {selected}
$mismatch = Test-StudioVenvArchMismatch -VenvPython "{venv_python}" -SelectedPython $selected
Write-Host ("MISMATCH=" + $mismatch)
"""
    )
    out = _run(shell, script)
    assert f"MISMATCH={expected}" in out


@pytest.mark.skipif(not POWERSHELLS, reason = "no PowerShell on this host")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_the_opt_out_also_protects_an_existing_arm64_environment(
    tmp_path: Path, shell: str, value: str
):
    """The opt-out has to be read HERE, not only where a fresh interpreter is chosen.

    Resolve-WindowsOnArmX64Python never runs when the ordinary probe already found an x64
    Python on the machine, which is an ARM64 box that has both interpreters installed. On
    such a host UNSLOTH_ALLOW_ARM64_PYTHON was never consulted at all, and this branch then
    moved aside the very native ARM64 environment the user had asked to keep. The rebuild
    is not a migration: the new venv is on a different architecture, so nothing the user
    installed into the old one is carried over.
    """
    venv_python = tmp_path / "python.exe"
    venv_python.write_text("", encoding = "utf-8")
    selected = '@{ Version = "3.13"; Path = "C:\\p\\python.exe"; Arch = "x86_64" }'
    script = (
        _preamble("arm64", "win-arm64", x64_available = True)
        + f"""
{_function("Test-StudioVenvArchMismatch")}
$selected = {selected}
$mismatch = Test-StudioVenvArchMismatch -VenvPython "{venv_python}" -SelectedPython $selected
Write-Host ("MISMATCH=" + $mismatch)
"""
    )
    out = _run(shell, script, env = {"UNSLOTH_ALLOW_ARM64_PYTHON": value})
    assert "MISMATCH=False" in out, (
        "the opt-out was ignored and an existing ARM64 environment was scheduled for replacement"
    )


@pytest.mark.skipif(not POWERSHELLS, reason = "no PowerShell on this host")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_arm64_environment_is_still_replaced_without_the_opt_out(tmp_path: Path, shell: str):
    """The other direction, so the guard above cannot be satisfied by never firing."""
    venv_python = tmp_path / "python.exe"
    venv_python.write_text("", encoding = "utf-8")
    selected = '@{ Version = "3.13"; Path = "C:\\p\\python.exe"; Arch = "x86_64" }'
    script = (
        _preamble("arm64", "win-arm64", x64_available = True)
        + f"""
{_function("Test-StudioVenvArchMismatch")}
$selected = {selected}
$mismatch = Test-StudioVenvArchMismatch -VenvPython "{venv_python}" -SelectedPython $selected
Write-Host ("MISMATCH=" + $mismatch)
"""
    )
    out = _run(shell, script)
    assert "MISMATCH=True" in out


def test_both_arm64_opt_out_readers_go_through_one_helper():
    """Two spellings of the same question drift. The fresh-selection path and the reuse
    re-check must read it the same way, or an accepted value works in one and not the
    other."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    assert source.count("function Test-Arm64PythonOptOut") == 1
    # The literal comparison lives in the helper and nowhere else.
    assert source.count("$env:UNSLOTH_ALLOW_ARM64_PYTHON -in") == 1, (
        "the opt-out is compared in more than one place"
    )
    assert source.count("Test-Arm64PythonOptOut") >= 3, (
        "both the selection path and the venv re-check have to call it"
    )


def test_the_arm64_rebuild_tells_the_user_what_it_does_not_carry_over():
    """A rebuild on a different architecture cannot reuse the old wheels, so packages the
    user added to the ARM64 environment are gone from the new one. That is unavoidable; a
    silent version of it is not. The branch has to name where the old tree went and how to
    opt out."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    start = source.find("Test-StudioVenvArchMismatch -VenvPython")
    assert start != -1
    block = source[start:source.find('step "venv" "creating Python', start)]
    assert "unsloth_studio.rollback" in block, "the rebuild does not say where the old venv went"
    assert "UNSLOTH_ALLOW_ARM64_PYTHON" in block, "the rebuild does not name its opt-out"


def test_the_installer_stops_when_no_x64_interpreter_can_be_installed():
    """The caller must treat $null as a failure, not as "no change".

    Resolve-WindowsOnArmX64Python returning $null is only a stop if the call site says so,
    and a call site that assigned the result and carried on would leave every case above
    passing while the installer behaved exactly as it did before.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    match = re.search(
        r"if \(\$DetectedPython -and \(Get-HostMachineArch\) -eq \"arm64\".*?\n    \}\n",
        source,
        flags = re.DOTALL,
    )
    assert match is not None, "install.ps1 no longer guards the Windows-on-ARM swap"
    block = match.group(0)
    assert "Resolve-WindowsOnArmX64Python" in block
    assert "Exit-InstallFailure" in block, (
        "an unobtainable x64 interpreter has to end the install here; continuing moves the "
        "failure to the pyarrow build, which is what #10875 reported"
    )


def test_the_venv_reuse_path_re_checks_before_it_reuses():
    """The re-check has to sit ahead of the branch that decides to reuse.

    Ordering is the whole property: placed after `step "venv" "using migrated environment"`
    it would run once the installer had already committed to the environment.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    recheck = source.find("Test-StudioVenvArchMismatch -VenvPython")
    create = source.find('step "venv" "creating Python')
    reuse = source.find('step "venv" "using migrated environment"')
    assert recheck != -1, "install.ps1 no longer re-checks the reused interpreter"
    assert create != -1 and reuse != -1
    assert recheck < create < reuse, "the re-check must precede the create/reuse branch"
    # And it must clear $_Migrated, or the install step downstream runs --no-deps into the
    # fresh venv it just created and installs nothing.
    tail = source[recheck:create]
    assert "$_Migrated = $false" in tail
