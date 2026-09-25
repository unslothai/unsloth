# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows on ARM has to end up on an x64 interpreter, or stop.

PyPI publishes no `win_arm64` wheel for pyarrow (reached through `datasets`) or
hf-transfer, at any version, so a native ARM64 CPython source-builds both and dies on CMake
or Rust minutes in. Measured on a windows-11-arm runner with ARM64 CPython 3.12.10:

    ERROR: Could not find a version that satisfies the requirement pyarrow (from versions: none)
    *** CMake configuration failed

The same probe on x64 resolved both, so it is the interpreter architecture. install.ps1
already swaps a freshly selected ARM64 interpreter for x64; these tests close what it does
when that swap CANNOT be made, and the fact that it only looks at a FRESH selection, so a
migrated environment is reused on whatever built it.

Provenance: #8495, #10875.
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
    """The opt-out has to opt out on a machine where the x64 install would SUCCEED."""
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
    """The swap only covers a fresh selection, so reuse has to be re-checked separately."""
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

    Resolve-WindowsOnArmX64Python never runs when the ordinary probe already found x64, so
    on an ARM64 box with both interpreters the variable was never consulted and this branch
    moved aside the very environment the user asked to keep. The rebuild is not a migration:
    nothing installed into the old venv carries over.
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
    assert (
        "MISMATCH=False" in out
    ), "the opt-out was ignored and an existing ARM64 environment was scheduled for replacement"


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
    assert (
        source.count("$env:UNSLOTH_ALLOW_ARM64_PYTHON -in") == 1
    ), "the opt-out is compared in more than one place"
    assert (
        source.count("Test-Arm64PythonOptOut") >= 3
    ), "both the selection path and the venv re-check have to call it"


def test_the_arm64_rebuild_tells_the_user_what_it_does_not_carry_over():
    """A rebuild on a different architecture cannot reuse the old wheels, so packages the
    user added to the ARM64 environment are gone from the new one. That is unavoidable; a
    silent version of it is not. The branch has to name where the old tree went and how to
    opt out."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    start = source.find("Test-StudioVenvArchMismatch -VenvPython")
    assert start != -1
    block = source[start : source.find('step "venv" "creating Python', start)]
    assert "unsloth_studio.arm64" in block, "the rebuild does not say where the old venv went"
    assert "UNSLOTH_ALLOW_ARM64_PYTHON" in block, "the rebuild does not name its opt-out"


def test_the_installer_stops_when_no_x64_interpreter_can_be_installed():
    """The caller must treat $null as a failure, not as "no change".

    Resolve-WindowsOnArmX64Python returning $null is only a stop if the call site says so,
    and a call site that assigned the result and carried on would leave every case above
    passing while the installer behaved exactly as it did before.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    # Anchored on the call, not on the shape of the if: the guard is multi-line since the
    # native CUDA torch check joined it, and an anchor on "$DetectedPython -and" silently
    # matched an unrelated earlier block instead of failing.
    call = source.index("$DetectedPython = Resolve-WindowsOnArmX64Python")
    opening = source.rindex("\n    if (", 0, call)
    match = re.search(r".*?\n    \}\n", source[opening + 1 :], flags = re.DOTALL)
    assert match is not None, "install.ps1 no longer guards the Windows-on-ARM swap"
    block = match.group(0)
    assert "Resolve-WindowsOnArmX64Python" in block
    assert "-not $script:WoaNativeCudaTorch" in block, (
        "a host with native ARM64 CUDA torch wants its ARM64 interpreter; swapping it for "
        "x64 there gives a Triton that cannot do sm_121"
    )
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


# ── The x64 bootstrap has to respect conda too ──
# The generic bootstrap takes the python.org route first inside an active conda environment,
# because winget's Python manifest hard-codes PrependPath=1 and there is no switch that asks
# it otherwise. Windows on ARM does not reach that bootstrap: it has a compatible (ARM64)
# interpreter already, so the "install Python" branch never runs and Install-X64Python is
# what actually performs the install. Ordered winget-first, it puts Python ahead of conda on
# PATH on exactly the machine this change exists to protect.


def _x64_bootstrap_preamble(
    conda_active: bool, python_org_arch: str, winget_available: bool, winget_arch: str
) -> str:
    """Stubs for everything Install-X64Python calls, so the case drives the decision.

    `python_org_arch` / `winget_arch` are what each source manages to produce: "x86_64",
    "arm64" (installed, but the wrong architecture, which the caller rejects) or "" for
    nothing at all.
    """

    def _result(arch: str, path: str) -> str:
        if not arch:
            return "$null"
        return f'@{{ Version = "3.13"; Path = "{path}"; Arch = "{arch}" }}'

    # Bound before the template rather than called inside it: a backslash inside an
    # f-string expression is a 3.12 syntax addition, and this repo's ruff target is 3.11.
    winget_installed = _result(winget_arch, "C:\\winget\\python.exe")
    python_org_installed = _result(python_org_arch, "C:\\pyorg\\python.exe")

    return f"""
$ErrorActionPreference = "Stop"
function Write-StudioLine {{
    param([string]$Message = "", [string]$ForegroundColor)
    Write-Host $Message
}}
function substep {{ param([string]$Message, [string]$Color = "DarkGray") Write-Host "SUBSTEP $Message" }}
function Test-ActiveCondaEnvironment {{ return ${str(conda_active).lower()} }}
$PythonVersion = "3.13"
$script:WingetAvailable = ${str(winget_available).lower()}
$script:WingetExe = "winget.exe"
$script:WingetInstalled = {winget_installed}
$script:PythonOrgCalls = 0
function Install-PythonFromPythonOrg {{
    param([string]$Arch = "")
    $script:PythonOrgCalls++
    Write-Host "PYTHON-ORG-CALLED arch=$Arch"
    return {python_org_installed}
}}
function Refresh-SessionPath {{ }}
function Find-CompatiblePython {{
    param([switch]$X64Only)
    return $script:WingetInstalled
}}
"""


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_x64_bootstrap_skips_winget_inside_conda(shell: str):
    """Inside conda, python.org is what installs the x64 interpreter."""
    script = (
        _x64_bootstrap_preamble(
            conda_active = True,
            python_org_arch = "x86_64",
            winget_available = True,
            winget_arch = "x86_64",
        )
        + f"""
function Invoke-Winget {{ Write-Host "WINGET-CALLED" }}
{_function("Install-X64Python").replace('& $script:WingetExe install', 'Invoke-Winget #')}
$found = Install-X64Python
Write-Host ("ARCH=" + $(if ($found) {{ $found.Arch }} else {{ "none" }}))
Write-Host ("PATH=" + $(if ($found) {{ $found.Path }} else {{ "none" }}))
Write-Host ("PYTHON-ORG-CALLS=" + $script:PythonOrgCalls)
"""
    )
    out = _run(shell, script)
    assert "ARCH=x86_64" in out
    assert (
        "PATH=C:\\pyorg\\python.exe" in out
    ), "the conda path must take the python.org interpreter"
    assert "WINGET-CALLED" not in out, (
        "winget hard-codes PrependPath=1, so reaching it inside conda puts Python ahead of "
        "the active environment with no way to ask for anything else"
    )
    assert "PYTHON-ORG-CALLS=1" in out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_x64_bootstrap_still_falls_back_to_winget_inside_conda(shell: str):
    """A warned-about PATH beats no x64 Python at all."""
    script = (
        _x64_bootstrap_preamble(
            conda_active = True,
            python_org_arch = "",
            winget_available = True,
            winget_arch = "x86_64",
        )
        + f"""
function Invoke-Winget {{ Write-Host "WINGET-CALLED" }}
{_function("Install-X64Python").replace('& $script:WingetExe install', 'Invoke-Winget #')}
$found = Install-X64Python
Write-Host ("ARCH=" + $(if ($found) {{ $found.Arch }} else {{ "none" }}))
Write-Host ("PYTHON-ORG-CALLS=" + $script:PythonOrgCalls)
"""
    )
    out = _run(shell, script)
    assert "WINGET-CALLED" in out
    assert "ARCH=x86_64" in out
    assert "ahead of conda on PATH" in out, "the fallback has to say what it costs"
    assert "PYTHON-ORG-CALLS=1" in out, (
        "the winget branch would otherwise repeat the same failing download, on the offline "
        "machine that is the usual reason it failed"
    )


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_x64_bootstrap_is_unchanged_outside_conda(shell: str):
    """No conda, no reordering: winget first, exactly as before.

    winget is the faster and more reliable source on a normal machine, and this change is
    only about the environment whose PATH it would break.
    """
    script = (
        _x64_bootstrap_preamble(
            conda_active = False,
            python_org_arch = "x86_64",
            winget_available = True,
            winget_arch = "x86_64",
        )
        + f"""
function Invoke-Winget {{ Write-Host "WINGET-CALLED" }}
{_function("Install-X64Python").replace('& $script:WingetExe install', 'Invoke-Winget #')}
$found = Install-X64Python
Write-Host ("PATH=" + $(if ($found) {{ $found.Path }} else {{ "none" }}))
Write-Host ("PYTHON-ORG-CALLS=" + $script:PythonOrgCalls)
"""
    )
    out = _run(shell, script)
    assert "WINGET-CALLED" in out
    assert "PATH=C:\\winget\\python.exe" in out
    assert "PYTHON-ORG-CALLS=0" in out


# ── The environment the rebuild promises to keep has to still be there ──


def _rollback_preamble(studio_home: Path) -> str:
    return f"""
$ErrorActionPreference = "Stop"
function Write-StudioLine {{
    param([string]$Message = "", [string]$ForegroundColor)
    Write-Host $Message
}}
function substep {{ param([string]$Message, [string]$Color = "DarkGray") Write-Host "SUBSTEP $Message" }}
$StudioHome = "{studio_home}"
function Remove-StudioVenvTreeWithRetry {{
    param([string]$Path, [string]$Label)
    Write-Host "REMOVED $Path"
    Remove-Item -LiteralPath $Path -Recurse -Force
    return $true
}}
{_function("Test-StudioPathPresent")}
{_function("Complete-StudioVenvRollback")}
"""


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_preserved_arm64_environment_survives_a_successful_install(tmp_path: Path, shell: str):
    """The rebuild promises the ARM64 environment is kept so packages can be copied out.

    Complete-StudioVenvRollback deletes the backup on success, which is right for every
    other rollback but wrong for this one: the old tree holds packages the new architecture
    cannot have, and deleting it discards them and the message that said otherwise.
    """
    home = tmp_path / "studio"
    backup = home / "unsloth_studio.rollback.20260101000000.4242"
    (backup / "Lib").mkdir(parents = True)
    (backup / "Lib" / "marker.txt").write_text("a package the user added", encoding = "utf-8")

    script = (
        _rollback_preamble(home)
        + f"""
$script:StudioVenvRollbackActive = $true
$script:StudioVenvRollbackDir = "{backup}"
$script:StudioVenvRollbackPartial = $false
$script:StudioVenvRollbackPreserve = $true
Complete-StudioVenvRollback
Write-Host ("COMMITTED=" + $script:StudioInstallCommitted)
Write-Host ("PRESERVE=" + $script:StudioVenvRollbackPreserve)
"""
    )
    out = _run(shell, script)
    assert "COMMITTED=True" in out, "the commit flag is what stops a later restore"
    assert "PRESERVE=False" in out, "the flag must not survive into a later rollback"
    assert "REMOVED" not in out, "the tree the user was told to read must not be deleted"
    assert not backup.exists(), "it has to leave the rollback namespace, see below"

    kept = [path for path in home.iterdir() if path.name.startswith("unsloth_studio.arm64.")]
    assert len(kept) == 1, f"expected one preserved ARM64 tree, found {[p.name for p in kept]}"
    assert (kept[0] / "Lib" / "marker.txt").read_text(encoding = "utf-8") == (
        "a package the user added"
    )
    assert (
        f"previous ARM64 environment kept at {kept[0]}" in out
    ), "a preserved tree nobody is told the path of is just disk usage"


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_an_ordinary_rollback_is_still_deleted_on_success(tmp_path: Path, shell: str):
    """The preserve flag is opt-in, and every other rollback keeps the old behaviour.

    Without this the fix trades one bug for a slow leak: a full environment left behind by
    every reinstall, on the machine of every user who never asked for one.
    """
    home = tmp_path / "studio"
    backup = home / "unsloth_studio.rollback.20260101000000.4242"
    backup.mkdir(parents = True)

    script = (
        _rollback_preamble(home)
        + f"""
$script:StudioVenvRollbackActive = $true
$script:StudioVenvRollbackDir = "{backup}"
$script:StudioVenvRollbackPartial = $false
$script:StudioVenvRollbackPreserve = $false
Complete-StudioVenvRollback
"""
    )
    out = _run(shell, script)
    assert f"REMOVED {backup}" in out
    assert not backup.exists()
    assert not list(home.iterdir()), "nothing is kept when nothing was promised"


def test_the_preserved_tree_leaves_the_swept_rollback_namespace():
    """Remove-StaleStudioVenvRollbacks globs unsloth_studio.rollback.*, and preserves an entry
    only while its owning PID is alive."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    complete = source.index("function Complete-StudioVenvRollback")
    block = source[complete : source.index("\n    }\n", complete)]
    # Comments out: this function's explain why it does NOT use the rollback name, and a
    # test that reads them as code would fail on the sentence describing the fix.
    code = "\n".join(line for line in block.splitlines() if not line.lstrip().startswith("#"))
    assert "unsloth_studio.arm64." in code, "the preserved copy is not renamed"
    assert (
        "unsloth_studio.rollback." not in code
    ), "a preserved copy under the swept name is removed by the next install"
    sweep = source.index("function Remove-StaleStudioVenvRollbacks")
    sweep_block = source[sweep : source.index("\n    }\n", sweep)]
    assert (
        "'unsloth_studio.rollback.*'" in sweep_block
    ), "the sweep no longer matches what this test reasons about"
    assert "unsloth_studio.arm64" not in sweep_block


# ── the interpreter is gone by the time the re-check runs ──
# An ordinary reinstall moves the whole environment to its rollback path, and that move
# happens BEFORE the x64 interpreter is chosen. Probing $VenvPython after it answers "no
# mismatch" for every existing install, so the rebuild proceeds on ARM64 and the tree is
# deleted with the ordinary rollback instead of being preserved.


def test_the_platform_tag_is_read_before_the_rollback_move():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    record = source.find("$script:PrevVenvPlatformTag = Get-PythonPlatformTag")
    move = source.find("Start-StudioVenvRollback -ExistingDir $VenvDir")
    recheck = source.find("Test-StudioVenvArchMismatch -VenvPython")
    assert record != -1, "install.ps1 no longer records the existing interpreter's tag"
    assert record < move, "the tag is read after the move that takes the interpreter away"
    assert (
        "-RecordedTag $script:PrevVenvPlatformTag" in source[recheck : recheck + 400]
    ), "the re-check does not use the recorded tag, so it cannot see a moved environment"


def _arch_mismatch_preamble(
    host_arch: str, opt_out: bool, venv_python_exists: bool, live_tag: str, recorded_tag: str
) -> str:
    selected = '@{ Version = "3.13"; Arch = "x86_64" }'
    exists = "$true" if venv_python_exists else "$false"
    recorded = f'"{recorded_tag}"' if recorded_tag else "$null"
    return f"""
$ErrorActionPreference = "Stop"
function substep {{ param([string]$Message = "", [string]$Color) Write-Host "SUBSTEP $Message" }}
function Get-HostMachineArch {{ return "{host_arch}" }}
function Test-Arm64PythonOptOut {{ return ${str(opt_out).lower()} }}
function Get-PythonPlatformTag {{ param($Exe) return "{live_tag}" }}
function Test-Path {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) return {exists} }}
{_function("Test-StudioVenvArchMismatch")}
$answer = Test-StudioVenvArchMismatch -VenvPython "C:\\venv\\Scripts\\python.exe" `
    -SelectedPython {selected} -RecordedTag {recorded}
Write-Host ("ANSWER=" + $answer)
"""


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_moved_arm64_environment_is_still_detected(shell: str):
    """The case the reinstall actually takes: the interpreter has been moved aside already."""
    out = _run(
        shell,
        _arch_mismatch_preamble("arm64", False, False, "win-amd64", "win-arm64"),
    )
    assert out.splitlines()[-1] == "ANSWER=True", out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_moved_x64_environment_is_not_mistaken_for_one(shell: str):
    """The other half. A recorded tag that is not win-arm64 answers no, and so does no
    recorded tag at all, which is the fresh-install case."""
    out = _run(shell, _arch_mismatch_preamble("arm64", False, False, "win-arm64", "win-amd64"))
    assert out.splitlines()[-1] == "ANSWER=False", out
    out = _run(shell, _arch_mismatch_preamble("arm64", False, False, "win-arm64", ""))
    assert out.splitlines()[-1] == "ANSWER=False", out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_the_opt_out_message_is_only_printed_for_an_arm64_environment(shell: str):
    """`UNSLOTH_ALLOW_ARM64_PYTHON=1` on an x64 environment used to announce that a native
    ARM64 environment was being kept, which was true of nothing on the machine."""
    out = _run(shell, _arch_mismatch_preamble("arm64", True, False, "win-amd64", "win-amd64"))
    assert "SUBSTEP" not in out, out
    assert out.splitlines()[-1] == "ANSWER=False", out

    # With a real ARM64 environment the opt-out still says what it is doing, and still
    # answers no.
    out = _run(shell, _arch_mismatch_preamble("arm64", True, True, "win-arm64", "win-arm64"))
    assert "UNSLOTH_ALLOW_ARM64_PYTHON" in out, out
    assert out.splitlines()[-1] == "ANSWER=False", out


def test_the_architecture_rebuild_does_not_move_the_venv_twice():
    """An ordinary reinstall has already moved $VenvDir into a rollback by the time this
    branch runs, so a second Start-StudioVenvRollback would move a directory that is no
    longer there, throw, and take every architecture migration out through
    Exit-InstallFailure. The tree is already where the branch wants it; only the flag that
    keeps it from being swept is missing."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    branch = source.index("windows on arm: the existing environment runs native ARM64 Python")
    # Searched forward from the branch rather than inside a fixed-size window: a byte count
    # measures how long the branch happens to be today, so adding an arm to the same if/elseif
    # chain pushes the move out of view and fails this for a reason that has nothing to do with
    # what it tests.
    guard = source.index("if ($script:StudioVenvRollbackActive)", branch)
    move = source.index("Start-StudioVenvRollback -ExistingDir $VenvDir", branch)
    assert guard < move, "the rebuild still moves the environment unconditionally"
    assert (
        "$script:StudioVenvRollbackPreserve = $true" in source[guard:move]
    ), "an already-active rollback is not marked for preservation"


def test_the_opt_out_keeps_the_environment_it_says_it_keeps():
    """`UNSLOTH_ALLOW_ARM64_PYTHON=1` returning $false is not, by itself, keeping anything:
    on a host that already has an x64 Python the reinstall has moved the ARM64 environment
    into a rollback that success deletes. The opt-out marks that rollback preserved."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    optout = source.index(
        '$script:PrevVenvPlatformTag -eq "win-arm64" -and (Test-Arm64PythonOptOut)'
    )
    create = source.index('step "venv" "creating Python')
    assert optout < create, "the opt-out preservation runs after the rebuild has started"
    assert "$script:StudioVenvRollbackPreserve = $true" in source[optout : optout + 600]
    assert (
        "unsloth_studio.arm64.*" in source[optout : optout + 600]
    ), "the user is not told where the kept environment is"


def test_the_opt_out_reaches_the_interpreter_selection():
    """On a host that already has an x64 Python the swap never runs."""
    body = _function("Find-CompatiblePython")
    assert (
        "Test-Arm64PythonOptOut" in body
    ), "the interpreter selection does not consult the ARM64 opt-out"
    prefer = body[body.index("$preferArm64 =") : body.index("$preferX64 =")]
    # Never for Install-X64Python's own lookup, which exists to find x64 specifically.
    assert "-not $X64Only" in prefer, prefer
    assert "Get-HostMachineArch" in prefer, prefer
    # The native-CUDA host keeps its own reason to prefer ARM64.
    assert "WoaNativeCudaTorch" in prefer, prefer
    # And x64 is still preferred on an ARM64 host WITHOUT the opt-out, which is the whole
    # reason the preference exists.
    x64 = body[body.index("$preferX64 =") :].split("\n", 1)[0]
    assert "-not $preferArm64" in x64, x64

    # The swap still keeps an ARM64 selection rather than bootstrapping over it.
    resolve = _function("Resolve-WindowsOnArmX64Python")
    assert "Test-Arm64PythonOptOut" in resolve
    assert "return $SelectedPython" in resolve
