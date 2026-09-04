# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused contracts for Windows Python wrapper and venv validation."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _link_dir(link: Path, target: Path) -> None:
    """Directory link, without needing SeCreateSymbolicLinkPrivilege on Windows.

    A junction is also the reparse point a Windows venv actually runs into, so this
    is the faithful construct there rather than a stand-in.
    """
    if os.name == "nt":
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            check = True,
            capture_output = True,
            text = True,
        )
    else:
        os.symlink(target, link, target_is_directory = True)


def _run_powershell(shell: str, script: str, env: dict[str, str]) -> str:
    # run_pwsh, not subprocess.run: $shell is always pwsh or powershell (see POWERSHELLS),
    # and every venv and rollback case in this file reads its stdout, so an interpreter that
    # died at startup would surface as install.ps1 losing half a moved environment.
    # See tests/_shared/unsloth_pwsh_runner.py.
    result = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        env = env,
        timeout = 30,
    )
    return result.stdout.strip()


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_path_python_wrapper_resolves_to_real_executable(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    finder = _extract(r"    function Find-CompatiblePython \{.*?\n    \}\n", source)
    (tmp_path / "sitecustomize.py").write_text('print("STARTUP_BANNER")\n', encoding = "utf-8")
    if os.name == "nt":
        wrapper = tmp_path / "python.bat"
        wrapper.write_text(f'@"{sys.executable}" %*\n', encoding = "utf-8")
    else:
        wrapper = tmp_path / "python-wrapper"
        wrapper.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n', encoding = "utf-8"
        )
        wrapper.chmod(0o755)

    script = f"""
$ErrorActionPreference = "Stop"
$PythonVersion = "3.13"
$script:CondaSkipPattern = '(?i)(conda|miniconda|anaconda)'
function Get-HostMachineArch {{ return "x86_64" }}
function Test-IsCondaPython {{ param([string]$Exe) return $false }}
function Get-PythonPlatformTag {{ param([string]$Exe) return "win-amd64" }}
function Get-Command {{
    param([Parameter(Position = 0)][string]$Name,
          [Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($Name -eq "python") {{
        return @([pscustomobject]@{{ Source = $env:TEST_PYTHON_WRAPPER }})
    }}
    return @()
}}
{finder}
$found = Find-CompatiblePython
Write-Output $found.Path
"""
    env = os.environ.copy()
    env["TEST_PYTHON_WRAPPER"] = str(wrapper)
    env["PYTHONPATH"] = str(tmp_path)
    assert Path(_run_powershell(shell, script, env)).resolve() == Path(sys.executable).resolve()


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_arch_probe_ignores_startup_output(tmp_path: Path, shell: str):
    """Startup output must not reach the arch tag.

    The caller compares the tag with -eq "win-amd64", so a contaminated answer reads
    as "unknown" and Windows on ARM silently settles for a native ARM64 interpreter.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    probe = _extract(r"    function Get-PythonPlatformTag \{.*?\n    \}\n", source)
    (tmp_path / "sitecustomize.py").write_text('print("STARTUP_BANNER")\n', encoding = "utf-8")

    script = f"""
$ErrorActionPreference = "Stop"
{probe}
Write-Output (Get-PythonPlatformTag $env:TEST_PYTHON)
"""
    env = os.environ.copy()
    env["TEST_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(tmp_path)
    tag = _run_powershell(shell, script, env)
    assert tag and "\n" not in tag and "startup_banner" not in tag, tag


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_venv_base_home_comes_from_pyvenv_config(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    reader = _extract(r"    function Get-VenvBaseHome \{.*?\n    \}\n", source)
    expected = tmp_path / "removed-base-python"
    (tmp_path / "pyvenv.cfg").write_text(f"home = {expected}\n", encoding = "utf-8")

    script = f"""
$ErrorActionPreference = "Stop"
{reader}
Write-Output (Get-VenvBaseHome -VenvRoot $env:TEST_VENV_ROOT)
"""
    env = os.environ.copy()
    env["TEST_VENV_ROOT"] = str(tmp_path)
    assert _run_powershell(shell, script, env) == str(expected)


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("case", ["partial", "clean"])
def test_rollback_keeps_state_when_the_move_stops_partway(tmp_path: Path, shell: str, case: str):
    """A half-finished rename must not be read as "the rename never happened".

    On Windows an open handle inside the tree fails Move-Item after it has already
    walked part of it, so entries exist at both paths. Testing only the source then
    clears StudioVenvRollbackDir -- the sole record of where the other half went --
    and the environment is stranded with no way to restore or even name it.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    rollback = _extract(r"    function Start-StudioVenvRollback \{.*?\n    \}\n", source)
    existing = tmp_path / "unsloth_studio"
    (existing / "Scripts").mkdir(parents = True)
    (existing / "Scripts" / "unsloth.exe").write_text("locked", encoding = "utf-8")

    script = f"""
$ErrorActionPreference = "Stop"
$StudioHome = $env:TEST_STUDIO_HOME
function substep {{ param([string]$Text, [string]$Color) }}
# The split-move warning goes through install.ps1's UTF-8 stdout sink. Echo it so
# the assertions below can read it; without this the call is a command-not-found
# terminating error that the try/catch swallows, and the warning is simply lost.
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) Write-Host $Message }}
function Move-Item {{
    param([string]$LiteralPath, [string]$Destination, [string]$ErrorAction, [switch]$Force)
    if ($env:TEST_ROLLBACK_CASE -eq "partial") {{
        # The entries walked before the locked one are already at the destination.
        [System.IO.Directory]::CreateDirectory((Join-Path $Destination "Lib")) | Out-Null
    }}
    throw "The process cannot access the file because it is being used by another process."
}}
{rollback}
try {{ Start-StudioVenvRollback -ExistingDir $env:TEST_EXISTING_DIR }} catch {{ }}
Write-Output ("active=" + $script:StudioVenvRollbackActive)
Write-Output ("dir=" + [string]$script:StudioVenvRollbackDir)
"""
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(tmp_path)
    env["TEST_EXISTING_DIR"] = str(existing)
    env["TEST_ROLLBACK_CASE"] = case
    out = _run_powershell(shell, script, env)
    state = dict(
        line.split("=", 1) for line in out.splitlines() if line.startswith(("active=", "dir="))
    )

    if case == "clean":
        # Nothing moved, so the original is intact and there is nothing to restore.
        assert state["active"] == "False", out
        assert state["dir"] == "", out
        return

    assert state["active"] == "True", out
    assert state["dir"].startswith(os.path.join(str(tmp_path), "unsloth_studio.rollback.")), out
    assert Path(state["dir"]).is_dir(), out
    # Both halves are named, so the user is not left hunting for the moved tree.
    # Match the warning lines themselves rather than the bare paths: $existing is a
    # prefix of the rollback dir, so "str(existing) in out" alone is satisfied by the
    # dir= line and would stay green even with the warning missing entirely.
    assert f"still in place: {existing}" in out, out
    assert f"moved aside:    {state['dir']}" in out, out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_restoring_a_split_move_never_deletes_the_half_left_behind(tmp_path: Path, shell: str):
    """Restoration must merge the two halves, not clear the destination first.

    After a partway move the target holds the entries the move never got to -- not
    an incomplete *new* environment. The committed-replacement path removes the
    target before moving the backup back, which for a split tree deletes files that
    exist nowhere else. Keeping the rollback active is only safe if restoration
    takes a merge path, so this pins the file that never moved to being still there.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    blocks = "".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            "Test-StudioPathPresent",
            "Remove-StudioVenvTreeWithRetry",
            "Merge-StudioVenvRollbackTree",
            "Restore-StudioVenvRollback",
        )
    )
    target = tmp_path / "unsloth_studio"
    backup = tmp_path / "unsloth_studio.rollback.20260804120000.999"
    # The half the interrupted move left behind, and the half that got across.
    (target / "Scripts").mkdir(parents = True)
    (target / "Scripts" / "unsloth.exe").write_text("irreplaceable", encoding = "utf-8")
    (backup / "Lib" / "site-packages").mkdir(parents = True)
    (backup / "Lib" / "site-packages" / "marker.txt").write_text("moved", encoding = "utf-8")

    script = f"""
$ErrorActionPreference = "Stop"
function substep {{ param([string]$Text, [string]$Color) }}
# The merge/restore helpers warn through install.ps1's UTF-8 stdout sink on their
# conflict branches. This run does not take one, but leaving the sink undefined
# means any future case that does would die on a command-not-found instead.
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) Write-Host $Message }}
{blocks}
$script:StudioVenvRollbackActive  = $true
$script:StudioVenvRollbackDir     = $env:TEST_BACKUP_DIR
$script:StudioVenvRollbackTarget  = $env:TEST_TARGET_DIR
$script:StudioVenvRollbackPartial = $true
Restore-StudioVenvRollback
Write-Output ("active=" + $script:StudioVenvRollbackActive)
"""
    env = os.environ.copy()
    env["TEST_BACKUP_DIR"] = str(backup)
    env["TEST_TARGET_DIR"] = str(target)
    out = _run_powershell(shell, script, env)

    # The file that never moved is the whole point: the pre-merge path deleted it.
    assert (target / "Scripts" / "unsloth.exe").read_text(encoding = "utf-8") == "irreplaceable", out
    # ...and the half that did move comes back rather than being stranded.
    assert (target / "Lib" / "site-packages" / "marker.txt").is_file(), out
    assert not backup.exists(), out
    assert "active=False" in out, out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_merging_a_split_move_keeps_every_sibling_at_its_own_path(tmp_path: Path, shell: str):
    """Each entry must land at its own path, not nested under the previous one.

    PowerShell variable names are case-insensitive, so a per-entry $destination
    reassigns the $Destination parameter. Only the first sibling at a level then
    lands correctly and the rest are appended to its path, so a restored venv comes
    back with pyvenv.cfg buried inside Lib. One entry per level hides it, so this
    uses several.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    blocks = "".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            "Test-StudioPathPresent",
            "Remove-StudioVenvTreeWithRetry",
            "Merge-StudioVenvRollbackTree",
            "Restore-StudioVenvRollback",
        )
    )
    target = tmp_path / "unsloth_studio"
    backup = tmp_path / "unsloth_studio.rollback.20260804120000.999"
    # The half left behind, and a moved half with several siblings at two levels.
    (target / "Scripts").mkdir(parents = True)
    (target / "Scripts" / "unsloth.exe").write_text("irreplaceable", encoding = "utf-8")
    (target / "Lib").mkdir()
    (target / "Lib" / "stayed.py").write_text("stayed", encoding = "utf-8")
    (backup / "Lib" / "site-packages").mkdir(parents = True)
    (backup / "Lib" / "site-packages" / "marker.txt").write_text("moved", encoding = "utf-8")
    (backup / "Lib" / "other.py").write_text("other", encoding = "utf-8")
    (backup / "pyvenv.cfg").write_text("cfg", encoding = "utf-8")
    (backup / "unsloth_install_manifest.json").write_text("{}", encoding = "utf-8")

    script = f"""
$ErrorActionPreference = "Stop"
function substep {{ param([string]$Text, [string]$Color) }}
# The merge/restore helpers warn through install.ps1's UTF-8 stdout sink on their
# conflict branches. This run does not take one, but leaving the sink undefined
# means any future case that does would die on a command-not-found instead.
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) Write-Host $Message }}
{blocks}
$script:StudioVenvRollbackActive  = $true
$script:StudioVenvRollbackDir     = $env:TEST_BACKUP_DIR
$script:StudioVenvRollbackTarget  = $env:TEST_TARGET_DIR
$script:StudioVenvRollbackPartial = $true
Restore-StudioVenvRollback
Write-Output ("active=" + $script:StudioVenvRollbackActive)
"""
    env = os.environ.copy()
    env["TEST_BACKUP_DIR"] = str(backup)
    env["TEST_TARGET_DIR"] = str(target)
    out = _run_powershell(shell, script, env)

    restored = sorted(
        str(p.relative_to(target)).replace("\\", "/") for p in target.rglob("*") if p.is_file()
    )
    assert restored == [
        "Lib/other.py",
        "Lib/site-packages/marker.txt",
        "Lib/stayed.py",
        "Scripts/unsloth.exe",
        "pyvenv.cfg",
        "unsloth_install_manifest.json",
    ], out
    assert not backup.exists(), out
    assert "active=False" in out, out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("side", ["destination", "source"])
def test_merging_a_split_move_never_walks_through_a_link(tmp_path: Path, shell: str, side: str):
    """A junction on either side is a leaf, not a subtree to recurse into.

    Recursing through one moves venv files to wherever the link points, outside
    $StudioHome, and replaces the link with a real directory. Either half can carry
    the link, so both directions are pinned here.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    blocks = "".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            "Test-StudioPathPresent",
            "Remove-StudioVenvTreeWithRetry",
            "Merge-StudioVenvRollbackTree",
            "Restore-StudioVenvRollback",
        )
    )
    target = tmp_path / "unsloth_studio"
    backup = tmp_path / "unsloth_studio.rollback.20260804120000.999"
    # A sibling of the environment, never under it.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("untouched", encoding = "utf-8")
    target.mkdir()
    backup.mkdir()

    linked, real = (target, backup) if side == "destination" else (backup, target)
    _link_dir(linked / "Lib", outside)
    (real / "Lib").mkdir()
    (real / "Lib" / "payload.txt").write_text("venv-only", encoding = "utf-8")

    script = f"""
$ErrorActionPreference = "Stop"
function substep {{ param([string]$Text, [string]$Color) }}
# Restore-StudioVenvRollback warns through install.ps1's UTF-8 stdout sink. Echo it
# rather than swallowing it, so a warning stays visible in the assertion message.
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) Write-Host $Message }}
{blocks}
$script:StudioVenvRollbackActive  = $true
$script:StudioVenvRollbackDir     = $env:TEST_BACKUP_DIR
$script:StudioVenvRollbackTarget  = $env:TEST_TARGET_DIR
$script:StudioVenvRollbackPartial = $true
Restore-StudioVenvRollback
Write-Output ("active=" + $script:StudioVenvRollbackActive)
"""
    env = os.environ.copy()
    env["TEST_BACKUP_DIR"] = str(backup)
    env["TEST_TARGET_DIR"] = str(target)
    out = _run_powershell(shell, script, env)

    # Nothing from the environment may be written through the link.
    assert not (outside / "payload.txt").exists(), out
    assert sorted(p.name for p in outside.iterdir()) == ["keep.txt"], out
    assert (outside / "keep.txt").read_text(encoding = "utf-8") == "untouched", out
    # An unresolved conflict keeps both copies, so the rollback stays tracked.
    assert "active=True" in out, out


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("case", ["missing", "unlaunchable", "working"])
def test_managed_python_readiness_probe(tmp_path: Path, shell: str, case: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    readiness = _extract(r"    function Test-VenvPythonReady \{.*?\n    \}\n", source)
    python_exe = tmp_path / "broken-python.cmd"
    expected = "False"
    if case == "unlaunchable":
        python_exe.write_text("@exit /b 17\n", encoding = "utf-8")
    elif case == "working":
        python_exe = Path(sys.executable)
        expected = "True"

    script = f"""
$ErrorActionPreference = "Stop"
{readiness}
Write-Output (Test-VenvPythonReady -PythonExe $env:TEST_MANAGED_PYTHON)
"""
    env = os.environ.copy()
    env["TEST_MANAGED_PYTHON"] = str(python_exe)
    assert _run_powershell(shell, script, env) == expected


def _run_uv_venv_creation_result(
    tmp_path: Path,
    shell: str,
    uv_mode: str,
    fallback_mode: str,
    migrated: bool = False,
    foreign: bool = False,
    detected_python_missing: bool = False,
    foreign_pyvenv: bool = False,
) -> dict[str, str]:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    readiness = _extract(r"    function Test-VenvPythonReady \{.*?\n    \}\n", source)
    creation = _extract(
        r"    \$venvDirExistedBeforeCreation = Test-Path -LiteralPath \$VenvDir\n    \$venvDirHasOwnershipEvidence = Test-Path -LiteralPath .*?\n    \$fallbackVenvExit = \$null\n    if \(-not \(Test-Path -LiteralPath \$VenvPython\)\) \{.*?(?=\n\n    # Mark the managed venv)",
        source,
    )
    marker_and_readiness = _extract(
        r"    # Mark the managed venv before probing.*?(?=\n\n    # .*Helper: run amd-smi)",
        source,
    )

    script_root = tmp_path / "nonstandard Python path with spaces"
    script_root.mkdir()
    uv_stub = script_root / "uv stub.ps1"
    detected_python = script_root / "selected detected Python.ps1"
    fallback_log = tmp_path / "fallback args.log"
    uv_log = tmp_path / "uv args.log"
    venv_dir = tmp_path / "managed venv with spaces"
    uv_stub.write_text(
        """
$target = $args[1]
New-Item -ItemType Directory -Force -Path $target | Out-Null
if ($env:TEST_UV_MODE -eq "ready") {
    New-Item -ItemType Directory -Force -Path (Join-Path $target "Scripts") | Out-Null
    Copy-Item -LiteralPath $env:TEST_REAL_PYTHON -Destination (Join-Path $target "Scripts\\python.exe") -Force
} elseif ($env:TEST_UV_MODE -eq "unlaunchable") {
    New-Item -ItemType Directory -Force -Path (Join-Path $target "Scripts\\python.exe") | Out-Null
} elseif ($env:TEST_UV_MODE -eq "nonzero_ready") {
    New-Item -ItemType Directory -Force -Path (Join-Path $target "Scripts") | Out-Null
    Copy-Item -LiteralPath $env:TEST_REAL_PYTHON -Destination (Join-Path $target "Scripts\\python.exe") -Force
}
[System.IO.File]::WriteAllText($env:TEST_UV_LOG, ($args -join "|"))
if ($env:TEST_UV_MODE -eq "nonzero" -or $env:TEST_UV_MODE -eq "nonzero_ready") { exit 17 }
exit 0
""".strip(),
        encoding = "utf-8",
    )
    detected_python.write_text(
        """
if ($args[0] -eq "-c") { exit 0 }
[System.IO.File]::WriteAllText($env:TEST_FALLBACK_LOG, ($args -join "|"))
$target = $args[2]
New-Item -ItemType Directory -Force -Path $target | Out-Null
$pythonPath = Join-Path $target "Scripts\\python.exe"
if (Test-Path -LiteralPath $pythonPath) {
    Remove-Item -LiteralPath $pythonPath -Force -Recurse
}
if ($env:TEST_FALLBACK_MODE -eq "ready") {
    New-Item -ItemType Directory -Force -Path (Join-Path $target "Scripts") | Out-Null
    Copy-Item -LiteralPath $env:TEST_REAL_PYTHON -Destination (Join-Path $target "Scripts\\python.exe") -Force
} elseif ($env:TEST_FALLBACK_MODE -eq "unusable") {
    New-Item -ItemType Directory -Force -Path (Join-Path $target "Scripts\\python.exe") | Out-Null
}
if ($env:TEST_FALLBACK_MODE -eq "nonzero") { exit 23 }
exit 0
""".strip(),
        encoding = "utf-8",
    )
    if detected_python_missing:
        detected_python.unlink()

    script = f"""
$ErrorActionPreference = "Stop"
$VenvDir = $env:TEST_VENV_DIR
$VenvPython = Join-Path $VenvDir "Scripts\\python.exe"
$StudioHome = $env:TEST_STUDIO_HOME
$studioUsesLegacyLayout = $false
$_Migrated = $false
$script:UvExe = $env:TEST_UV_EXE
$DetectedPython = [pscustomobject]@{{ Path = $env:TEST_DETECTED_PYTHON; Version = "3.13" }}
$script:TestCallLabels = @()
$script:FailureMessage = $null
$script:FailureExitCode = $null
$script:PackageInstallReached = $false

function step {{ param([string]$Name, [string]$Message, [string]$Color) }}
function substep {{ param([string]$Message, [string]$Color) }}
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) }}
function Exit-InstallFailure {{
    param([string]$Message, [int]$ExitCode = 1)
    $script:FailureMessage = $Message
    $script:FailureExitCode = $ExitCode
    return $Message
}}
function Get-VenvBaseHome {{ param([string]$VenvRoot) return $null }}
function Invoke-InstallCommand {{
    param([string]$Label, [scriptblock]$Command)
    $script:TestCallLabels += $Label
    & $Command
    return $LASTEXITCODE
}}

{readiness}

if ($env:TEST_MIGRATED -eq "1") {{
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvPython) | Out-Null
    Copy-Item -LiteralPath $env:TEST_REAL_PYTHON -Destination $VenvPython -Force
}}
if ($env:TEST_FOREIGN_DIR -eq "1") {{
    New-Item -ItemType Directory -Force -Path $VenvDir | Out-Null
    Set-Content -LiteralPath (Join-Path $VenvDir "foreign.txt") -Value "foreign"
    if ($env:TEST_FOREIGN_PYVENV -eq "1") {{
        Set-Content -LiteralPath (Join-Path $VenvDir "pyvenv.cfg") -Value "home = foreign"
    }}
}}

function Invoke-TestCreation {{
{creation}
{marker_and_readiness}
    $script:PackageInstallReached = $true
}}
Invoke-TestCreation | Out-Null
Write-Output ("calls=" + ($script:TestCallLabels -join ","))
if (Test-Path -LiteralPath $env:TEST_FALLBACK_LOG -PathType Leaf) {{
    Write-Output ("fallback_args=" + [System.IO.File]::ReadAllText($env:TEST_FALLBACK_LOG))
    }} else {{
        Write-Output "fallback_args="
    }}
    if (Test-Path -LiteralPath $env:TEST_UV_LOG -PathType Leaf) {{
        Write-Output ("uv_args=" + [System.IO.File]::ReadAllText($env:TEST_UV_LOG))
    }} else {{
        Write-Output "uv_args="
    }}
    Write-Output ("marker=" + (Test-Path -LiteralPath (Join-Path $VenvDir ".unsloth-studio-owned") -PathType Leaf))
Write-Output ("failure=" + ($null -ne $script:FailureMessage))
Write-Output ("failure_code=" + $script:FailureExitCode)
Write-Output ("failure_message=" + $script:FailureMessage)
Write-Output ("foreign=" + (Test-Path -LiteralPath (Join-Path $VenvDir "foreign.txt") -PathType Leaf))
Write-Output ("package=" + $script:PackageInstallReached)
"""
    env = os.environ.copy()
    env["TEST_UV_EXE"] = str(uv_stub)
    env["TEST_DETECTED_PYTHON"] = str(detected_python)
    env["TEST_REAL_PYTHON"] = sys.executable
    env["TEST_UV_MODE"] = uv_mode
    env["TEST_FALLBACK_MODE"] = fallback_mode
    env["TEST_FALLBACK_LOG"] = str(fallback_log)
    env["TEST_UV_LOG"] = str(uv_log)
    env["TEST_VENV_DIR"] = str(venv_dir)
    env["TEST_STUDIO_HOME"] = str(tmp_path / "studio home")
    env["TEST_MIGRATED"] = "1" if migrated else "0"
    env["TEST_FOREIGN_DIR"] = "1" if foreign else "0"
    env["TEST_FOREIGN_PYVENV"] = "1" if foreign_pyvenv else "0"
    env["PATH"] = os.pathsep.join((str(Path(sys.executable).parent), env.get("PATH", "")))
    output = _run_powershell(shell, script, env)
    return dict(line.split("=", 1) for line in output.splitlines())


def _assert_uv_invocation(state: dict[str, str], tmp_path: Path):
    args = state["uv_args"].split("|")
    assert args[0] == "venv", state
    assert Path(args[1]).resolve() == (tmp_path / "managed venv with spaces").resolve(), state
    assert args[2] == "--python", state
    assert Path(args[3]).name == "selected detected Python.ps1", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_venv_creation_result_ready_uv_skips_fallback(tmp_path: Path, shell: str):
    state = _run_uv_venv_creation_result(tmp_path, shell, "ready", "ready")
    assert state["calls"] == "create virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    assert state["fallback_args"] == "", state
    assert state["marker"] == "True", state
    assert state["failure"] == "False", state
    assert state["package"] == "True", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("uv_mode", ["missing", "unlaunchable"])
def test_uv_venv_creation_result_zero_exit_unusable_uv_uses_fallback(
    tmp_path: Path, shell: str, uv_mode: str
):
    state = _run_uv_venv_creation_result(tmp_path, shell, uv_mode, "ready")
    assert state["calls"] == "create virtual environment,repair virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    args = state["fallback_args"].split("|")
    assert args[:2] == ["-m", "venv"], state
    assert Path(args[2]).resolve() == (tmp_path / "managed venv with spaces").resolve(), state
    assert state["marker"] == "True", state
    assert state["failure"] == "False", state
    assert state["package"] == "True", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_venv_creation_result_nonzero_uv_uses_selected_python_fallback(
    tmp_path: Path, shell: str
):
    state = _run_uv_venv_creation_result(tmp_path, shell, "nonzero", "ready")
    assert state["calls"] == "create virtual environment,repair virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    args = state["fallback_args"].split("|")
    assert args[:2] == ["-m", "venv"], state
    assert Path(args[2]).resolve() == (tmp_path / "managed venv with spaces").resolve(), state
    assert state["marker"] == "True", state
    assert state["failure"] == "False", state
    assert state["package"] == "True", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_venv_creation_result_nonzero_ready_uv_still_uses_fallback(tmp_path: Path, shell: str):
    state = _run_uv_venv_creation_result(tmp_path, shell, "nonzero_ready", "ready")
    assert state["calls"] == "create virtual environment,repair virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    assert state["failure"] == "False", state
    assert state["package"] == "True", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_venv_creation_result_missing_selected_python_stops_before_packages(
    tmp_path: Path, shell: str
):
    state = _run_uv_venv_creation_result(
        tmp_path, shell, "nonzero_ready", "ready", detected_python_missing = True
    )
    assert state["calls"] == "create virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    assert state["marker"] == "True", state
    assert state["failure"] == "True", state
    assert state["failure_code"] == "1", state
    assert "Failed to repair virtual environment (exit code 1)" in state["failure_message"], state
    assert state["package"] == "False", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_venv_creation_result_nonzero_fallback_stops_before_packages(tmp_path: Path, shell: str):
    state = _run_uv_venv_creation_result(tmp_path, shell, "nonzero", "nonzero")
    assert state["calls"] == "create virtual environment,repair virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    assert state["marker"] == "True", state
    assert state["failure"] == "True", state
    assert state["failure_code"] == "23", state
    assert "Failed to repair virtual environment (exit code 23)" in state["failure_message"], state
    assert state["package"] == "False", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_venv_creation_result_unusable_fallback_stops_before_packages(
    tmp_path: Path, shell: str
):
    state = _run_uv_venv_creation_result(tmp_path, shell, "nonzero", "unusable")
    assert state["calls"] == "create virtual environment,repair virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    assert state["marker"] == "True", state
    assert state["failure"] == "True", state
    assert state["failure_code"] == "1", state
    assert "Managed Python is unavailable" in state["failure_message"], state
    assert state["package"] == "False", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("foreign_pyvenv", [False, True])
def test_uv_venv_creation_result_preserves_foreign_target(
    tmp_path: Path, shell: str, foreign_pyvenv: bool
):
    state = _run_uv_venv_creation_result(
        tmp_path, shell, "nonzero", "ready", foreign = True, foreign_pyvenv = foreign_pyvenv
    )
    assert state["calls"] == "create virtual environment", state
    _assert_uv_invocation(state, tmp_path)
    assert state["fallback_args"] == "", state
    assert state["marker"] == "False", state
    assert state["failure"] == "True", state
    assert state["failure_code"] == "1", state
    assert "unowned virtual environment directory" in state["failure_message"], state
    assert state["foreign"] == "True", state
    assert state["package"] == "False", state


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_venv_creation_result_migrated_environment_skips_creator_fallback(
    tmp_path: Path, shell: str
):
    state = _run_uv_venv_creation_result(tmp_path, shell, "nonzero", "ready", migrated = True)
    assert state["calls"] == "", state
    assert state["fallback_args"] == "", state
    assert state["uv_args"] == "", state
    assert state["marker"] == "True", state
    assert state["failure"] == "False", state
    assert state["package"] == "True", state


def test_readiness_gate_precedes_installs_and_names_both_interpreters():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    gate = source.index("if (-not (Test-VenvPythonReady -PythonExe $VenvPython))")
    marker = source.index(
        '[System.IO.File]::WriteAllText((Join-Path $VenvDir ".unsloth-studio-owned"), "")'
    )
    # Anchored past the command token: uv is invoked as the resolved $script:UvExe.
    first_uv_pip = source.index("pip install --python $VenvPython")
    gpu_detection = source.index("function Invoke-AmdSmiNoElevate")

    assert marker < gate < gpu_detection < first_uv_pip
    assert 'Write-StudioLine "        Managed Python: $VenvPython"' in source
    assert 'Write-StudioLine "        Recorded base Python home: $recordedBaseHome"' in source
    assert 'return (Exit-InstallFailure "Managed Python is unavailable' in source
