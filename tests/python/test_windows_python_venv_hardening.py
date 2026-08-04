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


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _run_powershell(shell: str, script: str, env: dict[str, str]) -> str:
    result = subprocess.run(
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
    assert str(existing) in out and state["dir"] in out, out


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


def test_readiness_gate_precedes_installs_and_names_both_interpreters():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    gate = source.index("if (-not (Test-VenvPythonReady -PythonExe $VenvPython))")
    marker = source.index(
        '[System.IO.File]::WriteAllText((Join-Path $VenvDir ".unsloth-studio-owned"), "")'
    )
    first_uv_pip = source.index("uv pip install --python $VenvPython")
    gpu_detection = source.index("function Invoke-AmdSmiNoElevate")

    assert marker < gate < gpu_detection < first_uv_pip
    assert 'Write-Host "        Managed Python: $VenvPython"' in source
    assert 'Write-Host "        Recorded base Python home: $recordedBaseHome"' in source
    assert 'return (Exit-InstallFailure "Managed Python is unavailable' in source
