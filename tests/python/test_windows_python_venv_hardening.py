# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused contracts for Windows Python wrapper and venv validation."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags=re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _run_powershell(shell: str, script: str, env: dict[str, str]) -> str:
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    return result.stdout.strip()


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_path_python_wrapper_resolves_to_real_executable(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding="utf-8")
    finder = _extract(r"    function Find-CompatiblePython \{.*?\n    \}\n", source)
    wrapper = tmp_path / "python.bat"
    wrapper.write_text(f'@"{sys.executable}" %*\n', encoding="utf-8")

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
    assert Path(_run_powershell(shell, script, env)).resolve() == Path(sys.executable).resolve()


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("case", ["missing", "unlaunchable", "working"])
def test_managed_python_readiness_probe(tmp_path: Path, shell: str, case: str):
    source = INSTALL_PS1.read_text(encoding="utf-8")
    readiness = _extract(r"    function Test-VenvPythonReady \{.*?\n    \}\n", source)
    python_exe = tmp_path / "broken-python.cmd"
    expected = "False"
    if case == "unlaunchable":
        python_exe.write_text("@exit /b 17\n", encoding="utf-8")
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
    source = INSTALL_PS1.read_text(encoding="utf-8")
    gate = source.index("if (-not (Test-VenvPythonReady -PythonExe $VenvPython))")
    first_uv_pip = source.index("uv pip install --python $VenvPython")
    gpu_detection = source.index("function Invoke-AmdSmiNoElevate")

    assert gate < gpu_detection < first_uv_pip
    assert 'Write-Host "        Managed Python: $VenvPython"' in source
    assert 'Write-Host "        Selected base Python: $($DetectedPython.Path)"' in source
    assert 'return (Exit-InstallFailure "Managed Python is unavailable' in source
