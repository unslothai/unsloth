# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Contracts for the Windows uv-managed Python fallback in install.ps1."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _run_powershell(shell: str, script: str) -> str:
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        env = os.environ.copy(),
        timeout = 30,
    )
    return result.stdout.strip()


@pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_managed_python_request_uses_requested_minor(shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    helper = _extract(r"    function New-UvManagedPythonRequest \{.*?\n    \}\n", source)

    script = f"""
$ErrorActionPreference = "Stop"
$PythonVersion = "3.13"
{helper}
$req = New-UvManagedPythonRequest
$req | ConvertTo-Json -Compress
"""
    payload = json.loads(_run_powershell(shell, script))
    assert payload == {
        "Version": "3.13",
        "Path": "3.13",
        "Arch": "x86_64",
        "ManagedByUv": True,
        "RequireManagedPython": True,
    }


def test_non_arm64_prefers_uv_managed_python_before_winget_bootstrap():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    uv_step = source.index('Write-TauriLog "STEP" "Installing uv package manager"')
    helper = source.index("function New-UvManagedPythonRequest")
    non_arm64 = source.index('} elseif ((Get-HostMachineArch) -ne "arm64") {')
    managed = source.index("$DetectedPython = New-UvManagedPythonRequest")
    managed_flag = source.index("--managed-python --python")
    venv_create = source.index('step "venv" "creating Python $($DetectedPython.Version) virtual environment"')
    winget = source.index(
        "winget install -e --id $pythonPackageId --source winget --architecture x64"
    )

    assert uv_step < helper < non_arm64 < managed < winget < venv_create < managed_flag
    assert 'step "python" "using uv-managed Python $($DetectedPython.Version)"' in source
    assert (
        'substep "no compatible system Python found; uv will download and manage it for this environment"'
        in source
    )
    assert (
        'uv venv $VenvDir --managed-python --python "$($DetectedPython.Path)"' in source
    )


def test_arm64_python_bootstrap_stays_x64_specific():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    assert "winget install -e --id $pythonPackageId --source winget --architecture x64" in source
    assert 'Install-PythonFromPythonOrg -Arch "x86_64"' in source
