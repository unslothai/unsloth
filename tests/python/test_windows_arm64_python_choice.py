# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows on ARM: install.ps1 must not settle for a native ARM64 interpreter.

pyarrow (via datasets) and hf-transfer publish no win_arm64 wheels, so an ARM64
Python source-builds both and dies minutes into the run. The resolver prefers an
x64 build of the requested minor and bootstraps one otherwise; the case pinned
here is the recovery path, where nothing can be downloaded but an x64 build of a
lower-priority supported minor is already installed.
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


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _resolver_script(installed: list[tuple[str, str]], can_download: bool) -> str:
    """Both production functions verbatim, over a fake set of interpreters.

    Extracted rather than reimplemented so the test cannot drift away from the
    text install.ps1 actually runs. `installed` is (minor, arch) in py-launcher
    order, so the first entry for a minor is what a bare `py -3.13` resolves to.
    The fake interpreters are named `*.exe` and invoked through the call operator,
    which resolves a string to a function, so no real binary is needed.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    finder = _extract(r"    function Find-CompatiblePython \{.*?\n    \}\n", source)
    installer = _extract(r"    function Install-X64Python \{.*?\n    \}\n", source)

    names = [f"Py{minor.replace('.', '')}{arch}.exe" for minor, arch in installed]
    table = ", ".join(
        f'@{{ Minor = "{minor}"; Arch = "{arch}"; Name = "{name}" }}'
        for (minor, arch), name in zip(installed, names)
    )
    downloaded = (
        '@{ Version = "3.13"; Path = "Downloaded.exe"; Arch = "x86_64" }'
        if can_download
        else "$null"
    )
    version_stubs = "\n".join(
        f"function {name} {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest)\n"
        f'    if ($Rest -contains "--version") {{ return "Python {minor}.0" }}\n'
        f'    return "{name}" }}'
        for (minor, _arch), name in zip(installed, names)
    )
    return f"""
$ErrorActionPreference = "Stop"
$PythonVersion = "3.13"
$script:WingetAvailable = $false
$script:CondaSkipPattern = 'conda'
$Interpreters = @({table})
{version_stubs}
# `py -0p` lists every registration; `py -3.x` runs the launcher's preferred build
# for that minor, which on an ARM64 host is normally the native one.
function FakePy {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($Rest -contains "-0p") {{
        return @($Interpreters | ForEach-Object {{ "  -V:$($_.Minor) *        $($_.Name)" }})
    }}
    $minor = ([string]$Rest[0]).TrimStart('-')
    $hit = @($Interpreters | Where-Object {{ $_.Minor -eq $minor }})
    if ($hit.Count -eq 0) {{ return "" }}
    if ($Rest -contains "--version") {{ return "Python $minor.0" }}
    return $hit[0].Name
}}
function substep {{ param($a, $b) }}
function Get-HostMachineArch {{ return "arm64" }}
function Get-Command {{
    param([Parameter(Position = 0)][string]$Name,
          [Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($Name -eq "py") {{ return @([pscustomobject]@{{ Source = "FakePy" }}) }}
    return @()
}}
function Test-Path {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) return $true }}
function Test-IsCondaPython {{ param([string]$Exe) return $false }}
function Get-PythonPlatformTag {{
    param([string]$Exe)
    foreach ($i in $Interpreters) {{
        if ($i.Name -eq $Exe) {{
            if ($i.Arch -eq "x86_64") {{ return "win-amd64" }} else {{ return "win-arm64" }}
        }}
    }}
    return "win-amd64"
}}
function Refresh-SessionPath {{ }}
function Install-PythonFromPythonOrg {{ param([string]$Arch = "") return {downloaded} }}
{finder}
{installer}
# The caller's ARM64 swap, condensed to what decides the interpreter.
$found = Find-CompatiblePython
if ($found -and $found.Arch -ne "x86_64") {{
    $x64 = Install-X64Python
    if ($x64) {{ $found = $x64 }}
}}
if ($found) {{ Write-Output "$($found.Version)|$($found.Arch)" }} else {{ Write-Output "none" }}
"""


def _pwsh(script: str) -> str:
    # Every ARM64 case below is decided by the one "version|arch" line this run prints, and check = True means a pwsh
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        env = os.environ.copy(),
    )
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("installed", "can_download", "expected"),
    [
        # An x64 build of the requested minor wins outright, downloads irrelevant.
        ([("3.13", "arm64"), ("3.13", "x86_64")], False, "3.13|x86_64"),
        # Requested minor is ARM64-only:
        ([("3.13", "arm64")], True, "3.13|x86_64"),
        # 3.13 cannot resolve pyarrow or hf-transfer, and this one can.
        # Offline, but an x64 build of a lower-priority minor is here.
        ([("3.13", "arm64"), ("3.11", "x86_64")], False, "3.11|x86_64"),
        # ARM64 everywhere: still returned, and the caller warns.
        ([("3.13", "arm64"), ("3.11", "arm64")], False, "3.13|arm64"),
    ],
)
def test_arm64_host_prefers_an_x64_interpreter(installed, can_download, expected):
    assert _pwsh(_resolver_script(installed, can_download)) == expected
