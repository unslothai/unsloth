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
import subprocess
from pathlib import Path

import pytest


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
    result = subprocess.run(
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
        # Requested minor is ARM64-only: bootstrap x64 rather than take the native one.
        ([("3.13", "arm64")], True, "3.13|x86_64"),
        # Offline, but an x64 build of a lower-priority minor is here. Use it: the native
        # 3.13 cannot resolve pyarrow or hf-transfer, and this one can.
        ([("3.13", "arm64"), ("3.11", "x86_64")], False, "3.11|x86_64"),
        # ARM64 everywhere: still returned, and the caller warns.
        ([("3.13", "arm64"), ("3.11", "arm64")], False, "3.13|arm64"),
    ],
)
def test_arm64_host_prefers_an_x64_interpreter(installed, can_download, expected):
    assert _pwsh(_resolver_script(installed, can_download)) == expected


# ── setup.ps1: every path that REUSES an interpreter has to re-ask its arch ──
# install.ps1's swap above only runs over a freshly selected interpreter. setup.ps1
# is handed one (UNSLOTH_SETUP_PYTHON, or the venv python), and validated it by
# version and conda-ness alone, so an ARM64 environment sailed through and every
# update it ran ended in the same pyarrow source build (issue #8495).

SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"


def _setup_arch_script(interpreter_tag: str, host_arch: str, no_datasets: bool) -> str:
    """Both production functions verbatim, over one fake interpreter.

    Extracted from setup.ps1 rather than reimplemented, for the same reason as the
    install.ps1 harness above: a copy here would keep passing after the original
    changed. The fake interpreter is a function invoked through the call operator,
    so `& $Exe -S -c ...` needs no real binary.
    """
    source = SETUP_PS1.read_text(encoding = "utf-8")
    tag_fn = _extract(r"function Get-PythonPlatformTag \{.*?\r?\n\}", source)
    arch_fn = _extract(r"function Test-CompatibleSetupPythonArch \{.*?\r?\n\}", source)
    return f"""
$ErrorActionPreference = "Stop"
$script:NoDatasetsMode = {'$true' if no_datasets else '$false'}
function Get-HostMachineArch {{ return "{host_arch}" }}
function FakePython {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    return "{interpreter_tag}"
}}
{tag_fn}
{arch_fn}
if (Test-CompatibleSetupPythonArch "FakePython") {{ Write-Output "accept" }} else {{ Write-Output "reject" }}
"""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("interpreter_tag", "host_arch", "no_datasets", "expected"),
    [
        # The reported failure: an ARM64 interpreter on an ARM64 host, full install.
        ("win-arm64", "arm64", False, "reject"),
        # The supported Windows-on-ARM configuration: x64 CPython under emulation.
        ("win-amd64", "arm64", False, "accept"),
        # The ARM64 inference-only tier runs on win-arm64 on purpose.
        ("win-arm64", "arm64", True, "accept"),
        # Every other host is unconstrained -- and pays no subprocess for the probe.
        ("win-amd64", "x86_64", False, "accept"),
        ("win-arm64", "x86_64", False, "accept"),
        # An unreadable interpreter is not evidence of an x64 build.
        ("", "arm64", False, "reject"),
    ],
)
def test_setup_arch_gate(interpreter_tag, host_arch, no_datasets, expected):
    assert _pwsh(_setup_arch_script(interpreter_tag, host_arch, no_datasets)) == expected


def test_every_setup_interpreter_path_consults_the_arch_gate():
    """One missed call site re-admits what the others reject, and the reused-venv
    path is precisely the one that reached a user (issue #8495)."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    # The 1g reuse gate, the py-launcher loop, the bare-python fallback, and both
    # phase-3 resolution paths.
    assert source.count("Test-CompatibleSetupPythonArch") >= 6


def test_setup_winget_fallback_asks_for_x64():
    """winget defaults to the ARM64 package on an ARM64 host, which is the build
    that cannot resolve the stack."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    assert '@("--architecture", "x64")' in source
    winget_call = source[source.index("Python.Python.3.12 --source winget") - 600 :]
    assert "_wingetArchArgs" in winget_call


def test_setup_refuses_an_arm64_environment_with_an_actionable_message():
    """Failing here beats spending minutes to fail inside a pyarrow build, but only
    if the message names the fix rather than the symptom."""
    source = SETUP_PS1.read_text(encoding = "utf-8")
    index = source.index("Environment uses ARM64 Python")
    message = source[index - 1500 : index + 200]
    assert "python.org" in message or "install.ps1" in message
    assert "UNSLOTH_NO_DATASETS" in message


def test_install_ps1_rechecks_a_migrated_venv():
    """A migrated environment never met the x64 swap, so its interpreter is still
    whatever built it. Probe it, set it aside through the existing rollback, and
    clear $_Migrated so the fresh-install path (not the migrated-upgrade path) runs."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("migrated environment is")
    block = source[index - 1200 : index + 1200]
    assert "Get-PythonPlatformTag $VenvPython" in block
    assert "Start-StudioVenvRollback" in block
    assert "$_Migrated = $false" in block


def test_install_ps1_falls_back_to_the_inference_only_tier():
    """When no x64 interpreter can be obtained, continuing into `uv pip install
    unsloth` only buys a CMake failure. The tier drops the wheel-less packages
    instead, and UNSLOTH_NO_DATASETS carries the choice into setup.ps1."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    index = source.index("Could not install an x64 Python")
    block = source[index - 500 : index + 1200]
    assert "$script:ArmInferenceOnly = $true" in block
    assert 'UNSLOTH_NO_DATASETS = "1"' in block
    assert "--no-deps" in source[source.index("arm64 inference-only") :][:800]
