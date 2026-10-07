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
    # Use the real opt-out reader, not a stub, or the extracted function calls a missing command.
    opt_out = _extract(r"    function Test-Arm64PythonOptOut \{.*?\n    \}\n", source)
    conda_active = _extract(r"    function Test-ActiveCondaEnvironment \{.*?\n    \}\n", source)

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
            if ($i.Arch -eq "x86_64") {{ return "win-amd64" }}
            elseif ($i.Arch -eq "arm64") {{ return "win-arm64" }}
            else {{ return $i.Arch }}
        }}
    }}
    return "win-amd64"
}}
function Refresh-SessionPath {{ }}
function Install-PythonFromPythonOrg {{ param([string]$Arch = "") return {downloaded} }}
{opt_out}
{conda_active}
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


def _opt_out_script(installed: list[tuple[str, str]], can_download: bool) -> str:
    """The same fakes, but ending in the REAL swap resolver rather than a condensed one.

    UNSLOTH_ALLOW_ARM64_PYTHON is read in two places -- the selection and the swap after it
    -- and the cases below are about both, so the swap cannot be a paraphrase here.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    resolver = _extract(r"    function Resolve-WindowsOnArmX64Python \{.*?\n    \}\n", source)
    base = _resolver_script(installed, can_download)
    tail = base.index("# The caller's ARM64 swap")
    return base[:tail] + (
        "function Write-StudioLine { param([Parameter(ValueFromRemainingArguments = $true)]$Rest) }\n"
        "function step { param($a, $b, $c) }\n" + resolver + "\n$found = Find-CompatiblePython\n"
        'if ($found -and $found.Arch -ne "x86_64") { $found = Resolve-WindowsOnArmX64Python $found }\n'
        'if ($found) { Write-Output "$($found.Version)|$($found.Arch)" } else { Write-Output "none" }\n'
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("installed", "can_download", "expected"),
    [
        # The opt-out wants a native interpreter, even over the preferred minor's x64 build.
        ([("3.13", "x86_64"), ("3.12", "arm64")], True, "3.12|arm64"),
        ([("3.13", "arm64"), ("3.12", "arm64")], True, "3.13|arm64"),
        ([("3.13", "x86_64")], False, "3.13|x86_64"),
        # A 32-bit interpreter is not ARM64; returning it here would label it native ARM64.
        ([("3.13", "win32")], True, "3.13|x86_64"),
        ([("3.13", "win32")], False, "none"),
    ],
)
def test_the_arm64_opt_out_selects_a_native_interpreter(installed, can_download, expected):
    environment = _environment_without_the_arm64_opt_out()
    environment["UNSLOTH_ALLOW_ARM64_PYTHON"] = "1"
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _opt_out_script(installed, can_download),
        ],
        check = True,
        capture_output = True,
        text = True,
        env = environment,
    )
    assert result.stdout.strip() == expected


def _environment_without_the_arm64_opt_out() -> dict:
    """These cases are the DEFAULT ARM64 host, which is the one that must prefer x64.

    UNSLOTH_ALLOW_ARM64_PYTHON now reaches the selection rather than only the swap after it,
    so a developer who happens to have it exported would flip every expectation below and
    read as a regression in the resolver.
    """
    environment = os.environ.copy()
    environment.pop("UNSLOTH_ALLOW_ARM64_PYTHON", None)
    # Install-X64Python takes a different branch inside a conda env.
    environment.pop("CONDA_PREFIX", None)
    environment.pop("CONDA_DEFAULT_ENV", None)
    return environment


def _pwsh(script: str) -> str:
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        env = _environment_without_the_arm64_opt_out(),
    )
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("installed", "can_download", "expected"),
    [
        ([("3.13", "arm64"), ("3.13", "x86_64")], False, "3.13|x86_64"),
        ([("3.13", "arm64")], True, "3.13|x86_64"),
        # Native 3.13 ARM64 cannot resolve pyarrow or hf-transfer, so the lower x64 minor wins.
        ([("3.13", "arm64"), ("3.11", "x86_64")], False, "3.11|x86_64"),
        ([("3.13", "arm64"), ("3.11", "arm64")], False, "3.13|arm64"),
    ],
)
def test_arm64_host_prefers_an_x64_interpreter(installed, can_download, expected):
    assert _pwsh(_resolver_script(installed, can_download)) == expected


def _venv_base_script(cfg_lines: str, base_arch: str, base_present: bool) -> str:
    """The real Get-StudioVenvBasePython over a fake pyvenv.cfg and base interpreter."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    reader = _extract(r"    function Get-StudioVenvBasePython \{.*?\n    \}\n", source)
    tag = {"arm64": "win-arm64", "x86_64": "win-amd64"}.get(base_arch, base_arch)
    present = "$true" if base_present else "$false"
    return f"""
$ErrorActionPreference = "Stop"
$PythonSkip = @()
$CfgLines = @'
{cfg_lines}
'@ -split "`n"
function Test-Path {{
    param([Parameter(Position = 0)][string]$LiteralPath,
          [Parameter(ValueFromRemainingArguments = $true)]$Rest)
    if ($LiteralPath -like "*pyvenv.cfg") {{ return $true }}
    return {present}
}}
function Get-Content {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    return $CfgLines
}}
function Test-IsCondaPython {{ param([string]$Exe) return $false }}
function Get-PythonPlatformTag {{ param([string]$Exe) return "{tag}" }}
function BasePython.exe {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    return "Python 3.12.4"
}}
# The home-only spelling resolves to <home>/python.exe, so that exact name answers too.
function /py312/python.exe {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    return "Python 3.12.4"
}}
{reader}
$found = Get-StudioVenvBasePython -VenvDir "/home/unsloth_studio"
if ($found) {{ Write-Output "$($found.Version)|$($found.Arch)" }} else {{ Write-Output "none" }}
"""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("cfg", "arch", "present", "expected"),
    [
        ("home = /py312\nbase-executable = BasePython.exe\n", "arm64", True, "3.12|arm64"),
        ("home = /py312\n", "arm64", True, "3.12|arm64"),
        ("base-executable = BasePython.exe\n", "arm64", False, "none"),
        ("base-executable = BasePython.exe\n", "x86_64", True, "none"),
        ("include-system-site-packages = false\n", "arm64", True, "none"),
    ],
)
def test_the_existing_environments_base_interpreter_is_the_last_arm64_python(
    cfg, arch, present, expected
):
    """UNSLOTH_ALLOW_ARM64_PYTHON with no ARM64 interpreter registered on the machine.

    The selection prefers ARM64 only among the interpreters discovery can see. Remove the
    ARM64 CPython that built the venv and there is nothing left to prefer: an x64 interpreter
    is selected, the reinstall has already moved the native environment aside, and the new one
    is built on x64 -- while the opt-out's own message said it would not be. pyvenv.cfg still
    names that base interpreter, so when it is on disk the variable has something to honour.
    """
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            _venv_base_script(cfg, arch, present),
        ],
        check = True,
        capture_output = True,
        text = True,
        env = _environment_without_the_arm64_opt_out(),
    )
    assert result.stdout.strip() == expected
