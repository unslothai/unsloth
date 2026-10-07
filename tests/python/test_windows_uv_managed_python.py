# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install.ps1 lets uv provide Python instead of a system-wide install (#7802)."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import pytest
from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SOURCE = INSTALL_PS1.read_text(encoding = "utf-8")
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]
HOST_MINOR = "{}.{}".format(*sys.version_info[:2])
HOST_FULL = "{}.{}.{}".format(*sys.version_info[:3])
RANGE = ">={0}.{1},<{0}.{2},!={3}".format(
    sys.version_info[0], sys.version_info[1], sys.version_info[1] + 1, HOST_FULL
)

FAKE_UV = r"""
Add-Content -LiteralPath $env:FAKE_UV_LOG -Value ($args -join ' ')
if ($args[0] -eq 'python' -and $args[1] -eq 'install') { exit [int]$env:FAKE_UV_INSTALL_EXIT }
if ($args[0] -eq 'python' -and $args[1] -eq 'find') {
    if ($env:FAKE_UV_FIND) { Write-Output $env:FAKE_UV_FIND; exit 0 }
    exit 2
}
exit 9
"""


def _function(name: str) -> str:
    match = re.search(rf"    function {name} \{{.*?\n    \}}\n", SOURCE, flags = re.DOTALL)
    assert match is not None, name
    return match.group(0)


def _resolve(
    shell: str,
    tmp_path: Path,
    *,
    install_exit: int = 0,
    find: str = "",
    skip: list[str] | None = None,
    extra_env: dict[str, str] | None = None,
):
    (tmp_path / "uv.ps1").write_text(FAKE_UV, encoding = "utf-8")
    log = tmp_path / "uv.log"
    skip_list = ", ".join(f"'{v}'" for v in (skip or []))
    script = f"""
$ErrorActionPreference = "Stop"
function substep {{ param($m, $c) }}
function Invoke-InstallCommand {{
    param([ScriptBlock]$Command, [string]$Label, [switch]$NoMirror)
    $global:LASTEXITCODE = 0
    & $Command | Out-Null
    return [int]$LASTEXITCODE
}}
$PythonVersion = "{HOST_MINOR}"
$PythonSkip = @({skip_list})
$script:UvExe = $env:FAKE_UV_EXE
{_function("Resolve-UvManagedPython")}
$r = Resolve-UvManagedPython
if ($r) {{ $r | ConvertTo-Json -Compress }} else {{ 'null' }}
"""
    env = dict(
        os.environ,
        FAKE_UV_LOG = str(log),
        FAKE_UV_EXE = str(tmp_path / "uv.ps1"),
        FAKE_UV_INSTALL_EXIT = str(install_exit),
        FAKE_UV_FIND = find,
        **(extra_env or {}),
    )
    out = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        env = env,
        timeout = 60,
    ).stdout.strip()
    calls = log.read_text(encoding = "utf-8").splitlines() if log.exists() else []
    return json.loads(out.splitlines()[-1]), calls


needs_pwsh = pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
needs_supported_host = pytest.mark.skipif(
    not re.fullmatch(r"3\.1[1-3]", HOST_MINOR),
    reason = "the probe runs this interpreter, which must be 3.11-3.13",
)


@needs_pwsh
@needs_supported_host
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_uv_managed_python_is_installed_without_touching_path(shell, tmp_path):
    result, calls = _resolve(shell, tmp_path, find = sys.executable)
    assert result == {"Version": HOST_MINOR, "Path": sys.executable, "Arch": ""}
    assert calls == [
        f"python install --no-bin --no-registry {HOST_MINOR}",
        f"python find --system --managed-python {HOST_MINOR}",
    ]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_failed_uv_install_falls_back(shell, tmp_path):
    result, calls = _resolve(shell, tmp_path, install_exit = 1, find = sys.executable)
    assert result is None
    assert len(calls) == 1


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_nothing_found_falls_back(shell, tmp_path):
    result, _ = _resolve(shell, tmp_path, find = "")
    assert result is None


@needs_pwsh
@needs_supported_host
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_skipped_patch_is_excluded_from_the_request(shell, tmp_path):
    result, calls = _resolve(shell, tmp_path, find = sys.executable, skip = [HOST_FULL])
    assert result is None
    assert calls[0] == f"python install --no-bin --no-registry {RANGE}"
    assert calls[1] == f"python find --system --managed-python {RANGE}"


@needs_pwsh
@needs_supported_host
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_startup_banner_does_not_hide_a_skipped_patch(shell, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text("print('banner')\n", encoding = "utf-8")
    result, _ = _resolve(
        shell,
        tmp_path,
        find = sys.executable,
        skip = [HOST_FULL],
        extra_env = {"PYTHONPATH": str(site)},
    )
    assert result is None


def test_non_arm64_defers_to_uv_and_keeps_the_system_install_as_fallback():
    detect = SOURCE.index("$DetectedPython = Remove-SkippedPython (Find-CompatiblePython)")
    defer = SOURCE.index(
        '$PythonFromUv = (-not $DetectedPython) -and ((Get-HostMachineArch) -ne "arm64")'
    )
    system_block = SOURCE.index("$InstallSystemPython = {")
    immediate = SOURCE.index(
        "if (-not $DetectedPython -and -not $PythonFromUv) {\n        . $InstallSystemPython"
    )
    uv_ready = SOURCE.index("Set-StudioUvCacheEnvironment -StudioRoot $StudioHome")
    resolve = SOURCE.index("$DetectedPython = Resolve-UvManagedPython")
    fallback = SOURCE.index(". $InstallSystemPython", resolve)
    venv = SOURCE.index("& $script:UvExe venv $VenvDir --python")
    assert detect < defer < system_block < immediate < uv_ready < resolve < fallback < venv
    assert SOURCE.count(". $InstallSystemPython") == 2
    winget = SOURCE.index("& $script:WingetExe install -e --id $pythonPackageId")
    assert system_block < winget < immediate
