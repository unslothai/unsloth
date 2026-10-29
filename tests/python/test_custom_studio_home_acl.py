# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install.ps1: a new custom UNSLOTH_STUDIO_HOME gets the private ACL (current user, SYSTEM,
Administrators), not only the desktop-chosen one, and an existing folder keeps its ACL."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
PWSH = shutil.which("pwsh") or shutil.which("powershell")


def _env_root_block() -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    start = source.index("    if ($envOverride) {\n        try {\n            # BEFORE the create")
    end = source.index('        $StudioDataDir = Join-Path $StudioHome "share"', start)
    return source[start:end] + "    }\n"


def _run(root: Path, *, tauri_chosen: bool) -> str:
    # Reports whether the root ended up protected (Windows) or whether the ACL write was tried
    # (elsewhere .NET has no Windows ACLs, so the attempt lands in the warning).
    script = f"""
$ErrorActionPreference = "Stop"
$script:lines = @()
function Write-StudioLine {{ param([string]$Text, $ForegroundColor) $script:lines += $Text }}
function Restore-StudioTempEnvironment {{ }}
$envOverride = $env:TEST_ROOT
$envOverrideVar = "UNSLOTH_STUDIO_HOME"
$tauriChosenRoot = ${"true" if tauri_chosen else "false"}
{_env_root_block()}
if ($IsWindows -or $env:OS -eq "Windows_NT") {{
    if ((Get-Acl -LiteralPath $StudioHome).AreAccessRulesProtected) {{ "restricted" }} else {{ "inherited" }}
}} elseif ($script:lines -match "Could not restrict access") {{ "restricted" }} else {{ "inherited" }}
"""
    env = os.environ.copy()
    env["TEST_ROOT"] = str(root)
    handle, name = tempfile.mkstemp(suffix = ".ps1")
    os.close(handle)
    try:
        Path(name).write_text(script, encoding = "utf-8-sig")
        result = run_pwsh(
            [PWSH, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", name],
            check = False,
            capture_output = True,
            text = True,
            env = env,
        )
    finally:
        os.unlink(name)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is required")
def test_new_custom_root_gets_the_private_acl(tmp_path: Path):
    assert _run(tmp_path / "custom" / "studio", tauri_chosen = False) == "restricted"


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is required")
def test_new_desktop_chosen_root_still_gets_the_private_acl(tmp_path: Path):
    assert _run(tmp_path / "chosen" / "studio", tauri_chosen = True) == "restricted"


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is required")
def test_existing_custom_root_keeps_its_acl(tmp_path: Path):
    root = tmp_path / "existing"
    root.mkdir()
    assert _run(root, tauri_chosen = False) == "inherited"
