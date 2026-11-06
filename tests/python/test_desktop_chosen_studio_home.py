# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""install.ps1 under --tauri: the desktop's chosen install folder is accepted, an inherited
UNSLOTH_STUDIO_HOME / STUDIO_HOME is still refused, and the Rust side passes the folder."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
TAURI_SRC = REPO_ROOT / "studio" / "src-tauri" / "src"
PWSH = shutil.which("pwsh") or shutil.which("powershell")


def _validation_block() -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    start = source.index("    # Custom Unsloth roots are not supported with --tauri")
    end = source.index("    # LOCALAPPDATA may be unset", start)
    return source[start:end]


def _run_validation(tmp_path: Path, override_var: str, chosen_flag: str | None) -> str:
    # Path helpers are stubbed to "different directory", so only the chosen-root gate decides.
    script = f"""
$ErrorActionPreference = "Stop"
function Write-StudioLine {{ param([string]$Text, $ForegroundColor) }}
function Restore-StudioTempEnvironment {{ }}
function Get-StudioFinalPath {{ param([string]$Path) return $Path }}
function Test-StudioSameDirectoryByProbe {{ param($Left, $Right) return $false }}
function Split-StudioExistingAncestor {{ param([string]$Path) return $null }}
$TauriMode = $true
$envOverride = $env:TEST_CHOSEN_ROOT
$envOverrideVar = "{override_var}"
$tauriProfile = $env:TEST_TAURI_PROFILE
try {{
{_validation_block()}
    Write-Output "accepted"
}} catch {{
    Write-Output "refused: $($_.Exception.Message)"
}}
"""
    env = os.environ.copy()
    env.pop("UNSLOTH_DESKTOP_CHOSEN_STUDIO_HOME", None)
    env["TEST_CHOSEN_ROOT"] = str(tmp_path / "Apps" / "Unsloth" / "studio")
    env["TEST_TAURI_PROFILE"] = str(tmp_path / "profile")
    if chosen_flag is not None:
        env["UNSLOTH_DESKTOP_CHOSEN_STUDIO_HOME"] = chosen_flag
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
            verdict = "accepted",
        )
    finally:
        os.unlink(name)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is required")
def test_desktop_chosen_root_is_accepted_under_tauri(tmp_path: Path):
    assert _run_validation(tmp_path, "UNSLOTH_STUDIO_HOME", "1") == "accepted"


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is required")
@pytest.mark.parametrize(
    "override_var, chosen_flag",
    [
        ("UNSLOTH_STUDIO_HOME", None),
        ("UNSLOTH_STUDIO_HOME", "0"),
        ("STUDIO_HOME", "1"),
    ],
)
def test_inherited_custom_root_is_still_refused_under_tauri(
    tmp_path: Path, override_var: str, chosen_flag: str | None
):
    outcome = _run_validation(tmp_path, override_var, chosen_flag)
    assert outcome == f"refused: {override_var} is not supported with --tauri."


def test_desktop_passes_the_chosen_root_only_to_the_installer_with_its_flag():
    install = (TAURI_SRC / "install.rs").read_text(encoding = "utf-8")
    scrub = install.index("for name in crate::process::MANAGED_CHILD_SCRUBBED_ENV")
    chosen = install.index("crate::studio_home::chosen_root()", scrub)
    block = install[chosen : install.index("}", install.index("UNSLOTH_ISOLATE_UV_CACHE", chosen))]
    assert "STUDIO_HOME_ENV" in block
    assert "DESKTOP_CHOSEN_HOME_ENV" in block
    studio_home = (TAURI_SRC / "studio_home.rs").read_text(encoding = "utf-8")
    assert '"UNSLOTH_DESKTOP_CHOSEN_STUDIO_HOME"' in studio_home
    assert "$env:UNSLOTH_DESKTOP_CHOSEN_STUDIO_HOME" in _validation_block()


def _custom_uninstall_roots(tmp_path: Path, pointer: str | None) -> list[str]:
    # Function definitions are lifted out of uninstall.ps1 by AST; its body stops processes and
    # edits the registry, so it is never run here.
    profile = tmp_path / "profile"
    (profile / ".unsloth").mkdir(parents = True)
    if pointer is not None:
        (profile / ".unsloth" / "desktop_studio_home").write_text(pointer, encoding = "utf-8")
    script = """
$ErrorActionPreference = "Stop"
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($env:TEST_UNINSTALL_PS1, [ref]$tokens, [ref]$errors)
foreach ($fn in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    . ([scriptblock]::Create($fn.Extent.Text))
}
foreach ($root in @(_CustomStudioRoots)) { Write-Output "ROOT=$root" }
Write-Output "done"
"""
    env = os.environ.copy()
    for name in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME", "UNSLOTH_HOME"):
        env.pop(name, None)
    env["USERPROFILE"] = str(profile)
    env["LOCALAPPDATA"] = str(tmp_path / "localappdata")
    env["TEST_UNINSTALL_PS1"] = str(REPO_ROOT / "scripts" / "uninstall.ps1")
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
            verdict = "done",
        )
    finally:
        os.unlink(name)
    assert result.returncode == 0 and "done" in result.stdout, result.stderr
    return [line[5:] for line in result.stdout.splitlines() if line.startswith("ROOT=")]


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is required")
def test_uninstaller_finds_the_folder_desktop_installed_into(tmp_path: Path):
    chosen = tmp_path / "D" / "Apps" / "Unsloth" / "studio"
    chosen.mkdir(parents = True)
    roots = _custom_uninstall_roots(tmp_path, f"{chosen}\n")
    assert [Path(root) for root in roots] == [chosen]


@pytest.mark.skipif(PWSH is None, reason = "PowerShell is required")
def test_uninstaller_without_a_desktop_choice_finds_no_custom_root(tmp_path: Path):
    assert _custom_uninstall_roots(tmp_path, None) == []
