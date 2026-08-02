# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused contracts for the Windows Studio install/runtime concurrency guard."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
COMMANDS_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "commands.rs"
PROCESS_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "process.rs"
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


def _mutex_helpers(source: str) -> str:
    return "\n".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            "Enter-StudioNamedMutex",
            "Get-StudioInstallMutexName",
            "Enter-StudioInstallMutex",
            "Exit-StudioInstallMutex",
        )
    )


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason="Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_running_venv_process_is_reported(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding="utf-8")
    detector = _extract(r"    function Get-RunningStudioVenvProcesses \{.*?\n    \}\n", source)
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents=True)
    probe = scripts / "guard-probe.exe"
    shutil.copy2(os.environ["COMSPEC"], probe)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    child = subprocess.Popen(
        [str(probe), "/d", "/c", "ping", "-n", "6", "127.0.0.1"],
        creationflags=creationflags,
    )
    try:
        script = f"""
$ErrorActionPreference = "Stop"
{detector}
@(Get-RunningStudioVenvProcesses -VenvPath $env:TEST_VENV) |
    ForEach-Object {{ Write-Output $_.Id }}
"""
        env = os.environ.copy()
        env["TEST_VENV"] = str(scripts.parent)
        observed = _run_powershell(shell, script, env).splitlines()
        assert str(child.pid) in observed
    finally:
        child.terminate()
        child.wait(timeout=10)


@pytest.mark.skipif(not POWERSHELLS, reason="PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_command_line_only_venv_consumer_is_reported(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding="utf-8")
    detector = _extract(r"    function Get-RunningStudioVenvProcesses \{.*?\n    \}\n", source)
    venv = tmp_path / "unsloth_studio"
    venv.mkdir()
    script = f"""
$ErrorActionPreference = "Stop"
function Get-Process {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) return @() }}
function Get-CimInstance {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    [pscustomobject]@{{
        ProcessId = 4242
        Name = "python.exe"
        ExecutablePath = $env:TEST_BASE_PYTHON
        CommandLine = ('"' + $env:TEST_BASE_PYTHON + '" "' + $env:TEST_VENV + '\\Lib\\site-packages\\worker.py"')
    }}
}}
{detector}
@(Get-RunningStudioVenvProcesses -VenvPath $env:TEST_VENV) |
    ForEach-Object {{ Write-Output $_.Id }}
"""
    env = os.environ.copy()
    env["TEST_VENV"] = str(venv)
    env["TEST_BASE_PYTHON"] = shutil.which("python") or "python.exe"
    assert _run_powershell(shell, script, env) == "4242"


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason="Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_install_mutex_blocks_before_target_mutation_and_recovers_abandonment(
    tmp_path: Path, shell: str
):
    source = INSTALL_PS1.read_text(encoding="utf-8")
    mutex_helpers = _mutex_helpers(source)
    studio_home = tmp_path / "studio"
    target = studio_home / "unsloth_studio"
    target.mkdir(parents=True)
    marker = target / "healthy.marker"
    marker.write_text("old", encoding="utf-8")
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(studio_home)
    env["TEST_TARGET"] = str(target)

    holder_script = f"""
$ErrorActionPreference = "Stop"
{mutex_helpers}
$mutex = Enter-StudioInstallMutex -Path $env:TEST_STUDIO_HOME
if ($null -eq $mutex) {{ throw "holder did not acquire mutex" }}
Write-Output "READY"
[Console]::ReadLine() | Out-Null
Exit-StudioInstallMutex -Mutex $mutex
"""
    holder = subprocess.Popen(
        [shell, "-NoProfile", "-NonInteractive", "-Command", holder_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "READY"
        contender_script = f"""
$ErrorActionPreference = "Stop"
{mutex_helpers}
$mutex = Enter-StudioInstallMutex -Path $env:TEST_STUDIO_HOME
if ($null -eq $mutex) {{ Write-Output "BLOCKED"; exit 0 }}
Move-Item -LiteralPath $env:TEST_TARGET -Destination ($env:TEST_TARGET + ".rollback") -ErrorAction Stop
Write-Output "MUTATED"
Exit-StudioInstallMutex -Mutex $mutex
"""
        assert _run_powershell(shell, contender_script, env) == "BLOCKED"
        assert marker.read_text(encoding="utf-8") == "old"
        assert not Path(f"{target}.rollback").exists()

        holder.kill()
        holder.wait(timeout=10)
        assert _run_powershell(shell, contender_script, env) == "MUTATED"
        assert not target.exists()
        assert (Path(f"{target}.rollback") / marker.name).read_text(encoding="utf-8") == "old"
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=10)


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason="Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_runtime_gate_blocks_a_late_backend_start(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding="utf-8")
    helper = _extract(r"    function Enter-StudioNamedMutex \{.*?\n    \}\n", source)
    release = _extract(r"    function Exit-StudioInstallMutex \{.*?\n    \}\n", source)
    mutex_name = f"Local\\UnslothStudioRuntimeGateTest-{os.getpid()}-{tmp_path.name}"
    env = os.environ.copy()
    env["TEST_RUNTIME_MUTEX"] = mutex_name

    holder_script = f"""
$ErrorActionPreference = "Stop"
{helper}
{release}
$mutex = Enter-StudioNamedMutex -Name $env:TEST_RUNTIME_MUTEX
if ($null -eq $mutex) {{ throw "holder did not acquire mutex" }}
Write-Output "READY"
[Console]::ReadLine() | Out-Null
Exit-StudioInstallMutex -Mutex $mutex
"""
    contender_script = f"""
$ErrorActionPreference = "Stop"
{helper}
{release}
$mutex = Enter-StudioNamedMutex -Name $env:TEST_RUNTIME_MUTEX
if ($null -eq $mutex) {{ Write-Output "BLOCKED"; exit 0 }}
Write-Output "ACQUIRED"
Exit-StudioInstallMutex -Mutex $mutex
"""
    holder = subprocess.Popen(
        [shell, "-NoProfile", "-NonInteractive", "-Command", holder_script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "READY"
        assert _run_powershell(shell, contender_script, env) == "BLOCKED"
        holder.kill()
        holder.wait(timeout=10)
        assert _run_powershell(shell, contender_script, env) == "ACQUIRED"
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=10)


def test_guard_and_mutex_precede_rollback_and_release_after_restore():
    source = INSTALL_PS1.read_text(encoding="utf-8")
    acquire = source.index("$studioInstallMutex = Enter-StudioInstallMutex -Path $StudioHome")
    runtime_lock = source.index(
        "Enter-StudioNamedMutex -Name $script:StudioManagedRuntimeMutexName",
        acquire,
    )
    runtime_guard = source.index(
        "Get-RunningStudioVenvProcesses -VenvPath $VenvDir",
        runtime_lock,
    )
    desktop_guard = source.index('Get-Process -Name "unsloth-studio"', runtime_guard)
    rollback = source.index("Start-StudioVenvRollback -ExistingDir $VenvDir", desktop_guard)
    restore = source.rindex("Restore-StudioVenvRollback")
    release_runtime = source.rindex("Exit-StudioInstallMutex -Mutex $studioRuntimeMutex")
    release_install = source.rindex("Exit-StudioInstallMutex -Mutex $studioInstallMutex")

    assert acquire < runtime_lock < runtime_guard < desktop_guard < rollback
    assert rollback < restore < release_runtime < release_install
    assert "--clear" not in source[source.index("uv venv $VenvDir") :][:200]


def test_tauri_runtime_uses_the_same_gate_before_backend_spawn():
    install_source = INSTALL_PS1.read_text(encoding="utf-8")
    process_source = PROCESS_RS.read_text(encoding="utf-8")

    assert '"Local\\UnslothStudioManagedEnvironment"' in install_source
    assert '"Local\\\\UnslothStudioManagedEnvironment"' in process_source
    start = process_source.index("pub fn start_backend(")
    guard = process_source.index("acquire_studio_runtime_launch_guard()?", start)
    resolve = process_source.index("resolve_backend_binary()", guard)
    spawn = process_source.index("cmd.spawn()", resolve)
    store = process_source.index("proc.owned = Some(", spawn)
    assert guard < resolve < spawn < store


def test_tauri_start_install_rejects_backend_conflicts_before_spawn():
    source = COMMANDS_RS.read_text(encoding="utf-8")
    start = source.index("pub async fn start_install(")
    end = source.index("\n}\n", start)
    body = source[start:end]

    owned_guard = body.index("has_owned_backend(&backend_state)?")
    external_guard = body.index("block_external_conflict(&[]).await?")
    spawn = body.index("install::run_install")
    assert owned_guard < external_guard < spawn
