# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Focused contracts for the Windows Studio install/runtime concurrency guard."""

from __future__ import annotations

import hashlib
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
PREFLIGHT_MANAGED_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "preflight" / "managed.rs"
DESKTOP_AUTH_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "desktop_auth.rs"
UPDATE_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "update.rs"
STUDIO_COMMAND = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"
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


def _mutex_helpers(source: str) -> str:
    return "\n".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            "Enter-StudioNamedMutex",
            "Get-StudioFinalPath",
            "Get-StudioPathHash",
            "Get-StudioInstallMutexName",
            "Test-StudioPathEqual",
            "Get-StudioRuntimeMutexNameForSid",
            "Get-StudioRuntimeMutexName",
            "Enter-StudioInstallMutex",
            "Exit-StudioInstallMutex",
        )
    )


def _process_helpers(source: str) -> str:
    return "\n".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            "Test-StudioCommandLinePathReference",
            "Get-RunningStudioVenvProcesses",
        )
    )


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_running_venv_process_is_reported(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    detector = _process_helpers(source)
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    probe = scripts / "guard-probe.exe"
    shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", probe)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    child = subprocess.Popen(
        [str(probe), "-n", "6", "127.0.0.1"],
        creationflags = creationflags,
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
        child.wait(timeout = 10)


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_command_line_only_venv_consumer_is_reported(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    detector = _process_helpers(source)
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


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_command_line_sibling_venv_is_not_reported(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    detector = _process_helpers(source)
    venv = tmp_path / "unsloth_studio"
    sibling = tmp_path / "unsloth_studio_backup" / "Lib" / "worker.py"
    venv.mkdir()
    script = f"""
$ErrorActionPreference = "Stop"
function Get-Process {{ param([Parameter(ValueFromRemainingArguments = $true)]$Rest) return @() }}
function Get-CimInstance {{
    param([Parameter(ValueFromRemainingArguments = $true)]$Rest)
    [pscustomobject]@{{
        ProcessId = 9191
        Name = "python.exe"
        ExecutablePath = $env:TEST_BASE_PYTHON
        CommandLine = ('"' + $env:TEST_BASE_PYTHON + '" "' + $env:TEST_SIBLING + '"')
    }}
}}
{detector}
@(Get-RunningStudioVenvProcesses -VenvPath $env:TEST_VENV) |
    ForEach-Object {{ Write-Output $_.Id }}
"""
    env = os.environ.copy()
    env["TEST_VENV"] = str(venv)
    env["TEST_SIBLING"] = str(sibling)
    env["TEST_BASE_PYTHON"] = shutil.which("python") or "python.exe"
    assert _run_powershell(shell, script, env) == ""


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_mutex_names_are_global_and_install_lock_is_path_scoped(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    studio_home = tmp_path / "studio"
    studio_home.mkdir()
    script = f"""
$ErrorActionPreference = "Stop"
{_mutex_helpers(source)}
Write-Output (Get-StudioInstallMutexName -Path $env:TEST_STUDIO_HOME)
Write-Output (Get-StudioRuntimeMutexNameForSid -Sid "S-1-5-21-111-222-333-1001")
Write-Output (Get-StudioRuntimeMutexNameForSid -Sid "S-1-5-21-111-222-333-1002")
$currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
Write-Output ((Get-StudioRuntimeMutexName) -eq `
    (Get-StudioRuntimeMutexNameForSid -Sid $currentSid))
"""
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(studio_home)
    canonical = str(studio_home.resolve()).rstrip("\\/").upper()
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    assert _run_powershell(shell, script, env).splitlines() == [
        f"Global\\UnslothStudioInstall-{digest}",
        "Global\\UnslothStudioManagedEnvironment-S-1-5-21-111-222-333-1001",
        "Global\\UnslothStudioManagedEnvironment-S-1-5-21-111-222-333-1002",
        "True",
    ]


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_tauri_managed_root_path_classification(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    profile = tmp_path / "profile"
    studio_home = profile / ".unsloth" / "studio"
    studio_home.mkdir(parents = True)
    alias = studio_home / ".." / "studio"
    junction = tmp_path / "profile-alias"
    junction_result = subprocess.run(
        [os.environ["COMSPEC"], "/d", "/c", "mklink", "/J", str(junction), str(profile)],
        capture_output = True,
        text = True,
    )
    if junction_result.returncode != 0:
        pytest.skip(f"Could not create a directory junction: {junction_result.stderr}")
    junction_studio = junction / ".unsloth" / "studio"
    script = f"""
$ErrorActionPreference = "Stop"
{_mutex_helpers(source)}
Write-Output (Test-StudioPathEqual -Left $env:TEST_STUDIO_HOME -Right $env:TEST_ALIAS)
Write-Output (Test-StudioPathEqual -Left $env:TEST_STUDIO_HOME -Right $env:TEST_CASE_VARIANT)
Write-Output (Test-StudioPathEqual -Left $env:TEST_STUDIO_HOME -Right $env:TEST_JUNCTION)
Write-Output (Test-StudioPathEqual -Left $env:TEST_STUDIO_HOME -Right $env:TEST_SIBLING)
"""
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(studio_home)
    env["TEST_ALIAS"] = str(alias)
    env["TEST_CASE_VARIANT"] = str(studio_home).upper()
    env["TEST_JUNCTION"] = str(junction_studio)
    env["TEST_SIBLING"] = str(profile / ".unsloth" / "studio-backup")
    assert _run_powershell(shell, script, env).splitlines() == ["True", "True", "True", "False"]


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_path_identity_failure_is_reported_as_unknown(shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    script = f"""
$ErrorActionPreference = "Stop"
{_mutex_helpers(source)}
function Get-StudioFinalPath {{ throw "identity unavailable" }}
$match = Test-StudioPathEqual -Left "C:\\one" -Right "C:\\two"
Write-Output ($null -eq $match)
"""
    assert _run_powershell(shell, script, os.environ.copy()).splitlines()[-1] == "True"


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_install_mutex_blocks_before_target_mutation_and_recovers_abandonment(
    tmp_path: Path, shell: str
):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    mutex_helpers = _mutex_helpers(source)
    studio_home = tmp_path / "studio"
    target = studio_home / "unsloth_studio"
    target.mkdir(parents = True)
    marker = target / "healthy.marker"
    marker.write_text("old", encoding = "utf-8")
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
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        env = env,
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
        assert marker.read_text(encoding = "utf-8") == "old"
        assert not Path(f"{target}.rollback").exists()

        holder.kill()
        holder.wait(timeout = 10)
        assert _run_powershell(shell, contender_script, env) == "MUTATED"
        assert not target.exists()
        assert (Path(f"{target}.rollback") / marker.name).read_text(encoding = "utf-8") == "old"
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout = 10)


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_runtime_gate_blocks_a_late_backend_start(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    helper = _extract(r"    function Enter-StudioNamedMutex \{.*?\n    \}\n", source)
    release = _extract(r"    function Exit-StudioInstallMutex \{.*?\n    \}\n", source)
    mutex_name = f"Global\\UnslothStudioRuntimeGateTest-{os.getpid()}-{tmp_path.name}"
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
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        env = env,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "READY"
        assert _run_powershell(shell, contender_script, env) == "BLOCKED"
        holder.kill()
        holder.wait(timeout = 10)
        assert _run_powershell(shell, contender_script, env) == "ACQUIRED"
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout = 10)


def test_guard_and_mutex_precede_rollback_and_release_after_restore():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    acquire = source.index("$studioInstallMutex = Enter-StudioInstallMutex -Path $StudioHome")
    root_match = source.index("$studioTauriRootMatch =", acquire)
    managed_root = source.index("$studioUsesTauriManagedRoot =", root_match)
    runtime_lock_needed = source.index("$studioNeedsRuntimeLock =", managed_root)
    legacy_layout = source.index("$studioUsesLegacyLayout =", runtime_lock_needed)
    runtime_name = source.index("Get-StudioRuntimeMutexName", legacy_layout)
    legacy_scan = source.index("if ($studioUsesLegacyLayout)", runtime_name)
    runtime_lock = source.index(
        "Enter-StudioNamedMutex -Name $studioRuntimeMutexName",
        runtime_name,
    )
    scan_candidates = source.index("$venvPathsToScan = @($VenvDir)", runtime_lock)
    legacy_source = source.index('Join-Path $StudioHome ".venv"', scan_candidates)
    cwd_source = source.index('Join-Path $env:USERPROFILE "unsloth_studio"', legacy_source)
    runtime_guard = source.index("foreach ($candidateVenv", cwd_source)
    desktop_guard = source.index('Get-Process -Name "unsloth-studio"', runtime_guard)
    rollback = source.index("Start-StudioVenvRollback -ExistingDir $VenvDir", desktop_guard)
    old_venv_move = source.index("Move-Item -LiteralPath $OldVenv", rollback)
    cwd_venv_move = source.index("Move-Item -LiteralPath $CwdVenv", old_venv_move)
    restore = source.rindex("Restore-StudioVenvRollback")
    prompt = source.index("Start Unsloth Studio now?", restore)
    autostart = source.index("Start-Process -FilePath $UnslothExe", prompt)
    release_runtime = source.rindex("Exit-StudioInstallMutex -Mutex $studioRuntimeMutex")
    release_install = source.rindex("Exit-StudioInstallMutex -Mutex $studioInstallMutex")
    wait_for_exit = source.rindex("$studioAutoStartProcess.WaitForExit()")

    assert (
        acquire
        < root_match
        < managed_root
        < runtime_lock_needed
        < legacy_layout
        < runtime_name
        < runtime_lock
    )
    assert runtime_lock < scan_candidates < legacy_scan < legacy_source < cwd_source
    assert cwd_source < runtime_guard < desktop_guard
    assert source.count("$studioUsesLegacyLayout `") >= 2
    assert "if ($studioNeedsRuntimeLock)" in source
    assert (
        "$studioUsesLegacyLayout = ($StudioRedirectMode -ne 'env') -or $studioUsesTauriManagedRoot"
    ) in source
    assert "-not $TauriMode -and $studioUsesLegacyLayout" in source
    assert runtime_guard < rollback < old_venv_move < cwd_venv_move
    assert (
        rollback < restore < prompt < autostart < release_runtime < release_install < wait_for_exit
    )
    assert "if ($StudioRedirectMode -eq 'legacy')" not in source
    assert "& $UnslothExe studio -p 8888" not in source
    assert "--clear" not in source[source.index("uv venv $VenvDir") :][:200]


def test_tauri_runtime_uses_the_same_gate_before_backend_spawn():
    install_source = INSTALL_PS1.read_text(encoding = "utf-8")
    process_source = PROCESS_RS.read_text(encoding = "utf-8")

    assert '"Global\\UnslothStudioManagedEnvironment-$Sid"' in install_source
    assert '"Global\\\\UnslothStudioManagedEnvironment-"' in process_source
    assert "Get-StudioRuntimeMutexName" in install_source
    assert "Get-StudioRuntimeMutexNameForSid" in install_source
    assert "studio_runtime_mutex_name_for_sid" in process_source
    assert "current_windows_user_sid()?" in process_source
    assert "studio_runtime_mutex_name_for_path" not in process_source
    start = process_source.index("pub fn start_backend(")
    guard = process_source.index("acquire_studio_runtime_launch_guard()?", start)
    resolve = process_source.index("resolve_backend_binary()", guard)
    handoff = process_source.index("STUDIO_RUNTIME_GATE_HANDOFF_ENV", resolve)
    spawn = process_source.index("cmd.spawn()", handoff)
    store = process_source.index("proc.owned = Some(", spawn)
    assert guard < resolve < handoff < spawn < store


def test_every_tauri_managed_child_spawn_uses_the_runtime_gate():
    process_source = PROCESS_RS.read_text(encoding = "utf-8")
    commands_source = COMMANDS_RS.read_text(encoding = "utf-8")
    preflight_source = PREFLIGHT_MANAGED_RS.read_text(encoding = "utf-8")
    desktop_auth_source = DESKTOP_AUTH_RS.read_text(encoding = "utf-8")
    update_source = UPDATE_RS.read_text(encoding = "utf-8")

    assert "pub(crate) fn with_studio_runtime_launch_guard<T>" in process_source

    install_check = commands_source.index("pub async fn check_install_status()")
    install_guard = commands_source.index("with_studio_runtime_launch_guard", install_check)
    install_spawn = commands_source.index("cmd.spawn()", install_guard)
    assert install_guard < install_spawn

    first_probe = preflight_source.index("async fn run_cli_probe(")
    first_guard = preflight_source.index("with_studio_runtime_launch_guard", first_probe)
    first_spawn = preflight_source.index("cmd.spawn()", first_guard)
    capability_probe = preflight_source.index("async fn probe_cli_capability(", first_spawn)
    capability_guard = preflight_source.index("with_studio_runtime_launch_guard", capability_probe)
    capability_spawn = preflight_source.index("cmd.spawn()", capability_guard)
    assert first_guard < first_spawn < capability_probe < capability_guard < capability_spawn

    provision = desktop_auth_source.index("async fn provision_desktop_auth()")
    provision_guard = desktop_auth_source.index("with_studio_runtime_launch_guard", provision)
    provision_spawn = desktop_auth_source.index("cmd.spawn()", provision_guard)
    provision_wait = desktop_auth_source.index("child.wait_with_output()", provision_spawn)
    assert provision_guard < provision_spawn < provision_wait

    update_call = update_source.index(
        "let result = crate::process::with_studio_runtime_launch_guard"
    )
    update_scan = update_source.index(
        "ensure_managed_environment_is_idle(&bin)",
        update_call,
    )
    update_spawn = update_source.index("spawn_update(&bin, &state)", update_scan)
    update_wait = update_source.index("wait_for_exit(&state)", update_spawn)
    update_guard_release = update_source.index("\n    });", update_wait)
    assert update_call < update_scan < update_spawn < update_wait < update_guard_release


def test_runtime_gate_handoff_covers_tauri_backend_and_installer_autostart():
    process_source = PROCESS_RS.read_text(encoding = "utf-8")
    install_source = INSTALL_PS1.read_text(encoding = "utf-8")
    studio_source = STUDIO_COMMAND.read_text(encoding = "utf-8")

    start = process_source.index("pub fn start_backend(")
    handoff = process_source.index('cmd.env(STUDIO_RUNTIME_GATE_HANDOFF_ENV, "1")', start)
    spawn = process_source.index("cmd.spawn()", handoff)
    assert handoff < spawn

    prompt = install_source.index("Start Unsloth Studio now?")
    save = install_source.index("$_runtimeGateHandoff =", prompt)
    set_handoff = install_source.index(
        '$env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"',
        save,
    )
    autostart = install_source.index("Start-Process -FilePath $UnslothExe", set_handoff)
    restore = install_source.index(
        "$env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = $_runtimeGateHandoff",
        autostart,
    )
    assert save < set_handoff < autostart < restore

    assert (
        studio_source.count(
            "runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()"
        )
        == 2
    )
    assert studio_source.count("inherited = runtime_gate_handoff") >= 4


def test_tauri_start_install_rejects_backend_conflicts_before_spawn():
    source = COMMANDS_RS.read_text(encoding = "utf-8")
    start = source.index("pub async fn start_install(")
    end = source.index("\n}\n", start)
    body = source[start:end]

    owned_guard = body.index("has_owned_backend(&backend_state)?")
    external_guard = body.index("block_external_conflict(&[]).await?")
    spawn = body.index("install::run_install")
    assert owned_guard < external_guard < spawn
