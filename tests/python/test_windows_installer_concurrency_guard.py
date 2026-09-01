# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
COMMANDS_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "commands.rs"
PROCESS_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "process.rs"
PREFLIGHT_MANAGED_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "preflight" / "managed.rs"
DESKTOP_AUTH_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "desktop_auth.rs"
UPDATE_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "update.rs"
MAIN_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "main.rs"
STUDIO_COMMAND = REPO_ROOT / "unsloth_cli" / "commands" / "studio.py"
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _run_powershell(shell: str, script: str, env: dict[str, str]) -> str:
    # Through a FILE, not -Command: these scripts carry the whole extracted helper chain, and Windows caps a command
    # line at 32767 characters.
    # Passed inline, the moment the chain grows past that every test here dies as WinError 206 "The filename or
    # extension is too long" instead of testing anything.
    # utf-8-sig because Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI.
    handle, name = tempfile.mkstemp(suffix = ".ps1")
    os.close(handle)
    try:
        Path(name).write_text(script, encoding = "utf-8-sig")
        result = subprocess.run(
            [shell, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", name],
            check = True,
            capture_output = True,
            text = True,
            # cannot decode what PowerShell writes and the whole test then dies as
            # Decoded as utf-8 with replacement, not the console codepage:
            encoding = "utf-8",
            errors = "replace",
            env = env,
            timeout = 60,
        )
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass
    return result.stdout.strip()


def _ps_file(directory: Path, name: str, script: str) -> str:
    """Same reason as _run_powershell: a 32 KB command line is not available here."""
    path = directory / name
    path.write_text(script, encoding = "utf-8-sig")
    return str(path)


# The chain Get-StudioFinalPath dispatches to.
# rather than as a missing helper (issue #9140).
_FINAL_PATH_CHAIN = (
    "Write-StudioLine",
    "Test-StudioDirectoryUsable",
    "Remove-StudioStalePrivateTempDirectories",
    "Get-StudioPrivateTempRoots",
    "New-StudioPrivateTempDirectory",
    "Initialize-StudioTempEnvironment",
    "Write-StudioFinalPathDegraded",
    "Initialize-StudioFinalPathNativeType",
    "Resolve-StudioLinkTarget",
    "Get-StudioSubstTarget",
    "Get-StudioLexicalPath",
    "Resolve-StudioFinalPathInfo",
    "Get-StudioFinalPath",
)


def _final_path_helpers(source: str) -> str:
    return "\n".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source) for name in _FINAL_PATH_CHAIN
    )


def _mutex_helpers(source: str) -> str:
    return "\n".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            # Test-StudioPathEqual reports an unresolvable identity through this, and these scripts run under
            # -ErrorActionPreference Stop, so leaving it out made the CATCH path throw CommandNotFound and every test
            # that reaches it fail for a reason that has nothing to do with what it measures.
            "Write-StudioLine",
            "Enter-StudioNamedMutex",
            # Get-StudioFinalPath is a dispatcher now:
            # PowerShell resolver when the native helper did not compile (#9140).
            # Get-StudioFinalPath is a dispatcher now:
            "Test-StudioDirectoryUsable",
            "Remove-StudioStalePrivateTempDirectories",
            "Get-StudioPrivateTempRoots",
            "New-StudioPrivateTempDirectory",
            "Initialize-StudioTempEnvironment",
            "Write-StudioFinalPathDegraded",
            "Initialize-StudioFinalPathNativeType",
            "Resolve-StudioLinkTarget",
            "Get-StudioSubstTarget",
            "Get-StudioLexicalPath",
            "Resolve-StudioFinalPathInfo",
            "Get-StudioFinalPath",
            "Get-StudioPathHash",
            "Get-StudioInstallMutexName",
            "Test-StudioPathEqual",
            "Get-StudioRuntimeMutexNameForSid",
            "Get-StudioRuntimePathHash",
            "Get-StudioRuntimeMutexNameForPath",
            "Get-StudioCurrentUserSid",
            "Get-StudioRuntimeMutexName",
            "Get-StudioRuntimeMutexNames",
            "Enter-StudioInstallMutex",
            "Exit-StudioInstallMutex",
        )
    )


def _process_helpers(source: str) -> str:
    return "\n".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", source)
        for name in (
            "Write-StudioLine",
            "Test-StudioDirectoryUsable",
            "Remove-StudioStalePrivateTempDirectories",
            "Get-StudioPrivateTempRoots",
            "New-StudioPrivateTempDirectory",
            "Initialize-StudioTempEnvironment",
            "Write-StudioFinalPathDegraded",
            "Initialize-StudioFinalPathNativeType",
            "Resolve-StudioLinkTarget",
            "Get-StudioSubstTarget",
            "Get-StudioLexicalPath",
            "Resolve-StudioFinalPathInfo",
            "Get-StudioFinalPath",
            "Test-StudioProtectedPathMatch",
            "Get-StudioProcessImagePath",
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
    # Long enough that the child outlives the scan itself.
    # Windows PowerShell 5.1 pays a cold start plus a real csc.exe compile of the native helper before it can look at
    # anything, which alone can outlast a six-ping child;
    child = subprocess.Popen(
        [str(probe), "-n", "120", "127.0.0.1"],
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
        deadline = time.monotonic() + 60
        observed = []
        while time.monotonic() < deadline:
            observed = _run_powershell(shell, script, env).splitlines()
            if str(child.pid) in observed:
                break
            time.sleep(0.1)
        assert str(child.pid) in observed
    finally:
        child.terminate()
        child.wait(timeout = 10)


@pytest.mark.skipif(
    os.name != "nt" or sys.maxsize <= 2**32,
    reason = "A 64-bit Windows test host is required",
)
def test_x86_powershell_reports_64_bit_managed_process(tmp_path: Path):
    x86_shell = (
        Path(os.environ["SystemRoot"])
        / "SysWOW64"
        / "WindowsPowerShell"
        / "v1.0"
        / "powershell.exe"
    )
    if not x86_shell.is_file():
        pytest.skip("32-bit Windows PowerShell is unavailable")

    source = INSTALL_PS1.read_text(encoding = "utf-8")
    detector = _process_helpers(source)
    scripts = tmp_path / "unsloth_studio" / "Scripts"
    scripts.mkdir(parents = True)
    probe = scripts / "guard-probe.exe"
    shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", probe)
    # Long-lived: a 32-bit shell pays a WOW64 start plus an Add-Type compile, so a short probe can exit before the scan
    child = subprocess.Popen(
        [str(probe), "-n", "120", "127.0.0.1"],
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
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
        deadline = time.monotonic() + 30
        observed = []
        while time.monotonic() < deadline:
            observed = _run_powershell(str(x86_shell), script, env).splitlines()
            if str(child.pid) in observed:
                break
            time.sleep(0.5)
        assert str(child.pid) in observed
    finally:
        child.terminate()
        child.wait(timeout = 10)


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_installer_decision_stops_active_process_and_allows_idle(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    decision_start = source.index("        $protectedProcessPaths = @(")
    decision_end = source.index(
        "        if (-not $TauriMode -and $studioUsesLegacyLayout)", decision_start
    )
    decision = source[decision_start:decision_end]

    studio_home = tmp_path / "studio"
    venv = studio_home / "unsloth_studio"
    scripts = venv / "Scripts"
    scripts.mkdir(parents = True)
    marker = venv / "must-remain.txt"
    marker.write_text("untouched", encoding = "utf-8")
    worker = scripts / "worker.exe"
    shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", worker)
    child = subprocess.Popen(
        [str(worker), "-n", "30", "127.0.0.1"],
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    script = f"""
$ErrorActionPreference = "Stop"
{_process_helpers(source)}
# The block names the blocking processes through install.ps1's UTF-8 stdout sink
# before it hands off to Exit-InstallFailure. Unstubbed that is a command-not-found
# terminating error under "Stop", so the active case never reaches RESULT:blocked.
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) Write-Host $Message }}
function Exit-InstallFailure {{
    param([string]$Message)
    return "blocked"
}}
function Invoke-InstallerDecision {{
    $VenvDir = $env:TEST_VENV
    $StudioHome = $env:TEST_STUDIO_HOME
    $studioUsesLegacyLayout = $false
{decision}
    return "continued"
}}
$result = Invoke-InstallerDecision
Write-Output ("RESULT:" + $result)
Write-Output ("MARKER:" + (Get-Content -LiteralPath $env:TEST_MARKER -Raw))
"""
    env = os.environ.copy()
    env["TEST_VENV"] = str(venv)
    env["TEST_STUDIO_HOME"] = str(studio_home)
    env["TEST_MARKER"] = str(marker)
    try:
        active = [
            line
            for line in _run_powershell(shell, script, env).splitlines()
            if line.startswith(("RESULT:", "MARKER:"))
        ]
        assert active == ["RESULT:blocked", "MARKER:untouched"]
    finally:
        child.terminate()
        child.wait(timeout = 10)

    idle = [
        line
        for line in _run_powershell(shell, script, env).splitlines()
        if line.startswith(("RESULT:", "MARKER:"))
    ]
    assert idle == ["RESULT:continued", "MARKER:untouched"]


def test_installer_ignores_command_line_and_cwd_only_path_mentions():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    for removed_helper in (
        "ConvertFrom-StudioWindowsCommandLine",
        "Test-StudioRawCommandLinePathReference",
        "Get-StudioProcessWorkingDirectories",
        "Test-StudioCommandLinePathReference",
    ):
        assert removed_helper not in source

    detector = source[
        source.index("    function Get-RunningStudioVenvProcesses {") : source.index(
            "    function Test-VenvPythonReady {"
        )
    ]
    assert "Get-CimInstance" not in detector
    assert ".CommandLine" not in detector
    assert "$process.Path" not in detector
    assert "Get-StudioProcessImagePath -ProcessId $process.Id" in detector

    # The same contract holds on every rung:
    # confirmed image, never a command line. Its Win32_Process rung exists because a
    # processes and overwrite a venv Unsloth has open (issue #9140).
    # The same contract has to hold on every rung of that helper's fallback:
    image = _extract(r"    function Get-StudioProcessImagePath \{.*?\n    \}\n", source)
    assert ".CommandLine" not in image
    assert "ExecutablePath" in image


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_versioned_native_helper_loads_after_older_installer_type(shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    final_path_helper = _mutex_helpers(source)
    script = f"""
$ErrorActionPreference = "Stop"
Add-Type -TypeDefinition @'
public static class UnslothStudioFinalPath
{{
    public static string Resolve(string path) {{ return path; }}
}}
'@
{final_path_helper}
Get-StudioFinalPath -Path $env:SystemRoot | Out-Null
Write-Output ([bool]("UnslothStudioFinalPathV2" -as [type]))
Write-Output ([bool]([UnslothStudioFinalPathV2]::GetProcessImagePath($PID)))
"""
    assert _run_powershell(shell, script, os.environ.copy()).splitlines() == ["True", "True"]


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_desktop_process_filter_keeps_only_the_current_user_sid(shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    helper = _extract(
        r"    function Get-StudioDesktopProcessesForCurrentUser \{.*?\n    \}\n", source
    )
    script = f"""
$ErrorActionPreference = "Stop"
{helper}
function Get-StudioCurrentUserSid {{ return "S-1-5-21-current" }}
function Get-CimInstance {{
    [CmdletBinding()]
    param([string]$ClassName, [string]$Filter)
    @(
        [pscustomobject]@{{ Name = "unsloth-studio.exe"; ProcessId = 101 }}
        [pscustomobject]@{{ Name = "unsloth-studio.exe"; ProcessId = 202 }}
        [pscustomobject]@{{ Name = "unsloth-studio.exe"; ProcessId = 303 }}
    )
}}
function Invoke-CimMethod {{
    [CmdletBinding()]
    param($InputObject, [string]$MethodName)
    if ($InputObject.ProcessId -eq 101) {{
        return [pscustomobject]@{{ ReturnValue = 0; Sid = "S-1-5-21-current" }}
    }}
    if ($InputObject.ProcessId -eq 202) {{
        return [pscustomobject]@{{ ReturnValue = 0; Sid = "S-1-5-21-other" }}
    }}
    throw "owner unavailable"
}}
@(Get-StudioDesktopProcessesForCurrentUser) | ForEach-Object {{ Write-Output $_.Id }}
"""
    assert _run_powershell(shell, script, os.environ.copy()).splitlines() == ["101"]
    assert "GetOwnerSid" in helper
    assert "SessionId" not in helper

    assert "QueryFullProcessImageNameW" in source


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_junction_alias_process_is_reported_for_physical_venv(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    detector = _process_helpers(source)
    physical = tmp_path / "physical" / "unsloth_studio"
    scripts = physical / "Scripts"
    scripts.mkdir(parents = True)
    alias = tmp_path / "alias"
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(alias), str(physical)],
        check = True,
        capture_output = True,
        text = True,
    )
    probe = alias / "Scripts" / "guard-probe.exe"
    shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", probe)
    child = subprocess.Popen(
        [str(probe), "-n", "6", "127.0.0.1"],
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        script = f"""
$ErrorActionPreference = "Stop"
{detector}
@(Get-RunningStudioVenvProcesses -VenvPath $env:TEST_VENV) |
    ForEach-Object {{ Write-Output $_.Id }}
"""
        env = os.environ.copy()
        env["TEST_VENV"] = str(physical)
        assert str(child.pid) in _run_powershell(shell, script, env).splitlines()
    finally:
        child.terminate()
        child.wait(timeout = 10)


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_exact_studio_bin_shim_process_is_reported(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    detector = _process_helpers(source)
    shim = tmp_path / "studio" / "bin" / "unsloth.exe"
    shim.parent.mkdir(parents = True)
    shutil.copy2(Path(os.environ["SystemRoot"]) / "System32" / "PING.EXE", shim)
    child = subprocess.Popen(
        [str(shim), "-n", "6", "127.0.0.1"],
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        script = f"""
$ErrorActionPreference = "Stop"
{detector}
@(Get-RunningStudioVenvProcesses -VenvPath $env:TEST_SHIM -Exact) |
    ForEach-Object {{ Write-Output $_.Id }}
"""
        env = os.environ.copy()
        env["TEST_SHIM"] = str(shim)
        assert str(child.pid) in _run_powershell(shell, script, env).splitlines()
    finally:
        child.terminate()
        child.wait(timeout = 10)


def test_installer_scan_protects_the_exact_studio_bin_shim():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    assert 'Join-Path $StudioHome "bin\\unsloth.exe"' in source
    assert "[pscustomobject]@{ Path = (Join-Path $StudioHome" in source
    assert "Exact = $true" in source


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
def test_tauri_override_accepts_junction_alias_of_managed_root(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    profile = tmp_path / "profile"
    studio_home = profile / ".unsloth" / "studio"
    studio_home.mkdir(parents = True)
    junction = tmp_path / "profile-alias"
    junction_result = subprocess.run(
        [os.environ["COMSPEC"], "/d", "/c", "mklink", "/J", str(junction), str(profile)],
        capture_output = True,
        text = True,
    )
    if junction_result.returncode != 0:
        pytest.skip(f"Could not create a directory junction: {junction_result.stderr}")

    validation_start = source.index("    # Custom Unsloth roots are not supported with --tauri")
    validation_end = source.index("    # LOCALAPPDATA may be unset", validation_start)
    validation = source[validation_start:validation_end]
    final_path_helper = _final_path_helpers(source)
    script = f"""
$ErrorActionPreference = "Stop"
{final_path_helper}
$TauriMode = $true
$envOverride = $env:TEST_TAURI_OVERRIDE
$envOverrideVar = "UNSLOTH_STUDIO_HOME"
$tauriProfile = $env:TEST_TAURI_PROFILE
{validation}
Write-Output "accepted"
"""
    env = os.environ.copy()
    env["TEST_TAURI_OVERRIDE"] = str(junction / ".unsloth" / "studio")
    env["TEST_TAURI_PROFILE"] = str(profile)
    assert _run_powershell(shell, script, env) == "accepted"


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_missing_root_beneath_junction_uses_the_physical_mutex_identity(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    profile = tmp_path / "profile"
    profile.mkdir()
    junction = tmp_path / "profile-alias"
    junction_result = subprocess.run(
        [os.environ["COMSPEC"], "/d", "/c", "mklink", "/J", str(junction), str(profile)],
        capture_output = True,
        text = True,
    )
    if junction_result.returncode != 0:
        pytest.skip(f"Could not create a directory junction: {junction_result.stderr}")

    script = f"""
$ErrorActionPreference = "Stop"
{_mutex_helpers(source)}
$aliasRoot = $env:TEST_ALIAS_ROOT
$physicalRoot = $env:TEST_PHYSICAL_ROOT
$aliasMatch = Test-StudioPathEqual -Left $aliasRoot -Right $physicalRoot
$physicalMatch = Test-StudioPathEqual -Left $physicalRoot -Right $physicalRoot
$aliasRuntime = @(Get-StudioRuntimeMutexNames -TauriRootMatch $aliasMatch -Path $aliasRoot)
$physicalRuntime = @(Get-StudioRuntimeMutexNames -TauriRootMatch $physicalMatch -Path $physicalRoot)
Write-Output $aliasMatch
Write-Output ((Get-StudioInstallMutexName -Path $aliasRoot) -eq (Get-StudioInstallMutexName -Path $physicalRoot))
Write-Output $aliasRuntime.Count
Write-Output $physicalRuntime.Count
Write-Output ($aliasRuntime[0] -eq $physicalRuntime[0])
Write-Output ($aliasRuntime[0].StartsWith("Global\\UnslothStudioManagedEnvironment-"))
"""
    env = os.environ.copy()
    env["TEST_ALIAS_ROOT"] = str(junction / ".unsloth" / "studio")
    env["TEST_PHYSICAL_ROOT"] = str(profile / ".unsloth" / "studio")
    assert _run_powershell(shell, script, env).splitlines() == [
        "True",
        "True",
        "1",
        "1",
        "True",
        "True",
    ]


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_path_identity_failure_is_reported_as_unknown(shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    script = f"""
$ErrorActionPreference = "Stop"
{_mutex_helpers(source)}
function Resolve-StudioFinalPathInfo {{ throw "identity unavailable" }}
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
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _ps_file(tmp_path, "holder.ps1", holder_script),
        ],
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        encoding = "utf-8",
        errors = "replace",
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
def test_unknown_root_identity_acquires_sid_and_path_mutexes(tmp_path: Path, shell: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    helpers = _mutex_helpers(source)
    custom_root = tmp_path / "custom-studio"
    custom_root.mkdir()
    script = f"""
$ErrorActionPreference = "Stop"
{helpers}
$names = @(
    Get-StudioRuntimeMutexNames -TauriRootMatch $null -Path $env:TEST_STUDIO_HOME
)
$names | ForEach-Object {{ Write-Output $_ }}
"""
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(custom_root)
    names = _run_powershell(shell, script, env).splitlines()
    assert len(names) == 2
    assert any(name.startswith("Global\\UnslothStudioManagedEnvironment-S-1-") for name in names)
    assert any(name.startswith("Global\\UnslothStudioManagedEnvironmentPath-") for name in names)


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_unicode_custom_root_mutex_name_matches_python(tmp_path: Path, shell: str):
    from unsloth_cli import _studio_runtime_gate as gate

    source = INSTALL_PS1.read_text(encoding = "utf-8")
    helpers = _mutex_helpers(source)
    custom_root = tmp_path / "Unsloth-ß"
    custom_root.mkdir()
    script = f"""
$ErrorActionPreference = "Stop"
{helpers}
Write-Output (Get-StudioRuntimeMutexNameForPath -Path $env:TEST_STUDIO_HOME)
"""
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(custom_root)
    powershell_name = _run_powershell(shell, script, env)
    assert powershell_name == gate.runtime_mutex_name_for_studio_home(custom_root)


@pytest.mark.skipif(os.name != "nt" or not POWERSHELLS, reason = "Windows PowerShell is required")
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_drive_root_identity_and_mutex_names_match_python(shell: str):
    from unsloth_cli import _studio_runtime_gate as gate

    source = INSTALL_PS1.read_text(encoding = "utf-8")
    helpers = _mutex_helpers(source)
    drive_root = Path(f"{os.environ['SystemDrive']}\\")
    script = f"""
$ErrorActionPreference = "Stop"
{helpers}
Write-Output (Get-StudioFinalPath -Path $env:TEST_STUDIO_HOME)
Write-Output (Get-StudioInstallMutexName -Path $env:TEST_STUDIO_HOME)
Write-Output (Get-StudioRuntimeMutexNameForPath -Path $env:TEST_STUDIO_HOME)
"""
    env = os.environ.copy()
    env["TEST_STUDIO_HOME"] = str(drive_root)
    final_path, install_name, runtime_name = _run_powershell(shell, script, env).splitlines()
    canonical = gate._resolved_windows_path(drive_root)
    install_digest = hashlib.sha256(canonical.upper().encode("utf-8")).hexdigest()

    assert final_path == canonical
    assert final_path.endswith("\\")
    assert install_name == f"Global\\UnslothStudioInstall-{install_digest}"
    runtime_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    expected_runtime = f"Global\\UnslothStudioManagedEnvironmentPath-{runtime_digest}"
    assert runtime_name == expected_runtime


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
        [
            shell,
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            _ps_file(tmp_path, "holder.ps1", holder_script),
        ],
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        encoding = "utf-8",
        errors = "replace",
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


def test_runtime_path_hash_is_defined_before_custom_root_lock_uses_it():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    path_hash = source.index("    function Get-StudioRuntimePathHash {")
    path_mutex = source.index("    function Get-StudioRuntimeMutexNameForPath {")
    acquire = source.index("$studioRuntimeMutexNames = @(")
    assert path_hash < path_mutex < acquire


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
    scan_candidates = source.index("$protectedProcessPaths = @(", runtime_lock)
    legacy_source = source.index('Join-Path $StudioHome ".venv"', scan_candidates)
    cwd_source = source.index('Join-Path $env:USERPROFILE "unsloth_studio"', legacy_source)
    runtime_guard = source.index("foreach ($candidate in $protectedProcessPaths)", cwd_source)
    desktop_guard = source.index("Get-StudioDesktopProcessesForCurrentUser", runtime_guard)
    dependency_check = source.index('Write-TauriLog "STEP" "Checking system dependencies"')
    rollback = source.index("Start-StudioVenvRollback -ExistingDir $VenvDir", desktop_guard)
    old_venv_move = source.index("Move-Item -LiteralPath $OldVenv", rollback)
    cwd_venv_move = source.index("Move-Item -LiteralPath $CwdVenv", old_venv_move)
    restore = source.rindex("Restore-StudioVenvRollback")
    prompt = source.index("Start Unsloth Studio now?", restore)
    autostart = source.index("Start-Process -FilePath $VenvPython", prompt)
    release_runtime = source.rindex("Exit-StudioInstallMutex -Mutex $studioRuntimeMutexes[$i]")
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
    assert cwd_source < runtime_guard < desktop_guard < dependency_check < rollback
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
    # Anchored past the command token:
    assert "--clear" not in source[source.index("venv $VenvDir") :][:200]


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
    acquire = process_source.index("STUDIO_RUNTIME_GATE_ACQUIRE_ENV", resolve)
    spawn = process_source.index("cmd.spawn()", acquire)
    store = process_source.index("proc.owned = Some(", spawn)
    assert guard < resolve < acquire < spawn < store


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

    update_child_fn = update_source.index("let run_child = || {")
    update_spawn = update_source.index("spawn_update(&bin, &state", update_child_fn)
    update_wait = update_source.index("wait_for_exit(&state)", update_spawn)
    update_call = update_source.index(
        "crate::process::with_studio_runtime_launch_guard(",
        update_wait,
    )
    exemption = update_source.index("fn mutates_live_environment(&self) -> bool {")
    assert (
        "!matches!(self, UpdateKind::Staged { .. })" in update_source[exemption : exemption + 200]
    )
    update_scan_gate = update_source.index("if kind.mutates_live_environment() {", update_wait)
    update_scan = update_source.index(
        "ensure_managed_environment_is_idle(&bin)",
        update_scan_gate,
    )
    update_gated_child = update_source.index("run_child()", update_scan)
    update_guard_release = update_source.index("\n        })", update_gated_child)
    assert update_child_fn < update_spawn < update_wait < update_scan_gate
    assert update_scan_gate < update_call < update_scan
    assert update_scan < update_gated_child < update_guard_release

    # a staged child acquires the gate itself so app death cannot release it early.
    configure_gate = update_source.index("fn configure_runtime_gate_environment(")
    staged_branch = update_source.index("cmd.env_remove(", configure_gate)
    stage_run = update_source.index("} else {\n        run_child()", update_call)
    assert configure_gate < staged_branch < update_child_fn < stage_run


def test_runtime_gate_handoff_covers_managed_children():
    process_source = PROCESS_RS.read_text(encoding = "utf-8")
    install_source = INSTALL_PS1.read_text(encoding = "utf-8")
    studio_source = STUDIO_COMMAND.read_text(encoding = "utf-8")

    start = process_source.index("pub fn start_backend(")
    clear_handoff = process_source.index("cmd.env_remove(STUDIO_RUNTIME_GATE_HANDOFF_ENV)", start)
    acquire = process_source.index('cmd.env(STUDIO_RUNTIME_GATE_ACQUIRE_ENV, "1")', clear_handoff)
    spawn = process_source.index("cmd.spawn()", acquire)
    assert clear_handoff < acquire < spawn

    prompt = install_source.index("Start Unsloth Studio now?")
    save = install_source.index("$_runtimeGateHandoff =", prompt)
    set_handoff = install_source.index(
        '$env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"',
        save,
    )
    autostart = install_source.index("Start-Process -FilePath $VenvPython", set_handoff)
    restore = install_source.index(
        "$env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = $_runtimeGateHandoff",
        autostart,
    )
    assert save < set_handoff < autostart < restore

    setup_python = install_source.index("$env:UNSLOTH_SETUP_PYTHON =")
    setup_save = install_source.index("$previousSetupRuntimeGateHandoff =", setup_python)
    setup_set = install_source.index(
        '$env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = "1"',
        setup_save,
    )
    setup_invoke = install_source.index(
        "Invoke-ManagedUnslothCli -Python $VenvPython -Arguments $studioArgs", setup_set
    )
    setup_restore = install_source.index(
        "$env:_UNSLOTH_STUDIO_RUNTIME_GATE_HANDOFF = $previousSetupRuntimeGateHandoff",
        setup_invoke,
    )
    tauri_remove = install_source.index("Remove-Item Env:UNSLOTH_TAURI_MODE", setup_invoke)
    assert setup_python < setup_save < setup_set < setup_invoke < tauri_remove < setup_restore

    assert (
        studio_source.count(
            "runtime_gate_handoff = _studio_runtime_gate.consume_runtime_gate_handoff()"
        )
        == 5
    )
    assert (
        studio_source.count(
            "runtime_gate_acquire = _studio_runtime_gate.consume_runtime_gate_acquire()"
        )
        == 1
    )
    assert studio_source.count("inherited = runtime_gate_handoff") >= 5


def test_a_reopened_app_cannot_replace_or_discard_an_externally_owned_stage():
    update_source = UPDATE_RS.read_text(encoding = "utf-8")
    commands_source = COMMANDS_RS.read_text(encoding = "utf-8")
    main_source = MAIN_RS.read_text(encoding = "utf-8")

    owner_check = update_source.index("pub(crate) fn staged_update_is_owned_elsewhere()")
    gate_probe = update_source.index("with_studio_runtime_launch_guard", owner_check)
    owner_helper = update_source.index("fn staged_update_is_owned_elsewhere_at(", gate_probe)
    stage_probe = update_source.index("crate::staged_update::STAGE_DIR", owner_helper)
    status = update_source.index("pub(crate) fn is_staged_update_running")
    status_uses_owner = update_source.index("staged_update_is_owned_elsewhere()", status)
    assert status < status_uses_owner < owner_check < gate_probe < owner_helper < stage_probe

    start = commands_source.index("pub async fn start_staged_update(")
    start_guard = commands_source.index("is_staged_update_running", start)
    start_spawn = commands_source.index("update::run_staged_update", start_guard)
    cancel = commands_source.index("pub fn cancel_staged_update(", start_spawn)
    cancel_stop = commands_source.index("update::stop_update", cancel)
    cancel_guard = commands_source.index("with_studio_runtime_launch_guard", cancel_stop)
    cancel_remove = commands_source.index("staged_update::discard", cancel_guard)
    discard = commands_source.index("pub fn discard_staged_update(")
    discard_guard = commands_source.index("with_studio_runtime_launch_guard", discard)
    discard_remove = commands_source.index("staged_update::discard", discard_guard)
    assert start < start_guard < start_spawn < cancel < cancel_stop < cancel_guard < cancel_remove
    assert cancel_remove < discard < discard_guard < discard_remove

    setup = main_source.index(".setup(|app| {")
    reconcile_gate = main_source.index("with_studio_runtime_launch_guard", setup)
    reconcile = main_source.index("staged_update::reconcile_at_launch", reconcile_gate)
    assert setup < reconcile_gate < reconcile


def test_tauri_start_install_rejects_backend_conflicts_before_spawn():
    source = COMMANDS_RS.read_text(encoding = "utf-8")
    start = source.index("pub async fn start_install(")
    end = source.index("\n}\n", start)
    body = source[start:end]

    owned_guard = body.index("has_owned_backend(&backend_state)?")
    external_guard = body.index("block_external_conflict(&[]).await?")
    spawn = body.index("install::run_install")
    assert owned_guard < external_guard < spawn


@pytest.mark.parametrize(
    "helpers",
    [_mutex_helpers, _process_helpers],
    ids = ["mutex", "process"],
)
def test_the_extracted_helpers_can_call_everything_they_call(helpers):
    """Every installer function these harnesses reach must be in the harness.

    The scripts above run under -ErrorActionPreference Stop, so a helper that
    calls an installer function nobody extracted dies with CommandNotFound, and
    the test fails for a reason unrelated to what it measures. That is not
    hypothetical: Test-StudioPathEqual reports an unresolvable path identity
    through Write-StudioLine, which was missing, so both
    test_path_identity_failure_is_reported_as_unknown cases failed on Windows
    while passing nowhere they could be noticed.

    Runs on every platform, unlike the scripts themselves, so the harness cannot
    drift out of step again where only a Windows runner would see it.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    extracted = helpers(source)
    # Every top-level installer function, i.e.
    # everything the harness COULD be missing.
    installer_functions = set(re.findall(r"^    function ([\w-]+) \{", source, flags = re.M))
    provided = set(re.findall(r"^    function ([\w-]+) \{", extracted, flags = re.M))
    assert provided, "the helper extraction produced nothing"

    called = set(re.findall(r"(?<![\w-])([A-Z][\w]*-[\w-]+)", extracted))
    missing = sorted((called & installer_functions) - provided)
    assert not missing, (
        f"{helpers.__name__} extracts functions that call {missing}, which the "
        "harness never defines; add them to the extraction list"
    )
