# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows prefers uv-managed CPython and preserves the requested version and architecture.
A system 3.12 must never silently replace a requested 3.13 (the cu130 torch route).
"""

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
needs_pwsh = pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")

X64 = "cpython-3.13-windows-x86_64-none"
ARM = "cpython-3.13-windows-aarch64-none"

# Records every call; `find` answers with the fake interpreter only for keys in FAKE_UV_FACTS,
# after writing the facts that interpreter will report.
FAKE_UV = r"""
Add-Content -LiteralPath $env:FAKE_UV_LOG -Value ($args -join ' ')
$facts = if ($env:FAKE_UV_FACTS) { $env:FAKE_UV_FACTS | ConvertFrom-Json } else { $null }
$key = $args[-1]
if ($args[0] -eq 'python' -and $args[1] -eq 'install') { exit [int]$env:FAKE_UV_INSTALL_EXIT }
if ($args[0] -eq 'python' -and $args[1] -eq 'find') {
    if ($facts -and $facts.PSObject.Properties.Name -contains $key) {
        Set-Content -LiteralPath $env:FAKE_PY_FACTS_FILE -Value $facts.$key
        Write-Output $env:FAKE_PY
        exit 0
    }
    exit 2
}
if ($args[0] -eq 'python' -and $args[1] -eq 'dir') { Write-Output $env:FAKE_UV_DIR; exit 0 }
exit 9
"""


def _function(name: str) -> str:
    match = re.search(rf"    function {name} \{{.*?\n    \}}\n", SOURCE, flags = re.DOTALL)
    assert match is not None, name
    return match.group(0)


def _between(start: str, end: str) -> str:
    a = SOURCE.index(start)
    return SOURCE[a : SOURCE.index(end, a)]


def _fake_python(tmp_path: Path) -> Path:
    """An interpreter stand-in that prints whatever facts the fake uv last wrote."""
    if os.name == "nt":
        exe = tmp_path / "python.cmd"
        exe.write_text('@type "%FAKE_PY_FACTS_FILE%"\r\n', encoding = "utf-8")
    else:
        exe = tmp_path / "python"
        exe.write_text('#!/bin/sh\ncat "$FAKE_PY_FACTS_FILE"\n', encoding = "utf-8")
        exe.chmod(0o755)
    return exe


def _pwsh(shell: str, script: str, env: dict[str, str]) -> str:
    return run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        env = dict(os.environ, **env),
        timeout = 60,
    ).stdout.strip()


def _resolve(
    shell: str,
    tmp_path: Path,
    *,
    facts: dict[str, str],
    arch: str = "x86_64",
    patch: str = "",
    skip: list[str] | None = None,
    install_exit: int = 0,
):
    (tmp_path / "uv.ps1").write_text(FAKE_UV, encoding = "utf-8")
    log = tmp_path / "uv.log"
    fake_py = _fake_python(tmp_path)
    skip_list = ", ".join(f"'{v}'" for v in (skip or []))
    script = f"""
$ErrorActionPreference = "Stop"
function Invoke-InstallCommand {{
    param([ScriptBlock]$Command, [string]$Label, [switch]$NoMirror)
    $global:LASTEXITCODE = 0
    & $Command | Out-Null
    return [int]$LASTEXITCODE
}}
$PythonSkip = @({skip_list})
$script:UvExe = $env:FAKE_UV_EXE
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) }}
{_function("Test-ApplicationControlBlock")}
{_function("Invoke-PythonSelectionProbe")}
function Redact-InstallOutput {{ param($Text) $Text }}
{_function("Resolve-UvManagedPython")}
$r = Resolve-UvManagedPython -Minor "3.13" -Patch "{patch}" -Arch "{arch}"
if ($r) {{ $r | ConvertTo-Json -Compress }} else {{ 'null' }}
"""
    out = _pwsh(
        shell,
        script,
        {
            "FAKE_UV_LOG": str(log),
            "FAKE_UV_EXE": str(tmp_path / "uv.ps1"),
            "FAKE_UV_INSTALL_EXIT": str(install_exit),
            "FAKE_UV_FACTS": json.dumps(facts),
            "FAKE_PY": str(fake_py),
            "FAKE_PY_FACTS_FILE": str(tmp_path / "facts.txt"),
        },
    )
    calls = log.read_text(encoding = "utf-8").splitlines() if log.exists() else []
    return json.loads(out.splitlines()[-1]), calls, str(fake_py)


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "arch, key, tag",
    [("x86_64", X64, "win-amd64"), ("arm64", ARM, "win-arm64")],
)
def test_the_request_names_the_architecture_and_touches_nothing_else(
    shell, tmp_path, arch, key, tag
):
    result, calls, fake_py = _resolve(shell, tmp_path, facts = {key: f"3.13.12 {tag} 0"}, arch = arch)
    assert result == {"Version": "3.13", "Full": "3.13.12", "Path": fake_py, "Arch": arch}
    assert calls == [
        f"python install --no-bin --no-registry {key}",
        f"python find --system --managed-python {key}",
    ]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_failed_download_still_finds_a_runtime_uv_installed_earlier(shell, tmp_path):
    result, calls, _ = _resolve(shell, tmp_path, facts = {X64: "3.13.12 win-amd64 0"}, install_exit = 1)
    assert result["Full"] == "3.13.12"
    assert calls[1] == f"python find --system --managed-python {X64}"


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_nothing_managed_is_nothing(shell, tmp_path):
    result, _, _ = _resolve(shell, tmp_path, facts = {}, install_exit = 1)
    assert result is None


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "arch, key, facts",
    [
        ("arm64", ARM, "3.13.12 win-amd64 0"),  # emulated x64 build for a native request
        ("x86_64", X64, "3.13.12 win-arm64 0"),
        ("x86_64", X64, "3.13.12 win-amd64 1"),  # free-threaded
        ("x86_64", X64, "3.12.9 win-amd64 0"),  # another minor
        ("x86_64", X64, "garbage"),
    ],
)
def test_an_interpreter_that_is_not_what_was_asked_for_is_refused(
    shell, tmp_path, arch, key, facts
):
    result, _, _ = _resolve(shell, tmp_path, facts = {key: facts}, arch = arch)
    assert result is None


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_skipped_patch_is_stepped_past(shell, tmp_path):
    next_key = "cpython-3.13.9-windows-x86_64-none"
    result, calls, _ = _resolve(
        shell,
        tmp_path,
        facts = {X64: "3.13.8 win-amd64 0", next_key: "3.13.9 win-amd64 0"},
        skip = ["3.13.8"],
    )
    assert result["Full"] == "3.13.9"
    assert calls[2:] == [
        f"python install --no-bin --no-registry {next_key}",
        f"python find --system --managed-python {next_key}",
    ]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_a_skipped_patch_falls_back_to_an_older_cached_one_offline(shell, tmp_path):
    older = "cpython-3.13.7-windows-x86_64-none"
    result, calls, _ = _resolve(
        shell,
        tmp_path,
        facts = {X64: "3.13.8 win-amd64 0", older: "3.13.7 win-amd64 0"},
        skip = ["3.13.8"],
        install_exit = 1,
    )
    assert result["Full"] == "3.13.7"
    assert f"python find --system --managed-python cpython-3.13.9-windows-x86_64-none" in calls
    assert calls[-1] == f"python find --system --managed-python {older}"


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_an_explicit_patch_is_exact(shell, tmp_path):
    key = "cpython-3.13.9-windows-x86_64-none"
    ok, calls, _ = _resolve(shell, tmp_path, facts = {key: "3.13.9 win-amd64 0"}, patch = "3.13.9")
    assert ok["Full"] == "3.13.9" and calls[0].endswith(key)
    wrong, _, _ = _resolve(shell, tmp_path, facts = {key: "3.13.12 win-amd64 0"}, patch = "3.13.9")
    assert wrong is None


# ── The request: --python > UNSLOTH_PYTHON > 3.13, exact, validated ──

REQUEST = _between(
    "    $argList = $args\n", "    # Env-var equivalent for web installs"
) + _between(
    "    # Request a specific CPython:",
    "    # An elevated run writes",
)


def _request(
    shell: str,
    args: list[str],
    env_python: str | None = None,
) -> str:
    quoted = ", ".join("'" + a.replace("'", "''") + "'" for a in args)
    script = f"""
$ErrorActionPreference = "Stop"
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) }}
function Exit-InstallFailure {{ param([string]$Message) return "EXIT" }}
function Probe {{
    $SkipTorch = $false
    $PackageName = "unsloth"
{REQUEST}
    return "$PythonVersion|$PythonPatchRequest"
}}
$a = @({quoted})
Probe @a
"""
    env = {"UNSLOTH_PYTHON": env_python} if env_python is not None else {"UNSLOTH_PYTHON": ""}
    return _pwsh(shell, script, env).splitlines()[-1]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "args, env_python, expected",
    [
        ([], None, "3.13|"),
        (["--python", "3.12"], None, "3.12|"),
        (["--python=3.11"], None, "3.11|"),
        (["--python", "3.13.9"], None, "3.13|3.13.9"),
        ([], "3.12", "3.12|"),
        (["--python", "3.11"], "3.12", "3.11|"),
        (["--no-torch", "--python", "3.13.8"], None, "3.13|3.13.8"),
        (["--python", "3.10"], None, "EXIT"),
        (["--python", "3.14"], None, "EXIT"),
        (["--python", "3.13.8"], None, "EXIT"),
        (["--python"], None, "EXIT"),
        (["--python="], None, "EXIT"),
        ([], "C:\\Python312\\python.exe", "EXIT"),
    ],
)
def test_the_request(shell, args, env_python, expected):
    assert _request(shell, args, env_python) == expected


# ── The choice made before uv exists: minor and architecture ──

CHOOSE = _function("Get-StudioVenvPythonMinor") + _between(
    "    # ── uv-managed Python: minor and arch now", "    $DetectedPython = $null\n"
)


def _choose(
    shell: str,
    tmp_path: Path,
    *,
    host: str = "x86_64",
    native: bool = False,
    opt_out: bool = False,
    venv_cfg: str | None = None,
    explicit: str = "",
    upgrade: bool = False,
    legacy_cfg: str | None = None,
    legacy_layout: bool = True,
) -> str:
    venv = tmp_path / "venv"
    venv.mkdir(exist_ok = True)
    if venv_cfg is not None:
        (venv / "pyvenv.cfg").write_text(venv_cfg, encoding = "utf-8")
    home = tmp_path / "home"
    (home / ".venv").mkdir(parents = True, exist_ok = True)
    if legacy_cfg is not None:
        (home / ".venv" / "pyvenv.cfg").write_text(legacy_cfg, encoding = "utf-8")
    script = f"""
$ErrorActionPreference = "Stop"
function Write-TauriLog {{ param($a, $b) }}
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) }}
function step {{ param($a, $b, $c) }}
function substep {{ param($a, $b) }}
function step {{ param($a, $b) }}
function Remove-IndexUrlCredentials {{ param($u) return $u }}
function Get-HostMachineArch {{ return "{host}" }}
function Initialize-WoaNativeCudaTorch {{
    param([string]$PythonMinor)
    $script:WoaNativeCudaTorch = ("{host}" -eq "arm64" -and ${str(native).lower()})
    $script:WoaTorchIndexUrl = "https://example.invalid/idx"
}}
{_function("Test-Arm64PythonOptOut")}
function Probe {{
    $PythonVersion = if ("{explicit}") {{ "{explicit}" }} else {{ "3.13" }}
    $PythonExplicitRequest = "{explicit}"
    $PythonPatchRequest = ""
    $VenvDir = $env:FAKE_VENV
    $StudioHome = $env:FAKE_HOME
    $studioUsesLegacyLayout = ${str(legacy_layout).lower()}
{CHOOSE}
    return "$PythonVersion|$PythonArch"
}}
Probe
"""
    env = {
        "FAKE_VENV": str(venv),
        "FAKE_HOME": str(home),
        "USERPROFILE": str(tmp_path / "profile"),
        "UNSLOTH_ALLOW_ARM64_PYTHON": "1" if opt_out else "",
        "UNSLOTH_TORCH_UPGRADE": "1" if upgrade else "",
    }
    return _pwsh(shell, script, env).splitlines()[-1]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "3.13|x86_64"),
        # A reinstall keeps the environment's minor (uv writes version_info, venv writes version).
        ({"venv_cfg": "home = C:\\py\nversion_info = 3.12.6\n"}, "3.12|x86_64"),
        ({"venv_cfg": "home = C:\\py\nversion = 3.11.9\n"}, "3.11|x86_64"),
        ({"venv_cfg": "version_info = 3.14.0\n"}, "3.13|x86_64"),
        ({"venv_cfg": "version_info = 3.12.6\n", "upgrade": True}, "3.13|x86_64"),
        ({"venv_cfg": "version_info = 3.12.6\n", "explicit": "3.13"}, "3.13|x86_64"),
        # A legacy layout keeps its minor too; the current layout wins when both exist.
        ({"legacy_cfg": "version_info = 3.12.6\n"}, "3.12|x86_64"),
        ({"legacy_cfg": "version_info = 3.12.6\n", "legacy_layout": False}, "3.13|x86_64"),
        (
            {"legacy_cfg": "version_info = 3.12.6\n", "venv_cfg": "version_info = 3.11.9\n"},
            "3.11|x86_64",
        ),
        ({"legacy_cfg": "version_info = 3.12.6\n", "upgrade": True}, "3.13|x86_64"),
        # Windows on ARM: x64 under emulation unless the native CUDA stack or the opt-out applies.
        ({"host": "arm64"}, "3.13|x86_64"),
        ({"host": "arm64", "native": True}, "3.13|arm64"),
        ({"host": "arm64", "opt_out": True}, "3.13|arm64"),
        ({"opt_out": True}, "3.13|x86_64"),
    ],
)
def test_the_choice(shell, tmp_path, kwargs, expected):
    assert _choose(shell, tmp_path, **kwargs) == expected


# ── Resolution after uv: a failed native ARM64 build degrades to x64, anything else stops ──

RESOLVE_SITE = _between(
    "    $DetectedPython = Resolve-UvManagedPython -Minor $PythonVersion",
    "    # ── Create the venv;",
)


def _site(
    shell: str,
    arch: str,
    ok_arches: str,
    opt_out: bool = False,
) -> str:
    script = f"""
$ErrorActionPreference = "Stop"
function Write-StudioLine {{ param([string]$Message, [string]$ForegroundColor) }}
function substep {{ param($a, $b) }}
function step {{ param($a, $b) }}
function Exit-InstallFailure {{ param([string]$Message) return "EXIT" }}
function Resolve-UvManagedPython {{
    param([string]$Minor, [string]$Patch, [string]$Arch)
    if (("{ok_arches}" -split ",") -contains $Arch) {{
        return @{{ Version = $Minor; Full = "$Minor.12"; Path = "p"; Arch = $Arch }}
    }}
    return $null
}}
{_function("Test-Arm64PythonOptOut")}
function Probe {{
    $PythonVersion = "3.13"; $PythonPatchRequest = ""; $PythonRequestLabel = "3.13"
    $PythonArch = "{arch}"
    $script:WoaNativeCudaTorch = ("{arch}" -eq "arm64")
{RESOLVE_SITE}
    return "$($DetectedPython.Arch)|$script:WoaNativeCudaTorch"
}}
Probe
"""
    return _pwsh(
        shell, script, {"UNSLOTH_ALLOW_ARM64_PYTHON": "1" if opt_out else ""}
    ).splitlines()[-1]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "arch, ok_arches, opt_out, expected",
    [
        ("x86_64", "x86_64", False, "x86_64|False"),
        ("arm64", "arm64,x86_64", False, "arm64|True"),
        ("arm64", "x86_64", False, "x86_64|False"),
        ("arm64", "x86_64", True, "EXIT"),
        ("x86_64", "", False, "EXIT"),
    ],
)
def test_the_resolution(shell, arch, ok_arches, opt_out, expected):
    assert _site(shell, arch, ok_arches, opt_out) == expected


# ── A legacy environment is reused only on the uv-managed Python this install resolved ──

STORE_HOME = "store/cpython-3.13.12-windows-x86_64-none"


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "home, version, exact, expected",
    [
        (STORE_HOME, "3.13.12", False, "True"),
        (STORE_HOME, "3.13.9", False, "True"),  # same minor, another patch
        (STORE_HOME, "3.13.9", True, "False"),  # --python 3.13.12 asked for exactly
        ("store/cpython-3.12.13-windows-x86_64-none", "3.12.13", False, "False"),
        ("Python312", "3.13.12", False, "False"),  # the user's own Python
        ("store/cpython-3.13.12-windows-aarch64-none", "3.13.12", False, "False"),  # another arch
        (STORE_HOME, None, False, "False"),
        (None, None, False, "False"),
    ],
)
def test_a_legacy_environment_is_reused_only_on_the_resolved_python(
    shell, tmp_path, home, version, exact, expected
):
    (tmp_path / "uv.ps1").write_text(FAKE_UV, encoding = "utf-8")
    venv = tmp_path / "legacy"
    venv.mkdir()
    if home is not None:
        cfg = f"home = {tmp_path / home}\n"
        if version:
            cfg += f"version_info = {version}\n"
        (venv / "pyvenv.cfg").write_text(cfg, encoding = "utf-8")
    script = f"""
$ErrorActionPreference = "Stop"
$script:UvExe = $env:FAKE_UV_EXE
{_function("Test-LegacyVenvReusable")}
$py = @{{ Version = "3.13"; Full = "3.13.12"; Path = "p"; Arch = "x86_64" }}
Test-LegacyVenvReusable -VenvDir $env:FAKE_VENV -Python $py -Exact ${str(exact).lower()}
"""
    out = _pwsh(
        shell,
        script,
        {
            "FAKE_UV_EXE": str(tmp_path / "uv.ps1"),
            "FAKE_UV_LOG": str(tmp_path / "uv.log"),
            "FAKE_UV_DIR": str(tmp_path / "store"),
            "FAKE_VENV": str(venv),
        },
    )
    assert out.splitlines()[-1] == expected


def test_pyvenv_cfg_is_read_as_utf8():
    # uv writes it without a BOM; Windows PowerShell 5.1 would otherwise decode a non-ASCII home as ANSI.
    for name in ("Get-StudioVenvPythonMinor", "Test-LegacyVenvReusable"):
        assert "Get-Content -LiteralPath $cfg -Encoding UTF8" in _function(name), name


def test_no_unrestricted_system_python_fallback_and_resolution_precedes_rollback():
    for gone in (
        "Find-CompatiblePython",
        "Install-PythonFromPythonOrg",
        "Install-X64Python",
        '"Python.Python.',
        "python.org/ftp/python",
        '"-0p"',
    ):
        assert gone not in SOURCE, gone
    uv_ready = SOURCE.index("Set-StudioUvCacheEnvironment -StudioRoot $StudioHome")
    resolve = SOURCE.index("$DetectedPython = Resolve-UvManagedPython -Minor")
    rollback = SOURCE.index("Start-StudioVenvRollback -ExistingDir $VenvDir")
    venv = SOURCE.index("& $script:UvExe venv $VenvDir --python")
    # Resolved before the existing environment is moved aside, so a failure leaves it intact.
    assert uv_ready < resolve < rollback < venv


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "code, message, blocked",
    [
        (1, "uv trampoline failed: uncategorized error (os error 4551)", True),
        (1, "[WinError 1260] blocked", True),
        (1, "network timeout 4551", False),
        (1, "os error 45510", False),
        (1, "[WinError 577] invalid image", False),
        (0, "os error 4551", False),
    ],
)
def test_native_stderr_retains_policy_evidence(shell, code, message, blocked):
    script = f"""
$ErrorActionPreference = 'Stop'
{_function('Test-ApplicationControlBlock')}
{_function('Invoke-PythonSelectionProbe')}
$r = Invoke-PythonSelectionProbe -Executable $env:PROBE_PYTHON -Arguments @('-c',
    "import sys; print('{message}', file=sys.stderr); sys.exit({code})")
@{{ Blocked = $r.Blocked; Success = $r.Success; Diagnostic = $r.Diagnostic; EAP = "$ErrorActionPreference" }} | ConvertTo-Json -Compress
"""
    r = json.loads(_pwsh(shell, script, {"PROBE_PYTHON": sys.executable}))
    assert r["Blocked"] is blocked
    assert r["Success"] is (code == 0)
    assert message in r["Diagnostic"]
    assert r["EAP"] == "Stop"


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("code, blocked", [(4551, True), (1260, True), (5, False), (577, False)])
def test_spawn_exception_does_not_use_stale_exit_code(shell, tmp_path, code, blocked):
    child = tmp_path / "blocked.ps1"
    child.write_text(f"throw [System.ComponentModel.Win32Exception]::new({code})", encoding = "utf-8")
    script = f"""
$ErrorActionPreference = 'Stop'
$LASTEXITCODE = 0
{_function('Test-ApplicationControlBlock')}
{_function('Invoke-PythonSelectionProbe')}
$r = Invoke-PythonSelectionProbe -Executable $env:PROBE_CHILD
@{{ Blocked = $r.Blocked; Success = $r.Success }} | ConvertTo-Json -Compress
"""
    r = json.loads(_pwsh(shell, script, {"PROBE_CHILD": str(child)}))
    assert r == {"Blocked": blocked, "Success": False}


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "managed, blocked, signed, expected, calls",
    [
        (True, False, True, "managed", 0),
        (False, False, True, "failed", 0),  # offline is not a reason to change interpreter source
        (False, True, True, "signed-managed", 1),
        (False, True, False, "failed", 1),
    ],
)
def test_installer_fallback_is_gated_by_policy_evidence(
    shell, managed, blocked, signed, expected, calls
):
    script = f"""
$ErrorActionPreference = 'Stop'
$script:Calls = 0
$script:Messages = @()
function Write-StudioLine {{ param($Message, $ForegroundColor) $script:Messages += $Message }}
function substep {{ param($Message, $Color) }}
function step {{ param($Name, $Message) }}
function Exit-InstallFailure {{ param($Message) return @{{ Source = 'failed' }} }}
function Resolve-UvManagedPython {{
    param($Minor, $Patch, $Arch)
    $script:PythonPolicyBlocked = ${str(blocked).lower()}
    if (${str(managed).lower()}) {{ return @{{ Full = '3.13.17'; Arch = $Arch; Source = 'managed' }} }}
}}
function Resolve-SignedManagedPython {{
    param($Minor, $Patch, $Arch)
    $script:Calls++
    if (${str(signed).lower()}) {{ return @{{ Full = '3.13.17'; Arch = $Arch; Source = 'signed-managed' }} }}
}}
function Run {{
    $PythonVersion = '3.13'; $PythonPatchRequest = ''; $PythonRequestLabel = '3.13'; $PythonArch = 'x86_64'
{RESOLVE_SITE}
    return $DetectedPython
}}
$r = Run
@{{ Source = $r.Source; Calls = $script:Calls; Messages = $script:Messages -join "`n" }} | ConvertTo-Json -Compress
"""
    r = json.loads(_pwsh(shell, script, {}))
    assert r["Source"] == expected
    assert r["Calls"] == calls
    if blocked and not signed:
        assert "Application Control blocked" in r["Messages"]
        assert "mirror" not in r["Messages"]
        assert "existing environment was not changed" in r["Messages"]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "minor, patch, arch, expected",
    [
        ("3.13", "", "x86_64", "python"),
        ("3.13", "3.13.17", "arm64", "pythonarm64"),
        ("3.13", "3.13.16", "x86_64", None),
        ("3.12", "", "x86_64", None),
        ("3.13", "", "x86", None),
    ],
)
def test_signed_package_never_changes_requested_version_or_arch(
    shell, minor, patch, arch, expected
):
    script = f"""
$PythonSkip = @('3.13.8')
{_function('Get-SignedPythonPackage')}
$r = Get-SignedPythonPackage -Minor '{minor}' -Patch '{patch}' -Arch '{arch}'
if ($r) {{ $r | ConvertTo-Json -Compress }} else {{ 'null' }}
"""
    r = json.loads(_pwsh(shell, script, {}))
    if expected is None:
        assert r is None
    else:
        assert r["Version"] == "3.13.17" and r["Arch"] == arch
        assert (
            r["Url"]
            == f"https://api.nuget.org/v3-flatcontainer/{expected}/3.13.17/{expected}.3.13.17.nupkg"
        )
        assert re.fullmatch("[0-9a-f]{64}", r["Sha256"])


@needs_pwsh
@pytest.mark.skipif(os.name != "nt", reason = "Windows runtime tree")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "bad_file, facts, success, expected, probed",
    [
        ("", "3.13.17 win-amd64 0", True, True, True),
        ("python.exe", "3.13.17 win-amd64 0", True, False, False),
        ("DLLs/_ssl.pyd", "3.13.17 win-amd64 0", True, False, False),
        ("Lib/venv/scripts/nt/venvlauncher.exe", "3.13.17 win-amd64 0", True, False, False),
        ("Lib/site-packages/unexpected.exe", "3.13.17 win-amd64 0", True, False, False),
        ("", "3.12.10 win-amd64 0", True, False, True),
        ("", "3.13.17 win-arm64 0", True, False, True),
        ("", "3.13.17 win-amd64 1", True, False, True),
        ("", "policy refused a runtime DLL", False, False, True),
    ],
)
def test_signed_runtime_checks_libraries_and_launchers_before_execution(
    shell, tmp_path, bad_file, facts, success, expected, probed
):
    files = [
        "python.exe",
        "pythonw.exe",
        "python3.dll",
        "python313.dll",
        "Lib/venv/scripts/nt/venvlauncher.exe",
        "Lib/venv/scripts/nt/venvwlauncher.exe",
        "DLLs/_ssl.pyd",
        "Lib/site-packages/pip/_vendor/distlib/t64.exe",
    ]
    if bad_file and bad_file not in files:
        files.append(bad_file)
    for name in files:
        f = tmp_path / name
        f.parent.mkdir(parents = True, exist_ok = True)
        f.touch()
    script = f"""
$ErrorActionPreference = 'Stop'
$script:Probed = $false
function substep {{ param($Message, $Color) }}
function Redact-InstallOutput {{ param($Text) $Text }}
function Get-AuthenticodeSignature {{
    param($LiteralPath, $ErrorAction)
    $cert = [pscustomobject]@{{}}
    $cert | Add-Member ScriptMethod GetNameInfo {{ param($type, $issuer) 'Python Software Foundation' }}
    $status = if ($LiteralPath -eq $env:BAD_FILE -or $LiteralPath.EndsWith('distlib\\t64.exe')) {{ 'NotSigned' }} else {{ 'Valid' }}
    [pscustomobject]@{{ Status = $status; SignerCertificate = $cert }}
}}
function Invoke-PythonSelectionProbe {{
    param($Executable, $Arguments)
    $script:Probed = $true
    @{{ Success = ${str(success).lower()}; Diagnostic = '{facts}' }}
}}
{_function('Test-SignedPythonRuntime')}
$r = Test-SignedPythonRuntime -Root $env:RUNTIME -Package @{{ Version='3.13.17'; Minor='3.13'; Arch='x86_64' }}
@{{ Accepted = [bool]$r; Probed = $script:Probed }} | ConvertTo-Json -Compress
"""
    r = json.loads(
        _pwsh(
            shell,
            script,
            {"RUNTIME": str(tmp_path), "BAD_FILE": str(tmp_path / bad_file) if bad_file else ""},
        )
    )
    assert r == {"Accepted": expected, "Probed": probed}


@needs_pwsh
@pytest.mark.skipif(os.name != "nt", reason = "Windows runtime tree")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "failure", ["checksum", "offline", "downloads-never", "signature", "download"]
)
def test_signed_runtime_failure_preserves_existing_environment(shell, tmp_path, failure):
    home = tmp_path / "home"
    old = home / "unsloth_studio" / "keep.txt"
    old.parent.mkdir(parents = True)
    old.write_text("previous environment")
    script = f"""
$ErrorActionPreference = 'Stop'
$StudioHome = $env:TEST_HOME
$script:Downloads = 0; $script:Extracts = 0
function substep {{ param($Message, $Color) }}
function Redact-InstallOutput {{ param($Text) $Text }}
function Get-SignedPythonPackage {{ param($Minor,$Patch,$Arch) @{{ Version='3.13.17'; Minor='3.13'; Arch='x86_64'; Sha256=('0'*64); Url='https://example.invalid/python.nupkg' }} }}
function Invoke-WebRequest {{
    param($UseBasicParsing, $Uri, $OutFile, $TimeoutSec, $ErrorAction)
    $script:Downloads++
    if ('{failure}' -eq 'download') {{ throw 'network unavailable' }}
    [IO.File]::WriteAllText($OutFile, 'wrong archive')
}}
function Get-FileHash {{ param($LiteralPath,$Algorithm) @{{ Hash = $(if ('{failure}' -eq 'checksum') {{ 'wrong' }} else {{ '0'*64 }}) }} }}
function Expand-Archive {{
    param($LiteralPath,$DestinationPath,$ErrorAction)
    $script:Extracts++
    [IO.Directory]::CreateDirectory((Join-Path $DestinationPath 'tools')) | Out-Null
}}
function Test-SignedPythonRuntime {{ param($Root,$Package) return $null }}
{_function('Test-UvEnvFlag')}
{_function('Resolve-SignedManagedPython')}
$r = Resolve-SignedManagedPython -Minor '3.13'
@{{ Accepted=[bool]$r; Downloads=$script:Downloads; Extracts=$script:Extracts; Stages=@(Get-ChildItem (Join-Path $StudioHome 'python-signed') -Filter '.stage-*' -ErrorAction SilentlyContinue).Count }} | ConvertTo-Json -Compress
"""
    # Switch parameters need their native declaration in the network double.
    script = script.replace("param($UseBasicParsing,", "param([switch]$UseBasicParsing,")
    r = json.loads(
        _pwsh(
            shell,
            script,
            {
                "TEST_HOME": str(home),
                "UV_OFFLINE": "1" if failure == "offline" else "",
                "UV_PYTHON_DOWNLOADS": "never" if failure == "downloads-never" else "",
            },
        )
    )
    assert not r["Accepted"] and r["Stages"] == 0
    assert r["Downloads"] == (0 if failure in ("offline", "downloads-never") else 1)
    assert r["Extracts"] == (1 if failure == "signature" else 0)
    assert old.read_text() == "previous environment"


@needs_pwsh
@pytest.mark.skipif(
    os.name != "nt" or not os.environ.get("UNSLOTH_TEST_PSF_PACKAGE"),
    reason = "requires verified official Python package",
)
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_policy_block_installs_private_signed_runtime_and_reuses_it_offline(shell, tmp_path):
    # Only transport and the original policy refusal are simulated. The archive hash,
    # signatures, native imports, install, cache reuse, and venv creation are real.
    uv = tmp_path / "blocked-uv.cmd"
    uv.write_text(
        "@echo uv trampoline failed to spawn Python child process: os error 4551 1>&2\n@exit /b 1\n",
        encoding = "utf-8",
    )
    create_venv = _between(
        "    if (-not (Test-Path -LiteralPath $VenvPython)) {",
        "    # Mark the managed venv before probing",
    )
    helpers = "\n".join(
        _function(name)
        for name in (
            "Test-ApplicationControlBlock",
            "Invoke-PythonSelectionProbe",
            "Redact-InstallOutput",
            "Resolve-UvManagedPython",
            "Get-SignedPythonPackage",
            "Test-SignedPythonRuntime",
            "Resolve-SignedManagedPython",
            "Test-UvEnvFlag",
            "Test-LegacyVenvReusable",
        )
    )
    script = f"""
$ErrorActionPreference = 'Stop'
$PythonSkip = @('3.13.8'); $StudioHome = $env:TEST_HOME
$script:UvExe = $env:BLOCKED_UV; $script:Downloads=0
function Write-StudioLine {{ param($Message,$ForegroundColor) }}
function substep {{ param($Name,$Message) }}
function step {{ param($Name,$Message) }}
function Exit-InstallFailure {{ param($Message) throw $Message }}
function Invoke-InstallCommand {{
    param([ScriptBlock]$Command,[string]$Label)
    $ErrorActionPreference='Continue'
    & $Command 2>&1 | Out-Null
    return [int]$global:LASTEXITCODE
}}
function Invoke-WebRequest {{
    param([switch]$UseBasicParsing,$Uri,$OutFile,$TimeoutSec,$ErrorAction)
    $script:Downloads++
    if ($Uri -ne 'https://api.nuget.org/v3-flatcontainer/python/3.13.17/python.3.13.17.nupkg') {{ throw 'Wrong package URL' }}
    Copy-Item -LiteralPath $env:UNSLOTH_TEST_PSF_PACKAGE -Destination $OutFile
}}
{helpers}
$PythonVersion='3.13'; $PythonPatchRequest=''; $PythonRequestLabel='3.13'; $PythonArch='x86_64'
{RESOLVE_SITE}
if ($DetectedPython.Source -ne 'signed-managed') {{ throw 'Signed managed runtime was not selected' }}
$env:UV_OFFLINE='1'
$again = Resolve-SignedManagedPython -Minor '3.13'
if ($again.Path -ne $DetectedPython.Path) {{ throw 'Cached runtime not reused offline' }}
$VenvDir=Join-Path $StudioHome 'unsloth_studio'
$VenvPython=Join-Path $VenvDir 'Scripts\\python.exe'
{create_venv}
$signature=Get-AuthenticodeSignature -LiteralPath $VenvPython
$facts=& $VenvPython -I -c "import sys,ssl,sqlite3,ctypes; print(sys.version.split()[0], int(sys.prefix != sys.base_prefix))"
if ($LASTEXITCODE -ne 0) {{ throw 'venv launch failed' }}
@{{ Status="$($signature.Status)"; Facts="$facts"; Path=$DetectedPython.Path; Downloads=$script:Downloads; Stages=@(Get-ChildItem (Join-Path $StudioHome 'python-signed') -Filter '.stage-*').Count; Legacy=(Test-LegacyVenvReusable -VenvDir $VenvDir -Python $DetectedPython) }} | ConvertTo-Json -Compress
"""
    home = tmp_path / "Unsloth home with spaces"
    r = json.loads(
        _pwsh(
            shell,
            script,
            {
                "TEST_HOME": str(home),
                "BLOCKED_UV": str(uv),
                "UV_OFFLINE": "",
                "UV_PYTHON_DOWNLOADS": "",
            },
        )
    )
    assert r["Status"] == "Valid" and r["Facts"] == "3.13.17 1"
    assert Path(r["Path"]).is_relative_to(home / "python-signed")
    assert r["Downloads"] == 1 and r["Stages"] == 0 and not r["Legacy"]
