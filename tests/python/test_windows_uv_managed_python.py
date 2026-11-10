# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Windows prefers uv-managed CPython and preserves the requested version and architecture.
A system 3.12 must never silently replace a requested 3.13 (the cu130 torch route).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
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
        if ($env:FAKE_UV_STDERR) { & $env:PROBE_PYTHON -c 'import os, sys; print(os.environ["FAKE_UV_STDERR"], file=sys.stderr)' }
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
        exe.write_text(
            '@type "%FAKE_PY_FACTS_FILE%"\r\n'
            "@if defined FAKE_PY_STDERR echo %FAKE_PY_STDERR% 1>&2\r\n",
            encoding = "utf-8",
        )
    else:
        exe = tmp_path / "python"
        exe.write_text(
            '#!/bin/sh\ncat "$FAKE_PY_FACTS_FILE"\n'
            'if [ -n "$FAKE_PY_STDERR" ]; then printf "%s\\n" "$FAKE_PY_STDERR" >&2; fi\n',
            encoding = "utf-8",
        )
        exe.chmod(0o755)
    return exe


def _pwsh(
    shell: str,
    script: str,
    env: dict[str, str | None],
    *,
    timeout: int = 60,
) -> str:
    module_setup = _between(
        "    if ($PSVersionTable.PSEdition -ne 'Core' -and $env:SystemRoot) {",
        "    # Same UTF-8 invariant",
    )
    result = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", module_setup + script],
        check = False,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        env = {key: value for key, value in dict(os.environ, **env).items() if value is not None},
        timeout = timeout,
    )
    if result.stderr:
        print(result.stderr)
    if result.returncode:
        print(result.stdout)
    result.check_returncode()
    return result.stdout.strip()


def _resolve(
    shell: str,
    tmp_path: Path,
    *,
    facts: dict[str, str],
    arch: str = "x86_64",
    patch: str = "",
    skip: list[str] | None = None,
    install_exit: int = 0,
    stderr: str = "",
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
            "FAKE_UV_STDERR": stderr,
            "FAKE_PY_STDERR": stderr,
            "PROBE_PYTHON": sys.executable,
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
function Resolve-SignedManagedPython {{ param($Minor, $Patch, $Arch, [switch]$CacheOnly) return $null }}
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
        (
            False,
            False,
            True,
            "signed-managed",
            1,
        ),  # reuse a validated cache after uv's runtime was removed
        (False, False, False, "failed", 1),
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
    param($Minor, $Patch, $Arch, [switch]$CacheOnly)
    if ([bool]$CacheOnly -ne (-not ${str(blocked).lower()})) {{ throw 'Downloads require policy evidence' }}
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
@pytest.mark.parametrize("hidden", [False, True])
def test_signed_runtime_checks_libraries_and_launchers_before_execution(
    shell, tmp_path, bad_file, facts, success, expected, probed, hidden
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
if (${str(hidden).lower()}) {{
    foreach ($file in Get-ChildItem -LiteralPath $env:RUNTIME -Recurse -File) {{
        $file.Attributes = $file.Attributes -bor [IO.FileAttributes]::Hidden -bor [IO.FileAttributes]::System
    }}
}}
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
    @{{ Success = ${str(success).lower()}; Output = @('{facts}'); Diagnostic = '{facts}' }}
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
function Test-SignedPythonDownloadAllowed {{ return (-not (Test-UvEnvFlag 'UV_OFFLINE') -and $env:UV_PYTHON_DOWNLOADS -ne 'never') }}
{_function('Test-UvEnvFlag')}
{_function('Resolve-SignedManagedPython')}
$r = Resolve-SignedManagedPython -Minor '3.13'
@{{ Accepted=[bool]$r; Downloads=$script:Downloads; Extracts=$script:Extracts; Stages=@(Get-ChildItem (Join-Path $StudioHome 'python-signed') -Filter '.stage-*' -Force -ErrorAction SilentlyContinue).Count }} | ConvertTo-Json -Compress
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
@pytest.mark.skipif(os.name != "nt", reason = "requires Windows runtime")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("arch", ["x86_64", "arm64"])
def test_policy_block_installs_private_signed_runtime_and_reuses_it_offline(shell, tmp_path, arch):
    # Only transport and the original policy refusal are simulated. The archive hash,
    # signatures, imports, repair, cache reuse, and venv creation are real.
    package_env = (
        "UNSLOTH_TEST_PSF_ARM64_PACKAGE" if arch == "arm64" else "UNSLOTH_TEST_PSF_PACKAGE"
    )
    if not os.environ.get(package_env):
        pytest.skip(f"requires {arch} official Python package")
    package_name = "pythonarm64" if arch == "arm64" else "python"
    uv = tmp_path / "blocked-uv.cmd"
    uv.write_text(
        '@if "%~3"=="--show-settings" goto settings\n'
        "@echo uv trampoline failed to spawn Python child process: os error 4551 1>&2\n@exit /b 1\n"
        ":settings\n@echo GlobalSettings {\n@echo     connectivity: Online,\n"
        "@echo     python_downloads: Automatic,\n@echo }\n@exit /b 0\n",
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
            "Test-SignedPythonDownloadAllowed",
            "Resolve-SignedManagedPython",
            "Test-UvEnvFlag",
            "Test-Arm64PythonOptOut",
            "Test-LegacyVenvReusable",
        )
    )
    script = f"""
$ErrorActionPreference = 'Stop'
$PythonSkip = @('3.13.8'); $StudioHome = $env:TEST_HOME
$script:UvExe = $env:BLOCKED_UV; $script:Downloads=0
function Write-StudioLine {{ param($Message,$ForegroundColor) }}
function substep {{ param($Name,$Message) [Console]::Error.WriteLine("$Name $Message") }}
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
    if ($Uri -ne 'https://api.nuget.org/v3-flatcontainer/{package_name}/3.13.17/{package_name}.3.13.17.nupkg') {{ throw 'Wrong package URL' }}
    Copy-Item -LiteralPath $env:TEST_PACKAGE -Destination $OutFile
}}
{helpers}
$PythonVersion='3.13'; $PythonPatchRequest=''; $PythonRequestLabel='3.13'; $PythonArch='{arch}'
{RESOLVE_SITE}
if ($DetectedPython.Source -ne 'signed-managed') {{ throw 'Signed managed runtime was not selected' }}
$env:UV_OFFLINE='1'
$again = Resolve-SignedManagedPython -Minor '3.13' -Arch '{arch}'
if ($again.Path -ne $DetectedPython.Path) {{ throw 'Cached runtime not reused offline' }}
$VenvDir=Join-Path $StudioHome 'unsloth_studio'
$VenvPython=Join-Path $VenvDir 'Scripts\\python.exe'
{create_venv}
$signature=Get-AuthenticodeSignature -LiteralPath $VenvPython
$facts=& $VenvPython -I -c "import sys,ssl,sqlite3,ctypes; print(sys.version.split()[0], int(sys.prefix != sys.base_prefix))"
if ($LASTEXITCODE -ne 0) {{ throw 'venv launch failed' }}
$initial = $DetectedPython
$oldVenv = $VenvPython
$cfg = Join-Path $VenvDir 'pyvenv.cfg'
$cfgHash = (Get-FileHash -LiteralPath $cfg).Hash
$exeHash = (Get-FileHash -LiteralPath $initial.Path).Hash
$missing = Join-Path (Split-Path $initial.Path -Parent) 'pythonw.exe'
Move-Item -LiteralPath $missing -Destination ($missing + '.saved')
$env:UV_OFFLINE=$null
{RESOLVE_SITE}
if ($DetectedPython.Path -eq $initial.Path) {{ throw 'Damaged runtime was not replaced' }}
if ((Get-FileHash -LiteralPath $cfg).Hash -ne $cfgHash -or
    (Get-FileHash -LiteralPath $initial.Path).Hash -ne $exeHash -or
    (Test-Path -LiteralPath $missing) -or -not (Test-Path -LiteralPath ($missing + '.saved'))) {{ throw 'Repair changed the old environment or base' }}
$VenvDir=Join-Path $StudioHome 'repaired environment'
$VenvPython=Join-Path $VenvDir 'Scripts\\python.exe'
{create_venv}
$tag = if ('{arch}' -eq 'arm64') {{ 'win-arm64' }} else {{ 'win-amd64' }}
foreach ($pair in @(@{{ Python=$oldVenv; Base=$initial.Path }}, @{{ Python=$VenvPython; Base=$DetectedPython.Path }})) {{
    if ((Get-AuthenticodeSignature -LiteralPath $pair.Python).Status -ne 'Valid') {{ throw 'Venv launcher signature invalid' }}
    $probe = Invoke-PythonSelectionProbe -Executable $pair.Python -Arguments @('-I', '-c',
        'import json,sys,sysconfig,ssl,sqlite3,ctypes; print(json.dumps(dict(version=sys.version.split()[0],tag=sysconfig.get_platform(),base=sys.base_prefix,venv=sys.prefix!=sys.base_prefix)))')
    if (-not $probe.Success) {{ throw $probe.Diagnostic }}
    $v = ($probe.Output -join "`n") | ConvertFrom-Json
    if ($v.version -ne '3.13.17' -or $v.tag -ne $tag -or -not $v.venv -or
        [IO.Path]::GetFullPath($v.base) -ne (Split-Path $pair.Base -Parent)) {{ throw 'Venv does not use its original signed base' }}
}}
$env:UV_OFFLINE='1'
$cached = Resolve-SignedManagedPython -Minor '3.13' -Arch '{arch}' -CacheOnly
if ($cached.Path -ne $DetectedPython.Path) {{ throw 'Repaired runtime was not reused offline' }}
@{{ Status="$($signature.Status)"; Facts="$facts"; Path=$DetectedPython.Path; Downloads=$script:Downloads; Stages=@(Get-ChildItem (Join-Path $StudioHome 'python-signed') -Filter '.stage-*' -Force).Count; Legacy=(Test-LegacyVenvReusable -VenvDir $VenvDir -Python $DetectedPython) }} | ConvertTo-Json -Compress
"""
    home = tmp_path / "Unsloth home with spaces"
    r = json.loads(
        _pwsh(
            shell,
            script,
            {
                "TEST_HOME": str(home),
                "TEST_PACKAGE": os.environ[package_env],
                "BLOCKED_UV": str(uv),
                "UV_OFFLINE": "",
                "UV_PYTHON_DOWNLOADS": "",
            },
            timeout = 180,
        )
    )
    assert r["Status"] == "Valid" and r["Facts"] == "3.13.17 1"
    assert Path(r["Path"]).is_relative_to(home / "python-signed")
    assert r["Downloads"] == 2 and r["Stages"] == 0 and not r["Legacy"]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_discovery_and_version_facts_ignore_trailing_native_stderr(shell, tmp_path):
    result, _, fake_py = _resolve(
        shell,
        tmp_path,
        facts = {X64: "3.13.17 win-amd64 0"},
        stderr = "TRACE Released lock at the uv cache",
    )
    assert result["Path"] == fake_py
    assert result["Full"] == "3.13.17"


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize("stderr_first", [False, True])
def test_probe_keeps_stdout_separate_without_losing_diagnostics(shell, stderr_first):
    writes = [
        "print('the interpreter path', flush=True)",
        "print('diagnostic', file=sys.stderr, flush=True)",
    ]
    if stderr_first:
        writes.reverse()
    code = "import sys, time; " + "; time.sleep(0.05); ".join(writes)
    script = f"""
{_function('Test-ApplicationControlBlock')}
{_function('Invoke-PythonSelectionProbe')}
$r = Invoke-PythonSelectionProbe -Executable $env:PROBE_PYTHON -Arguments @('-c', $env:PROBE_CODE)
$r | ConvertTo-Json -Compress
"""
    r = json.loads(_pwsh(shell, script, {"PROBE_PYTHON": sys.executable, "PROBE_CODE": code}))
    assert r["Success"] and not r["Blocked"]
    assert r["Output"] == ["the interpreter path"]
    assert "diagnostic" in r["Diagnostic"]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "arm_blocked, x64_blocked", [(False, False), (True, False), (False, True), (True, True)]
)
def test_policy_evidence_does_not_leak_between_architectures(shell, arm_blocked, x64_blocked):
    script = f"""
$ErrorActionPreference = 'Stop'
$PythonSkip = @()
$script:Selections = @()
function Invoke-InstallCommand {{ param($Command, $Label) return 1 }}
function Write-StudioLine {{ param($Message, $ForegroundColor) }}
function substep {{ param($Message, $Color) }}
function step {{ param($Name, $Message) }}
function Exit-InstallFailure {{ param($Message) return $null }}
function Test-Arm64PythonOptOut {{ return $false }}
function Invoke-PythonSelectionProbe {{
    param($Executable, $Arguments)
    $blocked = if ($Arguments[-1] -match 'aarch64') {{ ${str(arm_blocked).lower()} }} else {{ ${str(x64_blocked).lower()} }}
    return @{{ Success=$false; Blocked=$blocked; Diagnostic='' }}
}}
function Resolve-SignedManagedPython {{
    param($Minor, $Patch, $Arch, [switch]$CacheOnly)
    $script:Selections += @{{ Arch=$Arch; CacheOnly=[bool]$CacheOnly }}
    return $null
}}
{_function('Resolve-UvManagedPython')}
function Run {{
    $PythonVersion='3.13'; $PythonPatchRequest=''; $PythonArch='arm64'; $PythonRequestLabel='3.13'
{RESOLVE_SITE}
}}
$null = Run
ConvertTo-Json -InputObject $script:Selections -Compress
"""
    assert json.loads(_pwsh(shell, script, {})) == [
        {"Arch": "arm64", "CacheOnly": not arm_blocked},
        {"Arch": "x86_64", "CacheOnly": not x64_blocked},
    ]


SETTINGS = "GlobalSettings {\n    connectivity: Online,\n    python_downloads: Automatic,\n}"


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "output, diagnostic, success, allowed",
    [
        (SETTINGS, "", True, True),
        (SETTINGS.replace("Automatic", "Manual"), "", True, True),
        (SETTINGS.replace("Automatic", "Never"), "", True, False),
        (SETTINGS.replace("Online", "Offline"), "", True, False),
        (SETTINGS, "TRACE Released lock at cache", True, True),
        (
            SETTINGS,
            "warning: Failed to parse `pyproject.toml` during settings discovery:",
            True,
            False,
        ),
        (SETTINGS, "error: invalid settings", False, False),
        ("", "", True, False),
        (SETTINGS + "\n" + SETTINGS, "", True, False),
        (
            SETTINGS.replace("    connectivity:", "    connectivity: Offline,\n    connectivity:"),
            "",
            True,
            False,
        ),
        (
            SETTINGS.replace(
                "    python_downloads:", "    python_downloads: Unknown,\n    python_downloads:"
            ),
            "",
            True,
            False,
        ),
        (SETTINGS.replace("Automatic", "Unknown"), "", True, False),
        (SETTINGS.replace("    python_downloads: Automatic,\n", ""), "", True, False),
        (SETTINGS.replace("\n", "\r\n"), "", True, True),
    ],
)
def test_download_policy_requires_recognized_effective_settings(
    shell, output, diagnostic, success, allowed
):
    script = f"""
function substep {{ param($Message, $Color) }}
function Invoke-PythonSelectionProbe {{
    param($Executable, $Arguments)
    if (($Arguments -join ' ') -ne 'python install --show-settings --color never') {{ throw 'Wrong settings query' }}
    @{{ Success = ${str(success).lower()}; Output = @($env:SETTINGS); Diagnostic = $env:DIAGNOSTIC }}
}}
{_function('Test-SignedPythonDownloadAllowed')}
Test-SignedPythonDownloadAllowed | ConvertTo-Json -Compress
"""
    assert (
        json.loads(_pwsh(shell, script, {"SETTINGS": output, "DIAGNOSTIC": diagnostic})) is allowed
    )


@needs_pwsh
@pytest.mark.skipif(not shutil.which("uv"), reason = "requires uv to resolve real configuration")
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "config, env, allowed",
    [
        ("", {}, True),
        ('offline = true\npython-downloads = "never"\n', {}, False),
        ("offline = true\n", {}, False),
        ('python-downloads = "never"\n', {}, False),
        ('python-downloads = "manual"\n', {}, True),
        ('python-downloads = "never"\n', {"UV_PYTHON_DOWNLOADS": "automatic"}, True),
        ('python-downloads = "automatic"\n', {"UV_PYTHON_DOWNLOADS": "never"}, False),
        ("offline = false\n", {"UV_OFFLINE": "true"}, False),
        ("offline = not-valid-toml\n", {}, False),
        ("", {"RUST_LOG": "trace"}, True),
    ],
)
@pytest.mark.parametrize("location", ["explicit", "project", "pyproject", "user"])
def test_download_policy_uses_real_uv_configuration(
    shell, tmp_path, config, env, allowed, location
):
    project = tmp_path / "project"
    user = tmp_path / "config" / "uv"
    project.mkdir()
    user.mkdir(parents = True)
    config_path = (
        user / "uv.toml"
        if location == "user"
        else project / ("pyproject.toml" if location == "pyproject" else "uv.toml")
    )
    config_path.write_text(
        ("[tool.uv]\n" if location == "pyproject" else "") + config, encoding = "utf-8"
    )
    script = f"""
$script:UvExe = $env:TEST_UV
function substep {{ param($Message, $Color) }}
{_function('Test-ApplicationControlBlock')}
{_function('Invoke-PythonSelectionProbe')}
{_function('Test-SignedPythonDownloadAllowed')}
Test-SignedPythonDownloadAllowed | ConvertTo-Json -Compress
"""
    isolated = {
        "TEST_UV": shutil.which("uv"),
        "UV_WORKING_DIR": str(project),
        "UV_WORKING_DIRECTORY": None,
        "UV_PROJECT": None,
        "UV_CONFIG_FILE": str(config_path) if location == "explicit" else None,
        "UV_NO_CONFIG": None,
        "UV_NO_SYSTEM_CONFIG": "1",
        "UV_OFFLINE": None,
        "UV_PYTHON_DOWNLOADS": None,
        "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(tmp_path / "python"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "APPDATA": str(tmp_path / "config"),
        "RUST_LOG": "",
        **env,
    }
    assert json.loads(_pwsh(shell, script, isolated)) is allowed
    assert not (tmp_path / "python").exists()


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
@pytest.mark.parametrize(
    "initial, blocked, offline, failure, downloads, accepted",
    [
        ("missing", True, False, "", 1, True),
        ("missing", False, False, "", 0, False),
        ("missing", True, True, "", 0, False),
        ("valid", False, True, "", 0, True),
        ("damaged", True, False, "", 1, True),
        ("unreadable", True, False, "", 1, True),
        ("file", True, False, "", 1, True),
        ("damaged", False, False, "", 0, False),
        ("damaged", True, True, "", 0, False),
        ("damaged", True, False, "download", 1, False),
        ("damaged", True, False, "checksum", 1, False),
        ("damaged", True, False, "signature", 1, False),
        ("missing", True, False, "move-access", 1, True),
        ("damaged", True, False, "move-access", 1, True),
        ("damaged", True, False, "move-sharing", 1, True),
        ("damaged", True, False, "move-lock", 1, True),
        ("damaged", True, False, "move-unauthorized", 1, True),
        ("missing", True, False, "move-persistent", 1, False),
        ("damaged", True, False, "move-persistent", 1, False),
        ("damaged", True, False, "move-other", 1, False),
        ("damaged", True, False, "move-collision", 1, False),
        ("damaged", True, False, "final-signature", 1, False),
    ],
)
def test_signed_runtime_repair_preserves_old_bases_and_reuses_replacements(
    shell, tmp_path, initial, blocked, offline, failure, downloads, accepted
):
    archive = tmp_path / "package.zip"
    with zipfile.ZipFile(archive, "w") as package:
        package.writestr("tools/python.exe", "new runtime")
        package.writestr("tools/valid.txt", "validated runtime")
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    name = f"cpython-3.13.17-x86_64-{checksum[:12]}"
    home = tmp_path / "Studio home"
    root = home / "python-signed" / name
    old = home / "unsloth_studio" / "pyvenv.cfg"
    old.parent.mkdir(parents = True)
    old.write_text(f"home = {root}\nversion = 3.13.17\n", encoding = "utf-8")
    before = old.read_bytes()
    if initial == "file":
        root.parent.mkdir(parents = True)
        root.write_text("occupied path")
    elif initial != "missing":
        root.mkdir(parents = True)
        (root / "python.exe").write_text("old runtime")
        if initial == "valid":
            (root / "valid.txt").touch()
        if initial == "unreadable":
            (root / "unreadable.txt").touch()
    resolver = _function("Resolve-SignedManagedPython").replace(
        "[System.IO.Directory]::Move($staged, $destination)",
        "Move-TestRuntime -From $staged -To $destination",
    )
    script = f"""
$ErrorActionPreference = 'Stop'
$StudioHome = $env:TEST_HOME
$script:Downloads = 0
$script:Moves = 0
$script:Waits = @()
function Start-Sleep {{ param($Milliseconds) $script:Waits += $Milliseconds }}
function Move-TestRuntime {{
    param($From, $To)
    $script:Moves++
    if ('{failure}' -eq 'move-collision') {{
        [IO.Directory]::CreateDirectory($To) | Out-Null
        [IO.File]::WriteAllText((Join-Path $To 'foreign.txt'), 'another publisher')
    }}
    if ('{failure}' -eq 'move-persistent' -or $script:Moves -le 2) {{
        if ('{failure}' -eq 'move-unauthorized') {{ throw [UnauthorizedAccessException]::new('test refusal') }}
        $code = switch ('{failure}') {{
            'move-access' {{ 5 }} 'move-sharing' {{ 32 }} 'move-lock' {{ 33 }}
            'move-persistent' {{ 5 }} 'move-other' {{ 80 }} default {{ 0 }}
        }}
        if ($code) {{ throw [IO.IOException]::new('test publication failure', (-2147024896 + $code)) }}
    }}
    [IO.Directory]::Move($From, $To)
}}
$script:Blocked = ${str(blocked).lower()}
function substep {{ param($Message, $Color) [Console]::Error.WriteLine($Message) }}
function step {{ param($Name, $Message) }}
function Write-StudioLine {{ param($Message, $ForegroundColor) }}
function Redact-InstallOutput {{ param($Text) return $Text }}
function Exit-InstallFailure {{ param($Message) return $null }}
function Resolve-UvManagedPython {{ param($Minor, $Patch, $Arch) $script:PythonPolicyBlocked = $script:Blocked; return $null }}
function Get-SignedPythonPackage {{ param($Minor, $Patch, $Arch) @{{ Version='3.13.17'; Minor='3.13'; Arch='x86_64'; Sha256='{checksum}'; Url='https://example.invalid/python.nupkg' }} }}
function Test-SignedPythonDownloadAllowed {{ return (-not (Test-UvEnvFlag 'UV_OFFLINE')) }}
function Invoke-WebRequest {{
    param([switch]$UseBasicParsing, $Uri, $OutFile, $TimeoutSec, $ErrorAction)
    $script:Downloads++
    if ('{failure}' -eq 'download') {{ throw 'network unavailable' }}
    if ('{failure}' -eq 'checksum') {{ [IO.File]::WriteAllText($OutFile, 'wrong bytes'); return }}
    Copy-Item -LiteralPath $env:PACKAGE -Destination $OutFile
}}
function Test-SignedPythonRuntime {{
    param($Root, $Package)
    if (Test-Path -LiteralPath (Join-Path $Root 'unreadable.txt')) {{ throw 'unreadable runtime' }}
    if ('{failure}' -eq 'signature' -or -not (Test-Path -LiteralPath (Join-Path $Root 'valid.txt'))) {{ return $null }}
    if ('{failure}' -eq 'final-signature' -and $Root -notmatch '\\.stage-') {{ return $null }}
    return @{{ Version='3.13'; Full='3.13.17'; Arch='x86_64'; Source='signed-managed'; Path=(Join-Path $Root 'python.exe') }}
}}
{_function('Test-UvEnvFlag')}
{_function('Test-Arm64PythonOptOut')}
{resolver}
function Run {{
    $PythonVersion='3.13'; $PythonPatchRequest=''; $PythonArch='x86_64'; $PythonRequestLabel='3.13'
{RESOLVE_SITE}
    return $DetectedPython
}}
$result = Run
$again = $null
if ($result) {{
    $env:UV_OFFLINE='1'
    $script:Blocked=$false
    $again = Run
}}
@{{ Accepted=[bool]$result; Path=$result.Path; Reused=([bool]$again -and $again.Path -eq $result.Path); Downloads=$script:Downloads; Moves=$script:Moves; Waits=@($script:Waits); Stages=@(Get-ChildItem -LiteralPath (Join-Path $StudioHome 'python-signed') -Filter '.stage-*' -Force -ErrorAction SilentlyContinue).Count }} | ConvertTo-Json -Compress
"""
    r = json.loads(
        _pwsh(
            shell,
            script,
            {
                "TEST_HOME": str(home),
                "PACKAGE": str(archive),
                "UV_OFFLINE": "1" if offline else "",
                "UNSLOTH_ALLOW_ARM64_PYTHON": "",
            },
        )
    )
    assert r["Accepted"] is accepted
    assert r["Downloads"] == downloads
    assert r["Stages"] == 0
    assert r["Reused"] is accepted
    if failure == "move-persistent":
        assert r["Moves"] == 4 and r["Waits"] == [500, 1000, 2000]
    elif failure in ("move-other", "move-collision"):
        assert r["Moves"] == 1 and r["Waits"] == []
    elif failure.startswith("move-"):
        assert r["Moves"] == 3 and r["Waits"] == [500, 1000]
    else:
        assert r["Waits"] == []
    if failure == "move-collision":
        foreign = list((home / "python-signed").glob("*/foreign.txt"))
        assert len(foreign) == 1 and foreign[0].read_text() == "another publisher"
    assert old.read_bytes() == before
    if initial == "file":
        assert root.read_text() == "occupied path"
    elif initial != "missing":
        assert (root / "python.exe").read_text() == "old runtime"
    if accepted and initial in ("damaged", "unreadable", "file"):
        assert Path(r["Path"]).parent != root
        assert re.fullmatch(re.escape(name) + r"-[0-9a-f]{32}", Path(r["Path"]).parent.name)
