# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install.ps1 always builds the Studio environment on a uv-managed CPython, never on a Python
already installed on the machine (a system 3.12 used to win over uv's 3.13 and close the
cu130 torch 2.13 route)."""

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
    "    # Always a uv-managed CPython:",
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
    '    step "python" "using uv-managed Python',
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


def test_no_path_reaches_a_python_already_on_the_machine():
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
