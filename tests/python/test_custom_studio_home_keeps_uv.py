# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A custom UNSLOTH_STUDIO_HOME keeps uv, its cache and the uv-managed Python inside the home
(#11327), so a home on another drive leaves nothing large on the profile drive."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = (REPO_ROOT / "install.ps1").read_text(encoding = "utf-8")
INSTALL_SH = (REPO_ROOT / "install.sh").read_text(encoding = "utf-8")
POWERSHELLS = [shell for shell in ("pwsh", "powershell") if shutil.which(shell)]
needs_pwsh = pytest.mark.skipif(not POWERSHELLS, reason = "PowerShell is unavailable")
needs_sh = pytest.mark.skipif(
    sys.platform == "win32" or not shutil.which("sh"), reason = "POSIX sh is unavailable"
)


def _between(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def _uv_stub(directory: Path) -> None:
    name = "uv.exe" if os.name == "nt" else "uv"
    directory.mkdir(parents = True, exist_ok = True)
    (directory / name).write_text("#!/bin/sh\n")
    (directory / name).chmod(0o755)


# ── install.ps1 ──


def _ps1_private_uv(
    shell,
    home: Path,
    *,
    legacy = False,
    profile_default = False,
    private_ok = True,
    env = None,
):
    block = _between(
        INSTALL_PS1,
        "    $script:StudioKeepsTools = ",
        "    if ($script:StudioPrivateUvStale -or -not (Test-UvVersionOk)) {",
    )
    script = f"""
$ErrorActionPreference = "Stop"
$StudioHome = '{home}'
$studioUsesLegacyLayout = ${str(legacy).lower()}
function Test-StudioHomeIsCustom {{ return ${str(not profile_default).lower()} }}
function Write-StudioRootOwnerMarker {{ param($Root) $script:claimedBeforePython = -not (Test-Path (Join-Path $Root "python")) }}
function Test-UvCandidateVersion {{ param($exe) return ${str(private_ok).lower()} }}
$script:claimedBeforePython = $null
{block}
"claimed=$($script:claimedBeforePython)"
"stale=$($script:StudioPrivateUvStale)"
"keeps=$($script:StudioKeepsTools)"
"dir=$($script:StudioPrivateUvDir)"
"path=$env:PATH"
"""
    child_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL", "XDG_BIN_HOME")
    }
    child_env.update(env or {})
    out = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        env = child_env,
        timeout = 60,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_ps1_custom_home_installs_uv_into_its_own_dir(shell, tmp_path):
    bin_dir = str(tmp_path / "uv")
    found = _ps1_private_uv(shell, tmp_path)
    assert found["keeps"] == "True"
    assert Path(found["dir"]) == Path(bin_dir)
    # Nothing there yet, so PATH is untouched until a uv lands in it.
    assert bin_dir not in found["path"]

    # The root is claimed before python\ is created in it.
    assert found["claimed"] == "True"
    assert (tmp_path / "python").is_dir()

    (tmp_path / "uv").mkdir()
    (tmp_path / "uv" / "uv.exe").write_bytes(b"")
    found = _ps1_private_uv(shell, tmp_path)
    assert (found["stale"], found["path"].startswith(bin_dir)) == ("False", True)
    # A stale private uv is not put first; the installer replaces it instead.
    found = _ps1_private_uv(shell, tmp_path, private_ok = False)
    assert (found["stale"], found["path"].startswith(bin_dir)) == ("True", False)


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_ps1_default_root_and_explicit_uv_dir_keep_the_old_layout(shell, tmp_path):
    for found in (
        _ps1_private_uv(shell, tmp_path, legacy = True),
        # UNSLOTH_STUDIO_HOME naming the current USERPROFILE default is still the default.
        _ps1_private_uv(shell, tmp_path, profile_default = True),
    ):
        assert (found["keeps"], found["dir"]) == ("False", "")
    explicit = _ps1_private_uv(shell, tmp_path, env = {"UV_INSTALL_DIR": str(tmp_path / "mine")})
    assert (explicit["keeps"], explicit["dir"]) == ("True", "")


def test_ps1_private_uv_skips_winget_and_the_user_path():
    install = _between(INSTALL_PS1, "    function Install-UvFromRelease {", "\n    }\n")
    assert "Get-Variable -Name StudioPrivateUvDir -Scope Script" in install
    assert "$destDir = $privateDir" in install
    assert (
        "-not $env:UV_UNMANAGED_INSTALL -and -not $privateDir) {\n"
        "            Add-ToUserPath" in install
    )
    assert "if ($script:WingetAvailable -and -not $script:StudioPrivateUvDir) {" in INSTALL_PS1
    # Both guards on the way to Install-UvFromRelease honour the stale flag, not only the outer one.
    block = _between(
        INSTALL_PS1,
        "    if ($script:StudioPrivateUvStale -or -not (Test-UvVersionOk)) {",
        "    # A freshly installed uv can sit behind an older one on PATH",
    )
    assert block.count("$script:StudioPrivateUvStale -or -not (Test-UvVersionOk)") == 2


def test_ps1_custom_home_isolates_a_usable_cache_and_the_python_store():
    configured = INSTALL_PS1.index(
        "Set-StudioUvCacheEnvironment -StudioRoot $StudioHome -Isolated $IsolateUvCache"
    )
    isolate = INSTALL_PS1.rindex("if ($script:StudioKeepsTools -and", 0, configured)
    # An unusable Studio cache falls through to the selector, as it does without the override.
    assert "Test-StudioUvCacheUsable -Cache" in INSTALL_PS1[isolate:configured]
    python = _between(INSTALL_PS1, "    if ($PythonFromUv) {\n", "Resolve-UvManagedPython\n")
    assert "$script:StudioKeepsTools" in python
    assert '$env:UV_PYTHON_INSTALL_DIR = Join-Path $StudioHome "python"' in python
    # Restored for `irm | iex`, where the session is the user's own.
    restore = _between(
        INSTALL_PS1,
        "    } finally {\n        Restore-StudioUvCacheEnvironment",
        "Restore-StudioTempEnvironment",
    )
    assert "Env:UV_PYTHON_INSTALL_DIR" in restore


# ── install.sh ──


def _sh(
    home: Path,
    studio_home: Path,
    *,
    redirect = "env",
    env = None,
):
    keeps = _between(INSTALL_SH, "_STUDIO_KEEPS_TOOLS=false\n", "# The PATH we inherited")
    private = _between(
        INSTALL_SH,
        '_STUDIO_UV_DIR=""\n',
        'if ! command -v uv >/dev/null 2>&1 || ! _uv_version_ok uv || [ "$_STUDIO_UV_STALE"',
    )
    script = f"""
set -e
_uv_version_ok() {{ "$1" --version >/dev/null 2>&1; }}
_STUDIO_HOME_REDIRECT={redirect}
STUDIO_HOME='{studio_home}'
{keeps}
{private}
echo "keeps=$_STUDIO_KEEPS_TOOLS"
echo "uvdir=$_STUDIO_UV_DIR"
echo "stale=$_STUDIO_UV_STALE"
echo "pydir=${{UV_PYTHON_INSTALL_DIR:-}}"
echo "path=$PATH"
"""
    child_env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    child_env.update(env or {})
    out = subprocess.run(
        ["sh", "-c", script], check = True, capture_output = True, text = True, env = child_env
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@needs_sh
def test_sh_custom_home_keeps_uv_and_python(tmp_path):
    studio = tmp_path / "drive" / "studio"
    studio.mkdir(parents = True)
    found = _sh(tmp_path / "home", studio)
    assert (found["keeps"], found["uvdir"]) == ("true", f"{studio}/uv")
    assert found["pydir"] == f"{studio}/python"
    assert not found["path"].startswith(f"{studio}/uv")

    _uv_stub(studio / "uv")
    found = _sh(tmp_path / "home", studio)
    assert (found["stale"], found["path"].startswith(f"{studio}/uv:")) == ("false", True)
    # A stale private uv is not put first; the installer replaces it instead.
    (studio / "uv" / "uv").write_text("#!/bin/sh\nexit 1\n")
    found = _sh(tmp_path / "home", studio)
    assert (found["stale"], found["path"].startswith(f"{studio}/uv:")) == ("true", False)


def test_sh_python_marker_is_created_only_after_the_root_claim():
    claim = INSTALL_SH.index("\n_claim_studio_root\n")
    marker = INSTALL_SH.index('then mkdir -p "$STUDIO_HOME/python"')
    assert claim < marker
    assert INSTALL_SH.count('mkdir -p "$STUDIO_HOME/python"') == 1


@needs_sh
def test_sh_explicit_settings_and_the_default_root_win(tmp_path):
    home = tmp_path / "home"
    legacy = home / ".unsloth" / "studio"
    legacy.mkdir(parents = True)
    for found in (_sh(home, legacy), _sh(home, legacy, redirect = "default")):
        assert (found["keeps"], found["uvdir"], found["pydir"]) == ("false", "", "")

    studio = tmp_path / "studio"
    studio.mkdir()
    found = _sh(
        home,
        studio,
        env = {"UV_PYTHON_INSTALL_DIR": "/mine/python", "UV_INSTALL_DIR": "/mine/bin"},
    )
    assert (found["keeps"], found["uvdir"], found["pydir"]) == ("true", "", "/mine/python")


def test_sh_custom_home_isolates_a_usable_cache_and_unmanaged_fallback():
    cache = _between(INSTALL_SH, "_configure_uv_cache() {", "\n}\n")
    assert (
        'if [ "$_ISOLATE_UV_CACHE" = true ] || { [ "$_STUDIO_KEEPS_TOOLS" = true ] '
        '&& _uv_cache_is_writable "$_uv_studio_cache"; }; then' in cache
    )
    pinned = _between(INSTALL_SH, "_uv_install_pinned() {", "\n}\n")
    assert 'for _uip_candidate in "${_STUDIO_UV_DIR:-}" "${UV_INSTALL_DIR:-}"' in pinned
    assert (
        '(export UV_UNMANAGED_INSTALL="$_STUDIO_UV_DIR"; run_maybe_quiet sh "$_uv_tmp"'
        in INSTALL_SH
    )


# ── backend and CLI ──


def test_backend_points_on_demand_installs_at_the_home(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT / "studio" / "backend"))
    try:
        from utils.paths import storage_roots
    finally:
        sys.path.pop(0)

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    mine = tmp_path / "mine"
    _uv_stub(mine)
    monkeypatch.setenv("PATH", str(mine))
    monkeypatch.delenv("UV_PYTHON_INSTALL_DIR", raising = False)

    legacy = home / ".unsloth" / "studio"
    _uv_stub(legacy / "uv")
    monkeypatch.setattr(storage_roots, "studio_root", lambda: legacy)
    storage_roots._use_studio_uv_tools()
    assert os.environ["PATH"] == str(mine)
    assert "UV_PYTHON_INSTALL_DIR" not in os.environ

    # A custom root the opt-in installer never touched (UNSLOTH_HOME alone, say) is left alone.
    studio = tmp_path / "drive" / "studio"
    studio.mkdir(parents = True)
    monkeypatch.setattr(storage_roots, "studio_root", lambda: studio)
    storage_roots._use_studio_uv_tools()
    assert os.environ["PATH"] == str(mine)
    assert "UV_PYTHON_INSTALL_DIR" not in os.environ

    (studio / "python").mkdir()
    storage_roots._use_studio_uv_tools()
    assert os.environ["PATH"] == str(mine)
    assert os.environ["UV_PYTHON_INSTALL_DIR"] == str(studio / "python")

    # The installer only places a uv here when PATH had none or an old one, so it goes first,
    # also when it is already further down PATH, and only once.
    _uv_stub(studio / "uv")
    monkeypatch.setenv("PATH", os.pathsep.join([str(mine), str(studio / "uv")]))
    storage_roots._use_studio_uv_tools()
    storage_roots._use_studio_uv_tools()
    assert os.environ["PATH"] == str(studio / "uv") + os.pathsep + str(mine)

    # The engine installer's child keeps the Python store.
    sys.path.insert(0, str(REPO_ROOT / "studio" / "backend"))
    try:
        from core.inference import engine_install
    finally:
        sys.path.pop(0)

    assert engine_install.install_environment()["UV_PYTHON_INSTALL_DIR"] == str(studio / "python")

    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "/mine/python")
    storage_roots._use_studio_uv_tools()
    assert os.environ["UV_PYTHON_INSTALL_DIR"] == "/mine/python"


def test_cli_update_points_setup_at_the_home(tmp_path, monkeypatch):
    # `unsloth studio update` runs setup directly. setup uses a uv on PATH as is, and otherwise
    # version-checks UV_UNMANAGED_INSTALL and installs the pinned uv there, with no profile edit.
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from unsloth_cli.commands import studio as studio_cli
    finally:
        sys.path.pop(0)

    home = tmp_path / "studio"
    home.mkdir()
    monkeypatch.setattr(studio_cli, "STUDIO_HOME", home)
    monkeypatch.setattr(studio_cli, "_STUDIO_HOME_IS_CUSTOM", True)
    env = {"PATH": str(tmp_path / "mine")}

    # Not opted in: a custom root the opt-in installer never touched.
    assert studio_cli._with_studio_uv(env) is env

    (home / "python").mkdir()
    assert studio_cli._with_studio_uv(env) == {
        "PATH": str(tmp_path / "mine"),
        "UV_PYTHON_INSTALL_DIR": str(home / "python"),
        "UV_UNMANAGED_INSTALL": str(home / "uv"),
    }
    assert env == {"PATH": str(tmp_path / "mine")}
    # The caller's own choices win.
    mine = {"PATH": "p", "UV_INSTALL_DIR": "/mine/bin", "UV_PYTHON_INSTALL_DIR": "/mine/py"}
    assert studio_cli._with_studio_uv(mine) == mine

    monkeypatch.setattr(studio_cli, "_STUDIO_HOME_IS_CUSTOM", False)
    assert studio_cli._with_studio_uv(env) is env
