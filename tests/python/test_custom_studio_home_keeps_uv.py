# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A custom UNSLOTH_STUDIO_HOME keeps uv, its cache and the uv-managed Python inside the home
(#11327), so a home on another drive leaves nothing large on the profile drive."""

from __future__ import annotations

import os
import re
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


# ── install.ps1 ──

PS1_PRIVATE_UV = _between(
    INSTALL_PS1, "    $script:StudioPrivateUvDir = $null\n", "    if (-not (Test-UvVersionOk)) {"
)


def _ps1_private_uv(
    shell,
    home: Path,
    *,
    legacy: bool,
    env_extra = None,
):
    script = f"""
$ErrorActionPreference = "Stop"
$StudioHome = '{home}'
$studioUsesLegacyLayout = ${'true' if legacy else 'false'}
{PS1_PRIVATE_UV}
"dir=$($script:StudioPrivateUvDir)"
"path=$env:PATH"
"""
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL", "XDG_BIN_HOME")
    }
    env.update(env_extra or {})
    out = run_pwsh(
        [shell, "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
        encoding = "utf-8",
        env = env,
        timeout = 60,
    ).stdout
    found = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return found["dir"], found["path"]


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_ps1_custom_home_installs_uv_into_its_own_bin(shell, tmp_path):
    bin_dir = str(tmp_path / "bin")
    uv_dir, path = _ps1_private_uv(shell, tmp_path, legacy = False)
    assert uv_dir.replace("/", "\\") == bin_dir.replace("/", "\\") or uv_dir == bin_dir
    # Nothing there yet, so PATH is untouched until a uv lands in it.
    assert bin_dir not in path

    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "uv.exe").write_bytes(b"")
    _, path = _ps1_private_uv(shell, tmp_path, legacy = False)
    assert path.startswith(str(Path(bin_dir)))


@needs_pwsh
@pytest.mark.parametrize("shell", POWERSHELLS)
def test_ps1_default_home_and_explicit_uv_dir_keep_the_old_destination(shell, tmp_path):
    assert _ps1_private_uv(shell, tmp_path, legacy = True)[0] == ""
    explicit = {"UV_INSTALL_DIR": str(tmp_path / "mine")}
    assert _ps1_private_uv(shell, tmp_path, legacy = False, env_extra = explicit)[0] == ""


def test_ps1_private_uv_skips_winget_and_the_user_path():
    install = _between(INSTALL_PS1, "    function Install-UvFromRelease {", "\n    }\n")
    assert "$destDir = $script:StudioPrivateUvDir" in install
    assert (
        "-not $env:UV_UNMANAGED_INSTALL -and -not $script:StudioPrivateUvDir) {\n"
        "            Add-ToUserPath" in install
    )
    assert "if ($script:WingetAvailable -and -not $script:StudioPrivateUvDir) {" in INSTALL_PS1


def test_ps1_custom_home_isolates_the_cache_and_python_store():
    configured = INSTALL_PS1.index(
        "Set-StudioUvCacheEnvironment -StudioRoot $StudioHome -Isolated $IsolateUvCache"
    )
    isolate = INSTALL_PS1.index("if (-not $studioUsesLegacyLayout) { $IsolateUvCache = $true }")
    assert 0 < configured - isolate < 200
    python = _between(INSTALL_PS1, "    if ($PythonFromUv) {\n", "Resolve-UvManagedPython\n")
    assert '$env:UV_PYTHON_INSTALL_DIR = Join-Path $StudioHome "python"' in python
    assert "-not $studioUsesLegacyLayout" in python
    # Restored for `irm | iex`, where the session is the user's own.
    finally_block = _between(
        INSTALL_PS1,
        "    } finally {\n        Restore-StudioUvCacheEnvironment",
        "Restore-StudioTempEnvironment",
    )
    assert "Env:UV_PYTHON_INSTALL_DIR" in finally_block


# ── install.sh ──

SH_KEEPS_TOOLS = _between(INSTALL_SH, "_STUDIO_KEEPS_TOOLS=false\n", "# The PATH we inherited")
SH_PRIVATE_UV = _between(INSTALL_SH, '_STUDIO_UV_DIR=""\n', "if ! command -v uv >/dev/null 2>&1")


def _sh(
    home: Path,
    studio_home: Path,
    *,
    redirect: str = "env",
    env_extra = None,
):
    script = f"""
set -e
_STUDIO_HOME_REDIRECT={redirect}
STUDIO_HOME='{studio_home}'
{SH_KEEPS_TOOLS}
{SH_PRIVATE_UV}
echo "keeps=$_STUDIO_KEEPS_TOOLS"
echo "uvdir=$_STUDIO_UV_DIR"
echo "pydir=${{UV_PYTHON_INSTALL_DIR:-}}"
echo "path=$PATH"
"""
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    env.update(env_extra or {})
    out = subprocess.run(
        ["sh", "-c", script], check = True, capture_output = True, text = True, env = env
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@needs_sh
def test_sh_custom_home_keeps_uv_and_python(tmp_path):
    studio = tmp_path / "drive" / "studio"
    studio.mkdir(parents = True)
    found = _sh(tmp_path / "home", studio)
    assert found["keeps"] == "true"
    assert found["uvdir"] == f"{studio}/bin"
    assert found["pydir"] == f"{studio}/python"
    assert not found["path"].startswith(f"{studio}/bin")

    (studio / "bin").mkdir()
    (studio / "bin" / "uv").write_text("#!/bin/sh\n")
    (studio / "bin" / "uv").chmod(0o755)
    assert _sh(tmp_path / "home", studio)["path"].startswith(f"{studio}/bin:")


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
        env_extra = {"UV_PYTHON_INSTALL_DIR": "/mine/python", "UV_INSTALL_DIR": "/mine/bin"},
    )
    assert (found["keeps"], found["uvdir"], found["pydir"]) == ("true", "", "/mine/python")


def test_sh_custom_home_isolates_the_cache_and_unmanaged_fallback():
    cache = _between(INSTALL_SH, "_configure_uv_cache() {", "\n}\n")
    assert 'if [ "$_ISOLATE_UV_CACHE" = true ] || [ "$_STUDIO_KEEPS_TOOLS" = true ]; then' in cache
    pinned = _between(INSTALL_SH, "_uv_install_pinned() {", "\n}\n")
    assert 'for _uip_candidate in "${_STUDIO_UV_DIR:-}" "${UV_INSTALL_DIR:-}"' in pinned
    assert (
        '(export UV_UNMANAGED_INSTALL="$_STUDIO_UV_DIR"; run_maybe_quiet sh "$_uv_tmp"'
        in INSTALL_SH
    )


# ── backend ──


def test_backend_finds_the_private_uv_only_when_none_is_on_path(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO_ROOT / "studio" / "backend"))
    try:
        from utils.paths import storage_roots
    finally:
        sys.path.pop(0)

    studio = tmp_path / "studio"
    monkeypatch.setattr(storage_roots, "studio_root", lambda: studio)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    name = "uv.exe" if os.name == "nt" else "uv"

    storage_roots._expose_studio_uv()
    assert os.environ["PATH"] == str(empty)

    (studio / "bin").mkdir(parents = True)
    (studio / "bin" / name).write_text("#!/bin/sh\n")
    (studio / "bin" / name).chmod(0o755)
    storage_roots._expose_studio_uv()
    storage_roots._expose_studio_uv()
    assert os.environ["PATH"] == str(empty) + os.pathsep + str(studio / "bin")

    # A uv the user already has wins and PATH is left alone.
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / name).write_text("#!/bin/sh\n")
    (mine / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(mine))
    storage_roots._expose_studio_uv()
    assert os.environ["PATH"] == str(mine)
