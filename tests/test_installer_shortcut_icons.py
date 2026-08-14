"""Regression tests for installer shortcut icon selection and fallback behavior."""

from __future__ import annotations

import shlex
import shutil
import struct
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
PYPROJECT = REPO_ROOT / "pyproject.toml"
TAURI_ICON_PNG = REPO_ROOT / "studio" / "src-tauri" / "icons" / "icon.png"


def test_linux_tauri_icon_is_1024_square():
    data = TAURI_ICON_PNG.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    assert struct.unpack(">II", data[16:24]) == (1024, 1024)


def test_tauri_icons_are_declared_as_package_data():
    tomllib = pytest.importorskip("tomllib" if sys.version_info >= (3, 11) else "tomli")
    data = tomllib.loads(PYPROJECT.read_text(encoding = "utf-8"))
    package_data = data["tool"]["setuptools"]["package-data"]["studio"]
    assert "src-tauri/icons/icon.icns" in package_data
    assert "src-tauri/icons/icon.png" in package_data


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash is unavailable")
def test_linux_tauri_icon_is_copied_into_installed_data(tmp_path):
    source = INSTALL_SH.read_text(encoding = "utf-8")
    marker = source.index("# Prefer the higher-resolution Tauri icon.png")
    start = source.index('        _css_desktop_icon="$_css_icon_png"', marker)
    end = source.index('        cat > "$_css_desktop"', start)
    block = textwrap.dedent(source[start:end])

    checkout_icon = tmp_path / "checkout" / "icon.png"
    installed_icon = tmp_path / "data" / "unsloth-studio.png"
    checkout_icon.parent.mkdir()
    installed_icon.parent.mkdir()
    checkout_icon.write_text("high-resolution")
    installed_icon.write_text("fallback")
    script = "\n".join(
        [
            f"_css_tauri_png={shlex.quote(str(checkout_icon))}",
            f"_css_icon_png={shlex.quote(str(installed_icon))}",
            block,
            'printf "%s" "$_css_icon_escaped"',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": "/usr/bin:/bin"},
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout == str(installed_icon)
    assert installed_icon.read_text() == "high-resolution"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash is unavailable")
def test_macos_prebuilt_copy_failure_runs_generated_icns_fallback(tmp_path):
    source = INSTALL_SH.read_text(encoding = "utf-8")
    marker = source.index("# ── AppIcon ──")
    start = source.index('        if [ -f "$_css_tauri_icns"', marker)
    end = source.index("        # Touch so Finder indexes it", start)
    block = textwrap.dedent(source[start:end])

    tauri_icns = tmp_path / "source.icns"
    gem_png = tmp_path / "gem.png"
    fallback_png = tmp_path / "fallback.png"
    resources = tmp_path / "Resources"
    fake_bin = tmp_path / "bin"
    resources.mkdir()
    fake_bin.mkdir()
    tauri_icns.write_text("prebuilt")
    gem_png.write_text("gem")
    fallback_png.write_text("raw-png")

    (fake_bin / "cp").write_text(
        "#!/bin/sh\n"
        f'if [ "${{1:-}}" = {shlex.quote(str(tauri_icns))} ]; then exit 1; fi\n'
        'exec /bin/cp "$@"\n'
    )
    (fake_bin / "sips").write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --out ]; then shift; out=$1; fi\n'
        "  shift\n"
        "done\n"
        'mkdir -p "$(dirname "$out")"; printf resized > "$out"\n'
    )
    (fake_bin / "iconutil").write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = -o ]; then shift; out=$1; fi\n'
        "  shift\n"
        "done\n"
        'printf generated > "$out"\n'
    )
    for command in ("cp", "sips", "iconutil"):
        (fake_bin / command).chmod(0o755)

    script = "\n".join(
        [
            f"_css_tauri_icns={shlex.quote(str(tauri_icns))}",
            f"_css_gem_png={shlex.quote(str(gem_png))}",
            f"_css_icon_png={shlex.quote(str(fallback_png))}",
            f"_css_res_dir={shlex.quote(str(resources))}",
            block,
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": f"{fake_bin}:/usr/bin:/bin", **{"TMPDIR": str(tmp_path)}},
        capture_output = True,
        text = True,
    )
    assert result.returncode == 0, result.stderr
    assert (resources / "AppIcon.icns").read_text() == "generated"
