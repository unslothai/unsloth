"""Regression tests for installer shortcut icon selection and fallback behavior."""

from __future__ import annotations

import shlex
import shutil
import struct
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
TAURI_ICON_PNG = REPO_ROOT / "studio" / "src-tauri" / "icons" / "icon.png"


def test_linux_tauri_icon_is_1024_square():
    data = TAURI_ICON_PNG.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    assert struct.unpack(">II", data[16:24]) == (1024, 1024)


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
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = --out ]; then shift; out=$1; fi\n"
        "  shift\n"
        "done\n"
        'mkdir -p "$(dirname "$out")"; printf resized > "$out"\n'
    )
    (fake_bin / "iconutil").write_text(
        "#!/bin/sh\n"
        "while [ \"$#\" -gt 0 ]; do\n"
        "  if [ \"$1\" = -o ]; then shift; out=$1; fi\n"
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
