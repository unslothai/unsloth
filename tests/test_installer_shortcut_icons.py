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
from unsloth_pwsh_runner import run_pwsh

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


INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _wsl_packaged_icon_block() -> str:
    source = INSTALL_SH.read_text(encoding = "utf-8")
    start = source.index('        _css_wsl_ico_win=""')
    end = source.index("        # Create shortcuts via a temp PowerShell script", start)
    return textwrap.dedent(source[start:end])


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash is unavailable")
def test_wsl_shortcut_reads_the_packaged_ico_out_of_site_packages(tmp_path):
    """A WSL install that found the .ico in site-packages must hand the shortcut script a
    Windows path."""
    dist = (
        tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "studio" / "frontend" / "dist"
    )
    dist.mkdir(parents = True)
    (dist / "unsloth.ico").write_bytes(b"\x00\x00\x01\x00icon")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "wslpath").write_text("#!/bin/sh\nprintf 'C:\\\\wsl\\\\%s' \"${2##*/}\"\n")
    (fake_bin / "wslpath").chmod(0o755)

    script = "\n".join(
        [
            f"_css_venv_dir={shlex.quote(str(tmp_path / 'venv'))}",
            _wsl_packaged_icon_block(),
            'printf "%s" "$_css_wsl_ico_win_ps"',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": f"{fake_bin}:/usr/bin:/bin"},
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout == "C:\\wsl\\unsloth.ico"


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash is unavailable")
@pytest.mark.parametrize("ships_icon", [True, False])
def test_wsl_shortcut_falls_back_when_the_ico_cannot_be_translated(tmp_path, ships_icon):
    dist = (
        tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "studio" / "frontend" / "dist"
    )
    dist.mkdir(parents = True)
    if ships_icon:
        (dist / "unsloth.ico").write_bytes(b"\x00\x00\x01\x00icon")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "wslpath").write_text("#!/bin/sh\nprintf 'wslpath: cannot translate'\nexit 1\n")
    (fake_bin / "wslpath").chmod(0o755)

    script = "\n".join(
        [
            f"_css_venv_dir={shlex.quote(str(tmp_path / 'venv'))}",
            _wsl_packaged_icon_block(),
            'printf "%s" "$_css_wsl_ico_win_ps"',
        ]
    )
    result = subprocess.run(
        ["bash", "-c", script],
        env = {"PATH": f"{fake_bin}:/usr/bin:/bin"},
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout == ""


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh is unavailable")
def test_windows_shortcut_prefers_the_packaged_icon_over_the_download(tmp_path):
    """An irm|iex install has no $PSScriptRoot and so no $bundledIcon, which is where the
    packaged .ico has to win."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    start = source.index("            $hasValidIcon = $false")
    end = source.index("            if (Test-Path -LiteralPath $iconPath) {", start)
    block = textwrap.dedent(source[start:end])

    packaged = tmp_path / "packaged.ico"
    packaged.write_bytes(b"\x00\x00\x01\x00packaged")
    icon_path = tmp_path / "unsloth.ico"

    def run(packaged_icon: str) -> str:
        icon_path.unlink(missing_ok = True)
        script = "\n".join(
            [
                "function Write-StudioLine { }",
                "function Invoke-WebRequest { Write-Output 'DOWNLOADED' }",
                "$bundledIcon = $null",
                f"$packagedIcon = {packaged_icon}",
                f"$iconPath = '{icon_path}'",
                "$iconUrl = 'https://example.invalid/unsloth.ico'",
                block,
            ]
        )
        # run_pwsh, not subprocess.run: a pwsh killed at startup never read $packagedIcon,
        # and check = True would report that as the precedence being wrong.
        return run_pwsh(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
            check = True,
            capture_output = True,
            text = True,
        ).stdout

    assert "DOWNLOADED" not in run(f"'{packaged}'")
    assert icon_path.read_bytes() == b"\x00\x00\x01\x00packaged"
    assert "DOWNLOADED" in run("$null")


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "pwsh is unavailable")
def test_windows_packaged_icon_is_found_next_to_the_managed_python(tmp_path):
    """Resolve one level too few or too many and nothing is found, so the icon is downloaded."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    start = source.index("            $packagedIcon = $null")
    end = source.index('            $iconUrl = "https://raw.githubusercontent.com', start)
    block = textwrap.dedent(source[start:end])

    venv = tmp_path / "venv"
    (venv / "Scripts").mkdir(parents = True)
    (venv / "Scripts" / "python.exe").write_text("")
    icon = venv / "Lib" / "site-packages" / "studio" / "frontend" / "dist" / "unsloth.ico"
    icon.parent.mkdir(parents = True)
    icon.write_bytes(b"\x00\x00\x01\x00packaged")

    script = "\n".join(
        [
            f"$ManagedPythonPath = '{venv / 'Scripts' / 'python.exe'}'",
            block,
            "Write-Output $packagedIcon",
        ]
    )
    found = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
    ).stdout.strip()
    assert found.startswith(str(venv))
    assert found.endswith("unsloth.ico")


@pytest.mark.skipif(
    shutil.which("pwsh") is None or shutil.which("bash") is None,
    reason = "pwsh and bash are both required",
)
def test_wsl_shortcut_script_copies_the_packaged_ico_instead_of_downloading(tmp_path):
    """The half of the WSL path that runs under Windows, piped through bash because
    install.sh writes it in an unquoted heredoc: a `$` that lost its backslash expands to
    nothing and strips the path off Copy-Item, which no later check would catch."""
    source = INSTALL_SH.read_text(encoding = "utf-8")
    start = source.index("\\$packagedIcon = '$_css_wsl_ico_win_ps'")
    end = source.index("\\$hasIcon = \\$false", start)
    packaged = tmp_path / "packaged.ico"
    packaged.write_bytes(b"\x00\x00\x01\x00packaged")
    icon_dir = tmp_path / "Unsloth Studio"
    block = subprocess.run(
        [
            "bash",
            "-c",
            "\n".join(
                [
                    f"_css_wsl_ico_win_ps={shlex.quote(str(packaged))}",
                    "cat << WSLPS1_EOF",
                    source[start:end],
                    "WSLPS1_EOF",
                ]
            ),
        ],
        env = {"PATH": "/usr/bin:/bin"},
        check = True,
        capture_output = True,
        text = True,
    ).stdout

    script = "\n".join(
        [
            "function Invoke-WebRequest { Write-Output 'DOWNLOADED' }",
            f"$iconDir = '{icon_dir}'",
            f"$iconPath = '{icon_dir / 'unsloth.ico'}'",
            block,
        ]
    )
    result = run_pwsh(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        check = True,
        capture_output = True,
        text = True,
    )
    assert "DOWNLOADED" not in result.stdout
    assert (icon_dir / "unsloth.ico").read_bytes() == b"\x00\x00\x01\x00packaged"
