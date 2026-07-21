"""Regression tests for installer controls and process exits."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
README = REPO_ROOT / "README.md"

TRUTHY_VALUES = ("1", "true", "TRUE", "yes", "YES", "on", "ON")
FALSEY_VALUES = ("", "0", "false", "no", "off", "anything-else")


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"installer block not found: {pattern}"
    return match.group(0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(value, "true") for value in TRUTHY_VALUES]
    + [(value, "false") for value in FALSEY_VALUES],
)
@pytest.mark.skipif(shutil.which("sh") is None, reason = "POSIX shell is unavailable")
def test_posix_skip_autostart_value_parsing_with_no_torch(value: str, expected: str):
    source = INSTALL_SH.read_text(encoding = "utf-8")
    no_torch_parser = _extract(
        r'case "\$\{UNSLOTH_NO_TORCH:-\}" in.*?esac',
        source,
    )
    autostart_parser = _extract(
        r'case "\$\{UNSLOTH_SKIP_AUTOSTART:-\}" in.*?esac',
        source,
    )
    env = os.environ.copy()
    env["UNSLOTH_NO_TORCH"] = "1"
    env["UNSLOTH_SKIP_AUTOSTART"] = value
    result = subprocess.run(
        [
            "sh",
            "-c",
            (
                f"_NO_TORCH_FLAG=false\n_SKIP_AUTOSTART=false\n{no_torch_parser}\n"
                f'{autostart_parser}\nprintf "%s %s" "$_NO_TORCH_FLAG" "$_SKIP_AUTOSTART"'
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
        env = env,
    )
    assert result.stdout == f"true {expected}"


def test_posix_skip_autostart_bypasses_only_the_interactive_prompt():
    source = INSTALL_SH.read_text(encoding = "utf-8")
    gate = 'if [ "$_SKIP_AUTOSTART" != true ] && [ -t 1 ]; then'
    assert gate in source
    assert source.index(gate) < source.index("Start Unsloth Studio now? [Y/n]")
    assert source.count("Start Unsloth Studio now? [Y/n]") == 1
    assert source.index("Start Unsloth Studio now? [Y/n]") < source.index(
        'step "launch" "manual commands:"'
    )
    assert "export UNSLOTH_SKIP_AUTOSTART=1" in source


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize(
    ("value", "expected"),
    [(value, "True") for value in TRUTHY_VALUES]
    + [(value, "False") for value in FALSEY_VALUES],
)
def test_windows_skip_autostart_value_parsing_with_no_torch(value: str, expected: str):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    parser = _extract(
        r"\$SkipTorch = \$false\s+\$SkipAutostart = \$false\s+.*?"
        r"if \(\$env:UNSLOTH_NO_TORCH -in @\('1', 'true', 'yes', 'on'\)\) "
        r"\{ \$SkipTorch = \$true \}\s+"
        r"if \(\$env:UNSLOTH_SKIP_AUTOSTART -in @\('1', 'true', 'yes', 'on'\)\) "
        r"\{ \$SkipAutostart = \$true \}",
        source,
    )
    env = os.environ.copy()
    env["UNSLOTH_NO_TORCH"] = "1"
    env["UNSLOTH_SKIP_AUTOSTART"] = value
    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f'{parser}; "$SkipTorch $SkipAutostart"',
        ],
        check = True,
        capture_output = True,
        text = True,
        env = env,
    )
    assert result.stdout.strip() == f"True {expected}"


def test_windows_skip_autostart_bypasses_only_the_interactive_prompt():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    gate = (
        "$IsInteractive = (-not $SkipAutostart) -and "
        "[Environment]::UserInteractive -and (-not [Console]::IsInputRedirected)"
    )
    assert gate in source
    assert source.index(gate) < source.index("Start Unsloth Studio now? [Y/n]")
    assert source.count("Start Unsloth Studio now? [Y/n]") == 1
    assert source.index("Start Unsloth Studio now? [Y/n]") < source.index(
        'step "launch" "manual commands:"'
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_windows_installer_invalid_package_fails():
    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(INSTALL_PS1),
            "--package",
            "bad!",
        ],
        capture_output = True,
        text = True,
        timeout = 30,
    )

    assert result.returncode != 0
    assert "package name contains invalid characters" in result.stdout + result.stderr


def test_skip_autostart_is_documented_for_all_installers():
    readme = README.read_text(encoding = "utf-8")
    assert "UNSLOTH_SKIP_AUTOSTART=1 sh" in readme
    assert "$env:UNSLOTH_SKIP_AUTOSTART=1" in readme
