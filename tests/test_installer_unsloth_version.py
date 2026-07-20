"""Regression coverage for installer version reporting."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL | re.MULTILINE)
    assert match is not None, f"installer block not found: {pattern}"
    return match.group(0)


@pytest.mark.skipif(shutil.which("sh") is None, reason = "POSIX shell is unavailable")
def test_posix_installer_reports_installed_distribution_version():
    source = INSTALL_SH.read_text(encoding = "utf-8")
    reporter = _extract(
        r"_installed_package_version=\$\(.*?^fi",
        source,
    )
    result = subprocess.run(
        [
            "sh",
            "-c",
            (
                'step() { printf "%s %s\\n" "$1" "$2"; }\n'
                'substep() { printf "WARN %s\\n" "$1"; }\n'
                f"_VENV_PY={sys.executable!r}\n"
                "PACKAGE_NAME=pytest\n"
                f"{reporter}"
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout.strip() == f"pytest {version('pytest')} installed"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_windows_version_reporter_uses_distribution_metadata():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    reporter = _extract(
        r"    \$installedPackageVersion = .*?^    if .*?^    \} else \{.*?^    \}",
        source,
    )
    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                "function step { param($Label, $Value) Write-Output \"$Label $Value\" }; "
                "function substep { param($Message, $Color) Write-Output \"WARN $Message\" }; "
                f"$VenvPython = '{sys.executable}'; $PackageName = 'pytest'; "
                f"{reporter}"
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout.strip() == f"pytest {version('pytest')} installed"
