"""Regression coverage for installer version reporting."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

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
        r"^_installed_package_version_exit=0.*?^fi.*?^fi",
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
    # run_pwsh: a startup abort prints no version line and would read as bad metadata parsing.
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                'function step { param($Label, $Value) Write-Output "$Label $Value" }; '
                'function substep { param($Message, $Color) Write-Output "WARN $Message" }; '
                f"$VenvPython = '{sys.executable}'; $PackageName = 'pytest'; "
                f"{reporter}"
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout.strip() == f"pytest {version('pytest')} installed"


def _fake_venv(
    root: Path,
    records: dict[str, str],
    windows: bool = False,
) -> Path:
    site = root / ("Lib/site-packages" if windows else "lib/python3.13/site-packages")
    for dirname, version_line in records.items():
        info = site / dirname
        info.mkdir(parents = True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.4\nName: x\n{version_line}\n\nVersion: 0.0.0 in the body\n",
            encoding = "utf-8",
        )
    site.mkdir(parents = True, exist_ok = True)
    return root


DISK_CASES = [
    ({"unsloth-2026.9.3.dist-info": "Version: 2026.9.3"}, "2026.9.3"),
    ({"unsloth-2026.9.3.dist-info": "Version: 2026.9.3\r"}, "2026.9.3"),
    # unsloth_zoo must not be read as unsloth.
    ({"unsloth_zoo-2026.9.1.dist-info": "Version: 2026.9.1"}, ""),
    # Duplicate records: the version is ambiguous, so say nothing.
    (
        {
            "unsloth-2026.9.3.dist-info": "Version: 2026.9.3",
            "unsloth-2026.10.1.dist-info": "Version: 2026.10.1",
        },
        "",
    ),
    ({}, ""),
]


@pytest.mark.skipif(shutil.which("sh") is None, reason = "POSIX shell is unavailable")
@pytest.mark.parametrize("records, expected", DISK_CASES)
def test_posix_previous_version_read_from_disk(tmp_path, records, expected):
    source = INSTALL_SH.read_text(encoding = "utf-8")
    helper = _extract(r"^_dist_version_from_disk\(\) \{.*?^\}", source)
    venv = _fake_venv(tmp_path / "venv", records)
    result = subprocess.run(
        [
            "sh",
            "-c",
            f'set -eu\n{helper}\nprintf "[%s]" "$(_dist_version_from_disk "$1" unsloth)"',
            "sh",
            str(venv),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout == f"[{expected}]"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("records, expected", DISK_CASES)
def test_windows_previous_version_read_from_disk(tmp_path, records, expected):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    helper = _extract(r"^    function Get-PackageVersionFromDisk \{.*?^    \}", source)
    venv = _fake_venv(tmp_path / "venv", records, windows = True)
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"{helper}; Write-Output \"[$(Get-PackageVersionFromDisk -VenvRoot '{venv}' -Name 'unsloth')]\"",
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout.strip() == f"[{expected}]"


REPORT_CASES = [
    ("", ""),
    ("2026.9.3", " (was 2026.9.3)"),
    (None, " (unchanged)"),
]


@pytest.mark.skipif(shutil.which("sh") is None, reason = "POSIX shell is unavailable")
@pytest.mark.parametrize("previous, suffix", REPORT_CASES)
def test_posix_installer_reports_previous_version(previous, suffix):
    current = version("pytest")
    previous = current if previous is None else previous
    reporter = _extract(
        r"^_installed_package_version_exit=0.*?^fi.*?^fi",
        INSTALL_SH.read_text(encoding = "utf-8"),
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
                f"_PREV_PACKAGE_VERSION={previous!r}\n"
                f"{reporter}"
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout.strip() == f"pytest {current} installed{suffix}"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
@pytest.mark.parametrize("previous, suffix", REPORT_CASES)
def test_windows_installer_reports_previous_version(previous, suffix):
    current = version("pytest")
    previous = current if previous is None else previous
    reporter = _extract(
        r"    \$installedPackageVersion = .*?^    if .*?^    \} else \{.*?^    \}",
        INSTALL_PS1.read_text(encoding = "utf-8"),
    )
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                'function step { param($Label, $Value) Write-Output "$Label $Value" }; '
                'function substep { param($Message, $Color) Write-Output "WARN $Message" }; '
                f"$VenvPython = '{sys.executable}'; $PackageName = 'pytest'; "
                f"$script:PrevPackageVersion = '{previous}'; "
                f"{reporter}"
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert result.stdout.strip() == f"pytest {current} installed{suffix}"
