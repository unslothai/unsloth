"""Regression coverage for the pre-install "current version" / "new version" banner
lines requested by unslothai/unsloth#9910.

`Get-StudioVersionProbe` and the display block that calls it live between the banner
title and the winget check in install.ps1; these tests pin their relative order and
their behaviour for the states a pre-flight probe can actually land in: nothing
installed yet, something installed, an env-var override, and a broken interpreter.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

from unsloth_pwsh_runner import run_pwsh

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALL_PS1 = REPO_ROOT / "install.ps1"


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL | re.MULTILINE)
    assert match is not None, f"installer block not found: {pattern}"
    return match.group(0)


def _ps_literal(value: object) -> str:
    """A single-quoted PowerShell literal. Doubling escapes an apostrophe, which a Windows
    account named O'Brien puts in tmp_path; unsloth_cli/commands/studio.py does the same."""
    return "'" + str(value).replace("'", "''") + "'"


_RESOLVER_PATTERN = (
    r"    function Get-StudioVersionProbe \{.*?\n    \} catch \{\n"
    r'        \$CurrentVersionDisplay = "unknown"\n    \}\n'
)


def test_version_banner_sits_between_the_title_and_winget():
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    title_idx = source.index('" Unsloth Studio Installer (Windows)"')
    resolver_idx = source.index("function Get-StudioVersionProbe")
    display_idx = source.index('current version:{1} {2}"')
    winget_idx = source.index('step "winget" "available"')
    # The probe/resolution runs ahead of the banner text itself (it has to, so the
    # values exist by the time the banner prints), and the display lines land after
    # the title but still well before the winget check, per #9910.
    assert resolver_idx < title_idx < display_idx < winget_idx


def test_new_version_is_never_hardcoded_independently_of_the_install_spec():
    # #9910 explicitly forbids a second literal: the banner's floor and the one uv
    # actually installs must come from the same $_unslothReleaseInstallSpec.
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    assert (
        source.count('"unsloth>=2026.8.22"') == 1
    ), "the install floor must be defined exactly once and reused for the banner"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_fresh_install_reports_not_installed_and_the_pinned_floor(tmp_path):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    resolver = _extract(_RESOLVER_PATTERN, source)
    venv_dir = tmp_path / "studio-venv"  # never created: no Scripts\python.exe on disk
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                f"$VenvDir = {_ps_literal(venv_dir)}; $PackageName = 'unsloth'; "
                f"{resolver} "
                'Write-Output "CURRENT=$CurrentVersionDisplay"; '
                'Write-Output "NEW=$NewVersionDisplay"'
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    lines = result.stdout.strip().splitlines()
    assert "CURRENT=not installed" in lines
    assert "NEW=2026.8.22" in lines


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_desktop_backend_override_is_reflected_in_the_new_version(tmp_path):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    resolver = _extract(_RESOLVER_PATTERN, source)
    venv_dir = tmp_path / "studio-venv"
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                f"$VenvDir = {_ps_literal(venv_dir)}; $PackageName = 'unsloth'; "
                "$env:UNSLOTH_DESKTOP_BACKEND_VERSION = '9.9.9'; "
                f"{resolver} "
                'Write-Output "NEW=$NewVersionDisplay"'
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    assert "NEW=9.9.9" in result.stdout.strip().splitlines()


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_an_existing_install_is_reported_by_querying_its_own_venv(tmp_path):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    resolver = _extract(_RESOLVER_PATTERN, source)
    scripts_dir = tmp_path / "studio-venv" / "Scripts"
    scripts_dir.mkdir(parents = True)
    venv_python = scripts_dir / "python.exe"
    # A real interpreter so the embedded Python actually runs; invoked from tmp_path
    # (not the repo root) so `import studio.install_manifest` is unavailable and the
    # probe takes the plain importlib.metadata fallback, the same path a non-studio
    # `--package` install would take against a real venv.
    shutil.copy(sys.executable, venv_python)
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                f"$VenvDir = {_ps_literal(tmp_path / 'studio-venv')}; $PackageName = 'pytest'; "
                f"{resolver} "
                'Write-Output "CURRENT=$CurrentVersionDisplay"'
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
        cwd = tmp_path,
    )
    from importlib.metadata import version as dist_version

    assert f"CURRENT={dist_version('pytest')}" in result.stdout.strip().splitlines()


@pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")
def test_a_broken_interpreter_reports_unknown_instead_of_failing(tmp_path):
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    resolver = _extract(_RESOLVER_PATTERN, source)
    scripts_dir = tmp_path / "studio-venv" / "Scripts"
    scripts_dir.mkdir(parents = True)
    venv_python = scripts_dir / "python.exe"
    venv_python.write_text("not a real interpreter", encoding = "utf-8")
    venv_python.chmod(0o755)
    result = run_pwsh(
        [
            "pwsh",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            (
                f"$VenvDir = {_ps_literal(tmp_path / 'studio-venv')}; $PackageName = 'unsloth'; "
                f"{resolver} "
                'Write-Output "CURRENT=$CurrentVersionDisplay"'
            ),
        ],
        check = True,
        capture_output = True,
        text = True,
    )
    # Never blank, never an exception bubbling out -- a probe that cannot run
    # cleanly must still leave the installer with a value to print and keep going.
    assert "CURRENT=unknown" in result.stdout.strip().splitlines()
