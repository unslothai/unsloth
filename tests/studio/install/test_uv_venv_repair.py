"""Regression tests for install.ps1's uv venv bootstrap repair."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_PS1 = REPO_ROOT / "install.ps1"
PWSH = shutil.which("pwsh")


def _source(refspec: str | None = None) -> str:
    if refspec is None:
        return INSTALL_PS1.read_text(encoding = "utf-8")
    result = subprocess.run(
        ["git", "show", refspec],
        capture_output = True,
        check = True,
        cwd = REPO_ROOT,
        text = True,
    )
    return result.stdout


def _extract_venv_bootstrap_body(source: str) -> str:
    lines = source.splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == 'if (-not (Test-Path -LiteralPath $VenvPython)) {':
            start = idx + 1
            continue
        if start is not None and stripped == "} else {":
            end = idx
            break
    if start is None or end is None:
        raise AssertionError("failed to locate the install.ps1 venv bootstrap block")
    return "\n".join(lines[start:end])


def _write_uv_stub(stub_dir: Path, mode: str) -> Path:
    stub_dir.mkdir(parents = True, exist_ok = True)
    uv_path = stub_dir / "uv.cmd"
    if mode == "healthy":
        body = textwrap.dedent(
            """\
            @echo off
            >>"%UV_LOG%" echo %*
            "%~4" -m venv "%~2"
            exit /b %ERRORLEVEL%
            """
        )
    elif mode == "fail":
        body = "@echo off\r\n>>\"%UV_LOG%\" echo %*\r\nexit /b 7\r\n"
    else:
        body = "@echo off\r\n>>\"%UV_LOG%\" echo %*\r\nexit /b 0\r\n"
    with uv_path.open("w", encoding = "utf-8", newline = "") as fh:
        fh.write(body)
    return uv_path


def _run_bootstrap(tmp_path: Path, mode: str, source_text: str) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    if PWSH is None:
        pytest.skip("pwsh not available")

    venv_dir = tmp_path / "space path" / "unsloth_studio"
    stub_dir = tmp_path / "stub bin"
    log_file = tmp_path / "uv.log"
    _write_uv_stub(stub_dir, mode)

    env = os.environ.copy()
    env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")
    env["UV_LOG"] = str(log_file)

    source_body = _extract_venv_bootstrap_body(source_text)
    harness = tmp_path / "bootstrap.ps1"
    harness.write_text(
        textwrap.dedent(
            """\
            param(
                [Parameter(Mandatory = $true)][string]$VenvDir,
                [Parameter(Mandatory = $true)][string]$DetectedPythonPath
            )

            $ErrorActionPreference = "Stop"

            function Invoke-InstallCommand {
                param(
                    [Parameter(Mandatory = $true)][ScriptBlock]$Command
                )
                $prevEap = $ErrorActionPreference
                $ErrorActionPreference = "Continue"
                try {
                    $global:LASTEXITCODE = 0
                    $output = & $Command 2>&1 | Out-String
                    if ($LASTEXITCODE -ne 0) {
                        Write-Host $output -ForegroundColor Red
                    }
                    return [int]$LASTEXITCODE
                } finally {
                    $ErrorActionPreference = $prevEap
                }
            }

            function step {
                param([Parameter(ValueFromRemainingArguments = $true)]$Args)
            }

            function substep {
                param([Parameter(ValueFromRemainingArguments = $true)]$Args)
            }

            function Exit-InstallFailure {
                param(
                    [Parameter(Mandatory = $true)][string]$Message,
                    [int]$Code = 1
                )
                if ($Code -eq 0) { $Code = 1 }
                exit $Code
            }

            $DetectedPython = [pscustomobject]@{
                Path = $DetectedPythonPath
                Version = "3.13"
            }
            $VenvPython = Join-Path $VenvDir "Scripts\\python.exe"
            if (-not (Test-Path -LiteralPath $VenvPython)) {
            __SOURCE_BODY__
            }

            if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) { exit 91 }
            if ((Invoke-InstallCommand { & $VenvPython -c "import sys; print(sys.executable)" }) -ne 0) { exit 92 }
            exit 0
            """
        ).replace("__SOURCE_BODY__", source_body),
        encoding = "utf-8",
    )

    proc = subprocess.run(
        [
            PWSH,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-VenvDir",
            str(venv_dir),
            "-DetectedPythonPath",
            sys.executable,
        ],
        capture_output = True,
        text = True,
        env = env,
    )
    return proc, venv_dir, log_file


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_install_ps1_rechecks_uv_success_before_continuing():
    body = _extract_venv_bootstrap_body(_source())
    assert "uv venv returned success but left an unusable venv; rebuilding with python -m venv..." in body
    assert "& $DetectedPython.Path -m venv $VenvDir" in body


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_empty_head_passes(tmp_path):
    proc, venv_dir, log_file = _run_bootstrap(tmp_path, "empty", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_empty_origin_main_fails(tmp_path):
    proc, venv_dir, log_file = _run_bootstrap(tmp_path, "empty", _source("origin/main:install.ps1"))
    assert proc.returncode == 91, proc.stdout + proc.stderr
    assert not (venv_dir / "Scripts" / "python.exe").exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_healthy_skips_fallback(tmp_path):
    proc, venv_dir, log_file = _run_bootstrap(tmp_path, "healthy", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_nonzero_failure_is_preserved(tmp_path):
    proc, _, log_file = _run_bootstrap(tmp_path, "fail", _source())
    assert proc.returncode == 7, proc.stdout + proc.stderr
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
def test_migrated_branch_stays_side_effect_free():
    text = _source()
    anchor = text.index('step "venv" "using migrated environment"')
    window = text[anchor - 120 : anchor + 140]
    assert "python -m venv" not in window
    assert "uv venv" not in window
