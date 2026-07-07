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
    return INSTALL_PS1.read_text(encoding = "utf-8")


def _extract_venv_bootstrap_body(source: str) -> str:
    lines = source.splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "if (-not (Test-Path -LiteralPath $VenvPython)) {":
            start = idx + 1
            continue
        if start is not None and stripped == "} else {":
            end = idx
            break
    if start is None or end is None:
        raise AssertionError("failed to locate the install.ps1 venv bootstrap block")
    return "\n".join(lines[start:end])


def _source_without_uv_repair() -> str:
    source = _source()
    body = _extract_venv_bootstrap_body(source)
    anchor = "        # Trust neither uv's exit code nor a half-baked Scripts\\python.exe."
    if anchor not in body:
        raise AssertionError("failed to locate the uv repair block")
    stripped_body = body[: body.index(anchor)].rstrip()
    return source.replace(body, stripped_body, 1)


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
        body = '@echo off\r\n>>"%UV_LOG%" echo %*\r\nexit /b 7\r\n'
    elif mode == "corrupt":
        body = textwrap.dedent(
            """\
            @echo off
            >>"%UV_LOG%" echo %*
            if not exist "%~2\\Scripts" mkdir "%~2\\Scripts"
            type nul > "%~2\\Scripts\\python.exe"
            >"%~2\\uv-partial.txt" echo broken
            exit /b 0
            """
        )
    elif mode == "base_python":
        body = textwrap.dedent(
            """\
            @echo off
            >>"%UV_LOG%" echo %*
            if not exist "%~2\\Scripts" mkdir "%~2\\Scripts"
            copy /Y "%~4" "%~2\\Scripts\\python.exe" >nul
            >"%~2\\uv-partial.txt" echo broken
            exit /b 0
            """
        )
    else:
        body = textwrap.dedent(
            """\
            @echo off
            >>"%UV_LOG%" echo %*
            if not exist "%~2" mkdir "%~2"
            >"%~2\\uv-partial.txt" echo broken
            exit /b 0
            """
        )
    with uv_path.open("w", encoding = "utf-8", newline = "") as fh:
        fh.write(body)
    return uv_path


def _write_python_stub(stub_dir: Path, mode: str) -> Path:
    stub_dir.mkdir(parents = True, exist_ok = True)
    python_path = stub_dir / "python.cmd"
    if mode == "partial_fail":
        body = textwrap.dedent(
            """\
            @echo off
            if not exist "%~3\\Scripts" mkdir "%~3\\Scripts"
            >"%~3\\fallback-partial.txt" echo broken
            exit /b 9
            """
        )
    elif mode == "partial_success_corrupt":
        body = textwrap.dedent(
            """\
            @echo off
            if not exist "%~3\\Scripts" mkdir "%~3\\Scripts"
            type nul > "%~3\\Scripts\\python.exe"
            >"%~3\\fallback-partial.txt" echo broken
            exit /b 0
            """
        )
    else:
        body = f'@echo off\r\n"{sys.executable}" %*\r\nexit /b %ERRORLEVEL%\r\n'
    with python_path.open("w", encoding = "utf-8", newline = "") as fh:
        fh.write(body)
    return python_path


def _run_bootstrap(
    tmp_path: Path,
    mode: str,
    source_text: str,
    python_mode: str = "real",
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    if PWSH is None:
        pytest.skip("pwsh not available")

    venv_dir = tmp_path / "space path" / "unsloth_studio"
    stub_dir = tmp_path / "stub bin"
    log_file = tmp_path / "uv.log"
    _write_uv_stub(stub_dir, mode)
    detected_python = (
        _write_python_stub(stub_dir, python_mode) if python_mode != "real" else Path(sys.executable)
    )

    env = os.environ.copy()
    env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")
    env["UV_LOG"] = str(log_file)
    if extra_env:
        env.update(extra_env)

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
            str(detected_python),
        ],
        capture_output = True,
        text = True,
        env = env,
    )
    return proc, venv_dir, log_file, venv_dir / "uv-partial.txt"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_install_ps1_rechecks_uv_success_before_continuing():
    body = _extract_venv_bootstrap_body(_source())
    assert (
        "uv venv returned success but left an unusable venv; rebuilding with python -m venv..."
        in body
    )
    assert 'Join-Path $VenvRoot "pyvenv.cfg"' in body
    assert "sys.prefix" in body
    assert "sys.base_prefix" in body
    assert "Invoke-InstallCommand { & $PythonExe -c" not in body
    assert "Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue" in body
    assert "& $DetectedPython.Path -m venv $VenvDir" in body


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_empty_head_passes(tmp_path):
    proc, venv_dir, log_file, partial_marker = _run_bootstrap(tmp_path, "empty", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert not partial_marker.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_corrupt_python_falls_back(tmp_path):
    proc, venv_dir, log_file, partial_marker = _run_bootstrap(tmp_path, "corrupt", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert not partial_marker.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_base_python_without_config_falls_back(tmp_path):
    proc, venv_dir, log_file, partial_marker = _run_bootstrap(tmp_path, "base_python", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert (venv_dir / "pyvenv.cfg").is_file()
    assert not partial_marker.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_probe_allows_python_startup_stderr(tmp_path):
    sitecustomize_dir = tmp_path / "sitecustomize"
    sitecustomize_dir.mkdir()
    (sitecustomize_dir / "sitecustomize.py").write_text(
        "import sys; sys.stderr.write('startup warning\\n')\n",
        encoding = "utf-8",
    )

    proc, venv_dir, log_file, partial_marker = _run_bootstrap(
        tmp_path,
        "healthy",
        _source(),
        extra_env = {"PYTHONPATH": str(sitecustomize_dir)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert not partial_marker.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_empty_without_repair_fails(tmp_path):
    proc, venv_dir, log_file, partial_marker = _run_bootstrap(
        tmp_path, "empty", _source_without_uv_repair()
    )
    assert proc.returncode == 91, proc.stdout + proc.stderr
    assert not (venv_dir / "Scripts" / "python.exe").exists()
    assert partial_marker.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_healthy_skips_fallback(tmp_path):
    proc, venv_dir, log_file, _ = _run_bootstrap(tmp_path, "healthy", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_nonzero_failure_is_preserved(tmp_path):
    proc, _, log_file, _ = _run_bootstrap(tmp_path, "fail", _source())
    assert proc.returncode == 7, proc.stdout + proc.stderr
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_fallback_failure_removes_partial(tmp_path):
    proc, venv_dir, log_file, _ = _run_bootstrap(
        tmp_path, "empty", _source(), python_mode = "partial_fail"
    )
    assert proc.returncode == 9, proc.stdout + proc.stderr
    assert not venv_dir.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_fallback_success_with_unusable_python_removes_partial(tmp_path):
    proc, venv_dir, log_file, _ = _run_bootstrap(
        tmp_path, "empty", _source(), python_mode = "partial_success_corrupt"
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not venv_dir.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
def test_migrated_branch_stays_side_effect_free():
    text = _source()
    anchor = text.index('step "venv" "using migrated environment"')
    window = text[anchor - 120 : anchor + 140]
    assert "python -m venv" not in window
    assert "uv venv" not in window
