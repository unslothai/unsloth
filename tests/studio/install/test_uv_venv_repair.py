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
POWERSHELL_51 = shutil.which("powershell.exe")


def _source(refspec: str | None = None) -> str:
    return INSTALL_PS1.read_text(encoding = "utf-8")


def _extract_venv_bootstrap_body(source: str) -> str:
    lines = source.splitlines()
    helper_start = None
    start = None
    end = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "function Get-StudioVenvPathState {":
            helper_start = idx
        if stripped == "if (-not (Test-Path -LiteralPath $VenvPython)) {":
            start = idx + 1
            continue
        if start is not None and stripped == "} else {":
            end = idx
            break
    if helper_start is None or start is None or end is None:
        raise AssertionError("failed to locate the install.ps1 venv bootstrap block")
    return "\n".join(lines[helper_start : start - 1] + lines[start:end])


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
            >"%~2\\pyvenv.cfg" echo home = %~2
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
    preexisting_venv_file: bool = False,
    preexisting_venv_directory: bool = False,
    preexisting_venv_reparse: bool = False,
    legacy_root_sentinel: str | None = None,
    redirect_mode: str = "env",
    powershell_exe: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    powershell_exe = powershell_exe or PWSH
    if powershell_exe is None:
        pytest.skip("PowerShell not available")

    venv_dir = tmp_path / "space path" / "unsloth_studio"
    if preexisting_venv_file:
        venv_dir.parent.mkdir(parents = True)
        venv_dir.write_text("user-owned", encoding = "utf-8")
    elif preexisting_venv_directory:
        venv_dir.mkdir(parents = True)
        (venv_dir / "important.txt").write_text("user-owned", encoding = "utf-8")
        if legacy_root_sentinel == "share":
            sentinel = venv_dir.parent / "share" / "studio.conf"
            sentinel.parent.mkdir(parents = True, exist_ok = True)
            sentinel.write_text("legacy", encoding = "utf-8")
        elif legacy_root_sentinel == "bin":
            sentinel = venv_dir.parent / "bin" / "unsloth.exe"
            sentinel.parent.mkdir(parents = True, exist_ok = True)
            sentinel.write_text("legacy", encoding = "utf-8")
    elif preexisting_venv_reparse:
        target = tmp_path / "reparse-target"
        target.mkdir()
        (target / "important.txt").write_text("user-owned", encoding = "utf-8")
        try:
            venv_dir.symlink_to(target, target_is_directory = True)
        except OSError as exc:
            pytest.skip(f"directory reparse point unavailable: {exc}")

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
                [Parameter(Mandatory = $true)][string]$DetectedPythonPath,
                [Parameter(Mandatory = $true)][string]$RedirectMode
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
            $StudioRedirectMode = $RedirectMode
            $StudioHome = Split-Path -Parent $VenvDir
            $VenvPython = Join-Path $VenvDir "Scripts\\python.exe"
            $VenvOwnershipMarker = Join-Path $VenvDir ".unsloth-studio-owned"
            $VenvPathExistedBeforeCreate = Test-Path -LiteralPath $VenvDir
            $VenvDirExistedBeforeCreate = Test-Path -LiteralPath $VenvDir -PathType Container
            $VenvDirOwnedBeforeCreate = $VenvDirExistedBeforeCreate -and (
                (Test-Path -LiteralPath $VenvOwnershipMarker -PathType Leaf) -or
                (Test-Path -LiteralPath (Join-Path $StudioHome "share\\studio.conf") -PathType Leaf) -or
                (Test-Path -LiteralPath (Join-Path $StudioHome "bin\\unsloth.exe") -PathType Leaf)
            )
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
            powershell_exe,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(harness),
            "-VenvDir",
            str(venv_dir),
            "-DetectedPythonPath",
            str(detected_python),
            "-RedirectMode",
            redirect_mode,
        ],
        capture_output = True,
        text = True,
        env = env,
    )
    return proc, venv_dir, log_file, venv_dir / "uv-partial.txt"


def _run_full_install_script(
    tmp_path: Path,
    source_text: str,
    *,
    powershell_exe: str | None = None,
    preexisting_venv_directory: bool = False,
    legacy_root_sentinel: str | None = None,
    stop_after_rollback: bool = False,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    powershell_exe = powershell_exe or PWSH
    if powershell_exe is None:
        pytest.skip("PowerShell not available")

    studio_home = tmp_path / "studio-home"
    venv_dir = studio_home / "unsloth_studio"
    if preexisting_venv_directory:
        venv_dir.mkdir(parents = True)
        (venv_dir / "important.txt").write_text("user-owned", encoding = "utf-8")
        if legacy_root_sentinel == "share":
            sentinel = studio_home / "share" / "studio.conf"
            sentinel.parent.mkdir(parents = True, exist_ok = True)
            sentinel.write_text("legacy", encoding = "utf-8")
        elif legacy_root_sentinel == "bin":
            sentinel = studio_home / "bin" / "unsloth.exe"
            sentinel.parent.mkdir(parents = True, exist_ok = True)
            sentinel.write_text("legacy", encoding = "utf-8")

    stub_dir = tmp_path / "script-stub-bin"
    log_file = tmp_path / "script-uv.log"
    _write_uv_stub(stub_dir, "healthy")
    _write_python_stub(stub_dir, "real")

    stop_anchor = (
        "    # Env-mode session export AFTER Refresh-SessionPath; otherwise a legacy"
        if stop_after_rollback
        else "    # ── Helper: run amd-smi without triggering a UAC elevation prompt ──"
    )
    source_text = source_text.replace(
        stop_anchor,
        '    if ($env:UNSLOTH_TEST_STOP_AFTER_VENV -eq "1") { exit 0 }\n\n' + stop_anchor,
        1,
    )
    script_path = REPO_ROOT / f"install.full-entrypoint.{tmp_path.name}.ps1"
    script_path.write_text(source_text, encoding = "utf-8")

    env = os.environ.copy()
    env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")
    env["UV_LOG"] = str(log_file)
    env["UNSLOTH_STUDIO_HOME"] = str(studio_home)
    env["UNSLOTH_TEST_STOP_AFTER_VENV"] = "1"

    try:
        proc = subprocess.run(
            [
                powershell_exe,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "--local",
                "--no-torch",
            ],
            capture_output = True,
            text = True,
            env = env,
            cwd = str(REPO_ROOT),
        )
    finally:
        script_path.unlink(missing_ok = True)
    return proc, venv_dir, log_file


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_install_ps1_rechecks_uv_success_before_continuing():
    body = _extract_venv_bootstrap_body(_source())
    assert (
        "uv venv returned success but left an unusable venv; rebuilding with python -m venv..."
        in body
    )
    assert "uv venv failed; rebuilding with python -m venv..." in body
    assert "if ($needsVenvFallback)" in body
    assert 'Join-Path $VenvRoot "pyvenv.cfg"' in body
    assert "sys.prefix" in body
    assert "sys.base_prefix" in body
    assert "System.Diagnostics.ProcessStartInfo" in body
    assert "return ($proc.ExitCode -eq 0)" in body
    assert "Invoke-InstallCommand { & $PythonExe -c" not in body
    assert (
        "Remove-Item -LiteralPath $VenvDir -Recurse -Force -ErrorAction SilentlyContinue"
        not in body
    )
    assert "& $DetectedPython.Path -m venv $VenvStage" in body
    assert "Move-StudioVenvToQuarantine" in body
    assert "Publish-StudioVenvCandidate" in body


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
def test_uv_venv_base_python_with_config_falls_back(tmp_path):
    proc, venv_dir, log_file, partial_marker = _run_bootstrap(tmp_path, "base_python", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert (venv_dir / "pyvenv.cfg").is_file()
    assert partial_marker.exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_prefix_identity_rejection_runs_after_zero_exit(tmp_path):
    source = _source().replace(
        'Test-VenvPythonReady (Join-Path $VenvStage "Scripts\\python.exe") $VenvStage',
        'Test-VenvPythonReady (Join-Path $VenvStage "Scripts\\python.exe") (Join-Path $VenvStage "mismatch")',
        1,
    )
    proc, venv_dir, log_file, partial_marker = _run_bootstrap(tmp_path, "base_python", source)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "pyvenv.cfg").is_file()
    assert (
        not partial_marker.exists()
    ), "prefix-identity rejection must trigger fallback after zero-exit uv output"
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_publish_fails_closed_when_target_appears_after_absence_check(tmp_path):
    source = _source()
    needle = "        Assert-StudioVenvAbsent $VenvDir\n        [System.IO.Directory]::Move("
    replacement = (
        "        Assert-StudioVenvAbsent $VenvDir\n"
        "        [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null\n"
        "        [System.IO.Directory]::Move("
    )
    assert needle in source
    proc, venv_dir, _, _ = _run_bootstrap(
        tmp_path, "healthy", source.replace(needle, replacement, 1)
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert venv_dir.is_dir()
    assert not (
        venv_dir / "Scripts"
    ).exists(), "candidate must not be nested into substituted target"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(POWERSHELL_51 is None, reason = "Windows PowerShell 5.1 not available")
def test_uv_venv_guard_runs_under_windows_powershell_51(tmp_path):
    proc, venv_dir, _, _ = _run_bootstrap(
        tmp_path,
        "fail",
        _source(),
        preexisting_venv_directory = True,
        powershell_exe = POWERSHELL_51,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert (venv_dir / "important.txt").read_text(encoding = "utf-8") == "user-owned"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(POWERSHELL_51 is None, reason = "Windows PowerShell 5.1 not available")
def test_install_ps1_full_entrypoint_rejects_foreign_target_under_powershell_51(tmp_path):
    proc, venv_dir, _ = _run_full_install_script(
        tmp_path,
        _source(),
        powershell_exe = POWERSHELL_51,
        preexisting_venv_directory = True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "does not look like an Unsloth Studio install" in (proc.stdout + proc.stderr)
    assert (venv_dir / "important.txt").read_text(encoding = "utf-8") == "user-owned"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_reparse_target_is_rejected(tmp_path):
    proc, venv_dir, _, _ = _run_bootstrap(
        tmp_path, "fail", _source(), preexisting_venv_reparse = True
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert venv_dir.is_symlink()


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_file_reparse_target_is_rejected(tmp_path):
    file_target = tmp_path / "reparse-file-target.txt"
    file_target.write_text("user-owned", encoding = "utf-8")
    venv_dir = tmp_path / "space path" / "unsloth_studio"
    venv_dir.parent.mkdir(parents = True, exist_ok = True)
    try:
        venv_dir.symlink_to(file_target)
    except OSError as exc:
        pytest.skip(f"file reparse point unavailable: {exc}")

    proc, _, log_file, _ = _run_bootstrap(tmp_path, "fail", _source())
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert venv_dir.is_symlink()
    assert venv_dir.read_text(encoding = "utf-8") == "user-owned"
    assert not log_file.exists(), "foreign path must be rejected before uv runs"


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
@pytest.mark.skipif(POWERSHELL_51 is None, reason = "Windows PowerShell 5.1 not available")
def test_uv_venv_probe_allows_python_startup_stderr_under_powershell_51(tmp_path):
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
        powershell_exe = POWERSHELL_51,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert not partial_marker.exists()
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
def test_uv_venv_nonzero_failure_falls_back(tmp_path):
    proc, venv_dir, log_file, _ = _run_bootstrap(tmp_path, "fail", _source())
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_uv_venv_nonzero_preserves_preexisting_file_path(tmp_path):
    proc, venv_dir, log_file, _ = _run_bootstrap(
        tmp_path,
        "fail",
        _source(),
        preexisting_venv_file = True,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert venv_dir.is_file()
    assert venv_dir.read_text(encoding = "utf-8") == "user-owned"
    assert not log_file.exists(), "foreign path must be rejected before uv runs"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
@pytest.mark.parametrize("legacy_root_sentinel", ["share", "bin"])
def test_uv_venv_legacy_studio_root_is_accepted_for_final_target(tmp_path, legacy_root_sentinel):
    proc, venv_dir, log_file, _ = _run_bootstrap(
        tmp_path,
        "healthy",
        _source(),
        preexisting_venv_directory = True,
        legacy_root_sentinel = legacy_root_sentinel,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (venv_dir / ".unsloth-studio-owned").is_file()
    assert not (venv_dir / "important.txt").exists()
    assert log_file.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
@pytest.mark.parametrize("redirect_mode", ["env", "profile", "default"])
def test_uv_venv_foreign_directory_is_preserved_in_every_mode(tmp_path, redirect_mode):
    proc, venv_dir, log_file, _ = _run_bootstrap(
        tmp_path,
        "fail",
        _source(),
        preexisting_venv_directory = True,
        redirect_mode = redirect_mode,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert (venv_dir / "important.txt").read_text(encoding = "utf-8") == "user-owned"
    assert not (venv_dir / ".unsloth-studio-owned").exists()
    assert not log_file.exists(), "foreign path must be rejected before uv runs"


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
def test_uv_venv_retry_after_failed_fallback_uses_fresh_stage(tmp_path):
    first_proc, venv_dir, first_log, _ = _run_bootstrap(
        tmp_path, "empty", _source(), python_mode = "partial_fail"
    )
    assert first_proc.returncode == 9, first_proc.stdout + first_proc.stderr
    assert not venv_dir.exists()
    assert first_log.read_text(encoding = "utf-8").strip(), "uv stub did not run"

    second_proc, venv_dir, second_log, _ = _run_bootstrap(tmp_path, "healthy", _source())
    assert second_proc.returncode == 0, second_proc.stdout + second_proc.stderr
    assert (venv_dir / "Scripts" / "python.exe").is_file()
    assert second_log.read_text(encoding = "utf-8").strip(), "uv stub did not run"


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows installer test")
@pytest.mark.skipif(PWSH is None, reason = "pwsh not available")
def test_install_ps1_successful_reinstall_warns_when_rollback_cleanup_fails(tmp_path):
    source = _source()
    source = source.replace(
        "                Remove-StudioVenvDirectory $backup",
        '                throw "simulated rollback cleanup lock"',
        1,
    )
    source = source.replace(
        "        & $UnslothExe @studioArgs\n        $setupExit = $LASTEXITCODE",
        "        $setupExit = 0",
        1,
    )
    proc, venv_dir, log_file = _run_full_install_script(
        tmp_path,
        source,
        preexisting_venv_directory = True,
        legacy_root_sentinel = "share",
        stop_after_rollback = True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Could not remove rollback copy" in (proc.stdout + proc.stderr)
    assert (venv_dir / "Scripts" / "python.exe").is_file()
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
