# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the Node helper's real PowerShell wrapper with a download-free stub."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

import pytest


class _WarningPipelineAborted(AssertionError):
    pass


def _setup_source():
    configured = os.environ.get("UNSLOTH_REPO_ROOT")
    roots = [Path(configured)] if configured else []
    roots.extend(Path(__file__).resolve().parents)
    for root in roots:
        source = root / "studio/setup.ps1"
        if source.is_file():
            return source
    pytest.skip("Could not locate studio/setup.ps1")


PROBE = r"""
param([string]$SetupPath, [string]$Python, [string]$ReportPath)
$ErrorActionPreference = 'Stop'
$env:UNSLOTH_TAURI_UPDATE = $null
$tokens = $null
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($SetupPath, [ref]$tokens, [ref]$errors)
if ($errors.Count) { throw 'Setup source did not parse' }
$functions = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Show-DownloadProgress'
}, $true))
$assignments = @($ast.FindAll({
    param($node)
    $node -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $node.Left.Extent.Text -eq '$nodeOut' -and
    $node.Operator -eq [System.Management.Automation.Language.TokenKind]::Equals -and
    $node.Extent.Text -match 'install_node_prebuilt\.py'
}, $true))
if ($functions.Count -ne 1 -or $assignments.Count -ne 1) { throw 'Node pipeline source shape changed' }
Invoke-Expression $functions[0].Extent.Text
$NodeInstallPython = $Python
$ProbeDir = Split-Path -Parent $ReportPath
$NodeDir = Join-Path $ProbeDir 'unused-install-target'
$pipeline = $assignments[0].Extent.Text.Replace('$PSScriptRoot\', '$ProbeDir\')
$nodeOut = $null
$nodeExit = $null
$errorId = $null
$errorMessage = $null
try {
    Invoke-Expression $pipeline
    $nodeExit = $LASTEXITCODE
} catch {
    $errorId = $_.FullyQualifiedErrorId
    $errorMessage = $_.Exception.Message
}
$result = @{
    ExitCode=$nodeExit; ErrorId=$errorId; ErrorMessage=$errorMessage
    Completed=($nodeOut -match 'synthetic helper completed')
    ShellVersion=$PSVersionTable.PSVersion.ToString()
}
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($ReportPath, ($result | ConvertTo-Json), $utf8)
"""


pytestmark = pytest.mark.skipif(os.name != "nt", reason = "Windows native stderr wrapper behavior")


@pytest.mark.parametrize(
    "shell,warning,exit_code",
    [
        ("powershell.exe", False, 0),
        ("powershell.exe", False, 7),
        pytest.param(
            "powershell.exe",
            True,
            0,
            marks = pytest.mark.xfail(
                strict = True,
                raises = _WarningPipelineAborted,
                reason = "Issue #13466: Windows PowerShell stops on the helper's harmless stderr",
            ),
        ),
        ("pwsh.exe", False, 0),
        ("pwsh.exe", False, 7),
        ("pwsh.exe", True, 0),
    ],
)
def test_node_helper_wrapper_preserves_exit_status(tmp_path, shell, warning, exit_code):
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is unavailable")
    source = _setup_source()
    probe = tmp_path / "probe.ps1"
    probe.write_text(PROBE, encoding = "utf-8")
    helper = "import sys, warnings\n"
    if warning:
        helper += "warnings.warn('synthetic harmless warning', UserWarning)\n"
    helper += f"print('synthetic helper completed', flush=True)\nraise SystemExit({exit_code})\n"
    (tmp_path / "install_node_prebuilt.py").write_text(helper, encoding = "utf-8")
    report = tmp_path / "report.json"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "PATH") if key in os.environ}
    env.update(
        PATHEXT = ".COM;.EXE;.BAT;.CMD",
        TEMP = str(tmp_path),
        TMP = str(tmp_path),
        PSModulePath = str(Path(executable).parent / "Modules"),
        PSMODULEANALYSISCACHEPATH = str(tmp_path / "module-cache"),
    )
    process = subprocess.run(
        [
            executable,
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(probe),
            "-SetupPath",
            str(source),
            "-Python",
            sys.executable,
            "-ReportPath",
            str(report),
        ],
        capture_output = True,
        text = True,
        env = env,
        cwd = tmp_path,
        timeout = 15,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(report.read_text(encoding = "utf-8"))
    if (
        warning
        and shell == "powershell.exe"
        and result["ErrorId"] == "NativeCommandError"
        and "synthetic harmless warning" in (result["ErrorMessage"] or "")
    ):
        raise _WarningPipelineAborted("Native stderr stopped the wrapper before exit capture")
    assert result["ErrorId"] is None, result
    assert result["Completed"], result
    assert result["ExitCode"] == exit_code, result
