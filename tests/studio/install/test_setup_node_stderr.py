# SPDX-License-Identifier: AGPL-3.0-only
"""Exercise the Node helper's real PowerShell wrapper with a download-free stub."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import json

import pytest


def _setup_source():
    configured = os.environ.get("UNSLOTH_REPO_ROOT")
    roots = [Path(configured)] if configured else []
    roots.extend(Path(__file__).resolve().parents)
    for root in roots:
        source = root / "studio/setup.ps1"
        if source.is_file():
            return source
    pytest.fail("Could not locate studio/setup.ps1")


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
$source = [IO.File]::ReadAllText($SetupPath)
$start = $source.IndexOf('        $NodeInstallPython = if ($ValidatedSetupPython)')
$end = $source.IndexOf('        if ($nodeExit -eq 3)', $start)
if ($functions.Count -ne 1 -or $start -lt 0 -or $end -le $start) {
    throw 'Node invocation source shape changed'
}
Invoke-Expression $functions[0].Extent.Text
$ValidatedSetupPython = $Python
$ProbeDir = Split-Path -Parent $ReportPath
$NodeDir = Join-Path $ProbeDir 'unused-install-target'
$pipeline = $source.Substring($start, $end - $start).Replace('$PSScriptRoot\', '$ProbeDir\')
$script:MirrorCalls = 0
function Use-MirrorSpare {
    param($Name)
    if ($ErrorActionPreference -ne 'Stop') { throw 'Error preference leaked into mirror selection' }
    $script:MirrorCalls++
    return ($env:PROBE_RETRY -eq '1')
}
$PSNativeCommandUseErrorActionPreference = ($env:PROBE_NATIVE_ERRORS -eq '1')
$nodeOut = $null
$nodeExit = $null
$errorId = $null
$errorMessage = $null
try {
    Invoke-Expression $pipeline
} catch {
    $errorId = $_.FullyQualifiedErrorId
    $errorMessage = $_.Exception.Message
}
$result = @{
    ExitCode=$nodeExit; ErrorId=$errorId; ErrorMessage=$errorMessage
    Completed=($nodeOut -match 'synthetic helper completed')
    ShellVersion=$PSVersionTable.PSVersion.ToString()
    Output=$nodeOut; Preference="$ErrorActionPreference"; MirrorCalls=$script:MirrorCalls
    NativePreference=$PSNativeCommandUseErrorActionPreference
}
$utf8 = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($ReportPath, ($result | ConvertTo-Json), $utf8)
"""


pytestmark = pytest.mark.skipif(os.name != "nt", reason = "Windows native stderr wrapper behavior")


@pytest.mark.parametrize("shell", ["powershell.exe", "pwsh.exe"])
@pytest.mark.parametrize("native_errors", [False, True])
@pytest.mark.parametrize(
    "warning,exits,retry",
    [
        (False, [], False),  # Missing executable must still fail and restore Stop.
        (False, [0], False),
        (True, [0], False),
        (False, [7], False),
        (True, [7], False),
        (True, [3], True),
        (True, [4], True),
        (True, [7, 0], True),
        (True, [7, 7], True),
    ],
)
def test_node_helper_wrapper_preserves_exit_status(
    tmp_path, shell, native_errors, warning, exits, retry
):
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is unavailable")
    source = _setup_source()
    probe = tmp_path / "probe.ps1"
    probe_source = PROBE
    if not exits:
        probe_source = probe_source.replace(
            "$ValidatedSetupPython = $Python",
            "$ValidatedSetupPython = Join-Path (Split-Path -Parent $ReportPath) 'missing-python.exe'",
        )
    probe.write_text(probe_source, encoding = "utf-8")
    helper = "import sys, warnings\nfrom pathlib import Path\n"
    helper += "counter = Path(__file__).with_suffix('.count')\n"
    helper += "attempt = int(counter.read_text()) if counter.exists() else 0\n"
    helper += "counter.write_text(str(attempt + 1))\n"
    if warning:
        helper += "warnings.warn('synthetic harmless warning', UserWarning)\n"
    helper += "print('synthetic helper completed', flush=True)\n"
    helper += f"raise SystemExit({exits!r}[attempt])\n"
    (tmp_path / "install_node_prebuilt.py").write_text(helper, encoding = "utf-8")
    report = tmp_path / "report.json"
    env = {key: os.environ[key] for key in ("SystemRoot", "WINDIR", "PATH") if key in os.environ}
    env.update(
        PATHEXT = ".COM;.EXE;.BAT;.CMD",
        TEMP = str(tmp_path),
        TMP = str(tmp_path),
        PSModulePath = str(Path(executable).parent / "Modules"),
        PSMODULEANALYSISCACHEPATH = str(tmp_path / "module-cache"),
        PROBE_RETRY = "1" if retry else "0",
        PROBE_NATIVE_ERRORS = "1" if native_errors else "0",
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
    if not exits:
        assert "CommandNotFoundException" in (result["ErrorId"] or ""), result
        assert result["ExitCode"] is None, result
        assert result["Preference"] == "Stop", result
        return
    assert result["ErrorId"] is None, result
    assert result["Completed"], result
    assert result["ExitCode"] == exits[-1], result
    assert result["Preference"] == "Stop", result
    assert result["NativePreference"] is native_errors, result
    assert result["MirrorCalls"] == int(exits[0] not in (0, 3, 4)), result
    assert int((tmp_path / "install_node_prebuilt.count").read_text()) == len(exits)
    if warning:
        assert "synthetic harmless warning" in result["Output"], result
