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


def _run_node_probe(
    tmp_path,
    shell,
    helper,
    probe_source,
    retry = False,
    native_errors = False,
):
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is unavailable")
    source = _setup_source()
    probe = tmp_path / "probe.ps1"
    probe.write_text(probe_source, encoding = "utf-8")
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
        timeout = 120,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    return json.loads(report.read_text(encoding = "utf-8")), process.stdout


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
    probe_source = PROBE
    if not exits:
        probe_source = probe_source.replace(
            "$ValidatedSetupPython = $Python",
            "$ValidatedSetupPython = Join-Path (Split-Path -Parent $ReportPath) 'missing-python.exe'",
        )
    helper = "import sys, warnings\nfrom pathlib import Path\n"
    helper += "counter = Path(__file__).with_suffix('.count')\n"
    helper += "attempt = int(counter.read_text()) if counter.exists() else 0\n"
    helper += "counter.write_text(str(attempt + 1))\n"
    if warning:
        helper += "warnings.warn('synthetic harmless warning', UserWarning)\n"
    helper += "print('synthetic helper completed', flush=True)\n"
    helper += f"raise SystemExit({exits!r}[attempt])\n"
    result, _ = _run_node_probe(tmp_path, shell, helper, probe_source, retry, native_errors)
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


@pytest.mark.parametrize("shell", ["powershell.exe", "pwsh.exe"])
@pytest.mark.parametrize(
    "exit_code,diagnostic,route",
    [
        (0, "ordinary success", ""),
        (0, "keeping existing isolated Node; takeown /F repair", ""),
        (1, "download failed", "failure:1"),
        (2, "prebuilt unavailable", "failure:1"),
        (3, "install busy", "failure:3"),
        (4, "denied-scope: parent", "denied:parent"),
        (4, "denied-scope: install-dir", "denied:install"),
    ],
)
def test_node_setup_routes_failure_and_preserves_progress(
    tmp_path, shell, exit_code, diagnostic, route
):
    probe = PROBE.replace(
        "'        if ($nodeExit -eq 3)'",
        "'        # Same condition as the guard above'",
    ).replace("$env:UNSLOTH_TAURI_UPDATE = $null", "$env:UNSLOTH_TAURI_UPDATE = '1'")
    probe = probe.replace(
        "$ValidatedSetupPython = $Python",
        r"""
$writer = $ast.Find({
    param($node)
    $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $node.Name -eq 'Write-StudioLine'
}, $true)
Invoke-Expression $writer.Extent.Text
$script:StudioStdoutRedirected = $true
$script:Route = ''
$script:StepText = ''
$NodeParent = 'parent'
function step { param($Name, $Message, $Color); $script:StepText = $Message }
function Exit-SetupFailure {
    param($Message, $Code = 1)
    if ($ErrorActionPreference -ne 'Stop') { throw 'Preference leaked into failure handler' }
    $script:Route = "failure:$Code"
    throw 'expected routed failure'
}
function Exit-PathAccessDenied {
    param($Path, $Label, [switch]$OwnershipUnverified)
    if ($ErrorActionPreference -ne 'Stop') { throw 'Preference leaked into denial handler' }
    $script:Route = if ($Path -eq $NodeParent) { 'denied:parent' } else { 'denied:install' }
    throw 'expected routed failure'
}
$ValidatedSetupPython = $Python
""",
    ).replace(
        "Output=$nodeOut;", "Route=$script:Route; StepText=$script:StepText; Output=$nodeOut;"
    )
    progress = "Downloading runtime.zip: 25.0% (1.0 MiB/4.0 MiB) at 1.0 MiB/s"
    helper = (
        "import sys, warnings\n"
        f"print({progress!r}, flush=True)\n"
        "warnings.warn('synthetic harmless warning', UserWarning)\n"
        f"print({diagnostic!r}, flush=True)\n"
        f"raise SystemExit({exit_code})\n"
    )
    result, stdout = _run_node_probe(tmp_path, shell, helper, probe)
    assert result["Route"] == route, result
    assert result["ExitCode"] == exit_code, result
    assert result["Preference"] == "Stop", result
    assert progress in stdout
    assert diagnostic in result["Output"], result
    if route:
        assert result["ErrorMessage"] == "expected routed failure", result
        assert diagnostic in stdout
    else:
        assert result["ErrorId"] is None, result
    if "keeping existing" in diagnostic:
        assert result["StepText"] == "update not applied, existing isolated Node kept", result
        assert "takeown /F repair" in stdout
