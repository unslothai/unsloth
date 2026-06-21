# Regression test for the setup.ps1 system-node/npm probes.
#
# setup.ps1 runs under $ErrorActionPreference = "Stop". A bare `node -v` when node
# is absent throws a terminating CommandNotFoundException that `2>$null` does NOT
# swallow, which previously aborted setup on a fresh (no-Node) Windows machine
# before the bundled-Node decision could run. The probes are now guarded with
# Get-Command. This test extracts the real probe lines from setup.ps1 and runs
# them in a child pwsh with node/npm absent, asserting setup would NOT terminate.
$ErrorActionPreference = "Stop"
$script:failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$setupPath = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1"))).Path
$probeLines = (Get-Content $setupPath) | Where-Object {
    $_ -match 'Get-Command (node|npm) -ErrorAction SilentlyContinue'
}
Check "setup.ps1 guards both node and npm probes with Get-Command" ($probeLines.Count -eq 2)

# Resolve pwsh by absolute path BEFORE scrubbing PATH, so we can launch a child
# whose PATH has no node/npm while still invoking the interpreter.
$pwshExe = (Get-Command pwsh -ErrorAction SilentlyContinue).Source
if (-not $pwshExe) { $pwshExe = (Get-Command powershell).Source }
$emptyDir = Join-Path ([System.IO.Path]::GetTempPath()) ("uns_probe_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $emptyDir | Out-Null

function Invoke-WithoutNode([string]$body) {
    $script = "`$ErrorActionPreference = 'Stop'`n$body"
    $file = Join-Path $emptyDir ("probe_" + [guid]::NewGuid().ToString("N") + ".ps1")
    Set-Content -Path $file -Value $script -Encoding utf8
    $saved = $env:PATH
    try {
        $env:PATH = $emptyDir   # node/npm guaranteed absent for the child
        $out = & $pwshExe -NoProfile -File $file 2>&1 | Out-String
        $code = $LASTEXITCODE
    } finally {
        $env:PATH = $saved
    }
    return [pscustomobject]@{ ExitCode = $code; Output = $out }
}

# 1. The real guarded probes must NOT terminate, and must yield empty versions
#    (which Get-NodeDecision then maps to "bundled").
$guarded = ($probeLines -join "`n") + "`nWrite-Output ""RESULT node=[`$SysNodeVersion] npm=[`$SysNpmVersion]"""
$r = Invoke-WithoutNode $guarded
Check "guarded probes do not terminate when node is absent (exit 0)" ($r.ExitCode -eq 0)
Check "guarded probes yield empty node/npm versions" ($r.Output -match 'RESULT node=\[\] npm=\[\]')

# 2. Negative control: the OLD unguarded form DOES terminate -- proves this test
#    can actually distinguish the bug from the fix.
$unguarded = "`$SysNodeVersion = (node -v 2>`$null)`nWrite-Output ""REACHED"""
$n = Invoke-WithoutNode $unguarded
Check "unguarded bare probe terminates under Stop (negative control)" ($n.ExitCode -ne 0 -and $n.Output -notmatch 'REACHED')

Remove-Item -Recurse -Force $emptyDir -ErrorAction SilentlyContinue

if ($script:failures -gt 0) { Write-Host "$($script:failures) check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed"
