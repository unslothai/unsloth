# Regression test for the setup.ps1 system-node/npm probes. Under "Stop", a bare
# `node -v` for an absent/broken node throws a terminating error `2>$null` cannot
# swallow, which used to abort setup before the bundled-Node decision. The probes
# are now guarded (Get-Command + try/catch); this runs the real probe lines with
# node/npm absent or throwing and asserts setup would NOT terminate.
$ErrorActionPreference = "Stop"
$script:failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$setupPath = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1"))).Path
# Match specifically the two system-version probe assignments (not every
# Get-Command node/npm in the file, e.g. the OXC-runtime npm guard).
$probeLines = (Get-Content $setupPath) | Where-Object {
    $_ -match '\$Sys(Node|Npm)Version = try \{ if \(Get-Command (node|npm) -ErrorAction SilentlyContinue'
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

# 3. Present-but-broken shim: Get-Command finds it but invoking it throws (corrupt
#    Node / blocked npm.ps1). The try/catch must still degrade to empty, not abort.
$throwShims = "function node { throw 'boom' }`nfunction npm { throw 'boom' }`n"
$broken = $throwShims + ($probeLines -join "`n") + "`nWrite-Output ""RESULT node=[`$SysNodeVersion] npm=[`$SysNpmVersion]"""
$b = Invoke-WithoutNode $broken
Check "guarded probes do not terminate when a present shim throws (exit 0)" ($b.ExitCode -eq 0)
Check "guarded probes yield empty versions when a present shim throws" ($b.Output -match 'RESULT node=\[\] npm=\[\]')

# 4. Negative control: the if-guard WITHOUT try/catch terminates when a present
#    command throws -- proves the try/catch (not just Get-Command) is load-bearing.
$brokenUnguarded = "function node { throw 'boom' }`n`$SysNodeVersion = if (Get-Command node -ErrorAction SilentlyContinue) { (node -v 2>`$null) } else { '' }`nWrite-Output ""REACHED"""
$bn = Invoke-WithoutNode $brokenUnguarded
Check "if-guard without try/catch terminates on a throwing present command (negative control)" ($bn.ExitCode -ne 0 -and $bn.Output -notmatch 'REACHED')

Remove-Item -Recurse -Force $emptyDir -ErrorAction SilentlyContinue

if ($script:failures -gt 0) { Write-Host "$($script:failures) check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed"
