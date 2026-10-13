# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# setup.ps1 runs `npm run build` in the background on a source install (#8818). These run the
# real helpers (and the real Exit-SetupFailure) in child shells against a stand-in build: the
# join reports the build's exit code and output, a failure elsewhere kills the build's process
# tree and puts back what it hid while keeping the build mark, and the gate keeps Desktop,
# staged updates, verbose, opted-out and small machines serial.
$ErrorActionPreference = "Stop"
$script:failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setupPath = [System.IO.Path]::Combine($repoRoot, "studio", "setup.ps1")
. ([System.IO.Path]::Combine($repoRoot, "tests", "studio_setup_ps1", "Get-FunctionSource.ps1"))

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("fe-overlap-" + [guid]::NewGuid().ToString("N"))
$fe = Join-Path $work "outer/studio/frontend"
New-Item -ItemType Directory -Force -Path (Join-Path $fe "dist/assets") | Out-Null
$helpers = Join-Path $work "helpers.ps1"
$src = @()
foreach ($fn in @("New-UnslothTemporaryFile", "Exit-SetupFailure", "Test-FrontendBuildInBackground",
                  "Start-FrontendBuildBackground", "Restore-FrontendBuildGitignores",
                  "Remove-FrontendBuildFiles", "Stop-FrontendBuildBackground",
                  "Complete-FrontendBuildOutput", "Complete-FrontendBuildBackground", "Wait-FrontendBuildOwner")) {
    $body = Get-FunctionSource -Path $setupPath -Name $fn
    Check "setup.ps1 defines $fn" ($null -ne $body)
    $src += $body
}
$prelude = @'
$ErrorActionPreference = "Stop"
$script:UnslothVerbose = $false
$script:FrontendBuildProc = $null
$script:FrontendBuildFiles = @()
$script:FrontendBuildHidden = @()
function step($a, $b, $c) { Write-Output "step $a $b" }
function substep($a, $b) { Write-Output "substep $a" }
function Write-StudioLine($t) { Write-Output "line $t" }
function Redact-InstallOutput($t) { $t }
function Get-UsableMemoryMb { if ($env:FAKE_MB -eq "throw") { throw "no probe" }; [long]$env:FAKE_MB }
'@
Set-Content -LiteralPath $helpers -Value ($prelude + "`n" + ($src -join "`n")) -Encoding utf8
$trapLine = @(Get-Content -LiteralPath $setupPath | Where-Object { $_ -like 'trap { if ($script:FrontendBuildProc)*' })
Check "setup.ps1 has the frontend build trap" ($trapLine.Count -eq 1)
$pwshExe = (Get-Process -Id $PID).Path
$mark = Join-Path $fe ".unsloth-build.local"

function Invoke-Case([string]$Body) {
    $script = Join-Path $work ("case-" + [guid]::NewGuid().ToString("N") + ".ps1")
    $head = ". '$helpers'`n`$DistDir = '$fe/dist'`n`$FrontendBuildMark = '$mark'`n`$Build = '$pwshExe'`n"
    Set-Content -LiteralPath $script -Value ($head + $Body) -Encoding utf8
    $out = & $pwshExe -NoProfile -File $script 2>&1 | Out-String
    return @{ Out = $out; Code = $LASTEXITCODE }
}
function Reset-Tree {
    Set-Content -LiteralPath (Join-Path $work "outer/.gitignore._twbuild") -Value "*"
    Remove-Item -LiteralPath (Join-Path $work "outer/.gitignore") -Force -ErrorAction SilentlyContinue
    Set-Content -LiteralPath $mark -Value "1"
    Remove-Item -LiteralPath (Join-Path $fe "dist/assets/app.css") -Force -ErrorAction SilentlyContinue
}
$hidden = "@('" + (Join-Path $work "outer/.gitignore") + "')"

Write-Host "== a successful background build is joined like the serial one =="
Reset-Tree
$r = Invoke-Case @"
`$ok = Start-FrontendBuildBackground -WorkingDirectory '$fe' -HiddenGitignores $hidden -FilePath `$Build -ArgumentList @('-NoProfile', '-Command', "Start-Sleep 1; Set-Content -LiteralPath '$fe/dist/assets/app.css' -Value ('x' * 200000)")
Write-Output "started=`$ok"
Write-Output "parent-step"
Complete-FrontendBuildBackground
Write-Output "after-join"
"@
Check "build started in the background" ($r.Out -match "started=True")
Check "join prints the built line" ($r.Out -match "step frontend built")
Check "join returns and setup continues" ($r.Out -match "after-join" -and $r.Code -eq 0)
Check "hidden .gitignore restored" (Test-Path -LiteralPath (Join-Path $work "outer/.gitignore"))
Check "build mark cleared after success" (-not (Test-Path -LiteralPath $mark))

Write-Host "== a failed background build fails setup like the serial one =="
Reset-Tree
$r = Invoke-Case @"
`$null = Start-FrontendBuildBackground -WorkingDirectory '$fe' -HiddenGitignores $hidden -FilePath `$Build -ArgumentList @('-NoProfile', '-Command', "Write-Output 'tsc says boom'; exit 7")
Complete-FrontendBuildBackground
Write-Output "not-reached"
"@
Check "setup exits 1, as the serial path does" ($r.Code -eq 1)
Check "the build output is shown" ($r.Out -match "tsc says boom")
Check "the error names the build exit code" ($r.Out -match "npm run build failed \(exit code 7\)")
Check "setup stops" ($r.Out -notmatch "not-reached")
Check "hidden .gitignore restored" (Test-Path -LiteralPath (Join-Path $work "outer/.gitignore"))
Check "build mark kept, so the next run rebuilds" (Test-Path -LiteralPath $mark)

Write-Host "== -IfDone leaves a running build alone and joins a finished one =="
Reset-Tree
$r = Invoke-Case @"
`$null = Start-FrontendBuildBackground -WorkingDirectory '$fe' -FilePath `$Build -ArgumentList @('-NoProfile', '-Command', "Start-Sleep 2; Set-Content -LiteralPath '$fe/dist/assets/app.css' -Value ('x' * 200000)")
Complete-FrontendBuildBackground -IfDone
Write-Output "running-after-first=`$([bool]`$script:FrontendBuildProc)"
Start-Sleep 4
Complete-FrontendBuildBackground -IfDone
Write-Output "running-after-second=`$([bool]`$script:FrontendBuildProc)"
"@
Check "running build is left alone" ($r.Out -match "running-after-first=True")
Check "finished build is joined" ($r.Out -match "running-after-second=False" -and $r.Out -match "step frontend built")

Write-Host "== a failure elsewhere kills the whole build tree before setup exits =="
Reset-Tree
$pidFile = Join-Path $work "grandchild.pid"
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
$sleeper = if ($IsWindows) { "Start-Process -PassThru -NoNewWindow -FilePath ''$pwshExe'' -ArgumentList ''-NoProfile'',''-Command'',''Start-Sleep 300''" } else { "Start-Process -PassThru -FilePath ''sleep'' -ArgumentList ''300''" }
$r = Invoke-Case @"
`$null = Start-FrontendBuildBackground -WorkingDirectory '$fe' -HiddenGitignores $hidden -FilePath `$Build -ArgumentList @('-NoProfile', '-Command', '`$p = $sleeper; Set-Content -LiteralPath ''$pidFile'' -Value `$p.Id; Start-Sleep 300')
Start-Sleep 3
Exit-SetupFailure "python deps failed" -Code 3
"@
Check "setup exits with its own code" ($r.Code -eq 3)
$gc = if (Test-Path -LiteralPath $pidFile) { [int](Get-Content -LiteralPath $pidFile) } else { 0 }
$alive = $false
if ($gc) { $alive = $null -ne (Get-Process -Id $gc -ErrorAction SilentlyContinue | Where-Object { -not $_.HasExited }) }
Check "the build's child process is gone" ($gc -ne 0 -and -not $alive)
if ($alive) { Stop-Process -Id $gc -Force -ErrorAction SilentlyContinue }
Check "hidden .gitignore restored" (Test-Path -LiteralPath (Join-Path $work "outer/.gitignore"))
Check "build mark kept, so the next run rebuilds" (Test-Path -LiteralPath $mark)

Write-Host "== a terminating error anywhere stops the build too =="
Reset-Tree
Remove-Item -LiteralPath $pidFile -Force -ErrorAction SilentlyContinue
$r = Invoke-Case @"
$($trapLine[0])
`$null = Start-FrontendBuildBackground -WorkingDirectory '$fe' -HiddenGitignores $hidden -FilePath `$Build -ArgumentList @('-NoProfile', '-Command', '`$p = $sleeper; Set-Content -LiteralPath ''$pidFile'' -Value `$p.Id; Start-Sleep 300')
Start-Sleep 3
if (`$true) { throw "unexpected" }
Write-Output "not-reached"
"@
Check "setup still fails" ($r.Code -ne 0 -and $r.Out -notmatch "not-reached")
$gc = if (Test-Path -LiteralPath $pidFile) { [int](Get-Content -LiteralPath $pidFile) } else { 0 }
$alive = $false
if ($gc) { $alive = $null -ne (Get-Process -Id $gc -ErrorAction SilentlyContinue | Where-Object { -not $_.HasExited }) }
Check "the build's child process is gone" ($gc -ne 0 -and -not $alive)
if ($alive) { Stop-Process -Id $gc -Force -ErrorAction SilentlyContinue }
Check "hidden .gitignore restored" (Test-Path -LiteralPath (Join-Path $work "outer/.gitignore"))

Write-Host "== a rerun waits for a build its killed setup left running =="
$r = Invoke-Case @"
`$null = Start-FrontendBuildBackground -WorkingDirectory '$fe' -FilePath `$Build -ArgumentList @('-NoProfile', '-Command', 'Start-Sleep 4')
Write-Output "owner=`$(`$script:FrontendBuildProc.Id)"
"@
$ownerId = ([regex]::Match($r.Out, "owner=(\d+)")).Groups[1].Value
Check "the marker names the build process" ((Get-Content -LiteralPath $mark -Raw).Trim() -like "$ownerId *")
$sw = [Diagnostics.Stopwatch]::StartNew()
$r = Invoke-Case "Wait-FrontendBuildOwner"
Check "waited for the orphaned build" ($sw.Elapsed.TotalSeconds -ge 2)
Check "says why it waits" ($r.Out -match "waiting for an earlier frontend build")
$sw.Restart()
$null = Invoke-Case "Wait-FrontendBuildOwner"
Check "a finished build is not waited for" ($sw.Elapsed.TotalSeconds -lt 2)
Set-Content -LiteralPath $mark -Value "$PID 123"
$sw.Restart()
$null = Invoke-Case "Wait-FrontendBuildOwner"
Check "a reused process id is not waited for" ($sw.Elapsed.TotalSeconds -lt 2)

Write-Host "== gate =="
function Gate([hashtable]$Env, [string]$Extra = "") {
    $saved = @{}
    foreach ($k in $Env.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, $Env[$k]) }
    try { $r = Invoke-Case ($Extra + "`nif (Test-FrontendBuildInBackground) { 'gate=yes' } else { 'gate=no' }") }
    finally { foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
    if ($r.Out -match "gate=yes") { return "yes" } else { return "no" }
}
$base = @{ FAKE_MB = "65536"; UNSLOTH_STUDIO_SERIAL_SETUP = $null; UNSLOTH_TAURI_MODE = $null; UNSLOTH_TAURI_UPDATE = $null }
$big = [Environment]::ProcessorCount -ge 4
if ($big) { Check "4+ CPUs, 64 GiB usable overlaps" ((Gate $base) -eq "yes") }
if ($big) { Check "12 GiB usable overlaps" ((Gate ($base + @{}) ) -eq "yes" -and (Gate (@{ FAKE_MB = "12288" })) -eq "yes") }
Check "8 GiB usable stays serial" ((Gate @{ FAKE_MB = "8192" }) -eq "no")
Check "a failing memory probe stays serial" ((Gate @{ FAKE_MB = "throw" }) -eq "no")
Check "an unknown memory reading (-1) stays serial" ((Gate @{ FAKE_MB = "-1" }) -eq "no")
Check "opt-out stays serial" ((Gate @{ FAKE_MB = "65536"; UNSLOTH_STUDIO_SERIAL_SETUP = "1" }) -eq "no")
Check "opt-out (true) stays serial" ((Gate @{ FAKE_MB = "65536"; UNSLOTH_STUDIO_SERIAL_SETUP = "true" }) -eq "no")
Check "Desktop install stays serial" ((Gate @{ FAKE_MB = "65536"; UNSLOTH_TAURI_MODE = "1" }) -eq "no")
Check "Desktop update stays serial" ((Gate @{ FAKE_MB = "65536"; UNSLOTH_TAURI_UPDATE = "true" }) -eq "no")
Check "verbose stays serial" ((Gate @{ FAKE_MB = "65536" } '$script:UnslothVerbose = $true') -eq "no")
Check "staged update stays serial" ((Gate @{ FAKE_MB = "65536" } '$StageRoot = "C:\stage"') -eq "no")

Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
if ($script:failures -gt 0) { Write-Host "$($script:failures) check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed"
