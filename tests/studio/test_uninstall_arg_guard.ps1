#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Argument contract for scripts/uninstall.ps1. Tests run against a temporary
# copy that aborts before the first uninstall action, so a broken guard cannot
# perform a real uninstall.
#
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_arg_guard.ps1

$ErrorActionPreference = "Stop"
$sourceUninstallPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "scripts", "uninstall.ps1")
$sourceUninstallPath = (Resolve-Path $sourceUninstallPath).Path
$pwshPath = (Get-Process -Id $PID).Path
$bodyEntry = '    _Step "Stopping any running Unsloth Studio servers..."'
$bodyMarker = "__UNSLOTH_TEST_BODY_REACHED__"

$source = Get-Content -Raw -LiteralPath $sourceUninstallPath
$entryMatches = [regex]::Matches($source, [regex]::Escape($bodyEntry))
if ($entryMatches.Count -ne 1) {
    throw "Expected exactly one uninstall body entry point, found $($entryMatches.Count)"
}
$safeSource = $source.Replace($bodyEntry, "    throw `"$bodyMarker`"")
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-uninstall-guard-" + [guid]::NewGuid().ToString("N"))
$uninstallPath = Join-Path $tempRoot "uninstall.ps1"
New-Item -ItemType Directory -Path $tempRoot | Out-Null
try {
    [System.IO.File]::WriteAllText(
        $uninstallPath,
        $safeSource,
        [System.Text.UTF8Encoding]::new($false)
    )
} catch {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
    throw
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Windows PowerShell 5.1 turns a native command's stderr into a terminating
# NativeCommandError under $ErrorActionPreference = "Stop"; 7.1+ does not. This
# suite can run under 5.1, where every rejected-argument case writes to stderr.
function Invoke-Native([scriptblock]$Command) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { return (& $Command | Out-String) } finally { $ErrorActionPreference = $prev }
}

function Invoke-Uninstaller([string[]]$ScriptArgs) {
    $argv = @("-NoProfile", "-File", $uninstallPath) + $ScriptArgs
    $out = Invoke-Native { & $pwshPath @argv 2>&1 }
    return [pscustomobject]@{ Code = $LASTEXITCODE; Output = $out }
}

try {
    foreach ($parseCase in @(
        @{ Name = "source uninstall.ps1"; Path = $sourceUninstallPath },
        @{ Name = "instrumented uninstall.ps1"; Path = $uninstallPath }
    )) {
        $parsePath = $parseCase.Path
        $tokens = $null; $errors = $null
        [System.Management.Automation.Language.Parser]::ParseFile($parsePath, [ref]$tokens, [ref]$errors) | Out-Null
        Check "$($parseCase.Name) parses" (-not $errors)
        if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "uninstall.ps1 has parse errors" }
    }

    # Prove instrumentation did not break the normal entry path.
    $r = Invoke-Uninstaller @()
    Check "no arguments reach the instrumented body" ($r.Output -match $bodyMarker)
    Check "the instrumented body aborts"              ($r.Code -ne 0)

    Write-Host "help flags print usage and start nothing"
    foreach ($flag in @("-Help", "-h", "-help", "--help", "-?", "/?")) {
        $r = Invoke-Uninstaller @($flag)
        Check "$flag exits 0"                    ($r.Code -eq 0)
        Check "$flag prints usage"               ($r.Output -match "Unsloth Studio uninstaller")
        Check "$flag never starts the uninstall" ($r.Output -notmatch $bodyMarker)
    }

    Write-Host "unknown arguments abort before the body runs"
    foreach ($bad in @("--dry-run", "-n", "--version", "uninstall")) {
        $r = Invoke-Uninstaller @($bad)
        Check "'$bad' exits nonzero"               ($r.Code -ne 0)
        Check "'$bad' never starts the uninstall"  ($r.Output -notmatch $bodyMarker)
        Check "'$bad' is named in the error"        ($r.Output -match [regex]::Escape("unrecognized argument: $bad"))
    }

    $r = Invoke-Uninstaller @("--dry-run")
    Check "the error states nothing was removed" ($r.Output -match "Nothing was removed")

    $r = Invoke-Uninstaller @("--dry-run", "--help")
    Check "'--dry-run --help' exits nonzero"               ($r.Code -ne 0)
    Check "'--dry-run --help' never starts the uninstall"  ($r.Output -notmatch $bodyMarker)

    # Embedded invocation must report failure without exiting the caller.
    $probe = @'
$s = Get-Content -Raw '__PATH__'
try { & ([scriptblock]::Create($s)) --dry-run } catch { Write-Host $_.Exception.Message }
Write-Host 'SESSION SURVIVED'
'@ -replace '__PATH__', $uninstallPath
    $out = Invoke-Native { & $pwshPath -NoProfile -Command $probe 2>&1 }
    Check "a bad argument does not kill the caller's session" ($out -match "SESSION SURVIVED")
    Check "the scriptblock flow never starts the uninstall"   ($out -notmatch $bodyMarker)
} finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
