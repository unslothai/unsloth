#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The install lock must exclude two installers that reach one destination by different spellings.
#
# The named mutex alone cannot do that. Its name is a hash of the resolved path, so whenever the
# path resolver cannot give an exact answer the hash falls back to the normalised spelling, and a
# junction and its target produce two different mutex names that do not exclude each other. That
# is written down at install.ps1's Get-StudioPathHash. A lock file held inside the destination has
# no such failure mode: every alias reaches the same file because the filesystem resolves the
# alias, with no canonicalisation involved.
#
# Only the FILE half is exercised here, with the mutex stubbed out. The mutex half is unchanged by
# this work and is covered by tests/python/test_windows_installer_concurrency_guard.py; mixing the
# two in would mean dragging in the whole path-resolver apparatus to compute a name this test does
# not care about.
# Run: pwsh -NoProfile -File tests/studio/test_install_lock_excludes_aliases.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPs1 = Join-Path $root "install.ps1"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Get-HelperSources($path, $names) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    $out = @()
    foreach ($name in $names) {
        $fn = $ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
        }, $true)
        if ($fn.Count -lt 1) { throw "expected $name in $path, found none" }
        $out += $fn[0].Extent.Text
    }
    return $out
}

# The lock file's name is read out of install.ps1 rather than repeated here, so renaming it there
# without updating the uninstaller or this test shows up as a failure instead of a silent pass.
$installText = Get-Content -Raw -LiteralPath $installPs1
$nameMatch = [regex]::Match($installText,
    '\$script:StudioInstallLockFileName\s*=\s*"([^"]+)"')
if (-not $nameMatch.Success) { throw "could not find `$script:StudioInstallLockFileName in install.ps1" }
$lockFileName = $nameMatch.Groups[1].Value
Write-Host "  lock file name: $lockFileName"

# Stubs for the mutex half. Always granting it isolates the file half: every refusal this test
# observes is the file lock's doing, never the mutex's.
function Enter-StudioInstallMutex { param([string]$Path) return "stub-mutex" }
function Exit-StudioInstallMutex { param($Mutex) }

foreach ($src in (Get-HelperSources $installPs1 @("Enter-StudioInstallLock", "Exit-StudioInstallLock"))) {
    Invoke-Expression $src
}
$script:StudioInstallLockFileName = $lockFileName

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("instlock-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

# A child process is the only honest way to test exclusion: .NET tracks handles per process, so a
# same-process second open would be refused even if the OS were not enforcing anything.
$holderPs1 = Join-Path $tmp "holder.ps1"
@"
param([string]`$LockPath, [string]`$ReadyFile, [int]`$HoldSeconds = 25)
`$s = [System.IO.File]::Open(`$LockPath, [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
New-Item -ItemType File -Force -Path `$ReadyFile | Out-Null
Start-Sleep -Seconds `$HoldSeconds
`$s.Dispose()
"@ | Set-Content -LiteralPath $holderPs1 -Encoding UTF8

function Start-Holder($lockPath) {
    $ready = Join-Path $tmp ("ready-" + [guid]::NewGuid().ToString("N"))
    $exe = (Get-Process -Id $PID).Path
    # -WindowStyle is not supported by pwsh off Windows, so only ask for it where it exists.
    $startArgs = @{
        FilePath = $exe
        ArgumentList = @("-NoProfile", "-File", $holderPs1, $lockPath, $ready)
        PassThru = $true
    }
    if ($IsWindows -or $env:OS -eq "Windows_NT") { $startArgs["WindowStyle"] = "Hidden" }
    $proc = Start-Process @startArgs
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    while (-not (Test-Path -LiteralPath $ready) -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Milliseconds 100
    }
    if (-not (Test-Path -LiteralPath $ready)) {
        try { $proc.Kill() } catch {}
        throw "the holder process never took the lock"
    }
    return $proc
}

function Stop-Holder($proc) {
    try { if (-not $proc.HasExited) { $proc.Kill() } } catch {}
    try { $proc.WaitForExit(10000) | Out-Null } catch {}
}

try {
    $real = Join-Path $tmp "real"
    New-Item -ItemType Directory -Force -Path $real | Out-Null

    # 1. A destination that does not exist yet is created by taking the lock, since the installer
    #    locks before it has written anything.
    $fresh = Join-Path $tmp "not-created-yet"
    $freshLock = Enter-StudioInstallLock -Path $fresh
    Check "a destination that does not exist is created" (Test-Path -LiteralPath $fresh)
    Check "the lock is granted on a fresh destination" ($null -ne $freshLock)
    Check "the lock file lands inside the destination" (
        Test-Path -LiteralPath (Join-Path $fresh $lockFileName))
    Check "a root the lock created identifies itself to the uninstaller" (
        Test-Path -LiteralPath (Join-Path $fresh ".unsloth-studio-owned") -PathType Leaf)
    Exit-StudioInstallLock -Lock $freshLock
    Check "releasing does not delete the lock file" (
        Test-Path -LiteralPath (Join-Path $fresh $lockFileName))

    # The safety half of that marker. A UNSLOTH_STUDIO_HOME pointed at a directory the user
    # already had must never be claimed as ours, because scripts/uninstall.ps1 deletes what the
    # marker names. Only a directory this lock created may carry it.
    $preexisting = Join-Path $tmp "users-own-directory"
    New-Item -ItemType Directory -Force -Path $preexisting | Out-Null
    "keep me" | Set-Content -LiteralPath (Join-Path $preexisting "important.txt")
    $preLock = Enter-StudioInstallLock -Path $preexisting
    Check "an existing directory is NOT claimed as an Unsloth root" (
        -not (Test-Path -LiteralPath (Join-Path $preexisting ".unsloth-studio-owned")))
    Check "an existing directory's contents are untouched" (
        (Get-Content -Raw -LiteralPath (Join-Path $preexisting "important.txt")).Trim() -eq "keep me")
    Exit-StudioInstallLock -Lock $preLock

    # 2. Released means re-takeable. A lock that leaked would block every later run.
    $again = Enter-StudioInstallLock -Path $fresh
    Check "the lock can be taken again after release" ($null -ne $again)
    Exit-StudioInstallLock -Lock $again

    # 3. The control. Two genuinely different destinations must NOT exclude each other, or every
    #    check below would pass for the trivial reason that the lock always refuses.
    $other = Join-Path $tmp "other"
    New-Item -ItemType Directory -Force -Path $other | Out-Null
    $holder = Start-Holder (Join-Path $real $lockFileName)
    try {
        $otherLock = Enter-StudioInstallLock -Path $other
        Check "a different destination is not blocked (control)" ($null -ne $otherLock)
        if ($otherLock) { Exit-StudioInstallLock -Lock $otherLock }

        # 4. Same destination, same spelling: refused.
        $sameLock = Enter-StudioInstallLock -Path $real
        Check "the same destination is refused while held" ($null -eq $sameLock)
        if ($sameLock) { Exit-StudioInstallLock -Lock $sameLock }

        # 5. The point of the change. Same destination reached through an ALIAS: also refused.
        #    A junction on Windows and a symlink elsewhere; neither needs elevation for a
        #    directory junction, and the symlink path is what the Linux lane exercises.
        $alias = Join-Path $tmp "alias"
        $madeAlias = $false
        try {
            if ($IsWindows -or $env:OS -eq "Windows_NT") {
                New-Item -ItemType Junction -Path $alias -Target $real -ErrorAction Stop | Out-Null
            } else {
                New-Item -ItemType SymbolicLink -Path $alias -Target $real -ErrorAction Stop | Out-Null
            }
            $madeAlias = $true
        } catch {
            Write-Host "  SKIP  could not create a directory alias: $($_.Exception.Message)" -ForegroundColor Yellow
        }
        if ($madeAlias) {
            # Guard against a vacuous pass: if the alias does not actually reach the same
            # directory, the refusal below would prove nothing.
            $probe = Join-Path $real ("probe-" + [guid]::NewGuid().ToString("N"))
            New-Item -ItemType File -Force -Path $probe | Out-Null
            $viaAlias = Join-Path $alias (Split-Path -Leaf $probe)
            Check "the alias really reaches the same directory" (Test-Path -LiteralPath $viaAlias)
            Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue

            $aliasLock = Enter-StudioInstallLock -Path $alias
            Check "an ALIAS of the held destination is refused" ($null -eq $aliasLock)
            if ($aliasLock) { Exit-StudioInstallLock -Lock $aliasLock }

            # Proof that the check above is not vacuous. Get-StudioPathHash falls back to hashing
            # the normalised SPELLING when the resolver cannot answer exactly, and these two
            # spellings differ, so the mutex name derived from them differs too and the mutex
            # would have let both installers through. That is the regression this file exists to
            # prevent, restated as an assertion rather than as a claim in a comment.
            $spellReal = [System.IO.Path]::GetFullPath($real).ToUpperInvariant()
            $spellAlias = [System.IO.Path]::GetFullPath($alias).ToUpperInvariant()
            Check "the two spellings differ, so a name derived from them cannot exclude (bites)" (
                $spellReal -ne $spellAlias)
        }
    } finally {
        Stop-Holder $holder
    }

    # 6. A holder that dies without releasing must not wedge the destination: the OS closes the
    #    handle, so the next run takes the lock. This is why the file is not deleted on release.
    $crashDir = Join-Path $tmp "crashed"
    New-Item -ItemType Directory -Force -Path $crashDir | Out-Null
    $crashHolder = Start-Holder (Join-Path $crashDir $lockFileName)
    Stop-Holder $crashHolder
    $afterCrash = $null
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while ($null -eq $afterCrash -and [DateTime]::UtcNow -lt $deadline) {
        $afterCrash = Enter-StudioInstallLock -Path $crashDir
        if ($null -eq $afterCrash) { Start-Sleep -Milliseconds 200 }
    }
    Check "a killed holder leaves no stale lock" ($null -ne $afterCrash)
    if ($afterCrash) { Exit-StudioInstallLock -Lock $afterCrash }
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All install-lock alias checks passed"
exit 0
