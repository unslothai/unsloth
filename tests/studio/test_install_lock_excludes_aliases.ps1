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

foreach ($src in (Get-HelperSources $installPs1 @(
    "Enter-StudioInstallLock", "Exit-StudioInstallLock",
    "Test-StudioPlainFile", "Write-StudioRootOwnerMarker"))) {
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

    # 6. A planted link at the lock path must not have its target touched. File.Open follows a
    #    symbolic or hard link, so anything written through that handle lands in the link's
    #    TARGET. Another user who can write into a shared or custom root could then have an
    #    arbitrary writable file emptied merely by someone starting the installer. Taking the
    #    lock writes nothing at all, which is why this holds.
    $victimDir = Join-Path $tmp "victim-root"
    New-Item -ItemType Directory -Force -Path $victimDir | Out-Null
    $victim = Join-Path $tmp "precious.txt"
    $precious = "do not truncate me"
    $precious | Set-Content -LiteralPath $victim
    $planted = Join-Path $victimDir $lockFileName
    $plantedOk = $false
    try {
        New-Item -ItemType SymbolicLink -Path $planted -Target $victim -ErrorAction Stop | Out-Null
        $plantedOk = $true
    } catch {
        Write-Host "  SKIP  could not plant a file link: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    if ($plantedOk) {
        $plantedLock = $null
        try { $plantedLock = Enter-StudioInstallLock -Path $victimDir } catch {}
        # Checked WHILE the lock is held, which is the point: following the link would leave the
        # victim opened with FileShare.None for the whole install, so it would be unreadable here
        # even though nothing was written to it. That is a denial of service on somebody else's
        # file, and it is why the link is removed rather than merely not written to.
        $victimReadable = $false
        try { $victimReadable = ((Get-Content -Raw -LiteralPath $victim).Trim() -eq $precious) } catch {}
        Check "a planted link's target stays readable while the lock is held" $victimReadable
        Check "the lock file is a real file, not the planted link" (
            $null -ne (Get-Item -LiteralPath $planted -Force -ErrorAction SilentlyContinue) -and
            ((Get-Item -LiteralPath $planted -Force).Attributes -band
                [System.IO.FileAttributes]::ReparsePoint) -eq 0)
        if ($plantedLock) { Exit-StudioInstallLock -Lock $plantedLock }
        Check "a planted link's target is not truncated" (
            (Get-Content -Raw -LiteralPath $victim).Trim() -eq $precious)
    }

    # 6b. The same hazard planted as a HARD link. A hard link is a second directory entry for an
    #     existing file, so it carries no reparse point attribute and the attributes check cannot
    #     see it at all; File.Open follows it just the same and would hold the victim with
    #     FileShare.None for the whole install. That is a denial of service on somebody else's
    #     file, reached by a route the reparse check does not cover.
    $hardDir = Join-Path $tmp "hardlink-root"
    New-Item -ItemType Directory -Force -Path $hardDir | Out-Null
    $hardVictim = Join-Path $tmp "precious-hard.txt"
    $hardPrecious = "do not deny me"
    $hardPrecious | Set-Content -LiteralPath $hardVictim -NoNewline
    $hardVictimLength = (Get-Item -LiteralPath $hardVictim).Length
    $hardPlanted = Join-Path $hardDir $lockFileName
    $hardOk = $false
    try {
        New-Item -ItemType HardLink -Path $hardPlanted -Target $hardVictim -ErrorAction Stop | Out-Null
        $hardOk = $true
    } catch {
        Write-Host "  SKIP  this host cannot create a hard link: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    if ($hardOk) {
        # Proof the planted entry really is a hard link and not something the existing defence
        # already catches. Without this the checks below could pass for the wrong reason.
        $hardItem = Get-Item -LiteralPath $hardPlanted -Force
        Check "a planted hard link carries NO reparse attribute, so the attributes check is blind to it (bites)" (
            ($hardItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0)
        Check "the planted hard link really reaches the victim's contents" (
            $hardItem.Length -eq $hardVictimLength -and $hardVictimLength -gt 0)

        $hardLock = $null
        try { $hardLock = Enter-StudioInstallLock -Path $hardDir } catch {}
        Check "the lock is still granted with a hard link planted" ($null -ne $hardLock)
        # Read WHILE the lock is held. Following the hard link would leave the victim open with
        # FileShare.None until the install finishes, so a second exclusive open of it would be
        # refused. Get-Content is not enough on its own here, since a shared read can succeed
        # against a handle opened for reading; ask for the same exclusivity the installer takes.
        $victimFree = $false
        try {
            $probeStream = [System.IO.File]::Open($hardVictim, [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read, [System.IO.FileShare]::None)
            $probeStream.Dispose()
            $victimFree = $true
        } catch {}
        Check "a planted hard link's target is NOT held open by the lock" $victimFree
        Check "the lock file is a fresh empty file, not the victim" (
            (Get-Item -LiteralPath $hardPlanted -Force).Length -eq 0)
        if ($hardLock) { Exit-StudioInstallLock -Lock $hardLock }
        Check "a planted hard link's target keeps its contents" (
            (Get-Content -Raw -LiteralPath $hardVictim) -eq $hardPrecious)
        Check "a planted hard link's target keeps its length" (
            (Get-Item -LiteralPath $hardVictim).Length -eq $hardVictimLength)

        # 6c. Removing a non-empty entry must never become a way to steal a lock somebody else
        #     already holds. The file here is BOTH non-empty, which is what the new check keys
        #     on, and held by a real second process, so a fix that removed first and asked
        #     questions afterwards would hand out a second lock for one destination. The refusal
        #     has to come first, and the file has to survive intact.
        $stealDir = Join-Path $tmp "steal-root"
        New-Item -ItemType Directory -Force -Path $stealDir | Out-Null
        $stealLockPath = Join-Path $stealDir $lockFileName
        $stealBody = "held by somebody else"
        $stealBody | Set-Content -LiteralPath $stealLockPath -NoNewline
        $stealHolder = Start-Holder $stealLockPath
        try {
            $stolen = "not-run"
            try { $stolen = Enter-StudioInstallLock -Path $stealDir } catch {}
            Check "a held non-empty lock is refused, not removed and retaken" ($null -eq $stolen)
            if ($stolen -and $stolen -ne "not-run") { Exit-StudioInstallLock -Lock $stolen }
            Check "the held lock file still exists after the refusal" (
                Test-Path -LiteralPath $stealLockPath)
        } finally {
            Stop-Holder $stealHolder
        }
        # Its contents are only readable once the holder has let go, since the holder takes it
        # with FileShare.None exactly as the installer does.
        Check "the held lock file survives the refusal intact" (
            (Get-Content -Raw -LiteralPath $stealLockPath) -eq $stealBody)
    }

    # 7. A real I/O fault must not be reported as "another installer is running". Only a sharing
    #    or lock violation means that. Here the lock path already exists as a DIRECTORY, so the
    #    open fails for a reason that is nobody's concurrent install, and the caller must see the
    #    failure rather than a misleading "wait for the other install to finish".
    $blockedRoot = Join-Path $tmp "blocked-root"
    New-Item -ItemType Directory -Force -Path (Join-Path $blockedRoot $lockFileName) | Out-Null
    $blockedThrew = $false
    $blockedResult = "not-run"
    try { $blockedResult = Enter-StudioInstallLock -Path $blockedRoot } catch { $blockedThrew = $true }
    Check "a non-sharing failure is not silently reported as a busy lock" (
        $blockedThrew -or $null -ne $blockedResult)
    if ($blockedResult -and $blockedResult -ne "not-run") { Exit-StudioInstallLock -Lock $blockedResult }

    # 8. A holder that dies without releasing must not wedge the destination: the OS closes the
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

    # The lock file must not make a fresh root look like somebody else's directory.
    #
    # Enter-StudioInstallLock creates the lock inside $StudioHome before anything else runs, and
    # in env mode Write-StudioRootOwnerMarker refuses to claim a root that is not empty. Counting
    # the lock as occupancy would mean a fresh UNSLOTH_STUDIO_HOME never gets its owner marker,
    # and an install that then died before the venv marker was written would leave a root the
    # uninstaller does not recognise and will not remove. Nothing else pins this, so removing the
    # exclusion from the occupancy test would otherwise pass silently.
    $StudioRedirectMode = 'env'
    $freshRoot = Join-Path $tmp "fresh-env-root"
    New-Item -ItemType Directory -Force -Path $freshRoot | Out-Null
    $freshLock = Enter-StudioInstallLock -Path $freshRoot
    Check "the fresh root took its lock" ($null -ne $freshLock)
    Write-StudioRootOwnerMarker -Root $freshRoot
    Check "a fresh env-mode root carrying only the lock file is still claimed" (
        Test-Path -LiteralPath (Join-Path $freshRoot ".unsloth-studio-owned") -PathType Leaf)
    if ($freshLock) { Exit-StudioInstallLock -Lock $freshLock }

    # Bites control for the same code path: a root holding somebody else's file is NOT claimed,
    # so the check above is about the lock file specifically and not about env mode claiming
    # everything it is pointed at.
    $occupiedRoot = Join-Path $tmp "occupied-env-root"
    New-Item -ItemType Directory -Force -Path $occupiedRoot | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $occupiedRoot "someone-elses-notes.txt"), "mine")
    $occupiedLock = Enter-StudioInstallLock -Path $occupiedRoot
    Write-StudioRootOwnerMarker -Root $occupiedRoot
    Check "control: an env-mode root holding another file is NOT claimed" (
        -not (Test-Path -LiteralPath (Join-Path $occupiedRoot ".unsloth-studio-owned")))
    if ($occupiedLock) { Exit-StudioInstallLock -Lock $occupiedLock }

    # The owner marker must not follow a link either.
    #
    # The lock is now the first thing that can create the root, so on a root whose inherited
    # permissions let another user write to it, that user can plant a `.unsloth-studio-owned` link
    # between the directory appearing and the marker being written. A bare WriteAllText follows it
    # and truncates the target under the installer's rights. Same hazard the lock file itself is
    # guarded against a few checks above, so it gets the same treatment rather than a second one.
    $StudioRedirectMode = 'default'
    $markerVictim = Join-Path $tmp "marker-victim.txt"
    [System.IO.File]::WriteAllText($markerVictim, "keep me")
    $markerRoot = Join-Path $tmp "marker-root"
    New-Item -ItemType Directory -Force -Path $markerRoot | Out-Null
    $plantedMarker = Join-Path $markerRoot ".unsloth-studio-owned"
    $madeLink = $false
    try {
        New-Item -ItemType SymbolicLink -Path $plantedMarker -Target $markerVictim -ErrorAction Stop | Out-Null
        $madeLink = $true
    } catch {}
    if (-not $madeLink) {
        # Windows needs Developer Mode or elevation for this. Say so rather than reporting a pass.
        Write-Host "  SKIP  cannot create a symbolic link on this host" -ForegroundColor Yellow
    } else {
        Check "the planted marker link really reaches the victim (bites)" (
            (Get-Content -Raw -LiteralPath $plantedMarker) -eq "keep me")
        Write-StudioRootOwnerMarker -Root $markerRoot
        Check "the planted marker link's target is not truncated" (
            (Get-Content -Raw -LiteralPath $markerVictim) -eq "keep me")
        $written = Get-Item -LiteralPath $plantedMarker -Force -ErrorAction SilentlyContinue
        Check "the marker is a real file, not the planted link" (
            $null -ne $written -and $written.Length -eq 0 -and $null -eq $written.Target)
    }

    # And the lock's own fresh-root branch has to go THROUGH that helper. A second copy of the
    # write inline would pass every check above while carrying the hazard, because the checks
    # drive the helper directly. Read the branch instead of trusting it.
    # @() around the call, not just an index: a single-element return unrolls to a bare string,
    # and [0] on a string is its first CHARACTER, which matches neither pattern and passes the
    # negative check for free. Observed here before it was fixed.
    $lockFn = @(Get-HelperSources $installPs1 @("Enter-StudioInstallLock"))[0]
    Check "the lock claims a fresh root through Write-StudioRootOwnerMarker" (
        $lockFn -match 'Write-StudioRootOwnerMarker -Root \$Path')
    Check "and does not write the marker itself" (
        $lockFn -notmatch 'WriteAllText\([^)]*\.unsloth-studio-owned')
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All install-lock alias checks passed"
exit 0
