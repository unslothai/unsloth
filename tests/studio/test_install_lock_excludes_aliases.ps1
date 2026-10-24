#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The install lock must exclude two spellings of one destination. The mutex hashes the
# resolved path and cannot; only the FILE half is tested here, with the mutex stubbed.
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

# Read from install.ps1 so a rename there fails here instead of passing silently.
$installText = Get-Content -Raw -LiteralPath $installPs1
$nameMatch = [regex]::Match($installText,
    '\$script:StudioInstallLockFileName\s*=\s*"([^"]+)"')
if (-not $nameMatch.Success) { throw "could not find `$script:StudioInstallLockFileName in install.ps1" }
$lockFileName = $nameMatch.Groups[1].Value
Write-Host "  lock file name: $lockFileName"

# Always granting the mutex means every refusal seen is the file lock's.
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

# A child process: .NET would refuse a same-process second open regardless of the OS.
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
    # -WindowStyle is not supported by pwsh off Windows.
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

    # 1. A missing destination is created by taking the lock.
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

    # uninstall.ps1 deletes what the marker names, so a pre-existing directory must never get one.
    $preexisting = Join-Path $tmp "users-own-directory"
    New-Item -ItemType Directory -Force -Path $preexisting | Out-Null
    "keep me" | Set-Content -LiteralPath (Join-Path $preexisting "important.txt")
    $preLock = Enter-StudioInstallLock -Path $preexisting
    Check "an existing directory is NOT claimed as an Unsloth root" (
        -not (Test-Path -LiteralPath (Join-Path $preexisting ".unsloth-studio-owned")))
    Check "an existing directory's contents are untouched" (
        (Get-Content -Raw -LiteralPath (Join-Path $preexisting "important.txt")).Trim() -eq "keep me")
    Exit-StudioInstallLock -Lock $preLock

    # 2. Released means re-takeable.
    $again = Enter-StudioInstallLock -Path $fresh
    Check "the lock can be taken again after release" ($null -ne $again)
    Exit-StudioInstallLock -Lock $again

    # 3. Control: different destinations must not exclude each other.
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

        # 5. Same destination through an ALIAS (junction on Windows, symlink elsewhere): refused.
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
            # Guard against a vacuous pass: the alias must reach the same directory.
            $probe = Join-Path $real ("probe-" + [guid]::NewGuid().ToString("N"))
            New-Item -ItemType File -Force -Path $probe | Out-Null
            $viaAlias = Join-Path $alias (Split-Path -Leaf $probe)
            Check "the alias really reaches the same directory" (Test-Path -LiteralPath $viaAlias)
            Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue

            $aliasLock = Enter-StudioInstallLock -Path $alias
            Check "an ALIAS of the held destination is refused" ($null -eq $aliasLock)
            if ($aliasLock) { Exit-StudioInstallLock -Lock $aliasLock }

            # Not vacuous: these spellings hash to different mutex names.
            $spellReal = [System.IO.Path]::GetFullPath($real).ToUpperInvariant()
            $spellAlias = [System.IO.Path]::GetFullPath($alias).ToUpperInvariant()
            Check "the two spellings differ, so a name derived from them cannot exclude (bites)" (
                $spellReal -ne $spellAlias)
        }
    } finally {
        Stop-Holder $holder
    }

    # 6. A planted link at the lock path: File.Open follows it, so taking the lock writes nothing.
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
        # WHILE held: following the link would hold the victim with FileShare.None.
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

    # 6b. A HARD link has no reparse attribute, yet File.Open follows it just the same.
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
        # Proof the planted entry really is a hard link, not something already caught.
        $hardItem = Get-Item -LiteralPath $hardPlanted -Force
        Check "a planted hard link carries NO reparse attribute, so the attributes check is blind to it (bites)" (
            ($hardItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0)
        Check "the planted hard link really reaches the victim's contents" (
            $hardItem.Length -eq $hardVictimLength -and $hardVictimLength -gt 0)

        $hardLock = $null
        try { $hardLock = Enter-StudioInstallLock -Path $hardDir } catch {}
        Check "the lock is still granted with a hard link planted" ($null -ne $hardLock)
        # Ask for the installer's own exclusivity: a shared read can succeed against a read handle.
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

        # 6c. Removing a non-empty entry must never steal a lock another process holds.
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
        # Readable only once the holder lets go: it holds FileShare.None like the installer.
        Check "the held lock file survives the refusal intact" (
            (Get-Content -Raw -LiteralPath $stealLockPath) -eq $stealBody)
    }

    # 7. A real I/O fault (lock path is a directory) must not read as "another installer".
    $blockedRoot = Join-Path $tmp "blocked-root"
    New-Item -ItemType Directory -Force -Path (Join-Path $blockedRoot $lockFileName) | Out-Null
    $blockedThrew = $false
    $blockedResult = "not-run"
    try { $blockedResult = Enter-StudioInstallLock -Path $blockedRoot } catch { $blockedThrew = $true }
    Check "a non-sharing failure is not silently reported as a busy lock" (
        $blockedThrew -or $null -ne $blockedResult)
    if ($blockedResult -and $blockedResult -ne "not-run") { Exit-StudioInstallLock -Lock $blockedResult }

    # 8. A dead holder's handle is closed by the OS, which is why the file is not deleted on release.
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

    # The lock file must not make a fresh root look non-empty to the owner-marker check.
    $StudioRedirectMode = 'env'
    $freshRoot = Join-Path $tmp "fresh-env-root"
    New-Item -ItemType Directory -Force -Path $freshRoot | Out-Null
    $freshLock = Enter-StudioInstallLock -Path $freshRoot
    Check "the fresh root took its lock" ($null -ne $freshLock)
    Write-StudioRootOwnerMarker -Root $freshRoot
    Check "a fresh env-mode root carrying only the lock file is still claimed" (
        Test-Path -LiteralPath (Join-Path $freshRoot ".unsloth-studio-owned") -PathType Leaf)
    if ($freshLock) { Exit-StudioInstallLock -Lock $freshLock }

    # Script scope outlives a run and `irm | iex` can install twice, so stale state must reset.
    $StudioRedirectMode = 'env'
    $staleRoot = Join-Path $tmp "stale-env-root"
    Check "control: the stale root really is absent" (-not (Test-Path -LiteralPath $staleRoot))
    $script:StudioEnvRootPath = $staleRoot
    $script:StudioEnvRootExisted = $true
    $staleLock = Enter-StudioInstallLock -Path $staleRoot
    Check "the run still took its lock" ($null -ne $staleLock)
    Check "a root created by THIS run is claimed despite the stale record" (
        Test-Path -LiteralPath (Join-Path $staleRoot ".unsloth-studio-owned") -PathType Leaf)
    if ($staleLock) { Exit-StudioInstallLock -Lock $staleLock }
    $script:StudioEnvRootPath = $null
    $script:StudioEnvRootExisted = $false

    # The reset must run before the override is read; checked in the installer source.
    $srcAll = [System.IO.File]::ReadAllText((Join-Path $root "install.ps1"))
    $resetAt = $srcAll.IndexOf('$script:StudioEnvRootPath = $null')
    $recordAt = $srcAll.IndexOf('$script:StudioEnvRootPath = $envOverride')
    Check "the per-invocation reset exists" ($resetAt -gt 0)
    Check "the reset clears the existed flag too" (
        $srcAll -match '\$script:StudioEnvRootExisted = \$false')
    Check "the reset runs before the override records anything" (
        $resetAt -gt 0 -and $recordAt -gt 0 -and $resetAt -lt $recordAt)

    # Control: a root holding someone else's file is NOT claimed.
    $occupiedRoot = Join-Path $tmp "occupied-env-root"
    New-Item -ItemType Directory -Force -Path $occupiedRoot | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $occupiedRoot "someone-elses-notes.txt"), "mine")
    $occupiedLock = Enter-StudioInstallLock -Path $occupiedRoot
    Write-StudioRootOwnerMarker -Root $occupiedRoot
    Check "control: an env-mode root holding another file is NOT claimed" (
        -not (Test-Path -LiteralPath (Join-Path $occupiedRoot ".unsloth-studio-owned")))
    if ($occupiedLock) { Exit-StudioInstallLock -Lock $occupiedLock }

    # The owner marker must not follow a planted link either.
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
        # Windows needs Developer Mode or elevation for this.
        Write-Host "  SKIP  cannot create a symbolic link on this host" -ForegroundColor Yellow
    } else {
        Check "the planted marker link really reaches the victim (bites)" (
            (Get-Content -Raw -LiteralPath $plantedMarker) -eq "keep me")
        Write-StudioRootOwnerMarker -Root $markerRoot
        Check "the planted marker link's target is not truncated" (
            (Get-Content -Raw -LiteralPath $markerVictim) -eq "keep me")
        $written = Get-Item -LiteralPath $plantedMarker -Force -ErrorAction SilentlyContinue
        # Separate checks: this failed on 5.1 only, and one conjunction hides which part.
        Check "the marker exists after the claim" ($null -ne $written)
        Check "the marker is empty, as this installer only ever creates it" (
            $null -ne $written -and $written.Length -eq 0)
        # ReparsePoint, not .Target: .Target is an empty collection, not $null, on 5.1.
        Check "the marker is a real file, not the planted link" (
            $null -ne $written -and
            (($written.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -eq 0))
    }

    # No delete-then-write window: creation must FAIL on an existing entry.
    $raceVictim = Join-Path $tmp "race-victim.txt"
    [System.IO.File]::WriteAllText($raceVictim, "still here")
    $raceRoot = Join-Path $tmp "race-root"
    New-Item -ItemType Directory -Force -Path $raceRoot | Out-Null
    $raceMarker = Join-Path $raceRoot ".unsloth-studio-owned"
    $madeRaceLink = $false
    try {
        New-Item -ItemType SymbolicLink -Path $raceMarker -Target $raceVictim -ErrorAction Stop | Out-Null
        $madeRaceLink = $true
    } catch {}
    if (-not $madeRaceLink) {
        Write-Host "  SKIP  cannot create a symbolic link on this host" -ForegroundColor Yellow
    } else {
        $marker = $raceMarker
        $createThrew = $false
        $s = $null
        try {
            $s = [System.IO.File]::Open($marker, [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        } catch { $createThrew = $true } finally { if ($s) { $s.Dispose() } }
        Check "creating the marker refuses a name that is already taken" ($createThrew -eq $true)
        Check "and the link's target is untouched by the refusal" (
            (Get-Content -Raw -LiteralPath $raceVictim) -eq "still here")
    }

    # The drive above uses CreateNew directly, so the helper is read too.
    $markerFn = @(Get-HelperSources $installPs1 @("Write-StudioRootOwnerMarker"))[0]
    Check "the marker helper creates with CreateNew" ($markerFn -match 'FileMode\]::CreateNew')
    Check "and no longer writes the marker through WriteAllText" (
        $markerFn -notmatch 'WriteAllText\(\$marker')

    # A non-empty lock-path entry is moved aside, never destroyed: a hard link and a real file
    # cannot be told apart cheaply, and a rename needs no discrimination.
    $keepRoot = Join-Path $tmp "non-empty-lock-root"
    New-Item -ItemType Directory -Force -Path $keepRoot | Out-Null
    $keepLock = Join-Path $keepRoot $lockFileName
    [System.IO.File]::WriteAllText($keepLock, "somebody else's bytes")
    $keepLockObj = Enter-StudioInstallLock -Path $keepRoot
    Check "an install still starts with a non-empty entry at the lock path" ($null -ne $keepLockObj)
    Check "the lock file it ends up holding is a fresh empty one" (
        (Get-Item -LiteralPath $keepLock -Force).Length -eq 0)
    $displaced = @(Get-ChildItem -LiteralPath $keepRoot -Force |
        Where-Object { $_.Name -like "$lockFileName.displaced-*" })
    Check "the original was moved aside rather than deleted" ($displaced.Count -eq 1)
    Check "and it still has its contents" (
        $displaced.Count -eq 1 -and
        (Get-Content -Raw -LiteralPath $displaced[0].FullName) -eq "somebody else's bytes")
    if ($keepLockObj) { Exit-StudioInstallLock -Lock $keepLockObj }

    # Takes the lock then looks, proving the env-mode claim is reachable through the lock.
    $envFresh = Join-Path $tmp "env-root-created-earlier"
    [System.IO.Directory]::CreateDirectory($envFresh) | Out-Null
    $script:StudioEnvRootPath = $envFresh
    $script:StudioEnvRootExisted = $false
    $envLock = Enter-StudioInstallLock -Path $envFresh
    Check "a root the override created this run is still claimed by the lock" (
        Test-Path -LiteralPath (Join-Path $envFresh ".unsloth-studio-owned") -PathType Leaf)
    if ($envLock) { Exit-StudioInstallLock -Lock $envLock }

    # Control: a directory the user already had must never be claimed.
    $envOwned = Join-Path $tmp "env-root-user-already-had"
    [System.IO.Directory]::CreateDirectory($envOwned) | Out-Null
    $script:StudioEnvRootPath = $envOwned
    $script:StudioEnvRootExisted = $true
    $ownedLock = Enter-StudioInstallLock -Path $envOwned
    Check "control: a root that existed before the run is NOT claimed" (
        -not (Test-Path -LiteralPath (Join-Path $envOwned ".unsloth-studio-owned")))
    if ($ownedLock) { Exit-StudioInstallLock -Lock $ownedLock }
    $script:StudioEnvRootPath = $null

    # The lock must go through the helper, not an inline copy. @() is needed: [0] on a bare
    # string is its first character.
    $lockFn = @(Get-HelperSources $installPs1 @("Enter-StudioInstallLock"))[0]
    Check "the lock claims a fresh root through Write-StudioRootOwnerMarker" (
        $lockFn -match 'Write-StudioRootOwnerMarker -Root \$Path')
    Check "and does not write the marker itself" (
        $lockFn -notmatch 'WriteAllText\([^)]*\.unsloth-studio-owned')

    # A link that cannot be removed must STOP the install, not fall through to File.Open.
    $undeletableRoot = Join-Path $tmp "undeletable-link-root"
    New-Item -ItemType Directory -Force -Path $undeletableRoot | Out-Null
    $linkVictim = Join-Path $tmp "link-victim-empty.txt"
    [System.IO.File]::WriteAllText($linkVictim, "")
    $undeletableLock = Join-Path $undeletableRoot $lockFileName
    $madeUndeletable = $false
    try {
        New-Item -ItemType SymbolicLink -Path $undeletableLock -Target $linkVictim -ErrorAction Stop | Out-Null
        $madeUndeletable = $true
    } catch {}
    if (-not $madeUndeletable) {
        Write-Host "  SKIP  cannot create a symbolic link on this host" -ForegroundColor Yellow
    } else {
        # Refuse Remove-Item for this path only, like an ACL allowing inspection but not deletion.
        $savedRemove = ${function:Remove-Item}
        function Remove-Item {
            param(
                [Parameter(ValueFromPipeline = $true)]$InputObject,
                [string]$LiteralPath, [string]$Path, [switch]$Force, [switch]$Recurse,
                [string]$ErrorAction
            )
            if ($LiteralPath -and $LiteralPath.EndsWith($script:StudioInstallLockFileName)) {
                throw [System.UnauthorizedAccessException]::new("access denied by test")
            }
        }
        $linkThrew = $false
        $linkResult = "unset"
        try { $linkResult = Enter-StudioInstallLock -Path $undeletableRoot } catch { $linkThrew = $true }
        ${function:Remove-Item} = $savedRemove
        Check "an undeletable link at the lock path fails the install" (
            $linkThrew -eq $true -and $linkResult -eq "unset")
        Check "control: it did not quietly report a busy lock instead" ($linkResult -ne $null -or $linkThrew)
        Check "and the link's target was never held or replaced" (
            (Get-Item -LiteralPath $undeletableLock -Force).Target -eq $linkVictim)
    }

    # A non-contention rename failure must surface, not read as "already running".
    $mvRoot = Join-Path $tmp "unmovable-lock-root"
    New-Item -ItemType Directory -Force -Path $mvRoot | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $mvRoot $lockFileName), "not empty")
    $savedMove = ${function:Move-Item}
    function Move-Item {
        param([string]$LiteralPath, [string]$Destination, [switch]$Force, [string]$ErrorAction)
        throw [System.UnauthorizedAccessException]::new("rename denied by test")
    }
    $mvThrew = $false
    $mvResult = "unset"
    try { $mvResult = Enter-StudioInstallLock -Path $mvRoot } catch { $mvThrew = $true }
    ${function:Move-Item} = $savedMove
    Check "a rename denied by permissions is surfaced, not called a busy lock" (
        $mvThrew -eq $true -and $mvResult -eq "unset")

    # Control: a SHARING violation is still the busy path.
    [System.IO.File]::WriteAllText((Join-Path $mvRoot $lockFileName), "not empty")
    function Move-Item {
        param([string]$LiteralPath, [string]$Destination, [switch]$Force, [string]$ErrorAction)
        # The two-argument constructor sets HResult; the field is not settable from here.
        throw [System.IO.IOException]::new("sharing violation", 32)
    }
    $busyThrew = $false
    $busyResult = "unset"
    try { $busyResult = Enter-StudioInstallLock -Path $mvRoot } catch { $busyThrew = $true }
    ${function:Move-Item} = $savedMove
    Check "control: a sharing violation on the rename still reports a busy lock" (
        $busyThrew -eq $false -and $null -eq $busyResult)
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
# Swapped-link race: drive the SHIPPED predicate, not a retyped copy.
$lockSrc = [System.IO.File]::ReadAllText((Join-Path $root "install.ps1"))
$guardAt = $lockSrc.IndexOf('A link was planted at the install lock path')
Check "the post-open guard is present" ($guardAt -gt 0)

$openAt = $lockSrc.IndexOf('$stream = [System.IO.File]::Open(')
Check "the guard runs after the open, not before it" ($openAt -gt 0 -and $guardAt -gt $openAt)
# Guarded so a missing guard degrades to a failed check, not a crash.
$guardBlock = if ($openAt -gt 0 -and $guardAt -gt $openAt) { $lockSrc.Substring($openAt, $guardAt - $openAt) } else { "" }
Check "the guard releases the handle before it throws" ($guardBlock -match '\$stream\.Dispose\(\)[\s\S]*\$stream = \$null')

$predicate = @($lockSrc -split "`n" | Where-Object {
    $_ -match '\$entry -and \(\(\$entry\.Attributes -band \[System\.IO\.FileAttributes\]::ReparsePoint\) -ne 0\)' })
Check "the predicate is one line of the shipped source" ($predicate.Count -eq 1)
$test = $null
if ($predicate.Count -eq 1) {
    # Two statements, not one expression: [scriptblock]::Create(A -replace B, C) binds C as a
    # second ARGUMENT to Create rather than to -replace, and fails on the overload.
    $rewritten = ($predicate[0].Trim() -replace '^if \(', 'return (') -replace '\) \{$', ')'
    $test = [scriptblock]::Create($rewritten)
}

$raceDir = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-lockrace-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $raceDir | Out-Null
try {
    # The case the Length test cannot see: a link whose target is itself zero bytes.
    $victim = Join-Path $raceDir "someone-elses.lock"
    [System.IO.File]::WriteAllText($victim, "")
    $planted = Join-Path $raceDir "install.lock"
    New-Item -ItemType SymbolicLink -Path $planted -Target $victim | Out-Null

    $entry = Get-Item -LiteralPath $planted -Force
    Check "a zero-byte target is still zero bytes, so the Length test alone would accept it" (
        ([System.IO.FileInfo]$victim).Length -eq 0)
    Check "the guard refuses a link swapped in under the lock path" ($null -ne $test -and (& $test) -eq $true)

    $ordinary = Join-Path $raceDir "ordinary.lock"
    [System.IO.File]::WriteAllText($ordinary, "")
    $entry = Get-Item -LiteralPath $ordinary -Force
    Check "control: the guard leaves an ordinary empty lock file alone" ($null -ne $test -and (& $test) -eq $false)

    # A missing entry is not a link, or an AV-hidden fresh lock would fail the install.
    $entry = $null
    Check "control: a vanished entry is not a link" ($null -ne $test -and (& $test) -eq $false)
} finally {
    Remove-Item -LiteralPath $raceDir -Recurse -Force -ErrorAction SilentlyContinue
}

# The gate must be the LAST thing before the success line, or later failures exit 0.
$selfText = [System.IO.File]::ReadAllText($PSCommandPath)
$gateAt = $selfText.LastIndexOf('if ($failures -gt 0)')
$passAt = $selfText.LastIndexOf('All install-lock alias checks passed')
$lastCheckAt = $selfText.LastIndexOf("`nCheck ")
if ($gateAt -lt 0 -or $passAt -lt $gateAt -or $lastCheckAt -gt $gateAt) {
    Write-Host "  FAIL  the failure gate is not the last thing before the success line" -ForegroundColor Red
    exit 1
}

if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All install-lock alias checks passed"
exit 0
