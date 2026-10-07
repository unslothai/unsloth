#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for install.ps1's venv rollback helpers, AST-extracted so the installer never runs.

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

# Extract the transitive closure so a new callee is picked up automatically.
$subjectNames = @(
    "Start-StudioVenvRollback",
    "Remove-StudioVenvTreeWithRetry",
    "Test-StudioVenvRollbackMustBePreserved",
    "Remove-StaleStudioVenvRollbacks",
    "Restore-StudioVenvRollback",
    "Complete-StudioVenvRollback",
    "Restore-StudioUvCacheMarker",
    # Not a rollback callee: the disk-full diagnosis calls it.
    "Get-StudioFreeSpaceBytes"
)

$definitions = @{}
foreach ($node in $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
}, $true)) {
    if ($definitions.ContainsKey($node.Name)) {
        throw "install.ps1 defines $($node.Name) more than once; the extraction cannot tell which one is under test"
    }
    $definitions[$node.Name] = $node
}

foreach ($name in $subjectNames) {
    if (-not $definitions.ContainsKey($name)) {
        throw "expected $name in install.ps1, found no definition (renamed or removed?)"
    }
}

# Stubbed sinks: the walk stops at them.
$stubbedNames = @("substep", "Write-StudioLine")

$extracted = [System.Collections.Generic.List[string]]::new()
$seen = @{}
$queue = [System.Collections.Generic.Queue[string]]::new()
foreach ($name in $subjectNames) { $queue.Enqueue($name) }
while ($queue.Count -gt 0) {
    $name = $queue.Dequeue()
    if ($seen.ContainsKey($name)) { continue }
    $seen[$name] = $true
    $extracted.Add($name)
    foreach ($call in $definitions[$name].Body.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst]
    }, $true)) {
        $callee = $call.GetCommandName()
        if ($callee -and $definitions.ContainsKey($callee) -and
            $stubbedNames -notcontains $callee -and -not $seen.ContainsKey($callee)) {
            $queue.Enqueue($callee)
        }
    }
}

foreach ($name in $extracted) { Invoke-Expression $definitions[$name].Extent.Text }

$pulledIn = @($extracted | Where-Object { $subjectNames -notcontains $_ })
if ($pulledIn.Count -gt 0) { Write-Host "Extracted dependencies: $($pulledIn -join ', ')" }

function substep { param([string]$Message, [string]$Color) }
# Under "Stop", an undefined sink would abort the suite rather than fail a check.
function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }

# Fail here with the caller named, not later as "term not recognized".
$windowsOnlyCommands = @("Get-CimInstance", "Get-Acl")
$unresolved = [System.Collections.Generic.List[string]]::new()
foreach ($name in $extracted) {
    foreach ($call in $definitions[$name].Body.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.CommandAst]
    }, $true)) {
        $callee = $call.GetCommandName()
        if (-not $callee) { continue }
        # Keyed on the command being absent: $env:OS can say Windows without CimCmdlets.
        if ($windowsOnlyCommands -contains $callee -and
            -not (Get-Command -Name $callee -ErrorAction SilentlyContinue)) { continue }
        if (-not (Get-Command -Name $callee -ErrorAction SilentlyContinue)) {
            $unresolved.Add("$callee (called by $name)")
        }
    }
}
if ($unresolved.Count -gt 0) {
    throw "extracted helpers call commands that do not resolve: $(($unresolved | Sort-Object -Unique) -join '; ')"
}

$failures = 0
function Check($name, $condition) {
    if ($condition) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Reset-RollbackState($target) {
    $script:StudioVenvRollbackDir = $null
    $script:StudioVenvRollbackTarget = $target
    $script:StudioVenvRollbackActive = $false
    # The commit flag is per install; a previous section's commit would suppress this restore.
    $script:StudioInstallCommitted = $false
}

$StudioHome = Join-Path ([System.IO.Path]::GetTempPath()) "unsloth-rollback-$([guid]::NewGuid().ToString('N'))"
$VenvDir = Join-Path $StudioHome "unsloth_studio"
[System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null

try {
    Write-Host "Successful replacement"
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    Start-StudioVenvRollback -ExistingDir $VenvDir
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "new")
    Complete-StudioVenvRollback
    Check "new environment remains" ((Get-Content -LiteralPath (Join-Path $VenvDir "generation") -Raw) -eq "new")
    Check "current rollback is removed" (-not @(Get-ChildItem -LiteralPath $StudioHome -Directory |
        Where-Object { $_.Name -like "unsloth_studio.rollback.*" }))

    Write-Host "Stale cleanup"
    $stale = Join-Path $StudioHome "unsloth_studio.rollback.20000101000000.2147483647"
    $active = Join-Path $StudioHome "unsloth_studio.rollback.20000101000001.$PID"
    $unrecognized = Join-Path $StudioHome "unsloth_studio.rollback.user-data"
    [System.IO.Directory]::CreateDirectory($stale) | Out-Null
    [System.IO.Directory]::CreateDirectory($active) | Out-Null
    [System.IO.Directory]::CreateDirectory($unrecognized) | Out-Null
    Remove-StaleStudioVenvRollbacks
    Check "dead-owner rollback is removed" (-not (Test-Path -LiteralPath $stale))
    Check "live-owner rollback is preserved" (Test-Path -LiteralPath $active)
    Check "unrecognized rollback name is preserved" (Test-Path -LiteralPath $unrecognized)
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $active -Recurse -Force
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $unrecognized -Recurse -Force

    Write-Host "Failure restoration"
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old-again")
    Reset-RollbackState $VenvDir
    $committed = $false
    try {
        try {
            Start-StudioVenvRollback -ExistingDir $VenvDir
            [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
            [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "partial")
            throw "simulated install failure"
        } finally {
            if (-not $committed) { Restore-StudioVenvRollback }
        }
    } catch {
        if ($_.Exception.Message -ne "simulated install failure") { throw }
    }
    Check "finally restores the previous environment" (
        (Get-Content -LiteralPath (Join-Path $VenvDir "generation") -Raw) -eq "old-again"
    )
    Check "failure restoration consumes the rollback" (-not @(Get-ChildItem -LiteralPath $StudioHome -Directory |
        Where-Object { $_.Name -like "unsloth_studio.rollback.*" }))

    Write-Host "Locked-file retry"
    $retryDir = Join-Path $StudioHome "retry"
    [System.IO.Directory]::CreateDirectory($retryDir) | Out-Null
    $script:removeAttempts = 0
    function Remove-Item {
        param(
            [string]$LiteralPath,
            [switch]$Recurse,
            [switch]$Force,
            [object]$ErrorAction
        )
        $script:removeAttempts++
        if ($script:removeAttempts -lt 3) { throw "simulated lock" }
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $LiteralPath -Recurse:$Recurse -Force:$Force
    }
    try {
        $removed = Remove-StudioVenvTreeWithRetry -Path $retryDir -Label "test rollback"
    } finally {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath Function:\Remove-Item -Force
    }
    Check "locked rollback deletion retries" ($removed -and $script:removeAttempts -eq 3)

    Write-Host "--no-rollback discards the old environment instead of keeping a copy (#11313)"
    # uv creates only into an absent or empty path, so the rename still happens.
    foreach ($case in @(
            @{ Label = "default"; Flag = $false; ExpectCopy = $true },
            @{ Label = "--no-rollback"; Flag = $true; ExpectCopy = $false })) {
        [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
        [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
        Reset-RollbackState $VenvDir
        $script:StudioNoRollback = $case.Flag
        Start-StudioVenvRollback -ExistingDir $VenvDir
        $copies = @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)
        Check "$($case.Label): rollback copy kept = $($case.ExpectCopy)" (($copies.Count -gt 0) -eq $case.ExpectCopy)
        Check "$($case.Label): the old environment was moved aside" (-not (Test-Path -LiteralPath $VenvDir))
        if (-not $case.ExpectCopy) {
            # Cleared before the delete, so an interrupt is never handed a half-deleted backup.
            Check "--no-rollback clears the restore state" (
                (-not $script:StudioVenvRollbackActive) -and ($null -eq $script:StudioVenvRollbackDir))
            $restoreThrew = $false
            try { Restore-StudioVenvRollback } catch { $restoreThrew = $true }
            Check "--no-rollback leaves nothing for the restore to do" (
                (-not $restoreThrew) -and (-not (Test-Path -LiteralPath $VenvDir)))
        }
        foreach ($c in $copies) { Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue }
    }
    $script:StudioNoRollback = $false

    Write-Host "--no-rollback tells the ARM64 migration the tree is gone (#11313)"
    # "Inactive" means "not moved aside yet", so a separate discarded flag is needed.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:StudioVenvRollbackDiscarded = $false
    $script:StudioNoRollback = $true
    Start-StudioVenvRollback -ExistingDir $VenvDir
    Check "--no-rollback records that the tree was discarded" ($script:StudioVenvRollbackDiscarded)
    Check "and the tree really is gone" (-not (Test-Path -LiteralPath $VenvDir))
    $wouldMove = (-not $script:StudioVenvRollbackDiscarded) -and (-not $script:StudioVenvRollbackActive)
    Check "the migration would not try to move a tree that is gone" (-not $wouldMove)
    # Nothing may be marked preserved once the tree is gone.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:StudioVenvRollbackDiscarded = $false
    $script:StudioVenvRollbackPreserve = $false
    $script:StudioNoRollback = $true
    Start-StudioVenvRollback -ExistingDir $VenvDir
    Check "a discard inside the migration is not marked as a preserved tree" (
        $script:StudioVenvRollbackDiscarded -and -not $script:StudioVenvRollbackPreserve)
    $script:StudioNoRollback = $false
    $script:StudioVenvRollbackDiscarded = $false

    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    Start-StudioVenvRollback -ExistingDir $VenvDir
    Check "the ordinary path does not claim the tree was discarded" (-not $script:StudioVenvRollbackDiscarded)
    Check "and it is still there to be kept" ($null -ne $script:StudioVenvRollbackDir -and (Test-Path -LiteralPath $script:StudioVenvRollbackDir))
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "the volume helpers degrade to the drive root where mount points cannot be read"
    # Win32_Volume exists only on Windows; off it the fallback must say "cannot tell".
    $probed = $null
    $probeThrew = $false
    try { $probed = Get-StudioMountedVolume -Path $StudioHome } catch { $probeThrew = $true }
    Check "the mount-point probe never throws" (-not $probeThrew)
    if ($IsWindows -or $env:OS -eq "Windows_NT") {
        Check "on Windows the probe answers a volume or nothing, never something unusable" (
            $null -eq $probed -or $null -ne $probed.Name)
    } else {
        Check "off Windows the probe answers nothing" ($null -eq $probed)
    }
    Check "and an empty path is not an error" ($null -eq (Get-StudioMountedVolume -Path ""))
    # The WMI answer must be bounded, never throw, and be taken once.
    $script:StudioVolumeList = $null
    $listThrew = $false
    $t0 = [System.Diagnostics.Stopwatch]::StartNew()
    try { $null = Get-StudioVolumeList } catch { $listThrew = $true }
    $firstMs = $t0.ElapsedMilliseconds
    Check "the volume query returns rather than hanging" (-not $listThrew -and $firstMs -lt 60000)
    $t0.Restart()
    $null = Get-StudioVolumeList
    Check "and is answered from the cache the second time" ($t0.ElapsedMilliseconds -lt $firstMs + 50)
    Check "a query that answered nothing still caches an answer" (
        $null -ne $script:StudioVolumeList)
    # The cache must NOT answer free space: the disk fills during setup.
    $script:StudioVolumeList = @([pscustomobject]@{ Name = "sentinel"; DeviceID = "sentinel"; FreeSpace = 1 })
    $null = Get-StudioVolumeList
    Check "the cached list is reused when only identity is wanted" (
        @(Get-StudioVolumeList)[0].DeviceID -eq "sentinel")
    $null = Get-StudioVolumeList -Fresh
    Check "asking for a fresh answer replaces the cached one" (
        @($script:StudioVolumeList | Where-Object { $_.DeviceID -eq "sentinel" }).Count -eq 0)
    $script:StudioVolumeList = $null
    # A link is measured as the volume it points at.
    $linkTarget = Join-Path $StudioHome "link-target"
    [System.IO.Directory]::CreateDirectory($linkTarget) | Out-Null
    $linkPath = Join-Path $StudioHome "link"
    $linkMade = $true
    try { New-Item -ItemType SymbolicLink -Path $linkPath -Target $linkTarget -ErrorAction Stop | Out-Null }
    catch { $linkMade = $false }
    if ($linkMade) {
        # Free space on a live filesystem moves between calls.
        $viaLink = Get-StudioFreeSpaceBytes -Path $linkPath
        $viaTarget = Get-StudioFreeSpaceBytes -Path $linkTarget
        Check "free space through a link is the target's volume, not some other one" (
            $null -ne $viaLink -and $null -ne $viaTarget -and $viaLink -gt 0 -and
            [math]::Abs($viaLink - $viaTarget) -lt ([math]::Max($viaLink, $viaTarget) * 0.01))
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $linkPath -Force -ErrorAction SilentlyContinue
    } else {
        # Windows needs Developer Mode or elevation to create one.
        Write-Host "  SKIP  this host will not create a symbolic link"
    }
    $freeHere = Get-StudioFreeSpaceBytes -Path $StudioHome
    Check "free space still comes back from the fallback" ($null -ne $freeHere -and $freeHere -gt 0)

    Write-Host "picking the volume that holds a path, including a directory mount point"
    # On Windows a rooted path without a drive picks up the current drive, so both sides use
    # one GetFullPath call.
    $sep = [System.IO.Path]::DirectorySeparatorChar
    $mountPath = [System.IO.Path]::GetFullPath("${sep}studio")
    $otherPath = [System.IO.Path]::GetFullPath("${sep}studiofoo")
    $elsePath  = [System.IO.Path]::GetFullPath("${sep}elsewhere${sep}x")
    $rootVol  = [pscustomobject]@{ Name = [System.IO.Path]::GetPathRoot($mountPath); DeviceID = "root";  FreeSpace = 1GB }
    $mountVol = [pscustomobject]@{ Name = "$mountPath$sep";                          DeviceID = "mount"; FreeSpace = 2GB }
    $otherVol = [pscustomobject]@{ Name = "$otherPath$sep";                          DeviceID = "other"; FreeSpace = 3GB }
    $vols = @($rootVol, $mountVol, $otherVol)
    Check "a path under a mount point picks the mounted volume" (
        (Select-StudioVolumeForPath -Path (Join-Path $mountPath "cache") -Volumes $vols).DeviceID -eq "mount")
    # The mount point itself has no trailing separator.
    Check "the mount point itself picks the mounted volume, not the drive root" (
        (Select-StudioVolumeForPath -Path $mountPath -Volumes $vols).DeviceID -eq "mount")
    Check "a sibling whose name merely starts the same does not match it" (
        (Select-StudioVolumeForPath -Path $otherPath -Volumes $vols).DeviceID -eq "other")
    Check "a path under neither falls to the root volume" (
        (Select-StudioVolumeForPath -Path $elsePath -Volumes $vols).DeviceID -eq "root")
    # Volume{GUID}\... is not rooted; anchoring it to the current directory would mismatch.
    Check "an unrooted path matches no volume at all" (
        $null -eq (Select-StudioVolumeForPath -Path "Volume{00000000-0000-0000-0000-000000000000}" -Volumes $vols))
    Check "nothing to choose from is not an error" (
        $null -eq (Select-StudioVolumeForPath -Path $mountPath -Volumes @()))
    Check "an empty path chooses nothing" (
        $null -eq (Select-StudioVolumeForPath -Path "" -Volumes $vols))

    Write-Host "--no-rollback costs disk, never hardware (#11313)"
    # The Intel scan asks the PREVIOUS venv's torch about XPU; take it before discarding.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.Directory]::CreateDirectory((Join-Path $VenvDir "Scripts")) | Out-Null
    # Windows: a text .exe raises a modal 16-bit dialog, so the probe is stubbed there; on POSIX
    # the real probe runs a stand-in script that prints True.
    $fakePy = Join-Path (Join-Path $VenvDir "Scripts") "python.exe"
    $onWindows = ($IsWindows -or $env:OS -eq "Windows_NT")
    $probeCalls = [System.Collections.Generic.List[object]]::new()
    if ($onWindows) {
        Set-Content -LiteralPath $fakePy -Value "placeholder, not an interpreter"
        function Invoke-BoundedPythonProbe {
            param([string]$PythonExe, [string]$Code, [int]$TimeoutSec = 30)
            $probeCalls.Add([pscustomobject]@{ PythonExe = $PythonExe; Code = $Code })
            return [pscustomobject]@{ Ok = $true; Output = "True`r`n"; Error = "" }
        }
    } else {
        Set-Content -LiteralPath $fakePy -Value "#!/bin/sh`necho True"
        & chmod +x $fakePy
    }
    Reset-RollbackState $VenvDir
    $script:StudioPreservedXpuVerdict = $false
    $script:StudioNoRollback = $true
    $script:StudioRollbackCostsFullSize = $true
    $xpuThrew = $false
    try {
        try { Start-StudioVenvRollback -ExistingDir $VenvDir } catch { $xpuThrew = $true }
    } finally {
        if ($onWindows) {
            Invoke-Expression $definitions["Invoke-BoundedPythonProbe"].Extent.Text
        }
    }
    Check "taking the XPU verdict never costs the discard" (-not $xpuThrew)
    Check "the XPU verdict survives the environment being discarded" (
        $script:StudioPreservedXpuVerdict)
    if ($onWindows) {
        # The stand-in answers True to anyone, so pin which interpreter and question were asked.
        Check "the verdict came from the moved-aside interpreter" (
            $probeCalls.Count -eq 1 -and
            $probeCalls[0].PythonExe -match 'unsloth_studio\.rollback\.[^\\/]+[\\/]Scripts[\\/]python\.exe$' -and
            $probeCalls[0].Code -match 'torch\.xpu\.is_available\(\)')
        Check "the extracted probe is back in place" (
            (Get-Command Invoke-BoundedPythonProbe).ScriptBlock.ToString() -match 'ProcessStartInfo')
    }
    Check "and the environment really was discarded" (-not (Test-Path -LiteralPath $VenvDir))
    $script:StudioPreservedXpuVerdict = $false
    $script:StudioNoRollback = $false
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    Write-Host "the discard message tells the truth under --no-rollback"
    # A delete that failed frees no space, so it must not report discarded.
    $script:said = @()
    function substep { param([string]$Message, [string]$Color) $script:said += $Message }
    function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) $script:said += $Message }

    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:StudioNoRollback = $true
    Start-StudioVenvRollback -ExistingDir $VenvDir
    Check "--no-rollback reports the discard" (
        ($script:said -join "`n") -match 'discarded \(--no-rollback\)')

    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:StudioNoRollback = $false
    $script:said = @()
    Start-StudioVenvRollback -ExistingDir $VenvDir
    Check "an ordinary install keeps the rollback copy" ($script:StudioVenvRollbackActive)
    Check "and says nothing about discarding it" (
        ($script:said -join "`n") -notmatch 'discarded')
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }
    $script:StudioNoRollback = $true

    # Shadows the extracted definition, so Start-StudioVenvRollback resolves to this one.
    [System.IO.Directory]::CreateDirectory($VenvDir) | Out-Null
    [System.IO.File]::WriteAllText((Join-Path $VenvDir "generation"), "old")
    Reset-RollbackState $VenvDir
    $script:said = @()
    function Remove-StudioVenvTreeWithRetry { param([string]$Path, [string]$Label) return $false }
    try {
        $discardThrew = $false
        try { Start-StudioVenvRollback -ExistingDir $VenvDir } catch { $discardThrew = $true }
    } finally {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath Function:\Remove-StudioVenvTreeWithRetry -Force
    }
    $joined = ($script:said -join "`n")
    Check "a discard that could not delete never aborts the install" (-not $discardThrew)
    Check "a discard that could not delete does not claim success" (
        $joined -notmatch 'discarded \(--no-rollback\)')
    # Test-Path follows a dangling reparse point and calls it absent.
    $danglingRoot = Join-Path $StudioHome "dangling"
    [System.IO.Directory]::CreateDirectory($danglingRoot) | Out-Null
    $danglingTarget = Join-Path $danglingRoot "gone"
    [System.IO.Directory]::CreateDirectory($danglingTarget) | Out-Null
    $danglingLink = Join-Path $danglingRoot "link"
    $danglingMade = $true
    try { New-Item -ItemType SymbolicLink -Path $danglingLink -Target $danglingTarget -ErrorAction Stop | Out-Null }
    catch { $danglingMade = $false }
    if ($danglingMade) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $danglingTarget -Recurse -Force
        # Test-Path does not follow a dangling symlink on Linux, so only the helper is checked.
        Check "a dangling link is reported as present" (Test-StudioPathPresent -Path $danglingLink)
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $danglingLink -Force -ErrorAction SilentlyContinue
    } else {
        Write-Host "  SKIP  this host will not create a symbolic link"
    }
    Microsoft.PowerShell.Management\Remove-Item -LiteralPath $danglingRoot -Recurse -Force -ErrorAction SilentlyContinue

    Check "a discard that could not delete names the path left on disk" (
        $joined -match [regex]::Escape($StudioHome) -and $joined -match 'unsloth_studio\.rollback\.')
    $script:StudioNoRollback = $false
    foreach ($c in @(Get-ChildItem -LiteralPath $StudioHome -Directory -Filter "unsloth_studio.rollback.*" -ErrorAction SilentlyContinue)) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $c.FullName -Recurse -Force -ErrorAction SilentlyContinue
    }

    function substep { param([string]$Message, [string]$Color) }
    function Write-StudioLine { param([string]$Message, [string]$ForegroundColor) Write-Host $Message }
} finally {
    if (Test-Path -LiteralPath $StudioHome) {
        Microsoft.PowerShell.Management\Remove-Item -LiteralPath $StudioHome -Recurse -Force
    }
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
