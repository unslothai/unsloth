# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

<#
.SYNOPSIS
    Record what an Unsloth install left behind, so two commits can be compared.

.DESCRIPTION
    Runs after install.ps1 and writes three files that compare_installer_evidence.py consumes:

      shortcuts.json   every .lnk the installer created, read back through the shell, with the
                       fields that constitute the launch contract: target, arguments (which carry
                       -WindowStyle and -ExecutionPolicy), working directory, window style, icon.
      artifacts.json   the installed files, with the content of the two whose content is a contract
                       (launch-studio.ps1 and unsloth.cmd), plus which files a second install run
                       rewrote.
      (the transcript is captured by the workflow, which owns the installer's stdout)

    Reading the .lnk files back through the shell rather than trusting the installer's own log is
    the point: a change to the launch transport is invisible in the transcript, and the transcript
    is what a prose review looks at.

    This script never fails the job. A collection error is recorded in the JSON as an error field so
    the comparer can call the run VOID, which is a different and more honest outcome than a step
    that went red for a reason nobody reads.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$StudioHome,

    # Normally %LOCALAPPDATA%\Unsloth Studio, outside $StudioHome; launch-studio.ps1 lives here.
    [string]$StudioDataDir,

    [Parameter(Mandatory = $true)][string]$OutDir,

    [string[]]$ShortcutRoot,

    # Second install of the same commit: compare against the first run's manifest.
    [string]$CompareAgainst
)

$ErrorActionPreference = 'Continue'

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

if (-not $ShortcutRoot -or $ShortcutRoot.Count -eq 0) {
    $ShortcutRoot = @(
        [Environment]::GetFolderPath('Desktop'),
        [Environment]::GetFolderPath('CommonDesktopDirectory')
    )
    # %APPDATA% is unset off Windows and Join-Path throws on null.
    if ($env:APPDATA) {
        $ShortcutRoot += (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
    }
}

function Get-UnslothRootLabel {
    <#
    .SYNOPSIS
    A stable name for a shortcut root that does not vary between the two workspaces.

    Split-Path -Leaf was not stable ENOUGH: it maps the per-user Desktop and CommonDesktopDirectory
    to the same "Desktop", so a candidate that moved a shortcut from one to the other produced an
    identical key with identical fields and compared equal.
    #>
    param([Parameter(Mandatory = $true)][string]$Path)
    $known = [ordered]@{
        UserDesktop   = [Environment]::GetFolderPath('Desktop')
        CommonDesktop = [Environment]::GetFolderPath('CommonDesktopDirectory')
        CommonPrograms = [Environment]::GetFolderPath('CommonPrograms')
    }
    if ($env:APPDATA) {
        $known['UserPrograms'] = (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs')
    }
    foreach ($entry in $known.GetEnumerator()) {
        if ($entry.Value -and $Path -ieq $entry.Value) { return $entry.Key }
    }
    return "other:" + (Split-Path $Path -Leaf)
}

$shortcuts = New-Object System.Collections.ArrayList
$shell = $null
try {
    $shell = New-Object -ComObject WScript.Shell
} catch {
    [void]$shortcuts.Add([ordered]@{
        name  = '<collection failed>'
        error = "could not create WScript.Shell: $($_.Exception.Message)"
    })
}

if ($shell) {
    foreach ($root in ($ShortcutRoot | Where-Object { $_ })) {
        if (-not (Test-Path -LiteralPath $root)) { continue }
        $found = @(Get-ChildItem -LiteralPath $root -Filter '*.lnk' -Recurse -ErrorAction SilentlyContinue |
                   Where-Object { $_.Name -like '*nsloth*' })
        foreach ($file in $found) {
            try {
                $lnk = $shell.CreateShortcut($file.FullName)
                # Key on root label + relative path: workspaces differ, and leaf names collide across roots.
                $relative = $file.FullName
                if ($file.FullName.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
                    $relative = $file.FullName.Substring($root.Length).TrimStart('\', '/')
                }
                [void]$shortcuts.Add([ordered]@{
                    name             = $relative
                    root             = (Get-UnslothRootLabel $root)
                    # Catches an unconditional Save() that leaves every property identical.
                    lastWriteUtc     = $file.LastWriteTimeUtc.ToString('o')
                    targetPath       = $lnk.TargetPath
                    arguments        = $lnk.Arguments
                    workingDirectory = $lnk.WorkingDirectory
                    windowStyle      = [string]$lnk.WindowStyle
                    iconLocation     = $lnk.IconLocation
                    description      = $lnk.Description
                })
            } catch {
                [void]$shortcuts.Add([ordered]@{
                    name  = $file.Name
                    error = "could not read: $($_.Exception.Message)"
                })
            }
        }
    }
}

# -Depth because the default of 2 silently emits type names.
# Always an array: ConvertTo-Json unwraps single elements and 5.1 lacks -AsArray.
$shortcutJson = $shortcuts | ConvertTo-Json -Depth 6
if ($shortcuts.Count -le 1) { $shortcutJson = "[$($shortcutJson)]" }
Set-Content -LiteralPath (Join-Path $OutDir 'shortcuts.json') -Value $shortcutJson -Encoding utf8

Write-Host "collected $($shortcuts.Count) shortcut(s)"
foreach ($s in $shortcuts) {
    Write-Host ("  {0}: {1} {2}" -f $s.name, $s.targetPath, $s.arguments)
}

# Content is compared only for installer-generated templates, listed per possible layout.
# Windows-only contracts: listing a file the Windows installer never writes voids every run.
$contentFiles = [ordered]@{
    'launch-studio.ps1' = @('launch-studio.ps1', 'share\launch-studio.ps1')
    'unsloth.cmd'       = @('unsloth.cmd', 'bin\unsloth.cmd')
}

$files = [ordered]@{}
$artifactError = $null

# $StudioHome first so the canonical location wins.
$searchRoots = [ordered]@{ 'home' = $StudioHome }
if ($StudioDataDir -and $StudioDataDir -ne $StudioHome) { $searchRoots['data'] = $StudioDataDir }

if (Test-Path -LiteralPath $StudioHome) {
    foreach ($contract in $contentFiles.Keys) {
        $full = $null
        $foundAt = $null
        foreach ($rootLabel in $searchRoots.Keys) {
            $root = $searchRoots[$rootLabel]
            if (-not $root -or -not (Test-Path -LiteralPath $root)) { continue }
            foreach ($candidate in $contentFiles[$contract]) {
                $probe = Join-Path $root $candidate
                if (Test-Path -LiteralPath $probe) { $full = $probe; $foundAt = "$rootLabel/$candidate"; break }
            }
            if ($full) { break }
        }
        if (-not $full) {
            # Recorded as an error, not skipped, so a contract missing on both sides cannot compare equal.
            $files[$contract] = [ordered]@{
                error = "not found at any supported location under " +
                        (($searchRoots.Keys | ForEach-Object { $_ }) -join ', ')
            }
            continue
        }
        try {
            $item = Get-Item -LiteralPath $full -ErrorAction Stop
            $files[$contract] = [ordered]@{
                foundAt      = $foundAt
                content      = (Get-Content -Raw -LiteralPath $full -ErrorAction Stop)
                sha256       = (Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash
                # Track the BOM separately: Get-Content drops it, and PS 5.1 needs it for non-ASCII paths.
                bom          = (& {
                    $head = [byte[]]::new(3)
                    $stream = [System.IO.File]::OpenRead($full)
                    try { $read = $stream.Read($head, 0, 3) } finally { $stream.Dispose() }
                    if ($read -ge 3 -and $head[0] -eq 0xEF -and $head[1] -eq 0xBB -and $head[2] -eq 0xBF) { 'utf-8' }
                    elseif ($read -ge 2 -and $head[0] -eq 0xFF -and $head[1] -eq 0xFE) { 'utf-16le' }
                    elseif ($read -ge 2 -and $head[0] -eq 0xFE -and $head[1] -eq 0xFF) { 'utf-16be' }
                    else { 'none' }
                })
                # Write time catches an unconditional rewrite of identical bytes.
                lastWriteUtc = $item.LastWriteTimeUtc.ToString('o')
                length       = $item.Length
            }
        } catch {
            $files[$contract] = [ordered]@{ error = $_.Exception.Message }
        }
    }
    try {
        $tops = @(Get-ChildItem -LiteralPath $StudioHome -ErrorAction Stop |
                  Select-Object -ExpandProperty Name | Sort-Object)
        foreach ($name in $tops) {
            if (-not $files.Contains($name)) { $files[$name] = [ordered]@{ present = $true } }
        }
    } catch {
        $artifactError = "could not enumerate ${StudioHome}: $($_.Exception.Message)"
    }
} else {
    $artifactError = "the install root $StudioHome does not exist, so the installer did not finish"
}

# Built before the comparison below, which reads it; stored in artifacts.json for the second run.
$shortcutWrites = [ordered]@{}
foreach ($s in $shortcuts) {
    if ($s.Contains('lastWriteUtc')) { $shortcutWrites["$($s.root)/$($s.name)"] = $s.lastWriteUtc }
}

$rewritten = $null
if ($CompareAgainst) {
    if (-not (Test-Path -LiteralPath $CompareAgainst)) {
        # Not an empty list: $null means unmeasured, which the comparer treats differently.
        Write-Host "::warning::no first-run manifest at $CompareAgainst, so idempotency was not measured"
    } else {
        $rewritten = New-Object System.Collections.ArrayList
        try {
            $before = Get-Content -Raw -LiteralPath $CompareAgainst | ConvertFrom-Json
            foreach ($contract in $contentFiles.Keys) {
                $b = $before.files.$contract
                $a = $files[$contract]
                if (-not $b -and -not $a) { continue }
                if (-not $b -or -not $a) { [void]$rewritten.Add($contract); continue }
                if ($b.sha256 -and $a.sha256 -and $b.sha256 -ne $a.sha256) {
                    [void]$rewritten.Add("$contract (contents changed)")
                    continue
                }
                if ($b.lastWriteUtc -and $a.lastWriteUtc -and $b.lastWriteUtc -ne $a.lastWriteUtc) {
                    [void]$rewritten.Add("$contract (rewritten with identical bytes at $($a.lastWriteUtc))")
                    continue
                }
                if ($b.foundAt -and $a.foundAt -and $b.foundAt -ne $a.foundAt) {
                    [void]$rewritten.Add("$contract (moved from $($b.foundAt) to $($a.foundAt))")
                }
            }
            # Where-Object: an empty PSObject.Properties enumerates to $null, giving a phantom '' key.
            $beforeFileKeys = @($before.files.PSObject.Properties.Name | Where-Object { $_ })
            $nowFileKeys = @($files.Keys | Where-Object { $_ })
            $contractNames = @($contentFiles.Keys)
            foreach ($name in $beforeFileKeys) {
                if ($contractNames -contains $name) { continue }
                if ($nowFileKeys -notcontains $name) {
                    [void]$rewritten.Add("$name (removed by the second run)")
                }
            }
            foreach ($name in $nowFileKeys) {
                if ($contractNames -contains $name) { continue }
                if ($beforeFileKeys -notcontains $name) {
                    [void]$rewritten.Add("$name (created by the second run)")
                }
            }
            if ($before.PSObject.Properties.Name -contains 'shortcutWrites' -and $before.shortcutWrites) {
                $beforeKeys = @($before.shortcutWrites.PSObject.Properties.Name | Where-Object { $_ })
                foreach ($key in $shortcutWrites.Keys) {
                    if ($beforeKeys -notcontains $key) {
                        [void]$rewritten.Add("shortcut $key (created by the second run)")
                        continue
                    }
                    $wasWritten = $before.shortcutWrites.$key
                    if ($wasWritten -and $wasWritten -ne $shortcutWrites[$key]) {
                        [void]$rewritten.Add("shortcut $key (rewritten at $($shortcutWrites[$key]))")
                    }
                }
                # Also catch shortcuts the second run deleted.
                foreach ($key in $beforeKeys) {
                    # -contains on .Keys: OrderedDictionary has no ContainsKey.
                    if (@($shortcutWrites.Keys | Where-Object { $_ }) -notcontains $key) {
                        [void]$rewritten.Add("shortcut $key (removed by the second run)")
                    }
                }
            } elseif ($shortcuts.Count -gt 0) {
                Write-Host '::warning::the first-run manifest carries no shortcut write times, so second-run shortcut writes could not be measured'
                throw 'first-run shortcut evidence is missing'
            }
        } catch {
            Write-Host "::warning::could not compare against $CompareAgainst : $($_.Exception.Message)"
            $rewritten = $null
        }
    }
}

# The persisted install ID must match the one baked into the launcher, or Studio will not launch.
# They are expected to differ between base and head.
$installId = $null
foreach ($rootLabel in $searchRoots.Keys) {
    $root = $searchRoots[$rootLabel]
    if (-not $root) { continue }
    foreach ($candidate in @('share\studio_install_id', 'studio_install_id')) {
        $probe = Join-Path $root $candidate
        if (Test-Path -LiteralPath $probe) {
            try { $installId = (Get-Content -Raw -LiteralPath $probe -ErrorAction Stop).Trim() } catch {}
            break
        }
    }
    if ($installId) { break }
}

$embeddedId = $null
if ($files['launch-studio.ps1'] -and $files['launch-studio.ps1'].Contains('content')) {
    $m = [regex]::Match($files['launch-studio.ps1'].content, "_ExpectedStudioRootId\s*=\s*'([^']*)'")
    if ($m.Success) { $embeddedId = $m.Groups[1].Value }
}

$artifacts = [ordered]@{
    studioHome     = $StudioHome
    files          = $files
    shortcutWrites = $shortcutWrites
    installId      = $installId
    embeddedId     = $embeddedId
}
if ($artifactError) { $artifacts['error'] = $artifactError }
if ($null -ne $rewritten) { $artifacts['rewrittenOnSecondRun'] = @($rewritten) }

$artifacts | ConvertTo-Json -Depth 8 |
    Set-Content -LiteralPath (Join-Path $OutDir 'artifacts.json') -Encoding utf8

Write-Host "collected $($files.Count) artifact entr(ies)$(if ($artifactError) { " [$artifactError]" })"
if ($null -ne $rewritten) {
    Write-Host "second run rewrote: $(if ($rewritten.Count) { $rewritten -join ', ' } else { 'nothing' })"
}

# Always zero: a collection failure is data for the comparer.
exit 0
