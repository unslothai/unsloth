#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# uninstall.ps1 must remove the same ~/.unsloth artifacts as uninstall.sh; one leftover lock
# blocks the ~/.unsloth prune. The body cannot run here, so this asserts list parity.
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_prebuilt_parity.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$shPath  = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.sh")
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
[System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors) | Out-Null
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)
if ($errors -and $errors.Count -gt 0) { $errors | ForEach-Object { Write-Host "    $_" } }

$shText  = Get-Content -LiteralPath $shPath -Raw
$ps1Text = Get-Content -LiteralPath $ps1Path -Raw

# Only direct children: nested paths are removed with their parent.
$shArtifacts = [regex]::Matches($shText, '_remove_path\s+"\$HOME/\.unsloth/([^"/]+)"') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
Check "found the POSIX artifact list" ($shArtifacts.Count -ge 5)

$wslOnly = @("librocdxg", "rocm-smoketest")
foreach ($artifact in $shArtifacts) {
    if ($wslOnly -contains $artifact) { continue }
    Check "uninstall.ps1 removes ~/.unsloth/$artifact" ($ps1Text -match [regex]::Escape($artifact))
}

$shLocks = [regex]::Matches($shText, '_remove_lock_file\s+"\$HOME/\.unsloth/\.([^"/]+)\.install\.lock"') |
    ForEach-Object { $_.Groups[1].Value } | Sort-Object -Unique
Check "found the POSIX lock list" ($shLocks -contains "audio.cpp")
$stalePattern = [regex]::Match($ps1Text, "StaleLockPattern = '([^']+)'").Groups[1].Value
foreach ($lock in $shLocks) {
    Check "uninstall.ps1 removes ~/.unsloth/.$lock.install.lock" ($ps1Text -match [regex]::Escape("`".$lock.install.lock`""))
    Check "uninstall.ps1 sweeps stale .$lock.install.lock" (".$lock.install.lock.stale.123" -match $stalePattern)
}

Check "uninstall.sh removes the audio.cpp link farm"  ($shText  -match 'unsloth-audiocpp-links')
Check "uninstall.ps1 removes the audio.cpp link farm" ($ps1Text -match 'unsloth-audiocpp-links')

# A crash in install_node_prebuilt.py can strand a `.stale.<pid>` lock, blocking the prune.
Check "uninstall.sh sweeps stale install locks"  ($shText  -match '\.install\.lock\.stale\.')
Check "uninstall.ps1 sweeps stale install locks" ($ps1Text -match '\.install\.lock\.stale\.')

Check "uninstall.ps1 prunes ~/.unsloth only when empty" `
    ($ps1Text -match 'Get-ChildItem -LiteralPath \$defaultUnslothHome')

Check "uninstall.sh reads uv-cache-dir"  ($shText  -match 'uv-cache-dir')
Check "uninstall.ps1 reads uv-cache-dir" ($ps1Text -match 'uv-cache-dir')
Check "uninstall.sh names uv cache clean"  ($shText  -match 'uv cache clean')
Check "uninstall.ps1 names uv cache clean" ($ps1Text -match 'uv cache clean')

if ($failures -gt 0) { Write-Host ""; Write-Host "FAILED ($failures)" -ForegroundColor Red; exit 1 }
Write-Host ""; Write-Host "All tests passed."; exit 0
