# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Windows port of clean-machine-assert.sh nobuild; must not need bash (PATH scrub, servercore).
# Usage: assert-nobuild.ps1 -LogPath logs/install.log   (exit 1 = a source build)
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string] $LogPath)

if (-not (Test-Path -LiteralPath $LogPath)) {
    Write-Host "::error::nobuild requested but $LogPath is missing"
    exit 1
}

# Pure-Python sdists; keep in sync with clean-machine-assert.sh. Drop diffusers once a release has the wheel pin.
$allow = @('openai-whisper', 'argbind', 'randomname', 'antlr4-python3-runtime', 'triton-kernels', 'diffusers')
if ($env:UNSLOTH_ALLOW_SDIST) {
    $allow += ($env:UNSLOTH_ALLOW_SDIST -split '\s+' | Where-Object { $_ })
}
# Normalise case and separators: uv and the dist name can differ (triton_kernels).
$allow = @($allow | ForEach-Object { $_.ToLowerInvariant() -replace '_', '-' })

# [char]27, not "`e", which is PS 6+ only.
$esc = [char]27
$text = (Get-Content -LiteralPath $LogPath -Raw) -replace "$esc\[[0-9;]*[A-Za-z]", ''
$built = @()
foreach ($line in ($text -split "`r?`n")) {
    # A local file:// build is the overlay the caller chose, not resolution.
    if ($line -imatch 'building [a-z0-9._-]+ @ file://') { continue }
    # pip prints `Building wheel for <pkg>`, uv `Building <pkg>==<ver>`; `==`/` @ ` skips frontend text.
    foreach ($m in [regex]::Matches($line, '(?i)building wheel for ([a-z0-9._-]+)|building ([a-z0-9._-]+)(==| @ )')) {
        $name = if ($m.Groups[1].Success) { $m.Groups[1].Value } else { $m.Groups[2].Value }
        $built += ($name.ToLowerInvariant() -replace '_', '-')
    }
}
$built = @($built | Sort-Object -Unique)
$bad = @($built | Where-Object { $allow -notcontains $_ })

$rc = 0
if ($bad.Count -gt 0) {
    Write-Host "::error::built from source: $($bad -join ' ') -- these must resolve to wheels on a clean machine"
    $rc = 1
} else {
    Write-Host "[assert] OK  no non-allowlisted source build (built: $(if ($built) { $built -join ' ' } else { 'none' }))"
}
$compilerErr = Select-String -Path $LogPath -Pattern "error: command '(cc|gcc|clang|cl)' failed", 'clang: error', 'cargo: not found', 'Microsoft Visual C\+\+ 14.0 or greater is required'
if ($compilerErr) {
    Write-Host '::error::compiler invocation appears in the install log'
    $compilerErr | Select-Object -First 10 | ForEach-Object { Write-Host "  $($_.Line)" }
    $rc = 1
}
exit $rc
