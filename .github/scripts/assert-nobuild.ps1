# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# The `nobuild` contract from clean-machine-assert.sh, for Windows. A port, not
# `shell: bash`: the scrub drops every `*\Git\*` PATH entry the bash version needs
# sed/grep/tr/sort from, and it also runs inside the servercore container, which has no
# bash. Both Windows lanes call this one file so the sdist allowlist cannot drift.
#
# Usage: assert-nobuild.ps1 -LogPath logs/install.log   (exit 1 = a source build)
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string] $LogPath)

if (-not (Test-Path -LiteralPath $LogPath)) {
    Write-Host "::error::nobuild requested but $LogPath is missing"
    exit 1
}

# "Built an sdist" is NOT "needed a compiler": each name was verified against its own
# sdist (setuptools.build_meta, no ext_modules, no .c/.cpp/.pyx/.rs), so its PEP 517
# build is a pure-Python copy step. clean-machine-assert.sh carries the per-name detail.
# diffusers is pinned to a source archive because MiniMax-H3 support is not in any release
# yet; remove it here once a release carries H3. clean-machine-assert.sh carries the detail.
$allow = @('openai-whisper', 'argbind', 'randomname', 'antlr4-python3-runtime', 'triton-kernels', 'diffusers')
if ($env:UNSLOTH_ALLOW_SDIST) {
    $allow += ($env:UNSLOTH_ALLOW_SDIST -split '\s+' | Where-Object { $_ })
}
# Lowercased and underscore-folded on both sides: the distribution name and the name uv
# prints can differ on the separator (triton_kernels vs triton-kernels).
$allow = @($allow | ForEach-Object { $_.ToLowerInvariant() -replace '_', '-' })

# [char]27, not "`e": that escape is PowerShell 6+ and degrades to a literal "e" under
# 5.1, so the strip would eat real text instead of ANSI codes.
$esc = [char]27
$text = (Get-Content -LiteralPath $LogPath -Raw) -replace "$esc\[[0-9;]*[A-Za-z]", ''
$built = @()
foreach ($line in ($text -split "`r?`n")) {
    # A local-path build is one the caller pointed at (the overlay), never one
    # resolution chose; index deps always print `==<version>`.
    if ($line -imatch 'building [a-z0-9._-]+ @ file://') { continue }
    # pip prints `Building wheel for <pkg>`, uv `Building <pkg>==<ver>`
    # (astral-sh/uv#11165); the `==` / ` @ ` keeps this off the installer's own
    # "building frontend..." text.
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
# Independent of package names: a compiler error means a toolchain was needed.
$compilerErr = Select-String -Path $LogPath -Pattern "error: command '(cc|gcc|clang|cl)' failed", 'clang: error', 'cargo: not found', 'Microsoft Visual C\+\+ 14.0 or greater is required'
if ($compilerErr) {
    Write-Host '::error::compiler invocation appears in the install log'
    $compilerErr | Select-Object -First 10 | ForEach-Object { Write-Host "  $($_.Line)" }
    $rc = 1
}
exit $rc
