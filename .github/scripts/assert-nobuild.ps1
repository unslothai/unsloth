# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# The `nobuild` contract from clean-machine-assert.sh, for Windows.
#
# Why a port and not `shell: bash`: the clean-machine scrub drops every `*\Git\*`
# entry from PATH and from the Machine/User registry copies, and the bash version
# needs sed/grep/tr/sort out of C:\Program Files\Git\usr\bin. This also runs inside
# the servercore container, which has no bash at all. Both Windows lanes call this
# one file so the sdist allowlist cannot drift between them.
#
# Usage: assert-nobuild.ps1 -LogPath logs/install.log   (exit 1 = a source build)
[CmdletBinding()]
param([Parameter(Mandatory = $true)][string] $LogPath)

if (-not (Test-Path -LiteralPath $LogPath)) {
    Write-Host "::error::nobuild requested but $LogPath is missing"
    exit 1
}

# "Built an sdist" is NOT "needed a compiler". Every name here was checked against
# its own sdist: setuptools.build_meta backend, no ext_modules, no .c/.cpp/.pyx/.rs
# file, so its PEP 517 build is a pure-Python copy step. UNSLOTH_ALLOW_SDIST extends
# the list. Kept identical to clean-machine-assert.sh's `_allow`.
$allow = @('openai-whisper', 'argbind', 'randomname', 'antlr4-python3-runtime', 'triton-kernels')
if ($env:UNSLOTH_ALLOW_SDIST) {
    $allow += ($env:UNSLOTH_ALLOW_SDIST -split '\s+' | Where-Object { $_ })
}
# Lowercased and underscore-folded on both sides: a distribution name and the name uv
# prints can disagree on the separator (triton_kernels vs triton-kernels).
$allow = @($allow | ForEach-Object { $_.ToLowerInvariant() -replace '_', '-' })

# [char]27, not "`e": the `e escape is PowerShell 6+, and this runs under Windows
# PowerShell 5.1 too, where "`e" degrades to a literal "e" and the strip would eat
# real text instead of ANSI codes.
$esc = [char]27
$text = (Get-Content -LiteralPath $LogPath -Raw) -replace "$esc\[[0-9;]*[A-Za-z]", ''
$built = @()
foreach ($line in ($text -split "`r?`n")) {
    # A local-path build is something the caller pointed at (the CI source overlay),
    # never something dependency resolution chose. Index dependencies always print
    # `<name>==<version>`, so no signal is lost.
    if ($line -imatch 'building [a-z0-9._-]+ @ file://') { continue }
    # pip prints `Building wheel for <pkg>`, uv prints `Building <pkg>==<ver>`
    # (astral-sh/uv#11165). Requiring `==` or ` @ ` after the name keeps this off the
    # installer's own lowercase "building frontend..." progress text.
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
