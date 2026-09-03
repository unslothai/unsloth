#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Windows on ARM has no torchaudio wheel on ANY index, so every XPU spec list must drop it.
#
# The fresh XPU install already did; the flavor repair did not, and that is the path a MIGRATED
# win-arm64 venv takes -- the repair asks uv for a torchaudio that does not exist for win_arm64
# and fails outright, before setup.ps1 can reach its ARM-aware CPU fallback.
#
# One builder now serves both, since the two copies drifted the moment only one learned about
# ARM. The tests EXECUTE the builder, then assert by AST that the repair site calls it.
# Run: pwsh -NoProfile -File tests/studio/test_xpu_arm64_torchaudio.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPs1 = Join-Path $repo "install.ps1"

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPs1, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

function Get-FunctionAst {
    param([string] $Name)
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $Name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $Name in install.ps1, found $($fn.Count)" }
    return $fn[0]
}

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# --- Get-XpuTorchSpecs, executed ------------------------------------------------------------
$specsFn = Get-FunctionAst "Get-XpuTorchSpecs"
# An extraction that lost the ARM arm would make every case below pass vacuously.
Check "extraction kept the arm64 branch" ($specsFn.Extent.Text -match 'win-arm64')
. ([scriptblock]::Create($specsFn.Extent.Text))

$arm = Get-XpuTorchSpecs -Platform "win-arm64"
$x64 = Get-XpuTorchSpecs -Platform "win-amd64"
Check "arm64 drops torchaudio"        (-not ($arm -match '^torchaudio'))
Check "arm64 keeps torch"             (($arm | Where-Object { $_ -eq 'torch>=2.6,<2.11.0' }).Count -eq 1)
Check "arm64 keeps torchvision"       (($arm | Where-Object { $_ -eq 'torchvision>=0.21,<0.26.0' }).Count -eq 1)
Check "arm64 asks for exactly two"    ($arm.Count -eq 2)
Check "x64 keeps torchaudio"          (($x64 | Where-Object { $_ -eq 'torchaudio>=2.6,<2.11.0' }).Count -eq 1)
Check "x64 asks for the full trio"    ($x64.Count -eq 3)
# The floor is not cosmetic: unsloth/models/_utils.py raises at import for an XPU device on
# torch < 2.6, so a 2.4 floor installs an environment that cannot run.
Check "floor is 2.6 on both"          (($arm[0] -eq 'torch>=2.6,<2.11.0') -and ($x64[0] -eq 'torch>=2.6,<2.11.0'))
# An unaskable interpreter yields "", and a Linux/macOS platform is never win-arm64: both must
# keep torchaudio rather than dropping it everywhere.
foreach ($p in @("", "linux-x86_64", "macosx-14.0-arm64", "win32")) {
    $label = if ($p) { $p } else { "<empty>" }
    Check "platform '$label' keeps torchaudio" ((Get-XpuTorchSpecs -Platform $p).Count -eq 3)
}
# Belt and braces: -eq is case-insensitive here and the probe lowercases too.
Check "an uppercase arm64 is still arm64" ((Get-XpuTorchSpecs -Platform "WIN-ARM64").Count -eq 2)
Check "the probe lowercases its answer"   ((Get-FunctionAst "Get-VenvPlatformTag").Extent.Text -match 'ToLowerInvariant')

# --- every call site goes through it ----------------------------------------------------------
# Text, not AST, for the call count: asserting the COUNT catches a copy that quietly
# reintroduces its own literal trio. Three sites now -- the fresh XPU install, the flavor
# repair, and the release-preservation probe that vets a kept torch against the XPU window.
$src = Get-Content -Raw -LiteralPath $installPs1
Check "builder is used at 3 sites" (([regex]::Matches($src, 'Get-XpuTorchSpecs -Platform')).Count -eq 3)
# The literal trio must exist in exactly ONE place now (the builder itself), or the drift is back.
Check "one literal torchaudio 2.6 pin" (([regex]::Matches($src, '"torchaudio>=2\.6,<2\.11\.0"')).Count -eq 1)

# The repair site specifically. A kept-release pin substitutes into the trio one spec at a
# time and has to be restorable the same way, so the range trio is held in $_origFixSpecs
# and $_fixSpecs is rebuilt from it; assert on the former.
$origAssign = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $n.Left.Extent.Text -eq '$_origFixSpecs' -and
    $n.Right.Extent.Text -match 'Get-XpuTorchSpecs'
}, $true)
Check "the repair calls the builder" ($origAssign.Count -eq 1)
# The non-XPU arm keeps the generic 2.4 floor: only the ceiling moved to the 2.11 line.
$defaultAssign = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $n.Left.Extent.Text -eq '$_fixTorchSpec' -and
    $n.Right.Extent.Text -match '^"torch>='
}, $true)
Check "the repair keeps the 2.4 CUDA floor" ($defaultAssign.Count -eq 1 -and $defaultAssign[0].Right.Extent.Text -eq '"torch>=2.4,<2.12.0"')

if ($failures -gt 0) { Write-Host "FAILED: $failures" -ForegroundColor Red; exit 1 }
Write-Host "All XPU arm64 torchaudio checks passed."
