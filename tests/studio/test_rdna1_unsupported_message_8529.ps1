#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# RDNA 1 "not covered by ROCm" wording in install.ps1 and studio/setup.ps1. RDNA 1 now
# routes to AMD's multi-arch index, so Polaris (gfx803) exercises the unsupported wording.
# Runs the table under .NET, the only place -match semantics are real.
# Run: pwsh -NoProfile -File tests/studio/test_rdna1_unsupported_message_8529.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Invoke-Expression must run at script scope; inside a function the value is lost.
function Get-AssignmentSource($path, $varName) {
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$null, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    $hits = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
        $n.Left.Extent.Text -eq $varName
    }, $true)
    if ($hits.Count -lt 1) { throw "expected $varName in $path, found none" }
    return $hits[0].Extent.Text
}

foreach ($file in @("install.ps1", "studio/setup.ps1")) {
    $path = Join-Path $root $file
    Write-Host ""
    Write-Host "=== $file ==="

    Invoke-Expression (Get-AssignmentSource $path '$unsupportedNameArchTable')

    function Resolve-Unsupported($name) {
        foreach ($row in $unsupportedNameArchTable) {
            if ($name -match $row.P) { return $row.A }
        }
        return $null
    }

    Check "RX 5700 XT no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon RX 5700 XT"))
    Check "RX 5700 no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon RX 5700"))
    Check "RX 5600 XT no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon RX 5600 XT"))
    Check "Radeon Pro 5600 XT no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro 5600 XT"))
    Check "Radeon Pro V520 no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro V520"))
    Check "Radeon Pro 5600M no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro 5600M"))
    Check "RX 5500 XT no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon RX 5500 XT"))
    # Professional boards LLVM's table omits, mapped via libdrm amdgpu.ids and pci.ids.
    Check "Radeon Pro W5700 no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro W5700"))
    Check "Radeon Pro W5700X no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro W5700X"))
    Check "Radeon Pro W5500 no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro W5500"))
    Check "Radeon Pro W5500M no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro W5500M"))
    Check "Radeon Pro W5300M no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro W5300M"))
    Check "RX 5300 no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon RX 5300"))
    Check "RX 5300M no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon RX 5300M"))
    # Mac Pro MPX boards: Navi 10 parts with neither "RX 5700" nor a W prefix.
    Check "Radeon Pro 5700 XT no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro 5700 XT"))
    Check "Radeon Pro 5700 no longer unsupported (routes on Windows, #11614)" ($null -eq (Resolve-Unsupported "AMD Radeon Pro 5700"))
    # "W5700" must not be read out of "W7500".
    Check "PRO W7500 unclaimed"            ($null -eq (Resolve-Unsupported "AMD Radeon PRO W7500"))
    Check "PRO W6500 unclaimed"            ($null -eq (Resolve-Unsupported "AMD Radeon PRO W6500"))
    Check "PRO W6400 unclaimed"            ($null -eq (Resolve-Unsupported "AMD Radeon PRO W6400"))

    # RX 580: Polaris 20, gfx803, with no ROCm PyTorch wheels.
    Check "RX 580 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 580") -eq 'gfx803')
    Check "RX 580 Series -> gfx803"        ((Resolve-Unsupported "AMD Radeon RX 580 Series") -eq 'gfx803')
    Check "RX 570 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 570") -eq 'gfx803')
    Check "RX 590 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 590") -eq 'gfx803')
    Check "RX 480 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 480") -eq 'gfx803')
    Check "RX 470 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 470") -eq 'gfx803')
    # Polaris 10 workstation names carry no RX number.
    Check "Radeon Pro WX 7100 -> gfx803"   ((Resolve-Unsupported "AMD Radeon Pro WX 7100") -eq 'gfx803')
    Check "Radeon Pro WX 5100 -> gfx803"   ((Resolve-Unsupported "AMD Radeon Pro WX 5100") -eq 'gfx803')

    # Polaris 11/12 is a different die, deliberately absent: never guess an arch.
    Check "RX 560 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 560"))
    Check "RX 550 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 550"))
    Check "RX 460 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 460"))

    # "RX 570" prefixes "RX 5700": matched ALONE, since table order hides a dropped (?!0).
    $polarisPattern = @($unsupportedNameArchTable | Where-Object { $_.A -eq 'gfx803' })[0].P
    foreach ($rdna1 in @("AMD Radeon RX 5700 XT", "AMD Radeon RX 5700", "AMD Radeon RX 5500 XT")) {
        Check "Polaris pattern alone does not claim '$rdna1'" (-not ($rdna1 -match $polarisPattern))
    }

    # A hit here would print "ROCm does not cover this" at a card that has wheels.
    Check "RX 9070 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 9070 XT"))
    Check "RX 9060 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 9060 XT"))
    Check "RX 7900 XTX unclaimed"          ($null -eq (Resolve-Unsupported "AMD Radeon RX 7900 XTX"))
    Check "RX 6800 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 6800 XT"))
    Check "8060S Graphics unclaimed"       ($null -eq (Resolve-Unsupported "AMD Radeon 8060S Graphics"))
    Check "RTX 4090 unclaimed"             ($null -eq (Resolve-Unsupported "NVIDIA GeForce RTX 4090"))

    Invoke-Expression (Get-AssignmentSource $path '$nameArchTable')
    function Resolve-Supported($name) {
        foreach ($row in $nameArchTable) {
            if ($name -match $row.P) { return $row.A }
        }
        return $null
    }
    Check "RX 5700 XT -> gfx1010 (supported)"      ((Resolve-Supported "AMD Radeon RX 5700 XT") -eq 'gfx1010')
    Check "RX 5600 XT -> gfx1010 (supported)"      ((Resolve-Supported "AMD Radeon RX 5600 XT") -eq 'gfx1010')
    Check "Radeon Pro W5700 -> gfx1010 (supported)" ((Resolve-Supported "AMD Radeon Pro W5700") -eq 'gfx1010')
    Check "Radeon Pro V520 -> gfx1011 (supported)" ((Resolve-Supported "AMD Radeon Pro V520") -eq 'gfx1011')
    Check "RX 5500 XT -> gfx1012 (supported)"      ((Resolve-Supported "AMD Radeon RX 5500 XT") -eq 'gfx1012')
    Check "RX 570 still gets no supported arch"   ($null -eq (Resolve-Supported "AMD Radeon RX 570"))
    Check "RX 580 still gets no supported arch"   ($null -eq (Resolve-Supported "AMD Radeon RX 580"))
    Check "RX 9070 XT still maps to gfx1201"  ((Resolve-Supported "AMD Radeon RX 9070 XT") -eq 'gfx1201')

    # One table routes to a wheel index; the other exists because nothing routes.
    $bothTables = @($unsupportedNameArchTable | ForEach-Object { $_.A }) |
        Where-Object { @($nameArchTable | ForEach-Object { $_.A }) -contains $_ }
    Check "the two tables share no arch"   ($bothTables.Count -eq 0)

    # Both files ship CRLF, so normalise before matching multi-line needles.
    $src = (Get-Content -Raw $path) -replace "`r`n", "`n"

    # Ordering claims are guarded on "was found", so two -1s cannot pass vacuously.
    $unsupArm = 'step "gpu" "AMD GPU detected ($'
    $unknownArm = 'step "gpu" "AMD GPU detected -- arch unknown"'
    Check "unsupported gpu arm is present"  ($src.Contains($unsupArm))
    Check "arch-unknown gpu arm is present" ($src.Contains($unknownArm))
    Check "unsupported arm precedes arch-unknown arm" `
        ($src.IndexOf($unsupArm) -ge 0 -and $src.IndexOf($unknownArm) -ge 0 -and
         $src.IndexOf($unsupArm) -lt $src.IndexOf($unknownArm))

    # Scoped to the card: a host can pair an uncovered GPU with a covered one.
    $disclaimer = "setting UNSLOTH_ROCM_GFX_ARCH will not change that for it."
    Check "unsupported arm says the override cannot help" ($src.Contains($disclaimer))

    $sdkAdvice = 'substep "Could not determine the GPU arch'
    Check "HIP SDK advice still exists for genuinely unknown cards" ($src.Contains($sdkAdvice))
    Check "HIP SDK advice comes after the unsupported arm" `
        ($src.IndexOf($sdkAdvice) -ge 0 -and $src.IndexOf($unsupArm) -ge 0 -and
         $src.IndexOf($unsupArm) -lt $src.IndexOf($sdkAdvice))

    # PRINTING lines only: the same phrases appear in comments explaining the branch.
    $emitted = @(($src -split "`n") | Where-Object {
        $_ -match 'substep\s+["'']' -and $_.TrimStart() -notmatch '^#'
    })
    Check "the emitted advice offers Vulkan" `
        (($emitted -join "`n").Contains("through Vulkan"))
    # Verified by parsing: a bare NAME=vulkan parses as a command and sets nothing.
    $setter = '$env:UNSLOTH_LLAMA_CPP_BACKEND = "vulkan"'
    Check "the emitted advice teaches the current spelling" `
        (($emitted -join "`n").Contains($setter))
    $posix = @($emitted | Where-Object { $_ -match 'UNSLOTH_LLAMA_CPP_BACKEND=vulkan' })
    Check "no emitted line gives a POSIX assignment" ($posix.Count -eq 0)
    $setterAst = [System.Management.Automation.Language.Parser]::ParseInput(
        $setter, [ref]$null, [ref]$null)
    Check "the taught setter parses as an assignment, not a command" `
        (@($setterAst.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true)).Count -eq 1)

    # New text must not teach UNSLOTH_FORCE_VULKAN; setup.ps1 still reads it for back-compat.
    $teachesLegacy = @($emitted | Where-Object { $_ -match 'UNSLOTH_FORCE_VULKAN' })
    Check "the legacy spelling is not taught" ($teachesLegacy.Count -eq 0)

    # Per site: install.ps1 prints this advice twice.
    $mentions = @(0..($emitted.Count - 1) | Where-Object {
        $emitted[$_].Contains($setter)
    })
    Check "at least one site names the Vulkan variable" ($mentions.Count -ge 1)
    foreach ($i in $mentions) {
        $window = ($emitted[$i..([Math]::Min($i + 3, $emitted.Count - 1))]) -join "`n"
        Check "the advice at emitted line $($i + 1) says it applies at install time" `
            ($window.Contains("install time"))
    }

    # Evaluate the real if/elseif chain on the reported host and assert the FIRST true clause
    # is the unsupported arm; offsets cannot see unreachable branches.
    $chainAst = [System.Management.Automation.Language.Parser]::ParseFile(
        $path, [ref]$null, [ref]$null)
    $chains = @($chainAst.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.IfStatementAst] -and
        ($n.Clauses | Where-Object { $_.Item1.Extent.Text -match 'ROCmUnsupportedGfxArch' }) -and
        ($n.Clauses | Where-Object { $_.Item2.Extent.Text -match 'step "gpu"' })
    }, $true))
    Check "the gpu report chain was found" ($chains.Count -eq 1)
    if ($chains.Count -eq 1) {
        $HasNvidiaSmi = $false
        $script:IsIntelXpu = $false
        $HasROCm = $false
        $HipSdkInstalled = $true
        $ROCmGpuLabel = "AMD Radeon RX 5700 XT"
        $ROCmGfxArch = $null
        $script:ROCmGfxArch = $null
        $ROCmUnsupportedGfxArch = "gfx1010"
        $script:ROCmUnsupportedGfxArch = "gfx1010"
        $winner = -1
        $unsupIdx = -1
        for ($c = 0; $c -lt $chains[0].Clauses.Count; $c++) {
            $cond = $chains[0].Clauses[$c].Item1.Extent.Text
            if ($cond -match 'ROCmUnsupportedGfxArch' -and $cond -notmatch '-not') { $unsupIdx = $c }
            if ($winner -lt 0 -and [bool](& ([scriptblock]::Create($cond)))) { $winner = $c }
        }
        Check "the unsupported arm exists in the chain" ($unsupIdx -ge 0)
        Check "an RDNA 1 host with the HIP SDK installed reaches the unsupported arm" `
            ($winner -eq $unsupIdx)
    }

    # The unsupported lookup must never assign the arch the installers route on.
    $tableSrc = Get-AssignmentSource $path '$unsupportedNameArchTable'
    Check "the table assigns no routable arch" (-not ($tableSrc -match 'gfx1(0[3-9]|1|2)'))

    # The table has to be READ by the message arms, not just declared.
    $consumer = if ($file -eq "install.ps1") {
        'foreach ($row in $unsupportedNameArchTable) {'
    } else {
        '-Table $unsupportedNameArchTable'
    }
    Check "the table is consumed by the arch resolver" ($src.Contains($consumer))
    Check "the resolver feeds the variable the arms read" ($src -match 'ROCmUnsupportedGfxArch\s*=\s*(\$row\.A|Get-GfxArchFromGpuName)')
}

# Initialised OUTSIDE the `-not $HasNvidiaSmi` block, or Set-StrictMode aborts on NVIDIA.
$setupSrc = (Get-Content -Raw (Join-Path $root "studio/setup.ps1")) -replace "`r`n", "`n"
# Ask the parser: three `-not $HasNvidiaSmi` blocks exist and strings hold braces.
$setupAst = [System.Management.Automation.Language.Parser]::ParseFile(
    (Join-Path $root "studio/setup.ps1"), [ref]$null, [ref]$null)
$assignments = $setupAst.FindAll({
    param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $n.Left.Extent.Text -eq '$script:ROCmUnsupportedGfxArch'
}, $true)
Check "the unsupported-arch assignments were found" ($assignments.Count -gt 0)
$unconditional = @($assignments | Where-Object {
    $p = $_.Parent
    $nested = $false
    while ($null -ne $p) {
        if ($p -is [System.Management.Automation.Language.IfStatementAst]) { $nested = $true; break }
        $p = $p.Parent
    }
    -not $nested
})
Check "the unsupported-arch state is initialised outside every conditional" ($unconditional.Count -gt 0)

# The WMI fallback takes adapter [0], so a host pairing RX 5700 with RX 7900 must keep the
# arch-unknown arm. Runs the REAL block via AST.
Write-Host ""
Write-Host "=== install.ps1 mixed-adapter guard ==="
$installPath = Join-Path $root "install.ps1"
$installAst = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$null, [ref]$null)
$guardBlocks = @($installAst.FindAll({
    param($n)
    $n -is [System.Management.Automation.Language.IfStatementAst] -and
    $n.Extent.Text.Contains('$unsupportedNameArchTable') -and
    $n.Extent.Text.Contains('$ROCmUnsupportedGfxArch') -and
    $n.Extent.Text.Contains('$wmiAmdNames')
}, $true) | Sort-Object { $_.Extent.Text.Length })
# Requiring $wmiAmdNames makes deleting the peer scan fail here.
Check "the unsupported lookup block consults the peer list" ($guardBlocks.Count -gt 0)

# The peer list is report-only: it must never write the label or arch the inference reads.
$peerAssign = @($installAst.FindAll({
    param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $n.Left.Extent.Text -eq '$wmiAmdNames' -and
    $n.Right.Extent.Text.Contains('$usePeers')
}, $true))
Check "install.ps1 fills the peer list from every adapter" ($peerAssign.Count -eq 1)

$peerBlock = $null
foreach ($assign in $peerAssign) {
    $node = $assign.Parent
    while ($node) {
        if ($node -is [System.Management.Automation.Language.IfStatementAst]) {
            $peerBlock = $node
            break
        }
        $node = $node.Parent
    }
}
Check "install.ps1's peer scan writes neither the label nor the arch" (
    $peerBlock -and
    -not $peerBlock.Extent.Text.Contains('$ROCmGpuLabel =') -and
    -not $peerBlock.Extent.Text.Contains('$ROCmGfxArch =')
)

# $script:ROCmGpuLabels feeds the arch inference, so peer names must not go there.
$setupPath = Join-Path $root "studio/setup.ps1"
$setupAst = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$null, [ref]$null)
$setupPeer = @($setupAst.FindAll({
    param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    $n.Left.Extent.Text -eq '$script:ROCmPeerLabels' -and
    $n.Right.Extent.Text.Contains('$usePeers')
}, $true))
Check "setup.ps1 keeps a separate report-only peer list" ($setupPeer.Count -eq 1)
$setupPeerBlock = $null
foreach ($assign in $setupPeer) {
    $node = $assign.Parent
    while ($node) {
        if ($node -is [System.Management.Automation.Language.IfStatementAst]) { $setupPeerBlock = $node; break }
        $node = $node.Parent
    }
}
Check "setup.ps1's peer scan writes neither the label nor the inference list" (
    $setupPeerBlock -and
    -not $setupPeerBlock.Extent.Text.Contains('$ROCmGpuLabel =') -and
    -not $setupPeerBlock.Extent.Text.Contains('$script:ROCmGpuLabels =')
)

# Suppression yields to HIP/ROCR_VISIBLE_DEVICES; CUDA_VISIBLE_DEVICES must NOT count.
foreach ($pair in @(
    @{ F = "install.ps1";      P = $installPath },
    @{ F = "studio/setup.ps1"; P = (Join-Path $root "studio/setup.ps1") }
)) {
    $text = (Get-Content -Raw $pair.P) -replace "`r`n", "`n"
    $i = $text.IndexOf('$unsupMasked = @(')
    Check "$($pair.F) yields the peer suppression to an AMD visible-device mask" ($i -ge 0)
    if ($i -ge 0) {
        $decl = $text.Substring($i, [Math]::Min(240, $text.Length - $i))
        Check "$($pair.F) counts HIP and ROCR in that mask" (
            $decl.Contains('HIP_VISIBLE_DEVICES') -and $decl.Contains('ROCR_VISIBLE_DEVICES')
        )
        Check "$($pair.F) does not count CUDA_VISIBLE_DEVICES" (
            -not $decl.Contains('CUDA_VISIBLE_DEVICES')
        )
        # Both label sources keep adapter 0, so the masked branch goes silent once a peer exists.
        $window = $text.Substring($i, [Math]::Min(1200, $text.Length - $i))
        Check "$($pair.F) drops the verdict when the mask may name another adapter" (
            $window -match '(ROCmPeerLabels|wmiAmdNames)\.Count -gt (1|\$gpuNames\.Count)'
        )
    }
}

# Get-WmiObject is gone from PowerShell 7 and the scans catch silently; parse for calls.
foreach ($pair in @(
    @{ F = "install.ps1";      A = $installAst },
    @{ F = "studio/setup.ps1"; A = $setupAst }
)) {
    $wmiCalls = @($pair.A.FindAll({
        param($n)
        $n -is [System.Management.Automation.Language.CommandAst] -and
        $n.GetCommandName() -eq 'Get-WmiObject'
    }, $true))
    Check "$($pair.F) scans adapters with Get-CimInstance, not Get-WmiObject" ($wmiCalls.Count -eq 0)
}

Invoke-Expression (Get-AssignmentSource $installPath '$nameArchTable')
Invoke-Expression (Get-AssignmentSource $installPath '$unsupportedNameArchTable')

# SCRIPT scope on purpose: inside a function the block's assignment lands locally and
# every "claims nothing" case passes vacuously.
$guardCases = @(
    @{ N = "lone RX 580 is still named gfx803"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580"); E = 'gfx803' }
    # Mixed host: stay quiet and keep the arch-unknown arm.
    @{ N = "RX 580 beside an RX 7900 XTX claims nothing"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580", "AMD Radeon RX 7900 XTX"); E = $null }
    @{ N = "RX 580 beside an RX 9070 XT claims nothing"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580", "AMD Radeon RX 9070 XT"); E = $null }
    @{ N = "RX 580 beside an RX 5700 XT claims nothing (RDNA 1 routes, #11614)"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580", "AMD Radeon RX 5700 XT"); E = $null }
    @{ N = "lone RX 5700 XT is not called unsupported"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT"); E = $null }
    @{ N = "RX 580 beside an unmappable Radeon is still named"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580", "AMD Radeon Graphics"); E = 'gfx803' }
    # The amd-smi paths never enter the WMI scan, so the peer list is empty there.
    @{ N = "an empty peer list does not suppress the verdict"
       L = "AMD Radeon RX 580"; A = @(); E = 'gfx803' }
    @{ N = "a masked lone RX 580 is still named"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580"); M = "0"; E = 'gfx803' }
    @{ N = "a mask beside a second adapter claims nothing"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580", "AMD Radeon RX 7900 XTX"); M = "1"; E = $null }
    @{ N = "a mask beside a second uncovered adapter claims nothing"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580", "AMD Radeon RX 570"); M = "1"; E = $null }
)
$guardSource = $guardBlocks[0].Extent.Text
foreach ($case in $guardCases) {
    $ROCmGfxArch = $null
    $ROCmUnsupportedGfxArch = $null
    $ROCmGpuLabel = $case.L
    $wmiAmdNames = @($case.A)
    if ($case.M) { $env:HIP_VISIBLE_DEVICES = $case.M } else { Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue }
    Invoke-Expression $guardSource
    Remove-Item Env:HIP_VISIBLE_DEVICES -ErrorAction SilentlyContinue
    Check $case.N ($ROCmUnsupportedGfxArch -eq $case.E)
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
