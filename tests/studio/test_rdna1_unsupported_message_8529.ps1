#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Behavioural test for the RDNA 1 "detected but not covered by ROCm" wording in
# install.ps1 and studio/setup.ps1 (issue #8529). The reporter's card is an RX 5700 XT
# (Navi 10, gfx1010). AMD's Windows torch indexes are gfx103X/110X/1150/1151/120X only,
# so CPU torch is correct and stays; the bug was telling them to install the HIP SDK or
# set UNSLOTH_ROCM_GFX_ARCH, neither of which can succeed on gfx1010.
#
# The Python suite evaluates the table with Python's `re`; this runs it under the .NET
# engine that ships it, the only place -match semantics are real. Fixtures are raw WMI
# adapter names, which is what the table is matched against.
#
# Run: pwsh -NoProfile -File tests/studio/test_rdna1_unsupported_message_8529.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Returns the source text of a variable assignment. The caller Invoke-Expression's
# it at script scope; doing that inside a function would lose the value on return.
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

    # The same first-match-wins loop both installers run over the table.
    function Resolve-Unsupported($name) {
        foreach ($row in $unsupportedNameArchTable) {
            if ($name -match $row.P) { return $row.A }
        }
        return $null
    }

    # --- the reporter's card, and the rest of RDNA 1 ---------------------------
    Check "RX 5700 XT -> gfx1010"          ((Resolve-Unsupported "AMD Radeon RX 5700 XT") -eq 'gfx1010')
    Check "RX 5700 -> gfx1010"             ((Resolve-Unsupported "AMD Radeon RX 5700") -eq 'gfx1010')
    Check "RX 5600 XT -> gfx1010"          ((Resolve-Unsupported "AMD Radeon RX 5600 XT") -eq 'gfx1010')
    Check "Radeon Pro 5600 XT -> gfx1010"  ((Resolve-Unsupported "AMD Radeon Pro 5600 XT") -eq 'gfx1010')
    Check "Radeon Pro V520 -> gfx1011"     ((Resolve-Unsupported "AMD Radeon Pro V520") -eq 'gfx1011')
    Check "Radeon Pro 5600M -> gfx1011"    ((Resolve-Unsupported "AMD Radeon Pro 5600M") -eq 'gfx1011')
    Check "RX 5500 XT -> gfx1012"          ((Resolve-Unsupported "AMD Radeon RX 5500 XT") -eq 'gfx1012')
    # The professional boards LLVM's table omits, mapped from libdrm data/amdgpu.ids read
    # against pci.ids, which files 7312/7310 under Navi 10 and 7340/7341/7347/734f under
    # Navi 14.
    Check "Radeon Pro W5700 -> gfx1010"    ((Resolve-Unsupported "AMD Radeon Pro W5700") -eq 'gfx1010')
    Check "Radeon Pro W5700X -> gfx1010"   ((Resolve-Unsupported "AMD Radeon Pro W5700X") -eq 'gfx1010')
    Check "Radeon Pro W5500 -> gfx1012"    ((Resolve-Unsupported "AMD Radeon Pro W5500") -eq 'gfx1012')
    Check "Radeon Pro W5500M -> gfx1012"   ((Resolve-Unsupported "AMD Radeon Pro W5500M") -eq 'gfx1012')
    Check "Radeon Pro W5300M -> gfx1012"   ((Resolve-Unsupported "AMD Radeon Pro W5300M") -eq 'gfx1012')
    Check "RX 5300 -> gfx1012"             ((Resolve-Unsupported "AMD Radeon RX 5300") -eq 'gfx1012')
    Check "RX 5300M -> gfx1012"            ((Resolve-Unsupported "AMD Radeon RX 5300M") -eq 'gfx1012')
    # The Mac Pro MPX boards, pci.ids 7319 and 731b under Navi 10. The only Navi 10
    # retail parts whose name carries neither "RX 5700" nor a W prefix.
    Check "Radeon Pro 5700 XT -> gfx1010"  ((Resolve-Unsupported "AMD Radeon Pro 5700 XT") -eq 'gfx1010')
    Check "Radeon Pro 5700 -> gfx1010"     ((Resolve-Unsupported "AMD Radeon Pro 5700") -eq 'gfx1010')
    # The W-series that DOES have wheels: "W5700" must not be read out of "W7500".
    Check "PRO W7500 unclaimed"            ($null -eq (Resolve-Unsupported "AMD Radeon PRO W7500"))
    Check "PRO W6500 unclaimed"            ($null -eq (Resolve-Unsupported "AMD Radeon PRO W6500"))
    Check "PRO W6400 unclaimed"            ($null -eq (Resolve-Unsupported "AMD Radeon PRO W6400"))

    # --- Polaris, the second card in the cluster (#8458) ----------------------
    # #8458 is an RX 580: Polaris 20, gfx803, also with no ROCm PyTorch wheels.
    Check "RX 580 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 580") -eq 'gfx803')
    Check "RX 580 Series -> gfx803"        ((Resolve-Unsupported "AMD Radeon RX 580 Series") -eq 'gfx803')
    Check "RX 570 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 570") -eq 'gfx803')
    Check "RX 590 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 590") -eq 'gfx803')
    Check "RX 480 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 480") -eq 'gfx803')
    Check "RX 470 -> gfx803"               ((Resolve-Unsupported "AMD Radeon RX 470") -eq 'gfx803')
    # The Polaris 10 workstation boards, grouped on Ellesmere by pci.ids. Their names
    # carry no RX number, so the consumer rows could never have reached them.
    Check "Radeon Pro WX 7100 -> gfx803"   ((Resolve-Unsupported "AMD Radeon Pro WX 7100") -eq 'gfx803')
    Check "Radeon Pro WX 5100 -> gfx803"   ((Resolve-Unsupported "AMD Radeon Pro WX 5100") -eq 'gfx803')

    # Polaris 11/12 is a different die and is deliberately absent: this table is
    # only worth having while it never guesses an arch.
    Check "RX 560 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 560"))
    Check "RX 550 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 550"))
    Check "RX 460 unclaimed"               ($null -eq (Resolve-Unsupported "AMD Radeon RX 460"))

    # "RX 570" is a prefix of "RX 5700" and "RX 550" of "RX 5500". Matched ALONE: table
    # order already keeps RDNA 1 names away from Polaris, so a dropped (?!0) guard
    # changes nothing observable until someone reorders.
    $polarisPattern = @($unsupportedNameArchTable | Where-Object { $_.A -eq 'gfx803' })[0].P
    foreach ($rdna1 in @("AMD Radeon RX 5700 XT", "AMD Radeon RX 5700", "AMD Radeon RX 5500 XT")) {
        Check "Polaris pattern alone does not claim '$rdna1'" (-not ($rdna1 -match $polarisPattern))
    }

    # --- and nothing else -----------------------------------------------------
    # A hit here would print "ROCm does not cover this" at a card that has wheels.
    Check "RX 9070 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 9070 XT"))
    Check "RX 9060 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 9060 XT"))
    Check "RX 7900 XTX unclaimed"          ($null -eq (Resolve-Unsupported "AMD Radeon RX 7900 XTX"))
    Check "RX 6800 XT unclaimed"           ($null -eq (Resolve-Unsupported "AMD Radeon RX 6800 XT"))
    Check "8060S Graphics unclaimed"       ($null -eq (Resolve-Unsupported "AMD Radeon 8060S Graphics"))
    Check "RTX 4090 unclaimed"             ($null -eq (Resolve-Unsupported "NVIDIA GeForce RTX 4090"))

    # --- the supported table must still miss RDNA 1 ---------------------------
    # The behavioural half: CPU fallback is correct here and must not move.
    Invoke-Expression (Get-AssignmentSource $path '$nameArchTable')
    function Resolve-Supported($name) {
        foreach ($row in $nameArchTable) {
            if ($name -match $row.P) { return $row.A }
        }
        return $null
    }
    Check "RX 5700 XT gets no supported arch" ($null -eq (Resolve-Supported "AMD Radeon RX 5700 XT"))
    Check "RX 5500 XT gets no supported arch" ($null -eq (Resolve-Supported "AMD Radeon RX 5500 XT"))
    Check "RX 9070 XT still maps to gfx1201"  ((Resolve-Supported "AMD Radeon RX 9070 XT") -eq 'gfx1201')

    # No arch may appear in both tables: one routes to a wheel index, the other
    # exists precisely because nothing routes.
    $bothTables = @($unsupportedNameArchTable | ForEach-Object { $_.A }) |
        Where-Object { @($nameArchTable | ForEach-Object { $_.A }) -contains $_ }
    Check "the two tables share no arch"   ($bothTables.Count -eq 0)

    # --- the wording, in the source that prints it ----------------------------
    # CRLF normalisation is mandatory: both files ship CRLF, so any needle
    # spanning a line break never matches the raw text.
    $src = (Get-Content -Raw $path) -replace "`r`n", "`n"

    # Every ordering claim below is preceded by a "was found" guard, so a renamed
    # branch fails loudly instead of comparing two -1s and passing vacuously.
    $unsupArm = 'step "gpu" "AMD GPU detected ($'
    $unknownArm = 'step "gpu" "AMD GPU detected -- arch unknown"'
    Check "unsupported gpu arm is present"  ($src.Contains($unsupArm))
    Check "arch-unknown gpu arm is present" ($src.Contains($unknownArm))
    Check "unsupported arm precedes arch-unknown arm" `
        ($src.IndexOf($unsupArm) -ge 0 -and $src.IndexOf($unknownArm) -ge 0 -and
         $src.IndexOf($unsupArm) -lt $src.IndexOf($unknownArm))

    # Scoped to the card: a host can pair an uncovered GPU with a covered one, so the
    # sentence says what is true of the card just named and claims nothing beyond it.
    $disclaimer = "setting UNSLOTH_ROCM_GFX_ARCH will not change that for it."
    Check "unsupported arm says the override cannot help" ($src.Contains($disclaimer))

    $sdkAdvice = 'substep "Could not determine the GPU arch'
    Check "HIP SDK advice still exists for genuinely unknown cards" ($src.Contains($sdkAdvice))
    Check "HIP SDK advice comes after the unsupported arm" `
        ($src.IndexOf($sdkAdvice) -ge 0 -and $src.IndexOf($unsupArm) -ge 0 -and
         $src.IndexOf($unsupArm) -lt $src.IndexOf($sdkAdvice))

    # --- the Vulkan pointer (#8458) -------------------------------------------
    # Torch ends on these cards; llama.cpp does not. Asserted against PRINTING lines only:
    # every phrase below also appears in the comments explaining the branch, so a file-wide
    # search stays green after the message is gutted. Single quotes count too -- the setter
    # is a literal so PowerShell prints $env:... instead of expanding it.
    $emitted = @(($src -split "`n") | Where-Object {
        $_ -match 'substep\s+["'']' -and $_.TrimStart() -notmatch '^#'
    })
    Check "the emitted advice offers Vulkan" `
        (($emitted -join "`n").Contains("through Vulkan"))
    # PowerShell syntax, verified by PARSING rather than matching text: a bare
    # UNSLOTH_LLAMA_CPP_BACKEND=vulkan parses as a command name, so a user who pastes
    # it sets nothing and gets the same CPU bundle -- the #8458 failure mode again.
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

    # UNSLOTH_FORCE_VULKAN still works, but force_vulkan_requested() resolves
    # UNSLOTH_LLAMA_CPP_BACKEND first and only falls back to the legacy name, so new
    # text must not spread that spelling. Scoped to emitters: setup.ps1 legitimately
    # READS the legacy variable for back-compat and that is untouched here.
    $teachesLegacy = @($emitted | Where-Object { $_ -match 'UNSLOTH_FORCE_VULKAN' })
    Check "the legacy spelling is not taught" ($teachesLegacy.Count -eq 0)

    # WHEN to set it, per SITE: install.ps1 prints this advice twice, so a file-level
    # "install time" search is satisfied by whichever site still has it.
    $mentions = @(0..($emitted.Count - 1) | Where-Object {
        $emitted[$_].Contains($setter)
    })
    Check "at least one site names the Vulkan variable" ($mentions.Count -ge 1)
    foreach ($i in $mentions) {
        $window = ($emitted[$i..([Math]::Min($i + 3, $emitted.Count - 1))]) -join "`n"
        Check "the advice at emitted line $($i + 1) says it applies at install time" `
            ($window.Contains("install time"))
    }

    # --- which arm actually WINS, evaluated rather than read ------------------
    # Source offsets cannot see a branch made unreachable by an earlier condition.
    # Parse the real if/elseif chain, evaluate each clause in order against the #8529
    # host -- RX 5700 XT, no ROCm runtime, HIP SDK installed because the old advice
    # said to -- and assert the FIRST true clause is the unsupported arm.
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
        $HipSdkInstalled = $true            # they installed it because we told them to
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

    # The scope guard, in source: the unsupported lookup must never assign the
    # arch the installers route on.
    $tableSrc = Get-AssignmentSource $path '$unsupportedNameArchTable'
    Check "the table assigns no routable arch" (-not ($tableSrc -match 'gfx1(0[3-9]|1|2)'))

    # The table has to be READ, not just declared: everything above evaluates it in
    # isolation, so a lookup never wired to the message arms would sail through.
    $consumer = if ($file -eq "install.ps1") {
        'foreach ($row in $unsupportedNameArchTable) {'
    } else {
        '-Table $unsupportedNameArchTable'
    }
    Check "the table is consumed by the arch resolver" ($src.Contains($consumer))
    Check "the resolver feeds the variable the arms read" ($src -match 'ROCmUnsupportedGfxArch\s*=\s*(\$row\.A|Get-GfxArchFromGpuName)')
}

# studio/setup.ps1 reads $script:ROCmUnsupportedGfxArch in the unconditional ROCm summary,
# so it must be initialised OUTSIDE the `-not $HasNvidiaSmi` block. Left inside, an NVIDIA
# host never defines it and a caller's Set-StrictMode turns the summary into an aborting
# undefined-variable error (unslothai#8529).
$setupSrc = (Get-Content -Raw (Join-Path $root "studio/setup.ps1")) -replace "`r`n", "`n"
# Ask the parser, not the text: the file has three `-not $HasNvidiaSmi` blocks, so a
# line-order check picks the wrong one and brace counting trips over braces in strings.
# The invariant is that one assignment is NOT nested in any conditional.
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

# install.ps1's WMI fallback takes adapter [0], so on a host pairing an RX 5700 with an
# RX 7900 the label is the 5700 and "nothing can enable ROCm here" would be false: masking
# to the 7900 and setting gfx1100 installs. Such a host must keep the arch-unknown arm.
# Run the REAL block, pulled out by AST rather than retyped.
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
# Requiring $wmiAmdNames is what makes this load-bearing: delete the peer scan and no
# block matches, so this check fails rather than the cases below quietly going vacuous.
Check "the unsupported lookup block consults the peer list" ($guardBlocks.Count -gt 0)

# The cases below inject $wmiAmdNames, so they cannot see whether anything FILLS it.
# Assert the producer separately, and assert what it must NOT touch: the peer list is for
# the REPORT, so it lives in its own block and must never write the label or the arch the
# inference reads. Widening the existing scan instead turned a CPU install into a ROCm one
# on a host amd-smi had already claimed, which is a routing change, not a message change.
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

# studio/setup.ps1 keeps its own report-only list, for the same reason: $script:ROCmGpuLabels
# feeds $gpuNames and therefore the arch inference, so the peer names must not go there.
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

# A mask names the card, so a masked-out peer cannot answer for it: the suppression has to
# yield to HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES, as the arch-borrowing rule already
# does. CUDA_VISIBLE_DEVICES must NOT count; it masks NVIDIA devices, and counting it fired
# the verdict beside a covered Radeon on every host that sets it.
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
        # Yielding to the mask is only safe while the label is the pinned card: both label
        # sources keep adapter 0, so the masked branch has to fall silent once a peer it
        # never indexed exists.
        $window = $text.Substring($i, [Math]::Min(1200, $text.Length - $i))
        Check "$($pair.F) drops the verdict when the mask may name another adapter" (
            $window -match '(ROCmPeerLabels|wmiAmdNames)\.Count -gt (1|\$gpuNames\.Count)'
        )
    }
}

# Get-WmiObject is gone from PowerShell 7, and the scans below catch silently, so a supported
# Radeon named only by Windows took the CPU torch path there. Ask the parser for real calls;
# the name still appears in prose. Every adapter scan in these files is on CIM.
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

# Driven at SCRIPT scope on purpose. Invoke-Expression inside a function would let the
# block's own `$ROCmUnsupportedGfxArch = $row.A` land in that function's scope, so every
# "claims nothing" case would pass without the guard existing at all.
$guardCases = @(
    # The reporter's host: one RDNA 1 card, nothing else. The verdict must still be reached.
    @{ N = "lone RX 5700 XT is still named gfx1010"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT"); E = 'gfx1010' }
    # The mixed host: adapter 0 is the 5700, adapter 1 has wheels. Stay quiet and keep the
    # arch-unknown arm, which points at the override that really does work there.
    @{ N = "RX 5700 beside an RX 7900 XTX claims nothing"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT", "AMD Radeon RX 7900 XTX"); E = $null }
    @{ N = "RX 580 beside an RX 9070 XT claims nothing"
       L = "AMD Radeon RX 580"; A = @("AMD Radeon RX 580", "AMD Radeon RX 9070 XT"); E = $null }
    # Two uncovered cards are still an uncovered host.
    @{ N = "RX 5700 XT beside an RX 580 is still named"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT", "AMD Radeon RX 580"); E = 'gfx1010' }
    # A peer we cannot map is not a covered peer; it is the unknown the arm already handles.
    @{ N = "RX 5700 XT beside an unmappable Radeon is still named"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT", "AMD Radeon Graphics"); E = 'gfx1010' }
    # The amd-smi paths never enter the WMI scan, so the peer list is empty there.
    @{ N = "an empty peer list does not suppress the verdict"
       L = "AMD Radeon RX 5700 XT"; A = @(); E = 'gfx1010' }
    # Under a mask the peers no longer speak for the named card, but both label sources take
    # adapter 0, so the label is only the SELECTED card when there is nothing else to pick.
    @{ N = "a masked lone RX 5700 XT is still named"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT"); M = "0"; E = 'gfx1010' }
    @{ N = "a mask beside a second adapter claims nothing"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT", "AMD Radeon RX 7900 XTX"); M = "1"; E = $null }
    # Even two uncovered peers: the arch named would still be adapter 0's, not the pinned one.
    @{ N = "a mask beside a second uncovered adapter claims nothing"
       L = "AMD Radeon RX 5700 XT"; A = @("AMD Radeon RX 5700 XT", "AMD Radeon RX 580"); M = "1"; E = $null }
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
