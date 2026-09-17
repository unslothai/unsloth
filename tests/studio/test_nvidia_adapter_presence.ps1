#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The last-resort NVIDIA presence source, what it must never do, and the wheel it leads to.
#
# $HasNvidiaSmi does not mean "nvidia-smi exists". It means "an NVIDIA GPU is present", and
# setting it true SUPPRESSES all AMD and Intel detection downstream. So a presence source has
# three jobs, and only the first is the obvious one:
#
#   1. Say yes on a host with a working NVIDIA card that neither nvidia-smi nor the driver library
#      could report, instead of sending it to CPU-only wheels in silence (#9255).
#   2. Say no on everything else. A false yes on an AMD or Intel machine turns off that machine's
#      GPU support entirely, which is a worse failure than the one this fixes.
#   3. Lead somewhere useful. Promoting the flag without also giving the wheel selector a version
#      to work from was MEASURED to produce $HasNvidiaSmi true, AMD detection off, and a CPU index
#      anyway: strictly worse than doing nothing. That is why this file also drives the index.
#
# Marketing names in the fixtures are deliberately adversarial, because matching on a name rather
# than the PCI vendor ID is the obvious wrong implementation and it has to fail loudly.
# Run: pwsh -NoProfile -File tests/studio/test_nvidia_adapter_presence.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPs1 = Join-Path $root "install.ps1"
$setupPs1 = Join-Path $root "studio/setup.ps1"

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Get-FunctionText($file, $name) {
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$null, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$file has parse errors" }
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
    }, $true)
    if ($fn.Count -lt 1) { throw "expected $name in $file, found none" }
    return $fn[0].Extent.Text
}

# ------------------------------------------------------------------- the shared region
#
# install.ps1 nests one level deeper, so compare without indentation, the same way
# test_nvidia_library_inventory.ps1 and test_nvidia_smi_discovery.ps1 do.
$strip = { param($t) (($t -split "`n") | ForEach-Object { $_.TrimStart() }) -join "`n" }
$shared = @(
    "Invoke-BoundedVideoControllerScan", "Test-NvidiaAdapterPresent",
    "Test-OtherVendorAdapterPresent", "Get-XpuCapableNameRegex",
    "Get-NvidiaDriverRelease", "Get-NvidiaCudaFloorForRelease", "Get-NvidiaDriverCudaFloor",
    "Get-NvidiaAdapterDriverRelease", "Get-NvidiaAdapterCudaFloor",
    "Get-NvidiaRegistryAdapter"
)
foreach ($name in $shared) {
    Check "install.ps1 and setup.ps1 carry the same $name" (
        (& $strip (Get-FunctionText $installPs1 $name)) -eq (& $strip (Get-FunctionText $setupPs1 $name)))
}

foreach ($name in @("Test-NvidiaAdapterPresent", "Test-OtherVendorAdapterPresent", "Get-XpuCapableNameRegex",
                    "Get-NvidiaDriverRelease", "Get-NvidiaCudaFloorForRelease", "Get-NvidiaDriverCudaFloor",
                    "Get-NvidiaAdapterDriverRelease", "Get-NvidiaAdapterCudaFloor",
                    "Get-NvidiaRegistryAdapter")) {
    Invoke-Expression (Get-FunctionText $setupPs1 $name)
}

# The scan is stubbed, not run: this host has no Win32_VideoController, and what is under test is
# the judgement applied to an inventory rather than the ability to fetch one.
$script:FakeAdapters = @()
# $null means "derive it from the fixture", which is what every row below job 2b wants. A row
# that needs a scan which ANSWERED yet listed no healthy NVIDIA adapter sets it explicitly.
$script:FakeScanOk = $null
function Invoke-BoundedVideoControllerScan {
    param([int]$TimeoutSec = 15)
    return [pscustomobject]@{
        Ok = $(if ($null -eq $script:FakeScanOk) { $script:FakeAdapters.Count -gt 0 } else { $script:FakeScanOk })
        Names = @($script:FakeAdapters | ForEach-Object { $_.Name })
        Adapters = $script:FakeAdapters
    }
}
function Adapter($name, $id, $code, $driver = "32.0.15.6094") {
    return [pscustomobject]@{
        Name = $name; PNPDeviceID = $id; ConfigManagerErrorCode = $code; DriverVersion = $driver
    }
}

# ------------------------------------------------------------------- job 1 and job 2
#
# The registry fallback finds nothing on this host, so a $true below can only have come from the
# inventory. Checked rather than assumed, in the empty case first.
$script:FakeAdapters = @()
Check "control: nothing on the bus is not a GPU, and the registry adds nothing here" (
    (Test-NvidiaAdapterPresent) -eq $false)

$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684&SUBSYS_00000000" 0)
Check "a healthy NVIDIA adapter is present" ((Test-NvidiaAdapterPresent) -eq $true)

foreach ($case in @(
    @{ N = "an AMD-only machine";   A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 0) },
    @{ N = "an Intel-only machine"; A = @(Adapter "Intel Arc A770" "PCI\VEN_8086&DEV_56A0" 0) },
    @{ N = "a hybrid AMD and Intel machine"; A = @(
            (Adapter "Intel UHD Graphics" "PCI\VEN_8086&DEV_9A60" 0),
            (Adapter "AMD Radeon 780M" "PCI\VEN_1002&DEV_15BF" 0)) },
    @{ N = "a virtual adapter";     A = @(Adapter "Microsoft Basic Display Adapter" "ROOT\BASICDISPLAY" 0) }
)) {
    $script:FakeAdapters = $case.A
    Check "no NVIDIA reported for $($case.N)" ((Test-NvidiaAdapterPresent) -eq $false)
}

# A marketing name is not evidence. These two are the trap: an adapter whose name says NVIDIA but
# whose vendor ID is AMD's must be no, and an NVIDIA adapter with an OEM-rebranded or localized
# name that never says NVIDIA must still be yes. Matching on the name gets both backwards.
$script:FakeAdapters = @(Adapter "NVIDIA compatible display" "PCI\VEN_1002&DEV_744C" 0)
Check "a non-NVIDIA card whose NAME says NVIDIA is not counted" ((Test-NvidiaAdapterPresent) -eq $false)

$script:FakeAdapters = @(Adapter "Carte graphique" "PCI\VEN_10DE&DEV_2684" 0)
Check "an NVIDIA card whose name says nothing recognisable is still counted" (
    (Test-NvidiaAdapterPresent) -eq $true)

# A record is not a card. A driver configuration outlives removed hardware, and a disabled or
# faulted adapter is not usable. Counting any of these would suppress AMD and Intel on a machine
# whose only real GPU is one of those.
foreach ($case in @(
    @{ N = "a disabled adapter (code 22)";          C = 22 },
    @{ N = "a device-problem adapter (code 43)";    C = 43 },
    @{ N = "a driver record with no status at all"; C = $null }
)) {
    $script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" $case.C)
    Check "no NVIDIA reported for $($case.N)" ((Test-NvidiaAdapterPresent) -eq $false)
}

$script:FakeAdapters = @(
    (Adapter "NVIDIA GeForce GTX 1060" "PCI\VEN_10DE&DEV_1C03" 43),
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0))
Check "a healthy card after a faulted one is still found" ((Test-NvidiaAdapterPresent) -eq $true)

$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "pci\ven_10de&dev_2684" 0)
Check "the vendor ID match is case-insensitive" ((Test-NvidiaAdapterPresent) -eq $true)

$script:FakeAdapters = @(Adapter "Some Card" "PCI\VEN_110D&DEV_E10D" 0)
Check "a lookalike vendor ID is not matched" ((Test-NvidiaAdapterPresent) -eq $false)

# ------------------------------------------------- job 2, the gate that keeps this additive
#
# A hybrid host already gets AMD or Intel wheels today, so it is NOT the host this fixes.
# Promoting NVIDIA there would suppress a working path for a GPU whose CUDA version nothing on the
# machine can report.
foreach ($case in @(
    @{ N = "a healthy AMD adapter";   A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 0); Want = $true },
    @{ N = "a healthy Intel adapter"; A = @(Adapter "Intel Arc A770" "PCI\VEN_8086&DEV_56A0" 0); Want = $true },
    @{ N = "a faulted AMD adapter";   A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 43); Want = $false },
    @{ N = "an NVIDIA adapter only";  A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0); Want = $false },
    @{ N = "a virtual adapter only";  A = @(Adapter "Microsoft Basic Display Adapter" "ROOT\BASICDISPLAY" 0); Want = $false }
)) {
    $script:FakeAdapters = $case.A
    Check "other-vendor gate sees $($case.N) as $($case.Want)" (
        (Test-OtherVendorAdapterPresent) -eq $case.Want)
}

# And the two together on the machine that matters: NVIDIA plus AMD, both healthy. Presence says
# yes, the gate says another vendor is here, so the promotion must NOT fire.
$script:FakeAdapters = @(
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
    (Adapter "AMD Radeon 780M" "PCI\VEN_1002&DEV_15BF" 0))
Check "on a hybrid NVIDIA plus AMD host presence is true but the gate blocks promotion" (
    (Test-NvidiaAdapterPresent) -eq $true -and (Test-OtherVendorAdapterPresent) -eq $true)

# The commonest machine in this entire population: a laptop with an NVIDIA GPU and Intel
# integrated graphics. UHD and Iris get no XPU wheels, so counting them as an alternative blocks
# the NVIDIA promotion and then hands the host CPU wheels anyway. Worse than either outcome on
# its own, and it is the configuration the promotion was written for.
foreach ($case in @(
    @{ N = "Intel UHD Graphics";    Name = "Intel(R) UHD Graphics 770";        Blocks = $false },
    @{ N = "Intel Iris Xe";         Name = "Intel(R) Iris(R) Xe Graphics";     Blocks = $false },
    @{ N = "Intel HD Graphics";     Name = "Intel(R) HD Graphics 630";         Blocks = $false },
    # Arc and Data Center parts DO block: for those the Intel route has somewhere to go, and
    # suppressing it would cost the host wheels it can actually use.
    @{ N = "Intel Arc A770";        Name = "Intel(R) Arc(TM) A770 Graphics";   Blocks = $true },
    @{ N = "Intel Arc B580";        Name = "Intel(R) Arc(TM) B580 Graphics";   Blocks = $true },
    @{ N = "Intel Data Center GPU"; Name = "Intel(R) Data Center GPU Max 1100"; Blocks = $true },
    # A nameless Intel adapter cannot be shown to be XPU-capable, so it does not block.
    @{ N = "a nameless Intel adapter"; Name = "";                              Blocks = $false }
)) {
    $script:FakeAdapters = @(
        (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
        (Adapter $case.Name "PCI\VEN_8086&DEV_4680" 0))
    Check "NVIDIA plus $($case.N): the gate $(if ($case.Blocks) { 'blocks' } else { 'allows' }) promotion" (
        (Test-OtherVendorAdapterPresent) -eq $case.Blocks)
    Check "  and NVIDIA presence is unaffected either way" ((Test-NvidiaAdapterPresent) -eq $true)
}

# AMD is deliberately NOT narrowed: its route serves most Radeon parts through a name-to-gfx
# table rather than a short allowlist, so deciding capability here would duplicate that table.
$script:FakeAdapters = @(
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
    (Adapter "AMD Radeon(TM) Graphics" "PCI\VEN_1002&DEV_15BF" 0))
Check "an integrated AMD adapter still blocks, unlike the Intel one" (
    (Test-OtherVendorAdapterPresent) -eq $true)

# The gate's Intel pattern must be the SAME one the XPU route uses, or the two answer different
# questions and drift apart silently. Read both out of install.ps1 rather than restating either.
$gateText = Get-FunctionText $installPs1 "Test-OtherVendorAdapterPresent"
$installAll = [System.IO.File]::ReadAllText($installPs1)
Check "the gate defers to the shared XPU-capable pattern" ($gateText -match 'Get-XpuCapableNameRegex')
# @() around the WHOLE pipeline, not just the Matches. Sort-Object -Unique on one element
# returns a bare string, .Count on a string is 1, and [0] is then its first CHARACTER, so both
# checks below compare "(" with "(" and pass for free. Hit exactly that here before fixing it.
$patternDefs = @(@([regex]::Matches($installAll, 'function Get-XpuCapableNameRegex \{ return "([^"]+)" \}') |
    ForEach-Object { $_.Groups[1].Value }) | Sort-Object -Unique)
$routeDefs = @(@([regex]::Matches($installAll, '\$_xpuNameRe = "([^"]+)"') |
    ForEach-Object { $_.Groups[1].Value }) | Sort-Object -Unique)
Check "the extracted patterns are whole strings, not characters" (
    $patternDefs[0].Length -gt 5 -and $routeDefs[0].Length -gt 5)
Check "both patterns are defined exactly once" ($patternDefs.Count -eq 1 -and $routeDefs.Count -eq 1)
Check "the gate pattern and the XPU route pattern are identical" ($patternDefs[0] -ceq $routeDefs[0])

# -------------------------------------------------------- the driver version to CUDA floor
#
# Windows reports the DISPLAY driver version, not the NVIDIA release: "32.0.15.6094" is NVIDIA
# 560.94. Values are a floor, never the exact version.
foreach ($case in @(
    @{ V = "32.0.15.6094"; Want = "12.6" },   # 560.94
    @{ V = "32.0.15.7020"; Want = "12.8" },   # 570.20
    @{ V = "32.0.15.8000"; Want = "13.0" },   # 580.00
    @{ V = "31.0.15.5222"; Want = "12.4" },   # 552.22 -> 550 floor
    @{ V = "31.0.15.3667"; Want = "12.2" },   # 536.67 -> 535 floor
    @{ V = "30.0.14.4568"; Want = "11.0" },   # 445.68 is BELOW 450: no floor
    @{ V = "570.20";       Want = "12.8" },   # already NVIDIA-shaped, as nvidia-smi reports
    @{ V = "";             Want = "" },
    @{ V = "not a version"; Want = "" }
)) {
    $got = Get-NvidiaDriverCudaFloor -DriverVersion $case.V
    $text = if ($got) { "$($got[0]).$($got[1])" } else { "" }
    $want = if ($case.V -eq "30.0.14.4568") { "" } else { $case.Want }
    Check "driver $($case.V) floors at '$want'" ($text -eq $want)
}

# Read out of the adapter inventory, and only from a HEALTHY NVIDIA one.
$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0 "32.0.15.7020")
# The release parse on its own. This is what lets the caller tell "no version at all" apart from
# "a version, and it is below every row of the table": the floor helper answers $null for both,
# and they want opposite wheels.
foreach ($case in @(
    @{ V = "32.0.15.6094"; R = 560; N = "a Windows display-driver version" },
    @{ V = "30.0.14.4568"; R = 445; N = "a pre-R450 Windows version" },
    @{ V = "580.65.06";    R = 580; N = "an nvidia-smi style version" },
    @{ V = "445.68";       R = 445; N = "a pre-R450 nvidia-smi style version" },
    @{ V = "";             R = $null; N = "an empty version" },
    @{ V = "not a version"; R = $null; N = "an unparseable version" },
    @{ V = "1.2.3";        R = $null; N = "a three-field version" },
    # Three digits at least. The caller now acts on a low release by choosing CPU wheels, so a
    # malformed string must read as unknown rather than as an ancient driver.
    @{ V = "99.1";         R = $null; N = "a two-digit release" },
    @{ V = "100.1";        R = 100;  N = "the lowest release shape NVIDIA actually ships" }
)) {
    $got = Get-NvidiaDriverRelease -DriverVersion $case.V
    Check "$($case.N) reads as release $($case.R)" ($got -eq $case.R)
}
# And the two disagree exactly where they should: a readable pre-R450 version has a release but
# no floor, which is the whole distinction the wheel branch now depends on.
Check "a pre-R450 driver has a release but no floor" (
    (Get-NvidiaDriverRelease -DriverVersion "30.0.14.4568") -eq 445 -and
    $null -eq (Get-NvidiaDriverCudaFloor -DriverVersion "30.0.14.4568"))
Check "an unreadable version has neither" (
    $null -eq (Get-NvidiaDriverRelease -DriverVersion "junk") -and
    $null -eq (Get-NvidiaDriverCudaFloor -DriverVersion "junk"))

$floor = Get-NvidiaAdapterCudaFloor
Check "the floor is read from the healthy NVIDIA adapter" (
    $null -ne $floor -and $floor[0] -eq 12 -and $floor[1] -eq 8)

$script:FakeAdapters = @(Adapter "AMD Radeon" "PCI\VEN_1002&DEV_744C" 0 "32.0.15.7020")
Check "an AMD adapter's driver version is not read as an NVIDIA floor" (
    $null -eq (Get-NvidiaAdapterCudaFloor))

$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 43 "32.0.15.7020")
Check "a faulted NVIDIA adapter's driver version is not used" (
    $null -eq (Get-NvidiaAdapterCudaFloor))

# --------------------------------------------------------------- the table cannot drift
#
# The floors are ported from studio/nvidia_probe.py. Generated from that file rather than retyped,
# so the two cannot disagree without this failing.
$probePy = Join-Path $root "studio/nvidia_probe.py"
$probeText = Get-Content -Raw -LiteralPath $probePy
$pyRows = @()
$tableMatch = [regex]::Match($probeText, '(?s)_DRIVER_MAJOR_CUDA\s*=\s*\((.*?)\)\s*\n\s*\n')
Check "the python floor table was found" ($tableMatch.Success)
foreach ($m in [regex]::Matches($tableMatch.Groups[1].Value, '\((\d+),\s*\((\d+),\s*(\d+)\)\)')) {
    $pyRows += "$($m.Groups[1].Value):$($m.Groups[2].Value).$($m.Groups[3].Value)"
}
Check "the python table has rows to compare (bites)" ($pyRows.Count -ge 5)
# Read from the function that HOLDS the table. Get-NvidiaDriverCudaFloor now only parses a
# version and hands the release on, so reading it would compare an empty list and pass.
$psText = Get-FunctionText $setupPs1 "Get-NvidiaCudaFloorForRelease"
$psRows = @()
foreach ($m in [regex]::Matches($psText, '@\((\d+),\s*(\d+),\s*(\d+)\)')) {
    $psRows += "$($m.Groups[1].Value):$($m.Groups[2].Value).$($m.Groups[3].Value)"
}
Check "the PowerShell floor table matches nvidia_probe.py exactly" (
    ($pyRows -join ",") -eq ($psRows -join ","))

# ------------------------------------------------------------------ where it is consumed
#
# The promotion must sit BELOW the driver library, not above it. The library knows the CUDA
# version and the compute capabilities; this knows only that a card exists.
foreach ($file in @($installPs1, $setupPs1)) {
    $text = Get-Content -Raw -LiteralPath $file
    $name = Split-Path $file -Leaf
    $library = $text.IndexOf('if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory))')
    $presence = $text.IndexOf('Test-NvidiaAdapterPresent -Scan $presenceScan')
    Check "$name promotes on the driver library before falling back to the bus" (
        $library -gt 0 -and $presence -gt 0 -and $library -lt $presence)
    Check "$name only reaches the bus when nothing else answered" (
        $text.IndexOf('if (-not $HasNvidiaSmi) {', $library) -gt 0 -and
        $text.IndexOf('if (-not $HasNvidiaSmi) {', $library) -lt $presence)
    Check "$name gates the promotion on no other vendor being present" (
        $text -match '-not \(Test-OtherVendorAdapterPresent -Scan \$presenceScan\)')

    # PowerShell does not hoist. Both files are one long body executed top to bottom, so a
    # function called above its own definition is not defined yet and the call fails with "not
    # recognized". This was real, twice: the presence helper first landed BELOW the promotion that
    # calls it, and then the bounded scan did. Every pwsh suite still passed both times, because
    # they extract functions and define them in dependency order themselves. Only reading the file
    # in order catches it.
    foreach ($fn in $shared) {
        $def = $text.IndexOf("function $fn")
        Check "$name defines $fn before the promotion that uses it" ($def -gt 0 -and $def -lt $presence)
    }
    foreach ($m in [regex]::Matches($text, '(?<!function )Invoke-BoundedVideoControllerScan')) {
        $def = $text.IndexOf("function Invoke-BoundedVideoControllerScan")
        Check "$name calls Invoke-BoundedVideoControllerScan at offset $($m.Index) after its definition" (
            $m.Index -gt $def)
    }
}

# setup.ps1 must NOT guess a family. Get-PytorchCudaTag returning "" means unknown, and the
# do-not-wipe escape keys on that: guessing cu126 there replaced a working cu130 venv on every
# update (#9255). Only install.ps1, which is choosing an index for a fresh install, may floor.
$setupTag = Get-FunctionText $setupPs1 "Get-PytorchCudaTag"
Check "setup.ps1 still returns empty for unknown rather than flooring" (
    $setupTag -notmatch 'NvidiaPresenceCudaFloor')
Check "and install.ps1 is where the floor is consumed" (
    (Get-FunctionText $installPs1 "Get-TorchIndexUrl") -match 'NvidiaPresenceCudaFloor')

# The routing rows above set $script:NvidiaPresenceDriverRelease directly, so nothing in them
# exercises the promotion site that is supposed to RECORD it. Without this, deleting that
# recording leaves every row green while every real host reports no release at all, and the
# pre-R450 branch becomes unreachable. Found exactly that way: the mutation passed.
foreach ($file in @($installPs1, $setupPs1)) {
    $text = [System.IO.File]::ReadAllText($file)
    $leaf = Split-Path -Leaf $file
    Check "$leaf records the driver release at the promotion site" (
        $text -match '\$script:NvidiaPresenceDriverRelease = Get-NvidiaAdapterDriverRelease')
    Check "$leaf records it beside the floor, from the same scan" (
        $text -match 'Get-NvidiaAdapterCudaFloor -Scan \$presenceScan[\s\S]{0,400}?Get-NvidiaAdapterDriverRelease -Scan \$presenceScan')
    Check "$leaf initialises the release to null" (
        $text -match '\$script:NvidiaPresenceDriverRelease = \$null')
}

# setup.ps1 alone carries the rejection latch, and the bus promotion has to set it for the same
# reason the driver-library promotion above it does: reaching that block means discovery already
# walked every candidate path and every one was absent, hung, or reported no GPU. Left false,
# Get-CudaComputeCapability and Get-PytorchCudaTag rediscover that same executable, which costs up
# to two more bounded waits and can select a family from a banner detection already rejected.
$setupWhole = [System.IO.File]::ReadAllText($setupPs1)
$busBlock = [regex]::Match($setupWhole, '(?s)if \(-not \$HasNvidiaSmi\) \{\s*\$presenceScan = Invoke-BoundedVideoControllerScan.*?\n\}')
Check "setup.ps1's bus promotion block was found (bites)" ($busBlock.Success)
if ($busBlock.Success) {
    Check "the bus promotion marks nvidia-smi rejected" (
        $busBlock.Value -match '\$script:NvidiaSmiRejected = \$true')
    Check "and still records presence-only separately" (
        $busBlock.Value -match '\$script:NvidiaPresenceOnly = \$true')
}
Check "the driver-library promotion already did, which is the precedent" (
    $setupWhole -match '(?s)Get-NvidiaLibraryInventory\)\) \{[\s\S]{0,600}?\$script:NvidiaSmiRejected = \$true')
Check "install.ps1 has no rejection latch to set (bites)" (
    ([System.IO.File]::ReadAllText($installPs1)) -notmatch 'NvidiaSmiRejected')
Check "install.ps1 is where the release is consumed" (
    (Get-FunctionText $installPs1 "Get-TorchIndexUrl") -match 'NvidiaPresenceDriverRelease')
Check "setup.ps1's tag helper does not act on it, for the same reason it does not floor" (
    (Get-FunctionText $setupPs1 "Get-PytorchCudaTag") -notmatch 'NvidiaPresenceDriverRelease')

# But setup's ROUTER must, and for a while it did not. Get-PytorchCudaTag returns "" for a driver
# it cannot place, and the arm below that answered cu126 for it. Once setup gained the same bus
# promotion install.ps1 has, a pre-R450 host running setup.ps1 directly against a new or
# incomplete venv reached that arm and was handed CUDA wheels its driver cannot load, while
# install.ps1 answered CPU for the same machine. The two routers have to agree.
#
# Read as text because this is top-level script, not a function. Anchored on the ORDER, since an
# arm placed after the cu126 fallback is dead code that reads exactly like a fix.
$setupText = [System.IO.File]::ReadAllText($setupPs1)
$nvArm = [regex]::Match($setupText, '(?s)\} elseif \(\$HasNvidiaSmi\) \{.*?\n\} elseif \(\$script:IsIntelXpu\)')
Check "setup.ps1's NVIDIA routing arm was found (bites)" ($nvArm.Success)
if ($nvArm.Success) {
    $arm = $nvArm.Value
    $preR450 = $arm.IndexOf('NvidiaPresenceDriverRelease')
    $cu126 = $arm.IndexOf('"cu126"')
    Check "setup.ps1 routes a pre-R450 presence-only host to CPU" (
        $arm -match '\$script:NvidiaPresenceDriverRelease -lt 450' -and
        $arm -match '(?s)-lt 450\)? \{[^}]*\$CuTag = "cpu"')
    Check "and it decides before the cu126 fallback rather than after it" (
        $preR450 -ge 0 -and $cu126 -ge 0 -and $preR450 -lt $cu126)
    Check "and says why, rather than silently installing CPU wheels" (
        $arm -match 'predates R450')
    # Saying "installing CPU wheels" is not installing them. The CPU arm installs the bare torch
    # range when nothing pinned it, and an installed +cu build satisfies that range, so uv keeps
    # the very wheel this arm just said the driver cannot load. The stale-venv pass cannot rescue
    # it either: the expected family reads as unknown here, so $script:PinChangedForceReinstall
    # stays false. Same flag mechanism the ROCm and XPU fallbacks already use.
    Check "and marks the demotion so the CPU arm really replaces the CUDA wheel" (
        $arm -match '(?s)-lt 450\)? \{[^}]*NvidiaPreR450CpuFallback = \$true')
    Check "and only when a CUDA wheel is actually installed" (
        $arm -match 'Test-CudaFamilyLeaf \$installedTorchTag\) \{ \$script:NvidiaPreR450CpuFallback = \$true \}')
}
# The flag has to be READ where the wheels are installed, or setting it changes nothing. Anchored
# on the CPU arm's own force list, beside the two fallbacks that set theirs for the same reason.
$cpuArm = [regex]::Match($setupText, '(?s)substep "installing PyTorch \(CPU-only\)\.\.\.".{0,3000}')
Check "setup.ps1's CPU install arm was found (bites)" ($cpuArm.Success)
if ($cpuArm.Success) {
    Check "the CPU arm force-reinstalls after a pre-R450 demotion" (
        $cpuArm.Value -match 'if \(\$script:NvidiaPreR450CpuFallback\) \{ \$cpuForce = @\("--force-reinstall"\) \}')
    Check "the same way it does after the ROCm and XPU fallbacks (bites)" (
        $cpuArm.Value -match 'if \(\$ROCmCpuFallback\) \{ \$cpuForce = @\("--force-reinstall"\) \}')
}
# Reset per run, like the rest of the presence state: under `irm | iex` the script scope is the
# caller's own session, and a flag left set from a previous run would force a reinstall that this
# run never asked for.
Check "the flag starts false on every run" (
    $setupText -match '(?m)^\$script:NvidiaPreR450CpuFallback = \$false$')
# The bus-only host whose driver IS in the table. install.ps1 selects from the floor and setup did
# not, so the two routers disagreed by two whole families: an R450 to R524 driver maps to CUDA 11.0
# and install.ps1 picks cu118, while setup fell through to cu126, which that driver cannot load.
if ($nvArm.Success) {
    $arm = $nvArm.Value
    Check "setup.ps1 selects from the recorded floor" (
        $arm -match '\$script:NvidiaPresenceCudaFloor')
    # Only when nothing cu* is installed. An existing cu130 venv must not be pulled back to a
    # FLOOR: that is the do-not-wipe escape (#9255), and it is why Get-PytorchCudaTag still
    # returns empty rather than guessing.
    Check "and only when no CUDA family is already installed" (
        $arm -match '-not \(Test-CudaFamilyLeaf \$installedTorchTag\)[\s\S]{0,80}?NvidiaPresenceCudaFloor')
    # Through the shared ladder, never by formatting the digits: CUDA 11.0 is served by cu118, and
    # a leaf built from the digits would be "cu110", an index nothing publishes.
    Check "through the shared ladder rather than by formatting the digits" (
        $arm -match 'Get-CudaFamilyForVersion -Major \$floorMajor' -and
        $arm -notmatch 'cu\$floorMajor')
    # And the same pre-Turing cap install.ps1 applies, since nothing here named a capability.
    Check "and applies the same pre-Turing cap" (
        $arm -match '\$floorMajor -gt 12 -or \(\$floorMajor -eq 12 -and \$floorMinor -gt 6\)')
    Check "and the pre-R450 arm still decides first" (
        $arm.IndexOf('NvidiaPresenceDriverRelease') -lt $arm.IndexOf('NvidiaPresenceCudaFloor'))
}

# One ladder, two callers, and it must be the SAME ladder install.ps1 uses or the two installers
# hand one host different wheels. Compared as normalised text, then driven for real.
$ladderFn = Get-FunctionText $setupPs1 "Get-CudaFamilyForVersion"
Check "setup.ps1 names the ladder once (bites)" ($ladderFn.Length -gt 50)
$ladderRows = [regex]::Matches($ladderFn, '(?m)^\s*(?:if|elseif|return)[^\r\n]*cu\d+[^\r\n]*$')
$installLadder = [regex]::Matches(
    (Get-FunctionText $installPs1 "Get-TorchIndexUrl"), '(?m)^\s*(?:if|elseif)[^\r\n]*\$family = "cu\d+"[^\r\n]*$')
Check "install.ps1's inline ladder was found (bites)" ($installLadder.Count -ge 5)
$norm = {
    param($text)
    (($text -replace '(?i)\$Major', '$major') -replace '(?i)\$Minor', '$minor') -replace '\s+', ' '
}
$setupConds = @($ladderRows | ForEach-Object {
    $m = [regex]::Match($_.Value, '\((?<c>[^)]*(?:\([^)]*\)[^)]*)*)\)\s*\{')
    if ($m.Success) { (& $norm $m.Groups['c'].Value).Trim() }
})
$installConds = @($installLadder | ForEach-Object {
    $m = [regex]::Match($_.Value, '\((?<c>[^)]*(?:\([^)]*\)[^)]*)*)\)\s*\{')
    if ($m.Success) { (& $norm $m.Groups['c'].Value).Trim() }
})
Check "both ladders have the same number of rungs" ($setupConds.Count -eq $installConds.Count)
Check "and the rungs test the same conditions in the same order" (
    ($setupConds -join '|') -eq ($installConds -join '|'))

# Driven, because identical text is not the same as the same answer. These are the versions the
# driver floor table actually produces, plus the boundaries either side of every rung.
Invoke-Expression $ladderFn
foreach ($row in @(
    @{ M = 10; N = 2; Want = "cpu" },
    @{ M = 11; N = 0; Want = "cu118" },
    @{ M = 11; N = 8; Want = "cu118" },
    @{ M = 12; N = 0; Want = "cu124" },
    @{ M = 12; N = 5; Want = "cu124" },
    @{ M = 12; N = 6; Want = "cu126" },
    @{ M = 12; N = 7; Want = "cu126" },
    @{ M = 12; N = 8; Want = "cu128" },
    @{ M = 12; N = 9; Want = "cu128" },
    @{ M = 13; N = 0; Want = "cu130" }
)) {
    Check "CUDA $($row.M).$($row.N) is served by $($row.Want)" (
        (Get-CudaFamilyForVersion -Major $row.M -Minor $row.N) -eq $row.Want)
}
# The one that made this a bug rather than a nicety: an R450 to R524 driver.
Check "an R450 driver's floor lands on cu118, not a cu110 that does not exist" (
    (Get-CudaFamilyForVersion -Major 11 -Minor 0) -eq "cu118")

Check "install.ps1 answers the same way for the same host" (
    (Get-FunctionText $installPs1 "Get-TorchIndexUrl") -match '(?s)NvidiaPresenceDriverRelease -lt 450[\s\S]{0,1200}?/cpu')

# ------------------------------------------------- job 2b, the registry fallback is last resort
#
# These display-class keys outlive removed hardware and expose no ConfigManagerErrorCode, so a
# stale NVIDIA entry cannot be told apart from a working card. Reading one as a verified GPU
# would promote $HasNvidiaSmi, and that suppresses AMD and Intel detection as well as choosing
# CUDA wheels. The fallback therefore exists only for a WMI repository that could not answer.
#
# The registry is shadowed here rather than left to this host: a Linux runner has no HKLM, so
# without a stub every row below would pass for the wrong reason.
$script:RegistryConsulted = $false
function Get-ChildItem {
    param([string]$LiteralPath, [switch]$Directory, [string]$Filter, $ErrorAction)
    if ("$LiteralPath" -match 'Control\\Class') {
        $script:RegistryConsulted = $true
        return @([pscustomobject]@{ PSChildName = "0000"; PSPath = "fake::0000" })
    }
    return @()
}
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1" }
    }
    return $null
}

# A WMI scan that FAILED is the one case the fallback is for.
$script:FakeAdapters = @()
$script:FakeScanOk = $false
$script:RegistryConsulted = $false
Check "a WMI scan that could not answer falls back to the registry" ((Test-NvidiaAdapterPresent) -eq $true)
Check "and the registry really was the source" ($script:RegistryConsulted -eq $true)

# A WMI scan that ANSWERED is evidence, even when the answer is "no NVIDIA here".
foreach ($case in @(
    @{ N = "answered with no adapters at all"; A = @() },
    @{ N = "answered with an AMD adapter only"; A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 0) },
    @{ N = "answered with a faulted NVIDIA adapter"; A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 43) },
    @{ N = "answered with a disabled NVIDIA adapter"; A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 22) }
)) {
    $script:FakeAdapters = $case.A
    $script:FakeScanOk = $true
    $script:RegistryConsulted = $false
    Check "a scan that $($case.N) is absence, not a reason to ask the registry" (
        (Test-NvidiaAdapterPresent) -eq $false)
    Check "and the registry was never consulted" ($script:RegistryConsulted -eq $false)
}

# A healthy adapter still short-circuits before the registry is reached at all.
$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0)
$script:FakeScanOk = $true
$script:RegistryConsulted = $false
Check "a healthy NVIDIA adapter answers from WMI alone" ((Test-NvidiaAdapterPresent) -eq $true)
Check "and it did not need the registry either" ($script:RegistryConsulted -eq $false)

# A registry with nothing NVIDIA in it must still be a no, or the gate above is untestable.
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_1002&DEV_744C" }
}
$script:FakeAdapters = @()
$script:FakeScanOk = $false
Check "a failed scan with no NVIDIA in the registry is still absent" ((Test-NvidiaAdapterPresent) -eq $false)

# Both files must carry the gate, or setup.ps1 keeps the defect install.ps1 just lost.
foreach ($file in @($installPs1, $setupPs1)) {
    Check "$(Split-Path -Leaf $file) gates the registry fallback on a failed scan" (
        (Get-FunctionText $file "Test-NvidiaAdapterPresent") -match 'if \(\$Scan\.Ok\) \{ return \$false \}')
}

# ------------------------------------------ the fallback's own entry names a driver version too
#
# The presence check is not the only reader. A host in this fallback has WMI down, so
# $Scan.Adapters is empty and the floor had nothing to read: the routing below then took the
# unknown-version default of cu126, which an R450 to R524 driver cannot load. The class-key entry
# carries DriverVersion in the same spelling Win32_VideoController reports it, so the answer was
# there all along and was being discarded after the presence check.
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        # 31.0.15.3699 is the Windows display-driver form of NVIDIA release 536.99, which is the
        # form both WMI and this key report. R536 sits between the 535 and 545 rows, so its floor
        # is CUDA 12.2: a row that only a version READ FROM THE REGISTRY can produce.
        return [pscustomobject]@{
            MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"
            DriverVersion = "31.0.15.3699"
        }
    }
    return $null
}
$script:FakeAdapters = @()
$script:FakeScanOk = $false
Check "the failed-scan fallback still reports the GPU" ((Test-NvidiaAdapterPresent) -eq $true)
$regRelease = Get-NvidiaAdapterDriverRelease
Check "and the registry entry's driver release is read, not discarded" ($regRelease -eq 536)
$regFloor = @(Get-NvidiaAdapterCudaFloor)
Check "and it becomes a CUDA floor of 12.2 rather than nothing" (
    $regFloor.Count -eq 2 -and [int]$regFloor[0] -eq 12 -and [int]$regFloor[1] -eq 2)

# Several NVIDIA subkeys, and the first one does not name a version. A machine with retained
# driver configurations carries exactly that, and returning the first match would report the GPU
# and lose the floor, which is the whole reason this hands back an entry rather than a yes/no.
function Get-ChildItem {
    param([string]$LiteralPath, [switch]$Directory, [string]$Filter, $ErrorAction)
    if ("$LiteralPath" -match 'Control\\Class') {
        $script:RegistryConsulted = $true
        return @(
            [pscustomobject]@{ PSChildName = "0000"; PSPath = "fake::0000" },
            [pscustomobject]@{ PSChildName = "0001"; PSPath = "fake::0001" }
        )
    }
    return @()
}
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1" }
    }
    if ("$LiteralPath" -eq "fake::0001") {
        return [pscustomobject]@{
            MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"
            DriverVersion = "31.0.15.3699"
        }
    }
    return $null
}
$script:FakeAdapters = @()
$script:FakeScanOk = $false
Check "a later entry that names a version wins over an earlier one that does not" (
    (Get-NvidiaAdapterDriverRelease) -eq 536)
Check "and presence is unaffected by which entry answered" ((Test-NvidiaAdapterPresent) -eq $true)

# Two entries that disagree read as unknown, not as whichever came first. These keys outlive the
# hardware and enumeration order says nothing about which adapter is live, so a stale newer entry
# would pick wheels an older current driver cannot load, and a stale pre-R450 entry would demote a
# current GPU to CPU. Unknown is already handled: presence stands, the floor does not.
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"; DriverVersion = "31.0.15.3699" }
    }
    if ("$LiteralPath" -eq "fake::0001") {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"; DriverVersion = "30.0.14.4568" }
    }
    return $null
}
Check "two entries naming different releases read as unknown" ($null -eq (Get-NvidiaAdapterDriverRelease))
Check "and yield no floor either" ($null -eq (Get-NvidiaAdapterCudaFloor))
Check "while the GPU is still reported present" ((Test-NvidiaAdapterPresent) -eq $true)

# Two spellings of ONE release are agreement, not conflict: these entries can spell the same
# driver differently, and comparing the strings rather than the releases would throw the floor
# away on a machine that never disagreed with itself.
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"; DriverVersion = "31.0.15.3699" }
    }
    if ("$LiteralPath" -eq "fake::0001") {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"; DriverVersion = "536.99" }
    }
    return $null
}
Check "two spellings of one release still name it" ((Get-NvidiaAdapterDriverRelease) -eq 536)

# And with NO entry naming a version, the versionless one is still the presence answer.
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -match '^fake::000[01]$') {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1" }
    }
    return $null
}
Check "with no version anywhere the GPU is still found" ((Test-NvidiaAdapterPresent) -eq $true)
Check "and the release stays unknown" ($null -eq (Get-NvidiaAdapterDriverRelease))

# Back to one subkey for the rows below.
function Get-ChildItem {
    param([string]$LiteralPath, [switch]$Directory, [string]$Filter, $ErrorAction)
    if ("$LiteralPath" -match 'Control\\Class') {
        $script:RegistryConsulted = $true
        return @([pscustomobject]@{ PSChildName = "0000"; PSPath = "fake::0000" })
    }
    return @()
}

# Bites control: an entry with no DriverVersion is still unknown, not a guess.
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1" }
    }
    return $null
}
Check "an entry with no version stays unknown" ($null -eq (Get-NvidiaAdapterDriverRelease))
Check "and yields no floor" ($null -eq (Get-NvidiaAdapterCudaFloor))

# And a scan that ANSWERED never reaches the registry for a version either: the same evidence rule
# the presence check follows, or a stale key would set the family for a card that is gone.
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        return [pscustomobject]@{
            MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"
            DriverVersion = "31.0.15.3699"
        }
    }
    return $null
}
$script:FakeAdapters = @()
$script:FakeScanOk = $true
$script:RegistryConsulted = $false
Check "a scan that answered takes no version from the registry" (
    $null -eq (Get-NvidiaAdapterDriverRelease))
Check "and never consulted it" ($script:RegistryConsulted -eq $false)

# The OTHER half of the same exclusivity question, which for a while read a different source.
# Test-NvidiaAdapterPresent fell back to the class keys when WMI could not answer and
# Test-OtherVendorAdapterPresent did not, so a broken WMI repository on a hybrid NVIDIA plus Arc
# host saw NVIDIA and no alternative: the promotion fired, $HasNvidiaSmi suppressed the Intel
# branch, and the machine lost the XPU wheels it gets today.
foreach ($case in @(
    @{ N = "an Arc in the class keys";        Id = "PCI\\VEN_8086&DEV_56A0"; Desc = "Intel(R) Arc(TM) A770 Graphics"; Want = $true },
    @{ N = "a Data Center GPU";               Id = "PCI\\VEN_8086&DEV_0BD5"; Desc = "Intel(R) Data Center GPU Max 1100"; Want = $true },
    @{ N = "integrated Intel UHD";            Id = "PCI\\VEN_8086&DEV_9A49"; Desc = "Intel(R) UHD Graphics"; Want = $false },
    @{ N = "an AMD adapter of any kind";      Id = "PCI\\VEN_1002&DEV_744C"; Desc = "AMD Radeon RX 7900 XTX"; Want = $true },
    @{ N = "an NVIDIA adapter, which is not an alternative"; Id = "PCI\\VEN_10DE&DEV_1DB1"; Desc = "NVIDIA Tesla V100"; Want = $false }
)) {
    $script:CaseId = $case.Id
    $script:CaseDesc = $case.Desc
    function Get-ItemProperty {
        param([string]$LiteralPath, $ErrorAction)
        return [pscustomobject]@{ MatchingDeviceId = $script:CaseId; DriverDesc = $script:CaseDesc }
    }
    $script:FakeAdapters = @()
    $script:FakeScanOk = $false
    $script:RegistryConsulted = $false
    Check "registry fallback: $($case.N) reads as $(if ($case.Want) { 'an alternative' } else { 'no alternative' })" (
        (Test-OtherVendorAdapterPresent) -eq $case.Want)
    Check "  and the registry really was the source" ($script:RegistryConsulted -eq $true)
}

# Same last-resort gate as the NVIDIA side. A scan that ANSWERED is evidence either way, and a
# stale class key must not be allowed to block a promotion the live inventory permits.
$script:CaseId = "PCI\\VEN_1002&DEV_744C"
$script:CaseDesc = "AMD Radeon RX 7900 XTX"
$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0)
$script:FakeScanOk = $true
$script:RegistryConsulted = $false
Check "a scan that answered is not second-guessed by the registry" ((Test-OtherVendorAdapterPresent) -eq $false)
Check "and the registry was never consulted" ($script:RegistryConsulted -eq $false)

foreach ($file in @($installPs1, $setupPs1)) {
    Check "$(Split-Path -Leaf $file) gates the other-vendor fallback on a failed scan too" (
        (Get-FunctionText $file "Test-OtherVendorAdapterPresent") -match 'if \(\$Scan\.Ok\) \{ return \$false \}')
    # Read from the same shared pattern, not a second copy: DriverDesc is the registry's spelling
    # of Name, and the question is still "will the XPU route serve this", not "is it Intel".
    Check "$(Split-Path -Leaf $file) asks the shared XPU pattern of DriverDesc" (
        (Get-FunctionText $file "Test-OtherVendorAdapterPresent") -match 'DriverDesc[\s\S]{0,60}Get-XpuCapableNameRegex')
}

# ------------------------------------------------------- job 3, the wheel it actually leads to
#
# This is the half that was missing when the branch was first written, and leaving it out made the
# whole thing a net regression: promoting the flag without giving the selector a version produced
# $HasNvidiaSmi true, AMD detection suppressed, and a CPU index anyway. Measured, not reasoned
# about, by running the real Get-TorchIndexUrl.
$idxAst = [System.Management.Automation.Language.Parser]::ParseFile($installPs1, [ref]$null, [ref]$null)
foreach ($n in @("Get-TorchIndexUrl", "Trim-IndexPathSlashes", "Get-CudaFamilyCappedForPreTuring")) {
    $f = @($idxAst.FindAll({ param($x)
        $x -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $x.Name -eq $n }, $true))
    if ($f.Count -ge 1) { Invoke-Expression $f[0].Extent.Text }
}
# Stubbed only where real hardware is needed. Get-NvidiaCu126Verdict returns "" on a host with no
# compute capability to inspect, which is exactly what it returns here.
function Get-NvidiaCu126Verdict { param($a, $b) return "" }
function Get-NvidiaLibraryInventory { return $script:Inv }
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return $script:Banner }
function substep { param($a, $b) $script:Said = "$a" }

foreach ($case in @(
    # The ordinary non-NVIDIA host. Must stay SILENT and CPU: this branch is reached by every AMD,
    # Intel and CPU-only machine, and making it loud would warn all of them.
    @{ N = "an AMD, Intel or CPU-only host";        Exe = $null; Inv = $null; Pres = $false; Floor = $null; Want = "cpu";   Loud = $false },
    # A presence-only host has NO compute capability from any source, so the pre-Turing cap
    # cannot fire: Get-NvidiaCu126Verdict is handed no nvidia-smi and no caps and returns "".
    # A driver floor of 12.8 or 13.0 would then pick a family whose PyTorch 2.11 wheels start at
    # sm_75, and a pre-Turing card (a V100 is sm_70) would install torch and fail at the first
    # kernel. Unknown capability stops at cu126 whatever the driver says it could carry.
    @{ N = "presence-only with a 12.8 driver";      Exe = $null; Inv = $null; Pres = $true;  Floor = @(12,8); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with a 12.6 driver";      Exe = $null; Inv = $null; Pres = $true;  Floor = @(12,6); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with a 13.0 driver";      Exe = $null; Inv = $null; Pres = $true;  Floor = @(13,0); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with an 11.0 driver";     Exe = $null; Inv = $null; Pres = $true;  Floor = @(11,0); Want = "cu118"; Loud = $true },
    @{ N = "presence-only with no driver version";  Exe = $null; Inv = $null; Pres = $true;  Floor = $null;   Want = "cu126"; Loud = $true },
    # A version WAS readable and it is below every row of the table. Pre-R450 drivers carry no
    # CUDA runtime any offered wheel can use, so a CUDA family here downloads gigabytes that
    # cannot run. CPU is the honest answer, and it is the answer this host got before the
    # promotion existed; the only change is that it is no longer silent.
    @{ N = "presence-only on a pre-R450 driver"; Exe = $null; Inv = $null; Pres = $true; Floor = $null; Release = 445; Want = "cpu"; Loud = $true },
    @{ N = "presence-only on R450 exactly";      Exe = $null; Inv = $null; Pres = $true; Floor = @(11,0); Release = 450; Want = "cu118"; Loud = $true },
    # Unknown is NOT too old: a driver whose version could not be read at all keeps the
    # conservative CUDA family rather than being demoted to CPU with it.
    @{ N = "presence-only with an unreadable driver version"; Exe = $null; Inv = $null; Pres = $true; Floor = $null; Release = $null; Want = "cu126"; Loud = $true }
)) {
    $NvidiaSmiExe = $case.Exe
    $script:Inv = $case.Inv
    $script:Banner = ""
    $script:NvidiaPresenceOnly = $case.Pres
    $script:NvidiaPresenceCudaFloor = $case.Floor
    $script:NvidiaPresenceDriverRelease = $case.Release
    $script:Said = ""
    $leaf = ("" + (Get-TorchIndexUrl)) -replace '^.*/', ''
    Check "$($case.N) selects $($case.Want)" ($leaf -eq $case.Want)
    Check "and it $(if ($case.Loud) { 'says so' } else { 'stays silent' })" (
        [bool]$script:Said -eq $case.Loud)
}

# The cap is not silent. A host whose driver could carry 13.0 and which nonetheless gets cu126
# is told why and told the override, or the user has no way to ask for the other answer.
$NvidiaSmiExe = $null
$script:Inv = $null
$script:Banner = ""
$script:NvidiaPresenceOnly = $true
$script:NvidiaPresenceCudaFloor = @(13,0)
$script:NvidiaPresenceDriverRelease = 580
$script:Said = ""
$null = Get-TorchIndexUrl
Check "the capped row explains the cap" ($script:Said -match 'compute capability')
Check "the capped row names the override" ($script:Said -match 'UNSLOTH_TORCH_INDEX_FAMILY')

# The floor is still read: a driver too old for cu126 must not be lifted UP to it by the cap.
$script:NvidiaPresenceCudaFloor = @(11,0)
$script:Said = ""
Check "the cap lowers and never raises" ((("" + (Get-TorchIndexUrl)) -replace '^.*/', '') -eq "cu118")

# The rows this must NOT have touched. Every other path into the selector keeps its old answer,
# which is what makes the change additive rather than a rewrite of wheel selection.
foreach ($case in @(
    @{ N = "nvidia-smi with a readable banner"; Exe = "C:\it.exe"; Inv = $null; Banner = "CUDA Version: 12.8"; Want = "cu128"; Loud = $false },
    @{ N = "nvidia-smi with an unreadable banner"; Exe = "C:\it.exe"; Inv = $null; Banner = "broken"; Want = "cu126"; Loud = $true },
    @{ N = "the driver library answering"; Exe = $null; Banner = ""; Want = "cu128"; Loud = $false
       Inv = [pscustomobject]@{ CudaMajor = 12; CudaMinor = 8; ComputeCaps = @() } }
)) {
    $NvidiaSmiExe = $case.Exe
    $script:Inv = $case.Inv
    $script:Banner = $case.Banner
    $script:NvidiaPresenceOnly = $false
    $script:NvidiaPresenceCudaFloor = $null
    $script:NvidiaPresenceDriverRelease = $null
    $script:Said = ""
    $leaf = ("" + (Get-TorchIndexUrl)) -replace '^.*/', ''
    Check "unchanged: $($case.N) still selects $($case.Want)" ($leaf -eq $case.Want)
    Check "unchanged: and it $(if ($case.Loud) { 'still says so' } else { 'still stays silent' })" (
        [bool]$script:Said -eq $case.Loud)
}

# And an explicit pin still beats everything, including the new branch.
$env:UNSLOTH_TORCH_INDEX_FAMILY = "cu118"
try {
    $script:NvidiaPresenceOnly = $true
    $script:NvidiaPresenceCudaFloor = @(13, 0)
    Check "an explicit family override still wins over the bus floor" (
        (("" + (Get-TorchIndexUrl)) -replace '^.*/', '') -eq "cu118")
} finally { Remove-Item Env:UNSLOTH_TORCH_INDEX_FAMILY -ErrorAction SilentlyContinue }

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
