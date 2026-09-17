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
    "Test-OtherVendorAdapterPresent", "Get-NvidiaDriverRelease", "Get-NvidiaDriverCudaFloor",
    "Get-NvidiaAdapterDriverRelease", "Get-NvidiaAdapterCudaFloor"
)
foreach ($name in $shared) {
    Check "install.ps1 and setup.ps1 carry the same $name" (
        (& $strip (Get-FunctionText $installPs1 $name)) -eq (& $strip (Get-FunctionText $setupPs1 $name)))
}

foreach ($name in @("Test-NvidiaAdapterPresent", "Test-OtherVendorAdapterPresent",
                    "Get-NvidiaDriverRelease", "Get-NvidiaDriverCudaFloor",
                    "Get-NvidiaAdapterDriverRelease", "Get-NvidiaAdapterCudaFloor")) {
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
$psText = Get-FunctionText $setupPs1 "Get-NvidiaDriverCudaFloor"
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
Check "install.ps1 is where the release is consumed" (
    (Get-FunctionText $installPs1 "Get-TorchIndexUrl") -match 'NvidiaPresenceDriverRelease')
Check "setup.ps1 does not act on it, for the same reason it does not floor" (
    (Get-FunctionText $setupPs1 "Get-PytorchCudaTag") -notmatch 'NvidiaPresenceDriverRelease')

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
