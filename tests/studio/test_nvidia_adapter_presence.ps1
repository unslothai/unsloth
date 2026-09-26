#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# $HasNvidiaSmi suppresses AMD and Intel detection, so the presence source must say yes only for a
# healthy VEN_10DE adapter (names are deliberately adversarial) and must lead to a real index.
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

# install.ps1 nests one level deeper, so compare without indentation.
$strip = { param($t) (($t -split "`n") | ForEach-Object { $_.TrimStart() }) -join "`n" }
$shared = @(
    "Invoke-BoundedVideoControllerScan", "Test-NvidiaAdapterPresent",
    "Test-OtherVendorAdapterPresent", "Get-XpuCapableNameRegex",
    "Get-NvidiaDriverRelease", "Get-NvidiaCudaFloorForRelease", "Get-NvidiaDriverCudaFloor",
    "Get-NvidiaAdapterDriverRelease", "Get-NvidiaAdapterCudaFloor",
    "Get-NvidiaRegistryAdapter", "Test-NvidiaAdapterWithoutNvidiaDriver", "Test-IntelXpuRuntimeProven"
)
foreach ($name in $shared) {
    Check "install.ps1 and setup.ps1 carry the same $name" (
        (& $strip (Get-FunctionText $installPs1 $name)) -eq (& $strip (Get-FunctionText $setupPs1 $name)))
}

foreach ($name in @("Test-NvidiaAdapterPresent", "Test-OtherVendorAdapterPresent", "Get-XpuCapableNameRegex",
                    "Get-NvidiaDriverRelease", "Get-NvidiaCudaFloorForRelease", "Get-NvidiaDriverCudaFloor",
                    "Get-NvidiaAdapterDriverRelease", "Get-NvidiaAdapterCudaFloor",
                    "Get-NvidiaRegistryAdapter", "Test-NvidiaAdapterWithoutNvidiaDriver")) {
    Invoke-Expression (Get-FunctionText $setupPs1 $name)
}
$script:FakeRegistryNames = @()
$script:FakeRegistryThrows = $false
function Get-IntelRegistryAdapterNames {
    if ($script:FakeRegistryThrows) { throw "registry unreadable" }
    return $script:FakeRegistryNames
}

$script:FakeAdapters = @()
# $null derives Ok from the fixture.
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

# Names are the trap: only the vendor ID counts.
$script:FakeAdapters = @(Adapter "NVIDIA compatible display" "PCI\VEN_1002&DEV_744C" 0)
Check "a non-NVIDIA card whose NAME says NVIDIA is not counted" ((Test-NvidiaAdapterPresent) -eq $false)

$script:FakeAdapters = @(Adapter "Carte graphique" "PCI\VEN_10DE&DEV_2684" 0)
Check "an NVIDIA card whose name says nothing recognisable is still counted" (
    (Test-NvidiaAdapterPresent) -eq $true)

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

foreach ($case in @(
    @{ N = "a healthy AMD adapter";   A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 0); Want = $true },
    @{ N = "a healthy Intel adapter"; A = @(Adapter "Intel Arc A770" "PCI\VEN_8086&DEV_56A0" 0); Want = $true },
    # The AMD route keeps faulted, parked and PNP-less Radeons, so each must veto the promotion.
    @{ N = "a faulted AMD adapter";   A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 43); Want = $true },
    @{ N = "a parked AMD adapter";    A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 45); Want = $true },
    @{ N = "an AMD adapter with no PNP ID"; A = @(Adapter "AMD Radeon RX 7900 XTX" "" 0); Want = $true },
    @{ N = "a faulted Intel UHD adapter"; A = @(Adapter "Intel(R) UHD Graphics 770" "PCI\VEN_8086&DEV_4680" 43); Want = $false },
    @{ N = "an NVIDIA adapter only";  A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0); Want = $false },
    @{ N = "a virtual adapter only";  A = @(Adapter "Microsoft Basic Display Adapter" "ROOT\BASICDISPLAY" 0); Want = $false }
)) {
    $script:FakeAdapters = $case.A
    Check "other-vendor gate sees $($case.N) as $($case.Want)" (
        (Test-OtherVendorAdapterPresent) -eq $case.Want)
}

$script:FakeAdapters = @(
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
    (Adapter "AMD Radeon 780M" "PCI\VEN_1002&DEV_15BF" 0))
Check "on a hybrid NVIDIA plus AMD host presence is true but the gate blocks promotion" (
    (Test-NvidiaAdapterPresent) -eq $true -and (Test-OtherVendorAdapterPresent) -eq $true)

# UHD / Iris get no XPU wheels, so they must not block the promotion.
foreach ($case in @(
    @{ N = "Intel UHD Graphics";    Name = "Intel(R) UHD Graphics 770";        Blocks = $false },
    @{ N = "Intel Iris Xe";         Name = "Intel(R) Iris(R) Xe Graphics";     Blocks = $false },
    @{ N = "Intel HD Graphics";     Name = "Intel(R) HD Graphics 630";         Blocks = $false },
    @{ N = "Intel Arc A770";        Name = "Intel(R) Arc(TM) A770 Graphics";   Blocks = $true },
    @{ N = "Intel Arc B580";        Name = "Intel(R) Arc(TM) B580 Graphics";   Blocks = $true },
    @{ N = "Intel Data Center GPU"; Name = "Intel(R) Data Center GPU Max 1100"; Blocks = $true },
    @{ N = "a nameless Intel adapter"; Name = "";                              Blocks = $false }
)) {
    $script:FakeAdapters = @(
        (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
        (Adapter $case.Name "PCI\VEN_8086&DEV_4680" 0))
    Check "NVIDIA plus $($case.N): the gate $(if ($case.Blocks) { 'blocks' } else { 'allows' }) promotion" (
        (Test-OtherVendorAdapterPresent) -eq $case.Blocks)
    Check "  and NVIDIA presence is unaffected either way" ((Test-NvidiaAdapterPresent) -eq $true)
}

$script:FakeAdapters = @(
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
    (Adapter "AMD Radeon(TM) Graphics" "PCI\VEN_1002&DEV_15BF" 0))
Check "an integrated AMD adapter still blocks, unlike the Intel one" (
    (Test-OtherVendorAdapterPresent) -eq $true)

$gateText = Get-FunctionText $installPs1 "Test-OtherVendorAdapterPresent"
$installAll = [System.IO.File]::ReadAllText($installPs1)
Check "the gate defers to the shared XPU-capable pattern" ($gateText -match 'Get-XpuCapableNameRegex')
# @() around the WHOLE pipeline: one unique string would make [0] its first character.
$patternDefs = @(@([regex]::Matches($installAll, 'function Get-XpuCapableNameRegex \{ return "([^"]+)" \}') |
    ForEach-Object { $_.Groups[1].Value }) | Sort-Object -Unique)
$routeDefs = @(@([regex]::Matches($installAll, '\$_xpuNameRe = "([^"]+)"') |
    ForEach-Object { $_.Groups[1].Value }) | Sort-Object -Unique)
Check "the extracted patterns are whole strings, not characters" (
    $patternDefs[0].Length -gt 5 -and $routeDefs[0].Length -gt 5)
Check "both patterns are defined exactly once" ($patternDefs.Count -eq 1 -and $routeDefs.Count -eq 1)
Check "the gate pattern and the XPU route pattern are identical" ($patternDefs[0] -ceq $routeDefs[0])

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

$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0 "32.0.15.7020")
foreach ($case in @(
    @{ V = "32.0.15.6094"; R = 560; N = "a Windows display-driver version" },
    @{ V = "30.0.14.4568"; R = 445; N = "a pre-R450 Windows version" },
    @{ V = "580.65.06";    R = 580; N = "an nvidia-smi style version" },
    @{ V = "445.68";       R = 445; N = "a pre-R450 nvidia-smi style version" },
    @{ V = "";             R = $null; N = "an empty version" },
    @{ V = "not a version"; R = $null; N = "an unparseable version" },
    @{ V = "1.2.3";        R = $null; N = "a three-field version" },
    @{ V = "99.1";         R = $null; N = "a two-digit release" },
    @{ V = "100.1";        R = 100;  N = "the lowest release shape NVIDIA actually ships" },
    # Microsoft Basic Display versions are not NVIDIA releases.
    @{ V = "10.0.19041.3636"; R = $null; N = "a Microsoft Basic Display driver version" },
    @{ V = "10.0.22621.1";    R = $null; N = "a Windows 11 Basic Display driver version" },
    @{ V = "10.0.19041.1";    R = $null; N = "a Windows 10 Basic Display driver version" },
    @{ V = "31.0.15.3623";    R = 536; N = "a WDDM 3.1 NVIDIA version" },
    @{ V = "26.21.14.4250";   R = 442; N = "a WDDM 2.6 NVIDIA version" },
    @{ V = "21.21.13.7892";   R = 378; N = "a WDDM 2.1 NVIDIA version" }
)) {
    $got = Get-NvidiaDriverRelease -DriverVersion $case.V
    Check "$($case.N) reads as release $($case.R)" ($got -eq $case.R)
}
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

# Generated from studio/nvidia_probe.py so the tables cannot drift.
$probePy = Join-Path $root "studio/nvidia_probe.py"
$probeText = Get-Content -Raw -LiteralPath $probePy
$pyRows = @()
$tableMatch = [regex]::Match($probeText, '(?s)_DRIVER_MAJOR_CUDA\s*=\s*\((.*?)\)\s*\n\s*\n')
Check "the python floor table was found" ($tableMatch.Success)
foreach ($m in [regex]::Matches($tableMatch.Groups[1].Value, '\((\d+),\s*\((\d+),\s*(\d+)\)\)')) {
    $pyRows += "$($m.Groups[1].Value):$($m.Groups[2].Value).$($m.Groups[3].Value)"
}
Check "the python table has rows to compare (bites)" ($pyRows.Count -ge 5)
# Get-NvidiaDriverCudaFloor no longer holds the table; reading it would pass on an empty list.
$psText = Get-FunctionText $setupPs1 "Get-NvidiaCudaFloorForRelease"
$psRows = @()
foreach ($m in [regex]::Matches($psText, '@\((\d+),\s*(\d+),\s*(\d+)\)')) {
    $psRows += "$($m.Groups[1].Value):$($m.Groups[2].Value).$($m.Groups[3].Value)"
}
Check "the PowerShell floor table matches nvidia_probe.py exactly" (
    ($pyRows -join ",") -eq ($psRows -join ","))

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

    # PowerShell does not hoist, and the pwsh suites define functions in dependency order themselves.
    foreach ($fn in $shared) {
        $def = $text.IndexOf("function $fn")
        Check "$name defines $fn before the promotion that uses it" ($def -gt 0 -and $def -lt $presence)
    }
    $scanCalls = 0
    foreach ($m in [regex]::Matches($text, '(?<!function )Invoke-BoundedVideoControllerScan')) {
        $lineStart = $text.LastIndexOf("`n", [Math]::Max($m.Index - 1, 0)) + 1
        if ($text.Substring($lineStart, $m.Index - $lineStart).Contains("#")) { continue }
        $scanCalls++
        $def = $text.IndexOf("function Invoke-BoundedVideoControllerScan")
        Check "$name calls Invoke-BoundedVideoControllerScan at offset $($m.Index) after its definition" (
            $m.Index -gt $def)
    }
    Check "$name has real calls to the bounded scan to order (bites)" ($scanCalls -gt 0)
}

# setup.ps1 must NOT guess: "" keys the do-not-wipe escape (#9255).
$setupTag = Get-FunctionText $setupPs1 "Get-PytorchCudaTag"
Check "setup.ps1 still returns empty for unknown rather than flooring" (
    $setupTag -notmatch 'NvidiaPresenceCudaFloor')
Check "and install.ps1 is where the floor is consumed" (
    (Get-FunctionText $installPs1 "Get-TorchIndexUrl") -match 'NvidiaPresenceCudaFloor')

# The rows above set the release directly, so check the promotion site records it.
foreach ($file in @($installPs1, $setupPs1)) {
    $text = [System.IO.File]::ReadAllText($file)
    $leaf = Split-Path -Leaf $file
    Check "$leaf records the driver release at the promotion site" (
        $text -match '\$_presenceRelease = Get-NvidiaAdapterDriverRelease -Scan \$presenceScan[\s\S]{0,800}?\$script:NvidiaPresenceDriverRelease = \$_presenceRelease')
    Check "$leaf records it beside the floor, from the same scan" (
        $text -match 'Get-NvidiaAdapterCudaFloor -Scan \$presenceScan[\s\S]{0,400}?Get-NvidiaAdapterDriverRelease -Scan \$presenceScan')
    Check "$leaf initialises the release to null" (
        $text -match '\$script:NvidiaPresenceDriverRelease = \$null')
}

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

# Anchored on ORDER: an arm after the cu126 fallback is dead code.
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
    Check "and marks the demotion so the CPU arm really replaces the CUDA wheel" (
        $arm -match '(?s)-lt 450\)? \{[^}]*NvidiaPreR450CpuFallback = \$true')
    Check "and only when a CUDA wheel is actually installed" (
        $arm -match 'Test-CudaFamilyLeaf \$installedTorchTag\) \{ \$script:NvidiaPreR450CpuFallback = \$true \}')
}
$escapes = Get-FunctionText $setupPs1 "Invoke-FastPathEscapes"
Check "setup.ps1's fast-path escapes were found (bites)" (-not [string]::IsNullOrWhiteSpace($escapes))
if ($escapes) {
    Check "a bus-only NVIDIA host with CPU torch forces the dependency pass" (
        $escapes -match '\$script:NvidiaPresenceOnly' -and
        $escapes -match '(?s)NvidiaPresenceOnly[^}]*\$SkipPythonDeps = \$false')
    Check "and only when the installed wheel is a CPU one" (
        $escapes -match '\$installedTorchTag -eq "cpu"')
    Check "a pre-R450 driver does not trigger it" (
        $escapes -match '\$_nvidiaPreR450' -and $escapes -match 'NvidiaPresenceDriverRelease -lt 450')
    Check "nor does a pin that sends this host somewhere else" (
        $escapes -match '\$_nvidiaIsReachable' -and $escapes -match 'Test-CudaFamilyLeaf \$_pinLeafNow')
}
$cudaArm = [regex]::Match($setupText, '(?s)substep "installing PyTorch with CUDA support.{0,1500}')
Check "setup.ps1's CUDA install arm was found (bites)" ($cudaArm.Success)
if ($cudaArm.Success) {
    Check "the CUDA arm replaces an installed CPU wheel on a bus-only host" (
        $cudaArm.Value -match '(?s)\$script:NvidiaPresenceOnly -and \$installedTorchTag -eq "cpu"[^}]*--force-reinstall')
}

$cpuArm = [regex]::Match($setupText, '(?s)substep "installing PyTorch \(CPU-only\)\.\.\.".{0,3000}')
Check "setup.ps1's CPU install arm was found (bites)" ($cpuArm.Success)
if ($cpuArm.Success) {
    Check "the CPU arm force-reinstalls after a pre-R450 demotion" (
        $cpuArm.Value -match 'if \(\$script:NvidiaPreR450CpuFallback\) \{ \$cpuForce = @\("--force-reinstall"\) \}')
    Check "the same way it does after the ROCm and XPU fallbacks (bites)" (
        $cpuArm.Value -match 'if \(\$ROCmCpuFallback\) \{ \$cpuForce = @\("--force-reinstall"\) \}')
}
$staleHelper = Get-FunctionText $setupPs1 "Test-NvidiaPresenceStaleGpuWheel"
Check "setup.ps1 defines Test-NvidiaPresenceStaleGpuWheel (bites)" (-not [string]::IsNullOrWhiteSpace($staleHelper))
$staleEscape = $null
if ($escapes) {
    $escAst = [System.Management.Automation.Language.Parser]::ParseInput($escapes, [ref]$null, [ref]$null)
    $staleEscape = @($escAst.FindAll({ param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and
        $n.Clauses[0].Item1.Extent.Text -match 'Test-NvidiaPresenceStaleGpuWheel' }, $true))[0]
}
Check "the fast-path escape asks it (bites)" ($null -ne $staleEscape)
Check "the CUDA arm forces the reinstall off a stale +rocm/+xpu wheel" (
    $cudaArm.Success -and
    $cudaArm.Value -match '(?s)if \(Test-NvidiaPresenceStaleGpuWheel [^{]*\) \{\s*\$cudaForce = @\("--force-reinstall"\)')
$preArm = [regex]::Match($setupText, '(?s)NvidiaPresenceDriverRelease -lt 450\) \{.{0,2500}?predates R450')
Check "a pre-R450 demotion replaces a stale +rocm/+xpu wheel too" (
    $preArm.Success -and
    $preArm.Value -match '(?s)if \(Test-NvidiaPresenceStaleGpuWheel [^{]*\) \{\s*\$script:NvidiaPreR450CpuFallback = \$true')
Check "the cached answer starts empty on every run" (
    $setupText -match '(?m)^\$script:NvidiaPresenceStaleGpuWheel = \$null$')
if ($staleHelper -and $staleEscape) {
    foreach ($case in @(
        # tag, presence-only, probe answer, pre-R450, want stale, want skip afterwards
        @{ T = "rocm";  P = $true;  A = "DEV=False"; R = $false; Stale = $true;  Skip = $false; N = "+rocm that sees no GPU" },
        @{ T = "xpu";   P = $true;  A = "DEV=False"; R = $false; Stale = $true;  Skip = $false; N = "+xpu that sees no GPU" },
        @{ T = "rocm";  P = $true;  A = "DEV=False"; R = $true;  Stale = $true;  Skip = $false; N = "+rocm on a pre-R450 driver (goes to CPU)" },
        @{ T = "xpu";   P = $true;  A = "DEV=True";  R = $false; Stale = $false; Skip = $true;  N = "+xpu that still sees its GPU (kept, as the base did)" },
        @{ T = "rocm";  P = $true;  A = "DEV=True";  R = $false; Stale = $false; Skip = $true;  N = "+rocm that still sees a GPU" },
        @{ T = "rocm";  P = $true;  A = $null;       R = $false; Stale = $false; Skip = $true;  N = "+rocm whose probe did not answer" },
        @{ T = "cu128"; P = $true;  A = "DEV=False"; R = $false; Stale = $false; Skip = $true;  N = "a CUDA wheel" },
        @{ T = "cpu";   P = $true;  A = "DEV=False"; R = $true;  Stale = $false; Skip = $true;  N = "a CPU wheel on pre-R450 (this escape)" },
        @{ T = "rocm";  P = $false; A = "DEV=False"; R = $false; Stale = $false; Skip = $true;  N = "+rocm on a host with no bus promotion" })) {
        $got = & {
            param($c)
            $script:NvidiaPresenceOnly = $c.P
            $script:NvidiaPresenceStaleGpuWheel = $null
            $script:ProbeCalls = 0
            function Invoke-BoundedPythonProbe { param($PythonExe, $Code, $TimeoutSec)
                $script:ProbeCalls++
                if ($null -eq $c.A) { return [pscustomobject]@{ Ok = $false; Output = ""; TimedOut = $true } }
                return [pscustomobject]@{ Ok = $true; Output = "$($c.A)`n"; TimedOut = $false } }
            function substep { param($a, $b) }
            . ([scriptblock]::Create($staleHelper))
            $VenvDir = "venv"; $installedTorchTag = $c.T
            $_nvidiaIsReachable = $true; $_nvidiaPreR450 = $c.R; $SkipPythonDeps = $true
            . ([scriptblock]::Create($staleEscape.Extent.Text))
            $again = Test-NvidiaPresenceStaleGpuWheel -InstalledTag $c.T -PythonExe "C:\venv\Scripts\python.exe"
            [pscustomobject]@{ Skip = $SkipPythonDeps; Stale = $again; Calls = $script:ProbeCalls }
        } $case
        Check "bus-only escape, $($case.N): dependency pass = $(-not $case.Skip)" ($got.Skip -eq $case.Skip)
        Check "bus-only helper, $($case.N): stale = $($case.Stale), probed at most once" (
            $got.Stale -eq $case.Stale -and $got.Calls -le 1)
    }
    $script:NvidiaPresenceOnly = $false; $script:NvidiaPresenceStaleGpuWheel = $null
}
Check "the flag starts false on every run" (
    $setupText -match '(?m)^\$script:NvidiaPreR450CpuFallback = \$false$')
if ($nvArm.Success) {
    $arm = $nvArm.Value
    Check "setup.ps1 selects from the recorded floor" (
        $arm -match '\$script:NvidiaPresenceCudaFloor')
    Check "and only when no CUDA family is already installed" (
        $arm -match '-not \(Test-CudaFamilyLeaf \$installedTorchTag\)[\s\S]{0,80}?NvidiaPresenceCudaFloor')
    Check "through the shared ladder rather than by formatting the digits" (
        $arm -match 'Get-CudaFamilyForVersion -Major \$floorMajor' -and
        $arm -notmatch 'cu\$floorMajor')
    Check "and applies the same pre-Turing cap" (
        $arm -match '\$floorMajor -gt 12 -or \(\$floorMajor -eq 12 -and \$floorMinor -gt 6\)')
    Check "and the pre-R450 arm still decides first" (
        $arm.IndexOf('NvidiaPresenceDriverRelease') -lt $arm.IndexOf('NvidiaPresenceCudaFloor'))
}

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
Check "an R450 driver's floor lands on cu118, not a cu110 that does not exist" (
    (Get-CudaFamilyForVersion -Major 11 -Minor 0) -eq "cu118")

Check "install.ps1 answers the same way for the same host" (
    (Get-FunctionText $installPs1 "Get-TorchIndexUrl") -match '(?s)NvidiaPresenceDriverRelease -lt 450[\s\S]{0,1200}?/cpu')

# No HKLM on Linux: the registry is stubbed so rows cannot pass for the wrong reason.
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

# A failed scan must not fall back to stale class keys.
$script:FakeAdapters = @()
$script:FakeScanOk = $false
$script:RegistryConsulted = $false
Check "a WMI scan that could not answer is not presence, even with an NVIDIA class key" ((Test-NvidiaAdapterPresent) -eq $false)
Check "and the registry was never consulted for presence" ($script:RegistryConsulted -eq $false)

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

$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0)
$script:FakeScanOk = $true
$script:RegistryConsulted = $false
Check "a healthy NVIDIA adapter answers from WMI alone" ((Test-NvidiaAdapterPresent) -eq $true)
Check "and it did not need the registry either" ($script:RegistryConsulted -eq $false)

function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_1002&DEV_744C" }
}
$script:FakeAdapters = @()
$script:FakeScanOk = $false
Check "a failed scan with no NVIDIA in the registry is still absent" ((Test-NvidiaAdapterPresent) -eq $false)

foreach ($file in @($installPs1, $setupPs1)) {
    Check "$(Split-Path -Leaf $file) never reads presence from the class keys" (
        (Get-FunctionText $file "Test-NvidiaAdapterPresent") -notmatch 'Get-NvidiaRegistryAdapter|Control\\Class')
}

function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -eq "fake::0000") {
        # R536 floors to CUDA 12.2, a value only the registry version can produce.
        return [pscustomobject]@{
            MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"
            DriverVersion = "31.0.15.3699"
        }
    }
    return $null
}
$script:FakeAdapters = @()
$script:FakeScanOk = $false
Check "a failed scan is still not presence, whatever the class key names" ((Test-NvidiaAdapterPresent) -eq $false)
$regRelease = Get-NvidiaAdapterDriverRelease
Check "and the registry entry's driver release is read, not discarded" ($regRelease -eq 536)
$regFloor = @(Get-NvidiaAdapterCudaFloor)
Check "and it becomes a CUDA floor of 12.2 rather than nothing" (
    $regFloor.Count -eq 2 -and [int]$regFloor[0] -eq 12 -and [int]$regFloor[1] -eq 2)

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
Check "and presence is still not read from the class keys" ((Test-NvidiaAdapterPresent) -eq $false)

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
Check "and disagreeing entries are not presence either" ((Test-NvidiaAdapterPresent) -eq $false)

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

function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -match '^fake::000[01]$') {
        return [pscustomobject]@{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1" }
    }
    return $null
}
Check "with no version anywhere there is still no presence" ((Test-NvidiaAdapterPresent) -eq $false)
Check "and the release stays unknown" ($null -eq (Get-NvidiaAdapterDriverRelease))

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
Check "an entry with no version stays unknown" ($null -eq (Get-NvidiaAdapterDriverRelease))
Check "and yields no floor" ($null -eq (Get-NvidiaAdapterCudaFloor))

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

# The real helper, so the fallback reads the class keys through the Intel route's normalization.
$registryNamesStub = ${function:Get-IntelRegistryAdapterNames}
Invoke-Expression (Get-FunctionText $setupPs1 "Get-IntelRegistryAdapterNames")
foreach ($case in @(
    @{ N = "an Arc named without the vendor"; Id = "PCI\\VEN_8086&DEV_56A0"; Desc = "Arc A770 Graphics"; Want = $true },
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
${function:Get-IntelRegistryAdapterNames} = $registryNamesStub

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
    Check "$(Split-Path -Leaf $file) asks the shared XPU pattern of the Intel route's registry names" (
        (Get-FunctionText $file "Test-OtherVendorAdapterPresent") -match 'Get-IntelRegistryAdapterNames \| Where-Object \{ "\$_" -match \(Get-XpuCapableNameRegex\)')
}

# Runs the real Get-TorchIndexUrl: promoting without a version used to still yield a CPU index.
$idxAst = [System.Management.Automation.Language.Parser]::ParseFile($installPs1, [ref]$null, [ref]$null)
foreach ($n in @("Get-TorchIndexUrl", "Trim-IndexPathSlashes", "Get-CudaFamilyCappedForPreTuring")) {
    $f = @($idxAst.FindAll({ param($x)
        $x -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $x.Name -eq $n }, $true))
    if ($f.Count -ge 1) { Invoke-Expression $f[0].Extent.Text }
}
function Get-NvidiaCu126Verdict { param($a, $b) return "" }
function Get-NvidiaLibraryInventory { return $script:Inv }
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return $script:Banner }
function substep { param($a, $b) $script:Said = "$a" }

foreach ($case in @(
    # Must stay SILENT: every AMD, Intel and CPU-only host reaches this.
    @{ N = "an AMD, Intel or CPU-only host";        Exe = $null; Inv = $null; Pres = $false; Floor = $null; Want = "cpu";   Loud = $false },
    # No compute capability, so unknown stops at cu126 (pre-Turing).
    @{ N = "presence-only with a 12.8 driver";      Exe = $null; Inv = $null; Pres = $true;  Floor = @(12,8); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with a 12.6 driver";      Exe = $null; Inv = $null; Pres = $true;  Floor = @(12,6); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with a 13.0 driver";      Exe = $null; Inv = $null; Pres = $true;  Floor = @(13,0); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with an 11.0 driver";     Exe = $null; Inv = $null; Pres = $true;  Floor = @(11,0); Want = "cu118"; Loud = $true },
    @{ N = "presence-only with no driver version";  Exe = $null; Inv = $null; Pres = $true;  Floor = $null;   Want = "cu126"; Loud = $true },
    @{ N = "presence-only on a pre-R450 driver"; Exe = $null; Inv = $null; Pres = $true; Floor = $null; Release = 445; Want = "cpu"; Loud = $true },
    @{ N = "presence-only on R450 exactly";      Exe = $null; Inv = $null; Pres = $true; Floor = @(11,0); Release = 450; Want = "cu118"; Loud = $true },
    # Unknown is NOT too old.
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

$script:NvidiaPresenceCudaFloor = @(11,0)
$script:Said = ""
Check "the cap lowers and never raises" ((("" + (Get-TorchIndexUrl)) -replace '^.*/', '') -eq "cu118")

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

$env:UNSLOTH_TORCH_INDEX_FAMILY = "cu118"
try {
    $script:NvidiaPresenceOnly = $true
    $script:NvidiaPresenceCudaFloor = @(13, 0)
    Check "an explicit family override still wins over the bus floor" (
        (("" + (Get-TorchIndexUrl)) -replace '^.*/', '') -eq "cu118")
} finally { Remove-Item Env:UNSLOTH_TORCH_INDEX_FAMILY -ErrorAction SilentlyContinue }

# install_llama_prebuilt.py never sees the bus, so llama.cpp sites read $HasNvidiaDriverEvidence.
$setupAstErrors = $null
$setupAst = [System.Management.Automation.Language.Parser]::ParseFile($setupPs1, [ref]$null, [ref]$setupAstErrors)
Check "setup.ps1 parses" (-not $setupAstErrors)
$llamaStart = $setupAst.Extent.Text.IndexOf('$_arm64CudaOptOut = ')
$keepFn = $setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Get-GpuPrebuiltToKeepOverSourceBuild" }, $true)[0]
$nvReads = @($setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.VariableExpressionAst] -and $n.VariablePath.UserPath -eq "HasNvidiaSmi" }, $true))
$llamaReads = @($nvReads | Where-Object {
    $_.Extent.StartOffset -gt $llamaStart -or
    ($_.Extent.StartOffset -ge $keepFn.Extent.StartOffset -and $_.Extent.EndOffset -le $keepFn.Extent.EndOffset) })
Check "the llama.cpp section exists where this test looks for it" ($llamaStart -gt 0)
Check "no llama.cpp or CUDA-toolkit site reads the promoted `$HasNvidiaSmi" ($llamaReads.Count -eq 0)
foreach ($needle in @(
    '$_nvidiaEvidence = $HasNvidiaDriverEvidence -or',
    'if ($HasNvidiaDriverEvidence) { Resolve-CudaToolkit -RequireOrExit }',
    'if ($HasNvidiaDriverEvidence -and -not $cachedCuda) {',
    'if ($HasNvidiaDriverEvidence -and $NvccPath) {',
    '$nvidia = $HasNvidiaDriverEvidence')) {
    Check "setup.ps1 carries: $needle" ($setupAst.Extent.Text.Contains($needle))
}
$defAst = $setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and
    "$($n.Left)" -eq '$HasNvidiaDriverEvidence' }, $true)
Check "setup.ps1 defines `$HasNvidiaDriverEvidence exactly once" (@($defAst).Count -eq 1)
$defText = "$(@($defAst)[0].Extent.Text)"
Check "and defines it after the presence promotion" (
    @($defAst)[0].Extent.StartOffset -gt $setupAst.Extent.Text.IndexOf('$script:NvidiaPresenceOnly = $true'))
foreach ($case in @(
    @{ Smi = $true;  Only = $true;  Want = $false; N = "a presence-only promotion" },
    @{ Smi = $true;  Only = $false; Want = $true;  N = "nvidia-smi or the driver library" },
    @{ Smi = $false; Only = $false; Want = $false; N = "no NVIDIA at all" })) {
    $got = & { param($smi, $only) $HasNvidiaSmi = $smi; $script:NvidiaPresenceOnly = $only
        . ([scriptblock]::Create($defText)); $HasNvidiaDriverEvidence } $case.Smi $case.Only
    Check "$($case.N) reads as driver evidence = $($case.Want)" ($got -eq $case.Want)
}
$script:NvidiaPresenceOnly = $false

# (1) One bounded scan per run: Start-Job calls are counted on the REAL function.
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $scanText = Get-FunctionText $file "Invoke-BoundedVideoControllerScan"
    $r = & {
        param($scanText, $mode)
        $script:StartJobs = 0
        function Start-Job { param($ScriptBlock) $script:StartJobs++; return [pscustomobject]@{ Id = 1 } }
        function Wait-Job { param($Job, $Timeout) if ($mode -eq 'hang') { return $null } else { return $Job } }
        function Receive-Job { param($Job, $ErrorAction)
            [pscustomobject]@{ Name = "NVIDIA GeForce RTX 4090"; PNPDeviceID = "PCI\VEN_10DE&DEV_2684"; ConfigManagerErrorCode = 0; DriverVersion = "32.0.15.6094" } }
        function Stop-Job { param($Job, $ErrorAction) }
        function Remove-Job { param($Job, [switch]$Force, $ErrorAction) }
        . ([scriptblock]::Create($scanText))
        $script:VideoControllerScanResult = $null
        $a = Invoke-BoundedVideoControllerScan
        $b = Invoke-BoundedVideoControllerScan
        $afterTwo = $script:StartJobs
        $script:VideoControllerScanResult = $null
        $c = Invoke-BoundedVideoControllerScan
        [pscustomobject]@{ A = $a; B = $b; C = $c; AfterTwo = $afterTwo; AfterReset = $script:StartJobs }
    } $scanText 'hang'
    Check "${leaf}: a hung scan is waited out once, not twice" ($r.AfterTwo -eq 1)
    Check "${leaf}: and the reused answer is the hang answer (Ok false)" ($r.A.Ok -eq $false -and $r.B.Ok -eq $false)
    Check "${leaf}: a reset starts a fresh scan (bites: the cache is per run, not forever)" ($r.AfterReset -eq 2)
    $r = & {
        param($scanText)
        $script:StartJobs = 0
        function Start-Job { param($ScriptBlock) $script:StartJobs++; return [pscustomobject]@{ Id = 1 } }
        function Wait-Job { param($Job, $Timeout) return $Job }
        function Receive-Job { param($Job, $ErrorAction)
            [pscustomobject]@{ Name = "NVIDIA GeForce RTX 4090"; PNPDeviceID = "PCI\VEN_10DE&DEV_2684"; ConfigManagerErrorCode = 0; DriverVersion = "32.0.15.6094" } }
        function Stop-Job { param($Job, $ErrorAction) }
        function Remove-Job { param($Job, [switch]$Force, $ErrorAction) }
        . ([scriptblock]::Create($scanText))
        $script:VideoControllerScanResult = $null
        $a = Invoke-BoundedVideoControllerScan
        $b = Invoke-BoundedVideoControllerScan
        [pscustomobject]@{ A = $a; B = $b; N = $script:StartJobs }
    } $scanText
    Check "${leaf}: an answered scan is reused with the same adapters" (
        $r.N -eq 1 -and $r.B.Ok -eq $true -and @($r.B.Names)[0] -eq "NVIDIA GeForce RTX 4090")
    $text = [System.IO.File]::ReadAllText($file)
    $reset = $text.IndexOf('$script:VideoControllerScanResult = $null')
    $firstCall = $text.IndexOf('$presenceScan = Invoke-BoundedVideoControllerScan')
    Check "$leaf resets the cached scan before the promotion takes it" ($reset -gt 0 -and $reset -lt $firstCall)
    Check "$leaf's Intel route still goes through the (now shared) scan" (
        $text.IndexOf('$_gpuScan = Invoke-BoundedVideoControllerScan', $firstCall) -gt $firstCall)
}

# (2) Basic Display or no version: no promotion, same as base.
foreach ($case in @(
    @{ N = "Basic Display 10.0.19041.3636";  V = "10.0.19041.3636"; Want = $true },
    @{ N = "Basic Display 10.0.22621.1";     V = "10.0.22621.1";    Want = $true },
    @{ N = "an NVIDIA driver 560.94";        V = "32.0.15.6094";    Want = $false },
    @{ N = "a pre-R450 NVIDIA driver 442.50"; V = "26.21.14.4250";  Want = $false },
    @{ N = "no DriverVersion at all";        V = $null;             Want = $true },
    @{ N = "an empty DriverVersion";         V = "";                Want = $true },
    @{ N = "a blank DriverVersion";          V = "  ";              Want = $true }
)) {
    $script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0 $case.V)
    Check "an NVIDIA adapter with $($case.N) reads as no-NVIDIA-driver = $($case.Want)" (
        (Test-NvidiaAdapterWithoutNvidiaDriver) -eq $case.Want)
}
$script:FakeAdapters = @(
    (Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_1C03" 0 "10.0.19041.3636"),
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0 "32.0.15.6094"))
Check "one adapter on the NVIDIA driver is enough to answer no" ((Test-NvidiaAdapterWithoutNvidiaDriver) -eq $false)
$script:FakeAdapters = @(
    (Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_1C03" 0 $null),
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0 "32.0.15.6094"))
Check "a versionless adapter next to one on the NVIDIA driver still answers no" ((Test-NvidiaAdapterWithoutNvidiaDriver) -eq $false)
$script:FakeAdapters = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 22 "10.0.19041.3636")
Check "a disabled Basic Display NVIDIA adapter is not counted either way" ((Test-NvidiaAdapterWithoutNvidiaDriver) -eq $false)
$script:FakeAdapters = @(Adapter "Microsoft Basic Display Adapter" "ROOT\BASICDISPLAY" 0 "10.0.19041.3636")
Check "a non-NVIDIA Basic Display adapter is not an NVIDIA card without its driver" ((Test-NvidiaAdapterWithoutNvidiaDriver) -eq $false)

function Get-PromotionBlock($file) {
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$null, [ref]$errors)
    $blocks = @($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.IfStatementAst] -and
        $n.Extent.Text -match '^if \(-not \$HasNvidiaSmi\) \{\s*\$presenceScan = Invoke-BoundedVideoControllerScan' }, $true))
    return $blocks[0].Extent.Text
}
function Get-NvidiaAdapterCudaFloor { param($Scan) return $null }
Invoke-Expression (Get-FunctionText $setupPs1 "Test-IntelXpuRuntimeProven")
$script:PromoXpu = "False"
function Invoke-BoundedPythonProbe { param($PythonExe, $Code) return [pscustomobject]@{ Ok = $true; Output = $script:PromoXpu } }
$promoVenv = Join-Path ([System.IO.Path]::GetTempPath()) ("promo-venv-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $promoVenv | Out-Null
$promoPy = Join-Path $promoVenv "Scripts\python.exe"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $promoPy) | Out-Null
Set-Content -LiteralPath $promoPy -Value "" -NoNewline
$promoTorch = Join-Path $promoVenv "Lib\site-packages\torch"
New-Item -ItemType Directory -Force -Path $promoTorch | Out-Null
Set-Content -LiteralPath (Join-Path $promoTorch "version.py") -Value "__version__ = '2.8.0+xpu'"
Invoke-Expression (Get-FunctionText $setupPs1 "Test-VenvTorchIsXpu")
$script:FakeProbableVenv = $null
function Get-ProbableStudioVenvDir { return $script:FakeProbableVenv }
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $promo = Get-PromotionBlock $file
    Check "$leaf's promotion block was found (bites)" ([bool]$promo)
    foreach ($case in @(
        @{ N = "Basic Display (10.0.19041.3636)"; A = @(Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_2684" 0 "10.0.19041.3636"); Want = $false; Hint = $true },
        @{ N = "an NVIDIA driver (560.94)";        A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0 "32.0.15.6094");   Want = $true;  Hint = $false },
        @{ N = "no DriverVersion";                 A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0 $null);            Want = $false; Hint = $true },
        @{ N = "Basic Display with no DriverVersion"; A = @(Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_2684" 0 $null); Want = $false; Hint = $true },
        @{ N = "Basic Display with an empty DriverVersion"; A = @(Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_2684" 0 ""); Want = $false; Hint = $true },
        @{ N = "Basic Display plus Intel UHD";     A = @((Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_2684" 0 "10.0.19041.3636"), (Adapter "Intel(R) UHD Graphics" "PCI\VEN_8086&DEV_9A49" 0 "31.0.101.1")); Want = $false; Hint = $true },
        @{ N = "an NVIDIA driver plus a Meteor Lake GPU whose PyTorch has proven XPU"; A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" 0 "32.0.101.6078")); Want = $false; Hint = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver plus a Meteor Lake GPU with no XPU runtime";            A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" 0 "32.0.101.6078")); Want = $true;  Hint = $false; Xpu = "False" },
        # Base's Intel route ignored PNP ID and status, so these must still decline.
        @{ N = "an NVIDIA driver plus a Meteor Lake with no status, PyTorch XPU proven"; A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" $null "32.0.101.6078")); Want = $false; Hint = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver plus a Meteor Lake with no PNP ID, PyTorch XPU proven"; A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Graphics" $null 0 "32.0.101.6078")); Want = $false; Hint = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver and no Intel row at all, PyTorch XPU proven";           A = @(Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"); Want = $false; Hint = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver plus an Arc with no status, no XPU runtime";            A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Arc(TM) A770 Graphics" "PCI\VEN_8086&DEV_56A0" $null "32.0.101.6078")); Want = $false; Hint = $false; Xpu = "False" },
        @{ N = "an NVIDIA driver plus an Arc with no PNP ID, no XPU runtime";            A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Arc(TM) A770 Graphics" $null 0 "32.0.101.6078")); Want = $false; Hint = $false; Xpu = "False" },
        @{ N = "an NVIDIA driver plus a disabled Arc, no XPU runtime";                   A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Arc(TM) A770 Graphics" "PCI\VEN_8086&DEV_56A0" 22 "32.0.101.6078")); Want = $false; Hint = $false; Xpu = "False" },
        @{ N = "an NVIDIA driver plus an AMD adapter with no status";                    A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "AMD Radeon(TM) Graphics" "PCI\VEN_1002&DEV_15BF" $null "31.0.21001.45002")); Want = $false; Hint = $false; Xpu = "False" },
        @{ N = "an NVIDIA driver plus a UHD with no status, no XPU runtime";             A = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) UHD Graphics 770" "PCI\VEN_8086&DEV_4680" $null "31.0.101.1")); Want = $true; Hint = $false; Xpu = "False" },
        @{ N = "an unreadable status";                                                   A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" "bogus" "32.0.15.6094"); Want = $false; Hint = $false }
    )) {
        $script:PromoXpu = if ($case.Xpu) { $case.Xpu } else { "False" }
        $got = try { & { param($promo, $adapters)
            $VenvDir = $promoVenv; $VenvPython = $promoPy
            $script:StudioPreservedXpuVerdict = $false; $script:StudioVenvRollbackDir = $null
            $script:FakeAdapters = $adapters; $script:FakeScanOk = $null
            $script:Lines = @()
            function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") $script:Lines += $Line }
            $HasNvidiaSmi = $false
            $script:NvidiaPresenceOnly = $false; $script:NvidiaSmiRejected = $false
            . ([scriptblock]::Create($promo))
            [pscustomobject]@{ Has = $HasNvidiaSmi; Only = $script:NvidiaPresenceOnly; Lines = @($script:Lines) }
        } $promo $case.A } catch { [pscustomobject]@{ Has = "threw: $($_.Exception.Message)"; Only = $null; Lines = @() } }
        Check "$leaf, NVIDIA on $($case.N): promoted = $($case.Want)" ($got.Has -eq $case.Want -and $got.Only -eq $case.Want)
        $hinted = [bool](@($got.Lines) -match 'without the NVIDIA driver')
        Check "$leaf, NVIDIA on $($case.N): driver hint printed = $($case.Hint)" ($hinted -eq $case.Hint)
    }
}
# Base asked Get-ProbableStudioVenvDir, which is not $VenvDir under a stage root.
$emptyVenv = Join-Path ([System.IO.Path]::GetTempPath()) ("promo-empty-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $emptyVenv | Out-Null
$script:PromoXpu = "True"
$script:FakeProbableVenv = $promoVenv
$got = try { & { param($promo)
    $VenvDir = $emptyVenv; $VenvPython = Join-Path $emptyVenv "Scripts\python.exe"
    $script:FakeAdapters = @((Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094"), (Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" 0 "32.0.101.6078"))
    $script:FakeScanOk = $null
    function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }
    $HasNvidiaSmi = $false; $script:NvidiaPresenceOnly = $false; $script:NvidiaSmiRejected = $false
    . ([scriptblock]::Create($promo))
    $HasNvidiaSmi
} (Get-PromotionBlock $setupPs1) } catch { "threw: $($_.Exception.Message)" }
Check "setup.ps1 declines where the studio-home environment has proven XPU and the stage root has none" ($got -eq $false)
$script:FakeProbableVenv = $null
$script:PromoXpu = "False"
Remove-Item -LiteralPath $emptyVenv -Recurse -Force -ErrorAction SilentlyContinue

# A terminating Remove-Job error must not escape the scan.
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $scanText = Get-FunctionText $file "Invoke-BoundedVideoControllerScan"
    $r = try { & {
        param($scanText)
        function Start-Job { param($ScriptBlock) return [pscustomobject]@{ Id = 1 } }
        function Wait-Job { param($Job, $Timeout) return $Job }
        function Receive-Job { param($Job, $ErrorAction)
            [pscustomobject]@{ Name = "NVIDIA GeForce RTX 4090"; PNPDeviceID = "PCI\VEN_10DE&DEV_2684"; ConfigManagerErrorCode = 0; DriverVersion = "32.0.15.6094" } }
        function Stop-Job { param($Job, $ErrorAction) }
        function Remove-Job { param($Job, [switch]$Force, $ErrorAction) throw "job cleanup failed" }
        . ([scriptblock]::Create($scanText))
        $script:VideoControllerScanResult = $null
        Invoke-BoundedVideoControllerScan
    } $scanText } catch { "threw: $($_.Exception.Message)" }
    Check "${leaf}: a throwing job cleanup does not escape the scan" ($r -isnot [string])
    Check "${leaf}: and the answer it already had is kept" ($r -isnot [string] -and $r.Ok -eq $true)
}
$script:VideoControllerScanResult = $null

$script:NvidiaPresenceOnly = $false
Remove-Item -LiteralPath $promoVenv -Recurse -Force -ErrorAction SilentlyContinue

# (3) A localized Arc is reconciled through the registry, as the Intel route does.
$script:FakeRegistryNames = @()
# "Intel" in katakana, by code point so the file stays ASCII.
$jp = -join [char[]](0x30A4, 0x30F3, 0x30C6, 0x30EB)
foreach ($case in @(
    @{ N = "a Japanese Arc reconciled by the registry"; W = "$jp(R) Arc(TM) A770 Graphics";
       R = @("Intel $jp(R) Arc(TM) A770 Graphics"); Blocks = $true },
    @{ N = "an OEM-branded Arc reconciled by the registry"; W = "OEM Graphics A750";
       R = @("Intel(R) UHD Graphics 770", "Intel OEM Graphics A750 (Arc)"); Blocks = $true },
    @{ N = "a localized UHD whose registry entry is also UHD"; W = "$jp(R) UHD Graphics 770";
       R = @("Intel $jp(R) UHD Graphics 770"); Blocks = $false },
    @{ N = "a UHD next to a stale Arc key it does not name"; W = "Intel(R) UHD Graphics 770";
       R = @("Intel(R) UHD Graphics 770", "Intel(R) Arc(TM) A770 Graphics"); Blocks = $false },
    @{ N = "a localized Arc with no registry entry"; W = "$jp(R) Arc(TM) A770 Graphics"; R = @(); Blocks = $false }
)) {
    $script:FakeRegistryNames = $case.R
    $script:FakeAdapters = @(
        (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
        (Adapter $case.W "PCI\VEN_8086&DEV_56A0" 0))
    Check "NVIDIA plus $($case.N): the gate $(if ($case.Blocks) { 'blocks' } else { 'allows' }) promotion" (
        (Test-OtherVendorAdapterPresent) -eq $case.Blocks)
}
$script:FakeRegistryThrows = $true
$script:FakeRegistryNames = @("Intel $jp(R) Arc(TM) A770 Graphics")
$script:FakeAdapters = @(
    (Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" 0),
    (Adapter "$jp(R) Arc(TM) A770 Graphics" "PCI\VEN_8086&DEV_56A0" 0))
Check "an unreadable registry does not throw out of the gate and does not block" (
    (Test-OtherVendorAdapterPresent) -eq $false)
$script:FakeRegistryThrows = $false
$script:FakeRegistryNames = @()
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $gate = Get-FunctionText $file "Test-OtherVendorAdapterPresent"
    Check "$leaf's gate reconciles through Get-IntelRegistryAdapterNames" ($gate -match 'Get-IntelRegistryAdapterNames')
    $text = [System.IO.File]::ReadAllText($file)
    $def = $text.IndexOf("function Get-IntelRegistryAdapterNames")
    $promoAt = $text.IndexOf('$presenceScan = Invoke-BoundedVideoControllerScan')
    Check "$leaf defines Get-IntelRegistryAdapterNames before the promotion that now needs it" (
        $def -gt 0 -and $def -lt $promoAt)
    Check "$leaf defines it exactly once" (([regex]::Matches($text, 'function Get-IntelRegistryAdapterNames')).Count -eq 1)
}

Invoke-Expression (Get-FunctionText $setupPs1 "Test-IntelXpuRuntimeProven")
$script:ProbeCalls = 0
$script:ProbeAnswer = @{ Ok = $true; Output = "True" }
$script:ProbeThrows = $false
function Invoke-BoundedPythonProbe {
    param($PythonExe, $Code)
    $script:ProbeCalls++
    if ($script:ProbeThrows) { throw "probe failed" }
    return [pscustomobject]$script:ProbeAnswer
}
$fakePy = Join-Path ([System.IO.Path]::GetTempPath()) ("xpu-py-" + [guid]::NewGuid().ToString("N"))
Set-Content -LiteralPath $fakePy -Value "" -NoNewline
$mtl = Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" 0
$rtx = Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0
$scanOf = { param($list) [pscustomobject]@{ Ok = $true; Adapters = @($list); Names = @($list | ForEach-Object { $_.Name }) } }
$script:ProbeCalls = 0
Check "a Meteor Lake GPU whose PyTorch reports XPU is proven" (
    (Test-IntelXpuRuntimeProven -Scan (& $scanOf @($mtl, $rtx)) -PythonExe $fakePy) -eq $true)
$script:ProbeAnswer = @{ Ok = $true; Output = "False" }
Check "an XPU runtime that answers False is not proven" (
    (Test-IntelXpuRuntimeProven -Scan (& $scanOf @($mtl, $rtx)) -PythonExe $fakePy) -eq $false)
$script:ProbeAnswer = @{ Ok = $false; Output = "" }
Check "a probe that times out is not proven" (
    (Test-IntelXpuRuntimeProven -Scan (& $scanOf @($mtl, $rtx)) -PythonExe $fakePy) -eq $false)
$script:ProbeThrows = $true
Check "a probe that throws is not proven and does not throw out" (
    (Test-IntelXpuRuntimeProven -Scan (& $scanOf @($mtl, $rtx)) -PythonExe $fakePy) -eq $false)
$script:ProbeThrows = $false
$script:ProbeAnswer = @{ Ok = $true; Output = "True" }
Check "no environment to ask is not proven" (
    (Test-IntelXpuRuntimeProven -Scan (& $scanOf @($mtl, $rtx)) -PythonExe (Join-Path $fakePy "missing")) -eq $false)
foreach ($case in @(
    @{ N = "no Intel row at all";       A = @($rtx) },
    @{ N = "an Intel row with no status"; A = @((Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" $null), $rtx) },
    @{ N = "an Intel row with no PNP ID"; A = @((Adapter "Intel(R) Graphics" $null 0), $rtx) },
    @{ N = "a scan that did not answer"; A = @() }
)) {
    $script:ProbeCalls = 0
    $scan = & $scanOf $case.A
    if ($case.A.Count -eq 0) { $scan.Ok = $false }
    Check "a proven XPU runtime is proven with $($case.N)" (
        (Test-IntelXpuRuntimeProven -Scan $scan -PythonExe $fakePy) -eq $true -and $script:ProbeCalls -eq 1)
}
Remove-Item -LiteralPath $fakePy -Force -ErrorAction SilentlyContinue
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $text = [System.IO.File]::ReadAllText($file)
    $promo = $text.Substring($text.IndexOf('$presenceScan = Invoke-BoundedVideoControllerScan'))
    $promo = $promo.Substring(0, $promo.IndexOf('NvidiaPresenceOnly = $true'))
    Check "$leaf declines the promotion where PyTorch has proven XPU" (
        $promo -match 'Test-IntelXpuRuntimeProven -Scan \$presenceScan')
}
$installText = [System.IO.File]::ReadAllText($installPs1)
$installPromo = $installText.Substring($installText.IndexOf('$presenceScan = Invoke-BoundedVideoControllerScan'))
$installPromo = $installPromo.Substring(0, $installPromo.IndexOf('NvidiaPresenceOnly = $true'))
Check "install.ps1 declines it on a preserved XPU verdict too" ($installPromo -match '-not \$script:StudioPreservedXpuVerdict')

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
