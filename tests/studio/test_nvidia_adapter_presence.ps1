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
$installText = [System.IO.File]::ReadAllText($installPs1)
$setupText = [System.IO.File]::ReadAllText($setupPs1)
$texts = @{ $installPs1 = $installText; $setupPs1 = $setupText }

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}
# Passes when $text is non-empty, matches every $re and none of $not.
function CheckText($name, $text, [string[]]$re = @(), [string[]]$not = @()) {
    $ok = -not [string]::IsNullOrEmpty($text)
    foreach ($r in $re) { $ok = $ok -and ($text -match $r) }
    foreach ($r in $not) { $ok = $ok -and ($text -notmatch $r) }
    Check $name $ok
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
function Get-PromotionBlock($file) {
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($file, [ref]$null, [ref]$null)
    return @($ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.IfStatementAst] -and
        $n.Extent.Text -match '^if \(-not \$HasNvidiaSmi\) \{\s*\$presenceScan = Invoke-BoundedVideoControllerScan' }, $true))[0].Extent.Text
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
    if ($name -ne "Invoke-BoundedVideoControllerScan") { Invoke-Expression (Get-FunctionText $setupPs1 $name) }
}

$script:FakeRegistryNames = @()
$script:FakeRegistryThrows = $false
function Get-IntelRegistryAdapterNames {
    if ($script:FakeRegistryThrows) { throw "registry unreadable" }
    return $script:FakeRegistryNames
}
$script:FakeAdapters = @()
$script:FakeScanOk = $null   # $null derives Ok from the fixture
function Invoke-BoundedVideoControllerScan {
    param([int]$TimeoutSec = 15)
    return [pscustomobject]@{
        Ok = $(if ($null -eq $script:FakeScanOk) { $script:FakeAdapters.Count -gt 0 } else { $script:FakeScanOk })
        Names = @($script:FakeAdapters | ForEach-Object { $_.Name })
        Adapters = $script:FakeAdapters
    }
}
# No HKLM on Linux: the class keys are stubbed from $script:RegKeys so rows cannot pass for the wrong reason.
$script:RegKeys = @()
$script:RegistryConsulted = $false
function Get-ChildItem {
    param([string]$LiteralPath, [switch]$Directory, [string]$Filter, $ErrorAction)
    if ("$LiteralPath" -notmatch 'Control\\Class') { return @() }
    $script:RegistryConsulted = $true
    return @(for ($i = 0; $i -lt @($script:RegKeys).Count; $i++) {
        [pscustomobject]@{ PSChildName = "{0:D4}" -f $i; PSPath = "fake::{0:D4}" -f $i } })
}
function Get-ItemProperty {
    param([string]$LiteralPath, $ErrorAction)
    if ("$LiteralPath" -match '^fake::(\d{4})$') { return [pscustomobject]@($script:RegKeys)[[int]$Matches[1]] }
    return $null
}
$script:ProbeCalls = 0
$script:ProbeThrows = $false
$script:ProbeAnswer = @{ Ok = $true; Output = "True" }
function Invoke-BoundedPythonProbe {
    param($PythonExe, $Code, $TimeoutSec)
    $script:ProbeCalls++
    if ($script:ProbeThrows) { throw "probe failed" }
    return [pscustomobject]$script:ProbeAnswer
}
function Adapter($name, $id, $code, $driver = "32.0.15.6094") {
    return [pscustomobject]@{ Name = $name; PNPDeviceID = $id; ConfigManagerErrorCode = $code; DriverVersion = $driver }
}
function NvAdapter($code = 0, $driver = "32.0.15.6094") { Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684" $code $driver }
$nv = NvAdapter
$amd = Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 0
$basic = Adapter "Microsoft Basic Display Adapter" "ROOT\BASICDISPLAY" 0
# "Intel" in katakana, by code point so the file stays ASCII.
$jp = -join [char[]](0x30A4, 0x30F3, 0x30C6, 0x30EB)

# Names are the trap: only the vendor ID and a healthy status count.
foreach ($case in @(
    @{ N = "nothing on the bus";               A = @(); P = $false },
    @{ N = "a healthy NVIDIA adapter";         A = @(Adapter "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684&SUBSYS_00000000" 0); P = $true },
    @{ N = "an AMD-only machine";              A = @($amd); P = $false },
    @{ N = "an Intel-only machine";            A = @(Adapter "Intel Arc A770" "PCI\VEN_8086&DEV_56A0" 0); P = $false },
    @{ N = "a hybrid AMD and Intel machine";   A = @((Adapter "Intel UHD Graphics" "PCI\VEN_8086&DEV_9A60" 0), (Adapter "AMD Radeon 780M" "PCI\VEN_1002&DEV_15BF" 0)); P = $false },
    @{ N = "a virtual adapter";                A = @($basic); P = $false },
    @{ N = "a non-NVIDIA card whose NAME says NVIDIA"; A = @(Adapter "NVIDIA compatible display" "PCI\VEN_1002&DEV_744C" 0); P = $false },
    @{ N = "an NVIDIA card with an unrecognisable name"; A = @(Adapter "Carte graphique" "PCI\VEN_10DE&DEV_2684" 0); P = $true },
    @{ N = "a disabled adapter (code 22)";     A = @(NvAdapter 22); P = $false },
    @{ N = "a device-problem adapter (code 43)"; A = @(NvAdapter 43); P = $false },
    @{ N = "a driver record with no status";   A = @(NvAdapter $null); P = $false },
    @{ N = "a healthy card after a faulted one"; A = @((Adapter "NVIDIA GeForce GTX 1060" "PCI\VEN_10DE&DEV_1C03" 43), $nv); P = $true },
    @{ N = "a lower-case vendor ID";           A = @(Adapter "NVIDIA GeForce RTX 4090" "pci\ven_10de&dev_2684" 0); P = $true },
    @{ N = "a lookalike vendor ID";            A = @(Adapter "Some Card" "PCI\VEN_110D&DEV_E10D" 0); P = $false }
)) {
    $script:FakeAdapters = $case.A
    Check "NVIDIA presence on $($case.N) is $($case.P)" ((Test-NvidiaAdapterPresent) -eq $case.P)
}

foreach ($case in @(
    @{ N = "a healthy AMD adapter";   A = @($amd); Want = $true },
    @{ N = "a healthy Intel adapter"; A = @(Adapter "Intel Arc A770" "PCI\VEN_8086&DEV_56A0" 0); Want = $true },
    # The AMD route keeps faulted, parked and PNP-less Radeons, so each must veto the promotion.
    @{ N = "a faulted AMD adapter";   A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 43); Want = $true },
    @{ N = "a parked AMD adapter";    A = @(Adapter "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C" 45); Want = $true },
    @{ N = "an AMD adapter with no PNP ID"; A = @(Adapter "AMD Radeon RX 7900 XTX" "" 0); Want = $true },
    @{ N = "a faulted Intel UHD adapter"; A = @(Adapter "Intel(R) UHD Graphics 770" "PCI\VEN_8086&DEV_4680" 43); Want = $false },
    @{ N = "an NVIDIA adapter only";  A = @($nv); Want = $false },
    @{ N = "a virtual adapter only";  A = @($basic); Want = $false }
)) {
    $script:FakeAdapters = $case.A
    Check "other-vendor gate sees $($case.N) as $($case.Want)" ((Test-OtherVendorAdapterPresent) -eq $case.Want)
}

# NVIDIA plus one other adapter. UHD / Iris get no XPU wheels, so they must not block; a localized Arc is
# reconciled through the registry names (R), as the Intel route does.
$i4680 = "PCI\VEN_8086&DEV_4680"; $i56a0 = "PCI\VEN_8086&DEV_56A0"
foreach ($case in @(
    @{ N = "AMD Radeon 780M";         W = "AMD Radeon 780M";                   Id = "PCI\VEN_1002&DEV_15BF"; Blocks = $true },
    @{ N = "an integrated AMD Radeon(TM) Graphics"; W = "AMD Radeon(TM) Graphics"; Id = "PCI\VEN_1002&DEV_15BF"; Blocks = $true },
    @{ N = "Intel UHD Graphics";      W = "Intel(R) UHD Graphics 770";         Id = $i4680; Blocks = $false },
    @{ N = "Intel Iris Xe";           W = "Intel(R) Iris(R) Xe Graphics";      Id = $i4680; Blocks = $false },
    @{ N = "Intel HD Graphics";       W = "Intel(R) HD Graphics 630";          Id = $i4680; Blocks = $false },
    @{ N = "Intel Arc A770";          W = "Intel(R) Arc(TM) A770 Graphics";    Id = $i4680; Blocks = $true },
    @{ N = "Intel Arc B580";          W = "Intel(R) Arc(TM) B580 Graphics";    Id = $i4680; Blocks = $true },
    @{ N = "Intel Data Center GPU";   W = "Intel(R) Data Center GPU Max 1100"; Id = $i4680; Blocks = $true },
    @{ N = "a nameless Intel adapter"; W = "";                                 Id = $i4680; Blocks = $false },
    @{ N = "a Japanese Arc reconciled by the registry"; W = "$jp(R) Arc(TM) A770 Graphics"; Id = $i56a0;
       R = @("Intel $jp(R) Arc(TM) A770 Graphics"); Blocks = $true },
    @{ N = "an OEM-branded Arc reconciled by the registry"; W = "OEM Graphics A750"; Id = $i56a0;
       R = @("Intel(R) UHD Graphics 770", "Intel OEM Graphics A750 (Arc)"); Blocks = $true },
    @{ N = "a localized UHD whose registry entry is also UHD"; W = "$jp(R) UHD Graphics 770"; Id = $i56a0;
       R = @("Intel $jp(R) UHD Graphics 770"); Blocks = $false },
    @{ N = "a UHD next to a stale Arc key it does not name"; W = "Intel(R) UHD Graphics 770"; Id = $i56a0;
       R = @("Intel(R) UHD Graphics 770", "Intel(R) Arc(TM) A770 Graphics"); Blocks = $false },
    @{ N = "a localized Arc with no registry entry"; W = "$jp(R) Arc(TM) A770 Graphics"; Id = $i56a0; Blocks = $false },
    @{ N = "a localized Arc and an unreadable registry (no throw)"; W = "$jp(R) Arc(TM) A770 Graphics"; Id = $i56a0;
       R = @("Intel $jp(R) Arc(TM) A770 Graphics"); Throws = $true; Blocks = $false }
)) {
    $script:FakeRegistryNames = if ($case.R) { $case.R } else { @() }
    $script:FakeRegistryThrows = [bool]$case.Throws
    $script:FakeAdapters = @($nv, (Adapter $case.W $case.Id 0))
    Check "NVIDIA plus $($case.N): the gate $(if ($case.Blocks) { 'blocks' } else { 'allows' }) promotion" (
        (Test-OtherVendorAdapterPresent) -eq $case.Blocks)
    Check "  and NVIDIA presence is unaffected either way" ((Test-NvidiaAdapterPresent) -eq $true)
}
$script:FakeRegistryThrows = $false; $script:FakeRegistryNames = @()

CheckText "the gate defers to the shared XPU-capable pattern" (Get-FunctionText $installPs1 "Test-OtherVendorAdapterPresent") 'Get-XpuCapableNameRegex'
# @() around the WHOLE pipeline: one unique string would make [0] its first character.
$patternDefs = @(@([regex]::Matches($installText, 'function Get-XpuCapableNameRegex \{ return "([^"]+)" \}') |
    ForEach-Object { $_.Groups[1].Value }) | Sort-Object -Unique)
$routeDefs = @(@([regex]::Matches($installText, '\$_xpuNameRe = "([^"]+)"') |
    ForEach-Object { $_.Groups[1].Value }) | Sort-Object -Unique)
Check "the extracted patterns are whole strings, not characters" ($patternDefs[0].Length -gt 5 -and $routeDefs[0].Length -gt 5)
Check "both patterns are defined exactly once" ($patternDefs.Count -eq 1 -and $routeDefs.Count -eq 1)
Check "the gate pattern and the XPU route pattern are identical" ($patternDefs[0] -ceq $routeDefs[0])

foreach ($case in @(
    @{ V = "32.0.15.6094"; Want = "12.6" },   # 560.94
    @{ V = "32.0.15.7020"; Want = "12.8" },   # 570.20
    @{ V = "32.0.15.8000"; Want = "13.0" },   # 580.00
    @{ V = "31.0.15.5222"; Want = "12.4" },   # 552.22 -> 550 floor
    @{ V = "31.0.15.3667"; Want = "12.2" },   # 536.67 -> 535 floor
    @{ V = "30.0.14.4568"; Want = "" },       # 445.68 is BELOW 450: no floor
    @{ V = "570.20";       Want = "12.8" },   # already NVIDIA-shaped, as nvidia-smi reports
    @{ V = "";             Want = "" },
    @{ V = "not a version"; Want = "" },
    @{ V = "junk";         Want = "" }
)) {
    Check "driver $($case.V) floors at '$($case.Want)'" ((@(Get-NvidiaDriverCudaFloor -DriverVersion $case.V) -join '.') -eq $case.Want)
}
foreach ($case in @(
    @{ V = "32.0.15.6094"; R = 560; N = "a Windows display-driver version" },
    @{ V = "30.0.14.4568"; R = 445; N = "a pre-R450 Windows version" },
    @{ V = "580.65.06";    R = 580; N = "an nvidia-smi style version" },
    @{ V = "445.68";       R = 445; N = "a pre-R450 nvidia-smi style version" },
    @{ V = "";             R = $null; N = "an empty version" },
    @{ V = "not a version"; R = $null; N = "an unparseable version" },
    @{ V = "junk";         R = $null; N = "another unparseable version" },
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
    Check "$($case.N) reads as release $($case.R)" ((Get-NvidiaDriverRelease -DriverVersion $case.V) -eq $case.R)
}
foreach ($case in @(
    @{ N = "a healthy NVIDIA adapter";  A = (NvAdapter 0 "32.0.15.7020"); Want = "12.8" },
    @{ N = "an AMD adapter";            A = (Adapter "AMD Radeon" "PCI\VEN_1002&DEV_744C" 0 "32.0.15.7020"); Want = "" },
    @{ N = "a faulted NVIDIA adapter";  A = (NvAdapter 43 "32.0.15.7020"); Want = "" }
)) {
    $script:FakeAdapters = @($case.A)
    Check "the CUDA floor read from $($case.N) is '$($case.Want)'" ((@(Get-NvidiaAdapterCudaFloor) -join '.') -eq $case.Want)
}

# Generated from studio/nvidia_probe.py so the tables cannot drift.
$probeText = [System.IO.File]::ReadAllText((Join-Path $root "studio/nvidia_probe.py"))
$tableMatch = [regex]::Match($probeText, '(?s)_DRIVER_MAJOR_CUDA\s*=\s*\((.*?)\)\s*\n\s*\n')
Check "the python floor table was found" ($tableMatch.Success)
$pyRows = @([regex]::Matches($tableMatch.Groups[1].Value, '\((\d+),\s*\((\d+),\s*(\d+)\)\)') |
    ForEach-Object { "$($_.Groups[1].Value):$($_.Groups[2].Value).$($_.Groups[3].Value)" })
Check "the python table has rows to compare (bites)" ($pyRows.Count -ge 5)
# Get-NvidiaDriverCudaFloor no longer holds the table; reading it would pass on an empty list.
$psRows = @([regex]::Matches((Get-FunctionText $setupPs1 "Get-NvidiaCudaFloorForRelease"), '@\((\d+),\s*(\d+),\s*(\d+)\)') |
    ForEach-Object { "$($_.Groups[1].Value):$($_.Groups[2].Value).$($_.Groups[3].Value)" })
Check "the PowerShell floor table matches nvidia_probe.py exactly" (($pyRows -join ",") -eq ($psRows -join ","))

foreach ($file in @($installPs1, $setupPs1)) {
    $text = $texts[$file]
    $name = Split-Path $file -Leaf
    $library = $text.IndexOf('if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory))')
    $presence = $text.IndexOf('Test-NvidiaAdapterPresent -Scan $presenceScan')
    $promoAt = $text.IndexOf('$presenceScan = Invoke-BoundedVideoControllerScan')
    $busAt = $text.IndexOf('if (-not $HasNvidiaSmi) {', $library)
    Check "$name promotes on the driver library before falling back to the bus" (
        $library -gt 0 -and $presence -gt 0 -and $library -lt $presence)
    Check "$name only reaches the bus when nothing else answered" ($busAt -gt 0 -and $busAt -lt $presence)
    CheckText "$name gates the promotion on no other vendor being present" $text '-not \(Test-OtherVendorAdapterPresent -Scan \$presenceScan\)'
    # PowerShell does not hoist, and the pwsh suites define functions in dependency order themselves.
    foreach ($fn in $shared) {
        $def = $text.IndexOf("function $fn")
        Check "$name defines $fn before the promotion that uses it" ($def -gt 0 -and $def -lt $presence)
    }
    $scanDef = $text.IndexOf("function Invoke-BoundedVideoControllerScan")
    $scanCalls = 0
    foreach ($m in [regex]::Matches($text, '(?<!function )Invoke-BoundedVideoControllerScan')) {
        $lineStart = $text.LastIndexOf("`n", [Math]::Max($m.Index - 1, 0)) + 1
        if ($text.Substring($lineStart, $m.Index - $lineStart).Contains("#")) { continue }
        $scanCalls++
        Check "$name calls Invoke-BoundedVideoControllerScan at offset $($m.Index) after its definition" ($m.Index -gt $scanDef)
    }
    Check "$name has real calls to the bounded scan to order (bites)" ($scanCalls -gt 0)
    # Release rows below set the release directly, so check the promotion site records it.
    CheckText "$name records the driver release at the promotion site" $text '\$_presenceRelease = Get-NvidiaAdapterDriverRelease -Scan \$presenceScan[\s\S]{0,800}?\$script:NvidiaPresenceDriverRelease = \$_presenceRelease'
    CheckText "$name records it beside the floor, from the same scan" $text 'Get-NvidiaAdapterCudaFloor -Scan \$presenceScan[\s\S]{0,400}?Get-NvidiaAdapterDriverRelease -Scan \$presenceScan'
    CheckText "$name initialises the release to null" $text '\$script:NvidiaPresenceDriverRelease = \$null'
    CheckText "$name never reads presence from the class keys" (Get-FunctionText $file "Test-NvidiaAdapterPresent") @() 'Get-NvidiaRegistryAdapter|Control\\Class'
    $gate = Get-FunctionText $file "Test-OtherVendorAdapterPresent"
    CheckText "$name gates the other-vendor fallback on a failed scan too" $gate 'if \(\$Scan\.Ok\) \{ return \$false \}'
    CheckText "$name asks the shared XPU pattern of the Intel route's registry names" $gate 'Get-IntelRegistryAdapterNames \| Where-Object \{ "\$_" -match \(Get-XpuCapableNameRegex\)'
    $reset = $text.IndexOf('$script:VideoControllerScanResult = $null')
    Check "$name resets the cached scan before the promotion takes it" ($reset -gt 0 -and $reset -lt $promoAt)
    Check "$name's Intel route still goes through the (now shared) scan" (
        $text.IndexOf('$_gpuScan = Invoke-BoundedVideoControllerScan', $promoAt) -gt $promoAt)
    $def = $text.IndexOf("function Get-IntelRegistryAdapterNames")
    Check "$name defines Get-IntelRegistryAdapterNames before the promotion that now needs it" ($def -gt 0 -and $def -lt $promoAt)
    Check "$name defines it exactly once" (([regex]::Matches($text, 'function Get-IntelRegistryAdapterNames')).Count -eq 1)
    $promo = $text.Substring($promoAt)
    $promo = $promo.Substring(0, $promo.IndexOf('NvidiaPresenceOnly = $true'))
    CheckText "$name declines the promotion where PyTorch has proven XPU" $promo 'Test-IntelXpuRuntimeProven -Scan \$presenceScan'
    if ($file -eq $installPs1) { CheckText "install.ps1 declines it on a preserved XPU verdict too" $promo '-not \$script:StudioPreservedXpuVerdict' }
}

# setup.ps1 must NOT guess: "" keys the do-not-wipe escape (#9255); install.ps1 is where floor and release are consumed.
$setupTag = Get-FunctionText $setupPs1 "Get-PytorchCudaTag"
$installIdx = Get-FunctionText $installPs1 "Get-TorchIndexUrl"
CheckText "setup.ps1 still returns empty for unknown rather than flooring" $setupTag @() 'NvidiaPresenceCudaFloor'
CheckText "setup.ps1's tag helper does not act on the release either" $setupTag @() 'NvidiaPresenceDriverRelease'
CheckText "install.ps1 is where the floor and the release are consumed" $installIdx @('NvidiaPresenceCudaFloor', 'NvidiaPresenceDriverRelease')
CheckText "install.ps1 answers pre-R450 the same way (CPU)" $installIdx '(?s)NvidiaPresenceDriverRelease -lt 450[\s\S]{0,1200}?/cpu'
CheckText "install.ps1 has no rejection latch to set (bites)" $installText @() 'NvidiaSmiRejected'

$busBlock = [regex]::Match($setupText, '(?s)if \(-not \$HasNvidiaSmi\) \{\s*\$presenceScan = Invoke-BoundedVideoControllerScan.*?\n\}').Value
CheckText "setup.ps1's bus promotion marks nvidia-smi rejected" $busBlock '\$script:NvidiaSmiRejected = \$true'
CheckText "and still records presence-only separately" $busBlock '\$script:NvidiaPresenceOnly = \$true'
CheckText "the driver-library promotion already did, which is the precedent" $setupText '(?s)Get-NvidiaLibraryInventory\)\) \{[\s\S]{0,600}?\$script:NvidiaSmiRejected = \$true'

# Anchored on ORDER: an arm after the cu126 fallback is dead code.
$arm = [regex]::Match($setupText, '(?s)\} elseif \(\$HasNvidiaSmi\) \{.*?\n\} elseif \(\$script:IsIntelXpu\)').Value
CheckText "setup.ps1 routes a pre-R450 presence-only host to CPU" $arm @('\$script:NvidiaPresenceDriverRelease -lt 450', '(?s)-lt 450\)? \{[^}]*\$CuTag = "cpu"')
$preR450 = $arm.IndexOf('NvidiaPresenceDriverRelease'); $cu126 = $arm.IndexOf('"cu126"')
Check "and it decides before the cu126 fallback rather than after it" ($preR450 -ge 0 -and $cu126 -ge 0 -and $preR450 -lt $cu126)
CheckText "and says why, rather than silently installing CPU wheels" $arm 'predates R450'
CheckText "and marks the demotion so the CPU arm really replaces the CUDA wheel" $arm '(?s)-lt 450\)? \{[^}]*NvidiaPreR450CpuFallback = \$true'
CheckText "and only when a CUDA wheel is actually installed" $arm 'Test-CudaFamilyLeaf \$installedTorchTag\) \{ \$script:NvidiaPreR450CpuFallback = \$true \}'
CheckText "setup.ps1 selects from the recorded floor" $arm '\$script:NvidiaPresenceCudaFloor'
CheckText "and only when no CUDA family is already installed" $arm '-not \(Test-CudaFamilyLeaf \$installedTorchTag\)[\s\S]{0,80}?NvidiaPresenceCudaFloor'
CheckText "through the shared ladder rather than by formatting the digits" $arm 'Get-CudaFamilyForVersion -Major \$floorMajor' 'cu\$floorMajor'
CheckText "and applies the same pre-Turing cap" $arm '\$floorMajor -gt 12 -or \(\$floorMajor -eq 12 -and \$floorMinor -gt 6\)'
Check "and the pre-R450 arm still decides first" ($preR450 -ge 0 -and $preR450 -lt $arm.IndexOf('NvidiaPresenceCudaFloor'))
CheckText "the flag starts false on every run" $setupText '(?m)^\$script:NvidiaPreR450CpuFallback = \$false$'

$escapes = Get-FunctionText $setupPs1 "Invoke-FastPathEscapes"
CheckText "a bus-only NVIDIA host with CPU torch forces the dependency pass" $escapes @('\$script:NvidiaPresenceOnly', '(?s)NvidiaPresenceOnly[^}]*\$SkipPythonDeps = \$false')
CheckText "and only when the installed wheel is a CPU one" $escapes '\$installedTorchTag -eq "cpu"'
CheckText "a pre-R450 driver does not trigger it" $escapes @('\$_nvidiaPreR450', 'NvidiaPresenceDriverRelease -lt 450')
CheckText "nor does a pin that sends this host somewhere else" $escapes @('\$_nvidiaIsReachable', 'Test-CudaFamilyLeaf \$_pinLeafNow')
$cudaArm = [regex]::Match($setupText, '(?s)substep "installing PyTorch with CUDA support.{0,1500}').Value
CheckText "the CUDA arm replaces an installed CPU wheel on a bus-only host" $cudaArm '(?s)\$script:NvidiaPresenceOnly -and \$installedTorchTag -eq "cpu"[^}]*--force-reinstall'
CheckText "the CUDA arm forces the reinstall off a stale +rocm/+xpu wheel" $cudaArm '(?s)if \(Test-NvidiaPresenceStaleGpuWheel [^{]*\) \{\s*\$cudaForce = @\("--force-reinstall"\)'
$cpuArm = [regex]::Match($setupText, '(?s)substep "installing PyTorch \(CPU-only\)\.\.\.".{0,3000}').Value
CheckText "the CPU arm force-reinstalls after a pre-R450 demotion" $cpuArm 'if \(\$script:NvidiaPreR450CpuFallback\) \{ \$cpuForce = @\("--force-reinstall"\) \}'
CheckText "the same way it does after the ROCm and XPU fallbacks (bites)" $cpuArm 'if \(\$ROCmCpuFallback\) \{ \$cpuForce = @\("--force-reinstall"\) \}'
$preArm = [regex]::Match($setupText, '(?s)NvidiaPresenceDriverRelease -lt 450\) \{.{0,2500}?predates R450').Value
CheckText "a pre-R450 demotion replaces a stale +rocm/+xpu wheel too" $preArm '(?s)if \(Test-NvidiaPresenceStaleGpuWheel [^{]*\) \{\s*\$script:NvidiaPreR450CpuFallback = \$true'
CheckText "the cached answer starts empty on every run" $setupText '(?m)^\$script:NvidiaPresenceStaleGpuWheel = \$null$'

$staleHelper = Get-FunctionText $setupPs1 "Test-NvidiaPresenceStaleGpuWheel"
$escAst = [System.Management.Automation.Language.Parser]::ParseInput($escapes, [ref]$null, [ref]$null)
$staleEscape = @($escAst.FindAll({ param($n) $n -is [System.Management.Automation.Language.IfStatementAst] -and
    $n.Clauses[0].Item1.Extent.Text -match 'Test-NvidiaPresenceStaleGpuWheel' }, $true))[0]
Check "the fast-path escape asks Test-NvidiaPresenceStaleGpuWheel (bites)" ($null -ne $staleEscape)
if ($staleEscape) {
    foreach ($case in @(
        # tag, presence-only, probe answer ($null = no answer), pre-R450, want stale, want skip afterwards
        @{ T = "rocm";  P = $true;  A = "DEV=False"; R = $false; Stale = $true;  Skip = $false; N = "+rocm that sees no GPU" },
        @{ T = "xpu";   P = $true;  A = "DEV=False"; R = $false; Stale = $true;  Skip = $false; N = "+xpu that sees no GPU" },
        @{ T = "rocm";  P = $true;  A = "DEV=False"; R = $true;  Stale = $true;  Skip = $false; N = "+rocm on a pre-R450 driver (goes to CPU)" },
        @{ T = "xpu";   P = $true;  A = "DEV=True";  R = $false; Stale = $false; Skip = $true;  N = "+xpu that still sees its GPU (kept, as the base did)" },
        @{ T = "rocm";  P = $true;  A = "DEV=True";  R = $false; Stale = $false; Skip = $true;  N = "+rocm that still sees a GPU" },
        @{ T = "rocm";  P = $true;  A = $null;       R = $false; Stale = $false; Skip = $true;  N = "+rocm whose probe did not answer" },
        @{ T = "cu128"; P = $true;  A = "DEV=False"; R = $false; Stale = $false; Skip = $true;  N = "a CUDA wheel" },
        @{ T = "cpu";   P = $true;  A = "DEV=False"; R = $true;  Stale = $false; Skip = $true;  N = "a CPU wheel on pre-R450 (this escape)" },
        @{ T = "rocm";  P = $false; A = "DEV=False"; R = $false; Stale = $false; Skip = $true;  N = "+rocm on a host with no bus promotion" })) {
        $script:ProbeAnswer = if ($null -eq $case.A) { @{ Ok = $false; Output = ""; TimedOut = $true } }
                              else { @{ Ok = $true; Output = "$($case.A)`n"; TimedOut = $false } }
        $got = & {
            param($c)
            $script:NvidiaPresenceOnly = $c.P; $script:NvidiaPresenceStaleGpuWheel = $null; $script:ProbeCalls = 0
            function substep { param($a, $b) }
            . ([scriptblock]::Create($staleHelper))
            $VenvDir = "venv"; $installedTorchTag = $c.T
            $_nvidiaIsReachable = $true; $_nvidiaPreR450 = $c.R; $SkipPythonDeps = $true
            . ([scriptblock]::Create($staleEscape.Extent.Text))
            $again = Test-NvidiaPresenceStaleGpuWheel -InstalledTag $c.T -PythonExe "C:\venv\Scripts\python.exe"
            [pscustomobject]@{ Skip = $SkipPythonDeps; Stale = $again; Calls = $script:ProbeCalls }
        } $case
        Check "bus-only escape, $($case.N): dependency pass = $(-not $case.Skip)" ($got.Skip -eq $case.Skip)
        Check "bus-only helper, $($case.N): stale = $($case.Stale), probed at most once" ($got.Stale -eq $case.Stale -and $got.Calls -le 1)
    }
    $script:NvidiaPresenceOnly = $false; $script:NvidiaPresenceStaleGpuWheel = $null
}

$ladderFn = Get-FunctionText $setupPs1 "Get-CudaFamilyForVersion"
Check "setup.ps1 names the ladder once (bites)" ($ladderFn.Length -gt 50)
$installLadder = [regex]::Matches($installIdx, '(?m)^\s*(?:if|elseif)[^\r\n]*\$family = "cu\d+"[^\r\n]*$')
Check "install.ps1's inline ladder was found (bites)" ($installLadder.Count -ge 5)
$conds = { param($rows) @($rows | ForEach-Object {
    $m = [regex]::Match($_.Value, '\((?<c>[^)]*(?:\([^)]*\)[^)]*)*)\)\s*\{')
    if ($m.Success) { ((($m.Groups['c'].Value -replace '(?i)\$Major', '$major') -replace '(?i)\$Minor', '$minor') -replace '\s+', ' ').Trim() }
}) }
$setupConds = & $conds ([regex]::Matches($ladderFn, '(?m)^\s*(?:if|elseif|return)[^\r\n]*cu\d+[^\r\n]*$'))
$installConds = & $conds $installLadder
Check "both ladders have the same number of rungs" ($setupConds.Count -eq $installConds.Count)
Check "and the rungs test the same conditions in the same order" (($setupConds -join '|') -eq ($installConds -join '|'))
Invoke-Expression $ladderFn
foreach ($row in @(@(10, 2, "cpu"), @(11, 0, "cu118"), @(11, 8, "cu118"), @(12, 0, "cu124"), @(12, 5, "cu124"),
                   @(12, 6, "cu126"), @(12, 7, "cu126"), @(12, 8, "cu128"), @(12, 9, "cu128"), @(13, 0, "cu130"))) {
    Check "CUDA $($row[0]).$($row[1]) is served by $($row[2])" ((Get-CudaFamilyForVersion -Major $row[0] -Minor $row[1]) -eq $row[2])
}

# Presence never comes from the class keys; a failed scan may only borrow a unanimous driver release from them.
$nvKey = @{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1" }
$nv536 = @{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"; DriverVersion = "31.0.15.3699" }   # R536: floor 12.2
foreach ($case in @(
    # A = answered scan with these adapters; no A = a scan that could not answer.
    @{ N = "a failed scan with an NVIDIA class key"; K = @($nvKey); P = $false; Reg = $false },
    @{ N = "a scan that answered with no adapters at all"; K = @($nvKey); A = @(); P = $false; Reg = $false },
    @{ N = "a scan that answered with an AMD adapter only"; K = @($nvKey); A = @($amd); P = $false; Reg = $false },
    @{ N = "a scan that answered with a faulted NVIDIA adapter"; K = @($nvKey); A = @(NvAdapter 43); P = $false; Reg = $false },
    @{ N = "a scan that answered with a disabled NVIDIA adapter"; K = @($nvKey); A = @(NvAdapter 22); P = $false; Reg = $false },
    @{ N = "a healthy NVIDIA adapter from WMI alone"; K = @($nvKey); A = @($nv); P = $true; Reg = $false },
    @{ N = "a failed scan with only an AMD class key"; K = @(@{ MatchingDeviceId = "PCI\\VEN_1002&DEV_744C" }); P = $false },
    @{ N = "a failed scan with a versioned NVIDIA key"; K = @($nv536); P = $false; Rel = 536; Floor = "12.2" },
    @{ N = "an unversioned then a versioned NVIDIA key"; K = @($nvKey, $nv536); P = $false; Rel = 536 },
    @{ N = "two NVIDIA keys naming different releases"; K = @($nv536, @{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"; DriverVersion = "30.0.14.4568" });
       P = $false; Rel = $null; Floor = "" },
    @{ N = "two spellings of one release"; K = @($nv536, @{ MatchingDeviceId = "PCI\\VEN_10DE&DEV_1DB1"; DriverVersion = "536.99" }); Rel = 536 },
    @{ N = "two NVIDIA keys with no version"; K = @($nvKey, $nvKey); P = $false; Rel = $null },
    @{ N = "one NVIDIA key with no version"; K = @($nvKey); Rel = $null; Floor = "" },
    @{ N = "a scan that answered, beside a versioned key"; K = @($nv536); A = @(); Rel = $null; Reg = $false }
)) {
    $script:RegKeys = $case.K
    $script:FakeAdapters = if ($case.ContainsKey('A')) { $case.A } else { @() }
    $script:FakeScanOk = $case.ContainsKey('A')
    $script:RegistryConsulted = $false
    if ($case.ContainsKey('P')) { Check "registry: $($case.N) is presence = $($case.P)" ((Test-NvidiaAdapterPresent) -eq $case.P) }
    if ($case.ContainsKey('Rel')) { Check "registry: $($case.N) reads release '$($case.Rel)'" ("$(Get-NvidiaAdapterDriverRelease)" -eq "$($case.Rel)") }
    if ($case.ContainsKey('Floor')) { Check "registry: $($case.N) floors at '$($case.Floor)'" ((@(Get-NvidiaAdapterCudaFloor) -join '.') -eq $case.Floor) }
    if ($case.ContainsKey('Reg')) { Check "registry: $($case.N) consulted the registry = $($case.Reg)" ($script:RegistryConsulted -eq $case.Reg) }
}

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
    $script:RegKeys = @(@{ MatchingDeviceId = $case.Id; DriverDesc = $case.Desc })
    $script:FakeAdapters = @(); $script:FakeScanOk = $false; $script:RegistryConsulted = $false
    Check "registry fallback: $($case.N) reads as $(if ($case.Want) { 'an alternative' } else { 'no alternative' })" (
        (Test-OtherVendorAdapterPresent) -eq $case.Want)
    Check "  and the registry really was the source" ($script:RegistryConsulted -eq $true)
}
${function:Get-IntelRegistryAdapterNames} = $registryNamesStub
$script:RegKeys = @(@{ MatchingDeviceId = "PCI\\VEN_1002&DEV_744C"; DriverDesc = "AMD Radeon RX 7900 XTX" })
$script:FakeAdapters = @($nv); $script:FakeScanOk = $true; $script:RegistryConsulted = $false
Check "a scan that answered is not second-guessed by the registry" ((Test-OtherVendorAdapterPresent) -eq $false)
Check "and the registry was never consulted" ($script:RegistryConsulted -eq $false)
$script:RegKeys = @(); $script:FakeScanOk = $null

# Runs the real Get-TorchIndexUrl: promoting without a version used to still yield a CPU index.
$idxAst = [System.Management.Automation.Language.Parser]::ParseFile($installPs1, [ref]$null, [ref]$null)
foreach ($n in @("Get-TorchIndexUrl", "Trim-IndexPathSlashes", "Get-CudaFamilyCappedForPreTuring")) {
    $f = @($idxAst.FindAll({ param($x) $x -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $x.Name -eq $n }, $true))
    if ($f.Count -ge 1) { Invoke-Expression $f[0].Extent.Text }
}
function Get-NvidiaCu126Verdict { param($a, $b) return "" }
function Get-NvidiaLibraryInventory { return $script:Inv }
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return $script:Banner }
function substep { param($a, $b) $script:Said = "$a" }
$libInv = [pscustomobject]@{ CudaMajor = 12; CudaMinor = 8; ComputeCaps = @() }
foreach ($case in @(
    # Must stay SILENT: every AMD, Intel and CPU-only host reaches this.
    @{ N = "an AMD, Intel or CPU-only host";       Pres = $false; Want = "cpu"; Loud = $false },
    # No compute capability, so unknown stops at cu126 (pre-Turing). Unknown is NOT too old.
    @{ N = "presence-only with a 12.8 driver";     Pres = $true; Floor = @(12, 8); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with a 12.6 driver";     Pres = $true; Floor = @(12, 6); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with a 13.0 driver";     Pres = $true; Floor = @(13, 0); Want = "cu126"; Loud = $true },
    @{ N = "presence-only with an 11.0 driver";    Pres = $true; Floor = @(11, 0); Want = "cu118"; Loud = $true },
    @{ N = "presence-only with no or an unreadable driver version"; Pres = $true; Want = "cu126"; Loud = $true },
    @{ N = "presence-only on a pre-R450 driver";   Pres = $true; Release = 445; Want = "cpu"; Loud = $true },
    @{ N = "presence-only on R450 exactly";        Pres = $true; Floor = @(11, 0); Release = 450; Want = "cu118"; Loud = $true },
    @{ N = "the capped R580 row, which explains the cap and names the override"; Pres = $true; Floor = @(13, 0); Release = 580;
       Want = "cu126"; Loud = $true; Said = @('compute capability', 'UNSLOTH_TORCH_INDEX_FAMILY') },
    @{ N = "an R580 11.0 floor (the cap lowers and never raises)"; Pres = $true; Floor = @(11, 0); Release = 580; Want = "cu118" },
    @{ N = "unchanged: nvidia-smi with a readable banner"; Exe = "C:\it.exe"; Banner = "CUDA Version: 12.8"; Want = "cu128"; Loud = $false },
    @{ N = "unchanged: nvidia-smi with an unreadable banner"; Exe = "C:\it.exe"; Banner = "broken"; Want = "cu126"; Loud = $true },
    @{ N = "unchanged: the driver library answering"; Inv = $libInv; Want = "cu128"; Loud = $false },
    @{ N = "an explicit family override over the bus floor"; Inv = $libInv; Pres = $true; Floor = @(13, 0); Env = "cu118"; Want = "cu118" }
)) {
    $NvidiaSmiExe = $case.Exe; $script:Inv = $case.Inv; $script:Banner = "$($case.Banner)"
    $script:NvidiaPresenceOnly = [bool]$case.Pres
    $script:NvidiaPresenceCudaFloor = $case.Floor
    $script:NvidiaPresenceDriverRelease = $case.Release
    $script:Said = ""
    if ($case.Env) { $env:UNSLOTH_TORCH_INDEX_FAMILY = $case.Env }
    try { $leaf = ("" + (Get-TorchIndexUrl)) -replace '^.*/', '' }
    finally { if ($case.Env) { Remove-Item Env:UNSLOTH_TORCH_INDEX_FAMILY -ErrorAction SilentlyContinue } }
    Check "$($case.N) selects $($case.Want)" ($leaf -eq $case.Want)
    if ($case.ContainsKey('Loud')) {
        Check "and it $(if ($case.Loud) { 'says so' } else { 'stays silent' })" ([bool]$script:Said -eq $case.Loud)
    }
    foreach ($re in @($case.Said | Where-Object { $_ })) { Check "and the message matches '$re'" ($script:Said -match $re) }
}

# install_llama_prebuilt.py never sees the bus, so llama.cpp sites read $HasNvidiaDriverEvidence.
$setupAstErrors = $null
$setupAst = [System.Management.Automation.Language.Parser]::ParseFile($setupPs1, [ref]$null, [ref]$setupAstErrors)
Check "setup.ps1 parses" (-not $setupAstErrors)
$llamaStart = $setupText.IndexOf('$_arm64CudaOptOut = ')
$keepFn = $setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Get-GpuPrebuiltToKeepOverSourceBuild" }, $true)[0]
$llamaReads = @($setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.VariableExpressionAst] -and $n.VariablePath.UserPath -eq "HasNvidiaSmi" }, $true) |
    Where-Object { $_.Extent.StartOffset -gt $llamaStart -or
        ($_.Extent.StartOffset -ge $keepFn.Extent.StartOffset -and $_.Extent.EndOffset -le $keepFn.Extent.EndOffset) })
Check "the llama.cpp section exists where this test looks for it" ($llamaStart -gt 0)
Check "no llama.cpp or CUDA-toolkit site reads the promoted `$HasNvidiaSmi" ($llamaReads.Count -eq 0)
foreach ($needle in @(
    '$_nvidiaEvidence = $HasNvidiaDriverEvidence -or',
    'if ($HasNvidiaDriverEvidence) { Resolve-CudaToolkit -RequireOrExit }',
    'if ($HasNvidiaDriverEvidence -and -not $cachedCuda) {',
    'if ($HasNvidiaDriverEvidence -and $NvccPath) {',
    '$nvidia = $HasNvidiaDriverEvidence')) {
    Check "setup.ps1 carries: $needle" ($setupText.Contains($needle))
}
$defAst = @($setupAst.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.AssignmentStatementAst] -and "$($n.Left)" -eq '$HasNvidiaDriverEvidence' }, $true))
Check "setup.ps1 defines `$HasNvidiaDriverEvidence exactly once" ($defAst.Count -eq 1)
Check "and defines it after the presence promotion" ($defAst[0].Extent.StartOffset -gt $setupText.IndexOf('$script:NvidiaPresenceOnly = $true'))
foreach ($case in @(
    @{ Smi = $true;  Only = $true;  Want = $false; N = "a presence-only promotion" },
    @{ Smi = $true;  Only = $false; Want = $true;  N = "nvidia-smi or the driver library" },
    @{ Smi = $false; Only = $false; Want = $false; N = "no NVIDIA at all" })) {
    $got = & { param($smi, $only) $HasNvidiaSmi = $smi; $script:NvidiaPresenceOnly = $only
        . ([scriptblock]::Create("$($defAst[0].Extent.Text)")); $HasNvidiaDriverEvidence } $case.Smi $case.Only
    Check "$($case.N) reads as driver evidence = $($case.Want)" ($got -eq $case.Want)
}
$script:NvidiaPresenceOnly = $false

# One bounded scan per run (Start-Job counted on the REAL function), and a terminating Remove-Job must not escape it.
function Invoke-ScanHarness($scanText, $mode) {
    $script:StartJobs = 0
    function Start-Job { param($ScriptBlock) $script:StartJobs++; return [pscustomobject]@{ Id = 1 } }
    function Wait-Job { param($Job, $Timeout) if ($mode -eq 'hang') { return $null } else { return $Job } }
    function Receive-Job { param($Job, $ErrorAction) NvAdapter }
    function Stop-Job { param($Job, $ErrorAction) }
    function Remove-Job { param($Job, [switch]$Force, $ErrorAction) if ($mode -eq 'throw') { throw "job cleanup failed" } }
    . ([scriptblock]::Create($scanText))
    $script:VideoControllerScanResult = $null
    $a = Invoke-BoundedVideoControllerScan
    $b = Invoke-BoundedVideoControllerScan
    $afterTwo = $script:StartJobs
    $script:VideoControllerScanResult = $null
    $null = Invoke-BoundedVideoControllerScan
    return [pscustomobject]@{ A = $a; B = $b; AfterTwo = $afterTwo; AfterReset = $script:StartJobs }
}
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $scanText = Get-FunctionText $file "Invoke-BoundedVideoControllerScan"
    $r = Invoke-ScanHarness $scanText 'hang'
    Check "${leaf}: a hung scan is waited out once, not twice" ($r.AfterTwo -eq 1)
    Check "${leaf}: and the reused answer is the hang answer (Ok false)" ($r.A.Ok -eq $false -and $r.B.Ok -eq $false)
    Check "${leaf}: a reset starts a fresh scan (bites: the cache is per run, not forever)" ($r.AfterReset -eq 2)
    $r = Invoke-ScanHarness $scanText 'ok'
    Check "${leaf}: an answered scan is reused with the same adapters" (
        $r.AfterTwo -eq 1 -and $r.B.Ok -eq $true -and @($r.B.Names)[0] -eq "NVIDIA GeForce RTX 4090")
    $r = try { Invoke-ScanHarness $scanText 'throw' } catch { "threw: $($_.Exception.Message)" }
    Check "${leaf}: a throwing job cleanup does not escape the scan" ($r -isnot [string])
    Check "${leaf}: and the answer it already had is kept" ($r -isnot [string] -and $r.A.Ok -eq $true)
}
$script:VideoControllerScanResult = $null

# Basic Display or no version: no promotion, same as base.
foreach ($case in @(
    @{ N = "an NVIDIA adapter with Basic Display 10.0.19041.3636"; A = @(NvAdapter 0 "10.0.19041.3636"); Want = $true },
    @{ N = "an NVIDIA adapter with Basic Display 10.0.22621.1";    A = @(NvAdapter 0 "10.0.22621.1");    Want = $true },
    @{ N = "an NVIDIA adapter with an NVIDIA driver 560.94";       A = @(NvAdapter 0 "32.0.15.6094");    Want = $false },
    @{ N = "an NVIDIA adapter with a pre-R450 NVIDIA driver 442.50"; A = @(NvAdapter 0 "26.21.14.4250"); Want = $false },
    @{ N = "an NVIDIA adapter with no DriverVersion at all";       A = @(NvAdapter 0 $null);             Want = $true },
    @{ N = "an NVIDIA adapter with an empty DriverVersion";        A = @(NvAdapter 0 "");                Want = $true },
    @{ N = "an NVIDIA adapter with a blank DriverVersion";         A = @(NvAdapter 0 "  ");              Want = $true },
    @{ N = "a Basic Display adapter beside one on the NVIDIA driver"; Want = $false;
       A = @((Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_1C03" 0 "10.0.19041.3636"), $nv) },
    @{ N = "a versionless adapter beside one on the NVIDIA driver"; Want = $false;
       A = @((Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_1C03" 0 $null), $nv) },
    @{ N = "a disabled Basic Display NVIDIA adapter"; A = @(NvAdapter 22 "10.0.19041.3636"); Want = $false },
    @{ N = "a non-NVIDIA Basic Display adapter"; A = @(Adapter "Microsoft Basic Display Adapter" "ROOT\BASICDISPLAY" 0 "10.0.19041.3636"); Want = $false }
)) {
    $script:FakeAdapters = $case.A
    Check "$($case.N) reads as no-NVIDIA-driver = $($case.Want)" ((Test-NvidiaAdapterWithoutNvidiaDriver) -eq $case.Want)
}

function Get-NvidiaAdapterCudaFloor { param($Scan) return $null }
$promoVenv = Join-Path ([System.IO.Path]::GetTempPath()) ("promo-venv-" + [guid]::NewGuid().ToString("N"))
$promoPy = Join-Path $promoVenv "Scripts\python.exe"
$promoTorch = Join-Path $promoVenv "Lib\site-packages\torch"
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $promoPy), $promoTorch | Out-Null
Set-Content -LiteralPath $promoPy -Value "" -NoNewline
Set-Content -LiteralPath (Join-Path $promoTorch "version.py") -Value "__version__ = '2.8.0+xpu'"
Invoke-Expression (Get-FunctionText $setupPs1 "Test-VenvTorchIsXpu")
$script:FakeProbableVenv = $null
function Get-ProbableStudioVenvDir { return $script:FakeProbableVenv }
function Invoke-Promotion($promo, $adapters, $venv, $py) {
    try { & {
        $VenvDir = $venv; $VenvPython = $py
        $script:StudioPreservedXpuVerdict = $false; $script:StudioVenvRollbackDir = $null
        $script:FakeAdapters = $adapters; $script:FakeScanOk = $null
        $script:Lines = @()
        function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") $script:Lines += $Line }
        $HasNvidiaSmi = $false; $script:NvidiaPresenceOnly = $false; $script:NvidiaSmiRejected = $false
        . ([scriptblock]::Create($promo))
        [pscustomobject]@{ Has = $HasNvidiaSmi; Only = $script:NvidiaPresenceOnly; Lines = @($script:Lines) }
    } } catch { [pscustomobject]@{ Has = "threw: $($_.Exception.Message)"; Only = $null; Lines = @() } }
}
function Rtx4060 { Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0 "32.0.15.6094" }
function Mtl($id = "PCI\VEN_8086&DEV_7D55", $code = 0) { Adapter "Intel(R) Graphics" $id $code "32.0.101.6078" }
function Arc($id = "PCI\VEN_8086&DEV_56A0", $code = 0) { Adapter "Intel(R) Arc(TM) A770 Graphics" $id $code "32.0.101.6078" }
function BasicNv($v) { Adapter "Microsoft Basic Display Adapter" "PCI\VEN_10DE&DEV_2684" 0 $v }
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $promo = Get-PromotionBlock $file
    Check "$leaf's promotion block was found (bites)" ([bool]$promo)
    foreach ($case in @(
        @{ N = "Basic Display (10.0.19041.3636)";       A = @(BasicNv "10.0.19041.3636"); Want = $false; Hint = $true },
        @{ N = "an NVIDIA driver (560.94)";             A = @(NvAdapter 0 "32.0.15.6094"); Want = $true; Hint = $false },
        @{ N = "no DriverVersion";                      A = @(NvAdapter 0 $null); Want = $false; Hint = $true },
        @{ N = "Basic Display with no DriverVersion";   A = @(BasicNv $null); Want = $false; Hint = $true },
        @{ N = "Basic Display with an empty DriverVersion"; A = @(BasicNv ""); Want = $false; Hint = $true },
        @{ N = "Basic Display plus Intel UHD"; Want = $false; Hint = $true;
           A = @((BasicNv "10.0.19041.3636"), (Adapter "Intel(R) UHD Graphics" "PCI\VEN_8086&DEV_9A49" 0 "31.0.101.1")) },
        @{ N = "an NVIDIA driver plus a Meteor Lake GPU whose PyTorch has proven XPU"; A = @((Rtx4060), (Mtl)); Want = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver plus a Meteor Lake GPU with no XPU runtime";            A = @((Rtx4060), (Mtl)); Want = $true },
        # Base's Intel route ignored PNP ID and status, so these must still decline.
        @{ N = "an NVIDIA driver plus a Meteor Lake with no status, PyTorch XPU proven"; A = @((Rtx4060), (Mtl -code $null)); Want = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver plus a Meteor Lake with no PNP ID, PyTorch XPU proven"; A = @((Rtx4060), (Mtl $null)); Want = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver and no Intel row at all, PyTorch XPU proven";           A = @(Rtx4060); Want = $false; Xpu = "True" },
        @{ N = "an NVIDIA driver plus an Arc with no status, no XPU runtime";            A = @((Rtx4060), (Arc -code $null)); Want = $false },
        @{ N = "an NVIDIA driver plus an Arc with no PNP ID, no XPU runtime";            A = @((Rtx4060), (Arc $null)); Want = $false },
        @{ N = "an NVIDIA driver plus a disabled Arc, no XPU runtime";                   A = @((Rtx4060), (Arc -code 22)); Want = $false },
        @{ N = "an NVIDIA driver plus an AMD adapter with no status"; Want = $false;
           A = @((Rtx4060), (Adapter "AMD Radeon(TM) Graphics" "PCI\VEN_1002&DEV_15BF" $null "31.0.21001.45002")) },
        @{ N = "an NVIDIA driver plus a UHD with no status, no XPU runtime"; Want = $true;
           A = @((Rtx4060), (Adapter "Intel(R) UHD Graphics 770" "PCI\VEN_8086&DEV_4680" $null "31.0.101.1")) },
        @{ N = "an unreadable status"; A = @(NvAdapter "bogus" "32.0.15.6094"); Want = $false }
    )) {
        $script:ProbeAnswer = @{ Ok = $true; Output = $(if ($case.Xpu) { $case.Xpu } else { "False" }) }
        $got = Invoke-Promotion $promo $case.A $promoVenv $promoPy
        Check "$leaf, NVIDIA on $($case.N): promoted = $($case.Want)" ($got.Has -eq $case.Want -and $got.Only -eq $case.Want)
        $hint = [bool]$case.Hint
        Check "$leaf, NVIDIA on $($case.N): driver hint printed = $hint" ([bool](@($got.Lines) -match 'without the NVIDIA driver') -eq $hint)
    }
}
# Base asked Get-ProbableStudioVenvDir, which is not $VenvDir under a stage root.
$emptyVenv = Join-Path ([System.IO.Path]::GetTempPath()) ("promo-empty-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $emptyVenv | Out-Null
$script:ProbeAnswer = @{ Ok = $true; Output = "True" }
$script:FakeProbableVenv = $promoVenv
$got = Invoke-Promotion (Get-PromotionBlock $setupPs1) @((Rtx4060), (Mtl)) $emptyVenv (Join-Path $emptyVenv "Scripts\python.exe")
Check "setup.ps1 declines where the studio-home environment has proven XPU and the stage root has none" ($got.Has -eq $false)
$script:FakeProbableVenv = $null
Remove-Item -LiteralPath $emptyVenv, $promoVenv -Recurse -Force -ErrorAction SilentlyContinue
$script:NvidiaPresenceOnly = $false

$fakePy = Join-Path ([System.IO.Path]::GetTempPath()) ("xpu-py-" + [guid]::NewGuid().ToString("N"))
Set-Content -LiteralPath $fakePy -Value "" -NoNewline
$mtl = Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" 0
$rtx = Adapter "NVIDIA GeForce RTX 4060 Laptop GPU" "PCI\VEN_10DE&DEV_28E0" 0
foreach ($case in @(
    @{ N = "a Meteor Lake GPU whose PyTorch reports XPU is proven"; Want = $true },
    @{ N = "an XPU runtime that answers False is not proven"; Ans = @{ Ok = $true; Output = "False" }; Want = $false },
    @{ N = "a probe that times out is not proven"; Ans = @{ Ok = $false; Output = "" }; Want = $false },
    @{ N = "a probe that throws is not proven and does not throw out"; Throws = $true; Want = $false },
    @{ N = "no environment to ask is not proven"; Py = (Join-Path $fakePy "missing"); Want = $false },
    @{ N = "a proven XPU runtime is proven with no Intel row at all"; A = @($rtx); Want = $true; Calls = 1 },
    @{ N = "a proven XPU runtime is proven with an Intel row with no status";
       A = @((Adapter "Intel(R) Graphics" "PCI\VEN_8086&DEV_7D55" $null), $rtx); Want = $true; Calls = 1 },
    @{ N = "a proven XPU runtime is proven with an Intel row with no PNP ID";
       A = @((Adapter "Intel(R) Graphics" $null 0), $rtx); Want = $true; Calls = 1 },
    @{ N = "a proven XPU runtime is proven with a scan that did not answer"; A = @(); Want = $true; Calls = 1 }
)) {
    $list = if ($case.ContainsKey('A')) { @($case.A) } else { @($mtl, $rtx) }
    $scan = [pscustomobject]@{ Ok = $list.Count -gt 0; Adapters = $list; Names = @($list | ForEach-Object { $_.Name }) }
    $script:ProbeAnswer = if ($case.Ans) { $case.Ans } else { @{ Ok = $true; Output = "True" } }
    $script:ProbeThrows = [bool]$case.Throws
    $script:ProbeCalls = 0
    $got = Test-IntelXpuRuntimeProven -Scan $scan -PythonExe $(if ($case.Py) { $case.Py } else { $fakePy })
    Check $case.N ($got -eq $case.Want -and (-not $case.Calls -or $script:ProbeCalls -eq $case.Calls))
}
$script:ProbeThrows = $false
Remove-Item -LiteralPath $fakePy -Force -ErrorAction SilentlyContinue

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
