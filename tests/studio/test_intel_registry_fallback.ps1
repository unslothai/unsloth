#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit tests for Get-IntelRegistryAdapterNames, the display-class-key fallback used when the
# bounded CIM scan does not answer. AST-extracted from both installers (which must stay
# identical) and run in-process with Get-ChildItem / Get-ItemProperty mocked, so the hive is a
# fixture and no Windows registry is touched. Its oracle is windows_intel_gpu_in_registry() in
# studio/install_llama_prebuilt.py, which reads the same key: the two must agree on "is an
# Intel display adapter present".
# Run: pwsh -NoProfile -File tests/studio/test_intel_registry_fallback.ps1

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$fnName = "Get-IntelRegistryAdapterNames"

function Get-FunctionText {
    param([string] $Path, [string] $Name)
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$Path has parse errors" }
    $fn = $ast.FindAll({ param($n)
        $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $Name
    }, $true)
    if ($fn.Count -ne 1) { throw "expected exactly one $Name in $Path, found $($fn.Count)" }
    return $fn[0].Extent.Text
}

$installText = Get-FunctionText (Join-Path $repo "install.ps1") $fnName
$setupText = Get-FunctionText (Join-Path $repo "studio/setup.ps1") $fnName

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

# Both installers carry their own copy (install.ps1 nests it in a function, setup.ps1 does
# not), so compare with indentation normalised.
function Normalize($text) { (($text -replace "`r", "") -split "`n" | ForEach-Object { $_.Trim() }) -join "`n" }

Write-Host "$fnName is identical in both installers"
Check "install.ps1 and studio/setup.ps1 agree" ((Normalize $installText) -eq (Normalize $setupText))
# An empty or wrong extraction would make every case below pass vacuously.
Check "extraction kept the display class key" ($installText -match '4d36e968-e325-11ce-bfc1-08002be10318')
Check "extraction kept the PCI vendor id"     ($installText -match 'ven_8086')

# --- registry fixture -----------------------------------------------------------------------
# Throw='prop' models a provider blow-up on one subkey (corrupt hive); NullProps models the
# ordinary access-denied read, which is non-terminating and yields $null under -EA SilentlyContinue.
function New-Adapter {
    param([string] $Key, [string] $Desc, [string] $DevId, [string] $Throw = "none", [switch] $NullProps)
    [pscustomobject]@{ Key = $Key; Desc = $Desc; DevId = $DevId; Throw = $Throw; NullProps = [bool]$NullProps }
}

function Get-AdapterNames {
    param([object[]] $Adapters, [switch] $RootThrows)
    $sb = [scriptblock]::Create(@"
param(`$Adapters, `$RootThrows)
function Get-ChildItem {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    if (`$RootThrows) { throw "class key unreadable" }
    foreach (`$a in `$Adapters) { [pscustomobject]@{ PSChildName = `$a.Key; PSPath = "MOCK::`$(`$a.Key)" } }
}
function Get-ItemProperty {
    [CmdletBinding()] param([string] `$LiteralPath, [Parameter(ValueFromRemainingArguments = `$true)] `$Rest)
    `$a = `$Adapters | Where-Object { `$_.Key -eq (`$LiteralPath -replace '^MOCK::', '') } | Select-Object -First 1
    if (-not `$a) { return `$null }
    if (`$a.Throw -eq 'prop') { throw "access denied" }
    if (`$a.NullProps) { return `$null }
    `$o = [pscustomobject]@{}
    if (`$null -ne `$a.Desc)  { `$o | Add-Member DriverDesc `$a.Desc }
    if (`$null -ne `$a.DevId) { `$o | Add-Member MatchingDeviceId `$a.DevId }
    return `$o
}
$installText
@($fnName)
"@)
    # The caller wraps this in try/catch, so a throw here means "no adapters", not a crash.
    try { return , @(& $sb $Adapters $RootThrows) } catch { return @() }
}

# The caller's own filter, verbatim from install.ps1 and studio/setup.ps1.
function Test-Xpu { param([object[]] $Names)
    [bool]($Names | Where-Object { $_ -match "(?i)Intel.*(Arc|Data Center GPU)" } | Select-Object -First 1)
}

$VEN = "PCI\VEN_8086&DEV_56A0"
$ARC = "Intel(R) Arc(TM) A770 Graphics"
# Non-English Windows ships localized brand strings, leaving the vendor id as the only ASCII
# anchor -- which is why the match arm cannot rely on DriverDesc alone.
$ARC_JP = [char]0x30A4 + [char]0x30F3 + [char]0x30C6 + [char]0x30EB + "(R) Arc(TM) A770"

Write-Host "Intel adapters are found and classified"
Check "Arc, English DriverDesc"        (Test-Xpu (Get-AdapterNames @(New-Adapter "0000" $ARC $VEN)))
Check "Arc, localized DriverDesc"      (Test-Xpu (Get-AdapterNames @(New-Adapter "0000" $ARC_JP $VEN)))
Check "Data Center GPU Max"            (Test-Xpu (Get-AdapterNames @(New-Adapter "0000" "Intel(R) Data Center GPU Max 1550" $VEN)))
Check "Arc behind an NVIDIA primary"   (Test-Xpu (Get-AdapterNames @((New-Adapter "0000" "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684"), (New-Adapter "0001" $ARC $VEN))))
Check "Arc behind an AMD primary"      (Test-Xpu (Get-AdapterNames @((New-Adapter "0000" "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C"), (New-Adapter "0001" $ARC $VEN))))

Write-Host "Intel without XPU wheels stays off the xpu index"
Check "iGPU, English DriverDesc"       (-not (Test-Xpu (Get-AdapterNames @(New-Adapter "0000" "Intel(R) UHD Graphics 770" $VEN))))
Check "iGPU, OEM DriverDesc"           (-not (Test-Xpu (Get-AdapterNames @(New-Adapter "0000" "OEM Display Adapter" $VEN))))
Check "vendor id, empty DriverDesc"    (-not (Test-Xpu (Get-AdapterNames @(New-Adapter "0000" "" $VEN))))
# A vendor-id hit with a localized or OEM-branded name must still count as an Intel GPU, or
# the caller's Intel filter discards it.
Check "OEM DriverDesc still an adapter" ((Get-AdapterNames @(New-Adapter "0000" "OEM Display Adapter" $VEN)).Count -eq 1)

Write-Host "Non-Intel hosts report nothing"
Check "NVIDIA only"                    ((Get-AdapterNames @(New-Adapter "0000" "NVIDIA GeForce RTX 4090" "PCI\VEN_10DE&DEV_2684")).Count -eq 0)
Check "AMD only"                       ((Get-AdapterNames @(New-Adapter "0000" "AMD Radeon RX 7900 XTX" "PCI\VEN_1002&DEV_744C")).Count -eq 0)
Check "empty class key"                ((Get-AdapterNames @()).Count -eq 0)
Check "class key unreadable"           ((Get-AdapterNames @(New-Adapter "0000" $ARC $VEN) -RootThrows).Count -eq 0)
# "Properties" is ACL-restricted on every Windows install; only numeric subkeys are adapters.
Check "non-numeric subkey ignored"     ((Get-AdapterNames @(New-Adapter "Configuration" $ARC $VEN)).Count -eq 0)

Write-Host "One unreadable subkey does not hide the rest"
Check "throwing subkey before the Arc" (Test-Xpu (Get-AdapterNames @((New-Adapter "0000" $null $null "prop"), (New-Adapter "0001" $ARC $VEN))))
Check "throwing subkey after the Arc"  (Test-Xpu (Get-AdapterNames @((New-Adapter "0000" $ARC $VEN), (New-Adapter "0001" $null $null "prop"))))
Check "ACL Properties beside an Arc"   (Test-Xpu (Get-AdapterNames @((New-Adapter "Properties" $null $null "prop"), (New-Adapter "0000" $ARC $VEN))))
Check "denied read before the Arc"     (Test-Xpu (Get-AdapterNames @((New-Adapter "0000" $null $null -NullProps), (New-Adapter "0001" $ARC $VEN))))
Check "every subkey unreadable"        ((Get-AdapterNames @((New-Adapter "0000" $null $null "prop"), (New-Adapter "0001" $null $null "prop"))).Count -eq 0)

# --- drift guards on the two constants the installers assemble rather than ask for ---
$setupText = Get-Content -Raw (Join-Path $repo "studio/setup.ps1")
$installText2 = Get-Content -Raw (Join-Path $repo "install.ps1")
$manifestPy = Get-Content -Raw (Join-Path $repo "studio/install_manifest.py")

Write-Host "constants the installers hard-code stay in step with their source of truth"
# setup.ps1 joins this onto $VenvDir instead of asking install_manifest, so a rename there
# would silently stop the Triton swap from holding the manifest.
$manifestName = if ($manifestPy -match '(?m)^MANIFEST_NAME\s*=\s*"([^"]+)"') { $Matches[1] } else { "" }
Check "install_manifest exposes MANIFEST_NAME"   ($manifestName -ne "")
Check "setup.ps1 uses that exact file name"      ($manifestName -and $setupText.Contains("`"$manifestName`""))

# The reconciliation gate must key off the XPU match, not off "any Intel name": a hybrid laptop
# reports an ASCII "Intel UHD" next to a localized Arc, and keying on Intel stops at the UHD.
foreach ($pair in @(@("install.ps1", $installText2), @("studio/setup.ps1", $setupText))) {
    Check "$($pair[0]) gates reconciliation on the XPU regex" ($pair[1] -match '\$_xpuNameRe\s*=\s*"\(\?i\)Intel\.\*\(Arc\|Data Center GPU\)"')
    Check "$($pair[0]) reuses it for the gate"                ($pair[1] -match 'Where-Object \{ \$_ -match \$_xpuNameRe \}')
    # @() must wrap the WHOLE if: per-branch, a one-element array unrolls on its way out,
    # making $_gpuNames a String on a single-adapter host, and the += then concatenates.
    Check "$($pair[0]) forces an array for the WMI names" ($pair[1] -match '\$_gpuNames = @\(if \(\$_gpuScan\.Ok\)')
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
