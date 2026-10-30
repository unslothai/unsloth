#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# setup.ps1 pairs triton-windows with torch on the AMD ROCm path after the dependency stack: the stack resolves
# unsloth's unpinned triton-windows, and 3.9 removed the block pointers torch 2.11 / 2.12's flex_decoding builds
# (gpt-oss 4k prefill 0.1 -> 6.7 GB on gfx1151). A verified install that skips the dependency pass is sent back
# through it while the pair is stale. The helpers and the post-stack block are cut out of setup.ps1 and run against
# stubs. ASCII only: Windows PowerShell 5.1 reads a BOM-less script in the ANSI code page.
# Run: pwsh -NoProfile -File tests/studio/test_rocm_triton_windows_pairing.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setupPath = Join-Path (Join-Path $root "studio") "setup.ps1"
$setup = Get-Content -LiteralPath $setupPath -Raw -Encoding UTF8

function Cut([string]$From, [string]$To) {
    $start = $setup.IndexOf($From)
    $end = if ($start -ge 0) { $setup.IndexOf($To, $start) } else { -1 }
    if ($start -lt 0 -or $end -le $start) { throw "setup.ps1 no longer has the block from '$From' to '$To'" }
    return $setup.Substring($start, $end - $start)
}
$helpers = Cut "# AMD ROCm: triton-windows paired with torch." "# End of the AMD ROCm triton-windows pairing helpers."
$block = Cut "# AMD ROCm: pair triton-windows with torch (Get-RocmTritonPairing" "# Only bitsandbytes' XPU library"

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "  PASS  $Name" } else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

# Stubs.
$script:ProbeOutput = ""
$script:ProbeOk = $true
$script:Installs = @()
$script:Offline = $false
$script:OfflineFastPath = $false
function Invoke-BoundedPythonProbe { param($PythonExe, $Code, $TimeoutSec) return @{ Ok = $script:ProbeOk; Output = $script:ProbeOutput } }
function Fast-Install { $script:Installs += ,($args -join " "); $global:LASTEXITCODE = 0; return "" }
function Test-UvOfflineRequested { return $script:Offline }
function substep { param($Text, $Color) }
function Write-StudioLine { param($Text, $ForegroundColor) }
function Redact-InstallOutput { param($Text) return $Text }
Invoke-Expression $helpers

function Set-Probe([string]$Torch, [string]$Triton, [bool]$Ok = $true) {
    $script:ProbeOutput = "TORCH=$Torch`nTRITONWIN=$Triton"
    $script:ProbeOk = $Ok
}

function Invoke-Block([string]$Torch, [string]$Triton, [int]$StackExit = 0,
                      $IndexUrl = "https://repo.amd.com/rocm/whl/gfx1151/", [bool]$NoTorch = $false) {
    Set-Probe $Torch $Triton
    $script:Installs = @()
    $stackExit = $StackExit
    $ROCmIndexUrl = $IndexUrl
    $NoTorchMode = $NoTorch
    Invoke-Expression $block
    return ,$script:Installs
}

Write-Host "=== spec per torch ==="
Check "torch 2.11 ROCm pairs 3.8" ((Get-TritonWindowsRocmSpec -TorchVersion "2.11.0+rocm7.14.0") -eq "triton-windows>=3.8,<3.9")
Check "torch 2.12 ROCm pairs 3.8" ((Get-TritonWindowsRocmSpec -TorchVersion "2.12.1+rocm7.14.0") -eq "triton-windows>=3.8,<3.9")
Check "torch 2.13 is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion "2.13.0+rocm7.14.0"))
Check "torch 2.10 is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion "2.10.0+rocm7.2"))
Check "unreadable torch is left alone" ($null -eq (Get-TritonWindowsRocmSpec -TorchVersion ""))

Write-Host "=== post-stack step ==="
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "3.9 on torch 2.11 is replaced by 3.8" ($r.Count -eq 1 -and $r[0] -eq "triton-windows>=3.8,<3.9")
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.6.0.post25"
Check "3.6 (no HIP headers) is replaced by 3.8" ($r.Count -eq 1 -and $r[0] -eq "triton-windows>=3.8,<3.9")
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton ""
Check "a missing triton-windows is installed" ($r.Count -eq 1 -and $r[0] -eq "triton-windows>=3.8,<3.9")
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.8.0.post29"
Check "3.8 is kept" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.13.0+rocm7.14.0" -Triton "3.9.0.post30"
Check "torch 2.13 keeps what the stack resolved" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -IndexUrl $null
Check "not the ROCm path: nothing is touched" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -StackExit 1
Check "a failed stack is not touched" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "3.9.0.post30" -NoTorch $true
Check "no-torch mode leaves an existing Triton alone" ($r.Count -eq 0)
$r = Invoke-Block -Torch "2.11.0+rocm7.14.0" -Triton "" -NoTorch $true
Check "no-torch mode installs no Triton" ($r.Count -eq 0)

Write-Host "=== fast-path escape ==="
Set-Probe "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "verified install with 3.9 on ROCm torch 2.11 runs the pass" (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm")
Set-Probe "2.11.0+rocm7.14.0" ""
Check "a repair that failed last run (no triton-windows) is retried" (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm")
Set-Probe "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "a pinned gfx leaf tag counts as ROCm" (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "gfx1151")
Set-Probe "2.11.0+rocm7.14.0" "3.8.0.post29"
Check "a paired install keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
Set-Probe "2.13.0+rocm7.14.0" "3.9.0.post30"
Check "torch 2.13 keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
Set-Probe "2.11.0+cpu" "3.9.0.post30"
Check "a CPU torch keeps the fast path (the pass would not repair it)" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "cpu"))
Set-Probe "" "" -Ok $false
Check "an unreadable probe keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
$script:Offline = $true
Set-Probe "2.11.0+rocm7.14.0" "3.9.0.post30"
Check "UV_OFFLINE keeps the fast path" (-not (Test-RocmTritonPairingNeedsPass -InstalledTorchTag "rocm"))
$script:Offline = $false

Write-Host "=== wiring ==="
$stackAt = $setup.IndexOf('python "$PSScriptRoot\install_python_stack.py"')
Check "the repair runs after the dependency stack" ($stackAt -ge 0 -and $setup.IndexOf($block) -gt $stackAt)
$escape = 'if ($SkipPythonDeps -and -not $NoTorchMode -and (Test-RocmTritonPairingNeedsPass -InstalledTorchTag $installedTorchTag)) {'
$escapeAt = $setup.IndexOf($escape)
$gateAt = $setup.IndexOf('if (-not $SkipPythonDeps) {')
Check "the fast path asks for the pairing (outside no-torch mode)" ($escapeAt -ge 0)
Check "the helpers are defined before the fast-path escape" ($setup.IndexOf($helpers) -lt $escapeAt)
Check "the escape runs before the dependency gate" ($escapeAt -lt $gateAt)
$chainAt = if ($escapeAt -ge 0) { $setup.LastIndexOf('# After the chain', $escapeAt) } else { -1 }
Check "the escape sits in the AMD-GPU checks after the fast-path chain" ($chainAt -gt 0 -and $setup.IndexOf('if ($script:ROCmGfxArch) {', $chainAt) -lt $escapeAt -and $setup.IndexOf('if ($script:ROCmGfxArch) {', $chainAt) -gt $chainAt)

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed"
