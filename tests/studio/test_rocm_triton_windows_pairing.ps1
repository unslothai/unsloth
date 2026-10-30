#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# setup.ps1 pairs triton-windows with torch on the AMD ROCm path after the dependency stack: the stack resolves
# unsloth's unpinned triton-windows, and 3.9 removed the block pointers torch 2.11 / 2.12's flex_decoding builds
# (gpt-oss 4k prefill 0.1 -> 6.7 GB on gfx1151). The block is cut out of setup.ps1 and run against stubs.
# Run: pwsh -NoProfile -File tests/studio/test_rocm_triton_windows_pairing.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$setupPath = Join-Path (Join-Path $root "studio") "setup.ps1"
$setup = Get-Content -LiteralPath $setupPath -Raw

$start = $setup.IndexOf("# ── AMD ROCm: pair triton-windows with torch ──")
$end = $setup.IndexOf("# ── Intel XPU: bitsandbytes must carry XPU kernels ──")
if ($start -lt 0 -or $end -le $start) { throw "the AMD triton-windows block is missing from setup.ps1" }
$block = $setup.Substring($start, $end - $start)

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "  PASS  $Name" } else { Write-Host "  FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

# Stubs.
$script:ProbeOutput = ""
$script:Installs = @()
function Invoke-BoundedPythonProbe { param($PythonExe, $Code, $TimeoutSec) return @{ Ok = $true; Output = $script:ProbeOutput } }
function Fast-Install { $script:Installs += ,($args -join " "); $global:LASTEXITCODE = 0; return "" }
function substep { param($Text, $Color) }
function Write-StudioLine { param($Text, $ForegroundColor) }
function Redact-InstallOutput { param($Text) return $Text }

function Invoke-Block([string]$Torch, [string]$Triton, [int]$StackExit = 0, $IndexUrl = "https://repo.amd.com/rocm/whl/gfx1151/") {
    $script:ProbeOutput = "TORCH=$Torch`nTRITONWIN=$Triton"
    $script:Installs = @()
    $stackExit = $StackExit
    $ROCmIndexUrl = $IndexUrl
    Invoke-Expression $block
    return ,$script:Installs
}

Write-Host "=== spec per torch ==="
Invoke-Expression ($block.Substring(0, $block.IndexOf("if (`$stackExit")))
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

Write-Host "=== wiring ==="
$stackAt = $setup.IndexOf('python "$PSScriptRoot\install_python_stack.py"')
Check "runs after the dependency stack" ($stackAt -ge 0 -and $start -gt $stackAt)

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "All checks passed"
