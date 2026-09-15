#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Behavioural test for the NVIDIA driver-library inventory in install.ps1 and studio/setup.ps1:
# the torch family and compute capability come from NVML / the CUDA driver API when nvidia-smi
# cannot answer, an unknown driver version no longer means cu126, and a prebuilt update that
# fails keeps a working GPU prebuilt instead of building for the CPU (#9255).
# Run: pwsh -NoProfile -File tests/studio/test_nvidia_library_inventory.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

function Get-HelperSources($path, $names) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$path has parse errors" }
    $out = @()
    foreach ($name in $names) {
        $fn = $ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
        }, $true)
        if ($fn.Count -lt 1) { throw "expected $name in $path, found none" }
        $out += $fn[0].Extent.Text
    }
    return $out
}

# Printer stubs, so the file does not depend on the ANSI helpers.
function substep { param([string]$Message, [string]$Color = "DarkGray") $script:Substeps += $Message }
function step { param([string]$Component, [string]$Message, [string]$Color = "") $script:Steps += $Message }
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }
function Write-StudioStdoutMirror { param([string]$Line) }
function Write-LlamaFailureLog { param([string]$Output, [int]$MaxLines = 120) }
function Test-PathQuiet { param([string]$Path, [string]$PathType = "Any") Test-Path -LiteralPath $Path }
function Test-CudaFamilyLeaf { param([string]$Leaf) return ($Leaf -match '^cu\d+$') }

# nvidia-smi stand-in: $script:FakeSmiStdout answers every probe, $script:FakeSmiRc is its exit code.
function Invoke-NvidiaSmiBounded {
    param([string]$Exe, [string[]]$SmiArgs = @(), [int]$TimeoutSec = 10, [switch]$StdoutOnly)
    $global:LASTEXITCODE = $script:FakeSmiRc
    return $script:FakeSmiStdout
}

$installPs1 = Join-Path $root "install.ps1"
$setupPs1 = Join-Path $root "studio\setup.ps1"

Write-Host ""
Write-Host "=== shared inventory helper ==="
$installBlock = @(Get-HelperSources $installPs1 @("Get-NvidiaLibraryInventory"))[0]
$setupBlock = @(Get-HelperSources $setupPs1 @("Get-NvidiaLibraryInventory"))[0]
# install.ps1 nests its helpers one level deeper; compare the two copies without indentation.
$strip = { param($text) ($text -split "`n" | ForEach-Object { $_.TrimStart() }) -join "`n" }
Check "install.ps1 and setup.ps1 carry the same helper" ((& $strip $installBlock) -eq (& $strip $setupBlock))
Invoke-Expression $setupBlock

$script:NvidiaLibraryInventoryProbed = $false
$env:UNSLOTH_NVIDIA_LIBRARY_PROBE = "0"
Check "UNSLOTH_NVIDIA_LIBRARY_PROBE=0 turns the probe off" ($null -eq (Get-NvidiaLibraryInventory))
Remove-Item Env:UNSLOTH_NVIDIA_LIBRARY_PROBE -ErrorAction SilentlyContinue

$script:NvidiaLibraryInventoryProbed = $false
$real = Get-NvidiaLibraryInventory
if ($real) {
    Write-Host "  host inventory: $($real.Source) CUDA $($real.CudaMajor).$($real.CudaMinor) caps $($real.ComputeCaps -join ',')"
    Check "a real inventory names the driver family" ($real.CudaMajor -ge 11)
    Check "a real inventory lists one capability per GPU" ($real.Count -eq $real.ComputeCaps.Count -and $real.Count -ge 1)
    Check "capabilities are major.minor strings" (@($real.ComputeCaps | Where-Object { $_ -notmatch '^\d+\.\d+$' }).Count -eq 0)
} else {
    Write-Host "  (no NVIDIA driver library on this host; the real-inventory checks are skipped)"
}

# From here on the inventory is a stub the cases control.
function Get-NvidiaLibraryInventory { param([int]$TimeoutSec = 10) return $script:FakeInventory }

Write-Host ""
Write-Host "=== studio/setup.ps1 ==="
foreach ($src in (Get-HelperSources $setupPs1 @("Get-NvidiaCu126Verdict", "Get-CudaFamilyCappedForPreTuring",
        "Get-PytorchCudaTag", "Get-CudaComputeCapability", "Get-LlamaUpdateFailReason",
        "Get-GpuPrebuiltToKeepOverSourceBuild"))) {
    Invoke-Expression $src
}

$script:NvidiaSmiExe = "C:\fake\nvidia-smi.exe"
$script:FakeSmiRc = 0
$script:FakeSmiStdout = "| NVIDIA-SMI 580.00   Driver Version: 580.00   CUDA Version: 12.9 |"
$script:FakeInventory = $null
$script:PreTuringCapAnnounced = $false
Check "a readable banner still picks the family from nvidia-smi" ((Get-PytorchCudaTag) -eq "cu128")

$script:FakeSmiStdout = ""
$script:FakeSmiRc = 1
$script:FakeInventory = @{ Source = "nvml"; CudaMajor = 13; CudaMinor = 0; ComputeCaps = @("8.9"); Count = 1 }
Check "a failing nvidia-smi reads the family from the inventory" ((Get-PytorchCudaTag) -eq "cu130")
Check "the compute capability comes from the inventory too" ((Get-CudaComputeCapability) -eq "89")

$script:FakeInventory = @{ Source = "nvml"; CudaMajor = 13; CudaMinor = 0; ComputeCaps = @("6.1"); Count = 1 }
$script:PreTuringCapAnnounced = $false
Check "the pre-Turing cap reads the inventory capabilities" ((Get-PytorchCudaTag) -eq "cu126")

$script:NvidiaSmiExe = $null
$script:FakeInventory = @{ Source = "cuda"; CudaMajor = 12; CudaMinor = 8; ComputeCaps = @("12.0"); Count = 1 }
Check "no nvidia-smi at all still names the family" ((Get-PytorchCudaTag) -eq "cu128")

$script:FakeInventory = $null
Check "nothing answering means unknown, not cu126" ((Get-PytorchCudaTag) -eq "")
Check "an unknown capability stays null" ($null -eq (Get-CudaComputeCapability))

Check "a 429 reads as a rate limit" ((Get-LlamaUpdateFailReason "HTTP Error 429: too many requests") -eq "GitHub rate limit")
Check "a DNS failure reads as a network error" ((Get-LlamaUpdateFailReason "temporary failure in name resolution") -eq "network error")
Check "anything else is a download failure" ((Get-LlamaUpdateFailReason "sha256 mismatch") -eq "download failed")

# The keep guard: marker backend, GPU presence, this run's requests, and the installed tree running.
$install = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-keep-" + [System.IO.Path]::GetRandomFileName())
New-Item -ItemType Directory -Path $install -Force | Out-Null
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"backend": "cuda", "release_tag": "v1"}'
function python { $global:LASTEXITCODE = $script:FakePythonRc }
$script:FakePythonRc = 0
$HasNvidiaSmi = $true
$HasROCm = $false
$script:ROCmGfxArch = $null
$script:IsIntelXpu = $false
$LlamaPr = ""
$explicitLlamaSourceBackend = $null
foreach ($name in @("UNSLOTH_LLAMA_FORCE_COMPILE", "UNSLOTH_LLAMA_RELEASE_TAG", "UNSLOTH_LLAMA_TAG")) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
Check "a running CUDA prebuilt on an NVIDIA host is kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "cuda")
$env:UNSLOTH_LLAMA_TAG = "latest"
Check "the default tag spelling still keeps" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "cuda")
$env:UNSLOTH_LLAMA_TAG = "b7000"
Check "a version pin asked for that version, so nothing is kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
Remove-Item Env:UNSLOTH_LLAMA_TAG
$env:UNSLOTH_LLAMA_FORCE_COMPILE = "1"
Check "a forced compile is not kept over" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
Remove-Item Env:UNSLOTH_LLAMA_FORCE_COMPILE
$explicitLlamaSourceBackend = "cuda"
Check "an explicit backend request is not kept over" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
$explicitLlamaSourceBackend = $null
$script:FakePythonRc = 1
Check "an install that no longer runs is not kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
$script:FakePythonRc = 0
$HasNvidiaSmi = $false
Check "a CUDA prebuilt with the GPU gone is not kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"backend": "vulkan"}'
$HasROCm = $true
Check "a Vulkan prebuilt is kept for any GPU vendor" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "vulkan")
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"backend": "cpu"}'
Check "a CPU prebuilt has nothing to keep" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
Remove-Item -LiteralPath $install -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== install.ps1 ==="
foreach ($src in (Get-HelperSources $installPs1 @("Get-NvidiaCu126Verdict", "Get-CudaFamilyCappedForPreTuring",
        "Trim-IndexPathSlashes", "Get-TorchIndexUrl"))) {
    Invoke-Expression $src
}
foreach ($name in @("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY", "UNSLOTH_PYTORCH_MIRROR")) {
    Remove-Item "Env:$name" -ErrorAction SilentlyContinue
}
$NvidiaSmiExe = $null
$script:FakeInventory = $null
Check "no nvidia-smi and no inventory is the CPU index" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cpu")
$script:FakeInventory = @{ Source = "nvml"; CudaMajor = 13; CudaMinor = 1; ComputeCaps = @("10.0"); Count = 1 }
Check "no nvidia-smi but an inventory is the driver's family" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cu130")
$NvidiaSmiExe = "C:\fake\nvidia-smi.exe"
$script:FakeSmiRc = 0
$script:FakeSmiStdout = "no banner here"
Check "an unreadable banner falls back to the inventory" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cu130")
$script:FakeInventory = @{ Source = "nvml"; CudaMajor = 12; CudaMinor = 8; ComputeCaps = @("7.0"); Count = 1 }
$script:PreTuringCapAnnounced = $false
Check "the cap applies to inventory capabilities" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cu126")
$script:FakeInventory = $null
$script:Substeps = @()
Check "an unreadable banner with no inventory keeps the cu126 default" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cu126")
$script:FakeSmiStdout = "| CUDA Version: 12.6 |"
Check "a readable banner is still authoritative" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cu126")

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed"
