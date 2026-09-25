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
$script:FakeIntelAdapters = @()
function Get-IntelRegistryAdapterNames { return $script:FakeIntelAdapters }

# nvidia-smi stand-in: $script:FakeSmiStdout answers every probe, $script:FakeSmiRc is its exit code.
# $script:FakeSmiTimeouts calls time out first (exit 124); every bound asked for is recorded.
$script:FakeSmiTimeouts = 0
$script:FakeSmiBounds = @()
function Invoke-NvidiaSmiBounded {
    param([string]$Exe, [string[]]$SmiArgs = @(), [int]$TimeoutSec = 10, [switch]$StdoutOnly)
    $script:FakeSmiBounds += $TimeoutSec
    if ($script:FakeSmiTimeouts -gt 0) {
        $script:FakeSmiTimeouts--
        $global:LASTEXITCODE = 124
        return ""
    }
    $global:LASTEXITCODE = $script:FakeSmiRc
    return $script:FakeSmiStdout
}

$installPs1 = Join-Path $root "install.ps1"
$setupPs1 = Join-Path $root "studio\setup.ps1"

Write-Host ""
Write-Host "=== shared inventory helper ==="
$blockNames = @("Get-NvidiaNvmlLibraryPath", "Get-NvidiaLibraryProbeType", "Read-NvidiaLibraryRaw", "Get-NvidiaLibraryInventory")
$installParts = @(Get-HelperSources $installPs1 $blockNames)
$setupParts = @(Get-HelperSources $setupPs1 $blockNames)
$installBlock = $installParts[3]
$setupBlock = $setupParts[3]
$setupPath = $setupParts[0]
$readBlock = $setupParts[2]
# install.ps1 nests its helpers one level deeper; compare the two copies without indentation.
$strip = { param($text) ($text -split "`n" | ForEach-Object { $_.TrimStart() }) -join "`n" }
for ($k = 0; $k -lt $blockNames.Count; $k++) {
    Check "install.ps1 and setup.ps1 carry the same $($blockNames[$k])" ((& $strip $installParts[$k]) -eq (& $strip $setupParts[$k]))
}
# The installer must not spawn a C# compiler (windows-no-compiler-ci): the methods are emitted.
Check "the inventory compiles nothing" ((($setupParts -join "`n") -notmatch 'Add-Type') -and ($setupParts[1] -match 'New-StudioEmittedNativeType'))
Check "the emission is gated on the native-type capability" ($setupParts[1] -match 'if \(-not \(Test-StudioCanDefineNativeTypes\)\) \{ return \$null \}')
Invoke-Expression $setupPath
$pathRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-nvml-" + [guid]::NewGuid().ToString("N"))
$sys32 = Join-Path (Join-Path $pathRoot "root") "System32"
$nvsmi = Join-Path (Join-Path $pathRoot "pf") "NVIDIA Corporation\NVSMI"
New-Item -ItemType Directory -Path $sys32, $nvsmi -Force | Out-Null
$savedRoot = $env:SystemRoot; $savedPf = $env:ProgramFiles
try {
    $env:SystemRoot = Join-Path $pathRoot "root"; $env:ProgramFiles = Join-Path $pathRoot "pf"
    Check "no nvml.dll on disk keeps the bare name" ((Get-NvidiaNvmlLibraryPath) -eq "nvml.dll")
    Set-Content -Path (Join-Path $nvsmi "nvml.dll") -Value ""
    Check "an NVSMI-only nvml.dll is named by its path" ((Get-NvidiaNvmlLibraryPath) -eq (Join-Path $nvsmi "nvml.dll"))
    Set-Content -Path (Join-Path $sys32 "nvml.dll") -Value ""
    Check "System32 wins when both exist" ((Get-NvidiaNvmlLibraryPath) -eq (Join-Path $sys32 "nvml.dll"))
} finally {
    $env:SystemRoot = $savedRoot; $env:ProgramFiles = $savedPf
    Remove-Item -LiteralPath $pathRoot -Recurse -Force -ErrorAction SilentlyContinue
}
Check "a failed driver-version read is not an inventory" (
    $readBlock -match 'nvmlSystemGetCudaDriverVersion_v2\(\[ref\]\$ver\) -ne 0' -and
    $readBlock -match 'cuDriverGetVersion\(\[ref\]\$ver\) -ne 0')

# The real libraries, in a child so the fake type below can own this session.
$helperFile = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-inventory-" + [System.IO.Path]::GetRandomFileName() + ".ps1")
$emitters = @(Get-HelperSources $setupPs1 @("New-StudioDynamicAssembly", "New-StudioEmittedNativeType"))
# The capability gate is a Windows policy probe; the child stands in for it and answers yes.
$gateStub = 'function Test-StudioCanDefineNativeTypes { return $true }'
$helperBody = (@($gateStub) + $emitters + $setupParts) -join "`n"
Set-Content -LiteralPath $helperFile -Value ($helperBody + "`n`$inv = Get-NvidiaLibraryInventory`nif (`$inv) { `$inv | ConvertTo-Json -Compress } else { 'null' }")
$pwshExe = (Get-Process -Id $PID).Path
$realJson = & $pwshExe -NoProfile -File $helperFile 2>&1 | Select-Object -Last 1
$offJson = & $pwshExe -NoProfile -Command "`$env:UNSLOTH_NVIDIA_LIBRARY_PROBE = '0'; & '$helperFile'" 2>&1 | Select-Object -Last 1
Remove-Item -LiteralPath $helperFile -ErrorAction SilentlyContinue
Check "UNSLOTH_NVIDIA_LIBRARY_PROBE=0 turns the probe off" ("$offJson" -eq "null")
if ("$realJson" -ne "null" -and "$realJson" -match '^\{') {
    $real = "$realJson" | ConvertFrom-Json
    Write-Host "  host inventory: $($real.Source) CUDA $($real.CudaMajor).$($real.CudaMinor) caps $($real.ComputeCaps -join ',')"
    Check "a real inventory names the driver family" ($real.CudaMajor -ge 11)
    Check "a real inventory lists one capability per GPU" ($real.Count -eq @($real.ComputeCaps).Count -and $real.Count -ge 1)
    Check "capabilities are major.minor strings" (@(@($real.ComputeCaps) | Where-Object { $_ -notmatch '^\d+\.\d+$' }).Count -eq 0)
} else {
    Write-Host "  (no NVIDIA driver library on this host; the real-inventory checks are skipped)"
}

# The parser, over a stand-in for the library reader.
Invoke-Expression $setupBlock
$script:FakeRaw = ""
function Read-NvidiaLibraryRaw { param([int]$TimeoutMs = 10000) return $script:FakeRaw }
function Probe-Raw($raw) {
    $script:NvidiaLibraryInventoryProbed = $false
    $script:FakeRaw = $raw
    return Get-NvidiaLibraryInventory
}
$parsed = Probe-Raw "nvml;13;1;8.9,12.0"
Check "a probe line parses to the driver version and capabilities" (
    $parsed.Source -eq "nvml" -and $parsed.CudaMajor -eq 13 -and $parsed.CudaMinor -eq 1 -and
    $parsed.Count -eq 2 -and ($parsed.ComputeCaps -join ",") -eq "8.9,12.0")
Check "the CUDA driver API line parses too" ((Probe-Raw "cuda;12;8;12.0").Source -eq "cuda")
Check "an empty probe is no inventory" ($null -eq (Probe-Raw ""))
Check "a zero driver version is no inventory" ($null -eq (Probe-Raw "nvml;0;0;8.9"))
Check "no capabilities is no inventory" ($null -eq (Probe-Raw "nvml;13;0;"))
Check "an unreadable capability voids the inventory" ($null -eq (Probe-Raw "nvml;13;0;N/A,8.9"))
Check "one unreadable GPU voids the reader's source too" ($readBlock -notmatch '\bcontinue\b')
Check "the CUDA driver API is read with the mask lifted" ($readBlock -match 'Remove-Item Env:CUDA_VISIBLE_DEVICES' -and $readBlock -match '\$env:CUDA_VISIBLE_DEVICES = \$saved')
Check "setup.ps1 initialises the cache with its other script state" (
    (Get-Content -LiteralPath $setupPs1 -Raw) -match '(?m)^\$script:NvidiaLibraryInventoryProbed = \$false')
Check "the answer is cached" ($null -eq (Get-NvidiaLibraryInventory))

Write-Host ""
Write-Host "=== consumers of the inventory ==="
$resolve = @(Get-HelperSources $setupPs1 @("Resolve-CudaToolkit"))[0]
Check "Resolve-CudaToolkit takes the driver ceiling from the inventory" (
    $resolve -match 'Get-NvidiaLibraryInventory' -and $resolve -match '-not \$NvidiaSmiExe')
$installText = Get-Content -LiteralPath $installPs1 -Raw
$sharedEnd = $installText.IndexOf("END SHARED WITH studio/setup.ps1 (Get-NvidiaLibraryInventory)")
$detect = $installText.IndexOf("Detect GPU (robust", $sharedEnd)
Check "install.ps1 resets the cache before each invocation's detection" (
    $installText.Substring($sharedEnd, $detect - $sharedEnd) -match 'NvidiaLibraryInventoryProbed = \$false')

# From here on the inventory is a stub the cases control.
function Get-NvidiaLibraryInventory { param([int]$TimeoutSec = 10) return $script:FakeInventory }

Write-Host ""
Write-Host "=== studio/setup.ps1 ==="
foreach ($src in (Get-HelperSources $setupPs1 @("Get-NvidiaCu126Verdict", "Get-CudaFamilyCappedForPreTuring",
        "Get-CudaFamilyForVersion", "Get-PytorchCudaTag", "Get-CudaComputeCapability", "Get-LlamaUpdateFailReason",
        "Get-PrebuiltMarkerBackend", "Get-GpuPrebuiltToKeepOverSourceBuild"))) {
    Invoke-Expression $src
}

$script:NvidiaSmiRejected = $false
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

# Detection rejected nvidia-smi (absent or unusable) and the library answered: the consumers
# do not rediscover it, so a wedged or stale banner never overrides the inventory.
$script:NvidiaSmiRejected = $true
$script:FakeSmiStdout = "Driver Version: 560.00  CUDA Version: 12.6"; $script:FakeSmiRc = 0
$script:FakeInventory = @{ Source = "nvml"; CudaMajor = 13; CudaMinor = 1; ComputeCaps = @("8.9"); Count = 1 }
Check "a rejected nvidia-smi is not asked again for the tag" ((Get-PytorchCudaTag) -eq "cu130")
Check "a rejected nvidia-smi is not asked again for the capability" ((Get-CudaComputeCapability) -eq "89")
$script:NvidiaSmiRejected = $false
$setupText = Get-Content -LiteralPath $setupPs1 -Raw
Check "setup.ps1 marks nvidia-smi rejected when the library stands in" ($setupText -match '(?m)^\s+\$script:NvidiaSmiRejected = \$true' -and $setupText -match '(?m)^\$script:NvidiaSmiRejected = \$false')

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
$HasNvidiaDriverEvidence = $true
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
$env:UNSLOTH_LLAMA_TAG = "master"
Check "master asks for a source build, so nothing is kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
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
$HasNvidiaDriverEvidence = $false
Check "a CUDA prebuilt with the GPU gone is not kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
# A card on the bus with no driver stack behind it (the presence-only promotion) is not a CUDA
# host for llama.cpp: install_llama_prebuilt.py never sees the bus, so it would not pick CUDA either.
$HasNvidiaSmi = $true
$HasNvidiaDriverEvidence = $false
Check "a CUDA prebuilt on a presence-only NVIDIA host is not kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
$HasNvidiaSmi = $false
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"backend": "vulkan"}'
$HasROCm = $true
Check "a Vulkan prebuilt is kept for any GPU vendor" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "vulkan")
$HasROCm = $false
Check "a Vulkan prebuilt with no GPU at all is not kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
$script:FakeIntelAdapters = @("Intel(R) UHD Graphics 770")
Check "a Vulkan prebuilt is kept for an Intel adapter that is not an XPU part" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "vulkan")
$script:FakeIntelAdapters = @()
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"backend": "cpu"}'
Check "a CPU prebuilt has nothing to keep" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "")
# Markers from before the backend field, read as setup.sh reads them.
$HasNvidiaSmi = $true
$HasNvidiaDriverEvidence = $true
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"llama_backend": "cuda"}'
Check "a legacy marker naming the request is kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "cuda")
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"asset": "app-b8508-mix-windows-x64-cuda13-newer.zip"}'
Check "a legacy marker naming only the asset is kept" ((Get-GpuPrebuiltToKeepOverSourceBuild -InstallDir $install) -eq "cuda")
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value '{"backend": "HIP"}'
Check "hip reads as rocm, case-insensitively" ((Get-PrebuiltMarkerBackend -Marker (Join-Path $install "UNSLOTH_PREBUILT_INFO.json")) -eq "rocm")
Set-Content -LiteralPath (Join-Path $install "UNSLOTH_PREBUILT_INFO.json") -Value ''
Check "an empty marker has no backend" ((Get-PrebuiltMarkerBackend -Marker (Join-Path $install "UNSLOTH_PREBUILT_INFO.json")) -eq "")
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
Check "the cu126 default names the override" (@($script:Substeps | Where-Object { $_ -match 'UNSLOTH_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128' }).Count -eq 1)
# A congested driver: nvidia-smi times out once, and the longer retry reads the banner.
$script:FakeSmiStdout = "| CUDA Version: 13.1 |"
$script:FakeSmiTimeouts = 1
$script:FakeSmiBounds = @()
Check "a timed-out banner is retried and read" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cu130")
Check "the retry has the longer bound" (($script:FakeSmiBounds[0..1] -join ",") -eq "10,45")
$script:FakeSmiTimeouts = 0
$script:FakeSmiStdout = "| CUDA Version: 12.6 |"
Check "a readable banner is still authoritative" ((Get-TorchIndexUrl) -eq "https://download.pytorch.org/whl/cu126")

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed"
