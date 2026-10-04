#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for Format-CudaToolkitDir in studio/setup.ps1.
#
# Run: pwsh -NoProfile -File tests/studio/test_format_cuda_toolkit_dir.ps1

$ErrorActionPreference = "Stop"
$setupPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "studio", "setup.ps1")
$setupPath = (Resolve-Path $setupPath).Path

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($setupPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "setup.ps1 has parse errors" }
$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Format-CudaToolkitDir"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one Format-CudaToolkitDir, found $($fn.Count)" }
$cmakeFn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq "Format-CudaToolkitRootForCmake"
}, $true)
if ($cmakeFn.Count -ne 1) { throw "expected exactly one Format-CudaToolkitRootForCmake, found $($cmakeFn.Count)" }

. ([scriptblock]::Create($fn[0].Extent.Text))
. ([scriptblock]::Create($cmakeFn[0].Extent.Text))

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "PASS  $Name" -ForegroundColor Green }
    else { Write-Host "FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

Check "adds trailing slash" ((Format-CudaToolkitDir 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3') -eq 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\')
Check "idempotent when already slashed" ((Format-CudaToolkitDir 'C:\CUDA\v13.3\') -eq 'C:\CUDA\v13.3\')
Check "collapses repeated trailing slashes" ((Format-CudaToolkitDir 'C:\CUDA\v13.3\\') -eq 'C:\CUDA\v13.3\')
Check "cmake uses forward slashes and keeps the separator" ((Format-CudaToolkitRootForCmake 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3') -eq 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/')

$windowsPowerShell = Get-Command powershell.exe -ErrorAction SilentlyContinue
$python = Get-Command python.exe -ErrorAction SilentlyContinue
if ($windowsPowerShell -and $python) {
    $probePath = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-cuda-argv-{0}.ps1" -f [guid]::NewGuid().ToString('N'))
    try {
        @'
& python.exe -c 'import json, sys; print(json.dumps(sys.argv[1:]))' $env:UNSLOTH_CMAKE_ARG_1 $env:UNSLOTH_CMAKE_ARG_2
'@ | Set-Content -LiteralPath $probePath -Encoding UTF8
        $root = Format-CudaToolkitRootForCmake 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3'
        $env:UNSLOTH_CMAKE_ARG_1 = "-DCUDAToolkit_ROOT=$root"
        $env:UNSLOTH_CMAKE_ARG_2 = "-DCUDA_TOOLKIT_ROOT_DIR=$root"
        $json = & $windowsPowerShell.Source -NoProfile -ExecutionPolicy Bypass -File $probePath
        $received = $json | Select-Object -Last 1 | ConvertFrom-Json
        Check "Windows PowerShell preserves both native cmake arguments" (
            $received.Count -eq 2 -and
            $received[0] -eq $env:UNSLOTH_CMAKE_ARG_1 -and
            $received[1] -eq $env:UNSLOTH_CMAKE_ARG_2
        )
    } finally {
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
        Remove-Item Env:UNSLOTH_CMAKE_ARG_1 -ErrorAction SilentlyContinue
        Remove-Item Env:UNSLOTH_CMAKE_ARG_2 -ErrorAction SilentlyContinue
    }
} else {
    Write-Host "SKIP  Windows PowerShell native argv probe unavailable" -ForegroundColor Yellow
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
