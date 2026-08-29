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

. ([scriptblock]::Create($fn[0].Extent.Text))

$failures = 0
function Check([string]$Name, [bool]$Ok) {
    if ($Ok) { Write-Host "PASS  $Name" -ForegroundColor Green }
    else { Write-Host "FAIL  $Name" -ForegroundColor Red; $script:failures++ }
}

Check "adds trailing slash" ((Format-CudaToolkitDir 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3') -eq 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\')
Check "idempotent when already slashed" ((Format-CudaToolkitDir 'C:\CUDA\v13.3\') -eq 'C:\CUDA\v13.3\')
Check "collapses repeated trailing slashes" ((Format-CudaToolkitDir 'C:\CUDA\v13.3\\') -eq 'C:\CUDA\v13.3\')

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) FAILED" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed" -ForegroundColor Green
