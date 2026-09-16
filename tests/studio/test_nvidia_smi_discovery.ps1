#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Where the installer looks for nvidia-smi.
#
# $HasNvidiaSmi does not mean "nvidia-smi exists". It means "an NVIDIA GPU is present", and the
# whole wheel-index decision hangs off it: a host where nvidia-smi is not found gets CPU-only
# PyTorch with no warning. Two directories were searched, and there are real hosts where a real
# GPU sits in neither of them (#9255).
#
# The list below is the only thing that decides which binaries get probed, so it is driven here
# against a fake filesystem rather than asserted by reading the source. Each check plants exactly
# one copy and asserts the list finds it, so a location that stops being searched fails on its own
# row rather than being masked by another.
# Run: pwsh -NoProfile -File tests/studio/test_nvidia_smi_discovery.ps1

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

# The two copies have to stay the same helper. install.ps1 nests one level deeper, so compare
# without indentation, the same way test_nvidia_library_inventory.ps1 does.
$installText = Get-FunctionText $installPs1 "Get-NvidiaSmiCandidatePaths"
$setupText = Get-FunctionText $setupPs1 "Get-NvidiaSmiCandidatePaths"
$strip = { param($t) (($t -split "`n") | ForEach-Object { $_.TrimStart() }) -join "`n" }
Check "install.ps1 and setup.ps1 carry the same Get-NvidiaSmiCandidatePaths" (
    (& $strip $installText) -eq (& $strip $setupText))

Invoke-Expression $setupText

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("nvsmi-" + [guid]::NewGuid().ToString("N"))
$saved = @{
    SystemRoot = $env:SystemRoot
    ProgramFiles = $env:ProgramFiles
    ProgramW6432 = $env:ProgramW6432
    ProgramFilesX86 = ${env:ProgramFiles(x86)}
}

function Reset-FakeMachine {
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force }
    $env:SystemRoot = Join-Path $tmp "Windows"
    $env:ProgramFiles = Join-Path $tmp "Program Files"
    $env:ProgramW6432 = Join-Path $tmp "Program Files"
    ${env:ProgramFiles(x86)} = Join-Path $tmp "Program Files (x86)"
}

function Add-FakeSmi($relative) {
    $dir = Join-Path $tmp $relative
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $exe = Join-Path $dir "nvidia-smi.exe"
    [System.IO.File]::WriteAllText($exe, "")
    return $exe
}

try {
    # Each location on its own. A single planted copy has to be the single answer, so a rung
    # dropped from the list fails here and nowhere else.
    foreach ($case in @(
        @{ Name = "System32, where a current driver puts it"; Dir = "Windows/System32" },
        @{ Name = "the legacy NVSMI directory"; Dir = "Program Files/NVIDIA Corporation/NVSMI" },
        @{ Name = "SysNative, which is System32 seen from a 32-bit process"; Dir = "Windows/SysNative" },
        @{ Name = "Program Files (x86), left by an old 32-bit toolkit"; Dir = "Program Files (x86)/NVIDIA Corporation/NVSMI" },
        @{ Name = "the DriverStore package the driver ships in"; Dir = "Windows/System32/DriverStore/FileRepository/nvmii.inf_amd64_1234/" }
    )) {
        Reset-FakeMachine
        $planted = Add-FakeSmi $case.Dir
        $found = @(Get-NvidiaSmiCandidatePaths)
        Check "found in $($case.Name)" ($found.Count -eq 1 -and $found[0] -eq $planted)
    }

    # ProgramW6432 is the 64-bit Program Files whatever the process bitness. Model the 32-bit
    # host: $env:ProgramFiles is the x86 tree, and the driver is only in the 64-bit one.
    Reset-FakeMachine
    $env:ProgramFiles = Join-Path $tmp "Program Files (x86)"
    $planted = Add-FakeSmi "Program Files/NVIDIA Corporation/NVSMI"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "found through ProgramW6432 when ProgramFiles points at the x86 tree" (
        $found.Count -eq 1 -and $found[0] -eq $planted)

    # Bites control. A host with no copy anywhere must come back empty, or every check above
    # would pass on a helper that simply returns a fixed list without testing for the file.
    Reset-FakeMachine
    New-Item -ItemType Directory -Force -Path (Join-Path $tmp "Windows/System32") | Out-Null
    Check "control: nothing planted, nothing found" ((@(Get-NvidiaSmiCandidatePaths)).Count -eq 0)

    # A directory named but absent must not throw, since this runs before anything is installed.
    Reset-FakeMachine
    Check "a machine missing every directory is handled" ((@(Get-NvidiaSmiCandidatePaths)).Count -eq 0)

    # The current driver's copy wins over the legacy one. Both work; the System32 copy belongs to
    # the driver that is actually loaded, and the NVSMI one can be older and report an older CUDA.
    Reset-FakeMachine
    $current = Add-FakeSmi "Windows/System32"
    $null = Add-FakeSmi "Program Files/NVIDIA Corporation/NVSMI"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "System32 is offered before the legacy NVSMI directory" (
        $found.Count -eq 2 -and $found[0] -eq $current)

    # Each probe is a process spawn with a wall-clock bound behind it, so the same file reached
    # two ways must be offered once. ProgramFiles and ProgramW6432 are equal on a 64-bit host,
    # which is the ordinary case, not a contrived one.
    Reset-FakeMachine
    $planted = Add-FakeSmi "Program Files/NVIDIA Corporation/NVSMI"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "a directory reached twice is probed once" ($found.Count -eq 1 -and $found[0] -eq $planted)

    # DriverStore holds every package the machine has ever had. Walking it whole on a host with a
    # long driver history is the kind of thing that turns an install into a hang.
    Reset-FakeMachine
    for ($i = 0; $i -lt 40; $i++) { $null = Add-FakeSmi "Windows/System32/DriverStore/FileRepository/nv$i.inf_amd64_$i/" }
    $null = Add-FakeSmi "Windows/System32/DriverStore/FileRepository/unrelated.inf_amd64_9/"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "the DriverStore scan is bounded and filtered to NVIDIA's own packages" (
        $found.Count -le 8 -and -not ($found -match "unrelated"))
} finally {
    $env:SystemRoot = $saved.SystemRoot
    $env:ProgramFiles = $saved.ProgramFiles
    if ($null -eq $saved.ProgramW6432) { Remove-Item Env:ProgramW6432 -ErrorAction SilentlyContinue }
    else { $env:ProgramW6432 = $saved.ProgramW6432 }
    if ($null -eq $saved.ProgramFilesX86) { Remove-Item "Env:ProgramFiles(x86)" -ErrorAction SilentlyContinue }
    else { ${env:ProgramFiles(x86)} = $saved.ProgramFilesX86 }
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
}

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
