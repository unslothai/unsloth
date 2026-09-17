#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The NVIDIA inventory's Python rung: CPython's ctypes makes the same NVML and CUDA driver calls
# the emitted type makes, so a host that cannot emit a type (Constrained Language Mode, WDAC,
# Dynamic Code Security) still gets the CUDA version AND the per-device compute capabilities.
# Capabilities are the part WMI cannot supply: they feed $CudaArch into
# -DCMAKE_CUDA_ARCHITECTURES for the llama.cpp source build (#5854) and the pre-Turing cap.
#
# This file exists to stop three kinds of drift:
#   1. the two copies of the shared region drifting apart,
#   2. the embedded probe drifting from studio/nvidia_probe.py, which makes the same calls,
#   3. the Python rung being promoted ahead of the emitted one, or spending a second budget.
# Run: pwsh -NoProfile -File tests/studio/test_nvidia_python_probe_parity.ps1

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
    # A single-element return unrolls to a bare string, and [0] would then be its first
    # character. Callers wrap in @(), and this comment is why.
    return $out
}

$installPs1 = Join-Path $root "install.ps1"
$setupPs1 = Join-Path $root "studio\setup.ps1"
$probePy = Join-Path $root "studio\nvidia_probe.py"

# The embedded Python is the text between @' and '@ inside Read-NvidiaLibraryRawViaPython.
# Both files carry other here-strings, so anchor on the function rather than the first match.
# The variable is $probeSource, not $probe: install.ps1 and setup.ps1 each already carry a
# $probe here-string for the emit probe, and reusing the name made an older suite read this one.
function Get-EmbeddedProbe($path) {
    $text = [System.IO.File]::ReadAllText($path)
    $at = $text.IndexOf("function Read-NvidiaLibraryRawViaPython")
    if ($at -lt 0) { throw "no Read-NvidiaLibraryRawViaPython in $path" }
    $open = $text.IndexOf("`$probeSource = @'", $at)
    if ($open -lt 0) { throw "no probe here-string in $path" }
    $bodyStart = $text.IndexOf("`n", $open) + 1
    $close = $text.IndexOf("`n'@", $bodyStart)
    if ($close -lt 0) { throw "unterminated probe here-string in $path" }
    return $text.Substring($bodyStart, $close - $bodyStart)
}

Write-Host ""
Write-Host "=== the two copies agree ==="

$blockNames = @(
    "Get-NvidiaNvmlLibraryPath",
    "Get-NvidiaLibraryProbeType",
    "Read-NvidiaLibraryRawViaPython",
    "Read-NvidiaLibraryRaw",
    "Get-NvidiaLibraryInventory"
)
$installParts = @(Get-HelperSources $installPs1 $blockNames)
$setupParts = @(Get-HelperSources $setupPs1 $blockNames)
# install.ps1 nests its helpers one level deeper; compare the two copies without indentation.
$strip = { param($text) ($text -split "`n" | ForEach-Object { $_.TrimStart() }) -join "`n" }
for ($k = 0; $k -lt $blockNames.Count; $k++) {
    Check "install.ps1 and setup.ps1 carry the same $($blockNames[$k])" `
        ((& $strip $installParts[$k]) -eq (& $strip $setupParts[$k]))
}

$installProbe = Get-EmbeddedProbe $installPs1
$setupProbe = Get-EmbeddedProbe $setupPs1
# Stripping indentation would hide a real defect here: leading whitespace IS the Python syntax,
# so the two copies must match byte for byte, not merely after a TrimStart.
Check "the embedded Python is byte-identical in both files" ($installProbe -ceq $setupProbe)
Check "the embedded Python sits at column 0, so its indentation survives both files" `
    (($installProbe -split "`n" | Where-Object { $_ -match '^import ctypes' }).Count -eq 1)

Write-Host ""
Write-Host "=== the embedded probe matches studio/nvidia_probe.py ==="

$referenceProbe = [System.IO.File]::ReadAllText($probePy)

# Every native symbol the emitted type imports, and the two attribute numbers. If the embedded
# probe and nvidia_probe.py ever disagree on one of these, they are probing different things.
$symbols = @(
    "nvmlInit_v2", "nvmlShutdown", "nvmlSystemGetCudaDriverVersion_v2",
    "nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetCudaComputeCapability",
    "cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetAttribute"
)
foreach ($symbol in $symbols) {
    Check "both the embedded probe and nvidia_probe.py call $symbol" `
        (($installProbe -match [regex]::Escape($symbol)) -and ($referenceProbe -match [regex]::Escape($symbol)))
    Check "the emitted type imports $symbol too, so all three agree" `
        ($installParts[1] -match [regex]::Escape($symbol))
}

# CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR / _MINOR. A wrong number here returns a
# plausible integer rather than an error, which is exactly the drift a test has to catch.
Check "the embedded probe reads attribute 75 for the capability major" ($installProbe -match 'byref\(major\), 75,')
Check "the embedded probe reads attribute 76 for the capability minor" ($installProbe -match 'byref\(minor\), 76,')
Check "nvidia_probe.py reads the same two attribute numbers" `
    (($referenceProbe -match 'byref\(major\), 75,') -and ($referenceProbe -match 'byref\(minor\), 76,'))

# The packed CUDA version arithmetic, shared by all three implementations.
Check "the embedded probe unpacks the version as major*1000 + minor*10" `
    ($installProbe -match 'packed // 1000, \(packed % 1000\) // 10')
Check "nvidia_probe.py unpacks the version identically" `
    ($referenceProbe -match 'packed // 1000, \(packed % 1000\) // 10')
Check "the emitted rung unpacks the version identically" `
    (($installParts[3] -match 'Floor\(\$ver / 1000\)') -and ($installParts[3] -match 'Floor\(\(\$ver % 1000\) / 10\)'))

Check "the embedded probe is stdlib-only" `
    (($installProbe -match '(?m)^import ctypes, os, sys$') -and ($installProbe -notmatch '(?m)^\s*(import|from)\s+(?!ctypes|os|sys)'))

Write-Host ""
Write-Host "=== the rung stays underneath the emitted one ==="

$rawBlock = & $strip $setupParts[3]
Check "the emitted rung is attempted first" `
    ($rawBlock.IndexOf("Get-NvidiaLibraryProbeType") -lt $rawBlock.IndexOf("Read-NvidiaLibraryRawViaPython"))
Check "Python is only reached when the emitted rung returned nothing" `
    ($rawBlock -match 'if \(\$native\) \{ return \$native \}')
# One budget, not two: a driver that wedges the full timeout inside the emitted rung must not
# then buy a second timeout's worth of child process. Without this the worst case doubles.
Check "both rungs share one deadline" ($rawBlock -match '\$deadline = \(Get-Date\)\.AddMilliseconds\(\$TimeoutMs\)')
Check "Python gets only what the emitted rung left" ($rawBlock -match 'Read-NvidiaLibraryRawViaPython -TimeoutMs \$remainingMs')
Check "an exhausted budget skips the child process entirely" ($rawBlock -match 'if \(\$remainingMs -lt 2000\) \{ return "" \}')

$pyBlock = & $strip $setupParts[2]
# The banned constructs are NAMED in this rung's own comments, explaining why they are absent.
# Matching the raw text would fail on the explanation rather than on the code, so the CLM rows
# below run against comment-free source. (Found the honest way: they failed on the comments.)
$pyCode = ($pyBlock -split "`n" | Where-Object { $_.TrimStart() -notmatch '^#' }) -join "`n"
Check "the rung compiles nothing" ($pyCode -notmatch 'Add-Type')
Check "the rung emits no type" ($pyCode -notmatch 'New-StudioEmittedNativeType|DefinePInvokeMethod')
# Constrained Language Mode is one of the two policies that make the emitted rung decline, so a
# launcher this rung cannot use on a CLM host would be absent from the population it exists for.
Check "the launcher avoids ProcessStartInfo, which CLM refuses" ($pyCode -notmatch 'ProcessStartInfo')
Check "the launcher avoids [Process]::Start, which CLM refuses" ($pyCode -notmatch '\[(System\.)?Diagnostics\.Process\]::Start|\[Process\]::Start')
Check "the launcher avoids [System.IO.Path], which CLM refuses" ($pyCode -notmatch '\[System\.IO\.Path\]')
Check "the launcher avoids [math], which CLM refuses" ($pyCode -notmatch '\[math\]::')
# The stripping must not be so eager that it deletes the launcher itself.
Check "the comment-free source still contains the launcher" ($pyCode -match 'Start-Process -FilePath \$exe')
Check "the interpreter comes from a per-file hook, not from the shared region" `
    (($pyBlock -match 'Get-NvidiaProbePythonExe') -and ($pyBlock -notmatch 'function Get-NvidiaProbePythonExe'))
foreach ($file in @($installPs1, $setupPs1)) {
    Check "$(Split-Path -Leaf $file) defines its own Get-NvidiaProbePythonExe" `
        ([System.IO.File]::ReadAllText($file) -match 'function Get-NvidiaProbePythonExe')
}
Check "the probe is switchable off" ($pyBlock -match 'UNSLOTH_NVIDIA_PYTHON_PROBE')
Check "the temp files are always cleaned up" ($pyBlock -match 'finally \{[\s\S]*Remove-Item -LiteralPath \$stale')
# Windows PowerShell 5.1 appends each native argument verbatim, so anything with a space in it
# splits and the child silently runs something else. Nothing with a space may reach argv.
Check "only space-free arguments reach the command line" ($pyCode -match 'ArgumentList @\("-I", "-S", "-"\)')
Check "the script arrives on stdin, not as a path argument" ($pyCode -match '-RedirectStandardInput \$scriptFile')
Check "the library hints travel in the environment" `
    (($pyCode -match '\$env:UNSLOTH_NVML_HINT = \$nvmlHint') -and ($pyCode -match '\$env:UNSLOTH_CUDA_HINT = \$cudaHint'))
Check "the hint variables are restored afterwards" `
    ($pyCode -match 'finally \{[\s\S]*Remove-Item Env:UNSLOTH_NVML_HINT')
Check "the embedded probe reads the hints from the environment" `
    (($installProbe -match 'os\.environ\.get\("UNSLOTH_NVML_HINT"') -and ($installProbe -match 'os\.environ\.get\("UNSLOTH_CUDA_HINT"'))

Write-Host ""
Write-Host "=== the timeout is whole seconds, rounded up, without [math]::Ceiling ==="

# PowerShell's / is floating point and [int] rounds to NEAREST, so the C "+999" ceiling idiom
# turns 10000ms into 11s. Run the shipped expression rather than a retyped copy of it.
$ceilLines = @($pyBlock -split "`n" | Where-Object {
    $_ -match '^\$seconds = ' -or $_ -match '^if \(\(\$TimeoutMs % 1000\) -ne 0\)' -or $_ -match '^if \(\$seconds -lt 1\)'
})
Check "the ceiling is four lines of integer arithmetic" ($ceilLines.Count -eq 4)
$ceilExpr = [scriptblock]::Create(($ceilLines -join "`n") + "`nreturn `$seconds")
foreach ($row in @(
    @{ Ms = 10000; Want = 10 }, @{ Ms = 10001; Want = 11 }, @{ Ms = 9999; Want = 10 },
    @{ Ms = 1; Want = 1 }, @{ Ms = 0; Want = 1 }, @{ Ms = 2500; Want = 3 }, @{ Ms = 20000; Want = 20 }
)) {
    $TimeoutMs = $row.Ms
    $got = & $ceilExpr
    Check "$($row.Ms) ms is $($row.Want) s" ($got -eq $row.Want)
}

Write-Host ""
Write-Host "=== behaviour, driven through the real functions ==="

Invoke-Expression ($setupParts[0])   # Get-NvidiaNvmlLibraryPath
Invoke-Expression ($setupParts[2])   # Read-NvidiaLibraryRawViaPython
Invoke-Expression ($setupParts[3])   # Read-NvidiaLibraryRaw
Invoke-Expression ($setupParts[4])   # Get-NvidiaLibraryInventory
# The emitted rung declines, which is the whole point: this is what a Constrained Language Mode
# or WDAC host sees, and the capabilities below are recovered with nothing emitted.
function Get-NvidiaLibraryProbeType { return $null }

$script:PythonExe = ""
function Get-NvidiaProbePythonExe { return $script:PythonExe }

Check "no interpreter means no answer, not an error" ((Read-NvidiaLibraryRawViaPython -TimeoutMs 5000) -eq "")

$python = $null
foreach ($name in @("python3", "python")) {
    $cmd = Get-Command $name -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source; break }
}

if (-not $python) {
    Write-Host "  SKIP  no Python on this runner; the behavioural rows need one"
} else {
    $script:PythonExe = $python
    $saved = $env:UNSLOTH_NVIDIA_PYTHON_PROBE
    try {
        $env:UNSLOTH_NVIDIA_PYTHON_PROBE = "0"
        Check "UNSLOTH_NVIDIA_PYTHON_PROBE=0 declines before spawning anything" `
            ((Read-NvidiaLibraryRawViaPython -TimeoutMs 5000) -eq "")
    } finally {
        if ($null -eq $saved) { Remove-Item Env:UNSLOTH_NVIDIA_PYTHON_PROBE -ErrorAction SilentlyContinue }
        else { $env:UNSLOTH_NVIDIA_PYTHON_PROBE = $saved }
    }

    $answer = Read-NvidiaLibraryRawViaPython -TimeoutMs 30000
    # A runner without an NVIDIA driver is the common case and must answer "" rather than throw.
    if (-not $answer) {
        Check "no driver on this host answers empty, not an error" ($answer -eq "")
    } else {
        Check "the answer names a source the inventory understands" ($answer -match '^(nvml|cuda);')
        Check "the answer carries a version and at least one capability" `
            ($answer -match '^(nvml|cuda);\d+;\d+;\d+\.\d+(,\d+\.\d+)*$')
        # Through the real Read-NvidiaLibraryRaw, with the emitted rung declining. Every row
        # below is guarded on a non-null inventory: $null -eq $null would pass them all.
        $inventory = Get-NvidiaLibraryInventory -TimeoutSec 30
        Check "the inventory parses the answer with nothing emitted" ($null -ne $inventory)
        Check "the inventory recovered the compute capabilities" `
            ($null -ne $inventory -and $inventory.ComputeCaps.Count -ge 1 -and $inventory.ComputeCaps[0] -match '^\d+\.\d+$')
        Check "the inventory device count matches the capability list" `
            ($null -ne $inventory -and $inventory.Count -eq $inventory.ComputeCaps.Count)
        Check "the CUDA major version is plausible" ($null -ne $inventory -and $inventory.CudaMajor -ge 1)
        Check "the inventory names the source the raw answer named" `
            ($null -ne $inventory -and $answer.StartsWith("$($inventory.Source);"))
    }

    # The launcher must not leave the interpreter or its scratch files behind.
    $tempRoot = if ($env:TEMP) { $env:TEMP } elseif ($env:TMPDIR) { $env:TMPDIR } else { "/tmp" }
    $litter = @(Get-ChildItem -LiteralPath $tempRoot -Filter "unsloth-nvprobe-*" -ErrorAction SilentlyContinue)
    Check "the launcher leaves no scratch files behind" ($litter.Count -eq 0)
}

Write-Host ""
if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All checks passed"
