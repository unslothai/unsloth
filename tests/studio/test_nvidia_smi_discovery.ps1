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

    # The bound applies to directories that actually HOLD the binary, not to the nv* name matches.
    # NVIDIA ships several packages whose names start nv and contain no nvidia-smi at all: the HD
    # audio driver, the virtual audio device, the network service. A machine that installed any of
    # those after its display driver has them newer by write time, so a bound applied to the name
    # match spends the whole allowance on directories with nothing in them and reports the host as
    # having no nvidia-smi. Plant the real one FIRST so it is the oldest, then bury it.
    Reset-FakeMachine
    $repo = "Windows/System32/DriverStore/FileRepository"
    $planted = Add-FakeSmi "$repo/nvmii.inf_amd64_0001/"
    for ($i = 0; $i -lt 12; $i++) {
        $dir = Join-Path $tmp "$repo/nvhda.inf_amd64_10$i"
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        [System.IO.File]::WriteAllText((Join-Path $dir "nvhda.sys"), "")
    }
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "a display driver buried under newer nv* packages that ship no nvidia-smi is still found" (
        $found.Count -eq 1 -and $found[0] -eq $planted)

    # The 32-bit process. $env:ProgramFiles is the x86 tree there, so a stale copy left by an old
    # 32-bit CUDA toolkit sits in the FIRST Program Files rung unless the 64-bit locations are
    # searched ahead of it. That copy is a real binary: it lists the GPU and reports its own older
    # CUDA version, so the two-pass CUDA preference does not rescue this. Only the order does.
    Reset-FakeMachine
    $env:ProgramFiles = Join-Path $tmp "Program Files (x86)"
    $env:ProgramW6432 = Join-Path $tmp "Program Files"
    $stale = Add-FakeSmi "Program Files (x86)/NVIDIA Corporation/NVSMI"
    $current = Add-FakeSmi "Windows/SysNative"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "in a 32-bit process SysNative is probed before the stale x86 toolkit copy" (
        $found.Count -eq 2 -and $found[0] -eq $current -and $found[1] -eq $stale)

    # Same host, driver in the 64-bit Program Files rather than SysNative.
    Reset-FakeMachine
    $env:ProgramFiles = Join-Path $tmp "Program Files (x86)"
    $env:ProgramW6432 = Join-Path $tmp "Program Files"
    $stale = Add-FakeSmi "Program Files (x86)/NVIDIA Corporation/NVSMI"
    $current = Add-FakeSmi "Program Files/NVIDIA Corporation/NVSMI"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "in a 32-bit process the 64-bit NVSMI copy is probed before the x86 one" (
        $found.Count -eq 2 -and $found[0] -eq $current -and $found[1] -eq $stale)
} finally {
    $env:SystemRoot = $saved.SystemRoot
    $env:ProgramFiles = $saved.ProgramFiles
    if ($null -eq $saved.ProgramW6432) { Remove-Item Env:ProgramW6432 -ErrorAction SilentlyContinue }
    else { $env:ProgramW6432 = $saved.ProgramW6432 }
    if ($null -eq $saved.ProgramFilesX86) { Remove-Item "Env:ProgramFiles(x86)" -ErrorAction SilentlyContinue }
    else { ${env:ProgramFiles(x86)} = $saved.ProgramFilesX86 }
    if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
}

# ------------------------------------------------- which candidate the detection loop settles on
#
# The scenario is the one this reordering could have caused, and it was found by an adversarial
# audit rather than by me. Two binaries both answer -L. The one searched first prints a banner
# with no parseable CUDA version; the one searched second reports 12.8. Taking the first would
# send a cu128-capable host to cu126, purely because of which directory is searched first.
$setupAll = Get-Content -Raw -LiteralPath $setupPs1
$start = $setupAll.IndexOf('$smiCandidates = @(Get-NvidiaSmiCandidatePaths)')
if ($start -lt 0) { Check "the detection loop still makes two passes" $false }
else {
    # Bounded by the next statement rather than by a character count. A fixed 1400 stopped
    # short of the fallback the moment a comment was added above it, and the check failed
    # for a reason that had nothing to do with the loop.
    $loopEndForRead = $setupAll.IndexOf('if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory))', $start)
    $loop = $setupAll.Substring($start, $loopEndForRead - $start)
    Check "the loop prefers a candidate that also reports a CUDA version" (
        $loop -match "CUDA\(\?: UMD\)\? Version:")
    Check "and still settles for one that only lists a GPU when none reports a version" (
        $loop -match '\$firstListing' -and $loop -match 'if \(-not \$HasNvidiaSmi -and \$firstListing\)')
}

# Driven, not read. Both candidates list a GPU; only the second names a version.
$script:SmiCalls = @()
function Test-NvidiaSmiHasGpu { param([string]$Exe) $script:SmiCalls += "L:$Exe"; return $true }
function Invoke-NvidiaSmiBounded {
    param($Exe, $Arguments)
    $script:SmiCalls += "B:$Exe"
    if ($Exe -like "*second*") { return "CUDA Version: 12.8" }
    return "NVIDIA-SMI has failed because it couldn't communicate with the driver"
}
function Write-StudioLine { param([string]$Line, [string]$ForegroundColor = "") }
function Get-NvidiaSmiCandidatePaths { return @("C:\first\nvidia-smi.exe", "C:\second\nvidia-smi.exe") }
# Anchored on the deadline, which is the first statement of the loop's setup. Anchoring on
# $firstListing instead left $probeDeadline undefined inside the slice, and a comparison
# against $null broke out on the first iteration: every driven check below then passed or
# failed for the wrong reason. Observed here before it was fixed.
$loopStart = $setupAll.IndexOf('$smiCandidates = @(Get-NvidiaSmiCandidatePaths)')
$loopEnd = $setupAll.IndexOf('if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory))', $loopStart)
# The slice starts inside setup.ps1's "if (-not $HasNvidiaSmi) {" block, so it carries that
# block's closing brace and is short one opening brace. Supply the opener rather than trimming a
# brace off the end, which would silently drop the last statement.
$loopSrc = 'if ($true) {' + [System.Environment]::NewLine + $setupAll.Substring($loopStart, $loopEnd - $loopStart)
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $loopSrc
Check "the candidate that reports a CUDA version wins over the one searched first" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\second\nvidia-smi.exe")

# And when NOTHING reports a version, the first that lists a GPU is still used: this must not
# turn a detected GPU into no GPU at all.
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return "no version here" }
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $loopSrc
Check "with no version anywhere, the first listing candidate is still taken" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\first\nvidia-smi.exe")

# ----------------------------------------------------- and it gives up before setup does
#
# Each candidate gets its own bounded runner, and on a host with a wedged NVIDIA driver every one
# of them waits out that bound before failing. Broadening the search made the list longer, so what
# used to be two stalls can now be ten, all of it spent in front of a driver-library fallback that
# answers in milliseconds. Driven with probes that actually sleep, because the thing under test is
# elapsed time and a stubbed clock would only prove the arithmetic.
#
# The deadline in the shipped file is 30 seconds, which no test should sit through. The slice is
# rewritten to 2 before it is invoked, and the real value is asserted separately just below so
# that rewrite cannot quietly become the thing being tested.
Check "the shipped soft deadline is 30 seconds" ($setupAll -match '\$probeDeadline = \(Get-Date\)\.AddSeconds\(30\)')
Check "the shipped hard deadline is 60 seconds" ($setupAll -match '\$probeHardDeadline = \(Get-Date\)\.AddSeconds\(60\)')
$fastLoop = ($loopSrc -replace 'AddSeconds\(30\)', 'AddSeconds(2)') -replace 'AddSeconds\(60\)', 'AddSeconds(4)'
Check "the slice really carries both shortened deadlines (bites)" (
    $fastLoop -match 'AddSeconds\(2\)' -and $fastLoop -match 'AddSeconds\(4\)')

$script:Probed = 0
function Test-NvidiaSmiHasGpu { param([string]$Exe) $script:Probed++; Start-Sleep -Milliseconds 700; return $true }
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return "no version here" }
function Get-NvidiaSmiCandidatePaths {
    return @(1..10 | ForEach-Object { "C:\wedged$_\nvidia-smi.exe" })
}
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
$started = Get-Date
Invoke-Expression $fastLoop
$elapsed = ((Get-Date) - $started).TotalSeconds

Check "a wedged host stops probing instead of walking every candidate" ($script:Probed -lt 10)
Check "it probed at least once rather than skipping the loop outright" ($script:Probed -ge 1)
Check "the whole loop is bounded near its deadline, not by the candidate count" ($elapsed -lt 6)
# The bound must not cost a detection. A host that answers is still detected, and it is the FIRST
# listing candidate, exactly as it was without the deadline.
Check "giving up early still reports the GPU it did find" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\wedged1\nvidia-smi.exe")

# Bites control: the same loop on a host where nothing hangs walks the entire list, so the check
# above is about elapsed time and not about the loop simply stopping after one candidate.
$script:Probed = 0
function Test-NvidiaSmiHasGpu { param([string]$Exe) $script:Probed++; return $false }
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $fastLoop
Check "control: with fast probes every candidate is still tried" ($script:Probed -eq 10)

# ------------------------------------------- the wedge flag must describe the binary we chose
#
# Test-NvidiaSmiHasGpu records whether the binary it just probed timed out, and its own comment
# says what matters is "whether the binary it settled on answered". The two-pass loop broke that:
# it can keep probing after the candidate it will settle on, so the flag ends up describing a
# LATER binary. Downstream that flag decides whether to ask the selected nvidia-smi for the GPU
# name, the compute capability and the driver version, so a working binary would have all three
# skipped because a different copy elsewhere on the machine hung.
$script:NvidiaSmiWedged = $false
function Test-NvidiaSmiHasGpu {
    param([string]$Exe)
    # The first candidate answers and is the one the loop settles on. The second hangs.
    if ($Exe -like "*good*") { $script:NvidiaSmiWedged = $false; return $true }
    $script:NvidiaSmiWedged = $true
    return $false
}
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return "no version here" }
function Get-NvidiaSmiCandidatePaths {
    return @("C:\good\nvidia-smi.exe", "C:\hung\nvidia-smi.exe")
}
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $loopSrc
Check "the working candidate is the one selected" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\good\nvidia-smi.exe")
Check "a later candidate hanging does not mark the selected binary wedged" (
    $script:NvidiaSmiWedged -eq $false)

# Bites control: when the binary the loop settles on is ITSELF the one that hung, the flag must
# stay set, or the guard downstream would query a hung nvidia-smi and wait out the bound again.
function Test-NvidiaSmiHasGpu {
    param([string]$Exe)
    $script:NvidiaSmiWedged = $true
    return $true
}
$script:NvidiaSmiWedged = $false
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $loopSrc
Check "control: a selected binary that did hang is still reported wedged" (
    $script:NvidiaSmiWedged -eq $true)

# ------------------------------------- giving up must never cost a GPU that was there to find
#
# This is the regression the first version of the deadline introduced, found by an adversarial
# audit rather than by me. One budget, and a break in the failure branch, meant that a single
# slow FAILING probe ended the whole search. A machine whose System32 copy is broken and whose
# legacy NVSMI copy works lost its GPU entirely and went to CPU-only PyTorch, which is strictly
# worse than the stall the deadline was added to prevent. The comment above the old break even
# claimed this could not happen.
#
# Two budgets now: the soft one only applies once there is already a usable answer, and only the
# hard one may end the search empty-handed.
$script:Probed = 0
function Test-NvidiaSmiHasGpu {
    param([string]$Exe)
    $script:Probed++
    if ($Exe -like "*broken*") { Start-Sleep -Milliseconds 2500; return $false }   # slow AND failing
    return $true
}
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return "CUDA Version: 12.8" }
function Get-NvidiaSmiCandidatePaths {
    return @("C:\broken\nvidia-smi.exe", "C:\legacy\nvidia-smi.exe")
}
$script:NvidiaSmiWedged = $false
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $fastLoop
Check "a slow FAILING first probe does not end the search" ($script:Probed -ge 2)
Check "the working copy behind it is still found" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\legacy\nvidia-smi.exe")

# The hard bound still exists: with nothing working anywhere, the loop stops rather than walking
# an unbounded list. This is the property the deadline was added for, and it must survive the fix.
$script:Probed = 0
function Test-NvidiaSmiHasGpu { param([string]$Exe) $script:Probed++; Start-Sleep -Milliseconds 700; return $false }
function Get-NvidiaSmiCandidatePaths { return @(1..20 | ForEach-Object { "C:\wedged$_\nvidia-smi.exe" }) }
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
$t0 = Get-Date
Invoke-Expression $fastLoop
$spent = ((Get-Date) - $t0).TotalSeconds
Check "with nothing working the hard bound still stops the walk" ($script:Probed -lt 20)
Check "and it spends the HARD budget, not the soft one, before giving up" ($spent -ge 2)
Check "control: giving up empty-handed really reports no GPU" ($HasNvidiaSmi -eq $false)

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
