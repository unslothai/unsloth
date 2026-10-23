#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# $HasNvidiaSmi means "an NVIDIA GPU is present"; a missed nvidia-smi means CPU-only
# PyTorch. Each check plants exactly one copy in a fake filesystem.
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

# install.ps1 nests one level deeper, so compare without indentation.
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

    # ProgramW6432 is the 64-bit Program Files whatever the process bitness.
    Reset-FakeMachine
    $env:ProgramFiles = Join-Path $tmp "Program Files (x86)"
    $planted = Add-FakeSmi "Program Files/NVIDIA Corporation/NVSMI"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "found through ProgramW6432 when ProgramFiles points at the x86 tree" (
        $found.Count -eq 1 -and $found[0] -eq $planted)

    # Control: no copy anywhere must come back empty.
    Reset-FakeMachine
    New-Item -ItemType Directory -Force -Path (Join-Path $tmp "Windows/System32") | Out-Null
    Check "control: nothing planted, nothing found" ((@(Get-NvidiaSmiCandidatePaths)).Count -eq 0)

    # Runs before anything is installed, so an absent directory must not throw.
    Reset-FakeMachine
    Check "a machine missing every directory is handled" ((@(Get-NvidiaSmiCandidatePaths)).Count -eq 0)

    # The System32 copy belongs to the loaded driver; NVSMI can be older.
    Reset-FakeMachine
    $current = Add-FakeSmi "Windows/System32"
    $null = Add-FakeSmi "Program Files/NVIDIA Corporation/NVSMI"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "System32 is offered before the legacy NVSMI directory" (
        $found.Count -eq 2 -and $found[0] -eq $current)

    # Each probe is a bounded spawn; ProgramFiles == ProgramW6432 on a 64-bit host.
    Reset-FakeMachine
    $planted = Add-FakeSmi "Program Files/NVIDIA Corporation/NVSMI"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "a directory reached twice is probed once" ($found.Count -eq 1 -and $found[0] -eq $planted)

    # DriverStore holds every package ever installed; walking it whole can hang an install.
    Reset-FakeMachine
    for ($i = 0; $i -lt 40; $i++) { $null = Add-FakeSmi "Windows/System32/DriverStore/FileRepository/nv$i.inf_amd64_$i/" }
    $null = Add-FakeSmi "Windows/System32/DriverStore/FileRepository/unrelated.inf_amd64_9/"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "the DriverStore scan is bounded and filtered to NVIDIA's own packages" (
        $found.Count -le 8 -and -not ($found -match "unrelated"))

    # In a 32-bit process System32 redirects to SysWOW64, which has no DriverStore.
    Reset-FakeMachine
    $planted = Add-FakeSmi "Windows/SysNative/DriverStore/FileRepository/nvmii.inf_amd64_0001/"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "the DriverStore is reached through SysNative as well as System32" (
        $found.Count -eq 1 -and $found[0] -eq $planted)

    # SysNative and System32 name one directory, so the bound is on the whole contribution.
    Reset-FakeMachine
    for ($i = 0; $i -lt 12; $i++) { $null = Add-FakeSmi "Windows/System32/DriverStore/FileRepository/nva$i.inf_amd64_$i/" }
    for ($i = 0; $i -lt 12; $i++) { $null = Add-FakeSmi "Windows/SysNative/DriverStore/FileRepository/nvb$i.inf_amd64_$i/" }
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "both DriverStore spellings still share one bound of eight" ($found.Count -le 8)

    # The bound applies to dirs that HOLD the binary: newer nv* packages (audio, network)
    # lack it. Plant the real one FIRST so it is the oldest.
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

    # In a 32-bit process a stale x86 toolkit copy wins unless 64-bit locations come first.
    Reset-FakeMachine
    $env:ProgramFiles = Join-Path $tmp "Program Files (x86)"
    $env:ProgramW6432 = Join-Path $tmp "Program Files"
    $stale = Add-FakeSmi "Program Files (x86)/NVIDIA Corporation/NVSMI"
    $current = Add-FakeSmi "Windows/SysNative"
    $found = @(Get-NvidiaSmiCandidatePaths)
    Check "in a 32-bit process SysNative is probed before the stale x86 toolkit copy" (
        $found.Count -eq 2 -and $found[0] -eq $current -and $found[1] -eq $stale)

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

# The first binary prints no parseable CUDA version, the second 12.8; taking the first
# would send a cu128 host to cu126.
$setupAll = Get-Content -Raw -LiteralPath $setupPs1
$start = $setupAll.IndexOf('$smiCandidates = @(Get-NvidiaSmiCandidatePaths)')
if ($start -lt 0) { Check "the detection loop still makes two passes" $false }
else {
    # Bounded by the next statement, not a character count.
    $loopEndForRead = $setupAll.IndexOf('if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory))', $start)
    $loop = $setupAll.Substring($start, $loopEndForRead - $start)
    Check "the loop prefers a candidate that also reports a CUDA version" (
        $loop -match "CUDA\(\?: UMD\)\? Version:")
    Check "and still settles for one that only lists a GPU when none reports a version" (
        $loop -match '\$firstListing' -and $loop -match 'if \(-not \$HasNvidiaSmi -and \$firstListing\)')
}

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
# Anchored on the deadline, or $probeDeadline is undefined inside the slice.
$loopStart = $setupAll.IndexOf('$smiCandidates = @(Get-NvidiaSmiCandidatePaths)')
$loopEnd = $setupAll.IndexOf('if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory))', $loopStart)
# The slice carries the enclosing block's closing brace, so supply the opener.
$loopSrc = 'if ($true) {' + [System.Environment]::NewLine + $setupAll.Substring($loopStart, $loopEnd - $loopStart)
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $loopSrc
Check "the candidate that reports a CUDA version wins over the one searched first" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\second\nvidia-smi.exe")

# When nothing reports a version, the first GPU listing is still used.
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) return "no version here" }
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $loopSrc
Check "with no version anywhere, the first listing candidate is still taken" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\first\nvidia-smi.exe")

# Each candidate waits out its own bound on a wedged driver, so probes really sleep. The
# shipped 30 s deadline is rewritten to 2 and asserted separately.
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
Check "giving up early still reports the GPU it did find" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\wedged1\nvidia-smi.exe")

# Control: with nothing hanging the loop walks the whole list.
$script:Probed = 0
function Test-NvidiaSmiHasGpu { param([string]$Exe) $script:Probed++; return $false }
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
Invoke-Expression $fastLoop
Check "control: with fast probes every candidate is still tried" ($script:Probed -eq 10)

# The wedge flag must describe the binary the loop settles on, not a later one, or a
# working binary has its name/capability/driver queries skipped.
$script:NvidiaSmiWedged = $false
function Test-NvidiaSmiHasGpu {
    param([string]$Exe)
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

# Control: if the settled binary itself hung, the flag must stay set.
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

# Two budgets: the soft one applies only once there is a usable answer; only the hard one
# may end the search empty-handed.
$script:Probed = 0
function Test-NvidiaSmiHasGpu {
    param([string]$Exe)
    $script:Probed++
    if ($Exe -like "*broken*") { Start-Sleep -Milliseconds 2500; return $false }
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

# The hard bound still stops the loop when nothing works anywhere.
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

# The banner probe must respect the deadline too, or a late candidate adds nearly two
# probe timeouts.
$script:Listings = 0
$script:Banners = 0
function Test-NvidiaSmiHasGpu { param([string]$Exe) $script:Listings++; Start-Sleep -Milliseconds 900; return $true }
function Invoke-NvidiaSmiBounded {
    param($Exe, $Arguments)
    $script:Banners++
    Start-Sleep -Milliseconds 900
    return "no version here"
}
function Get-NvidiaSmiCandidatePaths { return @(1..10 | ForEach-Object { "C:\slow$_\nvidia-smi.exe" }) }
$script:NvidiaSmiWedged = $false
$HasNvidiaSmi = $false
$NvidiaSmiExe = $null
$t0 = Get-Date
Invoke-Expression $fastLoop
$spent = ((Get-Date) - $t0).TotalSeconds
# Counted, not timed: the timing checks held with the fix reverted.
Check "a slow banner probe does not start after the deadline" ($script:Banners -eq 1)
Check "control: a banner probe did run for the candidate before it" ($script:Listings -ge 2)
Check "the loop does not spend two probe bounds per candidate after expiry" ($spent -lt 8)
Check "control: the GPU found before the deadline is still reported" (
    $HasNvidiaSmi -eq $true -and $NvidiaSmiExe -eq "C:\slow1\nvidia-smi.exe")

# Get-WoaNvidiaSmiPath must answer with a WORKING binary: its callers treat a non-answer
# as "no NVIDIA".
$woaFn = Get-FunctionText $installPs1 "Get-WoaNvidiaSmiPath"
$script:WoaNvidiaSmiProbed = $false
$script:WoaNvidiaSmiPath = $null
$script:WoaProbes = @()
function Get-Command { param($Name, $ErrorAction) return $null }
function Invoke-NvidiaSmiBounded {
    param($Exe, $Arguments)
    $script:WoaProbes += $Exe
    if ($Exe -like "*working*") {
        $global:LASTEXITCODE = 0
        if ($Arguments -contains '-L') { return "GPU 0: NVIDIA GeForce RTX 4090" }
        return "CUDA Version: 12.8"
    }
    $global:LASTEXITCODE = 1
    return ""
}
function Get-NvidiaSmiCandidatePaths {
    return @("C:\stale\nvidia-smi.exe", "C:\working\nvidia-smi.exe")
}
Invoke-Expression $woaFn
$woaAnswer = Get-WoaNvidiaSmiPath
Check "the ARM64 picker skips a stale copy and returns one that answers" (
    $woaAnswer -eq "C:\working\nvidia-smi.exe")
Check "it really probed rather than pattern-matching the name" ($script:WoaProbes.Count -ge 2)

# Memoised: a second call must not re-probe.
$before = $script:WoaProbes.Count
$null = Get-WoaNvidiaSmiPath
Check "a second call is served from the memo" ($script:WoaProbes.Count -eq $before)

# When nothing answers it still returns the first that exists, as before.
$script:WoaNvidiaSmiProbed = $false
$script:WoaNvidiaSmiPath = $null
function Invoke-NvidiaSmiBounded { param($Exe, $Arguments) $global:LASTEXITCODE = 1; return "" }
Check "with nothing answering it falls back to the first existing copy" (
    (Get-WoaNvidiaSmiPath) -eq "C:\stale\nvidia-smi.exe")

# A null banner skips the CUDA-major guard in Initialize-WoaNativeCudaTorch, so a
# listing-only candidate must not win over one that names a version.
$script:WoaNvidiaSmiProbed = $false
$script:WoaNvidiaSmiPath = $null
function Invoke-NvidiaSmiBounded {
    param($Exe, $Arguments)
    $global:LASTEXITCODE = 0
    if ($Arguments -contains '-L') { return "GPU 0: NVIDIA GeForce RTX 4090" }
    if ($Exe -like "*versioned*") { return "CUDA Version: 12.8" }
    return "NVIDIA-SMI has failed because it couldn't communicate with the driver"
}
function Get-NvidiaSmiCandidatePaths {
    return @("C:\listingonly\nvidia-smi.exe", "C:\versioned\nvidia-smi.exe")
}
Check "a candidate that also names a CUDA version wins over a listing-only one" (
    (Get-WoaNvidiaSmiPath) -eq "C:\versioned\nvidia-smi.exe")

$script:WoaNvidiaSmiProbed = $false
$script:WoaNvidiaSmiPath = $null
function Invoke-NvidiaSmiBounded {
    param($Exe, $Arguments)
    $global:LASTEXITCODE = 0
    if ($Arguments -contains '-L') { return "GPU 0: NVIDIA GeForce RTX 4090" }
    return "no version here"
}
Check "with no version anywhere the first listing candidate is still used" (
    (Get-WoaNvidiaSmiPath) -eq "C:\listingonly\nvidia-smi.exe")

# Bounded as a whole: a wedged driver plus many DriverStore copies held the install.
Check "the ARM64 scan carries a soft and a hard deadline" (
    $woaFn -match '\$woaDeadline = \(Get-Date\)\.AddSeconds\(30\)' -and
    $woaFn -match '\$woaHardDeadline = \(Get-Date\)\.AddSeconds\(60\)')

$fastWoa = ($woaFn -replace 'AddSeconds\(30\)', 'AddSeconds(2)') -replace 'AddSeconds\(60\)', 'AddSeconds(4)'
Check "the shortened ARM64 slice really carries both (bites)" (
    $fastWoa -match 'AddSeconds\(2\)' -and $fastWoa -match 'AddSeconds\(4\)')
Invoke-Expression $fastWoa
$script:WoaNvidiaSmiProbed = $false
$script:WoaNvidiaSmiPath = $null
$script:WoaSlowProbes = 0
function Invoke-NvidiaSmiBounded {
    param($Exe, $Arguments)
    $script:WoaSlowProbes++
    Start-Sleep -Milliseconds 700
    $global:LASTEXITCODE = 1
    return ""
}
function Get-NvidiaSmiCandidatePaths { return @(1..20 | ForEach-Object { "C:\wedged$_\nvidia-smi.exe" }) }
$null = Get-WoaNvidiaSmiPath
Check "a wedged ARM64 host stops scanning instead of walking every candidate" (
    $script:WoaSlowProbes -lt 20)
Check "control: it probed rather than declining immediately" ($script:WoaSlowProbes -ge 2)
Invoke-Expression $woaFn
Remove-Item Function:\Get-Command -ErrorAction SilentlyContinue

if ($failures -gt 0) {
    Write-Host "$failures check(s) failed" -ForegroundColor Red
    exit 1
}
Write-Host "all checks passed" -ForegroundColor Green
