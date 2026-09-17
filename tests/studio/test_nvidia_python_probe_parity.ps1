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
    "New-StudioChildScriptDirectory",
    "Get-NvidiaNvmlLibraryPath",
    "Read-NvidiaLibraryRawViaPython",
    "Read-NvidiaLibraryRaw",
    "Get-NvidiaLibraryInventory",
    # Appended rather than inserted: the indices below are positional.
    "Test-StudioChildScriptDirectoryElevated"
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

# Every native symbol the probe calls, and the two attribute numbers. If the embedded probe and
# nvidia_probe.py ever disagree on one of these, they are probing different things.
#
# Two implementations now, not three: the emitted type that carried the same import list is
# gone, and comparing against its old index would compare the probe with itself.
$symbols = @(
    "nvmlInit_v2", "nvmlShutdown", "nvmlSystemGetCudaDriverVersion_v2",
    "nvmlDeviceGetCount_v2", "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetCudaComputeCapability",
    "cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetAttribute"
)
foreach ($symbol in $symbols) {
    Check "both the embedded probe and nvidia_probe.py call $symbol" `
        (($installProbe -match [regex]::Escape($symbol)) -and ($referenceProbe -match [regex]::Escape($symbol)))
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

Check "the embedded probe is stdlib-only" `
    (($installProbe -match '(?m)^import ctypes, os, sys$') -and ($installProbe -notmatch '(?m)^\s*(import|from)\s+(?!ctypes|os|sys)'))

Write-Host ""
Write-Host "=== the rung stays underneath the emitted one ==="

$rawBlock = & $strip $setupParts[3]
# ONE rung. The emitted UnslothNvidiaProbeV2 type is gone, so there is no first rung to sit
# behind, no shared deadline to split and no leftover budget to pass on. The reader hands the
# whole timeout to the interpreter and does nothing else.
Check "the reader delegates straight to the Python rung" `
    ($rawBlock -match 'Read-NvidiaLibraryRawViaPython -TimeoutMs \$TimeoutMs')
Check "the whole budget goes to it, not a remainder" ($rawBlock -notmatch 'remainingMs')
Check "there is no emitted rung left to attempt first" ($rawBlock -notmatch 'Get-NvidiaLibraryProbeType|\$native')
Check "a throw in the rung is an empty answer, not a failed install" ($rawBlock -match 'catch \{ return "" \}')
# The apparatus is gone from both files, not merely unused by this reader.
foreach ($file in @($installPs1, $setupPs1)) {
    $whole = [System.IO.File]::ReadAllText($file)
    Check "$(Split-Path -Leaf $file) defines no emitted native types at all" (
        $whole -notmatch 'New-StudioEmittedNativeType|New-StudioDynamicAssembly|DefinePInvokeMethod')
    Check "$(Split-Path -Leaf $file) no longer carries the capability gate they needed" (
        $whole -notmatch 'Test-StudioCanDefineNativeTypes')
}

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
# Constrained Language Mode permits property reads only on its documented allowed-type list, and
# System.Diagnostics.Process and System.IO.FileInfo are on neither. Reading .Id, .HasExited,
# .ExitCode or a path off a FileInfo throws on precisely the hosts where the emitted rung already
# declined, which is the whole population this rung serves. Handing the object to a cmdlet keeps
# the access inside compiled code. This is not hypothetical: the sibling early-python launcher
# shipped with exactly these reads and failed three checks on windows-latest under 5.1.
foreach ($blocked in @("\.Id\b", "\.HasExited\b", "\.ExitCode\b", "\.FullName\b", "New-TemporaryFile")) {
    Check "the launcher avoids $($blocked -replace '\\[.b]', '') , which CLM refuses on a Process or FileInfo" (
        $pyCode -notmatch $blocked)
}
Check "the launcher waits by object, not by pid" ($pyCode -match 'Wait-Process -InputObject \$proc')
Check "the launcher kills by object too" ($pyCode -match 'Stop-Process -InputObject \$proc')
Check "the timeout is detected through an error variable" ($pyCode -match '-ErrorVariable waitError')
# "[ref] - Casting an object to type [ref] ... is not permitted."
Check "the launcher casts nothing to [ref]" ($pyCode -notmatch '\[ref\]\s*\$')
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
Write-Host "=== the installer uses the interpreter this run installed ==="

# Get-StudioEarlyPython memoises, including a MISS. On a fresh host it is first called by the
# install-lock path, finds no Python, and caches $null for the rest of the run. The inventory is
# not read until thousands of lines later, by which point this install has created a managed
# interpreter and a venv. Asking only the cache would decline on exactly the fresh install where
# nvidia-smi is also most likely absent, and the host would take CPU wheels with a working card.
$hookSrc = @(Get-HelperSources $installPs1 @("Get-NvidiaProbePythonExe"))[0]
Invoke-Expression $hookSrc
$script:EarlyCalled = $false
function Get-StudioEarlyPython { $script:EarlyCalled = $true; return "/early/python3" }

$hookDir = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-hook-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $hookDir | Out-Null
try {
    $fakeVenv = Join-Path $hookDir "venv-python"
    $fakeManaged = Join-Path $hookDir "managed-python"
    [System.IO.File]::WriteAllText($fakeVenv, "")
    [System.IO.File]::WriteAllText($fakeManaged, "")

    $VenvPython = $fakeVenv; $ManagedPythonPath = $fakeManaged
    $script:EarlyCalled = $false
    Check "the venv interpreter wins when it exists" ((Get-NvidiaProbePythonExe) -eq $fakeVenv)
    Check "and the memoising ladder is not consulted at all" ($script:EarlyCalled -eq $false)

    # Before the venv is built but after the managed interpreter is installed.
    $VenvPython = Join-Path $hookDir "not-created-yet"
    $script:EarlyCalled = $false
    Check "the managed interpreter is next" ((Get-NvidiaProbePythonExe) -eq $fakeManaged)
    Check "and it too skips the ladder" ($script:EarlyCalled -eq $false)

    # Neither present: the ladder is the fallback, which is the pre-existing behaviour.
    $VenvPython = Join-Path $hookDir "nope-1"; $ManagedPythonPath = Join-Path $hookDir "nope-2"
    $script:EarlyCalled = $false
    Check "with neither on disk the ladder still answers" ((Get-NvidiaProbePythonExe) -eq "/early/python3")
    Check "and the ladder really was the source" ($script:EarlyCalled -eq $true)

    # A path recorded but never created must not be handed out: Test-Path is the discriminator,
    # not whether the variable happens to be set.
    $VenvPython = $null; $ManagedPythonPath = $null
    Check "unset variables fall through rather than returning empty" (
        (Get-NvidiaProbePythonExe) -eq "/early/python3")

    $saved = $env:UNSLOTH_EARLY_PYTHON_PROBE
    try {
        $env:UNSLOTH_EARLY_PYTHON_PROBE = "0"
        $VenvPython = $fakeVenv
        Check "the kill switch still wins over the venv interpreter" ((Get-NvidiaProbePythonExe) -eq "")
    } finally {
        if ($null -eq $saved) { Remove-Item Env:UNSLOTH_EARLY_PYTHON_PROBE -ErrorAction SilentlyContinue }
        else { $env:UNSLOTH_EARLY_PYTHON_PROBE = $saved }
    }
} finally {
    Remove-Item -LiteralPath $hookDir -Recurse -Force -ErrorAction SilentlyContinue
}

# The ordering that makes the above reachable at all: both variables are assigned before the
# first Get-NvidiaLibraryInventory call, or the hook would read $null however it is written.
$installText = [System.IO.File]::ReadAllText($installPs1)
$venvAt = $installText.IndexOf('$VenvPython = Join-Path $VenvDir')
$managedAt = $installText.IndexOf('$ManagedPythonPath = (Resolve-Path')
$firstInventoryAt = $installText.IndexOf('if (-not $HasNvidiaSmi -and (Get-NvidiaLibraryInventory))')
Check "the venv interpreter is named before the inventory is first read" (
    $venvAt -gt 0 -and $firstInventoryAt -gt $venvAt)
Check "the managed interpreter is named before it too" (
    $managedAt -gt 0 -and $firstInventoryAt -gt $managedAt)

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

Invoke-Expression ($setupParts[0])   # New-StudioChildScriptDirectory
Invoke-Expression ($setupParts[1])   # Get-NvidiaNvmlLibraryPath
Invoke-Expression ($setupParts[2])   # Read-NvidiaLibraryRawViaPython
Invoke-Expression ($setupParts[3])   # Read-NvidiaLibraryRaw
Invoke-Expression ($setupParts[4])   # Get-NvidiaLibraryInventory
Invoke-Expression ($setupParts[5])   # Test-StudioChildScriptDirectoryElevated
# The emitted rung declines, which is the whole point: this is what a Constrained Language Mode
# or WDAC host sees, and the capabilities below are recovered with nothing emitted.
function Get-NvidiaLibraryProbeType { return $null }

$script:PythonExe = ""
function Get-NvidiaProbePythonExe { return $script:PythonExe }

Check "no interpreter means no answer, not an error" ((Read-NvidiaLibraryRawViaPython -TimeoutMs 5000) -eq "")

# ---- where the program is written ----
#
# The program is written, closed, and then its PATHNAME is handed to Start-Process. Anything of
# the same user watching the directory can swap the file in that gap, and when this shell is
# elevated the child then runs that file with the administrator token. A directory of our own
# closes it: a name nothing can predict, and a High mandatory integrity label that a medium
# integrity process of the same user cannot write through. A DACL cannot express this, because
# the attacker is the OWNER and an owner can always rewrite its own DACL.
foreach ($file in @($installPs1, $setupPs1)) {
    $leaf = Split-Path -Leaf $file
    $dirFn = @(Get-HelperSources $file @("New-StudioChildScriptDirectory"))[0]
    Check "$leaf carries the private child-script directory" ($dirFn.Length -gt 100)
    Check "$leaf refuses a directory that already exists" (
        $dirFn -match 'New-Item -ItemType Directory[^\r\n]*-ErrorAction Stop' -and
        $dirFn -notmatch 'New-Item -ItemType Directory[^\r\n]*-Force')
    Check "$leaf raises the integrity label rather than trusting a DACL" (
        $dirFn -match 'icacls' -and $dirFn -match 'setintegritylevel')
    # icacls can be absent, blocked by application control, or fail for its own reasons, and none
    # of that throws. Setting the label and assuming it took hands back a directory that looks
    # protected and is not, which is the whole of the escalation this closes. So the label is read
    # back, and a run that could not raise it while elevated refuses instead of returning a path.
    Check "$leaf reads the label back rather than assuming it took" (
        $dirFn -match '\$labelled' -and $dirFn -match 'High Mandatory Level')
    Check "$leaf refuses an unlabelled directory when it is elevated" (
        $dirFn -match 'Test-StudioChildScriptDirectoryElevated' -and
        $dirFn -match 'Remove-Item[^\r\n]*\$dir')
    $elevFn = @(Get-HelperSources $file @("Test-StudioChildScriptDirectoryElevated"))[0]
    # Comments stripped first. The helper's own comment NAMES the construct it avoids, so the
    # check below read the explanation and failed on it rather than on any code.
    $elevCode = (($elevFn -split "`r?`n") | Where-Object { $_.Trim() -notmatch '^#' }) -join "`n"
    # whoami, not WindowsPrincipal: the managed identity types are not reachable under
    # Constrained Language Mode, which is the population this whole ladder exists for.
    Check "$leaf asks the token for its own label with an in-box tool" (
        $elevCode -match 'whoami' -and $elevCode -match 'S-1-16-')
    Check "$leaf does not use the managed principal types for it" (
        $elevCode -notmatch 'WindowsPrincipal')
    Check "$leaf comment stripper kept the code (bites)" ($elevCode -match 'return')
    # And every launcher in the file uses it, rather than naming the shared root itself. The
    # shared-root spelling is what the finding was about, so its absence is the check.
    $whole = [System.IO.File]::ReadAllText($file)
    Check "$leaf writes no child program straight into the shared temp root" (
        $whole -notmatch '\$stem = Join-Path \$tempRoot')
}
# A temp root with wildcard characters in it. "C:\Users\Mike [work]" is a legal profile path, and
# New-Item takes -Path with no -LiteralPath on 5.1, so the create there reads the brackets as a
# pattern. Nothing about such a host should cost it the Python rungs.
$bracketRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth [test] " + [guid]::NewGuid().ToString("N"))
$null = New-Item -ItemType Directory -Path $bracketRoot -Force
$savedTemp = $env:TEMP
$savedTmpdir = $env:TMPDIR
try {
    $env:TEMP = $bracketRoot
    $env:TMPDIR = $bracketRoot
    $bracketDir = New-StudioChildScriptDirectory
    Check "a temp root with brackets still yields a directory" (
        -not [string]::IsNullOrWhiteSpace($bracketDir))
    # The path it HANDED BACK has to be the one that exists. The escaped spelling can succeed at a
    # different path, so a helper that trusted New-Item not throwing would return a name with
    # nothing behind it and every write into it would fail.
    Check "and the path it returned is the directory that exists" (
        $bracketDir -and (Test-Path -LiteralPath $bracketDir -PathType Container))
    Check "and it is inside the bracketed root, not beside it" (
        "$bracketDir".StartsWith($bracketRoot))
    if ($bracketDir) { Remove-Item -LiteralPath $bracketDir -Recurse -Force -ErrorAction SilentlyContinue }
} finally {
    if ($null -eq $savedTemp) { Remove-Item Env:TEMP -ErrorAction SilentlyContinue } else { $env:TEMP = $savedTemp }
    if ($null -eq $savedTmpdir) { Remove-Item Env:TMPDIR -ErrorAction SilentlyContinue } else { $env:TMPDIR = $savedTmpdir }
    Remove-Item -LiteralPath $bracketRoot -Recurse -Force -ErrorAction SilentlyContinue
}

# Run it: a directory really is created, really is fresh, and really is cleaned up.
$madeDir = New-StudioChildScriptDirectory
Check "the directory is created" (-not [string]::IsNullOrWhiteSpace($madeDir))
if ($madeDir) {
    Check "and it is empty, so nothing was adopted" (
        @(Get-ChildItem -LiteralPath $madeDir -Force -ErrorAction SilentlyContinue).Count -eq 0)
    Check "and two calls never collide" ($madeDir -ne (New-StudioChildScriptDirectory))
    Remove-Item -LiteralPath $madeDir -Recurse -Force -ErrorAction SilentlyContinue
}

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
