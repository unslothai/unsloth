# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Runs INSIDE a Windows container, proving it is genuinely virgin BEFORE anything is
# installed. That is the whole point of the container lane: the hosted Windows legs of
# clean-machine-install-ci.yml only SIMULATE absence (rename the toolcache Python, scrub
# the registry PATH), while this asserts real absence on an image that never had a
# toolchain. Without it the lane proves nothing the masked legs already do.

$ErrorActionPreference = 'Continue'
$failures = @()

function Section($t) { Write-Host ""; Write-Host "=== $t ===" }

# ── The interpreter itself ────────────────────────────────────────────────────
# install.ps1 must run under Windows PowerShell 5.1, what a real Windows box ships; pwsh
# 7 is a runner-image extra. nanoserver has neither, hence servercore.
Section 'interpreter'
Write-Host "PSVersion  : $($PSVersionTable.PSVersion)"
Write-Host "PSEdition  : $($PSVersionTable.PSEdition)"
Write-Host "CLRVersion : $($PSVersionTable.CLRVersion)"
Write-Host "Host       : $($Host.Name)"
if ($PSVersionTable.PSEdition -ne 'Desktop') {
    $failures += "PSEdition is '$($PSVersionTable.PSEdition)', not Desktop -- this is not Windows PowerShell 5.1"
}
if ($PSVersionTable.PSVersion.Major -ne 5) {
    $failures += "PSVersion is $($PSVersionTable.PSVersion), not 5.x"
}

Section 'operating system'
cmd /c ver
$cv = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' -ErrorAction SilentlyContinue
if ($cv) {
    Write-Host "ProductName    : $($cv.ProductName)"
    Write-Host "EditionID      : $($cv.EditionID)"
    Write-Host "InstallationType: $($cv.InstallationType)"
    Write-Host "CurrentBuild   : $($cv.CurrentBuild).$($cv.UBR)"
}
Write-Host "USERNAME       : $env:USERNAME"
Write-Host "USERPROFILE    : $env:USERPROFILE"
Write-Host "LOCALAPPDATA   : $env:LOCALAPPDATA"
Write-Host "PROCESSOR_ARCH : $env:PROCESSOR_ARCHITECTURE"

# install.ps1:254/258 joins $env:USERPROFILE with no null guard, so an unset USERPROFILE
# aborts under ErrorActionPreference=Stop. The lane sets UNSLOTH_STUDIO_HOME; record
# whether a bare container would have survived without it.
if ([string]::IsNullOrWhiteSpace($env:USERPROFILE)) {
    Write-Host "::warning::USERPROFILE is unset in this container; install.ps1's default install root would abort"
}

# ── The assertion the whole lane exists for ───────────────────────────────────
Section 'virginity: developer toolchain must be ABSENT'
# uv is on the list because install.ps1 would reuse a preinstalled one and skip its own
# bootstrap.
$mustBeAbsent = @('python', 'python3', 'py', 'git', 'cmake', 'cl', 'winget', 'uv')
foreach ($t in $mustBeAbsent) {
    $c = Get-Command $t -ErrorAction SilentlyContinue
    $where = if ($c) { $c.Source } else { 'ABSENT' }
    Write-Host ("  {0,-10} {1}" -f $t, $where)
    if ($c) { $failures += "$t is present at $($c.Source) -- this container is NOT virgin" }
}

Section 'informational: present but not a developer toolchain'
# OS components, not a toolchain. curl.exe and tar.exe ship in System32 and are the only
# transport into a container with no git; naming them beats silently relying on them.
foreach ($t in 'cmd', 'powershell', 'curl', 'tar', 'certutil', 'msiexec', 'reg', 'where', 'pwsh', 'node', 'npm', 'msbuild', 'dotnet', 'gcc') {
    $c = Get-Command $t -ErrorAction SilentlyContinue
    Write-Host ("  {0,-10} {1}" -f $t, $(if ($c) { $c.Source } else { 'ABSENT' }))
}

Section 'virginity: no toolchain on disk either'
# A binary can be off PATH and still be found by uv's discovery or py.exe's registry
# view -- how the hosted leg once reported `python ABSENT` then installed with the
# runner's 3.13.14. So check disk and registry too.
$badPaths = @(
    'C:\Python27', 'C:\Python3*', 'C:\Program Files\Python*', 'C:\Program Files (x86)\Python*',
    'C:\Program Files\Git', 'C:\Program Files\CMake', 'C:\Program Files\Microsoft Visual Studio',
    'C:\Program Files (x86)\Microsoft Visual Studio', 'C:\hostedtoolcache', 'C:\ProgramData\chocolatey'
)
foreach ($p in $badPaths) {
    # Wildcards can match several dirs; take the first so the message names a real path.
    $hit = @(Get-Item -Path $p -ErrorAction SilentlyContinue) | Select-Object -First 1
    if ($hit) {
        Write-Host "  PRESENT  $($hit.FullName)"
        $failures += "toolchain directory exists on disk: $($hit.FullName)"
    } else {
        Write-Host "  absent   $p"
    }
}

$pyReg = @('HKLM:\SOFTWARE\Python', 'HKCU:\SOFTWARE\Python')
foreach ($k in $pyReg) {
    if (Test-Path $k) {
        Write-Host "  PRESENT  $k"
        $failures += "a registered Python install exists at $k"
    } else {
        Write-Host "  absent   $k"
    }
}

Section 'PATH as the container sees it'
Write-Host "Process PATH:"
($env:PATH -split ';') | Where-Object { $_ } | ForEach-Object { Write-Host "  $_" }
foreach ($scope in 'Machine', 'User') {
    Write-Host "$scope PATH: $([System.Environment]::GetEnvironmentVariable('Path', $scope))"
}

# ── The VC++ runtime question the hosted leg cannot answer ────────────────────
Section 'VC++ runtime (honest measurement)'
# The hosted image ships the VC++ 2015-2022 runtime in System32 and cannot lose it
# without breaking the runner (see the HONESTY NOTE in clean-machine-install-ci.yml), so
# `import torch` succeeding there does not prove a no-winget machine has it. This
# container is the only place in CI that can answer, so absence is ASSERTED, not
# recorded: a base image that starts shipping them would silently make this a masked leg.
foreach ($dll in 'vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll') {
    $p = Join-Path $env:WINDIR "System32\$dll"
    $present = Test-Path $p
    Write-Host ("  {0,-20} {1}" -f $dll, $(if ($present) { 'PRESENT' } else { 'ABSENT' }))
    if ($present) { $failures += "System32\$dll is present -- this image already ships the VC++ runtime, which is the one thing the hosted runner cannot un-ship" }
}
foreach ($k in 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64',
               'HKLM:\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64') {
    $r = Get-ItemProperty $k -ErrorAction SilentlyContinue
    Write-Host ("  {0} -> {1}" -f $k, $(if ($r) { "Installed=$($r.Installed) $($r.Major).$($r.Minor)" } else { 'absent' }))
}

# ── Can the installer's transport work at all here? ───────────────────────────
Section 'outbound HTTPS and TLS'
# install.ps1 never sets [Net.ServicePointManager]::SecurityProtocol, so it inherits the
# .NET Framework default. Test that first: default failing where Tls12 works is a real
# installer portability bug, not a container quirk.
Write-Host "default SecurityProtocol: $([Net.ServicePointManager]::SecurityProtocol)"
$probeUrls = @(
    'https://www.python.org/ftp/python/',
    'https://astral.sh/uv/install.ps1',
    'https://pypi.org/simple/',
    'https://aka.ms/vs/17/release/vc_redist.x64.exe'
)
$defaultOk = @{}
foreach ($u in $probeUrls) {
    try {
        $null = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 60 -Method Head -ErrorAction Stop
        Write-Host "  OK (default TLS)   $u"; $defaultOk[$u] = $true
    } catch {
        Write-Host "  FAIL (default TLS) $u -- $($_.Exception.Message)"; $defaultOk[$u] = $false
    }
}
if ($defaultOk.Values -contains $false) {
    Write-Host "retrying the failures with an explicit Tls12..."
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    foreach ($u in $probeUrls) {
        if ($defaultOk[$u]) { continue }
        try {
            $null = Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 60 -Method Head -ErrorAction Stop
            Write-Host "  OK (Tls12)   $u"
            Write-Host "::warning::$u needs an explicit Tls12; install.ps1 never sets SecurityProtocol, so this is a real installer portability gap"
        } catch {
            Write-Host "  FAIL (Tls12) $u -- $($_.Exception.Message)"
            $failures += "no outbound HTTPS to $u even with Tls12 -- the container cannot reach the installer's download hosts"
        }
    }
}

# ── Verdict ───────────────────────────────────────────────────────────────────
Section 'verdict'
if ($failures.Count -gt 0) {
    foreach ($f in $failures) { Write-Host "::error::$f" }
    Write-Host "VIRGINITY ASSERTION FAILED ($($failures.Count) problem(s))"
    exit 1
}
Write-Host "VIRGINITY ASSERTION PASSED"
Write-Host "no python, py, git, cmake, cl, winget or uv on PATH, on disk, or in the registry"
exit 0
