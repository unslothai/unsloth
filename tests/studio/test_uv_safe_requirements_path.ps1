#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Unit test for install.ps1's Get-UvSafeRequirementsPath (issue #11012). uv splits -r on
# whitespace with no way to quote around it, so a requirements path under a user-chosen
# $RepoRoot or $VenvDir has to be made space-free before it is handed over.
#
# The contract under test: a space-free path is returned untouched and never marked temporary,
# a spaced path never comes back spaced while any space-free writable directory exists, a copy
# is reported as temporary so only a copy is ever deleted, and when nothing space-free can be
# found the original comes back with a warning rather than silently.
# Run: pwsh -NoProfile -File tests/studio/test_uv_safe_requirements_path.ps1

$ErrorActionPreference = "Stop"
$installPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1")
$installPath = (Resolve-Path $installPath).Path

# Parse install.ps1 (also a syntax gate) and extract the helper.
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "install.ps1 has parse errors" }

$fn = $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
    $n.Name -eq "Get-UvSafeRequirementsPath"
}, $true)
if ($fn.Count -ne 1) { throw "expected exactly one Get-UvSafeRequirementsPath in install.ps1, found $($fn.Count)" }
Invoke-Expression $fn[0].Extent.Text

# The helper reports the give-up case through the installer's own output helper. Capture it here
# rather than stubbing it away, so the warning is part of the contract and not an implementation
# detail a later edit can drop unnoticed.
$script:Warnings = @()
function substep { param([string]$Text, [string]$Colour) $script:Warnings += $Text }

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$root = Join-Path ([System.IO.Path]::GetTempPath()) ("uvsafe-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $root | Out-Null
try {
    # 1. A path with no space is the common case and must be handed through untouched.
    $plainDir = Join-Path $root "plain"
    New-Item -ItemType Directory -Force -Path $plainDir | Out-Null
    $plain = Join-Path $plainDir "no-torch-runtime.txt"
    "typer" | Set-Content -LiteralPath $plain
    $r = Get-UvSafeRequirementsPath -Path $plain
    Check "a space-free path is returned unchanged" ($r.Path -eq $plain)
    Check "a space-free path is never marked temporary" (-not $r.Temporary)
    Check "the untouched path emits no warning" ($script:Warnings.Count -eq 0)

    # 2. A spaced path must not come back spaced while somewhere space-free is writable.
    $spacedDir = Join-Path $root "has space"
    New-Item -ItemType Directory -Force -Path $spacedDir | Out-Null
    $spaced = Join-Path $spacedDir "no-torch-runtime.txt"
    "typer" | Set-Content -LiteralPath $spaced
    $r2 = Get-UvSafeRequirementsPath -Path $spaced
    Check "a spaced path never comes back containing a space" (-not $r2.Path.Contains(" "))
    Check "the space-free path carries the same content" ((Get-Content -Raw -LiteralPath $r2.Path).Trim() -eq "typer")

    # 3. Only a copy may be deleted. Deleting what the helper reported must leave the user's file.
    #    There are TWO legitimate ways to get a space-free path, and which one happens is a
    #    property of the volume, not of the helper: where 8.3 creation is enabled the helper
    #    returns the ALIAS OF THE USER'S OWN FILE and reports Temporary = $false, because deleting
    #    it would delete the user's file; where 8.3 is unavailable it copies and reports
    #    Temporary = $true. Asserting only the copy made this test fail on any Windows runner with
    #    8.3 enabled, which is a fact about that runner rather than a defect. The invariant that
    #    actually has to hold on both is: Temporary tells the caller whether the path is safe to
    #    delete, and the user's file survives either way.
    #
    #    Identity, not spelling, on BOTH branches. An alias differs from the long name as a string
    #    while naming one file, so a string comparison cannot tell "a copy" from "the user's file
    #    under another name" -- and getting that backwards is precisely the mistake that deletes
    #    someone's requirements file.
    function Test-SameUnderlyingFile([string]$a, [string]$b) {
        try {
            if ((Get-Item -LiteralPath $a -Force).FullName -eq (Get-Item -LiteralPath $b -Force).FullName) { return $true }
        } catch { return $false }
        # Different strings can still be one file. Write through one, read back through the other,
        # and put the content back. Confined to this test's own fixture.
        $probe = "typer-probe-" + [guid]::NewGuid().ToString("N")
        try {
            $original = Get-Content -Raw -LiteralPath $b
            $probe | Set-Content -LiteralPath $b
            $same = ((Get-Content -Raw -LiteralPath $a).Trim() -eq $probe)
            $original.TrimEnd("`r", "`n") | Set-Content -LiteralPath $b
            return $same
        } catch { return $false }
    }

    if ($r2.Temporary) {
        # Reported safe to delete, so it had better not be the user's file.
        Check "a copy is not the user's own file" (-not (Test-SameUnderlyingFile $r2.Path $spaced))
        Remove-Item -LiteralPath $r2.Path -Force
        Check "deleting the copy leaves the original in place" (Test-Path -LiteralPath $spaced)
    } else {
        # Reported unsafe to delete, so it had better BE the user's file, reached by another name.
        Check "a non-temporary path is the user's own file under another name" (
            Test-SameUnderlyingFile $r2.Path $spaced)
        Check "the user's file is still there and untouched" (
            (Test-Path -LiteralPath $spaced) -and ((Get-Content -Raw -LiteralPath $spaced).Trim() -eq "typer"))
    }

    # 4. Every candidate unusable: the original comes back, and the user is told why. Without the
    #    warning this case is indistinguishable from success until uv fails with a message that
    #    names a fragment of the install root, which is what made #11012 hard to read.
    $script:Warnings = @()
    # TMPDIR as well as TEMP/TMP: .NET reads TMPDIR for GetTempPath off Windows, so without it
    # this case quietly takes the success branch on Linux and never exercises the warning.
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $env:TEMP = $spacedDir
        $env:TMP = $spacedDir
        $env:TMPDIR = $spacedDir
        $r3 = Get-UvSafeRequirementsPath -Path $spaced
        if ($r3.Path.Contains(" ")) {
            Check "the original is returned when nowhere space-free is usable" ($r3.Path -eq $spaced)
            Check "the give-up path is not marked temporary" (-not $r3.Temporary)
            Check "the give-up path warns the user" (($script:Warnings -join " ") -match "space")
            Check "the warning names the variables to change" (($script:Warnings -join " ") -match "TMP")
        } else {
            # A host that still found somewhere space-free (an 8.3 name, or a space-free drive
            # root) is a pass for the contract that matters: the path handed to uv has no space.
            Check "a space-free path was still found, which satisfies the contract" $true
        }
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
    # 5. A failed copy must return the original, report it as not temporary, and leave nothing
    #    behind. Every candidate points at a path that exists as a FILE, so Copy-Item throws.
    #
    #    Honest limit: in this scenario Copy-Item fails before creating anything, so these two
    #    checks pass with or without the cleanup in the catch and do not prove it. They pin the
    #    reachable contract. The cleanup exists for a copy that fails PARTWAY, a full disk or an
    #    interrupted write, which leaves a partial destination; that cannot be forced portably
    #    here, so it is defensive rather than covered.
    $script:Warnings = @()
    $blocker = Join-Path $root "blocker"
    "not a directory" | Set-Content -LiteralPath $blocker
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $env:TEMP = $blocker
        $env:TMP = $blocker
        $env:TMPDIR = $blocker
        $before = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "unsloth-reqs-*").Count
        $r4 = Get-UvSafeRequirementsPath -Path $spaced
        $after = @(Get-ChildItem -LiteralPath $root -Recurse -File -Filter "unsloth-reqs-*").Count
        Check "a failed copy leaves no stray unsloth-reqs file" ($after -eq $before)
        Check "a failed copy is never reported as temporary" (-not $r4.Temporary)
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
    # 6. A relative or malformed temp path must not abort the install. install.ps1 runs under
    #    ErrorActionPreference = "Stop", so a throwing expression in the candidate list would be
    #    evaluated before the loop and terminate the whole run, which is worse than the split
    #    path this helper exists to prevent. GetPathRoot returns "" for a relative path and
    #    Join-Path then rejects it, so this is reachable from a machine with an odd TMP.
    $script:Warnings = @()
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $ErrorActionPreference = "Stop"
        $env:TEMP = "relative-temp-dir"
        $env:TMP = "relative-temp-dir"
        $env:TMPDIR = "relative-temp-dir"
        $threw = $false
        try { $r5 = Get-UvSafeRequirementsPath -Path $spaced } catch { $threw = $true }
        Check "a relative temp path does not abort the helper" (-not $threw)
        if (-not $threw) {
            Check "a relative temp path still yields a usable result" ($null -ne $r5.Path)
        }
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
    # 7. A temp path on an unmapped drive must not abort the install either. Join-Path resolves
    #    the drive qualifier, so it throws DriveNotFoundException for a stale or disconnected
    #    drive, and under ErrorActionPreference = "Stop" that would terminate the run before the
    #    loop could reach a candidate that works.
    $script:Warnings = @()
    $savedTemp = $env:TEMP; $savedTmp = $env:TMP; $savedTmpDir = $env:TMPDIR
    try {
        $ErrorActionPreference = "Stop"
        $env:TEMP = "Z:\stale"
        $env:TMP = "Z:\stale"
        $env:TMPDIR = "Z:\stale"
        $threw = $false
        try { $r6 = Get-UvSafeRequirementsPath -Path $spaced } catch { $threw = $true }
        Check "an unmapped temp drive does not abort the helper" (-not $threw)
        if (-not $threw) {
            Check "an unmapped temp drive still yields a result" ($null -ne $r6.Path)
        }
    } finally {
        $env:TEMP = $savedTemp; $env:TMP = $savedTmp; $env:TMPDIR = $savedTmpDir
    }
} finally {
    Remove-Item -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "All Get-UvSafeRequirementsPath checks passed"
exit 0
