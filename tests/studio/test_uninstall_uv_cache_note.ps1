#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A recorded uv cache outside every removed root is named, one inside is not.
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_uv_cache_note.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors)
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)
$allFns = $ast.FindAll({
        param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
    }, $true)
foreach ($name in @("_RecordedUvCache", "_UvCacheUnderRoot", "_UvLeftoverCaches")) {
    $fn = $allFns | Where-Object { $_.Name -eq $name } | Select-Object -First 1
    if (-not $fn) {
        Write-Host "  FAIL  $name not found in uninstall.ps1" -ForegroundColor Red
        exit 1
    }
    . ([scriptblock]::Create($fn.Extent.Text))
}

if ([System.IO.Path]::DirectorySeparatorChar -ne '\') {
    Write-Host "  SKIP  uninstall.ps1 joins Windows paths; run on Windows"
    exit 0
}

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-uv-note-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
try {
    function Root([string]$Name, $Recorded) {
        $root = Join-Path $tmp $Name
        New-Item -ItemType Directory -Path (Join-Path $root "cache") -Force | Out-Null
        if ($null -ne $Recorded) {
            [System.IO.File]::WriteAllText((Join-Path $root "cache\uv-cache-dir"), $Recorded + "`r`n")
        }
        return $root
    }

    $shared = Join-Path $tmp "shared-uv"
    $a = Root "a" $shared
    $r = _UvLeftoverCaches @($a)
    Check "shared cache is a leftover" ($r.Saw -and $r.Leftovers.Count -eq 1 -and $r.Leftovers[0] -eq $shared)

    $own = Join-Path (Join-Path $tmp "b") "cache\uv"
    $b = Root "b" $own
    $r = _UvLeftoverCaches @($b)
    Check "Studio-owned cache is silent" ($r.Saw -and $r.Leftovers.Count -eq 0)

    $r = _UvLeftoverCaches @((Root "c" ((Join-Path $tmp "c").Replace('\', '/') + "/cache/uv")))
    Check "forward-slash cache under the root is silent" ($r.Saw -and $r.Leftovers.Count -eq 0)

    $d = Root "d" (Join-Path (Join-Path $tmp "b") "cache\uv")
    $r = _UvLeftoverCaches @($b, $d)
    Check "cache under another removed root is silent" ($r.Leftovers.Count -eq 0)

    $r = _UvLeftoverCaches @((Root "e" $null))
    Check "no marker: nothing seen, nothing named" (-not $r.Saw -and $r.Leftovers.Count -eq 0)

    $r = _UvLeftoverCaches @()
    Check "no roots" (-not $r.Saw -and $r.Leftovers.Count -eq 0)

    $target = Join-Path $tmp "real"
    $link = Join-Path $tmp "linked"
    New-Item -ItemType Directory -Path (Join-Path $target "cache") -Force | Out-Null
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    $linkedCache = Join-Path $link "cache\uv"
    [System.IO.File]::WriteAllText((Join-Path $target "cache\uv-cache-dir"), $linkedCache + "`n")
    $r = _UvLeftoverCaches @($link)
    Check "junctioned root: its cache is named" ($r.Leftovers.Count -eq 1 -and $r.Leftovers[0] -eq $linkedCache)
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host ""; Write-Host "FAILED ($failures)" -ForegroundColor Red; exit 1 }
Write-Host ""; Write-Host "All tests passed."; exit 0
