#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# A package index that lags PyPI must not make every install unsatisfiable (#14031): install.ps1 lowers
# its unsloth / unsloth-zoo floors to the newest release that index has, and only then. The Windows port
# of tests/sh/test_relax_unsloth_floors_to_index.sh; uv is a stub.
# Run: pwsh -NoProfile -File tests/studio/test_relax_unsloth_floors_to_index.ps1

$ErrorActionPreference = "Stop"
$root = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$installPath = Join-Path $root "install.ps1"

function Get-FunctionAsts([string]$Path, [string[]]$Names) {
    $tokens = $null; $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tokens, [ref]$errors)
    if ($errors) { $errors | ForEach-Object { $_.ToString() }; throw "$Path has parse errors" }
    foreach ($name in $Names) {
        $fn = $ast.FindAll({ param($n)
            $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name
        }, $true)
        if ($fn.Count -ne 1) { throw "expected exactly one $name in $Path, found $($fn.Count)" }
        $fn[0].Extent.Text
    }
}
foreach ($text in (Get-FunctionAsts $installPath @("Resolve-UnslothFloorsForIndex"))) {
    Invoke-Expression $text
}

$script:Pass = 0; $script:Fail = 0
function Assert-Eq([string]$Label, $Expected, $Actual) {
    if ("$Expected" -eq "$Actual") { Write-Host "  PASS: $Label"; $script:Pass++ }
    else { Write-Host "  FAIL: $Label (expected '$Expected', got '$Actual')"; $script:Fail++ }
}

$scratch = Join-Path ([System.IO.Path]::GetTempPath()) "relax-floors-$PID"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null
$marker = Join-Path $scratch "unsloth_feed_ceiling.json"
$VenvPython = Join-Path $scratch "python.exe"
$script:Substeps = @()
function substep([string]$Message, [string]$Color) { $script:Substeps += $Message }
foreach ($n in "UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_EXCLUDE_NEWER", "UV_EXCLUDE_NEWER_PACKAGE") {
    Remove-Item "Env:$n" -ErrorAction SilentlyContinue
}
$env:UV_CONFIG_FILE = ""; $env:APPDATA = $scratch; $env:ProgramData = $scratch
Push-Location $scratch

# The index serves unsloth $Feed.Unsloth and unsloth-zoo $Feed.Zoo; a floor above either is unsatisfiable.
$script:Feed = @{ Unsloth = "2026.10.3"; Zoo = "2026.10.3"; Down = $false }
$script:Calls = @()
$script:UvExe = {
    $script:Calls += ($args -join " ")
    $req = @($input) -join "`n"
    if ($script:Feed.Down) { $global:LASTEXITCODE = 2; return }
    if (($req -match 'unsloth>=2026\.10\.6' -or $req -match 'unsloth-zoo>=2026\.10\.6') -and
        -not ($script:Feed.Unsloth -eq "2026.10.6" -and $script:Feed.Zoo -eq "2026.10.6")) {
        $global:LASTEXITCODE = 1; return
    }
    $global:LASTEXITCODE = 0
    "unsloth==$($script:Feed.Unsloth)"
    "unsloth-zoo==$($script:Feed.Zoo)"
}
function Invoke-Resolve([string]$Desktop = "") {
    $script:Calls = @(); $script:Substeps = @()
    $release = if ($Desktop) { "unsloth>=$Desktop" } else { "unsloth>=2026.10.6" }
    Resolve-UnslothFloorsForIndex -ReleaseSpec $release -ZooSpec "unsloth-zoo>=2026.10.6" -DesktopVersion $Desktop -Marker $marker
}

Write-Host "=== index that has the release (PyPI): floors kept, one probe ==="
$script:Feed.Unsloth = "2026.10.6"; $script:Feed.Zoo = "2026.10.6"
Set-Content -LiteralPath $marker -Value "stale"
$r = Invoke-Resolve
Assert-Eq "release floor kept" "unsloth>=2026.10.6" $r.Release
Assert-Eq "one probe" 1 $script:Calls.Count
Assert-Eq "probe ignores dependencies" $true ($script:Calls[0] -like "pip compile --python $VenvPython --no-deps*")
Assert-Eq "nothing printed" 0 $script:Substeps.Count
Assert-Eq "a leftover marker is dropped" $false (Test-Path -LiteralPath $marker)

Write-Host "=== index a week behind: floors lowered to what it has ==="
$env:UV_DEFAULT_INDEX = "https://feed.example/simple"
$script:Feed.Unsloth = "2026.10.3"; $script:Feed.Zoo = "2026.10.2"
$r = Invoke-Resolve
Assert-Eq "release floor lowered" "unsloth>=2026.10.3" $r.Release
Assert-Eq "zoo floor lowered" "unsloth-zoo>=2026.10.2" $r.Zoo
Assert-Eq "says why" $true ($script:Substeps[0] -like "*does not have unsloth>=2026.10.6 yet; installing unsloth 2026.10.3*")
Assert-Eq "no marker without Desktop" $false (Test-Path -LiteralPath $marker)

Write-Host "=== Desktop on that index: the marker it reads ==="
$r = Invoke-Resolve -Desktop "2026.10.6"
Assert-Eq "release floor lowered" "unsloth>=2026.10.3" $r.Release
$m = Get-Content -LiteralPath $marker -Raw | ConvertFrom-Json
Assert-Eq "marker requested" "2026.10.6" $m.requested
Assert-Eq "marker available" "2026.10.3" $m.available
$bytes = [System.IO.File]::ReadAllBytes($marker)
Assert-Eq "marker has no BOM" $true ($bytes[0] -eq [byte][char]'{')

Write-Host "=== feed caught up: marker cleared on the next run ==="
$script:Feed.Unsloth = "2026.10.6"; $script:Feed.Zoo = "2026.10.6"
$r = Invoke-Resolve -Desktop "2026.10.6"
Assert-Eq "release floor kept" "unsloth>=2026.10.6" $r.Release
Assert-Eq "marker gone" $false (Test-Path -LiteralPath $marker)

Write-Host "=== unreachable index: floors kept ==="
$script:Feed.Down = $true
$r = Invoke-Resolve
Assert-Eq "release floor kept" "unsloth>=2026.10.6" $r.Release
Assert-Eq "zoo floor kept" "unsloth-zoo>=2026.10.6" $r.Zoo
$script:Feed.Down = $false

Write-Host "=== upload cutoff alone also counts ==="
Remove-Item Env:UV_DEFAULT_INDEX
$env:UV_EXCLUDE_NEWER = "2026-10-09T00:00:00Z"
$script:Feed.Unsloth = "2026.10.3"; $script:Feed.Zoo = "2026.10.3"
$r = Invoke-Resolve
Assert-Eq "release floor lowered" "unsloth>=2026.10.3" $r.Release
Remove-Item Env:UV_EXCLUDE_NEWER

Pop-Location
Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
Write-Host ""
Write-Host "Results: $($script:Pass) passed, $($script:Fail) failed"
if ($script:Fail -ne 0) { exit 1 }
