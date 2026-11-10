#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# Invoke-InstallCommandRetry clears wheels uv calls invalid from its cache (#17375).

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$installPath = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", "..", "install.ps1"))).Path
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($installPath, [ref]$tokens, [ref]$errors)
if ($errors) { throw "install.ps1 has parse errors" }
foreach ($name in 'Redact-InstallOutput', 'Invoke-InstallCommand', 'Invoke-InstallMirrorRetry', 'Repair-UvDamagedWheels', 'Invoke-InstallCommandRetry') {
    $fn = $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $n.Name -eq $name }, $true)
    Invoke-Expression $fn[0].Extent.Text
}
function Write-UvDownloadMarker { param($Line) }
function Clear-TauriInstallError { param($Message) }
function Write-TauriLog { param($Kind, $Message) }
function Write-StudioLine { param($Text, $ForegroundColor) }
$script:Substeps = [System.Collections.Generic.List[string]]::new()
function substep { param($Message, $Color) $script:Substeps.Add($Message) }
$script:InstallTorchMirror = $null
$script:UnslothVerbose = $false
$script:UvExe = 'uv'
$env:_UNSLOTH_MIRROR_SPARE = $null
$env:UNSLOTH_INSTALL_RETRY_DELAY = '0'

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" } else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

Write-Host "=== Repair-UvDamagedWheels ==="
$script:UvExe = { param([Parameter(ValueFromRemainingArguments)]$a) $script:Cleans.Add(($a -join ' ')); $global:LASTEXITCODE = 0 }
$script:Cleans = [System.Collections.Generic.List[string]]::new()
# Verbatim from #17375 (CRLF, as Windows collects it), plus the older "Caused by:" spelling.
$issue = (@(
        'Resolved 87 packages in 227ms',
        'error: Failed to install: tqdm-4.70.1-py3-none-any.whl (tqdm==4.70.1)',
        '  cause: The wheel is invalid: Missing .dist-info directory',
        'error: Failed to install: Nest_Asyncio-1.6.0-py3-none-any.whl (Nest-Asyncio==1.6.0)',
        '  Caused by: The wheel is invalid: Metadata field Name not found',
        'error: Failed to install: torch-2.10.0-cp313-cp313-win_amd64.whl (torch==2.10.0)',
        '  cause: failed to hardlink file: Access is denied. (os error 5)'
    ) -join "`r`n")
$cleaned = @{}
Check "repairs what uv called invalid wheels" (Repair-UvDamagedWheels -Output $issue -Cleaned $cleaned)
Check "cleans exactly tqdm and nest-asyncio" ($script:Cleans.Count -eq 1 -and $script:Cleans[0] -eq 'cache clean tqdm nest-asyncio')
Check "a package already cleaned is not cleaned again" (-not (Repair-UvDamagedWheels -Output $issue -Cleaned $cleaned) -and $script:Cleans.Count -eq 1)
Check "a network failure is not a damaged wheel" (-not (Repair-UvDamagedWheels -Output 'error: Request failed after 3 retries: error sending request for url (https://pypi.org/simple/tqdm/)' -Cleaned @{}))
Check "empty output is not a damaged wheel" (-not (Repair-UvDamagedWheels -Output '' -Cleaned @{}))

Write-Host "=== real uv, damaged cache archive ==="
$script:UvExe = 'uv'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Write-Host "  FAIL  uv is not on PATH (CI installs it)" -ForegroundColor Red; exit 1 }
$python = if ($env:OS -ne 'Windows_NT' -and (Get-Command python3 -ErrorAction SilentlyContinue)) { 'python3' } else { 'python' }
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("uvdamaged-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $work | Out-Null
$savedCache = $env:UV_CACHE_DIR
try {
    $wheels = Join-Path $work "wheels"
    $maker = @'
import os, sys, zipfile
os.makedirs(sys.argv[1])
for name in sys.argv[2:]:
    di = f"{name}-1.0.dist-info"
    files = {f"{name}/__init__.py": "", f"{di}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n",
             f"{di}/WHEEL": "Wheel-Version: 1.0\nGenerator: t\nRoot-Is-Purelib: true\nTag: py3-none-any\n"}
    files[f"{di}/RECORD"] = "".join(f"{p},,\n" for p in files) + f"{di}/RECORD,,\n"
    with zipfile.ZipFile(os.path.join(sys.argv[1], f"{name}-1.0-py3-none-any.whl"), "w") as z:
        for p, c in files.items():
            z.writestr(p, c)
'@
    $makerPath = Join-Path $work "make_wheels.py"
    [System.IO.File]::WriteAllText($makerPath, $maker)
    & $python $makerPath $wheels alpha_pkg beta_pkg
    $env:UV_CACHE_DIR = Join-Path $work "cache"
    $ErrorActionPreference = "Continue"
    & uv venv -q (Join-Path $work "v1") 2>&1 | Out-Null
    & uv venv -q (Join-Path $work "v2") 2>&1 | Out-Null
    $ErrorActionPreference = "Stop"
    $bin = if ($env:OS -eq 'Windows_NT') { 'Scripts\python.exe' } else { 'bin/python' }
    $py1 = Join-Path (Join-Path $work "v1") $bin
    $py2 = Join-Path (Join-Path $work "v2") $bin
    $ErrorActionPreference = "Continue"
    & uv pip install -q --python $py1 --no-index --find-links $wheels alpha_pkg beta_pkg 2>&1 | Out-Null
    $ErrorActionPreference = "Stop"
    foreach ($d in @(Get-ChildItem -Path (Join-Path $env:UV_CACHE_DIR "archive-v0") -Directory | ForEach-Object { Get-ChildItem -LiteralPath $_.FullName -Directory -Filter "*.dist-info" })) {
        Remove-Item -LiteralPath $d.FullName -Recurse -Force
    }
    $ErrorActionPreference = "Continue"
    $base = (& uv pip install -q --python $py2 --no-index --find-links $wheels alpha_pkg beta_pkg 2>&1 | ForEach-Object { "$_" }) -join "`n"
    $baseExit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    Check "control: the damaged cache fails a plain install" ($baseExit -ne 0 -and $base -match 'The wheel is invalid')
    $env:UNSLOTH_INSTALL_RETRIES = '1'
    $script:Substeps.Clear()
    $code = Invoke-InstallCommandRetry -Label "install unsloth" { & $script:UvExe pip install --python $py2 --no-index --find-links $wheels alpha_pkg beta_pkg }
    Check "retry loop repairs the cache and installs (exit $code)" ($code -eq 0)
    Check "one cache clean per damaged package" (@($script:Substeps | Where-Object { $_ -like 'clearing damaged uv cache entries*' }).Count -eq 2)
    $ErrorActionPreference = "Continue"
    & $py2 -c "import alpha_pkg, beta_pkg" 2>&1 | Out-Null
    $importExit = $LASTEXITCODE
    $ErrorActionPreference = "Stop"
    Check "both packages import" ($importExit -eq 0)
} finally {
    $env:UV_CACHE_DIR = $savedCache
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures) { Write-Host "$failures failure(s)" -ForegroundColor Red; exit 1 }
Write-Host "all passed"
