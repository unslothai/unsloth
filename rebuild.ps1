#Requires -Version 5.1
# SPDX-License-Identifier: Apache 2.0
<#
.SYNOPSIS
    Clean rebuild of Unsloth Studio (frontend + backend)
.DESCRIPTION
    Removes frontend build artifacts (dist/, node_modules/), 
    deletes the Python virtual environment, and runs a fresh
    local install. Useful for developers working on Studio code.
.EXAMPLE
    .\rebuild.ps1
    Full clean rebuild with normal output.
.EXAMPLE
    .\rebuild.ps1 -Verbose
    Full clean rebuild with detailed output.
#>

param(
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"

# Set verbose env var for child processes if requested
if ($Verbose) {
    $env:UNSLOTH_VERBOSE = '1'
}

Write-Host "`n=== Unsloth Studio Clean Rebuild ===" -ForegroundColor Cyan
Write-Host ""

# ── Clean Frontend ──
Write-Host "[1/3] Cleaning frontend build artifacts..." -ForegroundColor Yellow

$FrontendDist = "studio\frontend\dist"
$FrontendNodeModules = "studio\frontend\node_modules"

if (Test-Path $FrontendDist) {
    Write-Host "  → Removing $FrontendDist" -ForegroundColor Gray
    Remove-Item -Recurse -Force $FrontendDist -ErrorAction SilentlyContinue
    Write-Host "  ✓ Deleted frontend dist" -ForegroundColor Green
} else {
    Write-Host "  → $FrontendDist not found (already clean)" -ForegroundColor Gray
}

if (Test-Path $FrontendNodeModules) {
    Write-Host "  → Removing $FrontendNodeModules (this may take a moment...)" -ForegroundColor Gray
    Remove-Item -Recurse -Force $FrontendNodeModules -ErrorAction SilentlyContinue
    Write-Host "  ✓ Deleted node_modules" -ForegroundColor Green
} else {
    Write-Host "  → node_modules not found (already clean)" -ForegroundColor Gray
}

# ── Clean Python Environment ──
Write-Host "`n[2/3] Cleaning Python virtual environment..." -ForegroundColor Yellow

$VenvPath = Join-Path $env:USERPROFILE ".unsloth\studio\unsloth_studio"

if (Test-Path $VenvPath) {
    Write-Host "  → Removing $VenvPath" -ForegroundColor Gray
    Remove-Item -Recurse -Force $VenvPath -ErrorAction SilentlyContinue
    Write-Host "  ✓ Deleted venv successfully" -ForegroundColor Green
} else {
    Write-Host "  → Venv not found (already clean)" -ForegroundColor Gray
}

# ── Rebuild Everything ──
Write-Host "`n[3/3] Running fresh local install..." -ForegroundColor Yellow
Write-Host ""

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
& .\install.ps1 --local

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Rebuild Complete! ===" -ForegroundColor Green
} else {
    Write-Host "`n=== Rebuild Failed ===" -ForegroundColor Red
    exit $LASTEXITCODE
}
