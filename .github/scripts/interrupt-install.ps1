# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Run install.ps1 and kill its whole process TREE partway through, like quitting the desktop app.
# Usage:
#   pwsh -File .github/scripts/interrupt-install.ps1 -Marker 'studio deps' `
#        -LogPath logs/install.log -InstallArgs '--tauri --no-torch --local'
[CmdletBinding()]
param(
  [string]$Marker = '',
  [string]$LogPath = 'logs/install.log',
  [string]$InstallArgs = '',
  [int]$KillAtSeconds = 900
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
Set-Content -Path $LogPath -Value '' -Encoding utf8

# Stand in for the desktop app, which writes this marker before spawning the installer.
# Both roots: Rust hardcodes ~/.unsloth/studio, CI overrides UNSLOTH_STUDIO_HOME. Never cleared.
foreach ($dir in @($env:UNSLOTH_STUDIO_HOME, (Join-Path $HOME '.unsloth\studio'))) {
  if ([string]::IsNullOrWhiteSpace($dir)) { continue }
  try {
    New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
    Set-Content -Path (Join-Path $dir '.desktop-install-in-progress') -Value '' -ErrorAction Stop
  } catch { Write-Host "[interrupt] could not seed install marker in ${dir}: $_" }
}

# Windows PowerShell 5.1 with install.rs's exact flags: the only host a real desktop install uses.
$argList = @(
  '-NoLogo', '-NoProfile', '-NonInteractive',
  '-WindowStyle', 'Hidden',
  '-ExecutionPolicy', 'Bypass',
  '-File', 'install.ps1'
)
if ($InstallArgs) { $argList += $InstallArgs.Split(' ') }
$proc = Start-Process -FilePath 'powershell.exe' -ArgumentList $argList `
  -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.err" `
  -PassThru -NoNewWindow
Write-Host "[interrupt] installer pid=$($proc.Id) marker='$Marker' deadline=${KillAtSeconds}s"

# Proves the kill was delivered: a natural failure also exits non-zero on Windows.
$script:rootKilled = $false

function Get-Descendants([int]$RootId) {
  $ids = @()
  foreach ($k in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$RootId" -ErrorAction SilentlyContinue)) {
    $kid = [int]$k.ProcessId
    $ids += Get-Descendants $kid
    $ids += $kid
  }
  return $ids
}

function Stop-Tree([int]$RootId) {
  # Snapshot the tree before killing: orphaned children lose their ParentProcessId.
  $descendants = @(Get-Descendants $RootId)
  # Kill the root before its children so it cannot react to their death or respawn them.
  try {
    Stop-Process -Id $RootId -Force -ErrorAction Stop
    Write-Host "[interrupt] killed installer pid=$RootId"
    if ($RootId -eq $proc.Id) { $script:rootKilled = $true }
  }
  catch { if ($RootId -eq $proc.Id) { Write-Host "[interrupt] installer pid=$RootId was already gone: $_" } }
  foreach ($id in $descendants) {
    try { Stop-Process -Id $id -Force -ErrorAction Stop; Write-Host "[interrupt] killed pid=$id" }
    catch { }
  }
}

# The dep pass rewrites one line with \r, so sub-steps are CR-separated segments.
$SubRe = '\[[=-]+\]\s*\d+/\d+\s'

function Get-PhaseLines([string]$Path) {
  $raw = Get-Content -Path $Path -Raw -ErrorAction SilentlyContinue
  if (-not $raw) { return @() }
  return @(($raw -replace "`r", "`n") -split "`n")
}

function Get-LastPhase([string]$Path) {
  $p = @(Get-PhaseLines $Path | Where-Object { $_ -match '^\[TAURI:STEP\]' -or $_ -match $SubRe })
  if ($p.Count) { return $p[-1] }
  return ''
}

function Test-MarkedPhaseOver {
  if (-not $Marker) { return $false }
  $lines = @(Get-PhaseLines $LogPath)
  $subs = @($lines | Where-Object { $_ -match $SubRe })
  if ($subs | Where-Object { $_ -match $Marker }) {
    $last = Get-LastPhase $LogPath
    return -not ($last -match $SubRe -and $last -match $Marker)
  }
  $steps = @($lines | Where-Object { $_ -match '^\[TAURI:STEP\]' })
  if ($steps | Where-Object { $_ -match $Marker }) {
    return ($steps[-1] -notmatch $Marker)
  }
  return $false
}

$killed = $false
$reason = ''
# Fifth-of-a-second slices to match the POSIX driver: labels print before their work.
for ($i = 0; $i -lt ($KillAtSeconds * 5); $i++) {
  if ($proc.HasExited) { $reason = 'exited-before-marker'; break }
  if ($Marker) {
    $hit = Select-String -Path $LogPath -Pattern $Marker -SimpleMatch:$false -ErrorAction SilentlyContinue
    if ($hit) {
      # Signal at detection, never after a delay: labels print before the work they name.
      if ($proc.HasExited) { $reason = 'exited-before-signal'; break }
      $reason = 'marker-hit'
      $killed = $true
      break
    }
  }
  Start-Sleep -Milliseconds 200
}
if (-not $killed -and -not $proc.HasExited) { if (-not $reason) { $reason = 'deadline' }; $killed = $true }

if ($killed) {
  Write-Host "[interrupt] killing process tree of $($proc.Id) ($reason)"
  Stop-Tree $proc.Id
  # UNSLOTH_STUDIO_HOME may mix `/` and `\`, so separators are normalised before matching paths.
  $studioRoot = if ([string]::IsNullOrWhiteSpace($env:UNSLOTH_STUDIO_HOME)) { Join-Path $HOME '.unsloth\studio' }
                else { $env:UNSLOTH_STUDIO_HOME }
  $homeNorm = if ([string]::IsNullOrWhiteSpace($studioRoot)) { $null }
              else { ($studioRoot -replace '/', '\').TrimEnd('\') }
  foreach ($p in @(Get-Process -Name 'uv', 'python', 'pythonw' -ErrorAction SilentlyContinue)) {
    $path = $null
    try { $path = $p.Path } catch { }
    $inHome = $homeNorm -and $path -and ($path -like "$homeNorm\*")
    if ($p.ProcessName -eq 'uv' -or $inHome) {
      try { Stop-Process -Id $p.Id -Force; Write-Host "[interrupt] swept $($p.ProcessName) pid=$($p.Id)" } catch { }
    }
  }
}

try { $proc.WaitForExit(30000) | Out-Null } catch { }
$rc = if ($proc.HasExited) { $proc.ExitCode } else { 'running' }
Write-Host "[interrupt] installer exit=$rc reason=$reason killed=$killed root_killed=$($script:rootKilled)"
Write-Host '[interrupt] last log lines:'
Get-Content $LogPath -Tail 15 -ErrorAction SilentlyContinue

if ($Marker -and -not (Select-String -Path $LogPath -Pattern $Marker -ErrorAction SilentlyContinue)) {
  Write-Host "::warning::marker '$Marker' never appeared -- killed at the deadline, not the intended step"
}
$lastPhase = Get-LastPhase $LogPath
Write-Host "[interrupt] phase at kill: $lastPhase"
$mismatch = Test-MarkedPhaseOver
if ($mismatch) {
  Write-Host "::warning::killed in '$lastPhase', not the marked phase -- that phase was already over"
}
# Lower-cased simple values only: the POSIX side sources this file.
@(
  "interrupt_reason=$reason"
  "interrupt_killed=$killed"
  "interrupt_root_killed=$(if ($script:rootKilled) { 'true' } else { 'false' })"
  "installer_exit=$rc"
  "interrupt_phase_mismatch=$(if ($mismatch) { 'true' } else { 'false' })"
) | Set-Content -Path (Join-Path (Split-Path -Parent $LogPath) 'interrupt.env') -Encoding utf8
exit 0
