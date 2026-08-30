# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Windows counterpart of interrupt-install.sh: run install.ps1 and kill it partway
# through, reproducing a user quitting the desktop app mid-install.
#
# Windows has no process groups (hence the app's windows_job.rs), so this kills the whole
# process TREE: killing the leader alone leaves uv/python children to finish the dep pass.
#
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

# Stand in for the desktop app, which writes this before spawning the installer
# (install.rs). We kill install.ps1 directly, so without it #7490's marker is absent for an
# unrelated reason, exactly what the Windows legs reported. Both roots: Rust hardcodes
# ~/.unsloth/studio, CI overrides UNSLOTH_STUDIO_HOME. Never cleared, by design.
foreach ($dir in @($env:UNSLOTH_STUDIO_HOME, (Join-Path $HOME '.unsloth\studio'))) {
  if ([string]::IsNullOrWhiteSpace($dir)) { continue }
  try {
    New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
    Set-Content -Path (Join-Path $dir '.desktop-install-in-progress') -Value '' -ErrorAction Stop
  } catch { Write-Host "[interrupt] could not seed install marker in ${dir}: $_" }
}

# Its own host, so stdout can be redirected to the log while we poll. That host is WINDOWS
# PowerShell 5.1, not pwsh, with install.rs:325-339's exact flags: the only host a real
# desktop install uses, while every other Windows job in .github runs install.ps1 under
# pwsh 7, leaving 5.1 behaviour (.NET Framework, OEM/ANSI console encoding, different
# native-command and OSArchitecture reporting) covered by nothing. Only the installer child
# and the repair re-run change host; the driver stays under pwsh.
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

# Proof that the signal was DELIVERED, not merely attempted. The installer can fail on its
# own between the last HasExited check and Stop-Tree, and a natural failure carries a
# non-zero exit code just like a kill does, so the exit status alone cannot separate the two
# on Windows. Stop-Process throws on a process already gone, so this flag is false exactly
# when there was nothing left to interrupt.
$script:rootKilled = $false

function Get-Descendants([int]$RootId) {
  # Depth-first, deepest first. CIM gives the parent link Windows has no process groups for.
  $ids = @()
  foreach ($k in @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$RootId" -ErrorAction SilentlyContinue)) {
    $kid = [int]$k.ProcessId
    $ids += Get-Descendants $kid
    $ids += $kid
  }
  return $ids
}

function Stop-Tree([int]$RootId) {
  # Snapshot the whole tree BEFORE killing anything: once a parent is gone its children are
  # orphaned with no ParentProcessId left to walk, so the walk has to happen first.
  $descendants = @(Get-Descendants $RootId)
  # Then the ROOT, ahead of its children. install.ps1 watches the child it waits on: in
  # staging run 30424366953 it had already printed "unsloth studio setup failed (exit code
  # -1)" by the time Stop-Process reached it. Killing children first races the leader's own
  # exit, and a leader that wins makes Stop-Process throw over an interruption the driver
  # did deliver, failing the leg for nothing. Dead first, it can neither react nor respawn
  # what we are about to kill.
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

# A leg can be aimed at either kind of phase, and only one of them is a line. install.ps1
# prints "[TAURI:STEP] <name>" lines, while the dependency pass rewrites ONE physical line
# with \r (install_python_stack.py:2499), so its sub-steps are CR-separated SEGMENTS.
# Splitting on \r is what makes a sub-step's END observable at all.
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

# True when the phase the marker named is no longer the running one. A sub-step marker is
# judged against the running sub-step, a step marker against the running step -- a step is
# not "over" because the sub-steps beneath it advanced.
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
# Fifth-of-a-second slices, matching the POSIX driver: every phase label prints BEFORE its
# work, so this delay IS the whole distance between the label and the signal and the only
# thing that can push the kill past the end of a short phase. It slept 500ms while claiming
# a fifth, carrying 2.5 slices of overshoot the POSIX side does not.
for ($i = 0; $i -lt ($KillAtSeconds * 5); $i++) {
  if ($proc.HasExited) { $reason = 'exited-before-marker'; break }
  if ($Marker) {
    $hit = Select-String -Path $LogPath -Pattern $Marker -SimpleMatch:$false -ErrorAction SilentlyContinue
    if ($hit) {
      # Same as the POSIX driver: signal at detection, never after a delay. The label
      # prints before the work, so the kill is inside the phase the moment the line appears,
      # and any wait is a bet on the phase outlasting it that staging runs 30419729244 and
      # 30426111484 both lost.
      #
      # The installer can still exit on its own between the match and the signal, which
      # would record marker-hit over an install that interrupted nothing.
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
  # Any straggler uv/python that reparented away from the installer. The old sweep matched
  # nothing: UNSLOTH_STUDIO_HOME arrives as `D:\a\r\r/.studio-home` (github.workspace joined
  # with a forward slash) while Process.Path is all backslashes, so the literal -like missed
  # even the venv's own python -- hence the separator normalisation, and uv by name (it
  # lives outside the studio home and the ephemeral runner has no other uv). Under --tauri
  # there is no UNSLOTH_STUDIO_HOME, so fall back to install.ps1's root.
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
# Where the signal actually landed. A phase that ended before the poll saw the marker sends
# the kill into a LATER phase, so the leg duplicates whichever leg owns that phase while its
# own label claims otherwise.
$lastPhase = Get-LastPhase $LogPath
Write-Host "[interrupt] phase at kill: $lastPhase"
$mismatch = Test-MarkedPhaseOver
if ($mismatch) {
  Write-Host "::warning::killed in '$lastPhase', not the marked phase -- that phase was already over"
}
# Lower-cased so the workflow compares it the same way on every platform, and only simple
# values: the POSIX side sources this file.
@(
  "interrupt_reason=$reason"
  "interrupt_killed=$killed"
  "interrupt_root_killed=$(if ($script:rootKilled) { 'true' } else { 'false' })"
  "installer_exit=$rc"
  "interrupt_phase_mismatch=$(if ($mismatch) { 'true' } else { 'false' })"
) | Set-Content -Path (Join-Path (Split-Path -Parent $LogPath) 'interrupt.env') -Encoding utf8
exit 0
