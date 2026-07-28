# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Windows counterpart of interrupt-install.sh: run install.ps1 and kill it partway
# through, reproducing a user quitting the desktop app mid-install.
#
# Windows has no process groups, which is why the app carries windows_job.rs -- a Job
# Object is what makes "kill the installer and everything it spawned" work there. This
# script kills the whole process TREE for the same reason: killing only the leader
# leaves uv/python children to finish the dependency pass, and the interruption would
# prove nothing.
#
# Usage:
#   pwsh -File .github/scripts/interrupt-install.ps1 -Marker 'studio deps' `
#        -LogPath logs/install.log -InstallArgs '-SkipTorch'
[CmdletBinding()]
param(
  [string]$Marker = '',
  [string]$LogPath = 'logs/install.log',
  [string]$InstallArgs = '',
  [int]$KillAtSeconds = 900,
  [int]$KillAfterMarkerSeconds = 3
)

$ErrorActionPreference = 'Continue'
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
Set-Content -Path $LogPath -Value '' -Encoding utf8

# Stand in for the desktop app, which creates this before spawning the installer and
# clears it only on a terminal outcome (install.rs). We kill install.ps1 directly
# rather than driving the real app, so without this the marker #7490 relies on is
# absent for a reason that has nothing to do with #7490 -- which is exactly what the
# Windows legs reported. Both locations because the Rust side hardcodes
# ~/.unsloth/studio while CI overrides UNSLOTH_STUDIO_HOME. Never cleared: being
# killed is the whole point.
foreach ($dir in @($env:UNSLOTH_STUDIO_HOME, (Join-Path $HOME '.unsloth\studio'))) {
  if ([string]::IsNullOrWhiteSpace($dir)) { continue }
  try {
    New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
    Set-Content -Path (Join-Path $dir '.desktop-install-in-progress') -Value '' -ErrorAction Stop
  } catch { Write-Host "[interrupt] could not seed install marker in ${dir}: $_" }
}

# Run the installer in its own pwsh so stdout can be redirected to the log while we poll.
$argList = @('-NoProfile', '-NonInteractive', '-File', 'install.ps1')
if ($InstallArgs) { $argList += $InstallArgs.Split(' ') }
$proc = Start-Process -FilePath 'pwsh' -ArgumentList $argList `
  -RedirectStandardOutput $LogPath -RedirectStandardError "$LogPath.err" `
  -PassThru -NoNewWindow
Write-Host "[interrupt] installer pid=$($proc.Id) marker='$Marker' deadline=${KillAtSeconds}s"

function Stop-Tree([int]$RootId) {
  # Depth-first: children before parents, so a parent cannot respawn a child we already
  # killed. CIM gives us the parent link Windows does not expose via process groups.
  $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId=$RootId" -ErrorAction SilentlyContinue)
  foreach ($k in $kids) { Stop-Tree ([int]$k.ProcessId) }
  try { Stop-Process -Id $RootId -Force -ErrorAction Stop; Write-Host "[interrupt] killed pid=$RootId" }
  catch { }
}

$killed = $false
$reason = ''
for ($i = 0; $i -lt $KillAtSeconds; $i++) {
  if ($proc.HasExited) { $reason = 'exited-before-marker'; break }
  if ($Marker) {
    $hit = Select-String -Path $LogPath -Pattern $Marker -SimpleMatch:$false -ErrorAction SilentlyContinue
    if ($hit) {
      $reason = 'marker-hit'
      Start-Sleep -Seconds $KillAfterMarkerSeconds
      $killed = $true
      break
    }
  }
  Start-Sleep -Seconds 1
}
if (-not $killed -and -not $proc.HasExited) { if (-not $reason) { $reason = 'deadline' }; $killed = $true }

if ($killed) {
  Write-Host "[interrupt] killing process tree of $($proc.Id) ($reason)"
  Stop-Tree $proc.Id
  # Any straggler uv/python that reparented away from the installer.
  foreach ($name in 'uv', 'python') {
    Get-Process -Name $name -ErrorAction SilentlyContinue |
      Where-Object { $_.Path -and $_.Path -like "*$env:UNSLOTH_STUDIO_HOME*" } |
      ForEach-Object { try { Stop-Process -Id $_.Id -Force } catch { } }
  }
}

try { $proc.WaitForExit(30000) | Out-Null } catch { }
$rc = if ($proc.HasExited) { $proc.ExitCode } else { 'running' }
Write-Host "[interrupt] installer exit=$rc reason=$reason killed=$killed"
Write-Host '[interrupt] last log lines:'
Get-Content $LogPath -Tail 15 -ErrorAction SilentlyContinue

if ($Marker -and -not (Select-String -Path $LogPath -Pattern $Marker -ErrorAction SilentlyContinue)) {
  Write-Host "::warning::marker '$Marker' never appeared -- killed at the deadline, not the intended step"
}
@(
  "interrupt_reason=$reason"
  "interrupt_killed=$killed"
  "installer_exit=$rc"
) | Set-Content -Path (Join-Path (Split-Path -Parent $LogPath) 'interrupt.env') -Encoding utf8
exit 0
