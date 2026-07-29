# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Windows counterpart of interrupt-install.sh: run install.ps1 and kill it partway
# through, reproducing a user quitting the desktop app mid-install.
#
# Windows has no process groups, which is why the app carries windows_job.rs. This script
# kills the whole process TREE for the same reason: killing only the leader leaves
# uv/python children to finish the dependency pass, proving nothing.
#
# Usage:
#   pwsh -File .github/scripts/interrupt-install.ps1 -Marker 'studio deps' `
#        -LogPath logs/install.log -InstallArgs '--tauri --no-torch --local'
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

# Stand in for the desktop app, which writes this before spawning the installer
# (install.rs). We kill install.ps1 directly, so without it the marker #7490 relies on is
# absent for a reason unrelated to #7490 -- exactly what the Windows legs reported. Both
# locations because the Rust side hardcodes ~/.unsloth/studio while CI overrides
# UNSLOTH_STUDIO_HOME. Never cleared: being killed is the whole point.
foreach ($dir in @($env:UNSLOTH_STUDIO_HOME, (Join-Path $HOME '.unsloth\studio'))) {
  if ([string]::IsNullOrWhiteSpace($dir)) { continue }
  try {
    New-Item -ItemType Directory -Force -Path $dir -ErrorAction Stop | Out-Null
    Set-Content -Path (Join-Path $dir '.desktop-install-in-progress') -Value '' -ErrorAction Stop
  } catch { Write-Host "[interrupt] could not seed install marker in ${dir}: $_" }
}

# Its own host, so stdout can be redirected to the log while we poll. That host is
# WINDOWS PowerShell 5.1, not pwsh, with install.rs:325-339's exact flags: that is the
# only host a real desktop install ever uses, and every other Windows job in .github runs
# install.ps1 under pwsh 7, leaving 5.1 behaviour (.NET Framework, OEM/ANSI console
# encoding, different native-command and OSArchitecture reporting) covered by nothing.
# The driver itself stays under pwsh; only the installer child and the repair re-run
# change.
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

function Stop-Tree([int]$RootId) {
  # Depth-first, so a parent cannot respawn a child we already killed. CIM gives the
  # parent link Windows does not expose via process groups.
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
      Start-Sleep -Seconds $KillAfterMarkerSeconds
      # The installer can finish inside the delay; recording marker-hit before it
      # let a COMPLETED install satisfy the landing assertion and probe HEALTHY.
      if ($proc.HasExited) { $reason = 'exited-during-marker-delay'; break }
      $reason = 'marker-hit'
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
  # Any straggler uv/python that reparented away from the installer. The old sweep matched
  # nothing: UNSLOTH_STUDIO_HOME arrives as `D:\a\r\r/.studio-home` (github.workspace
  # joined with a forward slash) while Process.Path is all backslashes, so the literal
  # -like missed even the venv's own python. Hence the separator normalisation, and uv by
  # name (it lives outside the studio home, and the ephemeral runner has no other uv).
  # Under --tauri there is no UNSLOTH_STUDIO_HOME, so fall back to the root install.ps1
  # uses then, or the sweep would only ever see uv.
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
