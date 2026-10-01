# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Run on a real Windows runner after `tauri build --bundles nsis`.
param([Parameter(Mandatory=$true)][string]$Executable,
      [Parameter(Mandatory=$true)][string]$Installer,
      [Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $Output | Out-Null
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class NativeIcons {
  // lint-allow: AV001 narrowly scoped test harness calls Windows icon extraction only; no shipped code, dynamic load, or network.
  [DllImport("user32.dll", CharSet=CharSet.Unicode)]
  public static extern uint PrivateExtractIcons(string file, int index, int cx, int cy,
    IntPtr[] icons, uint[] ids, uint count, uint flags);
  // lint-allow: AV001 destroy the extracted native icon handle from the same test harness.
  [DllImport("user32.dll")] public static extern bool DestroyIcon(IntPtr icon);
}
'@
$report = [ordered]@{
  os = [Environment]::OSVersion.VersionString
  executable = (Get-Item -LiteralPath $Executable).Name
  installer = (Get-Item -LiteralPath $Installer).Name
  extracted = @()
  session_id = (Get-Process -Id $PID).SessionId
  explorer_session_ids = @((Get-Process explorer -ErrorAction SilentlyContinue | ForEach-Object SessionId))
  desktop_capture = 'not attempted'
}
# Install the actual unsigned NSIS bundle into a runner-owned per-user path.
# /D must be the last NSIS argument and unquoted.
$installDir = Join-Path $env:RUNNER_TEMP 'Unsloth-icon-evidence-install'
if (Test-Path -LiteralPath $installDir) { throw 'Refusing to reuse an existing installation path' }
$installation = Start-Process -FilePath $Installer -ArgumentList @('/S', "/D=$installDir") -Wait -PassThru
if ($installation.ExitCode -ne 0) { throw "NSIS install failed with exit code $($installation.ExitCode)" }
$installedExecutable = Join-Path $installDir 'unsloth-studio.exe'
if (!(Test-Path -LiteralPath $installedExecutable)) { throw 'NSIS install did not create the Studio executable' }
# Hosted NSIS install produced a different PE hash; icon pixel equality is the relevant assertion.
$report.built_exe_sha256 = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
$report.installed_exe_sha256 = (Get-FileHash -LiteralPath $installedExecutable -Algorithm SHA256).Hash
$report.install_exit = $installation.ExitCode
$report.installed_executable = (Get-Item -LiteralPath $installedExecutable).Name
foreach ($item in @(@('app', $Executable), @('nsis', $Installer), @('installed', $installedExecutable))) {
  foreach ($size in @(16, 24, 32, 48, 256)) {
    $handles = [IntPtr[]]::new(1)
    $ids = [uint32[]]::new(1)
    $count = [NativeIcons]::PrivateExtractIcons($item[1], 0, $size, $size, $handles, $ids, 1, 0)
    if ($count -ne 1 -or $handles[0] -eq [IntPtr]::Zero) {
      throw "Windows failed to extract $size-pixel $($item[0]) resource (count=$count)"
    }
    try {
      $icon = [System.Drawing.Icon]::FromHandle($handles[0])
      $bitmap = $icon.ToBitmap()
      try {
        if ($bitmap.Width -ne $size -or $bitmap.Height -ne $size) {
          throw "Incorrect Windows-extracted size: $($bitmap.Width)x$($bitmap.Height) != $size"
        }
        $name = "$($item[0])-$size.png"
        $bitmap.Save((Join-Path $Output $name), [System.Drawing.Imaging.ImageFormat]::Png)
        $report.extracted += [ordered]@{file=$name; size=$size; corner_alpha=$bitmap.GetPixel(0,0).A; resource_id=$ids[0]}
        if ($size -le 32 -and $bitmap.GetPixel(0,0).A -ne 0) {
          throw "Windows-extracted $($item[0]) $size-pixel corner is opaque"
        }
      } finally { $bitmap.Dispose() }
    } finally { [void][NativeIcons]::DestroyIcon($handles[0]) }
  }
}
foreach ($size in @(16, 24, 32, 48, 256)) {
  $built = (Get-FileHash -LiteralPath (Join-Path $Output "app-$size.png") -Algorithm SHA256).Hash
  $installed = (Get-FileHash -LiteralPath (Join-Path $Output "installed-$size.png") -Algorithm SHA256).Hash
  if ($built -ne $installed) { throw "NSIS installed $size-pixel icon differs from the built app icon" }
}
$report.installed_icon_matches_app = $true
# GitHub-hosted Windows workers normally run as noninteractive services. Do not
# manufacture a taskbar screenshot from PNG files or a hidden desktop. Only
# attempt a genuine capture if Explorer exists in this process's own session.
if ($report.explorer_session_ids -contains $report.session_id) {
  try {
    Add-Type -AssemblyName System.Windows.Forms
    $process = Start-Process -FilePath $installedExecutable -WorkingDirectory $installDir -PassThru
    $visible = $null
    for ($attempt = 0; $attempt -lt 4; $attempt++) {
      Start-Sleep -Seconds 5
      $visible = @(Get-Process -Name 'unsloth-studio' -ErrorAction SilentlyContinue |
        Where-Object { $_.SessionId -eq $report.session_id -and $_.Path -eq $installedExecutable -and $_.MainWindowHandle -ne [IntPtr]::Zero }) |
        Select-Object -First 1
      if ($visible) { break }
    }
    $process.Refresh()
    $report.app_exited_before_capture = $process.HasExited
    if ($process.HasExited) { $report.app_exit_code = $process.ExitCode }
    if ($visible) {
      # MainWindowHandle alone does not mean the window is on top: the hosted
      # runner's own console covered it in the previous capture. Foreground the
      # verified installed process before taking a real desktop screenshot.
      Add-Type -AssemblyName Microsoft.VisualBasic
      [Microsoft.VisualBasic.Interaction]::AppActivate([int]$visible.Id)
      Start-Sleep -Seconds 2
      $report.installed_window_process = $visible.Id
      $bounds = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
      $capture = New-Object System.Drawing.Bitmap($bounds.Width, $bounds.Height)
      $graphics = [System.Drawing.Graphics]::FromImage($capture)
      try {
        $graphics.CopyFromScreen($bounds.Location, [System.Drawing.Point]::Empty, $bounds.Size)
        $capture.Save((Join-Path $Output 'desktop.png'), [System.Drawing.Imaging.ImageFormat]::Png)
        $report.desktop_capture = 'captured interactive desktop after activating installed app; inspect pixels for actual visible window/taskbar before publication'
      } finally { $graphics.Dispose(); $capture.Dispose() }
    } else {
      $report.desktop_capture = 'interactive Explorer session exists but installed app never exposed a visible window/taskbar in 20 seconds; no taskbar screenshot available'
    }
  } catch {
    $report.desktop_capture = "interactive installed-app capture failed: $($_.Exception.Message)"
  } finally {
    if ($visible -and $visible.Id -ne $process.Id) { Stop-Process -Id $visible.Id -ErrorAction SilentlyContinue }
    if ($process) { Stop-Process -Id $process.Id -ErrorAction SilentlyContinue }
  }
} else {
  $report.desktop_capture = 'unavailable: no explorer.exe in runner process session; GitHub-hosted service is not a visible interactive taskbar'
}
# Clean up only this runner-owned install after stopping the exact process above.
$uninstaller = Join-Path $installDir 'uninstall.exe'
if (Test-Path -LiteralPath $uninstaller) {
  $uninstall = Start-Process -FilePath $uninstaller -ArgumentList '/S' -Wait -PassThru
  $report.uninstall_exit = $uninstall.ExitCode
  if ($uninstall.ExitCode -ne 0) { throw "NSIS uninstall failed with exit code $($uninstall.ExitCode)" }
} else {
  $report.uninstall_exit = 'missing uninstaller'
}
# lint-allow: AV010 writes only JSON; the installed .exe comes from real NSIS and is never overwritten with non-PE bytes.
$report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $Output 'windows-report.json') -Encoding utf8
Get-Content -LiteralPath (Join-Path $Output 'windows-report.json')
