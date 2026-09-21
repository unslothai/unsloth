@echo off
REM SPDX-License-Identifier: AGPL-3.0-only
REM Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

rem RemoteSigned, not Bypass. RemoteSigned refuses an unsigned script only in the Internet or
rem Untrusted zone, and setup.ps1 normally sits beside this file inside an installed package, so it
rem is MyComputer-zone and loads unsigned either way. Pairing a relaxed policy with a PowerShell
rem launch is a token endpoint products score, and it was buying nothing in that case. Which product
rem scored what: tests/studio/test_installer_av_shapes.py (AV_SHAPES_RECORD).
rem
rem Two populations could regress, and both are handled below rather than assumed away.
rem
rem 1. A package unzipped from a download, where setup.ps1 carries a mark of the web that
rem    RemoteSigned honours and Bypass ignored. Unblock-File clears it. Execution policy governs
rem    script FILES and not -Command, so that call runs whatever the machine policy is, and clearing
rem    an absent stream is a no-op that writes nothing.
rem
rem 2. setup.ps1 living on a share. Execution policy is judged on the script file's ZONE, and a
rem    dotted-FQDN UNC (\\corp.example.com\...), a DFS path or an IP-literal share lands in the
rem    Internet zone -- where RemoteSigned refuses an unsigned script and Unblock-File cannot save
rem    it, because with no Zone.Identifier stream present the PATH decides and there is nothing to
rem    clear. install.ps1:3405-3421 already steps down to Bypass for exactly this case when it
rem    writes the Start Menu shortcut; the probe below reuses that logic rather than inventing a
rem    second answer. A plain \\server\share and a mapped drive are Intranet and stay RemoteSigned.
rem
rem The probe also does the Unblock-File, so this costs one process rather than two.
rem
rem The path travels in an environment variable rather than being interpolated into the command
rem string, so an apostrophe in the install path cannot break the quoting.
rem
rem -NoProfile on the probe only, and deliberately NOT on the launch below: setup.ps1 is run with
rem profiles loaded on purpose, and tests/studio/test_amd_venv_repair_loop.ps1 drives a profile that
rem sets Set-StrictMode against it.
setlocal
set "UNSLOTH_SETUP_SCRIPT=%~dp0setup.ps1"
set "UNSLOTH_SETUP_POLICY=RemoteSigned"
for /f "usebackq delims=" %%P in (`powershell -NoProfile -Command "$p = $env:UNSLOTH_SETUP_SCRIPT; Unblock-File -LiteralPath $p -Confirm:$false -ErrorAction SilentlyContinue; $remote = $p -like '\\*'; if (-not $remote) { try { $remote = ([System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($p))).DriveType -eq 'Network' } catch {} }; if ($remote) { 'Bypass' } else { 'RemoteSigned' }"`) do set "UNSLOTH_SETUP_POLICY=%%P"

rem Defaulting to RemoteSigned above means a probe that fails to run leaves the tightened policy in
rem place rather than silently restoring the relaxed one.
powershell -ExecutionPolicy %UNSLOTH_SETUP_POLICY% -File "%UNSLOTH_SETUP_SCRIPT%" %*
exit /b %ERRORLEVEL%
