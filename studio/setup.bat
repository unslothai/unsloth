@echo off
REM SPDX-License-Identifier: AGPL-3.0-only
REM Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

rem RemoteSigned, not Bypass. RemoteSigned refuses an unsigned script only in the Internet or
rem Untrusted zone, and setup.ps1 sits beside this file inside an installed package, so it is
rem MyComputer-zone and loads unsigned either way. Pairing a relaxed policy with a PowerShell launch
rem is a token endpoint products score, and it was buying nothing here.
rem
rem The one population that could regress is a package unzipped from a download, where setup.ps1
rem carries a mark of the web that RemoteSigned honours and Bypass ignored. The call below clears it
rem first. Execution policy governs script FILES and not -Command, so that call runs whatever the
rem machine policy is, and clearing an absent stream is a no-op that writes nothing.
rem
rem The path travels in an environment variable rather than being interpolated into the command
rem string, so an apostrophe in the install path cannot break the quoting.
rem
rem -NoProfile on that call only, and deliberately NOT on the launch below: setup.ps1 is run with
rem profiles loaded on purpose, and tests/studio/test_amd_venv_repair_loop.ps1 drives a profile that
rem sets Set-StrictMode against it.
setlocal
set "UNSLOTH_SETUP_SCRIPT=%~dp0setup.ps1"
powershell -NoProfile -Command "Unblock-File -LiteralPath $env:UNSLOTH_SETUP_SCRIPT -Confirm:$false -ErrorAction SilentlyContinue"
endlocal

powershell -ExecutionPolicy RemoteSigned -File "%~dp0setup.ps1" %*
