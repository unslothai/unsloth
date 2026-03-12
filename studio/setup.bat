@echo off
REM SPDX-License-Identifier: AGPL-3.0-only
REM Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
