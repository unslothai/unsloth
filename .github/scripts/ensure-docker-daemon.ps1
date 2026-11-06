# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Waits for the Windows Docker daemon, starting the service if installed but not running.

[CmdletBinding()]
param([int] $TimeoutMinutes = 5)

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
while ($true) {
    docker info *>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "docker daemon is up"
        break
    }
    if ((Get-Date) -ge $deadline) {
        Write-Host "::error::the Docker daemon never became reachable within $TimeoutMinutes minutes"
        Get-Service docker -ErrorAction SilentlyContinue | Format-List | Out-String | Write-Host
        exit 1
    }
    $svc = Get-Service -Name docker -ErrorAction SilentlyContinue
    Write-Host "docker service status: $(if ($svc) { $svc.Status } else { 'NOT INSTALLED' }); retrying..."
    if ($svc -and $svc.Status -ne 'Running') {
        Start-Service docker -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 5
}

# The runner appends `exit $LASTEXITCODE` to pwsh steps, so failed probes would fail the step.
$global:LASTEXITCODE = 0
exit 0
