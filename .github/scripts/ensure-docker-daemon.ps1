# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

# Waits for the Windows Docker daemon on a hosted runner, starting the service if
# it is installed but not running.
#
# Docker is installed on every windows-2022 image (runner-images uses Microsoft's
# install-docker-ce.ps1 without -HyperV, so the daemon serves WINDOWS containers) but is
# not always RUNNING at job start: a spike run died 21s in with "failed to connect to the
# docker API at npipe:////./pipe/docker_engine" while a sibling job was fine. Without the
# wait that flake reads as "Windows containers are unavailable on hosted runners".

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

# Failing `docker info` probes leave $LASTEXITCODE non-zero and the runner appends
# `exit $LASTEXITCODE` to every pwsh step (actions/runner#351), so a successful wait
# would still fail the step.
$global:LASTEXITCODE = 0
exit 0
