# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
param(
    [Parameter(Mandatory = $true)]
    [string] $Path
)

$ErrorActionPreference = "Continue"

$maxAttempts = 3
$retryPatterns = @(
    "No subscriptions found",
    "login via azure cli",
    "az\.cmd.*exited with code 1"
)
$trustedSigningArgs = @(
    "-e",
    "https://eus.codesigning.azure.net",
    "-d",
    "Unsloth Studio (Desktop)",
    $Path
)

for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
    $output = & trusted-signing-cli @trustedSigningArgs 2>&1
    $exitCode = $LASTEXITCODE
    if ($null -eq $exitCode) {
        $exitCode = 1
    }
    $text = $output | Out-String

    foreach ($line in $output) {
        Write-Output $line
    }

    if ($exitCode -eq 0) {
        exit 0
    }

    $isRetryable = $false
    foreach ($pattern in $retryPatterns) {
        if ($text -match $pattern) {
            $isRetryable = $true
            break
        }
    }

    if (-not $isRetryable -or $attempt -eq $maxAttempts) {
        exit $exitCode
    }

    Write-Warning "trusted-signing-cli failed with transient Azure auth error; retrying."
    Start-Sleep -Seconds (5 * $attempt)
}

exit 1
