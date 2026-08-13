# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# See /studio/LICENSE.AGPL-3.0
#
# Regression tests for the npm-failure hint in studio/setup.ps1 (issue #8725).
#
# A Windows user installing Studio with Node 24 on PATH saw the OXC validator
# step die with:
#
#     npm error code EACCES
#     npm error FetchError: request to https://registry.npmjs.org/oxlint/-/oxlint-1.65.0.tgz failed
#     npm error The operation was rejected by your operating system.
#
# and the installer answered "registry.npmjs.org looks blocked (corporate
# firewall/proxy?)". That is a local failure -- a locked or unwritable npm cache
# -- so the reporter spent the evening looking for a corporate proxy that did
# not exist; re-running and running as Administrator changed nothing.
#
# Show-NpmRegistryHint used to take no arguments: it could not tell a blocked
# registry from a denied file, because Invoke-SetupCommand captured npm's output
# into a local variable and dropped it. It now receives that text and classifies
# local errno failures BEFORE the network markers -- npm's FetchError line names
# registry.npmjs.org even when the cause is local, which is exactly what made
# the old hint fire.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')

    $candidates = @(
        $env:SETUP_PS1_PATH,
        (Join-Path $PSScriptRoot '..\..\studio\setup.ps1')
    ) | Where-Object { $_ }
    $script:SetupPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:SetupPs1) { throw "Could not locate studio/setup.ps1 (set SETUP_PS1_PATH)." }
    Write-Host "setup.ps1 under test: $script:SetupPs1"

    # The classifiers live at script scope, outside any function, so pull the
    # real assignments out of the file rather than restating their values here.
    $setupText = Get-Content -Raw -LiteralPath $script:SetupPs1
    foreach ($varName in @('NpmLocalFailureRe', 'NpmNetworkFailureRe')) {
        $m = [regex]::Match($setupText, "(?m)^\s*\`$script:$varName\s*=\s*'[^']*'")
        if (-not $m.Success) { throw "Could not find `$script:$varName in $script:SetupPs1." }
        . ([scriptblock]::Create($m.Value))
    }

    foreach ($fn in @('Get-StudioAnsi', 'Write-StudioLine', 'Write-StudioStdoutMirror', 'step', 'substep',
                      'Show-NpmLocalFailureHint', 'Show-NpmRegistryHint')) {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
        if (-not $src) { throw "Function '$fn' not found in $script:SetupPs1 - cannot test the real code." }
        . ([scriptblock]::Create($src))
    }

    # The log from the issue, trimmed. It names registry.npmjs.org: that is the
    # line that used to send this failure down the firewall path.
    $script:EaccesOutput = @'
npm error code EACCES
npm error errno EACCES
npm error FetchError: request to https://registry.npmjs.org/oxlint/-/oxlint-1.65.0.tgz failed, reason:
npm error   code: 'EACCES',
npm error   type: 'system'
npm error The operation was rejected by your operating system.
'@

    $script:NetworkOutput = @'
npm error code ENOTFOUND
npm error network request to https://registry.npmjs.org/oxlint failed, reason: getaddrinfo ENOTFOUND registry.npmjs.org
'@

    $script:UnrelatedOutput = @'
npm error code ELIFECYCLE
npm error errno 1
npm error oxc-validator@1.0.0 postinstall script failed
'@

    function Invoke-CapturingHint {
        param([string]$FailureOutput = "")
        $prevOut = [Console]::Out
        $writer = New-Object System.IO.StringWriter
        try {
            [Console]::SetOut($writer)
            Show-NpmRegistryHint -FailureOutput $FailureOutput 6>&1 | Out-Null
        } finally {
            [Console]::SetOut($prevOut)
        }
        return $writer.ToString()
    }
}

Describe 'Show-NpmRegistryHint classifies the failure before guessing' {
    BeforeEach {
        $script:StudioVtOk = $false
        $script:StudioStdoutRedirected = $true
        $env:NO_COLOR = $null
        $env:UNSLOTH_NPM_REGISTRY = $null
        $env:NPM_CONFIG_REGISTRY = $null
    }

    It 'reports a permission failure as local, not as a blocked registry' {
        $out = Invoke-CapturingHint -FailureOutput $script:EaccesOutput
        $out | Should -Match 'blocked by the operating system'
        $out | Should -Not -Match 'looks blocked \(corporate firewall/proxy\?\)'
    }

    It 'still points at the registry when the failure really is the network' {
        $out = Invoke-CapturingHint -FailureOutput $script:NetworkOutput
        $out | Should -Match 'looks blocked \(corporate firewall/proxy\?\)'
        $out | Should -Not -Match 'blocked by the operating system'
    }

    It 'stays quiet when nothing points at either cause' {
        # The raw npm error is already on screen and beats a guess.
        $out = Invoke-CapturingHint -FailureOutput $script:UnrelatedOutput
        $out.Trim() | Should -BeNullOrEmpty
    }

    It 'keeps the old behaviour when no output was captured' {
        # Verbose runs stream npm output straight through, so there is nothing to
        # classify; the registry hint is then the only guidance available.
        $out = Invoke-CapturingHint -FailureOutput ""
        $out | Should -Match 'looks blocked \(corporate firewall/proxy\?\)'
    }

    It 'prints the local hint even when a mirror is already configured' {
        # A mirror does not make a locked cache writable.
        $env:UNSLOTH_NPM_REGISTRY = 'https://mirror.example/api/npm/'
        try {
            $out = Invoke-CapturingHint -FailureOutput $script:EaccesOutput
            $out | Should -Match 'blocked by the operating system'
        } finally {
            $env:UNSLOTH_NPM_REGISTRY = $null
        }
    }
}

Describe 'Source contracts' {
    It 'gives Show-NpmRegistryHint the captured output' {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'Show-NpmRegistryHint'
        $src | Should -Match '\[string\]\s*\$FailureOutput'
    }

    It 'checks local errno markers before network ones' {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'Show-NpmRegistryHint'
        $localAt = $src.IndexOf('NpmLocalFailureRe')
        $networkAt = $src.IndexOf('NpmNetworkFailureRe')
        $localAt | Should -BeGreaterThan -1
        $networkAt | Should -BeGreaterThan -1
        $localAt | Should -BeLessThan $networkAt
    }

    It 'passes the captured output at every call site' {
        $text = Get-Content -Raw -LiteralPath $script:SetupPs1
        $calls = ([regex]::Matches($text, '(?m)^\s*Show-NpmRegistryHint[^\r\n]*')) | ForEach-Object { $_.Value }
        $calls.Count | Should -BeGreaterThan 1
        foreach ($call in $calls) { $call | Should -Match '-FailureOutput' }
    }
}
