# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# See /studio/LICENSE.AGPL-3.0
#
# Show-NpmRegistryHint in studio/setup.ps1: a local npm errno gets a local hint, not
# "registry.npmjs.org looks blocked". #8725's EACCES came from the HTTP socket
# (FetchError), so it gets the "OS refused node's connection" variant.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')
    $candidates = @($env:SETUP_PS1_PATH, (Join-Path $PSScriptRoot '..\..\studio\setup.ps1')) | Where-Object { $_ }
    $script:SetupPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:SetupPs1) { throw "Could not locate studio/setup.ps1 (set SETUP_PS1_PATH)." }

    $setupText = Get-Content -Raw -LiteralPath $script:SetupPs1
    foreach ($varName in @('NpmLocalFailureRe', 'NpmNetworkFailureRe')) {
        $m = [regex]::Match($setupText, "(?m)^\s*\`$script:$varName\s*=\s*'[^']*'")
        if (-not $m.Success) { throw "Could not find `$script:$varName in $script:SetupPs1." }
        . ([scriptblock]::Create($m.Value))
    }
    foreach ($fn in @('Get-StudioAnsi', 'Write-StudioLine', 'Write-StudioStdoutMirror', 'step', 'substep',
                      'Show-NpmLocalFailureHint', 'Show-NpmRegistryHint')) {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
        if (-not $src) { throw "Function '$fn' not found in $script:SetupPs1." }
        . ([scriptblock]::Create($src))
    }

    $script:EaccesOutput = @'
npm error code EACCES
npm error FetchError: request to https://registry.npmjs.org/oxlint/-/oxlint-1.65.0.tgz failed, reason:
npm error The operation was rejected by your operating system.
'@
    $script:EpermFileOutput = @'
npm error code EPERM
npm error syscall rename
npm error Error: EPERM: operation not permitted, rename 'C:\npm-cache\_cacache\tmp\x'
'@
    $script:CacheAclOutput = @'
npm error code EPERM
npm error syscall mkdir
npm error path D:\a\_temp\x\cache\_cacache
npm error errno EPERM
npm error FetchError: Invalid response body while trying to fetch https://registry.npmjs.org/is-number: EPERM: operation not permitted, mkdir 'D:\a\_temp\x\cache\_cacache'
'@
    $script:NetworkOutput = @'
npm error code ENOTFOUND
npm error network request to https://registry.npmjs.org/oxlint failed, reason: getaddrinfo ENOTFOUND registry.npmjs.org
'@
    $script:NetworkCleanupOutput = @'
npm warn cleanup Failed to remove some directories [
npm warn cleanup   [Error: EPERM: operation not permitted, rmdir 'C:\x\node_modules\oxlint']
npm warn cleanup ]
npm error code ETIMEDOUT
npm error network request to https://registry.npmjs.org/oxlint failed
'@
    $script:UnrelatedOutput = @'
npm error code ELIFECYCLE
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

Describe 'Show-NpmRegistryHint' {
    BeforeEach {
        $script:StudioVtOk = $false
        $script:StudioStdoutRedirected = $true
        $env:NO_COLOR = $null
        $env:UNSLOTH_NPM_REGISTRY = $null
        $env:NPM_CONFIG_REGISTRY = $null
    }

    It 'reports the #8725 socket EACCES as a blocked node connection' {
        $out = Invoke-CapturingHint -FailureOutput $script:EaccesOutput
        $out | Should -Match "refused node's connection"
        $out | Should -Not -Match 'local file error'
        $out | Should -Not -Match 'looks blocked \(corporate firewall/proxy\?\)'
    }

    It 'reports EPERM on a cache file as a local file error' {
        $out = Invoke-CapturingHint -FailureOutput $script:EpermFileOutput
        $out | Should -Match 'local file error'
        $out | Should -Not -Match "refused node's connection"
        $out | Should -Not -Match 'looks blocked \(corporate firewall/proxy\?\)'
    }

    It 'reports an unwritable cache (FetchError with a path) as a file error' {
        $out = Invoke-CapturingHint -FailureOutput $script:CacheAclOutput
        $out | Should -Match 'local file error'
        $out | Should -Not -Match "refused node's connection"
    }

    It 'strips ANSI colour before matching the code line' {
        $e = [char]27
        $colored = "${e}[31mnpm${e}[39m ${e}[31merror${e}[39m ${e}[90mcode${e}[39m EPERM`n${e}[31mnpm${e}[39m ${e}[31merror${e}[39m syscall rename"
        Invoke-CapturingHint -FailureOutput $colored | Should -Match 'local file error'
    }

    It 'points at the registry for a network failure' {
        $out = Invoke-CapturingHint -FailureOutput $script:NetworkOutput
        $out | Should -Match 'looks blocked \(corporate firewall/proxy\?\)'
        $out | Should -Not -Match 'local file error'
    }

    It 'ignores EPERM cleanup warnings next to a network failure' {
        $out = Invoke-CapturingHint -FailureOutput $script:NetworkCleanupOutput
        $out | Should -Match 'looks blocked \(corporate firewall/proxy\?\)'
        $out | Should -Not -Match 'local file error'
    }

    It 'stays quiet when nothing points at either cause' {
        (Invoke-CapturingHint -FailureOutput $script:UnrelatedOutput).Trim() | Should -BeNullOrEmpty
    }

    It 'keeps the registry hint when no output was captured' {
        Invoke-CapturingHint -FailureOutput "" | Should -Match 'looks blocked \(corporate firewall/proxy\?\)'
    }

    It 'prints the local hint even with UNSLOTH_NPM_REGISTRY set' {
        $env:UNSLOTH_NPM_REGISTRY = 'https://mirror.example/api/npm/'
        Invoke-CapturingHint -FailureOutput $script:EaccesOutput | Should -Match "refused node's connection"
    }

    It 'passes the captured output at every call site' {
        $calls = [regex]::Matches((Get-Content -Raw -LiteralPath $script:SetupPs1), '(?m)^\s*Show-NpmRegistryHint\b[^\r\n]*') | ForEach-Object { $_.Value }
        $calls.Count | Should -Be 2
        foreach ($call in $calls) { $call | Should -Match '-FailureOutput \$script:SetupCommandOutput' }
    }
}
