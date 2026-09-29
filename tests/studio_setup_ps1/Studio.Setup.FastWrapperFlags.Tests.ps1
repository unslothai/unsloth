# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    Fast-* wrappers are advanced PowerShell functions. Short flags such as pip's
    `-d` can bind to common parameters such as `-Debug` instead of reaching pip.
    These tests require long flags at call sites and verify `--dest` is forwarded.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')

    $candidates = @(
        $env:SETUP_PS1_PATH,
        (Join-Path $PSScriptRoot '..\..\studio\setup.ps1')
    ) | Where-Object { $_ }
    $script:SetupPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:SetupPs1) { throw "Could not locate studio/setup.ps1 (set SETUP_PS1_PATH)." }
    Write-Host "setup.ps1 under test: $script:SetupPs1"

    $script:SetupText = Get-Content -Raw -LiteralPath $script:SetupPs1

    # Follow the common parameters declared by the current PowerShell host.
    $script:CommonParameterNames = [System.Management.Automation.Internal.CommonParameters].GetProperties().Name

    # Collect wrapper invocations, excluding declarations and comments.
    $script:WrapperCallLines = @(
        ($script:SetupText -split "`r?`n") |
            ForEach-Object -Begin { $n = 0 } -Process {
                $n++
                [pscustomobject]@{ Number = $n; Text = $_ }
            } |
            Where-Object {
                $_.Text -match 'Fast-(Install|Uninstall|Download)\b' -and
                $_.Text -notmatch '^\s*#' -and
                $_.Text -notmatch '^\s*function\s+Fast-'
            }
    )

    function Test-BindsAsCommonParameter {
        param([Parameter(Mandatory)][string]$Token)
        # Unique matches are swallowed; ambiguous matches fail binding.
        $bare = $Token.TrimStart('-')
        if (-not $bare) { return $false }
        return [bool](@($script:CommonParameterNames | Where-Object {
            $_.StartsWith($bare, [System.StringComparison]::OrdinalIgnoreCase)
        }).Count)
    }
}

Describe 'Fast-* wrappers are advanced functions (why short flags are unsafe)' {
    It 'declares ValueFromRemainingArguments on all three wrappers' {
        foreach ($fn in @('Fast-Install', 'Fast-Uninstall', 'Fast-Download')) {
            $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
            $src | Should -Not -BeNullOrEmpty -Because "$fn must exist to be tested"
            $src | Should -Match 'ValueFromRemainingArguments' -Because "$fn forwards its arguments verbatim"
        }
    }

    It 'swallows -d as -Debug instead of forwarding it' {
        # Match the wrappers' advanced-function parameter binding.
        function script:Probe-Args { param([Parameter(ValueFromRemainingArguments=$true)]$Args_) return , @($Args_) }

        $withShort = Probe-Args --no-deps -d 'C:\tmp' 'pkg==1.0'
        $withShort | Should -Not -Contain '-d' -Because '-d binds to the common parameter -Debug'
        $withShort.Count | Should -Be 3

        $withLong = Probe-Args --no-deps --dest 'C:\tmp' 'pkg==1.0'
        $withLong | Should -Contain '--dest' -Because 'a long flag is not a parameter-name token'
        $withLong.Count | Should -Be 4
    }
}

Describe 'No Fast-* call site passes a short flag the binder would capture (#10018)' {
    It 'finds the wrapper call sites at all' {
        # Prevent a refactor from making the scan vacuous.
        $script:WrapperCallLines.Count | Should -BeGreaterThan 10
    }

    It 'passes no single-dash flag that prefixes a common parameter' {
        $offenders = foreach ($line in $script:WrapperCallLines) {
            # Start at the wrapper name so flags on piped cmdlets are excluded.
            $call = $line.Text.Substring($line.Text.IndexOf('Fast-'))
            foreach ($m in [regex]::Matches($call, '(?<![\w-])-(?<name>[A-Za-z]{1,2})(?=\s|$)')) {
                $token = "-$($m.Groups['name'].Value)"
                if (Test-BindsAsCommonParameter -Token $token) {
                    "setup.ps1:$($line.Number): '$token' binds as a common parameter -- use the long flag"
                }
            }
        }
        ($offenders -join "`n") | Should -BeNullOrEmpty
    }

    It 'fetches the XPU triton wheel with --dest' {
        $destCalls = @($script:WrapperCallLines | Where-Object { $_.Text -match 'Fast-Download' -and $_.Text -match '--dest' })
        $destCalls.Count | Should -Be 2 -Because 'the swap fetches through both the verbose and the quiet branch'
        foreach ($line in $destCalls) {
            $line.Text | Should -Match '--dest\s+\$_tritonTmp' -Because 'the wheel must land in the staged temp dir'
        }
    }
}

Describe 'Fast-Download forwards the destination through to pip' {
    BeforeAll {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'Fast-Download'
        if (-not $src) { throw "Function 'Fast-Download' not found in $script:SetupPs1 - cannot test the real code." }
        . ([scriptblock]::Create($src))
    }

    It 'reaches pip with --dest and the directory adjacent' {
        # Shadow the external command to capture the wrapper's arguments.
        function script:python { $script:PipArgs = @($args); return '' }
        $script:PipArgs = @()

        Fast-Download --no-deps --only-binary=:all: --dest 'C:\tmp\stage' 'triton-xpu==3.6.0' --index-url 'https://example.invalid/xpu' | Out-Null

        $script:PipArgs | Should -Contain '--dest'
        $idx = [array]::IndexOf($script:PipArgs, '--dest')
        $script:PipArgs[$idx + 1] | Should -Be 'C:\tmp\stage' -Because "pip reads the directory as the value of --dest"
        $script:PipArgs[0..2] -join ' ' | Should -Be '-m pip download'
    }

    It 'loses the directory when the short flag is used (the #10018 failure)' {
        function script:python { $script:PipArgs = @($args); return '' }
        $script:PipArgs = @()

        Fast-Download --no-deps --only-binary=:all: -d 'C:\tmp\stage' 'triton-xpu==3.6.0' --index-url 'https://example.invalid/xpu' | Out-Null

        $script:PipArgs | Should -Not -Contain '-d' -Because 'the binder took it as -Debug'
        $script:PipArgs | Should -Contain 'C:\tmp\stage'
        [array]::IndexOf($script:PipArgs, 'C:\tmp\stage') | Should -BeGreaterThan 2
    }
}
