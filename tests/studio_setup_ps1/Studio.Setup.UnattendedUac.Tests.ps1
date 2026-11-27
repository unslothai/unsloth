# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0' }

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')
    $script:SetupPs1 = Join-Path $PSScriptRoot '..\..\studio\setup.ps1'
    $script:InstallPs1 = Join-Path $PSScriptRoot '..\..\install.ps1'
    foreach ($n in 'Test-SetupProcessElevated', 'Test-SetupUacUnanswerable', 'Ensure-VCRedist', 'Get-CudaTorchInstallLabel') {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $n
        if (-not $src) { throw "setup.ps1 has no function $n" }
        . ([scriptblock]::Create($src))
    }
    # The real probe answers for the process running the tests, and GitHub's Windows runners are
    # elevated, so each test states it: $script:Elevated, $false unless the test sets it.
    $script:RealElevatedProbe = ${function:Test-SetupProcessElevated}
    $script:Elevated = $false
    function Test-SetupProcessElevated { $script:Elevated }
    foreach ($n in 'Test-InstallUacUnanswerable', 'Write-Arm64ShellNotice', 'Get-ShellProcessArch') {
        $src = Get-FunctionSource -Path $script:InstallPs1 -Name $n
        if (-not $src) { throw "install.ps1 has no function $n" }
        . ([scriptblock]::Create($src))
    }
    function step { param($Label, $Value, $Color) $script:Steps += "$Label|$Value" }
    function substep { param($Text, $Color) $script:Substeps += "$Text" }
    function Write-StudioLine { param([string]$Text, [string]$ForegroundColor) $script:Lines += "$Text" }
    function Refresh-Environment { }
    function Test-VCRedistInstalled { $false }
    function Invoke-SetupCommand { param($Block) throw 'ran an installer' }
    $script:SetupText = Get-Content -LiteralPath $script:SetupPs1 -Raw
    $script:InstallText = Get-Content -LiteralPath $script:InstallPs1 -Raw
    $script:EnvNames = 'UNSLOTH_NONINTERACTIVE', 'CI', 'UNSLOTH_TAURI_MODE', 'UNSLOTH_TAURI_UPDATE', 'UNSLOTH_SETUP_RERUN'
}

Describe 'Test-SetupUacUnanswerable' {
    BeforeEach {
        foreach ($k in $script:EnvNames) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
        $script:Elevated = $false
    }
    AfterAll { foreach ($k in $script:EnvNames) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }

    It 'UNSLOTH_NONINTERACTIVE=<Value> answers <Expected>' -ForEach @(
        @{ Value = '1'; Expected = $true }, @{ Value = 'yes'; Expected = $true },
        @{ Value = '0'; Expected = $false }, @{ Value = 'false'; Expected = $false }
    ) {
        $env:UNSLOTH_NONINTERACTIVE = $Value
        Test-SetupUacUnanswerable | Should -Be $Expected
    }

    It 'a CI run cannot answer a prompt' {
        $env:CI = 'true'
        Test-SetupUacUnanswerable | Should -BeTrue
    }

    It 'UNSLOTH_NONINTERACTIVE=0 wins over CI' {
        $env:CI = 'true'; $env:UNSLOTH_NONINTERACTIVE = '0'
        Test-SetupUacUnanswerable | Should -BeFalse
    }

    It 'an elevated run is never asked, even when unattended (<Var>=<Value>)' -ForEach @(
        @{ Var = 'UNSLOTH_NONINTERACTIVE'; Value = '1' }, @{ Var = 'CI'; Value = 'true' }
    ) {
        Set-Item "Env:$Var" $Value
        $script:Elevated = $true
        Test-SetupUacUnanswerable | Should -BeFalse
    }

    It 'asks the elevation probe the tests pin, not the token directly' {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name 'Test-SetupUacUnanswerable'
        $src | Should -Match 'if \(Test-SetupProcessElevated\) \{ return \$false \}'
        $src | Should -Not -Match 'IsInRole'
    }

    It 'the real elevation probe answers a bool' {
        & $script:RealElevatedProbe | Should -BeOfType ([bool])
    }

    It 'Unsloth Desktop (<Var>) keeps its own UAC flow' -ForEach @(
        @{ Var = 'UNSLOTH_TAURI_MODE' }, @{ Var = 'UNSLOTH_TAURI_UPDATE' }
    ) {
        Set-Item "Env:$Var" '1'
        $env:UNSLOTH_NONINTERACTIVE = '1'
        Test-SetupUacUnanswerable | Should -BeFalse
    }
}

Describe 'Ensure-VCRedist unattended' {
    BeforeEach {
        foreach ($k in $script:EnvNames) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue }
        $script:Steps = @(); $script:Substeps = @(); $script:Lines = @()
        $StageRoot = $null
    }
    AfterAll { foreach ($k in $script:EnvNames) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }

    It 'prints the admin command instead of starting an elevating installer' {
        Mock Test-VCRedistInstalled { $false }
        Mock Test-SetupUacUnanswerable { $true }
        Mock Get-Command { [pscustomobject]@{ Name = 'winget' } } -ParameterFilter { $Name -eq 'winget' }
        Mock Invoke-WebRequest { throw 'downloaded the runtime' }
        Mock Start-Process { throw 'started an installer' }
        Ensure-VCRedist
        Should -Invoke Invoke-WebRequest -Times 0 -Exactly
        Should -Invoke Start-Process -Times 0 -Exactly
        ($script:Steps -join "`n") | Should -Match '^vcredist\|missing; not installed'
        ($script:Substeps -join "`n") | Should -Match 'winget install --id Microsoft\.VCRedist\.2015\+\.x64'
    }

    It 'an attended run still tries to install it' {
        Mock Test-VCRedistInstalled { $false }
        Mock Test-SetupUacUnanswerable { $false }
        Mock Get-Command { $null } -ParameterFilter { $Name -eq 'winget' }
        Mock Invoke-WebRequest { throw 'offline' }
        Ensure-VCRedist
        Should -Invoke Invoke-WebRequest -Times 1 -Exactly
    }
}

Describe 'LongPathsEnabled unattended' {
    It 'is decided before reg.exe is started elevated, and names the command' {
        $text = $script:SetupText
        $gate = $text.IndexOf('} elseif (Test-SetupUacUnanswerable) {')
        $runAs = $text.IndexOf("-ArgumentList 'add `"HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`" /v LongPathsEnabled")
        $gate | Should -BeGreaterThan 0
        $runAs | Should -BeGreaterThan $gate
        $branch = $text.Substring($gate, $runAs - $gate)
        $branch | Should -Match 'reg add "HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1 /f'
        $branch.Substring(0, $branch.IndexOf('} else {')) | Should -Not -Match '-Verb RunAs'
    }
}

Describe 'Get-CudaTorchInstallLabel' {
    It 'Windows on ARM reports the pinned build tag, not the leaf' {
        Get-CudaTorchInstallLabel -CuTag 'cu130' -WinArm64Venv $true -WinArm64TorchSpec 'torch==2.14.0+cu134' | Should -Be 'cu134'
    }
    It 'Windows on ARM without a pinned tag does not claim the leaf' {
        Get-CudaTorchInstallLabel -CuTag 'cu130' -WinArm64Venv $true -WinArm64TorchSpec 'torch>=2.4' | Should -Be 'win_arm64 CUDA build'
    }
    It 'x64 keeps the CuTag leaf' {
        Get-CudaTorchInstallLabel -CuTag 'cu128' -WinArm64Venv $false -WinArm64TorchSpec 'torch==2.14.0+cu134' | Should -Be 'cu128'
    }
    It 'the install step prints the helper, not the raw leaf' {
        $script:SetupText | Should -Not -Match 'installing PyTorch with CUDA support \(\$CuTag\)'
        $script:SetupText | Should -Match 'installing PyTorch with CUDA support \(\$\(Get-CudaTorchInstallLabel'
    }
}

Describe 'install.ps1 unattended UAC' {
    BeforeEach { foreach ($k in $script:EnvNames) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }
    AfterAll { foreach ($k in $script:EnvNames) { Remove-Item "Env:$k" -ErrorAction SilentlyContinue } }

    It 'UNSLOTH_NONINTERACTIVE=1 is unattended unless elevated or Desktop' {
        $env:UNSLOTH_NONINTERACTIVE = '1'
        Test-InstallUacUnanswerable -Elevated 'false' | Should -BeTrue
        Test-InstallUacUnanswerable -Elevated 'true' | Should -BeFalse
        Test-InstallUacUnanswerable -Tauri -Elevated 'false' | Should -BeFalse
    }
    It 'UNSLOTH_NONINTERACTIVE=0 keeps prompting under CI' {
        $env:CI = '1'; $env:UNSLOTH_NONINTERACTIVE = '0'
        Test-InstallUacUnanswerable -Elevated 'false' | Should -BeFalse
    }
    It 'uv skips winget when unattended and hands the decision to setup' {
        $t = $script:InstallText
        $t | Should -Match 'if \(\$script:WingetAvailable -and -not \$script:StudioPrivateUvDir\) \{\s*#[^\n]*\n\s*if \(\$script:InstallUacUnanswerable\) \{\s*substep "unattended run: installing uv from its release archive'
        $t | Should -Match 'if \(\$script:InstallUacUnanswerable -and -not \$_hadNonInteractive\) \{ \$env:UNSLOTH_NONINTERACTIVE = "1" \}'
    }
}

Describe 'Write-Arm64ShellNotice' {
    BeforeEach { $script:Lines = @() }

    It 'an emulated x64 shell on ARM64 is told how to relaunch natively' {
        Write-Arm64ShellNotice -HostArch 'arm64' -ProcessArch 'x86_64' -Elevated 'true'
        $all = $script:Lines -join "`n"
        $all | Should -Match 'emulated x86_64 process on Windows on ARM'
        $all | Should -Match 'explorer\.exe ".*powershell\.exe"'
    }
    It 'an elevated native shell on ARM64 gets the relaunch command' {
        Write-Arm64ShellNotice -HostArch 'arm64' -ProcessArch 'arm64' -Elevated 'true'
        ($script:Lines -join "`n") | Should -Match 'explorer\.exe'
    }
    It 'a native non-elevated shell on ARM64 prints nothing' {
        Write-Arm64ShellNotice -HostArch 'arm64' -ProcessArch 'arm64' -Elevated 'false'
        $script:Lines.Count | Should -Be 0
    }
    It 'x64 Windows prints nothing, elevated or not' {
        Write-Arm64ShellNotice -HostArch 'x86_64' -ProcessArch 'x86_64' -Elevated 'true'
        Write-Arm64ShellNotice -HostArch 'x86_64' -ProcessArch 'x86_64' -Elevated 'false'
        $script:Lines.Count | Should -Be 0
    }
}
