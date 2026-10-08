# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#Requires -Modules @{ ModuleName = 'Pester'; ModuleVersion = '5.0' }
# A failed llama.cpp prebuilt must not install a machine-wide toolchain without consent.

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')
    $script:SetupPs1 = Join-Path $PSScriptRoot '..\..\studio\setup.ps1'
    . ([scriptblock]::Create((Get-FunctionSource -Path $script:SetupPs1 -Name 'Test-LlamaBuildToolsInstallAllowed')))
    . ([scriptblock]::Create((Get-FunctionSource -Path $script:SetupPs1 -Name 'Test-LlamaBuildToolsMissing')))
    . ([scriptblock]::Create((Get-FunctionSource -Path $script:SetupPs1 -Name 'Test-SetupConsoleHeadless')))
    . ([scriptblock]::Create((Get-FunctionSource -Path $script:SetupPs1 -Name 'Get-VcBuildCustomizationsDir')))
    function Get-LlamaDriverMaxCuda { $null }
    function Write-StudioLine { param([string]$Text, [string]$ForegroundColor) }
    $script:SetupText = Get-Content -LiteralPath $script:SetupPs1 -Raw
}

Describe 'Test-LlamaBuildToolsInstallAllowed' {
    BeforeEach {
        foreach ($k in 'UNSLOTH_INSTALL_BUILD_TOOLS', 'UNSLOTH_TAURI_MODE', 'UNSLOTH_TAURI_UPDATE') {
            Remove-Item "Env:$k" -ErrorAction SilentlyContinue
        }
    }
    AfterAll {
        foreach ($k in 'UNSLOTH_INSTALL_BUILD_TOOLS', 'UNSLOTH_TAURI_MODE', 'UNSLOTH_TAURI_UPDATE') {
            Remove-Item "Env:$k" -ErrorAction SilentlyContinue
        }
    }

    It 'UNSLOTH_INSTALL_BUILD_TOOLS=<Value> answers without a prompt' -ForEach @(
        @{ Value = '1'; Expected = $true }, @{ Value = 'yes'; Expected = $true },
        @{ Value = '0'; Expected = $false }, @{ Value = 'no'; Expected = $false }
    ) {
        Mock Read-Host { throw 'prompted' }
        $env:UNSLOTH_INSTALL_BUILD_TOOLS = $Value
        Test-LlamaBuildToolsInstallAllowed | Should -Be $Expected
        Should -Invoke Read-Host -Times 0 -Exactly
    }

    It 'declines in Unsloth Desktop (<Var>) instead of prompting' -ForEach @(
        @{ Var = 'UNSLOTH_TAURI_MODE' }, @{ Var = 'UNSLOTH_TAURI_UPDATE' }
    ) {
        Mock Read-Host { throw 'prompted' }
        Set-Item "Env:$Var" '1'
        Test-LlamaBuildToolsInstallAllowed | Should -BeFalse
        Should -Invoke Read-Host -Times 0 -Exactly
    }

    It 'a <Kind> run outside Unsloth Desktop answers <Expected> without a prompt' -ForEach @(
        @{ Kind = 'headless'; Headless = $true; Expected = $true },
        @{ Kind = 'terminal'; Headless = $false; Expected = $false }
    ) {
        Mock Read-Host { throw 'prompted' }
        Mock Test-SetupConsoleHeadless { $Headless }
        Test-LlamaBuildToolsInstallAllowed | Should -Be $Expected
        Should -Invoke Read-Host -Times 0 -Exactly
    }

    It 'setup.ps1 never asks about the build tools' {
        $body = Get-FunctionSource -Path $script:SetupPs1 -Name 'Test-LlamaBuildToolsInstallAllowed'
        $body | Should -Not -Match 'Read-Host'
    }

    It 'the opt-in still wins inside Unsloth Desktop' {
        $env:UNSLOTH_TAURI_MODE = '1'
        $env:UNSLOTH_INSTALL_BUILD_TOOLS = '1'
        Test-LlamaBuildToolsInstallAllowed | Should -BeTrue
    }
}

Describe 'Test-LlamaBuildToolsMissing' {
    It 'reports missing when <Case>' -ForEach @(
        @{ Case = 'git is absent'; Git = $false; Cmake = $true; Vs = 'C:\VS'; Expected = $true },
        @{ Case = 'cmake is absent'; Git = $true; Cmake = $false; Vs = 'C:\VS'; Expected = $true },
        @{ Case = 'Build Tools are absent'; Git = $true; Cmake = $true; Vs = $null; Expected = $true },
        @{ Case = 'the CUDA Toolkit is absent on NVIDIA'; Git = $true; Cmake = $true; Vs = 'C:\VS'; Nvidia = $true; Nvcc = $null; Expected = $true },
        @{ Case = 'nothing is absent on NVIDIA'; Git = $true; Cmake = $true; Vs = 'C:\VS'; Nvidia = $true; Nvcc = 'C:\nvcc.exe'; Expected = $false },
        @{ Case = 'nothing is absent'; Git = $true; Cmake = $true; Vs = 'C:\VS'; Expected = $false }
    ) {
        Mock Get-Command { if ($Git) { [pscustomobject]@{ Name = 'git' } } } -ParameterFilter { $Name -eq 'git' }
        Mock Get-Command { if ($Cmake) { [pscustomobject]@{ Name = 'cmake' } } } -ParameterFilter { $Name -eq 'cmake' }
        function Find-VsBuildTools { $null }
        function Find-Nvcc { $Nvcc }
        $HasNvidiaDriverEvidence = [bool]$Nvidia
        $script:VsInstallPath = $Vs
        Test-LlamaBuildToolsMissing | Should -Be $Expected
    }

    It 'a CUDA toolkit newer than the driver counts as missing (toolkit <Major>, driver 12.8)' -ForEach @(
        @{ Major = 13; Expected = $true }, @{ Major = 12; Expected = $false }
    ) {
        Mock Get-Command { [pscustomobject]@{ Name = $Name } } -ParameterFilter { $Name -in 'git', 'cmake' }
        function Find-VsBuildTools { $null }
        function Find-Nvcc { param([string]$MaxVersion) if ($MaxVersion) { $null } else { 'C:\CUDA\v13.0\bin\nvcc.exe' } }
        function Get-LlamaDriverMaxCuda { '12.8' }
        function Get-NvccMajor { param($Nvcc) $Major }
        $HasNvidiaDriverEvidence = $true
        $CmakeGenerator = $null
        $script:VsInstallPath = $null
        $script:VsInstallPath = 'Z:\none'
        Test-LlamaBuildToolsMissing | Should -Be $Expected
    }

    It 'reports missing when CMake cannot drive the VS 2026 generator (<CanDrive>)' -ForEach @(
        @{ CanDrive = $false; Expected = $true }, @{ CanDrive = $true; Expected = $false }
    ) {
        Mock Get-Command { [pscustomobject]@{ Name = $Name } } -ParameterFilter { $Name -in 'git', 'cmake' }
        function Find-VsBuildTools { $null }
        function Find-Nvcc { $null }
        function Test-CmakeCanDriveGenerator { param($Generator) $CanDrive }
        $HasNvidiaDriverEvidence = $false
        $CmakeGenerator = 'Visual Studio 18 2026'
        $script:VsInstallPath = 'C:\VS'
        Test-LlamaBuildToolsMissing | Should -Be $Expected
    }

    It 'reports missing when VS lacks the CUDA integration Resolve-CudaToolkit would copy (<Targets>)' -ForEach @(
        @{ Targets = $false; Expected = $true }, @{ Targets = $true; Expected = $false }
    ) {
        Mock Get-Command { [pscustomobject]@{ Name = $Name } } -ParameterFilter { $Name -in 'git', 'cmake' }
        $cuda = Join-Path $TestDrive 'CUDA\v12.8'
        $vs = Join-Path $TestDrive 'VS'
        $null = New-Item -ItemType Directory -Force (Join-Path $cuda 'bin'), (Join-Path $cuda 'extras\visual_studio_integration\MSBuildExtensions')
        $custom = Get-VcBuildCustomizationsDir -VsInstallPath $vs -Generator 'Visual Studio 17 2022'
        Remove-Item -Recurse -Force $custom -ErrorAction SilentlyContinue
        $null = New-Item -ItemType Directory -Force $custom
        if ($Targets) { $null = New-Item -ItemType File (Join-Path $custom 'CUDA 12.8.targets') }
        $nvccPath = Join-Path $cuda 'bin\nvcc.exe'
        function Find-VsBuildTools { $null }
        function Find-Nvcc { $nvccPath }
        $HasNvidiaDriverEvidence = $true
        $CmakeGenerator = 'Visual Studio 17 2022'
        $script:VsInstallPath = $vs
        Test-LlamaBuildToolsMissing | Should -Be $Expected
    }
}

Describe 'only the automatic fallback is gated' {
    It 'marks the fallback right where the failed prebuilt hands over to a source build' {
        $at = $script:SetupText.IndexOf('failed validation -- falling back to source build')
        $at | Should -BeGreaterThan 0
        $next = $script:SetupText.IndexOf('$script:LlamaSourceBuildIsFallback = $true')
        $next | Should -BeGreaterThan $at
        $script:SetupText.Substring($at, $next - $at).Split("`n").Count | Should -BeLessOrEqual 3
    }

    It 'a declined toolchain finishes the install instead of failing it' {
        $declined = $script:SetupText.IndexOf('} elseif ($script:LlamaBuildToolsDeclined) {', $script:SetupText.IndexOf('[TAURI:DIAG] llama_cpp=unavailable'))
        $fatal = $script:SetupText.IndexOf('Exit-SetupFailure "llama.cpp setup did not produce a usable server"')
        $declined | Should -BeGreaterThan 0
        $declined | Should -BeLessThan $fatal
    }

    It 'decides consent before Phase 3.5 installs OpenSSL, and OpenSSL honours it' {
        $decide = $script:SetupText.IndexOf('$script:LlamaBuildToolsDeclined = $NeedLlamaSourceBuild')
        $openssl = $script:SetupText.IndexOf('ShiningLight.OpenSSL.Dev')
        $decide | Should -BeGreaterThan 0
        $decide | Should -BeLessThan $openssl
        $script:SetupText | Should -Match ([regex]::Escape('if ($NeedLlamaSourceBuild -and -not $script:LlamaBuildToolsDeclined) {'))
    }

    It 'a fallback without consent skips the OpenSSL install even when every other tool is present' {
        $skip = $script:SetupText.IndexOf('} elseif ($script:LlamaSourceBuildIsFallback -and -not (Test-LlamaBuildToolsInstallAllowed)) {')
        $install = $script:SetupText.IndexOf('winget install -e --id ShiningLight.OpenSSL.Dev')
        $skip | Should -BeGreaterThan 0
        $skip | Should -BeLessThan $install
        $script:SetupText.Substring($skip, $install - $skip) | Should -Match 'building without HTTPS'
    }

    It 'the gate checks the fallback flag before asking' {
        $script:SetupText | Should -Match ([regex]::Escape('$script:LlamaSourceBuildIsFallback -and (Test-LlamaBuildToolsMissing) -and'))
    }
}
