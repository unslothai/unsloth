<#
    Pester v5 unit tests for the Visual Studio 2026 completion helpers in
    studio/setup.ps1:
      - Get-VcBuildCustomizationsDir : derive the VC MSBuild BuildCustomizations
        folder (v160 / v170 / v180) from the detected VS generator.
      - Test-CmakeSupportsGenerator  : gate the "Visual Studio 18 2026" generator
        on CMake >= 4.2 (no-op for older VS generators).

    Both are pure functions (no GPU, no Visual Studio, no CUDA, no network), so the
    suite runs on a stock windows-latest runner - and on any pwsh host.

    The real functions are extracted from setup.ps1 and dot-sourced (the script is
    a top-level installer and cannot be loaded wholesale). Path resolution honors
    $env:SETUP_PS1_PATH (set by the PR-validate workflow) and falls back to the
    repo-relative path. If a target function cannot be found, the suite FAILS
    loudly rather than silently passing.
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

    foreach ($fn in @('Resolve-VsGeneratorFromLabel', 'Find-VsBuildTools', 'Get-VcBuildCustomizationsDir',
                      'Test-CmakeSupportsGenerator', 'Get-CmakeVersion', 'Test-CmakeListsGenerator',
                      'Test-CmakeCanDriveGenerator', 'Get-FallbackVsGenerator',
                      'Ensure-BuildToolsForLlamaSourceBuild', 'Test-VCRedistInstalled')) {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name $fn
        if (-not $src) { throw "Function '$fn' not found in $script:SetupPs1 - cannot test the real code." }
        . ([scriptblock]::Create($src))
    }
}

Describe 'Resolve-VsGeneratorFromLabel (vswhere/dir label -> generator)' {
    # Guards that detection accepts both '18' (the internal major vswhere reports
    # for VS 2026) and the year form.
    It 'maps the VS 2026 internal major "18" to the VS 2026 generator' {
        Resolve-VsGeneratorFromLabel '18' | Should -Be 'Visual Studio 18 2026'
    }
    It 'maps the VS 2026 year label "2026" to the VS 2026 generator' {
        Resolve-VsGeneratorFromLabel '2026' | Should -Be 'Visual Studio 18 2026'
    }
    It 'maps the VS 2022 year "2022" and major "17" to the VS 2022 generator' {
        Resolve-VsGeneratorFromLabel '2022' | Should -Be 'Visual Studio 17 2022'
        Resolve-VsGeneratorFromLabel '17'   | Should -Be 'Visual Studio 17 2022'
    }
    It 'maps 2019/2017 (year and major) to their generators' {
        Resolve-VsGeneratorFromLabel '2019' | Should -Be 'Visual Studio 16 2019'
        Resolve-VsGeneratorFromLabel '16'   | Should -Be 'Visual Studio 16 2019'
        Resolve-VsGeneratorFromLabel '2017' | Should -Be 'Visual Studio 15 2017'
        Resolve-VsGeneratorFromLabel '15'   | Should -Be 'Visual Studio 15 2017'
    }
    It 'trims whitespace (vswhere output can carry a trailing newline)' {
        Resolve-VsGeneratorFromLabel "  18 `n" | Should -Be 'Visual Studio 18 2026'
    }
    It 'returns null for unknown or empty labels' {
        Resolve-VsGeneratorFromLabel '2015' | Should -BeNullOrEmpty
        Resolve-VsGeneratorFromLabel ''     | Should -BeNullOrEmpty
        Resolve-VsGeneratorFromLabel $null  | Should -BeNullOrEmpty
    }
}

Describe 'Find-VsBuildTools (VS 2026 generator discovery)' {
    # Exercises the real discovery entry point. Windows-only: Find-VsBuildTools builds
    # backslash candidate paths that only resolve as directories on Windows.
    BeforeAll {
        # Define in BeforeAll, not the Describe body: Pester 5 runs the body only at
        # discovery, so body-level functions are not visible in the run-phase It blocks.
        function New-FakeVsTree {
            param([string]$Root, [string]$VersionDir, [string]$Edition = 'BuildTools')
            $clDir = Join-Path $Root "Microsoft Visual Studio\$VersionDir\$Edition\VC\Tools\MSVC\14.50.00000\bin\Hostx64\x64"
            New-Item -ItemType Directory -Path $clDir -Force | Out-Null
            New-Item -ItemType File -Path (Join-Path $clDir 'cl.exe') -Force | Out-Null
        }
    }
    BeforeEach {
        $script:OrigPF    = ${env:ProgramFiles}
        $script:OrigPFx86 = ${env:ProgramFiles(x86)}
    }
    AfterEach {
        ${env:ProgramFiles}      = $script:OrigPF
        ${env:ProgramFiles(x86)} = $script:OrigPFx86
    }

    It 'detects a filesystem-only VS 2026 BuildTools install (dir "18")' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF'
        New-FakeVsTree -Root $root -VersionDir '18'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86'   # no vswhere here -> filesystem fallback
        $r = Find-VsBuildTools
        $r.Generator | Should -Be 'Visual Studio 18 2026'
    }

    It 'detects a filesystem-only VS 2026 install under the year dir ("2026")' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF2026'
        New-FakeVsTree -Root $root -VersionDir '2026'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86b'
        (Find-VsBuildTools).Generator | Should -Be 'Visual Studio 18 2026'
    }

    It 'still detects VS 2022 (no regression)' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF2022'
        New-FakeVsTree -Root $root -VersionDir '2022'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86c'
        (Find-VsBuildTools).Generator | Should -Be 'Visual Studio 17 2022'
    }

    It 'detects an older VS installed under the Preview edition dir' -Skip:(-not $IsWindows) {
        # Preview installs under a "Preview" edition folder; the fallback must include it.
        $root = Join-Path $TestDrive 'PF2022prev'
        New-FakeVsTree -Root $root -VersionDir '2022' -Edition 'Preview'
        ${env:ProgramFiles}      = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86d'
        (Find-VsBuildTools).Generator | Should -Be 'Visual Studio 17 2022'
    }
}

Describe 'Get-VcBuildCustomizationsDir (CUDA to VS MSBuild integration path)' {
    # Use TestDrive as the root so Join-Path resolves on any OS; assertions accept
    # either path separator.

    It 'derives v180 for the VS 2026 generator' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 18 2026' |
            Should -Match 'VC[\\/]v180[\\/]BuildCustomizations$'
    }

    It 'derives v170 for VS 2022 (unchanged behavior)' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 17 2022' |
            Should -Match 'VC[\\/]v170[\\/]BuildCustomizations$'
    }

    It 'derives v160 for VS 2019' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 16 2019' |
            Should -Match 'VC[\\/]v160[\\/]BuildCustomizations$'
    }

    It 'falls back to v170 when the generator is empty/unparseable (backwards compatible)' {
        Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator '' |
            Should -Match 'VC[\\/]v170[\\/]BuildCustomizations$'
    }

    It 'roots the path under the supplied VS install path' {
        $p = Get-VcBuildCustomizationsDir -VsInstallPath "$TestDrive" -Generator 'Visual Studio 18 2026'
        $p.StartsWith("$TestDrive") | Should -BeTrue
    }
}

Describe 'Test-CmakeSupportsGenerator (CMake 4.2 guard for VS 2026)' {

    It 'rejects CMake 3.31.0 with the VS 2026 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '3.31.0' -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }

    It 'accepts CMake 4.2.1 with the VS 2026 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '4.2.1' -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'accepts CMake exactly 4.2 with the VS 2026 generator (boundary)' {
        Test-CmakeSupportsGenerator -CmakeVersion '4.2' -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'rejects CMake 4.1.0 with the VS 2026 generator (boundary)' {
        Test-CmakeSupportsGenerator -CmakeVersion '4.1.0' -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }

    It 'is a no-op (accepts any CMake) for the VS 2022 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '3.20.0' -Generator 'Visual Studio 17 2022' | Should -BeTrue
    }

    It 'is a no-op (accepts any CMake) for the VS 2019 generator' {
        Test-CmakeSupportsGenerator -CmakeVersion '3.10.0' -Generator 'Visual Studio 16 2019' | Should -BeTrue
    }
}

Describe 'Test-CmakeListsGenerator (probe cmake --help)' {
    # Mock cmake as a function (resolved before any on-PATH exe): PowerShell caches
    # its app-path table, so a $env:Path shim would not reliably beat a real cmake.

    It 'returns true when cmake --help lists the generator' {
        Mock cmake { "Generators`n  Visual Studio 18 2026        = Generates VS 2026 project files.`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        Test-CmakeListsGenerator -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'returns false when cmake --help does not list the generator' {
        Mock cmake { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        Test-CmakeListsGenerator -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }

    It 'returns false when cmake produces no help output' {
        Mock cmake { $null }
        Test-CmakeListsGenerator -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }
}

Describe 'Test-CmakeCanDriveGenerator (probe OR version floor)' {
    It 'accepts a sub-4.2 cmake that lists the VS 2026 generator (bundled cmake)' {
        # 3.31.0 is below the 4.2 floor but lists the generator, so the help-probe accepts it.
        Mock cmake {
            if ($args -contains '--version') { 'cmake version 3.31.0' }
            else { "Generators`n  Visual Studio 18 2026        = Generates VS 2026 project files." }
        }
        Test-CmakeCanDriveGenerator -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'accepts a 4.2 cmake via the version floor when the help probe misses it' {
        # Help omits the generator but 4.2.0 meets the floor, so the version branch accepts it.
        Mock cmake {
            if ($args -contains '--version') { 'cmake version 4.2.0' }
            else { 'Generators' }
        }
        Test-CmakeCanDriveGenerator -Generator 'Visual Studio 18 2026' | Should -BeTrue
    }

    It 'rejects a sub-4.2 cmake that does not list the VS 2026 generator' {
        Mock cmake {
            if ($args -contains '--version') { 'cmake version 3.31.0' }
            else { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        }
        Test-CmakeCanDriveGenerator -Generator 'Visual Studio 18 2026' | Should -BeFalse
    }
}

Describe 'Get-FallbackVsGenerator (older VS the cmake can drive)' {
    BeforeAll {
        function New-FakeVsTree2 {
            param([string]$Root, [string]$VersionDir, [string]$Edition = 'BuildTools')
            $clDir = Join-Path $Root "Microsoft Visual Studio\$VersionDir\$Edition\VC\Tools\MSVC\14.39.00000\bin\Hostx64\x64"
            New-Item -ItemType Directory -Path $clDir -Force | Out-Null
            New-Item -ItemType File -Path (Join-Path $clDir 'cl.exe') -Force | Out-Null
        }
    }
    BeforeEach {
        $script:OrigPF = ${env:ProgramFiles}
        $script:OrigPFx86 = ${env:ProgramFiles(x86)}
    }
    AfterEach {
        ${env:ProgramFiles} = $script:OrigPF
        ${env:ProgramFiles(x86)} = $script:OrigPFx86
    }

    It 'returns the VS 2022 generator when VS 2022 is installed and cmake lists it' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF_fb'
        New-FakeVsTree2 -Root $root -VersionDir '2022'
        ${env:ProgramFiles} = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86_fb'
        Mock cmake { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        $r = Get-FallbackVsGenerator
        $r.Generator | Should -Be 'Visual Studio 17 2022'
    }

    It 'returns null when the cmake cannot drive any installed older VS' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF_none'
        New-FakeVsTree2 -Root $root -VersionDir '2022'
        ${env:ProgramFiles} = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86_none'
        # cmake lists only VS 2026 (not 2022/2019/2017), so no older fallback is usable.
        Mock cmake { "Generators`n  Visual Studio 18 2026        = Generates VS 2026 project files." }
        $r = Get-FallbackVsGenerator
        $r | Should -BeNullOrEmpty
    }

    It 'falls back to an older VS installed under the Preview edition dir' -Skip:(-not $IsWindows) {
        $root = Join-Path $TestDrive 'PF_prev'
        New-FakeVsTree2 -Root $root -VersionDir '2022' -Edition 'Preview'
        ${env:ProgramFiles} = $root
        ${env:ProgramFiles(x86)} = Join-Path $TestDrive 'PFx86_prev'
        Mock cmake { "Generators`n  Visual Studio 17 2022        = Generates VS 2022 project files." }
        (Get-FallbackVsGenerator).Generator | Should -Be 'Visual Studio 17 2022'
    }
}

Describe 'Deferred build tools (prebuilt path needs no VS/CMake)' {
    # Phase-1 detection must be non-fatal (prebuilt path never blocked) and the
    # deferred installer must no-op when VS was already detected. The install +
    # exit-1 path is covered by studio-windows-no-vs-smoke.yml.
    BeforeEach {
        $script:OrigPF    = ${env:ProgramFiles}
        $script:OrigPFx86 = ${env:ProgramFiles(x86)}
    }
    AfterEach {
        ${env:ProgramFiles}      = $script:OrigPF
        ${env:ProgramFiles(x86)} = $script:OrigPFx86
        $script:VsInstallPath = $null
        $script:CmakeGenerator = $null
    }

    It 'Find-VsBuildTools returns null when no VS is present (probe stays non-fatal)' {
        # Empty discovery roots so no VS is found; the probe must return null
        # (then log and continue, never exit).
        ${env:ProgramFiles}      = (Join-Path $TestDrive 'EmptyPF')
        ${env:ProgramFiles(x86)} = (Join-Path $TestDrive 'EmptyPFx86')
        New-Item -ItemType Directory -Force -Path ${env:ProgramFiles}, ${env:ProgramFiles(x86)} | Out-Null
        Find-VsBuildTools | Should -BeNullOrEmpty
    }

    It 'Ensure-BuildToolsForLlamaSourceBuild no-ops when VS is already detected' {
        # With $VsInstallPath already set, the deferred installer must return without
        # re-scanning or installing.
        $script:VsInstallPath  = 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
        $script:CmakeGenerator = 'Visual Studio 17 2022'
        { Ensure-BuildToolsForLlamaSourceBuild } | Should -Not -Throw
        $script:VsInstallPath  | Should -Be 'C:\Program Files\Microsoft Visual Studio\2022\BuildTools'
        $script:CmakeGenerator | Should -Be 'Visual Studio 17 2022'
    }
}

Describe 'Source-build ordering invariant: CUDA integration runs AFTER the VS generator is finalized (#6473 review)' {
    # Resolve-CudaToolkit copies the CUDA .targets into the current generator's dir,
    # so it must run after the VS 2026 gate/fallback; otherwise a fallback to VS 2022
    # builds v170 while the .targets went to v180 ("No CUDA toolset found").
    It 'the source-build Resolve-CudaToolkit call appears AFTER the Get-FallbackVsGenerator fallback' {
        $text = Get-Content -Raw -LiteralPath $script:SetupPs1
        $idxFallback = $text.IndexOf('$fallback = Get-FallbackVsGenerator')
        $idxResolve  = $text.IndexOf('Resolve-CudaToolkit -RequireOrExit')
        $idxFallback | Should -BeGreaterThan 0
        $idxResolve  | Should -BeGreaterThan 0
        $idxResolve  | Should -BeGreaterThan $idxFallback
    }
}

Describe 'Get-FallbackVsGenerator discovery is symmetric with Find-VsBuildTools (#6473 review)' {
    # The fallback must also query vswhere, else a VS in a custom location is found
    # as primary but missed as fallback -> avoidable hard exit.
    It 'queries vswhere as part of fallback discovery' {
        $src = Get-FunctionSource -Path $script:SetupPs1 -Name Get-FallbackVsGenerator
        $src | Should -Match 'vswhere'
    }
}

Describe 'Test-VCRedistInstalled (VC++ 2015-2022 runtime needed by the prebuilt llama.cpp + PyTorch)' {
    # The prebuilts link the VC++ runtime DLLs (which the Universal CRT lacks);
    # detection is System32\vcruntime140_1.dll with a registry fallback.
    BeforeEach { $script:OrigSysRoot = $env:SystemRoot }
    AfterEach  { $env:SystemRoot = $script:OrigSysRoot }

    # Probes Test-Path once (System32 DLL), then the registry; mock both.
    It 'returns true when vcruntime140_1.dll is present in System32' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $true }
        Test-VCRedistInstalled | Should -BeTrue
    }
    It 'returns true via the registry when the DLL is not found (Installed=1, >= 14.20)' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $false }
        Mock Get-ItemProperty { [pscustomobject]@{ Installed = 1; Major = 14; Minor = 29 } }
        Test-VCRedistInstalled | Should -BeTrue
    }
    It 'returns false when neither the DLL nor a >= 14.20 registry entry exists' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $false }
        Mock Get-ItemProperty { throw 'no key' }
        Test-VCRedistInstalled | Should -BeFalse
    }
    It 'returns false for an old 2015-only redist (Installed=1 but < 14.20)' {
        $env:SystemRoot = 'C:\Windows'
        Mock Test-Path { $false }
        Mock Get-ItemProperty { [pscustomobject]@{ Installed = 1; Major = 14; Minor = 0 } }
        Test-VCRedistInstalled | Should -BeFalse
    }
}
