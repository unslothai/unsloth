# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    setup.ps1 writes uv to $USERPROFILE\.local\bin and used to probe for it with
    Get-Command alone, which reads PATH. Only astral's own installer edits the registry
    PATH, so a fresh update process never saw the uv already on disk and downloaded it
    again, every run. Get-UvInstallDir is the shared answer to "where does uv go", used by
    both the installer and the probe so the two cannot drift.

    These drive the real function out of setup.ps1 against the inline logic it replaced.
    The nonexistent-drive case is the one that matters: Join-Path returns null there under
    ErrorActionPreference Continue, and the original fell through to the home fallback.
#>

BeforeAll {
    $setup = Join-Path $PSScriptRoot "../../studio/setup.ps1"
    $match = [regex]::Match((Get-Content -Raw $setup), '(?ms)^function Get-UvInstallDir \{.*?^\}')
    if (-not $match.Success) { throw "Get-UvInstallDir is no longer a top-level function in setup.ps1" }
    Invoke-Expression $match.Value

    # The block Get-UvInstallDir was extracted from, kept verbatim as the oracle.
    function Original-Inline {
        $destDir = $null
        foreach ($candidate in @($env:UV_INSTALL_DIR, $env:UV_UNMANAGED_INSTALL, $env:XDG_BIN_HOME)) {
            if ($candidate) { $destDir = $candidate; break }
        }
        if (-not $destDir -and $env:XDG_DATA_HOME) { $destDir = Join-Path $env:XDG_DATA_HOME "../bin" }
        if (-not $destDir) {
            $userHome = if ($env:USERPROFILE) { $env:USERPROFILE } else { $HOME }
            if (-not $userHome) { return $null }
            $destDir = Join-Path $userHome ".local\bin"
        }
        return $destDir
    }

    $script:saved = @{}
    foreach ($name in 'UV_INSTALL_DIR', 'UV_UNMANAGED_INSTALL', 'XDG_BIN_HOME', 'XDG_DATA_HOME', 'USERPROFILE') {
        $script:saved[$name] = [Environment]::GetEnvironmentVariable($name)
    }
}

AfterAll {
    foreach ($name in $script:saved.Keys) {
        Set-Item -Path "Env:$name" -Value $script:saved[$name] -ErrorAction SilentlyContinue
    }
}

Describe "Get-UvInstallDir matches the inline logic it replaced" {
    It "agrees on every combination of the five variables it reads" {
        $ErrorActionPreference = 'Continue'
        $differing = @()
        # The third XDG_DATA_HOME value is a drive that does not exist: Join-Path answers null.
        foreach ($a in @($null, "/tmp/a")) {
        foreach ($b in @($null, "/tmp/b")) {
        foreach ($c in @($null, "/tmp/c")) {
        foreach ($d in @($null, "/tmp/d", "Q:\nosuch")) {
        foreach ($e in @($null, "/tmp/home")) {
            $env:UV_INSTALL_DIR = $a
            $env:UV_UNMANAGED_INSTALL = $b
            $env:XDG_BIN_HOME = $c
            $env:XDG_DATA_HOME = $d
            $env:USERPROFILE = $e
            $got = Get-UvInstallDir 2>$null
            $want = Original-Inline 2>$null
            if ("$got" -ne "$want") { $differing += "$a/$b/$c/$d/$e got=$got want=$want" }
        }}}}}
        $differing | Should -BeNullOrEmpty
    }

    It "falls through to the home directory when XDG_DATA_HOME cannot be joined" {
        $ErrorActionPreference = 'Continue'
        $env:UV_INSTALL_DIR = $null
        $env:UV_UNMANAGED_INSTALL = $null
        $env:XDG_BIN_HOME = $null
        $env:XDG_DATA_HOME = "Q:\nosuch"
        $env:USERPROFILE = "/tmp/home"
        Get-UvInstallDir 2>$null | Should -Be (Join-Path "/tmp/home" ".local\bin")
    }

    It "honours the documented priority order" {
        $env:USERPROFILE = "/tmp/home"
        $env:XDG_DATA_HOME = "/tmp/d"
        $env:XDG_BIN_HOME = "/tmp/c"
        $env:UV_UNMANAGED_INSTALL = "/tmp/b"
        $env:UV_INSTALL_DIR = "/tmp/a"
        Get-UvInstallDir | Should -Be "/tmp/a"
        $env:UV_INSTALL_DIR = $null
        Get-UvInstallDir | Should -Be "/tmp/b"
        $env:UV_UNMANAGED_INSTALL = $null
        Get-UvInstallDir | Should -Be "/tmp/c"
        $env:XDG_BIN_HOME = $null
        Get-UvInstallDir | Should -Be (Join-Path "/tmp/d" "../bin")
    }
}
