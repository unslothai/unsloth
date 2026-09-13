# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    install.ps1 records its uv cache in <StudioHome>/cache/uv-cache-dir; studio/setup.sh,
    unsloth_cli and the staging promote read it back. The contract is one absolute path and
    one LF in plain UTF-8: `Set-Content -Encoding utf8` emits a BOM under Windows PowerShell
    5.1 but not 7, and the update digests this file, so a marker flipping between CRLF+BOM
    and LF read as a change that never happened. pwsh 7 cannot show the 5.1 defect, hence
    the source guard, which fails on the pre-fix text on every platform.
#>

BeforeAll {
    . (Join-Path $PSScriptRoot 'Get-FunctionSource.ps1')

    $candidates = @(
        $env:INSTALL_PS1_PATH,
        (Join-Path $PSScriptRoot '..\..\install.ps1')
    ) | Where-Object { $_ }
    $script:InstallPs1 = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $script:InstallPs1) { throw "Could not locate install.ps1 (set INSTALL_PS1_PATH)." }
    Write-Host "install.ps1 under test: $script:InstallPs1"

    $script:InstallText = Get-Content -Raw -LiteralPath $script:InstallPs1

    # Only the three marker functions, dot-sourced: install.ps1 runs work at load and cannot be sourced whole.
    $script:MarkerFunctions = @(
        'Resolve-StudioUvCachePath'
        'Write-StudioUvCacheMarker'
        'Restore-StudioUvCacheMarker'
    )
    foreach ($name in $script:MarkerFunctions) {
        $src = Get-FunctionSource -Path $script:InstallPs1 -Name $name
        if (-not $src) { throw "$name is gone from install.ps1." }
        . ([scriptblock]::Create($src))
    }

    # The writer region, sliced as tests/python/test_cross_platform_parity.py slices it.
    $writeStart = $script:InstallText.IndexOf('function Write-StudioUvCacheMarker')
    $writeEnd = $script:InstallText.IndexOf('function Set-StudioUvCacheEnvironment')
    if ($writeStart -lt 0 -or $writeEnd -le $writeStart) {
        throw "could not slice the marker functions out of install.ps1."
    }
    $script:MarkerRegion = $script:InstallText.Substring($writeStart, $writeEnd - $writeStart)

    # The same region with comments blanked: they NAME the defect, so a prose match would fire
    # on the fix. Blanked rather than deleted so offsets and counts stay comparable.
    $script:MarkerCode = & {
        $tokens = $null
        $parseErrors = $null
        [System.Management.Automation.Language.Parser]::ParseInput(
            $script:MarkerRegion, [ref]$tokens, [ref]$parseErrors) | Out-Null
        if (@($parseErrors).Count -gt 0) {
            throw "the sliced marker functions do not parse: $($parseErrors[0].Message)"
        }
        $chars = [char[]]$script:MarkerRegion
        foreach ($token in $tokens) {
            if ($token.Kind -ne [System.Management.Automation.Language.TokenKind]::Comment) { continue }
            for ($i = $token.Extent.StartOffset; $i -lt $token.Extent.EndOffset; $i++) { $chars[$i] = ' ' }
        }
        return (-join $chars)
    }
    if ($script:MarkerCode -match 'BOM') {
        throw "comment stripping failed -- the source guards below would match prose."
    }

    function script:New-MarkerRoot {
        # A space (a Windows StudioHome usually has one) and a non-ASCII segment outside the ANSI code page.
        $root = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth marker käffee " + [guid]::NewGuid())
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        return $root
    }

    function script:Reset-MarkerState {
        $script:StudioUvMarkerSaved = $false
        $script:StudioUvMarkerExisted = $false
        $script:StudioUvMarkerPrevious = $null
        $script:StudioInstallCommitted = $false
    }

    function script:Read-MarkerLikeTheCli {
        <#
            unsloth_cli/commands/studio.py:_recorded_install_uv_cache, transliterated: decode
            utf-8-sig, drop ONE trailing "\n" and then ONE trailing "\r", keep everything
            else -- a path may legitimately begin or end with a space.
        #>
        param([Parameter(Mandatory = $true)][string]$Path)
        $bytes = [System.IO.File]::ReadAllBytes($Path)
        if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
            $bytes = $bytes[3..($bytes.Length - 1)]
        }
        $text = (New-Object System.Text.UTF8Encoding($false)).GetString($bytes)
        if ($text.EndsWith("`n")) { $text = $text.Substring(0, $text.Length - 1) }
        if ($text.EndsWith("`r")) { $text = $text.Substring(0, $text.Length - 1) }
        return $text
    }
}

Describe 'the uv cache marker install.ps1 writes' {
    BeforeEach {
        Reset-MarkerState
        $script:Root = New-MarkerRoot
        $script:Marker = Join-Path (Join-Path $script:Root 'cache') 'uv-cache-dir'
    }

    AfterEach {
        if ($script:Root) { Remove-Item -Recurse -Force -LiteralPath $script:Root -ErrorAction SilentlyContinue }
    }

    It 'does not start with a UTF-8 BOM' {
        Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache (Join-Path $script:Root 'uv')
        $bytes = [System.IO.File]::ReadAllBytes($script:Marker)
        $bytes.Length | Should -BeGreaterThan 3
        # 0xEF 0xBB 0xBF, byte-for-byte: both readers cope and would hide the defect.
        @($bytes[0], $bytes[1], $bytes[2]) -join ',' | Should -Not -Be '239,187,191' -Because (
            'a BOM is what `Set-Content -Encoding utf8` writes under Windows PowerShell 5.1, ' +
            'and _studio_stage.promote copies these bytes verbatim')
    }

    It 'ends in exactly one LF and no CR' {
        Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache (Join-Path $script:Root 'uv')
        $bytes = [System.IO.File]::ReadAllBytes($script:Marker)
        # One record, one delimiter, as install.sh and the CLI write and the update digests.
        $bytes[$bytes.Length - 1] | Should -Be 0x0A
        $bytes[$bytes.Length - 2] | Should -Not -Be 0x0D -Because 'every other writer of this file uses LF'
        $bytes[$bytes.Length - 2] | Should -Not -Be 0x0A -Because 'one trailing newline, not two'
    }

    It 'round-trips a non-ASCII path through the BOM-tolerant reader' {
        # Not the marker root, so the recorded value differs from the path the marker file derives from.
        $cache = Join-Path $script:Root "uv café キャッシュ"
        Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache $cache
        Read-MarkerLikeTheCli -Path $script:Marker | Should -BeExactly $cache
    }

    It 'writes the same bytes on rollback as it wrote on the way in' {
        # A restore in another encoding than it saved would itself be a digestible change.
        $previous = Join-Path $script:Root "previous café"
        New-Item -ItemType Directory -Path (Split-Path -Parent $script:Marker) -Force | Out-Null
        # As install.sh and the Python backfill write it: BOM-less UTF-8, one LF.
        [System.IO.File]::WriteAllBytes($script:Marker,
            (New-Object System.Text.UTF8Encoding($false)).GetBytes($previous + "`n"))
        $before = [System.IO.File]::ReadAllBytes($script:Marker)

        Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache (Join-Path $script:Root 'chosen')
        Restore-StudioUvCacheMarker -StudioRoot $script:Root

        $after = [System.IO.File]::ReadAllBytes($script:Marker)
        ($after -join ',') | Should -BeExactly ($before -join ',')
    }

    It 'never throws when the marker cannot be written' {
        # WriteAllText throws where -ErrorAction SilentlyContinue warned; a marker is a preference.
        # A directory in the file's place is the cheapest portable unwritable path.
        New-Item -ItemType Directory -Path $script:Marker -Force | Out-Null
        # A directory survives the writer's Remove-Item -Force, so its Get-Item guard returns;
        # the restore has no such guard and must be silent too.
        { Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache (Join-Path $script:Root 'uv') } |
            Should -Not -Throw
        $script:StudioUvMarkerSaved = $true
        $script:StudioUvMarkerExisted = $true
        $script:StudioUvMarkerPrevious = [System.Text.Encoding]::UTF8.GetBytes("/previous/cache`n")
        { Restore-StudioUvCacheMarker -StudioRoot $script:Root } | Should -Not -Throw
    }
}

Describe 'the marker writer source' {
    <#
        The byte cases above run under pwsh 7, which writes no BOM even with the old
        `Set-Content -Encoding utf8`, so on this host they pass either way. These do not:
        they read the text of install.ps1, and they fail on the pre-fix file on every
        platform. Point INSTALL_PS1_PATH at an older install.ps1 to see it.
    #>

    It 'writes the marker with an explicit BOM-less UTF8Encoding' {
        # One text write, the record; the restore writes saved bytes and names no encoding.
        ([regex]::Matches($script:MarkerCode,
            [regex]::Escape('New-Object System.Text.UTF8Encoding($false)'))).Count |
            Should -Be 1 -Because (
                'the marker write must name the encoder. Windows PowerShell 5.1 emits a ' +
                'UTF-8 BOM for -Encoding utf8 and PowerShell 7 does not, so anything that ' +
                'leaves the encoding to the host writes a different file on each')
        $script:MarkerCode | Should -Match ([regex]::Escape(
            '[System.IO.File]::WriteAllBytes($markerFile, [byte[]]$script:StudioUvMarkerPrevious)')) -Because (
                'the rollback puts back the bytes it saved, whichever writer produced them')
        $script:MarkerCode | Should -Match ([regex]::Escape('[System.IO.File]::ReadAllBytes($markerFile)'))
    }

    It 'writes the marker with no host-encoded cmdlet at all' {
        # Each takes the host encoding when none is named, and `utf8` is BOM-ed on 5.1. The
        # cmdlets are banned, not `-Encoding utf8`: the reader legitimately passes
        # `Get-Content -Encoding UTF8` (5.1 decodes BOM-less as ANSI, hence the mojibake).
        foreach ($cmdlet in @('Set-Content', 'Out-File', 'Add-Content', 'Tee-Object')) {
            $script:MarkerCode | Should -Not -Match ([regex]::Escape($cmdlet)) -Because (
                "$cmdlet picks its encoding from the host, and this file is read by " +
                'setup.sh, unsloth_cli and _studio_stage. Use [System.IO.File]::WriteAllText ' +
                'with a New-Object System.Text.UTF8Encoding($false)')
        }
    }

    It 'appends the trailing newline itself' {
        # WriteAllText adds no newline and the contract is one path, one delimiter. One write: the record.
        ([regex]::Matches($script:MarkerCode, [regex]::Escape('+ "`n"'))).Count |
            Should -Be 1 -Because 'Set-Content supplied the newline; WriteAllText does not'
        $script:MarkerCode | Should -Not -Match '\+ "`r`n"' -Because (
            'install.sh, unsloth_cli and _studio_stage all write LF, and the update ' +
            'digests these bytes')
    }

    It 'keeps the symlink defence in front of every write' {
        # WriteAllText follows a symlink as Set-Content did: remove first, re-check with Get-Item -Force.
        ([regex]::Matches($script:MarkerCode,
            [regex]::Escape('Remove-Item -LiteralPath $markerFile -Force -ErrorAction SilentlyContinue'))).Count |
            Should -Be 2 -Because 'a write follows a symlink and would truncate its target'
        ([regex]::Matches($script:MarkerCode,
            [regex]::Escape('Get-Item -LiteralPath $markerFile -Force -ErrorAction SilentlyContinue'))).Count |
            Should -Be 2 -Because 'Get-Item -Force reports the link itself; Test-Path would follow it'
    }

    It 'swallows a failed write instead of throwing' {
        # WriteAllText throws where the -ErrorAction SilentlyContinue it replaced did not.
        ([regex]::Matches($script:MarkerCode, [regex]::Escape('} catch { }'))).Count |
            Should -BeGreaterOrEqual 2 -Because (
                'a marker is a preference, and a throw out of Restore-StudioUvCacheMarker ' +
                'is reported as a failed environment restore')
        $script:MarkerCode | Should -Not -Match '-ErrorAction Stop' -Because (
            'no marker operation may fail the install')
    }
}
