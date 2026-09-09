# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
<#
    install.ps1 records the uv cache it chose in <StudioHome>/cache/uv-cache-dir, and
    three things outside PowerShell read that file back: studio/setup.sh
    (_recorded_uv_cache), unsloth_cli's _recorded_install_uv_cache, and the staging
    promote in unsloth_cli/_studio_stage.py, which copies its BYTES.

    The contract is one absolute path and one trailing newline, in plain UTF-8. Both
    readers merely TOLERATE a UTF-8 BOM -- setup.sh strips $_UV_MARKER_BOM, the CLI reads
    utf-8-sig -- because `Set-Content -Encoding utf8` emits one under Windows PowerShell
    5.1 while PowerShell 7 does not, so the installer used to write two different files
    depending on the host that ran it. Tolerance is not the contract: install.sh writes
    `printf '%s\n'` and the CLI writes os.fsencode(f"{chosen}\n"), and the update
    procedure digests this file to decide whether a rerun changed anything, so a marker
    that flips between CRLF+BOM and LF reads as a change that never happened.

    These tests run under pwsh 7, where the OLD code emitted no BOM either, so the
    byte-level cases below cannot see the 5.1 defect on their own. That is why the source
    guard exists as well: it fails on the pre-fix text on every platform.
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

    # Only the three functions the marker needs, dot-sourced into this scope: install.ps1
    # is an installer that runs work at load, so it can never be dot-sourced whole. The
    # same three the Python hardening suite slices out, and for the same reason.
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

    # The writer region, sliced exactly as tests/python/test_cross_platform_parity.py
    # slices it: both marker functions, and nothing else.
    $writeStart = $script:InstallText.IndexOf('function Write-StudioUvCacheMarker')
    $writeEnd = $script:InstallText.IndexOf('function Set-StudioUvCacheEnvironment')
    if ($writeStart -lt 0 -or $writeEnd -le $writeStart) {
        throw "could not slice the marker functions out of install.ps1."
    }
    $script:MarkerRegion = $script:InstallText.Substring($writeStart, $writeEnd - $writeStart)

    # The same region with every comment blanked out, because the comments in it NAME the
    # defect ("NOT `Set-Content -Encoding utf8`") and a guard that matched prose would fire
    # on the fix and pass on a file that only stopped explaining itself. Blanked rather
    # than deleted so offsets, and so the counts below, stay comparable.
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
        # A space in the name because a Windows StudioHome usually has one, and a
        # non-ASCII segment because the whole point of naming the encoder is that a path
        # outside the active ANSI code page survives the round trip.
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
        # 0xEF 0xBB 0xBF. Byte-for-byte rather than "does the reader cope", because both
        # readers cope and would hide the defect.
        @($bytes[0], $bytes[1], $bytes[2]) -join ',' | Should -Not -Be '239,187,191' -Because (
            'a BOM is what `Set-Content -Encoding utf8` writes under Windows PowerShell 5.1, ' +
            'and _studio_stage.promote copies these bytes verbatim')
    }

    It 'ends in exactly one LF and no CR' {
        Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache (Join-Path $script:Root 'uv')
        $bytes = [System.IO.File]::ReadAllBytes($script:Marker)
        # One record, one delimiter: install.sh's `printf '%s\n'` and the CLI's
        # f"{chosen}\n" produce exactly this, and the update digests the file.
        $bytes[$bytes.Length - 1] | Should -Be 0x0A
        $bytes[$bytes.Length - 2] | Should -Not -Be 0x0D -Because 'every other writer of this file uses LF'
        $bytes[$bytes.Length - 2] | Should -Not -Be 0x0A -Because 'one trailing newline, not two'
    }

    It 'round-trips a non-ASCII path through the BOM-tolerant reader' {
        # Not the marker root, so the recorded value is distinguishable from the path the
        # test derived the marker file from.
        $cache = Join-Path $script:Root "uv café キャッシュ"
        Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache $cache
        Read-MarkerLikeTheCli -Path $script:Marker | Should -BeExactly $cache
    }

    It 'writes the same bytes on rollback as it wrote on the way in' {
        # The rollback restores a marker it read from disk. Restoring it in a different
        # encoding than the one it saved would itself be a change the update can digest.
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
        # WriteAllText throws where `-ErrorAction SilentlyContinue` only warned. A marker
        # is a preference, not a requirement, and a throw out of the restore is reported
        # as a failed environment restore. A directory in the file's place is the cheapest
        # portable unwritable path: File.WriteAllText raises UnauthorizedAccessException.
        New-Item -ItemType Directory -Path $script:Marker -Force | Out-Null
        # A directory survives the Remove-Item -Force the writer does first, so the
        # Get-Item guard sees it and returns. Prove the restore, which has no such guard
        # once the removal fails, is silent too.
        { Write-StudioUvCacheMarker -StudioRoot $script:Root -Cache (Join-Path $script:Root 'uv') } |
            Should -Not -Throw
        $script:StudioUvMarkerSaved = $true
        $script:StudioUvMarkerExisted = $true
        $script:StudioUvMarkerPrevious = "/previous/cache`n"
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
        # Two writes: the record and the rollback restore.
        ([regex]::Matches($script:MarkerCode,
            [regex]::Escape('New-Object System.Text.UTF8Encoding($false)'))).Count |
            Should -Be 2 -Because (
                'both marker writes must name the encoder. Windows PowerShell 5.1 emits a ' +
                'UTF-8 BOM for -Encoding utf8 and PowerShell 7 does not, so anything that ' +
                'leaves the encoding to the host writes a different file on each')
    }

    It 'writes the marker with no host-encoded cmdlet at all' {
        # Every one of these takes its encoding from the host when none is named, and the
        # named `utf8` is BOM-ed on 5.1 and BOM-less on 7. Banning the cmdlets outright,
        # rather than banning `-Encoding utf8`, is deliberate: the reader a few lines above
        # legitimately passes `Get-Content -Encoding UTF8` (5.1 decodes a BOM-less file with
        # the active ANSI code page, which is how a non-ASCII path came back as mojibake),
        # so a blanket ban on the switch would forbid the fix for a different defect.
        foreach ($cmdlet in @('Set-Content', 'Out-File', 'Add-Content', 'Tee-Object')) {
            $script:MarkerCode | Should -Not -Match ([regex]::Escape($cmdlet)) -Because (
                "$cmdlet picks its encoding from the host, and this file is read by " +
                'setup.sh, unsloth_cli and _studio_stage. Use [System.IO.File]::WriteAllText ' +
                'with a New-Object System.Text.UTF8Encoding($false)')
        }
    }

    It 'appends the trailing newline itself' {
        # WriteAllText adds none, and the contract is one path and one delimiter, so a
        # write that dropped it would join this record to whatever read it next.
        ([regex]::Matches($script:MarkerCode, [regex]::Escape('+ "`n"'))).Count |
            Should -Be 2 -Because 'Set-Content supplied the newline; WriteAllText does not'
        $script:MarkerCode | Should -Not -Match '\+ "`r`n"' -Because (
            'install.sh, unsloth_cli and _studio_stage all write LF, and the update ' +
            'digests these bytes')
    }

    It 'keeps the symlink defence in front of every write' {
        # WriteAllText follows a symlink exactly as Set-Content did, so removing the link
        # first and re-checking with Get-Item -Force is still the only thing between this
        # writer and a truncated target elsewhere on disk.
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
