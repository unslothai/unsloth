#!/usr/bin/env pwsh
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
# The ownership gate must still recognise the layouts older installers left: on Windows every
# sentinel a root can show postdates the bin\ shim dir, while install.ps1 still migrates .venv.
# What is INSIDE a venv proves ownership only at the managed root, so every case names its mode.
# The uninstaller body writes to the registry, so the helpers are lifted out by AST.
# Run: pwsh -NoProfile -File tests/studio/test_uninstall_legacy_layout_gate.ps1

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path ([System.IO.Path]::Combine($PSScriptRoot, "..", ".."))).Path
$ps1Path = [System.IO.Path]::Combine($repoRoot, "scripts", "uninstall.ps1")

$failures = 0
function Check($name, $cond) {
    if ($cond) { Write-Host "  PASS  $name" }
    else { Write-Host "  FAIL  $name" -ForegroundColor Red; $script:failures++ }
}

$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($ps1Path, [ref]$tokens, [ref]$errors)
Check "uninstall.ps1 parses" ($null -eq $errors -or $errors.Count -eq 0)

# _IsStudioRoot calls the three helpers below, so all of them must come across or the suite is
# vacuous.
$allFns = $ast.FindAll({
        param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
    }, $true)
foreach ($name in @("_IsUnslothCmdShim", "_IsOwnerMarker", "_IsVenvDir", "_IsInstallerLeftoverName", "_IsStudioRoot")) {
    $fn = $allFns | Where-Object { $_.Name -eq $name } | Select-Object -First 1
    if (-not $fn) {
        Write-Host "  FAIL  $name not found in uninstall.ps1" -ForegroundColor Red
        exit 1
    }
    . ([scriptblock]::Create($fn.Extent.Text))
}

$shimText = @"
@echo off
rem unsloth-studio-managed-launcher
"%~dp0..\unsloth_studio\Scripts\python.exe" -X utf8 -c "from unsloth_cli import app" %*
"@

$tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("unsloth-legacy-gate-" + [System.Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null
try {
    function Make([string]$Name, [string[]]$Files) {
        $root = Join-Path $tmp $Name
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        foreach ($f in $Files) {
            $p = Join-Path $root $f
            New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($p)) -Force | Out-Null
            if ($f -eq "bin\unsloth.cmd") { Set-Content -LiteralPath $p -Value $shimText }
            else { Set-Content -LiteralPath $p -Value "x" }
        }
        # Every one of these stands for an install that holds chat history.
        Set-Content -LiteralPath (Join-Path $root "studio.db") -Value "chat history"
        return $root
    }

    # -ManagedDefaultRoot is passed for %USERPROFILE%\.unsloth\studio and for nothing else.
    Check "partial install: the root marker alone" `
        (_IsStudioRoot (Make "root-marker" @(".unsloth-studio-owned")) -ManagedDefaultRoot)
    Check "the root marker proves a custom root too" `
        (_IsStudioRoot (Make "custom-root-marker" @(".unsloth-studio-owned")))
    # A relocated install is still an install. Moving a multi-gigabyte venv to another drive and
    # leaving a junction behind puts a REAL marker behind one, and refusing it would strand the
    # install this gate exists to keep removable. Refusing links here would also buy nothing:
    # anyone who can plant one at these paths can plant a plain file instead. The link test
    # belongs in install.ps1's claim, where following one TRUNCATES the target.
    $markTarget = Join-Path $tmp "linked-marker-target.txt"
    Set-Content -LiteralPath $markTarget -Value "x"
    # (subdirectory, marker) rather than one backslash path: New-Item -Path treats a backslash
    # as an escape off Windows, so a combined literal would not create the link on the Linux row.
    foreach ($pair in @(@("", ".unsloth-studio-owned"), @("unsloth_studio", ".unsloth-studio-owned"),
                        @(".venv", ".unsloth-studio-owned"), @("share", "studio.conf"))) {
        $label = (($pair | Where-Object { $_ }) -join "/")
        $r = Make ("lm-" + ($label -replace '[^A-Za-z0-9]', '_')) @("keepme.txt")
        $at = $r
        if ($pair[0]) { $at = Join-Path $r $pair[0]; New-Item -ItemType Directory -Path $at -Force | Out-Null }
        $ok = $null
        try { $ok = New-Item -ItemType SymbolicLink -Path (Join-Path $at $pair[1]) -Target $markTarget -ErrorAction Stop } catch { $ok = $null }
        if ($ok) { Check "a relocated $label is still recognised" (_IsStudioRoot $r -ManagedDefaultRoot) }
        else { Write-Host "  SKIP  no symlink could be created for $label" }
    }
    # ... and the shape a user really produces: the whole venv directory moved and linked back.
    $realVenvDir = Make "lm-real-venv" @(".unsloth-studio-owned", "pyvenv.cfg")
    $linkedDir = Make "lm-linked-dir" @("keepme.txt")
    $dl = $null
    foreach ($kind in @("Junction", "SymbolicLink")) {
        try {
            $dl = New-Item -ItemType $kind -Path (Join-Path $linkedDir "unsloth_studio") -Target $realVenvDir -ErrorAction Stop
            if ($dl -and -not [string]::IsNullOrWhiteSpace(@($dl.Target)[0])) { break }
        } catch { $dl = $null }
    }
    if ($dl -and -not [string]::IsNullOrWhiteSpace(@($dl.Target)[0])) {
        Check "a relocated venv directory is still recognised" (_IsStudioRoot $linkedDir -ManagedDefaultRoot)
        # The same relocation for a PRE-MARKER install, which reaches the gate by the fallback.
        $preVenv = Make "lm-pre-venv" @("pyvenv.cfg", "Scripts\python.exe", "Scripts\unsloth.exe")
        $preRoot = Make "lm-linked-premarker" @("keepme.txt")
        $dl2 = $null
        foreach ($kind in @("Junction", "SymbolicLink")) {
            try {
                $dl2 = New-Item -ItemType $kind -Path (Join-Path $preRoot ".venv") -Target $preVenv -ErrorAction Stop
                if ($dl2 -and -not [string]::IsNullOrWhiteSpace(@($dl2.Target)[0])) { break }
            } catch { $dl2 = $null }
        }
        if ($dl2) { Check "a relocated pre-marker venv is still recognised" (_IsStudioRoot $preRoot -ManagedDefaultRoot) }
    } else {
        Write-Host "  SKIP  no reparse point could be created here"
    }
    Check "current layout (venv owner marker)" `
        (_IsStudioRoot (Make "cur-marker" @("unsloth_studio\.unsloth-studio-owned")) -ManagedDefaultRoot)
    Check "shim .exe" `
        (_IsStudioRoot (Make "cur-exe" @("bin\unsloth.exe")) -ManagedDefaultRoot)
    Check "shim .cmd alone, when a policy quarantined the .exe" `
        (_IsStudioRoot (Make "cur-cmd" @("bin\unsloth.cmd")) -ManagedDefaultRoot)

    Check "legacy .venv carrying our owner marker" `
        (_IsStudioRoot (Make "old-marker" @(".venv\.unsloth-studio-owned", ".venv\Scripts\python.exe")) -ManagedDefaultRoot)
    Check "legacy .venv carrying the unsloth console script" `
        (_IsStudioRoot (Make "old-cli" @(".venv\Scripts\python.exe", ".venv\Scripts\unsloth.exe")) -ManagedDefaultRoot)
    Check "pre-marker unsloth_studio venv carrying the unsloth console script" `
        (_IsStudioRoot (Make "pre-marker" @("unsloth_studio\Scripts\python.exe", "unsloth_studio\Scripts\unsloth.exe")) -ManagedDefaultRoot)
    # Antivirus takes the .exe out of a venv that still runs, leaving nothing but the package.
    Check "pre-marker venv whose unsloth.exe antivirus quarantined" `
        (_IsStudioRoot (Make "quarantined" @(".venv\Scripts\python.exe", ".venv\Lib\site-packages\unsloth_cli\__init__.py")) -ManagedDefaultRoot)
    Check "pre-marker venv proved by pyvenv.cfg alone" `
        (_IsStudioRoot (Make "cfg-only" @(".venv\pyvenv.cfg", ".venv\Scripts\unsloth.exe")) -ManagedDefaultRoot)

    # An install that died before the marker was written. The root is ours and holds nothing else.
    Check "partial install: rollback copy only" `
        (_IsStudioRoot (Make "partial-rollback" @("unsloth_studio.rollback.20260908120000.4242\pyvenv.cfg")) -ManagedDefaultRoot)
    Check "partial install: invalid legacy venv only" `
        (_IsStudioRoot (Make "partial-invalid" @(".venv.invalid.20260908120000.4242\pyvenv.cfg")) -ManagedDefaultRoot)
    Check "partial install: rollback with a numeric suffix" `
        (_IsStudioRoot (Make "partial-suffix" @("unsloth_studio.rollback.20260908120000.4242.2\pyvenv.cfg")) -ManagedDefaultRoot)
    # install.ps1 claims the root before the uv cache, so the cache is not a sentinel.
    $uvOnly = Make "uvcache-only" @("notes.md")
    New-Item -ItemType Directory -Path (Join-Path $uvOnly "cache\uv") -Force | Out-Null
    Check "a uv cache with no root marker is refused" (-not (_IsStudioRoot $uvOnly -ManagedDefaultRoot))
    Check "the same at a custom root is refused" (-not (_IsStudioRoot $uvOnly))

    # A marker is proof wherever the root sits; it is the only thing a custom root can offer.
    Check "a custom root with the venv owner marker" `
        (_IsStudioRoot (Make "custom-marker" @("unsloth_studio\.unsloth-studio-owned")))
    Check "a custom root with share\studio.conf" `
        (_IsStudioRoot (Make "custom-conf" @("share\studio.conf")))

    Check "a hand-made directory is refused" `
        (-not (_IsStudioRoot (Make "scratch" @("notes.md")) -ManagedDefaultRoot))
    Check "an ordinary project venv is refused" `
        (-not (_IsStudioRoot (Make "plain-venv" @(".venv\Scripts\python.exe", "notes.md")) -ManagedDefaultRoot))
    Check "a venv merely NAMED unsloth_studio is refused" `
        (-not (_IsStudioRoot (Make "named-venv" @("unsloth_studio\Scripts\python.exe")) -ManagedDefaultRoot))
    Check "somebody else's bin\unsloth.cmd is refused" `
        (-not (_IsStudioRoot (Make "foreign-cmd" @("bin\python.exe")) -ManagedDefaultRoot))
    # Why the flag exists: pip writes Scripts\unsloth.exe into ANY venv with the wheel.
    Check "a project venv with unsloth pip-installed is refused as a custom root" `
        (-not (_IsStudioRoot (Make "custom-project" @(".venv\Scripts\python.exe", ".venv\Scripts\unsloth.exe", "pyproject.toml"))))
    Check "that project's site-packages is refused as a custom root too" `
        (-not (_IsStudioRoot (Make "custom-pkg" @(".venv\Scripts\python.exe", ".venv\Lib\site-packages\unsloth_cli\__init__.py"))))
    Check "a partial-install leftover is refused at a custom root" `
        (-not (_IsStudioRoot (Make "custom-partial" @(".venv.invalid.20260908120000.4242\pyvenv.cfg"))))
    # The name alone is not proof: only a renamed venv carries the shape.
    Check "a FILE named like a leftover is refused" `
        (-not (_IsStudioRoot (Make "leftover-file" @(".venv.invalid.20260908120000.4242")) -ManagedDefaultRoot))
    $emptyLeftover = Make "leftover-empty" @()
    New-Item -ItemType Directory -Path (Join-Path $emptyLeftover "unsloth_studio.rollback.20260908120000.4242") -Force | Out-Null
    Check "an empty directory named like a leftover is refused" `
        (-not (_IsStudioRoot $emptyLeftover -ManagedDefaultRoot))
    Check "a leftover-named directory holding the user's own files is refused" `
        (-not (_IsStudioRoot (Make "leftover-user" @(".venv.invalid.20260908120000.4242\notes.md")) -ManagedDefaultRoot))
    # Scripts\unsloth.exe and site-packages are only pip's work when they are inside a venv.
    Check "a console script with no venv around it is refused" `
        (-not (_IsStudioRoot (Make "loose-exe" @(".venv\Scripts\unsloth.exe")) -ManagedDefaultRoot))
    Check "a site-packages tree with no interpreter is refused" `
        (-not (_IsStudioRoot (Make "loose-pkg" @("unsloth_studio\Lib\site-packages\unsloth\__init__.py")) -ManagedDefaultRoot))
    # install.ps1 keeps any rollback outside <stamp>.<pid>[.<n>] as user data.
    Check "a rollback-named venv outside the installer's format is refused" `
        (-not (_IsStudioRoot (Make "leftover-freeform" @("unsloth_studio.rollback.user-data\pyvenv.cfg")) -ManagedDefaultRoot))
    Check "an invalid-venv name outside the installer's format is refused" `
        (-not (_IsStudioRoot (Make "leftover-backup" @(".venv.invalid.backup\pyvenv.cfg")) -ManagedDefaultRoot))
    # Only the rollback name has a collision counter; .venv.invalid is written once per run.
    Check "an invalid-venv name with a rollback-style suffix is refused" `
        (-not (_IsStudioRoot (Make "leftover-suffixed-invalid" @(".venv.invalid.20260908120000.4242.2\pyvenv.cfg")) -ManagedDefaultRoot))
    Check "a rollback name with a non-numeric pid is refused" `
        (-not (_IsStudioRoot (Make "leftover-badpid" @("unsloth_studio.rollback.20260908120000.mine\pyvenv.cfg")) -ManagedDefaultRoot))
    # \d matches every Unicode decimal digit; install.ps1 only ever writes ASCII.
    $arabicDigits = -join (0..13 | ForEach-Object { [char](0x0660 + ($_ % 10)) })
    Check "a leftover stamp in non-ASCII digits is refused" `
        (-not (_IsStudioRoot (Make "leftover-unicode" @(".venv.invalid.$arabicDigits.4242\pyvenv.cfg")) -ManagedDefaultRoot))
    # "time" is install.sh's date(1) fallback; install.ps1 always formats yyyyMMddHHmmss.
    Check "install.sh's time fallback is not a Windows name" `
        (-not (_IsStudioRoot (Make "leftover-time" @(".venv.invalid.time.4242\pyvenv.cfg")) -ManagedDefaultRoot))
    # A reparse point with the right name points at a venv the installer did not put there.
    $linkRoot = Make "leftover-link" @("keepme.txt")
    $realVenv = Make "somebodys-venv" @("pyvenv.cfg")
    $made = $null
    foreach ($kind in @("Junction", "SymbolicLink")) {
        try {
            $made = New-Item -ItemType $kind -Path (Join-Path $linkRoot "unsloth_studio.rollback.20260908120000.4242") -Target $realVenv -ErrorAction Stop
            if ($made -and -not [string]::IsNullOrWhiteSpace(@($made.Target)[0])) { break }
        } catch { $made = $null }
    }
    if ($made -and -not [string]::IsNullOrWhiteSpace(@($made.Target)[0])) {
        Check "a linked leftover is refused" (-not (_IsStudioRoot $linkRoot -ManagedDefaultRoot))
    } else {
        Write-Host "  SKIP  no reparse point could be created here"
    }
    $foreign = Make "foreign-shim" @()
    New-Item -ItemType Directory -Path (Join-Path $foreign "bin") -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $foreign "bin\unsloth.cmd") -Value "@echo off`r`npython -m mytool %*`r`n"
    Check "a bin\unsloth.cmd without the trampoline is refused" (-not (_IsStudioRoot $foreign -ManagedDefaultRoot))
    Check "a missing path is refused" (-not (_IsStudioRoot (Join-Path $tmp "does-not-exist") -ManagedDefaultRoot))
    Check "an empty path is refused" (-not (_IsStudioRoot "" -ManagedDefaultRoot))

    # The other half of the question: install.ps1 decides when to WRITE the marker this gate
    # reads, and claiming a workspace it is about to refuse hands it to the uninstaller.
    $installPs1 = [System.IO.Path]::Combine($repoRoot, "install.ps1")
    $ie = $null; $it = $null
    $iast = [System.Management.Automation.Language.Parser]::ParseFile($installPs1, [ref]$it, [ref]$ie)
    $ifns = $iast.FindAll({
            param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst]
        }, $true)
    # Write-StudioRootOwnerMarker calls Test-StudioPlainFile, or every claim below is vacuous.
    foreach ($name in @("Test-StudioPlainFile", "Write-StudioRootOwnerMarker")) {
        $fn = $ifns | Where-Object { $_.Name -eq $name } | Select-Object -First 1
        if (-not $fn) {
            Write-Host "  FAIL  $name not found in install.ps1" -ForegroundColor Red
            $script:failures++
        } else {
            . ([scriptblock]::Create($fn.Extent.Text))
        }
    }
    function ClaimCheck([string]$Name, [bool]$WantClaimed, [string]$Mode, [string[]]$Files) {
        $root = Join-Path $tmp ("claim-" + $Name)
        New-Item -ItemType Directory -Path $root -Force | Out-Null
        foreach ($f in $Files) {
            $p = Join-Path $root $f
            New-Item -ItemType Directory -Path ([System.IO.Path]::GetDirectoryName($p)) -Force | Out-Null
            Set-Content -LiteralPath $p -Value "x"
        }
        $StudioRedirectMode = $Mode
        Write-StudioRootOwnerMarker -Root $root
        $claimed = Test-Path -LiteralPath (Join-Path $root ".unsloth-studio-owned") -PathType Leaf
        Check $Name ($claimed -eq $WantClaimed)
    }
    ClaimCheck "the default root, always" $true "default" @("notes.md")
    ClaimCheck "an empty custom root" $true "env" @()
    ClaimCheck "a custom root already carrying our marker" $true "env" @("unsloth_studio\.unsloth-studio-owned")
    ClaimCheck "a custom root with share\studio.conf" $true "env" @("share\studio.conf")
    ClaimCheck "somebody's workspace" $false "env" @("pyproject.toml")
    ClaimCheck "somebody's workspace with a venv of their own" $false "env" @("unsloth_studio\pyvenv.cfg")
    # A file called bin\unsloth.exe is any file of that name, and this list authorizes a delete.
    ClaimCheck "a workspace holding a plain bin\unsloth.exe" $false "env" @("bin\unsloth.exe", "notes.txt")
    # A LINKED share or unsloth_studio holding a genuine marker: the attribute answers for the
    # named file only.
    foreach ($pair in @(@("share", "studio.conf"), @("unsloth_studio", ".unsloth-studio-owned"))) {
        $wroot = Join-Path $tmp ("claim-linked-" + $pair[0])
        $wreal = Join-Path $tmp ("claim-linked-" + $pair[0] + "-real")
        New-Item -ItemType Directory -Path $wroot -Force | Out-Null
        New-Item -ItemType Directory -Path $wreal -Force | Out-Null
        Set-Content -LiteralPath (Join-Path $wreal $pair[1]) -Value "x"
        Set-Content -LiteralPath (Join-Path $wroot "notes.txt") -Value "mine"
        $dl = $null
        foreach ($kind in @("Junction", "SymbolicLink")) {
            try {
                $dl = New-Item -ItemType $kind -Path (Join-Path $wroot $pair[0]) -Target $wreal -ErrorAction Stop
                if ($dl -and -not [string]::IsNullOrWhiteSpace(@($dl.Target)[0])) { break }
            } catch { $dl = $null }
        }
        if ($dl -and -not [string]::IsNullOrWhiteSpace(@($dl.Target)[0])) {
            $StudioRedirectMode = "env"
            Write-StudioRootOwnerMarker -Root $wroot
            Check "a linked $($pair[0]) is not ownership proof" `
                (-not (Test-Path -LiteralPath (Join-Path $wroot ".unsloth-studio-owned") -PathType Leaf))
        }
    }
    # A link named like our marker: Test-Path follows it, which would skip the emptiness test.
    $lm = Join-Path $tmp "claim-linked-marker"
    New-Item -ItemType Directory -Path $lm -Force | Out-Null
    Set-Content -LiteralPath (Join-Path $lm "notes.txt") -Value "mine"
    $lmTarget = Join-Path $tmp "claim-linked-marker-target.txt"
    Set-Content -LiteralPath $lmTarget -Value "x"
    $lmLink = $null
    try {
        $lmLink = New-Item -ItemType SymbolicLink -Path (Join-Path $lm ".unsloth-studio-owned") -Target $lmTarget -ErrorAction Stop
    } catch { $lmLink = $null }
    if ($lmLink) {
        $StudioRedirectMode = "env"
        Write-StudioRootOwnerMarker -Root $lm
        $still = Get-Item -LiteralPath (Join-Path $lm ".unsloth-studio-owned") -Force
        Check "a linked marker is not read as proof, and is left as it was" `
            ((($still.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0))
    } else {
        Write-Host "  SKIP  no symlink could be created here"
    }
    # Called twice per install: the second call must not delete what the first one wrote.
    $twice = Join-Path $tmp "claim-twice"
    New-Item -ItemType Directory -Path $twice -Force | Out-Null
    $StudioRedirectMode = "default"
    Write-StudioRootOwnerMarker -Root $twice
    Set-Content -LiteralPath (Join-Path $twice ".unsloth-studio-owned") -Value "first" -NoNewline
    Write-StudioRootOwnerMarker -Root $twice
    Check "a second claim leaves a valid marker alone" `
        ((Get-Content -LiteralPath (Join-Path $twice ".unsloth-studio-owned") -Raw) -eq "first")
    # Test-DirectoryHasEntries is defined below the first call site, so the test must be inline.
    Check "the claim does not depend on Test-DirectoryHasEntries" `
        (-not ((($ifns | Where-Object { $_.Name -eq "Write-StudioRootOwnerMarker" } | Select-Object -First 1).Extent.Text) -match "Test-DirectoryHasEntries"))

    # A link at the marker path: WriteAllText would follow it and truncate the target.
    $linkClaim = Join-Path $tmp "claim-link"
    New-Item -ItemType Directory -Path $linkClaim -Force | Out-Null
    $target = Join-Path $tmp "precious.txt"
    Set-Content -LiteralPath $target -Value "precious"
    $linked = $null
    try {
        $linked = New-Item -ItemType SymbolicLink -Path (Join-Path $linkClaim ".unsloth-studio-owned") -Target $target -ErrorAction Stop
    } catch { $linked = $null }
    if ($linked) {
        $StudioRedirectMode = "default"
        Write-StudioRootOwnerMarker -Root $linkClaim
        Check "a linked marker path does not truncate its target" ((Get-Content -LiteralPath $target -Raw).Trim() -eq "precious")
    } else {
        Write-Host "  SKIP  no symlink could be created here"
    }

    # Same for install.ps1's replacement guard, and structural for the same reason.
    $installText = Get-Content -LiteralPath $installPs1 -Raw
    $guard = [regex]::Match($installText, "(?s)why: matching guard to the \.venv branch below.*?Move it aside or choose an empty").Value
    Check "install.ps1 has an env-mode replacement guard" ($guard.Length -gt 0)
    Check "and it reads the root marker" `
        ($guard -match [regex]::Escape('Join-Path $StudioHome ".unsloth-studio-owned"'))
    # ... the same way the claim writes it, or a link gets a foreign workspace past the guard.
    Check "and through Test-StudioPlainFile, not Test-Path" `
        ($guard -match [regex]::Escape('Test-StudioPlainFile -Path (Join-Path $StudioHome ".unsloth-studio-owned")'))
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($failures -gt 0) { Write-Host "$failures check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "all checks passed"
