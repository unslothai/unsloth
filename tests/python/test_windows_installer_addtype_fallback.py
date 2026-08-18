# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The installer must survive a host where Add-Type cannot compile.

install.ps1 resolves path identity through a C# helper it compiles at runtime.
Windows PowerShell 5.1 -- the interpreter studio/src-tauri/src/install.rs spawns
-- compiles by writing the source into %TEMP% and running csc.exe, so a %TEMP%
that cannot hold a file (or a scanner that eats what was just written there)
makes Add-Type throw CS2001. That exception used to travel up Get-StudioPathHash
and end a first launch as "Could not create the Studio install lock" (#9140),
with nothing about the message pointing at the compiler.

These run under pwsh on any platform: the failure being guarded is that a thrown
Add-Type aborts the install, and that is shell-independent. Windows PowerShell
5.1's CodeDom behaviour itself is exercised by the Windows-only tests in
test_windows_installer_concurrency_guard.py.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")

# Every Add-Type call in install.ps1 that compiles C# rather than loading an
# assembly, so a stub can stand in for the compiler.
SABOTAGE = (
    """function Add-Type { throw "(0) : error CS2001: Source file 'a.0.cs' could not be found" }"""
)


def _extract(pattern: str, source: str) -> str:
    match = re.search(pattern, source, flags = re.DOTALL)
    assert match is not None, f"install.ps1 block not found: {pattern}"
    return match.group(0)


def _helpers(*names: str) -> str:
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    return "\n".join(_extract(rf"    function {name} \{{.*?\n    \}}\n", source) for name in names)


# The whole chain the reported failure walked, plus the fallback it now lands on.
LOCK_CHAIN = (
    "Write-StudioLine",
    "Test-StudioDirectoryUsable",
    "Remove-StudioStalePrivateTempDirectories",
    "New-StudioPrivateTempDirectory",
    "Initialize-StudioTempEnvironment",
    "Restore-StudioTempEnvironment",
    "Write-StudioFinalPathDegraded",
    "Initialize-StudioFinalPathNativeType",
    "Resolve-StudioLinkTarget",
    "Get-StudioLexicalPath",
    "Resolve-StudioFinalPathInfo",
    "Get-StudioFinalPath",
    "Get-StudioPathHash",
    "Get-StudioInstallMutexName",
    "Test-StudioPathEqual",
    "Enter-StudioNamedMutex",
    "Enter-StudioInstallMutex",
    "Exit-StudioInstallMutex",
)


def _run_powershell(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 120,
        env = env if env is not None else os.environ.copy(),
    )


def _script(
    body: str,
    *,
    sabotage: bool = True,
    names: tuple[str, ...] = LOCK_CHAIN,
) -> str:
    return "\n".join(
        [
            '$ErrorActionPreference = "Stop"',
            # Write-StudioLine picks its sink from this; unset it reaches Write-Host,
            # which -NonInteractive still writes but the caller cannot capture.
            "$script:StudioStdoutRedirected = $true",
            _helpers(*names),
            SABOTAGE if sabotage else "",
            body,
        ]
    )


def _lines(result: subprocess.CompletedProcess, prefix: str) -> list[str]:
    return [line for line in result.stdout.splitlines() if line.startswith(prefix)]


@requires_pwsh
def test_install_lock_is_acquired_when_the_compiler_fails(tmp_path: Path):
    """The bug, exactly: a throwing Add-Type must not cost the install its lock."""
    studio_home = tmp_path / "studio"
    studio_home.mkdir()
    result = _run_powershell(
        _script(
            f"""
$name = Get-StudioInstallMutexName -Path '{studio_home}'
Write-Output "NAME:$name"
$mutex = Enter-StudioInstallMutex -Path '{studio_home}'
Write-Output "LOCK:$($null -ne $mutex)"
Exit-StudioInstallMutex -Mutex $mutex
"""
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "LOCK:") == ["LOCK:True"]
    assert re.fullmatch(
        r"NAME:Global\\UnslothStudioInstall-[0-9a-f]{64}", _lines(result, "NAME:")[0]
    )
    # And it says why it degraded, once, without dumping the C# it tried to build.
    warnings = [line for line in result.stdout.splitlines() if "native path resolver" in line]
    assert len(warnings) == 1
    assert "using System" not in result.stdout


@requires_pwsh
def test_the_compiler_is_not_retried_for_every_path(tmp_path: Path):
    result = _run_powershell(
        _script(
            f"""
$global:AddTypeCalls = 0
function Add-Type {{
    $global:AddTypeCalls++
    throw "(0) : error CS2001: Source file 'a.0.cs' could not be found"
}}
foreach ($i in 1..5) {{ Get-StudioFinalPath -Path '{tmp_path}' | Out-Null }}
Write-Output "CALLS:$global:AddTypeCalls"
""",
            sabotage = False,
        )
    )
    assert result.returncode == 0, result.stderr
    # One attempt, then one retry with a private %TEMP%. Never again after that:
    # the scan below resolves a path per running process.
    assert _lines(result, "CALLS:") == ["CALLS:2"]


@requires_pwsh
def test_the_private_temp_retry_recovers_the_native_helper(tmp_path: Path):
    """Surviving a dead compiler is the floor. The retry has to actually work.

    Add-Type is replaced by a stub that behaves the way 5.1's CodeDom does: it
    writes the source into %TEMP% and fails when that cannot hold a file, and it
    calls the real cmdlet once it can, so the type genuinely gets defined.
    """
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding = "utf-8")
    dead = str(blocker / "temp")
    local_app_data = tmp_path / "localappdata"
    local_app_data.mkdir()
    studio_home = tmp_path / "studio"
    studio_home.mkdir()

    env = os.environ.copy()
    env.update({"TMP": dead, "TEMP": dead, "LOCALAPPDATA": str(local_app_data)})
    env.pop("USERPROFILE", None)

    result = _run_powershell(
        _script(
            f"""
$script:RealAddType = Get-Command Add-Type -CommandType Cmdlet
$global:Attempts = 0
function Add-Type {{
    param([string]$TypeDefinition, [string]$ErrorAction)
    $global:Attempts++
    $probe = Join-Path $env:TMP ("codedom-" + [guid]::NewGuid().ToString('N').Substring(0, 8) + ".0.cs")
    try {{
        [System.IO.File]::WriteAllText($probe, $TypeDefinition)
        Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
    }} catch {{
        throw "(0) : error CS2001: Source file '$probe' could not be found"
    }}
    & $script:RealAddType -TypeDefinition $TypeDefinition -ErrorAction Stop
}}
# Skip the session-wide temp fix, so the RETRY is the only thing that can save it.
$script:StudioTempChecked = $true
$info = Resolve-StudioFinalPathInfo -Path '{studio_home}'
Write-Output "ATTEMPTS:$global:Attempts"
Write-Output "LOADED:$([bool]("UnslothStudioFinalPathV2" -as [type]))"
Write-Output "PATH:$($info.Path)"
Write-Output "TMP:$env:TMP"
Write-Output "TEMP:$env:TEMP"
""",
            sabotage = False,
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    # One failed attempt, then one retry with a private %TEMP%, which succeeds.
    assert _lines(result, "ATTEMPTS:") == ["ATTEMPTS:2"]
    assert _lines(result, "LOADED:") == ["LOADED:True"]
    assert not [line for line in result.stdout.splitlines() if "native path resolver" in line]
    # Restored exactly, broken values and all: they are the caller's, not ours.
    assert _lines(result, "TMP:") == [f"TMP:{dead}"]
    assert _lines(result, "TEMP:") == [f"TEMP:{dead}"]
    assert list((local_app_data / "Unsloth Studio" / "temp").glob("ust-*")) == []
    # The helper's P/Invoke targets kernel32, so off Windows the type loads but the
    # CALL fails. That is a path Windows will not take, and it has to degrade to a
    # usable answer rather than throw.
    assert _lines(result, "PATH:")[0].startswith("PATH:")
    assert _lines(result, "PATH:") != ["PATH:"]


@requires_pwsh
def test_resolver_follows_a_linked_ancestor(tmp_path: Path):
    physical = tmp_path / "physical" / "studio"
    physical.mkdir(parents = True)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(tmp_path / "physical", target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("creating a symlink requires privilege on this host")

    result = _run_powershell(
        _script(
            f"""
Write-Output "PATH:$(Get-StudioFinalPath -Path '{alias / "studio"}')"
Write-Output "EQUAL:$(Test-StudioPathEqual -Left '{alias / "studio"}' -Right '{physical}')"
Write-Output "MISSING:$(Get-StudioFinalPath -Path '{alias / "studio" / "not" / "there"}')"
"""
        )
    )
    assert result.returncode == 0, result.stderr
    # A link on a PARENT component is the ordinary Windows shape, and it is the
    # one GetFullPath alone gets wrong.
    assert _lines(result, "PATH:") == [f"PATH:{physical}"]
    assert _lines(result, "EQUAL:") == ["EQUAL:True"]
    # Segments that do not exist yet are reattached, as the native resolver does.
    assert _lines(result, "MISSING:") == [f"MISSING:{physical / 'not' / 'there'}"]


@requires_pwsh
def test_resolver_terminates_on_a_link_loop(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    try:
        left.symlink_to(right, target_is_directory = True)
        right.symlink_to(left, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("creating a symlink requires privilege on this host")

    result = _run_powershell(
        _script(
            f"""
Write-Output "PATH:$(Get-StudioFinalPath -Path '{left}')"
"""
        )
    )
    assert result.returncode == 0, result.stderr
    assert len(_lines(result, "PATH:")) == 1


@requires_pwsh
def test_unequal_paths_are_unknown_rather_than_different_when_inexact(tmp_path: Path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    result = _run_powershell(
        _script(
            f"""
$answer = Test-StudioPathEqual -Left '{left}' -Right '{right}'
Write-Output "NULL:$($null -eq $answer)"
"""
        )
    )
    assert result.returncode == 0, result.stderr
    # Without exact resolution two different spellings may still be one directory,
    # and $null is what makes the caller take BOTH runtime locks instead of
    # guessing. Returning $false here would silently drop one of them.
    assert _lines(result, "NULL:") == ["NULL:True"]


@requires_pwsh
def test_unusable_temp_is_replaced_and_then_restored(tmp_path: Path):
    # Under a regular file, so it cannot merely be created: the probe has to fail
    # the way a system-scoped or ACL-restricted temp directory fails.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    env = os.environ.copy()
    env["TMP"] = str(blocker / "temp")
    env.pop("TEMP", None)
    result = _run_powershell(
        _script(
            """
Initialize-StudioTempEnvironment
Write-Output "USABLE:$(Test-StudioDirectoryUsable -Path $env:TMP)"
Write-Output "MATCHED:$($env:TMP -eq $env:TEMP)"
$replacement = $env:TMP
Restore-StudioTempEnvironment
Write-Output "TMP:$env:TMP"
Write-Output "TEMPSET:$($null -ne $env:TEMP)"
Write-Output "KEPT:$(Test-Path -LiteralPath $replacement)"
""",
            names = (
                "Write-StudioLine",
                "Test-StudioDirectoryUsable",
                "Remove-StudioStalePrivateTempDirectories",
                "New-StudioPrivateTempDirectory",
                "Initialize-StudioTempEnvironment",
                "Restore-StudioTempEnvironment",
            ),
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "USABLE:") == ["USABLE:True"]
    # Both, because Windows reads TMP before TEMP.
    assert _lines(result, "MATCHED:") == ["MATCHED:True"]
    assert _lines(result, "TMP:") == [f"TMP:{env['TMP']}"]
    # TEMP was absent to begin with; restoring it as "" would change how every
    # later child resolves its own temp directory.
    assert _lines(result, "TEMPSET:") == ["TEMPSET:False"]
    # The replacement directory survives on purpose: a Studio autostarted by this
    # install inherited it as its own %TEMP%, and the host's real one is broken.
    # Stale ones are swept on the next run instead.
    assert _lines(result, "KEPT:") == ["KEPT:True"]


_NT_DEVICE_PREFIX = "\\??\\"
_LIST = "$l=[System.Collections.Generic.List[string]]::new(); {0}; $l"

# What (Get-Item).Target actually hands back. Windows PowerShell 5.1 returns a
# COLLECTION, not a string, and not an [array] either -- so a container test that
# names one type lets the real one fall through and be space-joined into a path
# that does not exist. Junctions store the NT device form. System junctions and
# Store AppExecLinks report nothing at all. None of these shapes can be produced
# on Linux, so Get-Item is stubbed and the paths are POSIX.
_TARGET_SHAPES = [
    ("generic collection", _LIST.format('$l.Add("/real/target")'), "/real/target"),
    (
        "generic collection, several entries",
        _LIST.format('$l.Add("/first"); $l.Add("/second")'),
        "/first",
    ),
    ("array", '@("/real/target")', "/real/target"),
    ("bare string", '"/real/target"', "/real/target"),
    ("nt device prefix", f'"{_NT_DEVICE_PREFIX}/real/target"', "/real/target"),
    (
        "nt device prefix in a collection",
        _LIST.format(f'$l.Add("{_NT_DEVICE_PREFIX}/real/target")'),
        "/real/target",
    ),
    ("relative to the link", '"sibling"', "/some/sibling"),
    ("pointing at itself", '"/some/link"', ""),
    ("null (system junction, AppExecLink)", "$null", ""),
    ("empty", '""', ""),
    ("whitespace", '"   "', ""),
    ("throws (access denied)", '$(throw "access denied")', ""),
]


@requires_pwsh
@pytest.mark.parametrize(
    "shape,expression,expected",
    _TARGET_SHAPES,
    ids = [shape.replace(" ", "-") for shape, _, _ in _TARGET_SHAPES],
)
def test_link_targets_of_every_windows_powershell_5_1_shape(
    shape: str, expression: str, expected: str
):
    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                _helpers("Resolve-StudioLinkTarget"),
                "function Get-Item {",
                "    param([string]$LiteralPath, [switch]$Force, [string]$ErrorAction)",
                "    $target = $null",
                f"    try {{ $target = {expression} }} catch {{ $target = $null }}",
                "    $item = New-Object psobject",
                "    $item | Add-Member -MemberType NoteProperty -Name Target -Value $target",
                "    Write-Output $item",
                "}",
                "Write-Output \"TARGET:[$(Resolve-StudioLinkTarget -Path '/some/link')]\"",
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "TARGET:") == [f"TARGET:[{expected}]"]


@requires_pwsh
def test_the_stale_sweep_never_deletes_through_a_link(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keepme.txt").write_text("do not delete", encoding = "utf-8")
    stale = root / "ust-1-old"
    stale.mkdir()
    (stale / "junk").write_text("x", encoding = "utf-8")
    fresh = root / "ust-2-new"
    fresh.mkdir()
    link = root / "ust-3-link"
    try:
        link.symlink_to(precious, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("creating a symlink requires privilege on this host")

    aged = time.time() - 3 * 24 * 3600
    os.utime(stale, (aged, aged))
    os.utime(link, (aged, aged), follow_symlinks = False)

    result = _run_powershell(
        _script(
            f"Remove-StudioStalePrivateTempDirectories -Root '{root}'",
            sabotage = False,
            names = ("Remove-StudioStalePrivateTempDirectories",),
        )
    )
    assert result.returncode == 0, result.stderr
    assert not stale.exists()
    assert fresh.exists()
    assert not link.is_symlink()
    # Windows PowerShell 5.1's Remove-Item -Recurse follows a junction and empties
    # what it points at. Nothing the installer creates under this root is a
    # reparse point, so one appearing there must cost only the link.
    assert (precious / "keepme.txt").exists()


@requires_pwsh
def test_a_failure_before_the_lock_still_restores_temp(tmp_path: Path):
    """Most install failures return long before the finally that frees the locks."""
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding = "utf-8")
    env = os.environ.copy()
    env["TMP"] = str(blocker / "temp")
    env["TEMP"] = str(blocker / "temp")

    result = _run_powershell(
        _script(
            """
$TauriMode = $false
function Write-TauriLog { param([string]$Tag, [string]$Message) }
Initialize-StudioTempEnvironment
Write-Output "REDIRECTED:$($env:TMP -ne $env:ORIGINAL_TMP)"
try { Exit-InstallFailure -Message "preflight failed" } catch {}
Write-Output "TMP:$env:TMP"
Write-Output "TEMP:$env:TEMP"
""",
            names = (
                "Write-StudioLine",
                "Exit-InstallFailure",
                "Test-StudioDirectoryUsable",
                "Remove-StudioStalePrivateTempDirectories",
                "New-StudioPrivateTempDirectory",
                "Initialize-StudioTempEnvironment",
                "Restore-StudioTempEnvironment",
            ),
        ),
        env = {**env, "ORIGINAL_TMP": str(blocker / "temp")},
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "REDIRECTED:") == ["REDIRECTED:True"]
    assert _lines(result, "TMP:") == [f"TMP:{env['TMP']}"]
    assert _lines(result, "TEMP:") == [f"TEMP:{env['TEMP']}"]


def test_the_private_temp_directory_is_somewhere_uninstall_reclaims():
    """It outlives the install, so it has to sit where the uninstaller looks."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")
    roots = _extract(r"    function New-StudioPrivateTempDirectory \{.*?\n    \}\n", source)

    # LOCALAPPDATA\"Unsloth Studio" is the data dir uninstall.ps1 removes wholesale.
    assert 'Join-Path $env:LOCALAPPDATA "Unsloth Studio\\temp"' in roots
    assert '"Unsloth Studio"' in uninstall
    # ~\.unsloth\.cache is on its explicit sibling list. Directly under ~\.unsloth
    # would be worse than litter: that directory is removed only when it is empty,
    # so anything left there stops the uninstaller clearing it at all.
    assert 'Join-Path $env:USERPROFILE ".unsloth\\.cache\\temp"' in roots
    assert (
        '$defaultCache = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".cache" }'
        in uninstall
    )
    assert ".unsloth\\temp" not in roots


def test_path_resolution_and_process_identity_no_longer_need_the_compiler():
    source = INSTALL_PS1.read_text(encoding = "utf-8")

    # The resolver callers reach is now a dispatcher; only the initializer builds.
    assert "Add-Type" not in _extract(r"    function Get-StudioFinalPath \{.*?\n    \}\n", source)
    assert "Add-Type" not in _extract(
        r"    function Resolve-StudioFinalPathInfo \{.*?\n    \}\n", source
    )
    assert "Add-Type" not in _extract(r"    function Get-StudioLexicalPath \{.*?\n    \}\n", source)

    # The process scan must keep its "confirmed executable image only" contract on
    # every rung of the fallback, or a degraded host would block on a name match.
    image = _extract(r"    function Get-StudioProcessImagePath \{.*?\n    \}\n", source)
    assert ".CommandLine" not in image
    assert "Win32_Process" in image
    assert "QueryFullProcessImageNameW" not in image

    # Asserted as source rather than behaviour on purpose: PowerShell 7 does not
    # follow a link on -Recurse, so the test above passes with or without this
    # guard. Windows PowerShell 5.1 -- the shell the desktop app actually spawns
    # -- does follow it, and there this is the difference between deleting a
    # stale scratch directory and emptying whatever it points at.
    sweep = _extract(
        r"    function Remove-StudioStalePrivateTempDirectories \{.*?\n    \}\n", source
    )
    assert "ReparsePoint" in sweep
    reparse_branch = sweep.index("ReparsePoint")
    assert "-Recurse" not in sweep[reparse_branch : sweep.index("continue", reparse_branch)]
