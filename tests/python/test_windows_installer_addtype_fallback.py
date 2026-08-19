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
import tempfile
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")

# Stands in for every Add-Type in install.ps1 that compiles C# rather than loading
# an assembly.
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
    "Get-StudioPrivateTempRoots",
    "New-StudioPrivateTempDirectory",
    "Initialize-StudioTempEnvironment",
    "Restore-StudioTempEnvironment",
    "Write-StudioFinalPathDegraded",
    "Initialize-StudioFinalPathNativeType",
    "Resolve-StudioLinkTarget",
    "Get-StudioSubstTarget",
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
    # Through a FILE, not -Command: these scripts carry the whole extracted helper
    # chain, and Windows caps a command line at 32767 characters. Passed inline,
    # the moment the chain grows past that every test here dies as WinError 206
    # rather than testing anything. utf-8-sig because Windows PowerShell 5.1 reads
    # a BOM-less .ps1 as ANSI; utf-8 with replacement on the way back because the
    # default console codepage there cannot decode what PowerShell writes.
    handle, name = tempfile.mkstemp(suffix = ".ps1")
    os.close(handle)
    try:
        Path(name).write_text(script, encoding = "utf-8-sig")
        return subprocess.run(
            ["pwsh", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", name],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 120,
            env = env if env is not None else os.environ.copy(),
        )
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def _script(
    body: str,
    *,
    sabotage: bool = True,
    names: tuple[str, ...] = LOCK_CHAIN,
) -> str:
    return "\n".join(
        [
            '$ErrorActionPreference = "Stop"',
            # Write-StudioLine picks its sink from this; unset, it reaches Write-Host,
            # which the caller cannot capture.
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
    # One attempt, one retry with a private %TEMP%, then cached: the scan below
    # resolves a path per running process.
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
    # The P/Invoke targets kernel32, so off Windows the type loads but the CALL
    # fails; it has to degrade to a usable answer rather than throw.
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
    # A link on a PARENT component is the ordinary Windows shape, and the one
    # GetFullPath alone gets wrong.
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
    # Without exact resolution two spellings may still be one directory; $null makes
    # the caller take BOTH runtime locks, where $false would silently drop one.
    assert _lines(result, "NULL:") == ["NULL:True"]


@requires_pwsh
def test_a_directory_that_will_not_give_a_file_back_is_not_usable(tmp_path: Path):
    """Create and read are not enough. csc.exe also has to clean up after itself.

    A directory that accepts a file and then refuses to delete it (a denied
    Delete ACE, a scanner sitting on the handle) is the shape that produced the
    reported failure, and it passes a create-and-read probe unchanged.
    """
    good = tmp_path / "good"
    good.mkdir()
    result = _run_powershell(
        _script(
            f"""
Write-Output "BEFORE:$(Test-StudioDirectoryUsable -Path '{good}')"
# Every delete is a no-op from here, which is what a denied Delete ACE looks
# like to the caller: no exception, the file simply stays.
function Remove-Item {{ param([string]$LiteralPath, [switch]$Force, [string]$ErrorAction) }}
Write-Output "AFTER:$(Test-StudioDirectoryUsable -Path '{good}')"
""",
            sabotage = False,
            names = ("Write-StudioLine", "Test-StudioDirectoryUsable"),
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "BEFORE:") == ["BEFORE:True"]
    assert _lines(result, "AFTER:") == ["AFTER:False"]
    # The healthy probe cleaned up after itself, so only the undeletable one is left.
    assert len(list(good.glob("unsloth-probe-*.tmp"))) == 1


@requires_pwsh
def test_unusable_temp_is_replaced_and_then_restored(tmp_path: Path):
    # Under a regular file so it cannot merely be created: the probe has to fail the
    # way an ACL-restricted temp directory fails.
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
                "Get-StudioPrivateTempRoots",
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
    # TEMP was absent; restoring it as "" would change how every later child
    # resolves its own temp directory.
    assert _lines(result, "TEMPSET:") == ["TEMPSET:False"]
    # It survives on purpose: an autostarted Studio inherited it as its own %TEMP%,
    # and the host's real one is broken. Stale ones are swept on the next run.
    assert _lines(result, "KEPT:") == ["KEPT:True"]


def _same_path(got: str, expected: str) -> bool:
    """Compare two resolved paths without pinning one platform's spelling.

    The resolver ends with [System.IO.Path]::GetFullPath, so on Windows the POSIX
    fixtures below come back rooted on the current drive and with backslashes:
    "/real/target" resolves to "D:\\real\\target". That is correct, and it has
    nothing to do with what these cases measure, which is whether the shape
    (Get-Item).Target arrived in was unwrapped to the right single target.
    """

    def norm(value: str) -> str:
        value = value.replace("\\", "/")
        if len(value) > 1 and value[1] == ":":
            value = value[2:]
        return value.rstrip("/").lower()

    return norm(got) == norm(expected)


_NT_DEVICE_PREFIX = "\\??\\"
_LIST = "$l=[System.Collections.Generic.List[string]]::new(); {0}; $l"

# What (Get-Item).Target actually hands back. 5.1 returns a COLLECTION, not a
# string and not an [array] either, so a container test naming one type lets the
# real one fall through and be space-joined. Junctions store the NT device form;
# system junctions and Store AppExecLinks report nothing. None of these shapes can
# be produced on Linux, so Get-Item is stubbed and the paths are POSIX-style; see
# _same_path for why the comparison cannot be a string equality.
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
    got = _lines(result, "TARGET:")
    assert len(got) == 1, got
    assert _same_path(got[0][len("TARGET:[") : -1], expected)


def _dead_pid() -> int:
    """A PID that is not running, so a ust-<pid>- directory reads as abandoned."""
    for candidate in range(4_000_000, 4_000_400):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except (OSError, PermissionError):
            continue
    return 4_000_000


_DEAD_PID = _dead_pid()


@requires_pwsh
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("\\??\\UNC\\server\\share\\dir", "\\\\server\\share\\dir"),
        ("\\??\\unc\\server\\share", "\\\\server\\share"),
        ("\\??\\C:\\real\\target", "C:\\real\\target"),
        (
            "\\??\\Volume{11111111-2222-3333-4444-555555555555}\\dir",
            "\\\\?\\Volume{11111111-2222-3333-4444-555555555555}\\dir",
        ),
    ],
    ids = [
        "unc-device-form",
        "lowercase-unc",
        "ordinary-device-form",
        "mounted-folder-volume-guid",
    ],
)
def test_the_nt_device_prefix_becomes_a_usable_path(raw: str, expected: str):
    """\\??\\UNC\\server\\share is the device spelling of \\\\server\\share.

    Dropping the first four characters leaves "UNC\\server\\share", which reads as
    a RELATIVE path and gets combined with the link's own parent, inventing an
    identity that matches nothing real -- and a wrong identity here is a wrong
    mutex. Whether the rewritten form counts as rooted is platform-dependent, so
    what is asserted is the rewrite; the Windows end of it is covered on the 5.1
    probe.
    """
    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                _helpers("Resolve-StudioLinkTarget"),
                "function Get-Item {",
                "    param([string]$LiteralPath, [switch]$Force, [string]$ErrorAction)",
                "    $item = New-Object psobject",
                f'    $item | Add-Member -MemberType NoteProperty -Name Target -Value "{raw}"',
                "    Write-Output $item",
                "}",
                "Write-Output \"TARGET:[$(Resolve-StudioLinkTarget -Path '/some/link')]\"",
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    got = _lines(result, "TARGET:")[0]
    assert expected in got
    assert "UNC\\" not in got and "unc\\" not in got


@requires_pwsh
def test_a_drive_less_rooted_target_lands_on_the_link_own_root(tmp_path: Path):
    """A symlink may store "\\real", which is rooted but names no drive.

    IsPathRooted says true, so anchoring against the link is skipped and
    GetFullPath then resolves the target against whatever drive the PROCESS
    happens to be sitting on. Windows resolves it on the LINK's volume, so a
    link on D: must never normalize to C:\\real: that is a different directory,
    and it would be the one the fallback mutex and the in-use scan protect.
    """
    link = tmp_path / "nested" / "link"
    link.parent.mkdir(parents = True)
    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                _helpers("Resolve-StudioLinkTarget"),
                "function Get-Item {",
                "    param([string]$LiteralPath, [switch]$Force, [string]$ErrorAction)",
                "    $item = New-Object psobject",
                '    $item | Add-Member -MemberType NoteProperty -Name Target -Value "\\real"',
                "    Write-Output $item",
                "}",
                f"$root = [System.IO.Path]::GetPathRoot('{link}')",
                f"Write-Output \"TARGET:[$(Resolve-StudioLinkTarget -Path '{link}')]\"",
                'Write-Output "ROOT:[$root]"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    got = _lines(result, "TARGET:")[0].split("[", 1)[1].rstrip("]")
    root = _lines(result, "ROOT:")[0].split("[", 1)[1].rstrip("]")
    assert got.startswith(root), f"{got} is not on the link's own root {root}"
    assert got.rstrip("\\/").endswith("real")
    # The whole point: it is NOT reparented under the link's own directory either.
    assert "nested" not in got


@requires_pwsh
@pytest.mark.parametrize(
    "tmp,expect_override",
    [
        ("   ", True),
        ("\t", True),
        ("", False),
        (None, False),
    ],
    ids = ["whitespace-tmp", "tab-tmp", "empty-tmp", "absent-tmp"],
)
def test_a_whitespace_only_tmp_is_set_as_far_as_windows_is_concerned(
    tmp_path: Path, tmp: str | None, expect_override: bool
):
    """GetTempPath takes the first of TMP/TEMP that is merely NON-empty.

    So a whitespace-only TMP is the one Windows and every child process will use.
    Reading it as "unset" probes a healthy TEMP, changes nothing, and leaves the
    compile and every later download pointed at a path that cannot exist.
    """
    good = tmp_path / "good"
    good.mkdir()
    local_app_data = tmp_path / "localappdata"
    local_app_data.mkdir()

    env = os.environ.copy()
    env.pop("TMP", None)
    if tmp is not None:
        env["TMP"] = tmp
    env["TEMP"] = str(good)
    env["LOCALAPPDATA"] = str(local_app_data)
    env["USERPROFILE"] = str(tmp_path / "profile")

    result = _run_powershell(
        _script(
            """
Initialize-StudioTempEnvironment
Write-Output "TMP:[$env:TMP]"
""",
            sabotage = False,
            names = (
                "Write-StudioLine",
                "Test-StudioDirectoryUsable",
                "Remove-StudioStalePrivateTempDirectories",
                "Set-StudioPrivateTempOwner",
                "Get-StudioPrivateTempRoots",
                "New-StudioPrivateTempDirectory",
                "Initialize-StudioTempEnvironment",
                "Restore-StudioTempEnvironment",
            ),
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    got = _lines(result, "TMP:")[0][len("TMP:[") : -1]
    if expect_override:
        assert got not in ("", tmp)
        assert "ust-" in got
    else:
        assert "ust-" not in got


@requires_pwsh
def test_final_normalization_keeps_a_volume_guid_rooted():
    """\\\\?\\C:\\x still names a drive once the prefix comes off. A volume GUID does not.

    Stripping it leaves the unrooted "Volume{GUID}\\x", which hashes to a
    different identity than the same directory reached by drive letter and leaves
    GetPathRoot empty, so the relaxed process comparison cannot run either.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    body = _extract(r"    function Resolve-StudioFinalPathInfo \{.*?\n    \}\n", source)
    guid = body.index("\\\\?\\Volume{")
    dos = body.index("$resolved.Substring(4)")
    # The volume-GUID branch has to come BEFORE the general one, which would
    # otherwise swallow it and strip the prefix anyway.
    assert guid < dos
    branch = body[guid:dos]
    assert "Substring" not in branch


@requires_pwsh
@pytest.mark.parametrize(
    "path,expected",
    [
        ("X:\\", "D:\\real\\dir"),
        ("X:", "D:\\real\\dir"),
        ("x:\\venv", "D:\\real\\dir"),
        ("C:\\other", ""),
    ],
    ids = ["drive-root", "bare-drive", "lowercase-letter", "unmapped-drive"],
)
def test_a_subst_drive_folds_onto_its_target(path: str, expected: str):
    """The Python runtime gate resolves a SUBST drive; Path.resolve does that.

    Left unresolved here, one directory reached under two spellings produces two
    different install mutexes, and a Studio running from the physical spelling is
    invisible to the in-use scan. Measured on windows-latest: subst.exe is the
    only source available without a compiler -- Get-PSDrive.DisplayRoot,
    Win32_LogicalDisk.ProviderName, GetFullPath and Resolve-Path all reveal
    nothing about the mapping.
    """
    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                _helpers("Get-StudioSubstTarget"),
                '$script:StudioSubstMap = @{ "X" = "D:\\real\\dir" }',
                f"Write-Output \"SUBST:[$(Get-StudioSubstTarget -Path '{path}')]\"",
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    got = _lines(result, "SUBST:")[0][len("SUBST:[") : -1]
    if expected:
        assert got.startswith(expected)
    else:
        assert got == ""


@requires_pwsh
def test_the_recorded_owner_outranks_the_name(tmp_path: Path):
    """The PID in the name is the INSTALLER's, and it exits.

    The process that keeps using the directory is the Studio the installer
    autostarted, which is what owner.pid records. Reading only the name would
    clear a live Studio's %TEMP% on the next run.
    """
    root = tmp_path / "root"
    root.mkdir()
    # Name says dead, owner.pid says alive: keep it.
    keep = root / f"ust-{_DEAD_PID}-a"
    keep.mkdir()
    (keep / "owner.pid").write_text(str(os.getpid()), encoding = "utf-8")
    # Name says alive, owner.pid says dead: the recorded owner wins, so sweep it.
    drop = root / f"ust-{os.getpid()}-b"
    drop.mkdir()
    (drop / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")

    aged = time.time() - 3 * 24 * 3600
    for d in (keep, drop):
        os.utime(d, (aged, aged))

    result = _run_powershell(
        _script(
            f"Remove-StudioStalePrivateTempDirectories -Root '{root}'",
            sabotage = False,
            names = ("Remove-StudioStalePrivateTempDirectories",),
        )
    )
    assert result.returncode == 0, result.stderr
    assert keep.exists()
    assert not drop.exists()


@requires_pwsh
def test_the_sweep_keeps_a_directory_whose_owner_is_still_running(tmp_path: Path):
    """A Studio autostarted by an earlier install owns one of these as its %TEMP%.

    It can outlive the one-day cutoff without ever writing to the directory, and
    the sweep runs before the runtime mutex is taken, so age alone must not be
    read as proof that nothing is using it.
    """
    root = tmp_path / "root"
    root.mkdir()
    live = root / f"ust-{os.getpid()}-old"
    live.mkdir()
    (live / "in-use.txt").write_text("a live process owns this", encoding = "utf-8")
    abandoned = root / f"ust-{_DEAD_PID}-old"
    abandoned.mkdir()

    aged = time.time() - 3 * 24 * 3600
    os.utime(live, (aged, aged))
    os.utime(abandoned, (aged, aged))

    result = _run_powershell(
        _script(
            f"Remove-StudioStalePrivateTempDirectories -Root '{root}'",
            sabotage = False,
            names = ("Remove-StudioStalePrivateTempDirectories",),
        )
    )
    assert result.returncode == 0, result.stderr
    assert (live / "in-use.txt").exists()
    assert not abandoned.exists()


@requires_pwsh
def test_a_healthy_temp_still_sweeps_what_an_earlier_degraded_run_left(tmp_path: Path):
    """The allocator is the only thing that sweeps, and a healthy host skips it.

    Once the host's own temp is fixed (an ACL correction, a cleaned environment)
    no further run allocates a private directory, so whatever the degraded runs
    left would age in place until an uninstall.
    """
    good = tmp_path / "good"
    good.mkdir()
    local_app_data = tmp_path / "localappdata"
    root = local_app_data / "Unsloth Studio" / "temp"
    root.mkdir(parents = True)
    abandoned = root / f"ust-{_DEAD_PID}-old"
    abandoned.mkdir()
    (abandoned / "leftover.bin").write_text("half a download", encoding = "utf-8")
    aged = time.time() - 3 * 24 * 3600
    os.utime(abandoned, (aged, aged))

    env = os.environ.copy()
    env.update({"TMP": str(good), "TEMP": str(good), "LOCALAPPDATA": str(local_app_data)})
    env.pop("USERPROFILE", None)

    result = _run_powershell(
        _script(
            """
Initialize-StudioTempEnvironment
Write-Output "TMP:$env:TMP"
Write-Output "OVERRIDE:$($null -ne $script:StudioTempOverride)"
""",
            sabotage = False,
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    assert not abandoned.exists()
    # And the healthy path is still the healthy path: nothing was redirected.
    assert _lines(result, "TMP:") == [f"TMP:{good}"]
    assert _lines(result, "OVERRIDE:") == ["OVERRIDE:False"]


@requires_pwsh
def test_the_stale_sweep_never_deletes_through_a_link(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keepme.txt").write_text("do not delete", encoding = "utf-8")
    # A dead owner PID, or the sweep keeps the directory for the live process its
    # name says owns it; that case is the test below.
    stale = root / f"ust-{_DEAD_PID}-old"
    stale.mkdir()
    (stale / "junk").write_text("x", encoding = "utf-8")
    fresh = root / f"ust-{_DEAD_PID}-new"
    fresh.mkdir()
    link = root / f"ust-{_DEAD_PID}-link"
    try:
        link.symlink_to(precious, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("creating a symlink requires privilege on this host")

    aged = time.time() - 3 * 24 * 3600
    os.utime(stale, (aged, aged))
    try:
        os.utime(link, (aged, aged), follow_symlinks = False)
    except (NotImplementedError, OSError):
        # Windows has no follow_symlinks=False for utime, and aging the link any
        # other way writes THROUGH it, leaving the link fresh so the sweep skips it.
        # The 5.1 staging probe ages the reparse point via a
        # FILE_FLAG_OPEN_REPARSE_POINT handle instead.
        pytest.skip("this host cannot age a link without writing through it")

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
    # 5.1's Remove-Item -Recurse follows a junction and empties what it points at.
    # Nothing the installer creates here is a reparse point, so one costs only the link.
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
                "Get-StudioPrivateTempRoots",
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
    roots = _extract(r"    function Get-StudioPrivateTempRoots \{.*?\n    \}\n", source)

    # LOCALAPPDATA\"Unsloth Studio" is the data dir uninstall.ps1 removes wholesale.
    assert 'Join-Path $env:LOCALAPPDATA "Unsloth Studio\\temp"' in roots
    assert '"Unsloth Studio"' in uninstall
    # ~\.unsloth\.cache is on its explicit sibling list. Directly under ~\.unsloth
    # would be worse: that directory is removed only when it is empty.
    assert 'Join-Path $env:USERPROFILE ".unsloth\\.cache\\temp"' in roots
    assert (
        '$defaultCache = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".cache" }'
        in uninstall
    )
    assert ".unsloth\\temp" not in roots

    # The GetFolderPath fallback is the whole point of the second root: LOCALAPPDATA
    # is dropped in service and CI contexts. The uninstaller has to resolve the data
    # dir the same way, or the tree it places there survives an uninstall on exactly
    # the hosts that needed the fallback.
    assert '[Environment]::GetFolderPath("LocalApplicationData")' in roots
    # And it has to consider BOTH spellings rather than the first that answers:
    # install.ps1 falls through from a set-but-unusable LOCALAPPDATA to the known
    # folder, so the variable being non-blank does not say where the tree landed.
    assert "foreach ($root in @($env:LOCALAPPDATA, $knownLocalAppData)) {" in uninstall
    assert "foreach ($d in $defaultDataDirs) { _RemoveDataDirKeepingWslIcon $d }" in uninstall
    assert "_RemoveDataDirKeepingWslIcon $defaultDataDir " not in uninstall


@requires_pwsh
def test_the_uninstaller_reclaims_both_local_app_data_spellings(tmp_path: Path):
    """A set-but-unusable LOCALAPPDATA is the case that produced two roots.

    install.ps1 skips such a path and places its private temp under the known
    folder instead. An uninstaller that stops at the first non-blank candidate
    then deletes a directory that was never used and leaves the real one behind.
    """
    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")
    block = _extract(r"    # BOTH LocalAppData spellings.*?\n    \}\n", uninstall)

    dead = str(tmp_path / "gone" / "localappdata")
    env = os.environ.copy()
    env["LOCALAPPDATA"] = dead
    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                block,
                '$defaultDataDirs | ForEach-Object { Write-Output "DIR:$_" }',
            ]
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    dirs = [line[len("DIR:") :] for line in _lines(result, "DIR:")]
    assert len(dirs) == 2, dirs
    assert any(d.startswith(dead) for d in dirs), dirs
    assert all(d.rstrip("\\/").endswith("Unsloth Studio") for d in dirs), dirs
    # Deduplicated, so the ordinary host where both spellings agree is untouched.
    assert len(set(dirs)) == len(dirs)


def test_path_resolution_and_process_identity_no_longer_need_the_compiler():
    source = INSTALL_PS1.read_text(encoding = "utf-8")

    # The resolver callers reach is now a dispatcher; only the initializer builds.
    assert "Add-Type" not in _extract(r"    function Get-StudioFinalPath \{.*?\n    \}\n", source)
    assert "Add-Type" not in _extract(
        r"    function Resolve-StudioFinalPathInfo \{.*?\n    \}\n", source
    )
    assert "Add-Type" not in _extract(r"    function Get-StudioLexicalPath \{.*?\n    \}\n", source)

    # Every rung of the fallback must keep the "confirmed executable image only"
    # contract, or a degraded host would block on a name match.
    image = _extract(r"    function Get-StudioProcessImagePath \{.*?\n    \}\n", source)
    assert ".CommandLine" not in image
    assert "Win32_Process" in image
    assert "QueryFullProcessImageNameW" not in image

    # Source rather than behaviour on purpose: PowerShell 7 does not follow a link
    # on -Recurse, so the test above passes with or without this guard. Windows
    # PowerShell 5.1, the shell the desktop app spawns, does follow it.
    sweep = _extract(
        r"    function Remove-StudioStalePrivateTempDirectories \{.*?\n    \}\n", source
    )
    assert "ReparsePoint" in sweep
    reparse_branch = sweep.index("ReparsePoint")
    branch = sweep[reparse_branch : sweep.index("continue", reparse_branch)]
    assert "-Recurse" not in branch
    # Remove-Item without -Recurse is no answer either: on 5.1 it reports the
    # junction target's contents and refuses as "directory not empty", so the link is
    # never reclaimed (observed on windows-latest).
    assert "Remove-Item" not in branch
    assert "[System.IO.Directory]::Delete(" in branch
    assert ", $false)" in branch
