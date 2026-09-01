# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The installer must survive a host where Add-Type cannot compile.

install.ps1 resolves path identity through a C# helper it compiles at runtime.
Windows PowerShell 5.1 -- the interpreter studio/src-tauri/src/install.rs spawns
-- compiles by writing the source into %TEMP% and running csc.exe, so a %TEMP%
that cannot hold a file (or a scanner that eats what was just written there)
makes Add-Type throw CS2001. That exception used to travel up Get-StudioPathHash
and end a first launch as "Could not create the Unsloth install lock" (#9140),
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
from unsloth_pwsh_runner import run_pwsh


REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_PS1 = REPO_ROOT / "install.ps1"

requires_pwsh = pytest.mark.skipif(shutil.which("pwsh") is None, reason = "PowerShell is unavailable")

# Stands in for every Add-Type in install.ps1 that compiles C# rather than loading an assembly.
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
    # Through a FILE, not -Command: these scripts carry the whole extracted helper chain, and Windows caps a command
    # line at 32767 characters.
    # Passed inline, the moment the chain grows past that every test here dies as WinError 206 rather than testing
    # anything.
    # utf-8-sig because Windows PowerShell 5.1 reads a BOM-less .ps1 as ANSI;
    handle, name = tempfile.mkstemp(suffix = ".ps1")
    os.close(handle)
    try:
        Path(name).write_text(script, encoding = "utf-8-sig")
        # run_pwsh, not subprocess.run: every test in this file reads this result as "did the Add-Type fallback chain
        # survive", and an interpreter that aborted at startup produces the same empty stdout as a helper that never ran
        # its fallback.
        # See tests/_shared/unsloth_pwsh_runner.py.
        return run_pwsh(
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
            # which the caller cannot capture.
            # Write-StudioLine picks its sink from this;
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
    # One attempt, one retry with a private %TEMP%, then cached:
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
    assert _lines(result, "ATTEMPTS:") == ["ATTEMPTS:2"]
    assert _lines(result, "LOADED:") == ["LOADED:True"]
    assert not [line for line in result.stdout.splitlines() if "native path resolver" in line]
    # Restored exactly, broken values and all:
    assert _lines(result, "TMP:") == [f"TMP:{dead}"]
    assert _lines(result, "TEMP:") == [f"TEMP:{dead}"]
    assert list((local_app_data / "Unsloth Studio" / "temp").glob("ust-*")) == []
    # The P/Invoke targets kernel32, so off Windows the type loads but the CALL fails;
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
    # GetFullPath alone gets wrong.
    # A link on a PARENT component is the ordinary Windows shape, and the one GetFullPath alone gets wrong.
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
    # Without exact resolution two spellings may still be one directory;
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
    # TEMP was absent; restoring it as "" would change how every later child resolves its own temp directory.
    assert _lines(result, "TEMPSET:") == ["TEMPSET:False"]
    # It survives on purpose: an autostarted Unsloth inherited it as its own %TEMP%, and the host's real one is broken.
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

# What (Get-Item).Target actually hands back.
# 5.1 returns a COLLECTION, not a string and not an [array] either, so a container test naming one type lets the real
# one fall through and be space-joined.
# None of these shapes can be produced on Linux, so Get-Item is stubbed and the paths are POSIX-style;
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
def test_final_normalization_strips_every_extended_prefix():
    r"""This string is hashed into the runtime mutex, so Python decides its shape.

    unsloth_cli/_studio_runtime_gate.py::_resolved_windows_path strips \\?\
    unconditionally, after one special case for \\?\UNC\. A volume GUID kept its
    prefix here for a while, on the reasoning that the bare "Volume{GUID}\x" is
    not rooted. That is true, and it matters while a LINK TARGET is being
    anchored, which is why Resolve-StudioLinkTarget still keeps the extended
    form. It does not matter here, and keeping it made the installer and a
    running Unsloth compute different names for one directory.
    """
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    body = _extract(r"    function Resolve-StudioFinalPathInfo \{.*?\n    \}\n", source)
    assert "'\\\\?\\Volume{'" not in body
    assert "$resolved.Substring(4)" in body

    gate = (REPO_ROOT / "unsloth_cli" / "_studio_runtime_gate.py").read_text(encoding = "utf-8")
    # The two sides have to agree about which prefixes come off, and there are exactly two rules on the Python side.
    assert 'resolved.startswith("\\\\\\\\?\\\\UNC\\\\")' in gate
    assert "resolved = resolved[4:]" in gate
    assert "Volume{" not in gate


@requires_pwsh
@pytest.mark.parametrize(
    "resolved,expected",
    [
        ("\\\\?\\C:\\Users\\bob\\studio", "C:\\Users\\bob\\studio"),
        ("\\\\?\\UNC\\server\\share\\studio", "\\\\server\\share\\studio"),
        (
            "\\\\?\\Volume{11111111-2222-3333-4444-555555555555}\\data\\studio",
            "Volume{11111111-2222-3333-4444-555555555555}\\data\\studio",
        ),
    ],
    ids = ["extended-dos", "extended-unc", "volume-guid"],
)
def test_the_installer_and_the_runtime_gate_normalize_alike(resolved: str, expected: str):
    """Byte-for-byte agreement, or the two sides key their lock on different names."""
    source = INSTALL_PS1.read_text(encoding = "utf-8")
    body = _extract(r"    function Resolve-StudioFinalPathInfo \{.*?\n    \}\n", source)
    branch = _extract(
        r"        if \(\$resolved\.StartsWith\('\\\\\?\\UNC\\'.*?\n        \}\n", body
    )
    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                f"$resolved = '{resolved}'",
                branch,
                'Write-Output "OUT:$resolved"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    got = _lines(result, "OUT:")[0][len("OUT:") :]
    assert got == expected

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_studio_runtime_gate", REPO_ROOT / "unsloth_cli" / "_studio_runtime_gate.py"
    )
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    python_side = resolved
    if python_side.startswith("\\\\?\\UNC\\"):
        python_side = "\\\\" + python_side[8:]
    elif python_side.startswith("\\\\?\\"):
        python_side = python_side[4:]
    assert got == python_side


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
    different install mutexes, and an Unsloth running from the physical spelling is
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

    The process that keeps using the directory is the Unsloth the installer
    autostarted, which is what owner.pid records. Reading only the name would
    clear a live Unsloth's %TEMP% on the next run.
    """
    root = tmp_path / "root"
    root.mkdir()
    # Name says dead, owner.pid says alive:
    keep = root / f"ust-{_DEAD_PID}-000000aa"
    keep.mkdir()
    (keep / "owner.pid").write_text(str(os.getpid()), encoding = "utf-8")
    # Name says alive, owner.pid says dead:
    drop = root / f"ust-{os.getpid()}-000000bb"
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
    """An Unsloth autostarted by an earlier install owns one of these as its %TEMP%.

    It can outlive the one-day cutoff without ever writing to the directory, and
    the sweep runs before the runtime mutex is taken, so age alone must not be
    read as proof that nothing is using it.
    """
    root = tmp_path / "root"
    root.mkdir()
    live = root / f"ust-{os.getpid()}-01d01d01"
    live.mkdir()
    (live / "in-use.txt").write_text("a live process owns this", encoding = "utf-8")
    abandoned = root / f"ust-{_DEAD_PID}-01d01d01"
    abandoned.mkdir()
    (abandoned / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")

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
    abandoned = root / f"ust-{_DEAD_PID}-01d01d01"
    abandoned.mkdir()
    # Recorded, not guessed: an unrecorded owner is now treated as unknown.
    (abandoned / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
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
    # And the healthy path is still the healthy path:
    assert _lines(result, "TMP:") == [f"TMP:{good}"]
    assert _lines(result, "OVERRIDE:") == ["OVERRIDE:False"]


@requires_pwsh
def test_probing_the_host_temp_never_creates_it(tmp_path: Path):
    """An inherited TMP that names nothing is unusable, not an instruction.

    New-Item -Force builds the whole parent chain, so treating the host's own
    TMP as creatable would have the installer materialize a tree at a path
    nobody chose, on a stale or mistyped value, and then trust it as temp.
    """
    ghost = tmp_path / "ghost" / "deeper"
    result = _run_powershell(
        _script(
            f"""
Write-Output "USABLE:$(Test-StudioDirectoryUsable -Path '{ghost}')"
Write-Output "EXISTS:$(Test-Path -LiteralPath '{ghost}')"
""",
            sabotage = False,
            names = ("Write-StudioLine", "Test-StudioDirectoryUsable"),
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "USABLE:") == ["USABLE:False"]
    assert _lines(result, "EXISTS:") == ["EXISTS:False"]
    assert not ghost.exists()
    # A directory the installer owns is a different matter:
    owned = tmp_path / "owned" / "ust-1-aaaaaaaa"
    result = _run_powershell(
        _script(
            f"Write-Output \"USABLE:$(Test-StudioDirectoryUsable -Path '{owned}' -CreateIfMissing)\"",
            sabotage = False,
            names = ("Write-StudioLine", "Test-StudioDirectoryUsable"),
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "USABLE:") == ["USABLE:True"]
    assert owned.is_dir()


@requires_pwsh
def test_an_undeletable_probe_file_is_reclaimed_next_time(tmp_path: Path):
    """The probe cannot clean up after itself when deletion is what failed.

    Without a sweep every such run leaves one more file in the host's own
    temp, forever, since nothing else knows the name. Aged, so a probe running
    concurrently in another process is never touched.
    """
    good = tmp_path / "good"
    good.mkdir()
    stale = good / "unsloth-probe-deadbeef.tmp"
    stale.write_text("left by a run that could not delete it", encoding = "utf-8")
    fresh = good / "unsloth-probe-cafebabe.tmp"
    fresh.write_text("another process is using this right now", encoding = "utf-8")
    # Same prefix, same suffix, not the shape the probe writes.
    # This sweep runs in the HOST's temp directory, so a name that merely starts the same way is somebody else's file
    # however old it is.
    theirs = good / "unsloth-probe-report.tmp"
    theirs.write_text("not ours", encoding = "utf-8")
    theirs_long = good / "unsloth-probe-deadbeefcafe.tmp"
    theirs_long.write_text("not ours either", encoding = "utf-8")
    aged = time.time() - 3 * 24 * 3600
    for path in (stale, theirs, theirs_long):
        os.utime(path, (aged, aged))

    result = _run_powershell(
        _script(
            f"Write-Output \"USABLE:$(Test-StudioDirectoryUsable -Path '{good}')\"",
            sabotage = False,
            names = ("Write-StudioLine", "Test-StudioDirectoryUsable"),
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "USABLE:") == ["USABLE:True"]
    assert not stale.exists()
    assert fresh.exists()
    assert theirs.exists(), "the sweep took a file that only shares the prefix"
    assert theirs_long.exists(), "the sweep took a file that only shares the prefix"


@requires_pwsh
def test_a_root_that_fails_its_probe_is_not_left_behind(tmp_path: Path):
    """A failed install must not conjure the data directory tree.

    The probe creates each candidate before testing it, so on a host where
    every root fails, giving up used to leave a "Unsloth Studio" tree on a
    machine Unsloth was never installed on.
    """
    local_app_data = tmp_path / "localappdata"
    local_app_data.mkdir()
    user_profile = tmp_path / "userprofile"
    user_profile.mkdir()

    # Pre-existing content under one of the same parents:
    keep = local_app_data / "Unsloth Studio" / "studio.port"
    keep.parent.mkdir(parents = True)
    keep.write_text("41343", encoding = "utf-8")

    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(local_app_data)
    env["USERPROFILE"] = str(user_profile)

    result = _run_powershell(
        _script(
            """
# Every candidate fails its probe, whatever the filesystem says.
function Test-StudioDirectoryUsable {
    param([string]$Path, [switch]$CreateIfMissing)
    # Creates unconditionally, the way the real probe did before it took the
    # switch, so this measures the caller rather than the stub.
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    return $false
}
Write-Output "PRIVATE:$(New-StudioPrivateTempDirectory)"
""",
            sabotage = False,
            names = (
                "Write-StudioLine",
                "Remove-StudioStalePrivateTempDirectories",
                "Get-StudioPrivateTempRoots",
                "New-StudioPrivateTempDirectory",
            ),
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "PRIVATE:") == ["PRIVATE:"]
    # and "temp" were conjured too and a run that gave up must not leave a data
    # directory tree on a machine Unsloth was never installed on.
    # Not just the ust-* leaf: -Force built the whole chain, so "Unsloth Studio" and "temp" were conjured too and a run
    assert not (user_profile / ".unsloth" / ".cache").exists()
    assert not list(user_profile.rglob("ust-*")), list(user_profile.rglob("*"))
    assert keep.exists()
    assert not list(local_app_data.rglob("ust-*")), list(local_app_data.rglob("*"))
    assert not (local_app_data / "Unsloth Studio" / "temp").exists()


@requires_pwsh
def test_the_sweep_only_takes_directories_the_allocator_could_have_made(tmp_path: Path):
    """Shape, not prefix. The delete is recursive and this is the only owner test.

    A prefix match takes "ust-legacy" and "ust-user-cache" as well, and neither
    has a parseable PID, so the liveness check is skipped for exactly the names
    least likely to be ours. scripts/uninstall.ps1 already required the shape;
    the installer sweep had drifted away from it.
    """
    root = tmp_path / "root"
    root.mkdir()
    aged = time.time() - 3 * 24 * 3600

    ours = root / f"ust-{_DEAD_PID}-abcdef01"
    ours.mkdir()
    (ours / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
    (ours / "scratch.bin").write_text("x", encoding = "utf-8")
    keep = []
    # Case-insensitively ours:
    # Case-insensitively ours: Windows filenames are case-insensitive, so refusing the uppercase spelling would leak a
    upper = root / f"ust-{_DEAD_PID}-ABCDEF01"
    upper.mkdir()
    (upper / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
    for name in ("ust-legacy", "ust-user-cache", "ust-notapid-abcdef01", "ust-", "ust-12-abcdefg1"):
        victim = root / name
        victim.mkdir()
        (victim / "keep.txt").write_text("not ours", encoding = "utf-8")
        keep.append(victim)
    for path in [ours, upper] + keep:
        os.utime(path, (aged, aged))

    result = _run_powershell(
        _script(
            f"Remove-StudioStalePrivateTempDirectories -Root '{root}'",
            sabotage = False,
            names = ("Remove-StudioStalePrivateTempDirectories",),
        )
    )
    assert result.returncode == 0, result.stderr
    assert not ours.exists()
    assert not upper.exists()
    for victim in keep:
        assert (victim / "keep.txt").exists(), f"{victim.name} was swept"


@requires_pwsh
def test_a_native_resolver_that_throws_says_so_once(tmp_path: Path):
    """Compiling and then failing to resolve is not the same as no compiler.

    The install still proceeds on the lexical answer, and Exact = $false already
    makes the runtime lock fail closed, but nothing said so: the degraded warning
    fires only when the COMPILE failed. That left an operator with a silently
    inexact identity on a host that looks perfectly healthy.
    """
    studio = tmp_path / "studio"
    studio.mkdir()
    result = _run_powershell(
        _script(
            f"""
# The helper is "available" and throws anyway, which is what a rename between
# the Test-Path walk and CreateFileW looks like.
function Initialize-StudioFinalPathNativeType {{ return $true }}
Add-Type -TypeDefinition @'
public class UnslothStudioFinalPathV2 {{
    public static string Resolve(string path) {{ throw new System.Exception("access is denied"); }}
}}
'@
foreach ($i in 1..3) {{ $null = Resolve-StudioFinalPathInfo -Path '{studio}' }}
$info = Resolve-StudioFinalPathInfo -Path '{studio}'
Write-Output "EXACT:$($info.Exact)"
Write-Output "PATH:$($info.Path)"
""",
            sabotage = False,
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "EXACT:") == ["EXACT:False"]
    assert _lines(result, "PATH:")[0].endswith("studio")
    warnings = [line for line in result.stdout.splitlines() if "native helper; continuing" in line]
    assert len(warnings) == 1, warnings


@requires_pwsh
def test_an_unrecorded_owner_is_unknown_rather_than_abandoned(tmp_path: Path):
    """The name's PID is the installer's, and it can be dead while Unsloth is not.

    An installer killed between Start-Process and the owner.pid write leaves a
    directory whose only owner evidence is a dead installer PID, while the
    Unsloth it started is using that directory as its own %TEMP%. Reading is
    proof; guessing from the name is not, so the two get different patience.
    """
    root = tmp_path / "root"
    root.mkdir()
    two_days = time.time() - 2 * 24 * 3600
    eight_days = time.time() - 8 * 24 * 3600

    # No owner.pid, dead name PID, two days old:
    unknown = root / f"ust-{_DEAD_PID}-aaaaaaaa"
    unknown.mkdir()
    (unknown / "in-use.txt").write_text("a live Unsloth may own this", encoding = "utf-8")
    os.utime(unknown, (two_days, two_days))

    # Same, but a week past: collected, so the pile still stays bounded.
    ancient = root / f"ust-{_DEAD_PID}-bbbbbbbb"
    ancient.mkdir()
    os.utime(ancient, (eight_days, eight_days))

    # Recorded dead owner, two days old:
    recorded = root / f"ust-{_DEAD_PID}-cccccccc"
    recorded.mkdir()
    (recorded / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
    os.utime(recorded, (two_days, two_days))

    result = _run_powershell(
        _script(
            f"Remove-StudioStalePrivateTempDirectories -Root '{root}'",
            sabotage = False,
            names = ("Remove-StudioStalePrivateTempDirectories",),
        )
    )
    assert result.returncode == 0, result.stderr
    assert (unknown / "in-use.txt").exists(), "an unrecorded owner was read as abandoned"
    assert not ancient.exists(), "an unrecorded owner is never collected at all"
    assert not recorded.exists(), "a recorded dead owner should still go at one day"


@requires_pwsh
def test_the_stale_sweep_never_deletes_through_a_link(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    precious = tmp_path / "precious"
    precious.mkdir()
    (precious / "keepme.txt").write_text("do not delete", encoding = "utf-8")
    # A dead owner PID, or the sweep keeps the directory for the live process its name says owns it;
    stale = root / f"ust-{_DEAD_PID}-01d01d01"
    stale.mkdir()
    (stale / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
    (stale / "junk").write_text("x", encoding = "utf-8")
    fresh = root / f"ust-{_DEAD_PID}-0e0e0e0e"
    fresh.mkdir()
    (fresh / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
    link = root / f"ust-{_DEAD_PID}-11111111"
    try:
        link.symlink_to(precious, target_is_directory = True)
    except (OSError, NotImplementedError):
        pytest.skip("creating a symlink requires privilege on this host")

    aged = time.time() - 3 * 24 * 3600
    os.utime(stale, (aged, aged))
    try:
        os.utime(link, (aged, aged), follow_symlinks = False)
    except (NotImplementedError, OSError):
        # Windows has no follow_symlinks=False for utime, and aging the link any other way writes THROUGH it, leaving
        # the link fresh so the sweep skips it.
        # The 5.1 staging probe ages the reparse point via a FILE_FLAG_OPEN_REPARSE_POINT handle instead.
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
    # 5.1's Remove-Item -Recurse follows a junction and empties what it points at;
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
    # ~\.unsloth\.cache is on its explicit sibling list.
    assert 'Join-Path $env:USERPROFILE ".unsloth\\.cache\\temp"' in roots
    assert (
        '$defaultCache = if ($defaultUnslothHome) { Join-Path $defaultUnslothHome ".cache" }'
        in uninstall
    )
    assert ".unsloth\\temp" not in roots

    # The GetFolderPath fallback is the point of the second root:
    assert '[Environment]::GetFolderPath("LocalApplicationData")' in roots
    # And it has to consider BOTH spellings rather than the first that answers:
    assert "foreach ($root in @($env:LOCALAPPDATA, $knownLocalAppData)) {" in uninstall
    # And the second spelling gets the TEMP TREE ONLY:
    assert 'Join-Path $root "Unsloth Studio\\temp"' in uninstall
    assert "_RemoveStudioPrivateTempTrees -Paths $privateTempDirs" in uninstall
    assert "foreach ($d in $defaultDataDirs)" not in uninstall


@requires_pwsh
def test_the_uninstall_sweep_leaves_a_live_owner_and_never_follows_a_link(tmp_path: Path):
    """An uninstall stops the Unsloth instances under the roots it knows about, not others.

    An Unsloth from another install root, or another user, can be alive on one of
    these directories as its %TEMP%; install.ps1's own sweep preserves it and so
    must this one. And a temp directory that is itself a link must not be walked:
    the target's children carry no ReparsePoint attribute, so a recursive delete
    would take an unrelated tree.
    """
    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")
    block = _extract(r"    function _RemoveStudioPrivateTempTrees \{.*?\n    \}\n", uninstall)
    preamble = (
        '$ErrorActionPreference = "Stop"\nfunction _Substep { param([string]$Msg, [string]$Color) }'
    )

    temp = tmp_path / "Unsloth Studio" / "temp"
    temp.mkdir(parents = True)
    live = temp / "ust-1234-abcdef01"
    live.mkdir()
    (live / "owner.pid").write_text(str(os.getpid()), encoding = "utf-8")
    dead = temp / "ust-1234-abcdef02"
    dead.mkdir()
    (dead / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")

    result = _run_powershell(
        "\n".join(
            [
                preamble,
                block,
                f"_RemoveStudioPrivateTempTrees -Paths @('{temp}') -PrimaryPath '{temp}'",
                'Write-Output "DONE:1"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "DONE:") == ["DONE:1"]
    assert (live / "owner.pid").exists(), "a live owner's temp was removed"
    assert not dead.exists()

    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "ust-1234-abcdef03").mkdir()
    (victim / "ust-1234-abcdef03" / "precious.txt").write_text("not ours", encoding = "utf-8")
    linked = tmp_path / "Linked Unsloth" / "temp"
    linked.parent.mkdir()
    linked.symlink_to(victim, target_is_directory = True)

    result = _run_powershell(
        "\n".join(
            [
                preamble,
                block,
                f"_RemoveStudioPrivateTempTrees -Paths @('{linked}') -PrimaryPath '{linked}'",
                'Write-Output "DONE:2"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "DONE:") == ["DONE:2"]
    assert (
        victim / "ust-1234-abcdef03" / "precious.txt"
    ).exists(), "the sweep walked through a link"


@requires_pwsh
def test_a_live_owner_survives_the_data_directory_removal(tmp_path: Path):
    """Preserving a directory is worth nothing if the next line deletes its parent.

    The primary private temp directory sits inside the data directory, and the
    data directory is removed wholesale. An Unsloth from another install root can
    be alive on that temp directory as its %TEMP%, so the sweep has to run first
    and the removal has to be told what the sweep kept.
    """
    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")

    # The order is a property of the script body, not of any one function, so it is checked as one:
    body = uninstall[uninstall.index("function Uninstall-UnslothStudio") :]
    calls = [
        line.strip()
        for line in body.splitlines()
        if (
            "_RemoveDataDirKeepingWslIcon $defaultDataDir" in line
            or "_RemoveStudioPrivateTempTrees -Paths" in line
        )
    ]
    assert calls, "neither call is in the script body"
    for index, line in enumerate(calls):
        if "_RemoveDataDirKeepingWslIcon" not in line:
            continue
        assert "-Preserve" in line, f"data dir removed without the preserved list: {line}"
        assert (
            index > 0 and "_RemoveStudioPrivateTempTrees" in calls[index - 1]
        ), f"the data dir is removed before the temp sweep runs: {line}"

    blocks = "\n".join(
        _extract(rf"    function {name} \{{.*?\n    \}}\n", uninstall)
        for name in (
            "_RemoveStudioPrivateTempTrees",
            "_RemoveTreeKeeping",
            "_RemoveDataDirKeepingWslIcon",
        )
    )

    data = tmp_path / "Unsloth Studio"
    temp = data / "temp"
    temp.mkdir(parents = True)
    live = temp / "ust-4321-abcdef05"
    live.mkdir()
    (live / "owner.pid").write_text(str(os.getpid()), encoding = "utf-8")
    (live / "scratch.bin").write_text("in use", encoding = "utf-8")
    dead = temp / "ust-4321-abcdef06"
    dead.mkdir()
    (dead / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
    other = data / "launcher.db"
    other.write_text("data", encoding = "utf-8")

    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                "function _Substep { param([string]$Msg, [string]$Color) }",
                "function _RemovePath { param([string]$Path)"
                " Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue }",
                blocks,
                f"$kept = @(_RemoveStudioPrivateTempTrees -Paths @('{temp}') -PrimaryPath '{temp}')",
                'Write-Output ("KEPT:" + @($kept).Count)',
                f"_RemoveDataDirKeepingWslIcon -DataDir '{data}' -ShortcutDirs @() -Preserve $kept",
                'Write-Output "DONE:1"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "DONE:") == ["DONE:1"]
    assert _lines(result, "KEPT:") == ["KEPT:1"]
    assert (live / "scratch.bin").exists(), "a live Unsloth's %TEMP% went with the data dir"
    assert not dead.exists()
    assert not other.exists(), "the rest of the data dir was left behind"


@requires_pwsh
def test_a_link_high_above_another_profile_is_still_a_link(tmp_path: Path):
    r"""A junction does not have to sit next to the temp directory to redirect it.

    LocalAppData itself, the profile, or the drive root can be the reparse
    point, and then "<root>\Unsloth Studio\temp" and its parent both look like
    perfectly ordinary directories while the enumeration lands somewhere else
    entirely. For a spelling that is not the profile being uninstalled, that
    somewhere else can be another user, so every ancestor has to be ordinary.

    The profile this uninstall IS for is deliberately not held to that: a
    redirected LocalAppData there is the same user's own storage, and refusing
    would leave the installer's own temp tree behind on every host that uses
    folder redirection.
    """
    if os.path.realpath(tmp_path) != str(tmp_path):
        pytest.skip("the temp root itself is a link, which is what this test plants")

    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")
    block = _extract(r"    function _RemoveStudioPrivateTempTrees \{.*?\n    \}\n", uninstall)
    preamble = (
        '$ErrorActionPreference = "Stop"\nfunction _Substep { param([string]$Msg, [string]$Color) }'
    )

    # The real profile, with an Unsloth temp tree in it that belongs to a dead owner:
    real = tmp_path / "real profile"
    real_temp = real / "localappdata" / "Unsloth Studio" / "temp"
    real_temp.mkdir(parents = True)
    stale = real_temp / "ust-1234-abcdef04"
    stale.mkdir()
    (stale / "owner.pid").write_text(str(_DEAD_PID), encoding = "utf-8")
    (stale / "precious.txt").write_text("another profile", encoding = "utf-8")

    # Three levels above the temp directory, so neither it nor its parent is a link.
    redirected = tmp_path / "redirected profile"
    redirected.symlink_to(real, target_is_directory = True)
    aliased = redirected / "localappdata" / "Unsloth Studio" / "temp"

    other = tmp_path / "mine" / "Unsloth Studio" / "temp"
    other.mkdir(parents = True)

    result = _run_powershell(
        "\n".join(
            [
                preamble,
                block,
                f"_RemoveStudioPrivateTempTrees -Paths @('{aliased}') -PrimaryPath '{other}'",
                'Write-Output "DONE:1"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "DONE:") == ["DONE:1"]
    assert (stale / "precious.txt").exists(), "the sweep walked through a link high above the root"

    result = _run_powershell(
        "\n".join(
            [
                preamble,
                block,
                f"_RemoveStudioPrivateTempTrees -Paths @('{aliased}') -PrimaryPath '{aliased}'",
                'Write-Output "DONE:2"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "DONE:") == ["DONE:2"]
    assert not stale.exists(), "a redirected profile cannot clean its own temp tree"


@requires_pwsh
def test_a_pre_existing_fallback_parent_is_not_unwound(tmp_path: Path):
    """Only what the probe created may be taken back.

    A pre-provisioned "Unsloth Studio\temp" with its own ACLs, or an empty
    relocation junction, is configuration this installer did not create. Empty
    and correctly named is not the same as ours.
    """
    local_app_data = tmp_path / "localappdata"
    provisioned = local_app_data / "Unsloth Studio" / "temp"
    provisioned.mkdir(parents = True)
    user_profile = tmp_path / "userprofile"
    user_profile.mkdir()

    env = os.environ.copy()
    env["LOCALAPPDATA"] = str(local_app_data)
    env["USERPROFILE"] = str(user_profile)

    result = _run_powershell(
        _script(
            """
function Test-StudioDirectoryUsable {
    param([string]$Path, [switch]$CreateIfMissing)
    New-Item -ItemType Directory -Path $Path -Force | Out-Null
    return $false
}
Write-Output "PRIVATE:$(New-StudioPrivateTempDirectory)"
""",
            sabotage = False,
            names = (
                "Write-StudioLine",
                "Remove-StudioStalePrivateTempDirectories",
                "Get-StudioPrivateTempRoots",
                "New-StudioPrivateTempDirectory",
            ),
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "PRIVATE:") == ["PRIVATE:"]
    # The candidate the probe made is gone;
    assert not list(provisioned.glob("ust-*"))
    assert provisioned.is_dir(), "a pre-existing temp directory was unwound"
    # And the tree the probe DID create under the other root is still taken back.
    assert not (user_profile / ".unsloth" / ".cache").exists()


@requires_pwsh
def test_the_uninstall_sweep_needs_a_recorded_owner_outside_its_own_profile(tmp_path: Path):
    """The alternate LocalAppData spelling can be another user's profile.

    install.ps1 reads a missing owner.pid as unknown rather than abandoned,
    because an installer killed before writing it leaves a live Unsloth holding
    the directory. Deleting that out of somebody else's profile is not this
    uninstall's business. Under our own profile the shape is enough, since that
    is what is being removed.
    """
    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")
    block = _extract(r"    function _RemoveStudioPrivateTempTrees \{.*?\n    \}\n", uninstall)
    preamble = (
        '$ErrorActionPreference = "Stop"\nfunction _Substep { param([string]$Msg, [string]$Color) }'
    )

    mine = tmp_path / "mine" / "Unsloth Studio" / "temp"
    theirs = tmp_path / "theirs" / "Unsloth Studio" / "temp"
    for root in (mine, theirs):
        root.mkdir(parents = True)
        (root / "ust-1234-abcdef01").mkdir()

    # Same shape, but now it is this uninstall's own profile:
    result = _run_powershell(
        "\n".join(
            [
                preamble,
                block,
                f"_RemoveStudioPrivateTempTrees -Paths @('{mine}','{theirs}') -PrimaryPath '{mine}'",
                'Write-Output "DONE:1"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "DONE:") == ["DONE:1"]
    assert not (mine / "ust-1234-abcdef01").exists(), "our own profile should be reclaimed"
    assert (
        theirs / "ust-1234-abcdef01"
    ).is_dir(), "another profile was swept without a recorded owner"


@requires_pwsh
def test_the_uninstaller_reclaims_the_temp_tree_at_both_spellings(tmp_path: Path):
    """A set-but-unusable LOCALAPPDATA is the case that produced two roots.

    install.ps1 skips such a path and places its private temp under the known
    folder instead, so an uninstaller that stops at the first non-blank
    candidate leaves the real tree behind. What the second root must NOT get is
    the recursive, sentinel-free data-dir delete: the two spellings differ
    mainly when one of them names a DIFFERENT USER's profile.
    """
    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")
    block = _extract(r"    # The SECOND LocalAppData spelling.*?\n    \}\n", uninstall)

    dead = str(tmp_path / "gone" / "localappdata")
    env = os.environ.copy()
    env["LOCALAPPDATA"] = dead
    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                block,
                '$privateTempDirs | ForEach-Object { Write-Output "DIR:$_" }',
            ]
        ),
        env = env,
    )
    assert result.returncode == 0, result.stderr
    dirs = [line[len("DIR:") :] for line in _lines(result, "DIR:")]
    assert len(dirs) == 2, dirs
    assert any(d.startswith(dead) for d in dirs), dirs
    assert all(d.rstrip("\\/").endswith("temp") for d in dirs), dirs
    assert len(set(dirs)) == len(dirs)


@requires_pwsh
def test_the_private_temp_removal_only_takes_what_it_created(tmp_path: Path):
    """Narrow on purpose: this runs against a root that may be another user's.

    Only ust-<pid>-<hex> directories go, matched by shape rather than prefix,
    and the temp directory and its parent only when they are left empty.
    """
    uninstall = (REPO_ROOT / "scripts" / "uninstall.ps1").read_text(encoding = "utf-8")
    block = _extract(r"    function _RemoveStudioPrivateTempTrees \{.*?\n    \}\n", uninstall)

    data = tmp_path / "Unsloth Studio"
    temp = data / "temp"
    temp.mkdir(parents = True)
    (temp / "ust-1234-abcdef01").mkdir()
    (temp / "ust-1234-abcdef01" / "scratch.bin").write_text("x", encoding = "utf-8")
    (temp / "ust-legacy").mkdir()
    (temp / "ust-notapid-abcdef01").mkdir()
    (temp / "somebody-elses").mkdir()
    (data / "studio.port").write_text("41343", encoding = "utf-8")

    result = _run_powershell(
        "\n".join(
            [
                '$ErrorActionPreference = "Stop"',
                "function _Substep { param([string]$Msg, [string]$Color) }",
                block,
                f"_RemoveStudioPrivateTempTrees -Paths @('{temp}') -PrimaryPath '{temp}'",
                'Write-Output "DONE:1"',
            ]
        )
    )
    assert result.returncode == 0, result.stderr
    assert _lines(result, "DONE:") == ["DONE:1"]
    assert not (temp / "ust-1234-abcdef01").exists()
    # Prefix, not shape, would have taken all three of these.
    assert (temp / "ust-legacy").is_dir()
    assert (temp / "ust-notapid-abcdef01").is_dir()
    assert (temp / "somebody-elses").is_dir()
    assert temp.is_dir()
    assert (data / "studio.port").exists()
