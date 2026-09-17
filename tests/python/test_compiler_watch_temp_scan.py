# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The no-compiler detector must not fail the job because something else touched TEMP.

Watch-ForCompiler.ps1 lists the temp roots to see what a compile left behind. It did that with
one `Get-ChildItem -Recurse -Force -ErrorAction SilentlyContinue` per root. A temp root is
shared with everything else on the machine, so a directory can disappear or stop being openable
partway through the walk, and the provider raises a Win32Exception. `-ErrorAction
SilentlyContinue` does not suppress that one: it governs non-terminating errors, and the step
sets `$ErrorActionPreference = 'Stop'`.

Observed on hosted runners, in the POSITIVE CONTROL, which is the worst place for it:

    Get-ChildItem : The system cannot find the file specified
    + CategoryInfo : NotSpecified: (:) [Get-ChildItem], Win32Exception

The job went red while the installer under test had done nothing at all. The walk is by hand
now, one directory at a time, so an unreadable directory costs that directory and nothing else.

Driven rather than read: the whole question is what happens when enumeration throws, and a
regex over the script cannot answer it.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "Watch-ForCompiler.ps1"

PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(PWSH is None, reason="needs PowerShell")


def _walk(root: pathlib.Path) -> list[str]:
    """Run Get-StudioTempSubtree over `root` under the same preference CI uses."""
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        f". '{SCRIPT}'\n"
        f"Get-StudioTempSubtree -Root '{root}' | ForEach-Object {{ Write-Output $_ }}\n"
    )
    proc = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True, text = True, timeout = 300,
    )
    assert proc.returncode == 0, f"the walk itself failed:\n{proc.stdout}\n{proc.stderr}"
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def test_an_unreadable_directory_costs_only_itself(tmp_path: pathlib.Path) -> None:
    (tmp_path / "keep").mkdir()
    (tmp_path / "keep" / "probe.dll").write_text("x")
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked" / "hidden.dll").write_text("x")
    (tmp_path / "later").mkdir()
    (tmp_path / "later" / "after.cmdline").write_text("x")
    os.chmod(tmp_path / "locked", 0o000)
    try:
        if os.geteuid() == 0:
            pytest.skip("root reads every directory, so nothing here can be made unreadable")
        found = _walk(tmp_path)
    finally:
        os.chmod(tmp_path / "locked", 0o700)

    # The bites control first: if the tree were readable throughout, this row would pass on a
    # walk that stopped at the first directory and never proved anything.
    assert any(name.endswith("probe.dll") for name in found), found
    assert any(name.endswith("after.cmdline") for name in found), (
        "the walk stopped at the unreadable directory instead of stepping over it"
    )
    assert not any(name.endswith("hidden.dll") for name in found), found


@pytest.mark.skipif(os.name != "nt", reason="the error this reproduces is a Windows one")
def test_the_walk_really_would_have_died_without_the_handling(tmp_path: pathlib.Path) -> None:
    """The control for the row above: the shape it replaced does end the run.

    Windows only, and deliberately so. The error that broke CI is a Win32Exception raised by the
    Windows provider partway through a recursive enumeration, and `-ErrorAction
    SilentlyContinue` does not suppress it. The POSIX provider reports an unreadable directory
    as an ordinary non-terminating error, which that parameter does suppress, so on Linux this
    control would assert something untrue and pass the rewrite off as unnecessary.
    """
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked" / "hidden.dll").write_text("x")
    # DENY on the directory itself, which is how a temp directory belonging to another user
    # presents to this process.
    subprocess.run(
        ["icacls", str(tmp_path / "locked"), "/deny", "*S-1-1-0:(OI)(CI)(RX)"],
        capture_output = True, text = True, timeout = 300,
    )
    try:
        proc = subprocess.run(
            [
                PWSH, "-NoProfile", "-NonInteractive", "-Command",
                "$ErrorActionPreference = 'Stop'\n"
                f"Get-ChildItem -LiteralPath '{tmp_path}' -Filter '*.dll' -File -Recurse "
                "-Force -ErrorAction SilentlyContinue | Out-Null\n"
                "Write-Output 'SURVIVED'\n",
            ],
            capture_output = True, text = True, timeout = 300,
        )
    finally:
        subprocess.run(
            ["icacls", str(tmp_path / "locked"), "/remove:d", "*S-1-1-0"],
            capture_output = True, text = True, timeout = 300,
        )
    assert "SURVIVED" not in proc.stdout, (
        "-ErrorAction SilentlyContinue suppressed the error on this host, so the reason for the "
        "rewrite needs re-checking"
    )


def test_the_artifact_filter_still_selects_by_extension(tmp_path: pathlib.Path) -> None:
    """Splitting the walk from the filter must not widen what counts as an artifact."""
    (tmp_path / "sub").mkdir()
    for name in ("a.dll", "b.cmdline", "c.rsp", "d.cs", "e.err", "f.out", "g.txt", "h.exe"):
        (tmp_path / "sub" / name).write_text("x")
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        f". '{SCRIPT}'\n"
        f"$env:TEMP = '{tmp_path}'\n"
        f"$env:TMP = '{tmp_path}'\n"
        "Get-StudioTempArtifacts | ForEach-Object { Write-Output (Split-Path -Leaf $_) }\n"
    )
    proc = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True, text = True, timeout = 300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    got = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    assert {"a.dll", "b.cmdline", "c.rsp", "d.cs", "e.err", "f.out"} <= got, got
    assert "g.txt" not in got and "h.exe" not in got, got


def test_no_recursive_listing_is_left_in_the_script() -> None:
    """The shape that raised, pinned out of the file it was removed from.

    A future edit that reaches for -Recurse again brings the whole failure back, and it only
    shows up on a runner whose temp directory happened to change under it.
    """
    text = SCRIPT.read_text(encoding = "utf-8")
    # Comments first: this file EXPLAINS the shape it removed, and prose naming it is not a
    # call. Stripping the comment-based help blocks as well, which is where that prose lives.
    body, inside_help = [], False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("<#"):
            inside_help = True
        if inside_help:
            if "#>" in stripped:
                inside_help = False
            continue
        if stripped.startswith("#"):
            continue
        body.append(line)
    offenders = [
        line.strip()
        for line in body
        if "Get-ChildItem" in line and "-Recurse" in line
    ]
    assert not offenders, offenders
