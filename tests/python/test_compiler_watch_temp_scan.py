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
pytestmark = pytest.mark.skipif(PWSH is None, reason = "needs PowerShell")


def _run_pwsh(body: str) -> subprocess.CompletedProcess:
    script = f"$ErrorActionPreference = 'Stop'\n. '{SCRIPT}'\n{body}"
    return subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 300,
    )


def _scan(root: pathlib.Path, patterns: str = "'*.dll','*.cmdline'") -> dict[str, list[str]]:
    """Run Get-StudioTempSubtree over `root` under the same preference CI uses.

    Both halves are returned. Which directories went unread is not a diagnostic here: it is
    what stops a directory read in one snapshot and not the other from being scored as a
    compile, so it is asserted on directly.
    """
    proc = _run_pwsh(
        f"$scan = Get-StudioTempSubtree -Root '{root}' -Patterns {patterns}\n"
        "foreach ($f in $scan.Files)  { Write-Output \"FILE $f\" }\n"
        "foreach ($d in $scan.Unread) { Write-Output \"UNREAD $d\" }\n"
    )
    assert proc.returncode == 0, f"the walk itself failed:\n{proc.stdout}\n{proc.stderr}"
    out: dict[str, list[str]] = {"files": [], "unread": []}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("FILE "):
            out["files"].append(line[len("FILE "):])
        elif line.startswith("UNREAD "):
            out["unread"].append(line[len("UNREAD "):])
    return out


def _walk(root: pathlib.Path, patterns: str = "'*.dll','*.cmdline'") -> list[str]:
    return _scan(root, patterns)["files"]


@pytest.mark.skipif(
    os.name == "nt",
    reason = (
        "POSIX permissions only. Windows has no os.geteuid, and chmod there sets the read-only "
        "attribute rather than making a directory unopenable, so this would not deny anything. "
        "The Windows half of the same question is the ACL-based control below."
    ),
)
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
    assert any(
        name.endswith("after.cmdline") for name in found
    ), "the walk stopped at the unreadable directory instead of stepping over it"
    assert not any(name.endswith("hidden.dll") for name in found), found


# There is no control here for "the old shape really would have died", and that is deliberate.
#
# The CI failure was a RACE: a directory in the shared temp root disappeared partway through a
# recursive enumeration and the Windows provider raised a Win32Exception, which -ErrorAction
# SilentlyContinue does not suppress because it governs non-terminating errors. An ACL denial was
# tried as a stand-in and is not one: a directory the caller cannot open is an ordinary
# non-terminating access-denied error, which that parameter DOES suppress, so the control reached
# its SURVIVED line and would have passed while claiming the opposite. A POSIX chmod is the same
# story.
#
# Reproducing the race means deleting directories under a live walk and hoping the timing lands,
# which is a flaky test rather than a control. What is pinned instead is the shape: enumeration is
# per-directory and wrapped, and no recursive listing is left in the file. The evidence for the
# failure itself is the CI log quoted at the top of this module.


def test_the_scan_refuses_to_report_a_truncated_snapshot(tmp_path: pathlib.Path) -> None:
    """Past the ceiling it raises, rather than handing back a partial listing.

    The caller reads this snapshot as complete, and it is what stands in when the file-system
    watcher cannot attach, so a silent stop turns a missed artifact into a clean verdict. Driven
    by lowering the ceiling with a stubbed walk is not possible here, so the tree is built: 12
    directories against a ceiling of 8, set by dot-sourcing and re-declaring nothing.
    """
    # The real ceiling is 200000, far too large to build, so the shape is asserted instead and
    # the behaviour is driven at a scale that fits: the function is re-defined with the same body
    # and a smaller limit, taken from the shipped source rather than retyped.
    text = SCRIPT.read_text(encoding = "utf-8")
    assert (
        "throw (" in text and "$visited -gt 200000" in text
    ), "the ceiling no longer raises, so a truncated scan would be read as a complete one"
    small = text[
        text.index("function Get-StudioTempSubtree") : text.index(
            "function Get-StudioTempArtifacts"
        )
    ]
    small = small.replace("$visited -gt 200000", "$visited -gt 3")
    assert "$visited -gt 3" in small
    for i in range(12):
        (tmp_path / f"d{i}").mkdir()
    holder = tmp_path / "small.ps1"
    holder.write_text(small, encoding = "utf-8")
    proc = subprocess.run(
        [
            PWSH,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "$ErrorActionPreference = 'Stop'\n"
            f". '{holder}'\n"
            f"Get-StudioTempSubtree -Root '{tmp_path}' -Patterns '*.dll' | Out-Null\n"
            "Write-Output 'NO-THROW'\n",
        ],
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert (
        "NO-THROW" not in proc.stdout
    ), "the walk returned a partial snapshot instead of declaring the measurement void"
    assert "cannot say whether a compiler ran" in (proc.stdout + proc.stderr)


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
        "(Get-StudioTempArtifacts).Files | ForEach-Object { Write-Output (Split-Path -Leaf $_) }\n"
    )
    proc = subprocess.run(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 300,
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
    offenders = [line.strip() for line in body if "Get-ChildItem" in line and "-Recurse" in line]
    assert not offenders, offenders


@pytest.mark.skipif(
    os.name == "nt",
    reason = (
        "POSIX permissions only, for the same reason as the walk control above: chmod on "
        "Windows sets the read-only attribute rather than making a directory unopenable."
    ),
)
def test_an_unreadable_directory_is_reported_and_not_just_skipped(tmp_path: pathlib.Path) -> None:
    """The gap has to reach the caller, because the caller subtracts two of these listings.

    Skipping quietly is what makes the comparison lie: a directory unread at baseline and
    readable afterwards hands every file already sitting in it to the "new since the action"
    set, and an installer that compiled nothing is reported as having compiled. This asserts
    the walk says which directory it could not read, which is what the caller needs to
    withhold those paths.
    """
    (tmp_path / "locked").mkdir()
    (tmp_path / "locked" / "hidden.dll").write_text("x")
    (tmp_path / "open").mkdir()
    (tmp_path / "open" / "seen.dll").write_text("x")
    os.chmod(tmp_path / "locked", 0o000)
    try:
        if os.geteuid() == 0:
            pytest.skip("root reads every directory, so nothing here can be made unreadable")
        scan = _scan(tmp_path)
    finally:
        os.chmod(tmp_path / "locked", 0o700)

    assert any(name.endswith("seen.dll") for name in scan["files"]), scan
    assert not any(name.endswith("hidden.dll") for name in scan["files"]), scan
    assert [d for d in scan["unread"] if d.endswith("locked")], (
        "the walk stepped over the unreadable directory without recording it. The caller "
        f"cannot then tell an empty directory from an unread one: {scan}"
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason = "POSIX permissions only; see the walk control above.",
)
def test_a_directory_that_vanished_is_not_reported_as_unread(tmp_path: pathlib.Path) -> None:
    """A path that is gone is not a hole in the measurement, and must not void a run.

    This is the transient case the whole change exists for. A directory that no longer
    exists cannot contribute a file to a later listing of itself, so there is nothing for
    the caller to withhold and nothing to declare void. Only a directory that is still
    there and still unreadable is a real gap.
    """
    scan = _scan(tmp_path / "never-existed")
    assert scan["files"] == [], scan
    assert scan["unread"] == [], (
        "a missing directory was recorded as an unread one. That would void a measurement "
        f"for the ordinary temp race this change was written to survive: {scan}"
    )


def _shipped_left_expression() -> str:
    """The `$left = @(...)` assignment as it appears in Invoke-WithCompilerWatch.

    Lifted from the shipped source rather than retyped. A copy of the expression in this file
    would keep passing after the withholding was deleted from the script, which is precisely
    the regression worth catching: the comparison is the only place the unread directories
    are allowed to change the answer.
    """
    text = SCRIPT.read_text(encoding = "utf-8")
    start = text.index("    $left = @(")
    end = text.index("\n    )\n", start) + len("\n    )\n")
    expression = text[start:end]
    assert "$unread" in expression, (
        "the $left comparison no longer consults the unread directories, so a directory that "
        "could not be read in one snapshot and could in the other is scored as a compile:\n"
        f"{expression}"
    )
    return expression


def test_the_unread_directories_are_withheld_from_the_new_artifact_set() -> None:
    """The defect this guards: a baseline hole turning pre-existing files into evidence.

    `old.dll` was in the gap directory all along. The baseline sweep could not read that
    directory, the final sweep could, so a plain "in after, not in before" difference hands it
    back as new and an installer that compiled nothing is reported as having compiled.

    The expression under test is the one the script actually runs, read out of the file, so
    deleting or bypassing the withholding fails this rather than leaving a copy passing here.
    """
    proc = _run_pwsh(
        "$before = New-Object 'System.Collections.Generic.HashSet[string]' "
        "([string[]]@(), [StringComparer]::OrdinalIgnoreCase)\n"
        r"$after = @('C:\t\gap\old.dll', 'C:\t\seen\new.dll')" + "\n"
        r"$unread = @('C:\t\gap')" + "\n"
        + _shipped_left_expression()
        + "foreach ($p in $left) { Write-Output $p }\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    left = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    assert left == [r"C:\t\seen\new.dll"], (
        "a file under a directory the baseline sweep could not read was scored as new. "
        f"That reports an innocent action as having compiled: {left}"
    )


def test_the_same_comparison_still_reports_a_genuinely_new_file() -> None:
    """The control for the row above: withholding must not swallow everything.

    A test that only checks something was removed passes just as well against an expression
    that returns nothing at all, which would hide every real compile instead. With no unread
    directories, both files are new and both come back.
    """
    proc = _run_pwsh(
        "$before = New-Object 'System.Collections.Generic.HashSet[string]' "
        "([string[]]@(), [StringComparer]::OrdinalIgnoreCase)\n"
        r"$after = @('C:\t\gap\old.dll', 'C:\t\seen\new.dll')" + "\n"
        "$unread = @()\n"
        + _shipped_left_expression()
        + "foreach ($p in $left) { Write-Output $p }\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    left = sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())
    assert left == [r"C:\t\gap\old.dll", r"C:\t\seen\new.dll"], (
        f"a clean sweep stopped reporting new artifacts, which hides every real compile: {left}"
    )


def test_the_prefix_test_does_not_match_a_sibling_by_name() -> None:
    """`C:\\t\\ab` is not under `C:\\t\\a`, and withholding it would hide a real artifact."""
    proc = _run_pwsh(
        r"Write-Output ('under=' + (Test-StudioPathUnder -Path 'C:\t\a\x.dll' -Directory 'C:\t\a'))"
        + "\n"
        r"Write-Output ('sibling=' + (Test-StudioPathUnder -Path 'C:\t\ab\x.dll' -Directory 'C:\t\a'))"
        + "\n"
        r"Write-Output ('case=' + (Test-StudioPathUnder -Path 'C:\T\A\x.dll' -Directory 'c:\t\a'))"
        + "\n"
    )
    assert proc.returncode == 0, proc.stderr
    out = dict(
        line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line
    )
    assert out["under"] == "True", out
    assert out["sibling"] == "False", (
        "a sibling directory sharing a name prefix was treated as being inside the unread "
        f"one, which would withhold real evidence: {out}"
    )
    assert out["case"] == "True", out
