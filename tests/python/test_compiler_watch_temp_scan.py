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
        'foreach ($f in $scan.Files)  { Write-Output "FILE $f" }\n'
        'foreach ($d in $scan.Unread) { Write-Output "UNREAD $d" }\n'
    )
    assert proc.returncode == 0, f"the walk itself failed:\n{proc.stdout}\n{proc.stderr}"
    out: dict[str, list[str]] = {"files": [], "unread": []}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("FILE "):
            out["files"].append(line[len("FILE ") :])
        elif line.startswith("UNREAD "):
            out["unread"].append(line[len("UNREAD ") :])
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
        r"$unread = @('C:\t\gap')"
        + "\n"
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
    assert left == [
        r"C:\t\gap\old.dll",
        r"C:\t\seen\new.dll",
    ], f"a clean sweep stopped reporting new artifacts, which hides every real compile: {left}"


def test_the_prefix_test_does_not_match_a_sibling_by_name() -> None:
    """`C:\\t\\ab` is not under `C:\\t\\a`, and withholding it would hide a real artifact."""
    proc = _run_pwsh(
        r"Write-Output ('under=' + (Test-StudioPathUnder -Path 'C:\t\a\x.dll' -Directory 'C:\t\a'))"
        + "\n"
        r"Write-Output ('sibling=' + (Test-StudioPathUnder -Path 'C:\t\ab\x.dll' -Directory 'C:\t\a'))"
        + "\n"
        r"Write-Output ('case=' + (Test-StudioPathUnder -Path 'C:\T\A\x.dll' -Directory 'c:\t\a'))"
        + "\n"
        # The directory itself. A temp ROOT can be what failed to enumerate, and the root is
        # also what the watcher attaches to, so descendants-only makes the coverage check
        # below declare a watched root uncovered and throw.
        r"Write-Output ('self=' + (Test-StudioPathUnder -Path 'C:\t\a' -Directory 'C:\t\a'))" + "\n"
        r"Write-Output ('selfslash=' + (Test-StudioPathUnder -Path 'C:\t\a\' -Directory 'C:\t\a'))"
        + "\n"
    )
    assert proc.returncode == 0, proc.stderr
    out = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    assert out["under"] == "True", out
    assert out["sibling"] == "False", (
        "a sibling directory sharing a name prefix was treated as being inside the unread "
        f"one, which would withhold real evidence: {out}"
    )
    assert out["case"] == "True", out
    assert out["self"] == "True" and out["selfslash"] == "True", (
        "the directory itself did not count as covered. An unreadable temp ROOT is then "
        f"declared to have no watcher on it and the run throws for nothing: {out}"
    )


def _shipped_coverage_check() -> str:
    """The `if ($unread.Count -gt 0) { ... }` block from Invoke-WithCompilerWatch.

    Read from the shipped file for the same reason as the `$left` expression: a copy here
    would keep passing after the real check changed.
    """
    text = SCRIPT.read_text(encoding = "utf-8")
    start = text.index("    if ($unread.Count -gt 0) {")
    end = text.index("\n    }\n", start) + len("\n    }\n")
    block = text[start:end]
    assert "Test-StudioPathUnder" in block and "throw" in block, block
    return block


def test_an_unreadable_root_with_a_watcher_on_it_does_not_void_the_run() -> None:
    """The case that would reintroduce the failure this change exists to contain.

    A temp ROOT can be the directory that could not be enumerated, and the root is also
    exactly what Start-StudioTempWatch attaches to. If the coverage check only recognises
    descendants of a watched root, an unreadable root is declared to have no watcher, the run
    throws, and the job goes red again for a transient condition in somebody else's TEMP.
    """
    proc = _run_pwsh(
        r"$unread = @('C:\t')" + "\n"
        r"$watch = @([pscustomobject]@{ Root = 'C:\t' })" + "\n"
        "$watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' "
        "([string[]]@($watch | ForEach-Object { $_.Root }), [StringComparer]::OrdinalIgnoreCase)\n"
        + _shipped_coverage_check()
        + "Write-Output 'SURVIVED'\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SURVIVED" in proc.stdout, (
        "an unreadable temp root voided the measurement even though the watcher was attached "
        f"to that very root:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_unreadable_root_with_no_watcher_still_voids_the_run() -> None:
    """The control: the throw has to survive, or the fix above would gut the guard.

    With nothing watching, the listing is the only evidence there is, and withholding part of
    it would hand back a hole as a clean result.
    """
    proc = _run_pwsh(
        r"$unread = @('C:\t')" + "\n"
        "$watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' "
        "([string[]]@(), [StringComparer]::OrdinalIgnoreCase)\n"
        + _shipped_coverage_check()
        + "Write-Output 'SURVIVED'\n"
    )
    assert "SURVIVED" not in proc.stdout, (
        "an unread directory with no watcher on its root was treated as a complete "
        f"measurement:\n{proc.stdout}"
    )
    assert "cannot say whether a compiler ran" in (proc.stdout + proc.stderr), (
        f"the run was voided without saying why:\n{proc.stdout}\n{proc.stderr}"
    )


def test_the_error_subscription_exists_and_condemns_its_root() -> None:
    """A watcher that dropped events must not count as covering its root.

    FileSystemWatcher raises Error on buffer overflow and drops the creations it could not
    queue. The handle stays in the list looking exactly like a working one, so without this the
    coverage check treats the root as watched, the unread directories under it are withheld,
    and an artifact missing from BOTH the live stream and the listing reports as a clean run.
    That is the only combination that turns a real compile into a pass.

    The wiring is asserted on the shipped source rather than by forcing a real overflow, which
    needs a Windows host and thousands of creations to land reliably.
    """
    text = SCRIPT.read_text(encoding = "utf-8")
    assert "-EventName Error" in text, (
        "no Error subscription on the watcher, so an overflow is not observable at all and the "
        "root keeps counting as covered"
    )
    assert "FailedRoots" in text, "the drained errors never reach the caller"
    assert "$watchFailedRoots -notcontains $_" in text, (
        "the coverage set no longer excludes roots whose watcher failed, so an overflowed "
        "watcher still counts as coverage"
    )


def test_a_failed_watcher_root_is_dropped_from_the_coverage_set() -> None:
    """Driven: the filter that builds $watchedRoots, read out of the shipped file."""
    text = SCRIPT.read_text(encoding = "utf-8")
    start = text.index("    $watchedRoots = New-Object")
    end = text.index("OrdinalIgnoreCase)", start) + len("OrdinalIgnoreCase)")
    expression = text[start:end]
    proc = _run_pwsh(
        r"$watch = @([pscustomobject]@{ Root = 'C:\good' }, [pscustomobject]@{ Root = 'C:\bad' })"
        + "\n"
        r"$watchFailedRoots = @('C:\bad')" + "\n"
        + expression
        + "\nforeach ($r in $watchedRoots) { Write-Output $r }\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    roots = sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())
    assert roots == [r"C:\good"], (
        f"a root whose watcher raised Error was still counted as covering it: {roots}"
    )


@pytest.mark.skipif(os.name == "nt", reason = "POSIX permissions only; see the note above.")
def test_an_inaccessible_directory_is_recorded_not_mistaken_for_a_deleted_one(
    tmp_path: pathlib.Path,
) -> None:
    """An ACL denial must not read as "the directory is gone".

    This is why the walk classifies the ERROR instead of probing the path. Measured under pwsh
    with $ErrorActionPreference = 'Stop':

        missing directory  ->  ItemNotFoundException, Test-Path returns $false
        denied directory   ->  Test-Path THROWS "Access to the path ... is denied"

    So a Test-Path probe both calls an unreadable directory deleted, dropping it silently out
    of the comparison, and can raise from inside the catch that was meant to contain the
    failure. Here the PARENT is denied, so the child cannot be enumerated or probed: the walk
    must still report the child's parent as unread rather than skipping it.
    """
    outer = tmp_path / "denied"
    outer.mkdir()
    (outer / "inner").mkdir()
    (outer / "inner" / "hidden.dll").write_text("x")
    os.chmod(outer, 0o000)
    try:
        if os.geteuid() == 0:
            pytest.skip("root reads every directory, so nothing here can be made unreadable")
        scan = _scan(tmp_path)
    finally:
        os.chmod(outer, 0o700)

    assert [d for d in scan["unread"] if d.endswith("denied")], (
        "an unreadable directory was treated as deleted and dropped. Pre-existing files under "
        f"it then read as new in the other snapshot, or a real artifact is hidden: {scan}"
    )


def test_the_walk_classifies_the_error_and_never_probes_with_test_path() -> None:
    """The shape that makes the row above possible, pinned so it cannot regress quietly."""
    text = SCRIPT.read_text(encoding = "utf-8")
    body = text[
        text.index("function Get-StudioTempSubtree") : text.index("function Get-StudioTempArtifacts")
    ]
    # Comments stripped first, like the -Recurse guard above: this function EXPLAINS why it
    # does not probe, and the prose naming Test-Path is not a call to it.
    code = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("#")
    )
    assert "Test-Path" not in code, (
        "the walk probes with Test-Path again. That throws on an ACL-denied directory and "
        "answers False for one that merely cannot be read, so it cannot decide deleted "
        "versus unreadable."
    )
    assert body.count("Test-StudioPathIsGone") >= 2, (
        "both the first failure and the retry must classify the error; otherwise a directory "
        "deleted inside the retry window is recorded as unread and can void the run"
    )


def test_a_missing_directory_is_still_not_recorded_as_unread(tmp_path: pathlib.Path) -> None:
    """The control for the row above: fail-safe must not become fail-always.

    Test-StudioPathIsGone returning $false for everything would make every temp deletion a
    gap, and an unread directory under an unwatched root voids the run. That would fail the
    job for exactly the race this change exists to tolerate.
    """
    scan = _scan(tmp_path / "never-existed")
    assert scan["files"] == [] and scan["unread"] == [], (
        f"a missing directory was recorded as a gap, which voids runs for ordinary temp "
        f"deletions: {scan}"
    )


def test_the_classifier_answers_both_cases() -> None:
    """Driven against the real helper: missing is gone, denied is not."""
    proc = _run_pwsh(
        "$missing = $null\n"
        "try { Get-ChildItem -LiteralPath '/nonexistent-xyz-123' -Force -ErrorAction Stop | Out-Null }\n"
        "catch { $missing = $_ }\n"
        "Write-Output ('missing=' + (Test-StudioPathIsGone -ErrorRecord $missing))\n"
        "$denied = New-Object System.Management.Automation.ErrorRecord ("
        "(New-Object System.UnauthorizedAccessException 'denied'), 'x', 'PermissionDenied', $null)\n"
        "Write-Output ('denied=' + (Test-StudioPathIsGone -ErrorRecord $denied))\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    assert out["missing"] == "True", out
    assert out["denied"] == "False", (
        "an access-denied error was classified as a deleted directory, so the directory is "
        f"dropped out of the comparison instead of recorded as a gap: {out}"
    )
