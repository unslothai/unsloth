# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The temp scan walks one directory at a time, so an unreadable or vanished dir costs only itself."""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess

import pytest
from unsloth_pwsh_runner import run_pwsh

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "Watch-ForCompiler.ps1"

PWSH = shutil.which("pwsh")
pytestmark = pytest.mark.skipif(PWSH is None, reason = "needs PowerShell")


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# PowerShell's error formatter gutters wrapped continuation lines with "   | ".
_GUTTER = re.compile(r"^\s*\|\s?")


def _says(proc: subprocess.CompletedProcess, phrase: str) -> bool:
    """Matches a phrase in PowerShell output after stripping ANSI colour codes, gutter bars and wrapping."""
    text = _ANSI.sub("", proc.stdout + proc.stderr)
    text = "\n".join(_GUTTER.sub("", line) for line in text.splitlines())
    return " ".join(phrase.split()) in " ".join(text.split())


def _run_pwsh(body: str) -> subprocess.CompletedProcess:
    script = f"$ErrorActionPreference = 'Stop'\n. '{SCRIPT}'\n{body}"
    return run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        timeout = 300,
    )


def _scan(root: pathlib.Path, patterns: str = "'*.dll','*.cmdline'") -> dict[str, list[str]]:
    """Runs Get-StudioTempSubtree under CI's preference, returning files and unread directories."""
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
        "There is deliberately NO Windows equivalent of this row: see the note below on what "
        "Windows does and does not cover here."
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

    # Control first: a readable tree would let a walk that stopped early pass.
    assert any(name.endswith("probe.dll") for name in found), found
    assert any(
        name.endswith("after.cmdline") for name in found
    ), "the walk stopped at the unreadable directory instead of stepping over it"
    assert not any(name.endswith("hidden.dll") for name in found), found


# Rows needing POSIX denial skip on Windows, where no ACL control reproduces the race.


def test_the_scan_refuses_to_report_a_truncated_snapshot(tmp_path: pathlib.Path) -> None:
    """Past the ceiling the scan must raise: a partial listing would read as a complete snapshot."""
    # The real ceiling (200000) is too large to build, so the function is redefined from the
    # shipped source with a smaller limit.
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
    proc = run_pwsh(
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
    assert _says(
        proc, "cannot say whether a compiler ran"
    ), f"the walk raised, but not with the message that explains why:\n{proc.stdout}\n{proc.stderr}"


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
    proc = run_pwsh(
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
    """Fails if -Recurse returns to the script: the recursive listing threw when TEMP changed under it."""
    text = SCRIPT.read_text(encoding = "utf-8")
    # Strip comments and help blocks first: their prose names the removed shape.
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
    """Lifts the $left assignment from the shipped script, so the test cannot pass against a stale copy."""
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
    """Unread directories are withheld from new artifacts, or a baseline hole makes old files look new."""
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
    """Control: with no unread directories the same comparison must still report genuinely new files."""
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
        # The root itself counts: it can fail to enumerate and is what the watcher attaches to.
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
    """Lifts the uncovered-directory loop and its throw from the shipped script, not a copy."""
    body = _measured_action_body()
    loop_start = body.index("    $uncovered = @()")
    loop_end = body.index("    $left = @(", loop_start)
    raise_start = body.index("    $incomplete = @()")
    raise_end = body.index("This run cannot say whether a compiler ran.", raise_start)
    raise_end = body.index("\n    }\n", raise_end) + len("\n    }\n")
    return body[loop_start:loop_end] + body[raise_start:raise_end]


def test_an_unreadable_root_with_a_watcher_on_it_does_not_void_the_run() -> None:
    """An unreadable root that has a watcher on it still counts as covered, so the run is not voided."""
    proc = _run_pwsh(
        r"$unread = @('C:\t')" + "\n"
        r"$watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' ([string[]]@('C:\t'), [StringComparer]::OrdinalIgnoreCase)"
        + "\n"
        "$watchFailedRoots = @()\n" + _shipped_coverage_check() + "Write-Output 'SURVIVED'\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SURVIVED" in proc.stdout, (
        "an unreadable temp root voided the measurement even though the watcher was attached "
        f"to that very root:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_unreadable_root_with_no_watcher_still_voids_the_run() -> None:
    """Control: an unreadable root with no watcher must still void the run, or the fix would hide a hole."""
    proc = _run_pwsh(
        r"$unread = @('C:\t')" + "\n"
        "$watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' "
        "([string[]]@(), [StringComparer]::OrdinalIgnoreCase)\n"
        "$watchFailedRoots = @()\n" + _shipped_coverage_check() + "Write-Output 'SURVIVED'\n"
    )
    assert "SURVIVED" not in proc.stdout, (
        "an unread directory with no watcher on its root was treated as a complete "
        f"measurement:\n{proc.stdout}"
    )
    assert _says(
        proc, "cannot say whether a compiler ran"
    ), f"the run was voided without saying why:\n{proc.stdout}\n{proc.stderr}"


def test_the_error_subscription_exists_and_condemns_its_root() -> None:
    """A watcher that overflowed must not count as covering its root, or a missed compile reads clean."""
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
        r"$watchFailedRoots = @('C:\bad')"
        + "\n"
        + expression
        + "\nforeach ($r in $watchedRoots) { Write-Output $r }\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    roots = sorted(line.strip() for line in proc.stdout.splitlines() if line.strip())
    assert roots == [
        r"C:\good"
    ], f"a root whose watcher raised Error was still counted as covering it: {roots}"


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
        text.index("function Get-StudioTempSubtree") : text.index(
            "function Get-StudioTempArtifacts"
        )
    ]
    # Comments stripped first: prose naming Test-Path is not a call.
    code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))
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
    """A missing directory must not be recorded as unread, or routine temp deletions void every run."""
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


def _measured_action_body() -> str:
    """Invoke-WithCompilerWatch, as shipped."""
    text = SCRIPT.read_text(encoding = "utf-8")
    return text[text.index("function Invoke-WithCompilerWatch") :]


def test_the_action_failure_is_persisted_and_rethrown_before_the_scan_is_rejected() -> None:
    """The action's failure is written and rethrown before the coverage void, so the real error survives."""
    body = _measured_action_body()
    write_error = body.index('"$stem-error.txt"')
    rethrow = body.index("if ($failure) { throw $failure }")
    void = body.index("$uncovered.Count -gt 0")

    assert write_error < rethrow, (
        "the action's failure is rethrown before it is written to disk, so the evidence file "
        "this function promises is never produced"
    )
    assert rethrow < void, (
        "the incomplete-scan throw runs before the action's own failure is rethrown. An "
        "installer that genuinely died is then reported as a scanner problem."
    )


def test_the_coverage_check_no_longer_throws_from_inside_the_loop() -> None:
    """The coverage check collects in the loop and throws only after evidence is written."""
    body = _measured_action_body()
    loop_start = body.index("foreach ($dir in $unread) {")
    loop_end = body.index("$left = @(", loop_start)
    loop = body[loop_start:loop_end]
    assert "throw" not in loop, (
        "the coverage loop raises directly again, which puts it ahead of the evidence write "
        f"and the action's own failure:\n{loop}"
    )
    assert "$uncovered += $dir" in loop, "the uncovered directories are no longer collected"


def test_an_overflowed_watcher_voids_the_measurement_on_its_own() -> None:
    """An overflowed watcher voids the measurement alone, since a compile can appear only as live events."""
    proc = _run_pwsh(
        "$unread = @()\n"
        "$watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' "
        "([string[]]@(), [StringComparer]::OrdinalIgnoreCase)\n"
        r"$watchFailedRoots = @('C:\t')"
        + "\n"
        + _shipped_coverage_check()
        + "Write-Output 'SURVIVED'\n"
    )
    assert "SURVIVED" not in proc.stdout, (
        "a watcher that raised was accepted as a complete measurement because no directory "
        f"happened to be unreadable:\n{proc.stdout}"
    )
    assert _says(
        proc, "may have been dropped"
    ), f"the run was voided without naming the dropped events:\n{proc.stdout}\n{proc.stderr}"


def test_a_healthy_watcher_and_a_readable_sweep_still_pass() -> None:
    """The control: voiding on watcher health must not void every ordinary run."""
    proc = _run_pwsh(
        "$unread = @()\n"
        "$watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' "
        r"([string[]]@('C:\t'), [StringComparer]::OrdinalIgnoreCase)" + "\n"
        "$watchFailedRoots = @()\n" + _shipped_coverage_check() + "Write-Output 'SURVIVED'\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        "SURVIVED" in proc.stdout
    ), f"a clean measurement was voided, which would fail every run:\n{proc.stdout}\n{proc.stderr}"


def test_both_incompleteness_reasons_are_reported_together() -> None:
    """One throw naming everything wrong, rather than whichever check happened to run first."""
    proc = _run_pwsh(
        r"$unread = @('C:\gap')" + "\n"
        "$watchedRoots = New-Object 'System.Collections.Generic.HashSet[string]' "
        "([string[]]@(), [StringComparer]::OrdinalIgnoreCase)\n"
        r"$watchFailedRoots = @('C:\t')"
        + "\n"
        + _shipped_coverage_check()
        + "Write-Output 'SURVIVED'\n"
    )
    assert "SURVIVED" not in proc.stdout, proc.stdout
    assert _says(proc, "may have been dropped"), proc.stdout + proc.stderr
    assert _says(proc, "could not read"), (
        "only one of the two reasons reached the caller, so fixing that one would leave the "
        f"run failing again for a reason never reported:\n{proc.stdout}\n{proc.stderr}"
    )
