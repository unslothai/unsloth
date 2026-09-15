# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The AMSI probe must keep its positive control, and must never actually run what it submits.

The probe answers the question #10805 is about: will a live AMSI provider let install.ps1 compile?
Its entire value rests on the control. A runner where no provider is live returns "not blocked" for
every input, including Microsoft's own test sample, and that outcome is byte-identical to a clean
verdict. release-desktop.yml's comments record three releases that shipped unscanned while their scan
step was green, for exactly this reason.

The second property is that the probe compiles and never invokes. It is handed the real installer, so
an `Invoke-Expression` or a `.Invoke()` creeping in would run a full install inside a measurement
step -- on the base side, from a commit nobody reviewed for that purpose.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest
import yaml

# The shared runner, not a direct subprocess call: a direct one shares a single
# $XDG_CACHE_HOME/powershell startup cache with every other xdist worker, and an interpreter that
# dies at startup then renders as this test failing rather than as the crash it was.
# tests/studio/test_pwsh_calls_use_the_shared_runner.py enforces this.
from unsloth_pwsh_runner import run_pwsh


REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / ".github" / "scripts" / "Probe-AmsiParse.ps1"
WORKFLOW = REPO / ".github" / "workflows" / "windows-amsi-defender-differential-ci.yml"


def _probe_text() -> str:
    return PROBE.read_text(encoding = "utf-8")


def test_the_probe_exists() -> None:
    assert PROBE.is_file(), f"missing {PROBE.relative_to(REPO)}"


def test_the_probe_parses() -> None:
    """It is invoked by a workflow under Windows PowerShell 5.1, where a syntax error would surface
    as "the probe wrote no result" and be indistinguishable from an absent scanner."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")
    probe = (
        "$errors = $null; $tokens = $null; "
        "$null = [System.Management.Automation.Language.Parser]::ParseFile("
        "$env:UNSLOTH_TARGET, [ref]$tokens, [ref]$errors); "
        "if ($errors.Count) { $errors | ForEach-Object { $_.Message }; exit 1 }"
    )
    import os

    result = run_pwsh(
        [pwsh, "-NoProfile", "-NonInteractive", "-Command", probe],
        capture_output = True,
        text = True,
        timeout = 120,
        env = {**os.environ, "UNSLOTH_TARGET": str(PROBE)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_probe_compiles_but_never_invokes() -> None:
    """It is handed the real installer. Running it would install, from a commit nobody reviewed."""
    text = _probe_text()
    code = "\n".join(
        line
        for line in text.splitlines()
        if not line.strip().startswith("#") and not line.strip().startswith(".")
    )
    for banned in (
        "Invoke-Expression",
        ".Invoke()",
        "InvokeReturnAsIs",
        "Start-Process",
        "& $sb",
        "iex ",
    ):
        assert banned not in code, (
            f"the probe contains {banned!r}. It must submit text to the compiler and stop there: "
            f"it is handed install.ps1, so invoking would run a full install inside a measurement."
        )
    assert "[scriptblock]::Create" in code, (
        "the probe no longer submits anything to the compiler, which is the only way to ask AMSI "
        "the question. Compilation is the AMSI trigger; execution is not needed and is not wanted."
    )


def test_the_positive_control_is_present_and_split() -> None:
    """Split so this repository is not itself a sample carrying the signature.

    A single literal here would mean every clone, every source tarball and every scan of this repo
    contains the AMSI test signature, which is the same self-inflicted detection class as the two
    fixture archives that made Panda flag the GitHub zip.
    """
    text = _probe_text()
    assert "$Control" in text, "the probe lost its -Control switch"
    assert "7e72c3ce" in text, (
        "the AMSI test sample is gone. Without it a runner with no live provider returns 'not "
        "blocked' for everything, which is indistinguishable from a clean verdict."
    )
    assert "7e72c3ce-861b-4339-8740-0ac1484c1386" not in text, (
        "the AMSI test sample appears as one literal. Assemble it from fragments, the way "
        "Microsoft's own documentation does, so this repository is not a sample of it."
    )


def test_a_block_is_matched_on_the_error_id_and_not_on_message_text() -> None:
    """Message text is localised. install.rs already keys on the id, and this follows it."""
    text = _probe_text()
    assert "ScriptContainedMaliciousContent" in text
    assert "ScriptHasAdminBlockedContent" in text, (
        "an administrative policy block is a different cause with a different fix from a signature "
        "match, and collapsing them loses the only distinction that tells a user what to do"
    )
    assert "FullyQualifiedErrorId" in text


def test_the_probe_never_exits_non_zero_on_a_detection() -> None:
    """Whether a detection is a failure depends on the controls and on which side it appeared on,
    and the caller holds both. A probe that exits non-zero on a detection cannot establish a
    baseline, which is the first thing this lane needs to do."""
    text = _probe_text()
    assert re.search(r"^exit 0\s*$", text, re.M), "the probe must end by exiting zero"
    assert "exit 1" not in text, (
        "the probe decides a verdict. It must report and let the workflow decide, because the "
        "workflow is the only place that knows whether the control fired."
    )


# ---------------------------------------------------------------------------
# The workflow around it
# ---------------------------------------------------------------------------


def test_the_workflow_exists_and_is_valid_yaml() -> None:
    assert WORKFLOW.is_file()
    data = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    assert list(data["jobs"]) == ["measure"]


def test_the_lane_is_never_a_required_check_by_schedule_alone() -> None:
    """Definitions change daily and outside our control. A required check that a Microsoft
    definition push can turn red is a release-blocking hazard, so the scheduled run exists to
    notice a new signature, not to gate a merge."""
    body = WORKFLOW.read_text(encoding = "utf-8")
    assert "schedule:" in body
    assert "required check" in body, (
        "the workflow no longer records that it must not be a required check, which is the one "
        "thing a reader needs to know before wiring it into branch protection"
    )


def test_an_absent_scanner_warns_and_never_reports_clean() -> None:
    body = WORKFLOW.read_text(encoding = "utf-8")
    for needle, why in (
        ("COULD NOT MEASURE", "the unmeasured outcome has to be named, or it reads as clean"),
        ("positive control", "both halves are gated on a control"),
        ("EICAR", "the file-scan half needs its own control"),
        ("did NOT fire", "the control failing has to be reported explicitly"),
    ):
        assert needle in body, f"{why}: {needle!r} is gone"
    # The EICAR literal must be split here too, for the same reason as the AMSI sample.
    assert (
        "X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*" not in body
    ), "EICAR appears as one literal in the workflow; assemble it from fragments"


def test_the_probe_runs_under_windows_powershell_five_one() -> None:
    """pwsh 7 is a different host with a different AMSI integration, so its answer does not
    transfer, and 5.1 is what install.rs spawns and what the #10805 reporter ran."""
    body = WORKFLOW.read_text(encoding = "utf-8")
    assert "WindowsPowerShell\\v1.0\\powershell.exe" in body
    assert (
        "-ExecutionPolicy RemoteSigned" in body
    ), "the lane that exists to remove relaxed execution policies must not use one itself"
    assert "Bypass" not in body.replace(
        "ScriptHasAdminBlockedContent", ""
    ), "the workflow relaxes an execution policy somewhere"


def test_the_probe_reports_on_a_host_without_amsi_rather_than_crashing() -> None:
    """Run for real, here, where no AMSI provider exists.

    This is the whole unmeasured path: the control does not fire, nothing is blocked, and the probe
    still has to produce well-formed JSON saying so. If it crashed instead, the workflow would read
    "the probe wrote no result" and a genuinely absent scanner would be reported as a broken lane.
    """
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.json"
        target = Path(tmp) / "sample.ps1"
        target.write_text("Write-Output 'hello'\n", encoding = "utf-8")
        result = run_pwsh(
            [
                pwsh,
                "-NoProfile",
                "-File",
                str(PROBE),
                "-Control",
                "-Path",
                str(target),
                "-OutFile",
                str(out),
            ],
            capture_output = True,
            text = True,
            timeout = 180,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert out.is_file(), f"the probe wrote no result file:\n{result.stdout}{result.stderr}"
        data = json.loads(out.read_text(encoding = "utf-8"))

    labels = [r["label"] for r in data["results"]]
    assert any(l.startswith("control") for l in labels), "the control is missing from the output"
    assert str(target) in labels or any(str(target) in l for l in labels)
    for row in data["results"]:
        assert row["blocked"] is False, (
            "something was reported blocked on a host with no AMSI provider, which means the "
            "'blocked' flag is being set by something other than a scanner verdict"
        )


def test_the_laid_out_copies_keep_the_bytes_that_ship(tmp_path: Path) -> None:
    """The lane scans the bytes users get, or its verdict does not transfer to them.

    The copies used to be reconstructed rather than copied: `git show` was captured into a
    PowerShell variable, which decodes the blob into lines, and the write back re-encoded them. That
    round trip converts CRLF to LF, collapses every trailing blank line into one, invents a final
    newline where the blob had none, and cannot represent a byte that is not valid UTF-8. The
    scanner was then judging a file this project never serves.

    Driven through the workflow's own extraction snippet against a real blob built to carry all
    three of those properties, rather than by asserting on the text of the snippet, so a future
    rewrite is judged on the bytes it produces.
    """
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")

    body = WORKFLOW.read_text(encoding = "utf-8")
    start = body.index("$psi = New-Object System.Diagnostics.ProcessStartInfo")
    end = body.index("$laid++", start)
    snippet = textwrap.dedent(body[start:end])

    repo = tmp_path / "repo"
    repo.mkdir()
    # CRLF, a byte that is not valid UTF-8, two trailing blank lines and no final newline: each one
    # is separately destroyed by the text round trip.
    blob = b"Write-Host 'one'\r\nWrite-Host 'two \xff'\r\n\r\n\r\nWrite-Host 'three'"
    (repo / "install.ps1").write_bytes(blob)
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    for argv in (
        ["git", "init", "-q"],
        # -text so the checkout cannot be what normalises the line endings: the question here is
        # only what the extraction does with the blob.
        ["git", "config", "core.autocrlf", "false"],
        ["git", "add", "install.ps1"],
        ["git", "commit", "-qm", "b"],
    ):
        subprocess.run(argv, cwd = repo, check = True, env = env)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd = repo, check = True, capture_output = True, text = True
    ).stdout.strip()

    dest = tmp_path / "copy.ps1"
    script = tmp_path / "lay.ps1"
    script.write_text(
        f"$sha = '{sha}'\n$f = 'install.ps1'\n$dest = '{dest.as_posix()}'\n" + snippet,
        encoding = "utf-8",
    )
    done = run_pwsh(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script)],
        cwd = str(repo),
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
    assert dest.read_bytes() == blob, (
        "the laid-out copy is not the committed blob, so the scanner is judging bytes this project "
        "never serves"
    )

    shipped = (REPO / "install.ps1").read_bytes()
    assert b"\r\n" not in shipped, (
        "install.ps1 now contains CRLF, so the assumption this lane is built on no longer holds. "
        "Check .gitattributes before changing the workflow: a committed Authenticode signature over "
        "CRLF-rewritten content reads as HashMismatch, which is worse than unsigned."
    )
    assert shipped[:3] != b"\xef\xbb\xbf", "install.ps1 gained a BOM"


def test_the_mark_of_the_web_is_written_the_documented_way() -> None:
    """`-Stream`, not a stream suffix appended to `-LiteralPath`.

    PowerShell documents exactly one way to address an alternate data stream, and
    `release-desktop.yml:1470` already uses it. The suffix form relies on the path being taken
    "exactly as typed" -- and if it does not bind, the `-ErrorAction SilentlyContinue` next to it
    swallows the error, the mark is never applied, block-at-first-sight never consults the cloud,
    and the step still prints "clean". That is a silent downgrade of the exact condition this lane
    exists to create, which is the worst failure shape available here.
    """
    body = WORKFLOW.read_text(encoding = "utf-8")
    assert "-Stream Zone.Identifier" in body, (
        "the mark-of-the-web stamp no longer uses -Stream, which is the only documented way to "
        "address an alternate data stream"
    )
    # Comment lines stripped before the ban is applied. The comment right above the fixed code
    # explains what the broken form looked like, and naming it is the point of that comment -- a
    # substring search over the raw file matches the explanation and fails. That has now happened
    # three times while writing these guards, which is a good argument for never grepping a file
    # for a string its own prose is obliged to contain.
    code = "\n".join(line for line in body.splitlines() if not line.strip().startswith("#"))
    assert (
        ':Zone.Identifier"' not in code
    ), "the workflow appends a stream name to a path again; use -Stream"
    assert "was scanned WITHOUT mark-of-the-web" in body, (
        "a failed stamp is no longer reported. A scan of a MyComputer-zone file is a weaker test "
        "than this lane claims to run, and a reader has to know which one they got."
    )


def _amsi_step_script() -> str:
    """The body of the step that turns probe results into a verdict, as CI runs it."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    for step in workflow["jobs"]["measure"]["steps"]:
        if step.get("id") == "amsi":
            return step["run"]
    raise AssertionError(
        "the workflow no longer has a step with id 'amsi'. This test drives the REAL verdict logic "
        "by extracting it, so that it cannot pass against a copy that has drifted from CI."
    )


def _verdict_logic() -> str:
    """Everything from the per-row control onwards, with the row collection left to the caller."""
    script = _amsi_step_script()
    marker = "# The control decides whether any of this means anything"
    assert (
        marker in script
    ), "the per-row control block is gone from the workflow, so there is nothing to exercise"
    return script[script.index(marker) :]


def _run_verdict(
    tmp_path: Path,
    rows_ps: str,
    no_result_ps: str = "@()",
) -> tuple[int, str]:
    """Run the extracted verdict logic against synthetic probe rows."""
    pwsh = shutil.which("pwsh")
    if pwsh is None:
        pytest.skip("pwsh is unavailable")
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    script = tmp_path / "verdict.ps1"
    script.write_text(
        f"$env:GITHUB_OUTPUT = '{out.as_posix()}'\n"
        f"$rows = {rows_ps}\n"
        f"$noResult = {no_result_ps}\n" + _verdict_logic(),
        encoding = "utf-8",
    )
    done = run_pwsh(
        [pwsh, "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output = True,
        text = True,
        timeout = 120,
    )
    return done.returncode, done.stdout + done.stderr + out.read_text(encoding = "utf-8")


def _row(side: str, name: str, *, control: bool, result: str) -> str:
    control_ps = (
        f"@{{ label = 'control (AMSI test sample)'; blocked = ${str(control).lower()}; errorId = "
        f"$(if (${str(control).lower()}) {{ 'ScriptContainedMaliciousContent' }} else {{ '' }}) }}"
    )
    return (
        f"[pscustomobject]@{{ side = '{side}'; name = '{name}'; "
        f"data = [pscustomobject]@{{ results = @({control_ps}, {result}) }} }}"
    )


_COMPILED = "@{ label = 'f.ps1'; blocked = $false; errorId = '' }"
_BLOCKED = "@{ label = 'f.ps1'; blocked = $true; errorId = 'ScriptContainedMaliciousContent' }"
_SYNTAX = "@{ label = 'f.ps1'; blocked = $false; errorId = 'ExpectedExpression' }"
_MISSING = "@{ label = 'f.ps1'; blocked = $false; errorId = ''; missing = $true }"
_UNREADABLE = "@{ label = 'f.ps1'; blocked = $false; errorId = ''; unreadable = $true }"


@pytest.mark.parametrize(
    ("label", "head", "head_control", "expect_code", "expect_text"),
    [
        ("both sides compiled", _COMPILED, True, 0, "verdict=clean"),
        ("head refused by a provider", _BLOCKED, True, 1, "AMSI refused head/"),
        ("head did not compile", _SYNTAX, True, 1, "verdict=broken"),
        ("head file missing", _MISSING, True, 0, "verdict=unmeasured"),
        ("head file unreadable", _UNREADABLE, True, 0, "verdict=unmeasured"),
        ("head control silent", _COMPILED, False, 0, "verdict=unmeasured"),
    ],
)
def test_the_head_verdict_is_never_clean_unless_head_was_really_measured(
    tmp_path: Path, label: str, head: str, head_control: bool, expect_code: int, expect_text: str
) -> None:
    """Every way a head measurement can fail must stop short of publishing "clean".

    A lane that reports clean when it did not measure is worse than no lane: it converts an absent
    scanner, an unreadable candidate or a candidate that never compiled into a green check, and this
    repository has already shipped three releases whose scan step was green because nothing scanned.
    The base side is held healthy in every case so that the head side is the only variable.
    """
    rows = (
        f"@({_row('base', 'install.ps1', control = True, result = _COMPILED)}, "
        + _row("head", "install.ps1", control = head_control, result = head)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == expect_code, f"{label}: expected exit {expect_code}, got {code}\n{text}"
    assert expect_text in text, f"{label}: expected {expect_text!r} in output\n{text}"
    if expect_text != "verdict=clean":
        assert "verdict=clean" not in text, f"{label}: published a clean verdict anyway\n{text}"


def test_a_probe_that_wrote_no_result_is_not_silently_dropped(tmp_path: Path) -> None:
    """The row simply vanished before, and a head file that vanishes leaves zero blocked: clean."""
    rows = (
        f"@({_row('base', 'install.ps1', control = True, result = _COMPILED)}, "
        + _row("head", "install.ps1", control = True, result = _COMPILED)
        + ")"
    )
    code, text = _run_verdict(
        tmp_path, rows, no_result_ps = "@('head/setup.ps1 [the probe wrote no result]')"
    )
    assert code == 0
    assert "verdict=unmeasured" in text, text
    assert "verdict=clean" not in text, text


def test_a_control_firing_elsewhere_does_not_vouch_for_this_process(tmp_path: Path) -> None:
    """The exact borrowing the per-row control exists to stop.

    AMSI initialises per process. A live base invocation followed by a head invocation where no
    provider loaded used to set one job-wide flag to true, and every row was then trusted, so the
    head answer -- taken in a process that would have said "not blocked" to anything at all --
    was published as clean.
    """
    rows = (
        f"@({_row('base', 'install.ps1', control = True, result = _BLOCKED)}, "
        + _row("head", "install.ps1", control = False, result = _COMPILED)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == 0
    assert "verdict=unmeasured" in text, text
    assert "verdict=clean" not in text, text


@pytest.mark.parametrize(
    ("label", "base_result", "base_control", "expect"),
    [
        ("base compiled the same file", _COMPILED, True, "This change introduced it"),
        (
            "base was refused for the same file",
            _BLOCKED,
            True,
            "pre-existing rather than introduced",
        ),
        ("base row is not validly measured", _COMPILED, False, "cannot be said from this run"),
        ("base did not compile the same file", _SYNTAX, True, "cannot be said from this run"),
    ],
)
def test_causality_is_decided_per_script_and_only_against_a_valid_base(
    tmp_path: Path, label: str, base_result: str, base_control: bool, expect: str
) -> None:
    """Whether a block is introduced or pre-existing is a statement about ONE script.

    Comparing counts across sides said "pre-existing" whenever the base had any block at all, even
    on a different file, and said "introduced" whenever it had none -- including when the base row
    for that script was never validly measured, where the only honest answer is that this run
    cannot tell. Both mistakes point a reader at the wrong commit.
    """
    rows = (
        "@("
        + _row("base", "install.ps1", control = base_control, result = base_result)
        + ", "
        + _row("head", "install.ps1", control = True, result = _BLOCKED)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == 1, f"{label}: a refused head script must fail the job\n{text}"
    assert expect in text, f"{label}: expected {expect!r}\n{text}"


def test_a_block_on_a_different_base_script_is_not_called_pre_existing(tmp_path: Path) -> None:
    """The count-based version's exact failure: base blocked on setup.ps1, head blocked on
    install.ps1, and the run announced the install.ps1 block as pre-existing."""
    rows = (
        "@("
        + _row("base", "setup.ps1", control = True, result = _BLOCKED)
        + ", "
        + _row("base", "install.ps1", control = True, result = _COMPILED)
        + ", "
        + _row("head", "install.ps1", control = True, result = _BLOCKED)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == 1
    assert "This change introduced it" in text, text
    assert "pre-existing" not in text.split("install.ps1")[-1], text


def test_a_real_block_is_reported_even_when_another_row_is_unmeasured(tmp_path: Path) -> None:
    """Incomplete coverage must not swallow a detection that was actually made.

    Returning early on any unmeasured head row meant a script a live provider genuinely REFUSED
    went unreported whenever some other invocation happened to write no result, and the job stayed
    green. A gap in coverage is a warning; a refusal is the finding this lane exists for.
    """
    rows = (
        "@("
        + _row("base", "install.ps1", control = True, result = _COMPILED)
        + ", "
        + _row("head", "install.ps1", control = True, result = _BLOCKED)
        + ")"
    )
    code, text = _run_verdict(
        tmp_path, rows, no_result_ps = "@('head/setup.ps1 [the probe wrote no result]')"
    )
    assert code == 1, f"a refused head script did not fail the job\n{text}"
    assert "AMSI refused head/install.ps1" in text, text
    assert "could not measure head/setup.ps1" in text, "the coverage gap was not reported too"
    assert "verdict=clean" not in text, text


def test_a_parse_error_is_classified_even_when_the_control_is_silent(tmp_path: Path) -> None:
    """A syntax error is a property of the script, not of the scanner.

    The compiler rejects it whether or not an AMSI provider is listening, so gating the
    classification on the per-row control filed a broken candidate as merely unmeasured and exited
    zero. Only the BLOCKED verdict genuinely depends on a live provider.
    """
    rows = (
        "@("
        + _row("base", "install.ps1", control = True, result = _COMPILED)
        + ", "
        + _row("head", "install.ps1", control = False, result = _SYNTAX)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == 1, f"a candidate that does not parse was not reported\n{text}"
    assert "verdict=broken" in text, text


def test_a_block_claimed_without_a_live_control_is_not_trusted(tmp_path: Path) -> None:
    """The other side of the same reordering: only BLOCKED needs the control, and it still needs it."""
    rows = (
        "@("
        + _row("base", "install.ps1", control = True, result = _COMPILED)
        + ", "
        + _row("head", "install.ps1", control = False, result = _BLOCKED)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == 0, text
    assert "verdict=unmeasured" in text, text
    assert "verdict=clean" not in text, text


def test_a_parse_error_is_reported_even_when_no_control_fired_anywhere(tmp_path: Path) -> None:
    """The per-row ordering was fixed and the global one was not, which left the same hole.

    Whether a script compiles does not depend on AMSI, so the compile verdict has to be decided
    before the gate that reports a scanner-less runner as unmeasured. With the gate first, a
    candidate with a syntax error on a runner where no control fired at all, which is the usual
    state of a hosted image, exited zero and said the probe had not completed.
    """
    rows = (
        "@("
        + _row("base", "install.ps1", control = False, result = _COMPILED)
        + ", "
        + _row("head", "install.ps1", control = False, result = _SYNTAX)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == 1, f"a candidate that does not parse was filed as unmeasured\n{text}"
    assert "verdict=broken" in text, text


def test_cloud_readiness_requires_block_at_first_sight_to_be_on() -> None:
    """Reachable is not the same as acting, and the message claimed the second.

    With `DisableBlockAtFirstSeen` set, MAPS answers and `ValidateMapsConnection` succeeds, so the
    gate passed and then told the reader that block-at-first-sight could act on the mark of the web.
    Acting on a zero-prevalence file is the one behaviour this lane exists to exercise, and
    release-desktop.yml treats the same preference as fatal for the same reason.
    """
    body = WORKFLOW.read_text(encoding = "utf-8")
    start = body.index("$cloudReady = $false")
    end = body.index("if ($cloudReady) {", start)
    gate = body[start:end]
    assert "DisableBlockAtFirstSeen" in gate, (
        "the cloud-readiness gate checks only MAPS reporting and connectivity, so a runner with "
        "block-at-first-sight disabled is still labelled cloud-protected"
    )
    assert "$cloudReady = $true" in gate
    assert gate.index("DisableBlockAtFirstSeen") < gate.index(
        "$cloudReady = $true"
    ), "the preference is read after readiness is already decided"


def test_a_real_block_is_still_reported_when_another_script_fails_to_parse(tmp_path: Path) -> None:
    """The compile-failure exit was taken before the blocks were even derived.

    Moving the compile verdict ahead of the control gate was right, but it then ran before
    `$headBlocked` existed, so a run with one unparseable candidate and one candidate genuinely
    refused by a live provider printed only the parse failure. The refusal is the finding this lane
    exists to surface, and it went unmentioned in the log and in the verdict.
    """
    rows = (
        "@("
        + _row("base", "install.ps1", control = True, result = _COMPILED)
        + ", "
        + _row("head", "install.ps1", control = True, result = _SYNTAX)
        + ", "
        + _row("head", "setup.ps1", control = True, result = _BLOCKED)
        + ")"
    )
    code, text = _run_verdict(tmp_path, rows)
    assert code == 1, text
    assert "verdict=broken" in text, text
    assert "AMSI also refused the candidate" in text, (
        "a genuine AMSI refusal was suppressed because another script failed to parse:\n" + text
    )


def test_an_unparseable_probe_result_is_not_silently_dropped() -> None:
    """A row with a null payload is neither a block nor an unmeasured candidate.

    The step runs under `$ErrorActionPreference = 'Continue'`, so an existing but truncated JSON
    file made `ConvertFrom-Json` emit a non-terminating error and return nothing, and the row was
    appended with `data = $null`. That row iterates no results at all, so the file quietly left the
    measured set while any other process whose control fired carried the job to a clean verdict.
    """
    body = WORKFLOW.read_text(encoding = "utf-8")
    start = body.index("$json = Join-Path $out")
    end = body.index("if ($rows.Count -eq 0)", start)
    collection = body[start:end]
    assert "ConvertFrom-Json -ErrorAction Stop" in collection, (
        "the probe result is still parsed without erroring, so an invalid file yields a null row"
    )
    assert "could not be parsed" in collection, (
        "a parse failure is not routed into the no-result list, so it shrinks the measured set "
        "instead of being accounted for"
    )
    assert "carries no results" in collection, (
        "a payload that parses but has no results is still accepted as a measured row"
    )


def test_the_defender_control_is_scanned_the_way_the_candidates_are() -> None:
    """A control that runs a different command than the measurement does not vouch for it.

    `-DisableRemediation` is not only about remediation: this repository's release scanner records
    that it makes the explicit scan ignore file exclusions
    (`.github/workflows/release-desktop.yml:1295` and `:1440-1442`), and the hosted images this lane
    describes ship with both drive roots excluded. The control scan therefore could be skipped
    while the candidate scans worked, leaving the step to exit without measuring anything.
    """
    body = WORKFLOW.read_text(encoding = "utf-8")
    scans = [line for line in body.splitlines() if "-Scan -ScanType 3" in line]
    assert scans, "nothing scans any more"
    for line in scans:
        assert "-DisableRemediation" in line, (
            f"this scan does not use the same flags as the others, so the control and the "
            f"measurement are not comparable:\n{line.strip()}"
        )


def test_the_laid_out_copies_are_exempt_from_on_access_scanning() -> None:
    """On-access quarantine turns a detection into a missing file, which this lane exits zero for.

    Real-time protection acts on open and on write, and `-DisableRemediation` governs only the
    explicit scan, so a live provider can take a copy away while it is being stamped or opened.
    release-desktop.yml adds filesystem exclusions for exactly this reason and notes that the
    verdict is unaffected, because the explicit scan ignores exclusions anyway.
    """
    body = WORKFLOW.read_text(encoding = "utf-8")
    start = body.index("Add-MpPreference -ExclusionPath")
    assert start < body.index("$controlOut = (& $mp -Scan"), (
        "the exclusion is added after the control has already been scanned"
    )
    assert "$env:ROOT" in body[start : start + 120], (
        "the exclusion does not cover the directory the base and head copies were laid out in"
    )
