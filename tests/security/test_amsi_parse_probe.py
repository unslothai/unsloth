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
import re
import shutil
import tempfile
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


def test_the_laid_out_copies_keep_the_bytes_that_ship() -> None:
    """The lane scans the bytes users get, or its verdict does not transfer to them.

    install.ps1 is LF-only with no BOM. `Set-Content` rewrites every line ending to CRLF and, on
    Windows PowerShell 5.1, prepends a BOM, so a copy written that way is a file this project never
    serves. Writing through .NET with a BOM-less encoder is the only way to keep them equal.
    """
    body = WORKFLOW.read_text(encoding = "utf-8")
    assert "UTF8Encoding($false)" in body, (
        "the base/head copies are no longer written with a BOM-less encoder, so the scanner is being "
        "handed bytes that differ from what ships"
    )
    assert "[System.IO.File]::WriteAllText" in body
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
