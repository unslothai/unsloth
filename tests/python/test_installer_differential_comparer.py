# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The base-vs-head comparer must exit non-zero on a real change, or hardening PRs ship unverified."""

from __future__ import annotations

import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / ".github" / "scripts" / "compare_installer_evidence.py"

sys.path.insert(0, str(SCRIPT.parent))
import compare_installer_evidence as cmp  # noqa: E402


BASELINE = "\n".join(
    [
        "  python         3.13.14 ready",
        "  uv             0.12.1 installed",
        "  studio         installed in 12.4s",
        "  shortcut       desktop and Start Menu",
        "  next           run: unsloth studio",
    ]
)

SHORTCUTS = [
    {
        "name": "Unsloth.lnk",
        "targetPath": r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
        "arguments": "-NoProfile -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File launch-studio.ps1",
        "workingDirectory": r"C:\Users\runneradmin\.unsloth\studio",
        "windowStyle": "1",
        "iconLocation": r"C:\Users\runneradmin\.unsloth\studio\unsloth.ico,0",
    }
]

# The collector always writes `bom` and the comparer requires it, so fixtures include it.
ARTIFACTS = {
    "files": {
        "launch-studio.ps1": {
            "content": "Start-Process -WindowStyle Hidden pwsh\n",
            "bom": "utf-8",
        },
        "unsloth.cmd": {"content": "@echo off\n", "bom": "none"},
    },
    "rewrittenOnSecondRun": [],
}

RUN = {"side": "base", "installExit": 0, "secondInstallExit": 0}


def _write(
    directory: Path,
    transcript: str = BASELINE,
    shortcuts = None,
    artifacts = None,
    run = None,
    second: str | None = None,
) -> Path:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "transcript.txt").write_text(transcript, encoding = "utf-8")
    (directory / "transcript-second-run.txt").write_text(
        transcript if second is None else second, encoding = "utf-8"
    )
    (directory / "shortcuts.json").write_text(
        json.dumps(SHORTCUTS if shortcuts is None else shortcuts), encoding = "utf-8"
    )
    (directory / "artifacts.json").write_text(
        json.dumps(ARTIFACTS if artifacts is None else artifacts), encoding = "utf-8"
    )
    (directory / "run.json").write_text(json.dumps(RUN if run is None else run), encoding = "utf-8")
    return directory


def _run(base: Path, head: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--base",
            str(base),
            "--head",
            str(head),
            "--base-sha",
            "a" * 40,
            "--head-sha",
            "b" * 40,
        ],
        capture_output = True,
        text = True,
        timeout = 120,
    )


@pytest.mark.parametrize(
    "mutation, why",
    [
        ("shortcut       desktop only", "a line of user-visible output changed"),
        ("  shortcut       desktop and Start Menu ", "trailing space only, which must NOT fail"),
    ],
)
def test_a_changed_output_line_is_reported(tmp_path: Path, mutation: str, why: str) -> None:
    base = _write(tmp_path / "base")
    mutated = BASELINE.replace("  shortcut       desktop and Start Menu", mutation)
    head = _write(tmp_path / "head", transcript = mutated)
    result = _run(base, head)
    if "must NOT fail" in why:
        assert (
            result.returncode == 0
        ), f"trailing whitespace was treated as a change: {result.stdout}"
    else:
        assert result.returncode == 2, f"{why} was not reported: {result.stdout}\n{result.stderr}"


def test_a_lost_indent_is_reported(tmp_path: Path) -> None:
    """`step` pads its label to exactly 15 columns and the output lock pins the indent."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = BASELINE.replace("  python", "   python"))
    assert _run(base, head).returncode == 2


def test_a_relaxed_execution_policy_in_a_shortcut_is_reported(tmp_path: Path) -> None:
    """The single substitution this whole effort is about, and invisible in the transcript."""
    base = _write(tmp_path / "base")
    relaxed = [
        dict(SHORTCUTS[0], arguments = SHORTCUTS[0]["arguments"].replace("RemoteSigned", "Bypass"))
    ]
    head = _write(tmp_path / "head", shortcuts = relaxed)
    result = _run(base, head)
    assert result.returncode == 2
    assert "arguments" in result.stdout and "Bypass" in result.stdout


def test_a_shortcut_that_stopped_being_created_is_reported(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", shortcuts = [])
    assert _run(base, head).returncode in (2, 3)


def test_a_changed_generated_launcher_is_reported(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    changed = json.loads(json.dumps(ARTIFACTS))
    changed["files"]["launch-studio.ps1"]["content"] = "Start-Process pwsh\n"
    head = _write(tmp_path / "head", artifacts = changed)
    assert _run(base, head).returncode == 2


def test_a_second_run_that_rewrites_files_is_reported(tmp_path: Path) -> None:
    """Several changes here touch content-comparison paths, so idempotency is a real risk."""
    base = _write(tmp_path / "base")
    noisy = json.loads(json.dumps(ARTIFACTS))
    noisy["rewrittenOnSecondRun"] = ["launch-studio.ps1"]
    head = _write(tmp_path / "head", artifacts = noisy)
    result = _run(base, head)
    assert result.returncode == 2
    assert "second time" in result.stdout


def test_known_noise_does_not_fail(tmp_path: Path) -> None:
    noisy = "\n".join(
        [
            "  python         3.13.9 ready",
            "  uv             0.12.4 installed",
            "  studio         installed in 41.9s",
            "  shortcut       desktop and Start Menu",
            "  next           run: unsloth studio",
        ]
    )
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = noisy)
    result = _run(base, head)
    assert result.returncode == 0, f"noise failed the lane: {result.stdout}"


def test_winget_spinner_frames_do_not_fail(tmp_path: Path) -> None:
    """How many spinner frames winget leaves in the log depends on how long its source query took."""
    lines = BASELINE.split("\n")
    base = _write(
        tmp_path / "base", transcript = "\n".join(lines[:1] + ["   - ", "   \\ "] + lines[1:])
    )
    head = _write(
        tmp_path / "head", transcript = "\n".join(lines[:1] + ["   - ", "   | ", "   - "] + lines[1:])
    )
    result = _run(base, head)
    assert result.returncode == 0, f"spinner frames failed the lane: {result.stdout}"


def test_winget_partial_progress_frames_do_not_fail(tmp_path: Path) -> None:
    """How many partly filled download bars winget leaves in the log depends on the network."""
    full = "  ██████████████████████████████  17.2 MB / 17.2 MB"
    lines = BASELINE.split("\n")
    base_bars = ["  █████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  5.1 MB / 17.2 MB", full]
    head_bars = [
        "  ██████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  3.0 MB / 17.2 MB",
        "  █████████████████████▒▒▒▒▒▒▒▒▒  70%",
        full,
    ]
    base = _write(tmp_path / "base", transcript = "\n".join(lines[:1] + base_bars + lines[1:]))
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + head_bars + lines[1:]))
    result = _run(base, head)
    assert result.returncode == 0, f"partial progress frames failed the lane: {result.stdout}"


def test_a_repeated_full_bar_does_not_fail(tmp_path: Path) -> None:
    """winget can redraw the finished bar once more; a second copy right after the first is noise."""
    full = "  ██████████████████████████████  17.2 MB / 17.2 MB"
    lines = BASELINE.split("\n")
    base = _write(tmp_path / "base", transcript = "\n".join(lines[:1] + [full, full] + lines[1:]))
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + [full] + lines[1:]))
    result = _run(base, head)
    assert result.returncode == 0, f"a repeated full bar failed the lane: {result.stdout}"


def test_a_second_download_still_compares(tmp_path: Path) -> None:
    """Two downloads leave two full bars with a "Downloading" line between them; losing one is reported."""
    full = "  ██████████████████████████████  17.2 MB / 17.2 MB"
    lines = BASELINE.split("\n")
    twice = [full, "Downloading https://example.invalid/uv.zip", full]
    base = _write(tmp_path / "base", transcript = "\n".join(lines[:1] + twice + lines[1:]))
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + twice[:2] + lines[1:]))
    assert _run(base, head).returncode == 2


def test_a_second_transfer_with_no_header_still_compares(tmp_path: Path) -> None:
    """A partial frame between two full bars is a new transfer, not a redraw of the first."""
    full = "  ██████████████████████████████  17.2 MB / 17.2 MB"
    partial = "  ███████████████▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  8.6 MB / 17.2 MB"
    lines = BASELINE.split("\n")
    base = _write(
        tmp_path / "base", transcript = "\n".join(lines[:1] + [full, partial, full] + lines[1:])
    )
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + [full] + lines[1:]))
    assert _run(base, head).returncode == 2


def test_a_redraw_that_only_adds_a_colour_escape_is_still_a_redraw(tmp_path: Path) -> None:
    """A style escape on one copy of the full bar does not make it a second download."""
    full = "  ██████████████████████████████  17.2 MB / 17.2 MB"
    styled = "\x1b[32m" + full + "\x1b[0m"
    lines = BASELINE.split("\n")
    base = _write(tmp_path / "base", transcript = "\n".join(lines[:1] + [full, styled] + lines[1:]))
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + [full] + lines[1:]))
    assert _run(base, head).returncode == 0


def test_two_finished_downloads_of_different_sizes_still_compare(tmp_path: Path) -> None:
    """Both bars normalise to <size> / <size>, but different raw sizes are two downloads."""
    first = "  ██████████████████████████████  17.2 MB / 17.2 MB"
    second = "  ██████████████████████████████  34.4 MB / 34.4 MB"
    lines = BASELINE.split("\n")
    base = _write(tmp_path / "base", transcript = "\n".join(lines[:1] + [first, second] + lines[1:]))
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + [first] + lines[1:]))
    assert _run(base, head).returncode == 2


def test_two_finished_downloads_differing_only_in_units_still_compare(tmp_path: Path) -> None:
    """`1.0 MB / 1.0 MB` then `1.0 GB / 1.0 GB` are two downloads, not a redraw."""
    first = "  ██████████████████████████████  1.0 MB / 1.0 MB"
    second = "  ██████████████████████████████  1.0 GB / 1.0 GB"
    lines = BASELINE.split("\n")
    base = _write(tmp_path / "base", transcript = "\n".join(lines[:1] + [first, second] + lines[1:]))
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + [first] + lines[1:]))
    assert _run(base, head).returncode == 2


def test_a_download_that_never_finishes_still_compares(tmp_path: Path) -> None:
    """The full bar a finished download ends on is kept, so losing it is reported."""
    lines = BASELINE.split("\n")
    full = "  ██████████████████████████████  17.2 MB / 17.2 MB"
    base = _write(tmp_path / "base", transcript = "\n".join(lines[:1] + [full] + lines[1:]))
    head = _write(tmp_path / "head", transcript = "\n".join(lines[:1] + lines[1:]))
    assert _run(base, head).returncode == 2


def test_a_dash_line_with_text_still_compares(tmp_path: Path) -> None:
    """Only a bare frame is dropped: winget's own "  - Packages" list is output the user reads."""
    base = _write(tmp_path / "base", transcript = BASELINE + "\n  - Packages")
    head = _write(tmp_path / "head", transcript = BASELINE + "\n  - Package")
    assert _run(base, head).returncode == 2


def test_version_drift_is_normalised_but_still_printed(tmp_path: Path) -> None:
    """Normalising something away without saying so is how a lane stops telling you anything."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = BASELINE.replace("0.12.1", "0.12.4"))
    result = _run(base, head)
    assert result.returncode == 0
    assert (
        "version drift in the first-run transcript" in result.stdout and "0.12.4" in result.stdout
    )


def test_missing_evidence_is_void_and_not_a_pass(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    result = _run(base, tmp_path / "absent")
    assert result.returncode == 3, "a side with no evidence must not read as agreement"
    assert "VOID" in result.stdout


def test_an_empty_transcript_is_void(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = "\n\n  \n")
    result = _run(base, head)
    assert result.returncode == 3
    assert "did not run" in result.stdout


def test_comparing_a_commit_with_itself_is_void() -> None:
    verdict = cmp.compare_directories(Path("/x"), Path("/y"), "c" * 40, "c" * 40)
    assert verdict.is_void
    assert verdict.exit_code() == 3
    assert any("itself" in reason for reason in verdict.void)


def test_two_empty_shortcut_manifests_are_void_not_agreement() -> None:
    verdict = cmp.Verdict()
    cmp.compare_shortcuts([], [], verdict)
    assert verdict.is_void


def test_void_and_different_have_distinct_exit_codes() -> None:
    """A workflow that treats every non-zero the same cannot tell "it changed" from "we did not
    look", and those need different reactions from whoever reads the run."""
    void = cmp.Verdict()
    void.void.append("x")
    different = cmp.Verdict()
    different.differences.append("y")
    assert void.exit_code() == 3
    assert different.exit_code() == 2
    assert cmp.Verdict().exit_code() == 0


def test_the_self_test_passes() -> None:
    """CI runs this before any real comparison, so a broken normaliser refuses to produce a
    verdict instead of producing a reassuring one."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--self-test"], capture_output = True, text = True, timeout = 120
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_self_test_can_fail() -> None:
    """A control that cannot fail is decoration. Widen the normaliser until it swallows a real
    change and the self-test must notice."""
    original = cmp._NORMALISERS
    try:
        import re

        cmp._NORMALISERS = original + ((re.compile(r".*"), "", "everything", True),)
        failures = cmp.self_test()
        assert failures, "a normaliser that erases every line still passed the controls"
        assert any("too loose" in f for f in failures)
    finally:
        cmp._NORMALISERS = original


def test_every_normaliser_records_why_it_exists() -> None:
    """A rule with no recorded cause is a rule nobody can argue with, and this list is exactly
    where a future 'just make the lane green' change would land."""
    for pattern, _replacement, why, in_scripts in cmp._NORMALISERS:
        assert why and len(why) > 8, f"the normaliser {pattern.pattern!r} has no stated reason"
        assert isinstance(in_scripts, bool), (
            f"the normaliser {pattern.pattern!r} does not say whether it applies to a generated "
            f"script, and defaulting that wrong is how a behaviour change gets normalised away"
        )


def test_the_shortcut_fields_compared_include_the_launch_contract() -> None:
    for field in ("arguments", "targetPath", "windowStyle"):
        assert field in cmp._SHORTCUT_FIELDS


def test_a_single_shortcut_serialised_as_an_object_still_compares(tmp_path: Path) -> None:
    """A single-element ConvertTo-Json output is a bare object, so iterating it yields dict keys."""
    base = _write(tmp_path / "base")
    (base / "shortcuts.json").write_text(json.dumps(SHORTCUTS[0]), encoding = "utf-8")
    head = _write(tmp_path / "head")
    relaxed = dict(
        SHORTCUTS[0], arguments = SHORTCUTS[0]["arguments"].replace("RemoteSigned", "Bypass")
    )
    (head / "shortcuts.json").write_text(json.dumps(relaxed), encoding = "utf-8")

    result = _run(base, head)
    assert (
        result.returncode == 2
    ), f"an unwrapped single shortcut was not compared as a shortcut: {result.stdout}"
    assert "Bypass" in result.stdout


def test_a_symmetric_collection_failure_is_void_not_a_pass(tmp_path: Path) -> None:
    """Symmetric collection failures on both sides must void the comparison, not pass it as equal."""
    failed = [{"name": "<collection failed>", "error": "could not create WScript.Shell"}]
    base = _write(tmp_path / "base", shortcuts = failed)
    head = _write(tmp_path / "head", shortcuts = failed)
    result = _run(base, head)
    assert result.returncode == 3, f"a shared collection failure read as agreement: {result.stdout}"
    assert "prove nothing" in result.stdout


# Shared failures (mirror outage, image change) match on both sides; exit status breaks the tie.


def test_two_installers_that_both_failed_are_void_not_a_pass(tmp_path: Path) -> None:
    dead = {"installExit": 1, "secondInstallExit": 1}
    base = _write(tmp_path / "base", transcript = BASELINE, run = dead)
    head = _write(tmp_path / "head", transcript = BASELINE, run = dead)
    result = _run(base, head)
    assert result.returncode == 3, f"two failed installs read as agreement: {result.stdout}"
    assert "exited 1" in result.stdout


def test_one_installer_that_failed_is_void(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", run = {"installExit": 5, "secondInstallExit": 0})
    assert _run(base, head).returncode == 3


def test_a_failed_second_install_is_void(tmp_path: Path) -> None:
    """The idempotency install is a measurement too, and one that did not run measured nothing."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", run = {"installExit": 0, "secondInstallExit": 3})
    assert _run(base, head).returncode == 3


def test_evidence_with_no_recorded_exit_status_is_void(tmp_path: Path) -> None:
    """Both sides are measured with the candidate's tools, so evidence without a run status came
    from a workflow that was never in a position to say the installer finished."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head")
    (head / "run.json").unlink()
    result = _run(base, head)
    assert result.returncode == 3
    assert "VOID" in result.stdout


@pytest.mark.parametrize("bad", [{"installExit": "0"}, {"installExit": True}, {}, []])
def test_an_unusable_exit_status_is_void(tmp_path: Path, bad) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head")
    (head / "run.json").write_text(json.dumps(bad), encoding = "utf-8")
    assert _run(base, head).returncode == 3


# An errored artifact entry loses only `content`, so it would silently skip comparison.


def test_an_unreadable_artifact_is_void_not_a_skipped_comparison(tmp_path: Path) -> None:
    broken = json.loads(json.dumps(ARTIFACTS))
    broken["files"]["launch-studio.ps1"] = {"error": "access is denied"}
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", artifacts = broken)
    result = _run(base, head)
    assert result.returncode == 3, f"an unreadable contract file read as agreement: {result.stdout}"
    assert "never read" in result.stdout


def test_a_symmetric_artifact_read_failure_is_void(tmp_path: Path) -> None:
    broken = json.loads(json.dumps(ARTIFACTS))
    broken["files"]["unsloth.cmd"] = {"error": "access is denied"}
    base = _write(tmp_path / "base", artifacts = broken)
    head = _write(tmp_path / "head", artifacts = broken)
    assert _run(base, head).returncode == 3


def test_an_install_root_that_does_not_exist_is_void(tmp_path: Path) -> None:
    """What the collector writes when the installer did not finish. It is an error field, and an
    error field on both sides compares equal to itself."""
    failed = {"studioHome": r"C:\x", "files": {}, "error": "the install root C:\\x does not exist"}
    base = _write(tmp_path / "base", artifacts = failed)
    head = _write(tmp_path / "head", artifacts = failed)
    result = _run(base, head)
    assert result.returncode == 3
    assert "not evidence" in result.stdout


def test_content_captured_on_one_side_only_is_void(tmp_path: Path) -> None:
    half = json.loads(json.dumps(ARTIFACTS))
    half["files"]["launch-studio.ps1"] = {"sha256": "a" * 64}
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", artifacts = half)
    assert _run(base, head).returncode == 3


@pytest.mark.parametrize("blob", ['"nothing here"', "42", '["Unsloth.lnk"]'])
def test_a_shortcut_manifest_of_the_wrong_shape_is_void(tmp_path: Path, blob: str) -> None:
    """Coercing it to an empty list made it compare against a populated side as "every shortcut
    disappeared", which is a behaviour difference reported about evidence nobody parsed."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head")
    (head / "shortcuts.json").write_text(blob, encoding = "utf-8")
    assert _run(base, head).returncode == 3


@pytest.mark.parametrize("blob", ["[]", '"x"'])
def test_an_artifact_manifest_of_the_wrong_shape_is_void(tmp_path: Path, blob: str) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head")
    (head / "artifacts.json").write_text(blob, encoding = "utf-8")
    assert _run(base, head).returncode == 3


@pytest.mark.parametrize("name", ["shortcuts.json", "artifacts.json", "run.json"])
def test_a_byte_order_mark_does_not_turn_good_evidence_into_void(tmp_path: Path, name: str) -> None:
    """Windows PowerShell 5.1 writes a BOM for `Set-Content -Encoding utf8`, and json.loads
    rejects one. A lane that goes VOID on a valid file stops being read."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head")
    text = (head / name).read_text(encoding = "utf-8")
    (head / name).write_bytes(b"\xef\xbb\xbf" + text.encode("utf-8"))
    assert _run(base, head).returncode == 0


def test_a_byte_order_mark_does_not_hide_a_real_change(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head")
    relaxed = [
        dict(SHORTCUTS[0], arguments = SHORTCUTS[0]["arguments"].replace("RemoteSigned", "Bypass"))
    ]
    (head / "shortcuts.json").write_bytes(b"\xef\xbb\xbf" + json.dumps(relaxed).encode("utf-8"))
    result = _run(base, head)
    assert result.returncode == 2
    assert "Bypass" in result.stdout


@pytest.mark.parametrize(
    "before, after",
    [
        (
            r"  work           C:\Temp\unsloth-uv-1a2b3c4d\bin",
            r"  work           C:\Temp\unsloth-uv-99ffee00\bin",
        ),
        (
            "  probe          unsloth-probe-1a2b3c4d.tmp",
            "  probe          unsloth-probe-ffee9900.tmp",
        ),
        (
            "  probe          .unsloth-write-probe.1a2b3c4d5e6f7a8b",
            "  probe          .unsloth-write-probe.ffee99001a2b3c4d",
        ),
    ],
)
def test_a_random_scratch_name_is_still_normalised(before: str, after: str) -> None:
    verdict = cmp.Verdict()
    cmp.compare_transcripts(BASELINE + "\n" + before, BASELINE + "\n" + after, verdict)
    assert not verdict.differences, verdict.differences


@pytest.mark.parametrize(
    "before, after",
    [
        ("unsloth-studio-managed-launcher", "unsloth-desktop-managed-launcher"),
        ("pip install unsloth-studio", "pip install unsloth-nightly"),
        (".unsloth-studio-owned", ".unsloth-desktop-owned"),
    ],
)
def test_a_renamed_unsloth_marker_is_not_a_scratch_name(before: str, after: str) -> None:
    """`unsloth-studio-managed-launcher` is written into unsloth.cmd and is how the installer
    recognises its own shim, so a rename is a behaviour change. The rule meant for
    `unsloth-uv-<hex8>` matched any six characters after `unsloth-` and swallowed it."""
    verdict = cmp.Verdict()
    cmp.compare_transcripts(
        BASELINE + "\n  cmd            " + before, BASELINE + "\n  cmd            " + after, verdict
    )
    assert verdict.differences, f"{before!r} -> {after!r} was normalised away"


def test_a_renamed_launcher_marker_inside_unsloth_cmd_is_reported(tmp_path: Path) -> None:
    marked = json.loads(json.dumps(ARTIFACTS))
    marked["files"]["unsloth.cmd"]["content"] = "@echo off\nrem unsloth-studio-managed-launcher\n"
    renamed = json.loads(json.dumps(marked))
    renamed["files"]["unsloth.cmd"]["content"] = "@echo off\nrem unsloth-desktop-managed-launcher\n"
    base = _write(tmp_path / "base", artifacts = marked)
    head = _write(tmp_path / "head", artifacts = renamed)
    assert _run(base, head).returncode == 2


def test_a_version_drift_inside_a_shortcut_is_printed_even_though_it_is_normalised(
    tmp_path: Path,
) -> None:
    """normalise_line runs on shortcut fields too, so a launcher retargeted at a different python
    is erased in exactly the same way a patch release is. That is the right call and the wrong one
    to make silently, so wherever the rule reaches, the drift is said out loud."""
    base = _write(tmp_path / "base")
    retargeted = [
        dict(
            SHORTCUTS[0],
            targetPath = r"C:\Users\runneradmin\.unsloth\python-3.13.14\python.exe",
        )
    ]
    bumped = [
        dict(
            SHORTCUTS[0],
            targetPath = r"C:\Users\runneradmin\.unsloth\python-3.11.9\python.exe",
        )
    ]
    base = _write(tmp_path / "base", shortcuts = retargeted)
    head = _write(tmp_path / "head", shortcuts = bumped)
    result = _run(base, head)
    assert result.returncode == 0
    assert "version drift in the shortcut fields" in result.stdout, result.stdout
    assert "3.11.9" in result.stdout


def test_the_shortcut_note_is_not_suppressed_by_a_transcript_difference(tmp_path: Path) -> None:
    """Reading the whole verdict meant a changed output line also stopped the run saying whether
    the launch contract had been looked at at all."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", transcript = BASELINE.replace("ready", "prepared"))
    result = _run(base, head)
    assert result.returncode == 2
    assert "every field equal" in result.stdout


def test_the_collector_parses_and_emits_json_the_comparer_accepts(tmp_path: Path) -> None:
    """The collector forces the array: ConvertTo-Json unwraps one element and PS 5.1 lacks -AsArray."""
    sys.path.insert(0, str(REPO / "tests" / "_shared"))
    from unsloth_pwsh_runner import PWSH, run_pwsh  # noqa: PLC0415

    if PWSH is None:
        pytest.skip("no PowerShell on this host")

    collector = REPO / ".github" / "scripts" / "Collect-InstallerEvidence.ps1"
    out = tmp_path / "evidence"
    script = f"""
$ErrorActionPreference = 'Stop'
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
    '{collector.as_posix()}', [ref]$null, [ref]$errors) | Out-Null
if ($errors) {{ $errors | ForEach-Object {{ "PARSE: $_" }}; exit 1 }}

& '{collector.as_posix()}' -StudioHome '{(tmp_path / "home").as_posix()}' -OutDir '{out.as_posix()}' |
    Out-Null

# The forcing, exercised directly with exactly one entry, which is the case that produced an
# object on a real run.
$one = New-Object System.Collections.ArrayList
[void]$one.Add([ordered]@{{ name = 'Unsloth.lnk'; arguments = '-ExecutionPolicy RemoteSigned' }})
$json = $one | ConvertTo-Json -Depth 6
if ($one.Count -le 1) {{ $json = "[$($json)]" }}
Set-Content -LiteralPath '{(tmp_path / "one.json").as_posix()}' -Value $json -Encoding utf8
Write-Output 'COLLECTOR-OK'
"""
    result = run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        verdict = "COLLECTOR-OK",
        timeout = 300,
    )
    assert "COLLECTOR-OK" in result.stdout, result.stdout + result.stderr

    one = json.loads((tmp_path / "one.json").read_text(encoding = "utf-8-sig"))
    assert (
        isinstance(one, list) and len(one) == 1
    ), f"a single shortcut did not serialise as a list: {one!r}"
    assert one[0]["arguments"] == "-ExecutionPolicy RemoteSigned"

    shortcuts = json.loads((out / "shortcuts.json").read_text(encoding = "utf-8-sig"))
    assert isinstance(shortcuts, list), f"shortcuts.json is not a list: {shortcuts!r}"

    # Off Windows there is no WScript.Shell; the collector records an error the comparer must VOID.
    artifacts = json.loads((out / "artifacts.json").read_text(encoding = "utf-8-sig"))
    verdict = cmp.Verdict()
    cmp.compare_shortcuts(shortcuts, shortcuts, verdict)
    cmp.compare_artifacts(artifacts, artifacts, verdict)
    assert verdict.is_void, "the collector's own failure output did not read as VOID"
    assert verdict.exit_code() == 3


def _differential_workflow() -> dict:
    import yaml
    path = REPO / ".github" / "workflows" / "windows-installer-differential-ci.yml"
    return yaml.safe_load(path.read_text(encoding = "utf-8"))


def test_the_lane_never_installs_in_env_override_mode() -> None:
    """Setting UNSLOTH_STUDIO_HOME skips shortcut writes, so the comparison compares empty against empty."""
    workflow = _differential_workflow()
    offenders = []
    for job_name, job in workflow["jobs"].items():
        for step in job.get("steps") or []:
            env = step.get("env") or {}
            if "UNSLOTH_STUDIO_HOME" in env:
                offenders.append(f"{job_name}/{step.get('name', '<unnamed>')}")
    assert not offenders, (
        f"these steps set UNSLOTH_STUDIO_HOME: {offenders}. That puts the installer into "
        f"env-override mode, where it creates no shortcuts at all, so the lane measures no "
        f"shortcuts on either side and reports them equal."
    )


def test_a_desktop_and_a_start_menu_shortcut_stay_separate() -> None:
    """Shortcut keys need the location: Unsloth Studio.lnk on Desktop and Start Menu would collapse."""
    both = [
        {"name": "Unsloth Studio.lnk", "root": "Desktop", "targetPath": "p", "arguments": "a"},
        {"name": "Unsloth Studio.lnk", "root": "Programs", "targetPath": "p", "arguments": "a"},
    ]
    desktop_only = [both[0]]

    keys = {cmp._shortcut_key(e) for e in both}
    assert len(keys) == 2, f"the two locations collapsed to one key: {keys}"

    verdict = cmp.Verdict()
    cmp.compare_shortcuts(both, desktop_only, verdict)
    assert verdict.differences, "losing the Start Menu shortcut was reported as no change"
    assert any("Programs" in d for d in verdict.differences), verdict.differences


def test_unmeasured_idempotency_is_void_not_a_note() -> None:
    """None means not measured and voids the check; [] means measured with no rewrites. Keep them apart."""
    # Identical apart from missing idempotency evidence, so only the thing under test can void it.
    complete = {
        "studioHome": "X",
        "files": {
            "launch-studio.ps1": {
                "foundAt": "data/launch-studio.ps1",
                "content": "a",
                "sha256": "A",
                "bom": "utf-8",
            },
            "unsloth.cmd": {
                "foundAt": "home/bin\\unsloth.cmd",
                "content": "b",
                "sha256": "B",
                "bom": "none",
            },
        },
    }
    measured = dict(complete, rewrittenOnSecondRun = [])

    ok = cmp.Verdict()
    cmp.compare_artifacts(measured, dict(measured), ok)
    assert not ok.is_void, f"the control case voided for an unrelated reason: {ok.void}"

    verdict = cmp.Verdict()
    cmp.compare_artifacts(complete, dict(complete), verdict)
    assert verdict.is_void, "a run that never measured idempotency was still eligible to pass"
    assert verdict.exit_code() == 3, verdict.void
    assert any("idempotency" in row for row in verdict.void), verdict.void


def test_the_lane_overlays_the_checkout_so_setup_ps1_is_the_candidates() -> None:
    """Without the overlay, both legs run the released studio/setup.ps1 from PyPI, not the checkout's."""
    workflow = _differential_workflow()
    installs = [
        step
        for job in workflow["jobs"].values()
        for step in (job.get("steps") or [])
        if "-File ./install.ps1" in str(step.get("run", ""))
    ]
    assert installs, "no step invokes install.ps1 any more"
    for step in installs:
        env = step.get("env") or {}
        assert "UNSLOTH_CI_SOURCE_OVERLAY" in env, (
            f"step {step.get('name')!r} runs install.ps1 without UNSLOTH_CI_SOURCE_OVERLAY, so the "
            f"Python side including studio/setup.ps1 comes from the released wheel rather than "
            f"from this leg's checkout"
        )


def test_the_collector_is_given_the_data_directory_too() -> None:
    """Pass the data directory too: normal-profile installs write the launcher outside $StudioHome."""
    workflow = _differential_workflow()
    calls = [
        step
        for job in workflow["jobs"].values()
        for step in (job.get("steps") or [])
        if "& (Join-Path $env:UNSLOTH_EVIDENCE_TOOLS 'Collect-InstallerEvidence.ps1')"
        in str(step.get("run", ""))
    ]
    assert calls, "nothing invokes the collector any more"
    for step in calls:
        assert "-StudioDataDir" in str(step["run"]), (
            f"step {step.get('name')!r} calls the collector without -StudioDataDir, so "
            f"launch-studio.ps1 is absent from its manifest on a normal-profile install"
        )


def test_a_contract_that_moved_without_changing_its_bytes_is_a_difference() -> None:
    """Compare foundAt too: a contract moved elsewhere with identical bytes is a real difference."""

    def side(found_at: str) -> dict:
        return {
            "studioHome": "X",
            "files": {"studio.conf": {"foundAt": found_at, "content": "a", "sha256": "A"}},
            "rewrittenOnSecondRun": [],
        }

    same = cmp.Verdict()
    cmp.compare_artifacts(side("home/studio.conf"), side("home/studio.conf"), same)
    assert not same.differences, f"an unmoved contract was reported as moved: {same.differences}"

    verdict = cmp.Verdict()
    cmp.compare_artifacts(side("home/studio.conf"), side("home/share\\studio.conf"), verdict)
    assert verdict.differences, "a relocated contract with identical bytes was reported as equal"
    assert any("moved" in row for row in verdict.differences), verdict.differences


def test_the_collector_builds_shortcut_timestamps_before_it_compares_them() -> None:
    """Build $shortcutWrites before CompareAgainst reads it; under StrictMode the early read hides
    rewrites."""
    script = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "scripts"
        / "Collect-InstallerEvidence.ps1"
    )
    text = script.read_text(encoding = "utf-8")
    built = text.index("$shortcutWrites = [ordered]@{}")
    read = text.index("foreach ($key in $shortcutWrites.Keys)")
    assert built < read, (
        "$shortcutWrites is still built after the comparison that reads it, so the shortcut half of "
        "the idempotency measurement cannot run"
    )


def test_a_shortcut_description_change_is_a_difference() -> None:
    """Shortcut Description must be a compared field, or a description-only change passes as equal."""

    def side(description: str) -> list[dict]:
        return [
            {
                "root": "desktop",
                "name": "Unsloth Studio.lnk",
                "targetPath": "C:\\ps.exe",
                "arguments": "-File x",
                "workingDirectory": "C:\\",
                "windowStyle": 7,
                "iconLocation": "C:\\i.ico,0",
                "description": description,
            }
        ]

    same = cmp.Verdict()
    cmp.compare_shortcuts(side("Launch Unsloth Studio"), side("Launch Unsloth Studio"), same)
    assert not same.differences, same.differences

    verdict = cmp.Verdict()
    cmp.compare_shortcuts(side("Launch Unsloth Studio"), side("Start Unsloth"), verdict)
    assert verdict.differences, "a changed shortcut description was reported as no change"
    assert any("description" in row for row in verdict.differences), verdict.differences


def test_the_collector_treats_a_second_run_shortcut_creation_as_a_write() -> None:
    """A shortcut created on a second run must count as a write; a key with no prior timestamp was
    missed."""
    script = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "scripts"
        / "Collect-InstallerEvidence.ps1"
    )
    text = script.read_text(encoding = "utf-8")
    assert "created by the second run" in text, (
        "a shortcut present now and absent from the first-run manifest is still ignored, so a "
        "reinstall that created one measures as writing nothing"
    )
    assert "first-run shortcut evidence is missing" in text, (
        "a first-run manifest with no shortcut write times still yields a clean measurement "
        "rather than voiding the half it could not measure"
    )


def test_a_contract_missing_on_both_sides_is_void_not_agreement() -> None:
    """Missing on both sides must void the comparison, not count as agreement about a file never checked."""
    missing = {
        "studioHome": "X",
        "files": {
            "launch-studio.ps1": {
                "foundAt": "data/launch-studio.ps1",
                "content": "a",
                "sha256": "A",
            },
            "unsloth.cmd": {"error": "not found at any supported location under home, data"},
        },
        "rewrittenOnSecondRun": [],
    }
    verdict = cmp.Verdict()
    cmp.compare_artifacts(missing, dict(missing), verdict)
    assert verdict.is_void, "a contract missing on both sides was treated as agreement"
    assert verdict.exit_code() == 3, verdict.void

    script = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "scripts"
        / "Collect-InstallerEvidence.ps1"
    )
    assert "not found at any supported location" in script.read_text(
        encoding = "utf-8"
    ), "the collector still drops an unresolved contract instead of recording it as an error"


def test_two_roots_with_the_same_leaf_name_stay_distinct() -> None:
    """Keying roots on `Split-Path -Leaf` merged the per-user and common Desktop into one key."""
    script = (
        Path(__file__).resolve().parents[2]
        / ".github"
        / "scripts"
        / "Collect-InstallerEvidence.ps1"
    )
    text = script.read_text(encoding = "utf-8")
    assert "Get-UnslothRootLabel" in text, "the root is no longer canonicalised"
    assert (
        "CommonDesktop" in text and "UserDesktop" in text
    ), "the two desktops are not distinguished, so a move between them compares equal"
    assert (
        "root             = (Split-Path $root -Leaf)" not in text
    ), "the shortcut root is still keyed on the leaf name alone"
    assert "$file.FullName.Substring($root.Length)" in text, (
        "a shortcut nested under Programs is still keyed on its bare file name, so a move into or "
        "out of a subdirectory is invisible"
    )

    assert cmp._shortcut_key(
        {"name": "Unsloth Studio.lnk", "root": "UserDesktop"}
    ) != cmp._shortcut_key({"name": "Unsloth Studio.lnk", "root": "CommonDesktop"})


def test_every_collected_contract_is_one_the_windows_installer_writes() -> None:
    """`studio.conf` is written only by `install.sh`, so the Windows contract list must not include it."""
    repo = Path(__file__).resolve().parents[2]
    script = (repo / ".github" / "scripts" / "Collect-InstallerEvidence.ps1").read_text(
        encoding = "utf-8"
    )
    block = script[script.index("$contentFiles = [ordered]@{") :]
    block = block[: block.index("\n}")]
    contracts = re.findall(r"^\s*'([^']+)'\s*=", block, re.M)
    assert contracts, "the contract list is empty"

    windows_sources = "\n".join(
        (repo / name).read_text(encoding = "utf-8", errors = "ignore")
        for name in ("install.ps1", "studio/setup.ps1")
    )
    # unsloth.cmd is written to a temp then renamed, so Move-Item counts as a write.
    writers = (
        "Set-Content",
        "Out-File",
        "WriteAllText",
        "WriteAllBytes",
        "WriteAllLines",
        "Move-Item",
    )
    for contract in contracts:
        holders = set(
            re.findall(r"\$(\w+)\s*=\s*Join-Path[^\n]*" + re.escape(contract), windows_sources)
        )
        assert holders, (
            f"{contract!r} is collected as a contract but no variable in install.ps1 or "
            f"studio/setup.ps1 is ever set to its path"
        )
        written = any(
            re.search(r"(?:" + "|".join(writers) + r")[^\n]*\$" + holder, windows_sources)
            for holder in holders
        )
        assert written, (
            f"{contract!r} is collected as a contract but nothing in install.ps1 or "
            f"studio/setup.ps1 writes it -- it is only read -- so a clean Windows run records it "
            f"as missing and the lane voids every time. That is exactly what studio.conf did."
        )


def test_a_launcher_that_loses_its_bom_is_a_difference() -> None:
    """PowerShell 5.1 reads a BOM-less launcher as ANSI, so the UTF-8 BOM is part of the contract."""

    def side(bom: str) -> dict:
        return {
            "studioHome": "X",
            "files": {
                "launch-studio.ps1": {
                    "foundAt": "data/launch-studio.ps1",
                    "content": "Write-Host 'hi'\n",
                    "sha256": "A",
                    "bom": bom,
                }
            },
            "rewrittenOnSecondRun": [],
        }

    same = cmp.Verdict()
    cmp.compare_artifacts(side("utf-8"), side("utf-8"), same)
    assert not same.differences, same.differences

    verdict = cmp.Verdict()
    cmp.compare_artifacts(side("utf-8"), side("none"), verdict)
    assert verdict.differences, "a launcher that lost its BOM compared equal"
    assert any("encoding" in row for row in verdict.differences), verdict.differences


def test_version_drift_in_a_generated_file_is_reported() -> None:
    """Generated-file comparison must call `report_version_drift` on the raw text before normalising."""

    def side(version: str) -> dict:
        return {
            "studioHome": "X",
            "files": {
                "launch-studio.ps1": {
                    "foundAt": "data/launch-studio.ps1",
                    "content": f"& 'C:\\\\py\\\\python-{version}\\\\python.exe' -m unsloth\n",
                    "sha256": "A",
                    "bom": "utf-8",
                }
            },
            "rewrittenOnSecondRun": [],
        }

    verdict = cmp.Verdict()
    cmp.compare_artifacts(side("3.11.9"), side("3.13.0"), verdict)
    reported = verdict.differences + verdict.notes
    assert any(
        "3.11.9" in row or "3.13.0" in row or "version" in row.lower() for row in reported
    ), f"a retargeted launcher produced no drift note at all: {reported}"


def test_a_launcher_that_expects_the_wrong_install_id_is_reported() -> None:
    """The launcher's `$_ExpectedStudioRootId` must match the persisted ID; normalising hex hides it."""

    def side(persisted: str, embedded: str) -> dict:
        return {
            "studioHome": "X",
            "files": {
                "launch-studio.ps1": {
                    "foundAt": "data/launch-studio.ps1",
                    "content": "x",
                    "sha256": "A",
                    "bom": "utf-8",
                }
            },
            "rewrittenOnSecondRun": [],
            "installId": persisted,
            "embeddedId": embedded,
        }

    good = cmp.Verdict()
    cmp.compare_artifacts(side("a" * 64, "a" * 64), side("b" * 64, "b" * 64), good)
    assert (
        not good.differences
    ), f"two healthy installs with different IDs were reported as a difference: {good.differences}"

    verdict = cmp.Verdict()
    cmp.compare_artifacts(side("a" * 64, "a" * 64), side("b" * 64, "c" * 64), verdict)
    assert verdict.differences, "a launcher expecting the wrong install ID was not reported"
    assert any("refuse its own backend" in row for row in verdict.differences), verdict.differences


def test_a_malformed_artifact_entry_is_void_not_skipped() -> None:
    """A non-object artifact entry must void the run, not be skipped, or nothing about it is compared."""

    def side(entry) -> dict:
        return {
            "studioHome": "X",
            "files": {
                "launch-studio.ps1": {
                    "foundAt": "data/launch-studio.ps1",
                    "content": "a",
                    "sha256": "A",
                    "bom": "utf-8",
                },
                "unsloth.cmd": entry,
            },
            "rewrittenOnSecondRun": [],
        }

    ok = cmp.Verdict()
    cmp.compare_artifacts(
        side({"foundAt": "home/bin\\unsloth.cmd", "content": "b", "sha256": "B", "bom": "none"}),
        side({"foundAt": "home/bin\\unsloth.cmd", "content": "b", "sha256": "B", "bom": "none"}),
        ok,
    )
    assert not ok.is_void, f"the control voided for an unrelated reason: {ok.void}"

    verdict = cmp.Verdict()
    cmp.compare_artifacts(side("not-an-object"), side("not-an-object"), verdict)
    assert verdict.is_void, "symmetrically malformed artifact evidence was reported as agreement"
    assert verdict.exit_code() == 3, verdict.void


def test_the_trigger_only_lists_files_this_lane_actually_runs() -> None:
    """Trigger paths must only list files this lane actually runs, or changes to others pass untested."""
    import yaml as _yaml

    repo = Path(__file__).resolve().parents[2]
    path = repo / ".github" / "workflows" / "windows-installer-differential-ci.yml"
    workflow = _yaml.safe_load(path.read_text(encoding = "utf-8"))
    # `on` parses as boolean True in YAML 1.1 (PyYAML).
    triggers = workflow.get("on", workflow.get(True))
    paths = triggers["pull_request"]["paths"]

    body = path.read_text(encoding = "utf-8")
    for script in ("studio/setup.bat", "scripts/uninstall.ps1"):
        if script in paths:
            assert re.search(
                re.escape(Path(script).name) + r"[^\n]*(&|Start-Process|cmd|-File)", body
            ), (
                f"{script} starts this workflow but nothing in it runs the file, so a PR that "
                f"changes only that script gets a PASS from a lane that never read it"
            )
    assert "install.ps1" in paths, "the lane no longer starts on the file it actually measures"


def test_generated_scripts_are_not_normalised_like_console_output() -> None:
    """Console normalisation drops blank lines and rstrips each line, which are real content in a script."""

    def side(body: str) -> dict:
        return {
            "studioHome": "X",
            "files": {
                "unsloth.cmd": {
                    "foundAt": "home/bin\\unsloth.cmd",
                    "content": body,
                    "sha256": "A",
                    "bom": "none",
                }
            },
            "rewrittenOnSecondRun": [],
        }

    for before, after, what in (
        (
            "set UNSLOTH_HOME=C:\\u \r\n",
            "set UNSLOTH_HOME=C:\\u\r\n",
            "a trailing space in a set value",
        ),
        ("@echo off\n\necho hi\n", "@echo off\necho hi\n", "a dropped blank line"),
        ("echo Run the installer\n", "echo Run the setup\n", "a line starting with Run"),
    ):
        verdict = cmp.Verdict()
        cmp.compare_artifacts(side(before), side(after), verdict)
        assert verdict.differences, f"{what} was normalised away and compared equal"

    same = cmp.Verdict()
    cmp.compare_artifacts(
        side("echo C:\\Users\\r\\AppData\\Local\\Temp\\unsloth-aaaaaa\\x\n"),
        side("echo C:\\Users\\r\\AppData\\Local\\Temp\\unsloth-bbbbbb\\x\n"),
        same,
    )
    assert not same.differences, f"a scratch name was not normalised: {same.differences}"


def test_a_reinstall_only_output_change_is_reported(tmp_path: Path) -> None:
    """The comparer must read `transcript-second-run.txt`, or a change to what a reinstall prints passes."""
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head", second = BASELINE + "\n  warning        already installed\n")
    result = _run(base, head)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "second-run transcript" in result.stdout
    assert "already installed" in result.stdout


def test_a_missing_second_run_transcript_is_void_not_a_pass(tmp_path: Path) -> None:
    base = _write(tmp_path / "base")
    head = _write(tmp_path / "head")
    (head / "transcript-second-run.txt").unlink()
    result = _run(base, head)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "second-run transcript" in result.stdout


def test_script_normalisation_does_not_erase_a_changed_port_or_limit(tmp_path: Path) -> None:
    """Transcript port and size tokens must not hide a changed port or limit in a generated script."""
    before = "$url = 'http://127.0.0.1:8888/api/health'\nset LIMIT=10MB\n"
    after = "$url = 'http://127.0.0.1:9999/api/health'\nset LIMIT=20MB\n"
    assert cmp.normalise_script(before) != cmp.normalise_script(after)
    root = "a" * 64
    assert cmp.normalise_script(f"$expected = '{root}'") == cmp.normalise_script(
        "$expected = '" + "b" * 64 + "'"
    )
    assert cmp.normalise_script("$p = 'C:\\Temp\\unsloth-probe-0a1b2c3d.tmp'") == (
        cmp.normalise_script("$p = 'C:\\Temp\\unsloth-probe-9f8e7d6c.tmp'")
    )


def test_a_shortcut_the_second_run_deletes_is_recorded(tmp_path: Path) -> None:
    """Walk the first run's shortcut keys too, or a shortcut the second run deletes is never recorded."""
    sys.path.insert(0, str(REPO / "tests" / "_shared"))
    from unsloth_pwsh_runner import PWSH, run_pwsh  # noqa: PLC0415

    if PWSH is None:
        pytest.skip("no PowerShell on this host")

    collector = REPO / ".github" / "scripts" / "Collect-InstallerEvidence.ps1"
    first = tmp_path / "first-run-artifacts.json"
    first.write_text(
        json.dumps(
            {
                "studioHome": str(tmp_path / "home"),
                "files": {},
                "shortcutWrites": {
                    "UserDesktop/Unsloth Studio.lnk": "2026-01-01T00:00:00.0000000Z"
                },
                "installId": None,
                "embeddedId": None,
            }
        ),
        encoding = "utf-8",
    )
    out = tmp_path / "evidence"
    (tmp_path / "home").mkdir()
    script = f"""
$ErrorActionPreference = 'Stop'
& '{collector.as_posix()}' -StudioHome '{(tmp_path / "home").as_posix()}' `
    -OutDir '{out.as_posix()}' -CompareAgainst '{first.as_posix()}' | Out-Null
Write-Output 'COLLECTOR-OK'
"""
    result = run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output = True,
        text = True,
        verdict = "COLLECTOR-OK",
        timeout = 300,
    )
    assert "COLLECTOR-OK" in result.stdout, result.stdout + result.stderr
    artifacts = json.loads((out / "artifacts.json").read_text(encoding = "utf-8-sig"))
    rewritten = artifacts.get("rewrittenOnSecondRun")
    assert rewritten is not None, "idempotency was not measured at all"
    assert any(
        "removed by the second run" in entry for entry in rewritten
    ), f"a shortcut deleted by the reinstall was not recorded: {rewritten!r}"


def _collect(tmp_path: Path, home_names: list[str], first_files: dict) -> list:
    """Run the collector against a home directory and a hand-made first-run manifest."""
    sys.path.insert(0, str(REPO / "tests" / "_shared"))
    from unsloth_pwsh_runner import PWSH, run_pwsh  # noqa: PLC0415

    if PWSH is None:
        pytest.skip("no PowerShell on this host")

    home = tmp_path / "home"
    home.mkdir()
    for name in home_names:
        (home / name).write_text("x", encoding = "utf-8")
    first = tmp_path / "first-run-artifacts.json"
    first.write_text(
        json.dumps(
            {
                "studioHome": str(home),
                "files": first_files,
                "shortcutWrites": {},
                "installId": None,
                "embeddedId": None,
            }
        ),
        encoding = "utf-8",
    )
    out = tmp_path / "evidence"
    collector = REPO / ".github" / "scripts" / "Collect-InstallerEvidence.ps1"
    result = run_pwsh(
        [
            PWSH,
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"& '{collector.as_posix()}' -StudioHome '{home.as_posix()}' "
            f"-OutDir '{out.as_posix()}' -CompareAgainst '{first.as_posix()}' | Out-Null; "
            f"Write-Output 'COLLECTOR-OK'",
        ],
        capture_output = True,
        text = True,
        verdict = "COLLECTOR-OK",
        timeout = 300,
    )
    assert "COLLECTOR-OK" in result.stdout, result.stdout + result.stderr
    artifacts = json.loads((out / "artifacts.json").read_text(encoding = "utf-8-sig"))
    rewritten = artifacts.get("rewrittenOnSecondRun")
    assert rewritten is not None, "idempotency was not measured at all"
    return rewritten


def test_a_transient_top_level_artifact_is_reported(tmp_path: Path) -> None:
    """Check every top-level name after reinstall, or an artifact that reinstall removes goes unseen."""
    rewritten = _collect(tmp_path, ["kept"], {"kept": {"present": True}, "gone": {"present": True}})
    assert any("gone (removed by the second run)" == e for e in rewritten), rewritten
    assert not any(
        e.startswith("kept ") for e in rewritten
    ), f"a file present on both runs was reported as a change: {rewritten!r}"


def test_an_empty_first_run_map_invents_no_phantom_key(tmp_path: Path) -> None:
    """Member enumeration over an EMPTY `PSObject.Properties` yields `$null`, and `@($null)` has
    Count 1. Without filtering, a first run that recorded nothing produced one key named `''` and
    every clean run reported it as removed, which fails a lane that should pass."""
    rewritten = _collect(tmp_path, [], {})
    assert not any("removed by the second run" in e for e in rewritten), rewritten


def test_an_unrelated_label_does_not_restart_the_two_installs() -> None:
    """The labeled event must check only the label just applied, or cancel-in-progress kills a live run."""
    body = (REPO / ".github" / "workflows" / "windows-installer-differential-ci.yml").read_text(
        encoding = "utf-8"
    )
    gate = body[body.index("name: Skip unless the label asked for it") :]
    gate = gate[: gate.index("name: Pick the two commits")]
    assert (
        "github.event.label.name" in gate
    ), "the gate never reads the label that was applied, so any label restarts two clean installs"
    assert '[ "$ACTION" = "labeled" ]' in gate, "there is no branch for the labeled event"
    assert '"installer-differential"' in gate
    assert gate.index('[ "$ACTION" = "labeled" ]') < gate.index(
        "grep -q"
    ), "the full label list is consulted before the labeled branch, so the branch cannot help"
    # Concurrency is evaluated before any job, so an unrelated label could cancel in-flight installs.
    group = body[body.index("concurrency:") : body.index("env:", body.index("concurrency:"))]
    assert "github.event.label.name" in group, (
        "the concurrency group does not distinguish the label, so an unrelated label still cancels "
        "a measurement in progress before this gate can run"
    )
    assert "unrelated-label" in group
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in group


def test_installer_output_that_looks_like_a_runner_header_survives() -> None:
    """Lines beginning `Run ` are installer output, not runner headers, and must survive normalisation."""
    real = [
        "       Run install.ps1 without --tauri for custom-root shell installs,",
        "    Run 'setx HIP_VISIBLE_DEVICES 1' and reopen your terminal",
        "       Run this manually in an Admin terminal:",
        "  env: the managed environment is ready",
        "  shell: powershell is what this install used",
    ]
    kept = cmp.normalise_transcript("\n".join(real))
    assert len(kept) == len(real), f"normalisation dropped installer output: {kept!r}"
    changed = list(real)
    changed[0] = changed[0].replace("without --tauri", "with --tauri")
    assert cmp.normalise_transcript("\n".join(real)) != cmp.normalise_transcript("\n".join(changed))
    assert cmp.normalise_transcript("##[group]Install\n::endgroup::\n##[debug]x") == []


def test_the_lines_this_rule_used_to_eat_are_really_in_the_installers() -> None:
    """The reason the rule was wrong is a fact about the shipped files, so it is asserted rather
    than described. If these lines ever stop existing the comment above the rule is stale."""
    found = 0
    for name in ("install.ps1", "studio/setup.ps1"):
        text = (REPO / name).read_text(encoding = "utf-8")
        for line in text.splitlines():
            if re.search(r'"\s*Run ', line) and ("Write-StudioLine" in line or "substep" in line):
                found += 1
    assert found >= 5, (
        f"only {found} printed lines start with 'Run '; the normaliser comment cites six and needs "
        f"updating if that changed"
    )


def test_a_manual_dispatch_compares_against_the_default_branch() -> None:
    """A manual dispatch with no input must use the default branch as base, not `HEAD~1`."""
    body = (REPO / ".github" / "workflows" / "windows-installer-differential-ci.yml").read_text(
        encoding = "utf-8"
    )
    step = body[body.index("name: Pick the two commits") :]
    step = step[: step.index("- name:", 10)]
    assert "DEFAULT_BRANCH" in step, "the resolver never looks at the default branch"
    assert (
        'merge-base "$default_tip"' in step
    ), "the manual-dispatch path does not take a merge base against the default branch"
    # HEAD~1 is legitimate only on the default branch or as a warned fallback.
    lines = step.splitlines()
    for i, line in enumerate(lines):
        if "rev-parse HEAD~1" not in line:
            continue
        window = "\n".join(lines[max(0, i - 6) : i])
        assert "::warning::" in window or "previous commit on that branch" in window, (
            "HEAD~1 is used here without saying that it is the previous commit and not the default "
            f"branch:\n{window}\n{line}"
        )


def test_a_default_branch_run_gets_a_distinct_base() -> None:
    """On the default branch, base is the previous commit: the merge base equals head and voids the run."""
    body = (REPO / ".github" / "workflows" / "windows-installer-differential-ci.yml").read_text(
        encoding = "utf-8"
    )
    assert "schedule:" in body, "there is no scheduled run any more, so this test is describing"
    step = body[body.index("name: Pick the two commits") :]
    step = step[: step.index("- name:", 10)]
    # Keyed on the message: the same-commit guard also compares `$base` and `$head`.
    assert "the tip of $DEFAULT_BRANCH" in step, (
        "nothing notices that the merge base resolved to head, so every scheduled run VOIDs before "
        "it reaches the measurement jobs"
    )
    assert step.index("the tip of $DEFAULT_BRANCH") < body.index(
        "base and head are the same commit"
    ), "the same-commit guard runs before the default-branch case is handled"
    after = step[step.index("the tip of $DEFAULT_BRANCH") :]
    assert (
        "rev-parse HEAD~1" in after[:900]
    ), "the default-branch case is detected and then does not pick a different baseline"
    # Compare head with the fetched tip; an ancestor check would substitute HEAD~1 wrongly.
    assert '[ "$head" = "$default_tip" ]' in step, (
        "the health-run baseline is chosen from the merge base equalling head, which is also true "
        "for any commit that is merely an ancestor of the default branch"
    )
    assert "default_tip=" in step, "the default branch tip is never resolved"


def test_an_empty_shortcut_object_is_void_not_one_compared(tmp_path: Path) -> None:
    """An empty shortcut object `{}` must void the run, not count as one `<unnamed>` shortcut compared."""
    base = _write(tmp_path / "base", shortcuts = [{}])
    head = _write(tmp_path / "head", shortcuts = [{}])
    result = _run(base, head)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "no name and no path" in result.stdout, result.stdout


def test_a_shortcut_with_a_name_but_no_launch_contract_is_void(tmp_path: Path) -> None:
    base = _write(
        tmp_path / "base", shortcuts = [{"name": "Unsloth Studio.lnk", "root": "UserDesktop"}]
    )
    head = _write(
        tmp_path / "head", shortcuts = [{"name": "Unsloth Studio.lnk", "root": "UserDesktop"}]
    )
    result = _run(base, head)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "none of the launch contract fields" in result.stdout, result.stdout


def test_a_content_contract_with_no_content_on_either_side_is_void(tmp_path: Path) -> None:
    """A content contract with no `content` on either side must void, not fall through uncompared."""
    artifacts = copy.deepcopy(ARTIFACTS)
    artifacts["files"]["launch-studio.ps1"] = {}
    base = _write(tmp_path / "base", artifacts = artifacts)
    head = _write(tmp_path / "head", artifacts = copy.deepcopy(artifacts))
    result = _run(base, head)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "whose text is the contract" in result.stdout, result.stdout
    ok_base = _write(tmp_path / "ok-base")
    ok_head = _write(tmp_path / "ok-head")
    assert _run(ok_base, ok_head).returncode == 0


def test_a_shortcut_argument_change_is_not_normalised_away(tmp_path: Path) -> None:
    """Shortcut arguments are the launch contract, so the console normaliser must not rewrite them."""

    def _sc(arguments: str) -> list[dict]:
        return [
            {
                "name": "Unsloth Studio.lnk",
                "root": "UserDesktop",
                "targetPath": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
                "arguments": arguments,
                "windowStyle": 7,
            }
        ]

    base = _write(
        tmp_path / "base", shortcuts = _sc("-File launcher.ps1 --limit 10MB --timeout 10.0s")
    )
    head = _write(
        tmp_path / "head", shortcuts = _sc("-File launcher.ps1 --limit 20MB --timeout 30.0s")
    )
    result = _run(base, head)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "arguments" in result.stdout and "20MB" in result.stdout, result.stdout
    same_base = _write(
        tmp_path / "sb", shortcuts = _sc("-File C:\\Temp\\unsloth-probe-0a1b2c3d\\l.ps1")
    )
    same_head = _write(
        tmp_path / "sh", shortcuts = _sc("-File C:\\Temp\\unsloth-probe-9f8e7d6c\\l.ps1")
    )
    assert _run(same_base, same_head).returncode == 0


def test_missing_bom_metadata_is_void_not_skipped(tmp_path: Path) -> None:
    """The collector always writes `bom` for a contract it could read, so its absence is a
    regression, and the same collector writes both manifests, so it goes missing symmetrically."""
    artifacts = copy.deepcopy(ARTIFACTS)
    del artifacts["files"]["launch-studio.ps1"]["bom"]
    base = _write(tmp_path / "base", artifacts = artifacts)
    head = _write(tmp_path / "head", artifacts = copy.deepcopy(artifacts))
    result = _run(base, head)
    assert result.returncode == 3, result.stdout + result.stderr
    assert "carries no bom metadata on base and head" in result.stdout, result.stdout


def test_no_rev_parse_fallback_can_echo_its_argument() -> None:
    """`git rev-parse` echoes an unresolvable argument to stdout, so use `--verify --quiet` in fallbacks."""
    body = (REPO / ".github" / "workflows" / "windows-installer-differential-ci.yml").read_text(
        encoding = "utf-8"
    )
    step = body[body.index("name: Pick the two commits") :]
    step = step[: step.index("- name:", 10)]
    for line in step.splitlines():
        if "rev-parse" not in line or "||" not in line:
            continue
        for call in line.split("||"):
            if "rev-parse" not in call:
                continue
            assert "--verify --quiet" in call, (
                "this rev-parse has a fallback and can echo its argument into the captured value:\n"
                f"{line.strip()}"
            )
