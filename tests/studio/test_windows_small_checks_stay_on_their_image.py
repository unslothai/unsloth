# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows checks must stay on the image under test; the VC++ phase must run last on any shared box."""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-windows-inference-smoke.yml"
JOB = "small-checks"

# Kept here deliberately: a test reading this from the workflow could not catch it being wrong.
EXPECTED_IMAGES = {
    "pester": {"windows-latest"},
    "no-vs-gpu": {"windows-latest"},
    "vcredist": {"windows-latest", "windows-2025-vs2026"},
    "vs-detect": {"windows-2022", "windows-2025-vs2026"},
}


def _job() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"][JOB]


def _cells() -> list[dict]:
    return _job()["strategy"]["matrix"]["include"]


def _phases(cell: dict) -> list[str]:
    return str(cell["phases"]).split()


def _tokens_in(cond: str) -> set[str]:
    return set(re.findall(r"contains\(matrix\.phases, ' ([a-z-]+) '\)", cond))


@pytest.mark.parametrize("phase", sorted(EXPECTED_IMAGES))
def test_each_phase_runs_on_exactly_the_images_it_validates(phase):
    got = {str(c["os"]) for c in _cells() if phase in _phases(c)}
    assert got == EXPECTED_IMAGES[phase], (
        f"{phase} runs on {sorted(got)}, expected {sorted(EXPECTED_IMAGES[phase])}. Windows "
        f"checks exist to prove behaviour on a specific Windows image; moving one to another "
        f"image usually still passes, which is why this is asserted rather than observed."
    )


def test_every_cell_is_a_windows_image():
    for cell in _cells():
        assert str(cell["os"]).startswith("windows-"), cell


def test_no_phase_is_stranded_and_no_cell_is_idle():
    declared = {p for c in _cells() for p in _phases(c)}
    assert declared == set(EXPECTED_IMAGES), (
        f"the matrix declares {sorted(declared)} but the contract covers "
        f"{sorted(EXPECTED_IMAGES)}"
    )
    used = set()
    for step in _job()["steps"]:
        used |= _tokens_in(str(step.get("if", "")))
    stranded = declared - used
    assert not stranded, (
        f"these phases are named in the matrix but gate no step, so a cell asks for work "
        f"that does not exist: {sorted(stranded)}"
    )
    unreachable = used - declared
    assert not unreachable, (
        f"these steps gate on a phase no cell declares, so they run nowhere and cannot "
        f"fail: {sorted(unreachable)}"
    )


def test_every_step_is_gated_so_a_cell_runs_only_its_own_phases():
    ungated = [
        s.get("name") or s.get("uses")
        for s in _job()["steps"]
        if "matrix.phases" not in str(s.get("if", "")) and "checkout" not in str(s.get("uses", ""))
    ]
    assert not ungated, f"these steps carry no phase gate and would run on every image: {ungated}"


def test_the_vcredist_round_trip_is_the_last_phase_on_any_box_it_shares():
    """It uninstalls the VC++ runtime and restores the registry in a `finally`."""
    steps = _job()["steps"]
    last_vcredist = max(
        i for i, s in enumerate(steps) if "vcredist" in _tokens_in(str(s.get("if", "")))
    )
    after = [
        s.get("name")
        for s in steps[last_vcredist + 1 :]
        if _tokens_in(str(s.get("if", ""))) - {"vcredist"}
    ]
    assert not after, (
        f"these phases run after the VC++ round-trip has torn down and restored the "
        f"runtime, so they see a perturbed machine: {after}"
    )


def test_a_failing_phase_still_lets_the_others_on_that_box_report():
    """Each phase's run: step must let the other phases on its box still report; setup-python is exempt."""
    offenders = [
        s.get("name") or s.get("uses")
        for s in _job()["steps"]
        if "matrix.phases" in str(s.get("if", ""))
        and "run" in s
        and not ("cancelled()" in str(s["if"]) or "always()" in str(s["if"]))
    ]
    assert not offenders, (
        f"these steps inherit the implicit success(), so an earlier phase failing on the "
        f"same box silently skips them: {offenders}"
    )


def test_the_merged_job_did_not_absorb_the_two_long_jobs():
    """inference-smoke and no-vs-cpu stay separate; merged, they would queue behind seconds-long checks."""
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    assert {"inference-smoke", "no-vs-cpu"} <= set(jobs), sorted(jobs)
