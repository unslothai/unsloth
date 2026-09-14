# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The three GGUF smoke phases share a runner without sharing anything else.

Unsloth GGUF CI used to be three ubuntu-latest jobs. It is now one job with three
sequential phases, matching what Windows and macOS already do. Three jobs gave a few
properties for free that a single job has to reproduce by hand, and each of them fails
silently rather than loudly:

  * A phase-1 failure must not skip phases 2 and 3. A step with no `if:` rides the
    implicit `success()`, which is exactly that skip -- and the run stays green-looking
    because skipped steps are not failures.
  * Two phases must not share a port, an HF_HOME, a server log or an artifact name. The
    symptom of a shared port is phase 2 asserting against phase 1's still-running server;
    the symptom of a shared log is an artifact that describes the wrong phase.
  * A step that can block must carry its own `timeout-minutes`. The job cap is not a
    substitute: `hf-download-with-retry.sh` retries forever by design and names the
    enclosing step's timeout as its only bound, so one stalled download would eat the
    whole job and take the other two phases' results with it.

Every assertion below reads the workflow rather than a list kept here, so adding a fourth
phase is caught by the same checks that guard the three.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-inference-smoke.yml"
JOB = "inference-smoke"

# The step that every phase gates on, directly or through the SDK install that follows it.
SHARED = ("steps.install", "steps.sdks")


def _steps() -> list[dict]:
    job = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"][JOB]
    return job["steps"]


def _phase_env() -> list[dict]:
    return [s for s in _steps() if str(s.get("name", "")).startswith("Phase ")]


def _phase_of(index: int, steps: list[dict]) -> int:
    """0 for the shared preamble, else the 1-based number of the phase in force."""
    phase = 0
    for i, step in enumerate(steps):
        name = str(step.get("name", ""))
        if name.startswith("Phase "):
            phase = int(name.split()[1])
        if i == index:
            return phase
    raise AssertionError(index)


def _exports(step: dict) -> dict[str, str]:
    """The KEY=VALUE pairs a `Phase N environment` step writes to $GITHUB_ENV."""
    found = re.findall(r'echo "([A-Z_]+)=(.*?)" >> "\$GITHUB_ENV"', step.get("run", ""))
    return dict(found)


def test_the_workflow_is_a_single_job():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    assert list(jobs) == [JOB], (
        f"Unsloth GGUF CI is meant to be one bundled job; found {list(jobs)}. Splitting it "
        f"back out returns two of the 60 concurrent slots to a pool that was measured "
        f"handing out staggered starts (median 175s between a run's first and last job)."
    )


def test_there_are_three_phases():
    names = [s["name"] for s in _phase_env()]
    assert len(names) == 3, names


@pytest.mark.parametrize("key", ["STUDIO_PORT", "HF_HOME", "STUDIO_LOG"])
def test_no_two_phases_share_a(key):
    values = [_exports(s).get(key) for s in _phase_env()]
    assert all(values), f"a phase never sets {key}: {values}"
    assert len(set(values)) == len(values), f"phases collide on {key}: {values}"


def test_every_phase_uploads_under_its_own_artifact_name():
    names = [s["with"]["name"] for s in _steps() if "upload-artifact" in str(s.get("uses", ""))]
    assert len(names) == 3, names
    assert len(set(names)) == 3, f"artifact names collide: {names}"


def test_every_phase_uploads_the_log_that_phase_actually_wrote():
    steps = _steps()
    logs = {
        _phase_of(i, steps): _exports(s)["STUDIO_LOG"]
        for i, s in enumerate(steps)
        if str(s.get("name", "")).startswith("Phase ")
    }
    for i, step in enumerate(steps):
        if "upload-artifact" not in str(step.get("uses", "")):
            continue
        phase = _phase_of(i, steps)
        want = logs[phase]
        paths = str(step["with"]["path"])
        assert want in paths, (
            f"phase {phase} boots its server with --log {want} but its artifact uploads "
            f"{paths!r}. The upload would carry another phase's log."
        )


def test_no_step_rides_the_implicit_success_into_skipping_a_later_phase():
    steps = _steps()
    offenders = []
    for i, step in enumerate(steps):
        if _phase_of(i, steps) == 0:
            continue  # the shared preamble is meant to fail-fast
        if "if" not in step:
            offenders.append(step.get("name") or step.get("uses"))
    assert not offenders, (
        "these phase steps carry no `if:`, so they inherit the implicit success() and a "
        f"failure in an earlier phase silently skips them: {offenders}"
    )


def test_every_phase_gates_on_the_shared_preamble_not_on_the_phase_before_it():
    steps = _steps()
    for i, step in enumerate(steps):
        phase = _phase_of(i, steps)
        cond = str(step.get("if", ""))
        if phase == 0 or "always()" in cond:
            continue
        assert any(s in cond for s in SHARED), (
            f"{step.get('name')!r} (phase {phase}) does not gate on the shared install: "
            f"{cond!r}"
        )


def test_step_ids_are_unique_and_every_reference_resolves():
    steps = _steps()
    ids = [s["id"] for s in steps if "id" in s]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, (
        f"duplicate step ids {duplicates}: `steps.<id>.outcome` then silently reads "
        f"whichever one Actions picked, and a phase gate can be answered by another "
        f"phase's step."
    )
    referenced = set(re.findall(r"steps\.([A-Za-z0-9_-]+)\.", yaml.dump(steps)))
    assert not referenced - set(ids), f"dangling step references: {referenced - set(ids)}"


def test_every_step_that_can_block_carries_its_own_timeout():
    steps = _steps()
    # A step is "blocking" if it downloads a model, waits on the server, or drives it.
    blocking = re.compile(r"hf-download-with-retry|wait-for-health|curl |seq 1 ")
    offenders = [
        step.get("name")
        for i, step in enumerate(steps)
        if _phase_of(i, steps) > 0
        and blocking.search(str(step.get("run", "")))
        and "timeout-minutes" not in step
        and "always()" not in str(step.get("if", ""))
    ]
    assert not offenders, (
        "hf-download-with-retry.sh retries forever and names the enclosing step's "
        f"timeout as its bound, so these need a `timeout-minutes:`: {offenders}"
    )


def test_the_cross_os_gemma_cache_entry_is_left_alone():
    """Phase 1's cache is byte-shared with the macOS and Windows gemma phases.

    actions/cache identifies an entry by key AND a hash of `path`, so renaming the
    directory on Linux alone would give a permanent restore miss on all three platforms
    without any of them failing -- they would just quietly re-download every run.
    """
    steps = _steps()
    entries = {
        str(s["with"]["path"]): str(s["with"]["key"])
        for s in steps
        if "actions/cache" in str(s.get("uses", ""))
    }
    assert entries.get("hf-cache", "").endswith("-v3"), entries
    assert "runner.os" not in entries.get(
        "hf-cache", ""
    ), "phase 1's key gained a runner.os scope, which un-shares it from macOS and Windows"
    assert len(entries) == 3, f"expected one cache path per phase, got {entries}"
