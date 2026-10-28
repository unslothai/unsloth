# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Folding phases into one macOS job shares ports, logs and outcomes, so each hazard is pinned."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "studio-mac-ui-smoke.yml"
BOOT_SCRIPT = REPO / ".github" / "scripts" / "boot-studio-api-only.sh"


def _boot_defaults() -> tuple[str, str]:
    """Reads LOG and PID_VAR defaults from boot-studio-api-only.sh, since an omitted --log is invisible."""
    src = BOOT_SCRIPT.read_text(encoding = "utf-8")
    log = re.search(r'^LOG="([^"]+)"', src, flags = re.M)
    pid = re.search(r'^PID_VAR="([^"]+)"', src, flags = re.M)
    assert log, f"{BOOT_SCRIPT.name} no longer sets a default LOG; this scan is blind"
    return log.group(1), pid.group(1) if pid else "STUDIO_PID"


PHASE_MARKERS = (
    "Drive the chat UI with Playwright",
    "Run Unsloth API & Auth tests",
    "Multi-turn determinism via OpenAI + Anthropic SDKs",
    "Tool calling, server-side tools, thinking on/off",
    "JSON schema decoding + image input",
    "Uninstall and verify clean",
)


@pytest.fixture(scope = "module")
def job() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    jobs = doc["jobs"]
    assert len(jobs) == 1, f"expected one bundled job, got {list(jobs)}"
    return next(iter(jobs.values()))


@pytest.fixture(scope = "module")
def steps(job: dict) -> list[dict]:
    return job["steps"]


def _script(step: dict) -> str:
    return step.get("run") or ""


def _phase_starts(steps: list[dict]) -> list[int]:
    """Phase starts match 'boot unsloth' only; 'Pass bootstrap password' steps would split a phase."""
    names = [str(s.get("name") or "") for s in steps]
    boots = [i for i, n in enumerate(names) if "boot unsloth" in n.lower()]
    assert boots, "no server boot step found; every scan below would be vacuous"

    # Phases declare their port in a "Phase N environment" step ahead of the boot, so the
    # boundary is drawn there.
    declarations = [i for i, n in enumerate(names) if re.fullmatch(r"Phase \d+ environment", n)]

    starts: list[int] = []
    previous = -1
    for boot in boots:
        candidates = [d for d in declarations if previous < d < boot]
        starts.append(candidates[0] if candidates else boot)
        previous = boot
    return starts


def _phase_of(starts: list[int], index: int) -> int:
    return max([b for b in starts if b <= index], default = -1)


def test_the_bundle_still_carries_every_phase(steps: list[dict]) -> None:
    """A scan that found no phases would pass every check below."""
    names = [str(s.get("name") or s.get("uses") or "") for s in steps]
    blob = "\n".join(names)
    for marker in PHASE_MARKERS:
        assert marker in blob, (
            f"{WORKFLOW.name} no longer runs {marker!r}. Four workflows were folded "
            f"into this job; a phase that quietly leaves takes its whole surface with "
            f"it and nothing else covers it."
        )


def test_the_uninstall_phase_runs_last(steps: list[dict]) -> None:
    """The uninstall step must be last in the job, or later steps run with no Unsloth installed."""
    names = [str(s.get("name") or "") for s in steps]
    uninstall = next(i for i, n in enumerate(names) if n == "Uninstall and verify clean")
    after = [n for n in names[uninstall + 1 :] if n]
    assert all("Upload" in n for n in after), (
        f"steps run after the uninstall phase: {after}. That phase removes Unsloth, so "
        f"anything below it that needs an install now runs against a machine it just "
        f"deleted."
    )


def test_no_two_phases_bind_the_same_port(steps: list[dict]) -> None:
    """A shared port is harmless only in step order; a leftover server makes a phase test the wrong
    model."""
    # Group by phase: one phase names its port in several steps.
    starts = _phase_starts(steps)
    by_phase: dict[str, set[int]] = defaultdict(set)
    for i, step in enumerate(steps):
        text = _script(step) + "\n" + yaml.safe_dump(step.get("env") or {})
        for found in re.findall(r"\b(188\d\d)\b", text):
            by_phase[found].add(_phase_of(starts, i))

    assert by_phase, "no ports found; this scan would be vacuous"
    collisions = {port: sorted(phases) for port, phases in by_phase.items() if len(phases) > 1}
    assert not collisions, (
        f"these ports are used by more than one phase of the bundled job "
        f"(values are the index of each phase's boot step): {collisions}. Give each "
        f"phase its own port; a phase that reaches a server another phase left behind "
        f"reports a pass against the wrong model."
    )


def test_no_two_phases_write_the_same_server_log(steps: list[dict]) -> None:
    """A shared server log path lets a later phase truncate the earlier phase's log before upload."""
    # Grouped by phase: the health wait is given the log path to read, not write.
    starts = _phase_starts(steps)
    default_log, _ = _boot_defaults()
    logs: dict[str, set[int]] = defaultdict(set)
    for i, step in enumerate(steps):
        script = _script(step)
        for pattern in (r"--log (logs/[\w.\-]+)", r"> (?:\")?(logs/[\w.\-]+)"):
            for found in re.findall(pattern, script):
                logs[found].add(_phase_of(starts, i))
        # An invocation with no --log writes the default file; checked over the whole step since
        # calls are backslash-continued.
        if "boot-studio-api-only.sh" in script and "--log" not in script:
            logs[default_log].add(_phase_of(starts, i))

    assert logs, "no server log targets found; this scan would be vacuous"
    collisions = {path: sorted(phases) for path, phases in logs.items() if len(phases) > 1}
    assert not collisions, (
        f"more than one phase writes these server logs (values are the index of each "
        f"phase's boot step): {collisions}. The second truncates the first, so the "
        f"uploaded artifact describes the wrong phase."
    )


def test_every_absorbed_phase_step_says_when_it_runs(steps: list[dict]) -> None:
    """Without an explicit `if:` a step gets a job-wide implicit success(), so one flaky run skips it."""
    names = [str(s.get("name") or "") for s in steps]
    start = names.index("Phase 1 environment")
    end = names.index("First update should be a no-op (prebuilt already validated)")

    ungated = [n for s, n in zip(steps[start:end], names[start:end]) if not s.get("if")]
    assert not ungated, (
        f"absorbed inference steps with no `if:`: {ungated}. Each inherits a job-wide "
        f"implicit success(), so a failure in any earlier phase skips them and the run "
        f"still reports green."
    )


def test_the_absorbed_phases_keep_the_host_offload_opt_out(job: dict) -> None:
    """Set at job level so later phases inherit the opt-out; without it the load returns HTTP 400."""
    assert (job.get("env") or {}).get("UNSLOTH_ALLOW_HOST_OFFLOAD") == "1", (
        "the bundled Mac job no longer opts out of the #8883 host-offload guard. "
        "GitHub's macOS runners have a paravirtual Metal device, so every phase here "
        "runs the whole model from host RAM and the guard declines the load."
    )
