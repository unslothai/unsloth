# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
The Mac bundle now carries four workflows' worth of phases in one job.

Concurrent macOS jobs are capped at 5 account-wide and that pool is shared with
unslothai/unsloth-zoo, so the queue, not the execution, is what a macOS slot
costs: measured over the last 8 green main runs, the UI job executed 1154s
behind a 17438s queue and the inference job 402s behind a 13252s queue. Folding
the second into the first returns a slot.

What that buys in queue it risks in isolation. Four phases that used to be four
runners are now steps in one job, sharing a filesystem, a port space, an
`$GITHUB_ENV` and a step-outcome graph. Each of the tests below is a way two
phases can quietly stop testing what their name says while the job stays green:

  - two phases on one port, where the second talks to the first's server;
  - two phases on one log file, where the second erases the evidence of the
    first's failure before the artifact upload runs;
  - a phase with no `if:`, which inherits an implicit `success()` that now means
    "every step of every earlier phase passed" rather than "the install worked";
  - the uninstall phase stopping being last, which would leave the phases after
    it with no Unsloth installed.

None of those is loud. Ports and logs collide silently, an implicit `success()`
reports as a skip rather than a failure, and a phase running after the uninstall
fails with an error that names neither the uninstall nor the ordering.
"""

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
    """
    The log path and PID variable boot-studio-api-only.sh uses when not told.

    Read from the script rather than written down here, because the whole point
    of the scan below is that an omitted `--log` is invisible: the collision this
    guard exists to catch was two phases both taking this default, and neither
    workflow line mentioned a file at all.
    """
    src = BOOT_SCRIPT.read_text(encoding = "utf-8")
    log = re.search(r'^LOG="([^"]+)"', src, flags = re.M)
    pid = re.search(r'^PID_VAR="([^"]+)"', src, flags = re.M)
    assert log, f"{BOOT_SCRIPT.name} no longer sets a default LOG; this scan is blind"
    return log.group(1), pid.group(1) if pid else "STUDIO_PID"


# The phases, in the order they must run.
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
    """
    Indices of the steps that boot a server, which is what delimits a phase.

    Matched on "boot Unsloth" rather than "boot": several steps in this job are
    named "Pass bootstrap password ...", and treating one of those as a phase
    boundary splits a phase in half and reports its own port as a collision.
    """
    names = [str(s.get("name") or "") for s in steps]
    boots = [i for i, n in enumerate(names) if "boot unsloth" in n.lower()]
    assert boots, "no server boot step found; every scan below would be vacuous"

    # The absorbed phases declare their port and model in a "Phase N environment" step several steps ahead of the boot,
    # so a boundary drawn at the boot alone files that port under the PREVIOUS phase and reports a collision against
    # itself.
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
    """
    It uninstalls Unsloth and asserts the machine is clean, which is the teardown
    for the whole job. Anything needing an install after it fails for a reason
    that names neither the uninstall nor the ordering.
    """
    names = [str(s.get("name") or "") for s in steps]
    uninstall = next(i for i, n in enumerate(names) if n == "Uninstall and verify clean")
    after = [n for n in names[uninstall + 1 :] if n]
    # Artifact upload is the only legitimate follower: it needs no install.
    assert all("Upload" in n for n in after), (
        f"steps run after the uninstall phase: {after}. That phase removes Unsloth, so "
        f"anything below it that needs an install now runs against a machine it just "
        f"deleted."
    )


def test_no_two_phases_bind_the_same_port(steps: list[dict]) -> None:
    """
    The phases boot servers in sequence and each kills its own, so a shared port
    is harmless only for as long as the step order stays exactly as it is. That
    is a property of the ordering, and the ordering is the thing an edit changes.
    A phase that finds a previous phase's server still listening does not error:
    it connects, and tests the wrong model.
    """
    # A port legitimately appears several times inside ONE phase: the boot step, the health wait and
    # the stop step all name it. So group by phase, not by step, and fail only when two phases share one.
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
    """
    The artifact upload publishes these by name. Two phases sharing one path means
    the later phase truncates the earlier one's log, so a run that went red in an
    early phase uploads the log of a later phase that passed.
    """
    # Grouped by phase for the same reason the port scan is: within one phase the health wait is
    # *given* the log path so it can tail it on failure, which is a read, not a second writer.
    starts = _phase_starts(steps)
    default_log, _ = _boot_defaults()
    logs: dict[str, set[int]] = defaultdict(set)
    for i, step in enumerate(steps):
        script = _script(step)
        for pattern in (r"--log (logs/[\w.\-]+)", r"> (?:\")?(logs/[\w.\-]+)"):
            for found in re.findall(pattern, script):
                logs[found].add(_phase_of(starts, i))
        # An invocation with no --log is the case that actually bit: neither workflow line named a
        # file, so a text scan saw no collision while both phases wrote the same one. Checked over
        # the whole step rather than the matched call, since the invocations are backslash-continued
        # across lines.
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
    """
    A step with no `if:` gets an implicit `success()`, which is job-wide. When these
    phases were their own workflows that meant "the install worked". Bundled behind
    the UI and API phases it means "and every Playwright test passed", so one flaky
    browser run silently drops all the inference coverage -- as a skip, which reads
    green.
    """
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
    """
    Set at job level so a phase added later inherits it. Without it the load
    returns HTTP 400 and the probe reports an unexpected status several layers
    from the cause -- which is how the first draft of this bundle broke.
    """
    assert (job.get("env") or {}).get("UNSLOTH_ALLOW_HOST_OFFLOAD") == "1", (
        "the bundled Mac job no longer opts out of the #8883 host-offload guard. "
        "GitHub's macOS runners have a paravirtual Metal device, so every phase here "
        "runs the whole model from host RAM and the guard declines the load."
    )
