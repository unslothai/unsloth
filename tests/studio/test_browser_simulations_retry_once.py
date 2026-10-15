# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Retries a flaky browser driver once; a driver that fails twice must still fail the leg."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "studio-composer-compatibility.yml"
)

DRIVERS = (
    "playwright_prompt_queue_actions.py",
    "playwright_composer_settings.py",
    "playwright_queue_localization.py",
)


def _step() -> str:
    document = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    for job in document["jobs"].values():
        for step in job.get("steps", []):
            if step.get("name") == "Browser simulations":
                return step["run"]
    raise AssertionError(f"{WORKFLOW.name} has no `Browser simulations` step")


def test_the_step_defines_the_retry_helper():
    assert "run_driver()" in _step(), (
        "`Browser simulations` no longer defines run_driver, so a single lost browser "
        "target fails the leg for whichever pull request happened to be running"
    )


@pytest.mark.parametrize("driver", DRIVERS)
def test_every_driver_goes_through_the_helper(driver):
    """A driver called directly is one that still fails on the first transient."""
    step = _step()
    for line in step.splitlines():
        if driver not in line:
            continue
        body = step[: step.index(line)]
        assert body.rstrip().endswith("\\") or line.strip().startswith("run_driver"), (
            f"{driver} is invoked without run_driver, so it gets no retry while the "
            f"other drivers in the same loop do"
        )
        break
    else:
        raise AssertionError(f"{driver} is not invoked by the step at all")


@pytest.mark.parametrize("driver", DRIVERS)
def test_every_driver_is_still_spelled_as_a_python_invocation(driver):
    """tests/studio/test_playwright_suites_run_in_ci.py reads this text for
    `python <driver>`, so passing the command through the helper rather than naming it
    inside is what keeps that guard able to see these three."""
    assert f"python tests/studio/{driver}" in _step(), (
        f"{driver} is no longer spelled as a python invocation in the workflow, so the "
        f"suites-run-in-CI guard can no longer tell that it runs"
    )


def test_a_second_failure_still_fails_the_leg():
    """The retry is a second chance, not an amnesty."""
    step = _step()
    assert "return 1" in step, "run_driver never reports failure, so nothing can go red"
    assert step.count("|| status=1") == len(DRIVERS), (
        f"expected each of the {len(DRIVERS)} drivers to set status=1 on a failed "
        f"retry; found {step.count('|| status=1')}"
    )
    assert 'exit "$status"' in step, "the step no longer exits on a failed driver"


def test_the_retry_is_bounded_at_one():
    """A loop here would turn a real regression into a slow green."""
    step = _step()
    helper = step[step.index("run_driver()") : step.index("for browser in")]
    attempts = helper.count('"$@"')
    assert attempts == 2, (
        f"run_driver runs the command {attempts} times; it must be exactly twice, once "
        f"and one retry, or a genuinely broken driver is only a slower green"
    )
    for looping in ("while", "until", "for attempt", "seq "):
        assert (
            looping not in helper
        ), f"run_driver contains `{looping}`: the retry has to stay bounded at one"


def test_the_retry_records_the_browser_process():
    """A second TargetClosedError should say why the browser went away.

    Only the retry carries DEBUG=pw:browser: a first-attempt pass stays quiet, and the
    attempt that decides the leg logs the browser's launch, stderr and exit.
    """
    step = _step()
    helper = step[step.index("run_driver()") : step.index("for browser in")]
    first, second = [line.strip() for line in helper.splitlines() if '"$@"' in line]
    assert "DEBUG=" not in first, "the first attempt should run exactly as before"
    assert second.startswith('if DEBUG=pw:browser "$@"'), (
        "the retry no longer records the browser process, so a twice-closed target "
        "leaves no reason in the artifact"
    )
