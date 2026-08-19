# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
Every apt step on a hosted runner must be bounded, retried, and paid for.

An unbounded apt step is the worst failure shape CI has. It does not go red: it
sits in apt's download loop, spends the job's entire `timeout-minutes`, and
GitHub reports the result as **cancelled** -- no reason printed, no failing step
named, and every step after it skipped. Three of these were caught by hand in a
single day, on three different workflows:

  Chat UI Tests (chat)        30m18s, the job's whole budget, in `Linux deps`
  Frontend build              16m38s in `playwright install --with-deps`
  Source lint                  5m02s in `Linux deps for shellcheck`, on a 5m job

None of the three reported anything about its actual subject. The last one was
noticed only because a human happened to look at a job that said "cancelled".

So the invariant is not "apt should be retried", it is: **the step is what
bounds itself, never the job**. A job timeout is a backstop for the unforeseen;
using it as the bound on a known-flaky step converts a diagnosable failure into
a silent one. These tests read the workflows and enforce that, plus the two
things that make the bound honest -- that the retry budget actually fits inside
the step timeout, and that the step timeout actually fits inside the job's.

`--with-deps` counts as an apt step, because that is precisely what it is; it
was excluded from the first pass by name and promptly cost a 16-minute job.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
HELPER = ".github/scripts/retry-with-apt-lock.sh"

# PyYAML reads the `on:` key as the boolean True.
ON = True

# apt reached directly, or via playwright, which shells out to it. `--with-deps`
# is qualified by `playwright install` on purpose: scan_packages.py takes a flag
# of the same name that has nothing to do with apt.
_APT = re.compile(r"\bapt-get\s+(?:update|install)\b|playwright\s+install\b[^\n]*--with-deps\b")

# `release_dpkg_lock()` waits up to 24 * 5s for the orphaned holder, then kills it
# and sleeps 5. It runs between attempts, so N attempts pay it N-1 times.
LOCK_WAIT_SECONDS = 125

# Workflows whose apt calls do not run on a hosted runner and cannot use the
# helper. Each is here for a reason that would survive a rewrite, not because it
# was inconvenient to convert.
EXEMPT_WORKFLOWS = {
    # Runs apt inside bare distro containers and inside WSL, as root, with no
    # checkout (the whole point is a machine with no git). There is no repo on
    # disk to read the helper from, and `sudo`/`fuser` are deliberately absent --
    # one leg asserts that `sudo` does not exist on the image at all.
    "clean-machine-install-ci.yml",
}

# Individual steps that reach apt as the thing under test rather than as setup.
# Retrying these would retry an assertion.
EXEMPT_STEPS = {
    # Removes curl to construct the no-transport case. A retry loop around a
    # removal is meaningless, and it is bounded by being a removal.
    ("clean-machine-install-ci.yml", "Take the transport away again"),
    # Installs the built .deb to find out whether the package declares its own
    # runtime dependencies. A failure here is the answer, not a transport
    # hiccup. Its apt *preamble* (update + xvfb) does go through the helper.
    ("desktop-app-clean-machine-ci.yml", "Install with NO dev tooling, only runtime libs"),
    # Greps the documentation for the apt line it tells users to run. It reads an
    # apt command as data; it never executes one.
    ("release-desktop.yml", "Verify desktop updater and Linux package config"),
}


def _workflows() -> list[Path]:
    paths = sorted(WORKFLOWS.glob("*.yml"))
    assert paths, "no workflows found; this guard would pass vacuously"
    return paths


def _steps(doc: dict) -> list[tuple[str, dict, dict]]:
    """(job id, job, step) for every step with a `run:`, across every job."""
    out = []
    for job_id, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        for step in job.get("steps") or []:
            if isinstance(step, dict) and isinstance(step.get("run"), str):
                out.append((job_id, job, step))
    return out


def _apt_steps(path: Path) -> list[tuple[str, dict, dict]]:
    doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
    return [
        (job_id, job, step)
        for job_id, job, step in _steps(doc)
        if _APT.search(step["run"]) and (path.name, step.get("name", "")) not in EXEMPT_STEPS
    ]


def _int_minutes(value: object) -> int | None:
    """A `timeout-minutes:` that is an expression is not a number we can check."""
    return value if isinstance(value, int) else None


def _worst_case_seconds(step: dict, run: str) -> int:
    """Seconds the helper can burn before it gives up, from the step's own budget."""
    env = step.get("env") or {}
    attempts = env.get("RETRY_ATTEMPTS")
    per_attempt = env.get("RETRY_ATTEMPT_TIMEOUT")

    # A step may also set them inline, as `RETRY_ATTEMPTS=3 bash ...helper`, which
    # is how the two steps that embed the call in a larger script pass them.
    if attempts is None:
        inline = re.search(r"RETRY_ATTEMPTS=(\d+)", run)
        attempts = inline.group(1) if inline else None
    if per_attempt is None:
        inline = re.search(r"RETRY_ATTEMPT_TIMEOUT=(\d+)", run)
        per_attempt = inline.group(1) if inline else None

    # The helper's own defaults, kept in sync by test_helper_defaults_are_what_the_budgets_assume.
    attempts = int(attempts) if attempts is not None else 3
    per_attempt = int(per_attempt) if per_attempt is not None else 480

    calls = run.count(HELPER)
    return calls * (attempts * per_attempt + (attempts - 1) * LOCK_WAIT_SECONDS)


@pytest.mark.parametrize("path", _workflows(), ids = lambda p: p.name)
def test_every_apt_step_goes_through_the_shared_helper(path: Path) -> None:
    if path.name in EXEMPT_WORKFLOWS:
        return
    for job_id, _job, step in _apt_steps(path):
        assert HELPER in step["run"], (
            f"{path.name}: job '{job_id}' step '{step.get('name', '<unnamed>')}' "
            f"reaches apt without going through {HELPER}. Unbounded, it will spend "
            f"the job's whole timeout and be reported as 'cancelled' with no reason "
            f"and every later step skipped."
        )


@pytest.mark.parametrize("path", _workflows(), ids = lambda p: p.name)
def test_every_apt_step_bounds_itself(path: Path) -> None:
    if path.name in EXEMPT_WORKFLOWS:
        return
    for job_id, _job, step in _apt_steps(path):
        assert step.get("timeout-minutes") is not None, (
            f"{path.name}: job '{job_id}' step '{step.get('name', '<unnamed>')}' "
            f"has no timeout-minutes, so the job timeout is what bounds it -- which "
            f"is the failure mode this guard exists for, retry helper or not."
        )


@pytest.mark.parametrize("path", _workflows(), ids = lambda p: p.name)
def test_the_retry_budget_fits_inside_the_step_timeout(path: Path) -> None:
    """
    A step timeout smaller than the retries it authorises silently deletes the
    last attempt, and reports a truncated step rather than the helper's warnings.
    """
    if path.name in EXEMPT_WORKFLOWS:
        return
    for job_id, _job, step in _apt_steps(path):
        budget = _int_minutes(step.get("timeout-minutes"))
        if budget is None:
            continue
        worst = _worst_case_seconds(step, step["run"])
        assert worst <= budget * 60, (
            f"{path.name}: job '{job_id}' step '{step.get('name', '<unnamed>')}' "
            f"authorises up to {worst}s of retries but is cut off at "
            f"{budget * 60}s, so the final attempt can never finish."
        )


@pytest.mark.parametrize("path", _workflows(), ids = lambda p: p.name)
def test_the_step_timeout_fits_inside_the_job_timeout(path: Path) -> None:
    """
    Otherwise the job timeout still fires first and the diagnosis is still lost:
    the step's bound only helps if the job is alive to report it.
    """
    if path.name in EXEMPT_WORKFLOWS:
        return
    for job_id, job, step in _apt_steps(path):
        step_budget = _int_minutes(step.get("timeout-minutes"))
        job_budget = _int_minutes(job.get("timeout-minutes"))
        if step_budget is None or job_budget is None:
            continue
        assert step_budget < job_budget, (
            f"{path.name}: job '{job_id}' allows {job_budget}m but its apt step "
            f"'{step.get('name', '<unnamed>')}' alone may take {step_budget}m, so "
            f"the job timeout fires first and the run is reported as 'cancelled' "
            f"with no step named."
        )


@pytest.mark.parametrize("path", _workflows(), ids = lambda p: p.name)
def test_a_workflow_that_calls_the_helper_reruns_when_the_helper_changes(path: Path) -> None:
    """
    A paths-filtered workflow that calls the helper but does not list it is not
    covered by an edit to it: the helper could be broken and every consumer would
    keep showing the last green run.
    """
    doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
    if not any(HELPER in step["run"] for _, _, step in _steps(doc)):
        return
    triggers = doc.get(ON) or {}
    if not isinstance(triggers, dict):
        return
    for event, spec in triggers.items():
        if not isinstance(spec, dict):
            continue
        paths = spec.get("paths")
        if paths is None:
            continue  # unfiltered: it runs for every change, helper included
        assert HELPER in paths, (
            f"{path.name}: the '{event}' trigger filters on paths but does not "
            f"list {HELPER}, which its steps call. A change to the helper would "
            f"not run this workflow."
        )


def test_helper_defaults_are_what_the_budgets_assume() -> None:
    """
    The worst-case arithmetic above falls back to the helper's own defaults for a
    step that sets neither variable. If those defaults move and this fallback does
    not, every such budget check silently starts measuring the wrong number.
    """
    source = (REPO_ROOT / HELPER).read_text(encoding = "utf-8")
    assert 'ATTEMPTS="${RETRY_ATTEMPTS:-3}"' in source
    assert 'ATTEMPT_TIMEOUT="${RETRY_ATTEMPT_TIMEOUT:-480}"' in source
    # 24 sleeps of 5s, then a kill and a final 5s.
    assert "seq 1 24" in source
    assert LOCK_WAIT_SECONDS == 24 * 5 + 5


def test_the_helper_makes_apt_fail_fast() -> None:
    """
    The bound is the backstop; this is the part that stops the stall happening.

    apt's `Acquire::http::Timeout` defaults to 120s and is an *idle* timeout, so a
    socket that is open and trickling never trips it. That is how a 126 kB index
    file cost 29 minutes: apt did not consider waiting on it an error at all.
    Without these four options the helper still works, but every stall costs the
    full step budget and ends in a kill -- and the kill is what orphans the dpkg
    lock, so an apt that gives up on its own is one we never have to kill.
    """
    source = (REPO_ROOT / HELPER).read_text(encoding = "utf-8")

    # The generated config only, not the whole file. Searching the file matches the
    # header comment, which names these options while explaining them -- so deleting
    # the line that actually sets one would have gone unnoticed. It did, on the
    # first version of this test.
    written = re.search(r'conf="(.*?)"\n', source, re.DOTALL)
    assert written, f"{HELPER} no longer builds an apt config to write"
    conf = written.group(1)

    for option in (
        "Acquire::Retries",
        "Acquire::http::Timeout",
        "Acquire::https::Timeout",
        "Acquire::ftp::Timeout",
    ):
        assert option in conf, f"{HELPER} no longer sets {option}; it writes: {conf!r}"

    # Well under apt's 120s default, or it buys nothing. The whole point is that a
    # dead transfer is abandoned in seconds rather than minutes.
    match = re.search(r'APT_TIMEOUT="\$\{APT_ACQUIRE_TIMEOUT:-(\d+)\}"', source)
    assert match, f"{HELPER} does not define a default transfer timeout"
    assert int(match.group(1)) <= 30, (
        f"a {match.group(1)}s transfer timeout is close enough to apt's 120s "
        f"default that a stalled mirror still eats the step budget"
    )

    # Ordering is the whole value: configured inside or after the retry loop, the
    # first attempt -- the one that actually stalls -- runs unconfigured.
    configure = source.index("configure_apt_fail_fast\n\nfor attempt")
    loop = source.index('for attempt in $(seq 1 "$ATTEMPTS")')
    assert configure < loop, (
        f"{HELPER} configures apt at or after the retry loop, so the first "
        f"attempt runs with apt's 120s idle timeout"
    )

    # Best effort: a runner that cannot write the file must still run the command.
    # `set -e` is absent here by design, but a bare failing `tee` would still leave
    # the function returning non-zero into a caller that may have it set.
    assert "could not write" in source, (
        f"{HELPER} does not handle an unwritable apt config, so a runner that "
        f"refuses the write would fail before running the command at all"
    )


def test_the_guard_is_not_vacuous() -> None:
    """
    Every assertion above is a for-loop over steps this finds. If the detector
    stopped matching, all of them would pass by finding nothing.
    """
    found = {path.name: len(_apt_steps(path)) for path in _workflows() if _apt_steps(path)}
    assert (
        len(found) >= 10
    ), f"only {len(found)} workflows matched; the detector looks broken: {found}"
    assert sum(found.values()) >= 15, f"only {sum(found.values())} apt steps matched: {found}"
