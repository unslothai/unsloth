# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Each apt step must bound itself, since an unbounded one stalls silently until the job is cancelled."""

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

# apt directly, or via `playwright install --with-deps` (scan_packages.py has an unrelated
# flag of the same name).
_APT = re.compile(r"\bapt-get\s+(?:update|install)\b|playwright\s+install\b[^\n]*--with-deps\b")

# `release_dpkg_lock()` waits up to 24 * 5s, then kills the holder and sleeps 5.
LOCK_WAIT_SECONDS = 125

# Workflows whose apt calls cannot use the helper, each for a durable reason.
EXEMPT_WORKFLOWS = {
    # Runs apt in bare containers and WSL with no checkout, and deliberately without `sudo`.
    "clean-machine-install-ci.yml",
}

# Steps that run apt as the thing under test.
EXEMPT_STEPS = {
    ("clean-machine-install-ci.yml", "Take the transport away again"),
    ("desktop-app-clean-machine-ci.yml", "Install with NO dev tooling, only runtime libs"),
    # Reads an apt command as data; never executes one.
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

    # Also accepts inline `RETRY_ATTEMPTS=3 bash ...helper`.
    if attempts is None:
        inline = re.search(r"RETRY_ATTEMPTS=(\d+)", run)
        attempts = inline.group(1) if inline else None
    if per_attempt is None:
        inline = re.search(r"RETRY_ATTEMPT_TIMEOUT=(\d+)", run)
        per_attempt = inline.group(1) if inline else None

    # The helper's defaults, kept in sync by test_helper_defaults_are_what_the_budgets_assume.
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
    """The retry budget must fit inside the step timeout, or the last attempt is silently cut off."""
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
    """The step timeout must fit inside the job timeout, or the job is killed before the step reports."""
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
    """A paths-filtered workflow that calls the helper must list the helper in its paths filter."""
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
            continue
        assert HELPER in paths, (
            f"{path.name}: the '{event}' trigger filters on paths but does not "
            f"list {HELPER}, which its steps call. A change to the helper would "
            f"not run this workflow."
        )


def test_helper_defaults_are_what_the_budgets_assume() -> None:
    """The helper's default retry and timeout values must match the worst-case budget arithmetic assumes."""
    source = (REPO_ROOT / HELPER).read_text(encoding = "utf-8")
    assert 'ATTEMPTS="${RETRY_ATTEMPTS:-3}"' in source
    assert 'ATTEMPT_TIMEOUT="${RETRY_ATTEMPT_TIMEOUT:-480}"' in source
    assert "seq 1 24" in source
    assert LOCK_WAIT_SECONDS == 24 * 5 + 5


def test_the_helper_makes_apt_fail_fast() -> None:
    """The helper must write apt's fail-fast options; an idle timeout never trips on a trickling socket."""
    source = (REPO_ROOT / HELPER).read_text(encoding = "utf-8")

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

    # Well under apt's 120s default.
    match = re.search(r'APT_TIMEOUT="\$\{APT_ACQUIRE_TIMEOUT:-(\d+)\}"', source)
    assert match, f"{HELPER} does not define a default transfer timeout"
    assert int(match.group(1)) <= 30, (
        f"a {match.group(1)}s transfer timeout is close enough to apt's 120s "
        f"default that a stalled mirror still eats the step budget"
    )

    # Must be configured before the retry loop, or the first (stalling) attempt runs unconfigured.
    configure = source.index("configure_apt_fail_fast\n\nfor attempt")
    loop = source.index('for attempt in $(seq 1 "$ATTEMPTS")')
    assert configure < loop, (
        f"{HELPER} configures apt at or after the retry loop, so the first "
        f"attempt runs with apt's 120s idle timeout"
    )

    # Best effort: a runner that cannot write the file must still run the command.
    assert "could not write" in source, (
        f"{HELPER} does not handle an unwritable apt config, so a runner that "
        f"refuses the write would fail before running the command at all"
    )


def test_the_guard_is_not_vacuous() -> None:
    """At least 10 workflows must match the apt detector, or the loops above pass by finding nothing."""
    found = {path.name: len(_apt_steps(path)) for path in _workflows() if _apt_steps(path)}
    assert (
        len(found) >= 10
    ), f"only {len(found)} workflows matched; the detector looks broken: {found}"
    assert sum(found.values()) >= 15, f"only {sum(found.values())} apt steps matched: {found}"


def test_install_deps_does_not_let_apt_retry_inside_the_attempt() -> None:
    """`playwright install-deps` must not let apt retry; retries belong in the outer loop, not apt."""
    steps = [
        (f"{path.name}: {step.get('name', '<unnamed>')}", step)
        for path in _workflows()
        for _job_id, _job, step in _steps(yaml.safe_load(path.read_text(encoding = "utf-8")))
        if "install-deps" in step["run"]
    ]
    assert steps, "no step runs `playwright install-deps`; this guard checks nothing"

    for name, step in steps:
        retries = (step.get("env") or {}).get("APT_ACQUIRE_RETRIES")
        assert retries is not None, (
            f"step '{name}' runs `playwright install-deps` without setting "
            f"APT_ACQUIRE_RETRIES, so apt falls back to 3 internal retries and a "
            f"stalled mirror costs roughly four times the per-URI timeout"
        )
        assert str(retries) == "0", (
            f"step '{name}' sets APT_ACQUIRE_RETRIES={retries!r}; anything above 0 "
            f"spends the attempt budget inside apt, where the retries are invisible "
            f"and the outer attempt loop cannot see or report them"
        )
