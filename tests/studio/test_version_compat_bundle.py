# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compat suites share one job; every suite on disk must be named by a PR job, so none go unrun."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "version-compat-ci.yml"
SUITE_DIRS = ("tests/version_compat", "tests/vllm_compat")

# Named, not detected, so a rename is a deliberate edit rather than a silent no-op.
BUNDLE_JOB = "pinned-symbol-matrix"

SWEEP_JOB = "daily-fresh-fetch"
# A None entry in sys.modules makes `import` raise and find_spec return None, as in that job.
NOT_IN_THE_SWEEP = ("torch", "numpy", "transformers", "trl", "peft", "accelerate", "unsloth_zoo")

# Recorded gap, not an approval: these need real torch or TRL, which the bundle never installs.
# The cron-only `daily-fresh-fetch` job sweeps them.
CRON_ONLY = {
    "tests/version_compat/test_import_leaves_torch_globals_alone.py",
    "tests/version_compat/test_trl_vllm_generation_lora_patch.py",
}


def _doc() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8")) or {}


def _jobs() -> dict:
    return _doc().get("jobs") or {}


def _runs_on_pull_request(job: dict) -> bool:
    """A job gated to schedule/dispatch cannot be what keeps a suite covered on a PR."""
    cond = str(job.get("if", ""))
    if not cond:
        return True
    return "pull_request" in cond or not re.search(r"schedule|workflow_dispatch", cond)


def _named_paths(job: dict) -> set[str]:
    named: set[str] = set()
    for step in job.get("steps") or []:
        body = str(step.get("run", ""))
        for m in re.finditer(r"tests/[\w/]+\.py", body):
            named.add(m.group(0))
        for m in re.finditer(r"(tests/(?:version_compat|vllm_compat))/(?:\s|\\|$)", body):
            named.add(m.group(1) + "/")
    return named


def _covers(named: set[str], suite: str) -> bool:
    return suite in named or any(n.endswith("/") and suite.startswith(n) for n in named)


def _all_suites() -> set[str]:
    found = set()
    for d in SUITE_DIRS:
        for p in sorted((REPO / d).glob("test_*.py")):
            found.add(f"{d}/{p.name}")
    return found


def test_every_suite_still_runs_on_a_pull_request() -> None:
    """The whole point: a path dropped from the bundle proves less and stays green."""
    pr_named: set[str] = set()
    for job in _jobs().values():
        if _runs_on_pull_request(job):
            pr_named |= _named_paths(job)

    uncovered = sorted(s for s in _all_suites() - CRON_ONLY if not _covers(pr_named, s))
    assert not uncovered, (
        f"these version-compat suites are not run by any pull_request job: {uncovered}. "
        f"Either a path was dropped from {BUNDLE_JOB}'s pytest line -- which removes a "
        f"compat surface without failing anything -- or a new suite was added and never "
        f"wired up. If a suite genuinely cannot run on a PR, add it to CRON_ONLY with the "
        f"reason."
    )


def test_the_bundle_exists_and_names_suites_explicitly() -> None:
    """Suites are named explicitly: a directory sweep would pull in TRL suites that error without torch."""
    job = _jobs().get(BUNDLE_JOB)
    assert job is not None, f"{BUNDLE_JOB} no longer exists; retarget or delete this file"
    named = _named_paths(job)
    assert named, f"{BUNDLE_JOB} names no test paths at all"
    sweeps = sorted(n for n in named if n.endswith("/"))
    assert not sweeps, (
        f"{BUNDLE_JOB} sweeps {sweeps} rather than naming files. That pulls in the suites "
        f"belonging to the install-bearing jobs, which have no torch here."
    )


def test_the_bundle_does_not_duplicate_the_install_bearing_jobs() -> None:
    """Running a suite twice per commit is the waste this change exists to remove."""
    bundle = _named_paths(_jobs()[BUNDLE_JOB])
    for jid, job in _jobs().items():
        if jid == BUNDLE_JOB or not _runs_on_pull_request(job):
            continue
        overlap = sorted(bundle & _named_paths(job))
        assert not overlap, (
            f"{jid} and {BUNDLE_JOB} both run {overlap} on a pull request, so every commit "
            f"pays for it twice"
        )


def test_the_daily_sweep_skips_what_it_cannot_import() -> None:
    """Suites needing torch must skip, not fail, in the daily sweep; bundle suites are left out."""
    sweep = _named_paths(_jobs()[SWEEP_JOB])
    bundle = _named_paths(_jobs()[BUNDLE_JOB])
    suites = sorted(s for s in _all_suites() if _covers(sweep, s) and s not in bundle)
    assert suites, f"{SWEEP_JOB} sweeps none of {SUITE_DIRS}; retarget this test"

    hide = f"import sys\nfor m in {NOT_IN_THE_SWEEP!r}: sys.modules[m] = None\nimport pytest\nsys.exit(pytest.main())"
    proc = subprocess.run(
        [sys.executable, "-c", hide, "-q", "-p", "no:cacheprovider", *suites],
        cwd = REPO,
        env = {**os.environ, "PYTHONPATH": str(REPO)},
        capture_output = True,
        text = True,
        timeout = 600,
    )
    failed = [line for line in proc.stdout.splitlines() if line.startswith(("FAILED", "ERROR"))]
    assert proc.returncode == 0 and not failed, (
        f"{SWEEP_JOB} would go red: these fail without {NOT_IN_THE_SWEEP[0]} rather than skip. Guard them the way "
        f"their neighbours do, `if importlib.util.find_spec('torch') is None: pytest.skip(...)`.\n"
        + ("\n".join(failed[:20]) or proc.stdout[-3000:] + proc.stderr[-3000:])
    )


def test_the_bundle_stays_parallel_and_file_scoped() -> None:
    """Keep -n 4 (-n 8 is throttled upstream) and --dist loadfile so one file stays on one worker."""
    steps = _jobs()[BUNDLE_JOB].get("steps") or []
    body = "\n".join(str(s.get("run", "")) for s in steps)
    assert re.search(r"-n\s+4\b", body), (
        "the bundled job lost its `-n 4`, so six jobs' worth of suites now run one after "
        "another on a single runner -- slower than the six jobs it replaced"
    )
    assert "--dist loadfile" in body, (
        "the bundled job lost `--dist loadfile`, so one suite's tests can be split across "
        "workers and interleaved with another's"
    )


def test_the_install_bearing_jobs_were_not_folded_in() -> None:
    """They install mutually exclusive TRL pins; one venv cannot hold both."""
    for jid in ("zoo-imports-under-spoof", "grpo-fake-run"):
        assert jid in _jobs(), (
            f"{jid} is gone. It installs a torch + TRL stack that conflicts with its "
            f"sibling's pins, so it cannot have been merged into anything -- check it was "
            f"not folded into {BUNDLE_JOB}, which has no torch at all."
        )


def test_every_test_file_a_job_runs_triggers_the_workflow() -> None:
    """A file only this workflow executes must be in its pull_request paths, or a PR that only
    edits (or weakens) that file never runs it. The modules.json trust gate is the case that
    prompted this: the CPU repo-test shards skip it, so this workflow is its only runner."""
    from fnmatch import fnmatch

    doc = _doc()
    # PyYAML reads the bare `on:` key as the boolean True.
    triggers = doc.get("on", doc.get(True)) or {}
    patterns = (triggers.get("pull_request") or {}).get("paths") or []
    assert patterns, "version-compat-ci.yml lost its pull_request paths filter"
    named = set()
    for job in _jobs().values():
        named |= {p for p in _named_paths(job) if not p.endswith("/") and (REPO / p).is_file()}
    assert named, "no test file found in any run step"
    missing = sorted(p for p in named if not any(fnmatch(p, pattern) for pattern in patterns))
    assert not missing, f"run by version-compat-ci.yml but not in its pull_request paths: {missing}"
