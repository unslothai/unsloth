# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Six version-compat jobs became one. None of their suites may stop running.

The six pinned-symbol jobs cost six runner slots for 287s of combined work, and on a
sampled main commit each waited 23 to 31 minutes to be admitted in order to execute for
15 to 128 seconds. The queue was the cost, so the slots were the target.

Co-locating them is safe for a reason specific to these suites, not a general one: they
fetch raw source from raw.githubusercontent.com and grep it for symbols (see
tests/version_compat/_fetch.py, `has_def` / `first_match`). Nothing is pip-installed, so
there is no venv for "transformers 4.57.6" and "transformers main" to fight over. That is
why the two HEAVY jobs in the same workflow -- which do install, and install mutually
exclusive TRL pins -- stay separate and must never be folded in.

The failure this file exists for is the silent one. Deleting a path from the bundled
pytest line removes a whole compat surface and turns nothing red: the job still runs, still
passes, and simply proves less. Nothing else in CI would notice, because the suite it
stopped running is the only thing that was checking that upstream symbol.

So the assertion is coverage, derived from the filesystem: every suite that exists must be
named by some job that actually runs on a pull request. A new suite file added and never
wired up fails here too, which is the same bug arriving from the other direction.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "version-compat-ci.yml"
SUITE_DIRS = ("tests/version_compat", "tests/vllm_compat")

# The bundled job. Named, not detected:
BUNDLE_JOB = "pinned-symbol-matrix"

# Suites with no pull_request home today.
# This is a RECORDED GAP, not an approval, and both entries pre-date the bundling change that added this file
# Neither can join the bundle, because the bundle installs nothing but pytest: test_import_leaves_torch_globals_alone.py
# runs `import torch` inside a subprocess probe (_PROBE at module scope), so it needs a real torch.
# test_trl_vllm_generation_lora_patch.py needs an installed TRL.
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
        # A bare directory sweep covers every file under it.
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
    """A directory sweep here would silently pull in the heavy jobs' suites.

    `tests/version_compat/` as a bare argument would drag in the TRL fake-run files, which
    need an installed torch + TRL that this dependency-free job does not have. They would
    error rather than skip, so this is a real constraint and not tidiness.
    """
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


def test_the_bundle_stays_parallel_and_file_scoped() -> None:
    """Without -n the bundle is six jobs' work run end to end on one runner.

    Measured on the full set: serial 182.8s, `-n 4` 27.8s, `-n 8` 75.2s, all three at the
    same 1788 collected. -n 8 being 2.7x slower than -n 4 is upstream throttling of a
    fetch-bound suite, not core contention, so the worker count is not "as high as
    possible" -- it is pinned at the runner's core count.

    The pass/skip SPLIT of those 1788 is environment-dependent, so do not pin a number to
    it. On CI the job is 1604 passed / 184 skipped; on a developer box with transformers
    and unsloth_zoo already installed it is 1606 / 182, because
    test_peft_conversion_symbol_backfill.py::test_the_moe_snapshot_matches_the_installed_transformers
    and test_vllm_pinned_symbols.py::test_unsloth_zoo_standby_guards_present both skip
    when their package is absent. The six jobs this replaced installed pytest and nothing
    else, so they skipped those two as well -- the split is unchanged by the bundling.

    --dist loadfile keeps a file on one worker so two suites cannot interleave.
    """
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
