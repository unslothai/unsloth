# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No workflow may spend the shared Actions cache budget carelessly.

This repo's Actions cache budget is 50 GiB (not GitHub's 10 GB default), and GitHub evicts
least-recently-used once it is exceeded. Measured before this file existed, unslothai/unsloth
held **49.63 GiB across 258 entries -- 99.3% full**, so eviction runs at the margin and every
new entry displaces an existing one. What fills it is almost entirely redundancy:

    20.74 GiB  duplicate waste: the SAME key held on several refs (42% of the cache)
                 6.67 GiB   13 copies  setup-python ... python-3.13.15-pip-85e247d7...
                 6.50 GiB   11 copies  setup-python ... python-3.11.15-pip-85e247d7...
                 3.83 GiB   10 copies  setup-python ... python-3.12.13-pip-85e247d7...
                 0.91 GiB    3 copies  ms-playwright-Linux-1.62.0-cfw-v1
    10.70 GiB  84 entries written and never read again, 10.60 GiB of it setup-python
    24.01 GiB  198 entries on PR refs, restorable only by re-runs of that same PR
     0.00 GiB  entries unread for 7+ days -- nothing is idle, the cache is churning

Every one of those duplicated keys already has a copy on `main`, which every PR can restore
from. The PR-scoped copies are therefore redundant by construction: they buy no hit rate and
evict the copy that does.

That is a self-reinforcing loop, and the repo had already diagnosed it once for the GGUF
caches (see the save step in studio-inference-smoke.yml: "PR misses -> downloads -> saves its
own copy -> evicts main's -> next PR misses"). It reappeared through two doors this file now
closes.

Both failure modes are silent. Nothing goes red when a cache is evicted; CI just quietly
re-downloads a 4.6 GB model and everyone assumes that is what it costs.

Door 1 -- saving on a PR ref. `actions/cache` (the read-write form) saves from its post-step
on every ref. A PR-scoped entry can never be read by anyone except re-runs of that same PR,
yet it competes for the budget against main's copy, which every PR *can* read. Saves belong
on main only, via `actions/cache/restore` plus a `github.ref == 'refs/heads/main'` save.

Door 2 -- `cache: 'pip'` on a job that installs almost nothing. `actions/setup-python`
derives one pip key per interpreter from dependency files across the whole repo, so dozens of
unrelated jobs share it and race to save under it. The entries measured 666-715 MB, and the
four interpreter keys between them account for 19.44 GiB of the 20.74 GiB of duplicate waste.
A job that only pip-installs `huggingface_hub` or `pytest` was paying that for the 0-7s its
restore step took. Jobs that really do install torch/transformers keep the cache; the rest do
not.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Jobs whose pip cache earns its place: they install a torch/transformers-class dependency
# set, where the download genuinely dominates. Anything not listed here must not ask for it.
PIP_CACHE_ALLOWED = {
    ("consolidated-tests-ci.yml", "consolidated"),
    ("consolidated-tests-ci.yml", "llama-cpp-smoke"),
    ("mlx-ci.yml", "dispatch"),
    ("notebooks-ci.yml", "api-introspect"),
    ("studio-backend-ci.yml", "pytest"),
    ("studio-backend-ci.yml", "repo-cpu-tests"),
    ("studio-export-capability-ci.yml", "capability"),
    ("version-compat-ci.yml", "zoo-imports-under-spoof"),
    ("version-compat-ci.yml", "grpo-fake-run"),
}

HEAVY = re.compile(
    r"torch|transformers|trl|peft|vllm|bitsandbytes|sentence-transformers|diffusers"
    r"|accelerate|datasets|requirements/"
)


def _workflows():
    for f in sorted(WORKFLOWS.glob("*.yml")):
        try:
            doc = yaml.safe_load(f.read_text(encoding = "utf-8"))
        except yaml.YAMLError as exc:  # a broken workflow is another test's problem
            pytest.fail(f"{f.name} does not parse: {exc}")
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), dict):
            yield f.name, doc


def _jobs():
    for name, doc in _workflows():
        for jid, job in doc["jobs"].items():
            if isinstance(job, dict):
                yield name, jid, job


def test_no_workflow_saves_a_cache_on_a_pull_request_ref():
    offenders = []
    for name, jid, job in _jobs():
        for step in job.get("steps") or []:
            uses = str(step.get("uses", ""))
            if "actions/cache" not in uses:
                continue
            saves = "/restore@" not in uses  # read-write and /save@ both write
            if not saves:
                continue
            if "refs/heads/main" not in str(step.get("if", "")):
                offenders.append(f"{name}:{jid}: {step.get('name') or uses}")
    assert not offenders, (
        "these steps save a cache on whatever ref they run on, so every PR writes its own "
        "copy and evicts the copy on main that all PRs share:\n  " + "\n  ".join(offenders)
    )


def test_only_jobs_that_install_heavy_dependencies_ask_for_the_pip_cache():
    asking = {
        (name, jid)
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in str(step.get("uses", ""))
        and (step.get("with") or {}).get("cache") == "pip"
    }
    extra = asking - PIP_CACHE_ALLOWED
    assert not extra, (
        f"these jobs ask for the shared pip cache without installing anything that justifies "
        f"a ~700MB entry: {sorted(extra)}. The setup-python pip key is one per interpreter "
        f"across the whole repo, so every extra claimant is another racer saving under it."
    )


@pytest.mark.parametrize("name,jid", sorted(PIP_CACHE_ALLOWED))
def test_every_allowed_pip_cache_job_still_exists_and_still_earns_it(name, jid):
    """The allowlist must not outlive the jobs, or it silently permits nothing."""
    doc = dict(_workflows()).get(name)
    assert doc is not None, f"{name} no longer exists; drop it from PIP_CACHE_ALLOWED"
    job = doc["jobs"].get(jid)
    assert job is not None, f"{name} no longer has job {jid}; drop it from PIP_CACHE_ALLOWED"
    body = "\n".join(str(s.get("run", "")) for s in job.get("steps") or [])
    assert HEAVY.search(body), (
        f"{name}:{jid} is allowed a pip cache but no longer installs anything heavy; it "
        f"should give the budget back"
    )


def test_the_cold_install_lanes_never_restore_a_cache():
    """These workflows exist to prove a cold install works. A warm one proves nothing.

    They would still pass with a cache in front of them, which is exactly why this is
    asserted rather than left to review.
    """
    cold = [
        "clean-machine-install-ci.yml",
        "desktop-app-clean-machine-ci.yml",
        "interrupted-install-ci.yml",
    ]
    offenders = []
    for name, jid, job in _jobs():
        if name not in cold:
            continue
        for step in job.get("steps") or []:
            uses = str(step.get("uses", ""))
            if "actions/cache" in uses:
                offenders.append(f"{name}:{jid}: {step.get('name') or uses}")
            if "setup-python" in uses and (step.get("with") or {}).get("cache"):
                offenders.append(f"{name}:{jid}: setup-python cache on a cold-install lane")
    assert not offenders, "a cold-install lane must not be warmed by a cache:\n  " + "\n  ".join(
        offenders
    )


def test_every_setup_python_step_still_pins_an_interpreter():
    """Guards the edit that produced this file.

    Removing `cache: 'pip'` from an inline-flow mapping (`with: { python-version: '3.12',
    cache: 'pip' }`) by deleting the line takes the interpreter pin with it, and the job then
    silently runs on whatever Python the image happens to ship.
    """
    offenders = [
        f"{name}:{jid}"
        for name, jid, job in _jobs()
        for step in job.get("steps") or []
        if "setup-python" in str(step.get("uses", ""))
        and not (step.get("with") or {}).get("python-version")
    ]
    assert not offenders, f"setup-python without an explicit python-version: {offenders}"
