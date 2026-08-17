# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``cancel-in-progress: false`` does not keep a main run alive, and the repo believed it did.

GitHub cancels any PENDING run in a concurrency group the moment a newer one is queued.
``cancel-in-progress`` governs only runs that are already EXECUTING: "any existing pending
job or workflow in the same concurrency group will be canceled and the new queued job or
workflow will take its place" (workflow-syntax reference).

So on a branch where pushes land in bursts, ``cancel-in-progress: false`` buys nothing for
the runs in the middle of the burst. Twenty-seven workflows carried a comment saying it did.
Both incidents the repo wrote down are this: studio-ui-smoke.yml records a break sitting on
main "for 14 hours behind four cancelled runs", and on 2026-08-17 five merges cancelled four
consecutive main runs of Backend CI, which completed none. Each was cancelled with ZERO jobs
recorded, so nothing ever showed up as a failure; the workflow simply never reported.

The only thing that protects a main run is a concurrency group it does not share with the
next commit.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Kaggle spends an external GPU quota rather than runner minutes. There a superseded run
# genuinely should be dropped instead of replayed per commit, so these keep one shared group
# and are the only workflows allowed to lose a main run.
QUOTA_BOUND = frozenset({"kaggle-t4-notebook-ci.yml", "kaggle-t4-studio-gpu-ci.yml"})

# Phrasings that assert main runs are not cancelled. A workflow may only say this if its
# group makes it true.
CLAIMS_PROTECTION = re.compile(
    r"never on main|never cancelled on main|not cancelled on main|never cancels on main",
    re.IGNORECASE,
)


def _documents() -> dict[str, dict]:
    out = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        try:
            document = yaml.safe_load(path.read_text(encoding = "utf-8"))
        except yaml.YAMLError:  # not this guard's job to report
            continue
        if isinstance(document, dict):
            out[path.name] = document
    return out


def _group(document: dict) -> str:
    concurrency = document.get("concurrency")
    if isinstance(concurrency, str):
        return concurrency
    if isinstance(concurrency, dict):
        return str(concurrency.get("group", ""))
    return ""


def _is_per_commit_on_main(group: str) -> bool:
    """Whether this group differs between two commits on main.

    ``github.sha`` is the only thing in the expression context that does. A group without it
    is shared by every main commit, so a pending run in it is replaced by the next merge.
    """
    return "github.sha" in group


def _runs_on_main_push(document: dict) -> bool:
    triggers = document.get(True) or document.get("on") or {}
    push = triggers.get("push") if isinstance(triggers, dict) else None
    return isinstance(push, dict) and "main" in (push.get("branches") or [])


def _protected() -> dict[str, dict]:
    return {
        name: document
        for name, document in _documents().items()
        if name not in QUOTA_BOUND and _runs_on_main_push(document)
    }


def test_every_workflow_that_runs_on_main_is_grouped_per_commit():
    """Otherwise a merge burst discards it before it starts.

    Checked across all of them rather than a named few, because the failure is invisible:
    the run is cancelled with zero jobs recorded, so nothing surfaces as a test failure and
    the workflow simply never reported on that commit.
    """
    offenders = sorted(
        name
        for name, document in _protected().items()
        if not _is_per_commit_on_main(_group(document))
    )
    assert not offenders, (
        f"{offenders} share one concurrency group across main commits, so a merge burst "
        f"cancels their pending runs before they start. cancel-in-progress does not prevent "
        f"that: it only spares runs already executing. Append "
        f"-${{{{ github.ref == 'refs/heads/main' && github.sha || '' }}}} to the group."
    )


def test_the_scan_actually_found_the_workflows():
    """A glob that matched nothing would pass every check above."""
    protected = _protected()
    assert len(protected) > 20, f"only found {len(protected)} main workflows; the scan is wrong"
    assert "studio-backend-ci.yml" in protected


def test_the_quota_bound_exemptions_still_exist():
    """An exemption naming a file that moved would silently widen to nothing."""
    documents = _documents()
    missing = sorted(name for name in QUOTA_BOUND if name not in documents)
    assert not missing, f"QUOTA_BOUND names workflows that no longer exist: {missing}"


def test_a_pull_request_still_gets_latest_only():
    """The protection is for main; superseding a PR push is still what we want.

    A group keyed on github.sha unconditionally would leave every abandoned PR run
    executing, which is the opposite of the intent and the expensive direction.
    """
    for name, document in _protected().items():
        group = _group(document)
        assert "github.ref" in group, f"{name} no longer separates refs: {group!r}"
        assert "refs/heads/main" in group, (
            f"{name} applies its per-commit group unconditionally ({group!r}), so pull "
            f"request pushes stop superseding each other and every abandoned run keeps "
            f"burning a runner"
        )


def test_no_workflow_claims_a_main_protection_it_does_not_have():
    """The comment that made this repo believe it was covered, in 27 files.

    Read from the raw text rather than the parsed document, because a comment is exactly what
    YAML throws away, and the comment is what people act on.
    """
    offenders = []
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        text = path.read_text(encoding = "utf-8")
        if not CLAIMS_PROTECTION.search(text):
            continue
        try:
            document = yaml.safe_load(text)
        except yaml.YAMLError:
            continue
        if isinstance(document, dict) and not _is_per_commit_on_main(_group(document)):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} say main runs are never cancelled while sharing one concurrency group "
        f"across main commits, which is the belief that let a regression sit on main for 14 "
        f"hours behind four cancelled runs. Either group per commit, or do not claim it."
    )
