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
main "for 14 hours behind four cancelled runs", and on 2026-08-17 four merges in 37 minutes
cancelled three consecutive main runs of Backend CI, which completed none.

The only thing that actually protects a main run is a concurrency group it does not share
with the next commit. This checks that the workflows claiming that protection have it.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Workflows whose main run must survive a merge burst. Both are here because a regression
# went unreported behind cancelled main runs of exactly these two.
PROTECTED_ON_MAIN = ("studio-backend-ci.yml", "studio-ui-smoke.yml")

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


@pytest.mark.parametrize("name", PROTECTED_ON_MAIN)
def test_the_protected_workflows_are_grouped_per_commit_on_main(name):
    documents = _documents()
    assert name in documents, f"{name} is gone; if it moved, move this guard with it"
    group = _group(documents[name])
    assert group, f"{name} has no concurrency group at all"
    assert _is_per_commit_on_main(group), (
        f"{name} shares one concurrency group across main commits ({group!r}), so a merge "
        f"burst cancels its pending runs before they start. cancel-in-progress does not "
        f"prevent that: it only spares runs that are already executing. Include github.sha "
        f"in the group on main."
    )


def test_no_workflow_claims_a_main_protection_it_does_not_have():
    """The comment that made this repo believe it was covered, in 27 files.

    Read from the raw text rather than the parsed document, because a comment is exactly
    what YAML throws away, and the comment is what people act on.
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


def test_a_pull_request_still_gets_latest_only():
    """The protection is for main; superseding a PR push is still what we want.

    A group keyed on github.sha for pull requests too would leave every superseded PR run
    executing, which is the opposite of the intent and the expensive direction.
    """
    for name in PROTECTED_ON_MAIN:
        document = _documents()[name]
        group = _group(document)
        assert "github.ref" in group, f"{name} no longer separates refs: {group!r}"
        assert "refs/heads/main" in group, (
            f"{name} applies its per-commit group unconditionally ({group!r}), so pull "
            f"request pushes stop superseding each other and every abandoned run keeps "
            f"burning a runner"
        )
        concurrency = document.get("concurrency") or {}
        assert "github.ref != 'refs/heads/main'" in str(
            concurrency.get("cancel-in-progress")
        ), f"{name} no longer cancels superseded PR runs"
