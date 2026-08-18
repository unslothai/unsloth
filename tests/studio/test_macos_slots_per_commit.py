# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""No macOS workflow may run on every commit to main.

GitHub caps macOS at **five concurrent jobs account-wide** -- across every repository,
on Free, Pro and Team alike. That makes a macOS runner slot the scarcest resource in this
repo's CI by a wide margin, and it is the reason macOS queue times dominate: measured over
the last 20 main runs, `studio-mac-ui-smoke` waited a median of 245 minutes to execute for
21, and `Unsloth Tauri CI :: Rust unit tests (macos)` waited a median of 270 minutes to run
for 3.

Four workflows used to declare `push: branches: [main]` with no `paths:` filter while their
`pull_request` trigger was carefully scoped. The effect was invisible on a PR and only
appeared after merge: commit 6371f46a changes README.md and nothing else, and it started
`Mac Studio GGUF CI`, `Mac Studio UI + API + Update CI`, `Mac Studio Install Matrix CI` and
`Unsloth Tauri CI` -- seven macOS legs, 40% over the entire account cap, for a
documentation typo. Every one of those runs then queued behind the others.

`clean-machine-install-ci.yml` and `mlx-ci.yml` already got this right and say why:
"Same list as the PR filter: without it a direct push to main touching any of these skipped
the workflow and the post-merge backstop never happened." This asserts the rest match.

The failure is silent in both directions, which is why it is a test rather than a review
note: an unfiltered push trigger costs nothing on the PR that introduces it, and the cost
lands on unrelated commits weeks later as queue time nobody attributes to it.
"""

import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

MACOS = re.compile(r"macos[-\w.]*", re.I)


def _on(doc):
    """The `on:` mapping, which PyYAML parses as the boolean True."""
    return doc.get(True) if True in doc else doc.get("on")


def _job_runs_on_macos(job) -> bool:
    """Whether ``job`` schedules a macOS runner.

    Reads `runs-on` and, when that is a matrix expression, the matrix values it selects
    from. Scanning the whole job instead was the first cut and it over-matched badly:
    `workflow-trigger-lint.yml` and `studio-inference-smoke.yml` both have ubuntu-only jobs
    that merely NAME macOS somewhere in a step, and both were reported as macOS workflows.
    A guard about runner slots has to read what actually allocates a runner.
    """
    runs_on = job.get("runs-on")
    values = runs_on if isinstance(runs_on, list) else [runs_on]
    for value in values:
        if not isinstance(value, str):
            continue
        if MACOS.search(value):
            return True
        # `runs-on: ${{ matrix.os }}` -- resolve against the matrix it names.
        for key in re.findall(r"matrix\.([\w-]+)", value):
            matrix = (job.get("strategy") or {}).get("matrix") or {}
            candidates = list(matrix.get(key) or [])
            for entry in matrix.get("include") or []:
                if isinstance(entry, dict) and key in entry:
                    candidates.append(entry[key])
            if any(isinstance(c, str) and MACOS.search(c) for c in candidates):
                return True
    return False


def _macos_workflows():
    """Workflows with at least one macOS leg."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding = "utf-8")
        doc = yaml.safe_load(text)
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            continue
        if any(_job_runs_on_macos(j) for j in doc["jobs"].values() if isinstance(j, dict)):
            yield path.name, doc, text


def test_the_scan_finds_the_macos_workflows_it_claims_to():
    """A scan that matched nothing would pass every check below."""
    names = {name for name, _, _ in _macos_workflows()}
    for expected in (
        "studio-mac-ui-smoke.yml",
        "studio-mac-inference-smoke.yml",
        "studio-mac-install-matrix.yml",
        "studio-tauri-smoke.yml",
        "mlx-ci.yml",
        "clean-machine-install-ci.yml",
    ):
        assert expected in names, f"{expected} is no longer detected as having a macOS leg"


def test_no_macos_workflow_runs_on_every_push_to_main():
    offenders = []
    for name, doc, _ in _macos_workflows():
        push = (_on(doc) or {}).get("push")
        if not isinstance(push, dict):
            continue  # no push trigger at all is the strongest form of this
        if not push.get("paths") and not push.get("paths-ignore"):
            offenders.append(name)
    assert not offenders, (
        f"these workflows run macOS jobs on EVERY commit to main: {offenders}. macOS is "
        f"capped at five concurrent jobs account-wide, so an unfiltered push trigger here "
        f"oversubscribes the whole account on commits that cannot affect what it tests. "
        f"Mirror the pull_request paths onto push, as clean-machine-install-ci.yml and "
        f"mlx-ci.yml do."
    )


@pytest.mark.parametrize(
    "name",
    [
        "studio-mac-ui-smoke.yml",
        "studio-mac-inference-smoke.yml",
        "studio-mac-install-matrix.yml",
        "studio-tauri-smoke.yml",
        "clean-machine-install-ci.yml",
        "mlx-ci.yml",
    ],
)
def test_the_push_filter_matches_the_pull_request_filter(name):
    """Narrower on push than on PR would drop the post-merge backstop.

    The two lists are the same question asked twice -- "could this commit break this
    workflow" -- so they drifting apart is always a bug, in whichever direction. A push
    list that is a strict subset silently stops testing something after merge that was
    tested before it, which is the more dangerous direction and the harder to notice.
    """
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    on = _on(doc) or {}
    pr_paths = (on.get("pull_request") or {}).get("paths")
    push_paths = (on.get("push") or {}).get("paths")
    assert pr_paths, f"{name} no longer scopes its pull_request trigger"
    assert push_paths, f"{name} no longer scopes its push trigger"
    assert sorted(pr_paths) == sorted(push_paths), (
        f"{name}: the push and pull_request path filters have drifted apart.\n"
        f"  only on pull_request: {sorted(set(pr_paths) - set(push_paths))}\n"
        f"  only on push:         {sorted(set(push_paths) - set(pr_paths))}"
    )


def test_a_commit_that_touches_nothing_relevant_starts_no_macos_job():
    """The property the whole change exists for, checked against a concrete commit.

    README-only is not a hypothetical: commit 6371f46a is exactly that, and it started
    four macOS workflows. Matching is by the same prefix/glob rules Actions uses, kept
    simple deliberately -- every filter in these files is either a literal, a `dir/**`
    prefix or a single `*` glob, and this asserts that stays true so the simple matcher
    cannot quietly become wrong.
    """
    import fnmatch

    changed = ["README.md"]
    triggered = []
    for name, doc, _ in _macos_workflows():
        push = (_on(doc) or {}).get("push")
        if not isinstance(push, dict):
            continue
        for pattern in push.get("paths") or []:
            assert "!" not in pattern, (
                f"{name} uses a negated push path ({pattern!r}); this matcher does not "
                f"model negation, so extend it before relying on this test"
            )
            for path in changed:
                if fnmatch.fnmatch(path, pattern) or (
                    pattern.endswith("/**") and path.startswith(pattern[:-2])
                ):
                    triggered.append(f"{name} via {pattern!r}")
    assert not triggered, (
        f"a README-only commit still starts macOS jobs: {triggered}. That was the "
        f"original symptom: seven macOS legs against a five-slot cap for a docs typo."
    )
