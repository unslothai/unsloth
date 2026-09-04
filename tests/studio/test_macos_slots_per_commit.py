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
`Mac Unsloth GGUF CI`, `Mac Unsloth UI + API + Update CI`, `Mac Unsloth Install Matrix CI` and
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


def _covered(path: str, patterns) -> bool:
    """Whether ``path`` matches any Actions path filter in ``patterns``."""
    import fnmatch

    for pattern in patterns:
        if pattern == path or fnmatch.fnmatch(path, pattern):
            return True
        if pattern.endswith("/**") and path.startswith(pattern[:-2]):
            return True
    return False


@pytest.mark.parametrize(
    "name",
    [
        "studio-mac-ui-smoke.yml",
        "studio-mac-install-matrix.yml",
        "studio-tauri-smoke.yml",
    ],
)
def test_every_helper_a_workflow_executes_is_in_its_trigger(name):
    """A scoped trigger must list the checked-in files the workflow actually runs.

    Scoping a trigger is only safe if the list is complete, and these lists were not: five
    helper scripts and one auditor were executed by name and matched no pattern. While the
    push trigger was unfiltered that gap was invisible, because every commit ran everything
    after merge; narrowing the trigger is what turns it into a real hole, where editing
    `assert-llama-loads.sh` stops running the workflow that asserts with it.

    Matched by looking for the path in a `run:` body, which is how every one of these is
    invoked. That deliberately says nothing about files a workflow depends on more
    loosely -- `studio/package.json` reaches the Tauri build through
    `npm install --prefix studio` and is listed by hand, not found here.
    """
    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    on = _on(doc) or {}
    runs = "\n".join(
        str(step.get("run", ""))
        for job in doc["jobs"].values()
        if isinstance(job, dict)
        for step in job.get("steps") or []
        if isinstance(step, dict)
    )
    referenced = {
        ref.strip()
        for pattern in (r"\.github/scripts/[\w./-]+", r"(?:^|\s)scripts/[\w./-]+")
        for ref in re.findall(pattern, runs, re.M)
    }
    existing = sorted(r for r in referenced if (REPO / r).is_file())
    assert existing, f"{name} appears to execute no checked-in helper; the scan is wrong"

    for trigger in ("pull_request", "push"):
        patterns = (on.get(trigger) or {}).get("paths") or []
        missing = [r for r in existing if not _covered(r, patterns)]
        assert not missing, (
            f"{name}: these files are executed by the workflow but match no {trigger} path "
            f"filter, so editing one of them does not run the workflow that uses it: "
            f"{missing}"
        )


@pytest.mark.parametrize(
    "name",
    [
        "studio-mac-ui-smoke.yml",
        "studio-mac-install-matrix.yml",
        "studio-tauri-smoke.yml",
    ],
)
def test_a_listed_python_input_brings_its_sibling_imports(name):
    """Listing a script but not the module it imports leaves half a dependency in the filter.

    `studio/install_llama_prebuilt.py` was listed; `studio/prebuilt_core.py`, which it
    imports at line 55, was not. Editing only the latter changed exactly what the install
    matrix asserts on and did not run it.

    Scoped to same-directory imports on purpose. The full transitive closure of an
    installer is most of the repo, and chasing it would put `pyproject.toml` and every
    requirements file into a macOS trigger, which is how a filter stops saving anything.
    Where a deeper dependency matters it is listed by hand with a comment saying why; this
    covers the one case that is mechanical and therefore easy to forget.
    """
    import ast

    doc = yaml.safe_load((WORKFLOWS / name).read_text(encoding = "utf-8"))
    on = _on(doc) or {}
    patterns = (on.get("pull_request") or {}).get("paths") or []

    missing = []
    for pattern in patterns:
        source = REPO / pattern
        if not (source.is_file() and source.suffix == ".py"):
            continue
        tree = ast.parse(source.read_text(encoding = "utf-8", errors = "replace"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
        for module in sorted(names):
            sibling = source.parent / f"{module}.py"
            if not sibling.is_file():
                continue  # stdlib or third-party, not a checked-in sibling
            rel = sibling.relative_to(REPO).as_posix()
            if not _covered(rel, patterns):
                missing.append(f"{rel} (imported by {pattern})")
    assert not missing, (
        f"{name} lists a Python input but not a module it imports from the same directory, "
        f"so editing that module does not run the workflow that depends on it: {missing}"
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


# Images GitHub still schedules.
LIVE_MACOS_IMAGES = {
    "macos-15",
    "macos-15-intel",
    "macos-26",
    "macos-26-intel",
    "macos-latest",
}


def _macos_labels():
    """Every concrete macOS image any job can be scheduled onto, with its origin."""
    found = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding = "utf-8"))
        if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), dict):
            continue
        for jid, job in doc["jobs"].items():
            if not isinstance(job, dict):
                continue
            # runs-on plus the matrix it may select from: a retired image hides in an `include:`
            # list just as easily as in a literal runs-on.
            blob = str(job.get("runs-on", ""))
            strategy = job.get("strategy") or {}
            blob += str((strategy.get("matrix") or {}) if isinstance(strategy, dict) else "")
            # Only things shaped like a GitHub image name. The loose MACOS pattern used elsewhere
            # in this file also matches build targets that merely contain "macos":
            # release-desktop's matrix carries `macos-aarch64`, which is a Rust triple's nickname
            # and never a runner label. Every real macOS image is macos-latest or
            # macos-<version>[-intel].
            for label in re.findall(r"\bmacos-(?:latest|\d+(?:-intel)?)\b", blob, re.I):
                found.append((path.name, jid, label.lower()))
    return found


def test_no_job_targets_a_retired_macos_image() -> None:
    """
    macos-14's retirement is already written into three comments in this repo, each
    explaining why some job moved off it. Comments do not fail, so the next
    retirement will be discovered the same way this one was: by a job that stops
    being scheduled, on a runner pool nobody is watching.

    This is the cheap version of that discovery. It cannot know GitHub's roadmap,
    but it does force the retirement to be recorded in one place, and it names
    every job that has to move on the day someone records it.
    """
    labels = _macos_labels()
    assert labels, "no macOS labels found at all; this guard would pass vacuously"

    retired = sorted(
        f"{name}:{jid} -> {label}" for name, jid, label in labels if label not in LIVE_MACOS_IMAGES
    )
    assert not retired, (
        f"these jobs target a macOS image not in LIVE_MACOS_IMAGES: {retired}. Either "
        f"GitHub ships it and it belongs in the set, or it is retired and these jobs "
        f"need moving before they stop being scheduled."
    )
