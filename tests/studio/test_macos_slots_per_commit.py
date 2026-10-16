# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Account-wide cap of five concurrent macOS jobs: push triggers to main need a `paths:` filter."""

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


_SELECTED_MATRIX = re.compile(r"fromJSON\(\s*needs\.([\w-]+)\.outputs\.([\w-]+)\s*\)")


def _matrix(job, doc) -> dict:
    """All legs are returned, PR subset or not, because this asks which images a job can allocate."""
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    if isinstance(matrix, dict):
        return matrix
    match = _SELECTED_MATRIX.search(str(matrix))
    if not match or not isinstance(doc, dict):
        return {}
    producer = (doc.get("jobs") or {}).get(match.group(1)) or {}
    for step in producer.get("steps") or []:
        matrix_file = ((step.get("env") or {}) if isinstance(step, dict) else {}).get("MATRIX_FILE")
        if matrix_file:
            legs = yaml.safe_load((REPO / matrix_file).read_text(encoding = "utf-8")) or {}
            return {"include": list(legs.get(match.group(2)) or [])}
    return {}


def _job_runs_on_macos(job, doc = None) -> bool:
    """Reads runs-on (through any matrix), not the whole job: a macOS name in a step allocates no runner."""
    runs_on = job.get("runs-on")
    values = runs_on if isinstance(runs_on, list) else [runs_on]
    for value in values:
        if not isinstance(value, str):
            continue
        if MACOS.search(value):
            return True
        for key in re.findall(r"matrix\.([\w-]+)", value):
            matrix = _matrix(job, doc)
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
        if any(_job_runs_on_macos(j, doc) for j in doc["jobs"].values() if isinstance(j, dict)):
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
            continue
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
    """Push and pull_request paths must match: a narrower push list silently drops the post-merge check."""
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
    """Every file a `run:` step executes must be in the trigger, or editing it skips the workflow."""
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
    """A listed script needs its same-directory imports too; a full closure is most of the repo."""
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
                continue
            rel = sibling.relative_to(REPO).as_posix()
            if not _covered(rel, patterns):
                missing.append(f"{rel} (imported by {pattern})")
    assert not missing, (
        f"{name} lists a Python input but not a module it imports from the same directory, "
        f"so editing that module does not run the workflow that depends on it: {missing}"
    )


def test_a_commit_that_touches_nothing_relevant_starts_no_macos_job():
    """A README-only commit starts no macOS job; the matcher handles literals, dir/** and single * globs."""
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


# Images GitHub still schedules. macos-14 is deliberately absent (being retired). Remove
# an image when GitHub announces its retirement.
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
            # A retired image can hide in a matrix `include:` too.
            blob = str(job.get("runs-on", ""))
            blob += str(_matrix(job, doc) if isinstance(job.get("strategy"), dict) else "")
            # Only image names: release-desktop's `macos-aarch64` is a Rust target, not a runner.
            for label in re.findall(r"\bmacos-(?:latest|\d+(?:-intel)?)\b", blob, re.I):
                found.append((path.name, jid, label.lower()))
    return found


def test_no_job_targets_a_retired_macos_image() -> None:
    """Retired macOS images must be recorded here, because a comment about a retirement never fails."""
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
