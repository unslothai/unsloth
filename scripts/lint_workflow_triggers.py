#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse dangerous GitHub Actions trigger patterns at PR time.

Bans patterns behind the TanStack GHSA-g7cv-rxg3-hmpx compromise:

1.  `pull_request_target` -- runs a fork's workflow against the base
    repo's secrets/permissions; use `pull_request` instead.
2.  `workflow_run` chained to a PR-triggered workflow -- same trust
    boundary problem one hop later (poisoned artifacts/caches run with
    elevated permissions).
3.  Cache keys shared between PR-triggered and publish/release/push
    workflows -- a fork PR could poison a cache the publish workflow
    restores. Partition the key namespaces.

Exit codes: 0 = no findings, 1 = findings (listed on stderr).
Run from repo root: python3 scripts/lint_workflow_triggers.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with 'pip install pyyaml'", file = sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

BANNED_TRIGGERS: tuple[str, ...] = ("pull_request_target",)
RESTRICTED_TRIGGERS: tuple[str, ...] = ("workflow_run",)
PUBLISH_WORKFLOW_NAMES: tuple[str, ...] = ("release-desktop.yml",)

# A workflow that runs this script is a "host". `pull_request` resolves the
# workflow file from the PR merge ref, so a host carrying a `paths` /
# `paths-ignore` filter can be skipped by the very PR that adds it -- the gate
# would then never run for workflow changes, which is the hole this whole
# script exists to close. Hosts must therefore trigger on unfiltered
# `pull_request`.
LINT_SCRIPT_NAME = "lint_workflow_triggers.py"
PATH_FILTER_KEYS: tuple[str, ...] = ("paths", "paths-ignore")


def _normalise_on(on_field):
    if isinstance(on_field, str):
        return {on_field}
    if isinstance(on_field, list):
        return set(on_field)
    if isinstance(on_field, dict):
        return set(on_field.keys())
    return set()


def _load_workflow(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding = "utf-8"))
    except Exception as exc:
        print(f"ERROR: failed to parse {path}: {exc}", file = sys.stderr)
        sys.exit(2)


def _extract_cache_keys(path: Path) -> list[str]:
    text = path.read_text(encoding = "utf-8")
    keys: list[str] = []
    for m in re.finditer(r"(?:^|\n)\s*key:\s*([^\n]+)", text):
        keys.append(m.group(1).strip())
    return keys


def _on_field(yaml_doc):
    # PyYAML resolves a bare `on:` key to the boolean True.
    on = yaml_doc.get(True) if isinstance(yaml_doc, dict) else None
    if on is None and isinstance(yaml_doc, dict):
        on = yaml_doc.get("on")
    return on


def _trigger_set(yaml_doc) -> set[str]:
    return _normalise_on(_on_field(yaml_doc))


def _runs_lint(yaml_doc) -> bool:
    """True when a job step actually runs this script.

    Reads the parsed steps rather than the raw text: a commented-out
    `# - run: python3 scripts/lint_workflow_triggers.py` executes nothing, and
    counting it as a host would let a deleted gate look wired.
    """
    jobs = yaml_doc.get("jobs") if isinstance(yaml_doc, dict) else None
    if not isinstance(jobs, dict):
        return False
    for job in jobs.values():
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            if LINT_SCRIPT_NAME in str(step.get("run") or ""):
                return True
    return False


def _pull_request_path_filters(yaml_doc) -> list[str]:
    """Path-filter keys on the `pull_request` trigger, if any."""
    on = _on_field(yaml_doc)
    if not isinstance(on, dict):
        return []
    pr = on.get("pull_request")
    if not isinstance(pr, dict):
        return []
    return [k for k in PATH_FILTER_KEYS if k in pr]


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--workflows-dir",
        type = Path,
        default = DEFAULT_WORKFLOWS_DIR,
        help = "Override the workflows directory (used by tests).",
    )
    parser.add_argument(
        "--require-host",
        action = "store_true",
        default = None,
        help = "Require a workflow that runs this script on unfiltered "
        "`pull_request`. Defaults on for the live tree, off for a "
        "fixture directory.",
    )
    parser.add_argument(
        "--no-require-host",
        dest = "require_host",
        action = "store_false",
        help = "Skip the host-wiring check.",
    )
    args = parser.parse_args()
    workflows_dir = args.workflows_dir

    require_host = args.require_host
    if require_host is None:
        require_host = workflows_dir.resolve() == DEFAULT_WORKFLOWS_DIR.resolve()

    findings: list[str] = []
    # GitHub Actions loads BOTH `.yml` and `.yaml` from .github/workflows/, so
    # scanning only `*.yml` leaves a rename-away bypass: `evil.yaml` with
    # `pull_request_target` would run for real and lint clean.
    workflows = sorted(list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")))
    pr_triggered: list[tuple[Path, list[str]]] = []
    publish_triggered: list[tuple[Path, list[str]]] = []
    unfiltered_hosts: list[Path] = []

    for path in workflows:
        doc = _load_workflow(path)
        triggers = _trigger_set(doc)

        for t in BANNED_TRIGGERS:
            if t in triggers:
                findings.append(
                    f"{path.name}: BANNED trigger '{t}' (GHSA-g7cv-rxg3-hmpx "
                    "pattern: fork PRs run in base-repo context). Switch to "
                    "'pull_request' and use a deploy-on-merge workflow for "
                    "any privileged step."
                )

        for t in RESTRICTED_TRIGGERS:
            if t in triggers:
                text = path.read_text(encoding = "utf-8")
                if "lint:workflow_triggers-allow-workflow_run" not in text:
                    findings.append(
                        f"{path.name}: RESTRICTED trigger '{t}' requires an "
                        "explicit `# lint:workflow_triggers-allow-workflow_run` "
                        "comment somewhere in the file, with a justification."
                    )

        if _runs_lint(doc):
            filters = _pull_request_path_filters(doc)
            if filters:
                findings.append(
                    f"{path.name}: runs {LINT_SCRIPT_NAME} but its "
                    f"'pull_request' trigger has {' + '.join(filters)}. A PR "
                    "adding that filter skips this workflow for its own PR, so "
                    "the trigger gate never runs on the workflow change it is "
                    "meant to review. Drop the filter, or host the lint in a "
                    "workflow that runs on every PR."
                )
            elif "pull_request" in triggers:
                unfiltered_hosts.append(path)

        if "pull_request" in triggers:
            pr_triggered.append((path, _extract_cache_keys(path)))
        is_dispatch_only = "workflow_dispatch" in triggers and not (
            "push" in triggers or "pull_request" in triggers
        )
        if path.name in PUBLISH_WORKFLOW_NAMES or is_dispatch_only:
            publish_triggered.append((path, _extract_cache_keys(path)))

    if require_host and not unfiltered_hosts:
        findings.append(
            f"no workflow runs {LINT_SCRIPT_NAME} on an unfiltered "
            "'pull_request' trigger, so this gate does not cover every PR. "
            "Restore the workflow-trigger-lint workflow."
        )

    pr_keys = {key for _, keys in pr_triggered for key in keys}
    for pub_path, pub_keys in publish_triggered:
        for k in pub_keys:
            if k in pr_keys:
                findings.append(
                    f"{pub_path.name}: cache key {k!r} is also declared in a "
                    "PR-triggered workflow. A fork PR could poison this cache "
                    "and the publish workflow would restore it on next run. "
                    "Add a unique suffix (e.g. '-publish-only') to partition "
                    "the namespaces."
                )

    if findings:
        print("Workflow trigger lint failed with the following issues:", file = sys.stderr)
        for f in findings:
            print(f"  - {f}", file = sys.stderr)
        return 1

    print(
        f"OK: scanned {len(workflows)} workflow file(s); "
        f"no pull_request_target, no unjustified workflow_run, "
        f"no PR/publish cache-key collision."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
