#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Refuse dangerous GitHub Actions trigger patterns at PR time.

Two patterns are banned outright, both of which powered the TanStack
GHSA-g7cv-rxg3-hmpx supply-chain compromise:

1.  `pull_request_target` -- runs a fork's workflow YAML against the
    BASE repository's secrets and permissions. The fork can inject
    arbitrary code into the base context. The TanStack worm used this
    to land base-context execution from a fork PR. There is essentially
    no safe use of this trigger for a public open-source project;
    `pull_request` is the safe alternative.

2.  `workflow_run` chained to a PR-triggered workflow -- carries the
    same trust boundary problem one hop later. If a PR-triggered
    workflow can poison artifacts/caches and a `workflow_run` trigger
    fires off the result with elevated permissions, the attacker still
    reaches the trusted context.

3.  Shared cache keys between PR-triggered workflows and publish /
    release / push-triggered workflows. The TanStack worm poisoned the
    Actions cache from a fork PR and the legitimate release workflow
    then restored the poisoned cache. Cache keys must be partitioned
    so that nothing a PR can write is ever read by a workflow that
    holds secrets.

Exit codes
==========

  0   no findings
  1   one or more findings; stderr lists each with file path

Run from repo root:
    python3 scripts/lint_workflow_triggers.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "ERROR: PyYAML is required. Install with 'pip install pyyaml'", file = sys.stderr
    )
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

BANNED_TRIGGERS: tuple[str, ...] = ("pull_request_target",)
RESTRICTED_TRIGGERS: tuple[str, ...] = ("workflow_run",)
PUBLISH_WORKFLOW_NAMES: tuple[str, ...] = ("release-desktop.yml",)


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
        return yaml.safe_load(path.read_text())
    except Exception as exc:
        print(f"ERROR: failed to parse {path}: {exc}", file = sys.stderr)
        sys.exit(2)


def _extract_cache_keys(path: Path) -> list[str]:
    text = path.read_text()
    keys: list[str] = []
    for m in re.finditer(r"(?:^|\n)\s*key:\s*([^\n]+)", text):
        keys.append(m.group(1).strip())
    return keys


def _trigger_set(yaml_doc) -> set[str]:
    on = yaml_doc.get(True)
    if on is None:
        on = yaml_doc.get("on")
    return _normalise_on(on)


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument(
        "--workflows-dir",
        type = Path,
        default = DEFAULT_WORKFLOWS_DIR,
        help = "Override the workflows directory (used by tests).",
    )
    args = parser.parse_args()
    workflows_dir = args.workflows_dir

    findings: list[str] = []
    workflows = sorted(workflows_dir.glob("*.yml"))
    pr_triggered: list[tuple[Path, list[str]]] = []
    publish_triggered: list[tuple[Path, list[str]]] = []

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
                text = path.read_text()
                if "lint:workflow_triggers-allow-workflow_run" not in text:
                    findings.append(
                        f"{path.name}: RESTRICTED trigger '{t}' requires an "
                        "explicit `# lint:workflow_triggers-allow-workflow_run` "
                        "comment somewhere in the file, with a justification."
                    )

        if "pull_request" in triggers:
            pr_triggered.append((path, _extract_cache_keys(path)))
        is_dispatch_only = "workflow_dispatch" in triggers and not (
            "push" in triggers or "pull_request" in triggers
        )
        if path.name in PUBLISH_WORKFLOW_NAMES or is_dispatch_only:
            publish_triggered.append((path, _extract_cache_keys(path)))

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
        print(
            "Workflow trigger lint failed with the following issues:", file = sys.stderr
        )
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
