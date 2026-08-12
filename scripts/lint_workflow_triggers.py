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
import shlex
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
# workflow file from the PR merge ref, so any narrowing a PR adds to its host
# takes effect for that same PR -- the gate then never runs on the workflow
# change it exists to review, which is the hole this whole script closes.
# `paths` is only the obvious key: `branches`, `branches-ignore` and `types`
# skip PRs just as well. So a host must trigger on a bare `pull_request:` with
# no configuration at all, and its lint step must be able to fail the run.
LINT_SCRIPT_NAME = "lint_workflow_triggers.py"


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


# Host detection. A host must run the repository's own lint script as a plain,
# standalone command with no arguments: `echo <script>` runs nothing;
# `/tmp/lint_workflow_triggers.py` is a decoy with the same basename; a
# pipeline, redirection or `|| true` detaches the step's exit status from the
# lint's (the default `run:` shell is `bash -e`, without pipefail); and any
# argument can point it elsewhere or turn it into a no-op, `--help` included.
_PYTHON = re.compile(r"[\w./-]*python[\w.]*")
LINT_SCRIPT_PATH = f"scripts/{LINT_SCRIPT_NAME}"
# Options that consume the next token, so the script path is not mistaken for
# their value.
_OPTS_WITH_VALUE = ("-X", "-W", "--check-hash-based-pycs")
_SHELL_OPERATORS = ("|", "&", ";", ">", "<", "`", "$(")


def _classify_lint_line(line: str) -> tuple[bool, str | None]:
    """(is an enforcing invocation, problem) for one line naming the script."""
    try:
        tokens = shlex.split(line.strip())
    except ValueError:
        return False, None
    if not tokens or not _PYTHON.fullmatch(tokens[0]):
        return False, None          # `echo <script>`, or not python at all

    args, i = tokens[1:], 0
    while i < len(args) and args[i].startswith("-"):
        if args[i].startswith(("-c", "-m")):
            return False, None      # runs that program, not this file
        i += 2 if args[i] in _OPTS_WITH_VALUE else 1

    rest = args[i:]
    if not rest or not rest[0].endswith(LINT_SCRIPT_PATH):
        return False, None
    if len(rest) == 1:
        return True, None

    trailing = " ".join(rest[1:])
    if any(op in tok for tok in rest[1:] for op in _SHELL_OPERATORS):
        return False, (
            f"its lint command is chained, piped or backgrounded ({trailing}), "
            "so the step's exit status need not be the lint's"
        )
    return False, (
        f"its lint command passes {trailing}, so it does not gate the live "
        "workflows with its own checks on"
    )


def _lint_step_report(run: str) -> tuple[bool, list[str]]:
    """(an enforcing invocation is present, problems with lint-ish commands)."""
    enforcing, problems = False, []
    for line in run.splitlines():
        if LINT_SCRIPT_NAME not in line:
            continue
        ok, problem = _classify_lint_line(line)
        enforcing = enforcing or ok
        if problem:
            problems.append(problem)
    if (enforcing or problems) and "set +e" in run:
        problems.append(
            "its lint step runs under 'set +e', so a non-zero exit does not "
            "fail the step"
        )
    return enforcing, problems


def _lint_steps(yaml_doc) -> list[tuple[dict, dict, bool, list[str]]]:
    """(job, step, enforcing, problems) for every step that runs the lint.

    Reads the parsed steps rather than the raw text: a commented-out
    `# - run: python3 scripts/lint_workflow_triggers.py` executes nothing, and
    counting it as a host would let a deleted gate look wired.
    """
    jobs = yaml_doc.get("jobs") if isinstance(yaml_doc, dict) else None
    if not isinstance(jobs, dict):
        return []
    found = []
    for job in jobs.values():
        steps = job.get("steps") if isinstance(job, dict) else None
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            enforcing, problems = _lint_step_report(str(step.get("run") or ""))
            if enforcing or problems:
                found.append((job, step, enforcing, problems))
    return found


def _pull_request_restrictions(yaml_doc) -> list[str]:
    """Keys narrowing the `pull_request` trigger. A gate wants none of them."""
    on = _on_field(yaml_doc)
    pr = on.get("pull_request") if isinstance(on, dict) else None
    return sorted(pr) if isinstance(pr, dict) else []


def _is_truthy(value) -> bool:
    """YAML truthiness, treating any `${{ ... }}` expression as possibly true."""
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().lower() not in ("", "false")


def main() -> int:
    # No abbreviations: `--workflows-d` must not silently mean
    # `--workflows-dir`, or a host could smuggle it past review.
    parser = argparse.ArgumentParser(description = __doc__, allow_abbrev = False)
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

        lint_steps = _lint_steps(doc)
        if lint_steps:
            problems = []
            restrictions = _pull_request_restrictions(doc)
            if restrictions:
                problems.append(
                    f"its 'pull_request' trigger is narrowed by "
                    f"{' + '.join(restrictions)}, so a PR adding that skips "
                    "this workflow for its own PR"
                )
            if any(
                _is_truthy(job.get("continue-on-error"))
                or _is_truthy(step.get("continue-on-error"))
                for job, step, _, _ in lint_steps
            ):
                problems.append(
                    "its lint step is continue-on-error, so findings cannot fail the run"
                )
            if any(
                job.get("if") is not None or step.get("if") is not None
                for job, step, _, _ in lint_steps
            ):
                problems.append(
                    "its lint step is gated by an 'if:' condition, so the gate "
                    "can be skipped while the run still succeeds"
                )
            if any(job.get("needs") is not None for job, _, _, _ in lint_steps):
                problems.append(
                    "its lint job declares 'needs:', so a skipped prerequisite "
                    "skips the gate without failing the run"
                )
            for _, _, _, step_problems in lint_steps:
                problems.extend(step_problems)
            if problems:
                findings.append(
                    f"{path.name}: runs {LINT_SCRIPT_NAME} but "
                    + ", and ".join(problems)
                    + ". The gate must run, and be able to fail, on every PR."
                )
            elif "pull_request" in triggers and any(
                enforcing for _, _, enforcing, _ in lint_steps
            ):
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
