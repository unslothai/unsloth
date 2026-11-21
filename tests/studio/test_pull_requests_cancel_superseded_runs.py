# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Superseded pull-request runs must cancel; main, scheduled and manual runs must never be cancelled."""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Cancelling the runner cannot stop a pushed Kaggle kernel, which keeps billing quota.
QUOTA_BOUND = frozenset({"kaggle-t4-notebook-ci.yml", "kaggle-t4-studio-gpu-ci.yml"})

# A `types: [labeled]`-only workflow would also need care: concurrency is claimed before
# the job `if`. Putting the label in the group is the better fix.
EXEMPT = QUOTA_BOUND

A_PULL_REQUEST = "refs/pull/9082/merge"
MAIN = "refs/heads/main"

_INTERPOLATION = re.compile(r"\$\{\{(.*?)\}\}", re.S)
_COMPARISON = re.compile(r"(.+?)(==|!=)(.+)")
_TERNARY = re.compile(r"(.+?)&&(.+?)\|\|(.+)")


class Unparsed(Exception):
    """Raised for an expression the evaluator does not model, rather than guessing its value."""


def _documents() -> dict[str, dict]:
    out = {}
    for path in sorted(WORKFLOWS.glob("*.y*ml")):
        try:
            document = yaml.safe_load(path.read_text(encoding = "utf-8"))
        except yaml.YAMLError:
            continue
        if isinstance(document, dict):
            out[path.name] = document
    return out


def _term(text: str, context: dict[str, str]) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == "'":
        return text[1:-1]
    if text in context:
        return context[text]
    raise Unparsed(text)


def _condition(text: str, context: dict[str, str]) -> bool:
    match = _COMPARISON.fullmatch(text.strip())
    if not match:
        return bool(_term(text, context))
    left, right = _term(match.group(1), context), _term(match.group(3), context)
    return left == right if match.group(2) == "==" else left != right


def _context(ref: str, event_name: str = "pull_request") -> dict[str, str]:
    return {
        "github.workflow": "a-workflow",
        "github.ref": ref,
        "github.sha": "a" * 40,
        "github.event_name": event_name,
        "github.repository": "unslothai/unsloth",
        "github.ref_name": ref.rsplit("/", 1)[-1],
    }


def _cancels(
    value,
    *,
    ref: str,
    event_name: str = "pull_request",
) -> bool:
    """Evaluates the expression rather than grepping it, since a reversed comparison has the same tokens."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip()
    match = _INTERPOLATION.fullmatch(text)
    if not match:
        raise Unparsed(text)
    body = match.group(1).strip()
    context = _context(ref, event_name)
    ternary = _TERNARY.fullmatch(body)
    if ternary:
        taken = ternary.group(2) if _condition(ternary.group(1), context) else ternary.group(3)
        return _term(taken, context).lower() not in ("", "false")
    return _condition(body, context)


def _group(document: dict) -> str:
    concurrency = document.get("concurrency")
    if isinstance(concurrency, str):
        return concurrency
    if isinstance(concurrency, dict):
        return str(concurrency.get("group", ""))
    return ""


def _cancel_setting(document: dict):
    concurrency = document.get("concurrency")
    if isinstance(concurrency, dict):
        return concurrency.get("cancel-in-progress")
    return None


def _on_pull_requests(document: dict) -> bool:
    triggers = document.get(True) or document.get("on") or {}
    return isinstance(triggers, dict) and "pull_request" in triggers


def _scanned() -> dict[str, dict]:
    return {
        name: document
        for name, document in _documents().items()
        if name not in EXEMPT and _on_pull_requests(document)
    }


def test_every_pull_request_workflow_declares_concurrency():
    """Checked apart from rendering; runner-pool-probe.yml once reached main with no block at all."""
    offenders = sorted(name for name, document in _scanned().items() if not _group(document))
    assert not offenders, (
        f"{offenders} are triggered by pull_request and declare no concurrency group, so "
        f"every superseded push keeps its runners until the jobs finish on their own. Add "
        f"group: ${{{{ github.workflow }}}}-${{{{ github.ref }}}} with cancel-in-progress."
    )


def test_every_pull_request_workflow_cancels_the_superseded_run():
    """Only cancel-in-progress reaches an executing run; a shared group discards just pending ones."""
    offenders = {}
    for name, document in _scanned().items():
        if not _group(document):
            continue
        try:
            cancels = _cancels(_cancel_setting(document), ref = A_PULL_REQUEST)
        except Unparsed as exc:
            continue
        if not cancels:
            offenders[name] = _cancel_setting(document)
    assert not offenders, (
        f"{offenders} do not cancel in progress on a pull request ref, so a push that "
        f"supersedes a RUNNING job leaves it holding its runners to completion. Either set "
        f"cancel-in-progress: ${{{{ github.event_name == 'pull_request' }}}}, or add the file to "
        f"EXEMPT in {Path(__file__).name} with the reason it must not be cancelled."
    )


def test_cancelling_is_still_gated_off_main():
    """A blanket cancel-in-progress: true would cancel main runs and recreate the merge-burst incident."""
    offenders = {}
    for name, document in _scanned().items():
        triggers = document.get(True) or document.get("on") or {}
        push = triggers.get("push") if isinstance(triggers, dict) else None
        if not (isinstance(push, dict) and "main" in (push.get("branches") or [])):
            continue
        try:
            if _cancels(_cancel_setting(document), ref = MAIN, event_name = "push"):
                offenders[name] = _cancel_setting(document)
        except Unparsed:
            continue
    assert not offenders, (
        f"{offenders} also run on pushes to main and cancel in progress there, so a merge "
        f"burst kills the main run mid-flight. Gate it on github.event_name == 'pull_request'."
    )


# schedule and repository_dispatch run on the default branch; workflow_dispatch on any ref.
_DISPATCH_REFS = ("refs/heads/main", "refs/heads/a-feature-branch", "refs/tags/v2026.9.1")
_DEFAULT_BRANCH_ONLY = ("schedule", "repository_dispatch")


def _non_pull_request_runs(triggers: dict):
    for event in _DEFAULT_BRANCH_ONLY:
        if event in triggers:
            yield event, MAIN
    if "workflow_dispatch" in triggers:
        for ref in _DISPATCH_REFS:
            yield "workflow_dispatch", ref
    push = triggers.get("push")
    if "push" in triggers:
        branches = (push or {}).get("branches") if isinstance(push, dict) else None
        tags = (push or {}).get("tags") if isinstance(push, dict) else None
        for branch in branches or ["main", "a-feature-branch"]:
            yield "push", f"refs/heads/{branch}"
        for tag in tags or []:
            yield "push", f"refs/tags/{tag}"


def test_only_a_superseded_pull_request_run_is_cancelled():
    """A ref check alone still cancels branch dispatches, so cancelling is gated per trigger and ref."""
    offenders = {}
    for name, document in _scanned().items():
        triggers = document.get(True) or document.get("on") or {}
        if not isinstance(triggers, dict):
            continue
        try:
            for event, ref in _non_pull_request_runs(triggers):
                if _cancels(_cancel_setting(document), ref = ref, event_name = event):
                    offenders.setdefault(name, []).append(f"{event}@{ref}")
        except Unparsed:
            continue
    assert not offenders, (
        f"{offenders} cancel a running job that is not a superseded pull request push. Use "
        f"cancel-in-progress: ${{{{ github.event_name == 'pull_request' }}}} so pushes to main, "
        f"schedules and dispatches always run to completion."
    )


def test_every_cancel_expression_is_understood():
    """A refusal to evaluate must be loud rather than a silent skip."""
    unreadable = {}
    for name, document in _scanned().items():
        if not _group(document):
            continue
        try:
            _cancels(_cancel_setting(document), ref = A_PULL_REQUEST)
        except Unparsed as exc:
            unreadable[name] = f"{_cancel_setting(document)!r} contains {exc}"
    assert not unreadable, (
        f"the evaluator in this file cannot read {unreadable}, so it cannot say whether "
        f"those workflows supersede a running job. Extend _cancels rather than exempting "
        f"the workflow."
    )


def test_the_evaluator_reads_the_direction_of_the_comparison():
    """Every string mentions github.ref and refs/heads/main, so only evaluation tells == from !=."""
    gated = "${{ github.ref != 'refs/heads/main' }}"
    reversed_ = "${{ github.ref == 'refs/heads/main' }}"

    assert _cancels(gated, ref = A_PULL_REQUEST)
    assert not _cancels(gated, ref = MAIN)
    assert not _cancels(reversed_, ref = A_PULL_REQUEST), "a reversed comparison cancels nothing"
    assert _cancels(reversed_, ref = MAIN)

    assert _cancels(True, ref = A_PULL_REQUEST)
    assert not _cancels(False, ref = A_PULL_REQUEST)
    assert not _cancels(None, ref = A_PULL_REQUEST), "absent means false, which is the default"

    by_event = "${{ github.event_name == 'pull_request' }}"
    assert _cancels(by_event, ref = A_PULL_REQUEST)
    assert not _cancels(by_event, ref = MAIN, event_name = "push")
    assert not _cancels(by_event, ref = "refs/heads/a-feature-branch", event_name = "workflow_dispatch")
    assert not _cancels(by_event, ref = MAIN, event_name = "schedule")
    assert _cancels(gated, ref = "refs/heads/a-feature-branch", event_name = "workflow_dispatch")

    # The ternary form renders to a string rather than a bool.
    ternary = "${{ github.ref == 'refs/heads/main' && 'false' || 'true' }}"
    assert _cancels(ternary, ref = A_PULL_REQUEST)
    assert not _cancels(ternary, ref = MAIN)


def test_the_scan_actually_found_the_workflows():
    """A glob that matched nothing would pass every check above."""
    scanned = _scanned()
    assert len(scanned) > 20, f"only found {len(scanned)} pull_request workflows; the scan is wrong"
    assert (
        "runner-pool-probe.yml" in scanned
    ), "the workflow that motivated this guard left the scan"
    assert "lint-ci.yml" in scanned


def test_the_exemptions_still_name_workflows_that_exist():
    """An exemption pointing at a moved file silently widens to nothing."""
    documents = _documents()
    missing = sorted(name for name in EXEMPT if name not in documents)
    assert not missing, f"EXEMPT names workflows that no longer exist: {missing}"

    # An exemption for a workflow that now cancels anyway hides the next real one.
    pointless = sorted(
        name
        for name in EXEMPT
        if name in documents and _cancels(_cancel_setting(documents[name]), ref = A_PULL_REQUEST)
    )
    assert not pointless, (
        f"{pointless} are exempt from this guard but cancel in progress regardless. Drop "
        f"them from EXEMPT so the list keeps meaning what it says."
    )


def test_this_guard_runs_on_a_workflow_only_pull_request():
    """Runs from workflow-trigger-lint.yml, which has no paths filter, so workflow-only PRs reach it."""
    lint = WORKFLOWS / "workflow-trigger-lint.yml"
    text = lint.read_text(encoding = "utf-8")
    assert Path(__file__).name in text, (
        f"{lint.name} no longer runs {Path(__file__).name}, so this guard is absent from "
        f"exactly the pull requests it exists to check: the ones that edit a workflow and "
        f"nothing else."
    )
