# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A second push to a pull request must cancel the first push's run.

``test_main_runs_survive_merge_bursts.py`` is the other half of this question and stops
one file short of it. That file scans ``_protected()``, which is every workflow with
``push: branches: [main]``, and asks whether two commits land in DIFFERENT groups; its
``test_a_pull_request_still_gets_latest_only`` then asks whether two commits on a pull
request land in the SAME group. Neither is the whole invariant:

  1. Its scan starts from ``push: branches: [main]``, so a workflow triggered ONLY by
     ``pull_request`` is outside it entirely. runner-pool-probe.yml sat there with no
     ``concurrency:`` block at all -- a ten-runner matrix, four cells macOS at 10x the
     minute rate, kept alive in full by every superseding push.

  2. A shared group is necessary and not sufficient. GitHub cancels a PENDING run when a
     newer one queues into its group, but a run that has already STARTED is only cancelled
     when ``cancel-in-progress`` is truthy. The expensive case is exactly the one a shared
     group does not cover: the old run is executing, which is when it is holding runners.

So this file asks the remaining question, of every pull-request-triggered workflow rather
than of the main-push ones: on a pull request ref, does ``cancel-in-progress`` evaluate
true? ``cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`` is the repo's usual
form and it is an expression, not a literal, so it is rendered rather than grepped -- the
reversed form is the same substrings in the same order and means the opposite.

Three workflows are exempt and each says why at its own ``concurrency:`` block. They are
listed below with the reason restated, because an exemption whose justification lives only
in another file is an exemption nobody re-reads.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Cancelling the GitHub runner cannot stop a Kaggle kernel it has already pushed, and an
# orphaned kernel bills quota to its own ceiling with nobody left to read the result.
# Superseded runs that have not STARTED are still discarded, which is the cheap half.
QUOTA_BOUND = frozenset({"kaggle-t4-notebook-ci.yml", "kaggle-t4-studio-gpu-ci.yml"})

# The other exemption this rule admits has no instance in THIS repo, and is written down
# because the next one will look like it: a workflow triggered by `types: [labeled]` only.
# Concurrency is claimed before the job-level `if` runs, so an unrelated label added to the
# same pull request takes the group and cancels a live matrix that the label has nothing to
# do with. unsloth-zoo's gemma4-audio-probe.yml is that shape. Both label-triggered
# workflows here -- windows-installer-differential-ci.yml and
# windows-amsi-defender-differential-ci.yml -- instead put the label in the GROUP, which
# separates the unrelated label rather than exempting the workflow, and is the better fix.
EXEMPT = QUOTA_BOUND

A_PULL_REQUEST = "refs/pull/9082/merge"
MAIN = "refs/heads/main"

_INTERPOLATION = re.compile(r"\$\{\{(.*?)\}\}", re.S)
_COMPARISON = re.compile(r"(.+?)(==|!=)(.+)")
_TERNARY = re.compile(r"(.+?)&&(.+?)\|\|(.+)")


class Unparsed(Exception):
    """An expression this evaluator does not model.

    Raised rather than guessed, for the same reason the merge-burst guard raises it: a
    guess would silently answer the one question the file exists to ask.
    """


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


def _context(ref: str) -> dict[str, str]:
    return {
        "github.workflow": "a-workflow",
        "github.ref": ref,
        "github.sha": "a" * 40,
        "github.event_name": "pull_request",
        "github.repository": "unslothai/unsloth",
        "github.ref_name": ref.rsplit("/", 1)[-1],
    }


def _cancels(value, *, ref: str) -> bool:
    """Whether ``cancel-in-progress: <value>`` is truthy for a run on ``ref``.

    A literal ``true`` is a bool once YAML has read it. Everything else in this repo is an
    expression, and an expression is the case worth evaluating: the whole of
    ``${{ github.ref != 'refs/heads/main' }}`` and its reversal are the same tokens.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip()
    match = _INTERPOLATION.fullmatch(text)
    if not match:
        raise Unparsed(text)
    body = match.group(1).strip()
    context = _context(ref)
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
    return None  # A bare string group is the default, which is false.


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
    """No block at all is the failure that has actually happened here.

    Checked separately from the rendering below so the message names the real cause. A
    missing block is not a mis-evaluated expression, and runner-pool-probe.yml reached main
    as one.
    """
    offenders = sorted(name for name, document in _scanned().items() if not _group(document))
    assert not offenders, (
        f"{offenders} are triggered by pull_request and declare no concurrency group, so "
        f"every superseded push keeps its runners until the jobs finish on their own. Add "
        f"group: ${{{{ github.workflow }}}}-${{{{ github.ref }}}} with cancel-in-progress."
    )


def test_every_pull_request_workflow_cancels_the_superseded_run():
    """The half a shared group does not cover: a run that has already started.

    GitHub discards a PENDING run when a newer one takes its group regardless of this
    setting. An EXECUTING one is the run holding the runners, and only cancel-in-progress
    reaches it.
    """
    offenders = {}
    for name, document in _scanned().items():
        if not _group(document):
            continue  # Reported by the test above; one cause, one failure.
        try:
            cancels = _cancels(_cancel_setting(document), ref = A_PULL_REQUEST)
        except Unparsed as exc:
            continue  # Reported by test_every_cancel_expression_is_understood.
        if not cancels:
            offenders[name] = _cancel_setting(document)
    assert not offenders, (
        f"{offenders} do not cancel in progress on a pull request ref, so a push that "
        f"supersedes a RUNNING job leaves it holding its runners to completion. Either set "
        f"cancel-in-progress: ${{{{ github.ref != 'refs/heads/main' }}}}, or add the file to "
        f"EXEMPT in {Path(__file__).name} with the reason it must not be cancelled."
    )


def test_cancelling_is_still_gated_off_main():
    """Fixing the pull-request half must not cancel main runs on the way past.

    The merge-burst incident this repo wrote down is the opposite failure, and a blanket
    ``cancel-in-progress: true`` on a workflow that also runs on main re-creates it. So the
    same expression is rendered on a main ref and required to be false wherever the
    workflow actually pushes to main.
    """
    offenders = {}
    for name, document in _scanned().items():
        triggers = document.get(True) or document.get("on") or {}
        push = triggers.get("push") if isinstance(triggers, dict) else None
        if not (isinstance(push, dict) and "main" in (push.get("branches") or [])):
            continue
        try:
            if _cancels(_cancel_setting(document), ref = MAIN):
                offenders[name] = _cancel_setting(document)
        except Unparsed:
            continue
    assert not offenders, (
        f"{offenders} also run on pushes to main and cancel in progress there, so a merge "
        f"burst kills the main run mid-flight. Gate it on github.ref != 'refs/heads/main'."
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
    """The assertions above are only worth anything if this holds.

    Every string here mentions github.ref and refs/heads/main, so a substring test calls
    them all the same. They are not.
    """
    gated = "${{ github.ref != 'refs/heads/main' }}"
    reversed_ = "${{ github.ref == 'refs/heads/main' }}"

    assert _cancels(gated, ref = A_PULL_REQUEST)
    assert not _cancels(gated, ref = MAIN)
    assert not _cancels(reversed_, ref = A_PULL_REQUEST), "a reversed comparison cancels nothing"
    assert _cancels(reversed_, ref = MAIN)

    assert _cancels(True, ref = A_PULL_REQUEST)
    assert not _cancels(False, ref = A_PULL_REQUEST)
    assert not _cancels(None, ref = A_PULL_REQUEST), "absent means false, which is the default"

    # The ternary form, which renders to a string rather than to a bool.
    ternary = "${{ github.ref == 'refs/heads/main' && 'false' || 'true' }}"
    assert _cancels(ternary, ref = A_PULL_REQUEST)
    assert not _cancels(ternary, ref = MAIN)


def test_the_scan_actually_found_the_workflows():
    """A glob that matched nothing would pass every check above."""
    scanned = _scanned()
    assert len(scanned) > 20, f"only found {len(scanned)} pull_request workflows; the scan is wrong"
    assert "runner-pool-probe.yml" in scanned, "the workflow that motivated this guard left the scan"
    assert "lint-ci.yml" in scanned


def test_the_exemptions_still_name_workflows_that_exist():
    """An exemption pointing at a moved file silently widens to nothing."""
    documents = _documents()
    missing = sorted(name for name in EXEMPT if name not in documents)
    assert not missing, f"EXEMPT names workflows that no longer exist: {missing}"

    # ... and each one must still be declining to cancel. An exemption for a workflow that
    # now cancels anyway is dead weight that hides the next real one.
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
    """Where it is invoked from is part of what it checks.

    The regression it catches is an edit to some other workflow's concurrency block. No
    workflow in this repo filters on .github/workflows/**, so a pull request touching only
    runner-pool-probe.yml collects no test that reads it. workflow-trigger-lint.yml carries
    no paths filter, by design, so it is the one job that sees such a pull request.
    """
    lint = WORKFLOWS / "workflow-trigger-lint.yml"
    text = lint.read_text(encoding = "utf-8")
    assert Path(__file__).name in text, (
        f"{lint.name} no longer runs {Path(__file__).name}, so this guard is absent from "
        f"exactly the pull requests it exists to check: the ones that edit a workflow and "
        f"nothing else."
    )
