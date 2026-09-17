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

# Kaggle spends an external GPU quota rather than runner minutes.
QUOTA_BOUND = frozenset({"kaggle-t4-notebook-ci.yml", "kaggle-t4-studio-gpu-ci.yml"})

# Phrasings that assert main runs are not cancelled.
CLAIMS_PROTECTION = re.compile(
    r"never on main|never cancelled on main|not cancelled on main|never cancels on main",
    re.IGNORECASE,
)


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


def _group(document: dict) -> str:
    concurrency = document.get("concurrency")
    if isinstance(concurrency, str):
        return concurrency
    if isinstance(concurrency, dict):
        return str(concurrency.get("group", ""))
    return ""


MAIN = "refs/heads/main"
A_PULL_REQUEST = "refs/pull/9082/merge"

_INTERPOLATION = re.compile(r"\$\{\{(.*?)\}\}", re.S)
_COMPARISON = re.compile(r"(.+?)(==|!=)(.+)")
_TERNARY = re.compile(r"(.+?)&&(.+?)\|\|(.+)")


class Unparsed(Exception):
    """A group expression this evaluator does not model.

    Raised rather than guessed. A guess here would silently answer the one question this
    file exists to ask, which is how a group behaves on main.
    """


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


def _evaluate(expression: str, context: dict[str, str]) -> str:
    ternary = _TERNARY.fullmatch(expression.strip())
    if ternary:
        taken = ternary.group(2) if _condition(ternary.group(1), context) else ternary.group(3)
        return _term(taken, context)
    return _term(expression, context)


def _render(
    group: str,
    *,
    ref: str,
    sha: str,
    event_name: str = "push",
) -> str:
    """The literal group string GitHub would compute for one run."""
    context = {
        "github.workflow": "a-workflow",
        "github.ref": ref,
        "github.sha": sha,
        "github.event_name": event_name,
        "github.repository": "unslothai/unsloth",
        "github.ref_name": ref.rsplit("/", 1)[-1],
    }
    return _INTERPOLATION.sub(lambda m: _evaluate(m.group(1), context), group)


def _is_per_commit_on_main(group: str) -> bool:
    """Whether two commits on main land in DIFFERENT groups.

    Evaluated rather than grepped. ``"github.sha" in group`` was the first form of this and
    it is not the invariant: reverse the conditional to
    ``github.ref != 'refs/heads/main' && github.sha || ''`` and the substring is still
    present, every workflow still names ``refs/heads/main``, and main commits share one
    group again. Rendering both sides asks the question directly, so which branch supplies
    the SHA is what decides the answer.
    """
    return _render(group, ref = MAIN, sha = "a" * 40) != _render(group, ref = MAIN, sha = "b" * 40)


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


def test_this_guard_runs_on_a_workflow_only_pull_request():
    """Where it is invoked from is part of what it checks.

    The regression this file catches is an edit to some OTHER workflow's concurrency block.
    No workflow filters on .github/workflows/**, so a PR touching only wheel-smoke.yml
    collects no test that reads it. workflow-trigger-lint.yml carries no paths filter at
    all, by design, so it is the one job that sees such a PR.
    """
    lint = WORKFLOWS / "workflow-trigger-lint.yml"
    text = lint.read_text(encoding = "utf-8")
    assert Path(__file__).name in text, (
        f"{lint.name} no longer runs {Path(__file__).name}. Backend CI does not filter on "
        f".github/workflows/**, so this guard would then be absent from exactly the pull "
        f"requests it exists to check: the ones that edit a workflow and nothing else."
    )
    triggers = yaml.safe_load(text).get(True) or yaml.safe_load(text).get("on") or {}
    pull_request = triggers.get("pull_request")
    assert isinstance(pull_request, dict) or pull_request is None, pull_request
    assert not (pull_request or {}).get("paths"), (
        f"{lint.name} gained a paths filter, so it stopped being the job that sees every "
        f"pull request and this guard is skippable again"
    )


def test_the_quota_bound_exemptions_still_exist():
    """An exemption naming a file that moved would silently widen to nothing."""
    documents = _documents()
    missing = sorted(name for name in QUOTA_BOUND if name not in documents)
    assert not missing, f"QUOTA_BOUND names workflows that no longer exist: {missing}"


def test_a_pull_request_still_gets_latest_only():
    """The protection is for main; superseding a PR push is still what we want.

    A group keyed on github.sha unconditionally would leave every abandoned PR run
    executing, which is the opposite of the intent and the expensive direction. Asked by
    rendering two commits on a pull request ref and requiring the SAME group.
    """
    offenders = {}
    for name, document in _protected().items():
        group = _group(document)
        first = _render(group, ref = A_PULL_REQUEST, sha = "a" * 40, event_name = "pull_request")
        second = _render(group, ref = A_PULL_REQUEST, sha = "b" * 40, event_name = "pull_request")
        if first != second:
            offenders[name] = group
    assert not offenders, (
        f"{offenders} put two commits on the same pull request in different concurrency "
        f"groups, so pushes stop superseding each other and every abandoned run keeps "
        f"burning a runner. Gate the SHA on github.ref == 'refs/heads/main'."
    )


def test_every_group_expression_is_understood():
    """The evaluator refuses to guess, and a refusal must be loud rather than a skip."""
    unreadable = {}
    for name, document in _protected().items():
        group = _group(document)
        try:
            _render(group, ref = MAIN, sha = "a" * 40)
        except Unparsed as exc:
            unreadable[name] = f"{group!r} contains {exc}"
    assert not unreadable, (
        f"the concurrency evaluator in this file cannot read {unreadable}, so it cannot say "
        f"whether those workflows survive a merge burst. Extend _term/_evaluate to cover the "
        f"new syntax rather than removing the workflow from the scan."
    )


def test_the_evaluator_reads_which_branch_supplies_the_sha():
    """The predicate above is only worth its assertions if this holds.

    Each of these renders to a string containing ``github.sha`` in its source text, and the
    substring test that this file first shipped with called all four per-commit. Only the
    first one is.
    """
    gated = "${{ github.ref }}-${{ github.ref == 'refs/heads/main' && github.sha || '' }}"
    reversed_ = "${{ github.ref }}-${{ github.ref != 'refs/heads/main' && github.sha || '' }}"
    wrong_branch = "${{ github.ref }}-${{ github.ref == 'refs/heads/main' && '' || github.sha }}"
    unconditional = "${{ github.ref }}-${{ github.sha }}"

    assert _is_per_commit_on_main(gated)
    assert not _is_per_commit_on_main(reversed_), "a reversed conditional shares one main group"
    assert not _is_per_commit_on_main(wrong_branch), "the SHA is on the pull request branch"
    assert _is_per_commit_on_main(unconditional)

    # ... and the pull request half, which is what disqualifies the unconditional form.
    def _same_on_a_pull_request(group: str) -> bool:
        return _render(group, ref = A_PULL_REQUEST, sha = "a" * 40) == _render(
            group, ref = A_PULL_REQUEST, sha = "b" * 40
        )

    assert _same_on_a_pull_request(gated)
    assert not _same_on_a_pull_request(unconditional)


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
