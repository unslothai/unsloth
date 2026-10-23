# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A merge burst on main runs the tip, never kills a running run, and nothing more.

How GitHub treats one concurrency group: at most one run EXECUTES and at most one is
PENDING. A newer run replaces the pending one ("any existing pending job or workflow in the
same concurrency group will be canceled and the new queued job or workflow will take its
place", workflow-syntax reference), and ``cancel-in-progress`` decides only whether the
EXECUTING run is killed as well.

This repo used to give every main commit its own group (``github.sha`` in the group on
main), so no main run ever replaced another. That bought a verdict per commit and cost
every workflow on every commit: about 550 runner-minutes per main commit at 100-160
commits a day, more than the org's runner pool supplies, so every queue (PRs, zoo, the
cache janitor) waited hours behind main.

The contract now, for every workflow that runs on pushes to main:

  1. One group per workflow, ref and event. A merge burst on main leaves the run already
     executing plus ONE pending run, which a newer push keeps replacing, so the tip always
     runs. The commits in between were tested as pull requests and the tip's tree contains
     them; a red tip is bisected with workflow_dispatch. That lost per-commit attribution on
     main is the trade, and it is deliberate.
  2. ``cancel-in-progress`` is false on main, so a run that has started always finishes
     (it is the one saving the main-branch caches, among other things).
  3. The event is in the group, so a nightly or a dispatch on main is never replaced by a
     push, and the reverse.
  4. On a pull request two pushes share a group, so the newer push supersedes the older.

Per-commit groups remain only where named below, with the reason.
"""

import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"

# Kaggle spends an external GPU quota rather than runner minutes.
QUOTA_BOUND = frozenset({"kaggle-t4-notebook-ci.yml", "kaggle-t4-studio-gpu-ci.yml"})

# Allowed to keep a group per commit on main. codeql.yml analyses each main commit for code
# scanning alerts and manages its own cache retention; it may coalesce too, so it is exempt
# from the rule rather than required to break it.
PER_COMMIT_ALLOWED = frozenset({"codeql.yml"})

# Phrasings that assert a main run is never cancelled or that main is grouped per commit.
# Either is false for a workflow sharing one group across main commits: its PENDING run is
# replaced by the next push.
CLAIMS_PROTECTION = re.compile(
    r"never on main|never cancelled on main|not cancelled on main|never cancels on main"
    r"|unique per commit on main|grouped per commit on main|one group per commit on main",
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
    """Raised rather than guessed: a guessed answer would silently misreport how a group behaves on main."""


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
    if _COMPARISON.fullmatch(expression.strip()):
        return "true" if _condition(expression, context) else "false"
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
    """Rendered, not grepped: a substring check still passes when the SHA is on the wrong branch."""
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


def _cancel_setting(document: dict) -> str:
    concurrency = document.get("concurrency")
    if isinstance(concurrency, dict):
        return str(concurrency.get("cancel-in-progress", False)).lower()
    return "false"


def _cancels_on_main_push(document: dict) -> bool:
    rendered = _render(_cancel_setting(document), ref = MAIN, sha = "a" * 40, event_name = "push")
    return rendered.strip().lower() == "true"


def _coalescing() -> dict[str, dict]:
    return {name: doc for name, doc in _protected().items() if name not in PER_COMMIT_ALLOWED}


def test_every_workflow_that_runs_on_main_coalesces_a_merge_burst():
    """Two main commits share one group, so a burst runs the executing commit plus the tip.

    A group per commit runs every workflow on every main commit, which is the load that
    queued the whole org for hours. Checked across all of them rather than a named few.
    """
    offenders = sorted(
        name for name, document in _coalescing().items() if _is_per_commit_on_main(_group(document))
    )
    assert not offenders, (
        f"{offenders} give every main commit its own concurrency group, so each main commit "
        f"runs the whole workflow and nothing is ever superseded. Use "
        f"${{{{ github.workflow }}}}-${{{{ github.ref }}}}-${{{{ github.event_name }}}} with "
        f"cancel-in-progress false on main, or add the file to PER_COMMIT_ALLOWED with a reason."
    )


def test_a_running_main_run_is_never_cancelled():
    """Coalescing replaces the PENDING run only; the one executing must finish."""
    offenders = {
        name: _cancel_setting(document)
        for name, document in _protected().items()
        if _cancels_on_main_push(document)
    }
    assert not offenders, (
        f"{offenders} cancel a running main run when the next commit lands, so a merge burst "
        f"can leave main with no finished run at all. Gate it on github.event_name == 'pull_request'."
    )


def test_a_push_never_replaces_a_nightly_or_a_dispatch_on_main():
    """The event is in the group, so runs of different kinds on main keep their own slot."""
    offenders = {}
    for name, document in _coalescing().items():
        triggers = document.get(True) or document.get("on") or {}
        group = _group(document)
        push = _render(group, ref = MAIN, sha = "a" * 40, event_name = "push")
        for event in ("schedule", "workflow_dispatch", "repository_dispatch"):
            if (
                event in triggers
                and _render(group, ref = MAIN, sha = "a" * 40, event_name = event) == push
            ):
                offenders.setdefault(name, []).append(event)
    assert not offenders, (
        f"{offenders} share the push group on main with another trigger, so a merge burst "
        f"replaces a pending nightly or dispatch. Put ${{{{ github.event_name }}}} in the group."
    )


def test_the_per_commit_allowance_still_names_workflows_that_exist():
    documents = _documents()
    missing = sorted(name for name in PER_COMMIT_ALLOWED if name not in documents)
    assert not missing, f"PER_COMMIT_ALLOWED names workflows that no longer exist: {missing}"


def test_the_scan_actually_found_the_workflows():
    """A glob that matched nothing would pass every check above."""
    protected = _protected()
    assert len(protected) > 20, f"only found {len(protected)} main workflows; the scan is wrong"
    assert "studio-backend-ci.yml" in protected
    assert len(_coalescing()) > 20, "the coalescing scan found almost nothing; the scan is wrong"


def test_this_guard_runs_on_a_workflow_only_pull_request():
    """Only workflow-trigger-lint.yml, with no paths filter, sees workflow-only PRs, so it must run this."""
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
    """PR runs share a group so superseded pushes cancel; per-SHA groups would leave them running."""
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
    """The per-commit predicate is only worth its assertions if this holds.

    Each of these contains ``github.sha`` in its source text, and a substring test would
    call all four per-commit. Only the first and the last are, so a grep cannot tell a
    coalescing group from a per-commit one; rendering can.
    """
    gated = "${{ github.ref }}-${{ github.ref == 'refs/heads/main' && github.sha || '' }}"
    reversed_ = "${{ github.ref }}-${{ github.ref != 'refs/heads/main' && github.sha || '' }}"
    wrong_branch = "${{ github.ref }}-${{ github.ref == 'refs/heads/main' && '' || github.sha }}"
    unconditional = "${{ github.ref }}-${{ github.sha }}"

    assert _is_per_commit_on_main(gated)
    assert not _is_per_commit_on_main(reversed_), "a reversed conditional shares one main group"
    assert not _is_per_commit_on_main(wrong_branch), "the SHA is on the pull request branch"
    assert _is_per_commit_on_main(unconditional)

    def _same_on_a_pull_request(group: str) -> bool:
        return _render(group, ref = A_PULL_REQUEST, sha = "a" * 40) == _render(
            group, ref = A_PULL_REQUEST, sha = "b" * 40
        )

    assert _same_on_a_pull_request(gated)
    assert not _same_on_a_pull_request(unconditional)


def test_no_workflow_claims_a_main_protection_it_does_not_have():
    """A comment promising per-commit or never-cancelled main runs must match the group.

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
        f"{offenders} claim main runs are per commit or never cancelled while sharing one "
        f"concurrency group across main commits, where a newer push replaces the pending run. "
        f"Describe what the group does (see studio-backend-ci.yml)."
    )
