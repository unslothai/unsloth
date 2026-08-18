# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Setting an action up must not be charged to the action.

`test_heavy_thread_harness_contract.py` pins the SHAPE of the harness and
`test_heavy_thread_repetition_rejection.py` pins the ARITHMETIC that turns three repetitions into
one number. This file pins the third way a measurement harness reports a number of something
other than what its label says: the counter window is opened too early, so work that establishes
the action's PRECONDITION lands in the action's own row.

The defect this file exists to keep out, previously live: the `scroll` and `jump` gestures both
have to start from the bottom of the thread to travel the distance they claim to travel, and both
did that anchoring in the first lines of their own evaluate. `window.__hv.begin()` came after it,
so the portable recorder correctly excluded it -- but `run_action` arms the long-task observer and
takes its CDP snapshot BEFORE the evaluate starts, so the anchoring was inside the Chromium-only
task, layout, style and long-task numbers.

That is not a uniform overhead that cancels out. Repetition 1 starts at the bottom already and
pays nothing; repetitions 2 and 3 start wherever the previous repetition left the viewport and pay
a full-height scroll, so the same gesture looks progressively more expensive down the column. A
`jump` arm ends at the top and pays a whole top-to-bottom reposition, while a control arm that
never moved pays nothing, so the two arms are not comparable in exactly the rows the comparison is
made from. Anchoring now runs from `ACTION_SETUPS`, before any snapshot, the mirror of the
`ACTION_RESETS` cleanup that runs after all of them.
"""

from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path

import pytest

WORKDIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _load_harness():
    """Import the harness module without needing a browser. See the sibling test for why."""
    os.environ.setdefault("PW_ART_DIR", str(WORKDIR / "logs" / "heavy-thread-artifacts"))
    if "playwright.sync_api" not in sys.modules:
        try:
            import playwright.sync_api  # noqa: F401
        except ImportError:
            package = types.ModuleType("playwright")
            module = types.ModuleType("playwright.sync_api")
            module.sync_playwright = None
            package.sync_api = module
            sys.modules["playwright"] = package
            sys.modules["playwright.sync_api"] = module
    import playwright_heavy_thread

    return playwright_heavy_thread


HARNESS = _load_harness()

# The gestures that cannot start from an arbitrary scroll position. Named explicitly rather than
# derived from ACTION_SETUPS, so that emptying ACTION_SETUPS fails this file instead of agreeing
# with it.
ANCHORED_ACTIONS = ("scroll", "jump")


# ── the ordering inside run_action ────────────────────────────────────


class _FakePage:
    """A page that records the order of the evaluates run against it."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def evaluate(
        self,
        script,
        arg = None,
    ):  # noqa: ANN001
        self.calls.append(f"evaluate:{_label(script)}")
        # Only the action script returns a payload; setups and resets return nothing.
        if _label(script) == "action":
            return {"metrics": {"frames": 1}, "scrolledPx": 100}
        return None


def _label(script) -> str:  # noqa: ANN001
    text = script if isinstance(script, str) else str(script)
    if text is _ACTION_SCRIPT or text == _ACTION_SCRIPT:
        return "action"
    if "scrollIntoView" in text or "scrollTo" in text:
        return "setup_or_reset"
    return "other"


_ACTION_SCRIPT = "async () => ({ metrics: {} })  // stand-in for the action itself"


@pytest.fixture()
def ordering(monkeypatch):
    """Drive `run_action` with every page and CDP touch recorded in call order."""
    calls: list[str] = []
    monkeypatch.setattr(HARNESS, "reset_long_tasks", lambda page: calls.append("arm_long_tasks"))
    monkeypatch.setattr(HARNESS, "cdp_metrics", lambda cdp: calls.append("cdp_snapshot") or {})
    # These two READ the counters, so they must appear in the recorded order as well. Stubbing
    # them silently made the cleanup test below unable to see the closing reads at all, and it
    # passed with the reset moved in front of them.
    monkeypatch.setattr(
        HARNESS,
        "cdp_counters",
        lambda before, after: calls.append("read_cdp_counters") or {"layout_count": 0},
    )
    monkeypatch.setattr(
        HARNESS,
        "long_task_summary",
        lambda page: calls.append("read_long_tasks") or {"long_tasks": 0},
    )
    return calls


def _run(calls, name: str):
    page = _FakePage(calls)
    return HARNESS.run_action(page, object(), name, _ACTION_SCRIPT, None)


@pytest.mark.parametrize("name", ANCHORED_ACTIONS)
def test_anchoring_runs_before_the_counters_are_armed(ordering, name):
    """The precondition scroll must be complete before anything starts counting.

    Goes red if the setup evaluate is moved after `reset_long_tasks` or after the `before`
    snapshot, and red if it is put back inside the action script, because then the only evaluate
    before the snapshot is gone.
    """
    _run(ordering, name)
    assert "evaluate:setup_or_reset" in ordering, f"{name} ran no setup evaluate at all: {ordering}"
    setup_at = ordering.index("evaluate:setup_or_reset")
    assert setup_at < ordering.index("arm_long_tasks"), (
        f"{name} anchored AFTER the long-task observer was armed, so the reposition lands in "
        f"this row's long_tasks/long_task_ms: {ordering}"
    )
    assert setup_at < ordering.index("cdp_snapshot"), (
        f"{name} anchored AFTER the opening CDP snapshot, so the reposition lands in this row's "
        f"task, layout and style counters: {ordering}"
    )


def test_an_action_with_no_precondition_runs_no_setup(ordering):
    """`keystroke` has nothing to position, and must not gain a scroll it did not ask for."""
    _run(ordering, "keystroke")
    assert (
        "evaluate:setup_or_reset" not in ordering
    ), f"keystroke ran a positioning evaluate it does not need: {ordering}"


def test_cleanup_still_runs_after_every_snapshot(ordering):
    """The mirror rule, kept here so setup and reset cannot drift apart.

    `jump` leaves the viewport at the top and restores it from ACTION_RESETS. That restore must
    stay after `long_task_summary`, or the row describes two jumps rather than one.
    """
    _run(ordering, "jump")
    reset_at = len(ordering) - 1 - ordering[::-1].index("evaluate:setup_or_reset")
    for read in ("cdp_snapshot", "read_cdp_counters", "read_long_tasks"):
        assert read in ordering, f"{read} never happened, so this test proves nothing: {ordering}"
        last_read = len(ordering) - 1 - ordering[::-1].index(read)
        assert reset_at > last_read, (
            f"jump's reset ran before {read}, so the restoring scroll is charged to the jump's "
            f"own row and the row describes two jumps: {ordering}"
        )


# ── the action scripts themselves ─────────────────────────────────────


def _script_source(name: str) -> str:
    return dict(
        scroll = HARNESS.SCROLL_JS,
        jump = HARNESS.JUMP_JS,
        keystroke = HARNESS.KEYSTROKE_JS,
        menu = HARNESS.MENU_JS,
        delete = HARNESS.DELETE_JS,
        reopen = HARNESS.REOPEN_JS,
    )[name]


@pytest.mark.parametrize("name", ANCHORED_ACTIONS)
def test_the_action_script_does_not_reposition_before_its_recorder_opens(name):
    """No viewport write may sit between the top of the script and `__hv.begin()`.

    This is the source-level half of the same rule. The ordering tests above prove the setup runs
    early; this proves the anchoring was actually REMOVED from the script rather than duplicated,
    which would leave the row correct and the gesture measuring a viewport that had already been
    moved twice.
    """
    prologue = _script_source(name).split("window.__hv.begin();")[0]
    code = "\n".join(line for line in prologue.splitlines() if not line.strip().startswith("//"))
    offenders = re.findall(r"scrollTo\(|scrollTop\s*=|scrollIntoView\(", code)
    assert not offenders, (
        f"{name} still repositions the viewport before its recorder window opens: {offenders}. "
        "The CDP counters are already armed at that point, so this work is charged to the action."
    )


@pytest.mark.parametrize("name", ANCHORED_ACTIONS)
def test_every_anchored_action_is_registered_for_setup(name):
    """A gesture that needs the bottom must be wired to something that puts it there.

    Without this, deleting an ACTION_SETUPS entry makes the ordering test above vacuous rather
    than red: there would simply be no setup evaluate to be early or late.
    """
    assert name in HARNESS.ACTION_SETUPS, (
        f"{name} starts from wherever the last repetition left the viewport, so it no longer "
        "travels the distance it reports"
    )


# ── the tests have to be run by something ─────────────────────────────
#
# Every defect this directory guards against was found because a check ran. A contract test that
# no workflow invokes is the same failure one level up: it passes locally forever, and the
# regression it exists to catch merges unnoticed. This has already happened twice on this branch,
# so it is pinned rather than remembered.

WORKFLOW = WORKDIR / ".github" / "workflows" / "studio-frontend-ci.yml"
CONTRACT_TESTS = sorted(
    path.name
    for path in (WORKDIR / "tests" / "studio").glob("test_*.py")
    if path.name.startswith(("test_heavy_thread_", "test_scroll_predecessor_"))
)


def test_there_are_contract_tests_to_check():
    """Without this the two checks below pass vacuously if the glob ever stops matching."""
    assert len(CONTRACT_TESTS) >= 3, CONTRACT_TESTS


@pytest.mark.parametrize("name", CONTRACT_TESTS)
def test_every_contract_test_is_invoked_by_the_frontend_workflow(name):
    text = WORKFLOW.read_text(encoding = "utf-8")
    run_blocks = text.split("python3 -m pytest")
    assert any(f"tests/studio/{name}" in block.split("\n\n")[0] for block in run_blocks[1:]), (
        f"{name} is not in any pytest invocation in studio-frontend-ci.yml, so nothing runs it "
        "there and a regression it covers can merge unnoticed"
    )


@pytest.mark.parametrize("name", CONTRACT_TESTS)
def test_every_contract_test_triggers_the_workflow_that_runs_it(name):
    """Listed in `paths` as well as in the run block. Without it an edit confined to the test
    itself changes no triggering path, so the job that would have run it never starts."""
    text = WORKFLOW.read_text(encoding = "utf-8")
    assert f"- 'tests/studio/{name}'" in text, (
        f"{name} is absent from the workflow's pull_request.paths, so editing it alone does not "
        "trigger the workflow that runs it"
    )


# ── the jump's start position survives aggregation ────────────────────


@pytest.mark.parametrize("field", ("startedFrom", "bottom"))
def test_the_jump_start_position_is_kept_per_repetition(field):
    """`action_failures` falls back to the aggregated value when the per-repetition list is
    missing, so dropping these from NUMERIC_PROOFS leaves the check running against one number
    instead of N and a single bad repetition among good ones stops being visible. That fallback is
    why asserting on `action_failures` alone could not catch this."""
    assert field in HARNESS.NUMERIC_PROOFS, (
        f"{field} is no longer retained per repetition, so a jump that started part-way up the "
        "thread on one repetition is hidden behind the aggregate"
    )


def test_one_bad_repetition_among_good_ones_is_rejected():
    """The behaviour the field retention exists for, driven rather than asserted structurally.

    Uses the sibling module's clean table so this cannot pass by being missing a key that
    `action_failures` would have tripped over anyway.
    """
    import test_heavy_thread_repetition_rejection as sibling

    actions = sibling.clean_actions()
    actions["jump"]["startedFrom_per_repetition"] = [8000, 3000, 8000]
    actions["jump"]["bottom_per_repetition"] = [8000, 8000, 8000]
    failures = HARNESS.action_failures("t", actions, sibling.COUNTS, sibling.VIEWPORT)
    assert any("above the bottom" in f and "repetition 2" in f for f in failures), failures


def test_a_short_per_repetition_list_does_not_silently_skip_repetitions():
    """`zip` truncates to the shorter list. If one of the two per-repetition lists is missing
    entries, the repetitions past that point would go unchecked and the check would quietly stop
    checking, which is the failure this whole section exists to prevent."""
    import test_heavy_thread_repetition_rejection as sibling

    actions = sibling.clean_actions()
    actions["jump"]["startedFrom_per_repetition"] = [8000, 8000, 8000]
    actions["jump"]["bottom_per_repetition"] = [8000]
    failures = HARNESS.action_failures("t", actions, sibling.COUNTS, sibling.VIEWPORT)
    assert any("repetition 3" in f for f in failures), (
        f"repetition 3 was never examined at all: {failures}"
    )
