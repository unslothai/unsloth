# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UI parity when one arm mounts a window, and the behavioural scoring that replaces it.

The structural digest asks "is the same DOM on screen". An arm whose entire purpose is to put less
DOM on screen answers no on every action, so the digest would print eighteen red rows that are all
the same non-finding and would bury anything real underneath them. These tests hold the line in
both directions: the digest must REFUSE such a pair rather than fail it, and the behavioural
scoring that stands in its place must still be able to fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.analysis import behaviour as B  # noqa: E402
from studiobench.analysis import parity as P  # noqa: E402


def _capture(
    mounted: int,
    total: int,
    digest: str = "d",
) -> dict:
    return {
        "parity_attempted": True,
        "root_kind": "thread",
        "digest": digest,
        "chars": 100,
        "messages": [
            {"i": i, "role": "assistant", "digest": f"{digest}{i}", "chars": 10}
            for i in range(mounted)
        ],
        "overlays": [],
        "styles": {"elements": mounted, "digest": "s", "capped": False},
        "mounted_messages": mounted,
        "thread_total": total,
    }


def _row(action: str, capture: dict, **expect) -> dict:
    return {
        "row_type": "action",
        "action": action,
        "ran": True,
        "expect_ok": True,
        "expect": dict(expect),
        "timings": {},
        "parity": capture,
        "census": {"viewport_scroll_height": expect.pop("_scroll_height", 10_000)},
    }


# ── the digest must refuse, not fail ────────────────────────────────


def test_a_windowed_capture_is_detected_from_its_own_numbers():
    assert P.windowed_mount(_capture(6, 18)) is True
    assert P.windowed_mount(_capture(18, 18)) is False
    # An old payload carries neither number and is treated as the full mount it was.
    assert P.windowed_mount({"parity_attempted": True, "digest": "d"}) is False


def test_the_digest_refuses_a_windowed_pair_rather_than_reporting_eighteen_differences():
    got = P.compare(_capture(18, 18, "base"), _capture(6, 18, "treat"))
    assert got["verdict"] == P.NOT_APPLICABLE
    assert got["verdict"] != P.DIFFER
    assert "mounts a WINDOW" in got["reason"]
    # And it localises nothing, because there is nothing here worth localising.
    assert got["moved"] == []
    assert got["style_verdict"] == P.NOT_APPLICABLE


def test_not_applicable_is_not_folded_into_a_pass_or_a_fail():
    tally = P.summarise([P.compare(_capture(18, 18, "b"), _capture(6, 18, "t"))])
    assert tally == {P.NOT_APPLICABLE: 1}
    assert tally.get(P.MATCH, 0) == 0
    assert tally.get(P.DIFFER, 0) == 0


def test_two_full_mounts_are_still_compared_exactly_as_before():
    """The refusal must not leak into a normal pair. Same digest is still MATCH, different is
    still DIFFER, and the windowed machinery is invisible to both."""
    same = P.compare(_capture(18, 18, "x"), _capture(18, 18, "x"))
    assert same["verdict"] == P.MATCH
    differ = P.compare(_capture(18, 18, "x"), _capture(18, 18, "y"))
    assert differ["verdict"] == P.DIFFER
    assert differ["moved"]


def test_unequal_mounted_counts_are_reported_even_without_a_declared_total():
    """THIS TEST USED TO ASSERT THE OPPOSITE, and it was wrong in the way the whole gate is meant
    to be proof against: it treated "these rows cannot be lined up" as "there is nothing to say".

    The rows genuinely cannot be compared position by position, and that is a statement about the
    ROWS. Two arms that each mounted their whole thread and arrived at different lengths have a
    difference between them whatever the rows can support, and calling the pair inapplicable
    hides exactly the regression -- a treatment rendering fewer messages -- that a parity check
    exists to catch.
    """
    got = P.compare(_capture(18, 18, "b"), _capture(12, 12, "t"))
    assert got["verdict"] == P.DIFFER
    assert "different numbers of messages" in got["reason"]
    assert got["moved"] == [], "the positional rows would all read as moved and bury the finding"


def test_a_refused_pair_is_not_evidence_of_stability_either():
    """`derive_unstable` counts observations. A pair the digest could not answer is not one."""
    derived = P.derive_unstable([("select_text", {"verdict": P.NOT_APPLICABLE})] * 4)
    assert derived["select_text"]["observations"] == 0
    assert derived["select_text"]["unstable"] is False
    assert derived["select_text"]["undetermined"] is True
    assert derived["select_text"]["not_comparable"] == 4


# ── the behavioural scoring that replaces it ────────────────────────


def test_the_scroll_extent_invariant_passes_a_virtualizer_that_sizes_its_spacers():
    base = _row("select_text", _capture(18, 18), selected_chars = 100, visible_chars = 100)
    treat = _row("select_text", _capture(6, 18), selected_chars = 100, visible_chars = 100)
    treat["census"]["viewport_scroll_height"] = 9_600  # 4% out, within the estimate tolerance
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.MATCH, got["reason"]


def test_the_scroll_extent_invariant_fails_a_virtualizer_that_simply_drops_rows():
    """A scrollbar that says the thread is a third of its real length is a user-visible defect,
    and it is the failure mode a windowed mount invites first."""
    base = _row("select_text", _capture(18, 18), selected_chars = 100, visible_chars = 100)
    treat = _row("select_text", _capture(6, 18), selected_chars = 100, visible_chars = 100)
    treat["census"]["viewport_scroll_height"] = 3_300
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN
    assert "scroll_extent" in got["reason"]


def _copy_row(
    capture,
    *,
    clipboard,
    selected,
    mounted,
    readable = True,
):
    return _row(
        "select_all_copy",
        capture,
        selected_chars = selected,
        clipboard_chars = clipboard,
        clipboard_readable = readable,
        clipboard_note = None,
        messages_total = 18,
        messages_mounted = mounted,
        mounted_fraction = round(mounted / 18, 3),
    )


def test_clipboard_truncation_is_reported_as_a_broken_invariant_not_as_noise():
    """A windowed thread whose copy path still reads the DOM loses conversation. That is data
    loss, and the report has to say so rather than file it under 'expected difference'."""
    base = _copy_row(_capture(18, 18), clipboard = 200_000, selected = 200_000, mounted = 18)
    treat = _copy_row(_capture(6, 18), clipboard = 66_000, selected = 66_000, mounted = 6)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN
    assert "clipboard_carries_the_whole_thread" in got["reason"]


def test_the_alarm_goes_quiet_when_the_copy_reads_the_store_and_not_before():
    """THE FIX, AND THE REASON IT COUNTS AS ONE.

    Same windowed arm, same shrunken selection -- six of eighteen messages mounted, so Ctrl+A can
    only reach a third of the text. What changed is that the app's copy handler serialises from
    the message store, so the CLIPBOARD is whole. The alarm has to go quiet for that and for
    nothing else: an alarm still wired to `selected_chars` would stay lit on a build that had
    fixed the defect, and an alarm that ignored the selection entirely could not tell this apart
    from a build that never virtualised at all.
    """
    base = _copy_row(_capture(18, 18), clipboard = 200_000, selected = 200_000, mounted = 18)
    treat = _copy_row(_capture(6, 18), clipboard = 200_000, selected = 66_000, mounted = 6)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.MATCH, got["reason"]
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_carries_the_whole_thread:base"]["ok"] is True
    assert checks["clipboard_carries_the_whole_thread:treatment"]["ok"] is True
    # And the shrunken selection is still on the record, as evidence rather than as a verdict.
    assert checks["selection_shrank_as_expected"]["ok"] is None
    assert "66000" in checks["selection_shrank_as_expected"]["detail"]


def test_an_unreadable_clipboard_is_never_a_pass():
    """The one invariant where "we could not tell" must not look like "it was fine"."""
    base = _copy_row(_capture(18, 18), clipboard = 200_000, selected = 200_000, mounted = 18)
    treat = _copy_row(
        _capture(6, 18),
        clipboard = None,
        selected = 66_000,
        mounted = 6,
        readable = False,
    )
    got = B.compare_behaviour(base, treat)
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_readable:treatment"]["ok"] is None
    assert got["verdict"] != P.MATCH


def test_a_reopen_that_loses_messages_is_broken():
    base = _row(
        "thread_reopen",
        _capture(18, 18),
        messages_before = 18,
        messages_after = 18,
        reopened_via = "click",
    )
    treat = _row(
        "thread_reopen",
        _capture(6, 18),
        messages_before = 18,
        messages_after = 6,
        reopened_via = "click",
    )
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN
    assert "reopen_keeps_every_message:treatment" in got["reason"]


def test_a_reopen_measured_through_a_page_navigation_is_broken():
    """A document reload is not a thread rebuild. See `_click_or_navigate`."""
    base = _row(
        "thread_reopen",
        _capture(18, 18),
        messages_before = 18,
        messages_after = 18,
        reopened_via = "click",
    )
    treat = _row(
        "thread_reopen",
        _capture(6, 18),
        messages_before = 18,
        messages_after = 18,
        reopened_via = "navigate",
    )
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN
    assert "reopen_used_the_control:treatment" in got["reason"]


# ── a rebuild that never finished is not a held invariant ───────────


def _reopen_row(
    mounted,
    *,
    before = 18,
    after = 18,
    ready = True,
):
    """A `thread_reopen` row in the shape scene/actions.py writes one.

    `ready = False` is what a rebuild that timed out produces: `ran` stays TRUE so the outstanding
    conditions travel with the row, `expect_ok` is false, `reopen_ms` is null, and the two message
    counts are UNCHANGED -- because both of them are `threadTotal()`, which is the total the store
    DECLARED and which the first reopened row publishes.
    """
    row = _row(
        "thread_reopen",
        _capture(mounted, 18),
        messages_before = before,
        messages_after = after,
        reopened_via = "click",
        reopen_ready_mode = "windowed" if mounted < 18 else "full",
        reopen_readiness = {
            "ready": ready,
            "mode": "windowed" if mounted < 18 else "full",
            "expected_messages": before,
            "conditions": {"settled": True, "end_present": bool(ready)},
            "probe": {"mounted": mounted, "setsize": after},
            "reason": None if ready else "the thread was not ready: end_present",
        },
    )
    row["expect_ok"] = bool(ready) and before == after
    row["timings"] = {"close_ms": 120.0, "reopen_ms": 900.0 if ready else None}
    if not row["expect_ok"]:
        row["reason"] = "the reopened thread never reached a ready state"
    return row


def test_a_reopen_that_never_became_ready_is_not_a_passed_invariant():
    """THE FALSE GREEN. The rebuild timed out at three of eighteen messages mounted, so the action
    reported `ran = True`, `expect_ok = False`, a null `reopen_ms` and the failed conditions -- and
    left `messages_before` and `messages_after` equal, because both are the total the store
    declared rather than a count of a rebuilt thread. Scored as equality that read as a held
    invariant, the route check passed because the sidebar click had worked, and the pair came out
    MATCH over a rebuild that never happened."""
    base = _reopen_row(18)
    treat = _reopen_row(3, ready = False)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] != P.MATCH, got
    assert got["verdict"] == P.NOT_COMPARABLE, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["reopen_keeps_every_message:treatment"]["ok"] is None, checks
    assert checks["reopen_keeps_every_message:treatment"]["required"] is True
    assert "never reached a ready state" in got["reason"], got["reason"]
    assert "end_present" in got["reason"], got["reason"]
    # The route check still passed, which is exactly what used to carry the pair to a MATCH.
    assert checks["reopen_used_the_control:treatment"]["ok"] is True


def test_a_reopen_that_lost_messages_is_still_broken_when_the_gate_also_refused_it():
    """The other half of the decision, and the reason it is not a blanket refusal. An arm whose
    store came back with six of eighteen messages FAILS the windowed readiness gate too -- its
    `aria-setsize` no longer matches the seeded count -- so a rule that voided every unready reopen
    would have turned the one finding this action exists for into "not comparable"."""
    base = _reopen_row(18)
    treat = _reopen_row(6, after = 6, ready = False)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN, got
    assert "reopen_keeps_every_message:treatment" in got["reason"]


def test_a_finished_rebuild_with_matching_counts_still_holds():
    """The control. A check that never passes is as useless as one that never fails: a thread that
    really did come back, past the same readiness gate that admitted the cell, still counts."""
    got = B.compare_behaviour(_reopen_row(18), _reopen_row(6))
    assert got["verdict"] == P.MATCH, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["reopen_keeps_every_message:treatment"]["ok"] is True


def test_an_old_payload_that_records_no_rebuild_evidence_is_not_a_pass():
    """A row from a checkout that predates the readiness gate carries neither `reopen_readiness`
    nor a meaningful `expect_ok`. Back then `messages_after` was read the moment the store published
    its total, which is the reading the gate was added to refuse, so those rows cannot support this
    invariant either and must not be counted as having held it."""
    base = _reopen_row(18)
    treat = _reopen_row(6)
    for row in (base, treat):
        row["expect"].pop("reopen_readiness")
        row["expect_ok"] = None
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got


def test_a_timed_out_rebuild_leaves_the_behavioural_run_with_no_verdict(tmp_path, capsys):
    """THE CONSEQUENCE, through the command that reports it. This pair was the run's only one, so
    counting it as a held invariant produced `invariants held: 1` and exit 0 over a rebuild that
    never finished. It is now the third outcome: nothing was compared, and that is not a pass.

    EXIT 1 RATHER THAN 2, and the difference is which of two true statements is the more specific.
    The invariant genuinely could not be read -- `invariants held: 0`, one pair NOT COMPARABLE,
    NOTHING WAS COMPARED, all still asserted below. But the pair also carries a DIRECTIONAL
    assertion failure: the action ran on both arms and its own `expect_ok` failed on the treatment
    alone, which is the two builds behaving differently rather than the tool being unable to tell.
    `report` already resolves this precedence for the structural mode and says so where it does it
    -- 'a run can find a real regression while deciding nothing, and "it failed" is the more
    specific answer than "it could not tell"' -- and running `report` over this very payload
    returns 1. The behavioural mode was reading only its invariants, so the same payload came out
    of the two modes with two different verdicts. It no longer does.
    """
    import json

    from studiobench.sweep import ui_parity as U

    rows = []
    for side, row in (("base", _reopen_row(18)), ("treatment", _reopen_row(3, ready = False))):
        row["cell_id"] = f"r100K.{side}.rep0"
        rows.append(row)
    shard = tmp_path / "payload.jsonl"
    shard.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")

    code = U.behaviour_report([shard], "UI PARITY: stalled reopen")
    out = capsys.readouterr().out
    assert "invariants held:            0" in out, out
    assert "NOT COMPARABLE:             1" in out, out
    assert "NOTHING WAS COMPARED" in out, out
    assert "ASSERTION FAILED ON ONE ARM" in out, out
    assert code == 1, out
    # THE TWO MODES AGREE ON THIS PAYLOAD, checked rather than asserted in prose. The structural
    # report is the one that already had this precedence rule written down, so it is the reference.
    assert U.report([shard], "structural on the same payload", frozenset()) == 1


def test_an_action_with_no_declared_invariant_is_unchecked_and_not_a_pass():
    """The scroll extent is a property of the THREAD and holds identically on all eighteen
    actions. Letting it carry an action to a pass would report `model_change` as verified on a
    windowed arm when nothing about `model_change` was looked at."""
    base = _row("model_change", _capture(18, 18))
    treat = _row("model_change", _capture(6, 18))
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.NOT_APPLICABLE
    assert got["verdict"] != P.MATCH
    assert "UNCHECKED" in got["reason"]
    # The scroll extent still held, and is still reported. It just does not vote.
    assert {c["invariant"]: c["ok"] for c in got["checks"]}["scroll_extent"] is True


def test_a_broken_scroll_extent_still_fails_an_action_with_no_invariant_of_its_own():
    """Not voting for a pass is not the same as not voting at all. A scrollbar that lies is a
    defect wherever it is observed."""
    base = _row("model_change", _capture(18, 18))
    treat = _row("model_change", _capture(6, 18))
    treat["census"]["viewport_scroll_height"] = 100
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN


def test_an_action_that_did_not_run_is_not_scored_at_all():
    base = _row("select_text", _capture(18, 18), selected_chars = 10, visible_chars = 10)
    treat = _row("select_text", _capture(6, 18))
    treat["ran"] = False
    treat["reason"] = "no assistant message"
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.NOT_EXERCISED
    assert got["checks"] == []


def test_every_named_first_to_break_action_has_an_invariant():
    """The five the brief names as the ones that break first. An entry silently missing from the
    table would make that action UNCHECKED, which reads as clean."""
    for action in (
        "select_all_copy",
        "select_text",
        "copy_markdown",
        "thread_reopen",
        "scroll_after",
    ):
        assert action in B.INVARIANTS, f"{action} has no behavioural invariant declared"


# ── the four false greens the first review round found ──────────────
#
# Every one of these returned SUCCESS before the fix, which is the only reason they are grouped:
# they are four different routes to a UI verdict of "fine" over a comparison that either found
# nothing or declined to look.


def test_two_full_mounts_of_different_lengths_is_a_difference_not_an_excuse():
    """THE MOST SERIOUS ONE. Neither arm is windowing, so neither is holding anything back on
    purpose, and the treatment renders fewer messages than the base -- a user-visible loss of
    conversation. It used to be waved through as NOT_APPLICABLE on the argument that the
    per-message rows are keyed by position, which is true of the ROWS and says nothing about the
    finding."""
    base = _capture(mounted = 18, total = 18)
    treat = _capture(mounted = 17, total = 17)
    assert P.windowed_mount(base) is False and P.windowed_mount(treat) is False
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert "NEITHER arm is windowing" in got["reason"]
    # And the positional noise is withheld, so the one finding that matters is not buried.
    assert got["moved"] == []


def test_a_windowed_pair_is_still_refused_rather_than_failed():
    """The fix must not turn the intended case red. A genuine window is still NOT_APPLICABLE."""
    got = P.compare(_capture(mounted = 18, total = 18), _capture(mounted = 9, total = 18))
    assert got["verdict"] == P.NOT_APPLICABLE, got
    assert "mounts a WINDOW" in got["reason"]


def test_equal_full_mounts_are_compared_as_before():
    got = P.compare(_capture(mounted = 18, total = 18), _capture(mounted = 18, total = 18))
    assert got["verdict"] == P.MATCH, got


def test_behavioural_scoring_that_validated_nothing_is_not_a_pass(tmp_path, capsys):
    """`broken` empty and `matched` zero used to return 0, under the heading "Every declared
    behavioural invariant held on both arms" -- a sentence that is true of an empty set and reads
    as a pass. On a windowed arm this is the ONLY UI verdict there is, so a silent no-op there
    leaves the arm with no verdict at all while appearing to have one."""
    from studiobench.sweep import ui_parity as U

    shard = tmp_path / "payload.jsonl"
    # Both arms recorded the action and NEITHER ran it: nothing to compare, nothing broken.
    import json

    rows = []
    for side in ("base", "treatment"):
        row = _row("thread_reopen", _capture(mounted = 9, total = 18))
        row["ran"] = False
        row["cell_id"] = f"r100K.{side}.rep0"
        rows.append(row)
    shard.write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")

    code = U.behaviour_report([shard], "UI PARITY: nothing")
    out = capsys.readouterr().out
    assert code == 2, out
    assert "NOTHING WAS COMPARED" in out


def _scroll_extent_broken_pair(cell_suffix = "rep0"):
    """A `select_text` pair whose declared invariant is BROKEN: the windowed arm's scrollbar says
    the thread is a third of its real length. `compare_behaviour` returns BROKEN, so this pair
    lands in `broken` and NOT in `matched`."""
    rows = []
    for side, mounted, scroll in (("base", 18, 10_000), ("treatment", 6, 3_300)):
        row = _row(
            "select_text",
            _capture(mounted, 18),
            selected_chars = 100,
            visible_chars = 100,
        )
        row["census"]["viewport_scroll_height"] = scroll
        row["cell_id"] = f"r100K.{side}.{cell_suffix}"
        rows.append(row)
    return rows


def test_a_run_whose_every_invariant_BROKE_does_not_also_claim_it_compared_nothing(
    tmp_path, capsys
):
    """THE SELF-CONTRADICTING ARTIFACT. `matched` and `broken` are independent counters, so a
    payload whose only pair BROKE its invariant satisfies `matched == 0` as well.

    The no-verdict banner was gated on `matched == 0` alone, so this run printed its broken
    invariant, then printed that not one behavioural invariant was evaluated and that it carried
    "neither a pass nor a failure" -- and then returned 1. Every clause of that banner is false
    here, and its instruction ("find out why the actions did not run") points away from the
    regression listed three lines above it.

    The exit code was right and stays right; asserted below, because a banner fix that quietened
    the failure would be far worse than the banner.
    """
    from studiobench.sweep import ui_parity as U

    shard = _write(tmp_path, "all_broken", _scroll_extent_broken_pair())
    code = U.behaviour_report([shard], "UI PARITY: all broken")
    out = capsys.readouterr().out

    assert code == 1, out
    assert "invariants held:            0" in out, out
    assert "INVARIANTS BROKEN:          1" in out, out
    assert "scroll_extent" in out, out
    assert "NOTHING WAS COMPARED" not in out, out
    assert "neither a pass nor a failure" not in out, out


def test_the_no_verdict_banner_still_fires_when_a_build_DIFFERENCE_is_the_only_failure(
    tmp_path, capsys
):
    """THE OTHER HALF OF THE GATE, pinned so the fix above cannot be widened into a regression.

    `build_differences` collects two shapes -- one-arm execution and a directional assertion
    failure -- from pairs whose INVARIANTS were unreadable. When one of those is the only reason a
    run fails, "not one behavioural invariant was evaluated" is literally true, and the banner
    belongs there alongside exit 1. Gating it on `build_failed` as well as on `broken` would
    silence a true statement.
    """
    from studiobench.sweep import ui_parity as U

    rows = []
    for side, row in (("base", _reopen_row(18)), ("treatment", _reopen_row(3, ready = False))):
        row["cell_id"] = f"r100K.{side}.rep0"
        rows.append(row)
    shard = _write(tmp_path, "build_diff_only", rows)

    code = U.behaviour_report([shard], "UI PARITY: build difference only")
    out = capsys.readouterr().out
    assert code == 1, out
    assert "INVARIANTS BROKEN:          0" in out, out
    assert "NOTHING WAS COMPARED" in out, out


def test_the_observation_cost_is_not_charged_to_the_action_budget():
    """`over_ms` used to be sampled AFTER the census and the multi-megabyte digest that this
    change had just moved outside the measured window, so an action that finished inside its
    budget was flagged `over_budget` for instrument time. The deadline is now read at the moment
    the window closes."""
    import inspect

    from studiobench.scene import schedule as S

    src = inspect.getsource(S.SceneRunner)
    close = src.index("window_closed_at = time.monotonic()")
    over = src.index("over_ms = ((window_closed_at - t0)", close)
    # The FIRST census taken after the deadline is read -- `_census` is called from more than one
    # place, so an unanchored search finds a gap window's copy and compares unrelated lines.
    census = src.index("census = self._census()", over)
    assert close < over < census, (
        "the deadline is sampled after the observations again, which charges instrument time to "
        "the action's budget"
    )


# ── a scan of nothing is not agreement ──────────────────────────────


def _styled(elements: int, digest: str = "s") -> dict:
    cap = _capture(mounted = 18, total = 18)
    cap["styles"] = {"elements": elements, "digest": digest, "capped": False}
    return cap


def test_a_style_probe_that_matched_no_elements_does_not_report_a_match():
    """THE POSITIVE CONTROL. Two probes that matched nothing have equal counts and equal digests
    -- both the hash of an empty string -- so the strongest verdict the function can return was
    being issued on the strength of no observation at all.

    Not hypothetical: the probe walks a hand-written selector list written against Unsloth's
    markup, and a class rename anywhere in it empties the scan silently.
    """
    verdict, reason = P.compare_styles(_styled(0), _styled(0))
    assert verdict == P.NOT_COMPARABLE, (verdict, reason)
    assert "matched no elements" in reason


def test_one_arm_scanning_nothing_is_also_not_a_difference_to_report():
    """It is a broken probe, not a finding about the build, and saying DIFFER here would send
    somebody looking for a UI change that nobody has evidence for."""
    verdict, _reason = P.compare_styles(_styled(0), _styled(12))
    assert verdict == P.NOT_COMPARABLE


def test_a_probe_that_actually_looked_still_matches_and_still_differs():
    """The control must not swallow the readings it exists to protect."""
    assert P.compare_styles(_styled(12, "a"), _styled(12, "a"))[0] == P.MATCH
    assert P.compare_styles(_styled(12, "a"), _styled(12, "b"))[0] == P.DIFFER


def test_the_passing_digest_verdict_states_what_it_did_not_look_at():
    """A PARITY OK line that reads as "the UI is unchanged" is a claim the instrument cannot
    support: run against a real sidebar-drag change the thread digest returned 0 of 34, and so did
    its null. The limitation is printed next to the verdict, not left in a source comment."""
    import inspect

    from studiobench.sweep import ui_parity as U

    src = inspect.getsource(U.report)
    assert "THREAD STRUCTURE" in src
    assert "sidebar-blind" in src and "layout-blind" in src


# ── the clipboard is scored against the thread, in both directions ──


def test_a_truncated_clipboard_still_fails():
    """The defect the invariant was written for. The windowed arm copies only what it mounted, so
    the clipboard is the visible fraction of the conversation and the rest is gone."""
    base = _copy_row(_capture(18, 18), clipboard = 200_000, selected = 200_000, mounted = 18)
    treat = _copy_row(_capture(6, 18), clipboard = 122_000, selected = 66_000, mounted = 6)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_carries_the_whole_thread:treatment"]["ok"] is False


def test_a_clipboard_that_carries_far_MORE_than_the_thread_also_fails():
    """THE DEFECT A FIX FOR THE FIRST ONE TURNS INTO. Serialising from the message store is the
    right repair, and the obvious serialiser is the "save this reply" one, which emits reasoning,
    tool-call arguments and tool results -- none of which a user can select, because the panes
    holding them are collapsed and a collapsed Radix Collapsible is not in the DOM at all.

    Measured on a real 100K arm: 420,911 characters against a 194,992-character thread, 2.16x.
    The truncation was fixed and the content was then wrong in the other direction. A check that
    only had a lower bound called that a pass.
    """
    base = _copy_row(_capture(18, 18), clipboard = 193_937, selected = 194_992, mounted = 18)
    treat = _copy_row(_capture(9, 18), clipboard = 420_911, selected = 118_089, mounted = 9)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_carries_the_whole_thread:treatment"]["ok"] is False
    assert "2.159" in checks["clipboard_carries_the_whole_thread:treatment"]["detail"]


def test_markdown_source_against_rendered_text_is_not_treated_as_a_difference():
    """The reason the two arms are NOT compared against each other. The base arm's clipboard is the
    DOM's rendered text and a store-based copy is markdown source, so fences, emphasis and LaTeX
    delimiters exist in one and not the other. A narrowed store serialiser measured about 1% over
    on a scale fixture. Comparing the two clipboards at a 2% tolerance fails a correct fix, and the
    only way to make it pass is to widen the tolerance until it tests nothing."""
    base = _copy_row(_capture(18, 18), clipboard = 193_937, selected = 194_992, mounted = 18)
    treat = _copy_row(_capture(9, 18), clipboard = 197_800, selected = 118_089, mounted = 9)
    got = B.compare_behaviour(base, treat)
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_carries_the_whole_thread:treatment"]["ok"] is True, checks


def test_without_a_fully_mounted_arm_there_is_no_reference_and_no_verdict():
    """The reference is the thread's visible text as measured by an arm that has all of it. If
    neither arm mounts everything, nobody in this payload knows how long the conversation is, and
    that is reported rather than guessed."""
    base = _copy_row(_capture(9, 18), clipboard = 190_000, selected = 118_000, mounted = 9)
    treat = _copy_row(_capture(9, 18), clipboard = 190_000, selected = 118_000, mounted = 9)
    got = B.compare_behaviour(base, treat)
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_carries_the_whole_thread"]["ok"] is None
    assert got["verdict"] == P.NOT_COMPARABLE, got


# ── visible-region parity needs a measured floor like everything else ──


def _visible_shard(
    tmp_path,
    name,
    differ_actions,
    actions = ("a", "b", "c"),
    rung = "r100K",
    reps = 2,
):
    """A payload shard whose visible-region captures differ on `differ_actions` and match elsewhere.

    TWO REPS BY DEFAULT, and the rung is a parameter. Both are what the floor is keyed and counted
    by: an entry is derived from repeated readings at one rung, so a shard with a single rep is a
    shard that cannot produce a floor at all, and one recorded at a different rung produces a floor
    that does not apply here.
    """
    import json

    rows = []
    for action in actions:
        for rep in range(reps):
            for side in ("base", "treatment"):
                digest = "X" if (action in differ_actions and side == "treatment") else "same"
                rows.append(
                    {
                        "row_type": "action",
                        "action": action,
                        "ran": True,
                        "cell_id": f"{rung}.{side}.rep{rep}",
                        "parity": _capture(mounted = 18, total = 18),
                        "visible": {
                            "visible_attempted": True,
                            "ever_visible": [1],
                            "ever_visible_count": 1,
                            "mounted_ever_visible": 1,
                            "unmounted_at_capture": 0,
                            "messages": {"1": {"role": "assistant", "digest": digest, "chars": 10}},
                        },
                    }
                )
    shard = tmp_path / name
    shard.mkdir()
    (shard / "payload.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
    return shard / "payload.jsonl"


def test_an_action_that_differs_against_an_identical_build_is_not_counted_against_the_arm(
    tmp_path, capsys
):
    """THE FLOOR, AND WHY IT IS NOT OPTIONAL.

    Measured on a real 100K run: the base-vs-base null control differed inside the viewport on 13
    of 64 action pairs, while the virtualization arm it was the control for differed on 5. Scored
    without the floor the arm ranks WORSE than two copies of the same build, so the verdict is not
    merely weak, it is backwards. The rows differ at identical character counts, which is a
    volatile attribute rather than changed content.
    """
    from studiobench.sweep import ui_parity as U

    null = _visible_shard(tmp_path, "null", differ_actions = {"a", "b"})
    arm = _visible_shard(tmp_path, "arm", differ_actions = {"a"})

    unstable = U.visible_unstable_set([null])
    assert unstable == frozenset({("r100K", "a"), ("r100K", "b")}), unstable

    # Unfloored: the arm's one difference is counted and the run fails.
    assert U.visible_report([arm], "unfloored") == 1
    # Floored by its own null: that action is known to differ against itself at this rung, so it
    # is reported and not counted.
    assert U.visible_report([arm], "floored", unstable) == 0
    out = capsys.readouterr().out
    assert "differ against an identical build" in out


def test_a_real_visible_difference_outside_the_floor_still_fails(tmp_path):
    """The floor must not become a way of passing anything. An action the null never saw differ is
    still a failure."""
    from studiobench.sweep import ui_parity as U

    null = _visible_shard(tmp_path, "null2", differ_actions = {"b"})
    arm = _visible_shard(tmp_path, "arm2", differ_actions = {"a"})
    assert U.visible_report([arm], "floored", U.visible_unstable_set([null])) == 1


def test_noise_at_one_rung_does_not_silence_a_regression_at_another(tmp_path, capsys):
    """THE FLOOR APPLIED WHERE IT WAS NEVER MEASURED.

    A payload holds several rungs -- the windowed readiness gate is written to permit an arm to
    mount everything at 1K and a window at 100K -- and the noise floor was a set of ACTION NAMES,
    so one differing null-control pair marked that action unstable everywhere. Here the null's
    100K `model_change` differs for its own reasons and the arm's 1K `model_change` differs
    reproducibly. Keyed by name alone the second is filed as noise, the other pairs supply
    `matched > 0`, and the command exits 0 having silenced the one real finding in the run.

    The 1K rung is a different thread at a different size with the film's slots landing somewhere
    else entirely; the null measured nothing there and the floor may not speak for it.
    """
    from studiobench.sweep import ui_parity as U

    both = ("model_change", "keystroke")
    null = [
        _visible_shard(
            tmp_path, "null_rungs", differ_actions = {"model_change"}, actions = both, rung = "r100K"
        )
    ]
    big = _visible_shard(
        tmp_path, "arm_100k", differ_actions = {"model_change"}, actions = both, rung = "r100K"
    )
    small = _visible_shard(
        tmp_path, "arm_1k", differ_actions = {"model_change"}, actions = both, rung = "r1K"
    )
    unstable = U.visible_unstable_set(null)
    assert unstable == frozenset({("r100K", "model_change")}), unstable
    assert U.visible_report([big, small], "mixed rungs", unstable) == 1
    out = capsys.readouterr().out
    assert "DIFFERENCES INSIDE THE VIEWPORT" in out
    # And it is the 1K pair that is counted, with the 100K one still reported as floored noise.
    assert "r1K rep0" in out
    assert "differ against an identical build" in out


def test_a_floor_derived_from_a_single_pair_is_not_a_floor(tmp_path):
    """ONE OCCURRENCE IS NOT EVIDENCE, which is the guard `derive_unstable` already applies to the
    structural set and this one had none of. A single flake in a null control would otherwise
    silence that action, at that rung, for every rep of every payload scored against it -- and the
    visible floor carries no declared mechanism behind it to justify the entry."""
    from studiobench.sweep import ui_parity as U

    thin = _visible_shard(tmp_path, "null_thin", differ_actions = {"a"}, reps = 1)
    assert U.visible_unstable_set([thin]) == frozenset()
    thick = _visible_shard(tmp_path, "null_thick", differ_actions = {"a"}, reps = 2)
    assert U.visible_unstable_set([thick]) == frozenset({("r100K", "a")})


def test_an_unfloored_visible_run_says_so(tmp_path, capsys):
    from studiobench.sweep import ui_parity as U

    arm = _visible_shard(tmp_path, "arm3", differ_actions = set())
    U.visible_report([arm], "no floor")
    assert "NO FLOOR WAS MEASURED" in capsys.readouterr().out


def test_the_noise_floor_cannot_silence_an_arm_that_lost_the_thread(tmp_path, capsys):
    """MEASURED. On the 100K run `model_change` is in the derived unstable set because the null
    control's copy of it differs on a volatile attribute at identical character counts -- AND it is
    the action on which the virtualization arm's thread went from 12 mounted messages to 0 and
    never came back. Routing the second finding into the floor because of the first suppresses a
    lost conversation on the strength of unrelated jitter."""
    import json

    from studiobench.sweep import ui_parity as U

    def _shard(name, treat_empty):
        rows = []
        for side in ("base", "treatment"):
            empty = treat_empty and side == "treatment"
            rows.append(
                {
                    "row_type": "action",
                    "action": "model_change",
                    "ran": True,
                    "cell_id": f"r100K.{side}.rep0",
                    "parity": _capture(mounted = 18, total = 18),
                    "visible": {
                        "visible_attempted": True,
                        "ever_visible": [14, 15],
                        "ever_visible_count": 2,
                        "mounted_ever_visible": 0 if empty else 2,
                        "unmounted_at_capture": 2 if empty else 0,
                        "messages": {}
                        if empty
                        else {
                            str(o): {"role": "assistant", "digest": "same", "chars": 10}
                            for o in (14, 15)
                        },
                    },
                }
            )
        shard = tmp_path / name
        shard.mkdir()
        (shard / "payload.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding = "utf-8"
        )
        return shard / "payload.jsonl"

    # The null differs on this action for its own reasons, so it lands in the unstable set...
    null = _visible_shard(
        tmp_path, "null_mc", differ_actions = {"model_change"}, actions = ("model_change",)
    )
    unstable = U.visible_unstable_set([null])
    assert ("r100K", "model_change") in unstable
    # ...and the arm that lost the thread on it is STILL a failure.
    assert U.visible_report([_shard("arm_mc", True)], "severe", unstable) == 1
    assert "one arm lost the thread" in capsys.readouterr().out


# ── the residue of a windowed capture is printed, not swallowed ─────


def _write(tmp_path, name, rows):
    import json

    shard = tmp_path / name
    shard.mkdir()
    (shard / "payload.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
    return shard / "payload.jsonl"


def _visible(
    ever,
    digested,
    digest = "same",
):
    """A visible-region capture that SAW `ever` and could still digest `digested` at capture time."""
    return {
        "visible_attempted": True,
        "ever_visible": sorted(ever),
        "ever_visible_count": len(ever),
        "mounted_ever_visible": len(digested),
        "unmounted_at_capture": len(ever) - len(digested),
        "messages": {
            str(o): {"role": "assistant", "digest": digest, "chars": 10} for o in sorted(digested)
        },
    }


def _action(
    action,
    cell_id,
    *,
    parity = None,
    visible = None,
    ran = True,
    reason = None,
    expect = None,
):
    row = {
        "row_type": "action",
        "action": action,
        "ran": ran,
        "cell_id": cell_id,
        "parity": parity if parity is not None else _capture(mounted = 18, total = 18),
        "census": {"viewport_scroll_height": 10_000},
        "expect": dict(expect or {}),
        "expect_ok": True,
        "timings": {},
    }
    if visible is not None:
        row["visible"] = visible
    if reason is not None:
        row["reason"] = reason
    return row


def test_a_style_regression_survives_a_structural_refusal(tmp_path, capsys):
    """THE ONLY THING HERE THAT SEES `display`, `visibility` OR `pointer-events`, dropped.

    The style probe is an INDEPENDENT reading: it is computed styles, not the DOM digest, so a pair
    refused for landing at two points in one live reply can still carry a real CSS regression on a
    settled surface. The refusal bucket sat above it and `continue`d, so that verdict was computed,
    attached to the row and then thrown away -- for exactly the pairs a streamed-parity refusal
    creates most of.

    The refusal itself is right and stays. What must not happen is losing the other reading with it.
    """
    from studiobench.sweep import ui_parity as U

    base = _capture(mounted = 4, total = 4)
    treat = _capture(mounted = 4, total = 4)
    # Structurally unreadable: the stream could not be placed on one arm.
    treat["in_flight_unplaced"] = True
    treat["in_flight"] = []
    base["streaming"] = treat["streaming"] = True
    # ...while the style probe read both arms cleanly and they disagree.
    treat["styles"] = {"elements": 4, "digest": "RESTYLED", "capped": False}

    rows = [_row("select_text", base), _row("select_text", treat)]
    rows[0]["cell_id"], rows[1]["cell_id"] = "r100K.base.rep0", "r100K.treatment.rep0"
    _write(tmp_path, "stylerefuse", rows)

    U.main([str(tmp_path / "stylerefuse"), "--mode", "structural", "--min-compared", "0"])
    out = capsys.readouterr().out
    assert "style probe differing:      1" in out, out


def test_a_visible_message_that_could_not_be_digested_is_printed_and_not_counted(tmp_path, capsys):
    """THE FALSE GREEN. `windowed` put ordinals 1 and 2 on screen during the action and had
    unmounted 2 again by the time the capture ran, so only ordinal 1 was ever compared. That pair
    used to return MATCH -- one mounted message is enough to carry it -- and `visible_report`
    never read `not_digested`, so a rendering difference in ordinal 2 exited 0 under a claim that
    every visible message was identical on both arms.

    It is now the third outcome, and the residue is on screen where a reader can see which message
    went uncompared."""
    from studiobench.sweep import ui_parity as U

    rows = []
    for side in ("base", "treatment"):
        rows.append(_action("settings", f"r100K.{side}.rep0", visible = _visible([1], [1])))
        rows.append(_action("scroll_after", f"r100K.{side}.rep0", visible = _visible([1, 2], [1])))
    shard = _write(tmp_path, "residue", rows)

    code = U.visible_report([shard], "residue")
    out = capsys.readouterr().out
    assert "VISIBLE BUT NOT DIGESTED" in out, out
    assert "ordinals [2]" in out, out
    # The fully digested pair still passes, so the refusal is scoped to the pair that earned it.
    assert "visible region matched:     1" in out, out
    assert "visible but NOT DIGESTED:   1" in out, out
    assert code == 0, out


def test_a_run_where_nothing_could_be_digested_carries_no_visible_verdict(tmp_path, capsys):
    """Every pair carrying a residue means every pair was refused, and a mode that refused
    everything has no verdict to report. 2, not 0."""
    from studiobench.sweep import ui_parity as U

    rows = [
        _action("scroll_after", f"r100K.{side}.rep0", visible = _visible([1, 2], [1]))
        for side in ("base", "treatment")
    ]
    code = U.visible_report([_write(tmp_path, "all_residue", rows)], "all residue")
    out = capsys.readouterr().out
    assert code == 2, out
    assert "NOTHING WAS COMPARED" in out, out


# ── an unmeasured windowed run cannot come out green ────────────────


def _failed_parity(why = "the parity probe timed out"):
    return {"parity_attempted": False, "reason": why}


def _declared_windowed_shard(
    tmp_path,
    name,
    *,
    arm = "treatment",
    parity = None,
):
    """A payload that DECLARES a windowed arm the way `__main__.py` records it, and measures nothing.

    Both records are written because both are read: the gate row `windowed_readiness:{arm}` that
    `--windowed-arm` produces, and `readiness.mode` on the cell row.
    """
    rows = [
        {
            "row_type": "gate",
            "name": f"windowed_readiness:{arm}",
            "passed": True,
            "detail": {"arm": arm, "reason": "declared on the command line with --windowed-arm"},
        }
    ]
    for side in ("base", "treatment"):
        rows.append(
            {
                "row_type": "cell",
                "cell_id": f"r100K.{side}.rep0",
                "readiness": {
                    "ready": True,
                    "mode": "windowed" if side == arm else "full",
                    "expected_messages": 18,
                },
            }
        )
        rows.append(
            _action(
                "select_all_copy",
                f"r100K.{side}.rep0",
                parity = parity if parity is not None else _failed_parity(),
                ran = False,
                reason = "the slot was missed",
            )
        )
    return _write(tmp_path, name, rows)


def test_a_declared_windowed_run_that_measured_nothing_is_still_windowed(tmp_path):
    """THE DETECTION HOLE. `any_windowed` scanned successful captures for
    `thread_total > mounted_messages`, so a run whose slots were all missed or whose parity probes
    all failed scanned exactly like a run that mounted its whole thread: no window found, `--mode
    auto` picks the digest, every pair comes out NOT_EXERCISED or NOT_COMPARABLE and the report
    exits 0. An entirely unmeasured windowed run produced a green structural result.

    The payload records the declaration independently of the measurement, so it decides when the
    measurement cannot."""
    from studiobench.sweep import ui_parity as U

    shard = _declared_windowed_shard(tmp_path, "declared")
    why = U.any_windowed([shard])
    assert why is not None, "a declared windowed run with no capture read as fully mounted"
    assert "DECLARED, not measured" in why, why
    assert all(mode == U.WINDOWED for mode, _why in U.decide_modes([shard]).values())


def test_an_unmeasured_windowed_run_does_not_exit_zero(tmp_path, capsys):
    """The consequence, end to end through `main`. Nothing was measured, so no mode has a verdict
    and the run must say so rather than report a structural pass."""
    from studiobench.sweep import ui_parity as U

    _declared_windowed_shard(tmp_path, "declared_main")
    code = U.main([str(tmp_path / "declared_main")])
    out = capsys.readouterr().out
    assert code != 0, out
    assert code == 2, out
    assert "NOTHING WAS COMPARED" in out, out


def test_a_payload_whose_captures_all_failed_is_not_a_structural_pass(tmp_path, capsys):
    """The same hole with no declaration to fall back on, at the report that would print the pass.
    Every pair is NOT COMPARABLE, so `stable_bad` is empty and `matched` is zero -- and the exit
    guard only refused this when the pairs were NOT APPLICABLE windowed mounts, so a payload whose
    parity probes all failed returned 0 under "No stable action rendered a different THREAD
    STRUCTURE"."""
    from studiobench.sweep import ui_parity as U

    rows = [
        _action("settings", f"r100K.{side}.rep0", parity = _failed_parity())
        for side in ("base", "treatment")
    ]
    code = U.report([_write(tmp_path, "all_failed", rows)], "all failed", frozenset())
    out = capsys.readouterr().out
    assert code == 2, out
    assert "NOTHING WAS COMPARED" in out, out
    assert "No stable action rendered a different THREAD STRUCTURE" not in out, out


# ── the mode is decided per action pair, not per payload ────────────


def _copy_expect(*, clipboard, selected, mounted):
    """The `select_all_copy` observations its behavioural invariant is scored on."""
    return {
        "selected_chars": selected,
        "clipboard_chars": clipboard,
        "clipboard_readable": True,
        "clipboard_note": None,
        "messages_total": 18,
        "messages_mounted": mounted,
        "mounted_fraction": round(mounted / 18, 3),
    }


def _mixed_rung_shard(tmp_path, name):
    """One payload holding a fully mounted 1K rung and a windowed 100K rung, which is exactly what
    the windowed readiness gate permits an arm to do, plus a DOM regression at the 1K rung.

    The 100K rung is a CLEAN windowed pair: it holds its behavioural invariants (the clipboard
    still carries the whole thread) and its visible region matches, so the only thing left that can
    fail this payload is the digest at the rung where both arms mount everything.
    """
    rows = [
        {
            "row_type": "gate",
            "name": "windowed_readiness:treatment",
            "passed": True,
            "detail": {"arm": "treatment"},
        }
    ]
    for side in ("base", "treatment"):
        digest = "regressed" if side == "treatment" else "shipped"
        rows.append(
            _action(
                "select_all_copy",
                f"r1K.{side}.rep0",
                parity = _capture(mounted = 18, total = 18, digest = digest),
                visible = _visible([1], [1], digest = digest),
                expect = _copy_expect(clipboard = 200_000, selected = 200_000, mounted = 18),
            )
        )
        windowed = side == "treatment"
        rows.append(
            _action(
                "select_all_copy",
                f"r100K.{side}.rep0",
                parity = _capture(mounted = 9 if windowed else 18, total = 18, digest = "shipped"),
                visible = _visible([1], [1]),
                expect = _copy_expect(
                    clipboard = 200_000,
                    selected = 66_000 if windowed else 200_000,
                    mounted = 9 if windowed else 18,
                ),
            )
        )
    return _write(tmp_path, name, rows)


def test_two_rungs_of_one_payload_are_two_pairs_and_not_one(tmp_path):
    """`make_cell_id` writes `r{rung}.{arm}.rep{n}` and the pair key took only the last segment, so
    `r1K.base.rep0` and `r100K.base.rep0` were the same key: one rung's rows overwrote the other's
    and half the payload disappeared before anything was compared. The per-pair mode decision needs
    both rungs to exist before it can score them differently."""
    from studiobench.sweep import ui_parity as U

    shard = _mixed_rung_shard(tmp_path, "mixed_keys")
    pairs = U.collect([shard])["pairs"]
    assert len(pairs) == 2, sorted(pairs)
    cells = {f"{rung} {rep}" for _shard, rung, rep, _sid, _action in pairs}
    assert cells == {"r1K rep0", "r100K rep0"}, cells
    assert all(set(sides) == {"base", "treatment"} for sides in pairs.values())


def test_the_windowed_large_rung_does_not_suppress_the_digest_on_the_mounted_small_one(
    tmp_path, capsys
):
    """THE FINDING. Deciding the mode per PAYLOAD meant one windowed 100K capture put every pair in
    that payload on the behavioural and visible-region scales -- including the 1K rung, where both
    arms mount their whole thread and the structural digest is exactly the right question. An
    ordinary DOM regression at that rung was then never looked for.

    Here the treatment's 1K digest differs and its 100K rung is a genuine window. The regression has
    to be found, by name, and the windowed rung has to stay out of the structural section."""
    from studiobench.sweep import ui_parity as U

    shard = _mixed_rung_shard(tmp_path, "mixed")
    modes = {
        f"{rung} {rep}": mode
        for (_s, rung, rep, _sid, _a), (mode, _why) in U.decide_modes([shard]).items()
    }
    assert modes == {"r1K rep0": U.STRUCTURAL, "r100K rep0": U.WINDOWED}, modes

    code = U.main([str(tmp_path / "mixed")])
    out = capsys.readouterr().out
    assert "MODE DECIDED PER ACTION PAIR: 1 of 2" in out, out
    assert "UI PARITY DIFFERENCES ON STABLE ACTIONS" in out, out
    assert "r1K rep0" in out.split("UI PARITY DIFFERENCES ON STABLE ACTIONS")[1], out
    # (1 fully mounted pair(s) of 2) in the heading: a structural section that silently covered
    # part of a payload would read as a verdict on all of it.
    assert "1 fully mounted pair(s) of 2" in out, out
    assert code == 1, out


def test_the_exit_status_combines_every_mode_that_ran(tmp_path, capsys):
    """Any mode's failure fails the run. The windowed rung passes its own two modes here and the
    fully mounted rung fails the digest, so the combined status is the failure."""
    from studiobench.sweep import ui_parity as U

    _mixed_rung_shard(tmp_path, "combined")
    code = U.main([str(tmp_path / "combined")])
    out = capsys.readouterr().out
    assert "COMBINED EXIT STATUS 1" in out, out
    assert code == 1


def test_an_arm_declared_windowed_is_still_digested_where_it_mounted_everything(tmp_path):
    """The declaration is a FALLBACK and never an override. `--windowed-arm treatment` is a
    statement about the arm, not about every rung it ran: at 1K it mounts the whole thread, the
    capture proves it, and that pair is owed a structural digest like any other."""
    from studiobench.sweep import ui_parity as U

    rows = [
        {
            "row_type": "gate",
            "name": "windowed_readiness:treatment",
            "passed": True,
            "detail": {"arm": "treatment"},
        }
    ]
    for side in ("base", "treatment"):
        rows.append(_action("settings", f"r1K.{side}.rep0", parity = _capture(mounted = 18, total = 18)))
    shard = _write(tmp_path, "declared_but_mounted", rows)
    assert all(mode == U.STRUCTURAL for mode, _why in U.decide_modes([shard]).values())
    assert U.any_windowed([shard]) is None


# ── the declared arm that never produced a row at all ───────────────


def _one_sided_shard(
    tmp_path,
    name,
    *,
    gate = True,
    cell_row = False,
):
    """A mixed-rung payload whose declared-windowed TREATMENT arm died at the large rung.

    The 1K rung is a clean, fully mounted pair that both arms recorded, so the structural report
    has something to pass on. At 100K only the base arm ever emitted an action row: `sides` holds
    one side, and the only thing left that can say the missing arm mounts a window is the run's own
    declaration -- the `--windowed-arm` gate row, or the cell row's `readiness.mode`.
    """
    rows = []
    if gate:
        rows.append(
            {
                "row_type": "gate",
                "name": "windowed_readiness:treatment",
                "passed": True,
                "detail": {"arm": "treatment"},
            }
        )
    if cell_row:
        rows.append(
            {
                "row_type": "cell",
                "cell_id": "r100K.treatment.rep0",
                "completed": False,
                "readiness": {"ready": True, "mode": "windowed", "expected_messages": 18},
            }
        )
    for side in ("base", "treatment"):
        rows.append(
            _action(
                "select_all_copy",
                f"r1K.{side}.rep0",
                parity = _capture(mounted = 18, total = 18),
                visible = _visible([1], [1]),
                expect = _copy_expect(clipboard = 200_000, selected = 200_000, mounted = 18),
            )
        )
    rows.append(
        _action(
            "select_all_copy",
            "r100K.base.rep0",
            parity = _capture(mounted = 18, total = 18),
            visible = _visible([1], [1]),
            expect = _copy_expect(clipboard = 200_000, selected = 200_000, mounted = 18),
        )
    )
    return _write(tmp_path, name, rows)


def test_a_declared_windowed_arm_with_no_row_is_not_scored_structurally(tmp_path):
    """THE ONE-SIDED HOLE. The declaration fallback was read off the rows the pair HAS, so an arm
    that failed before emitting an action row was never asked about: the loop saw the base row,
    found no declaration for the base arm, and classified the pair structural."""
    from studiobench.sweep import ui_parity as U

    shard = _one_sided_shard(tmp_path, "one_sided")
    modes = {
        f"{rung} {rep}": (mode, why)
        for (_s, rung, rep, _sid, _a), (mode, why) in U.decide_modes([shard]).items()
    }
    assert modes["r100K rep0"][0] == U.WINDOWED, modes
    assert "DECLARED, not measured" in modes["r100K rep0"][1], modes
    # And the rung that really did mount everything on both arms is still owed its digest.
    assert modes["r1K rep0"][0] == U.STRUCTURAL, modes


def test_a_windowed_cell_row_declares_the_arm_even_when_that_arm_has_no_action_row(tmp_path):
    """The other declaration, and the one that needs the missing arm's cell id to be derived at
    all: a run without `--windowed-arm` whose treatment cell was admitted by the WINDOWED readiness
    gate records that on the cell row, under a cell id no surviving row carries."""
    from studiobench.sweep import ui_parity as U

    shard = _one_sided_shard(tmp_path, "one_sided_cell", gate = False, cell_row = True)
    modes = {
        f"{rung} {rep}": mode
        for (_s, rung, rep, _sid, _a), (mode, _why) in U.decide_modes([shard]).items()
    }
    assert modes["r100K rep0"] == U.WINDOWED, modes
    assert modes["r1K rep0"] == U.STRUCTURAL, modes


def test_a_missing_windowed_arm_does_not_exit_zero_on_the_strength_of_the_other_rung(
    tmp_path, capsys
):
    """THE CONSEQUENCE. The fully mounted 1K pair supplies `matched > 0`, the 100K pair is filed as
    structurally NOT COMPARABLE, and the command exits 0 -- having never run a windowed report for
    the rung whose treatment arm produced nothing at all."""
    from studiobench.sweep import ui_parity as U

    _one_sided_shard(tmp_path, "one_sided_main")
    code = U.main([str(tmp_path / "one_sided_main")])
    out = capsys.readouterr().out
    assert "windowed:   " in out and "r100K rep0" in out, out
    assert "NOTHING WAS COMPARED" in out, out
    assert code == 2, out


def test_a_pair_missing_an_arm_with_no_declaration_anywhere_is_still_structural(tmp_path):
    """The fallback must not become an override in the other direction: with nothing declaring a
    window, a pair one arm failed to record is the structural report's problem, and it refuses it
    there. Otherwise every crashed cell in an ordinary A/B would be routed to a mode that cannot
    say anything about it either."""
    from studiobench.sweep import ui_parity as U

    shard = _one_sided_shard(tmp_path, "one_sided_undeclared", gate = False)
    modes = {
        f"{rung} {rep}": mode
        for (_s, rung, rep, _sid, _a), (mode, _why) in U.decide_modes([shard]).items()
    }
    assert modes == {"r1K rep0": U.STRUCTURAL, "r100K rep0": U.STRUCTURAL}, modes


def test_a_capture_that_saw_no_thread_at_all_falls_back_on_the_declaration(tmp_path):
    """MEASURED on the real 100K film: after `model_change` the treatment arm's captures read 0 of
    0 messages for the rest of the film. `windowed_mount` answers False for 0 of 0 -- nothing is
    missing from a thread of nothing -- so that capture would score a declared windowed arm
    structurally on the strength of a probe that observed no messages."""
    from studiobench.sweep import ui_parity as U

    lost = {
        "parity_attempted": True,
        "root_kind": "thread",
        "digest": "d",
        "chars": 0,
        "messages": [],
        "overlays": [],
        "styles": {"elements": 0, "digest": "s", "capped": False},
        "mounted_messages": 0,
        "thread_total": 0,
    }
    shard = _declared_windowed_shard(tmp_path, "lost_thread", parity = lost)
    assert all(mode == U.WINDOWED for mode, _why in U.decide_modes([shard]).values())


# ── a cell that failed its own completeness gate carries no UI verdict ──
#
# `probe_thread_completeness` runs before the film and `record_completeness_gate` writes the
# verdict against the cell, so a windowed arm that kept its first page and its last one and lost
# everything between them says so in its own payload. `report/payload.py::excluded_from_rows`
# drops that cell from the PERFORMANCE score. `ui_parity.py` read no gate row except the windowed
# declaration, so the same cell's eighteen action rows were still scored for UI parity -- and the
# visible region is a window on the END of the thread, which such a store still fills, so the
# pairs matched and `--mode auto` exited 0 over a payload that had already recorded the loss.


def _completeness_gate(
    cell_id,
    passed,
    reason = "the head of the thread mounted, but 12 of 18 ordinals never mounted",
):
    return {
        "row_type": "gate",
        "name": "thread_complete",
        "passed": passed,
        "cell_id": cell_id,
        "detail": {"probe_attempted": True, "head_reached": True, "reason": reason},
    }


def _matching_pair(cell_suffix = "rep0", ordinals = (17, 18)):
    """One action, both arms, identical inside the viewport: the pair that used to carry the run."""
    out = []
    for side in ("base", "treatment"):
        out.append(
            _action(
                "select_text",
                f"r100K.{side}.{cell_suffix}",
                parity = _capture(mounted = 6, total = 18),
                visible = _visible(ordinals, ordinals),
            )
        )
    return out


def test_a_cell_that_lost_messages_gets_no_visible_pass(tmp_path, capsys):
    """THE FALSE GREEN. The arm's own gate says it is missing the middle of the conversation and
    the visible region matched anyway, because the visible region is the end of the thread."""
    from studiobench.sweep import ui_parity as U

    rows = [_completeness_gate("r100K.treatment.rep0", False)] + _matching_pair()
    shard = _write(tmp_path, "lost_middle", rows)
    code = U.visible_report([shard], "lost middle")
    out = capsys.readouterr().out
    assert code == 2, out
    assert "visible region matched:     0" in out, out
    assert "FAILED its completeness gate" in out, out


def _held_invariant_pair(cell_suffix = "rep0"):
    """A `select_text` pair whose declared invariant HOLDS: the windowed arm selected the same
    characters and sized its spacers, so `compare_behaviour` returns MATCH. This is the shape a
    store that kept its first page and its last one still produces."""
    out = []
    for side, mounted in (("base", 18), ("treatment", 6)):
        row = _row(
            "select_text",
            _capture(mounted, 18),
            selected_chars = 100,
            visible_chars = 100,
        )
        row["cell_id"] = f"r100K.{side}.{cell_suffix}"
        out.append(row)
    return out


def test_a_cell_that_lost_messages_gets_no_behavioural_pass_either(tmp_path, capsys):
    """The behavioural invariants are what REPLACE the digest on a windowed arm, so a pass here is
    the whole UI verdict for that pair."""
    from studiobench.sweep import ui_parity as U

    rows = [_completeness_gate("r100K.treatment.rep0", False)] + _held_invariant_pair()
    shard = _write(tmp_path, "lost_middle_b", rows)
    code = U.behaviour_report([shard], "lost middle")
    out = capsys.readouterr().out
    assert code == 2, out
    assert "invariants held:            0" in out, out
    assert "NOTHING WAS COMPARED" in out, out


def _follow_gate(
    cell_id,
    passed,
    reason = "the thread fell behind the streamed reply for 38% of the streaming phase",
):
    return {
        "row_type": "gate",
        "name": "follows_the_stream",
        "passed": passed,
        "cell_id": cell_id,
        "detail": {"reason": reason},
    }


def test_a_cell_that_stopped_following_the_stream_gets_no_behavioural_pass(tmp_path, capsys):
    """REGRESSION. `follows_the_stream` invalidates a pair for the same reason `thread_complete`
    does, and this file recognised only the latter.

    A reply that scrolled out of the viewport and was unmounted stops costing anything to render,
    so the arm was not showing the thing under test -- yet the invariants that do not depend on it
    still hold. A `select_text` pair matches, `invariants held` grows, and `--mode auto` exits 0
    over a payload that has already recorded the arm losing the stream.
    """
    from studiobench.sweep import ui_parity as U

    rows = [_follow_gate("r100K.treatment.rep0", False)] + _held_invariant_pair()
    shard = _write(tmp_path, "lost_stream", rows)
    code = U.behaviour_report([shard], "lost stream")
    out = capsys.readouterr().out
    assert code == 2, out
    assert "invariants held:            0" in out, out
    assert "FAILED its stream-follow gate" in out, out


def test_a_cell_that_kept_following_the_stream_still_scores(tmp_path, capsys):
    """The positive control: the same pair, gate passed, is scored exactly as before."""
    from studiobench.sweep import ui_parity as U

    rows = [_follow_gate("r100K.treatment.rep0", True)] + _held_invariant_pair()
    shard = _write(tmp_path, "kept_stream", rows)
    code = U.behaviour_report([shard], "kept stream")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "invariants held:            1" in out, out


def test_a_complete_cell_still_earns_its_behavioural_pass(tmp_path, capsys):
    """The positive control for the test above: the same pair, gate passed, still scores."""
    from studiobench.sweep import ui_parity as U

    rows = [_completeness_gate("r100K.treatment.rep0", True)] + _held_invariant_pair()
    shard = _write(tmp_path, "complete_b", rows)
    code = U.behaviour_report([shard], "complete")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "invariants held:            1" in out, out


def test_a_cell_whose_completeness_gate_PASSED_is_scored_exactly_as_before(tmp_path, capsys):
    """The positive control. A refusal that fires on every cell measures nothing, and the shipped
    build passes this gate on every cell it runs."""
    from studiobench.sweep import ui_parity as U

    rows = [_completeness_gate("r100K.treatment.rep0", True)] + _matching_pair()
    shard = _write(tmp_path, "complete", rows)
    code = U.visible_report([shard], "complete")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "visible region matched:     1" in out, out


def test_only_the_cell_that_failed_is_refused(tmp_path, capsys):
    """Attribution. The gate names its cell, so a rep that lost messages must not silence the rep
    beside it -- that would be the mirror defect, a whole payload lost to one bad cell."""
    from studiobench.sweep import ui_parity as U

    rows = [_completeness_gate("r100K.treatment.rep0", False)]
    rows += _matching_pair("rep0")
    rows += _matching_pair("rep1")
    shard = _write(tmp_path, "one_bad_rep", rows)
    code = U.visible_report([shard], "one bad rep")
    out = capsys.readouterr().out
    assert code == 0, out
    assert "visible region matched:     1" in out, out
    assert "NOT COMPARABLE:             1" in out, out


# ── one glob pools separate runs, and a declaration belongs to the run that made it ──


def _legacy_capture(digest):
    """A capture from a checkout that predates `mounted_messages` / `thread_total`.

    `parity.windowed_mount` reads those two numbers and says of their absence: captures taken
    before they existed report neither and are treated as full-mount, which is what they were.
    Such a payload digests perfectly well and carries NO mount measurement, so every one of its
    pairs falls through to the declaration fallback.
    """
    return {
        "parity_attempted": True,
        "root_kind": "thread",
        "digest": digest,
        "chars": 100,
        "messages": [
            {"i": i, "role": "assistant", "digest": f"{digest}{i}", "chars": 10} for i in range(18)
        ],
        "overlays": [],
        "styles": {"elements": 18, "digest": "s", "capped": False},
    }


def _two_run_glob(tmp_path):
    """Two SEPARATE runs under one glob: `sb_win` was launched with `--windowed-arm treatment`,
    `sb_old` was an ordinary A/B from an older checkout whose treatment arm has a DOM regression."""
    win = [
        {
            "row_type": "gate",
            "name": "windowed_readiness:treatment",
            "passed": True,
            "detail": {"arm": "treatment"},
        }
    ]
    for side in ("base", "treatment"):
        win.append(
            _action(
                "select_all_copy",
                f"r100K.{side}.rep0",
                parity = _capture(mounted = 9 if side == "treatment" else 18, total = 18),
                visible = _visible([1], [1]),
                expect = _copy_expect(
                    clipboard = 200_000,
                    selected = 66_000 if side == "treatment" else 200_000,
                    mounted = 9 if side == "treatment" else 18,
                ),
            )
        )
    _write(tmp_path, "sb_win", win)
    old = [
        _action(
            "select_all_copy",
            f"r1K.{side}.rep0",
            parity = _legacy_capture("regressed" if side == "treatment" else "shipped"),
            visible = _visible([1], [1]),
            expect = _copy_expect(clipboard = 200_000, selected = 200_000, mounted = 18),
        )
        for side in ("base", "treatment")
    ]
    _write(tmp_path, "sb_old", old)


def _modes_by_shard_action(decided):
    """{(shard, action): mode}, so an assertion does not have to spell the whole pair key."""
    return {(key[0], key[-1]): mode for key, (mode, _why) in decided.items()}


def test_a_windowed_declaration_does_not_leak_into_another_run_under_one_glob(tmp_path, capsys):
    """DECLARATIONS ARE PER SHARD. `outputs/sbench_*` pools separate runs, and `cell_id` and arm
    label repeat identically in every one of them, so a `--windowed-arm treatment` gate row in one
    run became the fallback for every unmeasured pair in an ordinary run beside it. The ordinary
    run's pairs were scored on the visible region and on behavioural invariants instead, the
    structural digest they were owed never ran, and its DOM regression exited 0 because of a flag
    passed to a DIFFERENT run.
    """
    from studiobench.sweep import ui_parity as U

    _two_run_glob(tmp_path)
    modes = _modes_by_shard_action(U.decide_modes(U.shards_of(f"{tmp_path}/sb_*")))
    assert modes[("sb_old", "select_all_copy")] == U.STRUCTURAL, modes
    assert modes[("sb_win", "select_all_copy")] == U.WINDOWED, modes

    code = U.main([f"{tmp_path}/sb_*"])
    out = capsys.readouterr().out
    assert "UI PARITY DIFFERENCES ON STABLE ACTIONS" in out, out
    assert "sb_old" in out.split("UI PARITY DIFFERENCES ON STABLE ACTIONS")[1], out
    assert code == 1, out


def test_the_declaration_still_decides_the_run_that_made_it(tmp_path):
    """The positive control for the scoping above: a run's own declaration must still reach its own
    unmeasured pairs, which is the whole reason the fallback exists."""
    from studiobench.sweep import ui_parity as U

    shard = _declared_windowed_shard(tmp_path, "still_declared")
    assert all(mode == U.WINDOWED for mode, _why in U.decide_modes([shard]).values())
    other = _write(
        tmp_path,
        "unrelated",
        [
            _action("settings", f"r1K.{side}.rep0", parity = _capture(mounted = 18, total = 18))
            for side in ("base", "treatment")
        ],
    )
    modes = _modes_by_shard_action(U.decide_modes([shard, other]))
    assert modes[("still_declared", "select_all_copy")] == U.WINDOWED, modes
    assert modes[("unrelated", "settings")] == U.STRUCTURAL, modes


# ── a visible floor measured on another film tier is not this payload's floor ──


def _tiered_visible_shard(
    tmp_path,
    name,
    tier,
    differ_actions,
    windowed,
    corpus = "",
):
    """A visible-region payload that records the FILM TIER it was shot on, as `run_meta` does.

    `corpus` is the `corpus_hash` of the thread the film drove, written by the same row. Left out
    by default: a payload recorded before corpus hashes existed declares none, and that is the
    shape the tier tests below are about.
    """
    import json

    rows = [{"row_type": "run_meta", "tier": tier}]
    if corpus:
        rows[0]["corpus_hash"] = corpus
    for action in ("copy_markdown", "select_text"):
        for rep in range(2):
            for side in ("base", "treatment"):
                digest = "X" if (action in differ_actions and side == "treatment") else "same"
                mounted = 9 if (windowed and side == "treatment") else 18
                rows.append(
                    _action(
                        action,
                        f"r100K.{side}.rep{rep}",
                        parity = _capture(mounted = mounted, total = 18),
                        visible = _visible([1], [1], digest = digest),
                        expect = {
                            "clipboard_chars": 5000,
                            "selected_chars": 100,
                            "visible_chars": 100,
                        },
                    )
                )
    shard = tmp_path / name
    shard.mkdir()
    (shard / "payload.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
    return shard


def test_a_visible_floor_from_another_film_tier_is_not_applied(tmp_path, capsys):
    """THE INCOMPARABLE CONTROL. `tier_of` states the mechanism: on the fast film `copy_markdown`
    opens 2.7 s after a `send_turn` and lands inside that turn's stream, so it differs against
    itself; on the standard film it opens 26 s later against a finished reply. `--tier fast` then
    `--tier standard` is the documented way to work, so a stale fast null control on a standard
    run's command line is the ordinary mistake, not an exotic one.

    The floor derived from it was keyed by `(rung, action)` with no tier in it, so it silenced the
    standard payload's real visible difference, and an ALL-WINDOWED payload returns before the
    structural section that would have warned about the mismatch, so the command exited 0 without
    a word about the tier.
    """
    from studiobench.sweep import ui_parity as U

    null = _tiered_visible_shard(tmp_path, "null_fast", "fast", {"copy_markdown"}, windowed = False)
    arm = _tiered_visible_shard(
        tmp_path, "arm_standard", "standard", {"copy_markdown"}, windowed = True
    )
    assert U.visible_unstable_set(U.shards_of(str(null))) == frozenset({("r100K", "copy_markdown")})

    code = U.main([str(arm), "--null", str(null)])
    out = capsys.readouterr().out
    assert "FLOOR REFUSED" in out, out
    assert "DIFFERENCES INSIDE THE VIEWPORT" in out, out
    assert code == 1, out


def test_a_visible_floor_from_the_SAME_tier_still_applies(tmp_path, capsys):
    """The positive control, and the verdict this must not change: a null control shot on the same
    film still silences the action it measured differing against itself."""
    from studiobench.sweep import ui_parity as U

    null = _tiered_visible_shard(
        tmp_path, "null_std", "standard", {"copy_markdown"}, windowed = False
    )
    arm = _tiered_visible_shard(tmp_path, "arm_std", "standard", {"copy_markdown"}, windowed = True)
    code = U.main([str(arm), "--null", str(null)])
    out = capsys.readouterr().out
    assert "FLOOR REFUSED" not in out, out
    assert "differ against an identical build" in out, out
    assert code == 0, out


def test_a_visible_floor_from_another_corpus_is_not_applied(tmp_path, capsys):
    """THE OTHER AXIS OF THE SAME INCOMPARABLE CONTROL, and the quieter one. Which actions differ
    against an identical build is a property of the thread the film drove, not only of the film's
    spacing, so a null control recorded against corpus `c1` describes a thread a `c2` payload
    never rendered. The ordinary way in is a corpus revision landing while an older null control
    is still sitting in the directory the workflow globs.

    `cross_side_mismatch` already refuses this, but only in the structural section, and an
    ALL-WINDOWED payload -- which is every payload under `--mode visible` -- returns before it,
    so the wrong-corpus floor silenced the payload's real visible difference and the command
    exited 0 without a word about the corpus.
    """
    from studiobench.sweep import ui_parity as U

    null = _tiered_visible_shard(
        tmp_path, "null_c1", "standard", {"copy_markdown"}, windowed = False, corpus = "c1"
    )
    arm = _tiered_visible_shard(
        tmp_path, "arm_c2", "standard", {"copy_markdown"}, windowed = True, corpus = "c2"
    )
    assert U.visible_unstable_set(U.shards_of(str(null))) == frozenset({("r100K", "copy_markdown")})

    code = U.main([str(arm), "--null", str(null)])
    out = capsys.readouterr().out
    assert "FLOOR REFUSED" in out, out
    assert "corpus" in out, out
    assert "DIFFERENCES INSIDE THE VIEWPORT" in out, out
    assert code == 1, out


def test_a_visible_floor_from_the_SAME_corpus_still_applies(tmp_path, capsys):
    """The positive control for the corpus axis: a null control recorded against the same thread
    still silences the action it measured differing against itself."""
    from studiobench.sweep import ui_parity as U

    null = _tiered_visible_shard(
        tmp_path, "null_same", "standard", {"copy_markdown"}, windowed = False, corpus = "c1"
    )
    arm = _tiered_visible_shard(
        tmp_path, "arm_same", "standard", {"copy_markdown"}, windowed = True, corpus = "c1"
    )
    code = U.main([str(arm), "--null", str(null)])
    out = capsys.readouterr().out
    assert "FLOOR REFUSED" not in out, out
    assert "differ against an identical build" in out, out
    assert code == 0, out


# ── a resumed cell is judged on the attempt that survived, gates included ──


def _resumed_completeness_shard(tmp_path, name, *, retry_passes):
    """A payload where attempt 1 of a cell FAILED `thread_complete` and `--resume` re-ran it.

    `make_cell_id` is deterministic, so the retry carries the SAME `cell_id` and is told apart
    only by `session_id`, exactly as `latest_attempt_rows` documents.
    """
    import json

    rows = [{"row_type": "run_meta", "tier": "standard", "session_id": "s1"}]
    for side in ("base", "treatment"):
        cid = f"r100K.{side}.rep0"
        rows.append({"row_type": "cell", "cell_id": cid, "session_id": "s1"})
        rows.append(
            {
                "row_type": "gate",
                "name": "thread_complete",
                "passed": False,
                "cell_id": cid,
                "session_id": "s1",
                "detail": {"reason": "the head marker never mounted"},
            }
        )
    for side in ("base", "treatment"):
        cid = f"r100K.{side}.rep0"
        rows.append({"row_type": "cell", "cell_id": cid, "session_id": "s2"})
        rows.append(
            {
                "row_type": "gate",
                "name": "thread_complete",
                "passed": bool(retry_passes),
                "cell_id": cid,
                "session_id": "s2",
                "detail": {"reason": "" if retry_passes else "still short"},
            }
        )
        act = _action(
            "select_text",
            cid,
            parity = _capture(mounted = 18, total = 18),
            visible = _visible([1], [1]),
        )
        act["session_id"] = "s2"
        rows.append(act)
    shard = tmp_path / name
    shard.mkdir()
    (shard / "payload.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding = "utf-8")
    return shard


def test_a_successful_resume_clears_the_dead_attempts_completeness_failure(tmp_path):
    """THE SUPERSEDED GATE MUST NOT OUTLIVE ITS ATTEMPT. `latest_attempt_rows` drops a dead
    attempt's rows everywhere else, but its ATTEMPT_ROW_TYPES is {cell, action, window} and a gate
    is none of those, so `incomplete_cells` scanned raw rows and kept the failure from the attempt
    that died. `collect` then stamped `_incomplete` on the RETRY's action rows and `_refused`
    withheld a verdict from a cell that had just been re-measured successfully.
    """
    from studiobench.sweep import ui_parity as U

    shard = _resumed_completeness_shard(tmp_path, "resumed_ok", retry_passes = True)
    assert U.incomplete_cells([shard / "payload.jsonl"]) == {}


def test_a_resume_that_failed_again_is_still_refused(tmp_path):
    """The positive control: scoping the gate to the surviving attempt must not lose a real
    refusal when the retry failed too."""
    from studiobench.sweep import ui_parity as U

    shard = _resumed_completeness_shard(tmp_path, "resumed_bad", retry_passes = False)
    bad = U.incomplete_cells([shard / "payload.jsonl"])
    assert set(bad) == {"r100K.base.rep0", "r100K.treatment.rep0"}, bad
    assert "still short" in bad["r100K.base.rep0"], bad


# ── the coverage floor applies to every mode, not only the structural one ─────


def _windowed_only_shard(
    tmp_path,
    name,
    *,
    pairs = 1,
    differ = False,
):
    """A payload whose every pair is a WINDOWED mount, so `main` takes the structural early return.

    One clean, matching windowed pair per action. `--mode auto` classifies these on the capture's
    own `mounted_messages` / `thread_total`, so nothing here has to be declared.
    """
    rows = []
    for i in range(pairs):
        action = f"select_text{i}" if i else "select_text"
        for side in ("base", "treatment"):
            windowed = side == "treatment"
            digest = "regressed" if (differ and windowed) else "shipped"
            rows.append(
                _action(
                    action,
                    f"r100K.{side}.rep0",
                    parity = _capture(mounted = 9 if windowed else 18, total = 18),
                    visible = _visible([1], [1], digest = digest),
                    expect = {"selected_chars": 100, "visible_chars": 100},
                )
            )
    return _write(tmp_path, name, rows)


def _windowed_shard_visible_only_on_one(
    tmp_path,
    name,
    *,
    pairs = 4,
):
    """Every pair carries a behavioural invariant; only the FIRST carries a visible capture.

    One action at four cells, so all four are pairs behavioural mode declares an invariant for --
    what varies is whether the visible-region capture is there to be read at all.
    """
    rows = []
    for i in range(pairs):
        for side in ("base", "treatment"):
            windowed = side == "treatment"
            rows.append(
                _action(
                    "select_text",
                    f"r100K.{side}.rep{i}",
                    parity = _capture(mounted = 9 if windowed else 18, total = 18),
                    visible = _visible([1], [1], digest = "shipped") if i == 0 else None,
                    expect = {"selected_chars": 100, "visible_chars": 100},
                )
            )
    return _write(tmp_path, name, rows)


def test_behavioural_coverage_does_not_stand_in_for_visible_coverage(tmp_path, capsys):
    """THE SUBSTITUTION. The floor unioned the two windowed modes' compared sets.

    They are not two slices of one film. Under `--mode auto` they are two verdicts on the SAME
    pairs, so the union let one stand in for the other: behavioural mode reaching every pair while
    the visible capture could be read on only ONE still filled the set, cleared the floor, and
    exited 0 with essentially the whole appearance surface unmeasured. That is the same false green
    the floor exists to refuse, arriving through the floor itself.

    Four pairs here, all of them behaviourally checked, one of them visible. A floor of 4 has to
    fire, and the count has to be the visible one.
    """
    from studiobench.sweep import ui_parity as U

    _windowed_shard_visible_only_on_one(tmp_path, "winsub", pairs = 4)
    code = U.main([str(tmp_path / "winsub"), "--min-compared", "4"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" in out, out
    assert "1 of 4 pair(s) carry a verdict" in out, out
    assert code == 3, out


def test_a_pair_behavioural_mode_declares_no_invariant_for_is_not_a_shortfall(tmp_path, capsys):
    """THE OTHER DIRECTION, and the reason this is not an intersection.

    A behavioural invariant only exists for the handful of actions that declare one; every other
    pair is reported UNCHECKED, which is not a pass and equally not a coverage shortfall. Requiring
    both verdicts on every pair therefore fails a run for pairs behavioural mode never had an
    opinion about -- measured at 1 of 4 on this very fixture, where visible mode read all four.
    """
    from studiobench.sweep import ui_parity as U

    _windowed_only_shard(tmp_path, "winunchecked", pairs = 4)
    code = U.main([str(tmp_path / "winunchecked"), "--min-compared", "4"])
    out = capsys.readouterr().out
    assert "UNCHECKED:                  3" in out, out
    assert "TOO LITTLE COMPARED" not in out, out
    assert code == 0, out


def test_the_coverage_floor_applies_to_a_run_with_no_fully_mounted_pair(tmp_path, capsys):
    """THE HOLE. `--min-compared` was a parameter of `report()` and enforced at one site inside it,
    and `report()` is reached only from the structural loop BELOW `main`'s early return. A payload
    whose every pair is a windowed mount returns before that loop, so the floor was never applied
    to anything: `--min-compared 20` exited 0 having compared one pair.

    That floor is what replaces a slot-budget liveness gate for a job that reads no timings, and
    two blank pages have identical digests, so it is the guard against the easiest false green
    there is. The workflow passes `--min-compared 16` on every run.
    """
    from studiobench.sweep import ui_parity as U

    _windowed_only_shard(tmp_path, "winonly", pairs = 1)
    code = U.main([str(tmp_path / "winonly"), "--min-compared", "20"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" in out, out
    assert "below the floor of 20" in out, out
    assert code == 3, out


def test_the_coverage_floor_passes_a_windowed_run_that_compared_enough(tmp_path, capsys):
    """THE POSITIVE CONTROL, and it is the half that matters most here: a floor that fires on
    everything is not a floor. The same shape as above with the floor set beneath what the run
    actually compared has to come out 0, or the test above would pass against a `return 3`."""
    from studiobench.sweep import ui_parity as U

    _windowed_only_shard(tmp_path, "winenough", pairs = 4)
    code = U.main([str(tmp_path / "winenough"), "--min-compared", "4"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" not in out, out
    assert code == 0, out


def test_the_behaviour_policy_prints_the_coverage_band_it_actually_enforces(tmp_path, capsys):
    """The printed policy said the copy must be "complete", which reads as a content comparison.

    It is a LENGTH: each arm's clipboard over the thread's own visible text, inside a band. That
    band is the whole strength of the claim -- it is what refused truncation at 0.61 and
    substitution at 2.16 -- so the sentence has to carry the live numbers rather than a pair
    somebody typed once. The needle here is DERIVED from the constants, so widening the band
    without touching the wording is caught, and so is a sentence that stopped interpolating.
    """
    from studiobench.sweep import ui_parity as U

    band = P.behaviour_policy(B.MIN_CLIPBOARD_COVERAGE, B.MAX_CLIPBOARD_COVERAGE)
    assert f"{B.MIN_CLIPBOARD_COVERAGE}-{B.MAX_CLIPBOARD_COVERAGE}" in band, band

    # A different band has to travel through as well: a hard-coded sentence would pass the
    # assertion above by coincidence and fail this one.
    other = P.behaviour_policy(0.5, 2.0)
    assert "0.5-2.0" in other, other
    assert f"{B.MIN_CLIPBOARD_COVERAGE}-{B.MAX_CLIPBOARD_COVERAGE}" not in other, other

    # And the report has to be wired to it, not merely able to produce it.
    _mixed_rung_shard(tmp_path, "banner")
    U.main([str(tmp_path / "banner"), "--mode", "behaviour", "--min-compared", "0"])
    out = capsys.readouterr().out
    assert f"{B.MIN_CLIPBOARD_COVERAGE}-{B.MAX_CLIPBOARD_COVERAGE}" in out, out


def test_the_coverage_floor_is_applied_under_forced_visible_and_behaviour_modes(tmp_path, capsys):
    """The other route to an empty structural plan, and it does not need a windowed payload at all.
    `--mode visible` and `--mode behaviour` hard-set `structural` to an empty set for EVERY
    pattern, unconditionally, so under either of them the floor could never be applied to any
    payload whatsoever."""
    from studiobench.sweep import ui_parity as U

    _mixed_rung_shard(tmp_path, "forced")
    for mode in ("visible", "behaviour"):
        code = U.main([str(tmp_path / "forced"), "--mode", mode, "--min-compared", "50"])
        out = capsys.readouterr().out
        assert "TOO LITTLE COMPARED" in out, (mode, out)
        assert code == 3, (mode, out)


def test_the_coverage_floor_sums_the_windowed_and_structural_halves(tmp_path, capsys):
    """SUMMED, NOT CHECKED SEPARATELY. A combined run splits one film's coverage across three
    reports, and a floor applied inside each of them separately fails a run that compared plenty:
    the structural half here compares 1 pair and the windowed half 1, and a floor of 2 is cleared
    by the run and by neither half alone.

    The `--mode auto` pair is scored TWICE, once on the visible region and once on the invariants,
    and it still counts once: summing the two reports' counts would double every windowed pair and
    let a floor of 2 pass on 1."""
    from studiobench.sweep import ui_parity as U

    _mixed_rung_shard(tmp_path, "sums")
    code = U.main([str(tmp_path / "sums"), "--min-compared", "2"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" not in out, out
    # 1 is the digest regression the fixture carries at the 1K rung, so the floor did not mask it.
    assert code == 1, out
    # And one more than the run can reach refuses it, which is what proves the count above is 2
    # rather than the 3 a sum of the three reports would have produced.
    code = U.main([str(tmp_path / "sums"), "--min-compared", "3"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED: 2 of 2" in out, out
    assert code == 3, out


def test_the_coverage_floor_is_checked_per_payload_pattern(tmp_path, capsys):
    """PER FILM, not per invocation. The three modes are one film's coverage split three ways, so
    they are unioned; two positional payloads are two films, and pooling those lets each satisfy
    the other's floor. `ui_parity normal_run windowed_run` is a supported invocation -- the mode
    decision above already says why per-invocation was too coarse for it -- so a nearly empty or
    failed film could be carried over the floor by an unrelated one on the same command line.

    Two films of 2 pairs each against a floor of 4: pooled they clear it, and neither one did."""
    from studiobench.sweep import ui_parity as U

    _windowed_only_shard(tmp_path, "filmA", pairs = 2)
    _windowed_only_shard(tmp_path, "filmB", pairs = 2)
    code = U.main([str(tmp_path / "filmA"), str(tmp_path / "filmB"), "--min-compared", "4"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" in out, out
    assert out.count("TOO LITTLE COMPARED") == 2, out
    assert "filmA" in out and "filmB" in out, out
    assert code == 3, out


def test_a_floor_each_film_clears_on_its_own_still_passes(tmp_path, capsys):
    """THE POSITIVE CONTROL for the one above: per-pattern must not mean every multi-payload run
    fails. Two films of 2 pairs each against a floor of 2 is 0, or the test above would pass
    against an unconditional `return 3`."""
    from studiobench.sweep import ui_parity as U

    _windowed_only_shard(tmp_path, "filmA", pairs = 2)
    _windowed_only_shard(tmp_path, "filmB", pairs = 2)
    code = U.main([str(tmp_path / "filmA"), str(tmp_path / "filmB"), "--min-compared", "2"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" not in out, out
    assert code == 0, out


# ── a build difference the capture comparison cannot see ─────


def test_an_assertion_that_failed_on_one_arm_fails_the_windowed_verdict(tmp_path, capsys):
    """`stop_generation` returns `ran = True, expect_ok = stopped_ms is not None`, so a head on
    which Stop no longer ends the stream records a perfectly ordinary row with two viewports that
    look the same. The visible and behavioural paths built their whole result from a capture
    comparison plus one boolean -- the conjunction of `ran` -- so that shape left no trace at all,
    and on a windowed arm those two reports ARE the UI verdict: the digest is not applicable by
    construction and `stop_generation` has no behavioural invariant of its own, so a passing
    invariant on some other action carried the run to exit 0."""
    from studiobench.sweep import ui_parity as U

    rows = []
    for side in ("base", "treatment"):
        windowed = side == "treatment"
        mounted = 9 if windowed else 18
        # A held invariant on another action, which is what used to carry the run to a pass.
        rows.append(
            _action(
                "select_text",
                f"r100K.{side}.rep0",
                parity = _capture(mounted = mounted, total = 18),
                visible = _visible([1], [1]),
                expect = {"selected_chars": 100, "visible_chars": 100},
            )
        )
        stop = _action(
            "stop_generation",
            f"r100K.{side}.rep0",
            parity = _capture(mounted = mounted, total = 18),
            visible = _visible([1], [1]),
        )
        if windowed:
            stop["expect_ok"] = False
            stop["reason"] = "the stream did not stop"
        rows.append(stop)
    _write(tmp_path, "assertion", rows)

    code = U.main([str(tmp_path / "assertion")])
    out = capsys.readouterr().out
    assert "ASSERTION FAILED ON ONE ARM" in out, out
    assert "stop_generation" in out.split("ASSERTION FAILED ON ONE ARM")[1], out
    assert code == 1, out


def test_an_assertion_that_failed_on_BOTH_arms_is_not_a_build_difference(tmp_path, capsys):
    """THE CONTROL, and it is the reason the signal is an ASYMMETRY rather than a failure count.
    Both arms failing means the fixture cannot reach the state on either build -- coverage lost,
    not a difference between the builds -- and refusing on it would red every run whose scene
    cannot reach `stop_generation` at all."""
    from studiobench.sweep import ui_parity as U

    rows = []
    for side in ("base", "treatment"):
        windowed = side == "treatment"
        rows.append(
            _action(
                "select_text",
                f"r100K.{side}.rep0",
                parity = _capture(mounted = 9 if windowed else 18, total = 18),
                visible = _visible([1], [1]),
                expect = {"selected_chars": 100, "visible_chars": 100},
            )
        )
        stop = _action(
            "stop_generation",
            f"r100K.{side}.rep0",
            parity = _capture(mounted = 9 if windowed else 18, total = 18),
            visible = _visible([1], [1]),
        )
        stop["expect_ok"] = False
        stop["reason"] = "the stop button is not present"
        rows.append(stop)
    _write(tmp_path, "sym", rows)

    code = U.main([str(tmp_path / "sym")])
    out = capsys.readouterr().out
    assert "ASSERTION FAILED ON ONE ARM" not in out, out
    assert code == 0, out


def test_an_action_that_could_not_be_performed_on_one_arm_fails_the_windowed_verdict(
    tmp_path, capsys
):
    """The other shape a capture comparison erases. An action that RUNS on one build and cannot be
    performed on the other is the one regression class that leaves no digest to differ -- a control
    that stopped opening records exactly this -- and both windowed reports filed it under NOT
    EXERCISED, which each of them counts as lost coverage rather than as a finding."""
    from studiobench.sweep import ui_parity as U

    rows = []
    for side in ("base", "treatment"):
        windowed = side == "treatment"
        mounted = 9 if windowed else 18
        rows.append(
            _action(
                "select_text",
                f"r100K.{side}.rep0",
                parity = _capture(mounted = mounted, total = 18),
                visible = _visible([1], [1]),
                expect = {"selected_chars": 100, "visible_chars": 100},
            )
        )
        rows.append(
            _action(
                "message_menu",
                f"r100K.{side}.rep0",
                parity = _capture(mounted = mounted, total = 18),
                visible = _visible([1], [1]),
                ran = not windowed,
                reason = "no message menu control on the page" if windowed else None,
            )
        )
    _write(tmp_path, "onesided", rows)

    code = U.main([str(tmp_path / "onesided")])
    out = capsys.readouterr().out
    assert "RAN ON ONE ARM ONLY" in out, out
    assert code == 1, out


def test_a_slot_the_runner_arrived_too_late_for_is_still_not_a_build_difference(tmp_path, capsys):
    """THE CONTROL FOR THE ONE ABOVE, and it is the load-bearing half of `one_sided`. A missed slot
    is a slow machine and cannot move this verdict: the misses are correlated through the runner
    rather than being independent draws, so corroboration cannot separate them and reding on them
    would fail the job on machine speed."""
    from studiobench.sweep import ui_parity as U

    rows = []
    for side in ("base", "treatment"):
        windowed = side == "treatment"
        mounted = 9 if windowed else 18
        rows.append(
            _action(
                "select_text",
                f"r100K.{side}.rep0",
                parity = _capture(mounted = mounted, total = 18),
                visible = _visible([1], [1]),
                expect = {"selected_chars": 100, "visible_chars": 100},
            )
        )
        row = _action(
            "message_menu",
            f"r100K.{side}.rep0",
            parity = _capture(mounted = mounted, total = 18),
            visible = _visible([1], [1]),
            ran = not windowed,
            reason = "the slot closed before the runner reached it" if windowed else None,
        )
        if windowed:
            row["slot_missed"] = True
        rows.append(row)
    _write(tmp_path, "missed", rows)

    code = U.main([str(tmp_path / "missed")])
    out = capsys.readouterr().out
    assert "RAN ON ONE ARM ONLY" not in out, out
    assert code == 0, out
