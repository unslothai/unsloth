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

    Not hypothetical: the probe walks a hand-written selector list written against Studio's
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
):
    """A payload shard whose visible-region captures differ on `differ_actions` and match elsewhere."""
    import json

    rows = []
    for action in actions:
        for side in ("base", "treatment"):
            digest = "X" if (action in differ_actions and side == "treatment") else "same"
            rows.append(
                {
                    "row_type": "action",
                    "action": action,
                    "ran": True,
                    "cell_id": f"r100K.{side}.rep0",
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
    assert unstable == frozenset({"a", "b"}), unstable

    # Unfloored: the arm's one difference is counted and the run fails.
    assert U.visible_report([arm], "unfloored") == 1
    # Floored by its own null: that action is known to differ against itself, so it is reported
    # and not counted.
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
    assert "model_change" in unstable
    # ...and the arm that lost the thread on it is STILL a failure.
    assert U.visible_report([_shard("arm_mc", True)], "severe", unstable) == 1
    assert "one arm lost the thread" in capsys.readouterr().out
