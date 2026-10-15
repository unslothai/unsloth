# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windowed arms mount less DOM by design, so the digest must refuse such pairs, not fail them."""

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


def test_a_windowed_capture_is_detected_from_its_own_numbers():
    assert P.windowed_mount(_capture(6, 18)) is True
    assert P.windowed_mount(_capture(18, 18)) is False
    # An old payload carries neither number and is treated as a full mount.
    assert P.windowed_mount({"parity_attempted": True, "digest": "d"}) is False


def test_the_digest_refuses_a_windowed_pair_rather_than_reporting_eighteen_differences():
    got = P.compare(_capture(18, 18, "base"), _capture(6, 18, "treat"))
    assert got["verdict"] == P.NOT_APPLICABLE
    assert got["verdict"] != P.DIFFER
    assert "mounts a WINDOW" in got["reason"]
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
    """Unequal mounted message counts are a DIFFER, not inapplicable, even with no declared total."""
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


def test_the_scroll_extent_invariant_passes_a_virtualizer_that_sizes_its_spacers():
    base = _row("select_text", _capture(18, 18), selected_chars = 100, visible_chars = 100)
    treat = _row("select_text", _capture(6, 18), selected_chars = 100, visible_chars = 100)
    treat["census"]["viewport_scroll_height"] = 9_600
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
    """The alarm goes quiet only when copy reads the message store, not on the selected character count."""
    base = _copy_row(_capture(18, 18), clipboard = 200_000, selected = 200_000, mounted = 18)
    treat = _copy_row(_capture(6, 18), clipboard = 200_000, selected = 66_000, mounted = 6)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.MATCH, got["reason"]
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_carries_the_whole_thread:base"]["ok"] is True
    assert checks["clipboard_carries_the_whole_thread:treatment"]["ok"] is True
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


def _reopen_row(
    mounted,
    *,
    before = 18,
    after = 18,
    ready = True,
):
    """A timed-out reopen: ran stays true, reopen_ms is null, and both counts equal the declared total."""
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
    """An unready reopen must not pass as equal counts; the rebuild never happened."""
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
    assert checks["reopen_used_the_control:treatment"]["ok"] is True


def test_a_reopen_that_lost_messages_is_still_broken_when_the_gate_also_refused_it():
    """A reopen that lost messages stays BROKEN even when the readiness gate also refused it."""
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
    """Pre-readiness-gate rows carry no rebuild evidence and must not count as a held invariant."""
    base = _reopen_row(18)
    treat = _reopen_row(6)
    for row in (base, treat):
        row["expect"].pop("reopen_readiness")
        row["expect_ok"] = None
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got


def test_a_timed_out_rebuild_leaves_the_behavioural_run_with_no_verdict(tmp_path, capsys):
    """Behaviour mode must match `report`: a directional failure outranks 'could not tell', so exit 1."""
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


def test_two_full_mounts_of_different_lengths_is_a_difference_not_an_excuse():
    """Two full mounts of different lengths are a real user-visible difference, not NOT_APPLICABLE."""
    base = _capture(mounted = 18, total = 18)
    treat = _capture(mounted = 17, total = 17)
    assert P.windowed_mount(base) is False and P.windowed_mount(treat) is False
    got = P.compare(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert "NEITHER arm is windowing" in got["reason"]
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
    """Scoring that validated nothing (`matched` zero, `broken` empty) must not read as a pass."""
    from studiobench.sweep import ui_parity as U

    shard = tmp_path / "payload.jsonl"
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
    """An all-broken run must not also print that nothing was compared; the exit code stays 1."""
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
    """The no-verdict banner must still print when a build difference is the only failure."""
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
    """The budget deadline is read when the measured window closes, so observation cost is not charged."""
    import inspect

    from studiobench.scene import schedule as S

    src = inspect.getsource(S.SceneRunner)
    close = src.index("window_closed_at = time.monotonic()")
    over = src.index("over_ms = ((window_closed_at - t0)", close)
    # `_census` is called from several places, so anchor the search after the deadline.
    census = src.index("census = self._census()", over)
    assert close < over < census, (
        "the deadline is sampled after the observations again, which charges instrument time to "
        "the action's budget"
    )


def _styled(elements: int, digest: str = "s") -> dict:
    cap = _capture(mounted = 18, total = 18)
    cap["styles"] = {"elements": elements, "digest": digest, "capped": False}
    return cap


def test_a_style_probe_that_matched_no_elements_does_not_report_a_match():
    """Empty probes have equal digests, so a style probe matching no elements must be NOT_COMPARABLE."""
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
    """A clipboard holding far more than the thread must fail too; a lower bound alone passes it."""
    base = _copy_row(_capture(18, 18), clipboard = 193_937, selected = 194_992, mounted = 18)
    treat = _copy_row(_capture(9, 18), clipboard = 420_911, selected = 118_089, mounted = 9)
    got = B.compare_behaviour(base, treat)
    assert got["verdict"] == B.BROKEN, got
    checks = {c["invariant"]: c for c in got["checks"]}
    assert checks["clipboard_carries_the_whole_thread:treatment"]["ok"] is False
    assert "2.159" in checks["clipboard_carries_the_whole_thread:treatment"]["detail"]


def test_markdown_source_against_rendered_text_is_not_treated_as_a_difference():
    """Rendered DOM text and markdown source differ in form, so the two clipboards must not be compared."""
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


def _visible_shard(
    tmp_path,
    name,
    differ_actions,
    actions = ("a", "b", "c"),
    rung = "r100K",
    reps = 2,
):
    """Two reps by default, because a noise floor entry needs repeated readings at one rung."""
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
    """Identical builds differ on volatile attributes, so without a noise floor the arm ranks backwards."""
    from studiobench.sweep import ui_parity as U

    null = _visible_shard(tmp_path, "null", differ_actions = {"a", "b"})
    arm = _visible_shard(tmp_path, "arm", differ_actions = {"a"})

    unstable = U.visible_unstable_set([null])
    assert unstable == frozenset({("r100K", "a"), ("r100K", "b")}), unstable

    assert U.visible_report([arm], "unfloored") == 1
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
    """A noise floor from one rung does not apply to another, so it cannot silence a regression there."""
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
    assert "r1K rep0" in out
    assert "differ against an identical build" in out


def test_a_floor_derived_from_a_single_pair_is_not_a_floor(tmp_path):
    """One occurrence is not evidence: a floor derived from a single differing pair is not a floor."""
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
    """Noise-floor filtering must not hide an arm that lost the thread; a real loss is not jitter."""
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

    null = _visible_shard(
        tmp_path, "null_mc", differ_actions = {"model_change"}, actions = ("model_change",)
    )
    unstable = U.visible_unstable_set([null])
    assert ("r100K", "model_change") in unstable
    assert U.visible_report([_shard("arm_mc", True)], "severe", unstable) == 1
    assert "one arm lost the thread" in capsys.readouterr().out


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
    """A structural refusal must not discard the independent computed-style verdict on the same pair."""
    from studiobench.sweep import ui_parity as U

    base = _capture(mounted = 4, total = 4)
    treat = _capture(mounted = 4, total = 4)
    treat["in_flight_unplaced"] = True
    treat["in_flight"] = []
    base["streaming"] = treat["streaming"] = True
    treat["styles"] = {"elements": 4, "digest": "RESTYLED", "capped": False}

    rows = [_row("select_text", base), _row("select_text", treat)]
    rows[0]["cell_id"], rows[1]["cell_id"] = "r100K.base.rep0", "r100K.treatment.rep0"
    _write(tmp_path, "stylerefuse", rows)

    U.main([str(tmp_path / "stylerefuse"), "--mode", "structural", "--min-compared", "0"])
    out = capsys.readouterr().out
    assert "style probe differing:      1" in out, out


def test_a_visible_message_that_could_not_be_digested_is_printed_and_not_counted(tmp_path, capsys):
    """A visible message that could not be digested is printed as uncompared, never counted as a match."""
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


def _failed_parity(why = "the parity probe timed out"):
    return {"parity_attempted": False, "reason": why}


def _declared_windowed_shard(
    tmp_path,
    name,
    *,
    arm = "treatment",
    parity = None,
):
    """Declares a windowed arm in both the gate row and `readiness.mode`, since both are read."""
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
    """A declared windowed run that measured nothing stays windowed; the declaration decides then."""
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
    """A payload whose captures all failed must not print a structural pass, since no pair was compared."""
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
    """A fully mounted 1K rung beside a clean windowed 100K rung, so only the 1K digest can fail it."""
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
    """The pair key must include the rung, or `r1K.base.rep0` and `r100K.base.rep0` overwrite each other."""
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
    """Modes are decided per pair, so a windowed rung cannot hide the digest of a mounted rung."""
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


def _one_sided_shard(
    tmp_path,
    name,
    *,
    gate = True,
    cell_row = False,
):
    """One arm recorded at the large rung; the run's own declaration is the only evidence of windowing."""
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
    """With no declaration anywhere, a pair missing one arm stays structural and is refused there."""
    from studiobench.sweep import ui_parity as U

    shard = _one_sided_shard(tmp_path, "one_sided_undeclared", gate = False)
    modes = {
        f"{rung} {rep}": mode
        for (_s, rung, rep, _sid, _a), (mode, _why) in U.decide_modes([shard]).items()
    }
    assert modes == {"r1K rep0": U.STRUCTURAL, "r100K rep0": U.STRUCTURAL}, modes


def test_a_capture_that_saw_no_thread_at_all_falls_back_on_the_declaration(tmp_path):
    """A capture that saw no messages cannot show a window, so it falls back on the declaration."""
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


# A cell that failed its completeness gate carries no UI verdict, as for performance scoring.


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
    """A cell that stopped following the stream is invalidated, as `thread_complete` already does."""
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


def _legacy_capture(digest):
    """Captures lacking mount fields have no measurement, so their pairs fall back on the declaration."""
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
    """Declarations are per shard, so a `--windowed-arm` row in one run cannot leak into another."""
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


def _tiered_visible_shard(
    tmp_path,
    name,
    tier,
    differ_actions,
    windowed,
    corpus = "",
):
    """Records the film tier in run_meta, and omits the corpus hash by default, like old payloads."""
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
    """A floor from another film tier must not apply: fast-tier `copy_markdown` differs against itself."""
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
    """A floor from another corpus describes a thread this payload never rendered, so it must not apply."""
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


def _resumed_completeness_shard(tmp_path, name, *, retry_passes):
    """A failed-then-resumed cell keeps its `cell_id`, told apart only by `session_id`."""
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
    """A dead attempt's gate row must not outlive it, or a re-measured retry is wrongly refused."""
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


def _windowed_only_shard(
    tmp_path,
    name,
    *,
    pairs = 1,
    differ = False,
):
    """Every pair is a windowed mount, which `--mode auto` detects from the capture, not a declaration."""
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
    """Every pair has a behavioural invariant, but only the first has a visible-region capture to read."""
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
    """Behavioural coverage must not stand in for visible coverage when the coverage floor is counted."""
    from studiobench.sweep import ui_parity as U

    _windowed_shard_visible_only_on_one(tmp_path, "winsub", pairs = 4)
    code = U.main([str(tmp_path / "winsub"), "--min-compared", "4"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" in out, out
    assert "1 of 4 pair(s) carry a verdict" in out, out
    assert code == 3, out


def test_a_pair_behavioural_mode_declares_no_invariant_for_is_not_a_shortfall(tmp_path, capsys):
    """A pair with no behavioural invariant is UNCHECKED, not a shortfall, so it cannot fail the floor."""
    from studiobench.sweep import ui_parity as U

    _windowed_only_shard(tmp_path, "winunchecked", pairs = 4)
    code = U.main([str(tmp_path / "winunchecked"), "--min-compared", "4"])
    out = capsys.readouterr().out
    assert "UNCHECKED:                  3" in out, out
    assert "TOO LITTLE COMPARED" not in out, out
    assert code == 0, out


def test_the_coverage_floor_applies_to_a_run_with_no_fully_mounted_pair(tmp_path, capsys):
    """`--min-compared` must apply to a run with no fully mounted pair, not only the structural loop."""
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
    """The printed policy must state the live clipboard coverage band, derived from the constants."""
    from studiobench.sweep import ui_parity as U

    band = P.behaviour_policy(B.MIN_CLIPBOARD_COVERAGE, B.MAX_CLIPBOARD_COVERAGE)
    assert f"{B.MIN_CLIPBOARD_COVERAGE}-{B.MAX_CLIPBOARD_COVERAGE}" in band, band

    # A different band must travel through, so a hard-coded sentence cannot pass.
    other = P.behaviour_policy(0.5, 2.0)
    assert "0.5-2.0" in other, other
    assert f"{B.MIN_CLIPBOARD_COVERAGE}-{B.MAX_CLIPBOARD_COVERAGE}" not in other, other

    _mixed_rung_shard(tmp_path, "banner")
    U.main([str(tmp_path / "banner"), "--mode", "behaviour", "--min-compared", "0"])
    out = capsys.readouterr().out
    assert f"{B.MIN_CLIPBOARD_COVERAGE}-{B.MAX_CLIPBOARD_COVERAGE}" in out, out


def test_the_coverage_floor_is_applied_under_forced_visible_and_behaviour_modes(tmp_path, capsys):
    """`--mode visible` and `--mode behaviour` empty the structural set, so the floor must still apply."""
    from studiobench.sweep import ui_parity as U

    _mixed_rung_shard(tmp_path, "forced")
    for mode in ("visible", "behaviour"):
        code = U.main([str(tmp_path / "forced"), "--mode", mode, "--min-compared", "50"])
        out = capsys.readouterr().out
        assert "TOO LITTLE COMPARED" in out, (mode, out)
        assert code == 3, (mode, out)


def test_the_coverage_floor_sums_the_windowed_and_structural_halves(tmp_path, capsys):
    """The floor sums the windowed and structural halves of a run, and an auto pair still counts once."""
    from studiobench.sweep import ui_parity as U

    _mixed_rung_shard(tmp_path, "sums")
    code = U.main([str(tmp_path / "sums"), "--min-compared", "2"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED" not in out, out
    # 1 is the fixture's 1K digest regression, so the floor did not mask it.
    assert code == 1, out
    # Proves the count is 2, not the 3 a sum of the three reports would give.
    code = U.main([str(tmp_path / "sums"), "--min-compared", "3"])
    out = capsys.readouterr().out
    assert "TOO LITTLE COMPARED: 2 of 2" in out, out
    assert code == 3, out


def test_the_coverage_floor_is_checked_per_payload_pattern(tmp_path, capsys):
    """The coverage floor is per film, not pooled: one film must not carry another over it."""
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


def test_an_assertion_that_failed_on_one_arm_fails_the_windowed_verdict(tmp_path, capsys):
    """An assertion that fails on one arm only is a build difference and must fail the windowed verdict."""
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
    """Failing on both arms means the fixture cannot reach the state, not that the builds differ."""
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
    """An action that runs on one arm but cannot be performed on the other is a regression."""
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
    """A missed slot is machine speed, not a build difference: misses correlate through the runner."""
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
