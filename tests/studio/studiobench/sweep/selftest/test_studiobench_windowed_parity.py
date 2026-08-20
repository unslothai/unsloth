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


def test_unequal_mounted_counts_are_refused_even_without_a_declared_total():
    """Two arms that mounted different numbers of messages cannot be compared row by row, because
    the rows are keyed by position in the MOUNTED list."""
    got = P.compare(_capture(18, 18, "b"), _capture(12, 12, "t"))
    assert got["verdict"] == P.NOT_APPLICABLE
    assert "different numbers of messages" in got["reason"]


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
    assert checks["clipboard_carries_the_whole_thread"]["ok"] is True
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
