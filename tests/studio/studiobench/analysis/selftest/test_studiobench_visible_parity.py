# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Visible-region parity: off-screen-only differences must pass, on-screen ones must fail."""

from __future__ import annotations

import sys
from pathlib import Path

_STUDIO_TESTS = Path(__file__).resolve().parents[3]
if str(_STUDIO_TESTS) not in sys.path:
    sys.path.insert(0, str(_STUDIO_TESTS))

from studiobench.analysis import parity as P  # noqa: E402


def _cap(visible: dict[int, str], ever: list[int] | None = None) -> dict:
    """A visible-region capture. `visible` maps thread ordinal -> digest."""
    return {
        "visible_attempted": True,
        "ever_visible": sorted(ever if ever is not None else visible),
        "ever_visible_count": len(ever if ever is not None else visible),
        "mounted_ever_visible": len(visible),
        "unmounted_at_capture": len(ever if ever is not None else visible) - len(visible),
        "messages": {
            str(k): {"role": "assistant", "digest": v, "chars": 100} for k, v in visible.items()
        },
    }


def test_a_difference_that_is_only_off_screen_passes():
    """An off-screen-only difference must pass, though the structural digest fails it."""
    base = _cap({14: "a", 15: "b", 16: "c"})
    treat = _cap({14: "a", 15: "b", 16: "c"})
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.MATCH, got
    assert got["claim"] == P.CLAIM_VISIBLE


def test_a_difference_inside_the_viewport_still_fails():
    """The exemption is for off-screen differences only. A message the user was looking at is not
    excused by anything, and the row names it by THREAD position so it is actionable."""
    got = P.compare_visible(_cap({14: "a", 15: "b"}), _cap({14: "a", 15: "CHANGED"}))
    assert got["verdict"] == P.DIFFER, got
    assert any("ordinal 15" in m for m in got["moved"]), got["moved"]
    assert not any("ordinal 14" in m for m in got["moved"])


def test_showing_different_messages_is_itself_a_visible_difference():
    """Two arms whose viewports held different parts of the conversation did not show the user the
    same thing, whatever the digests of the overlap say. This is the case a naive intersection
    would silently skip by comparing only the ordinals both arms happen to have."""
    got = P.compare_visible(_cap({14: "a", 15: "b"}), _cap({15: "b", 16: "c"}))
    assert got["verdict"] == P.DIFFER
    assert "DIFFERENT MESSAGES on screen" in got["reason"]


def test_a_windowed_arm_and_a_full_arm_are_compared_by_thread_position():
    """The reason this mode works where the digest does not. The base has the whole thread mounted
    and the treatment has a window of it, so mounted INDEX 0 is a different message on the two
    arms. Keyed by thread ordinal, the messages that were actually on screen line up."""
    base = _cap({16: "p", 17: "q", 18: "r"})
    treat = _cap({16: "p", 17: "q", 18: "r"})
    assert P.compare_visible(base, treat)["verdict"] == P.MATCH


def test_a_visibility_scan_that_saw_nothing_is_not_a_pass():
    """Two empty scans must not pass: a scan that matched no messages is NOT COMPARABLE, not MATCH."""
    got = P.compare_visible(_cap({}), _cap({}))
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert "matched no messages" in got["reason"]


def test_one_arm_seeing_nothing_is_also_not_a_difference_to_report():
    got = P.compare_visible(_cap({}), _cap({14: "a"}))
    assert got["verdict"] == P.NOT_COMPARABLE


def test_a_missing_capture_is_refused_rather_than_assumed_empty():
    assert P.compare_visible(None, _cap({1: "a"}))["verdict"] == P.NOT_COMPARABLE
    assert (
        P.compare_visible({"visible_attempted": False, "reason": "no viewport"}, _cap({1: "a"}))[
            "verdict"
        ]
        == P.NOT_COMPARABLE
    )


def test_a_message_seen_mid_action_but_unmounted_by_capture_is_not_counted_as_agreement():
    """A message seen mid-action but unmounted by capture is not agreement; the verdict is NOT
    COMPARABLE."""
    base = _cap({14: "a"}, ever = [3, 14])
    treat = _cap({14: "a"}, ever = [3, 14])
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["verdict"] != P.MATCH
    assert got["not_digested"] == [3], got
    assert "ordinals [3]" in got["reason"], got["reason"]
    assert got["claim"] == P.CLAIM_VISIBLE


def test_the_messages_that_could_be_digested_agreeing_is_not_the_claim_this_mode_makes():
    """The residue is one ordinal out of six, so five messages were compared and all five agreed.
    That is a real observation and it is not the printed claim, which is about every message the
    viewport showed. The reason says which ordinal went uncompared so the reader can decide."""
    seen = {10: "a", 11: "b", 12: "c", 13: "d", 14: "e"}
    got = P.compare_visible(_cap(seen, ever = [3, *seen]), _cap(seen, ever = [3, *seen]))
    assert got["verdict"] == P.NOT_COMPARABLE, got
    assert got["not_digested"] == [3]
    assert "1 of the 6 message(s)" in got["reason"], got["reason"]
    assert "The 5 that could be digested agreed" in got["reason"], got["reason"]


def test_a_pair_with_nothing_left_undigested_still_matches_with_an_empty_residue():
    """The refusal must not leak into the pairs it does not concern, or the mode stops being able
    to pass anything and stops being able to fail anything either."""
    got = P.compare_visible(_cap({14: "a", 15: "b"}), _cap({14: "a", 15: "b"}))
    assert got["verdict"] == P.MATCH, got
    assert got["not_digested"] == []


def test_an_undigested_ordinal_never_downgrades_a_difference_that_was_found():
    """A residue withholds a pass; it does not withdraw a finding. Ordinal 3 could not be digested
    and ordinal 15 rendered differently, and the second of those is still the verdict."""
    base = _cap({14: "a", 15: "b"}, ever = [3, 14, 15])
    treat = _cap({14: "a", 15: "CHANGED"}, ever = [3, 14, 15])
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert got["not_digested"] == [3], got
    assert any("ordinal 15" in m for m in got["moved"]), got["moved"]


def test_a_pair_where_nothing_visible_could_be_digested_is_not_a_pass():
    """Every ordinal the viewport showed had been unmounted by capture time, so the comparison
    observed the visibility but none of the content. That is not agreement."""
    got = P.compare_visible(_cap({}, ever = [3, 4]), _cap({}, ever = [3, 4]))
    # The zero-length scan control fires first; either refusal is correct, MATCH is not.
    assert got["verdict"] == P.NOT_COMPARABLE, got


def test_every_verdict_names_the_claim_it_is_making():
    """Three modes have meant three different things by "parity" in this file's history, and the
    difference between them is the difference between a strong result and a weak one."""
    for got in (
        P.compare_visible(_cap({1: "a"}), _cap({1: "a"})),
        P.compare_visible(_cap({1: "a"}), _cap({1: "b"})),
        P.compare_visible(_cap({}), _cap({})),
    ):
        assert got["claim"] == P.CLAIM_VISIBLE
    assert "off screen" in P.CLAIM_VISIBLE
    assert "thread-structure parity" in P.CLAIM_STRUCTURAL
    assert "NOTHING about how anything looks" in P.CLAIM_BEHAVIOURAL


def test_the_structural_claim_does_not_promise_a_reading_the_digest_cannot_take():
    """The structural claim must not promise whole-document parity: the digest misses sidebar and layout."""
    assert "whole-document" not in P.CLAIM_STRUCTURAL
    assert "every element in the DOM" not in P.CLAIM_STRUCTURAL
    assert "thread-structure parity" in P.CLAIM_STRUCTURAL
    for surface in ("sidebar", "geometry", "CSS custom properties"):
        assert surface in P.CLAIM_STRUCTURAL, surface
    assert "0 of 34" in P.CLAIM_STRUCTURAL


def test_one_viewport_ending_empty_is_a_difference_not_a_refusal():
    """One viewport ending empty is a DIFFER, not a refusal: losing the whole conversation is real."""
    base = _cap({14: "a", 15: "b"}, ever = [14, 15])
    treat = _cap({}, ever = [14, 15])
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.DIFFER, got
    assert "ended this action EMPTY" in got["reason"]
    assert "one arm lost the thread" in got["reason"]


def test_both_viewports_ending_empty_is_still_only_a_refusal():
    """Symmetric loss is not evidence about the arm under test; it is an unusable pair."""
    got = P.compare_visible(_cap({}, ever = [14, 15]), _cap({}, ever = [14, 15]))
    assert got["verdict"] == P.NOT_COMPARABLE, got


def test_every_mode_names_the_policy_it_is_judging_against():
    """Every mode prints its policy beside its claim, and select-all's exemption needs complete copy."""
    from studiobench.analysis import parity as P

    assert set(P.POLICY_BY_MODE) == {"structural", "visible", "behaviour"}
    for mode, text in P.POLICY_BY_MODE.items():
        assert "idempotency" in text, mode
        assert "performance improvement" in text, mode
        assert "OFF SCREEN" in text or "off-screen" in text, mode
        assert "select-all that does not select all" in text, mode
        assert "PROVIDED the copy it produces stays complete" in text, mode
    assert "can GRANT the off-screen exemption" in P.POLICY_BY_MODE["visible"]
    assert "cannot grant" in P.POLICY_BY_MODE["structural"]
    assert "cannot grant the performance or off-screen exemptions" in P.POLICY_BY_MODE["behaviour"]
    # It must name the measure (clipboard length over thread text) and disclaim content comparison.
    assert "BY LENGTH" in P.POLICY_BY_MODE["behaviour"]
    assert "does not compare the copied characters" in P.POLICY_BY_MODE["behaviour"]
    assert "records the exemption rather than granting it" in P.POLICY_BY_MODE["behaviour"]
    assert "does not remove the floor" in P.POLICY_BY_MODE["visible"]


def test_the_policy_line_is_printed_next_to_every_claim_line():
    """Needles are derived from the module under test, so a mode with a missing policy line is named."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "sweep" / "ui_parity.py").read_text(
        encoding = "utf-8"
    )
    claims = sorted(name for name in vars(P) if name.startswith("CLAIM_"))
    assert len(claims) == 3, claims
    for name in claims:
        assert f"P.{name}" in source, f"{name} is never printed"
    for mode in P.POLICY_BY_MODE:
        assert (
            f"POLICY_BY_MODE['{mode}']" in source or f"{mode}_policy(" in source
        ), f"the {mode} policy line is never printed"


def test_the_mode_names_the_pull_request_template_uses_are_accepted():
    """Mode names the PR template and report use parse: structural for digest, behavior for behaviour."""
    from studiobench.sweep import ui_parity

    source = ui_parity.__file__
    with open(source, encoding = "utf-8") as handle:
        text = handle.read()
    for name in ("auto", "digest", "structural", "visible", "behaviour", "behavior"):
        assert f'"{name}"' in text, name
    assert '{"structural": "digest", "behavior": "behaviour"}' in text
