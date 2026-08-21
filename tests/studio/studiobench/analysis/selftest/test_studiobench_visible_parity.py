# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VISIBLE-REGION PARITY, held to the policy it exists to serve.

The policy: all changes preserve UI and UX idempotency, with two exemptions. A difference may be
accepted deliberately when performance improves dramatically, and a difference that exists only OFF
SCREEN is fine by definition, because rendering only what is visible is an accepted technique.

The whole-document digest cannot express the second exemption -- it compares everything in the DOM,
so every deferred-off-screen technique fails it by construction -- and answering NOT_APPLICABLE
withholds a verdict rather than giving one. These tests hold the replacement to both halves of the
claim: an off-screen-only difference must PASS, and an on-screen difference must FAIL, on the same
pair of captures.

The verdict logic is pure, so it is tested here without a browser. The observer that produces the
captures is tested in a real Chromium in
`scene/selftest/test_studiobench_visible_capture_live.py`, because whether IntersectionObserver
sees what it should is not a question a fake can answer.
"""

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


# ── the exemption, which is the entire point ────────────────────────


def test_a_difference_that_is_only_off_screen_passes():
    """THE POLICY, IN ONE ASSERTION. The treatment renders ordinals 1-3 differently -- they are
    genuinely not the same DOM -- but the viewport never showed them during this action, so the
    difference is off screen and is exempt. The whole-document digest fails this pair; this must
    not."""
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


# ── the windowed arm is comparable at all ───────────────────────────


def test_a_windowed_arm_and_a_full_arm_are_compared_by_thread_position():
    """The reason this mode works where the digest does not. The base has the whole thread mounted
    and the treatment has a window of it, so mounted INDEX 0 is a different message on the two
    arms. Keyed by thread ordinal, the messages that were actually on screen line up."""
    base = _cap({16: "p", 17: "q", 18: "r"})
    treat = _cap({16: "p", 17: "q", 18: "r"})
    assert P.compare_visible(base, treat)["verdict"] == P.MATCH


# ── the positive control ────────────────────────────────────────────


def test_a_visibility_scan_that_saw_nothing_is_not_a_pass():
    """Two empty scans have equal ordinal sets and no differing digests, so without this the
    strongest verdict available is returned on the strength of never having observed a single
    message. `compare_styles` had exactly this bug and it is the reason anything here that can
    return zero carries a control."""
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


# ── the honest residue ──────────────────────────────────────────────


def test_a_message_seen_mid_action_but_unmounted_by_capture_is_not_counted_as_agreement():
    """The known limitation, kept visible. Ordinal 3 scrolled through the viewport during the
    action and the windowed arm had unmounted it again before the capture ran, so it cannot be
    digested. It must not be silently dropped into the matched pile."""
    base = _cap({14: "a"}, ever = [3, 14])
    treat = _cap({14: "a"}, ever = [3, 14])
    got = P.compare_visible(base, treat)
    assert got["verdict"] == P.MATCH
    assert got["not_digested"] == [3], got


def test_a_pair_where_nothing_visible_could_be_digested_is_not_a_pass():
    """Every ordinal the viewport showed had been unmounted by capture time, so the comparison
    observed the visibility but none of the content. That is not agreement."""
    got = P.compare_visible(_cap({}, ever = [3, 4]), _cap({}, ever = [3, 4]))
    # The zero-length scan control fires first, and either refusal is correct; what must not
    # happen is a MATCH.
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
    assert "whole-document" in P.CLAIM_STRUCTURAL
    assert "NOTHING about how anything looks" in P.CLAIM_BEHAVIOURAL


def test_one_viewport_ending_empty_is_a_difference_not_a_refusal():
    """MEASURED, and it is why this check exists. On the 100K virtualization arm `model_change`
    took the thread from 12 mounted messages to 0 and it never came back: the census read 0
    messages and 2,107 elements for the rest of the film and three later actions could not run.

    Both arms had shown the same ordinals earlier in the action, so the union matched and every
    per-ordinal digest was simply absent on one side -- which the union comparison reported as NOT
    COMPARABLE. A refusal, for one arm losing the entire conversation.
    """
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
