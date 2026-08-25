# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""VISIBLE-REGION PARITY, held to the policy it exists to serve.

The policy: all changes preserve UI and UX idempotency, with three exemptions. A difference may be
accepted deliberately when performance improves dramatically; a difference that exists only OFF
SCREEN is fine by definition, because rendering only what is visible is an accepted technique; and
a select-all need not select all, PROVIDED the copy stays complete. Only the second is a question
this file can answer -- the third is scored behaviourally, on the clipboard.

The structural digest cannot express the second exemption -- it digests the thread on screen and
off, so every deferred-off-screen technique fails it by construction -- and answering NOT_APPLICABLE
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
    difference is off screen and is exempt. The structural digest fails this pair; this must
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
    """THIS TEST USED TO ASSERT MATCH, and it contradicted its own name.

    Ordinal 3 scrolled through the viewport during the action and had been unmounted again before
    the capture ran, so it cannot be digested. The old behaviour returned MATCH as long as one
    other message stayed mounted and left the residue in `not_digested`, which nothing printed:
    the run exited 0 under a claim that quantifies over EVERY message the viewport showed, while
    one of them had never been compared. A rendering difference in the missing message produced a
    clean pass.

    The residue is still reported, and the verdict is now the third outcome rather than the
    strongest one.
    """
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
    assert "thread-structure parity" in P.CLAIM_STRUCTURAL
    assert "NOTHING about how anything looks" in P.CLAIM_BEHAVIOURAL


def test_the_structural_claim_does_not_promise_a_reading_the_digest_cannot_take():
    """IT USED TO SAY "whole-document structural parity: every element in the DOM is identical on
    both arms", and that is false in a way that changes conclusions rather than wording.

    `scene/parity.js` digests the thread root plus a list of overlay selectors. It is sidebar-blind
    and layout-blind by construction and it never reads geometry or CSS custom properties. Measured:
    run against a real sidebar-drag change the thread digest returned 0 of 34 differing pairs, and
    its own null control returned 0 of 34 as well, so the instrument was not discriminating in
    either direction -- while the banner above the result said every element in the DOM was
    identical. Three purpose-built captures found the same change 34 of 34.
    """
    assert "whole-document" not in P.CLAIM_STRUCTURAL
    assert "every element in the DOM" not in P.CLAIM_STRUCTURAL
    assert "thread-structure parity" in P.CLAIM_STRUCTURAL
    # And it states what it does not cover, next to the claim rather than in a source comment.
    for surface in ("sidebar", "geometry", "CSS custom properties"):
        assert surface in P.CLAIM_STRUCTURAL, surface
    assert "0 of 34" in P.CLAIM_STRUCTURAL


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


def test_every_mode_names_the_policy_it_is_judging_against():
    """A BARE "PARITY OK" READS FAR STRONGER THAN ANY MODE CAN SUPPORT.

    Each mode already prints the CLAIM it is making. The claim says what was compared; it does not
    say what a pass is worth, and the three exemptions are exactly what decide that. So the policy
    is printed beside the claim, per mode, and this holds that every mode has one, that all three
    name all three exemptions, and that each says which of them it can grant.

    THE THIRD IS THE ONE A READER IS LIKELIEST TO BE MISSING, and it is the one with a condition
    attached: the copy must stay complete, only the visual fidelity of the selection is given up.
    A policy line that named the exemption without its condition would read as permission to lose
    conversation, so the condition is asserted alongside it.
    """
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
    # The behavioural mode grants neither of the first two and is the only one that speaks to the
    # third, so "either" would be the wrong word for it now.
    assert "cannot grant the performance or off-screen exemptions" in P.POLICY_BY_MODE["behaviour"]
    # AND IT SAYS HOW IT MEASURES THE CONDITION. "Complete" on its own reads as a comparison of
    # the copied content; what the gate does is divide each arm's clipboard length by the thread's
    # visible text and require the ratio to land in a band. Saying which of those it is decides
    # whether a reader is entitled to conclude the copy was intact, so the weaker wording is not
    # allowed back: the line has to name the measure AND disclaim the one it does not perform.
    assert "BY LENGTH" in P.POLICY_BY_MODE["behaviour"]
    assert "does not compare the copied characters" in P.POLICY_BY_MODE["behaviour"]
    assert "records the exemption rather than granting it" in P.POLICY_BY_MODE["behaviour"]
    # The floor survives the exemption. An exemption changes what counts as a pass; a measurement
    # with no floor under it is not a pass in the first place.
    assert "does not remove the floor" in P.POLICY_BY_MODE["visible"]


def test_the_policy_line_is_printed_next_to_every_claim_line():
    """A constant nothing prints is a constant nobody reads.

    Every needle below is DERIVED from the module under test rather than written out here: the
    claim names come from `vars(P)` and the mode names from `POLICY_BY_MODE` itself. Counting two
    hand-typed substrings instead could only ever report a total, so it said "3 claim lines but 2
    policy lines" without naming the mode that had gone quiet, and it broke the moment a policy
    line started being built by a function so that it could interpolate the band it enforces.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "sweep" / "ui_parity.py").read_text(
        encoding = "utf-8"
    )
    claims = sorted(name for name in vars(P) if name.startswith("CLAIM_"))
    assert len(claims) == 3, claims
    for name in claims:
        assert f"P.{name}" in source, f"{name} is never printed"
    for mode in P.POLICY_BY_MODE:
        # Either printed straight from the table, or through the per-mode helper that fills in
        # the numbers that mode is actually enforcing.
        assert (
            f"POLICY_BY_MODE['{mode}']" in source or f"{mode}_policy(" in source
        ), f"the {mode} policy line is never printed"


def test_the_mode_names_the_pull_request_template_uses_are_accepted():
    """THE TEMPLATE AND THE TOOL MUST AGREE ON WHAT THINGS ARE CALLED.

    The repository's pull request template asks for "the structural digest" and the report header
    prints "(STRUCTURAL MODE)", but the flag was spelled `--mode digest`, so a reader following
    either would type a word argparse rejected. `structural` is an alias for `digest`, not a
    fourth mode, and `behavior` for `behaviour` so the American spelling is not an error either.
    """
    from studiobench.sweep import ui_parity

    source = ui_parity.__file__
    with open(source, encoding = "utf-8") as handle:
        text = handle.read()
    for name in ("auto", "digest", "structural", "visible", "behaviour", "behavior"):
        assert f'"{name}"' in text, name
    assert '{"structural": "digest", "behavior": "behaviour"}' in text
