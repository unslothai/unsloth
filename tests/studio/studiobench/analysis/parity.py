# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Comparing two parity captures, and deciding what the comparison is allowed to mean.

`scene/parity.js` produces the reading; this decides what it says. The two are kept apart because
the reading is taken inside a browser and the decision has to be unit-testable without one.

THE RULE THIS FILE EXISTS TO ENFORCE. There are three outcomes of holding two captures side by
side, not two, and collapsing the third into either of the others is how an instrument starts
lying:

    MATCH            both arms were digested and the digests agree.
    DIFFER           both arms were digested and the digests disagree.
    NOT COMPARABLE   one arm has no digest, or the two were taken over different roots, or one
                     side's style probe was capped and the other's was not.

`NOT COMPARABLE` is never folded into `MATCH`. A capture that failed and a capture that matched
are the same absence of a complaint, and telling them apart is the entire difference between an
instrument and a decoration.
"""

from __future__ import annotations

import collections
from typing import Any, Iterable, Optional

# What a comparison concluded. Kept as plain strings so a payload row carries them verbatim.
#: WHAT "MATCH" ACTUALLY CLAIMS, because the name promises more than the instrument delivers.
#:
#: Everything in this module is computed from `scene/parity.js`, whose structural digest walks the
#: THREAD and nothing else. It is sidebar-blind and layout-blind by construction: a change confined
#: to the sidebar, or one that alters computed geometry without altering thread structure, passes
#: it while being entirely invisible to it. That is not a theory. A concurrent campaign measuring
#: the sidebar drag found the shipped thread digest returned 0 of 34 differing pairs on a real,
#: visible change -- and its null control returned 0 of 34 as well, so the instrument was not
#: discriminating in either direction. Three purpose-built captures (sidebar-inclusive structure,
#: sidebar inline style, custom-property reach) found the same change 34 of 34, with the null at
#: zero in every category.
#:
#: So MATCH means NO THREAD-STRUCTURE CHANGE WAS DETECTED. It does not mean the UI is unchanged.
#: Not covered, and not detectable here at all:
#:
#:   the sidebar                 outside the digest root
#:   computed layout / geometry  positions and sizes are never read
#:   CSS custom properties       never read
#:   stylesheet changes          only insofar as they alter `display`, `visibility` or
#:                               `pointer-events` on the <=64 elements the bounded style probe
#:                               reaches, which is reported separately and as an advisory
#:
#: Anything relying on a stronger reading than that needs its own capture.
MATCH = "match"
DIFFER = "differ"
NOT_COMPARABLE = "not_comparable"
NOT_EXERCISED = "not_exercised"
#: THE FOURTH OUTCOME, and it is not a softer NOT_COMPARABLE.
#:
#: NOT_COMPARABLE means the reading failed or the two readings are of different things by
#: accident. NOT_APPLICABLE means the question itself is wrong for this pair: the structural
#: digest asks "is the same DOM on screen", and an arm whose entire purpose is to put less DOM on
#: screen answers "no" by construction. Reporting that as a UI difference would be true and
#: useless -- eighteen red rows, none of them a finding, and the real behavioural questions
#: drowned underneath them.
#:
#: It is kept distinct from NOT_COMPARABLE so a reader can tell "we could not measure this" from
#: "this measurement does not apply here", and so `derive_unstable` counts neither as evidence.
NOT_APPLICABLE = "not_applicable"

# Actions whose rendered result legitimately differs between two runs of the SAME build, so a
# digest mismatch there carries no information about the pull request under test.
#
# A MECHANISM PER ENTRY, not a hunch, and the value is the mechanism rather than a comment beside
# it so that a test can require one. An action silenced without a stated reason is a hole somebody
# punched in the instrument and nobody can audit afterwards; the null control's empirically
# derived set (`derive_unstable`) is the cross-check that each of these earns its place.
UNSTABLE_ACTIONS: dict[str, str] = {
    "stop_generation": "stops a live stream, so how many characters arrived before the stop is a race with the "
    "network and differs run to run on one build. NOT reproduced by the 100K fast-tier null "
    "control (0 of 4), so this entry rests on the mechanism rather than on a measurement",
    "send_turn": "starts a new turn, so how far it has streamed when the digest is taken is wall clock. "
    "Measured differing on 2 of 4 base-vs-base pairs, with assistant_chars differing too",
    "scroll_during_generation": "the digest is taken after a scroll gesture against a growing thread; where it comes to "
    "rest depends on the autoscroll observer, which is the mechanism under study. Measured "
    "differing on 3 of 4 base-vs-base pairs",
    "scroll_after": "same gesture against a settled thread; the resting offset still depends on the observer. "
    "NOT reproduced by the 100K fast-tier null control (0 of 4); kept on the mechanism alone",
    # The four below were NOT in the hand-written set and were found by the null control, each
    # scoring as a hard UI-change signal on a build compared with itself. The mechanism for each
    # was read out of the payload, not guessed.
    "keystroke": "types characters into the composer over wall clock, and the composer's text is in the "
    "DOM, so how many keystrokes had landed by the capture deadline is a race. Measured 4 of "
    "4 differing, by 8 signature characters, with assistant_chars identical",
    "composer_fill": "fills the composer with a fixed string, but whether the send button has re-enabled by "
    "capture time is a race. A bare `disabled` serialises as exactly ten characters and the "
    "observed delta was exactly ten, in both directions across repetitions",
    "copy_markdown": "on the packed films this slot opens while the turn started by the preceding send_turn is "
    "still streaming, so the last assistant message is mid-tail. Measured 3 of 4 differing, "
    "with assistant_chars differing on every one",
    "message_menu": "same cause as copy_markdown: the slot lands inside the previous send_turn's stream. "
    "Measured 3 of 4 differing, with assistant_chars differing",
    "select_text": "same cause again, and the worst of the three at 4 of 4 differing; the selection is taken "
    "over a message whose tail is still arriving",
}


# WHOSE ABILITY TO RUN IS A RACE, which is a different claim from whose DIGEST varies and needs
# its own list. Every entry in UNSTABLE_ACTIONS above describes what makes the CAPTURE move --
# how many characters had arrived, where the scroll came to rest, whether the send button had
# re-enabled. Not one of them says the action sometimes cannot be performed at all. Using that
# list to excuse one-arm-only EXECUTION exempted nine of the sixteen scheduled actions from the
# one regression shape that leaves no digest to differ, on the strength of a measurement about
# something else.
#
# `slot_missed` already covers the runner arriving late, so what is left here is the narrow case
# of an action that legitimately cannot run because the stream it needs is not there. Read out of
# each action's own `not_run` reasons in `scene/actions.py` rather than assumed:
# `action -> (why, the not_run reasons that qualify)`. KEYED ON THE REASON, not just the action,
# because each of these has non-racy failure paths too: send_turn also returns not_run for "no
# composer on the page" (scene/actions.py:1016) and stop_generation for "the stop button is not
# present" (scene/actions.py:501). A treatment build that REMOVES either control records exactly
# the one-arm-only regression this instrument exists to catch, and keyed by name alone the
# exemption swallowed it. The mechanism was always written down here; it just was not what the
# code matched on.
RACY_EXECUTION: dict[str, tuple[str, tuple[str, ...]]] = {
    "stop_generation": (
        "needs a live stream to stop, and returns not_run when nothing was generating and a new "
        "turn did not start within 8s (scene/actions.py:493). Whether a stream was in flight on "
        "this arm at this moment is a race with the model, not a property of the build. Its "
        "OTHER not_run, the stop button being absent, is the build and is not exempt",
        ("nothing was generating",),
    ),
    "scroll_during_generation": (
        "returns not_run with 'nothing was generating, so this is not a scroll during "
        "generation' (scene/actions.py:334), and that is its only not_run path",
        ("nothing was generating",),
    ),
    "send_turn": (
        "returns not_run when 'a reply was still streaming, so this send would have been queued' "
        "(scene/actions.py:993), which is the previous turn's stream overrunning rather than "
        "this build being unable to send. Its composer-absent and queue-exhausted paths are not "
        "exempt",
        ("a reply was still streaming",),
    ),
}


def racy_execution(action: str, reason: str) -> bool:
    """Is THIS not-run a stream-timing race, rather than merely on a racy action?"""
    entry = RACY_EXECUTION.get(action)
    if entry is None:
        return False
    return any(marker in (reason or "") for marker in entry[1])


# The six that are NOT here, and why, since dropping an exemption needs as much justification as
# granting one. `composer_fill`, `keystroke`, `copy_markdown`, `message_menu` and `select_text`
# fail to run only when the control they need is absent or unresponsive -- "no composer on the
# page", "no Copy button on the last assistant message" -- and that IS the build. `scroll_after`
# has no `not_run` path at all, so a `ran: false` for it cannot be a race under any reading.


def _messages(capture: dict) -> dict[int, dict]:
    return {m["i"]: m for m in (capture.get("messages") or []) if "i" in m}


def _overlays(capture: dict) -> list[dict]:
    return list(capture.get("overlays") or [])


def windowed_mount(capture: Optional[dict]) -> bool:
    """Did this capture come from a thread that mounts a WINDOW rather than the whole thing?

    Read from the capture's own two numbers rather than from a flag the caller passed in, so an
    arm cannot be scored as full-mount because somebody forgot to set an option. Captures taken
    before those numbers existed report neither, and are treated as full-mount, which is what they
    were.
    """
    if not isinstance(capture, dict):
        return False
    mounted, total = capture.get("mounted_messages"), capture.get("thread_total")
    if not isinstance(mounted, int) or not isinstance(total, int):
        return False
    return total > mounted


def applicability(base: Optional[dict], treat: Optional[dict]) -> Optional[str]:
    """Why the STRUCTURAL DIGEST is the wrong question for this pair, or None if it is the right one.

    ONE shape, and it is an answer rather than a failure: an arm mounts a window of the thread on
    purpose, so the two digests describe different amounts of DOM by design and the comparison
    cannot tell the intended change from an unintended one.

    What replaces it is in `analysis/behaviour.py`: matched scroll positions plus the behavioural
    invariants that break first when a thread stops mounting everything.

    IT USED TO COVER A SECOND SHAPE and that was a false green. Two arms that both mount their
    whole thread, differing in how many messages that is, were also waved through here on the
    argument that the per-message rows are keyed by position and so describe different messages on
    the two sides. That argument is true and it is not a reason to withhold a verdict: if neither
    arm is windowing, a treatment that mounts fewer messages than the base has LOST MESSAGES, and
    that is the most serious thing this comparison could find. It is reported by
    `mount_count_mismatch` as a difference instead.
    """
    if not isinstance(base, dict) or not isinstance(treat, dict):
        return None
    bw, tw = windowed_mount(base), windowed_mount(treat)
    if bw or tw:
        which = "both arms" if bw and tw else ("the base arm" if bw else "the treatment arm")
        return (
            f"{which} mounts a WINDOW of the thread "
            f"(base {base.get('mounted_messages')} of {base.get('thread_total')}, treatment "
            f"{treat.get('mounted_messages')} of {treat.get('thread_total')} messages mounted). "
            "A structural DOM digest compares what is on screen, and this arm changes what is on "
            "screen by design, so the comparison cannot distinguish the intended change from an "
            "unintended one. Score this pair on behavioural invariants instead"
        )
    return None


def mount_count_mismatch(base: Optional[dict], treat: Optional[dict]) -> Optional[str]:
    """Two NON-windowed arms that mounted different numbers of messages, or None.

    Reached only when neither arm is windowing, so neither is holding anything back deliberately
    and the counts should be equal. They are not, which means one side is rendering a thread the
    other side is not -- a user-visible loss of conversation, reported as a difference rather than
    excused as an incomparable pair.

    The per-message digest rows are not quoted alongside it. They are keyed by position in the
    mounted list, so with different lengths no two rows describe the same message and every one of
    them would be listed as moved: a page of noise on top of the one finding that matters.
    """
    if not isinstance(base, dict) or not isinstance(treat, dict):
        return None
    bm, tm = base.get("mounted_messages"), treat.get("mounted_messages")
    if not isinstance(bm, int) or not isinstance(tm, int) or bm == tm:
        return None
    return (
        f"the arms mounted different numbers of messages ({bm} vs {tm}) and NEITHER arm is "
        "windowing, so this is not a difference of what is kept in the DOM on purpose -- one side "
        "is rendering messages the other is not. The per-message rows are keyed by position in "
        "the mounted list, so they are not quoted here: with different lengths every row would "
        "read as moved"
    )


def comparability(base: Optional[dict], treat: Optional[dict]) -> Optional[str]:
    """Why these two captures may NOT be compared, or None if they may be."""
    for label, side in (("base", base), ("treatment", treat)):
        if not isinstance(side, dict):
            return f"{label} recorded no parity capture at all"
        if not side.get("parity_attempted"):
            return (
                f"{label} could not be captured: " f"{side.get('reason') or 'no reason recorded'}"
            )
    assert base is not None and treat is not None
    # `root_kind` is absent in captures taken before it was recorded. Missing on BOTH sides is an
    # old payload and is allowed through; missing on one side means the two runs were produced by
    # two different versions of the instrument, which is not a comparison of two builds.
    bk, tk = base.get("root_kind"), treat.get("root_kind")
    if (bk is None) != (tk is None):
        return (
            "the two arms were captured by different versions of the parity instrument "
            f"(root_kind base={bk!r} treatment={tk!r})"
        )
    if bk is not None and bk != tk:
        return (
            f"the arms digested different roots (base={bk}, treatment={tk}); a body-root "
            "capture includes the sidebar and header and is not comparable with a thread one"
        )
    return None


def localise(base: dict, treat: dict) -> list[str]:
    """WHERE the two captures differ, as short human-readable claims.

    A whole-thread digest that differs is a fact nobody can act on. The per-message and per-overlay
    rows are what turn it into a bug report, and an unlocalised difference is reported as exactly
    that rather than left to look like a localised one.
    """
    moved: list[str] = []
    bm, tm = _messages(base), _messages(treat)
    for i in sorted(set(bm) | set(tm)):
        if i not in bm:
            moved.append(f"msg{i}({tm[i].get('role', '?')}):only treatment")
        elif i not in tm:
            moved.append(f"msg{i}({bm[i].get('role', '?')}):only base")
        elif bm[i].get("digest") != tm[i].get("digest"):
            moved.append(
                f"msg{i}({bm[i].get('role', '?')}):" f"{bm[i].get('chars')}->{tm[i].get('chars')}c"
            )

    bo, to = _overlays(base), _overlays(treat)
    if len(bo) != len(to):
        moved.append(f"overlays {len(bo)}->{len(to)}")
    else:
        for k, (b, t) in enumerate(zip(bo, to)):
            if b.get("digest") != t.get("digest"):
                moved.append(f"overlay{k}[{b.get('sel')}]:{b.get('chars')}->{t.get('chars')}c")

    if not moved:
        # The whole-thread digest moved but no message and no overlay did. That is the thread
        # scaffolding itself -- the viewport, the composer, the empty-state -- and saying so is
        # more useful than an empty list, which reads as "nothing differs".
        moved.append(
            f"thread scaffolding outside any message "
            f"({base.get('chars')}->{treat.get('chars')}c)"
        )
    return moved


def compare_styles(base: dict, treat: dict) -> tuple[str, str]:
    """The bounded computed-style probe, as its own verdict.

    Reported apart from the structural digest because it is the only reading here that can see a
    stylesheet change and, for the same reason, the one most likely to be caught mid-transition.
    """
    bs, ts = base.get("styles"), treat.get("styles")
    if not isinstance(bs, dict) or not isinstance(ts, dict):
        return NOT_COMPARABLE, "one or both arms carry no style probe"
    if bs.get("capped") or ts.get("capped"):
        return NOT_COMPARABLE, (
            f"the probe hit its element cap "
            f"(base={bs.get('elements')}, treatment={ts.get('elements')})"
        )
    # A POSITIVE CONTROL ON THE SCAN ITSELF. Two probes that matched no elements have equal
    # counts and equal digests -- both are the hash of an empty string -- so a probe that scanned
    # NOTHING reports MATCH, which is the strongest verdict this function can return and is
    # supported by no observation whatsoever. That is not hypothetical: the selector list is
    # written against Studio's markup, and a class rename anywhere in it silently empties the
    # scan. A DOM or CSSOM scan that can return zero has to be able to tell "I looked and they
    # agree" from "I did not look", and this one could not.
    if not bs.get("elements") or not ts.get("elements"):
        return NOT_COMPARABLE, (
            f"the style probe matched no elements (base={bs.get('elements')}, "
            f"treatment={ts.get('elements')}), so it observed nothing on at least one arm. Its "
            "selector list is written against Studio's markup and does not survive a rename; "
            "this is a probe that needs fixing, not two arms that agree"
        )
    if bs.get("elements") != ts.get("elements"):
        return DIFFER, (
            f"the probe matched a different number of elements "
            f"({bs.get('elements')} vs {ts.get('elements')})"
        )
    if bs.get("digest") == ts.get("digest"):
        return MATCH, ""
    return DIFFER, f"display/visibility/pointer-events differ over {bs.get('elements')} elements"


def _any_moved(base: dict, treat: dict) -> bool:
    """Did ANY digest in the capture move, not just the whole-thread one?

    THE HOLE THIS CLOSES, found by the spike control and not by reading the code. An overlay --
    an open menu, a dialog, the model picker -- lives OUTSIDE the thread root, so `parity.js`
    walks it separately and reports it separately. The comparison used to test only the
    whole-thread digest and then localise, which meant the overlay rows were unreachable: a menu
    that mounted when it should not, or one whose contents were rewritten, left the thread digest
    untouched and the check said MATCH. The overlay walk was carried in every payload and could
    never fire. Injecting a `role="menu"` element and watching the digest report a clean pass is
    what surfaced it.
    """
    if base.get("digest") != treat.get("digest"):
        return True
    bo, to = _overlays(base), _overlays(treat)
    if [(o.get("sel"), o.get("digest")) for o in bo] != [
        (o.get("sel"), o.get("digest")) for o in to
    ]:
        return True
    bm, tm = _messages(base), _messages(treat)
    if set(bm) != set(tm):
        return True
    return any(bm[i].get("digest") != tm[i].get("digest") for i in bm)


def compare(base: Optional[dict], treat: Optional[dict]) -> dict:
    """One base/treatment pair -> {verdict, reason, moved, style_verdict, style_reason}."""
    why = comparability(base, treat)
    if why is not None:
        return {
            "verdict": NOT_COMPARABLE,
            "reason": why,
            "moved": [],
            "style_verdict": NOT_COMPARABLE,
            "style_reason": why,
        }
    assert base is not None and treat is not None
    not_applicable = applicability(base, treat)
    if not_applicable is not None:
        return {
            "verdict": NOT_APPLICABLE,
            "reason": not_applicable,
            "moved": [],
            # The style probe goes with it. Its verdict is `elements` counts matching, and those
            # counts are element counts over `[data-role]` among other things, so it reports
            # DIFFER for exactly the same reason and with exactly as much meaning.
            "style_verdict": NOT_APPLICABLE,
            "style_reason": not_applicable,
        }
    lost = mount_count_mismatch(base, treat)
    if lost is not None:
        style_verdict, style_reason = compare_styles(base, treat)
        return {
            "verdict": DIFFER,
            "reason": lost,
            "moved": [],
            "style_verdict": style_verdict,
            "style_reason": style_reason,
        }
    style_verdict, style_reason = compare_styles(base, treat)
    if not _any_moved(base, treat):
        return {
            "verdict": MATCH,
            "reason": "",
            "moved": [],
            "style_verdict": style_verdict,
            "style_reason": style_reason,
        }
    return {
        "verdict": DIFFER,
        "reason": "",
        "moved": localise(base, treat),
        "style_verdict": style_verdict,
        "style_reason": style_reason,
    }


def compare_rows(base_row: Optional[dict], treat_row: Optional[dict]) -> dict:
    """`compare()`, gated on whether the action actually RAN on both arms.

    THE SECOND HOLE THE NULL CONTROL FOUND, and the more embarrassing one. Five of the eighteen
    actions did not run at all on either arm of a 100K fast-tier null control -- `image_upload`
    could not find a visible attachments button, `copy_markdown` found no Copy button, and so on.
    The digest was captured anyway, because the action window closes whether or not the action
    did anything, so the two arms agreed and `image_upload` was reported as a STABLE, MATCHING
    surface. It reads as "the attachment flow renders identically on both builds". What actually
    happened is that nobody opened it.

    A comparison of two pages that no action touched is a comparison of the thread's resting
    state. That is not worthless, but it is not coverage of the named action and must not be
    counted as it. So it gets its own verdict, and an action that never ran on either arm can no
    longer contribute a pass to anything.

    WHY AN ARM WENT IDLE IS PART OF THE READING, and `one_sided` carries it. An arm that could not
    PERFORM an action the other performed is a difference between the builds, which is the question
    this whole instrument asks: a control that no longer opens is exactly that shape, and it is the
    one regression class that produces no digest to differ. An arm that merely arrived after its
    slot closed is a slow machine and cannot move a verdict, whether one arm missed it or both.
    Only the first sets `one_sided`; the caller decides what to do with it.
    """
    ran: dict[str, bool] = {}
    for label, row in (("base", base_row), ("treatment", treat_row)):
        if not isinstance(row, dict):
            return {
                "verdict": NOT_COMPARABLE,
                "reason": f"the {label} arm has no row for this action",
                "moved": [],
                "one_sided": "",
                "style_verdict": NOT_COMPARABLE,
                "style_reason": "",
            }
        ran[label] = bool(row.get("ran"))
    if not all(ran.values()):
        # Named after the arm that DID run, so an empty string is the symmetric case.
        live = [label for label in ("base", "treatment") if ran[label]]
        label = "base" if not ran["base"] else "treatment"
        row = base_row if label == "base" else treat_row
        assert isinstance(row, dict)
        reason = row.get("reason") or "no reason recorded"
        # A MISSED SLOT IS NOT A BUILD DIFFERENCE, even when only one arm missed it, and this
        # distinction is the whole load-bearing part of `one_sided`. `ran=false` has two causes
        # that look identical in the count and mean opposite things. The runner arriving after
        # the slot closed says nothing about the build: the schedule is fixed, the budgets are
        # small, and a machine slow enough to miss a slot at 45700ms is slow enough to miss it
        # on the next repetition too, so corroboration does not separate them -- the misses are
        # correlated through the runner, not independent draws. Treating that as asymmetric
        # execution reds the job on machine speed and breaks the invariant the mutation study
        # established, that a missed slot cannot move this verdict. A precondition failure is
        # the opposite: `image_upload` records `slot_missed=false` with "no visible attachments
        # button", and a control that stops opening records exactly that. So the signal is
        # `slot_missed`, which the driver already writes, and not merely which arm went idle.
        missed_slot = bool(row.get("slot_missed"))
        detail = (
            f"the action did not run on the {label} arm ({reason}), so any "
            "agreement here is about a surface nothing touched"
        )
        if live and not missed_slot:
            detail = (
                f"the action RAN on the {live[0]} arm and could not be performed on the "
                f"{label} arm ({reason}), so the two builds did not behave the same way"
            )
        if missed_slot:
            live = []
        return {
            "verdict": NOT_EXERCISED,
            "moved": [],
            "one_sided": live[0] if live else "",
            # The idle arm's OWN not_run string, unwrapped. `reason` below is prose built for a
            # reader; the exemption has to match on what the action actually recorded.
            "idle_reason": reason,
            "reason": detail,
            "style_verdict": NOT_EXERCISED,
            "style_reason": "",
        }
    assert base_row is not None and treat_row is not None
    out = compare(base_row.get("parity"), treat_row.get("parity"))
    # AN ACTION THAT RAN AND FAILED ITS OWN ASSERTION IS NOT A COMPARISON OF WHAT IT NAMES, and
    # `ran` alone cannot see it. `stop_generation` returns `ran = True, expect_ok = stopped_ms is
    # not None`, so a head on which Stop no longer ends the stream records a row this layer reads
    # as a perfectly good observation. Every other reader already knows better --
    # `scoring.from_payload` drops such a row's timing and `report.payload` and `--assert-
    # liveness` both single it out -- and the parity layer was the one left reading only `ran`.
    #
    # CARRIED SEPARATELY FROM THE DIGEST, not folded into the verdict, and that is the whole point
    # of it. `stop_generation` is on the declared unstable list, so its DOM difference is excused;
    # applying that exemption to the assertion as well is applying a measurement of one quantity
    # (does this digest race) to a different one (did the button do its job). The two happen to
    # share an action name and nothing else. So the caller gets its own signal and decides.
    #
    # ONLY AN ASYMMETRY. `expect_ok is None` means the action asserts nothing, and both arms
    # failing means the fixture cannot reach the state on either build -- coverage lost, not a
    # difference between the builds. One arm asserting successfully while the other does not is
    # the two builds behaving differently, which is the question this instrument asks.
    failed = [
        label
        for label, row in (("base", base_row), ("treatment", treat_row))
        if row.get("expect_ok") is False
    ]
    passed = [
        label
        for label, row in (("base", base_row), ("treatment", treat_row))
        if row.get("expect_ok") is True
    ]
    out["expect_regressed"] = failed[0] if len(failed) == 1 and passed else ""
    if out["expect_regressed"]:
        _who = out["expect_regressed"]
        _row = base_row if _who == "base" else treat_row
        out["expect_reason"] = (
            f"the action ran on both arms and its own assertion failed on the {_who} arm "
            f"({_row.get('reason') or 'no reason recorded'}), so the two builds did not behave "
            "the same way"
        )
    else:
        out["expect_reason"] = ""
    return out


def derive_unstable(
    pairs: Iterable[tuple[str, dict]], *, min_observations: int = 2
) -> dict[str, dict]:
    """Which actions differ against THEMSELVES, measured from a base-vs-base null control.

    `pairs` is (action, compare() result) over a run whose two arms are the same build. Anything
    that differs there differs for a reason that has nothing to do with any pull request, so this
    is the empirical version of the hand-written UNSTABLE_ACTIONS list -- derived from a
    measurement instead of declared from memory.

    `min_observations` guards the obvious trap: one repetition of an action that happened to
    differ once is not evidence that it is unstable, and treating it as evidence would let a
    single flake permanently silence a real signal.
    """
    seen: dict[str, int] = collections.Counter()
    differ: dict[str, int] = collections.Counter()
    blind: dict[str, int] = collections.Counter()
    for action, result in pairs:
        if result.get("verdict") not in (MATCH, DIFFER):
            # Not comparable and not exercised are both "no reading", and neither may be counted
            # as an observation of stability. An action derived as stable from four pairs that
            # never ran would be permanently trusted on the strength of nothing.
            blind[action] += 1
            continue
        seen[action] += 1
        if result.get("verdict") == DIFFER:
            differ[action] += 1
    out: dict[str, dict] = {}
    for action in sorted(set(seen) | set(blind)):
        n, d = seen[action], differ[action]
        out[action] = {
            "observations": n,
            "differed": d,
            "not_comparable": blind[action],
            # Unstable only with enough observations to mean it. Below that the honest answer is
            # "not enough evidence", which is not the same as "stable" and is not reported as it.
            "unstable": bool(d and n >= min_observations),
            "undetermined": n < min_observations,
        }
    return out


def cross_check(derived: dict[str, dict], declared: Iterable[str]) -> dict[str, list[str]]:
    """Where the empirical instability and the hand-written list disagree.

    Both directions are findings and both are reported:
      `declared_stable_in_practice` an action the list calls unstable that the null control never
                                    saw differ. Its entry is costing real signal.
      `unstable_but_not_declared`   an action that differs against itself and is nonetheless being
                                    scored as a hard signal. Every mismatch it produces is noise.
    """
    declared = set(declared)
    unstable = {a for a, r in derived.items() if r["unstable"]}
    undetermined = {a for a, r in derived.items() if r["undetermined"]}
    return {
        "declared_stable_in_practice": sorted(
            a for a in declared if a in derived and a not in unstable and a not in undetermined
        ),
        "unstable_but_not_declared": sorted(unstable - declared),
        "declared_but_never_observed": sorted(a for a in declared if a not in derived),
        "undetermined": sorted(undetermined),
    }


def mutation_detected(before: dict, after: dict) -> dict:
    """Did a deliberately injected DOM change move the digest, and where?

    The spike control's decision function. Same shape as `compare`, and deliberately the SAME code
    path: a mutation harness that used its own comparison would prove that its own comparison
    works, which is not the question.
    """
    result = compare(before, after)
    result["detected"] = result["verdict"] == DIFFER
    return result


def summarise(results: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Tally of verdicts, with NOT_COMPARABLE counted as its own outcome rather than as a pass."""
    tally: dict[str, int] = collections.Counter()
    for r in results:
        tally[r.get("verdict", NOT_COMPARABLE)] += 1
    return dict(tally)


# ── visible-region parity ───────────────────────────────────────────
#
# THE POLICY. All changes must preserve UI and UX idempotency, with two exemptions: a difference
# may be accepted deliberately when performance improves dramatically, and a difference that exists
# only OFF SCREEN is fine by definition, because rendering only what is visible is an accepted
# technique rather than a parity violation.
#
# `compare()` above cannot express the second exemption. It digests the thread whether or not any
# of it is on screen, so deferred off-screen work fails it by construction: virtualization,
# deferred fence highlighting, content-visibility, lazy images. Returning NOT_APPLICABLE for those
# pairs withholds a verdict.
# This supplies one.

#: The claim each verdict makes, so a reader knows which of the three was actually checked. A bare
#: "PARITY OK" has meant three very different things in this file's history and the difference
#: between them is the difference between a strong result and a weak one.
#: NOT "whole-document structural parity: every element in the DOM is identical on both arms",
#: which is what this string used to say and which the instrument cannot support. `scene/parity.js`
#: digests the THREAD ROOT plus a list of overlay selectors; it never walks the sidebar, never reads
#: geometry and never reads CSS custom properties. Printing the stronger claim turns a limited
#: thread-structure reading into an experimental conclusion about the whole UI, which is how a
#: sidebar-drag campaign came to be scored 0 of 34 differing pairs by a digest whose own null
#: control also returned 0 of 34: the instrument was not discriminating in either direction, and
#: the banner said the DOM was identical.
CLAIM_STRUCTURAL = (
    "thread-structure parity: the thread root and the declared overlay selectors serialise "
    "identically on both arms, on screen and off. It does NOT cover the sidebar, computed layout "
    "or geometry, or CSS custom properties, so it is not a statement that the UI is unchanged: "
    "run against a real sidebar-drag change this digest reported 0 of 34 differing pairs, and so "
    "did its own null control"
)
CLAIM_VISIBLE = (
    "visible-region parity: every message the viewport showed at any point during the action is "
    "present on both arms and identical between them, and every difference lies off screen"
)
CLAIM_BEHAVIOURAL = (
    "behavioural parity: the scroll extent matches and the invariants a windowed mount breaks "
    "first still hold. Says NOTHING about how anything looks"
)

#: THE POLICY EVERY VERDICT IS JUDGED AGAINST, printed beside the claim.
#:
#: A bare "PARITY OK" reads as "the UI is unchanged" and none of the three modes can support that
#: sentence. Each supports a narrower one, and which one is being made changes what a pass MEANS,
#: so the policy is named next to the claim rather than left in a document somebody has to find.
#:
#: The exemptions are not loopholes. The first is a decision someone makes on the record with a
#: number attached; the second is a definition, because rendering only what is visible is an
#: accepted technique rather than a parity violation. Neither removes the need for a floor: a
#: difference that is exempt still has to be shown to be the difference you think it is, which is
#: what the null control is for.
POLICY = (
    "UI and UX idempotency is required, with two exemptions: a deliberate difference accepted "
    "for a dramatic performance improvement, and a difference that exists only OFF SCREEN"
)

#: What each mode can and cannot decide under that policy. The point of the second half of each
#: line is that a reader can tell which sentence they are being handed.
POLICY_BY_MODE = {
    "structural": (
        f"{POLICY}. This mode judges the FIRST requirement over the thread root and the declared "
        "overlays only. It cannot grant the off-screen exemption, because it does not know what "
        "was on screen: an arm that renders only the visible region reads as a difference here "
        "and needs --mode visible to get a verdict rather than a refusal"
    ),
    "visible": (
        f"{POLICY}. This mode is the one that can GRANT the off-screen exemption: it compares "
        "only what the viewport actually showed, so an off-screen-only difference passes on its "
        "merits rather than being argued about. The exemption changes what counts as a pass; it "
        "does not remove the floor, so read the null control printed on the same scale"
    ),
    "behaviour": (
        f"{POLICY}. This mode judges neither requirement directly. It checks invariants a "
        "windowed mount breaks first -- scroll extent, what the clipboard carries, whether the "
        "thread survives being reopened -- so a pass here is evidence about BEHAVIOUR and says "
        "nothing about appearance. It cannot grant either exemption"
    ),
}


def compare_visible(base: Optional[dict], treat: Optional[dict]) -> dict:
    """One base/treatment pair, scored on the VISIBLE REGION only.

    Returns {verdict, reason, moved, claim}. `moved` names the ordinals that differ, in thread
    position rather than mounted index, so the row is actionable on a windowed arm.
    """
    for label, side in (("base", base), ("treatment", treat)):
        if not isinstance(side, dict) or not side.get("visible_attempted"):
            why = (side or {}).get("reason") or "no visible-region capture"
            return {
                "verdict": NOT_COMPARABLE,
                "reason": f"the {label} arm produced no visible-region capture: {why}",
                "moved": [],
                "claim": CLAIM_VISIBLE,
            }
    assert base is not None and treat is not None

    # THE POSITIVE CONTROL. A visibility scan that matched nothing has equal (empty) ordinal sets
    # and no differing digests, so without this it returns the strongest verdict available on the
    # strength of never having seen a single message. Exactly the failure `compare_styles` had.
    bn, tn = len(base.get("ever_visible") or []), len(treat.get("ever_visible") or [])
    if bn == 0 or tn == 0:
        return {
            "verdict": NOT_COMPARABLE,
            "reason": (
                f"the visibility scan matched no messages (base={bn}, treatment={tn}). Nothing "
                "was observed on at least one arm, so there is nothing to agree about: this is a "
                "probe that needs fixing, not two arms that match"
            ),
            "moved": [],
            "claim": CLAIM_VISIBLE,
        }

    bev, tev = set(base.get("ever_visible") or []), set(treat.get("ever_visible") or [])
    if bev != tev:
        only_b, only_t = sorted(bev - tev), sorted(tev - bev)
        return {
            "verdict": DIFFER,
            "reason": (
                "the two arms put DIFFERENT MESSAGES on screen during this action, which is a "
                "visible difference and not an off-screen one "
                f"(only base: {only_b[:8]}, only treatment: {only_t[:8]})"
            ),
            "moved": [f"ordinal {o}" for o in (only_b + only_t)[:8]],
            "claim": CLAIM_VISIBLE,
        }

    bmsg, tmsg = base.get("messages") or {}, treat.get("messages") or {}
    # Ordinals seen during the window but unmounted by capture time cannot be digested. Reported,
    # never counted as agreement: this is the residue of comparing a windowed arm at one instant.
    uncomparable = sorted(int(o) for o in map(str, sorted(bev)) if o not in bmsg or o not in tmsg)
    moved = []
    for ordinal in sorted(bev):
        key = str(ordinal)
        b, t = bmsg.get(key), tmsg.get(key)
        if b is None or t is None:
            continue
        if b.get("digest") != t.get("digest"):
            moved.append(f"ordinal {ordinal}({b.get('role')}):{b.get('chars')}->{t.get('chars')}c")
    # ONE VIEWPORT ENDED EMPTY AND THE OTHER DID NOT, which is as visible a difference as there is
    # and used to be reported as NOT COMPARABLE -- a refusal, not a finding.
    #
    # This is not hypothetical and it is why the check exists. On the 100K virtualization arm,
    # `model_change` took the thread from 12 mounted messages to 0 and it never came back: the
    # census reads 0 messages and 2,107 elements for the rest of the film, and three later actions
    # could not run at all. Both arms had shown the same ordinals EARLIER in the action, so the
    # union matched and every per-ordinal digest was simply missing on one side. Comparing only the
    # union cannot see that; comparing what each arm could still show at the end can.
    b_left, t_left = len(bmsg), len(tmsg)
    if (b_left == 0) != (t_left == 0):
        empty, full = ("treatment", "base") if t_left == 0 else ("base", "treatment")
        return {
            "verdict": DIFFER,
            "reason": (
                f"the {empty} arm's viewport ended this action EMPTY while the {full} arm still "
                f"showed {max(b_left, t_left)} of the {len(bev)} message(s) that had been visible "
                "during it. Both arms displayed the same messages earlier in the action, so this "
                "is not a windowing difference: one arm lost the thread"
            ),
            "moved": [f"ordinal {o}" for o in sorted(bev)[:8]],
            "claim": CLAIM_VISIBLE,
            # NOT SUPPRESSIBLE BY THE NOISE FLOOR. The floor exists for actions whose visible
            # region differs between two runs of the same build, which is a volatile attribute on
            # rows of identical length. Losing the entire thread is a different kind of statement,
            # and an action can easily be BOTH: on the 100K run `model_change` is in the derived
            # unstable set because the null control's copy of it differs on an attribute, and that
            # would have silenced "the treatment arm's viewport ended empty" -- suppressing a lost
            # conversation on the strength of unrelated jitter in the same action.
            "severe": True,
        }
    if moved:
        return {
            "verdict": DIFFER,
            "reason": f"{len(moved)} visible message(s) rendered differently",
            "moved": moved,
            "claim": CLAIM_VISIBLE,
            "not_digested": uncomparable,
        }
    if uncomparable:
        # ANY residue refuses the verdict, not only a total one. This branch used to be entered
        # solely when EVERY visible ordinal was undigestable (`len(uncomparable) == len(bev)`), so a
        # scrolling or windowed action that put six messages on screen and unmounted one of them
        # before the capture returned MATCH on the strength of the other five. The claim printed
        # above quantifies over EVERY message the viewport showed; one message missing makes it
        # unsupported, and the rendering difference this mode exists to catch could be in exactly
        # the message that is missing. `visible_report` never printed the `not_digested` residue
        # either, so that pair exited 0 with nothing on screen to say a message went uncompared.
        #
        # WHY THE VERDICT AND NOT MERELY THE PASS COUNT. Demoting it inside `visible_report` alone
        # would leave the row itself reading `match`, and every other consumer of that row counts a
        # match as agreement: `visible_unstable_set` reads verdicts to derive the noise floor,
        # `summarise` tallies them, and the payload keeps them verbatim for whoever reads it next.
        # The honest outcome has to live in the verdict, which is the one thing they all share.
        #
        # THE COST, MEASURED on the two real 100K films rather than guessed. On the windowed arm 4
        # of 39 matching pairs carry a residue, all of them `stop_generation`; on the base-vs-base
        # null control 7 of 43 do. The mode keeps a discriminating majority on both, so this buys
        # honesty at four pairs in sixty-four.
        digested = len(bev) - len(uncomparable)
        return {
            "verdict": NOT_COMPARABLE,
            "reason": (
                f"{len(uncomparable)} of the {len(bev)} message(s) that were visible during this "
                "action had been unmounted again by the time the capture ran, so they could not be "
                f"digested (ordinals {uncomparable[:8]}). The {digested} that could be digested "
                "agreed, which is not the claim this mode makes"
            ),
            "moved": [],
            "claim": CLAIM_VISIBLE,
            "not_digested": uncomparable,
        }
    return {
        "verdict": MATCH,
        "reason": "",
        "moved": [],
        "claim": CLAIM_VISIBLE,
        "not_digested": [],
    }
