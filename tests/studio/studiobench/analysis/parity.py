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


def in_flight(base: Optional[dict], treat: Optional[dict]) -> set[int]:
    """Mounted indices of the messages that were still being WRITTEN on either arm.

    THE UNION, not the intersection, and that is the whole point. The ordinary case is precisely
    that one arm has finished the reply and the other has not: the two arms are two cells run back
    to back against one pacer, each with its own send click and its own paint clock, so the digest
    lands at two different points in the same stream. A message that is in flight on EITHER side
    has no defined moment on that side, and differencing it measures wall clock.

    Read from the capture rather than inferred. `scene/parity.js` takes it from the app's own
    published state -- assistant-ui's `data-status` on the text part, `aria-busy` on the reasoning
    content -- so this is the page's statement about itself and not the benchmark's guess.

    A capture recorded before the field existed reports nothing, and is treated as nothing in
    flight, which is what every such payload was scored as when it was written.
    """
    out: set[int] = set()
    for side in (base, treat):
        if isinstance(side, dict):
            for i in side.get("in_flight") or []:
                if isinstance(i, int):
                    out.add(i)
    return out


def _scaffold(capture: dict) -> tuple[Optional[str], Optional[int]]:
    """The thread with every message elided, falling back to the whole-thread digest.

    `digest` is this plus the per-message rows, so comparing the scaffold and the rows separately
    is the same reading taken apart -- and taken apart it can withhold one message. The fallback is
    what keeps an old payload comparable: a capture recorded before the scaffold existed carries
    `digest` alone, and it also carries no `in_flight`, so for those two this IS the old comparison.
    """
    if "digest_scaffold" in capture:
        return capture.get("digest_scaffold"), capture.get("chars_scaffold")
    return capture.get("digest"), capture.get("chars")


def _overlays(capture: dict) -> list[dict]:
    return list(capture.get("overlays") or [])


def overlays_moved(base: dict, treat: dict) -> list[str]:
    """The overlay rows that differ, as claims. Empty when the two arms agree about every overlay.

    THE ONE SURFACE IN THE STRUCTURAL CAPTURE THAT A STREAM CANNOT REACH. An overlay -- an open
    dialog, a menu, the model picker -- is walked from `document`, outside `.aui-thread-root`
    entirely, so its digest carries neither the streamed message nor the composer that reflects
    whether a reply is running. That makes it readable on a pair whose stream could not be placed,
    which is why it is factored out of `localise` and consulted before the refusal in `compare`.
    """
    bo, to = _overlays(base), _overlays(treat)
    if len(bo) != len(to):
        return [f"overlays {len(bo)}->{len(to)}"]
    return [
        f"overlay{k}[{b.get('sel')}]:{b.get('chars')}->{t.get('chars')}c"
        for k, (b, t) in enumerate(zip(bo, to))
        if b.get("digest") != t.get("digest")
    ]


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


def streaming_probe(base: dict, treat: dict) -> Optional[str]:
    """Why the STREAMING PROBE could not be believed on this pair, or None.

    THE POSITIVE CONTROL ON A SCAN THAT CAN RETURN ZERO. The elision below is only as good as its
    ability to find the streamed message, and `streamingMessages()` walks selectors written against
    Studio's markup: rename `data-status` and it goes quiet, matches nothing, and every capture then
    reports the strongest thing it can say about the stream -- that there was none -- on the
    strength of never having looked successfully. That is exactly the shape `compare_styles`
    already refuses for the style probe and `compare_visible` for the visibility scan.

    The app publishes the fact twice, through the Stop button and through `data-status`, and
    `capture()` carries the disagreement out of the page rather than resolving it there.

    NOT part of `comparability`, and the ordering is deliberate. This refusal is about the stream
    split only; a pair whose arms mounted different NUMBERS of messages is a lost conversation and
    is reported as a difference whether or not the stream could be placed, because that reading
    does not depend on the split at all. Put here, after `mount_count_mismatch`, a build that drops
    a message while a reply is running is still a finding rather than a shrug.
    """
    for label, side in (("base", base), ("treatment", treat)):
        if isinstance(side, dict) and side.get("in_flight_unplaced"):
            return (
                f"a reply was running on the {label} arm and no message published a streaming "
                "state, so the streamed message could not be identified and its digest cannot be "
                "held apart from the settled thread. This is a probe that needs fixing (the "
                "`data-status` / `aria-busy` hooks in scene/dom.js), not a settled thread"
            )
    return None


def localise(
    base: dict,
    treat: dict,
    skip: Optional[set[int]] = None,
) -> list[str]:
    """WHERE the two captures differ, as short human-readable claims.

    A whole-thread digest that differs is a fact nobody can act on. The per-message and per-overlay
    rows are what turn it into a bug report, and an unlocalised difference is reported as exactly
    that rather than left to look like a localised one.

    `skip` names the messages that were still being written on one arm or the other. They are left
    out of the list rather than quoted in it, because a mid-stream digest names a point in a stream
    and not a rendering, and a reader handed `msg22(assistant):17334->17358c` has no way to tell the
    two apart. What they were is reported separately, by the caller, as a residue.
    """
    skip = skip or set()
    moved: list[str] = []
    bm, tm = _messages(base), _messages(treat)
    for i in sorted(set(bm) | set(tm)):
        if i in skip and i in bm and i in tm:
            # In flight on BOTH sides of the comparison. A message present on one arm only is a
            # different statement -- one arm is not rendering a message the other is -- and is
            # reported below whatever its status.
            continue
        if i not in bm:
            moved.append(f"msg{i}({tm[i].get('role', '?')}):only treatment")
        elif i not in tm:
            moved.append(f"msg{i}({bm[i].get('role', '?')}):only base")
        elif bm[i].get("digest") != tm[i].get("digest"):
            moved.append(
                f"msg{i}({bm[i].get('role', '?')}):" f"{bm[i].get('chars')}->{tm[i].get('chars')}c"
            )

    moved.extend(overlays_moved(base, treat))

    if not moved:
        # The whole-thread digest moved but no message and no overlay did. That is the thread
        # scaffolding itself -- the viewport, the composer, the empty-state -- and saying so is
        # more useful than an empty list, which reads as "nothing differs".
        bc, tc = _scaffold(base)[1], _scaffold(treat)[1]
        moved.append(f"thread scaffolding outside any message ({bc}->{tc}c)")
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
    # THE RUN-STATE CONTROL IS IN THE SELECTOR LIST, so this reading straddles the same composer
    # swap the structural refusals below are about, and it has to be elided for the same reason.
    # `STYLE_SELECTORS` (scene/parity.js) names `button[aria-label="Send message"]` and
    # `button[aria-label="Stop generating"]`, and the signature carries the SELECTOR THAT MATCHED
    # (`parts.push(sel + "#" + n)`) -- so an arm still generating and a settled arm walk two
    # different entries for the one composer button. Measured on two byte-identical fixture threads
    # differing only in that control: Send against Stop holds all three properties at
    # `inline-block` / `visible` / `auto` on both arms and still moves the digest, and Send against
    # either queue control -- neither of which is in the selector list at all -- moves the element
    # COUNT from 7 to 6. Both come back DIFFER over identical CSS, and `sweep/ui_parity.report`
    # prints that as a style-regression advisory for precisely the ordinary cross-stream-state
    # timing the refusals exist to withhold.
    #
    # THE SAME CORROBORATION THE SCAFFOLD SUPPRESSION REQUIRES, and for the same reason: a
    # differing `composer_control` alone is what a treatment that DROPS or renames the control
    # produces, and that is a rendering regression this probe should still report. `streaming` and
    # `queued_idle` are read off the thread's run state rather than off the composer, so they are
    # evidence the composer cannot manufacture.
    #
    # NOT_COMPARABLE AND NOT MATCH. The probe reads ONE aggregate digest over up to `STYLE_CAP`
    # elements, so a genuine CSS difference elsewhere on the page is inside the same number and
    # cannot be separated from the swap here. This withholds the reading; it does not pass it.
    if (bs.get("elements") != ts.get("elements") or bs.get("digest") != ts.get("digest")) and (
        generation_disagrees(base, treat) and _run_state_disagrees(base, treat)
    ):
        return NOT_COMPARABLE, (
            "the two arms rendered different composer run-state controls "
            f"({base.get('composer_control')!r} against {treat.get('composer_control')!r}) and the "
            "run state itself says why, so the style probe walked a different selector for that "
            "one button -- its signature carries the selector that matched. The difference is the "
            "control swap and not a stylesheet change. The digest is ONE aggregate over every "
            "matched element, so a real CSS difference elsewhere cannot be separated from it here; "
            "this is a reading withheld, not a pass"
        )
    if bs.get("elements") != ts.get("elements"):
        return DIFFER, (
            f"the probe matched a different number of elements "
            f"({bs.get('elements')} vs {ts.get('elements')})"
        )
    if bs.get("digest") == ts.get("digest"):
        return MATCH, ""
    return DIFFER, f"display/visibility/pointer-events differ over {bs.get('elements')} elements"


def generation_disagrees(base: dict, treat: dict) -> bool:
    """Was one arm running a reply while the other was not, at the moment each was digested?

    THE COMPOSER IS A FUNCTION OF THIS, and the composer is inside `.aui-thread-root`:
    `ThreadPrimitive.Root` wraps `ThreadComposerDock`, so `digest_scaffold` carries whichever
    control the run state selected. Stop, Queue and Send are different subtrees, so two arms that
    disagree about generation disagree about the scaffold FOR THAT REASON, with no rendering
    difference between them.

    Read from `composer_control`, the TOKEN naming which control the composer put in that slot, and
    not from `streaming`. `streaming` is `isRunning()`, which is true for Stop and for Queue alike,
    so a queued-idle arm and a streaming arm agree on it while rendering two different subtrees --
    and the scaffold is exactly what those subtrees are in. Captures taken before the token existed
    fall back to `streaming`, and captures older than that report neither and are scored as they
    always were.
    """
    bc, tc = base.get("composer_control"), treat.get("composer_control")
    if bc is not None and tc is not None:
        return bc != tc
    return bool(base.get("streaming")) != bool(treat.get("streaming"))


def _run_state_disagrees(base: dict, treat: dict) -> bool:
    """Do the arms disagree about the run state, read OFF the run state rather than the composer?

    `generation_disagrees` reads `composer_control`, which is the composer. Using it alone to
    excuse a composer difference proves the premise with the conclusion, so this is the second,
    independent half: `streaming` is `isRunning()` and `queued_idle` is the queue waiting to be
    dispatched, both taken from the thread's own run state and neither derivable from which button
    was drawn.

    Captures older than these fields report neither, and two `False`s then read as agreement --
    which is the conservative direction here, since it makes the pair a reported difference rather
    than a silent refusal.
    """
    return bool(base.get("streaming")) != bool(treat.get("streaming")) or bool(
        base.get("queued_idle")
    ) != bool(treat.get("queued_idle"))


def _messages_moved(
    base: dict,
    treat: dict,
    skip: Optional[set[int]] = None,
) -> bool:
    """The message half of `_any_moved`, on its own."""
    skip = skip or set()
    bm, tm = _messages(base), _messages(treat)
    # A message PRESENT ON ONE ARM ONLY is a difference whether or not it was streaming. Being
    # mid-reply excuses a digest, never an absence.
    if set(bm) != set(tm):
        return True
    return any(bm[i].get("digest") != tm[i].get("digest") for i in bm if i not in skip)


def scaffold_moved(base: dict, treat: dict) -> bool:
    """Did the thread scaffolding -- viewport, composer, empty state -- move?"""
    return _scaffold(base)[0] != _scaffold(treat)[0]


def settled_messages_moved(base: dict, treat: dict) -> list[str]:
    """The per-message rows that differ and PROVABLY cannot be the reply being written.

    THE SAME RULE `compare_visible` APPLIES TO ITS OWN ROWS, and it has to be the same rule: the
    two modes must not disagree about which readings have a defined moment. Two shapes qualify.

    A ROW BOTH ARMS CALL THE USER'S. A stream writes into an assistant message -- assistant-ui's
    `status` (`running` / `complete` / ...) is defined on assistant messages only, and the
    `data-status` hook `scene/dom.js` walks is on the assistant text part -- so a user row is never
    the message being written, whether or not the probe could place that message. The two arms
    AGREEING about the role is what makes it provable; a row whose role disagrees is reported as a
    role change instead of being trusted as either.

    A ROW WHOSE ROLE CHANGED. The role is captured beside the digest rather than inside it, and how
    far a reply has arrived says nothing about whose message it is, so this is reported even for a
    row that is in flight -- where the digest itself is still withheld.

    Reached only past `mount_count_mismatch`, so the two arms mounted the same number of messages
    and index `i` names the same position on both sides.
    """
    bm, tm = _messages(base), _messages(treat)
    streaming = in_flight(base, treat)
    out: list[str] = []
    for i in sorted(set(bm) & set(tm)):
        b, t = bm[i], tm[i]
        if b.get("role") != t.get("role"):
            out.append(f"msg{i}:role {b.get('role')}->{t.get('role')}")
            continue
        # Flagged in flight by the arm that COULD place its stream. Its digest is a point in a
        # stream on that arm whatever role it carries, so it is withheld like any other.
        if i in streaming:
            continue
        if b.get("role") == "user" and b.get("digest") != t.get("digest"):
            out.append(f"msg{i}(user):{b.get('chars')}->{t.get('chars')}c")
    # WHY THERE IS NO ASSISTANT-ROW RULE HERE, since it looks like the obvious next one to add.
    #
    # An earlier assistant row on a fully mounted thread really is settled while the tail is being
    # written, and a counting argument even makes it provable without knowing WHICH row is in
    # flight: a thread writes one reply at a time, so two differing assistant rows cannot both be
    # it. Both readings are sound and both are still unusable, because of what BLINDS the probe in
    # the first place.
    #
    # The reachable blind pair is a build that renamed the `data-status` hook. That attribute is on
    # the assistant text part and the digest walks attributes, so the rename that blinds the probe
    # ALSO moves the digest of every assistant row, by itself. Reporting those rows therefore turns
    # every genuinely blind pair into a difference, which is the wall-clock false alarm this file
    # exists to remove, wearing the other hat. Measured: implementing the counting rule flipped
    # `test_a_settled_queued_idle_pair_is_scored_rather_than_refused`'s blind control from NOT
    # COMPARABLE to DIFFER.
    #
    # A user row is not exposed to that, which is exactly why it is the one shape that qualifies:
    # the hook is not on it. Separating "the hook attribute moved" from "the content moved" would
    # need a digest that excludes the attribute, which is not something this capture carries.
    return out


def _any_moved(
    base: dict,
    treat: dict,
    skip: Optional[set[int]] = None,
) -> bool:
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
    skip = skip or set()
    # THE SCAFFOLD, not the whole-thread digest. The whole-thread digest serialises the streamed
    # message too, so with a reply in flight it differs on essentially every pair -- measured at
    # 175 of 175 adjacent 24-character steps of the shipped corpus -- and every check below it
    # would be unreachable behind it, exactly as the whole-thread digest once made the overlay walk
    # unreachable. The scaffold plus the per-message rows is the same reading taken apart, so
    # nothing is lost by decomposing it. On a payload with no scaffold this IS the whole-thread
    # digest and the behaviour is the old one.
    if _scaffold(base)[0] != _scaffold(treat)[0]:
        return True
    bo, to = _overlays(base), _overlays(treat)
    if [(o.get("sel"), o.get("digest")) for o in bo] != [
        (o.get("sel"), o.get("digest")) for o in to
    ]:
        return True
    return _messages_moved(base, treat, skip)


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
    blind = streaming_probe(base, treat)
    if blind is not None:
        # THE REFUSAL MUST NOT SWALLOW A READING THAT DOES NOT DEPEND ON THE STREAM. An overlay is
        # walked from `document`, outside the thread root, so a dialog that mounted when it should
        # not, or one whose contents were rewritten, is a finding whether or not the streamed
        # message could be identified. Without this it went out with the refusal, and
        # `structural_report` buckets a refusal as blind and never consults it for the exit code,
        # so the run went green on it.
        #
        # THE SCAFFOLD IS DELIBERATELY NOT CONSULTED HERE, and this is the part of the review item
        # that measurement does not support. `ThreadPrimitive.Root` wraps `ThreadComposerDock`
        # (thread.tsx), so the composer is inside `.aui-thread-root` and inside the scaffold -- and
        # the composer is exactly the surface that changes when a reply starts and stops. Measured
        # on two byte-identical threads differing only in the composer control: Stop against Send
        # moves the scaffold from 373 to 381 characters and changes its digest, with no message
        # content involved at all. On the very pair this branch is about -- one arm generating with
        # a quiet hook, the other finished -- the scaffold therefore differs BECAUSE one arm is
        # generating, and reporting that as a rendering difference would manufacture the wall-clock
        # false alarm this whole file exists to remove.
        independent = overlays_moved(base, treat)
        # AND THE SETTLED MESSAGE ROWS, which are the other half of what this refusal must not
        # swallow. A row both arms call the user's, and a row whose role itself changed, cannot be
        # the reply being written -- so their meaning does not depend on where the stream had got
        # to, and discarding them costs exactly the finding this instrument exists to make. Without
        # this a user message rendered differently by the treatment left here as NOT COMPARABLE
        # with an empty `moved`, and `structural_report` buckets a refusal as blind and never
        # consults it for the exit code, so the run went green on it. `compare_visible` already
        # applies this rule to its own rows and says it is the same rule as this one; it has to be.
        independent = independent + settled_messages_moved(base, treat)
        # AND THE SCAFFOLD, but only when the two arms agree about whether a reply was running.
        # The composer is inside the scaffold and is a function of exactly that, so the scaffold is
        # readable here when they agree and meaningless when they do not. That is the correct form
        # of the half of this refusal that measurement did not support.
        #
        # AND THE DISAGREEMENT HAS TO BE CORROBORATED OFF THE RUN STATE, exactly as the composer
        # suppression below requires. `generation_disagrees` reads `composer_control`, which IS the
        # composer, so on its own it excuses a composer difference with the composer -- and a
        # treatment that DROPS the Stop control, renames it, or selects the wrong one makes the
        # tokens differ for that reason and suppressed the whole scaffold with it, while `streaming`
        # and `queued_idle` were saying the arms were in the same run state all along. That is a
        # rendering regression withheld by the branch written to admit rendering regressions, and
        # since a refusal is filed under `blind` the run goes green on it.
        if scaffold_moved(base, treat) and not (
            generation_disagrees(base, treat) and _run_state_disagrees(base, treat)
        ):
            bc, tc = _scaffold(base)[1], _scaffold(treat)[1]
            independent = independent + [f"thread scaffolding outside any message ({bc}->{tc}c)"]
        if independent:
            return {
                "verdict": DIFFER,
                "reason": (
                    "the streamed message could not be identified on one arm, but surfaces that "
                    "CANNOT be the reply being written differ: an overlay is walked outside the "
                    "thread root, so its digest carries neither the stream nor the composer, and a "
                    "row both arms call the user's, or a row whose role changed, is not the "
                    f"message a stream writes into. {blind}"
                ),
                "moved": independent,
                "style_verdict": style_verdict,
                "style_reason": style_reason,
            }
        return {
            "verdict": NOT_COMPARABLE,
            "reason": blind,
            "moved": [],
            "style_verdict": style_verdict,
            "style_reason": style_reason,
        }
    # ── THE STREAMED MESSAGE ────────────────────────────────────────────────────────────────────
    #
    # `streaming` holds the messages that were still being written on one arm or the other. They
    # are scored on NOTHING, and that is a refusal rather than a normalisation: the digest of a
    # half-written message names a point in a stream, the two arms are at two different points in
    # the same stream by construction, and no amount of text normalisation can turn one into the
    # other. See `in_flight`.
    #
    # THREE OUTCOMES, and the ordering between them is the load-bearing part.
    #
    #   the settled document differs      DIFFER. This is the case that used to be lost. An action
    #                                     landing inside a stream was silenced wholesale by
    #                                     UNSTABLE_ACTIONS, so a real regression anywhere else in
    #                                     the thread -- another message, an overlay, the composer --
    #                                     printed under "expected to vary" and the run exited 0.
    #                                     Now the streamed message is elided and the rest is scored.
    #   the settled document agrees and
    #   the in-flight message agrees      MATCH, unchanged. Two arms that landed on the same point
    #                                     in the stream serialised identically, which is the claim
    #                                     this mode makes, and demoting it would cost coverage for
    #                                     nothing.
    #   the settled document agrees and
    #   the in-flight message differs     NOT COMPARABLE. Not a pass. `CLAIM_STRUCTURAL` quantifies
    #                                     over the whole thread, one message did not serialise
    #                                     identically, and the reason it did not is a reading with
    #                                     no defined moment. Calling that MATCH would be the
    #                                     instrument certifying a surface it could not look at,
    #                                     which is the failure `compare_styles` and
    #                                     `compare_visible` each already refuse in their own way.
    #
    # WHAT THIS GIVES UP, plainly, and both of these are measured rather than reasoned about.
    #
    # 1. A genuine rendering regression INSIDE the message that happened to be streaming is no
    #    longer distinguishable here and lands as NOT COMPARABLE. It was not distinguishable before
    #    either -- every action that lands in a stream is on the declared unstable list -- so
    #    nothing that used to be caught stops being caught, and the outcome moves from "expected to
    #    vary, exit 0" to "not measured, not a pass".
    # 2. A REORDER that moves the streamed message past another message of the same role is
    #    demoted from DIFFER to NOT COMPARABLE. The per-message rows are keyed by mounted index, so
    #    swapping two messages puts a streaming row at one index on one arm and at the other index
    #    on the other; both indices are then in flight on one side or the other, both are withheld,
    #    and the scaffold markers that survive carry the same role in both orders. Measured on the
    #    live-DOM battery: 11 injected rendering differences, 10 still reported DIFFER and this one
    #    demoted. It is a demotion and not a hole -- NOT COMPARABLE is not a pass and the run does
    #    not go green on it -- and the only shape that reaches it needs the streamed message not to
    #    be the last one, which the app does not do today.
    streaming = in_flight(base, treat)
    # ── THE COMPOSER IS NOT A RENDERING DIFFERENCE ──────────────────────────────────────────────
    #
    # The pair this whole change is about is one arm that has finished its reply against one that
    # is still writing it. Its MESSAGES are withheld correctly. Its COMPOSER is not: the dock is
    # inside `.aui-thread-root`, so `digest_scaffold` carries Stop on the arm that is generating
    # and Send on the arm that is not, and `_any_moved` then reports the pair as DIFFER with the
    # single claim `thread scaffolding outside any message (373->381c)`. Measured on exactly that
    # pair, with every settled message row byte-identical across the two arms.
    #
    # THE NULL BATTERY CANNOT SEE THIS, which is why it survived a 15-of-15-to-0 null. The null is
    # one build against itself at six points in ONE stream, so BOTH arms are generating and both
    # render Stop; the bias is symmetric within the control and cancels exactly. A flat null proves
    # repeatability, never comparability.
    #
    # WITHHELD RATHER THAN IGNORED. If a message or an overlay also moved, that is reported as
    # usual and this never runs. It is only when the scaffold is the ONLY thing that moved, and the
    # arms disagree about generation, that the reading has no defined moment -- and then the honest
    # answer is a refusal, not a pass. Calling it MATCH would hide a genuine composer regression.
    #
    # AND THE RUN STATE HAS TO SAY SO INDEPENDENTLY, or the suppression argues in a circle.
    # `generation_disagrees` reads `composer_control`, the token naming which control the composer
    # rendered -- so "the arms were at different points in the turn" is being proved by the very
    # surface whose difference is in question. A treatment that DROPS the Send button, renames it,
    # or selects the wrong control reaches this branch with every message and overlay agreeing, and
    # a refusal here is a green run: `report` files NOT COMPARABLE under `blind` and takes its exit
    # code from `stable_bad or one_sided`. That is the regression the note above says must not be
    # hidden, hidden by the branch written to say so.
    #
    # `streaming` and `queued_idle` are read off the run state rather than off the composer, so
    # they are evidence the composer cannot manufacture. Every legitimate suppression still has
    # one: Stop against Send differs in `streaming`, and Queue against Send in the queued-idle
    # interval differs in `queued_idle`. What is left, the arms agreeing on both and still
    # rendering different controls, has no run-state explanation and is a rendering difference.
    if (
        generation_disagrees(base, treat)
        and _run_state_disagrees(base, treat)
        and scaffold_moved(base, treat)
        and not overlays_moved(base, treat)
        and not _messages_moved(base, treat, streaming)
    ):
        bc, tc = _scaffold(base)[1], _scaffold(treat)[1]
        return {
            "verdict": NOT_COMPARABLE,
            "reason": (
                "the only difference is the thread scaffolding "
                f"({bc}->{tc}c), and one arm was running a reply while the other was not. The "
                "composer dock is inside the thread root and renders Stop or Queue while a reply "
                "is running and Send when it is not, so the scaffolding differs BECAUSE the two "
                "arms were at different points in the same turn. Every message and every overlay "
                "agreed. The scaffold is read as ONE AGGREGATE digest -- viewport, composer dock "
                "and empty state together -- so this cannot separate the composer swap from a "
                "change elsewhere in the scaffold that happened at the same time, and it does not "
                "claim to have: that is part of why it refuses. Nothing here is a pass"
            ),
            "moved": [],
            "in_flight": sorted(streaming),
            "style_verdict": style_verdict,
            "style_reason": style_reason,
        }
    if not _any_moved(base, treat, streaming):
        bm, tm = _messages(base), _messages(treat)
        unsettled = sorted(
            i
            for i in streaming
            if i in bm and i in tm and bm[i].get("digest") != tm[i].get("digest")
        )
        if unsettled:
            names = ", ".join(
                f"msg{i}({bm[i].get('role', '?')}):{bm[i].get('chars')}->{tm[i].get('chars')}c"
                for i in unsettled[:4]
            )
            return {
                "verdict": NOT_COMPARABLE,
                "reason": (
                    f"the settled thread is identical on both arms and the only difference is in "
                    f"{len(unsettled)} message(s) that were STILL BEING WRITTEN when the digest "
                    f"was taken ({names}). The two arms are two cells against one pacer with their "
                    "own send clicks and their own paint clocks, so that digest names a point in a "
                    "stream rather than a rendering, and its size carries no information either: "
                    "the renderer repairs the half-arrived construct, so the serialisation is not "
                    "monotonic in how much text has landed. Nothing here is a pass"
                ),
                "moved": [],
                "in_flight": sorted(streaming),
                "not_digested": unsettled,
                "style_verdict": style_verdict,
                "style_reason": style_reason,
            }
        return {
            "verdict": MATCH,
            "reason": "",
            "moved": [],
            "in_flight": sorted(streaming),
            "style_verdict": style_verdict,
            "style_reason": style_reason,
        }
    return {
        "verdict": DIFFER,
        "reason": "",
        "moved": localise(base, treat, streaming),
        "in_flight": sorted(streaming),
        "style_verdict": style_verdict,
        "style_reason": style_reason,
    }


def execution_verdict(base_row: Optional[dict], treat_row: Optional[dict]) -> Optional[dict]:
    """The verdict that the DIGEST never gets to answer, or `None` when both arms ran.

    EXTRACTED SO THERE IS ONE COPY, not for tidiness. Only `compare_rows` ever consulted this; the
    visible-region and behavioural paths reduced execution to a boolean conjunction of `ran`, which
    cannot express WHICH arm went idle. An action that RUNS on one build and cannot be performed on
    the other leaves no digest to differ, and on a windowed arm those two reports are the whole UI
    verdict, so that class had nowhere left to be seen. A predicate copied into them would drift the
    way the gate-admission lists did before `INVALIDATING_CELL_GATES` was centralised.

    The returned dict is the caller's to merge or return whole: `compare_rows` returns it, the other
    two keep their own capture comparison and take `one_sided` and `idle_reason` off it.
    """
    ran: dict[str, bool] = {}
    for label, row in (("base", base_row), ("treatment", treat_row)):
        if not isinstance(row, dict):
            return {
                "verdict": NOT_COMPARABLE,
                "reason": f"the {label} arm has no row for this action",
                "moved": [],
                "one_sided": "",
                "idle_reason": "",
                "style_verdict": NOT_COMPARABLE,
                "style_reason": "",
            }
        ran[label] = bool(row.get("ran"))
    if all(ran.values()):
        return None
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
        # The idle arm's OWN not_run string, unwrapped. `reason` above is prose built for a
        # reader; the exemption has to match on what the action actually recorded.
        "idle_reason": reason,
        "reason": detail,
        "style_verdict": NOT_EXERCISED,
        "style_reason": "",
    }


def expect_regression(base_row: Optional[dict], treat_row: Optional[dict]) -> tuple[str, str]:
    """(the arm whose own assertion failed while the other's passed, why) or `("", "")`.

    AN ACTION THAT RAN AND FAILED ITS OWN ASSERTION IS NOT A COMPARISON OF WHAT IT NAMES, and
    `ran` alone cannot see it. `stop_generation` returns `ran = True, expect_ok = stopped_ms is not
    None`, so a head on which Stop no longer ends the stream records a row this layer read as a
    perfectly good observation. `scoring.from_payload`, `report.payload` and `--assert-liveness`
    already know better; the parity layer was the one left reading only `ran`.

    CARRIED SEPARATELY FROM THE DIGEST, not folded into the verdict. `stop_generation` is on the
    declared unstable list, so its DOM difference is excused -- but that exemption measures one
    quantity (does this digest race) and the assertion is another (did the button do its job). They
    share an action name and nothing else, so the caller gets its own signal and decides.

    ONLY AN ASYMMETRY. `expect_ok is None` means the action asserts nothing, and both arms failing
    means the fixture cannot reach the state on either build -- coverage lost, not a build
    difference. One arm asserting successfully while the other does not is the question asked here.

    SHARED WITH THE VISIBLE AND BEHAVIOURAL PATHS for the reason on `execution_verdict`: those two
    ARE the UI verdict on a windowed arm, and `stop_generation` has no behavioural invariant, so an
    assertion failing on one arm alone left no trace in either.
    """
    rows_ = (("base", base_row), ("treatment", treat_row))
    if not all(isinstance(row, dict) for _label, row in rows_):
        return "", ""
    failed = [label for label, row in rows_ if row.get("expect_ok") is False]  # type: ignore[union-attr]
    passed = [label for label, row in rows_ if row.get("expect_ok") is True]  # type: ignore[union-attr]
    if not (len(failed) == 1 and passed):
        return "", ""
    who = failed[0]
    row = base_row if who == "base" else treat_row
    assert isinstance(row, dict)
    return who, (
        f"the action ran on both arms and its own assertion failed on the {who} arm "
        f"({row.get('reason') or 'no reason recorded'}), so the two builds did not behave "
        "the same way"
    )


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

    Both halves live in `execution_verdict` and `expect_regression` so the visible-region and
    behavioural paths can ask the same two questions of the same code; see those docstrings.
    """
    idle = execution_verdict(base_row, treat_row)
    if idle is not None:
        return idle
    assert base_row is not None and treat_row is not None
    out = compare(base_row.get("parity"), treat_row.get("parity"))
    out["expect_regressed"], out["expect_reason"] = expect_regression(base_row, treat_row)
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
# THE POLICY. All changes must preserve UI and UX idempotency, with three exemptions: a difference
# may be accepted deliberately when performance improves dramatically; a difference that exists
# only OFF SCREEN is fine by definition, because rendering only what is visible is an accepted
# technique rather than a parity violation; and a select-all need not select all, PROVIDED the copy
# it produces stays complete.
#
# The third is what makes deferral and virtualization cheap: the copy path stops depending on what
# is mounted, so it may serialise the thread from the message store as markdown or plain text
# instead of reproducing a DOM selection. Completeness of the copied content is REQUIRED -- silent
# truncation is data loss -- and visual selection fidelity is NOT.
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
    "first still hold, including that each arm's clipboard covers the thread's own visible text "
    "BY LENGTH. Says NOTHING about how anything looks, and nothing about WHICH characters were "
    "copied"
)

#: THE POLICY EVERY VERDICT IS JUDGED AGAINST, printed beside the claim.
#:
#: A bare "PARITY OK" reads as "the UI is unchanged" and none of the three modes can support that
#: sentence. Each supports a narrower one, and which one is being made changes what a pass MEANS,
#: so the policy is named next to the claim rather than left in a document somebody has to find.
#:
#: The exemptions are not loopholes. The first is a decision someone makes on the record with a
#: number attached; the second is a definition, because rendering only what is visible is an
#: accepted technique rather than a parity violation; the third is CONDITIONAL, and the condition
#: is the part that is required -- the copy must be complete, and only the visual fidelity of the
#: selection is given up. None of the three removes the need for a floor: a difference that is
#: exempt still has to be shown to be the difference you think it is, which is what the null
#: control is for.
#:
#: NO MODE GRANTS THE THIRD ON A DIGEST. `--mode behaviour` is the only one that speaks to it at
#: all, through `clipboard_carries_the_whole_thread`, which scores the copy against the THREAD
#: rather than against the other arm. Where a payload carries no readable `select_all_copy` the
#: gate records that the exemption exists and does not grant it.
#:
#: And it scores it BY LENGTH. That is a proxy for completeness, not completeness itself, so the
#: printed line says so rather than leaving "complete" to be read as a content comparison. A
#: content comparison is not available here: the base arm's clipboard is the DOM's rendered text
#: while a store-based copy is markdown SOURCE, so the two are different serialisations of one
#: conversation and comparing their characters fails on a CORRECT build. The coverage band is what
#: carries the weight, and it is the band that refused both defects this was written for --
#: truncation at 0.61 of the thread and substitution at 2.16 -- so it is interpolated from the
#: live constants by `behaviour_policy()` rather than written out here, where it could drift.
POLICY = (
    "UI and UX idempotency is required, with three exemptions: a deliberate difference accepted "
    "for a dramatic performance improvement, a difference that exists only OFF SCREEN, and a "
    "select-all that does not select all PROVIDED the copy it produces stays complete"
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
        "nothing about appearance. It cannot grant the performance or off-screen exemptions, and "
        "it is the only mode that speaks to the select-all one, on the half of that exemption "
        "that is REQUIRED: whether the copy stays complete. It measures that BY LENGTH, scoring "
        "each arm's clipboard against the thread's own visible text and requiring coverage in "
        "{min}-{max}. It does not compare the copied characters and cannot: the base arm's "
        "clipboard is the DOM's rendered text and a store-based copy is markdown source, so a "
        "character comparison of two correct serialisations fails. Without a readable "
        "select_all_copy it records the exemption rather than granting it"
    ),
}


def behaviour_policy(min_coverage: float, max_coverage: float) -> str:
    """The behaviour-mode policy line with the coverage band it actually enforces filled in.

    The band lives in `analysis.behaviour`, which imports THIS module, so the caller passes it in
    rather than this module importing back and closing a cycle. Interpolating it is the point: a
    band written out by hand here drifts away from the one enforced, and the sentence a reader
    trusts is then describing a gate that no longer exists.
    """
    return POLICY_BY_MODE["behaviour"].format(min = min_coverage, max = max_coverage)


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

    # BEFORE THE TWO SETS ARE COMPARED, because a collision is what makes them untrustworthy.
    # `scene/parity.js` keys its per-message digests by ordinal, so two mounted rows sharing an
    # `aria-posinset` -- a virtualizer renumbering a recycled row wrongly -- leave one digest where
    # two rows were on screen, and `ever_visible` collapses them too. Both quantities below are then
    # short by a row, and whether the verdict came out MATCH or DIFFER would turn only on which of
    # the two survived, which is DOM order and nothing else.
    #
    # Read defensively so a payload recorded before the counter existed compares exactly as it did.
    collisions = {
        label: int(side.get("ordinal_collisions") or 0)
        for label, side in (("base", base), ("treatment", treat))
    }
    # A SEVERE FINDING OUTRANKS THIS REFUSAL, and exactly one finding qualifies. "One arm's
    # viewport ended EMPTY and the other's did not" is marked NOT SUPPRESSIBLE where it is raised,
    # because losing the thread is a different kind of statement from a capture that could not be
    # read -- and a collision provably cannot manufacture it. A collision needs TWO mounted rows
    # at one position, so the map it corrupts still has at least one entry in it; it can merge two
    # entries and never empty a map. So the one thing a collision cannot have caused is still
    # reported rather than swallowed by a blanket refusal.
    #
    # Only that one. Every other comparison below IS corrupted by a collision: `ever_visible` loses
    # an ordinal on one arm alone, so "the two arms put different messages on screen" can be
    # entirely an artefact of the collision, and the per-ordinal digests are short by a row.
    lost_the_thread = (len(base.get("messages") or {}) == 0) != (
        len(treat.get("messages") or {}) == 0
    )
    if any(collisions.values()) and not lost_the_thread:
        which = ", ".join(
            f"{label} {n} at ordinal(s) "
            f"{sorted((base if label == 'base' else treat).get('collided_ordinals') or [])[:8]}"
            for label, n in collisions.items()
            if n
        )
        return {
            "verdict": NOT_COMPARABLE,
            "reason": (
                "two or more mounted rows published the SAME thread position during this action, "
                f"so one of them is missing from everything compared below ({which}). The digest "
                "map is keyed by that position and the visible set is a set of them, so neither "
                "carries the extra row"
            ),
            "moved": [],
            "claim": CLAIM_VISIBLE,
            "not_digested": [],
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
    # The subset of `moved` that CANNOT be a point in a stream, kept so the blind-probe refusal
    # below does not take it out with the rest. See each `settled.append` for why that row is
    # provably not the reply being written.
    settled: list[str] = []
    for ordinal in sorted(bev):
        key = str(ordinal)
        b, t = bmsg.get(key), tmsg.get(key)
        if b is None or t is None:
            continue
        # A ROLE IS NOT A POSITION IN A STREAM. It is captured beside the digest rather than inside
        # it, and how far a reply has arrived says nothing about whether it is the assistant's. So
        # a row that changed role is reported even while that row is in flight, where the digest
        # itself is withheld: a treatment that renders the live assistant row as `data-role="user"`
        # is a structural regression that used to leave here as NOT COMPARABLE.
        if b.get("role") != t.get("role"):
            claim = f"ordinal {ordinal}:role {b.get('role')}->{t.get('role')}"
            moved.append(claim)
            settled.append(claim)
            continue
        if b.get("digest") != t.get("digest"):
            if b.get("in_flight") or t.get("in_flight"):
                # STILL BEING WRITTEN ON ONE ARM OR THE OTHER, so its digest names a point in a
                # stream. Residue, exactly like an ordinal that was unmounted before the capture:
                # it cannot be a difference, and it cannot be an agreement either, so it joins the
                # list that refuses the verdict below rather than the list that fails it. Same
                # rule as `compare` applies to the structural digest; the two modes must not
                # disagree about which readings have a defined moment.
                uncomparable.append(ordinal)
                continue
            claim = f"ordinal {ordinal}({b.get('role')}):{b.get('chars')}->{t.get('chars')}c"
            moved.append(claim)
            # A USER ROW IS NEVER THE REPLY BEING WRITTEN. Both arms agree this ordinal is the
            # user's, and a stream writes into an assistant message, so this difference survives
            # the blind-probe refusal below. The two arms agreeing is what makes it provable: a
            # row whose role DISAGREES is reported above as a role change, not silently trusted.
            if b.get("role") == "user":
                settled.append(claim)
    # Ordinals that were unmounted before the capture and ordinals that were still streaming are
    # the same kind of residue and are counted once each.
    uncomparable = sorted(set(uncomparable))
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
    # THE STREAMING PROBE, on the same footing it has in `compare`, because this mode is scored
    # from its own payload and never sees the structural verdict. `in_flight` above walks the
    # `data-status` / `aria-busy` hooks, so a build that renames them reports every row settled;
    # the two arms are two different builds, so that can be true on ONE of them while the other's
    # reply has genuinely finished, and the differing digests below are then two points in a stream
    # scored as a rendering difference.
    #
    # AFTER the two lost-conversation findings above and BEFORE the digest comparison, for the
    # reason `compare` puts it after `mount_count_mismatch`: different messages on screen, or one
    # viewport emptied, are readings that do not depend on the stream split at all, and a build
    # that loses the thread while a reply runs stays a finding rather than a shrug.
    blind = streaming_probe(base, treat)
    if blind is not None:
        # SAME RULE AS `compare`: the refusal covers the rows whose meaning depends on where the
        # stream had got to, and nothing else. `settled` holds the rows that provably cannot be the
        # reply being written -- a row both arms call the user's, and a row whose role itself
        # changed -- and those are reported rather than discarded. Without this a changed user
        # message left here as NOT COMPARABLE with an empty `moved`, and `visible_report` buckets
        # a refusal as blind and never consults it for the exit code.
        if settled:
            return {
                "verdict": DIFFER,
                "reason": (
                    f"{len(settled)} visible message(s) rendered differently on rows that cannot "
                    "be the reply being written (a user row, or a row whose role changed), so "
                    f"they are reported even though the stream could not be placed. {blind}"
                ),
                "moved": settled,
                "claim": CLAIM_VISIBLE,
                "not_digested": uncomparable,
            }
        return {
            "verdict": NOT_COMPARABLE,
            "reason": blind,
            "moved": [],
            "claim": CLAIM_VISIBLE,
            "not_digested": uncomparable,
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
