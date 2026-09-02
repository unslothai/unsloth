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

#: WHAT "MATCH" ACTUALLY CLAIMS: everything here is computed from `scene/parity.js`, whose
#: structural digest walks the THREAD only, so it is sidebar-blind and layout-blind by
#: construction. Measured: on a real visible sidebar-drag change the thread digest found 0 of 34
#: differing pairs, while three purpose-built captures found it 34 of 34. So MATCH means NO
#: THREAD-STRUCTURE CHANGE WAS DETECTED, not that the UI is unchanged.
MATCH = "match"
DIFFER = "differ"
NOT_COMPARABLE = "not_comparable"
NOT_EXERCISED = "not_exercised"
#: THE FOURTH OUTCOME, and not a softer NOT_COMPARABLE: that one means the reading failed, while
#: NOT_APPLICABLE means the question is wrong for this pair -- the digest asks "is the same DOM on
#: screen" and an arm whose purpose is to put less DOM on screen answers "no" by construction.
#: Kept distinct so `derive_unstable` counts neither as evidence.
NOT_APPLICABLE = "not_applicable"

#: A KEY ON A NOT_COMPARABLE RESULT, not a fifth verdict. `compare` has one refusal that also
#: carries a complete positive reading: everything agreed except the subtree of a reply that was
#: mid-tail. It stays a refusal, because a half-arrived digest names a point in a stream. THE
#: VERDICT DOES NOT MOVE; only `derive_unstable` reads this, since for "does this action differ
#: against ITSELF" a pair where everything readable agreed is an observation of non-difference.
#: It can only NARROW the excuse set, so the worst case is a loud false alarm.
SETTLED_MATCH = "settled_match"

# Actions whose rendered result legitimately differs between two runs of the SAME build, so a
# digest mismatch there says nothing about the pull request. A MECHANISM PER ENTRY, as the value
# rather than a comment beside it so a test can require one: an action silenced without a stated
# reason is an unauditable hole, and `derive_unstable` cross-checks that each earns its place.
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
    # scoring as a hard UI-change signal on a build compared with itself; the mechanism for each was
    # read out of the payload.
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


# WHOSE ABILITY TO RUN IS A RACE, a different claim from whose DIGEST varies. Every
# UNSTABLE_ACTIONS entry describes what makes the CAPTURE move, so using that list to excuse
# one-arm-only EXECUTION exempted nine of sixteen actions from the one regression shape that
# leaves no digest to differ. `slot_missed` already covers the runner arriving late, so what is
# left is an action that cannot run because the stream it needs is not there. KEYED ON THE
# REASON, because each has non-racy paths too, and a treatment that REMOVES a control records
# exactly the regression this exists to catch.
# `send_turn` also returns not_run for "no present" (scene/actions.py:501).
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


# The six that are NOT here: `composer_fill`, `keystroke`, `copy_markdown`, `message_menu` and
# `select_text` fail to run only when the control they need is absent or unresponsive, and that
# IS the build; `scroll_after` has no `not_run` path at all.


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
    # `root_kind` is absent in captures taken before it was recorded. Missing on BOTH sides is an old
    # payload and is allowed through; missing on one side means two different versions of the
    # instrument, which is not a comparison of two builds.
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
    Unsloth's markup: rename `data-status` and it goes quiet, matches nothing, and every capture then
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
            # In flight on BOTH sides of the comparison: a message present on one arm only is a different
            # statement and is reported below whatever its status.
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
        # The whole-thread digest moved but no message and no overlay did, so the difference is the
        # thread scaffolding itself; saying so beats an empty list that reads as "nothing differs".
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
    # A POSITIVE CONTROL ON THE SCAN ITSELF: two probes that matched no elements have equal counts
    # and equal digests (both the hash of an empty string), so a probe that scanned NOTHING reports
    # MATCH on no observation whatsoever. The selector list is written against Unsloth's markup, so a
    # class rename anywhere in it silently empties the scan.
    if not bs.get("elements") or not ts.get("elements"):
        return NOT_COMPARABLE, (
            f"the style probe matched no elements (base={bs.get('elements')}, "
            f"treatment={ts.get('elements')}), so it observed nothing on at least one arm. Its "
            "selector list is written against Unsloth's markup and does not survive a rename; "
            "this is a probe that needs fixing, not two arms that agree"
        )
    # THE RUN-STATE CONTROL IS IN THE SELECTOR LIST, so this reading straddles the same composer swap
    # the structural refusals below are about. `STYLE_SELECTORS` names both the Send and Stop buttons
    # and the signature carries the SELECTOR THAT MATCHED, so a generating arm and a settled one walk
    # two entries for one button: on two byte-identical threads Send against Stop moves the digest
    # with all three properties identical. THE SAME CORROBORATION THE SCAFFOLD SUPPRESSION REQUIRES,
    # because a differing `composer_control` alone is what a treatment that DROPS the control
    # produces. NOT_COMPARABLE AND NOT MATCH: the probe reads ONE aggregate digest, so a genuine CSS
    # difference elsewhere is inside the same number.
    # Over up to `STYLE_CAP` elements.
    # `streaming` and `queued_idle` are read off the run state, not the composer.
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
    # A message PRESENT ON ONE ARM ONLY is a difference whether or not it was streaming: being
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
        # Flagged in flight by the arm that COULD place its stream; its digest is a point in a stream on
        # that arm whatever role it carries, so it is withheld like any other.
        if i in streaming:
            continue
        if b.get("role") == "user" and b.get("digest") != t.get("digest"):
            out.append(f"msg{i}(user):{b.get('chars')}->{t.get('chars')}c")
    # WHY THERE IS NO ASSISTANT-ROW RULE HERE, though it looks like the obvious next one: an earlier
    # assistant row on a fully mounted thread really is settled, and a counting argument even proves
    # which cannot be in flight. Both readings are sound and unusable, because the reachable blind
    # pair is a build that renamed the `data-status` hook -- that attribute is on the assistant text
    # part and the digest walks attributes, so the rename that blinds the probe ALSO moves every
    # assistant row's digest. A user row is not exposed to that.
    # Measured: the counting rule flipped test_a_settled_queued_idle_pair_is_scored_rather_than_refused's
    # blind control from NOT COMPARABLE to DIFFER.
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
    # THE SCAFFOLD, not the whole-thread digest: the latter serialises the streamed message too, so
    # with a reply in flight it differs on essentially every pair (175 of 175 adjacent 24-character
    # steps) and every check below would be unreachable behind it. The scaffold plus the per-message
    # rows is the same reading taken apart.
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
            # The style probe goes with it: its verdict is `elements` counts matching, and those are element
            # counts over `[data-role]` among other things, so it reports DIFFER for the same reason.
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
        # THE REFUSAL MUST NOT SWALLOW A READING THAT DOES NOT DEPEND ON THE STREAM. An overlay is walked
        # from `document`, outside the thread root, so a dialog that mounted when it should not is a
        # finding either way; without this it went out with the refusal, which `structural_report` buckets
        # as blind, so the run went green. THE SCAFFOLD IS DELIBERATELY NOT CONSULTED HERE:
        # `ThreadPrimitive.Root` wraps `ThreadComposerDock`, so the composer is inside the scaffold and
        # Stop against Send moves it on two byte-identical threads.
        # `compare_visible` already refuses these.
        # thread.tsx; the swap moves the scaffold from 373 to 381 characters.
        independent = overlays_moved(base, treat)
        # AND THE SETTLED MESSAGE ROWS, the other half of what this refusal must not swallow: a row both
        # arms call the user's, and a row whose role itself changed, cannot be the reply being written,
        # so their meaning does not depend on the stream. Without this a user message rendered
        # differently by the treatment left as NOT COMPARABLE with an empty `moved`.
        independent = independent + settled_messages_moved(base, treat)
        # AND THE SCAFFOLD, but only when the two arms agree about whether a reply was running: the
        # composer is inside the scaffold and is a function of exactly that. AND THE DISAGREEMENT HAS TO
        # BE CORROBORATED OFF THE RUN STATE, since `generation_disagrees` reads `composer_control`, which
        # IS the composer -- so on its own it excuses a composer difference with the composer.
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
    # THE STREAMED MESSAGE. `streaming` holds messages still being written on one arm or the other;
    # they are scored on NOTHING, a refusal rather than a normalisation, because the two arms are at
    # two different points in one stream by construction. THREE OUTCOMES, and the ordering is
    # load-bearing: a settled document that differs is DIFFER; settled agreeing plus in-flight
    # agreeing is MATCH; settled agreeing with the in-flight message differing is NOT COMPARABLE, not
    # a pass, because `CLAIM_STRUCTURAL` quantifies over the whole thread. WHAT THIS GIVES UP: a
    # regression inside the streaming message lands as NOT COMPARABLE, and a REORDER past another
    # message of the same role is demoted from DIFFER (10 of 11 injected differences still DIFFER).
    streaming = in_flight(base, treat)
    # THE COMPOSER IS NOT A RENDERING DIFFERENCE. One arm finished and one still writing has its
    # MESSAGES withheld correctly, but the dock is inside `.aui-thread-root`, so `digest_scaffold`
    # carries Stop on one arm and Send on the other and `_any_moved` reported DIFFER while every
    # settled row was byte-identical. THE NULL BATTERY CANNOT SEE THIS -- one build against itself
    # has both arms generating and the bias cancels. WITHHELD RATHER THAN IGNORED: if a message or
    # overlay also moved this never runs. AND THE RUN STATE HAS TO SAY SO INDEPENDENTLY, or the
    # suppression argues in a circle.
    # ── THE COMPOSER IS NOT A RENDERING DIFFERENCE ──────────────────────────────────────────────
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
            # A ROW WHOSE ROLE CHANGED IS NOT SOMETHING THIS PAIR AGREED ON, and `_any_moved` cannot see it
            # because an in-flight digest is withheld while `settled_messages_moved` reports a role change
            # even in flight. The verdict is the refusal either way; this only decides whether the refusal
            # also carries a positive reading.
            roles_agree = all(bm[i].get("role") == tm[i].get("role") for i in set(bm) & set(tm))
            out = {
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
            if roles_agree:
                # THE ONE REFUSAL THAT ALSO CARRIES A POSITIVE READING (see `SETTLED_MATCH`). Set here and
                # nowhere else, because this is the only branch reached with `_any_moved` already false: the
                # scaffold, every overlay, the mounted message set and every settled message all agreed.
                out[SETTLED_MATCH] = True
            return out
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
    # A MISSED SLOT IS NOT A BUILD DIFFERENCE, even when only one arm missed it. `ran=false` has two
    # causes that look identical: a runner arriving after the slot closed says nothing about the
    # build, and because misses are correlated through the runner rather than independent draws,
    # corroboration does not separate them. A precondition failure is the opposite, so the signal is
    # `slot_missed` and not merely which arm went idle.
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
        # The idle arm's OWN not_run string, unwrapped: `reason` above is prose built for a reader, while
        # the exemption has to match what the action recorded.
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

    A SETTLED-MATCH REFUSAL IS AN OBSERVATION, and the only refusal that is. `SETTLED_MATCH` has
    the argument; the short version is that this function asks whether an action differs against
    ITSELF, and a pair whose scaffold, overlays, message set and every settled row agreed has
    answered that. What it withheld cannot produce a DIFFER on the result side either, so nothing
    is excused that anything was going to ask about.

    Counting it as blind is not conservative here, it is unreachable: both arms are driven through
    one film at wall-clock offsets, so on the packed 100K rungs every action landing inside a live
    turn reaches this branch whenever the two tails land a chunk apart. Measured on the payload
    that raised this, `keystroke@r100K` scored 0 observations from 2 pairs and `send_turn@r100K`
    1 from 2, and `audit_null` returned UNDECIDED on a run in which nothing had moved.
    """
    seen: dict[str, int] = collections.Counter()
    differ: dict[str, int] = collections.Counter()
    blind: dict[str, int] = collections.Counter()
    settled: dict[str, int] = collections.Counter()
    for action, result in pairs:
        verdict = result.get("verdict")
        if verdict not in (MATCH, DIFFER):
            if verdict == NOT_COMPARABLE and result.get(SETTLED_MATCH):
                # Everything this pair could read agreed: an observation of NON-difference, so `differed` cannot
                # grow. Tallied separately so a reader can see how much of a decision rests on it.
                seen[action] += 1
                settled[action] += 1
                continue
            # Not comparable and not exercised are both "no reading", and neither may count as an observation
            # of stability: an action derived as stable from four pairs that never ran would be permanently
            # trusted on nothing.
            blind[action] += 1
            continue
        seen[action] += 1
        if verdict == DIFFER:
            differ[action] += 1
    out: dict[str, dict] = {}
    for action in sorted(set(seen) | set(blind)):
        n, d = seen[action], differ[action]
        out[action] = {
            "observations": n,
            "differed": d,
            "not_comparable": blind[action],
            # How many of `observations` came from a settled-match refusal rather than a MATCH. Reported
            # rather than folded in: an action decided entirely this way was decided on the settled thread.
            SETTLED_MATCH: settled[action],
            # Unstable only with enough observations to mean it; below that the honest answer is "not enough
            # evidence". FROM THE COMPLETE COMPARISONS ONLY: a settled-match refusal can decide an action but
            # never help CLASSIFY one as unstable, or it would mint an exemption from a partial reading.
            "unstable": bool(d and (n - settled[action]) >= min_observations),
            # THE SAME COUNT `unstable` WAS DECIDED ON, once anything has differed. On the raw total, one
            # DIFFER beside one settled match read as "decided and stable", so `cross_check` filed a declared
            # action under `declared_stable_in_practice` on a run where it differed once.
            "undetermined": (n - settled[action] if d else n) < min_observations,
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


# THE POLICY: changes must preserve UI and UX idempotency, with three exemptions -- a dramatic
# performance win accepted on the record, a difference that exists only OFF SCREEN, and a
# select-all that need not select all PROVIDED the copy stays complete. The third is what makes
# deferral and virtualization cheap: the copy path may serialise from the message store, since
# completeness is REQUIRED and visual selection fidelity is not. `compare()` cannot express the
# second exemption, so it returns NOT_APPLICABLE and withholds a verdict. This supplies one.

#: The claim each verdict makes, so a reader knows which of the three was checked. NOT
#: "whole-document structural parity", which this instrument cannot support: `scene/parity.js`
#: digests the thread root plus overlay selectors, never the sidebar, geometry or custom
#: properties. Printing the stronger claim is how a sidebar-drag campaign came to be scored 0 of
#: 34 differing pairs under a banner saying the DOM was identical.
# ── visible-region parity ───────────────────────────────────────────
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

#: THE POLICY EVERY VERDICT IS JUDGED AGAINST, printed beside the claim, because a bare "PARITY
#: OK" reads as "the UI is unchanged" and no mode supports that sentence. The exemptions are not
#: loopholes: the first is a decision made on the record with a number, the second is a
#: definition, and the third is CONDITIONAL on the copy being complete. NO MODE GRANTS THE THIRD
#: ON A DIGEST: only `--mode behaviour` speaks to it, through
#: `clipboard_carries_the_whole_thread`, which scores the copy against the THREAD rather than the
#: other arm. And it scores BY LENGTH, a proxy for completeness: the base arm's clipboard is
#: rendered text while a store-based copy is markdown SOURCE, so comparing characters fails on a
#: CORRECT build. The coverage band carries the weight -- it refused truncation at 0.61 and
#: substitution at 2.16 -- so it is interpolated from the live constants.
POLICY = (
    "UI and UX idempotency is required, with three exemptions: a deliberate difference accepted "
    "for a dramatic performance improvement, a difference that exists only OFF SCREEN, and a "
    "select-all that does not select all PROVIDED the copy it produces stays complete"
)

#: What each mode can and cannot decide under that policy; the second half of each line is what
#: tells a reader which sentence they are being handed.
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

    # THE POSITIVE CONTROL: a visibility scan that matched nothing has equal (empty) ordinal sets and
    # no differing digests, so without this it returns the strongest verdict available on the strength
    # of never having seen a message. Exactly the failure `compare_styles` had.
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

    # BEFORE THE TWO SETS ARE COMPARED, because a collision is what makes them untrustworthy:
    # `scene/parity.js` keys per-message digests by ordinal, so two mounted rows sharing an
    # `aria-posinset` leave one digest where two rows were on screen and `ever_visible` collapses
    # them too. Read defensively so a payload recorded before the counter existed compares as it did.
    collisions = {
        label: int(side.get("ordinal_collisions") or 0)
        for label, side in (("base", base), ("treatment", treat))
    }
    # A SEVERE FINDING OUTRANKS THIS REFUSAL, and exactly one qualifies: "one arm's viewport ended
    # EMPTY and the other's did not" is marked NOT SUPPRESSIBLE, because a collision needs TWO mounted
    # rows at one position, so it can merge entries but never empty a map.
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
    # The subset of `moved` that CANNOT be a point in a stream, kept so the blind-probe refusal below
    # does not take it out with the rest. See each `settled.append` for why.
    settled: list[str] = []
    for ordinal in sorted(bev):
        key = str(ordinal)
        b, t = bmsg.get(key), tmsg.get(key)
        if b is None or t is None:
            continue
        # A ROLE IS NOT A POSITION IN A STREAM: it is captured beside the digest, and how far a reply has
        # arrived says nothing about whose message it is. So a row that changed role is reported even
        # while in flight -- a treatment rendering the live assistant row as `data-role="user"` used to
        # leave here as NOT COMPARABLE.
        if b.get("role") != t.get("role"):
            claim = f"ordinal {ordinal}:role {b.get('role')}->{t.get('role')}"
            moved.append(claim)
            settled.append(claim)
            continue
        if b.get("digest") != t.get("digest"):
            if b.get("in_flight") or t.get("in_flight"):
                # STILL BEING WRITTEN ON ONE ARM OR THE OTHER, so its digest names a point in a stream: residue,
                # exactly like an ordinal unmounted before the capture, so it joins the list that refuses the
                # verdict rather than the one that fails it. Same rule `compare` applies to the structural digest.
                uncomparable.append(ordinal)
                continue
            claim = f"ordinal {ordinal}({b.get('role')}):{b.get('chars')}->{t.get('chars')}c"
            moved.append(claim)
            # A USER ROW IS NEVER THE REPLY BEING WRITTEN, and both arms agreeing on the role is what makes
            # it provable; a row whose role DISAGREES is reported above as a role change.
            if b.get("role") == "user":
                settled.append(claim)
    # Ordinals unmounted before the capture and ordinals still streaming are the same kind of residue
    # and are counted once each.
    uncomparable = sorted(set(uncomparable))
    # ONE VIEWPORT ENDED EMPTY AND THE OTHER DID NOT, as visible a difference as there is, and it used
    # to be reported as NOT COMPARABLE. Not hypothetical: on the 100K virtualization arm
    # `model_change` took the thread from 12 mounted messages to 0 and it never came back. Both arms
    # had shown the same ordinals earlier, so the union matched and every per-ordinal digest was
    # simply missing on one side -- comparing what each arm could still show at the end sees that.
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
            # NOT SUPPRESSIBLE BY THE NOISE FLOOR: the floor exists for actions whose visible region differs
            # between two runs of one build, and losing the entire thread is a different kind of statement.
            # An action can be both -- `model_change` is in the derived unstable set for an unrelated
            # attribute, which would have silenced "the treatment arm's viewport ended empty".
            "severe": True,
        }
    # THE STREAMING PROBE, on the same footing it has in `compare`, because this mode is scored from
    # its own payload: `in_flight` walks the `data-status` / `aria-busy` hooks, so a build that
    # renames them reports every row settled while the other arm's reply has genuinely finished.
    # AFTER the two lost-conversation findings and BEFORE the digest comparison, since different
    # messages on screen do not depend on the stream split at all.
    blind = streaming_probe(base, treat)
    if blind is not None:
        # SAME RULE AS `compare`: the refusal covers the rows whose meaning depends on where the stream
        # had got to and nothing else. `settled` holds the rows that provably cannot be the reply being
        # written, and without this a changed user message left as NOT COMPARABLE with an empty `moved`.
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
        # ANY residue refuses the verdict, not only a total one. Entered solely when EVERY visible ordinal
        # was undigestable, a windowed action that put six messages on screen and unmounted one returned
        # MATCH on the strength of the other five, while the printed claim quantifies over EVERY message
        # the viewport showed. WHY THE VERDICT AND NOT MERELY THE PASS COUNT: demoting inside
        # `visible_report`, which never printed the `not_digested` residue either, alone leaves the row
        # reading `match`, which everything downstream counts as
        # agreement. THE COST, MEASURED on two real 100K films: four pairs in sixty-four.
        # `compare` puts this after `mount_count_mismatch`.
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
