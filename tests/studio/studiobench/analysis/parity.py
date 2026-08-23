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
MATCH = "match"
DIFFER = "differ"
NOT_COMPARABLE = "not_comparable"
NOT_EXERCISED = "not_exercised"

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
