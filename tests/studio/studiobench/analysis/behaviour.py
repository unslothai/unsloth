# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What replaces the structural digest when an arm changes what is mounted ON PURPOSE.

`sweep/ui_parity.py` asks "is the same DOM on screen on both arms". For every arm this project has
run, that is the right question. For an arm that mounts a window of the thread it is the wrong
one: the answer is no, by construction, on every action, and eighteen red rows that all say the
same thing are not a finding. Worse, they bury the differences that WOULD be findings.

So this asks the question that survives virtualization: THE DOM IS ALLOWED TO DIFFER, THE
BEHAVIOUR IS NOT. Five things a user does that stop working first when a list starts unmounting
rows, plus the scroll extent that every one of them depends on.

  SCROLL EXTENT      the viewport's `scrollHeight` must still describe the whole conversation. A
                     virtualizer that sizes its spacers correctly reproduces it within a few per
                     cent; one that simply drops rows produces a scrollbar that lies about how
                     much thread there is, and every scroll gesture, every jump-to-top and the
                     scrollbar thumb itself are then wrong. This is the invariant the other four
                     rest on, so it is checked on every action that carries a census rather than
                     on one named action.
  select_all_copy    Ctrl+A then Ctrl+C. The selection is taken over the DOM, so an unmounted
                     message CANNOT be on the clipboard. This is the one that is not a measurement
                     artefact and must not be filed as one: a user who copies their conversation
                     and gets a fraction of it has lost data, and an arm that virtualises without
                     answering for it has shipped that. It is checked as a coverage fraction of
                     the thread, and a shortfall is a FAILURE of the arm, not of the harness.
  select_text        selecting inside the last message. Single-message scope, so it must be
                     unaffected: if this moves, the arm has changed how a mounted message renders,
                     which is outside its remit.
  copy_markdown      the action bar's Copy on the last message. Also single-message scope, also
                     must not move. It is the control case for select_all_copy: if BOTH moved, the
                     change is not about what is mounted.
  thread_reopen      leaving the thread and coming back. The thread must come back the same LENGTH
                     -- `messages_before == messages_after` on both arms -- because the failure a
                     windowed mount invites is a reopen that restores only what fits on screen and
                     loses the rest of the conversation from the store.
  scroll_after       a scroll gesture against a settled thread. The gesture must still travel what
                     it commanded. A virtualizer whose row heights are estimated corrects them as
                     rows are measured, which moves the scroll target under the gesture; if that
                     correction is large enough to eat the travel, scrolling a long thread is
                     visibly broken however good the frame rate is.

WHAT THIS IS NOT. It is not a pixel comparison and it is not a substitute for looking at the
thing. It is the set of behaviours that a windowed mount breaks first, made into readings that can
be scored from a payload without a browser. An arm that passes all of it can still have changed
something nobody wrote an invariant for, and that is stated here rather than discovered later.
"""

from __future__ import annotations

from typing import Any, Optional

from .parity import MATCH, NOT_APPLICABLE, NOT_COMPARABLE, NOT_EXERCISED

#: Verdict for a behavioural invariant that moved. Distinct from the digest's DIFFER so a report
#: can never present the two as the same kind of evidence.
BROKEN = "broken"

#: How far a quantity that should be IDENTICAL may drift. 2%, the same figure the seeded-versus-
#: streamed equivalence check uses, and for the same reason: two runs of one build never produce
#: bit-identical character counts once a stream is involved.
EXACT_TOLERANCE = 0.02

#: How far the scroll extent may drift. 10%, much looser, because a windowed list computes its
#: total height from estimated row heights and corrects them as real rows are measured. Loose
#: enough to admit a correct virtualizer, nowhere near loose enough to admit one that dropped
#: rows: the failure this catches is an extent that is a FRACTION of the real one, not one that is
#: 6% out.
EXTENT_TOLERANCE = 0.10

#: How much of the thread the clipboard must carry, and how much more than the thread it may carry.
#:
#: TWO-SIDED, AND MEASURED AGAINST THE THREAD, because there are two ways to get a copy wrong and
#: the check used to see only one of them.
#:
#: The lower bound catches TRUNCATION, which is the defect this invariant was written for: a
#: windowed mount cannot select what it has not mounted, so a naive copy carries only the visible
#: fraction. Measured at 0.61 of the thread on a real 100K arm.
#:
#: The upper bound catches SUBSTITUTION, which is what a fix for the first defect turns into if
#: nobody is watching. Serialising from the message store is the right repair, but the obvious
#: serialiser is the "save this reply" one, which emits reasoning, tool-call arguments and tool
#: results -- none of which a user can select, because the panes holding them are collapsed and a
#: collapsed Radix Collapsible is not in the DOM at all. Measured at 2.16 of the thread on the same
#: arm: the truncation was fixed and the content was then wrong in the other direction.
#:
#: The gap between them is the honest allowance for two different serialisations of the same
#: content. The base arm's clipboard is the DOM's RENDERED TEXT; a store-based copy is markdown
#: SOURCE, so fences, emphasis and LaTeX delimiters exist in one and not the other. Measured at
#: ~0.9% on a scale fixture. Ten percent is generous against that and still refuses 2.16 by a
#: factor of twenty.
MIN_CLIPBOARD_COVERAGE = 0.95
MAX_CLIPBOARD_COVERAGE = 1.10

#: How much of the thread the clipboard must cover. 1.0: anything less is conversation the user
#: asked for and did not get.
CLIPBOARD_COVERAGE_REQUIRED = 1.0


def _drift(a: Any, b: Any) -> Optional[float]:
    """Proportional difference, or None when either side is missing or both are zero."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    biggest = max(abs(a), abs(b))
    if biggest == 0:
        return 0.0
    return abs(a - b) / biggest


def _check(
    name: str,
    ok: Optional[bool],
    detail: str,
    *,
    required: bool = False,
) -> dict:
    """One invariant's result.

    `required` marks a check WITHOUT WHICH THE REST CANNOT BE READ. It exists because of a hole
    this file's own tests found: when the treatment's clipboard could not be read back, the
    coverage check returned `None` (not applicable), the drift check returned `None` (one side
    missing), the base arm's own readability check returned `True` -- and the pair scored MATCH.
    An action whose entire subject went unmeasured was reporting that it was fine.

    A required check that could not be read makes the pair NOT COMPARABLE, which is the same rule
    the digest side already applies and the same principle throughout: silence is not a pass.
    """
    return {"invariant": name, "ok": ok, "detail": detail, "required": required}


def scroll_extent(base_row: dict, treat_row: dict) -> dict:
    """Does the scrollbar still describe the whole conversation on both arms?"""
    bc = base_row.get("census") or {}
    tc = treat_row.get("census") or {}
    b, t = bc.get("viewport_scroll_height"), tc.get("viewport_scroll_height")
    drift = _drift(b, t)
    if drift is None:
        return _check(
            "scroll_extent",
            None,
            f"no viewport scroll height in one of the censuses (base={b!r}, treatment={t!r})",
        )
    return _check(
        "scroll_extent",
        drift <= EXTENT_TOLERANCE,
        f"viewport scrollHeight {b} vs {t} ({drift:.1%} drift, {EXTENT_TOLERANCE:.0%} allowed)",
    )


def _expect(row: dict, key: str) -> Any:
    return (row.get("expect") or {}).get(key)


def clipboard_coverage(base_row: dict, treat_row: dict) -> list[dict]:
    """select_all_copy: did the user's copy carry the whole conversation?

    SCORED ON THE CLIPBOARD, NOT ON THE SELECTION, and the difference is the entire point.

    A windowed mount cannot SELECT what it has not mounted; `Selection.toString()` walks the DOM
    and the DOM is a window. But it can still COPY it, if the app handles the copy event and
    serialises from its message store. So a selection that shrank is not evidence of anything on
    its own, and an alarm wired to it would stay lit on a build that had fixed the data loss --
    which is how alarms get switched off.

    The base arm's number is not the reference either; the THREAD is. Each arm is asked the same
    question about itself: is what landed on the clipboard the whole conversation.
    """
    out = []
    base_clip = _expect(base_row, "clipboard_chars")
    treat_clip = _expect(treat_row, "clipboard_chars")
    for label, row, clip in (
        ("base", base_row, base_clip),
        ("treatment", treat_row, treat_clip),
    ):
        mounted, total = _expect(row, "messages_mounted"), _expect(row, "messages_total")
        if not _expect(row, "clipboard_readable"):
            # NOT A PASS. An unreadable clipboard is a surface that went unmeasured, and this is
            # the one invariant where "we could not tell" must never look like "it was fine".
            out.append(
                _check(
                    f"clipboard_readable:{label}",
                    None,
                    str(_expect(row, "clipboard_note") or "the clipboard could not be read back"),
                    required = True,
                )
            )
            continue
        fraction = _expect(row, "mounted_fraction")
        out.append(
            _check(
                f"clipboard_readable:{label}",
                clip is not None and clip > 0,
                f"{clip} characters reached the clipboard with {mounted} of {total} messages "
                f"mounted (mounted fraction {fraction})",
                required = True,
            )
        )
    # THE CHECK THAT DETECTS THE DATA LOSS, scored against THE THREAD rather than against the
    # other arm -- which is what the docstring above has always said and what the code did not do.
    #
    # Comparing the two clipboards directly, at a 2% tolerance, asks whether two different
    # serialisations of the same conversation are the same LENGTH. They are not and cannot be: the
    # base arm's clipboard is the DOM's rendered text and a store-based copy is markdown source.
    # A correct fix therefore fails that comparison, and the only way to make it pass is to widen
    # the tolerance until it stops testing anything.
    #
    # The reference is the thread's own visible text, measured by the arm that has all of it in
    # the DOM: on a fully mounted arm `Selection.toString()` over the whole thread IS the thread.
    # If neither arm mounts everything there is no reference and the pair is not comparable, which
    # is reported rather than assumed either way.
    reference = _expect(base_row, "selected_chars")
    base_full = _expect(base_row, "mounted_fraction")
    if not isinstance(reference, (int, float)) or reference <= 0 or base_full != 1:
        out.append(
            _check(
                "clipboard_carries_the_whole_thread",
                None,
                "the base arm did not mount the whole thread, so there is no measurement of how "
                "long the conversation's visible text actually is to score either clipboard "
                f"against (base selection {reference}, mounted fraction {base_full})",
                required = True,
            )
        )
        return out
    for label, clip in (("base", base_clip), ("treatment", treat_clip)):
        coverage = None if not isinstance(clip, (int, float)) else clip / reference
        out.append(
            _check(
                f"clipboard_carries_the_whole_thread:{label}",
                None
                if coverage is None
                else (MIN_CLIPBOARD_COVERAGE <= coverage <= MAX_CLIPBOARD_COVERAGE),
                f"the clipboard carried {clip} characters against a thread whose visible text is "
                f"{reference} characters"
                + (
                    ""
                    if coverage is None
                    else f" ({coverage:.3f} of it, allowed "
                    f"{MIN_CLIPBOARD_COVERAGE}-{MAX_CLIPBOARD_COVERAGE})"
                ),
                required = True,
            )
        )
    # Reported, never gated: on a windowed arm the selection is SUPPOSED to be short, and gating
    # on it would fail the fix.
    out.append(
        _check(
            "selection_shrank_as_expected",
            None,
            f"selected characters {_expect(base_row, 'selected_chars')} vs "
            f"{_expect(treat_row, 'selected_chars')} -- reported, not gated: a windowed mount "
            "cannot select what it has not mounted, and the clipboard above is what the user gets",
        )
    )
    return out


def _same_number(base_row: dict, treat_row: dict, key: str, name: str) -> dict:
    b, t = _expect(base_row, key), _expect(treat_row, key)
    drift = _drift(b, t)
    return _check(
        name,
        None if drift is None else drift <= EXACT_TOLERANCE,
        f"{key} {b} vs {t}" + ("" if drift is None else f" ({drift:.1%} drift)"),
    )


def _reopen_completed(row: dict) -> Optional[bool]:
    """Did the reopened thread finish REBUILDING, or does this row not say?

    Three values, and the third is the one that matters. `None` is a row that carries no evidence
    either way, which is what a payload written before `thread_reopen` waited on
    runtime/readiness.py looks like -- and back then `messages_after` was read off whatever was on
    screen when the store published its total, so those rows cannot support the invariant below
    either.

    `reopen_readiness.ready` first, because it is the gate's OWN verdict on the rebuilt thread.
    `expect_ok` second: on this action it is `reopen_ms is not None and after == before`, so a true
    value means the action's own assertion about the rebuild held under whatever gate that checkout
    applied.
    """
    readiness = _expect(row, "reopen_readiness")
    if isinstance(readiness, dict) and isinstance(readiness.get("ready"), bool):
        return readiness["ready"]
    ok = row.get("expect_ok")
    return ok if isinstance(ok, bool) else None


def thread_survives_reopen(base_row: dict, treat_row: dict) -> list[dict]:
    """thread_reopen: the thread came back the same length, and by the same route."""
    out = []
    for label, row in (("base", base_row), ("treatment", treat_row)):
        before, after = _expect(row, "messages_before"), _expect(row, "messages_after")
        completed = _reopen_completed(row)
        detail = f"the thread had {before} messages and came back with {after}"
        # MATCHING COUNTS ARE ONLY AN INVARIANT IF THE THREAD ACTUALLY CAME BACK.
        #
        # `messages_after` is `threadTotal()`, which is `aria-setsize` -- the store's DECLARATION of
        # how long the conversation is, published by the very first reopened row. When the rebuild
        # times out, scene/actions.py records `ran = True`, `expect_ok = False`, a null `reopen_ms`
        # and the outstanding conditions, and leaves the two counts equal because both are that same
        # declaration. Scored as equality it read as a held invariant, the route check passed
        # because the sidebar click had worked, and the pair came out MATCH with the rebuild having
        # timed out at three of eighteen messages mounted. "A declaration is not a rebuild" is the
        # defect the action itself was fixed for; this is the same defect one layer up.
        #
        # NOT COMPARABLE RATHER THAN BROKEN, AND ONLY FOR THE EQUAL CASE:
        #
        #   counts DISAGREE     BROKEN, whatever the gate said. The thread came back shorter than it
        #                       left, which is the data loss this invariant exists for, and a
        #                       readiness failure corroborates it rather than excusing it. Routing
        #                       this to NOT COMPARABLE would have downgraded the one finding the
        #                       whole action is written to catch.
        #   counts AGREE, no
        #   finished rebuild    NOT COMPARABLE. The comparison was not made: both numbers are the
        #                       same declaration read off a thread that never finished building. And
        #                       the timeout that produced it is bounded by the harness's OWN
        #                       remaining budget (a 10 s floor, a 60 s ceiling, against the 180 s the
        #                       cell's opening gate had), so calling it BROKEN would file a budget
        #                       exhaustion on a shared machine as a user-visible defect of the arm.
        #
        # The arm's failure is not lost by this: `ran = True, expect_ok = False` already excludes
        # the cell from scoring through `report/payload.py`, and the reason travels with the row.
        if before is None or after is None:
            ok: Optional[bool] = None
            required = False
        elif before != after:
            ok, required = False, False
        elif completed:
            ok, required = True, False
        else:
            ok, required = None, True
            readiness = _expect(row, "reopen_readiness")
            failed = (
                sorted(k for k, v in (readiness.get("conditions") or {}).items() if v is False)
                if isinstance(readiness, dict)
                else []
            )
            detail += (
                ", but both numbers are the total the store DECLARED and the reopened thread never "
                "reached a ready state, so nothing here says the thread came back "
                f"(outstanding {failed or 'unrecorded'})"
            )
        out.append(_check(f"reopen_keeps_every_message:{label}", ok, detail, required = required))
        # The route matters as much as the count. A row measured after a full page navigation is a
        # row about a page load; see `_click_or_navigate` in scene/actions.py.
        via = _expect(row, "reopened_via")
        out.append(
            _check(
                f"reopen_used_the_control:{label}",
                None if via is None else via == "click",
                f"the thread was reopened via {via!r}",
            )
        )
    return out


def scroll_travelled(base_row: dict, treat_row: dict) -> list[dict]:
    """scroll_after: the gesture still covers the ground it commanded, on both arms."""
    out = []
    for label, row in (("base", base_row), ("treatment", treat_row)):
        fraction = _expect(row, "travel_fraction")
        out.append(
            _check(
                f"scroll_travelled:{label}",
                None if fraction is None else fraction >= 0.9,
                f"the gesture travelled {fraction} of what it commanded",
            )
        )
    out.append(_same_number(base_row, treat_row, "bottom", "scroll_bottom_agrees"))
    return out


#: action -> the invariants that apply to it. An action absent from this table has no behavioural
#: invariant declared and is reported as UNCHECKED rather than as passing, on the same principle
#: that keeps NOT_COMPARABLE out of the pass column in the digest report.
INVARIANTS = {
    "select_all_copy": clipboard_coverage,
    "select_text": lambda b, t: [
        _same_number(b, t, "selected_chars", "selection_unchanged"),
        _same_number(b, t, "visible_chars", "visible_chars_unchanged"),
    ],
    "copy_markdown": lambda b, t: [
        _same_number(b, t, "clipboard_chars", "copy_unchanged"),
    ],
    "thread_reopen": thread_survives_reopen,
    "scroll_after": scroll_travelled,
}


def compare_behaviour(base_row: Optional[dict], treat_row: Optional[dict]) -> dict:
    """One base/treatment action pair, scored on behaviour instead of on structure.

    Returns the same verdict vocabulary the digest comparison uses, so one report can carry both:
    MATCH, BROKEN, NOT_EXERCISED, NOT_COMPARABLE, NOT_APPLICABLE.
    """
    for label, row in (("base", base_row), ("treatment", treat_row)):
        if not isinstance(row, dict):
            return {
                "verdict": NOT_COMPARABLE,
                "reason": f"the {label} arm has no row for this action",
                "checks": [],
            }
        if not row.get("ran"):
            return {
                "verdict": NOT_EXERCISED,
                "reason": f"the action did not run on the {label} arm "
                f"({row.get('reason') or 'no reason recorded'})",
                "checks": [],
            }
    assert base_row is not None and treat_row is not None
    action = base_row.get("action") or treat_row.get("action") or ""
    checks: list[dict] = [scroll_extent(base_row, treat_row)]
    rule = INVARIANTS.get(action)
    if rule is None:
        checks.append(
            _check(
                "behavioural_invariant_declared",
                None,
                f"no behavioural invariant is declared for {action!r}, so this action is "
                "UNCHECKED on a windowed arm rather than passing",
            )
        )
    else:
        got = rule(base_row, treat_row)
        checks.extend(got if isinstance(got, list) else [got])

    # A REQUIRED CHECK THAT COULD NOT BE READ VOIDS THE PAIR. See `_check`: without this, an
    # action whose entire subject went unmeasured scores MATCH on the strength of the checks that
    # happened to survive.
    unread = [c for c in checks if c.get("required") and c["ok"] is None]
    if unread:
        return {
            "verdict": NOT_COMPARABLE,
            "reason": "; ".join(f"{c['invariant']}: {c['detail']}" for c in unread),
            "checks": checks,
        }

    broken = [c for c in checks if c["ok"] is False]
    if broken:
        return {
            "verdict": BROKEN,
            "reason": "; ".join(f"{c['invariant']}: {c['detail']}" for c in broken),
            "checks": checks,
        }
    # A PASS REQUIRES AN ACTION-SPECIFIC INVARIANT TO HAVE HELD.
    #
    # The scroll extent is checked on every action and it is a property of the THREAD, not of the
    # action: it holds or fails identically across all eighteen. Counting an action as passing
    # because the thread's scrollbar was the right length would report `model_change`,
    # `image_upload` and `settings` as verified on a windowed arm when nothing whatsoever about
    # them was examined. That is the same mistake `compare_rows` exists to prevent on the digest
    # side, where an action that never ran used to contribute a match.
    specific = [c for c in checks if c["ok"] is not None and c["invariant"] != "scroll_extent"]
    if not specific:
        return {
            "verdict": NOT_APPLICABLE,
            "reason": (
                f"no behavioural invariant specific to {action!r} could be read from this payload, "
                "so this surface is UNCHECKED on a windowed arm rather than passing"
            ),
            "checks": checks,
        }
    return {"verdict": MATCH, "reason": "", "checks": checks}
