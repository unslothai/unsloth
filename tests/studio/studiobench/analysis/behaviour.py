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


def _extent_of(row: dict) -> tuple[Optional[float], bool]:
    """(the arm's scroll extent as `scroll_extent` measures it, was it reconstructed).

    `expect.bottom` is `scrollHeight - clientHeight`, read by `SCROLL_JS` before the gesture moves
    anything; `scroll_extent` compares `census.viewport_scroll_height`, which is `scrollHeight`.
    Same physical quantity offset by a constant, and the constant is not harmless: `_drift` is
    proportional, so subtracting a shared `clientHeight` AMPLIFIES the drift by `H / (H - C)` --
    1.087 on the 10,000 px extent and 800 px viewport measured here, so a tolerance applied to
    `bottom` is 8.7% tighter than the same number applied to the extent, and the two checks print
    two different percentages for one scrollbar.

    So the extent is reconstructed from `viewport_client_height`, which `scene/dom.js` has recorded
    in the census all along. `clientHeight` does not change while the viewport scrolls, so reading
    it from the census and `bottom` from the gesture is not mixing two instants of a moving
    quantity.

    Falls back to `bottom` when the census does not carry it -- a payload recorded before that
    field, or a census that failed -- and says which of the two it returned.
    """
    bottom = _expect(row, "bottom")
    if not isinstance(bottom, (int, float)):
        return None, False
    client = (row.get("census") or {}).get("viewport_client_height")
    if not isinstance(client, (int, float)):
        return float(bottom), False
    return float(bottom) + float(client), True


def _client_height(row: dict) -> Optional[float]:
    client = (row.get("census") or {}).get("viewport_client_height")
    return float(client) if isinstance(client, (int, float)) else None


def _bottom_of(row: dict) -> Optional[float]:
    bottom = _expect(row, "bottom")
    return float(bottom) if isinstance(bottom, (int, float)) else None


def _comparable_extents(base_row: dict, treat_row: dict) -> tuple[Any, Any, str]:
    """The two numbers `scroll_bottom_agrees` compares, and what they are.

    ONE DECISION FOR BOTH ARMS. Reconstructing per arm and comparing whatever each produced puts a
    `scrollHeight` beside a `bottom`: two arms with an identical 1,200 px `bottom` over an 800 px
    viewport, one of whose censuses failed, came out 2,000 against 1,200 and BROKEN at 40% drift,
    with a detail line that said `no client height on both arms` over the mixed pair.

    AND ONLY WHEN THE HEIGHTS AGREE. `clientHeight` is a shared offset only if it is shared. Two
    arms reporting the same 10,000 px `scrollHeight` at client heights of 800 and 2,000 have
    bottoms of 9,200 and 8,000 -- 1,200 px less room for the gesture on one of them -- and
    reconstructing both to 10,000 reports MATCH at 0.0% drift over that. `scroll_extent` already
    compares the scroll heights; what this check is for is the range the gesture actually had.

    So both arms are reconstructed together or neither is, and a fallback compares the raw bottoms
    at the same allowance, which is the quantity a differing viewport moves.
    """
    b_ext, b_full = _extent_of(base_row)
    t_ext, t_full = _extent_of(treat_row)
    b_bottom, t_bottom = _bottom_of(base_row), _bottom_of(treat_row)
    if not (b_full and t_full):
        return b_bottom, t_bottom, "bottom (no client height on both arms)"
    b_client, t_client = _client_height(base_row), _client_height(treat_row)
    if b_client != t_client:
        return (
            b_bottom,
            t_bottom,
            f"bottom (client heights differ: {b_client} vs {t_client})",
        )
    return b_ext, t_ext, "scroll extent"


def scroll_travelled(base_row: dict, treat_row: dict) -> list[dict]:
    """scroll_after: the gesture covers the ground it commanded and no more, on both arms."""
    out = []
    # THE PAIR'S REFERENCE EXTENT, NOT THE ARM'S OWN, and for the same reason `_drift` divides by
    # the larger of the two: an estimate correction is the arm closing the gap to the REAL extent,
    # so the gap is what bounds it, and the arm that is wrong is the one whose own extent is the
    # worse yardstick. Taken per arm the two checks read one tolerance off two denominators, and
    # disagreed inside it: extents of 10,000 and 9,050 pass `scroll_extent` at 9.5% drift while
    # the 950 px correction that closes that very gap came out BROKEN against a ceiling granting
    # 10% of 9,050. A false red is not free here -- it removes the cell from `readings_by_arm`,
    # takes its healthy partner with it through the arm intersection, and
    # `unmeasured_planned_cells` can then VOID the plan -- so the two now enforce the same
    # allowance on the same quantity.
    #
    # It only ever loosens: `max` is never below the arm's own extent. An arm carrying NO extent
    # still gets no ceiling rather than borrowing its partner's, because that would newly bound an
    # arm this check has always left unbounded above, which is the one direction that could invent
    # a red rather than retire one.
    reference = max(
        (
            abs(extent)
            for extent, _ in (_extent_of(base_row), _extent_of(treat_row))
            if isinstance(extent, (int, float))
        ),
        default = None,
    )
    for label, row in (("base", base_row), ("treatment", treat_row)):
        fraction = _expect(row, "travel_fraction")
        commanded = _expect(row, "commanded_px")
        travelled = _expect(row, "travelled_px")
        # THE ALLOWANCE IS A FRACTION OF THE EXTENT, so it is taken on the extent. `bottom` is
        # `scrollHeight - clientHeight`, and reading `EXTENT_TOLERANCE` off it is the same defect
        # `_extent_of` exists to fix one check below: on a 10,000 px extent behind an 800 px
        # viewport it grants 920 px where the tolerance says 1,000, so a 941 px correction -- 9.4%
        # of the extent, inside the declared 10% -- came out BROKEN.
        extent, _reconstructed = _extent_of(row)
        # BOUNDED ABOVE AS WELL AS BELOW, and the ceiling is DERIVED rather than chosen.
        #
        # The lower bound is what this invariant was written for: Studio's intent-aware autoscroll
        # snapping a programmatic move back to the bottom leaves the gesture having covered
        # nothing. But the predicate was `fraction >= 0.9` and nothing above it, so an arm whose
        # viewport moved TWICE as far as commanded passed exactly as 1.0 did and the pair returned
        # MATCH. `travelled` sums `|scrollTop_after - target_before|`, so every pixel above
        # `commanded` is the viewport being moved by something other than the gesture -- the same
        # anchor instability this action exists to detect, in the other direction.
        #
        # WHY THIS CEILING AND NOT A NUMBER PICKED TO LOOK SYMMETRIC WITH 0.9. Overshoot has one
        # legitimate source: a windowed list correcting estimated row heights, which moves the
        # offset by the error in the estimate. Those errors total the error in the arm's extent,
        # and this file already declares how far the extent may be wrong (`EXTENT_TOLERANCE`). So
        # the gesture may exceed its command by that fraction of the arm's own extent, both terms
        # read off the row. No second constant, and it scales with the rung.
        #
        # It degrades to no ceiling rather than to a guess: an arm carrying no extent gets the
        # lower bound alone, said in the detail, because a ceiling of zero fails every correct arm.
        ceiling: Optional[float] = None
        if (
            isinstance(commanded, (int, float))
            and commanded > 0
            and isinstance(extent, (int, float))
        ):
            ceiling = (commanded + EXTENT_TOLERANCE * reference) / commanded
        if fraction is None:
            ok: Optional[bool] = None
            detail = "the row records no travel fraction"
        elif ceiling is None:
            ok = fraction >= 0.9
            detail = (
                f"the gesture travelled {fraction} of what it commanded; NO CEILING was applied "
                f"because the row carries no scrollable extent to derive one from"
            )
        else:
            ok = 0.9 <= fraction <= ceiling
            detail = (
                f"the gesture travelled {fraction} of what it commanded "
                f"({travelled} of {commanded} px, allowed 0.9 to {ceiling:.3f}: "
                f"{EXTENT_TOLERANCE:.0%} of the pair's {reference} px reference extent)"
            )
        out.append(_check(f"scroll_travelled:{label}", ok, detail))
    # THE EXTENT, NOT `bottom`, AND AT THE EXTENT'S OWN ALLOWANCE. This compared `bottom` through
    # `_same_number` and so through `EXACT_TOLERANCE`, 2%, while `scroll_extent` grants the same
    # physical quantity 10% a few checks earlier and says why a correct virtualizer needs it. An
    # arm inside the declared allowance was reported behaviourally BROKEN, and a false red is not
    # free: it removes the cell from `readings_by_arm`, takes its healthy partner with it through
    # the arm intersection, and `unmeasured_planned_cells` can then VOID the plan.
    #
    # `_same_number` is deliberately NOT widened. Its other three keys -- `selected_chars`,
    # `visible_chars`, `clipboard_chars` -- are not extents and are correctly strict at 2%.
    b_ext, t_ext, what = _comparable_extents(base_row, treat_row)
    drift = _drift(b_ext, t_ext)
    out.append(
        _check(
            "scroll_bottom_agrees",
            None if drift is None else drift <= EXTENT_TOLERANCE,
            f"{what} {b_ext} vs {t_ext}"
            + ("" if drift is None else f" ({drift:.1%} drift, {EXTENT_TOLERANCE:.0%} allowed)"),
        )
    )
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
