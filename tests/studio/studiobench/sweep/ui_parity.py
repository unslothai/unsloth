# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""UI parity across the arms of a studiobench A/B: does the PR change what is on screen?

The performance question and the parity question are the same experiment run once. studiobench
already drives both arms through the same eighteen scripted actions against a byte-identical
seeded thread inside one session, so at the close of every action the two DOMs are supposed to
agree. This reads the digests taken there and reports, per action, whether they did.

WHAT A MISMATCH MEANS, AND WHAT IT DOES NOT. A differing digest is not automatically a defect and
is never reported as one. Three things produce it:

  a real UI change      the PR renders something different. This is the finding.
  a volatile the
  normaliser missed     a generated id, a rendered duration, a backend-minted record id. The fix
                        is in parity.js, and until it is fixed the reading is noise. The first
                        null control this tool ever ran failed on all eighteen actions for exactly
                        one such volatile. `sweep/parity_null_control.py --hunt`, over a null
                        control recorded with `--parity-raw`, prints the exact bytes that moved,
                        which is the only way to tell this case from a real change without
                        guessing. Run it before believing a wall of red.
  a legitimately
  divergent action      `stop_generation` stops a live stream, so how much text arrived before the
                        stop differs between two runs of ANY build.

So the output separates STABLE actions, where a mismatch is a genuine signal, from actions that
are expected to vary. A tool that reported all eighteen equally would be ignored within a day.

WHAT A PASS DOES NOT MEAN. The digest is a serialisation of the DOM. It cannot see CSS coming from
a stylesheet, computed layout, colour, typography, images or canvas pixels. `scene/parity.js`
states the full list at the top of the file, and the bounded computed-style probe reported here
covers three properties on a few dozen elements, no more. "18 actions, 0 differences" is not
"the UI is pixel-identical", and quoting it as that is the misreading this paragraph exists to
prevent.

THE THIRD OUTCOME. An action whose digest could not be captured on one side is NOT COMPARABLE, and
it is counted and printed as its own outcome rather than folded into the passes. An action that
was never measured and an action that matched produce the same silence otherwise, and telling them
apart is the entire difference between an instrument and a decoration.

    python -m tests.studio.studiobench.sweep.ui_parity outputs/mine
    python -m tests.studio.studiobench.sweep.ui_parity --null outputs/null outputs/mine
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    # Running the file directly rather than as a module, which is the first thing anyone tries.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.scoring.from_payload import latest_attempt_rows  # noqa: E402

# The DECLARED unstable set, each entry carrying its mechanism. It lives in the studiobench
# package rather than here so that a test can require a mechanism for every entry, and so that
# this script and the null control cannot drift into two different opinions about which actions
# are trusted.
#
# Declared is not the same as true. `--null` replaces it with the set MEASURED from a base-vs-base
# run and reports every disagreement in both directions, which is the only way an entry here gets
# audited rather than inherited.
UNSTABLE_ACTIONS = frozenset(P.UNSTABLE_ACTIONS)


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()
    ]


def arm_of(cell_id: str) -> str:
    return "treatment" if ".treatment." in cell_id else "base"


def rung_of(cell_id: str) -> str:
    return cell_id.split(".", 1)[0]


def collect(paths: list[Path], require_complete: bool = False) -> dict:
    """{(shard, rung, rep, session, action): {arm: action row}} plus a tally of what was captured.

    THE RUNG IS PART OF THE IDENTITY. A standard-tier run walks 1K, 10K and 100K inside one
    repetition, so keying on the repetition alone makes every action collide two rungs deep and
    only the last one read survives. A stable action that rendered differently at 1K was then
    overwritten by the matching 100K row and the gate reported success.

    THE SESSION IS PART OF IT FOR THE SAME REASON, and one shard is not one session. `--resume`
    re-runs a pair under a new session id, appending into the same shard directory, and it can
    stop again partway: a resume that re-ran the base arm and died before its partner leaves the
    payload holding one arm under the old session and the other under the new one. Keyed on the
    repetition alone those two arms pair, and the comparison is then across sessions:
    `sweep/floor_table.paired` and `scoring/ab.py` both refuse exactly that, the first for the ~8%
    session-to-session drift it charges to whichever arm was re-run, and this one because the
    digests are of two different browser sessions. A pair that differs only for that reason is
    counted by `derive_unstable` as evidence that the action is unstable, and two of them at one
    rung silence a real DOM regression at that rung for good.

    A payload recorded before session ids existed has `""` on both arms and pairs exactly as before.

    A SUPERSEDED ATTEMPT IS NOT AN OBSERVATION, so `latest_attempt_rows` drops it before any of
    that pairing happens. `runtime/ab.skippable_cells` skips a `(rung, rep)` pair only when EVERY
    arm of it completed, and re-runs BOTH arms otherwise, so an interruption leaves the payload
    holding a full base/treatment pair under the dead session AND another under the retry's. The
    scene emits its action rows before `stream:drain`, so a cell that dies draining has already
    written every digest it captured: keyed by session alone those two pairs are two comparable
    observations of ONE logical repetition, and the first of them compares a completed arm against
    an arm that was on its way down.

    That is the shape `derive_unstable` is least able to survive. One differing superseded pair
    plus one matching replacement is exactly `min_observations`, so the action is marked unstable
    at that rung on the strength of a reading the run itself threw away, a real DOM regression
    there prints under "expected to vary", and the command exits 0. `readings_by_arm` filters the
    same rows for the same reason on the timing side, and `sweep/floor_table.cell_metrics` gets it
    for free by keying completed cells on the cell id -- this was the one path left reading both.

    A row whose parity is missing or failed is KEPT, as the failed capture it is. Dropping it here
    would delete the evidence that the surface went unmeasured, and the comparison layer needs to
    see the failure in order to call the pair not comparable.

    The WHOLE ROW is carried, not just its digest, because `ran` is part of what the comparison
    means: a matching digest on an action that never ran is not coverage of that action.
    """
    out: dict[tuple, dict] = collections.defaultdict(dict)
    attempted = missing = 0
    incomplete = 0
    no_cell_rows = 0
    for path in paths:
        shard = path.parent.name
        raw_rows = rows(path)
        # A CELL THAT DID NOT FINISH IS NOT AN OBSERVATION -- ON THE NULL CONTROL ONLY, and the
        # asymmetry is the point rather than an omission.
        #
        # Action rows are emitted as the film runs; the `cell` row is written when it ends. So an
        # interrupted or in-flight cell leaves a complete-looking set of action rows that nothing
        # owns, and they pair and compare exactly like real ones.
        #
        # ON THE NULL they must be dropped. The null's job is to say which actions are unstable,
        # and an under-observed action reads as stable or undetermined instead -- which NARROWS
        # the excuse set, so the result's ordinary noise then has nothing to hide behind. Scoring
        # this sweep's own half-written film as a null control turned 0 false alarms into 5 of 30
        # for exactly that reason. `floor_table.cell_metrics` refuses the same rows on the timing
        # side.
        #
        # ON THE RESULT they must be kept, and `test_an_attempt_that_was_never_re_run_still_
        # carries_its_parity_verdict` holds that: a cell that died is the latest attempt at
        # itself, and a difference it observed is still a difference. Dropping it would silence a
        # regression, which is the worse direction of the two. So the conservative choice has
        # opposite signs on the two sides, because "conservative" means a different thing on each.
        #
        # READ THROUGH THE SAME FILTER THE ACTION ROWS ARE READ THROUGH. Taking `completed` from
        # the raw stream while taking the action rows from `latest_attempt_rows` is the guard
        # answering about a different attempt than the one it is guarding. `_resume_set` names the
        # path that gets there without anybody doing anything unusual: an A/B pair is re-run WHOLE
        # (`ab.skippable_cells`), so a resume re-runs an arm that had already succeeded, and if
        # that retry is interrupted the payload holds a completed row and a LATER, unfinished row
        # under the same deterministic `cell_id`. Read raw, the dead retry's action rows are
        # admitted on the strength of the completion the run before them earned. On the null that
        # is the worst direction: one valid repetition plus this one reaches `min_observations`,
        # the action is called unstable, and that excuse hides a real result difference.
        # Per file: the payload is append-only within one shard, and a cell id is reused across
        # shards, so superseding has to be resolved inside the stream that appended it.
        kept_rows = latest_attempt_rows(raw_rows)
        completed = {
            r.get("cell_id")
            for r in kept_rows
            if r.get("row_type") == "cell" and r.get("completed")
        }
        # A payload with no `cell` rows at all predates them, or is a fixture. Falling back is
        # right; falling back SILENTLY is how a guard stops guarding, so it is counted and said.
        # Asked of the raw stream on purpose: the question is whether this recorder ever wrote
        # cell rows, not whether the surviving attempt happened to reach one.
        has_cell_rows = any(r.get("row_type") == "cell" for r in raw_rows)
        if not has_cell_rows:
            no_cell_rows += 1
        for r in kept_rows:
            if r.get("row_type") != "action":
                continue
            if require_complete and has_cell_rows and r.get("cell_id") not in completed:
                incomplete += 1
                continue
            parity = r.get("parity")
            if isinstance(parity, dict) and parity.get("parity_attempted"):
                attempted += 1
            else:
                missing += 1
            cid = r.get("cell_id") or ""
            rep = cid.rsplit(".", 1)[-1]
            sid = str(r.get("session_id") or "")
            out[(shard, rung_of(cid), rep, sid, r.get("action"))][arm_of(cid)] = r
    return {
        "pairs": out,
        "attempted": attempted,
        "missing": missing,
        "incomplete": incomplete,
        "shards_without_cell_rows": no_cell_rows,
    }


def compare_all(paths: list[Path], require_complete: bool = False) -> tuple[list[tuple], dict]:
    """[(action, shard, cell, compare-result)] over every base/treatment pair found.

    `cell` is `rung rep`, so the two rungs of one repetition stay two observations.
    """
    got = collect(paths, require_complete = require_complete)
    results = []
    for (shard, rung, rep, sid, action), sides in sorted(got["pairs"].items()):
        cell = f"{rung} {rep}"
        if "base" not in sides or "treatment" not in sides:
            # One arm never produced this row at all. Recorded rather than skipped: an action that
            # ran on one arm and not the other is itself a difference between the arms.
            #
            # A resumed run lands here too, with the two arms under two session ids, and the
            # session is named so the reader can tell that case from a surface one build never
            # rendered. Both carry NOT_COMPARABLE, which is the honest verdict for either.
            results.append(
                (
                    action,
                    shard,
                    cell,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "moved": [],
                        "reason": f"only the {next(iter(sides))} arm recorded this action"
                        + (f" in session {sid}" if sid else ""),
                        "style_verdict": P.NOT_COMPARABLE,
                        "style_reason": "",
                    },
                )
            )
            continue
        results.append((action, shard, cell, P.compare_rows(sides["base"], sides["treatment"])))
    return results, got


def rung_of_cell(cell: str) -> str:
    """The rung half of a `compare_all` cell label, which is built as `f"{rung} {rep}"`."""
    return cell.split(" ", 1)[0]


def is_unstable(unstable: frozenset, action: str, cell: str) -> bool:
    """Is this action expected to vary AT THIS RUNG?

    Two kinds of entry live in the same set. A bare action name is a DECLARED entry and holds at
    every rung, because its mechanism is a property of the action. A `(rung, action)` tuple is a
    MEASURED entry and holds only at the rung it was measured at.
    """
    return action in unstable or (rung_of_cell(cell), action) in unstable


def unstable_label(entry) -> str:
    return entry if isinstance(entry, str) else f"{entry[1]}@{entry[0]}"


def compared_actions(paths: list[Path]) -> set[str]:
    """Actions the result put a real verdict on. Not the ones it scheduled."""
    results, _ = compare_all(paths)
    return {a for a, _s, _c, r in results if r["verdict"] in (P.MATCH, P.DIFFER)}


def actions_needing_an_excuse(paths: list[Path], min_reps: int) -> set[tuple[str, str]]:
    """(rung, action) entries of the result whose verdict the unstable set actually decides.

    SCOPED TO WHAT THE EXCUSE CHANGES, which is narrower than what the result compared, and the
    difference between a gate that is satisfiable and one that is not. Requiring the null to
    decide every action the result COMPARED sounds conservative and is not reachable: it demands
    a flawless null control. `derive_unstable` needs two comparable observations, an observation
    needs BOTH arms to reach the slot, and the arms miss slots independently, so with `--reps 2`
    there is no slack anywhere -- one missed slot on one arm in one repetition blinds that action
    for good. Measured on the run that raised this: 3 of 18 actions were lost per cell, and 3 of
    the 14 actions the result compared came back undecided in the null. Compounded over 14
    actions a clean sweep is not something a contended runner produces, so the audit failed a
    wave whose verdict, when finally run by hand, was a clean pass with zero stable differences.

    So the question is not which actions were compared but which ones the unstable set is load
    bearing for, and there are exactly two shapes. A CORROBORATED differing action is scored as a
    regression unless the unstable set excuses it. A CORROBORATED one-arm-only action is scored
    the same way, for the same reason. Everything else -- a match, or a difference seen in one
    repetition only -- is already reported without counting, so the unstable set cannot move it
    and the null owes it no opinion. `min_reps` is threaded through rather than assumed because
    the verdict's own threshold is what decides corroboration, and an audit scoped by a different
    number is auditing a verdict nobody is going to run.

    The safety direction is unchanged. Anything the unstable set can excuse still has to be
    MEASURED before it excuses it, so an undecided action that would have been waved through on
    the declared list still fails the audit. What no longer fails it is an action nothing was
    going to ask the null about.

    KEYED BY (RUNG, ACTION), because that is what `derive_unstable` decides and what
    `unstable_set` measures. Reducing the scope to bare action names re-widened it across the
    ladder: a result differing at 1K would demand the null decide that action at 100K too, where
    the result matched and no excuse was ever consulted, so one missed observation at an
    unrelated rung failed the audit again. The rung is part of the identity here for the same
    reason it is part of it there -- instability is a property of the rung, not only the action.
    """
    results, _ = compare_all(paths)
    differing = [e for e in results if e[3]["verdict"] == P.DIFFER]
    one_sided = [e for e in results if e[3]["verdict"] == P.NOT_EXERCISED and e[3].get("one_sided")]
    firm_differing, _ = corroborated(differing, min_reps)
    firm_one_sided, _ = corroborated(one_sided, min_reps)
    return {(rung_of_cell(e[2]), e[0]) for e in firm_differing + firm_one_sided}


def audit_null(
    paths: list[Path],
    allow_undecided: frozenset = frozenset(),
    scope: set[str] | None = None,
) -> tuple[int, dict]:
    """Did this base-vs-base run DECIDE the actions it exercised? Not: did it find any.

    A caller that is about to score a result against `--null` needs to know that the null control
    was capable of an opinion, because `unstable_set` falls back to the declared list when it was
    not, and prints the words "UNSTABLE SET DERIVED" either way. The obvious check -- require at
    least one measured `action@rung` entry -- is WRONG, and wrong in the worst direction: a null
    control in which every action reached `min_observations` and NONE of them differed is the best
    null control obtainable, and it emits no entries at all, because `derive_unstable` only sets
    `unstable` when something differed. That check fails exactly when the machine is quietest and
    the measurement is at its best, which trains everyone to re-run the job on its good days. Four
    consecutive nulls measured on CI runners here produced 11, 9, 10 and 0 stable differences, so
    the zero is not a hypothetical.

    So the question asked is whether each (rung, action) is DECIDED -- `undetermined` is
    `observations < min_observations`, which is the thing that actually breaks the derivation --
    and the number that differed is reported rather than required.

    UNDECIDED IS NOT ALWAYS A DEFECT. `derive_unstable` counts a pair that was not comparable or
    not exercised as blind rather than as an observation, so an action this fixture cannot perform
    at all -- `image_upload`, whose attachments button Studio never mounts without a model -- is
    permanently undecided for an honest reason. Those names are excused by `allow_undecided`, and
    every one of them is a hole, so they are printed.

    SCOPED TO WHAT THE EXCUSE CHANGES, when a scope is given, and this is the difference between
    a gate that means something and one no runner can satisfy. The null control exists to excuse
    the result's noise, so the only actions it owes an opinion on are the ones an excuse would
    move: `actions_needing_an_excuse` builds that set and the reasoning lives there. Scoping to
    what the result merely COMPARED was the first attempt and it was still unsatisfiable -- with
    `--reps 2` a single missed slot on one arm blinds an action permanently, and the run that
    raised this lost 3 of 18 actions per cell, leaving 3 of the 14 compared actions undecided on
    a wave whose verdict was a clean pass.

    Returns `(exit code, report)`. 0 decided, 1 undecided beyond the excused names, 2 no data.
    """
    results, _got = compare_all(paths, require_complete = True)
    if not results:
        return 2, {"reason": "no parity data", "decided": [], "undecided": [], "differed": []}

    by_rung: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for action, _shard, cell, r in results:
        by_rung[rung_of_cell(cell)].append((action, r))

    decided, undecided, differed, excused, out_of_scope = [], [], [], [], []
    for rung, pairs in sorted(by_rung.items()):
        for action, row in sorted(P.derive_unstable(pairs).items()):
            entry = (rung, action)
            # OUT OF SCOPE IS NOT AN EXCUSE, and conflating the two is what made the empty scope
            # read as a vacuous audit below. An excused action is one the fixture cannot perform
            # and it is a HOLE in a question that was asked. An out-of-scope action is a question
            # nobody asked, because no excuse could have changed the result's verdict for it.
            if scope is not None and entry not in scope:
                out_of_scope.append(entry)
                continue
            if row["undetermined"]:
                (excused if action in allow_undecided else undecided).append(entry)
                continue
            decided.append(entry)
            if row["unstable"]:
                differed.append(entry)

    report_ = {
        "decided": decided,
        "undecided": undecided,
        "excused": excused,
        "out_of_scope": out_of_scope,
        "differed": differed,
    }
    # NOTHING TO DECIDE IS A PASS, and only when a scope said so. A result whose every action
    # matched asks the null for no excuses at all, and there is no way to fail a question that
    # was never put. This is not the vacuous case guarded below: the scope was computed from the
    # result, so an empty one is a statement about the result rather than about the null.
    if scope is not None and not scope:
        report_["reason"] = "the result needs no excuse from this null control"
        return 0, report_
    # Everything excused is not a decided null control, it is a null control that measured
    # nothing while naming a reason for each blank. Passing it would let the excuse list grow
    # until the audit is vacuous.
    if not decided:
        report_["reason"] = "no (rung, action) reached min_observations"
        return 1, report_
    if undecided:
        report_["reason"] = "undecided actions outside the excused list"
        return 1, report_
    return 0, report_


def print_null_audit(rc: int, report_: dict, allow_undecided: frozenset) -> None:
    """The audit, said out loud, including the case where it passes with nothing measured."""
    label = {0: "DECIDED", 1: "UNDECIDED", 2: "NO DATA"}[rc]
    print(f"\nNULL CONTROL AUDIT: {label}")
    if rc == 2:
        print("  no comparable base/treatment pair was found in this payload at all.")
        print("  A null control that measured nothing cannot excuse anything, and a result")
        print("  scored against it would be scored against the declared list.")
        return
    print(f"  decided (rung, action):     {len(report_['decided'])}")
    print(f"  of which differed:          {len(report_['differed'])}  (the MEASURED unstable set)")
    print(f"  undecided:                  {len(report_['undecided'])}")
    if allow_undecided and report_["excused"]:
        print(
            f"  excused as undecided:       {len(report_['excused'])}  "
            f"({', '.join(sorted(allow_undecided))}) -- each one a hole"
        )
    if report_.get("out_of_scope"):
        # Counted apart from the excused, because it is not a hole: no excuse could have moved
        # the result's verdict on these, so the null was never asked about them.
        print(
            f"  not audited, out of scope:  {len(report_['out_of_scope'])}  "
            "(the result's verdict for these does not turn on an excuse)"
        )
    if rc == 0 and report_.get("reason") == "the result needs no excuse from this null control":
        print(
            "\n  The result matched on everything the unstable set could have moved, so it asked"
            "\n  this null control for no excuses and there was nothing for the audit to require."
        )
        return
    if rc == 0 and not report_["differed"]:
        # Said explicitly, because this is the reading a naive gate treats as breakage.
        print(
            "\n  Every action this run exercised reached the observation count and NONE of them"
            "\n  differed against itself. The measured unstable set is empty because there was"
            "\n  nothing to measure, which is the best null control obtainable, not a failure."
        )
    if rc == 1:
        if not report_["decided"]:
            print("\n  NOT ONE (rung, action) reached min_observations.")
        else:
            print("\n  These (rung, action) pairs never reached min_observations:")
            for rung, action in report_["undecided"][:12]:
                print(f"    {action}@{rung}")
        print(
            "\n  `analysis.parity.derive_unstable` needs two comparable observations of a"
            "\n  (rung, action) before it will decide anything about it, so the first thing to"
            "\n  check is --reps: a single repetition gives one observation per (rung, action)"
            "\n  and leaves every one of them undetermined. The unstable set would then come"
            "\n  back empty and the result would be scored against the DECLARED list."
        )


def unstable_set(paths: list[Path] | None) -> tuple[frozenset, dict, dict]:
    """The unstable set to score with, derived from a null control when one is supplied.

    DERIVED BEATS DECLARED, and the declared set is kept as the cross-check rather than thrown
    away: an entry that the null control never saw differ is costing real signal, and an action
    that differs against itself without being declared is producing noise. Both are printed.

    DERIVED PER RUNG, because instability is a property of the rung and not only of the action.
    `collect()` already keeps the rung in a pair's identity, and the reason it has to is the same
    reason this does: the mechanisms in `UNSTABLE_ACTIONS` are races between a scripted slot and a
    stream, and how that race lands depends on how much thread the rung mounted. The tool's own
    rung ladder says so out loud -- at 10K "the UI work disappears underneath the scene's own
    scripted timings", at 100K it does not.

    Pooling the rungs is not a smaller version of this. It is the failure mode the whole `--null`
    mechanism exists to avoid: one differing observation at 100K, which `derive_unstable` would
    call undetermined on its own, borrows the observation COUNT of the 1K and 10K pairs, clears
    `min_observations`, and silences that action at every rung. A genuine DOM regression at 1K then
    prints under "expected to vary" and the command exits 0.
    """
    if not paths:
        return frozenset(UNSTABLE_ACTIONS), {}, {}
    results, _ = compare_all(paths, require_complete = True)
    by_rung: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for action, _shard, cell, r in results:
        by_rung[rung_of_cell(cell)].append((action, r))
    derived: dict[str, dict] = {}
    measured: set[tuple[str, str]] = set()
    for rung, pairs in sorted(by_rung.items()):
        for action, row in P.derive_unstable(pairs).items():
            derived[f"{action}@{rung}"] = row
            if row["unstable"]:
                measured.add((rung, action))
    # The cross-check stays pooled ON PURPOSE. It audits the DECLARED list, whose entries claim to
    # hold at every rung, so the question it answers -- did this run ever see this action differ --
    # is an action-level question. It is advisory output and carries no verdict.
    checks = P.cross_check(
        P.derive_unstable([(a, r) for a, _s, _c, r in results]), UNSTABLE_ACTIONS
    )
    # UNION, not replacement. An action the null control could not reach -- `image_upload` has no
    # visible attachments button on this fixture -- would otherwise silently move from "declared
    # unstable" to "stable" on the strength of a measurement that never happened.
    return frozenset(measured) | frozenset(UNSTABLE_ACTIONS), derived, checks


def in_arm_repeatability(paths: list[Path]) -> tuple[set, set]:
    """A base-vs-base null taken ON THE RUNNER BEING SCORED, at no extra cost.

    Returns `(unstable, stable)` as `(rung, action)` sets. An action seen in fewer than two
    repetitions of side A is in NEITHER: undecided is not stable.

    WHY THIS EXISTS. The excuse set is measured by a different job. GitHub gives each matrix
    entry its own runner, and it does not start them together: across four consecutive waves of
    this workflow the two arms drew different runner ids every time, and the stagger between
    their start times ran from 1 second to 6 minutes 30. So the exemptions the verdict applies
    are a property of a machine and a moment that the result never touched, and the instability
    mechanisms they describe are timing races -- the one class of thing that does not transfer.
    The null cannot notice this: it is one machine measuring itself, and it agrees with itself.

    WHAT SIDE A GIVES US FOR FREE. The result arm is merge_base vs head, so side A is the SAME
    build in every repetition. Comparing side A at rep0 against side A at rep1 is therefore a
    base-vs-base comparison in the same session, on the same runner, minutes from the digests it
    is going to excuse. No extra install, no extra film, no change to what the job schedules.

    WHY IT IS THE RIGHT ANALOGUE, checked rather than argued. The cross-job null derives its set
    across ARMS within a repetition; this derives across REPETITIONS within an arm. On the null
    control's own payload, where both constructs can be computed, they name the same three
    actions at r100K -- keystroke, reasoning_toggle, scroll_during_generation -- so the axis is
    not what the measurement is picking up.
    """
    got = collect(paths, require_complete = True)
    # (shard, rung, session, action) -> {rep: side A row}
    side_a: dict[tuple, dict] = collections.defaultdict(dict)
    for (shard, rung, rep, sid, action), sides in got["pairs"].items():
        row = sides.get("base")
        if isinstance(row, dict):
            side_a[(shard, rung, sid, action)][rep] = row
    unstable: set = set()
    stable: set = set()
    for (_shard, rung, _sid, action), by_rep in side_a.items():
        reps = sorted(by_rep)
        if len(reps) < 2:
            continue
        verdicts = [
            P.compare(by_rep[reps[0]].get("parity"), by_rep[r].get("parity"))["verdict"]
            for r in reps[1:]
        ]
        # A capture that failed is blind, not agreement. Only a real MATCH earns "stable".
        if any(v == P.DIFFER for v in verdicts):
            unstable.add((rung, action))
        elif all(v == P.MATCH for v in verdicts):
            stable.add((rung, action))
    return unstable, stable


def confine_to_runner(
    unstable: frozenset,
    local_unstable: set,
    local_stable: set,
) -> tuple[frozenset, list]:
    """Drop an IMPORTED exemption the scored runner positively contradicts. Returns (set, dropped).

    ONE-SIDED ON PURPOSE, and the asymmetry is the whole safety argument. An exemption is removed
    only when this runner measured that (rung, action) and found it REPEATABLE -- two matching
    observations of one build. An action the local signal could not decide keeps its exemption,
    because "we did not look" must never read as "it is stable"; that is the direction which turns
    a quiet moment into a red job, and the false-alarm data this gate was tuned on says a null
    quieter than the result it scores is the failure mode that actually happens.

    So the only findings this can newly surface are ones where the scored runner ran the action
    twice against one build, got the same DOM both times, and then got a different DOM from head
    in both repetitions. That is not a race. That is a build difference.

    The DECLARED entries -- plain action names rather than `(rung, action)` -- are never touched.
    They are a standing claim about the app, not a measurement of a machine, so a machine cannot
    contradict them.
    """
    kept, dropped = set(), []
    for entry in unstable:
        if not isinstance(entry, tuple):
            kept.add(entry)
            continue
        if entry in local_stable and entry not in local_unstable:
            dropped.append(entry)
            continue
        kept.add(entry)
    return frozenset(kept), sorted(dropped)


def corroborated(entries: list[tuple], min_reps: int) -> tuple[list[tuple], list[tuple]]:
    """Split stable differences into those that REPEATED and those seen in one repetition only.

    A build renders the same way every time it renders. So a real UI change differs in EVERY
    repetition of an action, and a difference that shows up in one repetition and not the other
    is, by that fact alone, not a property of the build.

    This is the discriminator the false-alarm data pointed at rather than one chosen on taste.
    Over 132 scored pairs of base-vs-base films, every false alarm was a single-repetition
    difference; the injected-element probe -- one `<span>` added inside the thread root, which is
    as real as a UI change gets -- differed in every repetition of five separate actions.

    WHAT IT COSTS, said plainly: a genuine change that can only render in one repetition is
    demoted to a warning. That is a change whose visibility depends on state the film reaches
    once, and the repetitions are not independent runs of a fresh app but two passes in one
    session, so this is not free. It is printed as UNCORROBORATED rather than dropped, so the
    reading survives even when the verdict does not rest on it.
    """
    by_action: dict[tuple[str, str], list[tuple]] = collections.defaultdict(list)
    for entry in entries:
        by_action[(entry[0], rung_of_cell(entry[2]))].append(entry)
    firm, weak = [], []
    for group in by_action.values():
        # DISTINCT repetitions, not rows: one repetition seen twice is one observation.
        reps = {e[2] for e in group}
        (firm if len(reps) >= min_reps else weak).extend(group)
    return firm, weak


def report(
    paths: list[Path],
    label: str,
    unstable: frozenset,
    min_reps: int = 1,
    min_compared: int = 0,
) -> int:
    results, got = compare_all(paths)
    if not results:
        # An empty result is reported as an empty result. "No mismatches found" when nothing was
        # ever compared is the exact shape of a check that silently does nothing.
        print(
            f"\n{label}: NO PARITY DATA in {len(paths)} payload(s). "
            f"{got['missing']} action rows carried no digest. "
            f"Was this run recorded before the parity instrument existed?"
        )
        return 2

    stable_bad, unstable_bad, blind, style_bad, idle = [], [], [], [], []
    one_sided, one_sided_unstable = [], []
    matched = 0
    for action, shard, cell, r in results:
        if r["verdict"] == P.NOT_EXERCISED:
            entry = (action, shard, cell, [r.get("reason", "")])
            if r.get("one_sided"):
                # ONE ARM RAN IT AND THE OTHER COULD NOT, which is not the missed-slot case.
                # A missed slot is a runner losing a race and it costs coverage; an action that
                # RUNS on one build and cannot be performed on the other is the two builds
                # behaving differently, and it is the one regression shape that leaves no digest
                # to differ. A control that stopped opening produces exactly this and nothing
                # else, so filing it under coverage is how "the button no longer works" ships
                # green. Held to the SAME corroboration bar as a differing digest below rather
                # than failed on sight, because a contended runner can lose one arm's slot once.
                (one_sided_unstable if is_unstable(unstable, action, cell) else one_sided).append(
                    entry
                )
            else:
                idle.append(entry)
            continue
        if r["verdict"] == P.NOT_COMPARABLE:
            blind.append((action, shard, cell, [r.get("reason", "")]))
            continue
        if r["style_verdict"] == P.DIFFER:
            style_bad.append((action, shard, cell, [r.get("style_reason", "")]))
        if r["verdict"] == P.MATCH:
            matched += 1
            continue
        entry = (action, shard, cell, r["moved"])
        (unstable_bad if is_unstable(unstable, action, cell) else stable_bad).append(entry)

    # SPLIT BEFORE THE COUNTS ARE PRINTED. Printing "stable actions differing: 1" above a verdict
    # of 0 is how a reader concludes the tool is lying to them; the headline number has to be the
    # one the exit code is taken from.
    stable_bad, uncorroborated = corroborated(stable_bad, min_reps)
    one_sided, one_sided_weak = corroborated(one_sided, min_reps)

    print(f"\n{label}")
    print(f"  {len(results)} action pairs across {len(paths)} shard(s)")
    if got.get("incomplete"):
        print(
            f"  dropped:                    {got['incomplete']} action row(s) whose cell never "
            f"completed (not observations)"
        )
    if got.get("shards_without_cell_rows"):
        print(
            f"  NOT GUARDED:                {got['shards_without_cell_rows']} shard(s) carry no "
            f"cell rows, so completion could not be checked"
        )
    print(f"  matched:                    {matched}")
    print(
        f"  stable actions differing:   {len(stable_bad)}"
        + (f"  (in >= {min_reps} repetitions)" if min_reps > 1 else "")
    )
    if min_reps > 1:
        print(
            f"  uncorroborated:             {len(uncorroborated)}  (a stable action that differed "
            f"in fewer than {min_reps} repetitions; reported, not counted)"
        )
    print(
        f"  ran on ONE arm only:        {len(one_sided)}"
        + (f"  (in >= {min_reps} repetitions)" if min_reps > 1 else "")
    )
    print(f"  unstable actions differing: {len(unstable_bad)}  (expected to vary; not a verdict)")
    print(f"  NOT COMPARABLE:             {len(blind)}  (never measured; not a pass)")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")
    print(
        f"  style probe differing:      {len(style_bad)}  (advisory: display/visibility/"
        f"pointer-events)"
    )

    scored = matched + len(stable_bad) + len(uncorroborated) + len(unstable_bad)
    if min_compared and scored < min_compared:
        # NOT a detection threshold. This is the "did the film actually run" floor, and it is the
        # thing that replaces a slot-budget liveness gate for a job that reads no timings: a
        # missed slot costs COVERAGE, and coverage is what has to be defended, not punctuality.
        # Healthy runs measured here compared 22, 25 and 26 of 32 scheduled pairs; a payload
        # mutated until 24 slots were missed compared 9 and still produced a correct verdict, so
        # the floor sits between "lost some to a slow machine" and "did not run the film".
        print(
            f"\n  TOO LITTLE COMPARED: {scored} of {len(results)} pair(s) carry a verdict, "
            f"below the floor of {min_compared}. A run that compared almost nothing passes"
            f"\n  trivially, and this is that run. Nothing below is a pass."
        )
        return 3

    if one_sided:
        print(
            "\n  RAN ON ONE ARM ONLY -- one build could perform these and the other could not,"
            "\n  in every repetition. That is a difference between the builds, not lost coverage:"
        )
        for action, shard, cell, why in one_sided:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if one_sided_weak:
        print(
            f"\n  UNCORROBORATED one-arm-only -- ran on one arm in fewer than {min_reps} "
            f"repetitions; reported, not counted:"
        )
        for action, shard, cell, why in one_sided_weak[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if one_sided_unstable:
        print(
            "\n  (reported, not counted) one-arm-only on an action expected to vary between runs"
            "\n  of any build, so which arm reached its slot is a race rather than a signal:"
        )
        for action, shard, cell, why in one_sided_unstable[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if stable_bad:
        print("\n  UI PARITY DIFFERENCES ON STABLE ACTIONS -- these need explaining:")
        for action, shard, cell, moved in stable_bad:
            print(f"    {action:<26} {shard} {cell}: {', '.join(moved[:4])}")
    else:
        print("\n  No stable action rendered differently between the two arms.")

    if uncorroborated:
        # Printed in full. A build renders the same way every time, so one repetition out of two
        # is evidence of the run rather than of the build -- but it is still a reading, and a
        # reader chasing an intermittent regression needs to see it rather than be told nothing.
        print(
            f"\n  UNCORROBORATED -- differed in only one repetition, so not counted as a "
            f"regression:"
        )
        for action, shard, cell, moved in uncorroborated[:8]:
            print(f"    {action:<26} {shard} {cell}: {', '.join(moved[:3])}")

    if blind:
        print("\n  NOT COMPARABLE -- these surfaces carry no verdict in either direction:")
        for action, shard, cell, why in blind[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if idle:
        # Named surfaces, deduplicated: what matters is WHICH actions this run never opened, not
        # that it failed to open one of them sixteen separate times.
        names = sorted({action for action, _s, _c, _w in idle})
        print(
            f"\n  NOT EXERCISED -- {len(idle)} pair(s) over {len(names)} action(s) that did not "
            f"run. These surfaces are UNCHECKED, not unchanged:"
        )
        for name in names:
            why = next(w[0] for a, _s, _c, w in idle if a == name)
            print(f"    {name:<26} {why}")

    if style_bad:
        print("\n  (advisory) the bounded computed-style probe differed:")
        for action, shard, cell, why in style_bad[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if unstable_bad:
        print("\n  (reported, not counted) actions that vary between runs of any build:")
        for action, shard, cell, moved in unstable_bad[:8]:
            print(f"    {action:<26} {shard} {cell}: {', '.join(moved[:3])}")
    return 1 if (stable_bad or one_sided) else 0


def tier_of(paths: list[Path]) -> set[str]:
    """The tier(s) a payload was recorded at, from its own run_meta row.

    Instability is a property of the FILM, not only of the action. On the fast film `copy_markdown`
    opens 2.7 s after a `send_turn` and lands inside that turn's stream, so it differs against
    itself; on the standard film the same action opens 26 s later against a finished reply. An
    unstable set derived from one tier and applied to another would silence a real signal at one
    end and admit noise at the other, so the mismatch is said out loud rather than assumed away.
    """
    tiers: set[str] = set()
    for path in paths:
        for r in rows(path):
            if r.get("row_type") == "run_meta" and r.get("tier"):
                tiers.add(r["tier"])
    return tiers


def one_tier(paths: list[Path], label: str) -> set[str]:
    """The ONE tier `paths` were recorded at, refusing a set that holds more than one.

    The recorder appends, so a second run into the same `--out` leaves both films in one payload
    and `tier_of` reports both tiers. Comparing the two SETS is not enough on its own: when the
    null control and the payload have each been re-run at the other tier they hold the same two
    tiers, the sets are equal, and the mismatch warning below never fires.

    Equal is not the same as comparable here, because the harm is inside one set. `fast` and
    `standard` both walk 100K, so the two films' 100K pairs land on one rung in `unstable_set`,
    and `derive_unstable` counts them together: one differing observation from the fast film plus
    one matching observation from the standard film reads as two observations of one action, which
    is exactly `min_observations`, so the action is marked unstable at 100K. A real DOM regression
    at 100K in the payload then prints under "expected to vary" and the command exits 0.

    `sweep/floor_table.load` already refuses the same file for the same reason on the timing side.
    This is that refusal on the parity side, and it has to happen BEFORE the set is derived rather
    than after, because by then the pooling has already happened.
    """
    tiers = tier_of(paths)
    if len(tiers) > 1:
        raise SystemExit(
            f"refusing to score a {label} recorded at more than one tier: {sorted(tiers)}. "
            f"Which actions are unstable is a property of the film, and the fast and standard "
            f"films share the 100K rung, so their pairs would be pooled into one rung's "
            f"observation count. Record each tier into its own --out."
        )
    return tiers


def corpus_of(paths: list[Path]) -> set[str]:
    """The corpus hash(es) a payload was recorded against, from its own run_meta row."""
    out: set[str] = set()
    for path in paths:
        for r in rows(path):
            if r.get("row_type") == "run_meta" and r.get("corpus_hash"):
                out.add(r["corpus_hash"])
    return out


def one_corpus(paths: list[Path], label: str) -> set[str]:
    """The ONE corpus `paths` were recorded against, refusing a set that holds more than one.

    `one_tier` refuses two FILMS pooled into one reading; this refuses two CORPORA, and the harm
    is the same shape one level down. `derive_unstable` needs two observations of a (rung, action)
    before it will decide anything, and it counts observations without asking which thread they
    came from. So two runs that are each a single repetition -- neither of which can decide
    anything on its own -- pool into two observations and report DECIDED, on the strength of two
    different films of two different threads. An action that is stable on one corpus and races on
    the other is then classified from whichever pairing happened to land, and the result is scored
    against a set that was never measured on the corpus it is being applied to.

    A glob is the usual way in: `outputs/null-*` matching shards from two corpus revisions, or one
    append-only payload holding both. `sweep/floor_table.load` already refuses exactly this on the
    timing side, and the audit must not be the softer of the two.

    A payload recorded before corpus hashes existed declares none and is not refused, since there
    is nothing to disagree about.
    """
    corpora = corpus_of(paths)
    if len(corpora) > 1:
        raise SystemExit(
            f"refusing to score a {label} recorded against more than one corpus: "
            f"{sorted(corpora)}. Which actions are unstable is a property of the thread the film "
            f"drove, and `derive_unstable` pools observations without asking which corpus they "
            f"came from, so two one-repetition runs would satisfy min_observations together and "
            f"report DECIDED from two different films. Record each corpus into its own --out."
        )
    return corpora


def cross_side_mismatch(
    tiers: set[str], null_tiers: set[str], corpora: set[str], null_corpora: set[str]
) -> str:
    """Why the null control cannot be applied to this payload, or "" if it can.

    `one_tier` and `one_corpus` each refuse a pool INSIDE one side. This refuses a mismatch ACROSS
    the two sides, which is the same harm arriving by a different door: both sides can be perfectly
    single-tier and single-corpus and still have measured different things.

    The tier axis used to print a warning and then score anyway, which is the worst of the two
    options. The warning's own words were that the set does not transfer, and a set that does not
    transfer is not a weaker excuse than a real one, it is an arbitrary one. `fast` and `standard`
    both walk 100K, so a race measured on the fast film lands on exactly the rung key a standard
    film's regression would be scored under, prints beneath "expected to vary", and the command
    exits 0.

    The corpus axis has never been checked across sides at all, and it is the more likely of the
    two to happen quietly: a corpus revision lands, the result is re-recorded against it, and an
    older null control is still sitting in the directory the workflow globs. Which actions race is
    a property of the thread the film drove, so that null's set describes a thread the payload
    never rendered.

    An empty set on either side means the recorder predates that field; there is nothing to
    disagree about, so it is not refused. The caller exits 2 on a non-empty return: this is the
    tool declining to answer, not a parity failure.
    """
    for axis, mine, theirs in (
        ("tier", tiers, null_tiers),
        ("corpus", corpora, null_corpora),
    ):
        if mine and theirs and mine != theirs:
            return (
                f"the null control was recorded at {axis} {sorted(theirs)} and this payload at "
                f"{sorted(mine)}. Which actions are unstable depends on the film's slot spacing "
                f"and on the thread the film drove, so this unstable set does not transfer. "
                f"Re-record the null control alongside the payload."
            )
    return ""


def shards_of(pattern: str) -> list[Path]:
    root = Path(pattern).parent if "/" in pattern else Path(".")
    stem = Path(pattern).name
    return sorted(p / "payload.jsonl" for p in root.glob(stem) if (p / "payload.jsonl").exists())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("payloads", nargs = "+", help = "studiobench output dirs (globs allowed)")
    ap.add_argument(
        "--null",
        metavar = "OUTDIR",
        action = "append",
        default = [],
        help = "a base-vs-base run to derive the unstable set from",
    )
    ap.add_argument(
        "--min-reps",
        type = int,
        default = 1,
        dest = "min_reps",
        help = "how many DISTINCT repetitions of an action must show the same stable difference "
        "before it counts as a regression. 1 is the historical behaviour. 2 refuses to call a "
        "difference seen in one repetition of two a change to the build, which is what every "
        "measured false alarm has been",
    )
    ap.add_argument(
        "--min-compared",
        type = int,
        default = 0,
        dest = "min_compared",
        help = "fail unless at least this many action pairs carry a real verdict. Defends "
        "COVERAGE, which is what a missed slot actually costs a job that reads no timings, "
        "rather than punctuality, which costs it nothing",
    )
    ap.add_argument(
        "--audit-null",
        action = "store_true",
        dest = "audit_null",
        help = "treat the positional payload as a base-vs-base run and exit non-zero unless it "
        "DECIDED every action it exercised. Asks whether the null control was capable of an "
        "opinion, not whether it happened to find one: a null in which nothing differed is the "
        "best one obtainable and must pass",
    )
    ap.add_argument(
        "--compared-in",
        metavar = "OUTDIR",
        action = "append",
        default = [],
        dest = "compared_in",
        help = "with --audit-null: the RESULT run, so the audit only requires the null to have "
        "decided the actions whose verdict the unstable set actually changes -- the corroborated "
        "differences and the corroborated one-arm-only actions. An action nothing was going to "
        "excuse needs no excuse. Uses --min-reps, so pass the verdict's own value",
    )
    ap.add_argument(
        "--allow-undecided",
        metavar = "ACTIONS",
        dest = "allow_undecided",
        default = "",
        help = "comma-separated action names --audit-null may leave undecided. Only for an "
        "action the fixture genuinely cannot perform, and say which: every name here is a hole",
    )
    args = ap.parse_args(argv)

    if args.audit_null:
        allow = frozenset(a.strip() for a in args.allow_undecided.split(",") if a.strip())
        scope = None
        if args.compared_in:
            scope_paths: list[Path] = []
            for pattern in args.compared_in:
                scope_paths.extend(shards_of(pattern))
            if not scope_paths:
                print(f"--compared-in matched no payload: {args.compared_in}")
                return 2
            scope = actions_needing_an_excuse(scope_paths, args.min_reps)
            named = ", ".join(f"{action}@{rung}" for rung, action in sorted(scope))
            print(
                f"auditing against the {len(scope)} (rung, action) pair(s) whose verdict the "
                f"unstable set decides, of the {len(compared_actions(scope_paths))} action(s) "
                f"the result compared: {named or '(none -- nothing needs an excuse)'}"
            )
        worst = 0
        for pattern in args.payloads:
            paths = shards_of(pattern)
            if not paths:
                print(f"\nno payload found for {pattern}")
                worst = max(worst, 2)
                continue
            one_tier(paths, "null control")
            one_corpus(paths, "null control")
            rc, report_ = audit_null(paths, allow, scope)
            print(f"\nauditing {pattern} as a null control")
            print_null_audit(rc, report_, allow)
            worst = max(worst, rc)
        return worst

    null_paths: list[Path] = []
    for pattern in args.null:
        null_paths.extend(shards_of(pattern))
    null_tiers = one_tier(null_paths, "null control")
    null_corpora = one_corpus(null_paths, "null control")
    unstable, derived, checks = unstable_set(null_paths or None)
    if derived:
        print(f"UNSTABLE SET DERIVED from {len(null_paths)} null-control shard(s)")
        for key, entries in checks.items():
            print(f"  {key:<32} {', '.join(entries) if entries else '(none)'}")
        print(
            f"  scoring against {len(unstable)} unstable entr(ies), `action@rung` where the "
            f"instability was MEASURED at one rung: "
            f"{', '.join(sorted(unstable_label(e) for e in unstable))}"
        )
    else:
        print(
            f"UNSTABLE SET DECLARED, not measured: "
            f"{', '.join(sorted(unstable_label(e) for e in unstable))}"
        )
        print("  pass --null OUTDIR of a base-vs-base run to derive it instead.")

    worst = 0
    for pattern in args.payloads:
        paths = shards_of(pattern)
        if not paths:
            print(f"\nno payload found for {pattern}")
            worst = max(worst, 2)
            continue
        tiers = one_tier(paths, "payload")
        corpora = one_corpus(paths, "payload")
        mismatch = cross_side_mismatch(tiers, null_tiers, corpora, null_corpora)
        if mismatch:
            print(f"\n  REFUSING to score {pattern}: {mismatch}")
            worst = max(worst, 2)
            continue
        # CONFINED TO THE RUNNER BEING SCORED, and only ever downward. The exemptions above were
        # measured in the other matrix job, on another machine, minutes away; side A of THIS
        # payload is the same build in every repetition, so it can say whether this machine
        # reproduces them. One that it positively contradicts is dropped, one it could not decide
        # is kept.
        effective = unstable
        if derived:
            local_unstable, local_stable = in_arm_repeatability(paths)
            effective, dropped = confine_to_runner(unstable, local_unstable, local_stable)
            if dropped:
                print(
                    f"\n  {len(dropped)} imported exemption(s) DROPPED: this runner ran "
                    f"{'them' if len(dropped) > 1 else 'it'} twice against one build and got the "
                    f"same DOM, so the null's race did not reproduce here: "
                    f"{', '.join(unstable_label(e) for e in dropped)}"
                )
        worst = max(
            worst,
            report(paths, f"UI PARITY: {pattern}", effective, args.min_reps, args.min_compared),
        )
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
