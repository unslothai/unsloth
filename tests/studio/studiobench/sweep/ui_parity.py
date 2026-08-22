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
                        one such volatile; run `parity_null_control.py --hunt` before believing a
                        wall of red.
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
from typing import Optional

if __package__ in (None, ""):  # pragma: no cover
    # Running the file directly rather than as a module, which is the first thing anyone tries.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import behaviour as B  # noqa: E402
from tests.studio.studiobench.analysis import parity as P  # noqa: E402

# The DECLARED unstable set, each entry carrying its mechanism. It lives in the studiobench
# package rather than here so that a test can require a mechanism for every entry, and so that
# this script and the null control cannot drift into two different opinions about which actions
# are trusted.
#
# Declared is not the same as true. `--null` replaces it with the set MEASURED from a base-vs-base
# run and reports every disagreement in both directions, which is the only way an entry here gets
# audited rather than inherited.
UNSTABLE_ACTIONS = frozenset(P.UNSTABLE_ACTIONS)

#: THE RUN'S OWN DECLARATION that an arm mounts a window, as `__main__.py` records it: the gate row
#: `windowed_readiness:{arm}` written when `--windowed-arm` names that side, and `readiness.mode` on
#: every cell row. Matched as literal strings rather than imported from `runtime/readiness.py`
#: because this script reads payloads written by other checkouts of the tool, where what has to be
#: matched is the string in the file and not the constant in this working tree.
WINDOWED_GATE = "windowed_readiness:"
MODE_WINDOWED = "windowed"

#: THE CELL'S OWN VERDICT THAT IT LOST MESSAGES. `runtime/session.py::record_completeness_gate`
#: writes this row, against the cell, when the traversal to the top of the thread could not
#: establish that the arm still holds the whole conversation -- the head marker never mounted, or
#: the ordinals recorded on the way up have a hole in them. `report/payload.py::excluded_from_rows`
#: reads it as a failed gate and drops the cell from the PERFORMANCE score; this file read no gate
#: rows at all except the windowed declaration, so the same cell's action rows were still scored
#: for UI parity. An arm that keeps the first page and the last one and has lost the middle then
#: retains its first and last visible windows, and its bottom rows, its scroll extent and an
#: action-specific invariant such as `select_text` can all match: `--mode auto` printed a visible
#: pass and a behavioural pass and exited 0 over a payload that already recorded the loss.
#: Matched as a literal string for the same reason `WINDOWED_GATE` is.
COMPLETENESS_GATE = "thread_complete"

#: How one action pair is scored. PER PAIR, never per payload: the readiness gate deliberately
#: permits a payload to hold fully mounted small rungs and windowed large ones, so a payload-wide
#: decision lets one windowed 100K capture suppress the structural digest on every fully mounted 1K
#: pair beside it, and an ordinary DOM regression at those rungs is then never looked for.
WINDOWED = "windowed"
STRUCTURAL = "structural"

#: The arms a pair is expected to have. Consulted BY NAME rather than by walking the rows that are
#: present, because the case that matters is the arm with NO row: an arm that died before it emitted
#: an action row leaves `sides` holding only the other one, and reading the declaration off the rows
#: present asks the surviving arm whether the missing one was windowed.
ARMS = ("base", "treatment")


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()
    ]


def arm_of(cell_id: str) -> str:
    return "treatment" if ".treatment." in cell_id else "base"


def rep_of(cell_id: str) -> str:
    """The cell WITHOUT its arm, which is what makes a base row and a treatment row a pair.

    THE RUNG IS KEPT. `make_cell_id` writes `r{rung}.{arm}.rep{n}`, and taking only the last
    segment made `r1K.base.rep0` and `r100K.base.rep0` the same key, so a payload carrying more than
    one rung silently overwrote one rung's rows with the other's -- and could then pair a 1K base
    against a 100K treatment. That is also the shape the per-pair mode decision has to see: mixed
    rungs in one payload are exactly what the windowed readiness gate permits.
    """
    parts = cell_id.split(".")
    return f"{parts[0]}.{parts[-1]}" if len(parts) >= 3 else parts[-1]


def rung_of(rep_key: str) -> str:
    """The RUNG segment of a `rep_of` key: `r100K.rep0` -> `r100K`, and `""` when there is none.

    The scope a measured noise floor is allowed to be applied at. Not the shard: a null control is
    its own output directory, so a shard-scoped floor would match nothing in the payload it was
    measured for. Not the rep either: reps are repetitions of one configuration, and pooling them
    is what turns a single flake into the several observations a floor has to be built from.
    """
    parts = rep_key.split(".")
    return parts[0] if len(parts) >= 2 else ""


def incomplete_cells(paths: list[Path]) -> dict[str, str]:
    """{cell_id: why} for every cell whose `thread_complete` gate FAILED.

    THE PAYLOAD ALREADY KNOWS. `probe_thread_completeness` scrolls the whole thread before the
    film and `record_completeness_gate` writes the verdict against the cell, so a windowed arm
    that has really lost part of the conversation says so in its own payload. Nothing in this file
    read it, and the reports below then scored every action row of that cell: the visible region
    is a window on the end of the thread and a store that kept its first and last pages still
    fills it, so the pair matched, `matched` grew, and `--mode auto` exited 0 on a run whose own
    self-check had recorded conversation loss.

    A gate row with no `cell_id` is ignored here rather than applied run-wide. Only the per-cell
    writer emits this name, and attributing a nameless one to every cell would let one malformed
    row silence a whole payload -- the mirror of the defect being fixed.
    """
    out: dict[str, str] = {}
    for path in paths:
        for r in rows(path):
            if r.get("row_type") != "gate" or r.get("passed") is not False:
                continue
            if str(r.get("name") or "") != COMPLETENESS_GATE:
                continue
            cid = str(r.get("cell_id") or "")
            if not cid:
                continue
            detail = r.get("detail") if isinstance(r.get("detail"), dict) else {}
            out[cid] = (
                f"cell {cid} FAILED its completeness gate: "
                f"{detail.get('reason') or detail.get('coverage_reason') or 'the arm is not holding the whole conversation'}"
            )
    return out


def _refused(sides: dict) -> str:
    """Why this pair carries no UI verdict, or `""`. See `incomplete_cells`."""
    for _label, row in sorted(sides.items()):
        why = row.get("_incomplete")
        if why:
            return why
    return ""


def collect(paths: list[Path], select: Optional[set] = None) -> dict:
    """{(shard, rep, action): {arm: action row}} plus a tally of what was captured at all.

    A row whose parity is missing or failed is KEPT, as the failed capture it is. Dropping it here
    would delete the evidence that the surface went unmeasured, and the comparison layer needs to
    see the failure in order to call the pair not comparable.

    The WHOLE ROW is carried, not just its digest, because `ran` is part of what the comparison
    means: a matching digest on an action that never ran is not coverage of that action.

    `select`, when given, restricts the result to those pair keys. It is how one payload gets its
    windowed pairs scored on the visible region and its fully mounted pairs scored structurally in
    the same invocation, without either report seeing pairs the other one owns.
    """
    out: dict[tuple, dict] = collections.defaultdict(dict)
    attempted = missing = 0
    for path in paths:
        shard = path.parent.name
        incomplete = incomplete_cells([path])
        for r in rows(path):
            if r.get("row_type") != "action":
                continue
            cid = r.get("cell_id") or ""
            key = (shard, rep_of(cid), r.get("action"))
            if select is not None and key not in select:
                continue
            parity = r.get("parity")
            if isinstance(parity, dict) and parity.get("parity_attempted"):
                attempted += 1
            else:
                missing += 1
            # STAMPED, NOT DROPPED, for the same reason a failed capture is kept: the comparison
            # layer has to be able to say that this pair carries no verdict, and a row deleted here
            # would leave the pair looking like an action that simply never ran.
            if cid in incomplete:
                r = dict(r)
                r["_incomplete"] = incomplete[cid]
            out[key][arm_of(cid)] = r
    return {"pairs": out, "attempted": attempted, "missing": missing}


def declared_windowed(paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """What the RUN SAID about windowing, per cell and per arm: ({cell_id: why}, {arm: why}).

    The declaration is the fallback for a pair the measurement cannot answer, and only that. It is
    never allowed to override a capture that did succeed, because the arm named by `--windowed-arm`
    still mounts its whole thread at the small rungs and those pairs are owed a structural digest.
    """
    cells: dict[str, str] = {}
    arms: dict[str, str] = {}
    for path in paths:
        for r in rows(path):
            kind = r.get("row_type")
            if kind == "gate":
                name = str(r.get("name") or "")
                if name.startswith(WINDOWED_GATE):
                    arm = name[len(WINDOWED_GATE) :] or "?"
                    arms[arm] = f"the run declared the {arm} arm windowed (gate row {name})"
            elif kind == "cell":
                readiness = r.get("readiness")
                mode = readiness.get("mode") if isinstance(readiness, dict) else None
                if mode == MODE_WINDOWED:
                    cid = str(r.get("cell_id") or "")
                    cells[cid] = f"cell {cid} was admitted by the WINDOWED readiness gate"
    return cells, arms


def _mount_measured(parity: Optional[dict]) -> bool:
    """Did this capture actually MEASURE how much of the thread was mounted?

    `thread_total > 0` is part of it. A capture that reports 0 of 0 has not observed a thread at
    all -- on the real 100K film, 20 of the treatment arm's rows read 0 of 0 after `model_change`
    took the viewport to nothing -- and `windowed_mount` answers False for it, which would score a
    declared-windowed arm structurally on the strength of a capture that saw no messages.
    """
    if not isinstance(parity, dict) or not parity.get("parity_attempted"):
        return False
    mounted, total = parity.get("mounted_messages"), parity.get("thread_total")
    if not isinstance(mounted, int) or not isinstance(total, int):
        return False
    return total > 0


def _cell_id_for(sides: dict, arm: str) -> str:
    """This pair's cell id ON `arm`, derived when that arm recorded no row.

    `make_cell_id` writes `r{rung}.{arm}.rep{n}`, so the counterpart of a row that IS present is
    the same id with the arm segment swapped. Without this an arm that never reached the film has
    no cell id to look up in the per-cell declarations, and the only declaration left is the
    run-wide one from `--windowed-arm`.
    """
    row = sides.get(arm)
    if isinstance(row, dict) and row.get("cell_id"):
        return str(row["cell_id"])
    for other in sides.values():
        parts = str(other.get("cell_id") or "").split(".")
        if len(parts) >= 3:
            parts[1] = arm
            return ".".join(parts)
    return ""


def decide_modes(paths: list[Path]) -> dict[tuple, tuple[str, str]]:
    """{(shard, rep, action): (mode, why)} -- how each action pair is to be scored, and why.

    MEASURED FIRST, DECLARED ONLY AS A FALLBACK.

      measured windowed    either arm's capture says it mounted fewer messages than the thread
                           holds. This is the reading the tool has always used.
      measured full        both arms measured their mount and neither is windowing, so the pair is
                           owed the structural digest whatever the rest of the payload does.
      neither              no usable mount measurement on this pair -- the slot was missed, or the
                           parity probe failed, or the capture saw no thread at all -- so the run's
                           own declaration decides it.

    THE FALLBACK IS THE POINT. A declared windowed run in which every capture failed used to scan
    clean, `--mode auto` picked the digest, every pair came out NOT_EXERCISED or NOT_COMPARABLE,
    and `report()` returned 0 because its no-result guard only covered NOT_APPLICABLE. An entirely
    unmeasured run reported a green structural result.
    """
    cells, arms = declared_windowed(paths)
    decided: dict[tuple, tuple[str, str]] = {}
    for key, sides in collect(paths)["pairs"].items():
        for _label, row in sorted(sides.items()):
            parity = row.get("parity")
            if P.windowed_mount(parity):
                decided[key] = (
                    WINDOWED,
                    f"{row.get('cell_id')} / {row.get('action')} mounted "
                    f"{parity.get('mounted_messages')} of {parity.get('thread_total')} messages",
                )
                break
        if key in decided:
            continue
        if len(sides) == 2 and all(_mount_measured(r.get("parity")) for r in sides.values()):
            decided[key] = (STRUCTURAL, "")
            continue
        # EITHER EXPECTED ARM, INCLUDING ONE THAT HAS NO ROW AT ALL.
        #
        # The declaration fallback used to be read off the rows this pair actually has, which
        # covers a declared-windowed arm whose captures failed and misses the case one step worse:
        # the arm failed before it emitted an action row, so `sides` holds only the other side, the
        # loop never asks about the arm that is missing, and the pair is scored structurally. In a
        # mixed-rung payload the fully mounted pairs then supply `matched > 0`, the missing windowed
        # pair is filed as structurally NOT COMPARABLE, and the command exits 0 without ever running
        # a windowed report for the rung that has no verdict at all.
        why = ""
        for label in ARMS:
            cid = _cell_id_for(sides, label)
            why = (cells.get(cid) if cid else "") or arms.get(label) or ""
            if why:
                break
        decided[key] = (
            (
                WINDOWED,
                f"DECLARED, not measured: {why}, and this pair carries no usable mount "
                "measurement on at least one arm",
            )
            if why
            else (STRUCTURAL, "")
        )
    return decided


def any_windowed(paths: list[Path]) -> Optional[str]:
    """Did either arm of this payload mount a WINDOW of the thread rather than all of it?

    Measured from `mounted_messages` and `thread_total` where those exist, and falling back on the
    run's own declaration for pairs where they do not. Detection alone was not enough: it answers
    "no window here" identically for a payload that mounted everything and for one whose captures
    all failed, and the second of those is an unmeasured run being scored as a fully mounted one.
    """
    for _key, (mode, why) in sorted(decide_modes(paths).items()):
        if mode == WINDOWED:
            return why
    return None


def behaviour_report(
    paths: list[Path],
    label: str,
    windowed: bool = True,
    select: Optional[set] = None,
) -> int:
    """The windowed arm's report: behavioural invariants instead of a structural digest.

    Printed with the reason it is being printed, every time. The one way this could mislead is by
    quietly replacing a strict check with a looser one, so the banner says outright which question
    is no longer being asked.

    `select` scores only those pair keys, which is how a payload's windowed pairs are reported here
    while its fully mounted ones go to the structural digest they are owed.
    """
    got = collect(paths, select)
    results = []
    for (shard, rep, action), sides in sorted(got["pairs"].items()):
        if "base" not in sides or "treatment" not in sides:
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "reason": f"only the {next(iter(sides))} arm recorded this action",
                        "checks": [],
                    },
                )
            )
            continue
        why = _refused(sides)
        if why:
            results.append(
                (action, shard, rep, {"verdict": P.NOT_COMPARABLE, "reason": why, "checks": []})
            )
            continue
        results.append((action, shard, rep, B.compare_behaviour(sides["base"], sides["treatment"])))

    print(f"\n{label}  (BEHAVIOURAL MODE)")
    print(f"  CLAIM: {P.CLAIM_BEHAVIOURAL}.")
    print(f"  POLICY: {P.POLICY_BY_MODE['behaviour']}.")
    # WHICH REASON, and only if it is true. Forced behavioural mode on a payload that mounts its
    # whole thread -- which is exactly how a NULL CONTROL is scored on the same scale as the
    # windowed arm it is the control for -- used to print "one arm of this payload mounts a window
    # of the thread" about a payload where neither arm does. A control that misdescribes itself in
    # its own heading is not a small thing when the heading is what gets read.
    if windowed:
        print(
            "  One arm of this payload mounts a window of the thread, so the structural DOM digest"
            "\n  is NOT APPLICABLE: it compares what is on screen, and this arm changes what is on"
            "\n  screen by design. It would report a difference on every action and prove nothing."
        )
    else:
        print(
            "  NEITHER arm of this payload mounts a window; behavioural mode was FORCED. The"
            "\n  structural digest would have been applicable here and was not run, so nothing"
            "\n  below says anything about whether the mounted messages render identically."
        )
    print(
        "  What is scored is the scroll extent plus the behaviours a windowed mount breaks first.\n"
        "  WHAT IS NOT BEING ASKED: whether the mounted messages render identically. An arm that\n"
        "  passes everything below can still have changed how a message looks."
    )
    if not results:
        print("  NO ACTION DATA in this payload.")
        return 2

    broken, unchecked, idle, blind = [], [], [], []
    matched = 0
    for action, shard, rep, r in results:
        verdict = r["verdict"]
        if verdict == B.BROKEN:
            broken.append((action, shard, rep, r))
        elif verdict == P.MATCH:
            matched += 1
        elif verdict == P.NOT_EXERCISED:
            idle.append((action, shard, rep, r))
        elif verdict == P.NOT_COMPARABLE:
            blind.append((action, shard, rep, r))
        else:
            unchecked.append((action, shard, rep, r))

    print(f"\n  {len(results)} action pairs across {len(paths)} shard(s)")
    print(f"  invariants held:            {matched}")
    print(f"  INVARIANTS BROKEN:          {len(broken)}")
    print(f"  UNCHECKED:                  {len(unchecked)}  (no invariant declared; not a pass)")
    print(f"  NOT COMPARABLE:             {len(blind)}")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")

    if broken:
        print("\n  BEHAVIOURAL INVARIANTS BROKEN -- these are user-visible, not measurement noise:")
        for action, shard, rep, r in broken:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
    else:
        print("\n  Every declared behavioural invariant held on both arms.")

    if unchecked:
        names = sorted({a for a, _s, _r, _v in unchecked})
        print(
            f"\n  UNCHECKED -- {len(unchecked)} pair(s) over {len(names)} action(s) with no "
            f"declared invariant. These surfaces carry NO verdict on this arm:"
        )
        for name in names:
            print(f"    {name}")
    if blind:
        print("\n  NOT COMPARABLE:")
        for action, shard, rep, r in blind[:8]:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
    if idle:
        names = sorted({a for a, _s, _r, _v in idle})
        print(f"\n  NOT EXERCISED: {', '.join(names)}")

    if broken:
        return 1
    if matched == 0:
        # NOTHING WAS VALIDATED, which is not the same as nothing being wrong. With every pair
        # unchecked, not comparable or never exercised, `broken` is empty and the block above has
        # just printed "Every declared behavioural invariant held on both arms" -- a sentence that
        # is technically true of an empty set and reads as a pass. This is the same false green
        # the digest path returns 2 for, and it is the more dangerous of the two here, because
        # behavioural mode is what REPLACES the digest on a windowed arm: if it silently validates
        # nothing then a windowed arm has no UI verdict at all while appearing to have passed one.
        print(
            "\n  NOTHING WAS COMPARED. Not one behavioural invariant was evaluated on any pair, "
            "so this run carries no UI verdict -- neither a pass nor a failure. Treat it as an "
            "absent result and find out why the actions did not run."
        )
        return 2
    return 0


def visible_unstable_set(null_paths: list[Path] | None) -> frozenset[tuple[str, str]]:
    """(rung, action) pairs whose VISIBLE REGION differs between two runs of the SAME build.

    A floor, measured rather than assumed, and it has to be measured separately from the digest's
    because the two ask different questions. Observed on a 100K base-vs-base control: 13 of 64
    action pairs differed inside the viewport, against 5 for the virtualization arm the control was
    run for. Without this the arm under test scores WORSE than an identical pair of builds and the
    verdict is not merely weak, it is backwards.

    The mechanism is the same one the digest already normalises around and does not fully catch:
    the rows differ at identical character counts (`7609->7609c`), which is a volatile attribute
    rather than changed content.

    KEYED BY RUNG AS WELL AS BY ACTION, AND EARNED AT THAT KEY. A bare action name was both too
    broad and too cheap: ONE differing null-control pair silenced that action for every rep and
    every rung. A payload legitimately holds several rungs -- the windowed readiness gate is
    written to permit an arm to mount everything at 1K and a window at 100K -- so transient visible
    noise on the null's 100K `model_change` pair suppressed a reproducible visible regression on
    the target's 1K `model_change` pair, and `visible_report` exited 0 over it.

    The rung is where the instability lives. How much thread there is, and where the film's slots
    land against it, is what makes an action differ against itself; that is the same argument
    `tier_of` already makes about the film's spacing, applied one level down. `P.derive_unstable`
    then supplies the observation count, so an entry needs more than one reading at the key it will
    be applied at.

    WHERE THIS DIVERGES FROM THE STRUCTURAL FLOOR, deliberately. `unstable_set` keys on the action
    alone -- but it is UNIONED with the declared `P.UNSTABLE_ACTIONS`, where every entry carries a
    written mechanism, and its derived half already refuses to call anything unstable on a single
    reading. The visible floor has no declared set behind it, so being derived is the only claim it
    can make, and a claim with no mechanism attached has to be earned at the scope it silences.

    None of this reaches the SEVERE verdicts: `visible_report` never routes an arm whose viewport
    ended empty into the floor, whatever the floor is keyed by.
    """
    if not null_paths:
        return frozenset()
    results, _got = compare_all_with(null_paths, P.compare_visible, "visible")
    by_rung: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for action, _shard, rep, r in results:
        # A pair whose action never ran on both arms is not an observation of anything, in either
        # direction. `derive_unstable` refuses to count a verdict it cannot read; this refuses to
        # hand it one it should not read.
        if r.get("_ran"):
            by_rung[rung_of(rep)].append((action, r))
    out: set[tuple[str, str]] = set()
    for rung, pairs in by_rung.items():
        for action, row in P.derive_unstable(pairs).items():
            if row["unstable"]:
                out.add((rung, action))
    return frozenset(out)


def visible_report(
    paths: list[Path],
    label: str,
    unstable: frozenset[tuple[str, str]] = frozenset(),
    select: Optional[set] = None,
) -> int:
    """VISIBLE-REGION PARITY. The verdict the off-screen exemption asks for.

    Policy: all changes preserve UI and UX idempotency, except that a difference may be accepted
    deliberately when performance improves dramatically, and a difference that exists only OFF
    SCREEN is fine by definition. The structural digest cannot express the second exemption -- it
    digests the thread on screen and off, so it fails every deferred-off-screen technique by
    construction -- so this scores the claim the policy actually cares about.

    `select` scores only those pair keys; see `decide_modes`.
    """
    results, _got = compare_all_with(paths, P.compare_visible, "visible", select)

    print(f"\n{label}  (VISIBLE-REGION MODE)")
    print(f"  CLAIM: {P.CLAIM_VISIBLE}.")
    print(f"  POLICY: {P.POLICY_BY_MODE['visible']}.")
    print(
        "  Off-screen differences are EXEMPT by policy and are not reported below. A message that\n"
        "  was visible for any part of the action is compared, in full, even if only partly on\n"
        "  screen. Messages are keyed by THREAD POSITION, so a windowed arm and a fully mounted\n"
        "  one are comparable."
    )
    if not results:
        print("  NO ACTION DATA in this payload.")
        return 2

    differing, unstable_bad, blind, idle, matched = [], [], [], [], 0
    # THE RESIDUE, PRINTED. A message that was on screen during the action and had been unmounted
    # again by the time the capture ran cannot be digested, and `compare_visible` refuses the pair
    # for it. That refusal used to be invisible: the pair returned MATCH with the ordinals tucked
    # into `not_digested` and nothing here read the key, so a rendering difference in the missing
    # message left no trace anywhere in the output. It is collected across every verdict, because a
    # DIFFER pair with a residue is also a pair whose report is incomplete.
    residue = []
    for action, shard, rep, r in results:
        if r.get("not_digested"):
            residue.append((action, shard, rep, r))
        if not r.get("_ran"):
            idle.append((action, shard, rep, r))
        elif r["verdict"] == P.NOT_COMPARABLE:
            blind.append((action, shard, rep, r))
        elif r["verdict"] == P.DIFFER:
            # A SEVERE difference is never routed into the noise floor. See compare_visible: an
            # action can be in the derived unstable set for an unrelated attribute and still be
            # the action on which one arm lost the whole thread.
            #
            # AT THIS PAIR'S OWN RUNG. An action that differs against an identical build at 100K
            # says nothing about the same action at 1K, where the thread is a fraction of the size
            # and the film's slots land somewhere else entirely.
            noise = (rung_of(rep), action) in unstable and not r.get("severe")
            (unstable_bad if noise else differing).append((action, shard, rep, r))
        else:
            matched += 1

    print(f"\n  {len(results)} action pairs across {len(paths)} shard(s)")
    print(f"  visible region matched:     {matched}")
    print(f"  VISIBLE DIFFERENCES:        {len(differing)}")
    print(
        f"  unstable actions differing: {len(unstable_bad)}  (differ against an identical build; "
        "not a verdict)"
    )
    print(f"  NOT COMPARABLE:             {len(blind)}  (never observed; not a pass)")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")
    print(
        f"  visible but NOT DIGESTED:   {len(residue)}  (unmounted before the capture; no verdict "
        "covers them)"
    )

    if differing:
        print(
            "\n  DIFFERENCES INSIDE THE VIEWPORT -- the off-screen exemption does not cover these:"
        )
        for action, shard, rep, r in differing:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
            for m in r.get("moved", [])[:4]:
                print(f"        {m}")
    elif matched:
        print(
            "\n  Every message that reached the viewport was identical on both arms.\n"
            "  Differences outside the viewport are not reported here and are exempt by policy;\n"
            "  run --mode digest for the thread-structure comparison, on screen and off, instead."
        )
    if blind:
        print("\n  NOT COMPARABLE -- these carry no verdict in either direction:")
        for action, shard, rep, r in blind[:8]:
            print(f"    {action:<26} {shard} {rep}: {r['reason']}")
    if residue:
        print(
            "\n  VISIBLE BUT NOT DIGESTED -- these messages were on screen during the action and "
            "had\n  been unmounted again before the capture, so nothing below covers them:"
        )
        for action, shard, rep, r in residue[:8]:
            print(f"    {action:<26} {shard} {rep}: ordinals {r.get('not_digested')[:8]}")
    if unstable_bad:
        names = sorted({a for a, _s, _r, _v in unstable_bad})
        print(
            f"\n  (reported, not counted) {len(unstable_bad)} pair(s) over {len(names)} action(s) "
            f"whose visible region differs between two runs of the SAME build: {', '.join(names)}"
        )
    if idle:
        names = sorted({a for a, _s, _r, _v in idle})
        print(f"\n  NOT EXERCISED: {', '.join(names)}")
    if not unstable:
        print(
            "\n  NO FLOOR WAS MEASURED. Pass --null OUTDIR of a base-vs-base run: an identical\n"
            "  pair of builds has been observed differing on 13 of 64 pairs inside the viewport,\n"
            "  so an unfloored count here can rank a real arm below two copies of the same build."
        )
    else:
        # WHICH RUNG EACH FLOOR ENTRY WAS MEASURED AT, printed, because it is also the only rung it
        # silences anything at. A floor that reads as a list of action names would look like it
        # covers the whole payload.
        keys = sorted(unstable)
        shown = ", ".join(f"{rung or '?'} {action}" for rung, action in keys[:12])
        print(
            f"\n  FLOOR: {len(keys)} (rung, action) pair(s) measured differing against an "
            f"identical build,\n  and silenced ONLY at the rung they were measured at: {shown}"
            + (f", and {len(keys) - 12} more" if len(keys) > 12 else "")
        )

    if differing:
        return 1
    if matched == 0:
        # The same false green every other mode here has already been fixed for: nothing compared
        # is not the same as nothing wrong, and it must not exit 0.
        print(
            "\n  NOTHING WAS COMPARED. Not one action pair yielded a visible-region verdict, so\n"
            "  this run carries no UI verdict at all -- neither a pass nor a failure."
        )
        return 2
    return 0


def compare_all_with(
    paths: list[Path],
    compare,
    key: str,
    select: Optional[set] = None,
) -> tuple[list[tuple], dict]:
    """[(action, shard, rep, result)] using `compare` over payload sub-object `key`."""
    got = collect(paths, select)
    results = []
    for (shard, rep, action), sides in sorted(got["pairs"].items()):
        if "base" not in sides or "treatment" not in sides:
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "moved": [],
                        "_ran": True,
                        "reason": f"only the {next(iter(sides))} arm recorded this action",
                    },
                )
            )
            continue
        why = _refused(sides)
        if why:
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {"verdict": P.NOT_COMPARABLE, "moved": [], "_ran": True, "reason": why},
                )
            )
            continue
        ran = bool(sides["base"].get("ran")) and bool(sides["treatment"].get("ran"))
        out = compare(sides["base"].get(key), sides["treatment"].get(key))
        out["_ran"] = ran
        results.append((action, shard, rep, out))
    return results, got


def compare_all(paths: list[Path], select: Optional[set] = None) -> tuple[list[tuple], dict]:
    """[(action, shard, rep, compare-result)] over every base/treatment pair found."""
    got = collect(paths, select)
    results = []
    for (shard, rep, action), sides in sorted(got["pairs"].items()):
        if "base" not in sides or "treatment" not in sides:
            # One arm never produced this row at all. Recorded rather than skipped: an action that
            # ran on one arm and not the other is itself a difference between the arms.
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "moved": [],
                        "reason": f"only the {next(iter(sides))} arm recorded this action",
                        "style_verdict": P.NOT_COMPARABLE,
                        "style_reason": "",
                    },
                )
            )
            continue
        why = _refused(sides)
        if why:
            results.append(
                (
                    action,
                    shard,
                    rep,
                    {
                        "verdict": P.NOT_COMPARABLE,
                        "moved": [],
                        "reason": why,
                        "style_verdict": P.NOT_COMPARABLE,
                        "style_reason": "",
                    },
                )
            )
            continue
        results.append((action, shard, rep, P.compare_rows(sides["base"], sides["treatment"])))
    return results, got


def unstable_set(paths: list[Path] | None) -> tuple[frozenset[str], dict, dict]:
    """The unstable set to score with, derived from a null control when one is supplied.

    DERIVED BEATS DECLARED, and the declared set is kept as the cross-check rather than thrown
    away: an entry that the null control never saw differ is costing real signal, and an action
    that differs against itself without being declared is producing noise. Both are printed.
    """
    if not paths:
        return UNSTABLE_ACTIONS, {}, {}
    results, _ = compare_all(paths)
    derived = P.derive_unstable([(a, r) for a, _s, _rep, r in results])
    checks = P.cross_check(derived, UNSTABLE_ACTIONS)
    measured = frozenset(a for a, row in derived.items() if row["unstable"])
    # UNION, not replacement. An action the null control could not reach -- `image_upload` has no
    # visible attachments button on this fixture -- would otherwise silently move from "declared
    # unstable" to "stable" on the strength of a measurement that never happened.
    return measured | UNSTABLE_ACTIONS, derived, checks


def report(
    paths: list[Path],
    label: str,
    unstable: frozenset[str],
    select: Optional[set] = None,
) -> int:
    """THREAD-STRUCTURE PARITY. `select` scores only those pair keys; see `decide_modes`."""
    results, got = compare_all(paths, select)
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
    inapplicable: list[tuple] = []
    matched = 0
    for action, shard, rep, r in results:
        if r["verdict"] == P.NOT_APPLICABLE:
            # NEITHER A PASS NOR A FAIL. The digest is the wrong question for this pair; see
            # analysis/parity.applicability. It is bucketed separately so it cannot be summed into
            # either column, and `--mode behaviour` is what answers it instead.
            inapplicable.append((action, shard, rep, [r.get("reason", "")]))
            continue
        if r["verdict"] == P.NOT_EXERCISED:
            idle.append((action, shard, rep, [r.get("reason", "")]))
            continue
        if r["verdict"] == P.NOT_COMPARABLE:
            blind.append((action, shard, rep, [r.get("reason", "")]))
            continue
        if r["style_verdict"] == P.DIFFER:
            style_bad.append((action, shard, rep, [r.get("style_reason", "")]))
        if r["verdict"] == P.MATCH:
            matched += 1
            continue
        entry = (action, shard, rep, r["moved"])
        (unstable_bad if action in unstable else stable_bad).append(entry)

    print(f"\n{label}  (STRUCTURAL MODE)")
    print(f"  CLAIM: {P.CLAIM_STRUCTURAL}.")
    print(f"  POLICY: {P.POLICY_BY_MODE['structural']}.")
    print(
        "  Under the current policy an OFF-SCREEN-ONLY difference is exempt, and this mode cannot\n"
        "  tell an off-screen difference from an on-screen one. Use --mode visible for a payload\n"
        "  that defers off-screen work deliberately."
    )
    print(f"  {len(results)} action pairs across {len(paths)} shard(s)")
    print(f"  matched:                    {matched}")
    print(f"  stable actions differing:   {len(stable_bad)}")
    print(f"  unstable actions differing: {len(unstable_bad)}  (expected to vary; not a verdict)")
    print(f"  NOT COMPARABLE:             {len(blind)}  (never measured; not a pass)")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")
    if inapplicable:
        print(
            f"  NOT APPLICABLE:             {len(inapplicable)}  (a windowed mount; the digest "
            f"cannot answer this)"
        )
        print(f"    {inapplicable[0][3][0]}")
        print("    Re-run with --mode behaviour to score these on behavioural invariants.")
    print(
        f"  style probe differing:      {len(style_bad)}  (advisory: display/visibility/"
        f"pointer-events)"
    )

    if stable_bad:
        print("\n  UI PARITY DIFFERENCES ON STABLE ACTIONS -- these need explaining:")
        for action, shard, rep, moved in stable_bad:
            print(f"    {action:<26} {shard} {rep}: {', '.join(moved[:4])}")
    elif matched == 0:
        # NOT A PASS, and it must not print like one. Not one pair produced a digest verdict, so
        # "no stable action rendered differently" would be true, reassuring and about nothing --
        # the exact shape of a check that silently does nothing, which is what the NOT COMPARABLE
        # bucket exists to prevent elsewhere in this file.
        #
        # THE GUARD USED TO REQUIRE `inapplicable`, so it only caught the windowed-mount route to
        # an empty comparison. A run whose parity probes all failed, or whose slots were all
        # missed, produced nothing but NOT COMPARABLE and NOT EXERCISED rows, no inapplicable ones,
        # and exited 0 under a heading that reads as a clean structural pass.
        print(
            f"\n  NOTHING WAS COMPARED. Not one of the {len(results)} pair(s) here yielded a "
            f"structural verdict ({len(inapplicable)} windowed, {len(blind)} not comparable, "
            f"{len(idle)} never exercised, {len(unstable_bad)} differing on actions that differ "
            "against themselves). This is not a pass."
        )
        if inapplicable:
            print("  Re-run with --mode behaviour to score the windowed pairs on invariants.")
    else:
        print(
            "\n  No stable action rendered a different THREAD STRUCTURE between the two arms.\n"
            "  Read that literally. This digest walks the thread and nothing else: it is\n"
            "  sidebar-blind and layout-blind by construction, and it never reads geometry or\n"
            "  CSS custom properties. A change confined to the sidebar, or one that moves things\n"
            "  without restructuring the thread, passes here while being invisible to it.\n"
            "  This is not a statement that the UI is unchanged."
        )

    if blind:
        print("\n  NOT COMPARABLE -- these surfaces carry no verdict in either direction:")
        for action, shard, rep, why in blind[:8]:
            print(f"    {action:<26} {shard} {rep}: {why[0]}")

    if idle:
        # Named surfaces, deduplicated: what matters is WHICH actions this run never opened, not
        # that it failed to open one of them sixteen separate times.
        names = sorted({action for action, _s, _r, _w in idle})
        print(
            f"\n  NOT EXERCISED -- {len(idle)} pair(s) over {len(names)} action(s) that did not "
            f"run. These surfaces are UNCHECKED, not unchanged:"
        )
        for name in names:
            why = next(w[0] for a, _s, _r, w in idle if a == name)
            print(f"    {name:<26} {why}")

    if style_bad:
        print("\n  (advisory) the bounded computed-style probe differed:")
        for action, shard, rep, why in style_bad[:8]:
            print(f"    {action:<26} {shard} {rep}: {why[0]}")

    if unstable_bad:
        print("\n  (reported, not counted) actions that vary between runs of any build:")
        for action, shard, rep, moved in unstable_bad[:8]:
            print(f"    {action:<26} {shard} {rep}: {', '.join(moved[:3])}")
    if stable_bad:
        return 1
    # 2, the same code the empty-payload path uses, and for the same reason: the tool was asked a
    # question it could not answer. Exiting 0 here would let CI go green on a run where the parity
    # check was structurally incapable of firing -- whether because every pair was a windowed mount
    # or because every capture failed.
    if matched == 0:
        return 2
    return 0


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
        "--mode",
        # `structural` and `behavior` are ALIASES, not extra modes. The report header prints
        # "(STRUCTURAL MODE)" and the pull request template asks for "the structural digest", so
        # a reader who follows either and types `--mode structural` should not be told it is not
        # a choice. Same for the American spelling of behaviour.
        choices = ("auto", "digest", "structural", "visible", "behaviour", "behavior"),
        default = "auto",
        help = (
            "auto (default) decides per ACTION PAIR from the payload: a fully mounted pair is "
            "scored structurally, a windowed pair is scored on the VISIBLE REGION and then on "
            "behavioural invariants, and one payload can contain both; digest forces the "
            "thread-structure comparison on every pair; visible forces the visible-region one; "
            "behaviour forces the behavioural one. `structural` is an alias for `digest` and `behavior` for `behaviour`"
        ),
    )
    args = ap.parse_args(argv)
    args.mode = {"structural": "digest", "behavior": "behaviour"}.get(args.mode, args.mode)

    # THE MODE DECISION FIRST, before the unstable set is even derived. Deriving an unstable set
    # from a null control and then not using it because the payload is windowed would print a page
    # of scoring apparatus that has no bearing on the report underneath it.
    #
    # PER ACTION PAIR, and per payload and per invocation were both too coarse:
    #
    #   per invocation  `ui_parity normal_run windowed_run` scored the NORMAL run behaviourally
    #                   too, so an ordinary DOM regression in a fully mounted arm went unreported
    #                   because an unrelated payload on the same command line was windowed.
    #   per payload     one payload holds several rungs, and the readiness gate deliberately
    #                   permits an arm to mount everything at 1K and a window at 100K. One windowed
    #                   large-rung capture then suppressed the structural digest on every fully
    #                   mounted pair beside it, at the rungs where that digest is exactly the right
    #                   question.
    #
    # `--mode visible|behaviour|digest` still forces every pair, because that is what forcing means.
    plan: list[dict] = []
    for pattern in args.payloads:
        paths = shards_of(pattern)
        if not paths:
            # Kept with no pairs: the structural loop below is what reports a pattern that matched
            # no payload, and it exits 2 for it.
            plan.append({"pattern": pattern, "paths": [], "windowed": set(), "structural": None})
            continue
        if args.mode == "digest":
            plan.append({"pattern": pattern, "paths": paths, "windowed": set(), "structural": None})
            continue
        if args.mode in ("visible", "behaviour"):
            plan.append(
                {
                    "pattern": pattern,
                    "paths": paths,
                    "windowed": set(collect(paths)["pairs"]),
                    "structural": set(),
                    "why": f"forced by --mode {args.mode}",
                    "forced": True,
                }
            )
            continue
        decided = decide_modes(paths)
        windowed = {key for key, (mode, _why) in decided.items() if mode == WINDOWED}
        plan.append(
            {
                "pattern": pattern,
                "paths": paths,
                "windowed": windowed,
                # None, not an empty set, when the payload holds no action pairs at all: `report`
                # is what says NO PARITY DATA and exits 2, and it has to be reached to say it.
                "structural": (set(decided) - windowed) if decided else None,
                "why": next((w for _k, (m, w) in sorted(decided.items()) if m == WINDOWED), ""),
            }
        )

    worst = 0
    scored_windowed = 0
    vis_unstable: Optional[frozenset[tuple[str, str]]] = None
    for entry in plan:
        win, struct = entry["windowed"], entry["structural"] or set()
        if not win and not entry.get("forced"):
            continue
        pattern, paths = entry["pattern"], entry["paths"]
        print(f"\nWINDOWED PAIRS in {pattern}: {entry.get('why') or 'none measured or declared'}")
        if entry.get("forced"):
            print(
                f"  FORCED: all {len(win)} pair(s) are scored by --mode {args.mode}, whatever the "
                "payload says about itself."
            )
        else:
            print(
                f"  MODE DECIDED PER ACTION PAIR: {len(win)} of {len(win) + len(struct)} pair(s) "
                f"are scored on the visible region and on behavioural invariants here; "
                f"{len(struct)} fully mounted pair(s) are scored structurally further down."
            )
        for key in sorted(win)[:8]:
            print(f"    windowed:   {key[0]} {key[1]} {key[2]}")
        if len(win) > 8:
            print(f"    ... and {len(win) - 8} more windowed pair(s)")
        scored_windowed += len(win)
        # BOTH, and in this order. Visible-region parity is the verdict the off-screen exemption
        # asks for and it is the one that can FAIL a windowed arm for something a user would see.
        # The behavioural invariants are the complement: they catch what a viewport comparison
        # cannot, such as a clipboard that no longer carries the thread. Neither subsumes the other,
        # so a windowed pair gets both and the run fails if either does.
        if args.mode in ("auto", "visible"):
            if vis_unstable is None:
                # The floor, derived from the null control the caller passed. Without it an
                # identical pair of builds outscores the arm under test. Derived once: it is a
                # property of the null control, not of the payload being scored against it.
                vis_null: list[Path] = []
                for pat in args.null:
                    vis_null.extend(shards_of(pat))
                vis_unstable = visible_unstable_set(vis_null)
            worst = max(
                worst,
                visible_report(paths, f"UI PARITY: {pattern}", vis_unstable, select = win),
            )
        if args.mode in ("auto", "behaviour"):
            worst = max(
                worst,
                behaviour_report(
                    paths,
                    f"UI PARITY: {pattern}",
                    windowed = args.mode == "auto" or any_windowed(paths) is not None,
                    select = win,
                ),
            )

    remaining = [e for e in plan if e["structural"] is None or e["structural"]]
    if not remaining:
        return worst
    if scored_windowed:
        # The rest still get the digest they were owed, in the same run.
        print(
            f"\n  {len(remaining)} payload(s) still hold fully mounted pairs, which are scored "
            "structurally below and not behaviourally."
        )

    null_paths: list[Path] = []
    for pattern in args.null:
        null_paths.extend(shards_of(pattern))
    unstable, derived, checks = unstable_set(null_paths or None)
    if derived:
        print(f"UNSTABLE SET DERIVED from {len(null_paths)} null-control shard(s)")
        for key, entries in checks.items():
            print(f"  {key:<32} {', '.join(entries) if entries else '(none)'}")
        print(
            f"  scoring against {len(unstable)} unstable action(s): "
            f"{', '.join(sorted(unstable))}"
        )
    else:
        print(f"UNSTABLE SET DECLARED, not measured: {', '.join(sorted(unstable))}")
        print("  pass --null OUTDIR of a base-vs-base run to derive it instead.")

    null_tiers = tier_of(null_paths)
    scored_structural = 0
    for entry in remaining:
        pattern, paths, select = entry["pattern"], entry["paths"], entry["structural"]
        if not paths:
            print(f"\nno payload found for {pattern}")
            worst = max(worst, 2)
            continue
        tiers = tier_of(paths)
        if null_tiers and tiers and null_tiers != tiers:
            print(
                f"\n  WARNING: the null control was recorded at tier {sorted(null_tiers)} and "
                f"this payload at {sorted(tiers)}. Which actions are unstable depends on the "
                f"film's slot spacing, so this unstable set does not transfer."
            )
        label = f"UI PARITY: {pattern}"
        if select is not None and entry["windowed"]:
            # WHICH PAIRS THESE ARE, in the heading. A structural section that silently covered
            # part of a payload would read as a verdict on all of it.
            label += (
                f"  ({len(select)} fully mounted pair(s) of "
                f"{len(select) + len(entry['windowed'])})"
            )
        scored_structural += len(select) if select is not None else 0
        worst = max(worst, report(paths, label, unstable, select))
    # COMBINED, and the worst outcome wins. A behavioural failure on one payload is not cancelled
    # by a structural pass on another, and a structural failure at the fully mounted rungs is not
    # cancelled by the windowed rungs passing their own two modes.
    if scored_windowed and scored_structural:
        print(
            f"\nCOMBINED EXIT STATUS {worst}: {scored_windowed} pair(s) scored on the visible "
            f"region and behavioural invariants, {scored_structural} scored structurally. Any "
            "mode's failure fails the run."
        )
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
