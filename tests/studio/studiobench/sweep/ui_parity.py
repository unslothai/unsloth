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
from typing import Any, Optional

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.analysis import behaviour as B  # noqa: E402
from tests.studio.studiobench.analysis import parity as P  # noqa: E402
from tests.studio.studiobench.runtime.ab import gate_detail_is_unmeasured  # noqa: E402
from tests.studio.studiobench.scoring.from_payload import (  # noqa: E402
    ATTEMPT_ROW_TYPES,
    latest_attempt_rows,
)

# The DECLARED unstable set, each entry carrying its mechanism. It lives in the studiobench
# package so a test can require a mechanism per entry and this script cannot drift from the null
# control. Declared is not true: `--null` replaces it with the MEASURED set.
UNSTABLE_ACTIONS = frozenset(P.UNSTABLE_ACTIONS)

#: THE RUN'S OWN DECLARATION that an arm mounts a window, as `__main__.py` records it. Matched as
#: literal strings rather than imported, because this script reads payloads from other checkouts.
# From `runtime/readiness.py`.
WINDOWED_GATE = "windowed_readiness:"
MODE_WINDOWED = "windowed"

#: THE CELL'S OWN VERDICT THAT IT LOST MESSAGES, written by `record_completeness_gate` when the
#: traversal could not establish the arm still holds the whole conversation. `excluded_from_rows`
#: drops such a cell from the PERFORMANCE score, but this file read no gate rows except the
#: windowed declaration, so an arm that lost the middle still printed a pass and exited 0.
COMPLETENESS_GATE = "thread_complete"

#: THE CELL'S OWN VERDICT THAT IT STOPPED FOLLOWING THE REPLY IT WAS MEASURING, written when the
#: thread did not stay pinned for enough of the streaming phase. It invalidates the same thing as
#: the completeness gate: a reply scrolled out and unmounted stops costing anything to render.
FOLLOW_GATE = "follows_the_stream"

#: The per-cell gates whose failure means this cell's action rows are not a reading of the build.
#: A gate that only qualifies one column (`timer_clamp`) is deliberately NOT here.
INVALIDATING_GATES: tuple[str, ...] = (COMPLETENESS_GATE, FOLLOW_GATE)

#: How each is named and what its failure means, for the refusal a reader actually sees. Described
#: rather than spelled with its row name, because a person reads this deciding if a run is usable.
_GATE_LABEL: dict[str, str] = {
    COMPLETENESS_GATE: "completeness",
    FOLLOW_GATE: "stream-follow",
}
_GATE_REASON: dict[str, str] = {
    COMPLETENESS_GATE: "the arm is not holding the whole conversation",
    FOLLOW_GATE: "the arm stopped following the streamed reply",
}

#: How one action pair is scored. PER PAIR, never per payload: the readiness gate permits fully
#: mounted small rungs beside windowed large ones, so a payload-wide decision lets one windowed
#: 100K capture suppress the structural digest on every 1K pair.
WINDOWED = "windowed"
STRUCTURAL = "structural"

#: The arms a pair is expected to have, consulted BY NAME rather than by walking the rows present:
#: an arm that died before emitting an action row leaves `sides` holding only the other one, and
#: reading off the rows asks the surviving arm whether the missing one was windowed.
ARMS = ("base", "treatment")


class Outcome(int):
    """An exit code that carries WHICH PAIRS it was reached over.

    `--min-compared` used to be a parameter of `report()` alone, and `report()` sits below `main`'s
    structural early return. A run with no fully mounted pair -- every pair windowed, or a forced
    `--mode visible` / `--mode behaviour` -- therefore applied no coverage floor at all, and the
    workflow's `--min-compared 16` passed on one matched capture. Two blank pages have identical
    digests, so that floor is the guard against the easiest possible false green.

    AN `int` SUBCLASS, DELIBERATELY. PLEASE DO NOT SIMPLIFY THIS BACK TO A PLAIN RETURN. `main` does
    `worst = max(worst, ...)` and the selftests assert `== 0`/`== 1`/`== 2`/`== 3` directly against
    `report`, `visible_report` and `behaviour_report` in several dozen places. A tuple or dataclass
    would mean rewriting all of those in bulk to say what they already say, and this change is
    precisely about a gate nobody re-derived.

    PAIR KEYS RATHER THAN A COUNT: `--mode auto` scores the same windowed pairs on the visible
    region AND on the invariants, so summing would double them and let a floor of 16 pass on 8.

    The key is `(action, shard, cell)`. `latest_attempt_rows` has already reduced a cell to one
    attempt by the time `collect` builds these, so the session term dropped from `cell` cannot merge
    two live observations; if it ever did, the union would UNDER-count, which is a loud false red
    rather than a quiet false green.
    """

    compared: frozenset[tuple[str, str, str]]
    seen: frozenset[tuple[str, str, str]]

    def __new__(
        cls,
        rc: int,
        compared: frozenset[tuple[str, str, str]] = frozenset(),
        seen: frozenset[tuple[str, str, str]] = frozenset(),
    ) -> "Outcome":
        self = super().__new__(cls, rc)
        self.compared = compared
        self.seen = seen
        return self

    @property
    def total(self) -> int:
        """Pairs this report was offered, compared or not: the floor message's denominator."""
        return len(self.seen)


def _floored(
    worst: int, compared: dict[str, set[tuple]], seen: dict[str, set[tuple]], min_compared: int
) -> int:
    """`main`'s application of the coverage floor, once per payload pattern it scored.

    UNIONED ACROSS MODES, not checked per mode. One film's coverage is split three ways -- visible
    region, invariants, structural digest -- and a floor applied inside each report separately fails
    a run that compared plenty. `Outcome` says why the pairs are unioned rather than added.

    PER PATTERN, not per invocation, and that is the other half of the same rule. The three modes
    are one film seen three ways; two positional payloads are two films. Pooling those lets each
    satisfy the other's floor, so two payloads comparing 8 pairs apiece cleared `--min-compared 16`
    and a nearly empty film was hidden by coverage from an unrelated one on the same command line.
    `ui_parity normal_run windowed_run` is a supported invocation -- see the mode decision, which
    already rejected per-invocation as too coarse for exactly this shape.

    RAISED, NEVER LOWERED. A shortfall can only make the verdict worse: a run that already found a
    difference keeps that finding and gains the reason its coverage is too thin to trust either way.
    """
    for pattern in sorted(compared):
        shortfall = coverage_shortfall(
            len(compared[pattern]), len(seen.get(pattern) or ()), min_compared
        )
        if not shortfall:
            continue
        print(f"\n  {pattern}: {shortfall}")
        worst = max(worst, 3)
    return worst


def _keys(results: list[tuple]) -> frozenset[tuple[str, str, str]]:
    """The `(action, shard, cell)` key of every pair a report was offered. See `Outcome`."""
    return frozenset((action, shard, cell) for action, shard, cell, _r in results)


def coverage_shortfall(scored: int, of_pairs: int, min_compared: int) -> str:
    """The TOO LITTLE COMPARED message, or `""`. One rule, one wording, three call sites.

    NOT a detection threshold. This is the "did the film actually run" floor, and it is the
    thing that replaces a slot-budget liveness gate for a job that reads no timings: a
    missed slot costs COVERAGE, and coverage is what has to be defended, not punctuality.
    Healthy runs measured here compared 22, 25 and 26 of 32 scheduled pairs; a payload
    mutated until 24 slots were missed compared 9 and still produced a correct verdict, so
    the floor sits between "lost some to a slow machine" and "did not run the film".
    """
    if not min_compared or scored >= min_compared:
        return ""
    return (
        f"TOO LITTLE COMPARED: {scored} of {of_pairs} pair(s) carry a verdict, "
        f"below the floor of {min_compared}. A run that compared almost nothing passes"
        f"\n  trivially, and this is that run. Nothing below is a pass."
    )


def rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()
    ]


def arm_of(cell_id: str) -> str:
    return "treatment" if ".treatment." in cell_id else "base"


def rung_of(cell_id: str) -> str:
    """The RUNG segment of a cell id: `r100K.base.rep0` -> `r100K`.

    THE SCOPE A MEASURED NOISE FLOOR IS ALLOWED TO BE APPLIED AT, and half of a pair's identity.
    Not the shard: a null control is its own output directory, so a shard-scoped floor would match
    nothing in the payload it was measured for. Not the rep either: reps are repetitions of one
    configuration, and pooling them is what turns a single flake into the several observations a
    floor has to be built from.
    """
    return cell_id.split(".", 1)[0]


def incomplete_cells(paths: list[Path]) -> dict[str, str]:
    """{cell_id: why} for every cell that FAILED one of the gates in `INVALIDATING_GATES`.

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

    AND ONLY THE ATTEMPT THAT SURVIVED. `--resume` re-runs a cell that did not complete under the
    SAME `cell_id`, because `make_cell_id` is deterministic, so a payload can hold a failed gate
    from the attempt that died and a passing one from the retry that worked. `latest_attempt_rows`
    already drops the superseded attempt everywhere else, but its `ATTEMPT_ROW_TYPES` is
    `{cell, action, window}` and a gate is none of those, so scanning raw rows here kept the dead
    attempt's failure forever: `collect` then stamped `_incomplete` on the RETRY's action rows and
    `_refused` withheld a verdict from a cell that had been successfully re-measured. An attempt is
    `(cell_id, session_id)`, and the winner is named FROM THE SAME ATTEMPT-KEYED ROW TYPES
    `latest_attempt_rows` reads rather than from the terminal `cell` row alone. That row is written
    in `CellRunner.run`'s `finally`, which a SIGKILL or an OOM kill never reaches, while the
    Recorder has already fsynced every gate and action row before it. So a resume hard-killed
    inside a cell left the OLDER, completed session named as the winner: the new attempt's own
    failed gate was discarded here while `latest_attempt_rows` kept that same attempt's action
    rows, and `collect` scored a cell whose own self-check had recorded conversation loss with no
    `_incomplete` stamp on it -- a visible and behavioural pass, and exit 0, over exactly the
    payload this function exists to refuse. A gate with no session id is still honoured, for the
    same reason that function keeps unstamped rows: a payload written before the recorder stamped
    them cannot be split into attempts, and ignoring it would lose a real refusal.
    """
    out: dict[str, str] = {}
    for path in paths:
        payload = rows(path)
        winner: dict[str, Any] = {}
        for r in payload:
            if r.get("row_type") in ATTEMPT_ROW_TYPES and r.get("cell_id") is not None:
                winner[str(r.get("cell_id"))] = r.get("session_id")
        for r in payload:
            if r.get("row_type") != "gate" or r.get("passed") is not False:
                continue
            name = str(r.get("name") or "")
            if name not in INVALIDATING_GATES:
                continue
            cid = str(r.get("cell_id") or "")
            if not cid:
                continue
            keep = winner.get(cid)
            if keep is not None and r.get("session_id") not in (None, keep):
                continue
            detail = r.get("detail") if isinstance(r.get("detail"), dict) else {}
            # NOT MEASURED IS NOT FAILED, drawn where the performance side draws it:
            # `gate_detail_is_unmeasured` is THE definition, imported rather than restated. It matters most
            # for the stream-coverage clause, identical on both arms and untouched by a DOM digest, which
            # refused 32 of 32 pairs here including the null control.
            if gate_detail_is_unmeasured(detail):
                continue
            # FIRST FAILURE WINS, so a cell that failed both is not relabelled by whichever row comes second
            # in the file.
            if cid in out:
                continue
            out[cid] = (
                f"cell {cid} FAILED its {_GATE_LABEL[name]} gate: "
                f"{detail.get('reason') or detail.get('coverage_reason') or _GATE_REASON[name]}"
            )
    return out


def _refused(sides: dict) -> str:
    """Why this pair carries no UI verdict, or `""`. See `incomplete_cells`."""
    for _label, row in sorted(sides.items()):
        why = row.get("_incomplete")
        if why:
            return why
    return ""


def collect(
    paths: list[Path],
    select: Optional[set] = None,
    require_complete: bool = False,
) -> dict:
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

    `select`, when given, restricts the result to those pair keys. It is how one payload gets its
    windowed pairs scored on the visible region and its fully mounted pairs scored structurally in
    the same invocation, without either report seeing pairs the other one owns.
    """
    out: dict[tuple, dict] = collections.defaultdict(dict)
    attempted = missing = 0
    # TWO DIFFERENT "INCOMPLETE"S MEET HERE: `gate_failures` is per cell from a gate the payload
    # FAILED and those rows are kept, while `dropped_incomplete` counts action rows discarded because
    # their cell never wrote a `cell` row. One shared name would let the first overwrite the second.
    # Action rows come from `latest_attempt_rows`.
    # Which only happens under `require_complete`.
    dropped_incomplete = 0
    no_cell_rows = 0
    for path in paths:
        shard = path.parent.name
        gate_failures = incomplete_cells([path])
        raw_rows = rows(path)
        # A CELL THAT DID NOT FINISH IS NOT AN OBSERVATION -- ON THE NULL CONTROL ONLY. Action rows are
        # emitted as the film runs and the `cell` row when it ends, so an interrupted cell leaves a
        # complete-looking set nothing owns. ON THE NULL they must be dropped: an under-observed action
        # reads as stable and NARROWS the excuse set (0 false alarms became 5 of 30). ON THE RESULT they
        # must be kept, because a difference a dead cell observed is still a difference. Read through the
        # SAME filter as the action rows, or the guard answers about a different attempt. Per file, since
        # a cell id is reused across shards, so superseding has to be resolved inside the stream that
        # appended it. The keep-on-the-result direction is held by
        # test_an_attempt_that_was_never_re_run_still_carries_its_parity_verdict.
        kept_rows = latest_attempt_rows(raw_rows)
        completed = {
            r.get("cell_id")
            for r in kept_rows
            if r.get("row_type") == "cell" and r.get("completed")
        }
        # A payload with no `cell` rows predates them, or is a fixture. Falling back is right; falling
        # back SILENTLY is how a guard stops guarding, so it is counted and said. Asked of the raw
        # stream: the question is whether this recorder ever wrote cell rows.
        has_cell_rows = any(r.get("row_type") == "cell" for r in raw_rows)
        if not has_cell_rows:
            no_cell_rows += 1
        for r in kept_rows:
            if r.get("row_type") != "action":
                continue
            cid = r.get("cell_id") or ""
            rep = cid.rsplit(".", 1)[-1]
            sid = str(r.get("session_id") or "")
            key = (shard, rung_of(cid), rep, sid, r.get("action"))
            # BEFORE the tally, so a selected report's counts are counts of the pairs it scored.
            if select is not None and key not in select:
                continue
            # And before the tally for the same reason, but after the selection: a row this report never
            # asked for is not an incomplete observation of it.
            if require_complete and has_cell_rows and r.get("cell_id") not in completed:
                dropped_incomplete += 1
                continue
            parity = r.get("parity")
            if isinstance(parity, dict) and parity.get("parity_attempted"):
                attempted += 1
            else:
                missing += 1
            # STAMPED, NOT DROPPED: the comparison layer has to be able to say this pair carries no verdict,
            # and a deleted row would leave the pair looking like an action that never ran.
            if cid in gate_failures:
                r = dict(r)
                r["_incomplete"] = gate_failures[cid]
            out[key][arm_of(cid)] = r
    return {
        "pairs": out,
        "attempted": attempted,
        "missing": missing,
        "incomplete": dropped_incomplete,
        "shards_without_cell_rows": no_cell_rows,
    }


def declared_windowed(
    paths: list[Path],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    """What the RUN SAID about windowing: ({(shard, cell_id): why}, {(shard, arm): why}).

    The declaration is the fallback for a pair the measurement cannot answer, and only that. It is
    never allowed to override a capture that did succeed, because the arm named by `--windowed-arm`
    still mounts its whole thread at the small rungs and those pairs are owed a structural digest.

    SCOPED TO THE SHARD THAT DECLARED IT, for the same reason `incomplete_cells` is read per path.
    One glob pools separate runs -- that is what `outputs/sbench_*` is for -- and `cell_id` and arm
    label repeat identically in every one of them, so keying on those alone let a
    `--windowed-arm treatment` declaration from one run become the fallback for an unmeasured pair
    in an ordinary run beside it. The ordinary run's pairs were then scored on the visible region
    and on behavioural invariants, the structural digest they were owed never ran, and a plain DOM
    regression in a payload that declared nothing exited 0 because of a flag passed to a different
    run. A payload recorded before `mounted_messages` existed carries no mount measurement at all,
    so every one of its pairs takes that fallback.
    """
    cells: dict[tuple[str, str], str] = {}
    arms: dict[tuple[str, str], str] = {}
    for path in paths:
        shard = path.parent.name
        for r in rows(path):
            kind = r.get("row_type")
            if kind == "gate":
                name = str(r.get("name") or "")
                if name.startswith(WINDOWED_GATE):
                    arm = name[len(WINDOWED_GATE) :] or "?"
                    arms[(shard, arm)] = (
                        f"the run declared the {arm} arm windowed (gate row {name})"
                    )
            elif kind == "cell":
                readiness = r.get("readiness")
                mode = readiness.get("mode") if isinstance(readiness, dict) else None
                if mode == MODE_WINDOWED:
                    cid = str(r.get("cell_id") or "")
                    cells[(shard, cid)] = f"cell {cid} was admitted by the WINDOWED readiness gate"
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
    """{(shard, rung, rep, session, action): (mode, why)} -- how each pair is scored, and why.

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
        # EITHER EXPECTED ARM, INCLUDING ONE WITH NO ROW AT ALL. Reading the declaration off the rows a
        # pair has misses the case one step worse: the arm failed before emitting an action row, so the
        # loop never asks about it, the pair is scored structurally, and the command exits 0 with no
        # verdict for that rung. And only this pair's own shard declares it.
        # See `declared_windowed`.
        why = ""
        shard = key[0]
        for label in ARMS:
            cid = _cell_id_for(sides, label)
            why = (cells.get((shard, cid)) if cid else "") or arms.get((shard, label)) or ""
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


def build_differences(results: list[tuple], min_reps: int) -> dict[str, list[tuple]]:
    """The two shapes that are a difference between the BUILDS and leave no capture to differ.

    Shared by `visible_report` and `behaviour_report` so the windowed verdict cannot come to a
    different opinion in its two halves, and held to exactly the bar `report` holds the structural
    path to: `racy_execution` exempts the not-run reasons that are a lost race rather than a missing
    control, and `corroborated` demands the same repetition count, keyed on the arm the finding
    names so two repetitions blaming opposite arms cannot corroborate each other.

    Not folded into the visible or behavioural verdict itself. A digest or an invariant that agrees
    is a true statement about what was on screen; that it was reached on a build where the button no
    longer works is a second, separate statement, and merging them would let the noise floor
    measured for one silence the other.
    """
    one_sided: list[tuple] = []
    racy: list[tuple] = []
    expect_bad: list[tuple] = []
    for action, shard, cell, r in results:
        if r.get("one_sided"):
            entry = (action, shard, cell, [r.get("idle_detail") or ""], r["one_sided"])
            # ON THE RECORDED REASON, not the action name; see the same call in `report`.
            if P.racy_execution(action, r.get("idle_reason") or ""):
                racy.append(entry)
            else:
                one_sided.append(entry)
        if r.get("expect_regressed"):
            expect_bad.append(
                (action, shard, cell, [r.get("expect_reason") or ""], r["expect_regressed"])
            )
    firm_one, weak_one = corroborated(one_sided, min_reps)
    firm_expect, weak_expect = corroborated(expect_bad, min_reps)
    return {
        "one_sided": firm_one,
        "one_sided_weak": weak_one,
        "one_sided_racy": racy,
        "expect": firm_expect,
        "expect_weak": weak_expect,
    }


def print_build_differences(found: dict[str, list[tuple]], min_reps: int) -> bool:
    """Print what `build_differences` found; True when something in it fails the run."""
    if found["one_sided"]:
        print(
            "\n  RAN ON ONE ARM ONLY -- the action could be performed on one build and not the"
            "\n  other, which is the two builds behaving differently and leaves no capture to"
            "\n  differ:"
        )
        for action, shard, cell, why, *_dir in found["one_sided"]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")
    if found["expect"]:
        print(
            "\n  THE ACTION'S OWN ASSERTION FAILED ON ONE ARM -- it ran on both and did its job"
            "\n  on only one, in every repetition. The captures can agree and still be a reading"
            "\n  of a control that stopped working:"
        )
        for action, shard, cell, why, *_dir in found["expect"]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")
    for key, what in (
        ("one_sided_weak", "one-arm execution"),
        ("expect_weak", "assertion failure"),
    ):
        if found[key]:
            print(
                f"\n  UNCORROBORATED {what} -- in fewer than {min_reps} repetitions, or on "
                f"opposite arms;\n  reported, not counted:"
            )
            for action, shard, cell, why, *_dir in found[key]:
                print(f"    {action:<26} {shard} {cell}: {why[0]}")
    if found["one_sided_racy"]:
        names = sorted({a for a, _s, _c, _w, *_d in found["one_sided_racy"]})
        print(
            f"\n  (reported, not counted) {len(found['one_sided_racy'])} one-arm execution(s) whose "
            f"recorded reason is a lost race rather than a missing control: {', '.join(names)}"
        )
    return bool(found["one_sided"] or found["expect"])


def behaviour_report(
    paths: list[Path],
    label: str,
    windowed: bool = True,
    select: Optional[set] = None,
    min_reps: int = 1,
) -> Outcome:
    """The windowed arm's report: behavioural invariants instead of a structural digest.

    Printed with the reason it is being printed, every time. The one way this could mislead is by
    quietly replacing a strict check with a looser one, so the banner says outright which question
    is no longer being asked.

    `select` scores only those pair keys, which is how a payload's windowed pairs are reported here
    while its fully mounted ones go to the structural digest they are owed.
    """
    got = collect(paths, select)
    results = []
    for (shard, rung, rep, sid, action), sides in sorted(got["pairs"].items()):
        cell = f"{rung} {rep}"
        if "base" not in sides or "treatment" not in sides:
            results.append(
                (
                    action,
                    shard,
                    cell,
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
                (action, shard, cell, {"verdict": P.NOT_COMPARABLE, "reason": why, "checks": []})
            )
            continue
        out = B.compare_behaviour(sides["base"], sides["treatment"])
        # THE SAME TWO QUESTIONS THE STRUCTURAL PATH ASKS, off the same functions: `compare_behaviour`
        # cannot see either shape, since an action not performed on one arm records no quantities to
        # disagree about and one that failed its own assertion records ordinary ones.
        # `compare_all_with` is the other half.
        idle_ = P.execution_verdict(sides["base"], sides["treatment"])
        out["one_sided"] = (idle_ or {}).get("one_sided") or ""
        out["idle_reason"] = (idle_ or {}).get("idle_reason") or ""
        out["idle_detail"] = (idle_ or {}).get("reason") or ""
        out["expect_regressed"], out["expect_reason"] = P.expect_regression(
            sides["base"], sides["treatment"]
        )
        results.append((action, shard, cell, out))

    print(f"\n{label}  (BEHAVIOURAL MODE)")
    print(f"  CLAIM: {P.CLAIM_BEHAVIOURAL}.")
    # Through the helper, not straight off the table: this mode's line names the coverage band it
    # enforces, and printing the raw template would emit the placeholders instead of the numbers.
    print(f"  POLICY: {P.behaviour_policy(B.MIN_CLIPBOARD_COVERAGE, B.MAX_CLIPBOARD_COVERAGE)}.")
    # WHICH REASON, and only if it is true: forced behavioural mode on a fully mounted payload --
    # which is how a NULL CONTROL is scored -- used to print "one arm mounts a window of the thread"
    # about a payload where neither does.
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
        return Outcome(2)

    broken, unchecked, idle, blind = [], [], [], []
    compared: set[tuple[str, str, str]] = set()
    matched = 0
    for action, shard, cell, r in results:
        verdict = r["verdict"]
        if verdict == B.BROKEN:
            broken.append((action, shard, cell, r))
            compared.add((action, shard, cell))
        elif verdict == P.MATCH:
            matched += 1
            compared.add((action, shard, cell))
        elif verdict == P.NOT_EXERCISED:
            idle.append((action, shard, cell, r))
        elif verdict == P.NOT_COMPARABLE:
            blind.append((action, shard, cell, r))
        else:
            unchecked.append((action, shard, cell, r))
    found = build_differences(results, min_reps)

    print(f"\n  {len(results)} action pairs across {len(paths)} shard(s)")
    print(f"  invariants held:            {matched}")
    print(f"  INVARIANTS BROKEN:          {len(broken)}")
    print(f"  UNCHECKED:                  {len(unchecked)}  (no invariant declared; not a pass)")
    print(f"  NOT COMPARABLE:             {len(blind)}")
    print(f"  NOT EXERCISED:              {len(idle)}  (the action did not run; not coverage)")

    if broken:
        print("\n  BEHAVIOURAL INVARIANTS BROKEN -- these are user-visible, not measurement noise:")
        for action, shard, cell, r in broken:
            print(f"    {action:<26} {shard} {cell}: {r['reason']}")
    else:
        print("\n  Every declared behavioural invariant held on both arms.")
    build_failed = print_build_differences(found, min_reps)

    if unchecked:
        names = sorted({a for a, _s, _c, _v in unchecked})
        print(
            f"\n  UNCHECKED -- {len(unchecked)} pair(s) over {len(names)} action(s) with no "
            f"declared invariant. These surfaces carry NO verdict on this arm:"
        )
        for name in names:
            print(f"    {name}")
    if blind:
        print("\n  NOT COMPARABLE:")
        for action, shard, cell, r in blind[:8]:
            print(f"    {action:<26} {shard} {cell}: {r['reason']}")
    if idle:
        names = sorted({a for a, _s, _c, _v in idle})
        print(f"\n  NOT EXERCISED: {', '.join(names)}")

    # `not broken` IS THE PRECONDITION THE PARAGRAPH BELOW ARGUES FROM: an all-BROKEN payload reaches
    # `matched == 0` too, so a run that had just listed its broken invariants went on to say none was
    # evaluated and it carried no failure, immediately before returning 1. NOT gated on
    # `build_failed`, which is collected from pairs whose invariants were unreadable.
    # test_a_timed_out_rebuild_leaves_the_behavioural_run_with_no_verdict asserts that pairing.
    if matched == 0 and not broken:
        # NOTHING WAS VALIDATED is not the same as nothing being wrong: with every pair unchecked,
        # `broken` is empty and the block above has just printed a sentence that is true of an empty set
        # and reads as a pass. More dangerous here, because behavioural mode REPLACES the digest.
        print(
            "\n  NOTHING WAS COMPARED. Not one behavioural invariant was evaluated on any pair, "
            "so this run carries no UI verdict -- neither a pass nor a failure. Treat it as an "
            "absent result and find out why the actions did not run."
        )
    # AFTER the diagnosis prints and BEFORE the "could not tell" code: a run can find a real
    # regression while deciding nothing, and "it failed" is more specific.
    # The two shapes are what `build_differences` returns.
    if broken or build_failed:
        return Outcome(1, frozenset(compared), _keys(results))
    if matched == 0:
        return Outcome(2, frozenset(compared), _keys(results))
    return Outcome(0, frozenset(compared), _keys(results))


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
    broad and too cheap: ONE differing null-control pair silenced that action for every cell and
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
    # AND ONLY FROM CELLS THAT FINISHED, the same admission rule `unstable_set` and `audit_null`
    # apply. A null-control cell that died mid-film leaves captures nothing owns, and read raw one
    # DIFFERING unfinished observation plus one matching completed one is exactly `min_observations`,
    # so a real visible difference is filed under "differ against an identical build" and exits 0.
    results, _got = compare_all_with(
        null_paths, P.compare_visible, "visible", require_complete = True
    )
    by_rung: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for action, _shard, cell, r in results:
        # A pair whose action never ran on both arms is not an observation in either direction:
        # `derive_unstable` refuses to count a verdict it cannot read, and this refuses to hand it one.
        if r.get("_ran"):
            by_rung[rung_of_cell(cell)].append((action, r))
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
    min_reps: int = 1,
) -> Outcome:
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
        return Outcome(2)

    found = build_differences(results, min_reps)
    compared: set[tuple[str, str, str]] = set()
    differing, unstable_bad, blind, idle, matched = [], [], [], [], 0
    # THE RESIDUE, PRINTED. A message on screen during the action but unmounted by capture time
    # cannot be digested and `compare_visible` refuses the pair for it, which used to be invisible:
    # the pair returned MATCH with the ordinals in `not_digested` and nothing read the key. Collected
    # across every verdict, because a DIFFER pair with a residue is also incomplete.
    residue = []
    for action, shard, cell, r in results:
        if r.get("not_digested"):
            residue.append((action, shard, cell, r))
        if not r.get("_ran"):
            idle.append((action, shard, cell, r))
        elif r["verdict"] == P.NOT_COMPARABLE:
            blind.append((action, shard, cell, r))
        elif r["verdict"] == P.DIFFER:
            # A SEVERE difference is never routed into the noise floor: an action can be in the derived
            # unstable set for an unrelated attribute and still be the action on which one arm lost the whole
            # thread. AT THIS PAIR'S OWN RUNG, since an action differing at 100K says nothing about 1K.
            noise = (rung_of_cell(cell), action) in unstable and not r.get("severe")
            (unstable_bad if noise else differing).append((action, shard, cell, r))
            compared.add((action, shard, cell))
        else:
            matched += 1
            compared.add((action, shard, cell))

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
        for action, shard, cell, r in differing:
            print(f"    {action:<26} {shard} {cell}: {r['reason']}")
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
        for action, shard, cell, r in blind[:8]:
            print(f"    {action:<26} {shard} {cell}: {r['reason']}")
    if residue:
        print(
            "\n  VISIBLE BUT NOT DIGESTED -- these messages were on screen during the action and "
            "had\n  been unmounted again before the capture, so nothing below covers them:"
        )
        for action, shard, cell, r in residue[:8]:
            print(f"    {action:<26} {shard} {cell}: ordinals {r.get('not_digested')[:8]}")
    if unstable_bad:
        names = sorted({a for a, _s, _c, _v in unstable_bad})
        print(
            f"\n  (reported, not counted) {len(unstable_bad)} pair(s) over {len(names)} action(s) "
            f"whose visible region differs between two runs of the SAME build: {', '.join(names)}"
        )
    if idle:
        names = sorted({a for a, _s, _c, _v in idle})
        print(f"\n  NOT EXERCISED: {', '.join(names)}")
    if not unstable:
        print(
            "\n  NO FLOOR WAS MEASURED. Pass --null OUTDIR of a base-vs-base run: an identical\n"
            "  pair of builds has been observed differing on 13 of 64 pairs inside the viewport,\n"
            "  so an unfloored count here can rank a real arm below two copies of the same build."
        )
    else:
        # WHICH RUNG EACH FLOOR ENTRY WAS MEASURED AT, printed, because it is also the only rung it
        # silences anything at; a bare list of action names would look like it covers the payload.
        keys = sorted(unstable)
        shown = ", ".join(f"{rung or '?'} {action}" for rung, action in keys[:12])
        print(
            f"\n  FLOOR: {len(keys)} (rung, action) pair(s) measured differing against an "
            f"identical build,\n  and silenced ONLY at the rung they were measured at: {shown}"
            + (f", and {len(keys) - 12} more" if len(keys) > 12 else "")
        )

    build_failed = print_build_differences(found, min_reps)

    if matched == 0 and not differing:
        # The same false green every other mode here has been fixed for: nothing compared is not nothing
        # wrong, and it must not exit 0.
        print(
            "\n  NOTHING WAS COMPARED. Not one action pair yielded a visible-region verdict, so\n"
            "  this run carries no UI verdict at all -- neither a pass nor a failure."
        )
    # See the same three lines in `behaviour_report` and `report`: the diagnosis prints, then a
    # failure outranks a refusal.
    if differing or build_failed:
        return Outcome(1, frozenset(compared), _keys(results))
    if matched == 0:
        return Outcome(2, frozenset(compared), _keys(results))
    return Outcome(0, frozenset(compared), _keys(results))


def compare_all_with(
    paths: list[Path],
    compare,
    key: str,
    select: Optional[set] = None,
    require_complete: bool = False,
) -> tuple[list[tuple], dict]:
    """[(action, shard, cell, result)] using `compare` over payload sub-object `key`.

    `require_complete` drops the action rows of a cell that never wrote its terminal `cell` row;
    see `collect`, which states why that is right on a null control and wrong on a result.

    THE ACTION'S OUTCOME TRAVELS WITH ITS CAPTURE, and it did not use to: this path built its result
    from `compare()` plus one boolean, `_ran`, so the two shapes that are a difference between the
    BUILDS rather than between two DOMs were erased before any report saw them. An action performed
    on one arm only collapsed into the same "did not run" bucket as a missed slot, which counts as
    lost coverage and not as a finding; an action that ran on both and failed its OWN assertion on
    one -- `stop_generation` on a head where Stop no longer ends the stream -- left no trace, the two
    viewports looking the same.

    Survivable on the structural path, where `compare_rows` has always asked both questions. Not
    here: on a windowed arm THESE reports are the entire UI verdict, the digest being
    `not_applicable` by construction and `stop_generation` having no behavioural invariant, so in
    `--mode auto` a passing invariant on some other action carried the run to exit 0. Both questions
    now go to `analysis/parity.py`'s own functions, the ones `compare_rows` calls.
    """
    got = collect(paths, select, require_complete = require_complete)
    results = []
    for (shard, rung, rep, sid, action), sides in sorted(got["pairs"].items()):
        cell = f"{rung} {rep}"
        if "base" not in sides or "treatment" not in sides:
            results.append(
                (
                    action,
                    shard,
                    cell,
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
                    cell,
                    {"verdict": P.NOT_COMPARABLE, "moved": [], "_ran": True, "reason": why},
                )
            )
            continue
        ran = bool(sides["base"].get("ran")) and bool(sides["treatment"].get("ran"))
        out = compare(sides["base"].get(key), sides["treatment"].get(key))
        out["_ran"] = ran
        idle = P.execution_verdict(sides["base"], sides["treatment"])
        # WHICH ARM, not merely that one was idle: `execution_verdict` returns `one_sided` empty for a
        # missed slot and for the symmetric case, and the arm that DID run otherwise.
        out["one_sided"] = (idle or {}).get("one_sided") or ""
        out["idle_reason"] = (idle or {}).get("idle_reason") or ""
        out["idle_detail"] = (idle or {}).get("reason") or ""
        out["expect_regressed"], out["expect_reason"] = P.expect_regression(
            sides["base"], sides["treatment"]
        )
        results.append((action, shard, cell, out))
    return results, got


def compare_all(
    paths: list[Path],
    select: Optional[set] = None,
    require_complete: bool = False,
) -> tuple[list[tuple], dict]:
    """[(action, shard, cell, compare-result)] over every base/treatment pair found.

    `cell` is `rung rep`, so the two rungs of one repetition stay two observations.

    `select` scores only those pair keys; see `decide_modes`.
    """
    got = collect(paths, select, require_complete = require_complete)
    results = []
    for (shard, rung, rep, sid, action), sides in sorted(got["pairs"].items()):
        cell = f"{rung} {rep}"
        if "base" not in sides or "treatment" not in sides:
            # One arm never produced this row. Recorded rather than skipped: an action that ran on one arm
            # and not the other is itself a difference. A resumed run lands here too with two session ids, so
            # the session is named; both carry NOT_COMPARABLE, which is honest for either.
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
        why = _refused(sides)
        if why:
            results.append(
                (
                    action,
                    shard,
                    cell,
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

    THE SAME PREDICATES `report` USES, on both axes, or this scopes a verdict nobody runs. Two
    ways it drifted, and both fail the job rather than pass it, which is why they are worth as
    much care as the excusing direction:

      RACY_EXECUTION. `report` does not count a one-arm-only result for the three actions whose
      ABILITY to run is a race, so no excuse can move their verdict and the null owes them
      nothing. Scoped anyway, the null observing the same legitimate stream-timing race made
      `audit_null` return 1 and the workflow fail on stream timing.

      THE DIRECTION. `report` carries the live arm in the corroboration key, so a pair that blames
      opposite arms across the two repetitions is UNCORROBORATED and cannot move the verdict.
      Built here as bare four-element tuples it corroborated, entered the scope, and an undecided
      null then failed a job whose verdict would have been 0.

    The rule for this function is simply: scope is what the verdict's FATAL set turns on. Anything
    the verdict already declines to count is a question nobody is going to ask.
    """
    results, _ = compare_all(paths)
    differing = [e for e in results if e[3]["verdict"] == P.DIFFER]
    one_sided = [
        (e[0], e[1], e[2], e[3], e[3].get("one_sided") or None)
        for e in results
        if e[3]["verdict"] == P.NOT_EXERCISED
        and e[3].get("one_sided")
        and not P.racy_execution(e[0], e[3].get("idle_reason") or "")
    ]
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
    at all -- `image_upload`, whose attachments button Unsloth never mounts without a model -- is
    permanently undecided for an honest reason. Those names are excused by `allow_undecided`, and
    every one of them is a hole, so they are printed.

    ONE REFUSAL IS NOT BLIND, and it is the one that made this audit unsatisfiable on the packed
    rungs. A pair whose scaffold, overlays, message set and every SETTLED row agreed, with only
    the subtree of a still-arriving reply left over, counts as an observation of non-difference
    (`parity.SETTLED_MATCH`). How many decided entries rest on it is reported rather than folded
    away, because it is the weaker claim: the settled thread was read, the live tail was not.

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
    # Decided entries whose every observation came from a settled-match refusal, kept apart from
    # `decided` because the live tail was never read at all.
    on_settled = []
    for rung, pairs in sorted(by_rung.items()):
        for action, row in sorted(P.derive_unstable(pairs).items()):
            entry = (rung, action)
            # OUT OF SCOPE IS NOT AN EXCUSE: an excused action is a HOLE in a question that was asked, while
            # an out-of-scope one is a question nobody asked, and conflating them made the empty scope read
            # as a vacuous audit.
            if scope is not None and entry not in scope:
                out_of_scope.append(entry)
                continue
            if row["undetermined"]:
                (excused if action in allow_undecided else undecided).append(entry)
                continue
            decided.append(entry)
            if row.get(P.SETTLED_MATCH) and row[P.SETTLED_MATCH] >= row["observations"]:
                on_settled.append(entry)
            if row["unstable"]:
                differed.append(entry)

    # AN ACTION THE NULL NEVER MEASURED AT ALL IS UNDECIDED, not absent: the loop can only classify
    # what `derive_unstable` produced, so a scoped (rung, action) with no rows fell into neither list
    # and one other decided action was enough to return 0 -- with `unstable_set` unioning the DECLARED
    # names back in, a corroborated result difference on it then passed. Reproduced end to end: scope
    # {send_turn, settings}, no send_turn rows, audit 0, verdict 0, regression printed under
    # "expected to vary". Counted as MISSING as well as undecided, because measured-and-inconclusive
    # is a different reading from never measured.
    missing = []
    if scope is not None:
        seen = set(decided) | set(undecided) | set(excused)
        for entry in sorted(scope - seen):
            missing.append(entry)
            (excused if entry[1] in allow_undecided else undecided).append(entry)

    report_ = {
        "decided": decided,
        "undecided": undecided,
        "excused": excused,
        "out_of_scope": out_of_scope,
        "differed": differed,
        "missing": missing,
        "decided_on_settled_thread": on_settled,
    }
    # NOTHING TO DECIDE IS A PASS, and only when a scope said so: a result whose every action matched
    # asks the null for no excuses, and there is no way to fail a question never put. The scope is
    # computed from the result, so an empty one is a statement about the result.
    if scope is not None and not scope:
        report_["reason"] = "the result needs no excuse from this null control"
        return 0, report_
    # Everything excused is not a decided null control but one that measured nothing while naming a
    # reason for each blank; passing it would let the excuse list grow until the audit is vacuous.
    if not decided:
        report_["reason"] = "no (rung, action) reached min_observations"
        return 1, report_
    if undecided:
        report_["reason"] = (
            "scoped actions the null never measured at all"
            if missing and set(missing) >= set(undecided)
            else "undecided actions outside the excused list"
        )
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
    if report_.get("decided_on_settled_thread"):
        # A narrower claim than the line above, so it is said out loud.
        print(
            f"  of which on the settled thread only: "
            f"{len(report_['decided_on_settled_thread'])}  (a reply was still arriving on every "
            "observation, so its subtree was withheld and the rest of the thread agreed)"
        )
    print(f"  undecided:                  {len(report_['undecided'])}")
    if allow_undecided and report_["excused"]:
        print(
            f"  excused as undecided:       {len(report_['excused'])}  "
            f"({', '.join(sorted(allow_undecided))}) -- each one a hole"
        )
    if report_.get("out_of_scope"):
        # Counted apart from the excused, because it is not a hole: no excuse could have moved the
        # result's verdict on these.
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
            _missing = set(report_.get("missing") or ())
            for rung, action in report_["undecided"][:12]:
                # Said apart from the rest: measured and inconclusive means the null tried and could not decide,
                # while NO ROWS means it never measured the action at all.
                tail = "  -- NO ROWS in the null at all" if (rung, action) in _missing else ""
                print(f"    {action}@{rung}{tail}")
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
    # The cross-check stays pooled ON PURPOSE: it audits the DECLARED list, whose entries claim to
    # hold at every rung, so the question is action-level. Advisory output, carrying no verdict.
    checks = P.cross_check(
        P.derive_unstable([(a, r) for a, _s, _c, r in results]), UNSTABLE_ACTIONS
    )
    # UNION, not replacement: an action the null could not reach (`image_upload` has no attachments
    # button on this fixture) would otherwise move from "declared unstable" to "stable" on a
    # measurement that never happened.
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
    unstable: frozenset, local_unstable: set, local_stable: set
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

    THE DIRECTION IS PART OF THE CLAIM, where there is one. An entry may carry a fifth element
    naming the arm it is about: which arm went idle, or which arm's assertion failed. Two
    repetitions that disagree about WHICH BUILD failed are not one finding seen twice, they are a
    race that landed on either side, and grouping them only by action and rung let the pair reach
    `firm` and report "the two builds did not behave the same way" while its own two lines named
    opposite arms. Keyed on the direction they separate, and each side is then a single
    repetition, so both print as UNCORROBORATED and the verdict does not rest on them.

    A four-element entry has no direction and groups exactly as before, which is right for a
    digest difference: `stable_bad` is a statement about a PAIR of arms and has no failing side.
    """
    by_action: dict[tuple, list[tuple]] = collections.defaultdict(list)
    for entry in entries:
        direction = entry[4] if len(entry) > 4 else None
        by_action[(entry[0], rung_of_cell(entry[2]), direction)].append(entry)
    firm, weak = [], []
    for group in by_action.values():
        # DISTINCT repetitions, not rows: one repetition seen twice is one observation. DELIBERATELY NOT
        # KEYED ON THE SHARD -- two shards carrying the same (rung, rep) cannot be told from one film
        # recorded twice, where counting both lets a single flake corroborate itself. Keying on the cell
        # alone can only under-count; keying on the shard manufactures corroboration.
        # test_one_repetition_seen_twice_is_not_two_observations holds this direction.
        reps = {e[2] for e in group}
        (firm if len(reps) >= min_reps else weak).extend(group)
    return firm, weak


def report(
    paths: list[Path],
    label: str,
    unstable: frozenset,
    select: Optional[set] = None,
    min_reps: int = 1,
    min_compared: int = 0,
) -> Outcome:
    """THREAD-STRUCTURE PARITY. `select` scores only those pair keys; see `decide_modes`.

    `unstable` holds both kinds of entry `is_unstable` reads: a bare action name, which is a
    declared entry and holds at every rung, and a `(rung, action)` tuple, which was measured and
    holds only at the rung it was measured at.

    `min_reps` is the corroboration threshold and `min_compared` the coverage floor; see
    `corroborated` and the coverage check below.
    """
    results, got = compare_all(paths, select)
    if not results:
        # An empty result is reported as an empty result: "No mismatches found" when nothing was compared
        # is the exact shape of a check that silently does nothing.
        print(
            f"\n{label}: NO PARITY DATA in {len(paths)} payload(s). "
            f"{got['missing']} action rows carried no digest. "
            f"Was this run recorded before the parity instrument existed?"
        )
        return Outcome(2)

    stable_bad, unstable_bad, blind, style_bad, idle = [], [], [], [], []
    inapplicable: list[tuple] = []
    one_sided, one_sided_unstable = [], []
    expect_bad = []
    compared: set[tuple[str, str, str]] = set()
    matched = 0
    for action, shard, cell, r in results:
        # COLLECTED BEFORE THE VERDICT BRANCHES AND OUTSIDE THE EXEMPTION, because it is not a digest: an
        # action can produce two digests the instability exemption excuses and still have failed its own
        # assertion on one arm only, which is the one shape `ran` cannot see. `stop_generation` is the
        # live example: `ran = True, expect_ok = stopped_ms is not None`, and it is on the declared
        # unstable list, so a head that no longer stops generation was excused twice over.
        # The fifth element is the DIRECTION, and `corroborated` keys on it: a treatment
        # failure in one repetition and a base failure in the next is a race, not one finding twice.
        if r.get("expect_regressed"):
            expect_bad.append(
                (action, shard, cell, [r.get("expect_reason", "")], r["expect_regressed"])
            )
        if r["verdict"] == P.NOT_APPLICABLE:
            # NEITHER A PASS NOR A FAIL: the digest is the wrong question for this pair, so it is bucketed
            # separately and `--mode behaviour` answers it instead. BELOW the assertion collection above,
            # and that ordering is the merge's one real decision: this branch `continue`s, so putting it
            # first would drop a failed assertion whenever the pair was also a windowed mount.
            inapplicable.append((action, shard, cell, [r.get("reason", "")]))
            continue
        if r["verdict"] == P.NOT_EXERCISED:
            # Same, for which arm stayed live: `one_sided` names the arm that DID run.
            entry = (action, shard, cell, [r.get("reason", "")], r.get("one_sided") or None)
            if r.get("one_sided"):
                # ONE ARM RAN IT AND THE OTHER COULD NOT, which is not the missed-slot case: a missed slot costs
                # coverage, while an action that runs on one build and cannot be performed on the other is the
                # one regression shape that leaves no digest to differ. Held to the SAME corroboration bar as a
                # differing digest. EXEMPTED BY `RACY_EXECUTION`, NOT BY THE DIGEST SET: every UNSTABLE_ACTIONS
                # mechanism describes what makes the CAPTURE move, so keyed on that list nine of sixteen actions
                # were permanently exempt. ON THE RECORDED REASON, not the action name.
                _racy = P.racy_execution(action, r.get("idle_reason") or "")
                (one_sided_unstable if _racy else one_sided).append(entry)
            else:
                idle.append(entry)
            continue
        # THE STYLE VERDICT IS COLLECTED BEFORE THE STRUCTURAL REFUSAL IS BUCKETED, because it is an
        # INDEPENDENT reading: the bounded computed-style probe is the only thing here that sees
        # `display`, `visibility` or `pointer-events`, and a pair refused for landing at two points in
        # one stream can still carry a real CSS regression.
        if r.get("style_verdict") == P.DIFFER:
            style_bad.append((action, shard, cell, [r.get("style_reason", "")]))
        if r["verdict"] == P.NOT_COMPARABLE:
            blind.append((action, shard, cell, [r.get("reason", "")]))
            continue
        if r["verdict"] == P.MATCH:
            matched += 1
            compared.add((action, shard, cell))
            continue
        entry = (action, shard, cell, r["moved"])
        compared.add((action, shard, cell))
        (unstable_bad if is_unstable(unstable, action, cell) else stable_bad).append(entry)

    # SPLIT BEFORE THE COUNTS ARE PRINTED: "stable actions differing: 1" above a verdict of 0 is how
    # a reader concludes the tool is lying, so the headline number must be the one the exit code
    # comes from.
    stable_bad, uncorroborated = corroborated(stable_bad, min_reps)
    one_sided, one_sided_weak = corroborated(one_sided, min_reps)
    expect_bad, expect_weak = corroborated(expect_bad, min_reps)

    print(f"\n{label}  (STRUCTURAL MODE)")
    print(f"  CLAIM: {P.CLAIM_STRUCTURAL}.")
    print(f"  POLICY: {P.POLICY_BY_MODE['structural']}.")
    print(
        "  Under the current policy an OFF-SCREEN-ONLY difference is exempt, and this mode cannot\n"
        "  tell an off-screen difference from an on-screen one. Use --mode visible for a payload\n"
        "  that defers off-screen work deliberately."
    )
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
    print(
        f"  ASSERTION failed on one arm:  {len(expect_bad)}"
        + (f"  (in >= {min_reps} repetitions)" if min_reps > 1 else "")
    )
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

    # THE SAME SET, COUNTED TWICE, checked rather than assumed, so a later edit to either branch
    # cannot make the coverage floor answer about a different set than the summary does.
    scored = matched + len(stable_bad) + len(uncorroborated) + len(unstable_bad)
    assert len(compared) == scored, (len(compared), scored)
    shortfall = coverage_shortfall(scored, len(results), min_compared)
    if shortfall:
        print(f"\n  {shortfall}")
        return Outcome(3, frozenset(compared), _keys(results))

    if expect_bad:
        print(
            "\n  THE ACTION'S OWN ASSERTION FAILED ON ONE ARM -- it ran on both and did its job"
            "\n  on only one, in every repetition. Not excused by the unstable set: that set is a"
            "\n  measurement of whether a DIGEST races, and this is not a digest:"
        )
        for action, shard, cell, why, *_dir in expect_bad:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if expect_weak:
        print(
            f"\n  UNCORROBORATED assertion failure -- on one arm in fewer than {min_reps} "
            f"repetitions; reported, not counted:"
        )
        for action, shard, cell, why, *_dir in expect_weak[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if one_sided:
        print(
            "\n  RAN ON ONE ARM ONLY -- one build could perform these and the other could not,"
            "\n  in every repetition. That is a difference between the builds, not lost coverage:"
        )
        for action, shard, cell, why, *_dir in one_sided:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if one_sided_weak:
        print(
            f"\n  UNCORROBORATED one-arm-only -- ran on one arm in fewer than {min_reps} "
            f"repetitions; reported, not counted:"
        )
        for action, shard, cell, why, *_dir in one_sided_weak[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if one_sided_unstable:
        print(
            "\n  (reported, not counted) one-arm-only on an action expected to vary between runs"
            "\n  of any build, so which arm reached its slot is a race rather than a signal:"
        )
        for action, shard, cell, why, *_dir in one_sided_unstable[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    # A DEMOTED DIFFERENCE IS STILL A COMPARISON: `corroborated` moves a sub-`min_reps` difference out
    # of `stable_bad`, and reading `matched` alone a payload whose every difference was uncorroborated
    # exited 2 "NOTHING WAS COMPARED" over a run that compared everything it had.
    decided = matched + len(uncorroborated) + len(one_sided_weak)
    if stable_bad:
        print("\n  UI PARITY DIFFERENCES ON STABLE ACTIONS -- these need explaining:")
        for action, shard, cell, moved in stable_bad:
            print(f"    {action:<26} {shard} {cell}: {', '.join(moved[:4])}")
    elif decided == 0:
        # NOT A PASS, and it must not print like one: with no pair producing a digest verdict, "no stable
        # action rendered differently" is true, reassuring and about nothing. The guard used to require
        # `inapplicable`, so a run whose parity probes all failed exited 0 under a clean-looking heading.
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

    if uncorroborated:
        # Printed in full: one repetition out of two is evidence of the run rather than of the build, but
        # it is still a reading a reader chasing an intermittent regression needs to see.
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
        # Named surfaces, deduplicated: what matters is WHICH actions this run never opened, not that it
        # failed to open one of them sixteen times.
        names = sorted({entry[0] for entry in idle})
        print(
            f"\n  NOT EXERCISED -- {len(idle)} pair(s) over {len(names)} action(s) that did not "
            f"run. These surfaces are UNCHECKED, not unchanged:"
        )
        for name in names:
            why = next(entry[3][0] for entry in idle if entry[0] == name)
            print(f"    {name:<26} {why}")

    if style_bad:
        print("\n  (advisory) the bounded computed-style probe differed:")
        for action, shard, cell, why in style_bad[:8]:
            print(f"    {action:<26} {shard} {cell}: {why[0]}")

    if unstable_bad:
        print("\n  (reported, not counted) actions that vary between runs of any build:")
        for action, shard, cell, moved in unstable_bad[:8]:
            print(f"    {action:<26} {shard} {cell}: {', '.join(moved[:3])}")
    # A one-sided action is a failure on the same footing as a differing digest: the two builds
    # behaved differently and the difference leaves no digest to compare.
    # `expect_bad` is collected ahead of the verdict branches.
    if stable_bad or one_sided or expect_bad:
        return Outcome(1, frozenset(compared), _keys(results))
    # 2, the same code the empty-payload path uses: the tool was asked a question it could not
    # answer, and exiting 0 would let CI go green on a run structurally incapable of firing.
    # `decided`, not `matched`. AFTER the failure return, since "it failed" is more specific.
    if decided == 0:
        return Outcome(2, frozenset(compared), _keys(results))
    return Outcome(0, frozenset(compared), _keys(results))


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
        "--mode",
        # `structural` and `behavior` are ALIASES, not extra modes: the report header prints
        # "(STRUCTURAL MODE)" and the PR template asks for "the structural digest", so a reader who types
        # either should not be told it is not a choice.
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
    args.mode = {"structural": "digest", "behavior": "behaviour"}.get(args.mode, args.mode)

    # THE MODE DECISION FIRST, before the unstable set is derived, or a windowed payload prints a
    # page of scoring apparatus with no bearing on the report. PER ACTION PAIR: per invocation scored
    # a normal run behaviourally because an unrelated payload on the same command line was windowed,
    # and per payload let one windowed 100K capture suppress the digest on every mounted pair.
    plan: list[dict] = []
    for pattern in args.payloads:
        paths = shards_of(pattern)
        if not paths:
            # Kept with no pairs: the structural loop below is what reports a pattern that matched no
            # payload, and it exits 2 for it.
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
                # None, not an empty set, when the payload holds no action pairs at all: `report` is what says
                # NO PARITY DATA and exits 2, and it has to be reached to say it.
                "structural": (set(decided) - windowed) if decided else None,
                "why": next((w for _k, (m, w) in sorted(decided.items()) if m == WINDOWED), ""),
            }
        )

    worst = 0
    scored_windowed = 0
    # THE COVERAGE FLOOR'S OWN BOOKKEEPING, per payload pattern and across every mode. A SET rather
    # than a running total because `--mode auto` scores each windowed pair twice; keyed by pattern
    # because the floor is a question about one film.
    compared_pairs: dict[str, set[tuple]] = {}
    seen_pairs: dict[str, set[tuple]] = {}

    def note(pattern: str, out: "Outcome") -> None:
        compared_pairs.setdefault(pattern, set()).update(out.compared)
        seen_pairs.setdefault(pattern, set()).update(out.seen)

    vis_unstable: Optional[frozenset[tuple[str, str]]] = None
    vis_null_tiers: set[str] = set()
    vis_null_corpora: set[str] = set()
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
            # shard, rung, rep and the action. The session term is left out of the line: it is part of a
            # pair's identity, not part of naming it to a reader.
            print(f"    windowed:   {key[0]} {key[1]} {key[2]} {key[4]}")
        if len(win) > 8:
            print(f"    ... and {len(win) - 8} more windowed pair(s)")
        scored_windowed += len(win)
        # Per pattern, and reset here rather than outside the loop: two payload patterns are two films
        # and must not pool their coverage.
        windowed_compared: Optional[frozenset[tuple]] = None
        windowed_seen: set[tuple] = set()
        # BOTH, and in this order. Visible-region parity is the verdict the off-screen exemption asks for
        # and can FAIL a windowed arm for something a user would see; the behavioural invariants catch
        # what a viewport comparison cannot, such as a clipboard that no longer carries the thread.
        if args.mode in ("auto", "visible"):
            if vis_unstable is None:
                # The floor, derived from the null control the caller passed; without it an identical pair of
                # builds outscores the arm under test. Derived once: it is a property of the null control.
                vis_null: list[Path] = []
                for pat in args.null:
                    vis_null.extend(shards_of(pat))
                vis_unstable = visible_unstable_set(vis_null)
                vis_null_tiers = one_tier(vis_null, "null control")
                vis_null_corpora = one_corpus(vis_null, "null control")
            # A FLOOR FROM ANOTHER FILM IS NOT THIS PAYLOAD'S FLOOR, ON EITHER AXIS. Slot spacing decides
            # which actions land inside a live stream, and `--tier fast` then `--tier standard` is the
            # documented way to work, which is how a stale fast null control reaches a standard run; the
            # corpus axis is quieter, since a re-recorded payload leaves an older null in the globbed
            # directory. `one_tier`/`one_corpus` refuse an internally mixed null, `cross_side_mismatch`
            # refuses a consistent one belonging to a different film. Refused rather than applied.
            floor = vis_unstable
            mismatch = cross_side_mismatch(
                one_tier(paths, "payload"),
                vis_null_tiers,
                one_corpus(paths, "payload"),
                vis_null_corpora,
            )
            if floor and mismatch:
                print(
                    f"\n  FLOOR REFUSED: {mismatch} It is NOT applied, and the visible differences "
                    f"below are unfloored: record a null control alongside this payload before "
                    f"reading them as findings."
                )
                floor = frozenset()
            vis = visible_report(
                paths, f"UI PARITY: {pattern}", floor, select = win, min_reps = args.min_reps
            )
            worst = max(worst, int(vis))
            windowed_compared = frozenset(vis.compared)
            windowed_seen |= set(vis.seen)
        if args.mode in ("auto", "behaviour"):
            beh = behaviour_report(
                paths,
                f"UI PARITY: {pattern}",
                windowed = args.mode == "auto" or any_windowed(paths) is not None,
                select = win,
                min_reps = args.min_reps,
            )
            worst = max(worst, int(beh))
            # Only where behavioural mode is the ONLY mode, i.e. `--mode behaviour`. Under `--mode auto` it
            # adds nothing to the numerator; see below.
            if windowed_compared is None:
                windowed_compared = frozenset(beh.compared)
            windowed_seen |= set(beh.seen)
        # NOT UNIONED ACROSS THE TWO WINDOWED MODES. The union is right for the structural set below,
        # which is DISJOINT; here the two are two verdicts on the SAME pairs, and unioned one stood in
        # for the other -- behavioural mode reaching all 16 pairs while visible capture succeeded on 1
        # still cleared `--min-compared 16`. Not an intersection either: a behavioural invariant only
        # exists for the few actions that declare one, so intersecting fails a run for pairs it had no
        # opinion about. So in `auto` the numerator is the visible verdict; `seen` stays a union.
        if windowed_compared is not None:
            compared_pairs.setdefault(pattern, set()).update(windowed_compared)
            seen_pairs.setdefault(pattern, set()).update(windowed_seen)

    # AHEAD OF THE STRUCTURAL EARLY RETURN, because this mode does not read the plan at all: it takes
    # shards from `args.payloads`, audits each AS a null control and returns unconditionally. Below
    # the return it was unreachable whenever `remaining` came out empty, which `--mode visible` and
    # `--mode behaviour` guarantee, so `--audit-null` with either silently skipped the audit.
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

    # THE STRUCTURAL EARLY RETURN, and why there is now a floor check on both sides of it. Everything
    # below is the structural half, which a payload with no fully mounted pair has none of, but
    # `--min-compared` is a COVERAGE floor asking whether the film ran, and a windowed run's film runs
    # exactly as much. The FOURTH cross-cutting enforcement found sitting below it, so the floor goes
    # through `coverage_shortfall` on BOTH exits.
    remaining = [e for e in plan if e["structural"] is None or e["structural"]]
    if not remaining:
        return _floored(worst, compared_pairs, seen_pairs, args.min_compared)
    if scored_windowed:
        # The rest still get the digest they were owed, in the same run.
        print(
            f"\n  {len(remaining)} payload(s) still hold fully mounted pairs, which are scored "
            "structurally below and not behaviourally."
        )

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

    # `null_tiers` is the set `one_tier` already refused a mixed null control for. NOT reassigned
    # from `tier_of` here, which would be the same set without the refusal in front of it, and `worst`
    # is NOT reset either: a windowed failure above may not be dropped by the structural pass.
    scored_structural = 0
    for entry in remaining:
        pattern, paths, select = entry["pattern"], entry["paths"], entry["structural"]
        if not paths:
            print(f"\nno payload found for {pattern}")
            worst = max(worst, 2)
            continue
        tiers = one_tier(paths, "payload")
        # REFUSED, NOT WARNED: a tier or corpus the null was not recorded against makes its unstable set
        # inapplicable, and a warning printed above a verdict is read as a verdict. `cross_side_mismatch`
        # covers the tier this used to warn about and the corpus it did not check at all.
        corpora = one_corpus(paths, "payload")
        mismatch = cross_side_mismatch(tiers, null_tiers, corpora, null_corpora)
        if mismatch:
            print(f"\n  REFUSING to score {pattern}: {mismatch}")
            worst = max(worst, 2)
            continue
        # CONFINED TO THE RUNNER BEING SCORED, and only ever downward: the exemptions above were measured
        # in another matrix job on another machine, while side A of THIS payload is the same build in
        # every repetition. One it positively contradicts is dropped; one it could not decide is kept.
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
        label = f"UI PARITY: {pattern}"
        if select is not None and entry["windowed"]:
            # WHICH PAIRS THESE ARE, in the heading: a structural section that silently covered part of a
            # payload would read as a verdict on all of it.
            label += (
                f"  ({len(select)} fully mounted pair(s) of "
                f"{len(select) + len(entry['windowed'])})"
            )
        scored_structural += len(select) if select is not None else 0
        # BOTH SIDES OF THE MERGE, and the argument list is why this had to be read rather than chosen:
        # `select` is the FOURTH positional and the incoming call omitted it, which would have bound
        # `args.min_reps` to `select` and left the coverage floor at 0. `min_compared` is deliberately NOT
        # passed down any more: a floor applied inside each report is checked per report, so a run
        # comparing nine windowed and nine structural pairs would fail a floor of sixteen twice over.
        # `args.min_compared` was the argument that went missing.
        # `report` (`elif decided == 0`) and `visible_report` (`and not not_run`) each check their own.
        struct = report(paths, label, effective, select, args.min_reps)
        worst = max(worst, int(struct))
        note(pattern, struct)
    # COMBINED, and the worst outcome wins: a behavioural failure on one payload is not cancelled by
    # a structural pass on another.
    if scored_windowed and scored_structural:
        print(
            f"\nCOMBINED EXIT STATUS {worst}: {scored_windowed} pair(s) scored on the visible "
            f"region and behavioural invariants, {scored_structural} scored structurally. Any "
            "mode's failure fails the run."
        )
    return _floored(worst, compared_pairs, seen_pairs, args.min_compared)


if __name__ == "__main__":
    raise SystemExit(main())
