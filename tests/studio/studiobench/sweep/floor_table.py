# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Per-metric detection floor, and the three gates a result has to clear to be quotable.

    python -m tests.studio.studiobench.sweep.floor_table --floor OUT_NULL OUT_MINE

WHY PER METRIC. The A/B renderer prints ONE floor, which is the right default for a headline and
the wrong tool for deciding a pull request: a floor of 16.5% driven entirely by `jank_index` says
nothing about whether `message_menu.open_close_ms` can resolve a 5% change. Every numeric timing on
every action is harvested here and given its own floor, so a change is quoted against the floor of
the metric it was written to move rather than against the worst metric in the run.

WHY PAIRED BY REPETITION. The two arms of a repetition run adjacent in time, so pairing removes the
session drift that pooling leaves in. The element census climbs monotonically through a session,
and a pooled comparison charges that climb to whichever arm ran later.

WHAT THE FLOOR IS. Run the SAME build against itself, base versus base, and whatever spread that
produces is how far apart two identical builds land on this machine under this load. A difference
smaller than it is not a small effect. It is an effect that cannot be distinguished from zero, and
it prints as VOID rather than as a number.

The floor must be measured in band, concurrently with the comparison it judges. Session-to-session
drift on this metric set is about 8%, which is larger than most real effects, so a floor from a
different session is not a floor.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    # Running the file directly rather than as a module. Supported because the first thing a new
    # contributor does with a script is run it by path, and failing there with an import error is a
    # bad first minute.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.runtime.ab import failed_invalidating_gates  # noqa: E402
from tests.studio.studiobench.scoring.from_payload import (  # noqa: E402
    ACTION_SOURCES,
    FRAME_METRICS,
    UNSCORED_WINDOW_KINDS,
    STREAM_METRICS,
    _actions_for,
    _frame_measures,
    _stream_measures,
    latest_attempt_rows,
    refuse_if_probed,
)
from tests.studio.studiobench.scoring import payload_rules  # noqa: E402

METRICS = tuple(ACTION_SOURCES) + FRAME_METRICS + STREAM_METRICS

# The metrics compared by DIFFERENCE rather than by ratio, because zero is their CLEAN reading.
#
# Everything else here is a duration, and a duration is zero only when nothing happened. Jank is
# the opposite: a stream that never dropped a frame reports exactly 0.0, and that is the reading a
# good build is supposed to produce. Measured on this repository's recorded payloads,
# `stream_time_in_jank_pct` is exactly 0.0 on 765 of 1,438 scored cells, and 349 of 668 base-vs-
# treatment pairs across 117 payload files have a zero base arm.
#
# Paired as a ratio those pairs cannot survive: `t / 0.0` is undefined, so `paired` dropped them,
# and dropping them is not a conservative choice. It removes exactly the comparisons where the base
# was CLEAN -- the ones a treatment that introduces jank shows up in -- so a regression from 0.0%
# to 12.0% left no row at all, and a run where only some repetitions had a zero base kept the rest
# under the same `n`, pooling a metric over a set whose membership depends on the base arm's value.
#
# A difference is defined from zero, so these are pooled as `treatment - base` in the metric's own
# unit, for EVERY pair rather than only the zero ones: mixing the two within one mean would be a
# second unstable pool. The null control's floor is computed the same way, so the gates compare
# like with like, and the table marks these rows so the column is not read as a percentage change.
#
# Scoped to the two streaming metrics this file has just started harvesting. `time_in_jank_pct`
# has the same shape and loses 3% of its pairs, but it is already published against ratio-based
# floors, and re-basing it would silently re-score every table taken so far.
DIFFERENCE_METRICS: frozenset[str] = frozenset({"stream_time_in_jank_pct", "stream_jank_index"})


def _action_timings(records: list[dict], cid: str) -> dict[str, float]:
    """Every numeric timing on every action that RAN, as `action.timing`.

    Only three action timings are wired into the scoring anchors, and those three do not cover what
    most performance changes actually move: a menus change moves `message_menu`, a model-picker
    change moves `model_change`, a re-open change moves `thread_reopen`, and neither of the latter
    two is scored. Harvesting all of them is what lets each change be judged on its own metric.

    `ran` is checked first. An action that did not happen has no timing worth pairing, and folding
    its absence in as a fast number is the single most common way this harness has produced a
    confident wrong answer.

    AN ACTION WHOSE OWN ASSERTION FAILED IS TREATED THE SAME WAY. `report/payload.py` already
    records `ran = True` with `expect_ok = False` as a note saying its timings "exist and must not
    be quoted", and this is where they would be quoted. `keystroke` is the case: if half the
    characters never reach the controlled component the action still ran, its p95 reads lower for
    exactly that reason, and pairing it would print the failure as `faster`.
    """
    out: dict[str, float] = {}
    for name, row in _actions_for(records, cid).items():
        if not row.get("ran") or row.get("expect_ok") is False:
            continue
        for key, value in (row.get("timings") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{name}.{key}"] = float(value)
        # Correctness invariants, harvested alongside the timings and named apart from them.
        #
        # Same paired arithmetic, opposite meaning. A timing falling is the result a change is
        # trying to produce; a count falling is a regression, and one that no timing can reveal.
        # `select_all_copy.count.selected_chars` is the case this exists for: the selection is
        # taken over the viewport's DOM, so anything that stops mounting the whole thread
        # truncates the clipboard while every timing improves and the action still reports
        # `expect_ok`. Reading it against the other arm needs no calibration, because both arms
        # seed a byte-identical thread.
        for key, value in (row.get("counts") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{name}.count.{key}"] = float(value)
    return out


def sessions_in(records: list[dict]) -> set[str]:
    """Every `session_id` that produced a completed cell in this payload."""
    return {
        r.get("session_id")
        for r in records
        if r.get("row_type") == "cell" and r.get("completed") and r.get("session_id")
    }


def refuse_collisions(records: list[dict]) -> None:
    """Refuse a payload in which one cell completed under more than one session.

    CALLED FROM EVERY ENTRY POINT THAT POOLS, which is the whole point. The refusal used to live
    inside `cell_metrics` behind `session is None`, and the only production caller -- `paired` --
    always passes a session, so nothing in the shipped path ever reached it. On the real payload
    from two concurrent launchers it went straight through: `paired` returned four pairs from two
    cells, and `summarise` reported keystroke `p50_ms` up 93.4% on n=4. Neither session measured
    93.4%; they measured +37.0% and +149.8%. The pooled figure is the mean of two runs contending
    with each other, presented as four independent repetitions.

    A guard reachable only from a function nobody calls is the same defect as a guard that was
    never wired up at all, which this branch has now hit three times.
    """
    collided = collided_cells(records)
    if not collided:
        return
    # A SEQUENTIAL RE-RUN IS NOT A COLLISION, and refusing it rejected every resumed A/B.
    #
    # `ab.skippable_cells` skips a repetition only when EVERY arm of it completed, and an A/B with
    # any work left re-runs EVERY pair in one new session -- deliberately, because a lone arm
    # measured in a new session can never be paired and a partial table publishes a verdict over
    # whichever rungs survived. So a resumed comparison legitimately writes a SECOND completed row
    # under the same deterministic `cell_id`, and the writer's own docstring says what happens
    # next: "The old attempt stays in the payload and `latest_attempt_rows` supersedes it."
    # Nothing here superseded anything, so a resumed run could not be scored at all.
    #
    # WHAT IS ASKED IS WHETHER THE TWO SESSIONS WERE RUNNING AT ONCE, which is the property that
    # makes the readings contended, and it is recorded: each session's `run_meta` carries
    # `started_at` and every row carries `ts_ms` from that session's own clock. On the real
    # two-launcher payload of defect 9 the two sessions occupy 23:50:24-00:10:03 and
    # 23:53:45-00:12:43, sixteen minutes of overlap. A resume starts after its predecessor stopped.
    #
    # SILENCE IS NOT A LICENCE. A payload that does not carry the evidence -- no `started_at`, no
    # `ts_ms` -- is refused exactly as before, because "no proof they overlapped" is not "proof
    # they did not", and this is the guard that stands between a reader and a blend of two
    # contending runs. Only a payload that can SHOW its sessions were sequential is superseded.
    guilty: set[str] = set()
    for sessions in collided.values():
        guilty |= sessions
    verdict, both = concurrent_sessions(records, only = guilty)
    if verdict == "sequential":
        # The caller supersedes through `latest_attempt_rows`, which keeps the LAST attempt that
        # wrote anything -- so a resume that was itself hard-killed leaves an incomplete cell and
        # drops out, rather than resurrecting the older completed reading.
        return
    if verdict == "overlap":
        why = (
            f"sessions {both[0]} and {both[1]} were RUNNING AT THE SAME TIME by their own "
            f"`started_at` and `ts_ms`"
        )
    elif verdict == "interleaved":
        why = (
            f"the rows of sessions {both[0]} and {both[1]} INTERLEAVE in the file, which one "
            f"writer appending after another cannot produce"
        )
    else:
        why = (
            "this payload does not say when its sessions ran -- no `started_at`, or no `ts_ms` -- "
            "so it cannot show they were sequential rather than concurrent"
        )
    listed = ", ".join(f"{cid} in {sorted(s)}" for cid, s in sorted(collided.items())[:4])
    more = "" if len(collided) <= 4 else f" (and {len(collided) - 4} more)"
    raise SystemExit(
        f"refusing to pool {len(collided)} cell id(s) that completed under more than one "
        f"session in this payload, and {why}: {listed}{more}. `cell_id` is unique within a "
        f"session, not "
        f"across them, so keying on it alone would report whichever session was appended last, "
        f"and pairing across them would report two contending runs as repetitions of one. Two "
        f"concurrent runs sharing one --out is the usual cause. Split the payload by session, or "
        f"pass `session=` to score one of them."
    )


def collided_cells(records: list[dict]) -> dict[str, set[str]]:
    """{cell_id: sessions} for every cell id COMPLETED under more than one session id.

    THIS, AND NOT THE SESSION COUNT, IS WHAT SEPARATES THE TWO CASES. More than one session in a
    payload is ordinary and legitimate: `--resume` re-runs the arm that died under a new session id
    into the same shard directory, and sharding appends several sessions on purpose. What is never
    legitimate is the SAME cell completing twice, because a cell id is unique within a session, so
    two completed copies mean two runs measured the same thing and only one of them can be
    reported.

    The resumed case is distinguishable precisely because the attempt that died is not marked
    completed: only the retry is, so the id does not collide. The concurrent case is the one where
    every id is present twice with `completed: true` on both.
    """
    seen: dict[str, set[str]] = {}
    for r in records:
        if r.get("row_type") == "cell" and r.get("completed") and r.get("cell_id"):
            seen.setdefault(r["cell_id"], set()).add(str(r.get("session_id") or ""))
    return {cid: s for cid, s in seen.items() if len(s) > 1}


def session_spans(records: list[dict]) -> dict[str, tuple[int, int]]:
    """{session: (first row index, last row index)} over the file as written."""
    spans: dict[str, list[int]] = {}
    for i, row in enumerate(records):
        sid = row.get("session_id")
        if sid is None:
            continue
        sid = str(sid)
        if sid in spans:
            spans[sid][1] = i
        else:
            spans[sid] = [i, i]
    return {k: (v[0], v[1]) for k, v in spans.items()}


def session_clocks(records: list[dict]) -> dict[str, tuple[float, float]]:
    """{session: (start, end)} in seconds since the epoch, for sessions that can say.

    A session's `run_meta` records `started_at` in wall clock and every row it writes carries
    `ts_ms` from that session's own monotonic clock, so its occupancy is `started_at` to
    `started_at + max(ts_ms)`. A session missing either is absent from this map rather than
    given a guessed one: the caller treats absence as unknown and refuses.
    """
    import datetime

    starts: dict[str, float] = {}
    spans: dict[str, float] = {}
    for row in records:
        sid = row.get("session_id")
        if sid is None:
            continue
        sid = str(sid)
        if row.get("row_type") == "run_meta" and row.get("started_at") and sid not in starts:
            try:
                starts[sid] = datetime.datetime.fromisoformat(str(row["started_at"])).timestamp()
            except ValueError:
                continue
        ts = row.get("ts_ms")
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            spans[sid] = max(spans.get(sid, 0.0), float(ts) / 1000.0)
    return {sid: (t0, t0 + spans[sid]) for sid, t0 in starts.items() if sid in spans}


def concurrent_sessions(
    records: list[dict], only: set[str] | None = None
) -> tuple[str, tuple[str, str] | None]:
    """Were these sessions running at once? `("sequential"|"overlap"|"interleaved"|"unknown", pair)`.

    TWO INDEPENDENT WITNESSES, because either alone can be fooled. The clocks answer the question
    actually being asked -- contention is a property of time, not of file layout -- but they rest
    on a `started_at` a machine with a stepped clock can misreport. File order answers a narrower
    question that cannot be misreported: a payload is append-only with one writer per process, so
    a session's rows are one contiguous stretch unless somebody else was writing into the gaps. A
    run that stalls long enough for a whole second run to start and finish inside its own gap is
    caught too, since the nested stretch overlaps the enclosing one.

    Sorted by first row, so comparing each session with the next is enough: if any two stretches
    overlap then some ADJACENT pair does, because the later one's start is at or before the
    earlier one's end.

    UNKNOWN IS NOT SEQUENTIAL. A payload that cannot show when its sessions ran gets the refusal
    it got before this distinction existed.
    """
    spans = {k: v for k, v in session_spans(records).items() if only is None or k in only}
    order = sorted(spans.items(), key = lambda kv: kv[1][0])
    for (first, a), (second, b) in zip(order, order[1:]):
        if a[1] >= b[0]:
            return "interleaved", (first, second)
    clocks = {k: v for k, v in session_clocks(records).items() if only is None or k in only}
    if set(clocks) != set(spans) or len(clocks) < 2:
        return "unknown", None
    by_start = sorted(clocks.items(), key = lambda kv: kv[1][0])
    for (first, a), (second, b) in zip(by_start, by_start[1:]):
        if a[1] > b[0]:
            return "overlap", (first, second)
    return "sequential", None


def cell_metrics(records: list[dict], session: str | None = None) -> dict[str, dict[str, float]]:
    """{cell_id: {metric: value}} for every COMPLETED cell in ONE session of the payload.

    REFUSES rather than letting the last writer win, when and only when a cell id COMPLETED under
    more than one session. `cell_id` is unique within a session and NOT across sessions, so keying
    on it alone silently collapses two measurements of the same cell into whichever was appended
    last. Several sessions in one payload is not itself the fault -- `--resume` and sharding both
    produce that legitimately, and refusing them would delete good readings. See `collided_cells`.

    That is not hypothetical. A launcher started twice ran two full sessions concurrently against
    one `--out`, and both appended: three `run_meta` rows, every `cell_id` present twice, both
    marked completed, and the two copies carrying materially different timings because the runs
    were contending with each other -- `r1M.treatment.rep1` keystroke `p50_ms` read 73.4 ms in one
    session and 144.5 ms in the other. Scored last-wins it reported a 149.8% regression; scored
    per session it read +149.8% in one and +42.8% in the other.

    Pass `session` to select one, or use `paired`, which keys on the session and pairs within it.

    SCOPED TO THE CELL'S OWN SESSION, not to its cell id. The payload is append-only and a cell id
    is REUSED: `--resume` re-runs a cell that died, and a second run into the same output directory
    repeats every id. Selecting on the id alone pools the dead attempt's windows with the retry's
    and reports the average as the completed cell, which is a number nothing ever measured.

    IDLE WINDOWS ARE EXCLUDED, exactly as `scoring/from_payload` excludes them. Every cell records a
    1.5 s `idle:calibrate` window with the frame recorder running, and pooling that quiet into the
    frame metrics dilutes `time_in_jank_pct` and `jank_index` away from the film that was measured.
    A metric here has to be the same quantity the rest of the tool calls by that name.
    """
    if session is None:
        refuse_collisions(records)
        # SUPERSEDED ATTEMPTS ARE DROPPED, because a resumed A/B re-runs a repetition that had
        # already completed and both copies are in the file. Keyed on the LAST attempt that wrote
        # anything rather than the last that FINISHED: a resume killed inside a cell has flushed
        # its action rows and never reaches the cell row its `finally` would write, and treating
        # the older completed attempt as current would hand a stale reading to a floor. Dropped
        # here, that repetition has no completed cell and falls out of the table instead.
        #
        # ONLY WHEN NO SESSION WAS NAMED. `session=` is the escape hatch this module's own refusal
        # points a reader at -- "pass `session=` to score one of them" -- and superseding inside it
        # would empty the older of two concurrent sessions and take the hatch away. A caller who
        # names a session is asking for that session, not for the payload's latest view of it.
        records = list(latest_attempt_rows(records))
    # A CELL THAT FAILED AN INVALIDATING GATE IS NOT A READING, HERE EITHER. Both gates are
    # advisory where they are emitted, so such a cell arrives `completed=True` with a full set of
    # timings that are CHEAPER than a correct cell's, and this table is where a floor is built and
    # where a result is scored against it. `paired` divides treatment by base per pair, so a
    # one-sided failure is not cancelled by the null control being same-build: it lands in
    # `delta_pct` and `spread_pct` directly. On the result side it is worse, because a gate-failed
    # treatment cell pairs against a clean base cell and prints as `faster`.
    #
    # This is not hypothetical. The virtualization negative result recorded in CONTRIBUTING-perf.md
    # was measured through this workflow and contains both facts at once: the treatment arm's
    # census going from 12 mounted messages to 0 and never recovering, and a headline of 28.2%
    # faster weighted. Nothing between the two noticed.
    #
    # Scoped by hand, because a gate row is not one of the row types `latest_attempt_rows` covers:
    # that reduction has already dropped the superseded attempt's cells, actions and windows above,
    # and a `gate` row carries no attempt of its own to supersede.
    gate_failures = failed_invalidating_gates(records)

    # A SUPERSEDED ATTEMPT'S COMPLETION IS NOT THIS CELL'S READING, and the session scoping below
    # cannot say so: it answers "which rows belong to this completed cell row", not "is this
    # completed cell row still the cell". `ab.skippable_cells` re-runs an A/B pair WHOLE, so a
    # resume with any work left re-attempts cells that had already completed, and a retry that
    # dies leaves the older completed row as the only one carrying a full set of timings.
    #
    # Left raw, that row is admitted -- and the guard above it disagrees about which attempt is
    # being scored. `failed_invalidating_gates` names the winner from the LAST cell row, so a
    # retry that crashed and wrote its `completed=False` row supersedes the dead attempt's FAILED
    # gate while this loop keeps that same attempt's numbers. A cell refused for losing its
    # thread's middle before the resume was published after it, pairing its cheaper timings
    # against a clean base arm and printing as `faster`: the 28.2% failure described above,
    # arriving through the resume. Same rows, same rule, one lens -- as `runtime/ab.readings_by_arm`
    # and `sweep/ui_parity.collect` already read them.
    #
    # THE REDUCTION THAT DOES THIS NOW SITS AT THE TOP OF THE FUNCTION, and it is CONDITIONAL --
    # skipped when the caller named a `session`. A second unconditional call here would have been
    # a no-op on the default path and a silent regression on that one: it would supersede inside
    # the escape hatch, empty the older of two concurrent sessions, and take away the very thing
    # this module's collision refusal tells the reader to reach for. So the guard above and this
    # loop still read one lens, and which lens that is now depends on the argument, deliberately.
    #
    # Left open on purpose: with `session=` named, `failed_invalidating_gates` still resolves its
    # winner across every session in the file, so a caller who names the OLDER of two sessions can
    # have its gate discarded by the newer one while this loop admits its cells. Scoping the gate
    # search to the named session is the fix, and it is a change to the escape hatch's semantics
    # rather than to this merge.

    out: dict[str, dict[str, float]] = {}
    if session is not None:
        records = [r for r in records if r.get("session_id") in (session, None)]
    for row in records:
        if row.get("row_type") != "cell" or not row.get("completed"):
            continue
        if session is not None and row.get("session_id") != session:
            continue
        if str(row.get("cell_id")) in gate_failures:
            continue
        cid = row["cell_id"]
        # Scoped to this cell's OWN attempt: `--resume` re-runs a died cell under a new
        # session_id into the same file, and pooling both attempts mixes two measurements.
        sid = row.get("session_id")
        own = [r for r in records if r.get("cell_id") == cid and r.get("session_id") == sid]
        vals: dict[str, float] = _action_timings(own, cid)
        # `UNSCORED_WINDOW_KINDS` is idle plus setup, dropped here as in `measures_from_records`.
        # The setup window is the composer click that starts the film, which is mostly
        # Playwright's injected actionability script blocking the page's own main thread; pooled
        # in, an 11 s driver stall would set this table's `max_frame_ms` floor and hide every real
        # regression under it.
        windows = [
            w
            for w in own
            if w.get("row_type") == "window"
            and str(w.get("kind") or "") not in UNSCORED_WINDOW_KINDS
        ]
        for key, m in _frame_measures(windows).items():
            if m.value is not None:
                vals[key] = float(m.value)
        # The streaming phase on its own, per streamed character. Kept separate from the pooled
        # frame metrics rather than replacing them: the pooled ones answer "was this film janky",
        # which is still worth asking, and these answer "what did streaming one character cost",
        # which nothing answered before.
        for key, m in _stream_measures(windows).items():
            if m.value is not None:
                vals[key] = float(m.value)
        out[cid] = vals
    return out


def arm_of(cell_id: str) -> str:
    return "treatment" if ".treatment." in cell_id else "base"


def rep_of(cell_id: str) -> str:
    return cell_id.rsplit(".", 1)[-1]


def cell_sessions(records: list[dict]) -> dict[str, str]:
    """{cell_id: session_id} for every COMPLETED cell, resolved the way `cell_metrics` resolves it.

    Same last-writer-wins rule as `cell_metrics`, so the session reported here is the session whose
    numbers that function returned. Anything else would pair a reading against a session it did not
    come from, which is the thing the caller is trying to stop.

    THROUGH `latest_attempt_rows`, because `cell_metrics` is. A superseded attempt it no longer
    returns must not still be able to name a session here, or this answers about one attempt while
    the numbers came from another -- the same lens split this pair of functions exists to close.
    """
    out: dict[str, str] = {}
    for row in latest_attempt_rows(records):
        if row.get("row_type") == "cell" and row.get("completed"):
            out[row["cell_id"]] = str(row.get("session_id") or "")
    return out


def paired(records: list[dict], shard: str = "") -> dict[str, list[tuple[float, float]]]:
    """{metric: [(base, treatment), ...]} matched on (shard, rung, repetition, session).

    The shard is part of the key because sharding restarts the repetition counter: two independent
    sessions both produce `rep0`, and pairing on the repetition alone would silently overwrite one
    session's base with the other's. Pairing WITHIN a shard is also the correct thing to do, since
    a pair only means anything when both arms come from the same session.

    THE SESSION IS PART OF THE KEY FOR THE SAME REASON, and one shard is not one session. `--resume`
    is the case: when one arm completed and its partner died, the resumed run skips the completed
    arm and re-runs the dead one under a NEW session id, into the same shard directory. Keyed on the
    repetition alone those two arms pair, and the ~8% session-to-session drift this file's header
    measures is then charged in full to whichever arm was re-run. `scoring/ab.py` already refuses
    that comparison outright; this is the same refusal in the place the sweep does its pairing.

    A payload recorded before session ids existed has `""` on both arms and pairs exactly as before.
    """
    # THE SESSION IS PART OF THE KEY, for the same reason the shard is. Two sessions in one
    # payload both produce `rep0`, and pairing on the repetition alone matches one session's base
    # against the other's treatment -- which is a comparison between two different runs under two
    # different machine loads, reported as a repetition.
    #
    # Looped per session because several sessions in one payload is ORDINARY -- a sharded or
    # resumed run appends legitimately, and each session's repetitions must pair within themselves.
    # What is never ordinary is one cell completing twice, so that is refused first, here rather
    # than only inside `cell_metrics`: this loop always passes a session, so the refusal there was
    # unreachable from the shipped path and a payload of two concurrent runs pooled straight
    # through it.
    refuse_collisions(records)
    # The same superseding `cell_metrics` applies, done once here so the session list below is the
    # set of sessions that still own a completed cell. Without it a finished ladder re-run pairs
    # ONCE PER SESSION and one repetition is pooled twice, under two different sets of numbers.
    records = list(latest_attempt_rows(records))
    by_key: dict[tuple[str, str, str, str], dict[str, dict[str, float]]] = collections.defaultdict(
        dict
    )
    for sess in sorted(sessions_in(records)) or [None]:
        for cid, vals in cell_metrics(records, session = sess).items():
            rung = cid.split(".", 1)[0]
            by_key[(shard, str(sess), rung, rep_of(cid))][arm_of(cid)] = vals
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for sides in by_key.values():
        if "base" not in sides or "treatment" not in sides:
            continue
        for metric in set(sides["base"]) & set(sides["treatment"]):
            b, t = sides["base"][metric], sides["treatment"][metric]
            # `if b` is a ratio's precondition, not a validity test: a zero base is a real reading
            # for the jank metrics and only an undefined DENOMINATOR. Those are compared by
            # difference (see DIFFERENCE_METRICS), so they are kept whenever both sides are finite.
            if metric in DIFFERENCE_METRICS:
                if math.isfinite(b) and math.isfinite(t):
                    out[metric].append((b, t))
            elif b:
                out[metric].append((b, t))
    return out


def tiers_of(records: list[dict]) -> set[str]:
    """EVERY tier in one file, not the first.

    One payload can hold more than one run: the recorder appends, so a second invocation into the
    same output directory writes a second `run_meta` behind the first. Reading only the first is
    what let a fast-tier film and a standard-tier film sit in one file and pass the refusal below.
    """
    return {str(r.get("tier") or "?") for r in records if r.get("row_type") == "run_meta"} or {"?"}


def tier_of(records: list[dict]) -> str:
    for r in records:
        if r.get("row_type") == "run_meta":
            return str(r.get("tier") or "?")
    return "?"


def corpora_of(records: list[dict]) -> set[str]:
    """EVERY corpus hash the payload carries, not just the first one.

    The recorder appends, so one payload file can hold more than one `run_meta`: `--resume` (and
    any re-run into the same `--out`) writes a second header next to the first run's completed
    cells. `paired` matches base against treatment on (shard, rung, repetition) and does not care
    which run wrote either side, so a first-header-wins reading would pair a base recorded on the
    old corpus with a treatment recorded on the new one and print the corpus change as a
    performance change -- the exact thing the refusal below exists to prevent.
    """
    found = {str(r.get("corpus_hash") or "?") for r in records if r.get("row_type") == "run_meta"}
    return found or {"?"}


def corpus_of(records: list[dict]) -> str:
    """The one corpus a payload was recorded on, or a refusal if it holds more than one."""
    corpora = corpora_of(records)
    if len(corpora) > 1:
        raise SystemExit(
            f"refusing to read a payload recorded on more than one corpus: "
            f"{sorted(h[:16] for h in corpora)}. Its cells were recorded against different "
            f"films, so pairing them would read the corpus change as a performance change. "
            f"Re-run the whole payload on one corpus."
        )
    return next(iter(corpora))


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding = "utf-8").splitlines() if line.strip()
    ]


def load(paths: list[Path]) -> tuple[dict[str, list[tuple[float, float]]], set[str]]:
    """Pool paired ratios across every shard of one logical result, plus the tiers seen.

    The tiers come back with the data because they gate whether pooling was legitimate at all: the
    fast tier runs a 57 s film where the standard runs 243 s, so the same action is measured with
    different amounts of thread settled around it. Two such payloads are two different measurements
    of one quantity, not two samples of it.
    """
    pooled: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    tiers: set[str] = set()
    corpora: set[str] = set()
    for path in paths:
        records = read_rows(path)
        # REFUSED HERE, not warned about. This is the same class of refusal as the tier and
        # corpus checks below, one step earlier: those two say "these are different films", and
        # this one says "this film was shot with the camera in the shot". There is no flag to
        # override it, because the only correct response is to re-run without the probe. The
        # check itself lives in the scoring layer so that the A/B table and `--report` refuse on
        # exactly the same evidence rather than on a second copy of the rule.
        refuse_if_probed(records, str(path))
        tiers |= tiers_of(records)
        corpora.add(corpus_of(records))
        for metric, rows in paired(records, shard = str(path.parent.name)).items():
            pooled[metric].extend(rows)
    if len(tiers) > 1:
        raise SystemExit(
            f"refusing to pool payloads from different tiers: {sorted(tiers)}. A "
            f"fast-tier film and a standard-tier film are different measurements of "
            f"the same action, not repetitions of one."
        )
    # Same rule one level down. The tier fixes how long the film runs; the corpus hash fixes what
    # is IN it, and it covers the generator's parameters as well as every unit's bytes. Corpus v2
    # added math, so a v1 payload and a v2 payload measure two different documents under one name,
    # and pooling them would read the corpus change as a performance change.
    if len(corpora) > 1:
        raise SystemExit(
            f"refusing to pool payloads built on different corpora: "
            f"{sorted(h[:16] for h in corpora)}. The corpus hash covers every generated "
            f"byte and every generator parameter, so these are two different films. "
            f"Re-run the older side."
        )
    return pooled, tiers


def partial_censoring(paths: list[Path]) -> dict[str, str]:
    """{metric: why it must not be pooled across this ladder}, over every shard.

    THE GUARD WAS WRITTEN AND THEN NEVER CALLED. `payload_rules.refuse_partial_censoring` returned
    the right refusal from the moment it landed and nothing in the scoring or sweep path asked it
    anything, so the only code that ever saw the answer was its own selftest. That is the same
    shape as the row type that was registered nowhere: a guard that cannot fire is not a guard, and
    it is worse than an absent one because the reader believes the case is covered.

    What it catches is defect 2. `reasoning_toggle.open_ms` is censored on every cell above the
    100K rung, so `paired()` pools only the cells that could answer and `render()` prints the mean
    of those under a bare metric name. On a 100K/500K/1M ladder that row is a 100K-only number
    wearing a ladder label, and the only hint is a smaller `n` sitting beside the other rows --
    indistinguishable from a metric that simply had fewer repetitions.

    JUDGED OVER EVERY SHARD AT ONCE, because that is the set `summarise` pools. Asked per file,
    a ladder split across shards escapes: the shard holding the measured 100K cells sees no
    censoring at all, the shard holding the censored 500K cells sees censoring at every rung it
    contains, and neither returns a refusal -- while `load()` pools them together and prints the
    100K number under a ladder label. The same rows concatenated into one file are caught. Identical
    data, opposite treatment, decided by which file they happened to be written to.
    """
    everything: list[dict] = []
    for path in paths:
        # SUPERSEDED ATTEMPTS DROPPED, exactly as `paired` drops them, because this decides whether
        # the number `paired` produced may be quoted. A resume that died mid-cell leaves an action
        # row whose `expect_ok` is False, and `censored_metrics` counts that as a censored cell --
        # so reading the raw file here would mark a metric partially censored on the strength of an
        # attempt that contributed nothing to the mean. The two readers have to see one payload.
        #
        # NAMESPACED BY SHARD, for the same reason `paired` keys on it: sharding restarts the
        # repetition counter, so `r500K.base.rep0` names a DIFFERENT cell in each shard. Merged on
        # the bare id, the completed-cell filter below stops separating them -- one shard's failed
        # cell and another shard's successful one become a single id that is both censored and
        # completed, and a metric nothing censored anywhere is refused as partially censored on
        # the strength of a cell that contributed nothing to the mean. That is the very case the
        # paragraph above drops superseded attempts to avoid, arriving one file over.
        #
        # SUFFIXED rather than prefixed, so `refuse_partial_censoring` still reads the rung off
        # `cell_id.split(".", 1)[0]` and its message names rungs rather than directories.
        shard = str(path.parent.name)
        for row in latest_attempt_rows(read_rows(path)):
            row = dict(row)
            if row.get("cell_id") is not None:
                row["cell_id"] = f"{row['cell_id']}@{shard}"
            everything.append(row)
    out: dict[str, str] = {}
    for metric in sorted(censorable_metrics(everything)):
        why = payload_rules.refuse_partial_censoring(everything, metric)
        if why:
            out[metric] = why
    return out


def censorable_metrics(records: list[dict]) -> set[str]:
    """Every metric this payload censored anywhere, which is the set worth asking about."""
    return set(payload_rules.censored_metrics(records))


def summarise(paths: list[Path]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    pooled, _tiers = load(paths)
    censoring = partial_censoring(paths)
    for metric, rows in pooled.items():
        if metric in DIFFERENCE_METRICS:
            # Same three quantities, same three gates, measured in the metric's own unit because a
            # ratio from a clean zero base does not exist. `delta_pct` and `spread_pct` are then
            # percentage POINTS (or ms), which is why `render` marks the row.
            diffs = [t - b for b, t in rows]
            # A TIE IS NOT A DISAGREEMENT, AND HERE THE TIE IS THE MODAL READING.
            #
            # The ratio branch below asks `all(r > 1) or all(r < 1)`, and that is sound there
            # because a paired ratio of exactly 1.0 is vanishingly rare for a continuous timing.
            # Reused verbatim on differences it stops being sound, because the tie value is 0.0
            # and 0.0 is what a CLEAN reading of these metrics is: `stream_time_in_jank_pct` is
            # exactly 0.0 on 765 of 1,438 scored cells, so a repetition where neither arm dropped
            # a frame contributes a difference of exactly 0.0.
            #
            # A group of `[0.0, 12.0]` then failed both halves of the strict test and was reported
            # as VOID (pairs disagree on sign), which is not what happened: nothing moved the other
            # way, some repetitions were unchanged and the rest regressed together. Measured over
            # the recorded payloads, that voided 34 of 250 pooled groups, 13.6%, and every one of
            # the 34 came from a tie rather than from a real disagreement. Intermittent and
            # load-dependent jank is exactly the shape that produces ties, so the predicate was
            # discarding the regressions it was most needed for.
            #
            # Direction is therefore "no pair went the other way", with at least one that moved,
            # so an all-zero group stays inconsistent rather than being scored as a finding.
            nonzero = [d for d in diffs if d != 0.0]
            out[metric] = {
                "n": len(rows),
                "base": statistics.fmean(b for b, _ in rows),
                "treat": statistics.fmean(t for _, t in rows),
                "delta_pct": statistics.fmean(diffs),
                "consistent": bool(nonzero)
                and (all(d >= 0.0 for d in diffs) or all(d <= 0.0 for d in diffs)),
                "spread_pct": max(diffs) - min(diffs),
                "difference": True,
            }
            continue
        ratios = [t / b for b, t in rows]
        out[metric] = {
            "n": len(rows),
            "base": statistics.fmean(b for b, _ in rows),
            "treat": statistics.fmean(t for _, t in rows),
            "delta_pct": (statistics.fmean(ratios) - 1.0) * 100.0,
            # GATE 2. Do the pairs even agree on the SIGN? A metric whose four paired ratios
            # straddle 1.0 has no direction, however far its mean happens to land from the floor.
            # This is what separates "the change moved it" from "the metric is thrashing".
            "consistent": all(r > 1.0 for r in ratios) or all(r < 1.0 for r in ratios),
            # The spread of the paired ratios. From a null control it is the detection floor; from
            # a real comparison it is the effect's own scatter, which is GATE 3.
            "spread_pct": (max(ratios) - min(ratios)) * 100.0,
            "difference": False,
        }
        # LABELLED, NOT DROPPED AND NOT RAISED. Raising would be the defect-10 mistake: `open_ms`
        # is censored above 100K on EVERY standard and full run, so a refusal that exited would
        # abort the whole table on the normal case and the guard would be removed within a day.
        # Dropping the row silently would delete a reading somebody may still want at the one rung
        # that produced it. So the row survives, carries the rungs it actually covers, and is
        # denied a verdict below -- it can no longer clear a gate or be counted as a survivor.
        if metric in censoring:
            out[metric]["poolable"] = False
            out[metric]["censoring"] = censoring[metric]
    return out


def verdict_for(
    stat: dict,
    floor: dict | None,
    is_count: bool = False,
) -> tuple[float | None, str]:
    """The three gates, in the order that makes a failure most informative.

    Gate 1, the per-metric floor, is `max(|null delta|, null spread)` rather than the spread alone.
    In a null control several metrics show a systematic offset between the two arm LABELS with
    identical builds behind them: `stop_generation.stop_ms` reads 6.6% faster on the treatment side,
    tightly, across every repetition. Whatever causes it, arm B's page being created second inside
    each repetition being the likely candidate, it is charged to the treatment arm in a real A/B
    too. So the bar is the larger of the null control's own bias and its scatter.

    Gate 3 is applied last because it is the one that most often surprises: an effect can clear the
    floor on its mean while its own spread is an order of magnitude larger than the effect it
    claims. Twelve rows in a 40-comparison audit passed gates 1 and 2 and were junk.
    """
    if floor is None:
        return None, "no floor measured"
    f = max(abs(floor["delta_pct"]), floor["spread_pct"])
    if abs(stat["delta_pct"]) < f:
        return f, "VOID (under floor)"
    if not stat["consistent"]:
        return f, "VOID (pairs disagree on sign)"
    if stat["n"] > 1 and stat["spread_pct"] > abs(stat["delta_pct"]):
        return f, "VOID (effect under its own scatter)"
    if is_count:
        # A count is an invariant, so the sign means the opposite of what it means for a timing.
        # Less of the conversation copied is not an improvement, and calling it "faster" because
        # the number went down is exactly the kind of misreading this table exists to prevent.
        return f, ("LOST (invariant fell)" if stat["delta_pct"] < 0 else "gained")
    return f, ("faster" if stat["delta_pct"] < 0 else "SLOWER")


def is_count_metric(metric: str) -> bool:
    """`action.count.key` is an invariant; `action.key` and the frame metrics are timings."""
    return ".count." in metric


def merged_meta(paths: list[Path]) -> tuple[dict | None, list[str]]:
    """One `run_meta` describing every shard of one logical run, plus what forbids one.

    EVERY SHARD AND EVERY HEADER, for the same reason `tiers_of` and `corpora_of` read every
    header rather than the first: the recorder appends, so `--resume` writes a second header
    behind the first, and a sharded run spreads its headers across files. A first-header-wins
    reading describes the run that started the file rather than the cells now in it.
    """
    rows: list[dict] = []
    for path in paths:
        rows += read_rows(path)
    return payload_rules.merged_run_meta(rows)


def render(
    paths: list[Path],
    title: str,
    floors: dict | None = None,
    floor_tier: str | None = None,
    floor_corpus: str | None = None,
    floor_meta: dict | None = None,
) -> int:
    stats = summarise(paths)
    rows = read_rows(paths[0])
    tier = tier_of(rows)
    if floor_tier is not None and floor_tier != tier:
        raise SystemExit(
            f"refusing to score a {tier}-tier payload against a {floor_tier}-tier "
            f"floor: the two run different films, so their spreads are not the same "
            f"quantity. Run a null control at --tier {tier}."
        )
    corpus = corpus_of(rows)
    if floor_corpus is not None and floor_corpus != corpus:
        raise SystemExit(
            f"refusing to score a payload built on corpus {corpus[:16]} against a floor "
            f"built on {floor_corpus[:16]}: a floor is the scatter of THIS film, and a "
            f"different corpus is a different film. Re-run the null control."
        )
    # THE REST OF THE COMPARABILITY IDENTITY, which tier and corpus are only two axes of.
    #
    # `--compare` refuses two payloads whose `comparability_key`s differ, and this path -- the one
    # that actually certifies a number -- checked two of the eleven fields that key covers. So an
    # 0.1.0 floor could score an 0.2.0 payload on the same corpus and tier, and this commit is
    # precisely why that matters: it redefines `reasoning_toggle.open_ms` to terminate on a settled
    # DOM rather than on the `data-state` flip, so the older floor measures a different quantity
    # under the same name. `host`, `engine`, `cadence`, `headed`, `rungs` and the injection and
    # probe axes are the same class of difference, and `--compare` is a separate command nobody is
    # obliged to run first.
    #
    # Checked through `explain_incomparable` rather than against a second list of fields, so a
    # field added to `comparability_fields` is enforced here without being added here too. A guard
    # kept in step by hand is the drift `payload_rules` exists to argue against.
    if floor_meta is not None:
        meta, conflicts = merged_meta(paths)
        if meta is None:
            raise SystemExit(
                "refusing to score a payload that carries no run_meta row against a floor that "
                "does: nothing in it says which film, which harness or which host produced it, "
                "so there is no way to tell whether the floor describes the same quantity."
            )
        if conflicts:
            raise SystemExit(
                "refusing to score a payload that disagrees with ITSELF across its own run_meta "
                "rows:\n  " + "\n  ".join(conflicts) + "\nThis payload holds more than one run "
                "and they were not measuring the same thing, so no floor can be applied to it as "
                "a whole. Score the runs apart."
            )
        differ = payload_rules.explain_incomparable(floor_meta, meta)
        if differ:
            raise SystemExit(
                "refusing to score this payload against a floor that is not comparable with it. "
                "These differ (floor != payload):\n  "
                + "\n  ".join(differ)
                + "\nA floor is the scatter of THIS measurement on THIS machine with THIS "
                "harness. A field above changing means the two are not measuring the same "
                "quantity, whatever the metric is called. Re-run the null control in band."
            )
    if tier == "fast":
        print("\n  NOTE: fast tier. These are directions for iteration, not reportable numbers.")
    print(f"\n{title}")
    print(f"  payloads: {len(paths)} shard(s): {', '.join(p.parent.name for p in paths)}")
    head = f"  {'metric':<28}{'n':>3}{'base':>11}{'treat':>11}{'delta %':>10}{'spread %':>10}"
    print(head + ("      floor %  verdict" if floors else ""))
    print("  " + "-" * (len(head) + (26 if floors else 0)))
    survivors = 0
    censored_notes: list[str] = []
    marked = False
    for metric in sorted(stats, key = lambda m: (m in METRICS, m)):
        s = stats[metric]
        # TWO INDEPENDENT CAVEATS, BOTH CARRIED IN THE NAME COLUMN, and neither allowed to hide
        # the other. They answer different questions -- `(abs)` says what the delta MEANS, `[*]`
        # says which rungs it covers -- and a metric can need both at once.
        #
        # A metric censored at some rungs and measured at others is marked so the caveat cannot be
        # separated from the number when somebody copies one row out of this table into prose.
        # That is the route every withdrawn figure in this harness travelled.
        partial = s.get("poolable") is False
        # A difference row is named as one. Its delta is in the metric's own unit, and printing it
        # under a "delta %" heading would be the same class of mistake this file exists to stop.
        #
        # ORTHOGONAL TO THE TWO CENSORING MARKS BELOW, and applied first so it survives either of
        # them: `(abs)` says what the delta MEANS, `[*]` and `[f]` say who can judge it. A metric
        # can need `(abs)` and a censoring mark at once.
        label = metric
        if s.get("difference"):
            label, marked = f"{metric} (abs)", True
        # THE FLOOR IS A MEASUREMENT TOO, and it is censored by the same rule on its own cells.
        #
        # `summarise` marks the null control's entry `poolable = False` when the metric answered on
        # some of its cells and was censored on others, and this loop then read only the RESULT's
        # flag -- so a result measured in full was scored against a floor built from whichever
        # repetitions of the null happened to survive. Censoring is decided against a fixed budget,
        # so the repetitions it removes are the slow ones, and `spread_pct` is `max - min` over the
        # survivors: removing pairs can only narrow it. Observed on a real null control at 100K
        # plus 500K, `reasoning_toggle.close_ms` censored above 100K, the applied floor came out at
        # 17.1% from the four surviving 100K pairs while the censored repetitions of that same null
        # ran 1429-2149 ms against 498-681 ms and paired as far apart as 1.50 on identical builds.
        # A real result of -24.8% printed `faster` against the 17.1% and is not distinguishable
        # from the null's own scatter.
        floor_stat = None if floors is None else floors.get(metric)
        floor_partial = floor_stat is not None and floor_stat.get("poolable") is False
        # MARKED APART FROM `[*]`, because the two say different things. `[*]` means this row's own
        # number is not a ladder number; `[f]` means the number is fine and the null control cannot
        # judge it. Collapsing them would tell a reader to distrust a figure that is sound.
        shown = f"{label} [*]" if partial else (f"{label} [f]" if floor_partial else label)
        line = (
            f"  {shown:<28}{s['n']:>3}{s['base']:>11.1f}{s['treat']:>11.1f}"
            f"{s['delta_pct']:>+10.1f}{s['spread_pct']:>10.1f}"
        )
        if floors is not None:
            if partial:
                # DENIED A VERDICT. Passing three gates on the rungs that could answer is not
                # passing them on the ladder the row is labelled with.
                line += f"{'--':>13}  not pooled"
            elif floor_partial:
                # DENIED A VERDICT ON THE OTHER SIDE. The three gates are all measured against
                # this floor, so a floor that is itself a survivor sample cannot certify anything,
                # however clean the result is.
                line += f"{'--':>13}  no poolable floor"
            else:
                f, verdict = verdict_for(s, floor_stat, is_count_metric(metric))
                line += (f"{'--':>13}" if f is None else f"{f:>13.1f}") + f"  {verdict}"
                if verdict in ("faster", "SLOWER", "LOST (invariant fell)", "gained"):
                    survivors += 1
        if partial:
            censored_notes.append(f"[*] {s['censoring']}")
        elif floor_partial:
            censored_notes.append(
                f"[f] the null control's own {floor_stat['censoring']} So it is a floor for the "
                f"cells that survived, not for this row."
            )
        print(line)
    if marked:
        print(
            "\n  (abs) = compared by DIFFERENCE, in the metric's own unit rather than as a "
            "percentage\n        change: zero is these metrics' clean reading, and a ratio from "
            "zero does not exist."
        )
    if censored_notes:
        print("\n  NOT SCORED. [*] the row is NOT A LADDER NUMBER; [f] its floor is not one:")
        for note in censored_notes:
            print(f"      {note}")
    if floors is not None:
        print(f"\n  {survivors} metric(s) cleared all three gates.")
    return survivors


def shards_of(pattern: str) -> list[Path]:
    """`outputs/sbench_mine*` to every shard's payload, in a stable order."""
    root = Path(pattern).parent if "/" in pattern else Path(".")
    stem = Path(pattern).name
    found = sorted(p / "payload.jsonl" for p in root.glob(stem) if (p / "payload.jsonl").exists())
    return found or ([Path(pattern)] if Path(pattern).exists() else [])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog = "studiobench.sweep.floor_table",
        description = "Per-metric detection floor and the three verdict gates.",
    )
    ap.add_argument(
        "payloads",
        nargs = "+",
        help = "studiobench output directories or payload paths (globs allowed)",
    )
    ap.add_argument(
        "--floor",
        metavar = "OUTDIR",
        help = "the null control (base vs base) whose spread sets the floor. Without "
        "it this prints deltas and REFUSES to call any of them a result",
    )
    args = ap.parse_args(argv)

    floors, floor_tier, floor_corpus, floor_meta = None, None, None, None
    if args.floor:
        floor_paths = shards_of(args.floor)
        if not floor_paths:
            print(f"no null-control payload found for {args.floor}")
            return 2
        floors = summarise(floor_paths)
        floor_rows = read_rows(floor_paths[0])
        floor_tier = tier_of(floor_rows)
        floor_corpus = corpus_of(floor_rows)
        # The floor's own comparability identity, read across every shard and every header. A null
        # whose shards disagree with each other is not one null control, so it gets the same
        # refusal a result payload gets rather than a token averaged over the disagreement.
        floor_meta, floor_conflicts = merged_meta(floor_paths)
        if floor_conflicts:
            print(
                "refusing to use a null control that disagrees with ITSELF across its own "
                "run_meta rows:\n  " + "\n  ".join(floor_conflicts)
            )
            return 2

    seen = 0
    for arg in args.payloads:
        paths = shards_of(arg)
        if not paths:
            print(f"\nno payload found for {arg}")
            continue
        seen += 1
        render(
            paths,
            f"PAIRED PER-METRIC TABLE: {arg}",
            floors,
            floor_tier,
            floor_corpus,
            floor_meta,
        )
    if not seen:
        return 2
    if floors is None:
        print(
            "\n  NO FLOOR SUPPLIED. Nothing above is a result: without a null control there is "
            "\n  no way to tell any of these deltas from the noise of two identical builds."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
