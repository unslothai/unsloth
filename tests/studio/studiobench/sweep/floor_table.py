# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Per-metric detection floor and three gates; the floor is the same build run against itself."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    # Allows running the file directly by path.
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

# Compared by difference, not ratio: 0.0 is the clean reading (about half of scored cells), so `t / 0.0`
# would drop exactly the pairs where a treatment introduces jank.
DIFFERENCE_METRICS: frozenset[str] = frozenset({"stream_time_in_jank_pct", "stream_jank_index"})


def _action_timings(records: list[dict], cid: str) -> dict[str, float]:
    """Skip actions that did not run or whose own assertion failed; their timings read falsely fast."""
    out: dict[str, float] = {}
    for name, row in _actions_for(records, cid).items():
        if not row.get("ran") or row.get("expect_ok") is False:
            continue
        for key, value in (row.get("timings") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[f"{name}.{key}"] = float(value)
        # Counts are correctness invariants: a falling count (e.g. truncated clipboard) is a regression.
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
    """Refuse payloads where one cell completed under two sessions; pooling them averages contending
    runs."""
    collided = collided_cells(records)
    if not collided:
        return
    # Sequential re-runs are not collisions; refuse overlapping sessions and any with no timing evidence.
    guilty: set[str] = set()
    for sessions in collided.values():
        guilty |= sessions
    verdict, both = concurrent_sessions(records, only = guilty)
    if verdict == "sequential":
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
    """Refuse a cell id completed under two sessions; several sessions alone, as in a resume, are fine"""
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
    """Session wall-clock spans; a session missing started_at or ts_ms is omitted, never guessed."""
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
    """Overlap check from clock spans and file order; unknown is not sequential, so it is refused."""
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
    """Completed-cell metrics for one session; without one named, refuses cells completed in two
    sessions."""
    if session is None:
        refuse_collisions(records)
        # Drop superseded attempts (last attempt that wrote anything wins), except when `session=` is
        # named, which is the escape hatch for concurrent sessions.
        records = list(latest_attempt_rows(records))
    # A cell that failed an invalidating gate is not a reading: those gates are advisory upstream,
    # so such cells arrive completed with misleadingly cheap timings.
    gate_failures = failed_invalidating_gates(records)

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
        # Scoped to this cell's own attempt: `--resume` re-runs a died cell into the same file.
        sid = row.get("session_id")
        own = [r for r in records if r.get("cell_id") == cid and r.get("session_id") == sid]
        vals: dict[str, float] = _action_timings(own, cid)
        # Setup is dropped: Playwright's actionability script would set the `max_frame_ms` floor.
        windows = [
            w
            for w in own
            if w.get("row_type") == "window"
            and str(w.get("kind") or "") not in UNSCORED_WINDOW_KINDS
        ]
        for key, m in _frame_measures(windows).items():
            if m.value is not None:
                vals[key] = float(m.value)
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
    """Session id of each completed cell, resolved through latest_attempt_rows like cell_metrics."""
    out: dict[str, str] = {}
    for row in latest_attempt_rows(records):
        if row.get("row_type") == "cell" and row.get("completed"):
            out[row["cell_id"]] = str(row.get("session_id") or "")
    return out


def paired(records: list[dict], shard: str = "") -> dict[str, list[tuple[float, float]]]:
    """Pairs base with treatment per shard, rung, repetition and session, so resumed arms never pair."""
    # Session is part of the key: two sessions both produce `rep0`.
    refuse_collisions(records)
    # Supersede once here, or a re-run ladder pairs once per session and pools a rep twice.
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
            # Zero base is a real reading for jank metrics; those are compared by difference.
            if metric in DIFFERENCE_METRICS:
                if math.isfinite(b) and math.isfinite(t):
                    out[metric].append((b, t))
            elif b:
                out[metric].append((b, t))
    return out


def tiers_of(records: list[dict]) -> set[str]:
    """Every tier named by any run_meta row, since the recorder appends a second header per run."""
    return {str(r.get("tier") or "?") for r in records if r.get("row_type") == "run_meta"} or {"?"}


def tier_of(records: list[dict]) -> str:
    for r in records:
        if r.get("row_type") == "run_meta":
            return str(r.get("tier") or "?")
    return "?"


def corpora_of(records: list[dict]) -> set[str]:
    """Every corpus_hash across all run_meta rows, not the first, since the recorder appends headers."""
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
    """Pools paired ratios across shards; tiers come back too, since mixed tiers are not one measurement."""
    pooled: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    tiers: set[str] = set()
    corpora: set[str] = set()
    for path in paths:
        records = read_rows(path)
        # Refused, not warned: a probed run measured the probe. No flag overrides it.
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
    # Corpus hash covers generator params and unit bytes; pooling v1 with v2 is invalid.
    if len(corpora) > 1:
        raise SystemExit(
            f"refusing to pool payloads built on different corpora: "
            f"{sorted(h[:16] for h in corpora)}. The corpus hash covers every generated "
            f"byte and every generator parameter, so these are two different films. "
            f"Re-run the older side."
        )
    return pooled, tiers


def partial_censoring(paths: list[Path]) -> dict[str, str]:
    """Metrics censored on some rungs only, judged over every shard together, since pooling biases them."""
    everything: list[dict] = []
    for path in paths:
        # Drop superseded attempts as `paired` does, and suffix the shard because sharding restarts
        # the rep counter; suffix keeps `cell_id.split(".", 1)[0]` as the rung.
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
            # Same gates in the metric's own unit, so delta and spread are percentage points.
            diffs = [t - b for b, t in rows]
            # A zero difference is a tie, not a disagreement: direction means no pair went the other way
            # and at least one moved.
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
            # Gate 2: do the pairs agree on the sign?
            "consistent": all(r > 1.0 for r in ratios) or all(r < 1.0 for r in ratios),
            # Spread of paired ratios: the detection floor on a null control, gate 3 on a result.
            "spread_pct": (max(ratios) - min(ratios)) * 100.0,
            "difference": False,
        }
        # Labelled, not dropped or raised: `open_ms` is censored above 100K on every standard run.
        if metric in censoring:
            out[metric]["poolable"] = False
            out[metric]["censoring"] = censoring[metric]
    return out


def verdict_for(
    stat: dict,
    floor: dict | None,
    is_count: bool = False,
) -> tuple[float | None, str]:
    """Gate 1 floor is max(|null delta|, null spread), since null arms carry a systematic label offset."""
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
        # For a count invariant, falling is a loss, not an improvement.
        return f, ("LOST (invariant fell)" if stat["delta_pct"] < 0 else "gained")
    return f, ("faster" if stat["delta_pct"] < 0 else "SLOWER")


def is_count_metric(metric: str) -> bool:
    """`action.count.key` is an invariant; `action.key` and the frame metrics are timings."""
    return ".count." in metric


def merged_meta(paths: list[Path]) -> tuple[dict | None, list[str]]:
    """Merged run_meta over every shard and header, since a resumed run appends a second header."""
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
    # Check the full comparability identity via `explain_incomparable`, not just tier and corpus.
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
        # `(abs)` and `[*]` are independent caveats and both stay in the name column.
        partial = s.get("poolable") is False
        label = metric
        if s.get("difference"):
            label, marked = f"{metric} (abs)", True
        # The floor is censored by the same rule: a survivor-only null spread can only narrow.
        floor_stat = None if floors is None else floors.get(metric)
        floor_partial = floor_stat is not None and floor_stat.get("poolable") is False
        # `[*]`: this row's number is not a ladder number; `[f]`: the null cannot judge it.
        shown = f"{label} [*]" if partial else (f"{label} [f]" if floor_partial else label)
        line = (
            f"  {shown:<28}{s['n']:>3}{s['base']:>11.1f}{s['treat']:>11.1f}"
            f"{s['delta_pct']:>+10.1f}{s['spread_pct']:>10.1f}"
        )
        if floors is not None:
            if partial:
                line += f"{'--':>13}  not pooled"
            elif floor_partial:
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
        # A null whose shards disagree on identity is not one null control.
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
