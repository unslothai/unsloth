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
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    # Running the file directly rather than as a module. Supported because the first thing a new
    # contributor does with a script is run it by path, and failing there with an import error is a
    # bad first minute.
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tests.studio.studiobench.scoring.from_payload import (  # noqa: E402
    ACTION_SOURCES,
    FRAME_METRICS,
    STREAM_METRICS,
    _actions_for,
    _frame_measures,
    _stream_measures,
)

METRICS = tuple(ACTION_SOURCES) + FRAME_METRICS + STREAM_METRICS


def _action_timings(records: list[dict], cid: str) -> dict[str, float]:
    """Every numeric timing on every action that RAN, as `action.timing`.

    Only three action timings are wired into the scoring anchors, and those three do not cover what
    most performance changes actually move: a menus change moves `message_menu`, a model-picker
    change moves `model_change`, a re-open change moves `thread_reopen`, and neither of the latter
    two is scored. Harvesting all of them is what lets each change be judged on its own metric.

    `ran` is checked first. An action that did not happen has no timing worth pairing, and folding
    its absence in as a fast number is the single most common way this harness has produced a
    confident wrong answer.
    """
    out: dict[str, float] = {}
    for name, row in _actions_for(records, cid).items():
        if not row.get("ran"):
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


def cell_metrics(records: list[dict]) -> dict[str, dict[str, float]]:
    """{cell_id: {metric: value}} for every COMPLETED cell in the payload."""
    out: dict[str, dict[str, float]] = {}
    for row in records:
        if row.get("row_type") != "cell" or not row.get("completed"):
            continue
        cid = row["cell_id"]
        vals: dict[str, float] = _action_timings(records, cid)
        windows = [w for w in records if w.get("row_type") == "window" and w.get("cell_id") == cid]
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


def paired(records: list[dict], shard: str = "") -> dict[str, list[tuple[float, float]]]:
    """{metric: [(base, treatment), ...]} matched on (shard, rung, repetition).

    The shard is part of the key because sharding restarts the repetition counter: two independent
    sessions both produce `rep0`, and pairing on the repetition alone would silently overwrite one
    session's base with the other's. Pairing WITHIN a shard is also the correct thing to do, since
    a pair only means anything when both arms come from the same session.
    """
    cells = cell_metrics(records)
    by_key: dict[tuple[str, str, str], dict[str, dict[str, float]]] = collections.defaultdict(dict)
    for cid, vals in cells.items():
        rung = cid.split(".", 1)[0]
        by_key[(shard, rung, rep_of(cid))][arm_of(cid)] = vals
    out: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for sides in by_key.values():
        if "base" not in sides or "treatment" not in sides:
            continue
        for metric in set(sides["base"]) & set(sides["treatment"]):
            b, t = sides["base"][metric], sides["treatment"][metric]
            if b:
                out[metric].append((b, t))
    return out


def tier_of(records: list[dict]) -> str:
    for r in records:
        if r.get("row_type") == "run_meta":
            return str(r.get("tier") or "?")
    return "?"


def corpus_of(records: list[dict]) -> str:
    for r in records:
        if r.get("row_type") == "run_meta":
            return str(r.get("corpus_hash") or "?")
    return "?"


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
        tiers.add(tier_of(records))
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


def summarise(paths: list[Path]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    pooled, _tiers = load(paths)
    for metric, rows in pooled.items():
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
        }
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


def render(
    paths: list[Path],
    title: str,
    floors: dict | None = None,
    floor_tier: str | None = None,
    floor_corpus: str | None = None,
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
    if tier == "fast":
        print("\n  NOTE: fast tier. These are directions for iteration, not reportable numbers.")
    print(f"\n{title}")
    print(f"  payloads: {len(paths)} shard(s): {', '.join(p.parent.name for p in paths)}")
    head = f"  {'metric':<28}{'n':>3}{'base':>11}{'treat':>11}{'delta %':>10}{'spread %':>10}"
    print(head + ("      floor %  verdict" if floors else ""))
    print("  " + "-" * (len(head) + (26 if floors else 0)))
    survivors = 0
    for metric in sorted(stats, key = lambda m: (m in METRICS, m)):
        s = stats[metric]
        line = (
            f"  {metric:<28}{s['n']:>3}{s['base']:>11.1f}{s['treat']:>11.1f}"
            f"{s['delta_pct']:>+10.1f}{s['spread_pct']:>10.1f}"
        )
        if floors is not None:
            f, verdict = verdict_for(s, floors.get(metric), is_count_metric(metric))
            line += (f"{'--':>13}" if f is None else f"{f:>13.1f}") + f"  {verdict}"
            if verdict in ("faster", "SLOWER", "LOST (invariant fell)", "gained"):
                survivors += 1
        print(line)
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

    floors, floor_tier, floor_corpus = None, None, None
    if args.floor:
        floor_paths = shards_of(args.floor)
        if not floor_paths:
            print(f"no null-control payload found for {args.floor}")
            return 2
        floors = summarise(floor_paths)
        floor_rows = read_rows(floor_paths[0])
        floor_tier = tier_of(floor_rows)
        floor_corpus = corpus_of(floor_rows)

    seen = 0
    for arg in args.payloads:
        paths = shards_of(arg)
        if not paths:
            print(f"\nno payload found for {arg}")
            continue
        seen += 1
        render(paths, f"PAIRED PER-METRIC TABLE: {arg}", floors, floor_tier, floor_corpus)
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
