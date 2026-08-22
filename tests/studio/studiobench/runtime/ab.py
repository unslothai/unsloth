# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run two Studio builds against each other INSIDE ONE SESSION.

WHY THIS CANNOT BE TWO RUNS. Cross-session drift on this app measured 8%, which is larger than
most of the wins anybody argues about. `scoring.ab.assert_comparable` refuses two different
session ids outright, and that refusal is correct: running the base today and the treatment
tomorrow produces a ratio whose dominant term is the machine, not the change. So both builds are
installed up front, one browser drives both, and the cells alternate.

WHY THE ORDER FLIPS. Anything that drifts monotonically within a session -- thermal throttling, a
browser heap that never quite shrinks, another process ramping up -- is charged entirely to
whichever side runs second if the order is fixed. Alternating (base, treatment) on even reps and
(treatment, base) on odd ones cancels the linear part of that term instead of measuring it. With
`--reps 1` the order cannot be balanced, which is why a single-rep A/B prints the warning it does.

WHAT VOIDS THE RESULT. The null control -- the base build compared against ITSELF, interleaved the
same way -- runs first and must land inside its own noise band. If comparing a build to itself
produces a difference, then a difference between two builds means nothing, and no table is worth
printing. That check is the whole reason this file interleaves at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..fixture.corpus import Corpus, RungPlan
from .types import Cell


@dataclass
class Target:
    """One side of the comparison: a Studio to drive and everything needed to drive it."""

    label: str  # "base" or "treatment"
    ref: str  # the git ref, for the report
    base_url: str
    seeder: Any
    runner: Any  # a CellRunner bound to this target's base_url and seeder
    install: Any = None  # StudioInstall, when we own it
    owns_studio: bool = False


def origin_scoped(base_url: str, script: str) -> str:
    """Run `script` only on its own Studio's origin.

    `add_init_script` fires on every document in the context, and localStorage is per-origin under
    the SAME KEY NAMES on both builds. Seeding both unconditionally means whichever script runs
    last writes the other build's auth token into this build's storage, and the failure shows up
    much later as a logged-out SPA or a provider that renders as "No longer offered" -- neither of
    which points back here.
    """
    import json as _json
    return (
        "(() => { if (window.location.origin !== "
        + _json.dumps(base_url.rstrip("/"))
        + ") return; "
        + script
        + " })();"
    )


def interleave(
    cells: list[tuple[Cell, RungPlan]], targets: list[Target]
) -> list[tuple[Target, Cell, RungPlan]]:
    """Order the work so the two sides sit next to each other in time, not in separate halves.

    Adjacent in time is the point: the closer two paired readings are, the less of whatever the
    machine is doing separates them. Returned as a flat list so the caller's loop stays a loop
    and the ordering decision lives in one testable function.
    """
    out: list[tuple[Target, Cell, RungPlan]] = []
    for cell, plan in cells:
        order = list(targets) if cell.rep % 2 == 0 else list(reversed(targets))
        for target in order:
            out.append((target, cell.derive(arm = target.label), plan))
    return out


def skippable_cells(work: list[tuple[Any, Cell, RungPlan]], done: set) -> set:
    """Of the cells `--resume` COULD skip, the ones it may: only whole `(rung, rep)` pairs.

    A PAIR IS THE UNIT OF AN A/B, NOT A CELL. An interruption between the two adjacent cells of one
    pair is the ordinary way a run stops, and skipping the arm that completed then measured its
    partner ALONE in the new session. That lone reading can never be used: `readings_by_arm` scopes
    the comparison to one session on purpose -- cross-session drift measured 8%, larger than most
    wins anybody argues about -- so the completed arm from the old session is dropped, the new
    arm has nothing to pair with, and the run pays a full cell for a number no table can contain.
    What it renders instead is that repetition missing from the table, or NO READING with an exit
    code of 0 underneath it.

    So a pair is skipped only when EVERY arm of it is already complete, and otherwise both arms are
    re-run -- adjacent in time, in one session, which is the only arrangement `interleave` exists to
    produce. The old attempt stays in the payload and `latest_attempt_rows` supersedes it, exactly
    as it already does for a cell that died.

    Single-target work is a degenerate case of the same rule: each pair holds one cell, so nothing
    changes for a run without `--ab`.
    """
    by_pair: dict[tuple[str, int], list[str]] = {}
    for _target, cell, _plan in work:
        by_pair.setdefault((str(cell.rung), int(cell.rep)), []).append(str(cell.cell_id))
    out: set = set()
    for cell_ids in by_pair.values():
        if all(cell_id in done for cell_id in cell_ids):
            out.update(cell_ids)
    return out


def order_is_balanced(plan: list[tuple[Target, Cell, RungPlan]]) -> bool:
    """True when each side ran first equally often, so linear drift cancels rather than lands.

    Reported rather than enforced: an unbalanced plan is still worth running, it just carries a
    drift term that the reader has to be told about instead of discovering later.
    """
    labels = {target.label for target, _cell, _plan in plan}
    first_counts: dict[str, int] = {label: 0 for label in labels}
    seen: set[str] = set()
    for target, cell, _plan in plan:
        key = f"{cell.rung}:{cell.rep}"
        if key in seen:
            continue
        seen.add(key)
        first_counts[target.label] += 1
    # Every label is seeded at zero first. Counting only the labels that DID run first reports a
    # single-rep plan -- where one side always goes first and nothing cancels -- as balanced,
    # which is the one answer this function exists to prevent.
    return len(labels) > 1 and len(set(first_counts.values())) == 1


def readings_by_arm(
    records: list[dict], session_id: Optional[str] = None
) -> dict[str, dict[int, dict]]:
    """Split one payload's cell rows into `{arm: {rung_tokens: {metric: Measure}}}`.

    A CELL THAT DID NOT COMPLETE IS NOT AN ARM'S READING. The ladder scores an incomplete cell on
    purpose -- a build that dies at 500K is the most important thing the run has to say -- but a
    ratio is a different question. An arm that crashed after emitting one action row still carries
    that row's timings, and pairing them against a completed cell on the other side turns a crash
    into a win: a treatment cell holding nothing but a 50 ms keystroke, measured against a
    completed 100 ms base cell, reports IMPROVED. The crash is still in the payload, in the
    summary and in `excluded_cells`; it is only kept out of the ratios.

    `session_id`, when given, keeps the comparison inside ONE session. `--resume` appends to the
    payload a previous run wrote, so an interrupted A/B resumed into the same directory otherwise
    hands `compare_arms` cells from two browser sessions -- the 8% cross-session drift term that
    `assert_comparable` exists to refuse, arriving through the back door because both sides are
    labelled with the CURRENT session id.

    Deferred import: `scoring` pulls in the anchor table and this module is imported by the CLI
    before a run, where that cost buys nothing.
    """
    from ..scoring.from_payload import latest_attempt_rows, measures_by_cell

    # The session filter below scopes the CELL rows, but `action` and `window` rows are collected
    # by `cell_id` alone, and a resumed retry reuses the cell id of the attempt that died. Without
    # this the completed-cell filter admitted the dead attempt's windows into the retry's reading.
    records = list(latest_attempt_rows(records))

    arms: dict[str, list[dict]] = {}
    cell_ids: dict[str, set[str]] = {}
    for row in records:
        if row.get("row_type") == "cell":
            if row.get("completed") is not True:
                continue
            if session_id is not None and row.get("session_id") not in (None, session_id):
                continue
            arm = str((row.get("cell") or {}).get("arm") or row.get("arm") or "")
            if arm:
                cell_ids.setdefault(arm, set()).add(str(row.get("cell_id")))

    for arm, ids in cell_ids.items():
        subset = [
            r
            for r in records
            if r.get("row_type") not in {"cell", "action", "window"} or str(r.get("cell_id")) in ids
        ]
        arms[arm] = subset

    return {arm: measures_by_cell(rows) for arm, rows in arms.items()}


def compare_arms(
    records: list[dict],
    base_label: str,
    treatment_label: str,
    *,
    bench_version: str,
    corpus_hash: str,
    session_id: str,
    label: str,
    noise_floor_pct: Optional[float] = None,
    noise_floor_source: str = "declared default",
    is_null_control: bool = False,
) -> Any:
    """Build the A/B result for one pair of arms out of an already-recorded payload."""
    from ..scoring.ab import DEFAULT_NOISE_FLOOR_PCT, Pair, RunIdentity, compare
    from ..scoring.anchors import METRIC_BY_KEY, weights_id

    by_arm = readings_by_arm(records, session_id = session_id)
    base = by_arm.get(base_label, {})
    treatment = by_arm.get(treatment_label, {})

    rung_ladder_id = _ladder_id(sorted({rung for rung, _rep in set(base) | set(treatment)}))
    identity_kwargs = dict(
        bench_version = bench_version,
        corpus_hash = corpus_hash,
        rung_ladder_id = rung_ladder_id,
        weights_id = weights_id() if callable(weights_id) else str(weights_id),
        session_id = session_id,
    )
    # Paired PER REPETITION, matching (rung, rep) on both sides. Repetition r of the base and
    # repetition r of the treatment ran adjacent in time, so pairing them is what makes the
    # comparison paired at all; pooling reps into one reading per rung throws away every
    # observation but the first and leaves the bootstrap with nothing to resample.
    pairs = []
    for key in sorted(set(base) & set(treatment)):
        rung, _rep = key
        for metric_key in METRIC_BY_KEY:
            base_measure = base[key].get(metric_key)
            treatment_measure = treatment[key].get(metric_key)
            if base_measure is None or treatment_measure is None:
                continue
            pairs.append(
                Pair(
                    rung_tokens = int(rung),
                    metric_key = metric_key,
                    base = base_measure,
                    treatment = treatment_measure,
                )
            )

    return compare(
        label,
        pairs,
        RunIdentity(**identity_kwargs),
        RunIdentity(**identity_kwargs),
        noise_floor_pct = (DEFAULT_NOISE_FLOOR_PCT if noise_floor_pct is None else noise_floor_pct),
        noise_floor_source = noise_floor_source,
        is_null_control = is_null_control,
    )


def _ladder_id(rungs: list) -> str:
    import hashlib
    digest = hashlib.sha256(",".join(str(int(r)) for r in rungs).encode()).hexdigest()[:12]
    return f"r-{digest}"


def make_target(
    label: str,
    ref: str,
    base_url: str,
    *,
    pacer,
    model_id: str,
    corpus: Corpus,
    tier: str,
    paths,
    log: Callable[[str], None],
    cadence: str,
    image_path,
    session,
    username: str,
    password: str,
) -> Target:
    """Authenticate against one Studio, register the shared pacer on it, and bind a runner.

    Both sides talk to the SAME pacer, so the bytes on the wire are identical by construction
    rather than by two configurations that are meant to match.
    """
    from .lifecycle import authenticate, external_checkpoint_id, pacer_provider, register_provider
    from .seeder import Seeder
    from .session import CellRunner

    auth = authenticate(base_url, username, password)
    provider = pacer_provider(pacer.base_url, [model_id])
    register_provider(base_url, auth, provider)
    checkpoint = external_checkpoint_id(provider, model_id)
    log(f"  {label}: {base_url} -> pacer {pacer.base_url}, checkpoint {checkpoint}")

    seeder = Seeder(base_url = base_url, auth = auth, model_id = model_id, log = log)
    runner = CellRunner(
        session = session,
        pacer = pacer,
        seeder = seeder,
        corpus = corpus,
        base_url = base_url,
        model_id = model_id,
        tier = tier,
        paths = paths,
        log = log,
        cadence = cadence,
        image_path = image_path,
    )
    target = Target(label = label, ref = ref, base_url = base_url, seeder = seeder, runner = runner)
    target.auth = auth  # type: ignore[attr-defined]
    target.checkpoint = checkpoint  # type: ignore[attr-defined]
    return target
