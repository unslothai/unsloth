# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Builds share one session: cross-session drift can exceed the wins measured, and order alternates."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

from ..fixture.corpus import Corpus, RungPlan
from .types import Cell


@dataclass
class Target:
    """One side of the comparison: an Unsloth to drive and everything needed to drive it."""

    label: str  # "base" or "treatment"
    ref: str
    base_url: str
    seeder: Any
    runner: Any  # a CellRunner bound to this target's base_url and seeder
    install: Any = None
    owns_studio: bool = False


# Default ports are omitted by window.location.origin, so an explicit one is the same origin.
DEFAULT_PORTS = {"http": 80, "https": 443, "ws": 80, "wss": 443}


def browser_origin(url: str) -> str:
    """Canonical origin as window.location.origin spells it; localhost and 127.0.0.1 stay distinct."""
    from urllib.parse import urlsplit

    try:
        split = urlsplit(url.strip())
        host, port = split.hostname, split.port
    except ValueError:
        return url.rstrip("/")
    scheme = split.scheme.lower()
    if not scheme or not host:
        return url.rstrip("/")
    host = host.lower()
    if ":" in host:  # IPv6, which serialises with its brackets
        host = f"[{host}]"
    if port is None or port == DEFAULT_PORTS.get(scheme):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def origin_scoped(base_url: str, script: str) -> str:
    """Seeds the page only on its own build's origin, since both builds share localStorage key names."""
    import json as _json
    return (
        "(() => { if (window.location.origin !== "
        + _json.dumps(browser_origin(base_url))
        + ") return; "
        + script
        + " })();"
    )


def interleave(
    cells: list[tuple[Cell, RungPlan]], targets: list[Target]
) -> list[tuple[Target, Cell, RungPlan]]:
    """Sides of a pair run adjacent in time, not in halves, so machine drift between them is minimized."""
    out: list[tuple[Target, Cell, RungPlan]] = []
    for cell, plan in cells:
        order = list(targets) if cell.rep % 2 == 0 else list(reversed(targets))
        for target in order:
            out.append((target, cell.derive(arm = target.label), plan))
    return out


def skippable_cells(work: list[tuple[Any, Cell, RungPlan]], done: set) -> set:
    """Skips a (rung, rep) pair only if all its arms are complete; a lone completed arm has no partner."""
    by_pair: dict[tuple[str, int], list[str]] = {}
    for _target, cell, _plan in work:
        by_pair.setdefault((str(cell.rung), int(cell.rep)), []).append(str(cell.cell_id))
    out: set = set()
    for cell_ids in by_pair.values():
        if all(cell_id in done for cell_id in cell_ids):
            out.update(cell_ids)
    if any(len(cell_ids) > 1 for cell_ids in by_pair.values()):
        planned = sum(len(cell_ids) for cell_ids in by_pair.values())
        if len(out) != planned:
            return set()
    return out


def order_is_balanced(plan: list[tuple[Target, Cell, RungPlan]]) -> bool:
    """Reported, not enforced: an unbalanced order still runs, but its drift term must be disclosed."""
    labels = {target.label for target, _cell, _plan in plan}
    first_counts: dict[str, int] = {label: 0 for label in labels}
    seen: set[str] = set()
    for target, cell, _plan in plan:
        key = f"{cell.rung}:{cell.rep}"
        if key in seen:
            continue
        seen.add(key)
        first_counts[target.label] += 1
    # Labels are seeded at zero first, so a single-rep plan is not reported as balanced.
    return len(labels) > 1 and len(set(first_counts.values())) == 1


# Gates whose failure invalidates the whole cell; other gates (e.g. timer_clamp) only null a column.
INVALIDATING_CELL_GATES: frozenset[str] = frozenset({"thread_complete", "follows_the_stream"})


def gate_detail_is_unmeasured(detail: Mapping[str, Any]) -> bool:
    """Absent instruments are waived; no thread viewport is a defect in the arm, not an absent
    instrument."""

    unmeasured = (
        detail.get("follow_attempted") is False
        or detail.get("probe_attempted") is False
        or detail.get("stream_coverage_unmeasured") is True
    )
    return unmeasured and "viewport" not in str(detail.get("reason") or "").lower()


def failed_invalidating_gates(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Only per-cell gates disqualify; a resumed retry must not inherit a dead attempt's failed gate row."""
    winning: dict[str, Any] = {}
    for row in records:
        if row.get("row_type") == "cell" and row.get("cell_id") is not None:
            winning[str(row.get("cell_id"))] = row.get("session_id")

    failed: dict[str, str] = {}
    for row in records:
        if row.get("row_type") != "gate" or row.get("passed") is not False:
            continue
        name = str(row.get("name"))
        if name not in INVALIDATING_CELL_GATES or row.get("cell_id") is None:
            continue
        cell_id = str(row.get("cell_id"))
        keep = winning.get(cell_id)
        if keep is not None and row.get("session_id") not in (None, keep):
            continue
        detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
        # Not measured is not failed; gate_detail_is_unmeasured is shared with sweep/ui_parity.py.
        if gate_detail_is_unmeasured(detail):
            continue
        why = detail.get("reason") or detail.get("coverage_reason") or "the cell's own self-check"
        failed.setdefault(cell_id, f"gate {name}: {why}")
    return failed


def unmeasured_planned_cells(
    records: list[dict],
    planned: Sequence[str],
    session_id: Optional[str] = None,
) -> list[str]:
    """An incomplete plan must void the table: a failed cell also drops its healthy partner from the
    pair."""
    from ..scoring.from_payload import latest_attempt_rows

    # Apply the same two filters as readings_by_arm, so gate-failed cells also void the plan.
    failed = failed_invalidating_gates(records)
    complete: set = set()
    for row in latest_attempt_rows(records):
        if row.get("row_type") != "cell" or row.get("completed") is not True:
            continue
        if session_id is not None and row.get("session_id") not in (None, session_id):
            continue
        if str(row.get("cell_id")) in failed:
            continue
        complete.add(str(row.get("cell_id")))
    return [str(cell_id) for cell_id in planned if str(cell_id) not in complete]


def readings_by_arm(
    records: list[dict], session_id: Optional[str] = None
) -> dict[str, dict[int, dict]]:
    """Incomplete or gate-failed cells are excluded from ratios, or a crash reads as a win."""
    from ..scoring.from_payload import latest_attempt_rows, measures_by_cell

    # Resumed retries reuse the dead attempt's cell_id, so filter to the latest attempt first.
    records = list(latest_attempt_rows(records))
    failed_gates = failed_invalidating_gates(records)

    arms: dict[str, list[dict]] = {}
    cell_ids: dict[str, set[str]] = {}
    for row in records:
        if row.get("row_type") == "cell":
            if row.get("completed") is not True:
                continue
            if str(row.get("cell_id")) in failed_gates:
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
    # Pair per (rung, rep): reps of both arms ran adjacent in time, and the bootstrap needs them all.
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
    parity_raw: bool = False,
    parity_shots = None,
    username: str,
    password: str,
) -> Target:
    """Both sides share one pacer, so wire bytes are identical by construction, not by matching settings."""
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
        parity_raw = parity_raw,
        parity_shots = parity_shots,
        arm_label = label,
    )
    target = Target(label = label, ref = ref, base_url = base_url, seeder = seeder, runner = runner)
    target.auth = auth  # type: ignore[attr-defined]
    target.checkpoint = checkpoint  # type: ignore[attr-defined]
    return target
