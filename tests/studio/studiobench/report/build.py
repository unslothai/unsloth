# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Payload on disk -> scored, rendered summary.

The last mile. Everything either side of it existed: the session layer wrote rows, the scoring
layer scored readings, the renderer rendered scores. Nothing joined them, so a completed run
produced a JSONL file and no report.

The one policy decision that lives here: WHICH RUNGS ARE ON THE LADDER. `score_ladder` demands
every declared rung, present or not, because aggregating over only the rungs that survived is the
crash-beats-limp bug wearing a different hat. So a rung that was declared for the tier and never
produced a cell is passed through as INCOMPLETE with the reason, not quietly dropped.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..scoring.from_payload import (
    latest_attempt_rows,
    measures_from_records,
    refuse_if_probed,
)
from ..scoring.score import LadderScore, RungScore, score_ladder, score_rung
from .payload import assemble_rows
from .render import render_summary


def _records(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with Path(path).open(encoding = "utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _completion_by_rung(records: Sequence[Mapping[str, Any]]) -> dict[int, tuple[bool, str | None]]:
    from ..runtime.ab import failed_invalidating_gates

    gate_failures = failed_invalidating_gates(records)
    out: dict[int, tuple[bool, str | None]] = {}
    for r in records:
        if r.get("row_type") != "cell":
            continue
        tokens = r.get("target_tokens")
        if tokens is None:
            continue
        completed = bool(r.get("completed"))
        failure = r.get("failure") or {}
        reason = None
        if not completed:
            reason = f"{failure.get('kind') or 'unknown'}: {failure.get('message') or 'no message'}"
        # Gate-failed cells are marked incomplete, not dropped, so score_rung keeps their weight at 0.
        elif str(r.get("cell_id")) in gate_failures:
            completed = False
            reason = gate_failures[str(r.get("cell_id"))]
        prev = out.get(int(tokens))
        if prev is None or (prev[0] and not completed):
            out[int(tokens)] = (completed, reason)
    return out


def score_payload(path: str | Path, declared_rungs: Sequence[int] | None = None) -> LadderScore:
    """Score one run. `declared_rungs` is the ladder the tier promised, in tokens."""

    # Refuse probed payloads before scoring, on raw rows, so superseded probed attempts still count.
    raw = _records(path)
    refuse_if_probed(raw, str(path))
    # Score re-run cells on their latest attempt; --resume reuses cell_id in the same file.
    records = latest_attempt_rows(raw)
    measures = measures_from_records(records)
    completion = _completion_by_rung(records)

    rungs = sorted(set(declared_rungs) | set(measures)) if declared_rungs else sorted(measures)

    scored: list[RungScore] = []
    for tokens in rungs:
        if tokens not in measures:
            scored.append(
                score_rung(
                    tokens,
                    {},
                    completed = False,
                    failure_mode = "declared for this tier but no cell was recorded for it",
                )
            )
            continue
        complete, reason = completion.get(tokens, (True, None))
        scored.append(score_rung(tokens, measures[tokens], completed = complete, failure_mode = reason))
    return score_ladder(scored)


def build_report(
    path: str | Path,
    declared_rungs: Sequence[int] | None = None,
    *,
    extra_sections: Sequence[str] = (),
) -> tuple[str, LadderScore, dict[str, Any]]:
    """Return (rendered summary, ladder, assembled payload)."""

    # Refuse first so an unrelated schema error cannot pre-empt the probe refusal.
    refuse_if_probed(_records(path), str(path))
    payload = assemble_rows(path)
    ladder = score_payload(path, declared_rungs)
    text = render_summary(payload, ladder, extra_sections = extra_sections)
    return text, ladder, payload
