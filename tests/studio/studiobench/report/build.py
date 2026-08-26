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
                # A truncated final line is expected when a run was killed mid-write; the
                # records before it are still good and are still reported.
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
        # NOT COMPLETE, RATHER THAN NOT PRESENT. A cell whose thread lost messages, or that stopped
        # following the reply it was measuring, reaches here `completed=True` with a full set of
        # timings, because both gates are advisory where they are emitted. Those timings are
        # CHEAPER than a correct cell's -- fewer rows to render -- and the ladder is ABSOLUTE, with
        # no second arm to contradict them, so the rung was scored against fixed anchors and came
        # out green and fast on a cell whose own self-check had already recorded the loss.
        #
        # It is marked incomplete rather than dropped, which is the distinction this module is
        # built on: `score_rung` gives an incomplete rung 0 and KEEPS ITS WEIGHT, and aggregating
        # over only the rungs that survived is the crash-beats-limp bug wearing a different hat. A
        # build whose thread loses its middle at 100K is not usable at 100K, and lowering `onset`
        # is exactly the honest way to say so. Dropping the cell would delete the most important
        # thing the run has to say.
        elif str(r.get("cell_id")) in gate_failures:
            completed = False
            reason = gate_failures[str(r.get("cell_id"))]
        # If a rung was repeated and any rep failed, the rung is not clean. Recording the failure
        # rather than the success is deliberate: the interesting fact about a bimodal rung is
        # that it can fail, not that it can pass.
        prev = out.get(int(tokens))
        if prev is None or (prev[0] and not completed):
            out[int(tokens)] = (completed, reason)
    return out


def score_payload(path: str | Path, declared_rungs: Sequence[int] | None = None) -> LadderScore:
    """Score one run. `declared_rungs` is the ladder the tier promised, in tokens."""

    # BEFORE anything is scored, and against the RAW rows. `--report` reads the same file the run
    # wrote, so a probe run that was allowed to print its own A/B table would be scorable a second
    # time here, hours later, by somebody who was not the one who set the variable. It reads the
    # unfiltered rows deliberately: a probed attempt that was later superseded still means this
    # payload was recorded with the camera in the shot.
    raw = _records(path)
    refuse_if_probed(raw, str(path))
    # A CELL THAT WAS RE-RUN IS SCORED ON THE RUN THAT FINISHED IT. `--resume` re-runs the cells
    # that died, under the same `cell_id`, into the same file; scoring both attempts as one cell
    # kept the crash forever, so a rung that had since been re-run successfully still came out
    # INCOMPLETE and zero. The dead attempt is still in the payload and still in EXCLUDED CELLS,
    # which is where a superseded crash belongs -- it is only kept out of the score.
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

    # FIRST, before the payload is assembled. `score_payload` refuses too, but it runs second
    # here, and `assemble_rows` validates the schema on the way past: a probed payload that also
    # trips some unrelated schema complaint would report THAT instead, and the refusal would
    # never be reached. A refusal that any other failure can pre-empt is not a refusal.
    refuse_if_probed(_records(path), str(path))
    payload = assemble_rows(path)
    ladder = score_payload(path, declared_rungs)
    text = render_summary(payload, ladder, extra_sections = extra_sections)
    return text, ladder, payload
