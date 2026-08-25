# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Incremental JSONL payload: what survives when the renderer dies at rung 4.

A benchmark that builds its result in memory and writes it at the end has one output state and
one failure state, and the failure state is an empty directory. The runs that matter most are
exactly the ones that fail: a build that kills the renderer at 500K tokens is the most
interesting result the tool can produce, and losing the three rungs that DID complete because the
fourth crashed the browser turns the best evidence into no evidence.

So every window is appended to a JSONL file and flushed to the OS as it is produced. A crash at
rung 4 leaves rungs 1-3 on disk plus, if the harness got the chance, a `crash` record naming what
happened. `assemble()` reads whatever is there, tolerates a half-written final line (a process
killed mid-write leaves one), and reports how many records it had to discard rather than pretending
the file was complete.

`assemble()` also runs `validate_payload()`, so the schema-level ban on bare zeros is enforced on
the real payload and not only in the unit tests.

RECORD KINDS, all of which carry `kind` and `at_ms`:
    header      one per run, first: identity, machine, bench version, instrument levels
    selfcheck   the integrity gates and their verdicts; if any failed the run should have aborted
    window      one measured window: rung, arm, action, frame stats, metrics
    arm         one ablation arm outcome: invariance verdict, potency counters, cost
    excluded    one excluded cell with its reason
    crash       something died; carries the last sample row and RSS at death
    footer      one per run, last: totals, wall time, exit status
A file with no footer record is a run that did not finish, which `assemble()` reports as
`complete: false` rather than silently treating as finished.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..scoring.schema import ExcludedCell, Measure, validate_payload

RECORD_KINDS = (
    "header",
    "selfcheck",
    "window",
    "arm",
    "excluded",
    "crash",
    "footer",
)


def encode(node: Any) -> Any:
    """Recursively turn harness objects into JSON-safe data, preserving measure semantics."""

    if isinstance(node, Measure):
        return node.to_json()
    if isinstance(node, ExcludedCell):
        return node.to_json()
    if dataclasses.is_dataclass(node) and not isinstance(node, type):
        if hasattr(node, "to_json"):
            return encode(node.to_json())
        return {k: encode(v) for k, v in dataclasses.asdict(node).items()}
    if isinstance(node, Mapping):
        return {str(k): encode(v) for k, v in node.items()}
    if isinstance(node, (list, tuple, set)):
        return [encode(v) for v in node]
    if isinstance(node, Path):
        return str(node)
    return node


class PayloadWriter:
    """Append-only JSONL writer. One line per record, flushed as it is written.

    `fsync` is optional and off by default: it costs a few milliseconds per record, which on a
    per-window cadence is measurable against the thing being measured. The default (flush to the
    OS, no fsync) survives a renderer crash, a Python exception and a `SIGKILL` of the driver,
    which is every failure this file exists for; it does not survive the machine losing power,
    which is not a case worth slowing the benchmark down for.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fsync: bool = False,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self._fh = self.path.open("a", encoding = "utf-8")
        self._fsync = bool(fsync)
        self._started = time.monotonic()
        self.records_written = 0

    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        if kind not in RECORD_KINDS:
            raise ValueError(f"unknown record kind {kind!r}; expected one of {RECORD_KINDS}")
        record = {
            "kind": kind,
            "at_ms": round((time.monotonic() - self._started) * 1000.0, 3),
            **{k: encode(v) for k, v in fields.items()},
        }
        self._fh.write(json.dumps(record, separators = (",", ":"), sort_keys = False) + "\n")
        self._fh.flush()
        if self._fsync:
            os.fsync(self._fh.fileno())
        self.records_written += 1
        return record

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "PayloadWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # Best effort: if the driver is dying, say so in the file before it goes.
            try:
                self.write(
                    "crash",
                    where = "driver",
                    error_type = getattr(exc_type, "__name__", str(exc_type)),
                    error = str(exc),
                )
            except Exception:
                pass
        self.close()


def read_records(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Read a JSONL payload, tolerating a truncated final line.

    Returns `(records, discarded)`. `discarded` is almost always 0 or 1: a process killed
    mid-write leaves at most one partial line, and more than that means the file was corrupted
    some other way, which the report prints rather than hides.
    """

    records: list[dict[str, Any]] = []
    discarded = 0
    file_path = Path(path)
    if not file_path.exists():
        return records, discarded
    with file_path.open("r", encoding = "utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                discarded += 1
    return records, discarded


def assemble(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Turn a JSONL stream into one payload dict, complete or not.

    `excluded_cells` is materialised here and is always a list, never absent and never null,
    because a report that cannot say what it dropped is a report that dropped things silently.
    """

    records, discarded = read_records(path)
    by_kind: dict[str, list[dict[str, Any]]] = {kind: [] for kind in RECORD_KINDS}
    for record in records:
        by_kind.setdefault(record.get("kind", "unknown"), []).append(record)

    header = by_kind["header"][0] if by_kind["header"] else {}
    footer = by_kind["footer"][-1] if by_kind["footer"] else None

    payload: dict[str, Any] = {
        "schema": "studiobench/payload/1",
        "complete": footer is not None,
        "truncated_records": discarded,
        "record_counts": {k: len(v) for k, v in by_kind.items() if v},
        "header": header,
        "selfcheck": by_kind["selfcheck"],
        "windows": by_kind["window"],
        "arms": by_kind["arm"],
        "crashes": by_kind["crash"],
        "footer": footer,
        "excluded_cells": [
            {
                "cell_id": rec.get("cell_id", "unknown"),
                "reason": rec.get("reason", "unknown"),
                "count": int(rec.get("count", 1)),
                "detail": rec.get("detail"),
            }
            for rec in by_kind["excluded"]
        ],
    }
    if not payload["complete"]:
        payload["incomplete_note"] = (
            "no footer record: this run did not reach the end. Everything above it was still "
            "measured and is reported; nothing below it exists."
        )
    if validate:
        validate_payload(payload)
    return payload


#: How the harness layer's `row_type` values map onto the sections of an assembled payload.
#: Layer 1 writes rows through its `Recorder`; this layer reads them. Keeping the mapping in one
#: table rather than spread through the renderer means a new row type is one line here and a
#: visible `unknown_rows` entry until somebody decides where it belongs.
ROW_TYPE_SECTIONS: Mapping[str, str] = {
    "run_meta": "header",
    "gate": "selfcheck",
    "cell": "cells",
    "window": "windows",
    "action": "actions",
    "sample": "samples",
    "failure": "crashes",
    # Bookkeeping about HOW the A/B was run, not a measurement of the app. A section of its own
    # BESIDE the identity fields rather than inside them: `header` is reported as a single record,
    # so filing a second row type there had the mapping claim a destination and the reader throw
    # the row away (#9580). See `COLLAPSED_SECTIONS`.
    "ab_plan": "ab_plan",
    # The optional surface sweep. Its own section: a surface row is a coverage fact about the UI,
    # not a timing, and folding it into `actions` would put it in front of the scorer.
    "surface": "surfaces",
    # The comparability key. Its own section rather than `header`, for two independent reasons.
    #
    # First, `header` is collapsed to its FIRST row when the payload is assembled, so a second row
    # filed there is dropped without a word. Second, the row's `fields` block is identity
    # bookkeeping and not a measurement: `instrument_level: 0` is a true statement about how the
    # run was instrumented, exactly like the `identity` and `config` subtrees, so the section is
    # exempted from the bare-zero ban rather than made to fake a Measure. Left unmapped the row
    # fell into `unknown_rows`, which nothing exempts, and the walker killed every real-path
    # session on `$.unknown_rows[0].fields.instrument_level = 0`.
    "comparability": "comparability",
    # The terminal marker for a cell that did not finish. NOT `cells`, which is what the scorer
    # reads, and NOT an exclusion source: the `cell` row it follows is emitted with
    # `completed: false` immediately before it on the same path, and `excluded_from_rows` already
    # turns that into a `rung_incomplete` exclusion. Filing this row as a second exclusion would
    # count one abort twice. It exists so a reader scanning FORWARD can discard the cell's window
    # rows, and that is all it is kept for here.
    "cell_aborted": "aborted_cells",
}


#: The sections reported as ONE record rather than a list, and the row type each collapses to.
#: A section named here may hold exactly one row type: a second type filed into it is routed by
#: `ROW_TYPE_SECTIONS` and then discarded by the collapse, with no `unknown_rows` entry to show
#: for it. `report/selftest/test_studiobench_payload_sections.py` enforces that.
COLLAPSED_SECTIONS: Mapping[str, str] = {
    "header": "run_meta",
    "ab_plan": "ab_plan",
}


def _collapsed(sections: Mapping[str, list[dict[str, Any]]], name: str) -> dict[str, Any]:
    """The single record a collapsed section reports, chosen by row type rather than position.

    The FIRST match, not the last, because `Recorder` appends: a resumed payload holds several
    sessions, and `header` and `ab_plan` have to describe the same one.
    """

    row_type = COLLAPSED_SECTIONS[name]
    for row in sections.get(name, []):
        if row.get("row_type") == row_type:
            return row
    return {}


def assemble_rows(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Assemble a payload from the HARNESS layer's row stream (`row_type`, not `kind`).

    Two writers exist on purpose and they are not redundant. `PayloadWriter` is this layer's own
    stream, used by the ablation batches, which run outside a Layer 1 session and have no
    Recorder. `Recorder` is Layer 1's, and its rows are what a full session produces. Both end up
    in the same assembled shape so the renderer has one input, and neither has to know about the
    other while it is writing.

    A completed run is one that emitted at least one `run_meta` row and at least one `cell` row
    with `completed` true. There is no footer row in the harness contract, so completeness is
    inferred from content rather than from a marker that a crash would remove.
    """

    records, discarded = read_records(path)
    sections: dict[str, list[dict[str, Any]]] = {
        name: [] for name in sorted(set(ROW_TYPE_SECTIONS.values()))
    }
    unknown: list[dict[str, Any]] = []
    for record in records:
        row_type = record.get("row_type")
        section = ROW_TYPE_SECTIONS.get(str(row_type)) if row_type else None
        if section is None:
            unknown.append(record)
            continue
        sections[section].append(record)

    cells = sections.get("cells", [])
    completed_cells = [c for c in cells if c.get("completed") is True]
    # The `run_meta` row itself, not merely a non-empty `header` section: the section used to hold
    # `ab_plan` too, so an emptiness test read a run that had recorded only its A/B order as one
    # that had recorded its identity.
    header = _collapsed(sections, "header")
    payload: dict[str, Any] = {
        "schema": "studiobench/payload/1",
        "source": "recorder_rows",
        "complete": bool(header) and bool(completed_cells),
        "truncated_records": discarded,
        "record_counts": {name: len(rows) for name, rows in sections.items() if rows},
        "header": header,
        "ab_plan": _collapsed(sections, "ab_plan"),
        "selfcheck": sections.get("selfcheck", []),
        "windows": sections.get("windows", []),
        "actions": sections.get("actions", []),
        "cells": cells,
        "samples": sections.get("samples", []),
        "surfaces": sections.get("surfaces", []),
        "aborted_cells": sections.get("aborted_cells", []),
        "comparability": (sections["comparability"][0] if sections.get("comparability") else {}),
        "crashes": sections.get("crashes", []),
        "arms": [],
        "unknown_rows": unknown,
        "footer": None,
        "excluded_cells": excluded_from_rows(records),
    }
    if not payload["complete"]:
        payload["incomplete_note"] = (
            "no run_meta row, or no cell completed. Everything that WAS measured is reported; "
            "nothing that was not is invented"
        )
    if validate:
        validate_payload(payload)
    return payload


def excluded_from_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Derive `excluded_cells` from the harness rows that describe an exclusion.

    Three row shapes mean a cell does not enter scoring, and all three have been seen to vanish
    from a report by simply not being looked for: a cell that did not complete, a failed gate,
    and an action that ran but whose own assertion said it did not do what it claimed.
    """

    out: list[dict[str, Any]] = []
    for row in records:
        row_type = row.get("row_type")
        if row_type == "cell" and row.get("completed") is False:
            out.append(
                {
                    "cell_id": row.get("cell_id", "unknown"),
                    "reason": "rung_incomplete",
                    "count": 1,
                    "detail": str(
                        row.get("failure_mode") or row.get("reason") or "cell did not complete"
                    ),
                }
            )
        elif row_type == "gate" and row.get("passed") is False:
            out.append(
                {
                    "cell_id": row.get("cell_id") or "run",
                    "reason": "selfcheck_failed",
                    "count": 1,
                    "detail": f"gate {row.get('name')}: {row.get('detail')}",
                }
            )
        elif row_type == "action" and row.get("ran") is True and row.get("expect_ok") is False:
            out.append(
                {
                    "cell_id": row.get("cell_id", "unknown"),
                    "reason": "slot_missed",
                    "count": 1,
                    "detail": (
                        f"action {row.get('action')} ran but its own assertion failed: "
                        f"{row.get('reason')}. Its timings exist and must not be quoted"
                    ),
                }
            )
        elif row_type == "failure":
            out.append(
                {
                    "cell_id": row.get("cell_id") or "run",
                    "reason": "renderer_crash",
                    "count": 1,
                    "detail": f"{row.get('kind')}: {row.get('detail')}",
                }
            )
    return out


def iter_windows(payload: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    for window in payload.get("windows", []):
        yield window


def excluded_totals(payload: Mapping[str, Any]) -> dict[str, int]:
    """Per-reason totals for the excluded-cells block. Always rendered, even when empty."""

    totals: dict[str, int] = {}
    for cell in payload.get("excluded_cells", []):
        reason = cell.get("reason", "unknown")
        totals[reason] = totals.get(reason, 0) + int(cell.get("count", 1))
    return totals


def write_excluded(writer: PayloadWriter, cells: Iterable[ExcludedCell]) -> int:
    written = 0
    for cell in cells:
        writer.write("excluded", **cell.to_json())
        written += 1
    return written
