# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Appends every window to JSONL as produced, so a crash mid-run keeps every rung already written."""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..scoring.from_payload import ATTEMPT_ROW_TYPES
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
    """Flushes each record to the OS without fsync: survives crashes and SIGKILL, not power loss."""

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
    """A kill mid-write leaves one partial last line; it is skipped and counted in discarded."""

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
    """excluded_cells is always a list, never absent or null, so the report always says what it dropped."""

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


ROW_TYPE_SECTIONS: Mapping[str, str] = {
    "run_meta": "header",
    "gate": "selfcheck",
    "cell": "cells",
    "window": "windows",
    "action": "actions",
    "sample": "samples",
    "failure": "crashes",
    # Own section: header is collapsed to its first row, which would silently drop this row.
    "ab_plan": "ab_plan",
    "surface": "surfaces",
    # Own section: header keeps only its first row, and these identity fields are exempt from the zero ban.
    "comparability": "comparability",
    # Not an exclusion source: the preceding completed=false cell row already counts the abort.
    "cell_aborted": "aborted_cells",
}


def executed_balance(order: Sequence[Any], attempted: set[str]) -> bool | None:
    """None for other id shapes, which means cannot tell rather than unbalanced."""

    labels: set[str] = set()
    first: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for cell_id in order:
        if str(cell_id) not in attempted:
            continue
        try:
            head, arm, rep = str(cell_id).rsplit(".", 2)
        except ValueError:
            return None
        labels.add(arm)
        if (head, rep) in seen:
            continue
        seen.add((head, rep))
        first[arm] = first.get(arm, 0) + 1
    if not labels:
        return None
    return len(labels) > 1 and len({first.get(label, 0) for label in labels}) == 1


def merged_ab_plan(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merges all ab_plan rows; balanced is recomputed per session over the cells it actually attempted."""

    plans = [r for r in records if r.get("row_type") == "ab_plan"]
    if not plans:
        return {}
    plan = dict(plans[0])
    plan["sessions"] = [row.get("session_id") for row in plans]
    for stamp in ("session_id", "ts_ms"):
        plan.pop(stamp, None)
    order: list[Any] = []
    for row in plans:
        for cell_id in row.get("order", []):
            if cell_id not in order:
                order.append(cell_id)
    plan["order"] = order
    owner: dict[str, Any] = {}
    for record in records:
        if record.get("row_type") in ATTEMPT_ROW_TYPES and record.get("cell_id") is not None:
            owner[str(record["cell_id"])] = record.get("session_id")
    owning = set(owner.values())
    live = [row for row in plans if row.get("session_id") in owning] or [plans[-1]]
    verdicts = []
    for row in live:
        session = row.get("session_id")
        attempted = {cell for cell, owned_by in owner.items() if owned_by == session}
        ran = executed_balance(row.get("order", []), attempted)
        verdicts.append(bool(row.get("balanced")) if ran is None else ran)
    plan["balanced"] = all(verdicts)
    return plan


def assemble_rows(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Harness rows have no footer, so completeness is inferred from run_meta and completed cell rows."""

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
    payload: dict[str, Any] = {
        "schema": "studiobench/payload/1",
        "source": "recorder_rows",
        "complete": bool(sections.get("header")) and bool(completed_cells),
        "truncated_records": discarded,
        "record_counts": {name: len(rows) for name, rows in sections.items() if rows},
        "header": sections.get("header", [{}])[0] if sections.get("header") else {},
        "selfcheck": sections.get("selfcheck", []),
        "windows": sections.get("windows", []),
        "actions": sections.get("actions", []),
        "cells": cells,
        "samples": sections.get("samples", []),
        "surfaces": sections.get("surfaces", []),
        "aborted_cells": sections.get("aborted_cells", []),
        "comparability": (sections["comparability"][0] if sections.get("comparability") else {}),
        "ab_plan": merged_ab_plan(records),
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
    """A cell is excluded if it did not complete, failed a gate, or an action's own assertion failed."""

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
