# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The studiobench layer contract. See INTERFACES.md, which this file implements.

Stdlib only, and it imports nothing else from studiobench, so Layer 2 and Layer 3 can import it
without dragging in Playwright, psutil or the fixture generator. `python -c "import
tests.studio.studiobench.runtime.types"` works on a machine with nothing installed, which is what
makes `--doctor` able to report what is missing rather than crash on the way to finding out.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

SCHEMA = "studiobench/1"

ROW_TYPES = frozenset(
    {
        "run_meta",
        "gate",
        "cell",
        "window",
        "action",
        "sample",
        "failure",
        # The A/B run order, recorded before the first cell. Written even when the order is
        # UNBALANCED, because whether linear drift cancelled is a property of the run that a reader
        # of the table has no other way to recover.
        "ab_plan",
        # One UI surface swept by the optional `--surfaces` phase: a route, a settings tab, a menu.
        # A row type of its own rather than an `action` row with a different name, because a surface
        # has no slot, no budget and no timing to miss -- and reusing `action` would put forty rows
        # with a null `timings` into the column the report scores actions from.
        "surface",
    }
)

# Required keys per row type. Enforced in Recorder.emit, because a row that silently lost its
# `ran` flag reads downstream as a fast action rather than as a missing one.
ROW_REQUIRED: dict[str, tuple[str, ...]] = {
    "run_meta": (
        "tier",
        "tool_version",
        "corpus_hash",
        "studio_ref",
        "bundle",
        "platform",
        "started_at",
    ),
    "gate": ("name", "passed", "detail"),
    "cell": ("cell", "completed", "fidelity"),
    "window": ("name", "kind", "t_open_ms", "duration_ms"),
    "action": ("action", "ran", "expect_ok", "expect", "timings", "slot_missed"),
    "sample": ("t_ms",),
    "failure": ("kind", "detail"),
    # `reason` is REQUIRED, not optional. A surface row that lost its reason reads as a surface
    # that was reached, which is the one thing a coverage sweep may never claim by default. It is
    # null only on the success path, where `reached` is true.
    "surface": ("surface", "reached", "reason", "parity"),
}


# ── the cell ────────────────────────────────────────────────────────


@dataclass(frozen = True)
class Cell:
    """One measured configuration: one rung, one arm, one repetition."""

    cell_id: str
    rung: str
    rung_tokens: int
    arm: str = "A0"
    rep: int = 0
    tier: str = "quick"
    transport: str = "provider"
    instrument_level: int = 0
    seed: int = 0
    corpus_hash: str = ""
    session_id: str = ""
    meta: dict = field(default_factory = dict)

    def derive(self, **changes: Any) -> "Cell":
        """A sibling cell with a regenerated `cell_id`. What Layer 3 builds its arms with."""
        out = replace(self, **changes)
        return replace(out, cell_id = make_cell_id(out.rung, out.arm, out.rep))

    def as_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "rung": self.rung,
            "rung_tokens": self.rung_tokens,
            "arm": self.arm,
            "rep": self.rep,
            "tier": self.tier,
            "transport": self.transport,
            "instrument_level": self.instrument_level,
            "seed": self.seed,
            "corpus_hash": self.corpus_hash,
            "session_id": self.session_id,
            "meta": self.meta,
        }


def make_cell_id(rung: str, arm: str, rep: int) -> str:
    return f"r{rung}.{arm}.rep{rep}"


# ── the window ──────────────────────────────────────────────────────

WINDOW_KINDS = frozenset({"action", "stream", "idle", "settle", "teardown"})


@dataclass
class Window:
    """A bracketed interval on the driver's monotonic clock. Windows never nest."""

    name: str
    kind: str
    cell: Cell
    t_open_ms: float
    t_close_ms: Optional[float] = None
    notes: dict = field(default_factory = dict)
    instruments: dict = field(default_factory = dict)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.t_close_ms is None:
            return None
        return round(self.t_close_ms - self.t_open_ms, 2)

    def note(self, key: str, value: Any) -> None:
        self.notes[key] = value

    def row(self) -> dict:
        return {
            "row_type": "window",
            "cell_id": self.cell.cell_id,
            "name": self.name,
            "kind": self.kind,
            "t_open_ms": round(self.t_open_ms, 2),
            "duration_ms": self.duration_ms,
            "instruments": self.instruments,
            "notes": self.notes,
        }


# ── actions ─────────────────────────────────────────────────────────


@dataclass
class ActionResult:
    """The outcome of one action.

    `ran = False` is the ONLY way to report an action that did not happen. It is never a fast
    timing: `timings` is forced empty in that case by __post_init__, so a caller that forgets
    cannot leak a paint-floor number into a table as if it were a measurement.
    """

    ran: bool
    expect_ok: Optional[bool] = None
    expect: dict = field(default_factory = dict)
    timings: dict = field(default_factory = dict)
    reason: Optional[str] = None
    slot_missed: bool = False

    def __post_init__(self) -> None:
        if not self.ran:
            self.timings = {}
            self.expect_ok = None
            if not self.reason:
                self.reason = "action did not run and gave no reason"
        elif self.expect_ok is False and not self.reason:
            self.reason = "expectation failed and gave no reason"

    def row(self, action: str, window: str, cell_id: str) -> dict:
        return {
            "row_type": "action",
            "cell_id": cell_id,
            "action": action,
            "window": window,
            "ran": self.ran,
            "expect_ok": self.expect_ok,
            "expect": self.expect,
            "timings": self.timings,
            "reason": self.reason,
            "slot_missed": self.slot_missed,
        }


def not_run(
    reason: str,
    *,
    slot_missed: bool = False,
    expect: Optional[dict] = None,
) -> ActionResult:
    return ActionResult(ran = False, reason = reason, slot_missed = slot_missed, expect = expect or {})


@dataclass(frozen = True)
class Slot:
    """A fixed (start, budget) on the session wall clock. The scene is a film, not a task list."""

    action: str
    t_start_ms: int
    budget_ms: int
    args: dict = field(default_factory = dict)
    required: bool = True


@dataclass
class ActionContext:
    page: Any
    cdp: Any
    cell: Cell
    window: Window
    args: dict
    budget_ms: int
    dom: Any
    log: Callable[[str], None]


# ── instruments ─────────────────────────────────────────────────────


class Instrument:
    """Base class. Subclassing is optional; duck typing on `name`/`level` is enough."""

    name: str = "unnamed"
    level: int = 0

    def attach(self, ctx: "BenchContext") -> None: ...
    def start_cell(self, cell: Cell) -> None: ...
    def open(self, window: Window) -> None: ...
    def close(self, window: Window) -> Optional[dict]:
        return None

    def end_cell(self, cell: Cell) -> Optional[dict]:
        return None

    def detach(self) -> None: ...


# ── paths and context ───────────────────────────────────────────────


@dataclass
class Paths:
    out: Path
    payload_jsonl: Path
    traces: Path
    symbols: Path
    corpus: Path
    logs: Path

    @classmethod
    def under(cls, out: Path) -> "Paths":
        out = Path(out).resolve()
        p = cls(
            out = out,
            payload_jsonl = out / "payload.jsonl",
            traces = out / "traces",
            symbols = out / "symbols",
            corpus = out / "corpus",
            logs = out / "logs",
        )
        for d in (p.out, p.traces, p.symbols, p.corpus, p.logs):
            d.mkdir(parents = True, exist_ok = True)
        return p


@dataclass
class BenchContext:
    browser: Any = None
    context: Any = None
    page: Any = None
    cdp: Any = None
    base_url: str = ""
    session_id: str = ""
    tier: str = "quick"
    instrument_level: int = 0
    paths: Optional[Paths] = None
    recorder: Optional["Recorder"] = None
    log: Callable[[str], None] = print
    browser_procs: list = field(default_factory = list)


# ── the recorder ────────────────────────────────────────────────────


class Recorder:
    """Append-only JSONL. Every line is flushed and fsynced, so a renderer crash at rung 4 still
    ships rungs 1 to 3 plus the crash record."""

    def __init__(
        self,
        path: Path,
        session_id: str,
        t0: Optional[float] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self.session_id = session_id
        self.t0 = t0 if t0 is not None else time.monotonic()
        self._fh = self.path.open("a", encoding = "utf-8")
        self._count = 0

    def now_ms(self) -> float:
        return round((time.monotonic() - self.t0) * 1000, 2)

    def emit(self, row: dict) -> None:
        row_type = row.get("row_type")
        if row_type not in ROW_TYPES:
            raise ValueError(f"row_type must be one of {sorted(ROW_TYPES)}, got {row_type!r}")
        missing = [k for k in ROW_REQUIRED.get(row_type, ()) if k not in row]
        if missing:
            raise ValueError(f"{row_type} row is missing required keys: {missing}")
        row.setdefault("schema", SCHEMA)
        row.setdefault("ts_ms", self.now_ms())
        row.setdefault("session_id", self.session_id)
        # default = str so a stray Path or dataclass degrades to a string instead of losing the
        # whole row, and the run keeps going.
        self._fh.write(json.dumps(row, default = str) + "\n")
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())
        except OSError:
            pass
        self._count += 1

    def gate(
        self,
        name: str,
        passed: bool,
        detail: Optional[dict] = None,
    ) -> None:
        self.emit(
            {"row_type": "gate", "name": name, "passed": bool(passed), "detail": detail or {}}
        )

    def failure(
        self,
        cell_id: Optional[str],
        kind: str,
        detail: Optional[dict] = None,
    ) -> None:
        self.emit({"row_type": "failure", "cell_id": cell_id, "kind": kind, "detail": detail or {}})

    def rows(self, row_type: Optional[str] = None) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding = "utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row_type is None or row.get("row_type") == row_type:
                    yield row

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]


__all__ = [
    "SCHEMA",
    "ROW_TYPES",
    "ROW_REQUIRED",
    "Cell",
    "make_cell_id",
    "Window",
    "WINDOW_KINDS",
    "ActionResult",
    "not_run",
    "Slot",
    "ActionContext",
    "Instrument",
    "Paths",
    "BenchContext",
    "Recorder",
    "new_session_id",
]
