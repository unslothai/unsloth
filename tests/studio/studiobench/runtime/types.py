# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stdlib only and no studiobench imports, so --doctor can report missing deps rather than crash."""

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
        # Written even when unbalanced: whether linear drift cancelled is otherwise unrecoverable.
        "ab_plan",
        "cell_aborted",
        # Everything that must match for two payloads to be comparable, hashed into one token.
        "comparability",
        # Own row type: a surface has no slot or timing, and `action` rows would pollute scoring.
        "surface",
    }
)

# Enforced in Recorder.emit: a row missing `ran` would read as a fast action.
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
    "cell_aborted": ("cell_id", "reason"),
    "comparability": ("key", "fields"),
    "sample": ("t_ms",),
    "failure": ("kind", "detail"),
    # `reason` required: a row missing it would read as a reached surface.
    "surface": ("surface", "reached", "reason", "parity"),
}


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


# `gap` is the quiet stretch between slots, not `stream`. See SceneRunner._gap_window.
# `setup` is driver-dominated pre-film work, unscored. See from_payload.UNSCORED_WINDOW_KINDS.

WINDOW_KINDS = frozenset({"action", "stream", "gap", "idle", "setup", "settle", "teardown"})


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


@dataclass
class ActionResult:
    """ran = False is the only way to report an action that did not happen; timings are forced empty."""

    ran: bool
    expect_ok: Optional[bool] = None
    expect: dict = field(default_factory = dict)
    timings: dict = field(default_factory = dict)
    # Correctness invariants, kept apart from timings; only meaningful paired against the other arm.
    counts: dict = field(default_factory = dict)
    reason: Optional[str] = None
    slot_missed: bool = False

    def __post_init__(self) -> None:
        if not self.ran:
            self.timings = {}
            # An action that did not happen has no invariant; a zero would read as a done job.
            self.counts = {}
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
            "counts": self.counts,
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


class OutDirLock:
    """Taken by run() before anything moves or starts; a refusal must leave the live payload untouched."""

    def __init__(self, out: Path) -> None:
        self.out = Path(out)
        self.path = self.out / ".running.lock"
        self._fd: Optional[int] = None

    @classmethod
    def take(
        cls,
        out: Path,
        session_id: str = "starting",
    ) -> "OutDirLock":
        """The refusal names the holder from the marker; the default session_id stands in until claim()."""
        lock = cls(out)
        lock.out.mkdir(parents = True, exist_ok = True)
        # Legacy per-session marker names are still checked, but only the fixed name is a mutex.
        lock._refuse_if_legacy_marker_is_live()
        lock._acquire(session_id)
        return lock

    def claim(self, session_id: str) -> None:
        """Names the session in the marker, so a contender's refusal can say who holds the directory."""
        if self._fd is not None:
            self._write(session_id)

    def _acquire(self, session_id: str) -> None:
        """A kernel lock, since reclaiming a dead pid's O_EXCL marker races; the file is never unlinked."""
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            self._lock_fd_exclusive(fd)
        except OSError:
            held = self._read_marker_once_written(self.path)
            os.close(fd)
            who = (
                f"session {held[0]} is still running as pid {held[1]}"
                if held
                else "another run is still holding it"
            )
            raise SystemExit(
                f"refusing to append to {self.out}: {who}. Two concurrent runs sharing "
                f"one --out contend with each other and write the same cell ids into one file. "
                f"Give this run its own --out."
            ) from None
        self._fd = fd
        self._write(session_id)

    def _write(self, session_id: str) -> None:
        fd = self._fd
        if fd is None:
            return
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, f"{os.getpid()} {session_id}\n".encode())
        try:
            os.fsync(fd)
        except OSError:
            pass

    def release(self) -> None:
        """Idempotent. Releases the lock but never unlinks the file, which would reopen the lock race."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        # Blank the marker under the lock before release so a stale `pid session` is never read as holder.
        try:
            os.ftruncate(fd, 0)
        except OSError:
            pass
        self._unlock_fd(fd)
        try:
            os.close(fd)
        except OSError:
            pass

    @staticmethod
    def _lock_fd_exclusive(fd: int) -> None:
        """Take a non-blocking exclusive lock, raising OSError if somebody else holds it."""
        if os.name == "nt":  # pragma: no cover - exercised on the Windows CI leg
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _unlock_fd(fd: int) -> None:
        if os.name == "nt":  # pragma: no cover - exercised on the Windows CI leg
            import msvcrt
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            import fcntl
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass

    @staticmethod
    def _read_marker(path: Path) -> "Optional[tuple[str, int]]":
        """(session, pid) written in a marker, or None if it does not yet say."""
        try:
            parts = path.read_text(encoding = "utf-8").split()
            return (
                parts[1] if len(parts) > 1 else path.name.removeprefix(".running."),
                int(parts[0]),
            )
        except (OSError, ValueError, IndexError):
            return None

    @classmethod
    def _read_marker_once_written(
        cls,
        path: Path,
        budget_s: float = 0.5,
    ) -> "Optional[tuple[str, int]]":
        """The holder locks before it writes, so wait out the empty marker; a single read would name
        no one."""
        deadline = time.monotonic() + budget_s
        while True:
            got = cls._read_marker(path)
            # The marker is never unlinked, so a retained record only counts as a holder if its pid is alive.
            if got is not None and cls._alive(got[1]):
                return got
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.01)

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def _refuse_if_legacy_marker_is_live(self) -> None:
        """Honour `.running.<session>` markers from older builds, and clear the dead ones."""
        for other in sorted(self.out.glob(".running.*")):
            if other == self.path:
                continue
            got = self._read_marker(other)
            if got is not None and self._alive(got[1]):
                session, pid = got
                raise SystemExit(
                    f"refusing to append to {self.out}: session {session} is still "
                    f"running as pid {pid}. Two concurrent runs sharing one --out contend with "
                    f"each other and write the same cell ids into one file. Give this run its "
                    f"own --out."
                )
            other.unlink(missing_ok = True)


class Recorder:
    """Append-only JSONL. Every line is flushed and fsynced, so a renderer crash at rung 4 still
    ships rungs 1 to 3 plus the crash record."""

    def __init__(
        self,
        path: Path,
        session_id: str,
        t0: Optional[float] = None,
        lock: Optional[OutDirLock] = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents = True, exist_ok = True)
        self.session_id = session_id
        self.t0 = t0 if t0 is not None else time.monotonic()
        # Refuse a second live session in one output directory: concurrent runs corrupt each other.
        # One fixed name via O_CREAT|O_EXCL (no fcntl on Windows); only a self-taken lock is released.
        self._owns_lock = lock is None
        if lock is None:
            lock = OutDirLock.take(self.path.parent, session_id)
        else:
            lock.claim(session_id)
        self._lock = lock
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
        # default = str so a stray Path or dataclass degrades to a string instead of losing the whole row.
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
        cell_id: Optional[str] = None,
    ) -> None:
        """Pass cell_id whenever the verdict is about a cell; without it the failure is attributed
        to run."""
        row = {"row_type": "gate", "name": name, "passed": bool(passed), "detail": detail or {}}
        if cell_id is not None:
            row["cell_id"] = cell_id
        self.emit(row)

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
        # Release only a self-taken lock: `run()` reads the payload back after close under its own lock.
        lock = getattr(self, "_lock", None)
        if lock is not None and getattr(self, "_owns_lock", True):
            lock.release()


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
