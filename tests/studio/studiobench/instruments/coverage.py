# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exact invocation counts via `Profiler.startPreciseCoverage`.

This is the only non-statistical instrument in the tool. A CPU profile says "a
frame was on the stack for 12% of samples"; precise coverage says "this function
was entered exactly 4,110 times". That integer is what turns

    36% of TaskDuration is unnamed script

into

    cloneChildFibers ran exactly 4,110 times = 6 renders x 685 blocks

and the second statement is falsifiable against a structural quantity we
measured separately. That is the whole point of the layer.

THE COST, STATED UP FRONT: precise coverage makes V8 keep count-collecting
bytecode alive, which suppresses the optimising tiers for covered functions.
Every timing taken while coverage is on is therefore wrong, and wrong in a
direction that varies per function. So this module DISCARDS TIME BY
CONSTRUCTION: `CoverageSnapshot` carries no durations, and the only values it
exposes across the boundary are integers. There is no flag to turn that off,
because the moment a millisecond from this arm reaches a table, every number
next to it becomes unsafe.

`detailed: false` asks for function-level granularity rather than per-block
ranges. The first range of a function covers the whole function, so its `count`
is the invocation count of the function itself; block granularity would give a
larger, less interpretable set of ranges and a much bigger payload for no gain
here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..analysis import CellFailure


@dataclass(frozen = True)
class FunctionCount:
    """One function and the exact number of times it was entered."""

    script_id: str
    url: str
    function_name: str
    start_offset: int
    end_offset: int
    count: int

    @property
    def key(self) -> tuple[str, int, int]:
        """Identity within one build.

        Keyed on script and byte offsets, not on name. Minified code reuses
        names aggressively and gives many functions no name at all, so a
        name-keyed map silently merges unrelated functions.
        """
        return (self.script_id, self.start_offset, self.end_offset)

    def label(self) -> str:
        name = self.function_name or "(anonymous)"
        return f"{name} @ {self.url or 'script#' + self.script_id}[{self.start_offset}:{self.end_offset}]"


@dataclass
class CoverageSnapshot:
    """Counts taken between two `takePreciseCoverage` calls.

    Deliberately has no time fields. Not "unused time fields": none.
    """

    functions: list[FunctionCount] = field(default_factory = list)
    script_urls: dict[str, str] = field(default_factory = dict)
    # Set when this snapshot is a difference of two absolute snapshots.
    is_delta: bool = False

    def by_key(self) -> dict[tuple[str, int, int], FunctionCount]:
        return {f.key: f for f in self.functions}

    def total_calls(self) -> int:
        return sum(f.count for f in self.functions)

    def nonzero(self) -> list[FunctionCount]:
        return [f for f in self.functions if f.count > 0]

    def top(
        self,
        limit: int = 40,
        url_filter: str | None = None,
    ) -> list[FunctionCount]:
        rows = self.nonzero()
        if url_filter:
            rows = [f for f in rows if url_filter in f.url]
        rows.sort(key = lambda f: -f.count)
        return rows[:limit]

    def find(self, name: str) -> list[FunctionCount]:
        return [f for f in self.functions if f.function_name == name]

    def search(self, pattern: str) -> list[FunctionCount]:
        rx = re.compile(pattern)
        return [f for f in self.functions if rx.search(f.function_name or "")]

    def count_vector(self, keys: Sequence[tuple[str, int, int]]) -> tuple[int, ...]:
        m = self.by_key()
        return tuple(m[k].count if k in m else 0 for k in keys)


def _parse(result: dict[str, Any]) -> CoverageSnapshot:
    snap = CoverageSnapshot()
    for script in result.get("result", []):
        sid = str(script.get("scriptId", ""))
        url = str(script.get("url", ""))
        snap.script_urls[sid] = url
        for fn in script.get("functions", []):
            ranges = fn.get("ranges") or []
            if not ranges:
                continue
            # With `detailed: false` there is one range per function and it
            # spans the function; even with block coverage the FIRST range is
            # the function-level one, so this stays correct either way.
            head = ranges[0]
            snap.functions.append(
                FunctionCount(
                    script_id = sid,
                    url = url,
                    function_name = str(fn.get("functionName", "")),
                    start_offset = int(head.get("startOffset", 0)),
                    end_offset = int(head.get("endOffset", 0)),
                    count = int(head.get("count", 0)),
                )
            )
    return snap


class PreciseCoverage:
    """Bracket a window with exact call counts.

    Usage is deliberately two-phase. `takePreciseCoverage` returns counts
    accumulated since coverage started, not since the last call, so a window is
    measured as the difference of two snapshots. Reporting an absolute snapshot
    as if it were a window is how "this ran 4,110 times during the stream"
    becomes "this ran 4,110 times since the page loaded".
    """

    def __init__(
        self,
        cdp: Any,
        *,
        detailed: bool = False,
        allow_triggered_updates: bool = False,
    ) -> None:
        self.cdp = cdp
        self.detailed = detailed
        self.allow_triggered_updates = allow_triggered_updates
        self._started = False
        self._baseline: CoverageSnapshot | None = None

    def start(self) -> None:
        if self._started:
            raise RuntimeError("PreciseCoverage.start called twice")
        self.cdp.send("Profiler.enable")
        self.cdp.send(
            "Profiler.startPreciseCoverage",
            {
                "callCount": True,
                "detailed": self.detailed,
                "allowTriggeredUpdates": self.allow_triggered_updates,
            },
        )
        self._started = True

    def snapshot(self) -> CoverageSnapshot:
        if not self._started:
            raise RuntimeError("PreciseCoverage.snapshot before start")
        return _parse(self.cdp.send("Profiler.takePreciseCoverage"))

    def mark(self) -> None:
        """Take the baseline that a later `window()` is measured against."""
        self._baseline = self.snapshot()

    def window(self) -> CoverageSnapshot:
        """Counts accrued since `mark()`."""
        if self._baseline is None:
            raise RuntimeError("PreciseCoverage.window before mark")
        return diff(self._baseline, self.snapshot())

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.cdp.send("Profiler.stopPreciseCoverage")
        finally:
            self._started = False

    def __enter__(self) -> "PreciseCoverage":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def diff(before: CoverageSnapshot, after: CoverageSnapshot) -> CoverageSnapshot:
    """Counts accrued between two absolute snapshots.

    A function present in `after` but not in `before` is a script compiled
    inside the window; its full count belongs to the window. A count that went
    DOWN is impossible for a monotonic counter and means the two snapshots came
    from different coverage sessions, so it fails rather than clamping.
    """
    prev = before.by_key()
    out = CoverageSnapshot(is_delta = True, script_urls = dict(after.script_urls))
    for f in after.functions:
        base = prev.get(f.key)
        delta = f.count - (base.count if base else 0)
        if delta < 0:
            raise CellFailure(
                "coverage_counter_went_backwards",
                f"{f.label()} counted {base.count if base else 0} then {f.count}; "
                "precise coverage counters are monotonic, so these snapshots are "
                "not from the same coverage session",
            )
        out.functions.append(
            FunctionCount(
                script_id = f.script_id,
                url = f.url,
                function_name = f.function_name,
                start_offset = f.start_offset,
                end_offset = f.end_offset,
                count = delta,
            )
        )
    return out


def assert_integers_only(payload: dict[str, Any]) -> None:
    """Refuse to let a coverage-arm float reach a report.

    Timings from a coverage arm are meaningless because optimised code is
    suppressed. This is the boundary guard: it is called on anything derived
    from a coverage run before it is written out, and it raises on any
    non-integral number. It is a cheap check that makes a silent category error
    into a loud one.
    """

    def check(node: Any, path: str) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, float):
            raise CellFailure(
                "coverage_float_leak",
                f"{path} is a float ({node!r}). Precise coverage disables optimised "
                "code, so every duration measured under it is wrong. Only integers "
                "may cross this boundary.",
            )
        if isinstance(node, dict):
            for k, v in node.items():
                check(v, f"{path}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                check(v, f"{path}[{i}]")

    check(payload, "coverage")


def counts_for(snapshot: CoverageSnapshot, names: Iterable[str]) -> dict[str, int]:
    """Total exact calls per function NAME.

    Names are summed across every function carrying them, and the caller is told
    how many distinct functions contributed, because "React has three functions
    called `Zk`" is a fact the caller needs in order to know whether the number
    means anything.
    """
    out: dict[str, int] = {}
    for name in names:
        matches = snapshot.find(name)
        out[name] = sum(m.count for m in matches)
    return out


def ambiguity(snapshot: CoverageSnapshot, name: str) -> int:
    return len(snapshot.find(name))


# ═══════════════════════════════════════════════════════════════════════════
# Harness adapter (INTERFACES.md section 3)
# ═══════════════════════════════════════════════════════════════════════════
#
# Level 3, and every window this instrument touches is TIMING-VOID by
# construction. Precise coverage keeps count-collecting bytecode alive, which
# disables TurboFan and Maglev for the whole isolate, so the durations that
# `tracing` reports in the same cell describe a program nobody ships.
#
# The payload therefore carries `timings_void: true` at both window and cell
# level. That flag exists so the report layer can refuse to quote a duration
# from this cell without having to know why. Only integers cross the boundary,
# enforced by `assert_integers_only` on the way out rather than by convention.

import time  # noqa: E402

from ..analysis import assert_no_bare_zero, measured, merge, unmeasured  # noqa: E402
from . import register_instrument  # noqa: E402


class CoverageInstrument:
    """Exact invocation counts per window. Integers only, timings void."""

    name = "coverage"
    level = 3

    def __init__(self, top_n: int = 40) -> None:
        self.ctx: Any = None
        self.cdp: Any = None
        self.cov: PreciseCoverage | None = None
        self.top_n = top_n
        self._overhead_ms = 0.0
        self._windows = 0
        self._start_reason = ""

    def attach(self, ctx: Any) -> None:
        self.ctx = ctx

    def start_cell(self, cell: Any) -> None:
        self.cdp = getattr(self.ctx, "cdp", None)
        self._overhead_ms = 0.0
        self._windows = 0
        self.cov = None
        self._start_reason = ""
        if self.cdp is None:
            self._start_reason = "no CDP session; precise coverage is Chromium only"
            return
        t0 = time.perf_counter()
        try:
            # Started ONCE per cell, never per window. Restarting coverage
            # re-runs V8's DeoptimizeAll, which changes what gets compiled and
            # therefore what gets counted, so per-window restarts would make the
            # counts depend on the window boundaries.
            self.cov = PreciseCoverage(self.cdp)
            self.cov.start()
        except Exception as exc:  # noqa: BLE001
            self.cov = None
            self._start_reason = f"{type(exc).__name__}: {exc}"
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0

    def open(self, window: Any) -> None:
        if self.cov is None:
            return
        t0 = time.perf_counter()
        try:
            self.cov.mark()
        except Exception:
            pass
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0

    def close(self, window: Any) -> dict | None:
        if self.cov is None:
            return merge(
                unmeasured("total_calls", self._start_reason or "coverage not running"),
                {"timings_void": True, "active": False},
            )
        t0 = time.perf_counter()
        try:
            snap = self.cov.window()
            top = snap.top(self.top_n)
            payload = merge(
                measured("total_calls", int(snap.total_calls())),
                measured("functions_invoked", len(snap.nonzero())),
                measured(
                    "top_functions",
                    [
                        {
                            "function": f.function_name or "(anonymous)",
                            "url": f.url,
                            "start_offset": int(f.start_offset),
                            "end_offset": int(f.end_offset),
                            "count": int(f.count),
                        }
                        for f in top
                    ],
                ),
                {
                    "timings_void": True,
                    "active": True,
                    "note": (
                        "precise coverage disables TurboFan and Maglev isolate-wide; "
                        "no duration from this cell may be quoted"
                    ),
                },
            )
            # The boundary guard. A float here would be a category error, not a
            # rounding problem, so it raises rather than warns.
            assert_integers_only(
                {
                    k: v
                    for k, v in payload.items()
                    if k in ("total_calls", "functions_invoked", "top_functions")
                }
            )
        except Exception as exc:  # noqa: BLE001
            payload = merge(
                unmeasured("total_calls", f"{type(exc).__name__}: {exc}"),
                {"timings_void": True, "active": True},
            )
        self._windows += 1
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0
        assert_no_bare_zero(payload, "coverage")
        return payload

    def end_cell(self, cell: Any) -> dict | None:
        if self.cov is not None:
            try:
                self.cov.stop()
            except Exception:
                pass
            self.cov = None
        out = merge(
            measured("overhead_ms", round(self._overhead_ms, 3)),
            measured("windows_counted", self._windows),
            {"timings_void": True, "headline_safe": False},
            {"start_reason": self._start_reason} if self._start_reason else {},
        )
        assert_no_bare_zero(out, "coverage.end_cell")
        return out

    def detach(self) -> None:
        if self.cov is not None:
            try:
                self.cov.stop()
            except Exception:
                pass
            self.cov = None


@register_instrument(name = "coverage", level = 3)
def _make_coverage() -> CoverageInstrument:
    return CoverageInstrument()
