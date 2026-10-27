# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exact call counts via Profiler.startPreciseCoverage. Timings are discarded: coverage skews them."""

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
        """Keyed on script and byte offsets, since minified code reuses names and a name key merges
        functions."""
        return (self.script_id, self.start_offset, self.end_offset)

    def label(self) -> str:
        name = self.function_name or "(anonymous)"
        return f"{name} @ {self.url or 'script#' + self.script_id}[{self.start_offset}:{self.end_offset}]"


@dataclass
class CoverageSnapshot:
    """Call counts only, with no time fields; coverage skews timings, so none are carried."""

    functions: list[FunctionCount] = field(default_factory = list)
    script_urls: dict[str, str] = field(default_factory = dict)
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
            # The first range is function-level with or without block coverage.
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
    """takePreciseCoverage counts since coverage started, so a window is the difference of two snapshots."""

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
    """A count that drops means the two snapshots came from different sessions; diff raises, not clamps."""
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
    """Raises on any non-integer number, so a timing from a coverage arm cannot reach a report silently."""

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
    """Sums calls over every function sharing a name; names are not unique in minified code."""
    out: dict[str, int] = {}
    for name in names:
        matches = snapshot.find(name)
        out[name] = sum(m.count for m in matches)
    return out


def ambiguity(snapshot: CoverageSnapshot, name: str) -> int:
    return len(snapshot.find(name))


# Harness adapter (INTERFACES.md section 3)
# Precise coverage disables TurboFan/Maglev, so timings in this cell are void (`timings_void`).

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
            # Started once per cell: restarting re-runs DeoptimizeAll and changes what gets counted.
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
