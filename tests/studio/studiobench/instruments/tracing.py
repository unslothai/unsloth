# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Use ReturnAsStream, not ReportEvents: ReportEvents streams trace data during the measured window."""

from __future__ import annotations

import base64
import gzip
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..analysis import CellFailure


# L0 is the only level headline numbers may come from.
L0 = "L0"
L1 = "L1"
L2 = "L2"
# Every timing from L3 is discarded; only integers cross the boundary.
L3 = "L3"

LEVELS = (L0, L1, L2, L3)

_TIMELINE_CATEGORIES: tuple[str, ...] = (
    "devtools.timeline",
    "disabled-by-default-devtools.timeline",
    "disabled-by-default-devtools.timeline.frame",
    "blink.user_timing",
    "toplevel",
    "toplevel.flow",
    # scheduler adds no events, only typed args (task_type, queue_name) on toplevel slices.
    "scheduler",
    # sequence_manager does NOT carry queue names.
    "sequence_manager",
    "latencyInfo",
    "benchmark",
    "input",
)

_STACK_CATEGORIES: tuple[str, ...] = (
    "disabled-by-default-devtools.timeline.stack",
    "disabled-by-default-v8.cpu_profiler",
    "v8",
    "disabled-by-default-v8.gc",
)

# V8 hard-codes a 100 us tracing profiler interval; the .hires category no longer works.
TRACING_PROFILER_INTERVAL_US = 100

CATEGORIES_BY_LEVEL: dict[str, tuple[str, ...]] = {
    L0: (),
    L1: _TIMELINE_CATEGORIES,
    L2: _TIMELINE_CATEGORIES + _STACK_CATEGORIES,
    L3: _TIMELINE_CATEGORIES + _STACK_CATEGORIES,
}

# Larger than the 200 MB default: an overflow costs a whole cell.
DEFAULT_BUFFER_KB = 640 * 1024

# tracing_handler.cc clamps this to a 250 ms floor; 500 ms matches DevTools.
BUFFER_POLL_MS = 500

BUFFER_WARN_FRACTION = 0.80


@dataclass
class TraceResult:
    """One captured trace plus every integrity fact needed to trust it."""

    level: str
    categories: tuple[str, ...]
    text: str
    path: str | None
    data_loss_occurred: bool
    max_percent_full: float
    buffer_polls: int
    buffer_kb: int
    wall_ms: float
    drain_ms: float
    drain_chunks: int
    trace_format: str
    stream_compression: str
    started_at_wall: float
    ended_at_wall: float

    @property
    def bytes(self) -> int:
        return len(self.text)

    def integrity(self) -> dict[str, Any]:
        """No buffer events means max_percent_full is unmeasured, not zero, so overflow cannot pass
        as empty."""
        from ..analysis import measured, merge, unmeasured

        if self.buffer_polls > 0:
            usage = merge(
                measured("max_percent_full", round(self.max_percent_full, 5)),
                {"near_buffer_limit": self.max_percent_full >= BUFFER_WARN_FRACTION},
            )
        else:
            usage = merge(
                unmeasured(
                    "max_percent_full",
                    "no Tracing.bufferUsage events were received, so buffer headroom is "
                    "unknown for this window; dataLossOccurred is the only overflow signal here",
                ),
                {"near_buffer_limit": None, "near_buffer_limit_attempted": False},
            )
        return merge(
            usage,
            measured("trace_bytes", self.bytes),
            measured("wall_ms", round(self.wall_ms, 2)),
            measured("drain_ms", round(self.drain_ms, 2)),
            measured("drain_chunks", self.drain_chunks),
            measured("buffer_kb", self.buffer_kb),
            {
                "level": self.level,
                "data_loss_occurred": self.data_loss_occurred,
            },
        )

    def assert_intact(self) -> None:
        if self.data_loss_occurred:
            raise CellFailure(
                "trace_buffer_overflow",
                f"Tracing reported dataLossOccurred at level {self.level} with a "
                f"{self.buffer_kb} KB buffer ({self.max_percent_full * 100:.1f}% peak). "
                "A truncated trace is indistinguishable from the expensive work not "
                "happening, so this cell is void. Raise buffer_kb or shorten the window.",
            )
        if not self.text.strip():
            raise CellFailure("trace_empty", f"level {self.level} drained zero bytes")


class TraceCapture:
    """Only one tracing session may exist per browser, so a second Tracing.start is refused here."""

    def __init__(
        self,
        cdp: Any,
        *,
        level: str = L2,
        buffer_kb: int = DEFAULT_BUFFER_KB,
        extra_categories: Sequence[str] = (),
        record_mode: str = "recordAsMuchAsPossible",
        wait: Callable[[float], None] | None = None,
    ) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown instrument level {level!r}; expected one of {LEVELS}")
        self.cdp = cdp
        self.level = level
        self.buffer_kb = int(buffer_kb)
        self.record_mode = record_mode
        cats = list(CATEGORIES_BY_LEVEL[level]) + [c for c in extra_categories if c]
        self.categories: tuple[str, ...] = tuple(dict.fromkeys(cats))
        self._wait = wait or (lambda ms: time.sleep(ms / 1000.0))
        self._usage: list[float] = []
        self._complete: dict[str, Any] = {}
        self._running = False
        self._t_start = 0.0
        self._subscribed = False

    def _subscribe(self) -> None:
        if self._subscribed:
            return
        # Playwright passes params as the single positional argument, None for a param-less event.
        self.cdp.on("Tracing.bufferUsage", self._on_buffer_usage)
        self.cdp.on("Tracing.tracingComplete", self._on_complete)
        self._subscribed = True

    def _on_buffer_usage(self, ev: dict[str, Any] | None) -> None:
        pct = (ev or {}).get("percentFull")
        if isinstance(pct, (int, float)):
            self._usage.append(float(pct))

    def _on_complete(self, ev: dict[str, Any] | None) -> None:
        self._complete.update(ev or {})

    def start(self) -> None:
        if self.level == L0:
            # L0 is a no-op rather than an error so one code path runs every level.
            self._running = True
            self._t_start = time.perf_counter()
            return
        if self._running:
            raise RuntimeError("TraceCapture.start called twice; Tracing is per-browser")
        self._subscribe()
        self._usage.clear()
        self._complete.clear()
        self.cdp.send(
            "Tracing.start",
            {
                "transferMode": "ReturnAsStream",
                "streamFormat": "json",
                "streamCompression": "none",
                "bufferUsageReportingInterval": BUFFER_POLL_MS,
                "traceConfig": {
                    "recordMode": self.record_mode,
                    "traceBufferSizeInKb": self.buffer_kb,
                    "includedCategories": list(self.categories),
                    "excludedCategories": [],
                    "enableSampling": False,
                    "enableSystrace": False,
                    "enableArgumentFilter": False,
                },
            },
        )
        self._running = True
        self._t_start = time.perf_counter()

    def stop(
        self,
        *,
        save_to: str | None = None,
        timeout_s: float = 120.0,
    ) -> TraceResult:
        if not self._running:
            raise RuntimeError("TraceCapture.stop called without start")
        t_end = time.perf_counter()
        wall_ms = (t_end - self._t_start) * 1000.0
        self._running = False

        if self.level == L0:
            return TraceResult(
                level = L0,
                categories = (),
                text = "",
                path = None,
                data_loss_occurred = False,
                max_percent_full = 0.0,
                buffer_polls = 0,
                buffer_kb = 0,
                wall_ms = wall_ms,
                drain_ms = 0.0,
                drain_chunks = 0,
                trace_format = "",
                stream_compression = "",
                started_at_wall = self._t_start,
                ended_at_wall = t_end,
            )

        self.cdp.send("Tracing.end")
        deadline = time.time() + timeout_s
        while "stream" not in self._complete and "dataLossOccurred" not in self._complete:
            if time.time() > deadline:
                raise CellFailure(
                    "tracing_complete_timeout",
                    f"Tracing.tracingComplete did not arrive within {timeout_s:.0f}s",
                )
            self._wait(50)
        # dataLossOccurred can arrive before the stream handle on a very small trace.
        grace = time.time() + 5.0
        while "stream" not in self._complete and time.time() < grace:
            self._wait(50)

        data_loss = bool(self._complete.get("dataLossOccurred"))
        handle = self._complete.get("stream")
        if handle is None:
            raise CellFailure(
                "tracing_no_stream",
                "tracingComplete carried no stream handle; transferMode was not ReturnAsStream",
            )

        t_drain = time.perf_counter()
        text, chunks = self._drain(str(handle), self._complete.get("streamCompression") or "none")
        drain_ms = (time.perf_counter() - t_drain) * 1000.0

        path = None
        if save_to:
            os.makedirs(os.path.dirname(os.path.abspath(save_to)) or ".", exist_ok = True)
            with open(save_to, "w", encoding = "utf-8") as fh:
                fh.write(text)
            path = save_to

        return TraceResult(
            level = self.level,
            categories = self.categories,
            text = text,
            path = path,
            data_loss_occurred = data_loss,
            max_percent_full = max(self._usage) if self._usage else 0.0,
            buffer_polls = len(self._usage),
            buffer_kb = self.buffer_kb,
            wall_ms = wall_ms,
            drain_ms = drain_ms,
            drain_chunks = chunks,
            trace_format = str(self._complete.get("traceFormat") or "json"),
            stream_compression = str(self._complete.get("streamCompression") or "none"),
            started_at_wall = self._t_start,
            ended_at_wall = t_end,
        )

    def _drain(self, handle: str, compression: str) -> tuple[str, int]:
        """Never read by offset (it counts pre-base64 bytes); append the final chunk before the EOF
        check."""
        parts: list[str] = []
        binary = False
        chunks = 0
        while True:
            r = self.cdp.send("IO.read", {"handle": handle, "size": 1 << 20})
            binary = binary or bool(r.get("base64Encoded"))
            parts.append(r.get("data") or "")
            chunks += 1
            if r.get("eof"):
                break
        try:
            self.cdp.send("IO.close", {"handle": handle})
        except Exception:
            # Failing to close the stream leaks a browser temp file but does not invalidate the trace.
            pass
        blob = "".join(parts)
        raw = base64.b64decode(blob) if binary else blob.encode("utf-8")
        if compression == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors = "strict"), chunks


@dataclass
class OverheadLedger:
    """The gate is that overhead does not grow with length; constant overhead only shifts the intercept."""

    cells: dict[str, dict[str, float]] = field(default_factory = dict)
    growth_tolerance: float = 0.15

    def record(self, rung: str, level: str, cost: float) -> None:
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}")
        self.cells.setdefault(rung, {})[level] = float(cost)

    def overhead(self, rung: str, level: str) -> float | None:
        row = self.cells.get(rung) or {}
        base = row.get(L0)
        got = row.get(level)
        if not base or got is None or base <= 0:
            return None
        return (got - base) / base

    def per_rung(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for rung in self.cells:
            row: dict[str, Any] = {}
            for level in (L1, L2, L3):
                ov = self.overhead(rung, level)
                if ov is not None:
                    row[f"overhead_{level}_vs_{L0}"] = round(ov, 4)
            out[rung] = row
        return out

    def disqualified_levels(self, rung_order: Sequence[str]) -> dict[str, str]:
        """rung_order must run smallest first; rungs lacking L0 or the level are skipped, narrowing
        the claim."""
        out: dict[str, str] = {}
        for level in (L1, L2, L3):
            series = [
                (rung, self.overhead(rung, level))
                for rung in rung_order
                if self.overhead(rung, level) is not None
            ]
            if len(series) < 2:
                continue
            first_rung, first = series[0]
            last_rung, last = series[-1]
            assert first is not None and last is not None
            if last - first > self.growth_tolerance:
                out[level] = (
                    f"overhead rose from {first * 100:.1f}% at {first_rung} to "
                    f"{last * 100:.1f}% at {last_rung}; overhead correlated with the "
                    "treatment cannot support an exponent claim at this level"
                )
        return out

    def report(self, rung_order: Sequence[str]) -> dict[str, Any]:
        return {
            "per_rung": self.per_rung(),
            "disqualified": self.disqualified_levels(rung_order),
            "growth_tolerance": self.growth_tolerance,
        }


def cross_check_with_metrics(
    trace_run_task_us: int,
    metrics: dict[str, float],
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Disagreement with Performance.getMetrics TaskDuration is strong evidence the trace is incomplete."""
    task_duration_s = metrics.get("TaskDuration")
    if task_duration_s is None:
        raise CellFailure(
            "no_task_duration",
            "Performance.getMetrics returned no TaskDuration; enable the Performance domain",
        )
    trace_s = trace_run_task_us / 1e6
    if task_duration_s <= 0:
        raise CellFailure("task_duration_zero", "TaskDuration was not positive")
    drift = abs(trace_s - task_duration_s) / task_duration_s
    out = {
        "trace_run_task_s": trace_s,
        "cdp_task_duration_s": task_duration_s,
        "drift": drift,
        "tolerance": tolerance,
    }
    if drift > tolerance:
        raise CellFailure(
            "task_duration_mismatch",
            f"trace RunTask total {trace_s * 1000:.1f} ms disagrees with CDP "
            f"TaskDuration {task_duration_s * 1000:.1f} ms by {drift * 100:.1f}%",
        )
    return out


class MetricsWindow:
    """Take getMetrics strictly inside the trace window; a wider metrics window causes false
    disagreement."""

    def __init__(self, cdp: Any) -> None:
        self.cdp = cdp
        self.before: dict[str, float] = {}
        self.after: dict[str, float] = {}

    def open(self) -> None:
        self.before = read_metrics(self.cdp)

    def close(self) -> None:
        self.after = read_metrics(self.cdp)

    def delta(self) -> dict[str, float]:
        if not self.before or not self.after:
            raise CellFailure(
                "metrics_window_unclosed", "MetricsWindow.open/close were not both called"
            )
        return {
            k: self.after[k] - self.before.get(k, 0.0)
            for k in self.after
            if isinstance(self.after[k], (int, float))
        }


def read_metrics(cdp: Any) -> dict[str, float]:
    """getMetrics on a disabled Performance domain errors; swallowing that error silently skips the
    check."""
    res = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in res.get("metrics", [])}


def enable_metrics(cdp: Any, *, time_domain: str = "timeTicks") -> None:
    """Use timeTicks, not threadTicks: RunTask.dur is wall time, so CPU time would read as disagreement."""
    cdp.send("Performance.enable", {"timeDomain": time_domain})


def save_trace(result: TraceResult, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok = True)
    with open(path, "w", encoding = "utf-8") as fh:
        fh.write(result.text)
    return path


def load_trace_json(result: TraceResult) -> dict[str, Any]:
    return json.loads(result.text)


# One trace per window so per-window numbers exist at close(); Tracing.start is timed as overhead.

from ..analysis import assert_no_bare_zero, measured, merge, unmeasured  # noqa: E402
from . import register_instrument  # noqa: E402


class TracingInstrument:
    """Timeline trace per window, with task origins and, at L2+, named frames."""

    name = "tracing"
    level = 1

    def __init__(self) -> None:
        self.ctx: Any = None
        self.cdp: Any = None
        self.cell: Any = None
        self.capture: TraceCapture | None = None
        self.metrics: MetricsWindow | None = None
        self.trace_level: str = L1
        self._overhead_ms = 0.0
        self._windows = 0
        self._failed_windows: list[str] = []
        self._save_traces = True

    def attach(self, ctx: Any) -> None:
        self.ctx = ctx

    def start_cell(self, cell: Any) -> None:
        # ctx.page and ctx.cdp may be replaced after crash recovery, so never cache them in attach.
        self.cell = cell
        self.cdp = getattr(self.ctx, "cdp", None)
        self._overhead_ms = 0.0
        self._windows = 0
        self._failed_windows = []
        lvl = int(getattr(cell, "instrument_level", 1) or 1)
        # L3 reuses L2's categories; its extras are other instruments.
        self.trace_level = L1 if lvl <= 1 else (L2 if lvl == 2 else L3)
        if self.cdp is not None:
            try:
                enable_metrics(self.cdp)
            except Exception:
                # Metrics are only a cross-check; losing them costs nothing else.
                pass

    def open(self, window: Any) -> None:
        if self.cdp is None:
            return
        t0 = time.perf_counter()
        self.capture = TraceCapture(
            self.cdp,
            level = self.trace_level,
            wait = self._wait,
        )
        try:
            self.capture.start()
        except Exception:
            self.capture = None
            self.metrics = None
        else:
            # Keep a started capture: dropping it leaves the per-browser tracing session running.
            try:
                self.metrics = MetricsWindow(self.cdp)
                self.metrics.open()
            except Exception:
                self.metrics = None
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0

    def close(self, window: Any) -> dict | None:
        if self.capture is None:
            return merge(
                unmeasured("task_ms", "tracing did not start for this window"),
                {"trace_level": self.trace_level, "active": False},
            )
        t0 = time.perf_counter()
        self._windows += 1
        try:
            if self.metrics is not None:
                self.metrics.close()
            result = self.capture.stop(save_to = self._trace_path(window))
            result.assert_intact()
            payload = self._analyse(result, window)
        except CellFailure as exc:
            self._failed_windows.append(f"{getattr(window, 'name', '?')}: {exc.gate}")
            payload = merge(
                unmeasured("task_ms", f"{exc.gate}: {exc.detail}"),
                {"trace_level": self.trace_level, "active": True, "cell_failed": True},
            )
        except Exception as exc:  # noqa: BLE001
            self._failed_windows.append(f"{getattr(window, 'name', '?')}: {type(exc).__name__}")
            payload = merge(
                unmeasured("task_ms", f"{type(exc).__name__}: {exc}"),
                {"trace_level": self.trace_level, "active": True, "cell_failed": True},
            )
        finally:
            self.capture = None
            self.metrics = None
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0
        assert_no_bare_zero(payload, f"tracing.{getattr(window, 'name', '?')}")
        return payload

    def end_cell(self, cell: Any) -> dict | None:
        # overhead_ms is measured wall cost, required at level >= 1 for the growth gate.
        out = merge(
            measured("overhead_ms", round(self._overhead_ms, 3)),
            measured("windows_traced", self._windows),
            {
                "trace_level": self.trace_level,
                "headline_safe": self.trace_level == L0,
                "failed_windows": self._failed_windows,
            },
        )
        assert_no_bare_zero(out, "tracing.end_cell")
        return out

    def detach(self) -> None:
        if self.capture is not None:
            try:
                self.capture.stop()
            except Exception:
                pass
            self.capture = None

    def _wait(self, ms: float) -> None:
        page = getattr(self.ctx, "page", None)
        if page is not None:
            try:
                page.wait_for_timeout(ms)
                return
            except Exception:
                pass
        time.sleep(ms / 1000.0)

    def _trace_path(self, window: Any) -> str | None:
        paths = getattr(self.ctx, "paths", None)
        if paths is None or not self._save_traces:
            return None
        cell_id = getattr(self.cell, "cell_id", "cell")
        safe = "".join(
            c if c.isalnum() or c in "-_." else "_" for c in str(getattr(window, "name", "w"))
        )
        return str(getattr(paths, "traces") / f"{cell_id}.{safe}.json")

    def _analyse(self, result: TraceResult, window: Any) -> dict:
        from ..analysis import classify as K
        from ..analysis import cpuprofile as C
        from ..analysis.traceparse import Trace

        trace = Trace.from_json_text(result.text)
        cls = K.classify_thread(trace)
        payload: dict = merge(
            measured("task_ms", round(cls.total_us / 1000.0, 3)),
            measured("unclassified_task_pct", round(cls.unclassified_pct, 4)),
            {
                "trace_level": self.trace_level,
                "active": True,
                "task_ms_by_origin": {
                    k: round(v / 1000.0, 3) for k, v in sorted(cls.by_origin_us.items())
                },
                "task_count_by_origin": dict(sorted(cls.by_origin_count.items())),
                "integrity": result.integrity(),
                "trace_path": result.path,
            },
        )

        # A disagreement means the trace is missing tasks; reported but does not void the window.
        try:
            if self.metrics is not None:
                payload.update(
                    measured(
                        "task_duration_crosscheck_drift",
                        round(
                            cross_check_with_metrics(cls.total_us, self.metrics.delta())["drift"], 5
                        ),
                    )
                )
        except CellFailure as exc:
            payload.update(
                unmeasured("task_duration_crosscheck_drift", f"{exc.gate}: {exc.detail}")
            )
        except Exception as exc:  # noqa: BLE001
            payload.update(
                unmeasured("task_duration_crosscheck_drift", f"{type(exc).__name__}: {exc}")
            )

        # Named frames need L2+; at L1 return null with a reason, never an empty list.
        if self.trace_level == L1:
            payload.update(
                unmeasured(
                    "named_frames",
                    "the v8 CPU profiler category is off at instrument level 1, so samples "
                    "have no stacks and no frame can be named. Raise the level to 2.",
                )
            )
            return payload

        try:
            prof = C.main_thread_profile(trace)
            prof.assert_deltas_match_wall()
            rows, diag = C.self_time_in_windows(
                prof,
                [(prof.chunk_ts_first, prof.chunk_ts_last)],
                limit = 12,
            )
            payload.update(
                measured(
                    "named_frames",
                    [{"frame": f.label(), "self_ms": round(us / 1000.0, 3)} for f, us in rows],
                )
            )
            payload["frame_ranking_underpowered"] = bool(diag["underpowered"])
            payload["js_sample_count"] = int(diag["js_sample_count"])
        except CellFailure as exc:
            payload.update(unmeasured("named_frames", f"{exc.gate}: {exc.detail}"))
        return payload


@register_instrument(name = "tracing", level = 1)
def _make_tracing() -> TracingInstrument:
    return TracingInstrument()
