# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Capture a Chrome trace without contaminating the window being measured.

`transferMode: "ReturnAsStream"`, never `ReportEvents`. `ReportEvents` pushes
the trace back over the devtools pipe as `Tracing.dataCollected` notifications
WHILE THE WINDOW IS OPEN. Every one of those is renderer-visible work, and its
volume scales with how much the page is doing, which is to say it is correlated
with the treatment. `ReturnAsStream` writes to a temp file in the browser
process and hands back a stream handle at the end, so the drain happens after
the measurement is over. (Confirmed in `content/browser/devtools/protocol/
tracing_handler.cc`: `ReturnAsStream` builds a `DevToolsStreamFile` endpoint,
while `OnTraceDataCollected` splices events into a notification per chunk.)

A trace that hit its buffer is a FAILED CELL, not a short trace. A truncated
trace reads exactly like "the expensive thing did not happen", which is the most
dangerous possible failure for a tool whose whole job is to find an expensive
thing. `Tracing.tracingComplete.dataLossOccurred` is the authoritative signal;
it is a sticky OR over perfetto's `chunks_overwritten`, `chunks_discarded`,
`abi_violations` and `trace_writer_packet_loss`, and it is valid even if buffer
usage polling is off. `Tracing.bufferUsage.percentFull` is subscribed as an
early warning; its `eventCount` is hardcoded to 0 on modern Chrome and its
`value` is a legacy duplicate of `percentFull`, so neither is used.

TRACING OVERHEAD IS MEASURED, NEVER ASSUMED. `OverheadLedger` records the same
cell at L0 and at each higher level and reports `overhead_L1_vs_L0` and
`overhead_L2_vs_L0` per rung. A level whose overhead GROWS WITH LENGTH is
disqualified from exponent claims at that rung. Constant overhead is harmless
to a slope; overhead correlated with the treatment manufactures one.
"""

from __future__ import annotations

import base64
import gzip
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..analysis import CellFailure


# L0 is the only level headline numbers may come from: nothing is attached beyond the renderer's own metrics counters.
# --------------------------------------------------------------------- ladder

L0 = "L0"
# L1 adds the timeline trace (task boundaries, frames, layout, user timing). No CPU profiler,
# so no stacks and no naming, but the cheapest level giving a real task tree.
L1 = "L1"
# L2 adds the V8 CPU profiler. This is the level that NAMES A FRAME.
L2 = "L2"
# L3 adds precise coverage and heap sampling. Every timing from L3 is discarded by
# construction; only integers cross the boundary.
L3 = "L3"

LEVELS = (L0, L1, L2, L3)

_TIMELINE_CATEGORIES: tuple[str, ...] = (
    "devtools.timeline",
    "disabled-by-default-devtools.timeline",
    "disabled-by-default-devtools.timeline.frame",
    "blink.user_timing",
    # `toplevel` carries `ThreadControllerImpl::RunTask` with `src_file` / `src_func`, the C++ that
    # POSTED each task; `RunTask` itself says nothing about origin.
    "toplevel",
    "toplevel.flow",
    # `scheduler` matters most and is easy to miss: it adds no events, only TYPED ARGS to the
    # `toplevel` slice (task_type, queue_name), which turn task origin from an inference into a
    # read value. A React scheduler callback arrives labelled `TASK_TYPE_*POSTED_MESSAGE` on
    # `FRAME_PAUSABLE_TQ`, a timer as `TASK_TYPE_JAVASCRIPT_TIMER_*`.
    # Spelled `args.renderer_main_thread_task_execution.task_type` and
    # `args.sequence_manager_task.queue_name`.
    "scheduler",
    # `sequence_manager` adds the DoWork / SelectNextTask / DoIdleWork scoping slices. It does NOT
    # carry queue names, contrary to a natural reading.
    "sequence_manager",
    "latencyInfo",
    "benchmark",
    "input",
)

# `...timeline.stack` attaches call-site information to timeline events;
# `disabled-by-default-v8.cpu_profiler` is what produces `Profile` / `ProfileChunk`, and
# without it every microsecond stays a bucket with no stack.
# In full, `disabled-by-default-devtools.timeline.stack`.
_STACK_CATEGORIES: tuple[str, ...] = (
    "disabled-by-default-devtools.timeline.stack",
    "disabled-by-default-v8.cpu_profiler",
    "v8",
    "disabled-by-default-v8.gc",
)

# There is NO knob for the tracing CPU profiler's sampling interval:
# `v8/src/profiler/tracing-cpu-profiler.cc` hard-codes 100 us, and the `...cpu_profiler.hires`
# category that used to lower it is no longer registered in V8's category list, so DevTools
# still sends it and it does nothing. Measured spacing on a real capture is ~150 us.
# That category is `disabled-by-default-v8.cpu_profiler.hires`.
# The consequence is structural: a set of 40 us scheduler tasks yields at most one sample each,
# so leaf rankings over short task windows are underpowered.
# `analysis.cpuprofile.self_time_in_windows` reports `underpowered` rather than relying on a
# category flag that does not work.
TRACING_PROFILER_INTERVAL_US = 100

CATEGORIES_BY_LEVEL: dict[str, tuple[str, ...]] = {
    L0: (),
    L1: _TIMELINE_CATEGORIES,
    L2: _TIMELINE_CATEGORIES + _STACK_CATEGORIES,
    L3: _TIMELINE_CATEGORIES + _STACK_CATEGORIES,
}

# The default perfetto buffer is 200 MB. A long rung at L2 produces many ProfileChunks, so the
# buffer is set explicitly and large: an overflow costs a whole cell, and memory is cheaper
# than a re-run.
DEFAULT_BUFFER_KB = 640 * 1024

# `bufferUsageReportingInterval` is clamped to a 250 ms floor in `tracing_handler.cc`, so asking
# for less is silently ignored. 500 ms is what the DevTools frontend uses.
# `kMinimumReportingInterval`.
BUFFER_POLL_MS = 500

# A trace this close to full did not lose data but was about to, and the next rung will.
# Surfaced as a warning so a rung ladder does not walk off a cliff.
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
        """Integrity facts, under the no-bare-zero rule.

        `max_percent_full` is the interesting case. A reading of 0.0 with buffer
        usage events received means the buffer really was empty; a reading of
        0.0 with NO events received means we never heard from the buffer at all,
        and those two must not look the same. The second is precisely the state
        in which an overflow would go unnoticed, so it is reported as unmeasured
        with a reason rather than as a reassuring zero.
        """
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
    """Drive `Tracing` over one CDP session.

    Only one tracing session may exist per browser: a second `Tracing.start`
    fails with "Tracing has already been started (possibly in another tab)". The
    class refuses to double-start rather than letting that surface later as an
    unrelated protocol error.
    """

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
        # Order-stable de-duplication so the recorded category list is reproducible between runs.
        self.categories: tuple[str, ...] = tuple(dict.fromkeys(cats))
        self._wait = wait or (lambda ms: time.sleep(ms / 1000.0))
        self._usage: list[float] = []
        self._complete: dict[str, Any] = {}
        self._running = False
        self._t_start = 0.0
        self._subscribed = False

    # ------------------------------------------------------------------ driving

    def _subscribe(self) -> None:
        if self._subscribed:
            return
        # Playwright delivers the event's `params` object as the single positional argument, and it is
        # `None` for a param-less event.
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
            # L0 attaches nothing. A no-op rather than an error is what lets one code path run every level,
            # including the one whose definition is 'do not instrument'.
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
        # `dataLossOccurred` can arrive first on a very small trace; give the stream handle a moment
        # before deciding there is not one.
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
        """Read the stream to EOF, then close it.

        `IO.read` returns `base64Encoded: true` only for gzip or proto payloads,
        and its `offset`/`size` are raw pre-base64 byte counts, so the offset is
        never derived from `len(data)`; sequential reads with no offset are the
        only safe form. The read that reaches EOF returns an empty `data`, so
        the chunk is appended BEFORE the eof check.
        """
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
            # The stream is temp storage in the browser process; failing to close it leaks a file, it does
            # not invalidate the trace we hold.
            pass
        blob = "".join(parts)
        raw = base64.b64decode(blob) if binary else blob.encode("utf-8")
        if compression == "gzip":
            raw = gzip.decompress(raw)
        return raw.decode("utf-8", errors = "strict"), chunks


@dataclass
class OverheadLedger:
    """Per-rung measured cost of instrumentation, and the disqualification gate.

    The danger is not overhead. Constant overhead shifts an intercept and leaves
    an exponent alone. The danger is overhead CORRELATED WITH THE TREATMENT,
    because that manufactures exactly the slope the tool is looking for. So the
    gate is not "overhead is small", it is "overhead does not grow with length".

    HOW THIS RELATES TO THE HARNESS PATH, since there are two and they are not
    competitors. Under Layer 1, each instrument reports its own measured
    `overhead_ms` from `end_cell`, and the report layer assembles those across
    rungs into its `overhead_growth_with_length` gate. That is the production
    route and it needs nothing from this class. This ledger is the OFFLINE route:
    it takes the same cell run at L0 and at a higher level and produces the
    ratio and the disqualification verdict directly, which is what you want when
    calibrating a machine or investigating a suspicious slope outside a full
    run. Same rule, same tolerance, two entry points.
    """

    # rung label -> level -> observed cost of the identical cell (ms, or any single consistent
    # scalar such as median frame time)
    cells: dict[str, dict[str, float]] = field(default_factory = dict)
    # A level whose overhead ratio rises by more than this across the ladder is disqualified from exponent claims.
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
        """Levels whose overhead grows with length, with the reason.

        `rung_order` must be smallest-first. Only rungs that actually recorded
        both L0 and the level are considered, so a partial ladder narrows the
        claim instead of inventing one.
        """
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
    """Summed trace `RunTask` vs `Performance.getMetrics` `TaskDuration`.

    Two independent accountings of the same physical quantity, produced by
    different subsystems. Agreement is weak evidence that the trace is complete;
    disagreement is strong evidence that it is not, and the cell fails.
    """
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
    """Bracket `Performance.getMetrics` INSIDE the trace window.

    `TaskDuration` is a monotonic renderer-wide counter, so a window is the
    difference of two readings. The readings must be taken just AFTER
    `Tracing.start` and just BEFORE `Tracing.end`, never outside, or the metrics
    window is wider than the trace window and the cross-check reports a
    disagreement that is entirely an artefact of how it was taken. A 5.7% false
    failure was produced exactly this way while building this module, which is
    why the ordering lives in a class instead of in a comment.

    Taking the metrics strictly inside also makes the residual one-sided: the
    trace should account for at least as much task time as the metrics do, so a
    trace total BELOW the metrics total means missing events.
    """

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
    """`Performance.getMetrics` flattened to a plain mapping.

    The Performance domain must be enabled first; calling `getMetrics` on a
    disabled domain returns an error rather than an empty result, and swallowing
    that is how a cross-check silently stops checking.
    """
    res = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in res.get("metrics", [])}


def enable_metrics(cdp: Any, *, time_domain: str = "timeTicks") -> None:
    """Enable the Performance domain on the WALL clock.

    `timeTicks` is the default and it is the right one here: trace `RunTask.dur`
    is wall duration, so cross-checking it against a `TaskDuration` accumulated
    in `threadTicks` would compare CPU time to elapsed time and read as a real
    disagreement whenever the thread was descheduled.
    """
    cdp.send("Performance.enable", {"timeDomain": time_domain})


def save_trace(result: TraceResult, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok = True)
    with open(path, "w", encoding = "utf-8") as fh:
        fh.write(result.text)
    return path


def load_trace_json(result: TraceResult) -> dict[str, Any]:
    return json.loads(result.text)


# Harness adapter (INTERFACES.md section 3)
# ONE TRACE PER WINDOW, not one per cell. The alternative, a cell-long trace plus
# `performance.mark` bracketing, was rejected because a mark is a `Runtime.evaluate` round trip
# inside the measured interval; because a cell-long L2 trace accumulates ProfileChunks so
# overflow risk grows with cell length; and above all because per-window numbers must exist at
# `close()` time, since the harness emits `window.row()` right after and a trace drained at
# `end_cell` would leave every window row null.
# `Tracing.end` and the drain happen inside `close()`, which the harness calls AFTER stamping
# `t_close_ms`, so the window duration is already fixed and the drain cannot inflate it. The
# `Tracing.start` in `open()` does land inside the window, so it is timed and reported as
# `overhead_ms`.

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

    # ---------------------------------------------------------------- lifecycle

    def attach(self, ctx: Any) -> None:
        self.ctx = ctx

    def start_cell(self, cell: Any) -> None:
        # `ctx.page` and `ctx.cdp` may be REPLACED between cells when a crashed renderer is recovered,
        # so they are re-read here and never cached in `attach`. INTERFACES.md section 7.
        self.cell = cell
        self.cdp = getattr(self.ctx, "cdp", None)
        self._overhead_ms = 0.0
        self._windows = 0
        self._failed_windows = []
        lvl = int(getattr(cell, "instrument_level", 1) or 1)
        # The instrument ladder is expressed HERE. L0 never reaches this method because the registry
        # filters on `level`, and L3 uses the same category set as L2 (its coverage and heap sampling
        # are other instruments, not more categories).
        self.trace_level = L1 if lvl <= 1 else (L2 if lvl == 2 else L3)
        if self.cdp is not None:
            try:
                enable_metrics(self.cdp)
            except Exception:
                # Metrics are a cross-check, not the measurement. Losing them costs the cross-check and nothing else.
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
            # Nothing was started, so there is nothing to end.
            self.capture = None
            self.metrics = None
        else:
            # THE METRICS PROBE MAY NOT TAKE THE TRACE WITH IT. Dropping a STARTED capture here cost the
            # whole run: tracing is per-browser, so with `self.capture` cleared `close()` returns early and
            # never sends `Tracing.end`, `detach()` has nothing to stop, and the next window's
            # `Tracing.start` fails with 'Tracing has already been started', so every remaining window
            # reports `tracing did not start` while the abandoned session keeps recording underneath. One
            # failed `Performance.getMetrics` was enough. `close()` and `_analyse` already guard every use
            # of `self.metrics`, so only the cross-check goes.
            # `start_cell` says what losing the metrics probe costs, and it can be off.
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
        # `overhead_ms` is required from every instrument at level >= 1 (INTERFACES.md section 3) and
        # feeds the report layer's `overhead_growth_with_length` gate. It is a MEASURED wall cost of
        # this instrument's own calls, not an estimate from a table.
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

    # ------------------------------------------------------------------ helpers

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

        # Cross-check against the renderer's own accounting. A disagreement means the trace is missing
        # tasks, so it is reported rather than silently tolerated, but it does not void the window.
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

        # Named frames need the V8 profiler, which only L2+ turns on. At L1 this is an honest null with
        # a reason, never an empty list that reads as 'nothing was hot'.
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
