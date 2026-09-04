# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The V8 CPU profiler, both ways round.

There are two routes to sampled stacks and they are not interchangeable:

* IN-TRACE (`disabled-by-default-v8.cpu_profiler`). Samples arrive as
  `Profile` / `ProfileChunk` events on the SAME timeline as `RunTask`,
  `TimerFire` and `EventDispatch`, on one clock. That is what lets a sample be
  attributed to a classified task, which is the entire naming pipeline. This is
  the route `instruments/tracing.py` at L2 takes, and it is the default.

* STANDALONE (`Profiler.start` / `Profiler.stop`). One self-contained profile,
  no trace, no task tree. Useful as a cross-check on the in-trace parser and as
  a cheap way to get stacks when a full trace would overflow its buffer, but the
  samples cannot be joined to task origins.

`Profiler.setSamplingInterval` is in MICROSECONDS and must be set BEFORE
`Profiler.start`; setting it on a running profiler is silently ignored, which
looks exactly like the interval you asked for being unavailable.

The cross-check `compare_with_trace_profile` exists because the in-trace parser
is the load-bearing piece of this layer and it has several ways to be quietly
wrong (incremental nodes, deltas on a sibling key, chunks on the profiler's own
thread). Two independent extractions agreeing is worth having.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..analysis import CellFailure
from ..analysis.cpuprofile import CallFrame, CpuProfile, Sample

# V8's own default is 1000 us. 100 us resolves a sub-millisecond frame without the sampler becoming the workload.
DEFAULT_SAMPLING_INTERVAL_US = 100

# Minimum non-synthetic samples on BOTH sides before the two extraction routes may be declared to agree or disagree.
MIN_JS_SAMPLES_FOR_COMPARISON = 200


@dataclass
class StandaloneProfiler:
    """`Profiler.start` / `Profiler.stop` over one CDP session."""

    cdp: Any
    sampling_interval_us: int = DEFAULT_SAMPLING_INTERVAL_US
    _running: bool = field(default = False, init = False)

    def start(self) -> None:
        if self._running:
            raise RuntimeError("StandaloneProfiler.start called twice")
        self.cdp.send("Profiler.enable")
        # Order matters: the interval is latched at start.
        self.cdp.send("Profiler.setSamplingInterval", {"interval": int(self.sampling_interval_us)})
        self.cdp.send("Profiler.start")
        self._running = True

    def stop(self) -> CpuProfile:
        if not self._running:
            raise RuntimeError("StandaloneProfiler.stop without start")
        res = self.cdp.send("Profiler.stop")
        self._running = False
        return from_profiler_result(res.get("profile") or {})

    def __enter__(self) -> "StandaloneProfiler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._running:
            try:
                self.cdp.send("Profiler.stop")
            finally:
                self._running = False


def from_profiler_result(raw: dict[str, Any]) -> CpuProfile:
    """Adapt a `Profiler.stop` payload into the same object the trace path yields.

    The standalone payload is the complete profile in one piece: `nodes` is the
    full node list, `samples` and `timeDeltas` are flat arrays, and `startTime`
    and `endTime` are microseconds. Reusing `CpuProfile` means every aggregation
    and every gate in `analysis/cpuprofile.py` applies unchanged to both routes,
    so the cross-check compares the same arithmetic on two inputs.
    """
    nodes = raw.get("nodes")
    if not nodes:
        raise CellFailure("cpuprofile_empty", "Profiler.stop returned a profile with no nodes")
    prof = CpuProfile(pid = 0, tid = 0, start_time = int(raw.get("startTime", 0)))
    for n in nodes:
        nid = int(n["id"])
        cf = n.get("callFrame") or {}
        prof.nodes[nid] = CallFrame(
            function_name = str(cf.get("functionName", "")),
            script_id = str(cf.get("scriptId", "")),
            url = str(cf.get("url", "")),
            line = int(cf.get("lineNumber", -1)),
            column = int(cf.get("columnNumber", -1)),
        )
        for child in n.get("children") or ():
            prof.parents[int(child)] = nid

    samples = raw.get("samples") or []
    deltas = raw.get("timeDeltas") or []
    if len(samples) != len(deltas):
        raise CellFailure(
            "cpuprofile_ragged",
            f"{len(samples)} samples against {len(deltas)} timeDeltas",
        )
    cursor = prof.start_time
    for nid, d in zip(samples, deltas):
        d = int(d)
        if d < 0:
            prof.negative_deltas += 1
        cursor += d
        prof.samples.append(Sample(ts = cursor, node_id = int(nid), delta = d))

    prof.chunk_count = 1
    prof.chunk_ts_first = prof.start_time
    prof.chunk_ts_last = int(raw.get("endTime", cursor))
    return prof


def compare_with_trace_profile(
    standalone: CpuProfile,
    in_trace: CpuProfile,
    *,
    tolerance: float = 0.25,
) -> dict[str, Any]:
    """Do the two extraction routes agree on where the time went?

    Compared on the SHARE of JS self time per frame, not on absolute
    microseconds, because the two profiles cover different windows and different
    sampling intervals. A large disagreement means one of the two parsers is
    wrong, and since the in-trace one is the one this layer depends on, that is
    a finding and not a warning.
    """

    def shares(p: CpuProfile) -> dict[str, float]:
        by_name: dict[str, int] = {}
        for s in p.samples:
            f = p.nodes.get(s.node_id)
            if f is None or f.is_synthetic:
                continue
            by_name[f.function_name or "(anonymous)"] = (
                by_name.get(f.function_name or "(anonymous)", 0) + s.delta
            )
        total = sum(by_name.values())
        if total <= 0:
            return {}
        return {k: v / total for k, v in by_name.items()}

    def js_samples(p: CpuProfile) -> int:
        return sum(
            1 for s in p.samples if (f := p.nodes.get(s.node_id)) is not None and not f.is_synthetic
        )

    # A comparison built on a handful of JS samples cannot distinguish the two parsers from sampling
    # noise, so it declines to rule. And V8 has ONE CpuProfiler: if the trace carried the profiler
    # categories, the 'standalone' profile IS the in-trace profile and comparing them always agrees.
    # Detected structurally rather than trusted to the caller.
    if (
        len(standalone.samples) == len(in_trace.samples)
        and len(standalone.nodes) == len(in_trace.nodes)
        and standalone.samples[:8]
        and in_trace.samples[:8]
        and [s.node_id for s in standalone.samples[:8]] == [s.node_id for s in in_trace.samples[:8]]
    ):
        return {
            "agrees": None,
            "verdict": "same_profile",
            "reason": (
                "both profiles have identical sample and node counts and identical leading "
                "samples. V8 has a single CpuProfiler, so a Profiler.start taken while the "
                "trace already records disabled-by-default-v8.cpu_profiler returns the same "
                "samples. This is a comparison against itself and is not evidence."
            ),
        }

    power = {
        "standalone_js_samples": js_samples(standalone),
        "in_trace_js_samples": js_samples(in_trace),
    }
    if min(power.values()) < MIN_JS_SAMPLES_FOR_COMPARISON:
        return {
            "agrees": None,
            "verdict": "underpowered",
            "reason": (
                f"need at least {MIN_JS_SAMPLES_FOR_COMPARISON} non-synthetic samples on both "
                f"sides; got {power}. Lengthen the window or enable the hires CPU profiler category."
            ),
            **power,
        }

    a, b = shares(standalone), shares(in_trace)
    common = sorted(set(a) & set(b), key = lambda k: -(a[k] + b[k]))[:10]
    rows = [
        {
            "function": k,
            "standalone_share": round(a[k], 4),
            "in_trace_share": round(b[k], 4),
            "abs_diff": round(abs(a[k] - b[k]), 4),
        }
        for k in common
    ]
    worst = max((r["abs_diff"] for r in rows), default = 0.0)
    return {
        "compared_frames": len(common),
        "worst_share_difference": worst,
        "tolerance": tolerance,
        "agrees": worst <= tolerance,
        "rows": rows,
    }


# Harness adapter (INTERFACES.md section 3)
# MEASURED, NOT ASSUMED, and it changed this design: V8 HAS ONE CPU PROFILER. Starting
# `Profiler.start` while a trace with `disabled-by-default-v8.cpu_profiler` is running returns THE
# SAME SAMPLES: on a real capture both routes reported 2169 samples over 17 nodes, because the
# inspector and tracing profilers share one `CpuProfiler` on the isolate.
# Two consequences: `compare_with_trace_profile` must refuse to compare a profile with itself,
# since a cross-check that can never fail reads as corroboration; and this instrument is only
# meaningful at level 1, where the trace carries NO profiler categories and it gives stacks
# without the ProfileChunk volume that dominates an L2 buffer (a real L1 capture had 1748 of 1748
# standalone samples inside the trace's own `RunTask` span). At level 2 and above it stands down
# and says so.

import time  # noqa: E402

from ..analysis import assert_no_bare_zero, measured, merge, unmeasured  # noqa: E402
from . import register_instrument  # noqa: E402

_STAND_DOWN = (
    "instrument level {lvl} already records the v8 CPU profiler inside the trace. "
    "V8 has a single CpuProfiler, so a second Profiler.start returns the same "
    "samples; running it here would double the cost and produce a cross-check "
    "against itself."
)


class CpuProfileInstrument:
    """Standalone V8 sampling, for the level where the trace has no stacks."""

    name = "cpu_profile"
    level = 1

    def __init__(self) -> None:
        self.ctx: Any = None
        self.cdp: Any = None
        self.active = False
        self.stand_down_reason = ""
        self.profiler: StandaloneProfiler | None = None
        self._overhead_ms = 0.0
        self._windows = 0

    def attach(self, ctx: Any) -> None:
        self.ctx = ctx

    def start_cell(self, cell: Any) -> None:
        self.cdp = getattr(self.ctx, "cdp", None)
        self._overhead_ms = 0.0
        self._windows = 0
        lvl = int(getattr(cell, "instrument_level", 1) or 1)
        self.active = lvl == 1 and self.cdp is not None
        if lvl != 1:
            self.stand_down_reason = _STAND_DOWN.format(lvl = lvl)
        elif self.cdp is None:
            self.stand_down_reason = "no CDP session; the V8 profiler is Chromium only"
        else:
            self.stand_down_reason = ""

    def open(self, window: Any) -> None:
        if not self.active:
            return
        t0 = time.perf_counter()
        self.profiler = StandaloneProfiler(self.cdp)
        try:
            self.profiler.start()
        except Exception:
            self.profiler = None
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0

    def close(self, window: Any) -> dict | None:
        if not self.active:
            return merge(
                unmeasured("self_ms_top", self.stand_down_reason or "instrument inactive"),
                {"active": False},
            )
        if self.profiler is None:
            return merge(
                unmeasured("self_ms_top", "Profiler.start failed for this window"),
                {"active": True},
            )
        t0 = time.perf_counter()
        try:
            prof = self.profiler.stop()
            payload = self._summarise(prof)
        except CellFailure as exc:
            payload = merge(
                unmeasured("self_ms_top", f"{exc.gate}: {exc.detail}"), {"active": True}
            )
        except Exception as exc:  # noqa: BLE001
            payload = merge(
                unmeasured("self_ms_top", f"{type(exc).__name__}: {exc}"), {"active": True}
            )
        finally:
            self.profiler = None
        self._windows += 1
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0
        assert_no_bare_zero(payload, "cpu_profile")
        return payload

    def end_cell(self, cell: Any) -> dict | None:
        out = merge(
            measured("overhead_ms", round(self._overhead_ms, 3)),
            measured("windows_profiled", self._windows),
            {"active": self.active},
            {"stand_down_reason": self.stand_down_reason} if self.stand_down_reason else {},
        )
        assert_no_bare_zero(out, "cpu_profile.end_cell")
        return out

    def detach(self) -> None:
        if self.profiler is not None:
            try:
                self.profiler.stop()
            except Exception:
                pass
            self.profiler = None

    def _summarise(self, prof: Any) -> dict:
        from ..analysis import cpuprofile as C

        rows, diag = C.self_time_in_windows(
            prof,
            [(prof.chunk_ts_first, prof.chunk_ts_last)],
            limit = 12,
        )
        payload = merge(
            measured(
                "self_ms_top",
                [{"frame": f.label(), "self_ms": round(us / 1000.0, 3)} for f, us in rows],
            ),
            measured("js_sample_count", int(diag["js_sample_count"])),
            measured("sampling_interval_us", round(prof.sampling_interval_us(), 2)),
            {
                "active": True,
                "frame_ranking_underpowered": bool(diag["underpowered"]),
                "profiler": "standalone",
            },
        )
        return payload


@register_instrument(name = "cpu_profile", level = 1)
def _make_cpu_profile() -> CpuProfileInstrument:
    return CpuProfileInstrument()
