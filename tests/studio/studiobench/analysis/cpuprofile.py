# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turns ProfileChunk events into timed stacks; deltas can be negative, so sum them, never clamp."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import CellFailure
from .traceparse import Trace

# Excluded from frame ranking, not from totals.
SYNTHETIC_FRAMES = frozenset({"(root)", "(program)", "(idle)", "(garbage collector)"})

DELTA_WALL_TOLERANCE = 0.02

# At ~150 us sampling, short tasks give too few samples for a ranking; reported as `underpowered`.
MIN_JS_SAMPLES_FOR_RANKING = 100


@dataclass(frozen = True)
class CallFrame:
    function_name: str
    script_id: str
    url: str
    line: int
    column: int
    code_type: str = ""

    @property
    def key(self) -> tuple[str, str, int, int]:
        """Keyed on script id and source position, not name, which is often empty or collides when
        minified."""
        return (self.function_name, self.script_id, self.line, self.column)

    @property
    def is_synthetic(self) -> bool:
        return self.function_name in SYNTHETIC_FRAMES

    def label(self) -> str:
        name = self.function_name or "(anonymous)"
        where = self.url or f"script#{self.script_id}"
        return f"{name} @ {where}:{self.line}:{self.column}"


@dataclass
class Sample:
    ts: int  # microseconds, monotonic, same clock as trace `ts`
    node_id: int
    delta: int  # microseconds attributed to this sample


@dataclass
class CpuProfile:
    """Accumulated V8 CPU profile for one profiled thread."""

    pid: int
    tid: int
    start_time: int
    nodes: dict[int, CallFrame] = field(default_factory = dict)
    parents: dict[int, int] = field(default_factory = dict)
    samples: list[Sample] = field(default_factory = list)
    chunk_count: int = 0
    chunk_ts_first: int = 0
    chunk_ts_last: int = 0
    negative_deltas: int = 0
    # Kept only to report clock skew against `ts`; never an anchor.
    declared_start_time: int = 0
    source: str = ""
    # Exact stacks captured at instrumentation points, not sampled ones.
    trace_ids: dict[str, int] = field(default_factory = dict)

    @property
    def clock_skew_us(self) -> int:
        if not self.declared_start_time:
            return 0
        return self.start_time - self.declared_start_time

    def stack_for_sample_trace_id(self, sample_trace_id: int | str) -> list[CallFrame]:
        """Exact stack of a tagged event via sampleTraceId, with none of the sampling error of short
        windows."""
        node = self.trace_ids.get(str(sample_trace_id))
        if node is None:
            return []
        return self.stack(node)

    def stack(self, node_id: int) -> list[CallFrame]:
        """Stops at a repeated node: a corrupt trace can hold a cyclic parent chain that hangs the
        analysis."""
        out: list[CallFrame] = []
        seen: set[int] = set()
        cur: int | None = node_id
        while cur is not None and cur not in seen:
            seen.add(cur)
            frame = self.nodes.get(cur)
            if frame is None:
                break
            out.append(frame)
            cur = self.parents.get(cur)
        return out

    def stacked_samples(
        self,
        t0: int | None = None,
        t1: int | None = None,
    ) -> list[tuple[int, list[CallFrame]]]:
        """The deliverable: (timestamp, stack) pairs, leaf first."""
        return [(s.ts, self.stack(s.node_id)) for s in self.window(t0, t1)]

    def window(
        self,
        t0: int | None = None,
        t1: int | None = None,
    ) -> list[Sample]:
        lo = t0 if t0 is not None else -(1 << 62)
        hi = t1 if t1 is not None else (1 << 62)
        return [s for s in self.samples if lo <= s.ts < hi]

    def self_time_us(
        self,
        t0: int | None = None,
        t1: int | None = None,
    ) -> dict[tuple[str, str, int, int], int]:
        """Microseconds where a frame was the LEAF of the stack, by frame key."""
        out: dict[tuple[str, str, int, int], int] = {}
        for s in self.window(t0, t1):
            frame = self.nodes.get(s.node_id)
            if frame is None:
                continue
            out[frame.key] = out.get(frame.key, 0) + s.delta
        return out

    def total_time_us(
        self,
        t0: int | None = None,
        t1: int | None = None,
    ) -> dict[tuple[str, str, int, int], int]:
        """Microseconds where a frame appeared ANYWHERE on the stack."""
        out: dict[tuple[str, str, int, int], int] = {}
        for s in self.window(t0, t1):
            for frame in dict.fromkeys(self.stack(s.node_id)):
                out[frame.key] = out.get(frame.key, 0) + s.delta
        return out

    def frame_by_key(self, key: tuple[str, str, int, int]) -> CallFrame | None:
        for f in self.nodes.values():
            if f.key == key:
                return f
        return None

    def assert_deltas_match_wall(self, tolerance: float = DELTA_WALL_TOLERANCE) -> dict[str, Any]:
        """Deltas must sum to the profiled wall span; if not, every self time is off by an unknown
        factor."""
        if not self.samples:
            raise CellFailure("cpuprofile_empty", "profile contained no samples")
        delta_sum = sum(s.delta for s in self.samples)
        wall = self.chunk_ts_last - self.chunk_ts_first
        if wall <= 0:
            raise CellFailure("cpuprofile_wall", f"non-positive chunk wall span {wall} us")
        drift = abs(delta_sum - wall) / wall
        report = {
            "delta_sum_us": delta_sum,
            "chunk_wall_us": wall,
            "drift": drift,
            "tolerance": tolerance,
            "sample_count": len(self.samples),
            "negative_deltas": self.negative_deltas,
        }
        if drift > tolerance:
            raise CellFailure(
                "cpuprofile_delta_wall_mismatch",
                f"summed timeDeltas {delta_sum} us vs chunk wall {wall} us "
                f"= {drift * 100:.2f}% drift, above {tolerance * 100:.0f}%",
            )
        return report

    def sampling_interval_us(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return (self.chunk_ts_last - self.chunk_ts_first) / max(1, len(self.samples))


def _call_frame(raw: dict[str, Any]) -> CallFrame:
    return CallFrame(
        function_name = str(raw.get("functionName", "")),
        script_id = str(raw.get("scriptId", "")),
        url = str(raw.get("url", "")),
        line = int(raw.get("lineNumber", -1)),
        column = int(raw.get("columnNumber", -1)),
        code_type = str(raw.get("codeType", "")),
    )


def parse_cpu_profiles(trace: Trace) -> dict[str, CpuProfile]:
    """Chunks sit on the V8 profiler thread, so they are matched by (pid, id), never by thread id."""
    profiles: dict[str, CpuProfile] = {}
    for e in trace.events:
        if e.get("name") != "Profile" or "cpu_profiler" not in str(e.get("cat", "")):
            continue
        data = (e.get("args") or {}).get("data") or {}
        pid = int(e.get("pid", 0))
        key = f"{pid}:{e.get('id')}"
        profiles[key] = CpuProfile(
            pid = pid,
            tid = int(e.get("tid", 0)),
            # Anchor on `ts`: V8 writes `startTime` without converting to the trace clock domain.
            start_time = int(e.get("ts", data.get("startTime", 0))),
            declared_start_time = int(data.get("startTime", 0)),
            source = str(data.get("source", "")),
        )

    chunks = [
        e
        for e in trace.events
        if e.get("name") == "ProfileChunk" and "cpu_profiler" in str(e.get("cat", ""))
    ]
    chunks.sort(key = lambda e: int(e.get("ts", 0)))

    for e in chunks:
        pid = int(e.get("pid", 0))
        key = f"{pid}:{e.get('id')}"
        prof = profiles.get(key)
        if prof is None:
            continue
        data = (e.get("args") or {}).get("data") or {}
        cpu = data.get("cpuProfile") or {}

        for raw in cpu.get("nodes") or ():
            nid = int(raw["id"])
            prof.nodes[nid] = _call_frame(raw.get("callFrame") or {})
            parent = raw.get("parent")
            if parent is not None:
                prof.parents[nid] = int(parent)
            # Some producers give `children` instead of `parent`.
            for child in raw.get("children") or ():
                prof.parents.setdefault(int(child), nid)

        for tid_key, node_id in (cpu.get("trace_ids") or {}).items():
            prof.trace_ids[str(tid_key)] = int(node_id)

        sample_ids = cpu.get("samples") or []
        deltas = data.get("timeDeltas") or []
        if len(sample_ids) != len(deltas):
            raise CellFailure(
                "cpuprofile_chunk_ragged",
                f"chunk at ts={e.get('ts')} has {len(sample_ids)} samples and "
                f"{len(deltas)} timeDeltas; they must be 1:1",
            )
        cursor = prof.samples[-1].ts if prof.samples else prof.start_time
        for nid, delta in zip(sample_ids, deltas):
            d = int(delta)
            if d < 0:
                prof.negative_deltas += 1
            cursor += d
            prof.samples.append(Sample(ts = cursor, node_id = int(nid), delta = d))

        prof.chunk_count += 1
        ts = int(e.get("ts", 0))
        prof.chunk_ts_first = ts if prof.chunk_count == 1 else prof.chunk_ts_first
        prof.chunk_ts_last = max(prof.chunk_ts_last, ts)

    return profiles


def main_thread_profile(trace: Trace) -> CpuProfile:
    """The CPU profile for the thread the renderer runs its main loop on."""
    pid, tid = trace.profiled_thread()
    profiles = parse_cpu_profiles(trace)
    for prof in profiles.values():
        if prof.pid == pid and prof.tid == tid:
            return prof
    raise CellFailure(
        "no_cpu_profile",
        f"no v8 CPU profile for the profiled thread {pid}:{tid}; "
        "was disabled-by-default-v8.cpu_profiler included in the categories?",
    )


def rank_self_time(
    profile: CpuProfile,
    t0: int | None = None,
    t1: int | None = None,
    *,
    include_synthetic: bool = False,
    limit: int = 40,
) -> list[tuple[CallFrame, int]]:
    """Leaf frames by self time, descending."""
    by_key = profile.self_time_us(t0, t1)
    frames: dict[tuple[str, str, int, int], CallFrame] = {}
    for f in profile.nodes.values():
        frames.setdefault(f.key, f)
    rows = []
    for key, us in by_key.items():
        frame = frames.get(key)
        if frame is None:
            continue
        if frame.is_synthetic and not include_synthetic:
            continue
        rows.append((frame, us))
    rows.sort(key = lambda r: -r[1])
    return rows[:limit]


def self_time_in_windows(
    profile: CpuProfile,
    windows: Sequence[tuple[int, int]],
    *,
    include_synthetic: bool = False,
    limit: int = 40,
) -> tuple[list[tuple[CallFrame, int]], dict[str, Any]]:
    """Check js_sample_count: synthetic V8 frames dominate short windows, so a few JS samples is noise."""
    totals: dict[tuple[str, str, int, int], int] = {}
    frames: dict[tuple[str, str, int, int], CallFrame] = {}
    js_samples = 0
    synthetic_us = 0
    js_us = 0
    for lo, hi in windows:
        for s in profile.window(lo, hi):
            frame = profile.nodes.get(s.node_id)
            if frame is None:
                continue
            if frame.is_synthetic:
                synthetic_us += s.delta
                if not include_synthetic:
                    continue
            else:
                js_samples += 1
                js_us += s.delta
            frames.setdefault(frame.key, frame)
            totals[frame.key] = totals.get(frame.key, 0) + s.delta
    rows = [(frames[k], v) for k, v in totals.items() if k in frames]
    rows.sort(key = lambda r: -r[1])
    diagnostics = {
        "windows": len(windows),
        "window_us": sum(max(0, hi - lo) for lo, hi in windows),
        "js_sample_count": js_samples,
        "js_us": js_us,
        "synthetic_us": synthetic_us,
        "underpowered": js_samples < MIN_JS_SAMPLES_FOR_RANKING,
        "min_js_samples_for_ranking": MIN_JS_SAMPLES_FOR_RANKING,
    }
    return rows[:limit], diagnostics


def stacks_under(
    profile: CpuProfile,
    predicate_function_name: str,
    t0: int | None = None,
    t1: int | None = None,
) -> list[tuple[int, list[CallFrame]]]:
    """Every sampled stack that contains a named function, leaf first."""
    out = []
    for ts, stack in profile.stacked_samples(t0, t1):
        if any(f.function_name == predicate_function_name for f in stack):
            out.append((ts, stack))
    return out


def summarise(
    profile: CpuProfile,
    t0: int | None = None,
    t1: int | None = None,
) -> dict[str, Any]:
    win = profile.window(t0, t1)
    total = sum(s.delta for s in win)
    synthetic = 0
    for s in win:
        f = profile.nodes.get(s.node_id)
        if f is not None and f.is_synthetic:
            synthetic += s.delta
    return {
        "samples": len(win),
        "sampled_us": total,
        "synthetic_us": synthetic,
        "js_us": total - synthetic,
        "sampling_interval_us": profile.sampling_interval_us(),
        "nodes_known": len(profile.nodes),
        "chunks": profile.chunk_count,
        "negative_deltas": profile.negative_deltas,
    }


def iter_leaf_frames(
    profile: CpuProfile, samples: Iterable[Sample] | None = None
) -> Iterable[tuple[Sample, CallFrame | None]]:
    for s in samples if samples is not None else profile.samples:
        yield s, profile.nodes.get(s.node_id)


def frames_matching(profile: CpuProfile, needles: Sequence[str]) -> list[CallFrame]:
    """Every known frame whose function name contains any of `needles`."""
    out: list[CallFrame] = []
    seen: set[tuple[str, str, int, int]] = set()
    for f in profile.nodes.values():
        if f.key in seen:
            continue
        if any(n in f.function_name for n in needles):
            seen.add(f.key)
            out.append(f)
    return out
