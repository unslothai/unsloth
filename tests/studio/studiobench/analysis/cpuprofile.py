# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn `ProfileChunk` trace events into (timestamp, stack) pairs.

This is the module that deletes the concept of a residual. A renderer bucket
like `TaskOtherDuration` is a number the renderer computed for its own
accounting and it has no stack; a V8 CPU sample has a stack, so every
microsecond it covers resolves to a leaf call frame with an ancestry.

Four properties of the real wire format, each confirmed against a captured
trace rather than assumed, and each of which silently produces a wrong answer if
you get it wrong:

* `nodes` arrive INCREMENTALLY. In a real 118-chunk capture only 6 chunks
  carried a `nodes` array at all. A parser that reads nodes per chunk and
  discards them resolves almost every sample to "unknown node".
* `timeDeltas` lives at `args.data.timeDeltas`, a SIBLING of `cpuProfile`, not
  inside it. `samples` lives at `args.data.cpuProfile.samples`.
* `ts`, `Profile.args.data.startTime` and every entry of `timeDeltas` are
  MICROSECONDS on the same monotonic clock as the rest of the trace.
* `timeDeltas` entries can be NEGATIVE. The V8 sampler timestamps samples on the
  sampling thread and small reorderings happen. Summing is still correct;
  clamping each delta at zero is not, because it inflates the total.

The sum of `timeDeltas` over a window must equal the window wall duration.
Assert it, and FAIL THE CELL when it does not, rather than rescaling: a scale
factor that reconciles a broken profile with wall clock is a way of making every
downstream number look plausible and be wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from . import CellFailure
from .traceparse import Trace

# V8 synthetic frames. Their time is real but they are not call frames anyone can fix, so they are
# labelled and excluded from frame ranking rather than dropped from totals.
SYNTHETIC_FRAMES = frozenset({"(root)", "(program)", "(idle)", "(garbage collector)"})

# How far the summed sample deltas may drift from the wall span of the chunks before the cell is unusable.
DELTA_WALL_TOLERANCE = 0.02

# Below this many non-synthetic samples inside a window set, a leaf ranking is not a ranking: at a
# ~150 us interval a 40 us task contributes at most one sample, so short tasks can produce a
# confident-looking table built on five data points. Reported as `underpowered`, never hidden.
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
        """Identity of a function, stable across chunks and across a session.

        Keyed on script id and source position rather than on name, because the
        name is very often the empty string for a module top level or an
        anonymous callback, and because two different functions minified to the
        same short name are different functions.
        """
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
    # `Profile.args.data.startTime`, kept only so the clock skew against `ts` can be reported. Never used as an anchor.
    declared_start_time: int = 0
    source: str = ""
    # `sampleTraceId` -> node id, from `ProfileChunk.args.data.cpuProfile.trace_ids`. An EXACT stack,
    # not a sampled one: with `disabled-by-default-devtools.timeline.stack` on, Blink calls
    # `v8::CpuProfiler::CollectSample` with a tagged id at specific instrumentation points, so the
    # stack is the real one at that instant.
    # Not whatever the 100 us sampler happened to catch.
    trace_ids: dict[str, int] = field(default_factory = dict)

    @property
    def clock_skew_us(self) -> int:
        if not self.declared_start_time:
            return 0
        return self.start_time - self.declared_start_time

    def stack_for_sample_trace_id(self, sample_trace_id: int | str) -> list[CallFrame]:
        """The EXACT stack recorded for a tagged instrumentation point.

        Timeline events such as `EventDispatch`, `FunctionCall`,
        `RequestAnimationFrame` and `TimerInstall` carry
        `args.data.sampleTraceId` when the timeline.stack category is enabled.
        Looking that id up here gives the true stack at that event, with none of
        the sampling error that makes short windows unrankable.
        """
        node = self.trace_ids.get(str(sample_trace_id))
        if node is None:
            return []
        return self.stack(node)

    # ------------------------------------------------------------------ stacks

    def stack(self, node_id: int) -> list[CallFrame]:
        """Leaf-first ancestry of a sample node.

        Guards against a cyclic `parent` chain, which a corrupt trace can
        produce and which would otherwise hang the whole analysis.
        """
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

    # ------------------------------------------------------------- aggregation

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

    # ------------------------------------------------------------------- gates

    def assert_deltas_match_wall(self, tolerance: float = DELTA_WALL_TOLERANCE) -> dict[str, Any]:
        """Sum of deltas must equal the profiled wall span within `tolerance`.

        A profile whose deltas do not add up to the wall clock is either
        truncated or came from a different clock, and either way every self time
        derived from it is wrong by an unknown factor.
        """
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
    """Build every CPU profile in a trace, keyed by the `Profile` event id.

    The `Profile` event carries the pid/tid of the PROFILED thread and the
    `startTime` anchor. `ProfileChunk` events carry the same `id` but are
    emitted on the V8 profiler's own thread (`v8:ProfEvntProc`), so they are
    correlated by `(pid, id)` and never by thread id. Filtering chunks by the
    renderer main thread id returns zero samples, which is indistinguishable
    from the CPU profiler having been left off.
    """
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
            # ANCHOR ON `ts`, NOT ON `args.data.startTime`. V8 writes `startTime` straight from
            # `base::TimeTicks` without converting into the trace's clock domain, and both V8's source
            # comment and the DevTools frontend say to use the event timestamp. They agree to within
            # microseconds on a healthy capture, but the drift is unbounded in principle and would attribute
            # every sample to the neighbouring task.
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
            # A chunk with no matching Profile event cannot be anchored to a start time, so its sample
            # timestamps would be meaningless.
            continue
        data = (e.get("args") or {}).get("data") or {}
        cpu = data.get("cpuProfile") or {}

        for raw in cpu.get("nodes") or ():
            nid = int(raw["id"])
            prof.nodes[nid] = _call_frame(raw.get("callFrame") or {})
            parent = raw.get("parent")
            if parent is not None:
                prof.parents[nid] = int(parent)
            # Some producers give `children` instead of `parent`; honour both so ancestry does not silently
            # collapse to a forest of roots.
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
    """Rank leaf frames across a set of disjoint time windows.

    This is how a task ORIGIN becomes a NAMED FRAME: take the windows of every
    task classified as, say, message-channel, and rank the leaves sampled inside
    them. Windows are half-open and assumed non-overlapping, which holds for
    top-level `RunTask` intervals on one thread.

    The returned diagnostics carry `js_sample_count`, and callers must look at
    it. Synthetic V8 frames ((program), (idle), (garbage collector)) routinely
    dominate short windows at a 150 us sampling interval, so a ranking built
    from a handful of JS samples is noise wearing a function name. Dropping the
    synthetic frames silently would hide exactly that.
    """
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
