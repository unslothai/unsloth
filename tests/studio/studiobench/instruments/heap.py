# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sampling heap profiler, configured to see TRANSIENT garbage.

`HeapProfiler.startSampling` with `includeObjectsCollectedByMajorGC: true`. That
flag is load-bearing, not a nicety. Without it the returned profile contains
only objects that SURVIVED, and the hypothesis under test is the opposite: that
one work-in-progress fiber is cloned per sibling per render, allocated in a
burst and collected almost entirely at the next major GC. A survivors-only
profile of that mechanism is empty, and an empty profile reads as "no allocation
here", which is the exact wrong conclusion.

So the flag is REQUIRED. If the browser rejects it, this module raises rather
than retrying without it, because a quiet fallback would turn a missing
capability into a false negative and nothing downstream could tell.

The shape that confirms M1 is: allocation total proportional to sibling count,
attributed to a react-dom frame, with near-zero survival past a forced major GC.
`survival_ratio()` measures exactly that by taking a second profile after
`HeapProfiler.collectGarbage`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..analysis import CellFailure

# 4096 bytes between samples. Small enough to resolve a per-sibling allocation at a few hundred
# siblings, large enough not to perturb the allocation path.
DEFAULT_SAMPLING_INTERVAL = 4096

# `includeObjectsCollectedByMajorGC` / `MinorGC` landed in V8 10.8, which shipped in Chrome 108.
# Below that the parameter is accepted and ignored.
MIN_CHROME_FOR_GC_FLAGS = 108


@dataclass(frozen = True)
class HeapFrame:
    function_name: str
    url: str
    line: int
    column: int
    script_id: str = ""

    def label(self) -> str:
        return f"{self.function_name or '(anonymous)'} @ {self.url or 'script#' + self.script_id}:{self.line}:{self.column}"

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (self.function_name, self.url, self.line, self.column)


@dataclass
class HeapProfile:
    """Flattened sampling profile: bytes attributed to allocation sites."""

    self_bytes: dict[tuple[str, str, int, int], int] = field(default_factory = dict)
    frames: dict[tuple[str, str, int, int], HeapFrame] = field(default_factory = dict)
    total_bytes: int = 0
    sample_count: int = 0
    included_major_gc: bool = False

    def top(self, limit: int = 30) -> list[tuple[HeapFrame, int]]:
        rows = [(self.frames[k], v) for k, v in self.self_bytes.items() if k in self.frames]
        rows.sort(key = lambda r: -r[1])
        return rows[:limit]

    def bytes_matching(self, needles: Iterable[str]) -> int:
        needles = tuple(needles)
        total = 0
        for key, size in self.self_bytes.items():
            frame = self.frames.get(key)
            if frame and any(n in frame.function_name or n in frame.url for n in needles):
                total += size
        return total

    def summary(self) -> dict[str, Any]:
        return {
            "total_bytes": self.total_bytes,
            "sample_count": self.sample_count,
            "distinct_sites": len(self.self_bytes),
            "included_objects_collected_by_major_gc": self.included_major_gc,
        }


def _flatten(node: dict[str, Any], profile: HeapProfile) -> None:
    cf = node.get("callFrame") or {}
    frame = HeapFrame(
        function_name = str(cf.get("functionName", "")),
        url = str(cf.get("url", "")),
        line = int(cf.get("lineNumber", -1)),
        column = int(cf.get("columnNumber", -1)),
        script_id = str(cf.get("scriptId", "")),
    )
    size = int(node.get("selfSize", 0) or 0)
    if size:
        profile.frames.setdefault(frame.key, frame)
        profile.self_bytes[frame.key] = profile.self_bytes.get(frame.key, 0) + size
        profile.total_bytes += size
    for child in node.get("children") or ():
        _flatten(child, profile)


class SamplingHeapProfiler:
    def __init__(
        self,
        cdp: Any,
        *,
        sampling_interval: int = DEFAULT_SAMPLING_INTERVAL,
        include_major_gc: bool = True,
        include_minor_gc: bool = False,
    ) -> None:
        self.cdp = cdp
        self.sampling_interval = int(sampling_interval)
        self.include_major_gc = bool(include_major_gc)
        self.include_minor_gc = bool(include_minor_gc)
        self._running = False

    def assert_gc_flags_supported(self) -> int:
        """Check the browser is new enough for the GC-inclusion flags.

        THIS CANNOT BE FEATURE-DETECTED BY CATCHING AN ERROR. V8's inspector
        silently ignores unknown parameters to `HeapProfiler.startSampling`, so
        an old browser accepts `includeObjectsCollectedByMajorGC` with a cheerful
        empty success result and then hands back a survivors-only profile. The
        flag would appear to work and the answer would be wrong in the exact
        direction that hides the hypothesis. So the check is on the version:
        the flags landed in V8 10.8, which shipped in Chrome 108.
        """
        version = self.cdp.send("Browser.getVersion") or {}
        product = str(version.get("product", ""))
        major = 0
        for part in product.split("/")[-1].split("."):
            if part.isdigit():
                major = int(part)
                break
        if major and major < MIN_CHROME_FOR_GC_FLAGS:
            raise CellFailure(
                "heap_gc_flag_unsupported",
                f"{product} is older than Chrome {MIN_CHROME_FOR_GC_FLAGS}, where "
                "includeObjectsCollectedByMajorGC landed. Older browsers ignore the "
                "parameter silently and return a survivors-only profile, which for a "
                "transient-allocation hypothesis is empty and reads as 'no allocation "
                "here'. Refusing rather than reporting that.",
            )
        return major

    def start(self) -> None:
        if self._running:
            raise RuntimeError("SamplingHeapProfiler.start called twice")
        if self.include_major_gc or self.include_minor_gc:
            self.assert_gc_flags_supported()
        self.cdp.send("HeapProfiler.enable")
        self.cdp.send(
            "HeapProfiler.startSampling",
            {
                "samplingInterval": self.sampling_interval,
                "includeObjectsCollectedByMajorGC": self.include_major_gc,
                "includeObjectsCollectedByMinorGC": self.include_minor_gc,
            },
        )
        self._running = True

    def stop(self) -> HeapProfile:
        if not self._running:
            raise RuntimeError("SamplingHeapProfiler.stop without start")
        res = self.cdp.send("HeapProfiler.stopSampling")
        self._running = False
        return self._parse(res)

    def _parse(self, res: dict[str, Any]) -> HeapProfile:
        raw = res.get("profile") or {}
        head = raw.get("head")
        if head is None:
            raise CellFailure("heap_profile_empty", "stopSampling returned no profile head")
        profile = HeapProfile(included_major_gc = self.include_major_gc)
        _flatten(head, profile)
        profile.sample_count = len(raw.get("samples") or ())
        return profile

    def peek(self) -> HeapProfile:
        """Read the profile without stopping the profiler.

        Used to take the survivors arm: force a major GC, then peek. The
        profiler keeps accumulating afterwards, so this is non-destructive.
        """
        if not self._running:
            raise RuntimeError("SamplingHeapProfiler.peek without start")
        return self._parse(self.cdp.send("HeapProfiler.getSamplingProfile"))

    def collect_garbage(self) -> None:
        self.cdp.send("HeapProfiler.collectGarbage")

    def __enter__(self) -> "SamplingHeapProfiler":
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._running:
            try:
                self.cdp.send("HeapProfiler.stopSampling")
            finally:
                self._running = False


def survival_ratio(
    allocated: HeapProfile, survivors: HeapProfile, needles: Iterable[str]
) -> dict[str, Any]:
    """How much of what a site allocated is still alive after a major GC.

    Takes TWO profiles from TWO arms of the identical workload, because one
    session cannot produce both. A sampling profiler only records allocations
    made after it starts, so you cannot start a second profiler after a GC and
    learn anything about objects allocated before it. The two arms are:

    * `allocated`: `include_major_gc=True`. Everything the workload allocated,
      collected or not.
    * `survivors`: `include_major_gc=False`, with a forced
      `HeapProfiler.collectGarbage` before `stop()`. The profiler drops
      collected objects, so what remains is what outlived the GC.

    A ratio near zero is the signature of per-render churn; a ratio near one is
    retention, which is a different bug with a different fix.
    """
    needles = tuple(needles)
    if not allocated.included_major_gc:
        raise CellFailure(
            "heap_survival_arms_swapped",
            "the `allocated` profile was captured without includeObjectsCollectedByMajorGC, "
            "so it already excludes the transient garbage the ratio is about",
        )
    if survivors.included_major_gc:
        raise CellFailure(
            "heap_survival_arms_swapped",
            "the `survivors` profile was captured WITH includeObjectsCollectedByMajorGC, "
            "so it counts collected objects as survivors and the ratio would read 1.0",
        )
    alloc = allocated.bytes_matching(needles)
    alive = survivors.bytes_matching(needles)
    return {
        "needles": list(needles),
        "allocated_bytes": alloc,
        "surviving_bytes": alive,
        "survival_ratio": (alive / alloc) if alloc else None,
        "interpretation_note": (
            "near 0 means transient per-render churn; near 1 means retention. "
            "None means the site allocated nothing measurable, which is not the "
            "same as allocating nothing."
        ),
    }


# Harness adapter (INTERFACES.md section 3)
# Level 3. One sampling session per window, started with `includeObjectsCollectedByMajorGC` so
# transient garbage is visible: the hypothesis is per-render churn, and a survivors-only profile
# of that is empty.
# The survivors arm is NOT taken here. It needs a second run of the identical workload with the
# flag off, which is an ablation arm belonging to Layer 3. This emits the allocation side plus the
# site breakdown, so `analysis.heap.survival_ratio` can be applied across two arms afterwards.

import time  # noqa: E402

from ..analysis import assert_no_bare_zero, measured, merge, unmeasured  # noqa: E402
from . import register_instrument  # noqa: E402


class HeapInstrument:
    """Sampling allocation profile per window, including collected objects."""

    name = "heap"
    level = 3

    def __init__(
        self,
        top_n: int = 20,
        include_major_gc: bool = True,
    ) -> None:
        self.ctx: Any = None
        self.cdp: Any = None
        self.top_n = top_n
        self.include_major_gc = include_major_gc
        self.profiler: SamplingHeapProfiler | None = None
        self._overhead_ms = 0.0
        self._windows = 0
        self._reason = ""

    def attach(self, ctx: Any) -> None:
        self.ctx = ctx

    def start_cell(self, cell: Any) -> None:
        self.cdp = getattr(self.ctx, "cdp", None)
        self._overhead_ms = 0.0
        self._windows = 0
        self._reason = (
            "" if self.cdp is not None else "no CDP session; HeapProfiler is Chromium only"
        )

    def open(self, window: Any) -> None:
        if self.cdp is None:
            return
        t0 = time.perf_counter()
        self.profiler = SamplingHeapProfiler(self.cdp, include_major_gc = self.include_major_gc)
        try:
            self.profiler.start()
        except CellFailure as exc:
            # The version gate. Refusing is correct: an older browser ignores the flag silently and hands back
            # survivors only, which for a transient-allocation hypothesis reads as "no allocation here".
            self._reason = f"{exc.gate}: {exc.detail}"
            self.profiler = None
        except Exception as exc:  # noqa: BLE001
            self._reason = f"{type(exc).__name__}: {exc}"
            self.profiler = None
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0

    def close(self, window: Any) -> dict | None:
        if self.profiler is None:
            return merge(
                unmeasured("allocated_bytes", self._reason or "heap sampling not running"),
                {"active": False, "included_objects_collected_by_major_gc": self.include_major_gc},
            )
        t0 = time.perf_counter()
        try:
            prof = self.profiler.stop()
            payload = merge(
                measured("allocated_bytes", int(prof.total_bytes)),
                measured("allocation_sites", len(prof.self_bytes)),
                measured(
                    "top_sites",
                    [{"site": f.label(), "bytes": int(n)} for f, n in prof.top(self.top_n)],
                ),
                {
                    "active": True,
                    "included_objects_collected_by_major_gc": bool(prof.included_major_gc),
                    "survivors_arm_note": (
                        "this is the ALLOCATION side only. A survival ratio needs a second "
                        "arm of the identical workload with the flag off; see "
                        "analysis.heap.survival_ratio"
                    ),
                },
            )
        except Exception as exc:  # noqa: BLE001
            payload = merge(
                unmeasured("allocated_bytes", f"{type(exc).__name__}: {exc}"),
                {"active": True, "included_objects_collected_by_major_gc": self.include_major_gc},
            )
        finally:
            self.profiler = None
        self._windows += 1
        self._overhead_ms += (time.perf_counter() - t0) * 1000.0
        assert_no_bare_zero(payload, "heap")
        return payload

    def end_cell(self, cell: Any) -> dict | None:
        # A prose key is OMITTED when there is nothing to say, never set to None. `None` is reserved for a
        # QUANTITY that could not be measured.
        out = merge(
            measured("overhead_ms", round(self._overhead_ms, 3)),
            measured("windows_sampled", self._windows),
            {"headline_safe": False},
            {"reason": self._reason} if self._reason else {},
        )
        assert_no_bare_zero(out, "heap.end_cell")
        return out

    def detach(self) -> None:
        if self.profiler is not None:
            try:
                self.profiler.stop()
            except Exception:
                pass
            self.profiler = None


@register_instrument(name = "heap", level = 3)
def _make_heap() -> HeapInstrument:
    return HeapInstrument()
