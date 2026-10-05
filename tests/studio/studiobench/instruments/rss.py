# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser-tree RSS, sampled on a thread. SALVAGED from playwright_reasoning_pane.py.

WHY NOT `performance.memory`. `usedJSHeapSize` is the JS heap of one renderer. Three things are
wrong with reading it as the memory cost of a long thread. It excludes the DOM, which is where a
thread of 31,637 elements actually lives, and which is C++ on the other side of the heap boundary.
It excludes every other process in the browser tree, and the GPU process holds the layer buffers
for a tall page. And it is a heap SIZE, so a garbage collection that has not run yet reads as
growth and one that just ran reads as a leak repaired.

RSS of the whole browser tree has none of those problems and one of its own: the browser's pid is
not something Playwright exposes. The browser is launched by the node driver, which is a child of
this process, so the browser is a grandchild with an engine-specific name and a per-platform
executable path. Diffing the descendant set across the launch identifies it without knowing any of
that.

None, never 0.0. A renderer that has exited and a permission failure both produce no reading, and
reporting either as zero would put a fabricated floor under `rss_growth_mb`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from ..runtime.types import BenchContext, Cell, Instrument, Window
from . import register_instrument

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

SAMPLE_MS = 500


def snapshot_children(root_pid: int) -> dict:
    if psutil is None:
        return {}
    try:
        return {p.pid: p for p in psutil.Process(root_pid).children(recursive = True)}
    except Exception:  # noqa: BLE001
        return {}


def new_roots(root_pid: int, before: dict) -> list:
    """Processes under `root_pid` that were not there in `before`, topmost first."""
    if psutil is None:
        return []
    after = snapshot_children(root_pid)
    new = {pid: proc for pid, proc in after.items() if pid not in before}
    roots = []
    for pid, proc in new.items():
        try:
            if proc.ppid() not in new:
                roots.append(proc)
        except Exception:  # noqa: BLE001
            continue
    return roots


def tree_rss_mb(roots: list) -> Optional[float]:
    if psutil is None or not roots:
        return None
    total = 0
    read_any = False
    for root in roots:
        try:
            procs = [root, *root.children(recursive = True)]
        except Exception:  # noqa: BLE001
            continue
        for proc in procs:
            try:
                total += proc.memory_info().rss
                read_any = True
            except Exception:  # noqa: BLE001
                continue
    return round(total / 1048576, 1) if read_any else None


class RssSampler:
    """Samples on a THREAD, because the driver is blocked inside Playwright's sync API for the
    whole run and cannot sample anything itself. Touches no Playwright object, only psutil, so it
    is safe alongside the sync API's greenlet."""

    def __init__(
        self,
        roots: list,
        period_ms: int = SAMPLE_MS,
    ) -> None:
        self.roots = roots
        self.period_s = max(0.05, period_ms / 1000)
        self.samples: list[tuple[float, Optional[float]]] = []
        self.reason: Optional[str] = None
        if psutil is None:
            self.reason = "psutil is not installed"
        elif not roots:
            self.reason = "the browser process could not be identified"
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.monotonic()
        if self.reason is not None:
            return
        self._thread = threading.Thread(target = self._loop, daemon = True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            at = time.monotonic()
            self.samples.append(((at - self._t0) * 1000, tree_rss_mb(self.roots)))
            self._stop.wait(self.period_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout = 5)
        if self.reason is None and not [s for _, s in self.samples if s is not None]:
            self.reason = "no RSS reading succeeded for the browser tree"

    def between(self, t0_ms: float, t1_ms: float) -> list[float]:
        return [v for t, v in self.samples if v is not None and t0_ms <= t <= t1_ms]

    def latest(self) -> Optional[float]:
        usable = [v for _, v in self.samples if v is not None]
        return usable[-1] if usable else None

    def growth_mb(self) -> Optional[float]:
        """Peak minus the FIRST reading of this run, not minus zero. A browser that is already up
        carries a base footprint that has nothing to do with the thread, and the run before this
        one in the same browser leaves its own. What is claimed is what this run ADDED."""
        usable = [v for _, v in self.samples if v is not None]
        if len(usable) < 2:
            return None
        return round(max(usable) - usable[0], 1)


@register_instrument(name = "rss", level = 0)
def _rss():
    return RssInstrument()


class RssInstrument(Instrument):
    name = "rss"
    level = 0

    def __init__(self) -> None:
        self.sampler: Optional[RssSampler] = None
        self.ctx: Optional[BenchContext] = None
        self._cell_start_ms: Optional[float] = None
        self._open_ms: Optional[float] = None

    def attach(self, ctx: BenchContext) -> None:
        self.ctx = ctx
        self.sampler = RssSampler(ctx.browser_procs)
        self.sampler.start()

    def start_cell(self, cell: Cell) -> None:
        self._cell_start_ms = self._now_ms()

    def _now_ms(self) -> float:
        return (time.monotonic() - (self.sampler._t0 if self.sampler else 0.0)) * 1000

    def open(self, window: Window) -> None:
        self._open_ms = self._now_ms()

    def close(self, window: Window) -> Optional[dict]:
        if self.sampler is None:
            return {"rss_attempted": False, "reason": "no sampler"}
        if self.sampler.reason:
            return {"rss_mb": None, "rss_attempted": False, "reason": self.sampler.reason}
        window_samples = self.sampler.between(self._open_ms or 0.0, self._now_ms())
        return {
            "rss_attempted": True,
            "rss_mb": window_samples[-1] if window_samples else self.sampler.latest(),
            "rss_peak_mb": max(window_samples) if window_samples else None,
            "rss_samples": len(window_samples),
        }

    def end_cell(self, cell: Cell) -> Optional[dict]:
        if self.sampler is None or self.sampler.reason:
            return {
                "rss_growth_mb": None,
                "rss_attempted": False,
                "reason": (self.sampler.reason if self.sampler else "no sampler"),
            }
        cell_samples = self.sampler.between(self._cell_start_ms or 0.0, self._now_ms())
        growth = None
        if len(cell_samples) >= 2:
            growth = round(max(cell_samples) - cell_samples[0], 1)
        return {
            "rss_attempted": True,
            "rss_growth_mb": growth,
            "rss_start_mb": cell_samples[0] if cell_samples else None,
            "rss_peak_mb": max(cell_samples) if cell_samples else None,
            "rss_run_growth_mb": self.sampler.growth_mb(),
            "overhead_ms": 0.0,
            "overhead_attempted": True,
        }

    def detach(self) -> None:
        if self.sampler is not None:
            self.sampler.stop()
