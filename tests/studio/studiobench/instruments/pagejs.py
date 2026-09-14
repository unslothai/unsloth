# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Registers the three page-side instruments: `frames`, `input`, `glass`.

One module for all three because they share the same shape -- an init script installed before app
code, drained per window -- and because a file per instrument would be three copies of the same
error handling. The JS lives in the sibling .js files so it can be read and edited as JavaScript.

THE TRI-CLOCK GATE lives here, in `frames`. Three independent measures of whether frames happened:
the rAF loop, a 1ms timer's lag, and CDP `Page.startScreencast` presented frames. rAF unscheduled
reads as "no dropped frames" -- a page whose main thread never yields at all schedules no rAF, and
a naive reader calls that a clean window. When the three disagree by more than 20% the window is
marked `clocks_disagree` and the report layer excludes it from scoring rather than averaging a
number that three instruments cannot agree happened.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..runtime.types import BenchContext, Cell, Instrument, Window
from . import register_instrument

_HERE = Path(__file__).resolve().parent

# How far the three clocks may disagree before the window is not scoreable. Not tuned: it is the
# threshold declared in the design, and the point is that it is declared rather than discovered.
CLOCK_DISAGREEMENT_LIMIT = 0.20


def _js(name: str) -> str:
    from ..runtime import resources
    return resources.read_text(f"instruments/{name}")


class _PageInstrument(Instrument):
    """Shared plumbing: install an init script once, drain a page function per window."""

    script_name = ""
    read_expr = ""

    def __init__(self) -> None:
        self.ctx: Optional[BenchContext] = None
        self.page: Any = None
        self._open_at: Optional[float] = None
        self.unavailable: Optional[str] = None

    def attach(self, ctx: BenchContext) -> None:
        self.ctx = ctx
        try:
            ctx.context.add_init_script(_js(self.script_name))
        except Exception as exc:  # noqa: BLE001
            self.unavailable = f"could not install {self.script_name}: {exc}"

    def start_cell(self, cell: Cell) -> None:
        # Re-read the page every cell: a crashed renderer is recovered by opening a NEW page, so an
        # instrument that cached it in attach() would spend the rest of the run evaluating against a closed
        # one.
        self.page = self.ctx.page if self.ctx else None

    def _eval(
        self,
        expr: str,
        arg: Any = None,
    ) -> Any:
        if self.page is None:
            return None
        try:
            return self.page.evaluate(expr, arg) if arg is not None else self.page.evaluate(expr)
        except Exception as exc:  # noqa: BLE001
            self.unavailable = f"{type(exc).__name__}: {exc}"
            return None


@register_instrument(name = "frames", level = 0)
def _frames():
    return FramesInstrument()


class FramesInstrument(_PageInstrument):
    name = "frames"
    level = 0
    script_name = "frames.js"

    def __init__(self) -> None:
        super().__init__()
        self.clamp: Optional[dict] = None
        self._screencast_frames = 0
        self._screencast_on = False
        self._lock = threading.Lock()

    def attach(self, ctx: BenchContext) -> None:
        super().attach(ctx)
        self._arm_screencast()

    def _arm_screencast(self) -> None:
        """CDP presented frames: the only one of the three clocks that is not the page's own
        opinion of itself. A page whose main thread is wedged reports nothing from rAF and nothing
        from a timer; the compositor still presents, or visibly does not."""
        ctx = self.ctx
        if ctx is None or ctx.cdp is None:
            return
        try:

            def on_frame(params):
                with self._lock:
                    self._screencast_frames += 1
                try:
                    ctx.cdp.send(
                        "Page.screencastFrameAck", {"sessionId": params.get("sessionId", 0)}
                    )
                except Exception:  # noqa: BLE001
                    pass

            ctx.cdp.on("Page.screencastFrame", on_frame)
            # Tiny frames: the point is the COUNT and its timing, and a full-size capture would cost the
            # renderer real encode time inside the window being measured.
            ctx.cdp.send(
                "Page.startScreencast",
                {
                    "format": "jpeg",
                    "quality": 1,
                    "maxWidth": 32,
                    "maxHeight": 32,
                    "everyNthFrame": 1,
                },
            )
            self._screencast_on = True
        except Exception:  # noqa: BLE001
            self._screencast_on = False

    def calibrate(self, idle_ms: int = 1200) -> dict:
        """Calibrate the timer clamp during an ENFORCED IDLE WINDOW.

        Called by the session immediately before each measured window, with nothing streaming and
        no action in flight. Calibrating from the first ticks of a page that already has 31,637
        elements standing measures the app's steady-state load and calls it the timer floor, then
        subtracts that floor out of every window and reports a saturated page as 0.2% busy.
        """
        if self.page is None:
            self.clamp = {"clampMs": None, "reason": "no page"}
            return self.clamp
        self._eval("() => window.__sb.frames.beginCalibration()")
        time.sleep(idle_ms / 1000)
        self.clamp = self._eval("() => window.__sb.frames.endCalibration()") or {
            "clampMs": None,
            "reason": "calibration did not return",
        }
        return self.clamp

    def open(self, window: Window) -> None:
        self._open_at = time.monotonic()
        with self._lock:
            self._screencast_frames = 0
        self._eval("() => window.__sb.frames.reset()")

    def close(self, window: Window) -> Optional[dict]:
        if self.unavailable:
            return {"unavailable": self.unavailable, "frames_attempted": False}
        elapsed_ms = (time.monotonic() - (self._open_at or time.monotonic())) * 1000
        out = self._eval("(ms) => window.__sb.frames.read(ms)", elapsed_ms)
        if out is None:
            return {
                "unavailable": self.unavailable or "the page did not answer",
                "frames_attempted": False,
            }
        with self._lock:
            presented = self._screencast_frames
        out["driver_elapsed_ms"] = round(elapsed_ms, 2)
        out.update(self._clock_agreement(out, presented, elapsed_ms))
        return out

    def _clock_agreement(self, out: dict, presented: int, elapsed_ms: float) -> dict:
        """Three clocks, and a window they disagree about is not a window worth scoring.

        WHAT THE SCREENCAST CLOCK IS, AND WHAT IT IS NOT. `Page.startScreencast` was first used
        here as a third FRAME COUNT, on the reasoning that the compositor is the one observer that
        is not the page's own opinion of itself. Measured, it presents 4 to 5 frames in a 4-second
        window where the rAF loop counts 240 -- a 98% disagreement, in EVERY window, on a page
        that was demonstrably running at a steady 60 fps with 2% blocked time. Chromium's
        screencast emits on VISUAL CHANGE and is rate-limited; it is not a vsync counter. A gate
        wired to it would have excluded every window in every run from scoring, and a gate that
        always fires is a gate someone turns off.

        So it is kept as a LIVENESS signal -- did the compositor present anything at all, which
        separates "the page is idle" from "the renderer is wedged" -- and the agreement check is
        between the two clocks that do measure the same thing: the rAF loop and the 1ms timer,
        both of which are main-thread progress. If the main thread is blocked, rAF callbacks stop
        AND timer ticks stop, and they must stop together.
        """
        raf = out.get("frames")
        lag_ticks = out.get("lag_ticks")
        clamp = out.get("clamp_ms")
        expected_ticks = (elapsed_ms / clamp) if clamp else None
        result: dict = {
            "compositor_presented_frames": presented if self._screencast_on else None,
            "compositor_presented": (presented > 0) if self._screencast_on else None,
            "compositor_attempted": self._screencast_on,
            "compositor_note": (
                "a liveness signal, NOT a frame rate: Chromium's screencast "
                "emits on visual change and is rate-limited"
            ),
            "timer_ticks_expected": None if expected_ticks is None else round(expected_ticks, 1),
        }
        if raf is None or not expected_ticks or lag_ticks is None:
            result["clocks_agree"] = None
            result["clocks_reason"] = (
                "the timer clamp was not established, so the rAF loop has nothing to be checked "
                "against and frame counts rest on the page's own report alone"
            )
            return result
        # THE THIRD CLOCK IS NOT RESOLVED, AND THIS SAYS SO RATHER THAN INVENTING IT. Two of the three are
        # sound: the 1ms timer has a real expectation and `timer_clock_ratio` measures how much of its
        # budget the main thread could answer, and the compositor is a liveness signal only. The rAF loop
        # has NO sound expectation on a headless engine, since there is no display and it runs as fast as
        # it can. Normalising against the best window in the cell was tried and is wrong: an early idle
        # window runs far above 60, so every later window scores about 0.44 and `clocks_agree` came out
        # FALSE on 34 of 34 windows of a page demonstrably steady at 60 fps with 2% blocked time.
        # So `clocks_agree` is null WITH A REASON, `timer_clock_ratio` is the load-bearing availability
        # signal, and the frame columns are the page's own account of itself. Restoring the third clock
        # needs a vsync-locked source: a headed run, or `Page.startScreencast` correlated against
        # `LatencyInfo` in a trace, which is Layer 2's surface.
        result["timer_clock_ratio"] = round(lag_ticks / expected_ticks, 3)
        result["clocks_agree"] = None
        result["clocks_reason"] = (
            "the tri-clock check is not implementable on a headless engine as designed: rAF has "
            "no vsync to be checked against and the compositor screencast is rate-limited to "
            "visual change. timer_clock_ratio is the sound availability signal; the frame columns "
            "are the page's own report"
        )
        return result

    def end_cell(self, cell: Cell) -> Optional[dict]:
        return {"clamp": self.clamp, "overhead_ms": None, "overhead_attempted": False}

    def detach(self) -> None:
        if self._screencast_on and self.ctx and self.ctx.cdp:
            try:
                self.ctx.cdp.send("Page.stopScreencast")
            except Exception:  # noqa: BLE001
                pass


@register_instrument(name = "input", level = 0)
def _input():
    return InputInstrument()


class InputInstrument(_PageInstrument):
    """Armed and drained by the keystroke action rather than per window: a window that contained
    no typing has nothing to report, and reporting a zero for it would be a bare zero."""

    name = "input"
    level = 0
    script_name = "input.js"

    def arm(self, selector: str) -> dict:
        return self._eval("(s) => window.__sb.input.arm(s)", selector) or {
            "armed": False,
            "reason": self.unavailable or "the page did not answer",
        }

    def settled(self) -> dict:
        """Whether a keystroke's paint is still in flight. `None` when the page cannot answer, so
        a caller polling on it stops rather than looping to its bound."""
        return self._eval("() => window.__sb.input.settled()")

    def collect(self, expected: int) -> dict:
        return self._eval("(n) => window.__sb.input.collect(n)", expected) or {
            "samples": 0,
            "samples_attempted": False,
            "reason": self.unavailable or "the page did not answer",
        }

    def close(self, window: Window) -> Optional[dict]:
        return None


@register_instrument(name = "glass", level = 1)
def _glass():
    return GlassInstrument()


class GlassInstrument(_PageInstrument):
    """Level 1: it wraps hot accessors on Element.prototype and perturbs what it measures. The
    headline numbers come from level 0, where it is not installed at all."""

    name = "glass"
    level = 1
    script_name = "glass.js"

    def open(self, window: Window) -> None:
        self._eval("() => window.__sb.glass && window.__sb.glass.read()")

    def close(self, window: Window) -> Optional[dict]:
        out = self._eval("() => window.__sb.glass && window.__sb.glass.read()")
        if out is None:
            return {
                "glass_attempted": False,
                "unavailable": self.unavailable or "glass.js is not installed",
            }
        return out
