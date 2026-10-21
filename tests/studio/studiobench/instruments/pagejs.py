# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Frames' tri-clock gate: a window where rAF, timer and screencast disagree by over 20% is excluded."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional

from ..runtime.types import BenchContext, Cell, Instrument, Window
from . import register_instrument

_HERE = Path(__file__).resolve().parent

# Declared design threshold, not tuned.
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
        # Re-read every cell: crash recovery opens a NEW page.
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
            # Tiny frames so encoding does not cost renderer time inside the window.
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
        """Calibrate only in an enforced idle window: a loaded page's load would be read as the
        timer floor."""
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
        """Screencast is liveness only, as it emits on visual change; agreement is rAF against the
        1ms timer."""
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
        # The rAF clock has no sound expectation headless, so clocks_agree is null with a reason.
        # timer_clock_ratio is the load-bearing availability signal.
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
