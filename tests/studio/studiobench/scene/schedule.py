# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scene: a FIXED-DURATION FILM, slot-scheduled on wall clock.

THIS IS NOT A TASK LIST, and the difference is the whole point.

A sequential script runs action 1, waits for it, runs action 2. On a slow machine each action
takes longer, so the session is longer, so the stream has delivered more characters by the time
action 7 runs, so action 7 happens against a bigger thread, at a different point in the stream,
with a different amount of content mounted. The slow machine has taken a DIFFERENT PATH through a
DIFFERENT-LENGTH session, and none of its columns are comparable with the fast machine's. Every
number is then a mixture of "this machine is slower" and "this machine measured something else",
and no amount of care downstream can separate them.

So every action has a fixed `(t_start_ms, budget_ms)` on the session clock. The scheduler waits
until `t_start_ms`, runs the action with the remaining budget, and moves on. A machine too slow to
reach a slot in time records `slot_missed: true` and the film rolls on. Both machines see the same
thread at the same point in the same stream, and a missed slot is an honest, first-class reading
of "this machine could not do this here", which is exactly the finding worth having.

The deficit-scheduled pacer is the other half of this. Slots are only meaningful if the stream's
own progress is a function of wall clock, which is what deficit scheduling buys.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..runtime.types import ActionContext, ActionResult, Cell, Slot, Window, not_run
from . import default_budget_ms, get_action


@dataclass
class Scene:
    name: str
    slots: list[Slot]

    @property
    def duration_ms(self) -> int:
        return max((s.t_start_ms + s.budget_ms for s in self.slots), default = 0)

    def scaled(self, factor: float) -> "Scene":
        """A scene of the same SHAPE over a longer film.

        Used to give a big rung more room without changing the order or the relative spacing of
        the actions, so a 1K scene and a 1M scene are the same film at different speeds rather
        than two different films.
        """
        return Scene(name = self.name, slots = [
            Slot(action = s.action, t_start_ms = int(s.t_start_ms * factor),
                 budget_ms = int(s.budget_ms * factor), args = s.args, required = s.required)
            for s in self.slots])


def _slots(spec: list[tuple[str, int, Optional[int]]]) -> list[Slot]:
    return [Slot(action = name, t_start_ms = start,
                 budget_ms = budget if budget is not None else default_budget_ms(name))
            for name, start, budget in spec]


# The standard film. Ordered so that the actions that must happen DURING generation come first,
# then the ones that need a finished reply, then the destructive ones last -- delete and reopen
# change the thread, so anything after them would be measuring a different thread.
#
# Timings are offsets from the moment the send button was pressed.
STANDARD = Scene(name = "standard", slots = _slots([
    # ── during generation ────────────────────────────────────────
    ("scroll_during_generation", 3_000, 8_000),
    ("keystroke",                12_000, 6_000),
    ("scroll_during_generation", 19_000, 8_000),
    ("stop_generation",          28_000, 8_000),
    # ── after the reply is complete ──────────────────────────────
    ("scroll_after",             38_000, 8_000),
    ("reasoning_toggle",         47_000, 12_000),
    ("message_menu",             60_000, 12_000),
    ("copy_markdown",            73_000, 6_000),
    ("select_text",              80_000, 6_000),
    ("select_all_copy",          87_000, 10_000),
    ("composer_fill",            98_000, 10_000),
    ("model_change",            109_000, 10_000),
    ("settings",                120_000, 12_000),
    ("image_upload",            133_000, 12_000),
    # ── destructive, last ────────────────────────────────────────
    ("thread_reopen",           146_000, 30_000),
    ("delete_message",          177_000, 15_000),
]))

# The quick film. The SAME fifteen actions in the same order -- a tier that drops actions cannot be
# compared with one that does not -- on a shorter clock, for the small rungs where the stream is
# over in seconds.
QUICK = Scene(name = "quick", slots = _slots([
    ("scroll_during_generation",  1_500, 4_000),
    ("keystroke",                 6_000, 5_000),
    ("scroll_during_generation", 11_500, 4_000),
    ("stop_generation",          16_000, 6_000),
    ("scroll_after",             22_500, 5_000),
    ("reasoning_toggle",         28_000, 8_000),
    ("message_menu",             36_500, 8_000),
    ("copy_markdown",            45_000, 5_000),
    ("select_text",              50_500, 5_000),
    ("select_all_copy",          56_000, 7_000),
    ("composer_fill",            63_500, 7_000),
    ("model_change",             71_000, 7_000),
    ("settings",                 78_500, 9_000),
    ("image_upload",             88_000, 9_000),
    ("thread_reopen",            97_500, 25_000),
    ("delete_message",          123_000, 12_000),
]))

SCENES = {"quick": QUICK, "standard": STANDARD, "full": STANDARD}


@dataclass
class SceneRunner:
    """Runs a scene against a live page, one window per slot."""

    cell: Cell
    page: Any
    cdp: Any
    dom: Any
    recorder: Any
    open_window: Callable[[str, str], Any]
    log: Callable[[str], None]
    base_args: dict = field(default_factory = dict)

    def run(self, scene: Scene, t0: float) -> list[dict]:
        """`t0` is the driver monotonic time the film started, i.e. when send was pressed."""
        rows: list[dict] = []
        for i, slot in enumerate(scene.slots):
            # The GAP before this slot is itself a measured window. Without it, frame rate and
            # blocked time would only ever be sampled inside actions -- and the quiet stretches
            # between them, which is where the stream is doing its work unaided, would be the one
            # part of the session nothing observed. Continuous coverage, no sampler thread, and
            # no window ever overlaps another.
            self._gap_window(f"stream:gap{i}", slot.t_start_ms, t0)
            row = self._run_slot(slot, t0)
            rows.append(row)
            self.recorder.emit(row)
        return rows

    def _census(self) -> dict:
        try:
            return self.page.evaluate("() => window.__sb.dom.counts()")
        except Exception as exc:                                    # noqa: BLE001
            return {"census_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _gap_window(self, name: str, until_ms: int, t0: float) -> None:
        now_ms = (time.monotonic() - t0) * 1000
        if until_ms - now_ms < 250:
            return
        with self.open_window(name, "stream") as window:
            while (time.monotonic() - t0) * 1000 < until_ms:
                time.sleep(min(0.2, max(0.01, (until_ms - (time.monotonic() - t0) * 1000) / 1000)))
            window.note("waited_to_ms", until_ms)
            window.note("census", self._census())

    def _run_slot(self, slot: Slot, t0: float) -> dict:
        entry = get_action(slot.action)
        window_name = f"action:{slot.action}"
        if entry is None:
            return ActionResult(ran = False,
                                reason = f"no action named {slot.action!r} is registered").row(
                slot.action, window_name, self.cell.cell_id)

        now_ms = (time.monotonic() - t0) * 1000
        # Wait for the slot to open. Sleeping in small steps rather than one long sleep so a
        # renderer crash is noticed within a fifth of a second rather than at the end of the wait.
        while now_ms < slot.t_start_ms:
            time.sleep(min(0.2, (slot.t_start_ms - now_ms) / 1000))
            now_ms = (time.monotonic() - t0) * 1000

        deadline_ms = slot.t_start_ms + slot.budget_ms
        remaining = deadline_ms - now_ms
        if remaining <= 0:
            # THE SLOT WAS MISSED. Not an error and not a slow timing: this machine could not get
            # here in time, the film carries on, and the row says exactly that.
            self.log(f"    slot missed: {slot.action} "
                     f"(due at {slot.t_start_ms}ms, reached at {now_ms:.0f}ms)")
            return ActionResult(
                ran = False, slot_missed = True,
                reason = (f"the slot opened at {slot.t_start_ms}ms and this machine reached it at "
                          f"{now_ms:.0f}ms, past its {slot.budget_ms}ms budget"),
                expect = {"t_start_ms": slot.t_start_ms, "reached_at_ms": round(now_ms, 1)},
            ).row(slot.action, window_name, self.cell.cell_id)

        with self.open_window(window_name, "action") as window:
            ctx = ActionContext(
                page = self.page, cdp = self.cdp, cell = self.cell, window = window,
                args = {**self.base_args, **slot.args}, budget_ms = int(remaining),
                dom = self.dom, log = self.log)
            try:
                result = entry.fn(ctx)
            except Exception as exc:                                # noqa: BLE001
                self.log(f"    action {slot.action} raised: {type(exc).__name__}: {exc}")
                result = not_run(f"the action raised {type(exc).__name__}: {exc}")
            window.note("action", slot.action)
            window.note("ran", result.ran)
            # A CENSUS PER ACTION, taken at the close of every window.
            #
            # Not a nicety. The end-of-cell census is taken after the film has finished, and the
            # film ENDS with thread_reopen and delete_message -- so the "final" census of the
            # first working run read 0 assistant messages and 0 characters, having faithfully
            # measured a thread the benchmark had just deleted. A census per window gives the
            # occupancy at the moment each action ran, which is the denominator every per-action
            # cost needs anyway, and it makes the peak recoverable no matter what the last action
            # did. Measured cost on a 1,500-element tree: 0.2ms.
            window.note("census", self._census())

        over_ms = ((time.monotonic() - t0) * 1000) - deadline_ms
        row = result.row(slot.action, window_name, self.cell.cell_id)
        row["window_ms"] = window.duration_ms
        row["census"] = window.notes.get("census")
        # An action that ran but overran its budget has pushed nothing (the next slot has its own
        # absolute start), but it has overlapped the next one, so it is flagged.
        row["over_budget_ms"] = round(over_ms, 1) if over_ms > 0 else 0.0
        row["over_budget"] = over_ms > 0
        status = ("ran" if result.ran else "NOT RUN")
        verdict = "" if result.expect_ok is not False else " EXPECT FAILED"
        self.log(f"    {slot.action}: {status}{verdict}"
                 f"{'' if result.reason is None else ' -- ' + result.reason}")
        return row
