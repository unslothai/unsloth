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
from pathlib import Path
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
        return Scene(
            name = self.name,
            slots = [
                Slot(
                    action = s.action,
                    t_start_ms = int(s.t_start_ms * factor),
                    budget_ms = int(s.budget_ms * factor),
                    args = s.args,
                    required = s.required,
                )
                for s in self.slots
            ],
        )


def _slots(spec: list[tuple[str, int, Optional[int]]]) -> list[Slot]:
    return [
        Slot(
            action = name,
            t_start_ms = start,
            budget_ms = budget if budget is not None else default_budget_ms(name),
        )
        for name, start, budget in spec
    ]


# Ordered: during-generation actions first, then settled-reply ones, destructive last.
# Timings are offsets from the send button press.
STANDARD = Scene(
    name = "standard",
    slots = _slots(
        [
            ("scroll_during_generation", 3_000, 8_000),
            ("keystroke", 12_000, 6_000),
            # The tail drain varies 14s to 18s, so during-generation slots must open before the shortest.
            ("scroll_during_generation", 12_000, 8_000),
            ("stop_generation", 28_000, 8_000),
            ("scroll_after", 38_000, 8_000),
            ("reasoning_toggle", 47_000, 12_000),
            ("send_turn", 60_000, 12_000),
            ("message_menu", 73_000, 12_000),
            ("copy_markdown", 86_000, 6_000),
            ("select_text", 93_000, 6_000),
            ("send_turn", 100_000, 12_000),
            # Budget sized from 1M, where this action took ~27.7s.
            ("select_all_copy", 113_000, 35_000),
            ("composer_fill", 149_000, 10_000),
            ("model_change", 160_000, 10_000),
            ("settings", 171_000, 12_000),
            ("image_upload", 184_000, 12_000),
            ("thread_reopen", 197_000, 30_000),
            ("delete_message", 228_000, 15_000),
        ]
    ),
)

# Same actions in the same order as STANDARD so tiers stay comparable; shorter clock.
QUICK = Scene(
    name = "quick",
    slots = _slots(
        [
            # Budgets are ~2.5x measured action cost at 100K. The 12s to 20s gap is the stream drain
            # constraint, not slack.
            ("scroll_during_generation", 1_500, 2_500),
            ("keystroke", 5_000, 3_000),
            ("scroll_during_generation", 9_500, 2_500),
            ("stop_generation", 20_000, 3_000),
            ("scroll_after", 23_500, 2_500),
            ("reasoning_toggle", 26_500, 4_500),
            # A short window: a late send silently eats message_menu's drain window.
            ("send_turn", 31_500, 1_500),
            ("message_menu", 36_000, 3_000),
            ("copy_markdown", 39_500, 2_500),
            ("select_text", 42_500, 2_000),
            ("send_turn", 45_000, 1_500),
            ("select_all_copy", 49_500, 6_000),
            ("composer_fill", 56_000, 2_500),
            ("model_change", 59_000, 2_500),
            ("settings", 62_000, 2_500),
            ("image_upload", 65_000, 3_000),
            ("thread_reopen", 68_500, 6_000),
            ("delete_message", 75_000, 2_500),
        ]
    ),
)

# The fast film, for iteration, not reporting. Same actions, budgets ~1.5x measured cost so
# overruns record slot_missed. The 8s to 19s gap is the opening tail drain and cannot shrink.
FAST = Scene(
    name = "fast",
    slots = _slots(
        [
            # stop_generation opens after STREAM_TAIL_CHARS at field cadence (18.25s) so it does not
            # truncate the measured reply. Post-send slots must clear the 4.6s follow-up drain (see
            # test_settled_actions_open_after_the_follow_up_drains).
            ("scroll_during_generation", 1_500, 1_200),
            ("keystroke", 3_000, 1_800),
            ("scroll_during_generation", 6_000, 1_200),
            ("stop_generation", 18_400, 3_000),
            ("scroll_after", 21_500, 1_200),
            # Sized from CI cost (~4.4s); an overrun here lands on send_turn.
            ("reasoning_toggle", 23_000, 5_000),
            # Must not overlap reasoning_toggle, which nominally ends at 28,000.
            ("send_turn", 28_100, 1_500),
            # Measured from 29,600, the latest the send can fire; the follow-up drains in ~4.6s.
            ("message_menu", 34_400, 1_000),
            ("copy_markdown", 35_400, 600),
            ("select_text", 36_200, 400),
            ("send_turn", 36_800, 1_500),
            ("select_all_copy", 41_900, 4_000),
            ("composer_fill", 46_100, 600),
            ("model_change", 46_900, 1_000),
            ("settings", 48_100, 1_200),
            ("image_upload", 49_500, 800),
            # The film's last slots have nowhere to push an overrun, so budgets carry extra room.
            ("thread_reopen", 50_500, 6_500),
            ("delete_message", 57_200, 2_500),
        ]
    ),
)
SCENES = {"fast": FAST, "quick": QUICK, "standard": STANDARD, "full": STANDARD}


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
            # Gap windows measure the quiet stretches where the stream runs unaided.
            self._gap_window(f"stream:gap{i}", slot.t_start_ms, t0)
            row = self._run_slot(slot, t0)
            rows.append(row)
            self.recorder.emit(row)
        return rows

    def _census(self) -> dict:
        try:
            return self.page.evaluate("() => window.__sb.dom.counts()")
        except Exception as exc:  # noqa: BLE001
            return {"census_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _watch_visible(self) -> None:
        """Install the visible-region observer BEFORE the window opens.

        Before, not after, because the compared set is the union of everything the viewport showed
        during the action. An action that scrolls reveals messages and hides them again, and an
        observer installed at the close would compare only wherever the scroll happened to stop.
        """
        try:
            self.page.evaluate("() => window.__sb.parityVisible.watch()")
        except Exception:  # noqa: BLE001
            # The visible-region capture then reports itself absent, which the analysis refuses.
            pass

    def _visible(self) -> dict:
        """The visible-region capture, taken at the close and BEFORE the census and the digest.

        It closes an accumulating observation rather than reading a static DOM, so it goes first;
        see the comment at the call site for what its tail costs when it does not.
        """
        try:
            got = self.page.evaluate("async () => await window.__sb.parityVisible.capture()")
            self.page.evaluate("() => window.__sb.parityVisible.stop()")
            return got
        except Exception as exc:  # noqa: BLE001
            return {"visible_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _parity(self) -> dict:
        """A structural digest of what is on screen, for the UI-parity check across arms.

        Taken at the CLOSE of the action window, at the same moment as the census, so the digest
        and the occupancy it should be read against come from one reading of one DOM.
        """
        want_raw = bool(self.base_args.get("parity_raw"))
        try:
            return self.page.evaluate("(raw) => window.__sb.parity.capture({ raw })", want_raw)
        except Exception as exc:  # noqa: BLE001
            return {"parity_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}

    def _parity_shot(self, action: str) -> dict:
        """A viewport PNG taken at the same instant as the digest, when `--parity-shots` asked.

        WHY THE VIEWPORT AND NOT THE DIFFERING ELEMENT. An element screenshot is the better
        picture and Playwright takes it by SCROLLING the element into view, which mutates the
        page. The next slot in the film is scripted against where the thread actually is, so a
        shot that scrolls has changed the run it was supposed to be observing. The viewport is
        what the user sees and costs nothing beyond the encode.

        SCROLL IS RECORDED RATHER THAN FORCED, for the same reason. Both arms are driven by one
        script inside one session so their offsets agree by construction, but "by construction"
        is a claim, and a pair of shots at different offsets looks exactly like a UI change. The
        number travels with the image and the composite refuses to present a mismatched pair as
        a comparison.
        """
        out = self.base_args.get("parity_shots")
        if not out:
            return {}
        label = self.base_args.get("arm_label") or "?"
        try:
            scroll = self.page.evaluate(
                "() => { const v = document.querySelector('.aui-thread-viewport');"
                " return v ? Math.round(v.scrollTop) : -1; }"
            )
        except Exception:  # noqa: BLE001
            scroll = -1
        name = f"{self.cell.cell_id}__{action}__{label}.png"
        path = Path(out) / name
        try:
            path.parent.mkdir(parents = True, exist_ok = True)
            self.page.screenshot(path = str(path))
        except Exception as exc:  # noqa: BLE001
            return {"shot_error": f"{type(exc).__name__}: {exc}"}
        return {"shot": name, "shot_scroll_top": scroll}

    def _gap_window(self, name: str, until_ms: int, t0: float) -> None:
        now_ms = (time.monotonic() - t0) * 1000
        if until_ms - now_ms < 250:
            return
        # Kind is `gap`, not `stream`: most gap windows contain no streaming. The name stays
        # `stream:gapN` because it is the join key in existing payloads. The census runs before the
        # window opens so its cost does not land on the idle-phase reading.
        census = self._census()
        with self.open_window(name, "gap") as window:
            window.note("census_before_gap", census)
            while (time.monotonic() - t0) * 1000 < until_ms:
                time.sleep(min(0.2, max(0.01, (until_ms - (time.monotonic() - t0) * 1000) / 1000)))
            window.note("waited_to_ms", until_ms)

    def _run_slot(self, slot: Slot, t0: float) -> dict:
        entry = get_action(slot.action)
        window_name = f"action:{slot.action}"
        if entry is None:
            return ActionResult(
                ran = False, reason = f"no action named {slot.action!r} is registered"
            ).row(slot.action, window_name, self.cell.cell_id)

        now_ms = (time.monotonic() - t0) * 1000
        # Small steps so a renderer crash is noticed quickly.
        while now_ms < slot.t_start_ms:
            time.sleep(min(0.2, (slot.t_start_ms - now_ms) / 1000))
            now_ms = (time.monotonic() - t0) * 1000

        deadline_ms = slot.t_start_ms + slot.budget_ms
        remaining = deadline_ms - now_ms
        if remaining <= 0:
            # A missed slot is not an error: the film carries on and the row records it.
            self.log(
                f"    slot missed: {slot.action} "
                f"(due at {slot.t_start_ms}ms, reached at {now_ms:.0f}ms)"
            )
            return ActionResult(
                ran = False,
                slot_missed = True,
                reason = (
                    f"the slot opened at {slot.t_start_ms}ms and this machine reached it at "
                    f"{now_ms:.0f}ms, past its {slot.budget_ms}ms budget"
                ),
                expect = {"t_start_ms": slot.t_start_ms, "reached_at_ms": round(now_ms, 1)},
            ).row(slot.action, window_name, self.cell.cell_id)

        self._watch_visible()
        with self.open_window(window_name, "action") as window:
            ctx = ActionContext(
                page = self.page,
                cdp = self.cdp,
                cell = self.cell,
                window = window,
                args = {**self.base_args, **slot.args},
                budget_ms = int(remaining),
                dom = self.dom,
                log = self.log,
            )
            try:
                result = entry.fn(ctx)
            except Exception as exc:  # noqa: BLE001
                self.log(f"    action {slot.action} raised: {type(exc).__name__}: {exc}")
                result = not_run(f"the action raised {type(exc).__name__}: {exc}")
            window.note("action", slot.action)
            window.note("ran", result.ran)

        # Census and parity digest run outside the window: their cost grows with the DOM and was being
        # charged to the preceding action. The deadline is sampled before them for the same reason.
        window_closed_at = time.monotonic()
        over_ms = ((window_closed_at - t0) * 1000) - deadline_ms

        # The visible capture goes first: it closes an accumulating observation, so taking it after
        # arm-dependent probes kept the two arms' observers open for different intervals.
        visible = self._visible()
        census = self._census()
        parity = self._parity()
        observation_ms = (time.monotonic() - window_closed_at) * 1000
        row = result.row(slot.action, window_name, self.cell.cell_id)
        row["window_ms"] = window.duration_ms
        row["census"] = census
        row["parity"] = parity
        row["visible"] = visible
        row["observation_outside_window"] = True
        row["observation_ms"] = round(observation_ms, 1)
        # Screenshot encoding is outside the window so it does not cause missed slots on slow runners.
        if isinstance(row.get("parity"), dict) and row["parity"].get("parity_attempted"):
            row["parity"].update(self._parity_shot(slot.action))
        # Slots have absolute starts, so an overrun overlaps the next slot instead of pushing it.
        row["over_budget_ms"] = round(over_ms, 1) if over_ms > 0 else 0.0
        row["over_budget"] = over_ms > 0
        status = "ran" if result.ran else "NOT RUN"
        verdict = "" if result.expect_ok is not False else " EXPECT FAILED"
        self.log(
            f"    {slot.action}: {status}{verdict}"
            f"{'' if result.reason is None else ' -- ' + result.reason}"
        )
        return row
