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


# The standard film, ordered so actions that must happen DURING generation come first, then
# ones needing a finished reply, then the destructive ones last, since delete and reopen change
# the thread. Timings are offsets from the send button press.
STANDARD = Scene(
    name = "standard",
    slots = _slots(
        [
            # ── during generation ────────────────────────────────────────
            ("scroll_during_generation", 3_000, 8_000),
            ("keystroke", 12_000, 6_000),
            # 12s: the tail is a JITTERED clip capped at 6,000 characters, so its drain varies from about
            # 14s to 18s and a during-generation slot must open before the SHORTEST of those. At 19s, then
            # 15s, this slot ran against a finished reply at the top rung.
            ("scroll_during_generation", 12_000, 8_000),
            ("stop_generation", 28_000, 8_000),
            # ── after the reply is complete ──────────────────────────────
            ("scroll_after", 38_000, 8_000),
            ("reasoning_toggle", 47_000, 12_000),
            ("send_turn", 60_000, 12_000),
            ("message_menu", 73_000, 12_000),
            ("copy_markdown", 86_000, 6_000),
            ("select_text", 93_000, 6_000),
            ("send_turn", 100_000, 12_000),
            # 35s, not 10s: selecting and copying the whole thread at 1M took 27,690 ms, nearly three times
            # a 10s budget, and the OVERRUN pushed the next slot past its own start, recorded as
            # `composer_fill: slot missed`. This film is what 500K and 1M use, so its budgets are sized
            # from what those rungs cost; at 100K the same action takes 2,476 ms.
            ("select_all_copy", 113_000, 35_000),
            ("composer_fill", 149_000, 10_000),
            ("model_change", 160_000, 10_000),
            ("settings", 171_000, 12_000),
            ("image_upload", 184_000, 12_000),
            # 22,382 ms at 1M, so 30s stands.
            # ── destructive, last ────────────────────────────────────────
            ("thread_reopen", 197_000, 30_000),
            ("delete_message", 228_000, 15_000),
        ]
    ),
)

# The quick film: the SAME fifteen actions in the same order, since a tier that drops actions
# cannot be compared with one that does not, on a shorter clock for the small rungs.
QUICK = Scene(
    name = "quick",
    slots = _slots(
        [
            # BUDGETS ARE SIZED FROM MEASURED ACTION COST: at 100K, the largest rung this film is used for,
            # every action finished inside 2.5 s (select_all_copy 2,476 ms, thread_reopen 2,234 ms,
            # reasoning_toggle 1,788 ms, keystroke 1,041 ms, the rest under 500 ms), and the budgets carry
            # roughly 2.5x headroom. The film previously ran 162 s while its actions used 6.4 s, and that
            # waiting is multiplied by every cell, arm and repetition. What CANNOT be compressed is the
            # stream phase: the opening turn drains in 12 to 18 s, during-generation slots must open inside
            # the shortest (14.1 s at 1M) and after-generation slots after the longest (17.8 s at 100K), so
            # the gap between 12 s and 20 s is that constraint, not slack.
            ("scroll_during_generation", 1_500, 2_500),
            ("keystroke", 5_000, 3_000),
            ("scroll_during_generation", 9_500, 2_500),
            ("stop_generation", 20_000, 3_000),
            ("scroll_after", 23_500, 2_500),
            ("reasoning_toggle", 26_500, 4_500),
            # 1,500 rather than 4,000, because of the gap AFTER it rather than the cost of the send:
            # `send_turn` is a sub-100 ms action, so a 4,000 ms window is permission to begin the follow-up
            # four seconds late without recording a miss, and every one of those seconds comes off
            # `message_menu`'s window. Measured from the latest legal send, this film left 3,500 ms for a
            # 4,562 ms drain; the fast film had the same shape and CI failed on it.
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

# The fast film. FOR ITERATION, NOT FOR REPORTING. Same eighteen actions in the same order as
# the other two, because a tier that drops actions cannot tell you that your fix broke the one
# it dropped; what changes is the waiting. Budgets are sized from costs measured at 100K over
# eight null-control cells at roughly 1.5x rather than the quick film's 2.5x (thread_reopen
# 3,163 ms and select_all_copy 2,454 ms at the top, everything past model_change under 500 ms).
# At 1.5x headroom an action that overruns records `slot_missed` instead of silently pushing
# the next slot. WHAT CANNOT BE COMPRESSED, and why this film is 47 s: the opening turn streams
# a 6,000 character tail at field cadence (328.8 chars/s), draining in 14 to 18 s, so the gap
# between 8 s and 19 s below is that constraint, and shrinking it means shrinking the streamed
# tail, which changes the load being measured.
FAST = Scene(
    name = "fast",
    slots = _slots(
        [
            # 18.4 s, not 8 s: `stop_generation` starts and stops its OWN turn, so opening it while the
            # opening tail is still draining truncates the reply being measured, the defect that made the
            # seeded-vs-streamed check read a false 20% drift. KEYED TO THE DECLARED TAIL, not to what the
            # corpus happens to stream: at 17.8 s it passed only because the 1M rung was streaming recycled
            # text, and STREAM_TAIL_CHARS (6,000) at field cadence is 18.25 s, the ceiling to clear.
            # THE SECOND PACKING CONSTRAINT, learned the expensive way: `send_turn` starts a FOLLOW-UP turn
            # of FOLLOW_UP_CHARS (1,500) at field cadence, so it streams for 4.6 s, and anything needing a
            # SETTLED reply does not exist while it runs. The first fast film opened `message_menu` 1.7 s
            # after a `send_turn` and recorded `NOT RUN: no More button` on 312 of 312 attempts across a 36
            # job sweep. Every post-send slot below clears 4.6 s, and
            # test_settled_actions_open_after_the_follow_up_drains fails any film that does not.
            ("scroll_during_generation", 1_500, 1_200),
            ("keystroke", 3_000, 1_800),
            ("scroll_during_generation", 6_000, 1_200),
            ("stop_generation", 18_400, 3_000),
            ("scroll_after", 21_500, 1_200),
            # THE BUDGET IS SIZED FROM WHAT CI MEASURES, not the null control's 2,250 ms: on the GitHub
            # runner this action costs about 4,420 ms, of which reaching the open state alone is 2,898.8
            # ms. At 3,500 it overran by 934 ms on every run, and an overrun here lands on `send_turn`, the
            # one slot whose downstream gap is load-bearing.
            # 5,000 ms covers the measured cost with room.
            ("reasoning_toggle", 23_000, 5_000),
            # MOVED OUT FROM UNDER `reasoning_toggle`: a send slot that opens before the action before it
            # can finish starts late, and a late send shortens the drain window of everything after it
            # while reporting no miss of its own.
            # Which now nominally ends at 28,000.
            ("send_turn", 28_100, 1_500),
            # THE GAP IS THE POINT, measured from 29,600, the latest this film's send can fire, not from
            # 28,100. The follow-up drains in 4,562 ms nominal and 4,400 to 4,700 observed, so a window
            # closing at 35,400 leaves about 1.1 s of margin; the old packing left 38 ms, which is why CI
            # failed on a runner no slower than the one that passed.
            ("message_menu", 34_400, 1_000),
            ("copy_markdown", 35_400, 600),
            ("select_text", 36_200, 400),
            ("send_turn", 36_800, 1_500),
            ("select_all_copy", 41_900, 4_000),
            ("composer_fill", 46_100, 600),
            ("model_change", 46_900, 1_000),
            ("settings", 48_100, 1_200),
            ("image_upload", 49_500, 800),
            # thread_reopen 6.5 s and delete 2.5 s, not 5 s and 0.6 s: measured at 100K the pair costs about
            # 3.2 s, but the ACTION overran its 5 s window and pushed the last slot, so delete recorded NOT
            # EXERCISED on the base arm of every null-control cell in a 36 job sweep. The film's own end is
            # the one place an overrun has nowhere to go.
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
            # The GAP before this slot is itself a measured window: without it, frame rate and blocked time
            # would only be sampled inside actions, and the quiet stretches where the stream does its work
            # unaided would be the one part of the session nothing observed.
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
            # A page that cannot install it still gets a structural digest; the visible-region capture
            # reports itself absent rather than empty, which the analysis refuses.
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
        # `gap`, NOT `stream`. These windows were labelled `stream` and read as meaning the stream was
        # running in them. A gap window opens before EVERY slot, so on the standard film eighteen exist
        # and only the first four contain any streaming: on a 100K cell, `stream:gap12` ran 32.9 s at
        # 1.6% busy with the reply finished thirty seconds earlier, while `stream:drain` was 7 ms.
        # Anyone filtering on `kind == "stream"` therefore selected mostly post-stream idle. The window
        # NAME is deliberately left as `stream:gapN`, since it is the join key in every payload already
        # written.
        # THE CENSUS IS TAKEN BEFORE THE GAP WINDOW OPENS, not at its close (workspace task #102): a gap
        # window is the QUIET stretch, and nineteen querySelectorAll passes over 195,000 elements at the
        # end of it landed on the frame-rate reading for the idle phase. Before rather than after,
        # because the gap ends the instant the next slot is due, so taking it afterwards would turn an
        # instrument cost into a missed slot.
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
        # Wait for the slot to open, in small steps rather than one long sleep, so a renderer crash is
        # noticed within a fifth of a second.
        while now_ms < slot.t_start_ms:
            time.sleep(min(0.2, (slot.t_start_ms - now_ms) / 1000))
            now_ms = (time.monotonic() - t0) * 1000

        deadline_ms = slot.t_start_ms + slot.budget_ms
        remaining = deadline_ms - now_ms
        if remaining <= 0:
            # THE SLOT WAS MISSED. Not an error and not a slow timing: this machine could not get here in
            # time, the film carries on, and the row says exactly that.
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

        # THE CENSUS AND THE PARITY DIGEST ARE TAKEN OUTSIDE THE WINDOW (workspace task #102). Both used
        # to run inside the `with`, on the strength of "0.2ms on a 1,500-element tree" measured on a
        # tree three orders of magnitude smaller: at 100K+ the census walks nineteen querySelectorAll
        # passes over ~195,000 elements and the digest serialises 5.6 MB, and every millisecond was
        # charged to the preceding action, so delete_message reported 14.3 fps against a true 49.0. That
        # inverted the ranking of the actions this campaign is about, in the direction that makes
        # standing DOM look like a smaller problem, because the instrument's cost grows with the
        # quantity under investigation. They still run at the same MOMENT, so the digest and occupancy
        # come from one reading; only the accounting moved, and they now live on the ACTION row.
        # And message_menu 17.1 fps against a true 73.8.
        # THE DEADLINE IS SAMPLED HERE, BEFORE THE OBSERVATIONS: leaving `over_ms` to be taken after them
        # would still flag an action `over_budget` on the strength of a multi-megabyte serialisation
        # that is explicitly not its cost.
        window_closed_at = time.monotonic()
        over_ms = ((window_closed_at - t0) * 1000) - deadline_ms

        # THE VISIBLE CAPTURE GOES FIRST, AND THE ORDER IS THE MEASUREMENT. The other two read a DOM
        # sitting still; this one CLOSES an observation accumulating since `_watch_visible`, so every
        # millisecond it stays open is another in which a row can scroll into view and be counted as
        # visible during the action. Taken after the census and digest, that tail was as long as those
        # two probes take, and they are the one thing here whose cost is proportional to the arm, so the
        # two arms' observers stayed open for materially different intervals (14.3 fps against 49.0 is
        # the same asymmetry expressed as time). `compare_visible` returns DIFFER on strict set
        # inequality of `ever_visible` and the null control cannot absorb it, being same-build against
        # same-build. Taking it first does not make the tail zero, since `capture()` still waits two
        # frames on purpose; it makes it the SAME ON BOTH ARMS.
        visible = self._visible()
        census = self._census()
        parity = self._parity()
        # The observations DO consume wall clock before the next slot, so their cost is recorded as
        # theirs rather than dropped.
        observation_ms = (time.monotonic() - window_closed_at) * 1000
        row = result.row(slot.action, window_name, self.cell.cell_id)
        row["window_ms"] = window.duration_ms
        # READ FROM THE LOCALS ABOVE, not from `window.notes`: the three observations were moved out of
        # the measured window, so by now they are values this method holds.
        row["census"] = census
        row["parity"] = parity
        row["visible"] = visible
        # The observation cost itself, so it is never invisible again. `census_cost_ms` is the page's
        # own timing of the walk.
        row["observation_outside_window"] = True
        row["observation_ms"] = round(observation_ms, 1)
        # THE SCREENSHOT IS TAKEN OUTSIDE THE WINDOW: the film runs on a wall clock with absolute slot
        # starts, so an encode charged to the measured window eats the gap before the next slot, which
        # is how an action comes to report a MISSED SLOT on a contended runner, and `--assert-liveness`
        # counts those. Out here it costs the gap, which is what the gap is for.
        if isinstance(row.get("parity"), dict) and row["parity"].get("parity_attempted"):
            row["parity"].update(self._parity_shot(slot.action))
        # An action that ran but overran its budget has pushed nothing (the next slot has its own
        # absolute start), but it has overlapped the next one, so it is flagged.
        row["over_budget_ms"] = round(over_ms, 1) if over_ms > 0 else 0.0
        row["over_budget"] = over_ms > 0
        status = "ran" if result.ran else "NOT RUN"
        verdict = "" if result.expect_ok is not False else " EXPECT FAILED"
        self.log(
            f"    {slot.action}: {status}{verdict}"
            f"{'' if result.reason is None else ' -- ' + result.reason}"
        )
        return row
