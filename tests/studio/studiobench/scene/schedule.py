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


# The standard film. Ordered so that the actions that must happen DURING generation come first,
# then the ones that need a finished reply, then the destructive ones last -- delete and reopen
# change the thread, so anything after them would be measuring a different thread.
#
# Timings are offsets from the moment the send button was pressed.
STANDARD = Scene(
    name = "standard",
    slots = _slots(
        [
            # ── during generation ────────────────────────────────────────
            ("scroll_during_generation", 3_000, 8_000),
            ("keystroke", 12_000, 6_000),
            # 12s. The tail is a JITTERED clip capped at 6,000 characters, so its drain time varies from
            # about 14s to 18s across the ladder, and a during-generation slot has to open before the
            # SHORTEST of those, not the longest. At 19s, then 15s, this slot ran against a finished reply
            # at the top rung while still being labelled "during generation".
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
            # 35s, not 10s. Selecting and copying the whole thread at the 1M rung took 27,690 ms -- two
            # million characters through the clipboard -- so it blew a 10s budget by nearly three times
            # and the OVERRUN pushed the next slot past its own start, which was recorded as
            # `composer_fill: slot missed`. The 1M cell lost two slots that way and the cause looked like
            # a slow machine rather than one under-budgeted action.
            #
            # This film is the one the 500K and 1M rungs use, so its budgets are sized from what those
            # rungs actually cost, not from what the small ones do. At 100K the same action takes 2,476 ms.
            ("select_all_copy", 113_000, 35_000),
            ("composer_fill", 149_000, 10_000),
            ("model_change", 160_000, 10_000),
            ("settings", 171_000, 12_000),
            ("image_upload", 184_000, 12_000),
            # ── destructive, last ────────────────────────────────────────
            # 22,382 ms at 1M, so 30s stands.
            ("thread_reopen", 197_000, 30_000),
            ("delete_message", 228_000, 15_000),
        ]
    ),
)

# The quick film. The SAME fifteen actions in the same order -- a tier that drops actions cannot be
# compared with one that does not -- on a shorter clock, for the small rungs where the stream is
# over in seconds.
QUICK = Scene(
    name = "quick",
    slots = _slots(
        [
            # BUDGETS ARE SIZED FROM MEASURED ACTION COST, not guessed. At the 100K rung -- the largest
            # this film is used for -- every action finished inside 2.5 s: select_all_copy 2,476 ms,
            # thread_reopen 2,234 ms, reasoning_toggle 1,788 ms, keystroke 1,041 ms, everything else under
            # 500 ms. The budgets below carry roughly 2.5x headroom over those.
            #
            # The film previously ran 162 s while its actions used 6.4 s, so 96% of it was waiting for the
            # next slot to open. That waiting is not free: it is multiplied by every cell, every arm and
            # every repetition, and an A/B at four reps is sixteen films. Halving the film halves the cost
            # of every comparison this tool exists to make.
            #
            # What CANNOT be compressed is the stream phase. The opening turn drains in 12 to 18 s, the
            # during-generation slots have to open inside the shortest of those (14.1 s at 1M) and the
            # after-generation slots have to open after the longest (17.8 s at 100K). The gap between
            # 12 s and 20 s below is that constraint, not slack.
            ("scroll_during_generation", 1_500, 2_500),
            ("keystroke", 5_000, 3_000),
            ("scroll_during_generation", 9_500, 2_500),
            ("stop_generation", 20_000, 3_000),
            ("scroll_after", 23_500, 2_500),
            ("reasoning_toggle", 26_500, 4_500),
            ("send_turn", 31_500, 4_000),
            ("message_menu", 36_000, 3_000),
            ("copy_markdown", 39_500, 2_500),
            ("select_text", 42_500, 2_000),
            ("send_turn", 45_000, 4_000),
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

# The fast film. FOR ITERATION, NOT FOR REPORTING -- see the banner the CLI prints.
#
# Same eighteen actions in the same order as the other two films, because a tier that drops
# actions cannot tell you that your fix broke the one it dropped. What changes is the waiting.
#
# Budgets are sized from the costs this tool MEASURED at the 100K rung over eight cells of a null
# control, at roughly 1.5x rather than the 2.5x the quick film carries:
#
#   thread_reopen  3,163 ms (close 2,205 + reopen 958)   select_all_copy  2,454 ms
#   reasoning_toggle 2,250 ms (open 1,688 + close 563)   keystroke        1,002 ms
#   settings 485 ms   model_change 483 ms   scroll 466 ms   stop 335 ms   copy 204 ms
#
# Everything else is under 100 ms. The eighteen actions cost about 12 s of real work; the standard
# film spends 243 s and the quick film 77.5 s to deliver them. At 1.5x headroom an action that
# overruns records `slot_missed` instead of silently pushing the next slot, which is the honest
# failure mode and is exactly what the fixed-duration design exists to provide.
#
# WHAT CANNOT BE COMPRESSED, and why this film is 47 s rather than 20 s: the opening turn streams
# a 6,000 character tail at field cadence (328.8 chars/s), so it drains in 14 to 18 s. The
# during-generation slots must open inside the SHORTEST of those and the after-generation slots
# after the LONGEST. The gap between 8 s and 19 s below is that constraint, not slack. Shrinking
# it means shrinking the streamed tail, which changes the load being measured -- and streaming
# frame cost at length is the thing most of these fixes are about.
FAST = Scene(
    name = "fast",
    slots = _slots(
        [
            # 18.2 s, not 8 s. `stop_generation` starts and stops its OWN turn, so opening it while the
            # opening tail is still draining starts a second turn on top of the first and truncates the
            # reply being measured -- which is the defect that made the seeded-vs-streamed equivalence
            # check read a false 20% drift earlier in this project. The worst-case drain across the ladder
            # is 17.8 s, and the packing test in fixture/selftest holds every film to it.
            #
            # THE SECOND PACKING CONSTRAINT, learned the expensive way. `send_turn` starts a FOLLOW-UP
            # turn, and a follow-up is FOLLOW_UP_CHARS (1,500) at field cadence, so it streams for 4.6 s.
            # Anything needing a SETTLED reply -- the action bar's More and Copy buttons, select-all,
            # delete -- does not exist while that turn is running. The first fast film opened
            # `message_menu` 1.7 s after a `send_turn` and it recorded `NOT RUN: no More button` on
            # 312 of 312 attempts across a 36 job sweep, silently removing the four actions that carry
            # the largest known effect in this codebase (the message menu is the 19x one). Every
            # post-send slot below clears 4.6 s, and test_settled_actions_open_after_the_follow_up_drains
            # now fails any film that does not.
            ("scroll_during_generation", 1_500, 1_200),
            ("keystroke", 3_000, 1_800),
            ("scroll_during_generation", 6_000, 1_200),
            ("stop_generation", 18_200, 3_000),
            ("scroll_after", 21_500, 1_200),
            ("reasoning_toggle", 23_000, 3_500),
            ("send_turn", 26_700, 1_500),
            ("message_menu", 32_000, 800),
            ("copy_markdown", 33_000, 600),
            ("select_text", 33_800, 400),
            ("send_turn", 34_400, 1_500),
            ("select_all_copy", 39_500, 4_000),
            ("composer_fill", 43_700, 600),
            ("model_change", 44_500, 1_000),
            ("settings", 45_700, 1_200),
            ("image_upload", 47_100, 800),
            # thread_reopen 6.5 s and delete 2.5 s, not 5 s and 0.6 s. Measured at 100K the pair
            # costs about 3.2 s, but the ACTION overran its 5 s window and pushed the last slot: the
            # machine arrived at delete_message 834 ms after its 600 ms budget had already closed, so
            # on a 36 job sweep delete recorded NOT EXERCISED on the base arm of every null-control
            # cell. A last slot with no slack is a slot that measures nothing, and the film's own end
            # is the one place an overrun has nowhere to go.
            ("thread_reopen", 48_100, 6_500),
            ("delete_message", 54_800, 2_500),
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
        except Exception as exc:  # noqa: BLE001
            return {"census_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}

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
        with self.open_window(name, "stream") as window:
            while (time.monotonic() - t0) * 1000 < until_ms:
                time.sleep(min(0.2, max(0.01, (until_ms - (time.monotonic() - t0) * 1000) / 1000)))
            window.note("waited_to_ms", until_ms)
            window.note("census", self._census())

    def _run_slot(self, slot: Slot, t0: float) -> dict:
        entry = get_action(slot.action)
        window_name = f"action:{slot.action}"
        if entry is None:
            return ActionResult(
                ran = False, reason = f"no action named {slot.action!r} is registered"
            ).row(slot.action, window_name, self.cell.cell_id)

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
            window.note("parity", self._parity())

        over_ms = ((time.monotonic() - t0) * 1000) - deadline_ms
        row = result.row(slot.action, window_name, self.cell.cell_id)
        row["window_ms"] = window.duration_ms
        row["census"] = window.notes.get("census")
        row["parity"] = window.notes.get("parity")
        # THE SCREENSHOT IS TAKEN OUTSIDE THE WINDOW, on purpose and not as a tidiness point.
        # The film runs on a wall clock and every slot has an absolute start, so an encode charged
        # to the measured window eats the gap before the next slot -- which is how an action comes
        # to report a MISSED SLOT on a contended runner. `--assert-liveness` counts those, so a
        # camera inside the window could turn a healthy run red and it would look like the harness
        # failing rather than like the instrument taxing itself. Out here it costs the gap, which
        # is what the gap is for. The DOM has moved on by a few milliseconds; for a picture that
        # is nothing, and for the digest, which was taken inside, it is not true at all.
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
