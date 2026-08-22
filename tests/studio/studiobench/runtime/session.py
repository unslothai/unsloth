# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One session: one browser, one Studio, one pacer, N cells.

A SESSION IS THE UNIT OF COMPARISON. Every slope, ratio and A/B pair must be read within one of
these, because cross-session drift on this app has been measured at 8% -- larger than most of the
effects worth arguing about. `Cell.session_id` carries the truth so the report layer can refuse a
comparison that spans two.

The measured window structure per cell, in order:

  1. seed the thread over REST, navigate, wait for the thread to mount
  2. ENFORCED IDLE WINDOW -- nothing streaming, no action, the page at rest -- and the timer clamp
     is calibrated inside it. This is the fix for the salvaged recorder's worst bug: calibrating
     from the first 60 ticks of a page that already has 31,637 elements standing measures the
     app's steady-state load, calls it the timer floor, subtracts it out of every window, and
     reports a saturated page as 0.2% busy.
  3. a resting census, so the growth axis has a denominator that was actually counted
  4. press send; the film starts; the scene's slots run against wall clock
  5. drain the stream, final census, teardown
"""

from __future__ import annotations

import contextlib
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..fixture.corpus import Corpus, RungPlan, plan_rung
from ..instruments import build as build_instruments
from ..instruments import import_errors
from ..pacer import Pacer
from ..scene import schedule as scene_schedule
from ..scene.actions import paint_floor_ms
from ..scene.schedule import SceneRunner
from .browser import cdp_counters, cdp_metrics, dump_diagnostics
from .lifecycle import StudioAuth
from .seeder import Seeder, compare_signatures, dom_signature, measure_chars_per_token
from .types import BenchContext, Cell, Paths, Recorder, Window, make_cell_id, new_session_id

# A 1x1 PNG, written to disk once per run so the image-upload action has a real file to attach.
# Generated rather than shipped as a binary asset: the artifact is a zipapp and a tester's
# machine has no fixture directory, and an action that reports `ran = False, no image path` at
# every rung is a hole in the suite that looks like a decision.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000d49444154789c6360000002000100ffff03000006"
    "0005570b8f0000000049454e44ae426082"
)


def ensure_probe_image(paths: Paths) -> Path:
    png = paths.out / "probe.png"
    if not png.exists():
        png.write_bytes(_PNG_1X1)
    return png


IDLE_CALIBRATION_MS = 1500
# The rung the seeded-vs-streamed equivalence is CHECKED at. 10K, because both paths are
# affordable there: streaming it takes under a minute at field cadence, and seeding it is instant.
# Below it the thread is one turn and there is nothing to seed; above it, streaming the whole
# thread is hours.
EQUIVALENCE_RUNG = "10K"
MOUNT_TIMEOUT_S = 180

# How long the composer may take to accept the click that starts the film. Not a performance
# budget: the point is that the cell survives and the number gets recorded. See `_press_send`.
# 90s because it has to be well clear of the worst real reading and still bounded, since a click
# that never lands is a different fact from a slow one.
COMPOSER_CLICK_TIMEOUT_S = 90
# Above this the log says so out loud, because a multi-second click is the user complaint itself
# and it should not be something you only find by reading the payload afterwards.
SLOW_COMPOSER_CLICK_MS = 1_000


class WindowInUse(RuntimeError):
    pass


@dataclass
class Session:
    ctx: BenchContext
    instruments: list = field(default_factory = list)
    _open: Optional[Window] = None
    cell: Optional[Cell] = None

    # ── windows ─────────────────────────────────────────────────────

    @contextlib.contextmanager
    def window(
        self,
        name: str,
        kind: str = "action",
    ):
        """Open a measurement window. Windows do NOT nest and do NOT overlap."""
        if self._open is not None:
            raise WindowInUse(
                f"cannot open window {name!r}: {self._open.name!r} is still open. Overlapping "
                "windows would charge the same work to both."
            )
        w = Window(name = name, kind = kind, cell = self.cell, t_open_ms = self._now_ms())
        self._open = w
        for inst in sorted(self.instruments, key = lambda i: i.name):
            self._safe(inst, "open", w)
        try:
            yield w
        finally:
            w.t_close_ms = self._now_ms()
            # REVERSE order on close, so an instrument that wrapped another's state unwinds after
            # the one it wrapped.
            for inst in sorted(self.instruments, key = lambda i: i.name, reverse = True):
                got = self._safe(inst, "close", w)
                if got is not None:
                    w.instruments[inst.name] = got
            self._open = None
            self.ctx.recorder.emit(w.row())

    def each_instrument(self, method: str, *args) -> dict:
        """Run one lifecycle hook on every instrument, and collect what each returned.

        OVER A SNAPSHOT, not the live list. `_safe` drops an instrument that raises, and removing
        from the list being iterated makes Python skip whichever instrument shifted into the freed
        index, so one broken optional instrument silently cost its neighbour's hook as well --
        `heap` failing took `input`, and with it the highest-weight metric in the table, while the
        cell still completed and reported.
        """
        out: dict = {}
        for inst in list(self.instruments):
            got = self._safe(inst, method, *args)
            if got is not None:
                out[inst.name] = got
        return out

    def _safe(self, inst, method: str, *args):
        """One broken instrument never costs the window."""
        fn = getattr(inst, method, None)
        if fn is None:
            return None
        try:
            return fn(*args)
        except Exception as exc:  # noqa: BLE001
            self.ctx.log(
                f"    instrument {inst.name}.{method} failed: " f"{type(exc).__name__}: {exc}"
            )
            if inst in self.instruments:
                self.instruments.remove(inst)
            return {"error": f"{type(exc).__name__}: {exc}", "disabled": True}

    def _now_ms(self) -> float:
        return self.ctx.recorder.now_ms()


@dataclass
class CellRunner:
    """Runs one cell end to end and always emits a `cell` row, completed or not."""

    session: Session
    pacer: Pacer
    seeder: Seeder
    corpus: Corpus
    base_url: str
    model_id: str
    tier: str
    paths: Paths
    log: Callable[[str], None] = print
    image_path: Optional[Path] = None
    cadence: str = "field"
    # Set once the 10K check fails, and it then labels every LARGER rung, which is the whole
    # point: those rungs are mostly seeded and their fidelity depends on this one answer.
    equivalence_failed: bool = False
    # Run the click attribution probe before the film starts. Off by default: it costs a handful
    # of seconds at small rungs and a great deal at large ones, and it makes the cell's timings
    # incomparable with a cell that did not run it. See `_click_attribution`.
    click_probe: bool = False

    def run(self, cell: Cell, plan: RungPlan) -> dict:
        s = self.session
        s.cell = cell
        rec = s.ctx.recorder
        page = s.ctx.page
        self.log(
            f"\n=== cell {cell.cell_id}: {plan.rung} "
            f"({plan.seeded_chars:,} seeded + {plan.streamed_chars:,} streamed chars)"
        )
        s.each_instrument("start_cell", cell)

        row: dict = {
            "row_type": "cell",
            "cell_id": cell.cell_id,
            "cell": cell.as_dict(),
            "completed": False,
            "fidelity": "unknown",
            "seeded_chars": plan.seeded_chars,
            "streamed_chars": plan.streamed_chars,
            "target_chars": plan.target_chars,
            "target_tokens": plan.target_tokens,
            "instruments": {},
        }
        # Cleared HERE, not beside the click that produces it, so the preservation in the `finally`
        # below cannot attach the PREVIOUS cell's attribution to a cell that died before its own
        # probe ran. A reading filed under the wrong cell id is worse than a missing one.
        self._click_attribution_result = None
        try:
            self._run_inner(cell, plan, row)
            row["completed"] = True
        except Exception as exc:  # noqa: BLE001
            row["failure"] = {
                "kind": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-3000:],
            }
            self.log(f"  cell FAILED: {type(exc).__name__}: {exc}")
            rec.failure(cell.cell_id, type(exc).__name__, {"message": str(exc)})
            with contextlib.suppress(Exception):
                dump_diagnostics(page, self.paths.logs, f"fail_{cell.cell_id}", self.log)
        finally:
            row["instruments"].update(s.each_instrument("end_cell", cell))
            # A cell that could not complete is a FIRST-CLASS RESULT with its failure mode and its
            # RSS at death, not a gap in the table.
            rss = row["instruments"].get("rss") or {}
            row["rss_at_death_mb"] = rss.get("rss_peak_mb") if not row["completed"] else None
            # AND SO IS THE PROBE THAT ALREADY RAN. `--click-probe` finishes inside `_press_send`,
            # and everything after it there -- the `page.fill`, the send button lookup and its
            # click -- still runs under the default 8s action timeout, which is exactly what a
            # large rung exceeds. Assigned only on the way out of `_run_inner`, the whole
            # attribution was dropped from the cell it was measured for, and unlike
            # `composer_click_ms` it has no window row of its own to survive in: nothing else in
            # the payload records it. So it is filed from here, on both paths.
            if self._click_attribution_result is not None:
                row["click_attribution"] = self._click_attribution_result
            rec.emit(row)
        return row

    # ── the cell ────────────────────────────────────────────────────

    def _run_inner(self, cell: Cell, plan: RungPlan, row: dict) -> None:
        s = self.session
        page = s.ctx.page
        rec = s.ctx.recorder

        seeded = self.seeder.seed(plan)
        row["thread_id"] = seeded.thread_id
        row["seed_seconds"] = round(seeded.seconds, 2)
        row["seeded_messages"] = seeded.messages

        page.goto(
            f"{self.base_url}/chat?thread={seeded.thread_id}",
            wait_until = "domcontentloaded",
            timeout = 120_000,
        )
        self._wait_for_thread(page, seeded.messages)

        # ── the enforced idle window ────────────────────────────────
        frames = next((i for i in s.instruments if i.name == "frames"), None)
        with s.window("idle:calibrate", kind = "idle") as w:
            clamp = (
                frames.calibrate(IDLE_CALIBRATION_MS)
                if frames
                else {"clampMs": None, "reason": "the frames instrument is not loaded"}
            )
            w.note("clamp", clamp)
        row["clamp"] = clamp
        if clamp.get("clampMs") is None:
            # NOT fatal, and NOT silently zero. Blocked time is a subtraction against this floor,
            # so without it busy_pct is null with the reason attached and every other column
            # stands.
            self.log(f"  timer clamp NOT established: {clamp.get('reason')}")
            rec.gate("timer_clamp", False, clamp)
        else:
            self.log(
                f"  timer clamp {clamp['clampMs']:.2f}ms " f"over {clamp.get('samples')} idle ticks"
            )
            rec.gate("timer_clamp", True, clamp)

        row["paint_floor_ms"] = paint_floor_ms(page)
        row["census_before"] = dom_signature(page)
        self.log(
            f"  at rest: {row['census_before']['messages']} messages, "
            f"{row['census_before']['elements']:,} elements, "
            f"{row['census_before']['highlight_spans']:,} highlight spans"
        )

        cpt = measure_chars_per_token(
            (plan.streamed_unit.reasoning + plan.streamed_unit.content)
            if plan.streamed_unit
            else "",
            self.base_url,
            self.seeder.auth,
            self.model_id,
        )
        row.update(
            {
                "chars_per_token": cpt.get("chars_per_token"),
                "chars_per_token_source": cpt.get("source"),
                "chars_per_token_detail": cpt,
            }
        )
        self.log(
            f"  chars per token: {cpt.get('chars_per_token')} "
            f"(measured via {cpt.get('source')})"
        )

        # ── the film ────────────────────────────────────────────────
        unit = plan.streamed_unit
        self.pacer.reset()
        self.pacer.load(
            unit.reasoning,
            unit.content,
            cadence = self.cadence,
            tag = cell.cell_id,
            model = self.model_id,
        )
        expected_ms = self.pacer.expected_duration_ms(unit.reasoning, unit.content, self.cadence)
        row["stream_expected_ms"] = expected_ms
        self.log(
            f"  streaming {len(unit.reasoning):,} reasoning + {len(unit.content):,} "
            f"content chars, cadence {self.cadence}, {expected_ms / 1000:.0f}s expected"
        )

        before_metrics = cdp_metrics(s.ctx.cdp)
        self._composer_click_ms = None
        # `click_attribution` is NOT filed here. It is filed in `run`'s `finally`, because a cell
        # that dies after the probe has to keep it: see there.
        t0 = self._press_send(page)
        # On the cell rather than in `actions`, because it happens before the first slot opens and
        # filing it as an action would put a reading outside the film into a list the scoring layer
        # pairs by slot. It is still a per-cell timing and grows with the rung like any other.
        row["composer_click_ms"] = self._composer_click_ms

        scene = scene_schedule.SCENES.get(self.tier, scene_schedule.QUICK)
        runner = SceneRunner(
            cell = cell,
            page = page,
            cdp = s.ctx.cdp,
            dom = None,
            recorder = rec,
            open_window = s.window,
            log = self.log,
            base_args = {
                "base_url": self.base_url,
                "thread_id": seeded.thread_id,
                "cell_id": cell.cell_id,
                "cadence": self.cadence,
                "image_path": str(self.image_path) if self.image_path else None,
                # The follow-up turns `send_turn` streams mid-film, and
                # the pacer it reloads to serve them.
                "_pacer": self.pacer,
                "_stream_queue": [
                    {"reasoning": u.reasoning, "content": u.content, "kind": u.kind}
                    for u in (plan.follow_up_units or [])
                ],
                # Shared and MUTABLE, so consecutive sends advance
                # through the queue. See send_turn.
                "_stream_cursor": {"i": 0},
                "_input_instrument": next((i for i in s.instruments if i.name == "input"), None),
            },
        )
        row["actions"] = runner.run(scene, t0)
        row["scene"] = scene.name
        row["scene_duration_ms"] = scene.duration_ms
        row["slots_missed"] = sum(1 for a in row["actions"] if a.get("slot_missed"))
        row["actions_not_run"] = sum(1 for a in row["actions"] if not a.get("ran"))
        row["expect_failures"] = sum(1 for a in row["actions"] if a.get("expect_ok") is False)

        with s.window("stream:drain", kind = "stream") as w:
            drained = self._drain_stream(page, expected_ms)
            w.note("drained", drained)
        row["stream"] = drained
        row["pacer"] = self.pacer.last_stats()
        # A REPLY THAT NEVER FINISHED IS A FAILED CELL, not a completed one with a note.
        #
        # `_drain_stream` reports rather than raises, and the value it reports was recorded here
        # and read by nothing: no gate, no report column, no `--assert-liveness` check and no
        # exit code. So the one state this tool exists to catch -- the app still generating three
        # times past its own cadence and 120 s beyond that, after the whole film has run -- came
        # back as `completed: true`, `ok` in the summary and exit 0, with its actions and its
        # frame windows scored and paired into the A/B ratio against an arm that DID finish.
        # That is the crash-beats-limp rule inverted: a build that cannot finish reads as a build
        # that had nothing to say.
        #
        # Raised AFTER the drain reading AND the pacer's own counters are on the row, so the two
        # facts that say WHY it never finished -- how long was waited, and how much the pacer
        # actually delivered -- ship in the same row as the failure. `CellRunner.run` catches
        # this, records `failure`, dumps the diagnostics and keeps the cell as a first-class
        # incomplete result; the rung then scores INCOMPLETE, which is exactly how this harness
        # reports a build that died at 500K. The censuses below are not taken, because a census of
        # a thread that is still growing describes nothing that was measured.
        if not drained.get("finished"):
            raise RuntimeError(
                f"the reply never finished: {drained.get('reason') or 'the run was still going'} "
                f"({drained.get('drain_ms')}ms waited, {drained.get('expected_ms')}ms expected)"
            )
        row["census_after"] = dom_signature(page)
        row["cdp"] = cdp_counters(before_metrics, cdp_metrics(s.ctx.cdp))

        # THE PEAK, over every window's census, not the state at the end of the film.
        #
        # The film ENDS with thread_reopen and delete_message, so an end-of-cell census reports
        # the thread the benchmark has just deleted. The first working run recorded 0 assistant
        # messages and 0 characters against a reply the pacer's own log proved it had delivered in
        # full: 150 chunks, 3,581 characters, at exactly the 73ms cadence. The occupancy that
        # every per-action cost has to be read against is the peak, and it is now recovered from
        # the per-window censuses rather than from a single reading taken at the worst moment.
        censuses = [w.get("census") for w in row["actions"] if isinstance(w.get("census"), dict)]
        censuses = [c for c in censuses if c.get("elements")]
        peak = max(censuses, key = lambda c: c.get("elements", 0)) if censuses else {}
        row["census_peak"] = peak
        row["census_peak_attempted"] = bool(censuses)

        census = peak or row["census_after"]
        spans = census.get("highlight_spans") or 0
        chars = census.get("assistant_chars")
        if chars is None:
            chars = page.evaluate("() => window.__sb.dom.assistantChars()")
        row["assistant_chars_in_dom"] = chars
        # The span density the fixture ACHIEVED, measured in the DOM rather than assumed. The
        # field capture ran 5.6 characters per span; a corpus that lands far from that is not
        # standing in for the same highlighter load per character, and the report should say so
        # rather than quietly compare two different workloads.
        row["chars_per_span"] = round(chars / spans, 2) if spans else None
        row["chars_per_span_target"] = 5.6
        self.log(
            f"  after: {census['elements']:,} elements, {spans:,} spans, "
            f"{chars:,} assistant chars -> {row['chars_per_span']} chars/span"
        )

        row["fidelity"] = "streamed_and_seeded" if plan.seeded_units else "streamed_only"

        # ── the seeded-vs-streamed equivalence check ────────────────
        if cell.rung == EQUIVALENCE_RUNG and plan.streamed_unit is not None:
            eq = self._check_equivalence(plan, row)
            row["equivalence"] = eq
            rec.gate("seeded_equals_streamed", bool(eq.get("equivalent")), eq)
            if not eq.get("equivalent"):
                # A FINDING, printed, not a bug to hide. It says exactly which of this tool's
                # numbers are about the streaming path and which are about a thread that was put
                # there, and it is the reason the higher rungs carry a fidelity label at all.
                self.log(
                    "  SEEDED IS NOT EQUIVALENT TO STREAMED at the 10K rung. Rungs above it "
                    "are labelled fidelity: seeded_only."
                )
                for key, field in (eq.get("fields") or {}).items():
                    if field.get("within_tolerance") is False:
                        self.log(
                            f"    {key}: streamed {field['streamed']} vs seeded "
                            f"{field['seeded']} ({field['drift']:.1%} drift)"
                        )
                self.equivalence_failed = True
            else:
                self.log(
                    "  seeded and streamed agree on CONTENT at the 10K rung within "
                    f"{eq['tolerance']:.0%}"
                )
                # Passing the content gate is not the same as the two threads being identical,
                # and the difference is large enough that leaving it unsaid would mislead: a
                # seeded rung carries the same rendered content and materially less mounted DOM.
                fields = eq.get("fields") or {}
                for key in ("reasoning_spans", "highlight_spans", "assistant_chars"):
                    field = fields.get(key) or {}
                    if field.get("drift"):
                        self.log(
                            f"    but {key}: streamed {field['streamed']} vs seeded "
                            f"{field['seeded']} ({field['drift']:.1%}) -- a collapsed "
                            "reasoning pane mounts its children only when the text was "
                            "streamed into it"
                        )
        if self.equivalence_failed and plan.seeded_units:
            row["fidelity"] = "seeded_only"

    @staticmethod
    def _streamed_follow_ups(plan: RungPlan, row: dict) -> list:
        """The follow-up units that actually reached the thread during the film.

        EVERY TURN THAT STREAMED, not just the opening one. From 10K upwards the plan carries two
        follow-ups and the scene streams both through `send_turn` before the peak census is taken,
        so a mirror seeded from the prefix plus the opening unit is two assistant turns short of
        the thread it is being compared against. `assistant_messages` is a GATED key, and two
        missing turns out of six is 33% drift against a 2% tolerance: the check then failed on
        every healthy cell and labelled every larger rung `seeded_only` for a difference the
        mirror had introduced itself.

        Counted from the recorded action rows rather than from the plan, because a `send_turn`
        that did not run (an exhausted queue at the small rungs, a slot missed on a slow machine)
        put nothing in the thread and must not be seeded into the mirror either.
        """
        streamed = 0
        for action in row.get("actions") or []:
            if action.get("action") != "send_turn":
                continue
            if action.get("ran") and action.get("expect_ok") is not False:
                streamed += 1
        return list(plan.follow_up_units or [])[:streamed]

    def _check_equivalence(self, plan: RungPlan, row: dict) -> dict:
        """Build the SAME content as a fully seeded thread and compare what the app made of it.

        The streamed reply has just been measured. This seeds a second thread containing every
        unit including that one -- so the two threads carry identical text -- loads it, and
        compares the DOM the app built. Two paths, one corpus, one comparison.
        """
        s = self.session
        page = s.ctx.page
        streamed = row.get("census_peak") or row.get("census_after") or {}
        follow_ups = self._streamed_follow_ups(plan, row)
        try:
            all_units = list(plan.seeded_units) + [plan.streamed_unit] + follow_ups
            mirror = RungPlan(
                rung = plan.rung,
                target_tokens = plan.target_tokens,
                target_chars = plan.target_chars,
                seeded_units = all_units,
                streamed_unit = None,
            )
            seeded_thread = self.seeder.seed(mirror)
            page.goto(
                f"{self.base_url}/chat?thread={seeded_thread.thread_id}",
                wait_until = "domcontentloaded",
                timeout = 120_000,
            )
            self._wait_for_thread(page, seeded_thread.messages)
            # Let the highlighter finish, or the span count is a race rather than a comparison.
            page.wait_for_timeout(4000)
            seeded_sig = dom_signature(page)
        except Exception as exc:  # noqa: BLE001
            return {
                "equivalent": None,
                "checked_attempted": False,
                "reason": f"the mirror thread could not be built: " f"{type(exc).__name__}: {exc}",
            }
        out = compare_signatures(streamed, seeded_sig)
        out["streamed_census"] = streamed
        out["seeded_census"] = seeded_sig
        # What the mirror was built from, so a drift can be read against the corpus it compared
        # rather than against an assumption about which turns were in the thread.
        out["mirrored_follow_ups"] = len(follow_ups)
        out["planned_follow_ups"] = len(plan.follow_up_units or [])
        return out

    def _wait_for_thread(self, page, expected_messages: int) -> None:
        if expected_messages <= 0:
            page.wait_for_selector('textarea[aria-label="Message input"]', timeout = 60_000)
            return
        deadline = time.monotonic() + MOUNT_TIMEOUT_S
        last = -1
        while time.monotonic() < deadline:
            got = page.evaluate("() => window.__sb.dom.messageCount()")
            if got >= expected_messages:
                return
            if got != last:
                last = got
                self.log(f"  mounting: {got}/{expected_messages} messages")
            page.wait_for_timeout(500)
        raise TimeoutError(
            f"the thread mounted {last} of {expected_messages} messages in " f"{MOUNT_TIMEOUT_S}s"
        )

    def _click_attribution(self, page, selector: str) -> dict:
        """Split the composer click into what a user pays and what the DRIVER pays.

        `page.click` is not a click. Before dispatching it resolves the selector, waits for the
        element to be visible, enabled and stable, scrolls it into view, then hit-tests the point
        with `elementsFromPoint` and checks that what is under the cursor is what was asked for,
        retrying until it agrees. Every one of those steps is O(DOM), and a human does none of
        them. The Chromium CPU profile of that window at 500K is dominated by Playwright's own
        injected script, so a number taken from `page.click` cannot be reported as user cost
        without first showing how much of it is the driver.

        Four paths, ordered by how much machinery each one skips:

          click     `page.click`         full actionability, which is what the ladder recorded
          mouse     `page.mouse.click`   real input at a point: the browser hit-tests, the driver
                                         does not resolve or re-check anything
          dispatch  `dispatch_event`     a synthesised event, no hit test at all
          focus     `el.focus()`         no event and no hit test, just focus and its handlers

        And one that involves no click whatsoever:

          hover     move the cursor from a corner into the transcript, flipping `:hover` down the
                    whole hover chain. If THIS costs seconds then focus was never the variable and
                    the cost is style invalidation from a pseudo-class flip, which is worse news
                    than a slow click: a user pays it on every mouse movement over the thread.

        Each is preceded by a blur and a settle so no repetition inherits the previous one's state.
        """

        def blur() -> None:
            page.evaluate("() => document.activeElement && document.activeElement.blur()")
            page.wait_for_timeout(250)

        def settled(fn) -> float:
            """Time `fn` AND the wait for the main thread to be free again.

            Timing the call alone measures the wrong thing, and differently wrong per engine.
            `page.mouse.click` hands an input event to the browser over the debug protocol and
            returns; whether the acknowledgement waits for the renderer to process it is an
            implementation detail of each engine's Playwright backend, not a property of the app.
            Read that way, Chromium came back at 3 ms for both 100K and 500K, which does not mean
            the work was free, only that the ack did not wait for it.

            So every path is followed by a round trip into the page. `page.evaluate` CANNOT return
            while the main thread is blocked, and `offsetHeight` forces any pending style and
            layout to be resolved rather than deferred. The reading is then "how long until the
            page could serve me again", which is the thing a user actually experiences and is
            comparable across engines.
            """
            started = time.monotonic()
            fn()
            page.evaluate("() => document.body.offsetHeight")
            return (time.monotonic() - started) * 1000.0

        # FIRST, before anything else touches the page: the same trivial operation N times.
        #
        # This exists because the biggest number in the whole ladder is one I could not attribute.
        # The first thing touched after a large thread mounts costs 11 to 24 seconds, and in every
        # probe the cost vanished, because whatever ran first absorbed it and by the time the
        # interesting path ran the page was warm. Measuring the decay directly is the way out: if
        # the first reading is seconds and the rest are milliseconds, the cost is ONE TIME and its
        # size is the first reading minus the steady state. If they are all expensive, it is a
        # per-interaction cost and the ladder's number was never a one-off.
        #
        # A no-op body on purpose. Every reading is the same work, so any difference between them
        # is the page's state and not the operation.
        decay = [settled(lambda: None) for _ in range(5)]
        out: dict[str, Any] = {
            # The harness layer's attestation, and it is load-bearing rather than decorative.
            # `scoring/schema._walk_for_bare_zeros` rejects any bare zero that is not covered by a
            # sibling `*_attempted` flag, and this block has legitimate zeros: a thread with no
            # mounted code blocks records `code_token_spans: 0`, and `blur_inpage_ms` and
            # `forced_layout_ms` come from `performance.now()`, which Chromium coarsens to 100 us
            # outside a cross-origin-isolated context, so an operation shorter than that reads
            # exactly 0. Without this flag `--report` refuses the whole payload of a completed
            # probe run.
            "click_attribution_attempted": True,
            "first_touch_ms": decay[0],
            "settled_touch_ms": min(decay[1:]),
            "touch_decay_ms": [round(v, 1) for v in decay],
        }
        # The blur, timed, and timed again from INSIDE the page.
        #
        # The decay series above says the first touch after mount costs 10.6 ms at 500K, so there
        # is no expensive first interaction. Yet the reading taken right after the first blur came
        # back at 10,052 ms and 10,017 ms in two of three runs. Two things about that: it sits
        # between the cheap decay series and the cheap readings that follow, so the blur is what it
        # attaches to; and 10.0 seconds to three digits is the shape of a TIMEOUT, not of work that
        # happens to take ten seconds.
        #
        # `blur_inpage_ms` decides which. It runs the same blur and the same forced layout inside a
        # single evaluate and times them with `performance.now()`, so the protocol is excluded. If
        # the page reports ten seconds, it is real work. If the page reports a millisecond while
        # the outer reading is ten seconds, the ten seconds is the driver or the transport and not
        # the app, and must never be quoted as a user cost.
        out["blur_outer_ms"] = settled(
            lambda: page.evaluate("() => document.activeElement && document.activeElement.blur()")
        )
        out["blur_inpage_ms"] = page.evaluate(
            "() => { const t = performance.now();"
            " document.activeElement && document.activeElement.blur();"
            " void document.body.offsetHeight;"
            " return performance.now() - t; }"
        )
        blur()
        out["roundtrip_ms"] = settled(lambda: None)
        box = page.query_selector(selector).bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        blur()
        out["click_ms"] = settled(
            lambda: page.click(selector, timeout = COMPOSER_CLICK_TIMEOUT_S * 1000)
        )
        blur()
        out["mouse_ms"] = settled(lambda: page.mouse.click(x, y))
        blur()
        out["dispatch_ms"] = settled(lambda: page.dispatch_event(selector, "click"))
        blur()
        out["focus_ms"] = settled(lambda: page.eval_on_selector(selector, "e => { e.focus(); }"))
        blur()
        page.mouse.move(2, 2)
        page.wait_for_timeout(250)
        out["hover_thread_ms"] = settled(lambda: page.mouse.move(x, 300))
        # The reading that decides what `roundtrip_ms` meant. If the first forced layout of a huge
        # thread is expensive and every later one is cheap, this comes back near zero and the cost
        # is a ONE-TIME layout of the mounted thread, not a per-interaction cost. If it is
        # expensive again, every interaction pays it and the story is the opposite.
        blur()
        out["roundtrip_again_ms"] = settled(lambda: None)
        # Measured INSIDE the page, so the protocol round trip is not in the number. `offsetHeight`
        # is read after a write that dirties layout, so it cannot be served from a clean tree.
        out["forced_layout_ms"] = page.evaluate(
            "() => { const t = performance.now();"
            " document.body.style.minHeight = (1 + Math.random()) + 'px';"
            " void document.body.offsetHeight;"
            " document.body.style.minHeight = '';"
            " void document.body.offsetHeight;"
            " return performance.now() - t; }"
        )
        out["code_token_spans"] = page.evaluate(
            "() => document.querySelectorAll('[data-streamdown=\"code-block\"] code > span').length"
        )
        self.log(
            "  click attribution: "
            + ", ".join(
                f"{k.replace('_ms', '')}={v:,.0f}ms"
                for k, v in out.items()
                if k.endswith("_ms") and isinstance(v, (int, float))
            )
            + f", code token spans={out['code_token_spans']:,}"
            + f"\n  touch decay: {out['touch_decay_ms']}"
        )
        return out

    def _press_send(self, page) -> float:
        """Type a prompt and press send. Returns the driver monotonic time the film starts.

        THE CELL MUST SURVIVE THIS, and it took losing a whole rung to notice it did not.

        This runs before the film starts, so it was written as setup and inherited the default
        8s action timeout. At 500K `page.click` exceeds it and the exception killed the cell
        before a single slot opened, three times out of three across two runs, so the ladder had
        NO data at 500K at all. `COMPOSER_CLICK_TIMEOUT_S` fixes that: the cell survives, the film
        runs, and the cost is recorded whatever it comes to. Still bounded, because a click that
        never lands is a different fact from a slow one.

        `composer_click_ms` IS NOT WHAT A USER PAYS, and must never be quoted as though it were.
        I made exactly that mistake and published it. `page.click` resolves the selector, waits
        for visible, enabled and stable, scrolls into view, hit-tests the point with
        `elementsFromPoint` and re-checks that the element under the cursor is the one asked for,
        retrying until it agrees. All of that is O(DOM) and a human does none of it. Measured at
        500K on WebKit with `--click-probe`: `page.click` 11,036 ms, a real mouse click at the
        same point 573 ms. About 95% of the number is the driver.

        So this reading is a HARNESS health number: it says whether the cell can start. For what
        the user pays, run `--click-probe` and read `mouse_ms` and `focus_ms`.
        """
        selector = 'textarea[aria-label="Message input"]'
        page.wait_for_selector(selector, timeout = 60_000)
        if self.click_probe:
            self._click_attribution_result = self._click_attribution(page, selector)
        # In a window, so every instrument covers it. At 500K this single click is the largest
        # cost in the whole run by an order of magnitude, and it was the one moment the tool could
        # not see inside: no window, so no frame recorder, no CPU profile, no RSS delta. A cost
        # that big being outside every instrument is how it stayed unattributed.
        #
        # `setup`, NOT `action`. The scoring layer pools every non-excluded window of the cell into
        # `max_frame_ms`, `jank_index` and `time_in_jank_pct`, and this window is mostly Playwright's
        # own injected actionability script running ON THE PAGE'S MAIN THREAD -- it blocks frames
        # exactly as app work would, so the frame recorder cannot tell them apart. Filed as an
        # `action` it would put an 11 s driver stall into three weighted headline metrics whose
        # `max_frame_ms` anchor calls 2,000 ms the worst case, on EVERY run including the ones that
        # never asked for --click-probe. It is measured, reported and kept out of the film.
        with self.session.window("setup:composer_click", kind = "setup"):
            # Timed INSIDE the window, like `Window.duration_ms` itself: the session opens every
            # instrument before this block and closes them after it, and at instrument level 1-3
            # those hooks stop a CPU profile, collect coverage and write and analyse a trace.
            # Timed around the `with`, `composer_click_ms` would grow with the instrument level
            # and stop being the click.
            clicked_at = time.monotonic()
            page.click(selector, timeout = COMPOSER_CLICK_TIMEOUT_S * 1000)
            self._composer_click_ms = (time.monotonic() - clicked_at) * 1000.0
        if self._composer_click_ms > SLOW_COMPOSER_CLICK_MS:
            self.log(
                f"  page.click on the composer took {self._composer_click_ms / 1000:.1f}s. "
                f"MOST OF THAT IS THE DRIVER, not the app: run --click-probe to split it."
            )
        page.fill(selector, "continue")
        page.wait_for_timeout(150)
        send = page.query_selector('button[aria-label="Send message"]')
        if send is None:
            raise RuntimeError("the send button is not on the page, so no reply can be started")
        t0 = time.monotonic()
        send.click()
        # The composer must be EMPTY for the rest of the film, or the Stop control is replaced by
        # a Queue control and the stop action presses the wrong button. Sending clears it, but the
        # keystroke action refills it, which is why the stop action clears it again itself.
        return t0

    def _drain_stream(self, page, expected_ms: float) -> dict:
        """Wait for the run to end, or say plainly that it did not."""
        # Generous: the whole point of deficit scheduling is that the stream's own duration is
        # machine-independent, so anything much past it is the RENDERER failing to keep up, which
        # is a finding rather than a timeout to paper over.
        deadline = time.monotonic() + (expected_ms / 1000) * 3 + 120
        started = time.monotonic()
        while time.monotonic() < deadline:
            if not page.evaluate("() => window.__sb.dom.isRunning()"):
                return {
                    "finished": True,
                    "drain_ms": round((time.monotonic() - started) * 1000, 1),
                    "expected_ms": expected_ms,
                }
            page.wait_for_timeout(250)
        return {
            "finished": False,
            "drain_ms": round((time.monotonic() - started) * 1000, 1),
            "expected_ms": expected_ms,
            "reason": "the run was still going three times past its own cadence",
        }


def make_context(
    browser_bundle,
    base_url: str,
    tier: str,
    instrument_level: int,
    paths: Paths,
    log: Callable[[str], None],
    browser_procs: Optional[list] = None,
) -> tuple[BenchContext, Session]:
    session_id = new_session_id()
    recorder = Recorder(paths.payload_jsonl, session_id)
    ctx = BenchContext(
        browser = browser_bundle.browser,
        context = browser_bundle.context,
        page = browser_bundle.page,
        cdp = browser_bundle.cdp,
        base_url = base_url,
        session_id = session_id,
        tier = tier,
        instrument_level = instrument_level,
        paths = paths,
        recorder = recorder,
        log = log,
        browser_procs = browser_procs or [],
    )
    instruments = build_instruments(instrument_level)
    errors = import_errors()
    for name, err in errors.items():
        recorder.gate(f"instrument_unavailable:{name}", False, {"error": err})
        log(f"  instrument {name} unavailable: {err}")
    for inst in instruments:
        try:
            inst.attach(ctx)
        except Exception as exc:  # noqa: BLE001
            log(f"  instrument {inst.name} failed to attach: {exc}")
    log(
        f"  instruments at level {instrument_level}: "
        f"{', '.join(i.name for i in instruments) or 'none'}"
    )
    return ctx, Session(ctx = ctx, instruments = instruments)


def build_cells(
    rungs: list[str],
    corpus: Corpus,
    tier: str,
    session_id: str,
    instrument_level: int,
    reps: int = 1,
    chars_per_token: float = 4.0,
    stream_tail_chars: int | None = None,
    corpus_dollars: bool = False,
) -> list[tuple[Cell, RungPlan]]:
    out: list[tuple[Cell, RungPlan]] = []
    for rung in rungs:
        plan = plan_rung(
            corpus,
            rung,
            chars_per_token,
            stream_tail_chars = stream_tail_chars,
            dollars = corpus_dollars,
        )
        for rep in range(reps):
            cell = Cell(
                cell_id = make_cell_id(rung, "A0", rep),
                rung = rung,
                rung_tokens = plan.target_tokens,
                arm = "A0",
                rep = rep,
                tier = tier,
                transport = "provider",
                instrument_level = instrument_level,
                seed = corpus.seed,
                corpus_hash = corpus.corpus_hash,
                session_id = session_id,
            )
            out.append((cell, plan))
    return out
