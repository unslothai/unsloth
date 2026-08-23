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

from ..fixture.corpus import PROVISIONAL_CHARS_PER_TOKEN, Corpus, RungPlan, plan_rung
from ..instruments import build as build_instruments
from ..instruments import import_errors
from ..pacer import Pacer, check_planned_streams
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
        t0 = self._press_send(page)

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
        # EVERY STREAM THE CELL SERVED, not just the last one. `last_stats()` describes whichever
        # turn finished last, so for a multi-turn cell it says nothing at all about the opening
        # reply -- and the opening reply is the one the rung is named for. Everything stays under
        # the `pacer` key because that subtree is exempt from the payload's bare-zero rule: a
        # pacer counter of 0 is a true reading, not a missing one.
        streams = self.pacer.all_stats()
        planned = self._planned_streams(cell, plan, row)
        row["pacer"] = {
            "last": self.pacer.last_stats(),
            "streams": streams,
            "check": check_planned_streams(streams, planned),
        }
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
        # A CELL THAT DID NOT STREAM WHAT IT PLANNED IS A FAILED CELL, for the same reason.
        #
        # The drain check above only asks whether the UI stopped running, and a later turn that
        # finishes satisfies it on behalf of an earlier one that did not. So an opening reply that
        # disconnected part way through left a complete-looking cell whose thread was thousands of
        # characters short of its rung, averaged into the A/B ratio against arms that streamed in
        # full. Raised here, after the check is on the row, so the reason and the per-turn evidence
        # ship with the failure and the cell is excluded by name rather than quietly included.
        check = row["pacer"]["check"]
        if check["checked"] and not check["ok"]:
            self.log(f"  the cell did not stream what it planned: {check['reason']}")
            raise RuntimeError(f"the cell did not stream what it planned: {check['reason']}")
        # AND THE SAME RULE FOR THE TURN THAT STREAMED BUT NEVER LANDED.
        #
        # The stream check asks whether the bytes went out. `send_turn` asserts something the pacer
        # cannot see: that the thread GREW. A send whose bytes were served in full but whose reply
        # never joined the thread leaves every later action, the peak census and the equivalence
        # mirror reading a thread one turn short, and the stream check alone would pass it.
        #
        # Scoped to `send_turn` DELIBERATELY, and this is the whole of the generalisation. An
        # action whose own assertion fails already has its own timing voided by
        # `scoring/from_payload._action_measure`, which returns `Measure.failed` for
        # `expect_ok is False` and says so. What that does NOT cover is an action whose failure
        # changed the workload the REST of the cell measured, and `send_turn` is the only one that
        # can: it is the only action whose outcome decides how much content the thread carries for
        # everything after it. A `select_text` that selected nothing or a `message_menu` that did
        # not open leaves the workload intact, and failing the cell for those would throw away a
        # whole cell's frame readings -- which really were taken, over the real thread -- for a
        # gesture that missed.
        missed_turns = [
            a
            for a in (row["actions"] or [])
            if a.get("action") == "send_turn" and a.get("ran") and a.get("expect_ok") is False
        ]
        if missed_turns:
            reason = "; ".join(
                f"follow-up turn {(a.get('expect') or {}).get('turn_index')} was sent but "
                f"{a.get('reason') or 'its own assertion failed'}"
                for a in missed_turns
            )
            self.log(f"  the cell did not stream what it planned: {reason}")
            raise RuntimeError(f"the cell did not stream what it planned: {reason}")
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
    def _planned_streams(cell: Cell, plan: RungPlan, row: dict) -> list[dict]:
        """The turns this cell MEANT to stream, each with its tag and its character count.

        The opening reply, plus one entry per `send_turn` that was ATTEMPTED -- taken from the
        recorded action rows and from the tag the action itself reports, so the naming rule lives
        in one place rather than two.

        THE TWO KINDS OF "DID NOT RUN" ARE NOT THE SAME, and treating them alike was a hole in the
        first version of this check. `ran = False` means the turn was never attempted: an exhausted
        queue at the small rungs, a slot missed on a slow machine. Nothing was loaded into the
        pacer, the cell simply has fewer turns, and demanding one would fail every small rung.
        `ran = True, expect_ok = False` is the opposite: the turn WAS attempted, `send_turn` loaded
        the pacer with it and pressed Enter, and no reply started. That is a planned turn that did
        not stream, and skipping it let a cell whose follow-up never arrived pass the check with
        `planned_turns: 1`, complete, and score 91.6 against a thread one turn short of its rung.
        """
        planned: list[dict] = []
        unit = plan.streamed_unit
        if unit is not None:
            planned.append(
                {
                    "tag": cell.cell_id,
                    "turn": "opening",
                    "chars": len(unit.reasoning) + len(unit.content),
                }
            )
        for action in row.get("actions") or []:
            if action.get("action") != "send_turn":
                continue
            if not action.get("ran"):
                continue
            expect = action.get("expect") or {}
            tag = expect.get("pacer_tag")
            if not tag:
                continue
            planned.append(
                {
                    "tag": tag,
                    "turn": f"follow_up{expect.get('turn_index')}",
                    "chars": int(expect.get("streamed_chars") or 0),
                }
            )
        return planned

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

    def _press_send(self, page) -> float:
        """Type a prompt and press send. Returns the driver monotonic time the film starts."""
        selector = 'textarea[aria-label="Message input"]'
        page.wait_for_selector(selector, timeout = 60_000)
        page.click(selector)
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


#: The sources that may SIZE a rung. `measure_chars_per_token` also has a last-resort
#: whitespace-and-punctuation estimate, which it labels itself as "off by tens of percent on dense
#: code" -- and this corpus is mostly dense code, where that estimate reads 6.7 against tiktoken's
#: 3.3. Sizing the ladder from it would not make the axis honest, it would move the error and, past
#: `MANIFEST_CHARS_PER_TOKEN`, make `plan_rung` refuse the whole run on any machine with no
#: tokeniser. A real tokeniser sizes the rungs; the estimate is still measured and still reported.
LADDER_RATIO_SOURCES = ("tiktoken/cl100k", "studio /api/inference/chat/count_tokens")


def _corpus_sample(corpus: Corpus, chars: int = 200_000) -> str:
    """A prefix of the frozen corpus, in the order a thread receives it."""
    out: list[str] = []
    size = 0
    for entry in corpus.manifest["units"]:
        unit = corpus.unit(entry["index"])
        out.append(unit.reasoning + unit.content)
        size += unit.chars
        if size >= chars:
            break
    return "".join(out)[:chars]


def ladder_chars_per_token(
    corpus: Corpus,
    base_url: str = "",
    auth: Optional[StudioAuth] = None,
    model_id: str = "",
    log: Callable[[str], None] = lambda _m: None,
) -> dict:
    """The ratio THE RUNGS ARE SIZED BY, measured on the corpus BEFORE anything is planned.

    A rung is named in tokens and the corpus is built in characters, so the ratio is what makes the
    two the same claim. `PROVISIONAL_CHARS_PER_TOKEN` is what a rung is planned with when nothing
    has been tokenised yet, and it was previously what EVERY production rung was planned with: the
    per-cell measurement runs after the thread is already seeded and its result is recorded and
    nothing else, so a cell labelled 1M tokens carried 4,000,000 characters of a corpus that
    tiktoken reads at 3.34 -- about 1.2M tokens, a fifth over its own label, on the axis the onset
    headline is quoted against.

    Measured once for the whole ladder rather than per rung: the per-cell number is taken from that
    rung's streamed unit alone, which is 6,000 characters of either reasoning or code and swings
    between 3.2 and 4.9 by which kind the rung happens to land on. The axis needs the corpus's
    ratio, not one turn's.
    """
    got = measure_chars_per_token(_corpus_sample(corpus), base_url, auth, model_id)
    measured = got.get("chars_per_token")
    source = got.get("source")
    if measured and measured > 0 and source in LADDER_RATIO_SOURCES:
        used, reason = float(measured), None
    else:
        used = PROVISIONAL_CHARS_PER_TOKEN
        reason = (
            f"no tokeniser answered (source {source!r}), so the rungs keep the provisional "
            f"{PROVISIONAL_CHARS_PER_TOKEN} rather than being sized from an estimate"
        )
    log(
        f"  ladder sized at {used} chars per token "
        f"(measured {measured} via {source}){'; ' + reason if reason else ''}"
    )
    return {
        "chars_per_token": used,
        "measured": measured,
        "source": source,
        "provisional": reason is not None,
        "reason": reason,
        "detail": got,
    }


def build_cells(
    rungs: list[str],
    corpus: Corpus,
    tier: str,
    session_id: str,
    instrument_level: int,
    reps: int = 1,
    chars_per_token: Optional[float] = None,
    base_url: str = "",
    auth: Optional[StudioAuth] = None,
    model_id: str = "",
    log: Callable[[str], None] = lambda _m: None,
) -> list[tuple[Cell, RungPlan]]:
    """The ladder's cells, sized by the MEASURED ratio unless a caller names one.

    `chars_per_token = None` means "measure the corpus first", which is what the production caller
    does. The ratio that sized the ladder travels on every cell's `meta` so a reader of the payload
    can see which one it was and whether a tokeniser answered.
    """
    if chars_per_token is None:
        ratio = ladder_chars_per_token(corpus, base_url, auth, model_id, log)
    else:
        ratio = {
            "chars_per_token": float(chars_per_token),
            "measured": None,
            "source": "caller",
            "provisional": False,
            "reason": None,
        }
    out: list[tuple[Cell, RungPlan]] = []
    for rung in rungs:
        plan = plan_rung(corpus, rung, ratio["chars_per_token"])
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
                meta = {"ladder_chars_per_token": ratio},
            )
            out.append((cell, plan))
    return out
