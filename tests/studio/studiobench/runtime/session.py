# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One session: one browser, one Unsloth, one pacer, N cells.

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
from collections.abc import Mapping
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
from .readiness import (
    COVERAGE_STATES_SCOREABLE,
    MODE_FULL,
    MODE_WINDOWED,
    MODES,
    Readiness,
    ThreadNotReady,
    probe_thread_completeness,
    wait_for_thread_ready,
)
from .seeder import Seeder, SeededThread, compare_signatures, dom_signature, measure_chars_per_token
from .types import BenchContext, Cell, Paths, Recorder, Window, make_cell_id, new_session_id

# A 1x1 PNG, written once per run so image-upload has a real file. Generated rather than shipped:
# the artifact is a zipapp with no fixture directory, and an action reporting `ran = False` at
# every rung is a hole that looks like a decision.
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
# affordable there; below it there is nothing to seed, above it streaming the thread is hours.
EQUIVALENCE_RUNG = "10K"
MOUNT_TIMEOUT_S = 180

#: How much of the streaming phase the thread must stay pinned for `follows_the_stream` to pass.
#: 0.95 rather than 1.0, since the sampler ticks four times a second and a legitimate pin can land
#: a tick late. Paired with `ever_fell_behind`, which is absolute: being yanked back is the other
#: half of the intent contract being broken.
FOLLOW_PINNED_MIN = 0.95

#: How much of the STREAMING TIME the attached phases must cover before `pinned_fraction` may
#: stand as a verdict. It is computed over attached samples only, and with `detached` latching on
#: the first deliberate scroll the shipped film produced a verdict from the first ~3s -- 13%
#: coverage, reported as 100% pinned. The latch is fixed; this makes the coverage a condition.
# That scroll is 1.5s into an 18s opening stream.
FOLLOW_MIN_STREAM_COVERAGE = 0.50


def follow_verdict(follow: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """The `follows_the_stream` verdict, and the coverage fields recorded whatever it says.

    A module-level function rather than four lines inside the cell runner because the SPLIT it
    encodes is the whole point and has to be testable without a browser: which shortfalls are a
    reading of the BUILD and which are a property of the FILM.

    FATAL, because they describe how the arm behaved while it was attached and can genuinely
    differ between two builds:

      `pinned_fraction`   below `FOLLOW_PINNED_MIN`, or absent while the sampler was present
      `ever_fell_behind`  absolute; one drift past tolerance is a failure however fast the recovery

    NOT MEASURED, because it is set by the SCENE SCHEDULE and not by the build under test:

      `attached_fraction_of_stream` below `FOLLOW_MIN_STREAM_COVERAGE`, AND ONLY WHEN THE ARM
                                    re-attached at least once, so the shortfall is the schedule's
                                    and not this build's refusal to come back

    The film scrolls away twice inside an ~18s opening stream and the app then correctly declines
    to yank the reader back, so roughly half the streaming time is detached BY CONSTRUCTION.
    Measured over 32 cells it is 0.481 +/- 0.009, range 0.4625 to 0.5063 -- so a floor of 0.50 sat
    above the mean of the quantity it was gating and made the verdict a coin flip. It refused 32 of
    32 pairs with `TOO LITTLE COMPARED`, exit 3, and the NULL CONTROL -- the same commit on both
    arms -- refused identically. A gate that fails its own null is not measuring what it names. It
    was blinding the null audit too, which could not establish its own noise floor: all 16 actions
    came back `undetermined`.

    Both arms run the same film, so the shortfall is symmetric by construction. It cannot
    discriminate between them; it can only void the run. It still qualifies an ABSOLUTE quote -- a
    thread that spent half the stream detached did render less -- which is why the coverage is
    RECORDED AS A NUMBER rather than dropped, and why this stays a failed gate row a reader has to
    step over. What it may no longer do is take the cell out of a COMPARISON, where the confound is
    common to both sides and cancels. Raising the constant instead would be the same trap one turn
    later: it would need re-deriving every time the film's scroll schedule moves.

    `pinned_ok and not fell_behind` in the `stream_coverage_unmeasured` conjunction is
    LOAD-BEARING. Without it a genuine follow failure on a large rung, where coverage is low
    anyway, would ride out on this allowance -- the same defect in the opposite direction.
    """

    pinned = follow.get("pinned_fraction")
    coverage = follow.get("attached_fraction_of_stream")
    pinned_ok = pinned is not None and pinned >= FOLLOW_PINNED_MIN
    fell_behind = bool(follow.get("ever_fell_behind"))
    coverage_short = coverage is None or coverage < FOLLOW_MIN_STREAM_COVERAGE
    # THE DETACHMENT HAS TO BE THE SCHEDULE'S, AND ONLY A RE-ATTACHMENT PROVES IT. The waiver rests
    # on `attached_fraction_of_stream` being set by the film and identical on both arms, but half of
    # it is the BUILD'S: `scene/dom.js` clears `detached` only when a run begun after the gesture is
    # observed at the bottom, so an arm that stops re-pinning never re-attaches and every sample
    # lands in the detached branch, leaving `pinned_fraction: 1.0` intact while coverage collapses.
    # A reply that leaves the viewport stops costing anything to paint, so the cell is CHEAPER for
    # the defect and `readings_by_arm` would have admitted it against a healthy partner.
    # `reattachments` is the sampler's own record of the app coming back.
    reattached = bool(follow.get("reattachments"))
    # THE COVERAGE AS A NUMBER, WHATEVER THE VERDICT, next to the floor it is read against: it was
    # only ever visible as a pass or a fail, which is why it took a campaign to notice the floor sat
    # above the film's ceiling.
    # Every reader saw "FAILED its stream-follow gate"; nobody saw the 0.481.
    recorded: dict[str, Any] = {
        "stream_coverage": coverage,
        "stream_coverage_floor": FOLLOW_MIN_STREAM_COVERAGE,
        "stream_coverage_unmeasured": bool(
            coverage_short and pinned_ok and not fell_behind and reattached
        ),
    }
    if coverage_short:
        recorded["stream_coverage_reason"] = (
            "the thread was attached for "
            + ("an unknown share" if coverage is None else f"{coverage:.1%}")
            + f" of the streaming time, under the {FOLLOW_MIN_STREAM_COVERAGE:.0%} floor; "
            "the follow verdict is NOT MEASURED for this cell, not failed"
        )
    return bool(pinned_ok and not coverage_short and not fell_behind), recorded


# How long the composer may take to accept the click that starts the film. Not a performance
# budget: the point is that the cell survives and the number gets recorded. 90s, well clear of
# the worst real reading and still bounded.
COMPOSER_CLICK_TIMEOUT_S = 90
# Above this the log says so out loud, because a multi-second click is the user complaint itself.
SLOW_COMPOSER_CLICK_MS = 1_000


class WindowInUse(RuntimeError):
    pass


def record_completeness_gate(recorder: Recorder, cell: Cell, completeness: dict) -> bool:
    """Write the completeness verdict as a gate row AGAINST THE CELL THAT PRODUCED IT.

    WHY THIS IS NOT `recorder.gate(...)`. `Recorder.gate` writes `{row_type, name, passed,
    detail}` and no cell_id, and `report/payload.py::excluded_from_rows` reads a failed gate as
    `row.get("cell_id") or "run"`. So a windowed cell that had really lost messages was excluded
    under the synthetic cell id "run": the report could say a self-check failed somewhere in the
    run and could not say which arm or which rung lost them, which is the one thing this probe
    exists to find out. `cell_id` is `r{rung}.{arm}.rep{rep}`, so attributing the row names all
    three. `Recorder.failure` already takes a cell_id for the same reason.

    THE VERDICT ITSELF is the head marker AND the ordinal coverage, and coverage is three-valued.
    `False` is a finding. `None` is two different answers wearing one value, and they are told
    apart by `ordinal_coverage_state`:

      not_applicable  no row published an `aria-posinset` for the traversal to count. A fully
                      mounted arm publishes none anywhere -- the shipped build publishes none --
                      so the question does not arise, and failing on it would fail the shipped
                      build's own completeness gate on every cell.
      unmeasured      the question arises and the sweep could not answer it: the gesture stopped
                      short of the top, or its consecutive stops did not overlap so the middle of
                      the thread was never in view.

    Only the first is a pass. A store that retains the first page and the last one and has lost
    everything between them is the exact arm this probe was written to catch, and accepting
    `unmeasured` let it back in through the unknown state: the head marker arrives, the coverage
    sweep never looks, and the cell stays scoreable. "We could not tell" must not be recorded as
    "it was fine". The remedy for a coarse sweep is a smaller `step_px`, not a pass.

    A completeness dict carrying no state at all is treated the same way as `unmeasured`, because
    an undifferentiated `None` is precisely the ambiguity above and resolving it in favour of a
    pass is the defect.
    """
    coverage = completeness.get("ordinal_coverage_complete")
    state = completeness.get("ordinal_coverage_state")
    passed = (
        bool(completeness.get("head_reached"))
        and coverage is not False
        and state in COVERAGE_STATES_SCOREABLE
    )
    recorder.emit(
        {
            "row_type": "gate",
            "name": "thread_complete",
            "passed": passed,
            "detail": completeness,
            "cell_id": cell.cell_id,
        }
    )
    return passed


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
            # REVERSE order on close, so an instrument that wrapped another's state unwinds after the one it wrapped.
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
    #: Record the NORMALISED signature text beside each digest. Off by default: the text is megabytes
    #: per capture, and `sweep/parity_null_control.py --hunt` is the only consumer -- it needs it
    #: because a digest pair can say THAT two DOMs differ, never which bytes moved.
    parity_raw: bool = False
    #: Directory for the per-action viewport PNGs, or None. Set only when the caller intends
    #: before/after evidence; the encode is cheap but the files are not free.
    parity_shots: Optional[str] = None
    #: Which ARM this runner drives, burned into every filename: both arms share a fixture, a password
    #: and a film, so the image itself carries nothing that identifies the side.
    arm_label: str = "base"
    # Set once the 10K check fails, and it then labels every LARGER rung: those rungs are mostly
    # seeded and their fidelity depends on this one answer.
    equivalence_failed: bool = False
    # Run the click attribution probe before the film. Off by default: it costs a great deal at large
    # rungs and makes the cell's timings incomparable with a cell that did not run it.
    click_probe: bool = False
    # WHICH READINESS GATE. `full` is the default every normal arm runs; an arm that mounts a WINDOW
    # on purpose sets `windowed` and is held to a different set of conditions, none of them weaker.
    # See runtime/readiness.py.
    # Per TARGET, so a base-versus-virtualised A/B keeps the base arm on its strict gate.
    readiness_mode: str = MODE_FULL
    # Scroll a windowed thread to its top once per cell, before the measured window, to prove the arm
    # still holds the head of the conversation. Costs a full traversal, so it is off for `full` and
    # on by default for `windowed`, where it is the only check that separates a virtualised thread
    # from one that has lost most of its messages.
    completeness_probe: Optional[bool] = None

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
        # Cleared HERE, not beside the click that produces it, so the preservation in the `finally` below
        # cannot attach the PREVIOUS cell's attribution to a cell that died before its own probe ran.
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
            if isinstance(exc, ThreadNotReady):
                # WHICH CONDITION, not just that it timed out: the old message named a count and left the reader
                # to guess whether the app was slow, the thread short, or the arm windowed on purpose.
                row["failure"]["readiness"] = exc.detail
                row["readiness"] = exc.detail
                rec.gate(
                    f"thread_ready:{self.readiness_mode}",
                    False,
                    exc.detail,
                    cell_id = cell.cell_id,
                )
            self.log(f"  cell FAILED: {type(exc).__name__}: {exc}")
            rec.failure(cell.cell_id, type(exc).__name__, {"message": str(exc)})
            with contextlib.suppress(Exception):
                dump_diagnostics(page, self.paths.logs, f"fail_{cell.cell_id}", self.log)
        finally:
            row["instruments"].update(s.each_instrument("end_cell", cell))
            # A cell that could not complete is a FIRST-CLASS RESULT with its failure mode and its RSS at
            # death, not a gap in the table.
            rss = row["instruments"].get("rss") or {}
            row["rss_at_death_mb"] = rss.get("rss_peak_mb") if not row["completed"] else None
            # AND SO IS THE PROBE THAT ALREADY RAN: `--click-probe` finishes inside `_press_send`, and
            # everything after it there still runs under the default 8s action timeout, which a large rung
            # exceeds. Assigned only on the way out of `_run_inner`, the attribution was dropped from the
            # cell it was measured for, and unlike `composer_click_ms` it has no window row to survive in.
            if self._click_attribution_result is not None:
                row["click_attribution"] = self._click_attribution_result
            rec.emit(row)
            # A TERMINAL MARKER FOR A CELL THAT DID NOT FINISH, so a reader scanning FORWARD can discard its
            # windows without joining backwards. `window` rows are written as the film runs and the `cell`
            # row when it ends, so an aborted cell leaves a complete-looking set nothing owns. It cost a
            # headline: reading `stream:gap` windows without the guard reported the 1M rung at 28.7 fps
            # against a 46.7 fps baseline, drawn entirely from an unfinished cell.
            # And 21.8% of frames over 33 ms.
            if not row["completed"]:
                rec.emit(
                    {
                        "row_type": "cell_aborted",
                        "cell_id": cell.cell_id,
                        "reason": (row.get("failure") or {}).get("message", "did not complete"),
                        "kind": (row.get("failure") or {}).get("kind"),
                        "note": (
                            "every window row carrying this cell_id measures an unfinished film "
                            "and must not be pooled with completed cells"
                        ),
                    }
                )
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
        if self.readiness_mode not in MODES:
            raise ValueError(f"unknown readiness mode {self.readiness_mode!r}")
        readiness = self._wait_for_thread(page, seeded)
        row["readiness"] = readiness.as_dict()
        # RECORDED AS A GATE, so no reader can pick up a windowed cell's frame rate without also seeing
        # that its readiness was established a different way.
        rec.gate(
            f"thread_ready:{self.readiness_mode}",
            True,
            readiness.as_dict(),
            cell_id = cell.cell_id,
        )

        # THE COMPLETENESS PROBE, before the idle window and therefore before anything is measured. It
        # scrolls the whole thread, which mounts rows and dirties the page, so the idle window follows.
        do_probe = (
            self.completeness_probe
            if self.completeness_probe is not None
            else self.readiness_mode == MODE_WINDOWED
        )
        if do_probe and seeded.first_marker and seeded.messages > 0:
            completeness = probe_thread_completeness(
                page,
                first_marker = seeded.first_marker,
                expected_messages = seeded.messages,
                log = self.log,
            )
            row["completeness"] = completeness
            record_completeness_gate(rec, cell, completeness)
            # Back to the resting state the gate described, or the idle calibration below runs against a page
            # still settling from the traversal.
            self._wait_for_thread(page, seeded)

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
            # NOT fatal, and NOT silently zero: blocked time is a subtraction against this floor, so without
            # it busy_pct is null with the reason attached and every other column stands.
            self.log(f"  timer clamp NOT established: {clamp.get('reason')}")
            rec.gate("timer_clamp", False, clamp, cell_id = cell.cell_id)
        else:
            self.log(
                f"  timer clamp {clamp['clampMs']:.2f}ms " f"over {clamp.get('samples')} idle ticks"
            )
            rec.gate("timer_clamp", True, clamp, cell_id = cell.cell_id)

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

        # RESET THE FOLLOW SAMPLER FOR THIS CELL, immediately before the film starts: the counters now
        # survive a navigation via sessionStorage, and a cell boundary IS a navigation, so without this
        # cell 2 reports cell 1's samples plus its own and one bad cell poisons every later reading.
        with contextlib.suppress(Exception):
            page.evaluate("() => window.__sb.follow && window.__sb.follow.reset()")

        before_metrics = cdp_metrics(s.ctx.cdp)
        self._composer_click_ms = None
        # `click_attribution` is NOT filed here but in `run`'s `finally`, because a cell that dies after
        # the probe has to keep it.
        t0 = self._press_send(page)
        # On the cell rather than in `actions`, because it happens before the first slot opens and filing
        # it as an action would put a reading outside the film into a list scoring pairs by slot.
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
                "parity_raw": self.parity_raw,
                "parity_shots": self.parity_shots,
                "arm_label": self.arm_label,
                "image_path": str(self.image_path) if self.image_path else None,
                # The follow-up turns `send_turn` streams mid-film, and the pacer it reloads to serve them.
                "_pacer": self.pacer,
                "_stream_queue": [
                    {"reasoning": u.reasoning, "content": u.content, "kind": u.kind}
                    for u in (plan.follow_up_units or [])
                ],
                # Shared and MUTABLE, so consecutive sends advance through the queue. See send_turn.
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
        # DID THE THREAD FOLLOW THE STREAM? Read once, here, after the last window has closed, so the
        # reading is charged to nothing. A GATE, not a column, because of how it fails: a thread that
        # stops following lets the streamed message leave the viewport, a windowed list unmounts it, and
        # the streaming cost collapses -- an excellent frame rate about nothing. A number with a caveat
        # attached is still quoted without the caveat.
        follow = self._read_follow(page)
        row["follow"] = follow
        pinned = follow.get("pinned_fraction")
        coverage = follow.get("attached_fraction_of_stream")
        passed, recorded = follow_verdict(follow)
        follow.update(recorded)
        rec.gate("follows_the_stream", passed, follow, cell_id = cell.cell_id)
        # THE OTHER HALF OF THE CONTRACT, RECORDED AND DELIBERATELY NOT GATED. It was a gate for one run
        # and failed on BOTH arms at nearly the same rate, the signature of a reading about the film:
        # `send_turn` and `stop_generation` each START A RUN, where pinning is intended, and
        # `scroll_after` ends its gesture near the bottom by design. Separating a legitimate re-pin from
        # a yank needs to know which pins the app was ASKED for, which this sampler does not know, so it
        # is a per-arm figure compared BETWEEN arms where the confounds cancel.
        #
        row["scroll_intent"] = {
            # THE ATTESTATION, without which this block fails the bare-zero ban: `yanked_back_samples: 0`
            # beside a non-zero `detached_samples` is the GOOD outcome, and the walker in scoring/schema.py
            # cannot tell that from a counter nobody wrote. False here rather than absent, so "not measured"
            # stays distinguishable from "measured zero".
            "follow_attempted": bool(follow.get("follow_attempted")),
            "detached_samples": follow.get("detached_samples"),
            "yanked_back_samples": follow.get("yanked_back_samples"),
            "gated": False,
            "reason": (
                "the film starts runs of its own (send_turn, stop_generation) and each start pins "
                "to the bottom by design, so this counts legitimate re-pins as well as yanks. "
                "Meaningful only as a difference between two arms of one session"
            ),
        }
        if pinned is None:
            self.log(f"  follow: NOT MEASURED ({follow.get('pinned_fraction_reason')})")
        else:
            cov = follow.get("attached_fraction_of_stream")
            self.log(
                f"  follow: pinned for {pinned:.0%} of the samples taken while attached and "
                f"streaming, over "
                + ("an unknown share" if cov is None else f"{cov:.0%}")
                + " of the streaming time"
                + f", worst drift {follow.get('max_distance_while_running')}px"
                + (", AND IT FELL BEHIND" if follow.get("ever_fell_behind") else "")
            )
        if follow.get("detached_samples"):
            self.log(
                f"  scroll intent: {follow.get('yanked_back_samples')} of "
                f"{follow.get('detached_samples')} samples found the thread back at the bottom "
                f"after the user scrolled away"
                + (" -- THE USER WAS YANKED DOWN" if follow.get("yanked_after_scroll") else "")
            )
        # EVERY STREAM THE CELL SERVED, not just the last one: `last_stats()` describes whichever turn
        # finished last, so for a multi-turn cell it says nothing about the opening reply the rung is
        # named for. Everything stays under `pacer` because that subtree is exempt from the bare-zero
        # rule.
        streams = self.pacer.all_stats()
        planned = self._planned_streams(cell, plan, row)
        row["pacer"] = {
            "last": self.pacer.last_stats(),
            "streams": streams,
            "check": check_planned_streams(streams, planned),
        }
        # A REPLY THAT NEVER FINISHED IS A FAILED CELL, not a completed one with a note. `_drain_stream`
        # reports rather than raises, and the value was read by nothing, so an app still generating three
        # times past its own cadence came back as `completed: true`, exit 0, paired into the A/B ratio.
        # And 120 s beyond that, after the whole film had run.
        # Raised AFTER the drain reading and the pacer's counters are on the row, so how long was waited
        # ships with the failure. The censuses below are not taken, because a census of a still-growing
        # thread describes nothing that was measured.
        if not drained.get("finished"):
            raise RuntimeError(
                f"the reply never finished: {drained.get('reason') or 'the run was still going'} "
                f"({drained.get('drain_ms')}ms waited, {drained.get('expected_ms')}ms expected)"
            )
        # A CELL THAT DID NOT STREAM WHAT IT PLANNED IS A FAILED CELL, for the same reason: the drain
        # check only asks whether the UI stopped running, and a later turn that finishes satisfies it on
        # behalf of an earlier one that did not, so an opening reply that disconnected left a
        # complete-looking cell thousands of characters short of its rung.
        check = row["pacer"]["check"]
        if check["checked"] and not check["ok"]:
            self.log(f"  the cell did not stream what it planned: {check['reason']}")
            raise RuntimeError(f"the cell did not stream what it planned: {check['reason']}")
        # AND THE SAME RULE FOR THE TURN THAT STREAMED BUT NEVER LANDED: the stream check asks whether
        # the bytes went out, while `send_turn` asserts that the thread GREW. Scoped to `send_turn`
        # DELIBERATELY -- an action whose own assertion fails already has its timing voided, and what
        # that does not cover is an action whose failure changed the workload the REST of the cell
        # measured. `send_turn` is the only one that can.
        # The covered case is `expect_ok is False`.
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

        # THE PEAK, over every window's census, not the state at the end of the film: the film ENDS with
        # thread_reopen and delete_message, so an end-of-cell census reports the thread the benchmark has
        # just deleted. The first working run recorded 0 assistant messages against a delivered reply.
        # The pacer's own log proved it: 150 chunks, 3,581 characters, at exactly the 73 ms cadence.
        # 0 messages and 0 characters.
        censuses = [w.get("census") for w in row["actions"] if isinstance(w.get("census"), dict)]
        censuses = [c for c in censuses if c.get("elements")]
        peak = max(censuses, key = lambda c: c.get("elements", 0)) if censuses else {}
        row["census_peak"] = peak
        row["census_peak_attempted"] = bool(censuses)

        # WHICH ACTION THE PEAK CAME FROM, and a standing refusal to compare it across arms. The census
        # attached to an action is taken after it returns, and `reasoning_toggle` opens every pane and
        # closes them again, so that census races the close and which action wins the max() DIFFERS
        # BETWEEN ARMS. Measured on a null control the winner flipped between two actions, a 70.1% swing
        # WITHIN one arm, which produced a published wrong number. Kept as a diagnostic high-water mark,
        # carrying its provenance and an explicit refusal.
        # The winner flipped between `settings` at 64,648 elements and `reasoning_toggle` at 106,067,
        # published as main mounting 48% more Shiki spans.
        # Settled, the two trees mount the same document to within 0.3%.
        peak_from = next(
            (
                w.get("action")
                for w in row["actions"]
                if isinstance(w.get("census"), dict)
                and w["census"].get("elements") == peak.get("elements")
            ),
            None,
        )
        row["census_peak_from_action"] = peak_from
        row["census_peak_comparable_across_arms"] = False
        row["census_peak_note"] = (
            "diagnostic high-water mark only. The action it comes from is chosen by a max() over "
            "per-action censuses that race the action's own teardown, so it is not the same "
            "moment on two arms and must not be differenced across them. For a cross-arm census "
            "use a measure taken at a defined, settled moment."
        )

        census = peak or row["census_after"]
        spans = census.get("highlight_spans") or 0
        chars = census.get("assistant_chars")
        if chars is None:
            chars = page.evaluate("() => window.__sb.dom.assistantChars()")
        row["assistant_chars_in_dom"] = chars
        # The span density the fixture ACHIEVED, measured in the DOM rather than assumed: the field
        # capture ran 5.6 characters per span, and a corpus far from that is not standing in for the same
        # highlighter load per character.
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
            rec.gate(
                "seeded_equals_streamed",
                bool(eq.get("equivalent")),
                eq,
                cell_id = cell.cell_id,
            )
            if not eq.get("equivalent"):
                # A FINDING, printed, not a bug to hide: it says which of this tool's numbers are about the
                # streaming path and which are about a thread that was put there, and it is why the higher rungs
                # carry a fidelity label at all.
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
                # Passing the content gate is not the same as the two threads being identical: a seeded rung
                # carries the same rendered content and materially less mounted DOM.
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
        # THE STREAMED SIDE IS THE PEAK, AND THE PEAK IS TAKEN AT AN UNSTABLE MOMENT, recorded on the row
        # because the seeded side below is read after an explicit 4 s wait -- so this gate differences a
        # racing census against a stable one, the same shape as the defect that made `census_peak`
        # unquotable across arms. NOT the same harm: both sides come from ONE cell on ONE build, so the
        # instability widens the tolerance rather than pointing a difference in a direction.
        streamed = row.get("census_peak") or row.get("census_after") or {}
        streamed_from = "census_peak" if row.get("census_peak") else "census_after"
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
            self._wait_for_thread(page, seeded_thread)
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
        out["streamed_census_from"] = streamed_from
        out["streamed_census_settled"] = streamed_from != "census_peak"
        out["seeded_census"] = seeded_sig
        out["seeded_census_settled"] = True
        out["readiness_mode"] = self.readiness_mode
        if self.readiness_mode == MODE_WINDOWED:
            # SAID OUT LOUD RATHER THAN SCORED QUIETLY: both sides are loaded by the SAME build, so under a
            # windowed arm both censuses count the mounted window. The comparison is still like for like, but
            # it is no longer evidence that seeding reproduces the whole streamed thread.
            out["scope"] = "the mounted window only, not the whole thread"
            out["caveat"] = (
                "this arm mounts a window, so `assistant_messages`, `content_spans` and "
                "`content_code_blocks` are counts over what is mounted at the end of the thread. "
                "A pass is equivalence of the WINDOW, not of the thread."
            )
        # What the mirror was built from, so a drift can be read against the corpus it compared rather
        # than an assumption about which turns were in the thread.
        out["mirrored_follow_ups"] = len(follow_ups)
        out["planned_follow_ups"] = len(plan.follow_up_units or [])
        return out

    def _read_follow(self, page) -> dict:
        """Drain the page-side follow sampler. Never raises: a missing sampler is a reason, not
        a lost cell, and it must not read as a thread that followed."""
        try:
            got = page.evaluate("() => window.__sb.follow && window.__sb.follow.read()")
        except Exception as exc:  # noqa: BLE001
            return {"follow_attempted": False, "reason": f"{type(exc).__name__}: {exc}"}
        if not isinstance(got, dict):
            return {"follow_attempted": False, "reason": "the follow sampler is not installed"}
        return got

    def _wait_for_thread(self, page, seeded: SeededThread) -> Readiness:
        """The readiness gate. See runtime/readiness.py for what it asserts and why.

        The mode is the CELL RUNNER's, not the thread's: an arm declares that it mounts a window
        and the whole run is then gated that way and labelled that way in every row it writes. A
        thread cannot be allowed to talk its way past the gate by looking virtualised, because
        "looks like it mounted fewer nodes on purpose" is indistinguishable from "did not finish".
        """
        return wait_for_thread_ready(
            page,
            seeded.messages,
            marker = seeded.last_marker,
            mode = self.readiness_mode,
            timeout_s = MOUNT_TIMEOUT_S,
            log = self.log,
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

        # FIRST, before anything else touches the page: the same trivial operation N times. The biggest
        # number in the ladder is one nobody could attribute -- the first thing touched after a large
        # thread mounts costs 11 to 24 seconds, and in every probe the cost vanished because whatever ran
        # first absorbed it. Measuring the decay directly is the way out: a first reading in seconds and
        # the rest in milliseconds means the cost is ONE TIME. A no-op body on purpose.
        decay = [settled(lambda: None) for _ in range(5)]
        out: dict[str, Any] = {
            # The harness layer's attestation, load-bearing rather than decorative:
            # `scoring/schema._walk_for_bare_zeros` rejects a bare zero with no sibling `*_attempted` flag,
            # and this block has legitimate zeros -- a thread with no mounted code blocks records
            # `code_token_spans: 0`, and Chromium coarsens `performance.now()` to 100 us.
            # Which is what `forced_layout_ms` is built from.
            "click_attribution_attempted": True,
            "first_touch_ms": decay[0],
            "settled_touch_ms": min(decay[1:]),
            "touch_decay_ms": [round(v, 1) for v in decay],
        }
        # The blur, timed, and timed again from INSIDE the page. The decay series says the first touch
        # after mount costs 10.6 ms at 500K, yet the reading right after the first blur came back at
        # 10,052 ms in two of three runs -- and 10.0 seconds to three digits is the shape of a TIMEOUT.
        # And 10,017 ms in the other.
        # `blur_inpage_ms` decides which: it runs the same blur inside one evaluate, so if the page
        # reports a millisecond while the outer reading is ten seconds, the ten seconds is the driver.
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
        # The reading that decides what `roundtrip_ms` meant: near zero means the cost is a ONE-TIME
        # layout of the mounted thread, expensive again means every interaction pays it.
        blur()
        out["roundtrip_again_ms"] = settled(lambda: None)
        # Measured INSIDE the page, so the protocol round trip is not in the number. `offsetHeight` is
        # read after a write that dirties layout, so it cannot be served from a clean tree.
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
        # In a window, so every instrument covers it: at 500K this single click is the largest cost in the
        # run by an order of magnitude and was the one moment the tool could not see inside. `setup`, NOT
        # `action`: the scoring layer pools every non-excluded window into the three frame metrics, and
        # this window is mostly Playwright's own actionability script running ON THE PAGE'S MAIN THREAD,
        # so filed as an `action` it would put an 11 s driver stall into three weighted headline metrics.
        # It would peg `max_frame_ms`, `jank_index` and `time_in_jank_pct`.
        with self.session.window("setup:composer_click", kind = "setup"):
            # Timed INSIDE the window, like `Window.duration_ms`: the session opens every instrument before
            # this block and closes them after, and at instrument level 1-3 those hooks stop a CPU profile,
            # collect coverage and analyse a trace, so timing around the `with` would grow with the level.
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
        # The composer must be EMPTY for the rest of the film, or the Stop control is replaced by a Queue
        # control and the stop action presses the wrong button. Sending clears it, but the keystroke
        # action refills it, which is why the stop action clears it again itself.
        return t0

    def _drain_stream(self, page, expected_ms: float) -> dict:
        """Wait for the run to end, or say plainly that it did not."""
        # Generous: deficit scheduling makes the stream's own duration machine-independent, so anything
        # much past it is the RENDERER failing to keep up, which is a finding rather than a timeout.
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
    out_lock = None,
) -> tuple[BenchContext, Session]:
    session_id = new_session_id()
    # THE LOCK THE CALLER IS ALREADY HOLDING: `run()` takes the output directory before it archives a
    # payload, so the `Recorder` adopts that lock rather than opening a second against the same path.
    # Without a caller's lock it takes its own.
    recorder = Recorder(paths.payload_jsonl, session_id, lock = out_lock)
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


#: The sources that may SIZE a rung. `measure_chars_per_token`'s last-resort whitespace estimate
#: labels itself "off by tens of percent on dense code", and this corpus is mostly dense code,
#: where it reads 6.7 against tiktoken's 3.3 -- sizing from it would move the error and, past
#: `MANIFEST_CHARS_PER_TOKEN`, make `plan_rung` refuse the whole run on any machine with no
#: tokeniser. The estimate is still measured and reported.
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
    stream_tail_chars: Optional[int] = None,
    corpus_dollars: bool = False,
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
        # The ladder is sized by `ratio`, not the raw `chars_per_token` argument: the argument may be None
        # meaning "measure it", and the measured value is what every cell's `meta` reports, so passing the
        # argument through would size the rungs from a number the payload does not carry.
        plan = plan_rung(
            corpus,
            rung,
            ratio["chars_per_token"],
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
                meta = {"ladder_chars_per_token": ratio},
            )
            out.append((cell, plan))
    return out
