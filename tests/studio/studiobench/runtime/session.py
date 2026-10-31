# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One session (one browser, Unsloth, pacer, N cells) is the unit every comparison must stay within."""

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

# A 1x1 PNG generated at runtime: the zipapp artifact has no fixture directory.
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
# 10K is the only rung where both seeding and streaming are affordable.
EQUIVALENCE_RUNG = "10K"
MOUNT_TIMEOUT_S = 180

# 0.95, not 1.0: the sampler ticks at 4 Hz and a legitimate pin can land a tick late.
FOLLOW_PINNED_MIN = 0.95

# Minimum stream-time coverage before `pinned_fraction` counts, else an early verdict reads 100%.
FOLLOW_MIN_STREAM_COVERAGE = 0.50


def follow_verdict(follow: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Pinned and fell-behind are fatal; low stream coverage is set by the schedule, not the build."""

    pinned = follow.get("pinned_fraction")
    coverage = follow.get("attached_fraction_of_stream")
    pinned_ok = pinned is not None and pinned >= FOLLOW_PINNED_MIN
    fell_behind = bool(follow.get("ever_fell_behind"))
    coverage_short = coverage is None or coverage < FOLLOW_MIN_STREAM_COVERAGE
    # Only a re-attachment proves detachment came from the film: a build that never re-pins
    # keeps `pinned_fraction` at 1.0 while coverage collapses.
    reattached = bool(follow.get("reattachments"))
    # Record coverage beside its floor whatever the verdict, so the number is visible.
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


# Not a performance budget: just long enough that the cell survives and the number is recorded.
COMPOSER_CLICK_TIMEOUT_S = 90
SLOW_COMPOSER_CLICK_MS = 1_000


class WindowInUse(RuntimeError):
    pass


def record_completeness_gate(recorder: Recorder, cell: Cell, completeness: dict) -> bool:
    """Gate row must carry cell_id so lost messages name their arm; unmeasured coverage must not pass."""
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
            # Reverse order, so an instrument that wrapped another's state unwinds after it.
            for inst in sorted(self.instruments, key = lambda i: i.name, reverse = True):
                got = self._safe(inst, "close", w)
                if got is not None:
                    w.instruments[inst.name] = got
            self._open = None
            self.ctx.recorder.emit(w.row())

    def each_instrument(self, method: str, *args) -> dict:
        """Iterate a snapshot: _safe may drop a failing instrument, which would skip its neighbour's
        hook."""
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
    # Off by default: the text is megabytes per capture and only `parity_null_control --hunt` uses it.
    parity_raw: bool = False
    parity_shots: Optional[str] = None
    # Burned into filenames: both arms share fixture and film, so images cannot tell sides apart.
    arm_label: str = "base"
    # Set when the 10K check fails; labels every larger (mostly seeded) rung.
    equivalence_failed: bool = False
    # Off by default: costly at large rungs and makes timings incomparable with cells that skip it.
    click_probe: bool = False
    # Per target, so a base-vs-virtualised A/B keeps the base arm on the strict gate. See readiness.py.
    readiness_mode: str = MODE_FULL
    # Proves a windowed arm still holds the thread head; costs a full traversal, so off for `full`.
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
        # Cleared here so the `finally` below cannot attach the previous cell's attribution.
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
            # A failed cell is a first-class result with its failure mode and RSS at death.
            rss = row["instruments"].get("rss") or {}
            row["rss_at_death_mb"] = rss.get("rss_peak_mb") if not row["completed"] else None
            # Keep a click probe that already ran: the cell may die later in `_press_send`.
            if self._click_attribution_result is not None:
                row["click_attribution"] = self._click_attribution_result
            rec.emit(row)
            # Terminal marker so forward readers can discard window rows of a cell that did not finish.
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
        rec.gate(
            f"thread_ready:{self.readiness_mode}",
            True,
            readiness.as_dict(),
            cell_id = cell.cell_id,
        )

        # Completeness probe runs before the idle window: it dirties the page by mounting rows.
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
            self._wait_for_thread(page, seeded)

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
            # Not fatal: without the clamp floor busy_pct is null with a reason; other columns stand.
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

        # Reset per cell: sampler counters survive navigation via sessionStorage.
        with contextlib.suppress(Exception):
            page.evaluate("() => window.__sb.follow && window.__sb.follow.reset()")

        before_metrics = cdp_metrics(s.ctx.cdp)
        self._composer_click_ms = None
        # `click_attribution` is filed in `run`'s `finally` so a cell dying after the probe keeps it.
        t0 = self._press_send(page)
        # On the cell, not in `actions`: it happens before the first slot and is not a slot reading.
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
                "_pacer": self.pacer,
                "_stream_queue": [
                    {"reasoning": u.reasoning, "content": u.content, "kind": u.kind}
                    for u in (plan.follow_up_units or [])
                ],
                # Shared and mutable so consecutive sends advance through the queue. See send_turn.
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
        # A gate, not a column: a thread that stops following unmounts the stream and fakes a fast frame rate.
        follow = self._read_follow(page)
        row["follow"] = follow
        pinned = follow.get("pinned_fraction")
        coverage = follow.get("attached_fraction_of_stream")
        passed, recorded = follow_verdict(follow)
        follow.update(recorded)
        rec.gate("follows_the_stream", passed, follow, cell_id = cell.cell_id)
        # Recorded but not gated: run-starting actions legitimately re-pin, so compare it between arms.
        row["scroll_intent"] = {
            # Attestation for the bare-zero ban in scoring/schema.py; False rather than absent on purpose.
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
        # Every stream, not `last_stats()`; kept under `pacer`, which is exempt from the bare-zero rule.
        streams = self.pacer.all_stats()
        planned = self._planned_streams(cell, plan, row)
        row["pacer"] = {
            "last": self.pacer.last_stats(),
            "streams": streams,
            "check": check_planned_streams(streams, planned),
        }
        # An unfinished reply fails the cell; raised after the drain stats are on the row.
        if not drained.get("finished"):
            raise RuntimeError(
                f"the reply never finished: {drained.get('reason') or 'the run was still going'} "
                f"({drained.get('drain_ms')}ms waited, {drained.get('expected_ms')}ms expected)"
            )
        # A later finished turn can satisfy the drain check for an earlier disconnected one.
        check = row["pacer"]["check"]
        if check["checked"] and not check["ok"]:
            self.log(f"  the cell did not stream what it planned: {check['reason']}")
            raise RuntimeError(f"the cell did not stream what it planned: {check['reason']}")
        # Only `send_turn` failures change the workload the rest of the cell measured.
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

        # Peak over all window censuses: the film ends by deleting messages, so the final census is empty.
        censuses = [w.get("census") for w in row["actions"] if isinstance(w.get("census"), dict)]
        censuses = [c for c in censuses if c.get("elements")]
        peak = max(censuses, key = lambda c: c.get("elements", 0)) if censuses else {}
        row["census_peak"] = peak
        row["census_peak_attempted"] = bool(censuses)

        # Diagnostic only, never compare across arms: which action wins the max() races and differs by arm.
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
        # Measured span density; the field capture ran 5.6 characters per span.
        row["chars_per_span"] = round(chars / spans, 2) if spans else None
        row["chars_per_span_target"] = 5.6
        self.log(
            f"  after: {census['elements']:,} elements, {spans:,} spans, "
            f"{chars:,} assistant chars -> {row['chars_per_span']} chars/span"
        )

        row["fidelity"] = "streamed_and_seeded" if plan.seeded_units else "streamed_only"

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
                # Passing the content gate does not mean identical: seeded rungs mount materially less DOM.
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
        """A send_turn that ran but got no reply is a planned turn that did not stream; it must fail."""
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
        """Counts every follow-up that streamed, from the action rows, so the mirror seeds only what
        landed."""
        streamed = 0
        for action in row.get("actions") or []:
            if action.get("action") != "send_turn":
                continue
            if action.get("ran") and action.get("expect_ok") is not False:
                streamed += 1
        return list(plan.follow_up_units or [])[:streamed]

    def _check_equivalence(self, plan: RungPlan, row: dict) -> dict:
        """Seeds the same text as a full thread and compares the DOM the app built from both paths."""
        s = self.session
        page = s.ctx.page
        # The streamed peak is racy vs the stable seeded read; it widens the tolerance, not the direction.
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
            # Under a windowed arm both censuses count only the mounted window.
            out["scope"] = "the mounted window only, not the whole thread"
            out["caveat"] = (
                "this arm mounts a window, so `assistant_messages`, `content_spans` and "
                "`content_code_blocks` are counts over what is mounted at the end of the thread. "
                "A pass is equivalence of the WINDOW, not of the thread."
            )
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
        """The mode is the cell runner's, so a thread cannot pass the gate by looking virtualised."""
        return wait_for_thread_ready(
            page,
            seeded.messages,
            marker = seeded.last_marker,
            mode = self.readiness_mode,
            timeout_s = MOUNT_TIMEOUT_S,
            log = self.log,
        )

    def _click_attribution(self, page, selector: str) -> dict:
        """page.click adds O(DOM) driver work a user never pays; the other paths isolate the user's cost."""

        def blur() -> None:
            page.evaluate("() => document.activeElement && document.activeElement.blur()")
            page.wait_for_timeout(250)

        def settled(fn) -> float:
            """Times fn plus a round trip into the page, since the call can return before the page
            processes it."""
            started = time.monotonic()
            fn()
            page.evaluate("() => document.body.offsetHeight")
            return (time.monotonic() - started) * 1000.0

        # First, before anything else touches the page: the first touch after mount absorbs a one-time cost.
        decay = [settled(lambda: None) for _ in range(5)]
        out: dict[str, Any] = {
            # Required by `scoring/schema._walk_for_bare_zeros`: this block has legitimate zeros
            # (performance.now() is coarsened to 100 us), and without the flag `--report` refuses it.
            "click_attribution_attempted": True,
            "first_touch_ms": decay[0],
            "settled_touch_ms": min(decay[1:]),
            "touch_decay_ms": [round(v, 1) for v in decay],
        }
        # Inner `blur_inpage_ms` vs outer timing tells a 10 s driver timeout apart from page cost.
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
        # Near zero means a one-time layout cost; expensive again means every interaction pays it.
        blur()
        out["roundtrip_again_ms"] = settled(lambda: None)
        # Measured in-page; `offsetHeight` follows a layout-dirtying write so it cannot hit a clean tree.
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
        """Bounded by COMPOSER_CLICK_TIMEOUT_S; composer_click_ms is harness health, not what a user
        pays."""
        selector = 'textarea[aria-label="Message input"]'
        page.wait_for_selector(selector, timeout = 60_000)
        if self.click_probe:
            self._click_attribution_result = self._click_attribution(page, selector)
        # Kind `setup`, not `action`: this is mostly Playwright driver stall and would peg frame metrics.
        with self.session.window("setup:composer_click", kind = "setup"):
            # Timed inside the window: instrument open/close cost grows with the instrument level.
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
        # The composer must stay empty, or Stop becomes Queue; the stop action clears it again itself.
        return t0

    def _drain_stream(self, page, expected_ms: float) -> dict:
        """Wait for the run to end, or say plainly that it did not."""
        # Deficit scheduling makes stream duration machine-independent; overrun is a renderer finding.
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
    # Adopt the caller's output-directory lock; without one the Recorder takes its own.
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


# Whitespace estimates misread dense code (6.7 vs 3.3 chars/token), so they may not size rungs.
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
    """Measured once on the corpus before planning, not per rung, since one unit's ratio swings."""
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
    """chars_per_token None measures the corpus first; the ratio used is recorded in each cell's meta."""
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
        # Size from `ratio`, the measured value the payload reports, not the possibly-None argument.
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
