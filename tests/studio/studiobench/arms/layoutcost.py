# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Python side of `instruments/layoutcost.js`: an instrument that measures its own cost.

WHY THIS FILE LIVES UNDER `arms/`. `instruments/layoutcost.js` is the browser half; the harness
layer owns `instruments/` as a package and its registration machinery. This is the ablation
layer's adapter over it, and it exists for the ablation plane: the counters it reads
(`scrollHeight` reads, `scrollTop` writes, MutationObserver callbacks and records, custom
property writes) are the potency evidence for arms D and E, and the thing they instrument is the
mechanism those arms remove.

WHY IT IS OFF BY DEFAULT AND WHY IT RUNS TWICE. Wrapping the `scrollHeight` getter to time it
adds a call frame and two `performance.now()` reads to the very operation under suspicion. That
is not a small effect on a counter that fires per streamed character. So:

  * the instrument declares level 3, and never runs at the levels the headline numbers come from;
  * the deep tier runs the SAME CELL twice, once with it and once without, and reports the
    difference as the instrument's in-situ cost.

The second point is the one that matters. `selfCostEstimate()` inside the JS measures the wrapper
against a detached, clean element, which is a lower bound and a fair one; the paired cell measures
what it actually cost in the page, with a dirty layout tree and a real observer running. Those two
numbers are usually different, and quoting the cheap one because it is easier to obtain is how an
instrument's cost gets assumed rather than known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..scoring.schema import Measure

LAYOUTCOST_JS_PATH = Path(__file__).resolve().parents[1] / "instruments" / "layoutcost.js"

#:The instrument level at which this may run. Headline numbers come from level 0 only.
LAYOUTCOST_LEVEL = 3

#:Counter families the browser side reports, each with its own `attempted` flag.
COUNTER_FAMILIES = (
    "scrollHeightReads",
    "scrollTopWrites",
    "scrollToCalls",
    "moCallbacks",
    "moRecords",
    "customPropSets",
)


def load_layoutcost_js() -> str:
    return LAYOUTCOST_JS_PATH.read_text(encoding = "utf-8")


@dataclass
class LayoutCostReading:
    """One window's layout-cost counters, as Measures rather than bare integers."""

    counters: dict[str, Measure] = field(default_factory = dict)
    timings: dict[str, Measure] = field(default_factory = dict)
    unavailable: list[str] = field(default_factory = list)
    self_cost_ms_per_call: Measure = field(
        default_factory = lambda: Measure.not_attempted("ms", "self cost not estimated")
    )
    stabilizer_sets: Measure = field(
        default_factory = lambda: Measure.not_attempted("count", "not read")
    )
    viewport_observer_callbacks: Measure = field(
        default_factory = lambda: Measure.not_attempted("count", "not read")
    )

    def to_json(self) -> dict[str, Any]:
        return {
            "counters": {k: v.to_json() for k, v in self.counters.items()},
            "timings": {k: v.to_json() for k, v in self.timings.items()},
            "unavailable": list(self.unavailable),
            "self_cost_ms_per_call": self.self_cost_ms_per_call.to_json(),
            "stabilizer_sets": self.stabilizer_sets.to_json(),
            "viewport_observer_callbacks": self.viewport_observer_callbacks.to_json(),
        }


def reading_from_snapshot(snapshot: Mapping[str, Any] | None) -> LayoutCostReading:
    """Turn the browser snapshot into Measures, preserving `attempted` per family.

    A patch the engine refused (a non-configurable descriptor on some WebKit builds) comes back
    in `unavailable`, and every counter under it becomes NOT ATTEMPTED rather than zero. That
    distinction is the whole reason this adapter exists instead of the raw dict being written
    straight into the payload: a WebKit run that could not install the getter patch would
    otherwise report zero forced layouts, which is the most confident possible way to be wrong.
    """

    if not snapshot:
        return LayoutCostReading(
            unavailable = list(COUNTER_FAMILIES),
            counters = {
                name: Measure.not_attempted("count", "layoutcost produced no snapshot")
                for name in COUNTER_FAMILIES
            },
        )

    unavailable = list(snapshot.get("unavailable") or [])
    raw_counters = dict(snapshot.get("counters") or {})
    raw_timings = dict(snapshot.get("timings") or {})
    attempted_map = dict(snapshot.get("attempted") or {})

    reading = LayoutCostReading(unavailable = unavailable)
    for name in COUNTER_FAMILIES:
        family_ok = attempted_map.get(name, name not in unavailable)
        if not family_ok:
            reading.counters[name] = Measure.not_attempted(
                "count", f"{name}: the patch could not be installed on this engine"
            )
            continue
        value = raw_counters.get(name)
        if value is None:
            reading.counters[name] = Measure.failed("count", f"{name} missing from the snapshot")
        else:
            reading.counters[name] = Measure.read(float(value), "count")

    for name, value in raw_timings.items():
        if value is None:
            reading.timings[name] = Measure.failed("ms", f"{name} missing from the snapshot")
        else:
            reading.timings[name] = Measure.read(float(value), "ms")

    self_cost = snapshot.get("overheadMsPerCall")
    if self_cost is not None:
        reading.self_cost_ms_per_call = Measure.read(
            float(self_cost),
            "ms/call",
            note = (
                "measured against a detached clean element, so this is a LOWER BOUND on the "
                "in-page cost; the paired with/without cell is the real number"
            ),
        )

    mo = dict(snapshot.get("mo") or {})
    if "viewportCallbacks" in mo:
        reading.viewport_observer_callbacks = Measure.read(float(mo["viewportCallbacks"]), "count")
    if "stabilizerSets" in raw_counters:
        reading.stabilizer_sets = Measure.read(float(raw_counters["stabilizerSets"]), "count")
    return reading


def in_situ_overhead(with_instrument_ms: Measure, without_instrument_ms: Measure) -> Measure:
    """The instrument's real cost, from the paired cell. Not its own estimate of itself.

    Positive means the instrumented cell was slower, which is the expected direction. A negative
    result larger than the noise means the pair is not measuring what it thinks it is, and it is
    reported as a reading rather than clamped to zero, because a clamp would turn a broken pair
    into a plausible one.
    """

    if not (with_instrument_ms.has_reading and without_instrument_ms.has_reading):
        return Measure.failed(
            with_instrument_ms.unit,
            "the with/without pair is incomplete, so the instrument's cost is unknown rather "
            "than zero",
        )
    return Measure.read(
        float(with_instrument_ms.value) - float(without_instrument_ms.value),
        with_instrument_ms.unit,
    )


class LayoutCostInstrument:
    """Adapter satisfying the harness layer's `Instrument` protocol.

    Registration is deliberately not done at import time here: `instruments/__init__.py` and its
    `register_instrument` decorator belong to the harness layer, and importing them from this
    layer would make the ablation package fail to import whenever the harness package is being
    edited. `register()` below is called by whoever wires the two together.
    """

    name = "layoutcost"
    level = LAYOUTCOST_LEVEL

    def __init__(self) -> None:
        self._ctx: Any = None
        self._page: Any = None
        self._installed = False
        self._error: str | None = None

    def attach(self, ctx: Any) -> None:
        self._ctx = ctx

    def start_cell(self, cell: Any) -> None:
        # `ctx.page` may be replaced between cells when a crashed renderer is recovered, so the page is
        # re-read here rather than cached in attach().
        self._page = getattr(self._ctx, "page", None)

    def open(self, window: Any) -> None:
        if self._page is None:
            return
        try:
            self._page.evaluate(
                "() => { if (window.__sbLayoutCost) { window.__sbLayoutCost.reset(); } }"
            )
            self._installed = True
        except Exception as error:  # pragma: no cover - browser-side failure path
            self._error = str(error)

    def close(self, window: Any) -> dict[str, Any] | None:
        if self._page is None:
            return None
        try:
            snapshot = self._page.evaluate(
                "() => (window.__sbLayoutCost ? window.__sbLayoutCost.snapshot() : null)"
            )
        except Exception as error:  # pragma: no cover - browser-side failure path
            return {"error": str(error), "attempted": True}
        return reading_from_snapshot(snapshot).to_json()

    def end_cell(self, cell: Any) -> dict[str, Any] | None:
        # The harness contract requires every instrument at level >= 1 to declare its own cost. This one
        # declares the LOWER BOUND it can measure itself, because the real number comes from the paired
        # with/without cell that only the deep tier runs.
        if self._page is None:
            return {"overhead_ms": None, "overhead_attempted": False}
        try:
            estimate = self._page.evaluate(
                "() => (window.__sbLayoutCost ? window.__sbLayoutCost.selfCostEstimate() : null)"
            )
        except Exception:  # pragma: no cover - browser-side failure path
            estimate = None
        if not estimate:
            return {"overhead_ms": None, "overhead_attempted": False}
        return {
            "overhead_ms": estimate.get("overheadMsPerCall"),
            "overhead_attempted": True,
            "overhead_is_lower_bound": True,
            "overhead_note": (
                "per-call wrapper cost against a detached clean element; the in-page cost is "
                "measured by the paired with/without cell"
            ),
        }

    def detach(self) -> None:
        self._page = None


def register(register_instrument: Any) -> Any:
    """Wire this instrument into the harness layer's registry.

    Takes the decorator rather than importing it, so this module has no import-time dependency on
    a package another layer is still building.
    """

    @register_instrument(name = LayoutCostInstrument.name, level = LAYOUTCOST_LEVEL)
    def _make() -> LayoutCostInstrument:
        return LayoutCostInstrument()

    return _make
