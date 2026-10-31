# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Off by default: wrapping scrollHeight adds cost, so its overhead is measured by a paired cell."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..scoring.schema import Measure

LAYOUTCOST_JS_PATH = Path(__file__).resolve().parents[1] / "instruments" / "layoutcost.js"

# Instrument level this runs at; headline numbers come from level 0 only.
LAYOUTCOST_LEVEL = 3

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
    """A refused patch reads NOT ATTEMPTED, not zero, so WebKit cannot falsely show zero forced layouts."""

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
    """Paired-cell cost, not self-estimate; a negative beyond noise is reported, never clamped to zero."""

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
    """Registration happens in register(), not at import, so a harness edit cannot break this package."""

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
        # `ctx.page` may be replaced after a renderer crash, so re-read it.
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
        # Declares only the lower bound it can measure; the real number needs the deep-tier paired cell.
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
    """Takes the decorator as an argument so this module has no import-time dependency on the harness."""

    @register_instrument(name = LayoutCostInstrument.name, level = LAYOUTCOST_LEVEL)
    def _make() -> LayoutCostInstrument:
        return LayoutCostInstrument()

    return _make
