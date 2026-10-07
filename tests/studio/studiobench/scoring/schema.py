# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The value type that makes a bare zero unrepresentable.

Every number this benchmark prints is one of three things and they are not interchangeable:

    * a reading            -- the instrument ran and saw this
    * a reading below the  -- the instrument ran and saw nothing it can distinguish from
      detection floor         nothing, which is NOT the same as seeing nothing
    * not attempted        -- the instrument did not run here at all

A day of measurement was lost to the third being printed as `0.00`. `LayoutDuration` read as a
flat `0.105 -> 0.134` across a 300x range in thread size, which is exactly what a harness prints
when the mechanism it is charging never executed; a React stage read `0.00` because `<Profiler>`
is stripped from a production build, and the run was quoted as "React costs nothing". Both are
the same defect: a slot that was never filled rendered as a measurement of zero.

So `Measure` carries `attempted` next to `value`, always, in memory and in the JSON, and
`display()` refuses to emit a naked `0`:

    Measure(0.04, attempted=True, unit="ms/update", floor=0.12)
        -> "< 0.12 ms/update (instrument floor)"
    Measure.not_attempted("ms", "profiling alias not verified")
        -> "not attempted (profiling alias not verified)"

`validate_payload()` is the enforcement arm: it walks an assembled payload and fails on any
numeric zero that is not inside a measure object or explicitly exempted by key. That check runs
in the unit tests over synthetic payloads AND over the real payload before the report renders,
so the ban is a property of the schema rather than a rule contributors are asked to remember.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

MEASURE_KIND = "measure"

#: Keys whose numeric zero is genuine rather than missing; anything else must be a `Measure`.
ZERO_OK_KEYS = frozenset(
    {
        "index",
        # parity.js numbers message rows from `i: 0`.
        "i",
        # Only the counters, not the `visible`/`wire` subtrees: a zero-char signature or zero
        # `wireChars` must stay loud.
        "wire_parse_failures",
        "wire_pending_chars",
        "wire_parse_failures_in_window",
        "wire_pending_chars_at_close",
        "ever_visible_count",
        "mounted_ever_visible",
        "unmounted_at_capture",
        "rung_index",
        "arm_index",
        "window_index",
        "slot_index",
        "step_index",
        "bootstrap_seed",
        "seed",
        "count",  # only under excluded_cells / histogram buckets, see validate_payload
        "bucket_ms",
        "bucket_count",
        "spike_ms",
        "dose",
        "score",
        "weight",
        "n_excluded",
        "exit_code",
        "residual_ms",
        "at_ms",
        "truncated_records",
        "records_written",
        "frames_total",
        "n_pairs",
        # Harness-row counters whose zero is the good result; written unconditionally per cell.
        "slots_missed",
        "expect_failures",
        "over_budget_ms",
        "seeded_chars",
        "seeded_messages",
        "seed_seconds",
        "bursts",
        # Zero seeded-vs-streamed drift is the passing result that licenses seeding rungs above 10K.
        "drift",
        # The two counts `drift` is computed from; both can legitimately be 0.
        "streamed",
        "seeded",
    }
)

#: Subtrees the bare-zero walker skips (config, identity, bookkeeping): a contract with the
#: harness, which puts measurements under `metrics` and bookkeeping under `info`.
EXEMPT_SUBTREE_KEYS = frozenset(
    {
        "info",
        "raw",
        "config",
        "env",
        "identity",
        "header",
        "comparability",
        "ab_plan",
        "footer",
        "record_counts",
        "histogram",
        "potency_counters",
        # Observation subtrees (DOM censuses) where 0 is a true statement; never scored.
        "census",
        "census_before",
        "census_after",
        "census_peak",
        "streamed_census",
        "seeded_census",
        "expect",
        "readiness",
        "completeness",
        "notes",
        "scene",
        "cell",
        "pacer",
        "stream",
        "clamp",
    }
)


class PayloadSchemaError(AssertionError):
    """Raised when a payload contains a number that cannot be interpreted."""


@dataclass(frozen = True)
class Measure:
    """One number, plus everything needed to know whether it means anything.

    `value` is `None` whenever there is no reading: either the instrument was never attempted
    (`attempted is False`) or it was attempted and failed (`attempted is True`, `note` says how).
    `floor` is the instrument's detection floor in `unit`; a magnitude under it renders as a
    bound, never as a value and never as zero.
    """

    value: float | None
    attempted: bool
    unit: str = "ms"
    floor: float | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.attempted and self.value is not None:
            raise PayloadSchemaError(
                "a Measure that was not attempted cannot carry a value; got "
                f"value={self.value!r}"
            )
        if self.value is not None and not math.isfinite(float(self.value)):
            raise PayloadSchemaError(f"non-finite Measure value {self.value!r}")
        if self.floor is not None and self.floor <= 0:
            raise PayloadSchemaError(f"detection floor must be positive; got {self.floor!r}")
        if not self.attempted and not self.note:
            raise PayloadSchemaError("a not-attempted Measure must say why")

    @classmethod
    def not_attempted(cls, unit: str, reason: str) -> "Measure":
        return cls(value = None, attempted = False, unit = unit, note = reason)

    @classmethod
    def failed(cls, unit: str, reason: str) -> "Measure":
        """Attempted, but produced no usable reading. Distinct from both zero and skipped."""
        return cls(value = None, attempted = True, unit = unit, note = reason)

    @classmethod
    def read(
        cls,
        value: float,
        unit: str = "ms",
        floor: float | None = None,
        note: str | None = None,
    ) -> "Measure":
        return cls(value = float(value), attempted = True, unit = unit, floor = floor, note = note)

    @property
    def has_reading(self) -> bool:
        return self.attempted and self.value is not None

    @property
    def sub_floor(self) -> bool:
        """True when the instrument ran and could not distinguish the result from nothing."""
        return (
            self.has_reading
            and self.floor is not None
            and abs(float(self.value)) < float(self.floor)
        )

    def display(self) -> str:
        if not self.attempted:
            return f"not attempted ({self.note})"
        if self.value is None:
            return f"no reading ({self.note or 'instrument failed'})"
        if self.sub_floor:
            # A negative sub-floor reading is bounded from below; `< floor` would misread it.
            if float(self.value) < 0:
                return f"> -{_fmt(self.floor)} {self.unit} (instrument floor)"
            return f"< {_fmt(self.floor)} {self.unit} (instrument floor)"
        return f"{_fmt(self.value)} {self.unit}"

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": MEASURE_KIND,
            "value": None if self.value is None else float(self.value),
            "attempted": bool(self.attempted),
            "unit": self.unit,
            "floor": self.floor,
            "sub_floor": self.sub_floor,
            "note": self.note,
            "display": self.display(),
        }

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        key: str,
        *,
        unit: str = "ms",
        floor: float | None = None,
    ) -> "Measure":
        """Build a Measure from the harness layer's sibling-key convention.

        Layer 1 emits JSON-safe scalars, so it cannot emit a Measure object. Its contract is the
        same idea in flat form: a numeric key that can legitimately be zero carries
        `<key>_attempted: bool`, and a quantity that could not be measured is `None` with
        `<key>_reason: str`. This is the one place the two representations meet, so that the ban
        on bare zeros survives the boundary instead of being re-argued on the other side of it.
        """

        value = row.get(key)
        reason = row.get(f"{key}_reason")
        attempted_key = f"{key}_attempted"
        attempted = bool(row.get(attempted_key, key in row))
        if value is None:
            if attempted:
                return cls.failed(unit, reason or f"{key} produced no reading")
            return cls.not_attempted(unit, reason or f"{key} was not attempted")
        if not attempted:
            raise PayloadSchemaError(
                f"{key} carries a value but {attempted_key} is false; a row cannot both have a "
                "reading and claim it was never attempted"
            )
        return cls.read(float(value), unit, floor = floor, note = reason)

    @classmethod
    def from_json(cls, blob: Mapping[str, Any]) -> "Measure":
        if blob.get("kind") != MEASURE_KIND:
            raise PayloadSchemaError(f"not a measure object: {blob!r}")
        return cls(
            value = blob.get("value"),
            attempted = bool(blob.get("attempted")),
            unit = blob.get("unit", ""),
            floor = blob.get("floor"),
            note = blob.get("note"),
        )


def _fmt(value: float | None) -> str:
    if value is None:
        return "None"
    value = float(value)
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.3g}"


@dataclass
class ExcludedCell:
    """One cell that did not make it into scoring, and why.

    `excluded_cells` is mandatory and non-null in every payload. An empty list is a claim ("we
    excluded nothing"); a missing key is an unanswered question, and the two used to print the
    same way.
    """

    cell_id: str
    reason: str
    count: int = 1
    detail: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "reason": self.reason,
            "count": int(self.count),
            "detail": self.detail,
        }


EXCLUSION_REASONS = frozenset(
    {
        "clock_disagreement",
        "selfcheck_failed",
        "renderer_crash",
        "goto_timeout",
        "slot_missed",
        "arm_voided_invariance",
        "arm_not_run_potency",
        "bundle_arm_unavailable",
        "overhead_correlated_with_treatment",
        "seeding_fidelity_unverified",
        "rung_incomplete",
    }
)


def check_exclusion_reasons(cells: Iterable[ExcludedCell]) -> None:
    for cell in cells:
        if cell.reason not in EXCLUSION_REASONS:
            raise PayloadSchemaError(
                f"unknown exclusion reason {cell.reason!r} for cell {cell.cell_id!r}; "
                "add it to EXCLUSION_REASONS so the report can total it"
            )


def _is_measure(node: Any) -> bool:
    return isinstance(node, Mapping) and node.get("kind") == MEASURE_KIND


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Fail loudly on the two schema violations that made earlier reports unreadable.

    1. a numeric zero outside a measure object and outside the declared exemptions, which is
       indistinguishable from "we never ran that";
    2. a missing or null `excluded_cells`.

    Raises `PayloadSchemaError`. Callers run this before rendering anything.
    """

    if "excluded_cells" not in payload:
        raise PayloadSchemaError("payload is missing the mandatory `excluded_cells` key")
    if payload["excluded_cells"] is None:
        raise PayloadSchemaError("`excluded_cells` is null; use [] to claim nothing was excluded")
    if not isinstance(payload["excluded_cells"], Sequence):
        raise PayloadSchemaError("`excluded_cells` must be a list")

    problems: list[str] = []
    _walk_for_bare_zeros(payload, path = "$", problems = problems)
    if problems:
        joined = "\n  ".join(problems)
        raise PayloadSchemaError(
            "bare zeros found; every zero must be a Measure carrying `attempted`, or an "
            f"exempted key:\n  {joined}"
        )


def _is_number_list(node: Any) -> bool:
    """A list of plain numbers, i.e. an instrument's raw samples rather than a structure."""

    if not isinstance(node, (list, tuple)) or not node:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in node)


def _walk_for_bare_zeros(node: Any, path: str, problems: list[str]) -> None:
    if _is_measure(node):
        if node.get("attempted") is None:
            problems.append(f"{path}: measure object without `attempted`")
        return
    if isinstance(node, Mapping):
        # Harness rows attest with one `<name>_attempted` flag per instrument block, so it covers
        # numeric zeros directly inside the mapping, not nested mappings.
        attested = any(k.endswith("_attempted") and v is True for k, v in node.items())
        for key, child in node.items():
            if key in EXEMPT_SUBTREE_KEYS:
                continue
            covered = attested or f"{key}_attempted" in node
            if (
                isinstance(child, (int, float))
                and not isinstance(child, bool)
                and float(child) == 0.0
                and covered
            ):
                continue
            # Also covers a numeric list directly inside: `frame_gaps_ms` can hold a legitimate 0.
            if covered and _is_number_list(child):
                continue
            _walk_for_bare_zeros(child, f"{path}.{key}", problems)
        return
    if isinstance(node, (list, tuple)):
        for index, child in enumerate(node):
            _walk_for_bare_zeros(child, f"{path}[{index}]", problems)
        return
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)) and float(node) == 0.0:
        leaf = path.rsplit(".", 1)[-1].split("[")[0]
        if leaf not in ZERO_OK_KEYS:
            problems.append(f"{path} = 0")
