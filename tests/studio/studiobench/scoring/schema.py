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

#: Keys whose numeric zero is a genuine, meaningful zero rather than a missing measurement.
#: Everything here is an index, an identifier, a declared constant or a cardinality that is
#: interesting precisely when it is zero. Anything not on this list must be a `Measure`.
ZERO_OK_KEYS = frozenset(
    {
        "index",
        # The parity capture in scene/parity.js numbers its message rows with a bare loop
        # counter (`i`), not `index`, so the first message of every capture is `i: 0`. That is
        # an ordinal like every other name on this list, and rejecting it made `--report` fail
        # on any payload that carried parity rows at all.
        "i",
        # ── the visible-region capture, whose zeros are all true statements ──
        #
        # NOT the whole `visible` subtree, deliberately. Exempting a subtree here once already
        # defeated a test that required a zero-length `chars` inside `parity` to stay a loud
        # failure, and the same argument applies with more force to this one: a visible message
        # whose signature is zero characters long is a broken capture and must stay loud. Only the
        # counters are named, and each is a fact rather than a measurement:
        #   unmounted_at_capture: 0  every message the viewport showed was still mounted
        #   mounted_ever_visible: 0  a windowed arm had unmounted them all again, which the
        #                            analysis reports as NOT COMPARABLE rather than as agreement
        #   ever_visible_count: 0    the scan observed nothing, which the positive control in
        #                            compare_visible refuses; it is not scored anywhere
        # ── the SSE wire's integrity counters, whose zeros are the good outcome ──
        #
        # Again the scalars and not the `wire` subtree. "No frame failed to parse" and "no
        # unterminated frame was left buffered" are the statements that make the character count
        # trustworthy, and a run where they are zero is the run whose denominator can be scored.
        # `wireChars` itself is NOT on this list: a zero there means nothing was ever counted and
        # must stay loud.
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
        "score",  # a rung score of 0 is the whole point of the incomplete-rung rule
        "weight",
        "n_excluded",
        "exit_code",
        "residual_ms",  # telescoping ladders are residual-free BY CONSTRUCTION; 0 is the claim
        "at_ms",  # record timestamps relative to the writer's start
        "truncated_records",
        "records_written",
        "frames_total",  # a window with no frames is a SYMPTOM; frames.py already flags it
        "n_pairs",
        # -- harness-row counters whose zero is the GOOD result, not a missing reading. Each of
        # these is written unconditionally by the session layer for every cell, so "absent" and
        # "zero" are already distinguishable: absent means the row was never written at all.
        "slots_missed",  # 0 means the film kept time, which is exactly what we want to read
        "expect_failures",  # 0 means every action's expectation held
        "over_budget_ms",  # 0 means the action fitted inside its slot
        "seeded_chars",  # the small rungs seed nothing; 0 is the truth about the cell
        "seeded_messages",
        "seed_seconds",
        "bursts",  # a pacer that never had to burst is a pacer that was never jammed
        # Seeded-vs-streamed equivalence: 0 drift on a field is the PASSING result, and it is the
        # single most load-bearing zero in the payload, because it is what licenses every rung
        # above 10K to be seeded rather than streamed.
        "drift",
        # The two counts `drift` is computed FROM, in that same block. They are censuses of the
        # DOM in each arm, so a field that is absent from both reads 0/0 and is the passing
        # result: on a real quick-tier run `reasoning_spans` is 0 streamed and 0 seeded, and the
        # payload marks the field `gating: false` in the same breath. Exempting the quotient but
        # not the two numbers under it failed every payload that carried an equivalence block.
        "streamed",
        "seeded",
    }
)

#: Subtrees the bare-zero walker does not descend into. These hold configuration, identity,
#: histogram buckets and raw instrument bookkeeping, none of which is a measurement of the app.
#: The boundary is a contract with the harness layer: anything that is a MEASUREMENT goes under
#: a `metrics` key as a measure object, anything that is bookkeeping goes under `info`. Without
#: that split the walker either fails on every legitimate counter or is switched off entirely,
#: and switched off is how the ban stops existing.
EXEMPT_SUBTREE_KEYS = frozenset(
    {
        "info",
        "raw",
        "config",
        "env",
        "identity",
        "header",
        # The comparability key's own section. `fields` is the identity block the key is hashed
        # over -- `instrument_level`, `stream_tail_chars`, `corpus_dollars` -- every one of which
        # is a true statement about how the run was configured rather than a measurement of the
        # app, and every one of which is legitimately 0. Same rule as `identity` and `config`
        # directly above.
        "comparability",
        "footer",
        "record_counts",
        "histogram",
        "potency_counters",
        # -- OBSERVATION subtrees written by the session layer. These are censuses of the DOM and
        # the expectations checked against them: "0 code blocks" is a true statement about the
        # page at that instant, not a failed measurement. They are never scored and never quoted
        # as a cost; everything that IS scored reaches the report as a Measure via
        # scoring/from_payload.py, and that path stays under the strict rule.
        "census",
        "census_before",
        "census_after",
        "census_peak",
        "streamed_census",
        "seeded_census",
        "expect",
        # The readiness gate's verdict and the completeness probe's, both written once per cell by
        # the session layer. Same rule as a census: `from_bottom: 0` is a true statement that the
        # viewport is exactly at the bottom, `waited_ms: 0` is a thread that was ready on the
        # first sample, and neither is a measurement of the app. Nothing here is scored; what the
        # gate produces for the report is a `gate` row, which carries a boolean.
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

    # -- constructors -------------------------------------------------------------------

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

    # -- predicates ---------------------------------------------------------------------

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

    # -- rendering ----------------------------------------------------------------------

    def display(self) -> str:
        if not self.attempted:
            return f"not attempted ({self.note})"
        if self.value is None:
            return f"no reading ({self.note or 'instrument failed'})"
        if self.sub_floor:
            # A negative reading under the floor is bounded from below, not from above; printing
            # it as `< floor` would read as "small and positive" for a value that is neither.
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


#: Reasons a cell may be excluded. Free-text reasons are refused so the report can total them.
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


# ---------------------------------------------------------------------------------------
# payload validation
# ---------------------------------------------------------------------------------------


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
        # A measure object is self-describing; its own zero is fine because `attempted` sits
        # next to it. Do not descend, the inner `value` is exactly the thing being exempted.
        if node.get("attempted") is None:
            problems.append(f"{path}: measure object without `attempted`")
        return
    if isinstance(node, Mapping):
        # The harness layer cannot emit measure objects (its rows are JSON-safe scalars), so it
        # attests with a `<name>_attempted` flag. That flag is written ONCE PER INSTRUMENT BLOCK,
        # not once per counter: the frames instrument emits `frames_attempted: true` alongside
        # `frames_over_33`, `long_tasks` and the rest. An exact-name-sibling rule would demand
        # `frames_over_33_attempted` and fail on every real row, and a check that fails on every
        # real row gets switched off, which is how the ban stops existing.
        #
        # So a mapping that positively attests (any `*_attempted` key that is True) covers the
        # numeric zeros DIRECTLY inside it. It does not cover nested mappings, which carry their
        # own attestation or none.
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
            # The same attestation, for a plain numeric ARRAY directly inside the block. An
            # instrument that attests reports samples as well as counters: `frames` writes
            # `frames_attempted: true` next to `frame_gaps_ms`, whose entries are inter-frame
            # gaps in whole milliseconds, so two frames in the same millisecond record a
            # legitimate 0. Walking into the list dropped the attestation on the floor and every
            # such run failed validation, which is the scalar case one line above with an extra
            # pair of brackets around it.
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
