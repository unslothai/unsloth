# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What every ablation arm must declare before it is allowed to produce a number.

An ablation is a claim of the form "removing X made it faster, therefore X was the cost". That
claim has exactly two ways of being false, and both of them are silent:

  1. THE ARM CHANGED THE OUTPUT. Turning off the thing also turned off some of the work, so the
     two sides are not rendering the same page and the difference is not attributable to X. The
     worst case is not a visible difference; it is a small invisible one. An earlier stub arm in
     this codebase produced 552 syntax-highlighted spans where the real page produces 2,561, and
     it read as a clean 4x win. The arm was not fast. It was rendering a fifth of the content.
  2. THE ARM DID NOT FIRE. The knob was injected, the selector matched nothing, the run completed
     and the difference was zero. Reported as "no effect", that is evidence AGAINST the mechanism.
     It is not evidence of anything; the treatment was never applied. Four instrument defects
     found in a single day were all this shape, which is why the inversion is enforced here:
     an arm whose potency counter did not move reads NOT RUN, never "no effect".

So every arm declares:

  INVARIANCE -- what proves the rendered output did not change, in one of three classes:
      EXACT         the output digest is byte-identical. Anything else voids the arm.
      EQUIVALENT    identical after a declared, reviewed normaliser, AND the observed diff is
                    EXACTLY the declared diff. An extra difference, however small, voids it.
      DOM_CHANGING  the output legitimately differs. Usable only as a BOUND, printed as `<= x`,
                    never as a point estimate.
  POTENCY -- a counter, read before and after, that proves the arm actually fired, with the
      minimum movement that counts as fired. The counter must be something the arm CAUSES, not
      something correlated with it.

The distinction between VOIDED and NOT RUN matters more than either of them: VOIDED means we
measured something real and cannot attribute it, NOT RUN means we measured nothing. Reporting
both as "no effect" is how a whole day of work concluded that nothing was slow.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..scoring.schema import Measure


class Invariance(enum.Enum):
    """How strongly an arm claims the rendered output is unchanged."""

    EXACT = "EXACT"
    EQUIVALENT = "EQUIVALENT"
    DOM_CHANGING = "DOM_CHANGING"


class ArmStatus(enum.Enum):
    """The five things an arm run can be. Only two of them produce a number."""

    QUOTED = "QUOTED"  # invariance held, potency fired: a point estimate
    BOUND = "BOUND"  # DOM-changing but potent: an upper bound, printed as `<= x`
    VOIDED = "VOIDED"  # claimed invariance and drifted: measured, unattributable, not quoted
    NOT_RUN = "NOT_RUN"  # potency counter did not move: the treatment never happened
    UNAVAILABLE = "UNAVAILABLE"  # the arm needs a build that this install is not


@dataclass(frozen = True)
class PotencyCounter:
    """The proof that the arm fired, and the minimum movement that counts.

    `direction` is `"increase"`, `"decrease"` or `"any"`. A knob that is supposed to REMOVE work
    usually proves itself by a counter going DOWN, and accepting movement in either direction
    would let an unrelated regression pass as potency.
    """

    name: str
    min_delta: float
    direction: str = "any"
    description: str = ""

    def fired(self, before: float | None, after: float | None) -> bool:
        if before is None or after is None:
            return False
        delta = float(after) - float(before)
        if self.direction == "increase":
            return delta >= self.min_delta
        if self.direction == "decrease":
            return -delta >= self.min_delta
        return abs(delta) >= self.min_delta

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "min_delta": float(self.min_delta),
            "direction": self.direction,
            "description": self.description,
        }


@dataclass(frozen = True)
class DeclaredDiff:
    """For EQUIVALENT arms: exactly what is allowed to differ, and nothing else.

    `keys` are the normaliser-visible fields permitted to change. The verification is an equality
    check against the OBSERVED set, not a subset check: an arm that declares one difference and
    produces two is voided, because the second one is precisely the thing nobody looked at.
    """

    normaliser: str
    keys: tuple[str, ...]
    rationale: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "normaliser": self.normaliser,
            "keys": list(self.keys),
            "rationale": self.rationale,
        }


@dataclass(frozen = True)
class Arm:
    """One ablation arm: what it does, what it claims, and how it proves it fired."""

    arm_id: str
    title: str
    mechanism: str
    invariance: Invariance
    potency: PotencyCounter
    implies_fix: str
    kind: str = "runtime"  # "runtime" | "bundle" | "calibration" | "control" | "dose"
    declared_diff: DeclaredDiff | None = None
    init_script: str | None = None
    requires_armpack: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.invariance is Invariance.EQUIVALENT and self.declared_diff is None:
            raise ValueError(
                f"{self.arm_id}: an EQUIVALENT arm must declare its normaliser and the exact "
                "diff it is allowed to produce"
            )
        if self.invariance is not Invariance.EQUIVALENT and self.declared_diff is not None:
            raise ValueError(
                f"{self.arm_id}: only an EQUIVALENT arm may declare a diff; an EXACT arm that "
                "needs one is an EQUIVALENT arm that has not admitted it"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "title": self.title,
            "mechanism": self.mechanism,
            "invariance": self.invariance.value,
            "kind": self.kind,
            "implies_fix": self.implies_fix,
            "requires_armpack": bool(self.requires_armpack),
            "potency": self.potency.to_json(),
            "declared_diff": self.declared_diff.to_json() if self.declared_diff else None,
            "notes": self.notes,
        }


@dataclass
class ArmOutcome:
    """One arm, run once, with its verdict and the evidence behind it."""

    arm: Arm
    cost: Measure
    status: ArmStatus
    reason: str
    digest_before: str | None = None
    digest_after: str | None = None
    normalised_before: str | None = None
    normalised_after: str | None = None
    observed_diff_keys: tuple[str, ...] = ()
    potency_before: float | None = None
    potency_after: float | None = None
    potency_counters: dict[str, Any] = field(default_factory = dict)

    @property
    def quotable(self) -> bool:
        return self.status in (ArmStatus.QUOTED, ArmStatus.BOUND)

    def quote(self) -> str:
        """Render this arm's cost with the qualifier its status demands.

        A DOM-changing arm is printed as a bound and cannot be printed any other way; that is the
        entire difference between "this mechanism costs 40 ms" and "this mechanism plus whatever
        else changed costs at most 40 ms".
        """

        if self.status is ArmStatus.QUOTED:
            return self.cost.display()
        if self.status is ArmStatus.BOUND:
            return f"<= {self.cost.display()}"
        if self.status is ArmStatus.NOT_RUN:
            return f"NOT RUN ({self.reason})"
        if self.status is ArmStatus.VOIDED:
            return f"VOIDED ({self.reason})"
        return f"UNAVAILABLE ({self.reason})"

    def to_json(self) -> dict[str, Any]:
        return {
            "arm": self.arm.to_json(),
            "status": self.status.value,
            "reason": self.reason,
            "quote": self.quote(),
            "quotable": self.quotable,
            "cost": self.cost.to_json(),
            "invariance_evidence": {
                "digest_before": self.digest_before,
                "digest_after": self.digest_after,
                "normalised_before": self.normalised_before,
                "normalised_after": self.normalised_after,
                "observed_diff_keys": list(self.observed_diff_keys),
            },
            "potency_evidence": {
                "counter": self.arm.potency.name,
                "before": self.potency_before,
                "after": self.potency_after,
                "fired": self.arm.potency.fired(self.potency_before, self.potency_after),
                "potency_counters": dict(self.potency_counters),
            },
        }


def judge(
    arm: Arm,
    *,
    cost: Measure,
    digest_before: str | None,
    digest_after: str | None,
    normalised_before: str | None = None,
    normalised_after: str | None = None,
    observed_diff_keys: Sequence[str] = (),
    potency_before: float | None = None,
    potency_after: float | None = None,
    potency_counters: Mapping[str, Any] | None = None,
    available: bool = True,
    unavailable_reason: str = "",
) -> ArmOutcome:
    """Apply the manifest to one run of one arm and produce its verdict.

    ORDER MATTERS AND IS DELIBERATE.

    Availability is checked first: an arm that needs an armed bundle and did not get one has no
    evidence of any kind. Invariance is checked SECOND, before potency, because a voided arm has
    already produced a number that must not be quoted, and letting a NOT RUN verdict pre-empt it
    would hide a real drift behind a benign-sounding label. Potency is checked LAST, so the only
    way to reach QUOTED is: available, invariance held, and the counter moved.
    """

    counters = dict(potency_counters or {})

    if not available:
        return ArmOutcome(
            arm = arm,
            cost = Measure.not_attempted(cost.unit, unavailable_reason or "arm unavailable"),
            status = ArmStatus.UNAVAILABLE,
            reason = unavailable_reason or "arm unavailable for this build",
            potency_counters = counters,
        )

    outcome = ArmOutcome(
        arm = arm,
        cost = cost,
        status = ArmStatus.QUOTED,
        reason = "",
        digest_before = digest_before,
        digest_after = digest_after,
        normalised_before = normalised_before,
        normalised_after = normalised_after,
        observed_diff_keys = tuple(observed_diff_keys),
        potency_before = potency_before,
        potency_after = potency_after,
        potency_counters = counters,
    )

    if arm.invariance is Invariance.EXACT:
        if digest_before is None or digest_after is None:
            outcome.status = ArmStatus.VOIDED
            outcome.reason = (
                "this arm claims EXACT invariance but no output digest was captured, so the "
                "claim was never checked"
            )
            return outcome
        if digest_before != digest_after:
            outcome.status = ArmStatus.VOIDED
            outcome.reason = (
                f"claims EXACT invariance and the output digest changed "
                f"({digest_before[:12]} -> {digest_after[:12]}). The two sides did not render "
                "the same thing, so the difference cannot be attributed to the mechanism"
            )
            return outcome
    elif arm.invariance is Invariance.EQUIVALENT:
        declared = arm.declared_diff
        assert declared is not None  # enforced in Arm.__post_init__
        if normalised_before is None or normalised_after is None:
            outcome.status = ArmStatus.VOIDED
            outcome.reason = (
                "claims EQUIVALENT invariance but the normaliser produced no output, so the "
                "equivalence was never checked"
            )
            return outcome
        if normalised_before != normalised_after:
            outcome.status = ArmStatus.VOIDED
            outcome.reason = (
                f"claims EQUIVALENT under normaliser {declared.normaliser!r} and the normalised "
                "digests still differ"
            )
            return outcome
        observed = tuple(sorted(set(observed_diff_keys)))
        expected = tuple(sorted(set(declared.keys)))
        if observed != expected:
            extra = sorted(set(observed) - set(expected))
            missing = sorted(set(expected) - set(observed))
            outcome.status = ArmStatus.VOIDED
            outcome.reason = (
                "the observed diff is not exactly the declared diff"
                + (f"; undeclared: {extra}" if extra else "")
                + (f"; declared but absent: {missing}" if missing else "")
                + ". An undeclared difference is the one nobody looked at."
            )
            return outcome

    if not arm.potency.fired(potency_before, potency_after):
        outcome.status = ArmStatus.NOT_RUN
        outcome.reason = (
            f"potency counter {arm.potency.name!r} did not move "
            f"({potency_before} -> {potency_after}, needs {arm.potency.direction} of at least "
            f"{arm.potency.min_delta}). The treatment was not applied, so this is not a "
            "measurement of no effect"
        )
        outcome.cost = Measure.not_attempted(cost.unit, "arm did not fire")
        return outcome

    if arm.invariance is Invariance.DOM_CHANGING:
        outcome.status = ArmStatus.BOUND
        outcome.reason = (
            "the arm legitimately changes the rendered output, so its cost is an upper bound on "
            "the mechanism and not a point estimate"
        )
        return outcome

    outcome.status = ArmStatus.QUOTED
    outcome.reason = "invariance held and the potency counter moved"
    return outcome
