# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A batch: the unit that is planned, run, calibrated and then either quoted or not.

A batch is every cell needed to answer one question at one rung: both ladder routes, their shared
floor, the control, and the non-droppable calibration arms. It is the unit because calibration is
the unit: the noise floor and the detection floor are properties of a machine at a moment, and
carrying them across batches is how a number measured on a cold laptop gets quoted against one
measured while a build was running.

WHAT THIS MODULE ENFORCES, BEFORE ANYTHING RUNS:

  * calibration arms are present. `assert_batch_includes_calibration` refuses a plan without them,
    because "we will check afterwards whether that batch was resolvable" is not a thing that can
    be done afterwards.
  * every rung both declared routes need has a cell. A route with a hole cannot telescope, and
    discovering that after an hour of measurement wastes the hour.
  * the scene is the SAME LENGTH in every arm. This is the quiet one. The scene is a film on a
    wall clock, so if one arm's film is 40 seconds and another's is 44, the two arms saw different
    amounts of streaming and their difference includes that. Layer 1's contract allows an arm to
    supply its own slot list; this check is what stops that freedom from silently breaking
    additivity.

WHAT IT DOES AFTER:
  judges each arm against its manifest, computes both routes' adjacent differences, computes the
  interaction terms between them, and hands the whole thing to the report layer. It does not
  decide what is quotable; `CalibrationVerdict` does, and `BatchResult.quotable` just reads it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..scoring.schema import Measure
from .bundle import ArmpackResolution
from .calibration import (
    CALIBRATION_ARM_IDS,
    CalibrationVerdict,
    assert_batch_includes_calibration,
)
from .dose import DoseFit
from .ladder import (
    DECLARED_ROUTES,
    InteractionTerm,
    LadderError,
    LadderRoute,
    RouteResult,
    arms_key,
    differences,
    interaction_terms,
    required_rungs,
)
from .manifest import ArmOutcome, ArmStatus
from .recovery import RecoveryResult


class BatchPlanError(AssertionError):
    """Raised when a batch plan cannot answer the question it is being run to answer."""


@dataclass(frozen = True)
class PlannedCell:
    """One cell of a batch: which arms are applied, and what it is for."""

    arms: frozenset[str]
    role: str  # "ladder" | "calibration" | "control" | "dose" | "recovery"
    label: str = ""

    @property
    def key(self) -> str:
        return arms_key(self.arms)

    def to_json(self) -> dict[str, Any]:
        return {"arms": sorted(self.arms), "key": self.key, "role": self.role, "label": self.label}


def plan_batch(
    routes: Sequence[LadderRoute] = DECLARED_ROUTES,
    *,
    calibration_arm_ids: Sequence[str] = CALIBRATION_ARM_IDS,
) -> list[PlannedCell]:
    """Every cell this batch must run, ladder rungs first, calibration always."""

    cells = [PlannedCell(arms = rung, role = "ladder") for rung in required_rungs(routes)]
    cells.extend(
        PlannedCell(arms = frozenset({arm_id}), role = "calibration", label = arm_id)
        for arm_id in calibration_arm_ids
    )
    planned_ids = {arm_id for cell in cells for arm_id in cell.arms}
    assert_batch_includes_calibration(planned_ids)
    return cells


def assert_equal_scene_duration(
    scene_durations_ms: Mapping[str, float], *, tolerance_ms: float = 1.0
) -> None:
    """Every arm in a batch must run a scene of the same length.

    The scene is slot-scheduled on the wall clock, so its duration sets how much streaming each
    arm saw. Two arms with different scene lengths differ by the treatment AND by the workload,
    and no amount of care in the ladder recovers that. Layer 1 permits per-arm slot lists; this
    is the check that keeps the permission from quietly breaking additivity.
    """

    if not scene_durations_ms:
        raise BatchPlanError("no scene durations were supplied, so equality was never checked")
    values = list(scene_durations_ms.values())
    spread = max(values) - min(values)
    if spread > tolerance_ms:
        offenders = ", ".join(
            f"{arm}={duration:.1f} ms" for arm, duration in sorted(scene_durations_ms.items())
        )
        raise BatchPlanError(
            f"scene durations differ across arms by {spread:.1f} ms (tolerance {tolerance_ms:g} "
            f"ms): {offenders}. Arms with different-length scenes saw different amounts of "
            "streaming, so their difference is not the treatment"
        )


@dataclass
class BatchResult:
    """One batch, judged. Nothing in here is quotable unless `quotable` is true."""

    rung_tokens: int
    calibration: CalibrationVerdict
    outcomes: dict[str, ArmOutcome] = field(default_factory = dict)
    routes: list[RouteResult] = field(default_factory = list)
    interactions: list[InteractionTerm] = field(default_factory = list)
    armpack: ArmpackResolution | None = None
    dose: DoseFit | None = None
    recovery: RecoveryResult | None = None
    plan_notes: list[str] = field(default_factory = list)

    @property
    def quotable(self) -> bool:
        return self.calibration.quotable

    @property
    def detection_floor_ms(self) -> float | None:
        floor = self.calibration.detection_floor_ms
        return float(floor.value) if floor.has_reading else None

    def voided_arms(self) -> list[ArmOutcome]:
        return [o for o in self.outcomes.values() if o.status is ArmStatus.VOIDED]

    def not_run_arms(self) -> list[ArmOutcome]:
        return [o for o in self.outcomes.values() if o.status is ArmStatus.NOT_RUN]

    def to_json(self) -> dict[str, Any]:
        return {
            "rung_tokens": int(self.rung_tokens),
            "quotable": self.quotable,
            "calibration": self.calibration.to_json(),
            "outcomes": {key: outcome.to_json() for key, outcome in self.outcomes.items()},
            "routes": [route.to_json() for route in self.routes],
            "interactions": [term.to_json() for term in self.interactions],
            "armpack": self.armpack.to_json() if self.armpack else None,
            "dose": self.dose.to_json() if self.dose else None,
            "recovery": self.recovery.to_json() if self.recovery else None,
            "plan_notes": list(self.plan_notes),
            "voided_arms": [o.arm.arm_id for o in self.voided_arms()],
            "not_run_arms": [o.arm.arm_id for o in self.not_run_arms()],
        }


def judge_batch(
    *,
    rung_tokens: int,
    outcomes: Mapping[str, ArmOutcome],
    calibration: CalibrationVerdict,
    routes: Sequence[LadderRoute] = DECLARED_ROUTES,
    armpack: ArmpackResolution | None = None,
    dose: DoseFit | None = None,
    recovery: RecoveryResult | None = None,
) -> BatchResult:
    """Turn a batch's raw arm outcomes into route differences and interaction terms.

    The detection floor comes from the calibration arms of THIS batch and is threaded into every
    difference, so a step below what this machine could resolve prints as a bound rather than as
    a small number. That is the whole reason calibration is per batch.
    """

    result = BatchResult(
        rung_tokens = int(rung_tokens),
        calibration = calibration,
        outcomes = dict(outcomes),
        armpack = armpack,
        dose = dose,
        recovery = recovery,
    )
    floor = result.detection_floor_ms

    for route in routes:
        result.routes.append(differences(route, outcomes, detection_floor_ms = floor))

    for index, left in enumerate(result.routes):
        for right in result.routes[index + 1 :]:
            try:
                result.interactions.extend(interaction_terms(left, right, detection_floor_ms = floor))
            except LadderError as error:
                result.plan_notes.append(str(error))

    if not result.quotable:
        result.plan_notes.append("this batch is NOT quotable: " + calibration.reason)
    return result


def missing_cells(planned: Iterable[PlannedCell], outcomes: Mapping[str, ArmOutcome]) -> list[str]:
    """Which planned cells produced no outcome. Checked before the report, not after."""

    return [cell.key for cell in planned if cell.key not in outcomes]
