# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ablation arms: the layer that turns a correlation into a cause.

Four pieces, in the order they constrain each other:

    manifest.py     what every arm must declare (INVARIANCE, POTENCY) and the verdict logic that
                    turns a run into QUOTED, BOUND, VOIDED, NOT RUN or UNAVAILABLE
    knobs.py/.js    the seven runtime-injected knobs on the shipped build, as a decision table
    ladder.py       the telescoping ladder, its adjacent differences, and the interaction term
                    between two routes to the same floor
    calibration.py  NULL and SPIKE, non-droppable, which decide whether the batch is quotable

plus `bundle.py` for the arms that need an armed dist, `dose.py` for the dose-response that makes
a null informative, and `recovery.py` for occupancy versus retention.
"""

from .batch import (  # noqa: F401
    BatchPlanError,
    BatchResult,
    PlannedCell,
    assert_equal_scene_duration,
    judge_batch,
    missing_cells,
    plan_batch,
)
from .bundle import (  # noqa: F401
    BANNER,
    BUNDLE_ARMS,
    ArmpackManifest,
    ArmpackResolution,
    ArmpackUnavailable,
    discover_armpack,
)
from .calibration import (  # noqa: F401
    CALIBRATION_ARM_IDS,
    CalibrationMissing,
    CalibrationVerdict,
    SPIKE_SIZES_MS,
    SpikeRecovery,
    assert_batch_includes_calibration,
    calibration_arms,
    evaluate_batch,
    null_arm,
    null_delta_from_outcomes,
    spike_arm,
    spike_init_script,
)
from .dose import DOSES, DoseFit, DosePoint, fit_dose_response  # noqa: F401
from .knobs import (  # noqa: F401
    ARM_BY_ID,
    PREBOOT_ARM_IDS,
    RUNTIME_ARM_IDS,
    RUNTIME_ARMS,
    arms_json,
    config_init_script,
    decision_table,
    init_scripts_for,
    load_knobs_js,
    render_decision_table,
    split_arms,
)
from .ladder import (  # noqa: F401
    DECLARED_ROUTES,
    MECHANISMS,
    MECHANISM_FIX,
    InteractionTerm,
    LadderError,
    LadderRoute,
    RouteResult,
    Step,
    StepResult,
    arms_key,
    differences,
    interaction_terms,
    required_rungs,
)
from .manifest import (  # noqa: F401
    Arm,
    ArmOutcome,
    ArmStatus,
    DeclaredDiff,
    Invariance,
    PotencyCounter,
    judge,
)
from .recovery import RECOVERY_TURNS, RecoveryResult, classify_recovery  # noqa: F401
