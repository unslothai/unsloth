# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ablation arms turn a correlation into a cause; calibration decides which batches may be quoted."""

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
