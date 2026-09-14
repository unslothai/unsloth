# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scoring: pure functions over a payload, with no dependency on the harness.

Everything in this package takes plain data and returns plain data. It never touches Playwright,
a browser, a socket or the clock. That is deliberate: the arithmetic that turns milliseconds into
a verdict is the part most likely to be wrong in a way nobody notices, so it is the part that
must be unit-testable against synthetic payloads including adversarial ones (a crashed rung that
must not outscore a slow one, a regression that must surface despite a positive headline).
"""

from .anchors import (  # noqa: F401
    DEFAULT_NOISE_FLOOR_PCT,
    METRIC_ANCHORS,
    METRIC_BY_KEY,
    MIN_WEIGHT_COVERAGE,
    ONSET_METRIC_FLOOR,
    ONSET_SCORE_THRESHOLD,
    RUNG_TOKENS,
    MetricAnchor,
    rung_ladder_id,
    weights_id,
)
from .ab import (  # noqa: F401
    AbResult,
    IncomparableRuns,
    MetricComparison,
    Pair,
    RunIdentity,
    assert_comparable,
    bootstrap_geomean_ci,
    compare,
    noise_floor_from_null_control,
    pairs_from_cells,
)
from .frames import (  # noqa: F401
    JANK_FRAME_MS,
    FrameStats,
    build_histogram,
    compute_frame_stats,
    measure_refresh_interval_ms,
)
from .schema import (  # noqa: F401
    EXCLUSION_REASONS,
    ExcludedCell,
    Measure,
    PayloadSchemaError,
    check_exclusion_reasons,
    validate_payload,
)
from .score import (  # noqa: F401
    LadderScore,
    MetricScore,
    RungScore,
    log_rung_weights,
    score_ladder,
    score_metric,
    score_rung,
)

__all__ = [
    "AbResult",
    "DEFAULT_NOISE_FLOOR_PCT",
    "EXCLUSION_REASONS",
    "ExcludedCell",
    "FrameStats",
    "IncomparableRuns",
    "JANK_FRAME_MS",
    "LadderScore",
    "METRIC_ANCHORS",
    "METRIC_BY_KEY",
    "MIN_WEIGHT_COVERAGE",
    "Measure",
    "MetricAnchor",
    "MetricComparison",
    "MetricScore",
    "ONSET_METRIC_FLOOR",
    "ONSET_SCORE_THRESHOLD",
    "Pair",
    "PayloadSchemaError",
    "RUNG_TOKENS",
    "RunIdentity",
    "RungScore",
    "assert_comparable",
    "bootstrap_geomean_ci",
    "build_histogram",
    "check_exclusion_reasons",
    "compare",
    "compute_frame_stats",
    "log_rung_weights",
    "measure_refresh_interval_ms",
    "noise_floor_from_null_control",
    "pairs_from_cells",
    "rung_ladder_id",
    "score_ladder",
    "score_metric",
    "score_rung",
    "validate_payload",
    "weights_id",
]
