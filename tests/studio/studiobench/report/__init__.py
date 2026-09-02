# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Report: incremental payload on the way in, editorial policy on the way out."""

from .ablation import (  # noqa: F401
    render_arm_verdicts,
    render_batch,
    render_fix_implications,
    render_interactions,
    render_route,
)
from .payload import (  # noqa: F401
    COLLAPSED_SECTIONS,
    PayloadWriter,
    RECORD_KINDS,
    ROW_TYPE_SECTIONS,
    assemble,
    assemble_rows,
    encode,
    excluded_from_rows,
    excluded_totals,
    iter_windows,
    read_records,
    write_excluded,
)
from .render import (  # noqa: F401
    HEADLINE_FRAME_METRICS,
    HeadlinePolicyError,
    assert_headline_pair,
    render_ab_table,
    render_ceiling_shift,
    render_excluded,
    render_frame_health,
    render_harness_bias,
    render_headline,
    render_rung_metrics,
    render_rung_table,
    render_summary,
)

__all__ = [
    "HEADLINE_FRAME_METRICS",
    "render_arm_verdicts",
    "render_batch",
    "render_fix_implications",
    "render_interactions",
    "render_route",
    "COLLAPSED_SECTIONS",
    "HeadlinePolicyError",
    "PayloadWriter",
    "RECORD_KINDS",
    "ROW_TYPE_SECTIONS",
    "assemble",
    "assemble_rows",
    "assert_headline_pair",
    "encode",
    "excluded_from_rows",
    "excluded_totals",
    "iter_windows",
    "read_records",
    "render_ab_table",
    "render_ceiling_shift",
    "render_excluded",
    "render_frame_health",
    "render_harness_bias",
    "render_headline",
    "render_rung_metrics",
    "render_rung_table",
    "render_summary",
    "write_excluded",
]
