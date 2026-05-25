# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any

from eval.json_score.comparators import string_comparator
from .base import ConfigField, MetricResult, MetricSpec, Scorer


def _build(config: dict) -> Scorer:
    threshold = float(config.get("threshold", 0.5))
    cmp = string_comparator(threshold=threshold)

    def score(prediction: str, reference: Any) -> MetricResult:
        ref = "" if reference is None else str(reference)
        return MetricResult(score=float(cmp(ref, prediction)))

    return score


SPEC = MetricSpec(
    name="text_similarity",
    label="Text similarity (ANLS)",
    reference_kind="text",
    config_fields=[
        ConfigField("threshold", "float", 0.5, "Similarity threshold"),
    ],
    build=_build,
)
