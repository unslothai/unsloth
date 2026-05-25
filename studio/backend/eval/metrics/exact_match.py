# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Any

from .base import ConfigField, MetricResult, MetricSpec, Scorer


def _build(config: dict) -> Scorer:
    case_insensitive = bool(config.get("case_insensitive", False))
    strip = bool(config.get("strip", True))

    def score(prediction: str, reference: Any) -> MetricResult:
        a = "" if prediction is None else str(prediction)
        b = "" if reference is None else str(reference)
        if strip:
            a, b = a.strip(), b.strip()
        if case_insensitive:
            a, b = a.lower(), b.lower()
        return MetricResult(score=1.0 if a == b else 0.0)

    return score


SPEC = MetricSpec(
    name="exact_match",
    label="Exact match",
    reference_kind="text",
    config_fields=[
        ConfigField("case_insensitive", "bool", False, "Case-insensitive"),
        ConfigField("strip", "bool", True, "Trim whitespace"),
    ],
    build=_build,
)
