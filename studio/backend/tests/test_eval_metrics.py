# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from eval.metrics.registry import make_scorer, list_metrics
from eval.metrics.base import MetricResult


def test_exact_match_basic():
    score = make_scorer("exact_match", {})
    assert score("Yes", "Yes").score == 1.0
    assert score("Yes", "No").score == 0.0


def test_exact_match_case_insensitive_and_strip():
    score = make_scorer("exact_match", {"case_insensitive": True, "strip": True})
    assert score("  yes ", "YES").score == 1.0


def test_exact_match_strip_default_true():
    score = make_scorer("exact_match", {})
    assert score("yes ", "yes").score == 1.0  # trailing space stripped


def test_make_scorer_unknown_raises():
    with pytest.raises(ValueError, match="Unknown metric"):
        make_scorer("nope", {})


def test_list_metrics_includes_exact_match_schema():
    metrics = {m["name"]: m for m in list_metrics()}
    assert "exact_match" in metrics
    em = metrics["exact_match"]
    assert em["reference_kind"] == "text"
    field_names = {f["name"] for f in em["config_fields"]}
    assert {"case_insensitive", "strip"} <= field_names


def test_metric_result_shape():
    r = MetricResult(score=0.5)
    assert r.score == 0.5 and r.breakdown is None and r.error is None
