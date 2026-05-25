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


def test_text_similarity_identical_and_different():
    score = make_scorer("text_similarity", {"threshold": 0.5})
    assert score("Acme Corporation", "Acme Corporation").score == 1.0
    assert score("abcdefgh", "zzzzzzzz").score == 0.0  # below threshold -> 0


def test_json_document_perfect_and_partial():
    schema = {"total": {"type": "money"}, "currency": "categorical"}
    score = make_scorer("json_document", {"schema": schema})
    gt = {"total": 100, "currency": "USD"}
    perfect = score('{"total": 100, "currency": "USD"}', gt)
    assert perfect.score == 1.0
    assert perfect.breakdown is not None and "children" in perfect.breakdown
    partial = score('{"total": 90, "currency": "EUR"}', gt)
    assert abs(partial.score - 0.45) < 1e-9  # money .9 + categorical 0, /2


def test_json_document_reference_as_json_string():
    score = make_scorer("json_document", {})
    r = score('{"a": "x"}', '{"a": "x"}')
    assert r.score == 1.0


def test_json_document_unparseable_prediction_is_error_zero():
    score = make_scorer("json_document", {})
    r = score("the model refused", {"a": "x"})
    assert r.score == 0.0 and r.error is not None


def test_json_document_bad_reference_is_error_zero():
    score = make_scorer("json_document", {})
    r = score('{"a": "x"}', "not json")
    assert r.score == 0.0 and r.error is not None


def test_text_similarity_and_json_in_registry():
    names = {m["name"] for m in list_metrics()}
    assert {"text_similarity", "json_document"} <= names
    kinds = {m["name"]: m["reference_kind"] for m in list_metrics()}
    assert kinds["json_document"] == "json"
