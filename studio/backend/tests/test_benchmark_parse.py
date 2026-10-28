# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.benchmark.parse import _sample_correct, pick_default_metric


def _m(*names):
    return [{"name": n, "score": 0.5, "stderr": None} for n in names]


def test_priority_prefers_normalized_then_strict():
    assert pick_default_metric(_m("acc", "acc_norm")) == "acc_norm"
    assert (
        pick_default_metric(_m("exact_match,flexible-extract", "exact_match,strict-match"))
        == "exact_match,strict-match"
    )


def test_priority_order_across_families():
    # acc_norm wins over exact_match even when the latter is listed first.
    assert pick_default_metric(_m("exact_match,strict-match", "acc_norm")) == "acc_norm"
    # The list is a strict rank: exact_match beats pass@1, which beats f1.
    assert pick_default_metric(_m("f1", "pass@1", "exact_match")) == "exact_match"
    assert pick_default_metric(_m("f1", "pass@1")) == "pass@1"


def test_unmapped_or_empty_metrics_yield_no_default():
    assert pick_default_metric([]) == ""
    assert pick_default_metric(_m("brier_score", "word_perplexity")) == ""


def test_sample_correct_uses_the_same_priority():
    # A sample is correct when the highest-priority metric present hits 1.0;
    # lower-priority fields never override the headline choice.
    assert _sample_correct({"acc_norm": 0.0, "acc": 1.0}) is False
    assert _sample_correct({"acc_norm": 1.0, "acc": 0.0}) is True
    assert _sample_correct({"exact_match,strict-match": 1.0}) is True
    assert _sample_correct({}) is False
