# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from core.benchmark.parse import _sample_correct, extract_samples, pick_default_metric


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


def test_task_specific_filters_still_get_a_headline():
    assert (
        pick_default_metric(_m("exact_match,custom-extract", "exact_match_stderr,custom-extract"))
        == "exact_match,custom-extract"
    )
    assert pick_default_metric(_m("exact_match,none")) == "exact_match,none"
    assert pick_default_metric(_m("exact_match,get-answer")) == "exact_match,get-answer"
    ifeval = _m(
        "inst_level_loose_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_strict_acc,none",
        "prompt_level_strict_acc,none",
    )
    assert pick_default_metric(ifeval) == "prompt_level_strict_acc,none"


def _gsm8k(filters):
    samples = []
    for doc_id in range(2):
        for filt, em in filters:
            samples.append(
                {
                    "doc_id": doc_id,
                    "doc": {"question": f"q{doc_id}"},
                    "target": "4",
                    "filter": filt,
                    "metrics": ["exact_match"],
                    "exact_match": em,
                    "filtered_resps": [filt],
                    "resps": [["raw"]],
                }
            )
    return {
        "results": {
            "gsm8k": {
                "exact_match,strict-match": 0.0,
                "exact_match,flexible-extract": 1.0,
            }
        },
        "samples": {"gsm8k": samples},
    }


def test_samples_keep_only_the_headline_filter():
    data = _gsm8k([("strict-match", 0.0), ("flexible-extract", 1.0)])
    samples = extract_samples(data, "gsm8k")
    assert [s["doc_id"] for s in samples] == [0, 1]
    assert {s["response"] for s in samples} == {"strict-match"}
    assert not any(s["correct"] for s in samples)


def test_samples_with_one_filter_are_untouched():
    samples = extract_samples(_gsm8k([("strict-match", 1.0)]), "gsm8k")
    assert len(samples) == 2 and all(s["correct"] for s in samples)
