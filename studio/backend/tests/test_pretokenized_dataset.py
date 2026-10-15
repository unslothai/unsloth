# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-tokenized datasets (input_ids / attention_mask / labels) route to training as-is (#7139), and nothing else moves."""

import sys
from pathlib import Path

import pytest
from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils.dataset_format import check_dataset_format as hub_check_dataset_format  # noqa: E402
from utils.datasets.dataset_utils import (  # noqa: E402
    check_dataset_format,
    format_and_template_dataset,
    format_dataset,
)
from utils.datasets.pretokenized import (  # noqa: E402
    has_custom_labels,
    is_pretokenized_dataset,
    prepare_pretokenized_dataset,
)


def _pretokenized(with_labels = True, n = 4):
    rows = {
        "input_ids": [[1, 2, 3, 4, 5][: 3 + i % 3] for i in range(n)],
        "attention_mask": [[1] * (3 + i % 3) for i in range(n)],
    }
    if with_labels:
        rows["labels"] = [[-100] + ids[1:] for ids in rows["input_ids"]]
    return Dataset.from_dict(rows)


NORMAL_DATASETS = {
    "alpaca": (
        Dataset.from_dict({"instruction": ["Say hi"], "input": [""], "output": ["hi"]}),
        "alpaca",
    ),
    "chatml": (
        Dataset.from_dict(
            {
                "messages": [
                    [
                        {"role": "user", "content": "Say hi"},
                        {"role": "assistant", "content": "hi"},
                    ]
                ]
            }
        ),
        "chatml",
    ),
    "sharegpt": (
        Dataset.from_dict(
            {
                "conversations": [
                    [{"from": "human", "value": "Say hi"}, {"from": "gpt", "value": "hi"}]
                ]
            }
        ),
        "sharegpt",
    ),
    # A text dataset that also ships token ids keeps its text route.
    "question_answer_with_ids": (
        Dataset.from_dict({"question": ["Say hi"], "answer": ["hi"], "input_ids": [[1, 2, 3]]}),
        "custom_heuristic",
    ),
}


@pytest.mark.parametrize("with_labels", [True, False])
def test_pretokenized_is_detected_and_needs_no_mapping(with_labels):
    ds = _pretokenized(with_labels)
    assert is_pretokenized_dataset(ds)
    for check in (check_dataset_format, hub_check_dataset_format):
        result = check(ds)
        assert result["requires_manual_mapping"] is False
        assert result["detected_format"] == "pretokenized"


def test_format_passes_pretokenized_rows_through_untouched():
    ds = _pretokenized()
    info = format_and_template_dataset(ds, model_name = "unsloth/Qwen3-0.6B", tokenizer = None)
    assert info["success"] is True
    assert info["final_format"] == "pretokenized"
    assert info["dataset"].column_names == ds.column_names
    assert info["dataset"].to_dict() == ds.to_dict()


@pytest.mark.parametrize("name", sorted(NORMAL_DATASETS))
def test_normal_datasets_keep_their_route(name):
    ds, expected = NORMAL_DATASETS[name]
    result = hub_check_dataset_format(ds)
    assert result["detected_format"] == expected
    assert check_dataset_format(ds)["detected_format"] == expected
    assert format_dataset(ds, format_type = "auto")["final_format"] != "pretokenized"


@pytest.mark.parametrize(
    "rows",
    [
        {"input_ids": [["a", "b"]]},
        {"input_ids": [[]]},
        {"input_ids": [[1, 2, 3]], "labels": [[1, 2]]},
        {"tokens": [[1, 2, 3]]},
    ],
)
def test_malformed_token_rows_still_ask_for_mapping(rows):
    ds = Dataset.from_dict(rows)
    assert not is_pretokenized_dataset(ds)
    assert hub_check_dataset_format(ds)["requires_manual_mapping"] is True


def test_prepare_truncates_every_token_column_and_is_idempotent():
    ds = _pretokenized()
    once = prepare_pretokenized_dataset(ds, max_seq_length = 3, vocab_size = 100)
    for column in ("input_ids", "attention_mask", "labels"):
        assert max(len(v) for v in once[column]) == 3
    assert once[0]["labels"] == [-100, 2, 3]
    twice = prepare_pretokenized_dataset(once, max_seq_length = 3, vocab_size = 100)
    assert twice.to_dict() == once.to_dict()


def test_prepare_leaves_short_rows_alone():
    ds = _pretokenized()
    assert prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100) is ds


def test_prepare_rejects_ids_outside_the_vocabulary():
    with pytest.raises(ValueError, match = "outside this model's vocabulary"):
        prepare_pretokenized_dataset(_pretokenized(), max_seq_length = 2048, vocab_size = 4)


def test_prepare_rejects_labels_of_another_length():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]], "labels": [[1, 2]]})
    with pytest.raises(ValueError, match = "same length"):
        prepare_pretokenized_dataset(ds, max_seq_length = 2048, vocab_size = 100)


def test_only_user_labels_turn_off_response_masking():
    assert has_custom_labels(_pretokenized(with_labels = True))
    assert not has_custom_labels(_pretokenized(with_labels = False))
    assert not has_custom_labels(_pretokenized(with_labels = False).to_iterable_dataset())


@pytest.mark.parametrize("format_type", ["auto", "alpaca", "chatml", "sharegpt"])
def test_any_format_but_raw_routes_token_ids(format_type):
    info = format_dataset(_pretokenized(), format_type = format_type)
    assert info["final_format"] == "pretokenized"
    assert info["requires_manual_mapping"] is False
