# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset, IterableDataset

from utils.datasets.raw_text import prepare_raw_text_dataset
from utils.datasets.text_validation import validate_text_sft_dataset


def test_materialized_text_dataset_accepts_non_empty_rows():
    dataset = [{"text": "hello"}, {"text": " world "}]

    assert validate_text_sft_dataset(dataset, split_name = "train") is dataset


def test_materialized_text_dataset_rejects_an_empty_split():
    with pytest.raises(ValueError, match = "the train split is empty"):
        validate_text_sft_dataset([], split_name = "train")


def test_materialized_text_dataset_is_validated_completely():
    dataset = Dataset.from_dict({"text": ["hello", "world", "   "]})

    with pytest.raises(ValueError, match = r"train row 2 has an empty `text` field"):
        validate_text_sft_dataset(dataset, split_name = "train")


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"prompt": "missing"}, "eval row 0 is missing the `text` field"),
        ({"text": None}, "eval row 0 has a non-string `text` field"),
    ],
)
def test_materialized_text_dataset_reports_invalid_field(row, message):
    with pytest.raises(ValueError, match = message):
        validate_text_sft_dataset([row], split_name = "eval")


def test_streaming_text_validation_is_lazy_and_rejects_a_later_blank():
    visited: list[int] = []

    def rows():
        for index, text in enumerate(["first", "second", ""]):
            visited.append(index)
            yield {"text": text}

    dataset = IterableDataset.from_generator(rows)
    validated = validate_text_sft_dataset(dataset, split_name = "train")

    assert visited == []
    iterator = iter(validated)
    assert next(iterator)["text"] == "first"
    assert next(iterator)["text"] == "second"
    with pytest.raises(ValueError, match = r"train row 2 has an empty `text` field"):
        next(iterator)
    assert visited == [0, 1, 2]


def test_streaming_validation_does_not_scan_an_unbounded_invalid_prefix():
    visited = 0

    def rows():
        nonlocal visited
        while True:
            visited += 1
            yield {"text": None}

    dataset = IterableDataset.from_generator(rows)
    result = prepare_raw_text_dataset(
        dataset,
        mode_label = "CPT",
        split_name = "train",
        eos_token = "<eos>",
        append_eos = True,
    )

    # Raw-text column discovery may perform its existing one-row schema probe,
    # but validation must not continue through the unbounded invalid prefix.
    assert visited == 1
    assert result.dataset is not dataset


def test_streaming_raw_text_drops_invalid_prefix_lazily_then_appends_eos():
    visited: list[int] = []

    def rows():
        source = [None, 7, "valid"]
        for index, text in enumerate(source):
            visited.append(index)
            yield {"text": text}

    dataset = IterableDataset.from_generator(rows)
    result = prepare_raw_text_dataset(
        dataset,
        mode_label = "CPT",
        split_name = "train",
        eos_token = "<eos>",
        append_eos = True,
    )

    # Column discovery performs one bounded probe. HF IterableDataset starts a
    # fresh generator when training later consumes the prepared pipeline.
    assert visited == [0]
    assert next(iter(result.dataset))["text"] == "valid<eos>"
    assert visited == [0, 0, 1, 2]


def test_raw_blank_is_rejected_before_eos_append():
    dataset = Dataset.from_dict({"text": ["   "]})

    with pytest.raises(ValueError, match = r"train row 0 has an empty `text` field"):
        prepare_raw_text_dataset(
            dataset,
            mode_label = "CPT",
            split_name = "train",
            eos_token = "<eos>",
            append_eos = True,
        )


def test_streaming_raw_blank_is_rejected_before_eos_append():
    dataset = IterableDataset.from_generator(lambda: iter([{"text": " "}]))
    result = prepare_raw_text_dataset(
        dataset,
        mode_label = "CPT",
        split_name = "train",
        eos_token = "<eos>",
        append_eos = True,
    )

    with pytest.raises(ValueError, match = r"train row 0 has an empty `text` field"):
        next(iter(result.dataset))


def test_vlm_dataset_bypasses_text_validation_without_iteration():
    iterated = False

    def rows():
        nonlocal iterated
        iterated = True
        yield {"image": "pixels", "messages": []}

    dataset = IterableDataset.from_generator(rows)

    assert validate_text_sft_dataset(dataset, is_vlm = True) is dataset
    assert iterated is False
