# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset, IterableDataset

from utils.datasets.raw_text import prepare_raw_text_dataset


def test_streaming_raw_text_drops_invalid_prefix_lazily_then_appends_eos():
    visited: list[int] = []

    def rows():
        for index, text in enumerate([None, "  ", "valid"]):
            visited.append(index)
            yield {"text": text}

    result = prepare_raw_text_dataset(
        IterableDataset.from_generator(rows),
        mode_label = "CPT",
        split_name = "train",
        eos_token = "<eos>",
        append_eos = True,
    )

    # Column discovery reads one row; the filter must not scan further until training.
    assert visited == [0]
    assert next(iter(result.dataset))["text"] == "valid<eos>"


@pytest.mark.parametrize("streaming", [False, True])
def test_raw_blanks_are_dropped_before_eos_append(streaming):
    texts = ["", "   ", None, "hello", "\n", "world"]
    dataset = (
        IterableDataset.from_generator(lambda: ({"text": text} for text in texts))
        if streaming
        else Dataset.from_dict({"text": texts})
    )
    result = prepare_raw_text_dataset(
        dataset,
        mode_label = "CPT",
        split_name = "train",
        eos_token = "<eos>",
        append_eos = True,
    )
    assert [row["text"] for row in result.dataset] == ["hello<eos>", "world<eos>"]
    assert any("blank" in notice.message for notice in result.notices)


def test_all_blank_raw_eval_split_is_left_for_the_trainer_to_skip():
    result = prepare_raw_text_dataset(
        Dataset.from_dict({"text": ["", "  "]}),
        mode_label = "CPT",
        split_name = "eval",
        eos_token = "<eos>",
        append_eos = True,
    )
    assert len(result.dataset) == 0
    assert any("blank" in notice.message for notice in result.notices)


def test_all_blank_raw_train_split_still_fails():
    with pytest.raises(ValueError, match = "at least one non-blank string"):
        prepare_raw_text_dataset(Dataset.from_dict({"text": ["", "  "]}), split_name = "train")
