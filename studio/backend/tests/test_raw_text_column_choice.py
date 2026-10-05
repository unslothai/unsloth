# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets.raw_text import prepare_raw_text_dataset

CODE = "def f():\n    return 1\n" * 20
ROWS = {
    "repo_name": ["pansapiens/mytardis", "a/b"],
    "path": ["tardis/models.py", "x.py"],
    "content": [CODE, CODE + "print(f())\n"],
}


@pytest.mark.parametrize("streaming", [False, True])
def test_raw_text_trains_the_body_column_not_the_first_string_column(streaming):
    dataset = Dataset.from_dict(ROWS)
    if streaming:
        dataset = dataset.to_iterable_dataset()

    result = prepare_raw_text_dataset(dataset, mode_label = "CPT", split_name = "train")

    assert [row["text"] for row in result.dataset] == ROWS["content"]
    assert any("auto-selecting 'content'" in notice.message for notice in result.notices)


def test_raw_text_keeps_the_first_column_when_no_column_is_longer():
    result = prepare_raw_text_dataset(
        Dataset.from_dict({"content": ["101", "102"], "source": ["corpus", "corpus"]})
    )
    assert list(result.dataset["text"]) == ["101", "102"]
