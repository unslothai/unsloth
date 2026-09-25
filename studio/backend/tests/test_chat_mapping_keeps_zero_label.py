# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset

from utils.datasets import format_dataset


def test_chat_mapping_keeps_a_zero_label_and_blanks_a_missing_one():
    # The chat-format twin of test_alpaca_mapping_keeps_a_zero_label_and_blanks_a_missing_one:
    # a label of 0 is an answer, not an empty cell, and a missing label must not train "nan".
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b", "c"], "label": [1, 0, None]}),
        format_type = "auto",
        batch_size = 3,
        custom_format_mapping = {"text": "user", "label": "assistant"},
    )

    replies = [convo[-1] for convo in result["dataset"]["conversations"]]
    assert replies == [
        {"role": "assistant", "content": "1"},
        {"role": "assistant", "content": "0"},
        {"role": "assistant", "content": ""},
    ]


def test_chat_mapping_blanks_a_nan_cell():
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b"], "score": [0.0, float("nan")]}),
        format_type = "auto",
        batch_size = 2,
        custom_format_mapping = {"text": "user", "score": "assistant"},
    )

    replies = [convo[-1]["content"] for convo in result["dataset"]["conversations"]]
    assert replies == ["0.0", ""]
