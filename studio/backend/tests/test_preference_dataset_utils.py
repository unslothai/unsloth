# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset

from utils.datasets.dataset_utils import format_dataset
from utils.datasets.format_detection import detect_dataset_format


def test_detect_dataset_format_supports_implicit_prompt_preference():
    dataset = Dataset.from_list([{"chosen": " blue.", "rejected": " green."}])

    result = detect_dataset_format(dataset)

    assert result["format"] == "preference"


def test_detect_dataset_format_prefers_chat_structure_over_preference_metadata():
    dataset = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "The sky is"},
                    {"role": "assistant", "content": " blue."},
                ],
                "chosen": " blue.",
                "rejected": " green.",
            }
        ]
    )

    result = detect_dataset_format(dataset)

    assert result["format"] == "chatml"
    assert result["chat_column"] == "messages"


def test_detect_dataset_format_prefers_preference_over_plain_texts_column():
    dataset = Dataset.from_list(
        [{"texts": "The sky is", "chosen": " blue.", "rejected": " green."}]
    )

    result = detect_dataset_format(dataset)

    assert result["format"] == "preference"


def test_format_dataset_preference_mapping_ignores_metadata_keys():
    dataset = Dataset.from_list([{"winner": " blue.", "loser": " green."}])

    result = format_dataset(
        dataset,
        format_type = "preference",
        custom_format_mapping = {
            "winner": "chosen",
            "loser": "rejected",
            "__label_mapping": {"winner": {"1": "preferred"}},
        },
    )

    assert result["final_format"] == "preference"
    assert result["dataset"][0] == {
        "chosen": " blue.",
        "rejected": " green.",
    }
