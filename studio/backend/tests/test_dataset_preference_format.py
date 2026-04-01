# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset

from utils.datasets.dataset_utils import (
    check_dataset_format,
    format_and_template_dataset,
    format_dataset,
)


def _preference_dataset():
    return Dataset.from_list(
        [
            {
                "prompt": "The sky is",
                "chosen": " blue.",
                "rejected": " green.",
            }
        ]
    )


def test_check_dataset_format_detects_preference_dataset():
    result = check_dataset_format(_preference_dataset())

    assert result["detected_format"] == "preference"
    assert result["requires_manual_mapping"] is False


def test_format_dataset_auto_preserves_preference_dataset():
    result = format_dataset(_preference_dataset(), format_type = "auto")

    assert result["detected_format"] == "preference"
    assert result["final_format"] == "preference"
    assert result["requires_manual_mapping"] is False
    assert result["dataset"][0]["chosen"] == " blue."


def test_format_dataset_preference_mapping_converts_custom_columns():
    dataset = Dataset.from_list(
        [{"question": "The sky is", "winner": " blue.", "loser": " green."}]
    )

    result = format_dataset(
        dataset,
        format_type = "preference",
        custom_format_mapping = {
            "question": "prompt",
            "winner": "chosen",
            "loser": "rejected",
        },
    )

    assert result["final_format"] == "preference"
    assert result["dataset"][0] == {
        "prompt": "The sky is",
        "chosen": " blue.",
        "rejected": " green.",
    }


def test_format_and_template_dataset_skips_chat_template_for_preference():
    result = format_and_template_dataset(
        _preference_dataset(),
        model_name = "unsloth/Qwen2.5-0.5B-Instruct",
        tokenizer = object(),
        format_type = "auto",
    )

    assert result["success"] is True
    assert result["final_format"] == "preference"
    assert result["dataset"][0] == {
        "prompt": "The sky is",
        "chosen": " blue.",
        "rejected": " green.",
    }
