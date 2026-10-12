# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest
from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from hub.utils import dataset_format  # noqa: E402
from utils.datasets import format_detection  # noqa: E402

_CHATML = [
    {"role": "user", "content": "a"},
    {"role": "assistant", "content": "b"},
]
_SHAREGPT = [
    {"from": "human", "value": "a"},
    {"from": "gpt", "value": "b"},
]

_CASES = [
    pytest.param([{"messages": _CHATML}], "chatml", "messages", id = "messages"),
    pytest.param([{"conversations": _SHAREGPT}], "sharegpt", "conversations", id = "conversations"),
    pytest.param([{"texts": _CHATML}], "chatml", "texts", id = "texts"),
    pytest.param([{"dialog": _CHATML}], "chatml", "dialog", id = "dialog"),
    pytest.param([{"conversation": _SHAREGPT}], "sharegpt", "conversation", id = "conversation"),
    pytest.param([{"id": "1", "agent_trace": _CHATML}], "chatml", "agent_trace", id = "agent_trace"),
    pytest.param(
        [{"messages": []}, {"messages": _CHATML}], "chatml", "messages", id = "empty-first-row"
    ),
]


@pytest.mark.parametrize("rows, expected_format, expected_column", _CASES)
def test_check_accepts_the_chat_column_the_trainer_uses(rows, expected_format, expected_column):
    dataset = Dataset.from_list(rows)
    trainer = format_detection.detect_dataset_format(dataset)
    result = dataset_format.check_dataset_format(dataset)

    assert trainer["format"] == expected_format
    assert trainer["chat_column"] == expected_column
    assert result["requires_manual_mapping"] is False
    assert result["detected_format"] == expected_format
    assert result["chat_column"] == expected_column


def test_alpaca_still_passes_without_a_chat_column():
    dataset = Dataset.from_list([{"instruction": "a", "input": "", "output": "b"}])
    result = dataset_format.check_dataset_format(dataset)

    assert result["requires_manual_mapping"] is False
    assert result["detected_format"] == "alpaca"
    assert result["chat_column"] is None


def test_plain_columns_still_fall_back_to_the_heuristic():
    dataset = Dataset.from_list([{"question": "a", "answer": "b"}])
    result = dataset_format.check_dataset_format(dataset)

    assert result["detected_format"] == "custom_heuristic"
    assert result["suggested_mapping"] == {"question": "user", "answer": "assistant"}


def test_preview_standardizes_a_sharegpt_column_under_another_name():
    dataset = Dataset.from_list([{"conversation": _SHAREGPT}])
    preview = dataset_format.format_dataset_preview(dataset)

    assert preview[0]["conversation"] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]
