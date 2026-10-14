# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset, IterableDataset

from hub.utils.dataset_format import check_dataset_format
from utils.datasets import format_dataset

_CONVERSATION = [
    {"role": "user", "content": "Help me budget."},
    {"role": "assistant", "content": "What is your monthly income?"},
    {"role": "user", "content": "3000, rent is 1000."},
    {"role": "assistant", "content": "Save 600 and keep 1400 for the rest."},
]
_SHAREGPT = [
    {"from": "human" if turn["role"] == "user" else "gpt", "value": turn["content"]}
    for turn in _CONVERSATION
]
_WRAPPERS = pytest.mark.parametrize(
    "wrapped",
    [{"conversation": _CONVERSATION}, {"messages": _SHAREGPT}],
    ids = ["role_content", "from_value"],
)


def _recipe_output(wrapped):
    return Dataset.from_list([{"user_goal": "Plan a monthly budget.", "output_format": wrapped}])


@_WRAPPERS
def test_check_detects_a_conversation_wrapped_in_a_structured_column(wrapped):
    result = check_dataset_format(_recipe_output(wrapped))

    assert result["detected_format"] in {"chatml", "sharegpt"}
    assert result["chat_column"] == "output_format"
    assert result["suggested_mapping"] is None


@_WRAPPERS
def test_wrapped_conversation_trains_as_its_turns(wrapped):
    result = format_dataset(_recipe_output(wrapped))

    assert result["chat_column"] == "output_format"
    assert result["dataset"][0]["output_format"] == _CONVERSATION


@_WRAPPERS
def test_streamed_wrapped_conversation_trains_as_its_turns(wrapped):
    rows = [{"user_goal": "Plan a monthly budget.", "output_format": wrapped}]
    result = format_dataset(IterableDataset.from_generator(lambda: iter(rows)))

    assert next(iter(result["dataset"]))["output_format"] == _CONVERSATION


@_WRAPPERS
def test_mapped_wrapped_conversation_trains_as_its_turns(wrapped):
    result = format_dataset(
        _recipe_output(wrapped), custom_format_mapping = {"output_format": "assistant"}
    )

    assert result["dataset"][0]["conversations"] == _CONVERSATION


def test_single_field_object_without_messages_still_trains_as_json():
    row = {"question": "Name two colors.", "answer": {"colors": ["red", "blue"]}}
    result = format_dataset(Dataset.from_list([row]))

    assert result["dataset"][0]["conversations"][-1]["content"] == '{"colors": ["red", "blue"]}'
