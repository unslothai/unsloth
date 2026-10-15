# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest
from datasets import Dataset, IterableDataset, load_dataset

from hub.utils.dataset_format import check_dataset_format, format_dataset_preview
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


@pytest.mark.parametrize("batch_size", [1000, 1])
def test_empty_wrapped_conversation_does_not_break_the_dataset(tmp_path, batch_size):
    rows = [
        {"user_goal": "Nothing came back.", "output_format": {}},
        {"user_goal": "Plan a monthly budget.", "output_format": {"conversation": _CONVERSATION}},
    ]
    path = tmp_path / "recipe.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    dataset = load_dataset("json", data_files = str(path), split = "train")

    result = format_dataset(dataset, batch_size = batch_size)

    assert result["chat_column"] == "output_format"
    assert result["dataset"]["output_format"] == [None, _CONVERSATION]


@_WRAPPERS
def test_preview_shows_the_wrapped_conversation_as_its_turns(wrapped):
    preview = format_dataset_preview(_recipe_output(wrapped))

    assert preview[0]["output_format"] == _CONVERSATION


def test_mapped_empty_wrapped_conversation_adds_no_turns():
    rows = [
        {"output_format": {"conversation": _CONVERSATION}},
        {"output_format": {"conversation": []}},
    ]
    result = format_dataset(
        Dataset.from_list(rows), custom_format_mapping = {"output_format": "assistant"}
    )

    assert result["dataset"]["conversations"] == [_CONVERSATION, []]
