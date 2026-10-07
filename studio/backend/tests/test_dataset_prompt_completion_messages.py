# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from hub.utils.dataset_format import check_dataset_format
from utils.datasets import format_dataset

_SYSTEM = {"role": "system", "content": "Answer in one word."}


def _user(text):
    return {"role": "user", "content": text}


def _assistant(text):
    return {"role": "assistant", "content": text}


_CASES = [
    pytest.param(
        {"prompt": [_user("What color is the sky?")], "completion": [_assistant("Blue")]},
        [_user("What color is the sky?"), _assistant("Blue")],
        id = "trl",
    ),
    pytest.param(
        {"prompt": [_SYSTEM, _user("Sky?")], "completion": [_assistant("Blue")]},
        [_SYSTEM, _user("Sky?"), _assistant("Blue")],
        id = "system-in-prompt",
    ),
    pytest.param(
        {
            "prompt": [_user("Hi"), _assistant("Hello"), _user("Sky?")],
            "completion": [_assistant("Blue")],
        },
        [_user("Hi"), _assistant("Hello"), _user("Sky?"), _assistant("Blue")],
        id = "multi-turn-prompt",
    ),
    pytest.param(
        {
            "prompt": [_user("Sky?")],
            "completion": [_assistant("Blue"), _user("Grass?"), _assistant("Green")],
        },
        [_user("Sky?"), _assistant("Blue"), _user("Grass?"), _assistant("Green")],
        id = "multi-message-completion",
    ),
    pytest.param(
        {"prompt": "Sky?", "completion": [_assistant("Blue")]},
        [_user("Sky?"), _assistant("Blue")],
        id = "text-prompt",
    ),
    pytest.param(
        {"prompt": [_SYSTEM, _user("Sky?")], "completion": "Blue"},
        [_SYSTEM, _user("Sky?"), _assistant("Blue")],
        id = "text-completion",
    ),
]

_MAPPING = {"prompt": "user", "completion": "assistant"}


@pytest.mark.parametrize("mapping", [None, _MAPPING], ids = ["auto_mapping", "user_mapping"])
@pytest.mark.parametrize("row, conversation", _CASES)
def test_prompt_completion_messages_train_as_one_conversation(row, conversation, mapping):
    result = format_dataset(Dataset.from_list([row]), custom_format_mapping = mapping)

    assert result["chat_column"] == "conversations"
    assert result["dataset"][0]["conversations"] == conversation


@pytest.mark.parametrize("row, conversation", _CASES)
def test_check_suggests_prompt_and_completion_together(row, conversation):
    result = check_dataset_format(Dataset.from_list([row]))

    assert result["detected_format"] == "custom_heuristic"
    assert result["suggested_mapping"] == _MAPPING


@pytest.mark.parametrize("prompt", [[_user("Sky?")], "Sky?"], ids = ["messages", "text"])
def test_prompt_completion_pair_wins_over_another_chat_column(prompt):
    row = {
        "prompt": prompt,
        "completion": [_assistant("Blue")],
        "history": [_user("Hi"), _assistant("Hello")],
    }
    dataset = Dataset.from_list([row])

    assert check_dataset_format(dataset)["suggested_mapping"] == _MAPPING
    result = format_dataset(dataset)
    assert result["dataset"][0]["conversations"] == [_user("Sky?"), _assistant("Blue")]


@pytest.mark.parametrize("chat_column", ["messages", "conversations", "texts"])
def test_prompt_completion_pair_wins_over_an_exact_chat_column(chat_column):
    row = {
        "prompt": [_user("Sky?")],
        "completion": [_assistant("Blue")],
        chat_column: [_user("Unrelated?"), _assistant("Unrelated")],
    }
    dataset = Dataset.from_list([row])

    assert check_dataset_format(dataset)["suggested_mapping"] == _MAPPING
    result = format_dataset(dataset)
    assert result["dataset"][0]["conversations"] == [_user("Sky?"), _assistant("Blue")]


def test_plain_prompt_completion_keeps_existing_column_priority():
    question = "This longer question column keeps its prior user-role priority over prompt text."
    dataset = Dataset.from_list([{"prompt": "p", "completion": "c", "question": question}])

    assert check_dataset_format(dataset)["suggested_mapping"] == {
        "completion": "assistant",
        "question": "user",
    }
    result = format_dataset(dataset)
    assert result["dataset"][0]["conversations"] == [_user(question), _assistant("c")]


@pytest.mark.parametrize("mapping", [None, {"question": "user", "answer": "assistant"}])
def test_empty_generic_list_remains_training_text(mapping):
    rows = [
        {"question": "Return an empty JSON array", "answer": []},
        {"question": "Return one labelled item", "answer": [{"label": "x"}]},
    ]

    result = format_dataset(Dataset.from_list(rows), custom_format_mapping = mapping)

    assert result["dataset"][0]["conversations"] == [
        _user("Return an empty JSON array"),
        _assistant("[]"),
    ]


_CALL = {"id": "c1", "type": "function", "function": {"name": "weather", "arguments": "{}"}}


@pytest.mark.parametrize("mapping", [None, _MAPPING], ids = ["auto_mapping", "user_mapping"])
def test_tool_calls_and_empty_message_lists(mapping):
    rows = [
        {
            "prompt": [_user("Weather?")],
            "completion": [
                {"role": "assistant", "content": None, "tool_calls": [_CALL]},
                {"role": "tool", "content": "Sunny", "tool_call_id": "c1"},
                _assistant("Sunny"),
            ],
        },
        {"prompt": [_user("Sky?")], "completion": []},
    ]
    result = format_dataset(Dataset.from_list(rows), custom_format_mapping = mapping)

    conversations = [
        [{key: value for key, value in turn.items() if value is not None} for turn in row]
        for row in result["dataset"]["conversations"]
    ]
    assert conversations == [
        [
            _user("Weather?"),
            {"role": "assistant", "tool_calls": [_CALL]},
            {"role": "tool", "content": "Sunny", "tool_call_id": "c1"},
            _assistant("Sunny"),
        ],
        [_user("Sky?")],
    ]


@pytest.mark.parametrize("mapping", [None, _MAPPING], ids = ["auto_mapping", "user_mapping"])
def test_sharegpt_message_lists_train_as_one_conversation(mapping):
    row = {
        "prompt": [{"from": "system", "value": "Be brief."}, {"from": "human", "value": "Sky?"}],
        "completion": [{"from": "gpt", "value": "Blue"}],
    }
    result = format_dataset(Dataset.from_list([row]), custom_format_mapping = mapping)

    assert result["dataset"][0]["conversations"] == [
        {"role": "system", "content": "Be brief."},
        _user("Sky?"),
        _assistant("Blue"),
    ]
