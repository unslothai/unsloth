# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset, IterableDataset

from utils.datasets import convert_chatml_to_alpaca, format_and_template_dataset

SHAREGPT_ROLES = {"system": "system", "user": "human", "assistant": "gpt"}


class _Tokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"


def _messages(row):
    return [
        {"role": "system", "content": f"SYS-{row}"},
        {"role": "user", "content": f"Q1-{row}"},
        {"role": "assistant", "content": f"A1-{row}"},
        {"role": "user", "content": f"Q2-{row}"},
        {"role": "assistant", "content": f"A2-{row}"},
        {"role": "user", "content": f"Q3-{row}"},
        {"role": "assistant", "content": f"A3-{row}"},
    ]


def _sharegpt(row):
    return [
        {"from": SHAREGPT_ROLES[message["role"]], "value": message["content"]}
        for message in _messages(row)
    ]


def _expected(row):
    first = f"User: Q1-{row}\nAssistant: A1-{row}"
    second = f"User: Q2-{row}\nAssistant: A2-{row}"
    return [
        {"instruction": f"Q1-{row}", "input": f"SYS-{row}", "output": f"A1-{row}"},
        {"instruction": f"Q2-{row}", "input": f"SYS-{row}\n\n{first}", "output": f"A2-{row}"},
        {
            "instruction": f"Q3-{row}",
            "input": f"SYS-{row}\n\n{first}\n{second}",
            "output": f"A3-{row}",
        },
    ]


@pytest.mark.parametrize(
    ("column", "conversation"),
    [("messages", _messages), ("conversations", _sharegpt)],
)
def test_every_reply_becomes_a_row_with_the_system_prompt_and_earlier_exchanges(
    column, conversation
):
    dataset = Dataset.from_dict({column: [conversation(1), conversation(2)]})

    converted = convert_chatml_to_alpaca(dataset, batch_size = 1, num_proc = 1, chat_column = column)

    assert converted.column_names == ["instruction", "input", "output"]
    assert converted.to_list() == _expected(1) + _expected(2)


def test_a_single_exchange_converts_as_before():
    dataset = Dataset.from_dict(
        {
            "messages": [
                [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            ]
        }
    )

    converted = convert_chatml_to_alpaca(dataset, num_proc = 1)

    assert converted.select_columns(["instruction", "input", "output"]).to_list() == [
        {"instruction": "Hello", "input": "", "output": "Hi there"}
    ]


def test_unpaired_empty_and_unknown_turns_are_skipped():
    dataset = Dataset.from_dict(
        {
            "messages": [
                None,
                [],
                [
                    {"role": "assistant", "content": "orphan"},
                    {"role": "user", "content": "Q"},
                    {"role": "tool", "content": "tool output"},
                    {"role": "user", "content": "more"},
                    {"role": "assistant", "content": ""},
                    {"role": "assistant", "content": "A"},
                    {"role": "assistant", "content": "A again"},
                    {"role": "user", "content": "trailing"},
                ],
            ]
        }
    )

    converted = convert_chatml_to_alpaca(dataset, num_proc = 1)

    assert converted.to_list() == [
        {"instruction": "Q\n\nmore", "input": "", "output": "A\n\nA again"}
    ]


def test_parallel_conversion_keeps_every_reply_in_order(monkeypatch):
    monkeypatch.setattr("utils.hardware.dataset_map_num_proc", lambda num_proc = None: num_proc)
    dataset = Dataset.from_dict({"messages": [_messages(row) for row in range(1, 9)]})

    converted = convert_chatml_to_alpaca(dataset, batch_size = 1, num_proc = 2)

    assert converted.to_list() == [line for row in range(1, 9) for line in _expected(row)]


@pytest.mark.parametrize(
    "stream",
    [
        lambda rows: IterableDataset.from_generator(lambda: iter(rows)),
        lambda rows: Dataset.from_list(rows).to_iterable_dataset(),
    ],
    ids = ["generator", "typed"],
)
def test_streaming_conversion_splits_replies_the_same_way(stream):
    rows = [{"id": row, "messages": _messages(row)} for row in (1, 2)]

    converted = convert_chatml_to_alpaca(stream(rows), batch_size = 1)

    assert list(converted) == _expected(1) + _expected(2)


def test_structured_content_converts_to_text():
    structured = [
        {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}
        for message in _messages(1)
    ]
    dataset = Dataset.from_dict({"messages": [structured]})

    converted = convert_chatml_to_alpaca(dataset, num_proc = 1)

    assert converted.to_list() == _expected(1)


def test_alpaca_format_with_a_processor_keeps_the_system_prompt():
    class _Processor(_Tokenizer):
        image_processor = object()

    dataset = Dataset.from_dict({"conversations": [_sharegpt(1)]})

    result = format_and_template_dataset(
        dataset,
        model_name = "Gemma3ForConditionalGeneration",
        tokenizer = _Processor(),
        format_type = "alpaca",
        batch_size = 1,
        num_proc = 1,
    )

    assert result["success"] is True
    assert result["final_format"] == "alpaca"
    texts = list(result["dataset"]["text"])
    assert len(texts) == 3
    assert "### Input:\nSYS-1\n\nUser: Q1-1\nAssistant: A1-1\n" in texts[2]


def test_alpaca_format_trains_on_every_sharegpt_exchange():
    dataset = Dataset.from_dict({"conversations": [_sharegpt(1), _sharegpt(2)]})

    result = format_and_template_dataset(
        dataset,
        model_name = "Qwen2ForCausalLM",
        tokenizer = _Tokenizer(),
        format_type = "alpaca",
        batch_size = 1,
        num_proc = 1,
    )

    assert result["success"] is True
    assert result["detected_format"] == "sharegpt"
    assert result["final_format"] == "alpaca"
    texts = list(result["dataset"]["text"])
    assert len(texts) == 6
    assert texts[5].endswith(
        "### Instruction:\nQ3-2\n\n### Input:\nSYS-2\n\nUser: Q1-2\nAssistant: A1-2\n"
        "User: Q2-2\nAssistant: A2-2\n\n### Response:\nA3-2</s>"
    )
