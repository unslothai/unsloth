# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

import utils.hardware
from utils.datasets import apply_chat_template_to_dataset, format_and_template_dataset


class _StrictTokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"

    def apply_chat_template(self, conversation, **_kwargs):
        if any(turn["role"] == "system" for turn in conversation):
            raise ValueError("System role not supported")
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)


def _convo(index, with_system = False):
    turns = [
        {"role": "user", "content": f"question {index}"},
        {"role": "assistant", "content": f"answer {index}"},
    ]
    if with_system:
        turns.insert(0, {"role": "system", "content": "be brief"})
    return turns


def _dataset_info(dataset):
    return {
        "dataset": dataset,
        "detected_format": "chatml_messages",
        "final_format": "chatml_messages",
        "chat_column": "messages",
        "is_standardized": True,
        "warnings": [],
    }


def _mixed_dataset():
    return Dataset.from_dict(
        {"messages": [_convo(i, with_system = i in (1, 4, 5)) for i in range(8)]}
    )


@pytest.mark.parametrize("num_proc", [None, 2])
def test_rows_whose_template_raised_are_dropped_and_counted(monkeypatch, num_proc):
    monkeypatch.setattr(utils.hardware, "dataset_map_num_proc", lambda *_a, **_k: num_proc)

    result = apply_chat_template_to_dataset(_dataset_info(_mixed_dataset()), _StrictTokenizer())

    assert result["success"] is True
    formatted = result["dataset"]
    assert len(formatted) == 5
    assert [text.splitlines()[0] for text in formatted["text"]] == [
        f"user: question {i}" for i in (0, 2, 3, 6, 7)
    ]
    assert formatted.column_names == ["messages", "text"]
    assert result["dropped_rows_warning"] == (
        "Dropped 3 of 8 rows because the chat template failed: System role not supported"
    )
    assert result["dropped_rows_warning"] in result["warnings"]


def test_clean_dataset_reports_no_dropped_rows():
    dataset = Dataset.from_dict({"messages": [_convo(i) for i in range(3)]})

    result = apply_chat_template_to_dataset(_dataset_info(dataset), _StrictTokenizer())

    assert result["success"] is True
    assert len(result["dataset"]) == 3
    assert result["dataset"].column_names == ["messages", "text"]
    assert result.get("dropped_rows_warning") is None


def test_every_row_failing_is_an_error():
    dataset = Dataset.from_dict({"messages": [_convo(i, with_system = True) for i in range(3)]})

    result = apply_chat_template_to_dataset(_dataset_info(dataset), _StrictTokenizer())

    assert result["success"] is False
    assert result["errors"] == ["Chat template failed on all 3 rows: System role not supported"]
    assert result["dataset"] is dataset


def test_streaming_rows_whose_template_raised_are_filtered():
    result = apply_chat_template_to_dataset(
        _dataset_info(_mixed_dataset().to_iterable_dataset()), _StrictTokenizer()
    )

    assert result["success"] is True
    rows = list(result["dataset"])
    assert len(rows) == 5
    assert all(set(row) == {"messages", "text"} and row["text"] for row in rows)


def test_format_and_template_dataset_passes_the_dropped_rows_warning_through():
    result = format_and_template_dataset(
        _mixed_dataset(), model_name = "stub-model", tokenizer = _StrictTokenizer()
    )

    assert result["success"] is True
    assert len(result["dataset"]) == 5
    assert result["dropped_rows_warning"].startswith("Dropped 3 of 8 rows")
