# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset

from utils.datasets import format_dataset

PROMPT = "Classify the sentiment."
LABEL_MAPPING = {"label": {"0": "neg", "1": "pos"}}


def _dataset():
    return Dataset.from_dict({"text": ["Loved it", "Hated it"], "label": [1, 0]})


def _format(format_type, mapping):
    return format_dataset(
        _dataset(),
        format_type = format_type,
        batch_size = 2,
        custom_format_mapping = {"text": "instruction", "label": "output", **mapping},
    )


def test_alpaca_mapping_keeps_label_names_and_system_prompt():
    result = _format("alpaca", {"__label_mapping": LABEL_MAPPING, "__system_prompt": PROMPT})

    assert result["final_format"] == "alpaca", result["warnings"]
    rows = result["dataset"]
    assert list(rows["output"]) == ["pos", "neg"]
    assert list(rows["instruction"]) == [f"{PROMPT}\n\nLoved it", f"{PROMPT}\n\nHated it"]
    assert list(rows["input"]) == ["", ""]


def test_alpaca_mapping_uses_system_prompt_alone_for_a_blank_instruction():
    result = format_dataset(
        Dataset.from_dict({"text": ["Loved it", ""], "label": [1, 0]}),
        format_type = "alpaca",
        batch_size = 2,
        custom_format_mapping = {"text": "instruction", "label": "output", "__system_prompt": PROMPT},
    )

    assert result["final_format"] == "alpaca", result["warnings"]
    assert list(result["dataset"]["instruction"]) == [f"{PROMPT}\n\nLoved it", PROMPT]


def test_alpaca_mapping_keeps_a_zero_label_and_blanks_a_missing_one():
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b", "c"], "label": [1, 0, None]}),
        format_type = "alpaca",
        batch_size = 3,
        custom_format_mapping = {"text": "instruction", "label": "output"},
    )

    assert list(result["dataset"]["output"]) == ["1", "0", ""]


def test_alpaca_mapping_keeps_every_column_the_advisor_assigns_to_a_role():
    result = format_dataset(
        Dataset.from_dict({"premise": ["P1", "P2"], "hypothesis": ["H1", "H2"], "label": [0, 1]}),
        format_type = "alpaca",
        batch_size = 2,
        custom_format_mapping = {
            "premise": "instruction",
            "hypothesis": "instruction",
            "label": "output",
            "__label_mapping": LABEL_MAPPING,
            "__system_prompt": PROMPT,
        },
    )

    assert result["final_format"] == "alpaca", result["warnings"]
    assert list(result["dataset"]["instruction"]) == [
        f"{PROMPT}\n\nP1\nH1",
        f"{PROMPT}\n\nP2\nH2",
    ]
    assert list(result["dataset"]["output"]) == ["neg", "pos"]


def test_alpaca_mapping_renders_list_and_text_struct_cells_as_text():
    result = format_dataset(
        Dataset.from_dict(
            {
                "text": ["a", "b"],
                "labels": [[0], [1, 2]],
                "answers": [{"text": ["Paris", "paris"]}, {"text": ["Rome"]}],
            }
        ),
        format_type = "alpaca",
        batch_size = 2,
        custom_format_mapping = {"text": "instruction", "labels": "input", "answers": "output"},
    )

    assert result["final_format"] == "alpaca", result["warnings"]
    assert list(result["dataset"]["input"]) == ["0", "1, 2"]
    assert list(result["dataset"]["output"]) == ["Paris", "Rome"]


def test_alpaca_mapping_names_each_label_in_a_list_cell():
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b"], "labels": [[0], [1, 2]]}),
        format_type = "alpaca",
        batch_size = 2,
        custom_format_mapping = {
            "text": "instruction",
            "labels": "output",
            "__label_mapping": {"labels": {"0": "joy", "1": "anger", "2": "fear"}},
        },
    )

    assert result["final_format"] == "alpaca", result["warnings"]
    assert list(result["dataset"]["output"]) == ["joy", "anger, fear"]


def test_chatml_advisor_mapping_adds_system_turn_and_label_names():
    result = _format("chatml", {"__label_mapping": LABEL_MAPPING, "__system_prompt": PROMPT})

    assert result["final_format"] == "chatml_conversations", result["warnings"]
    assert list(result["dataset"]["conversations"]) == [
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "Loved it"},
            {"role": "assistant", "content": "pos"},
        ],
        [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "Hated it"},
            {"role": "assistant", "content": "neg"},
        ],
    ]


def test_chatml_advisor_mapping_renders_list_cells_as_text():
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b"], "labels": [[0], [1, 2]]}),
        format_type = "chatml",
        batch_size = 2,
        custom_format_mapping = {"text": "user", "labels": "assistant", "__system_prompt": PROMPT},
    )

    assert result["final_format"] == "chatml_conversations", result["warnings"]
    assert [convo[-1]["content"] for convo in result["dataset"]["conversations"]] == ["0", "1, 2"]
