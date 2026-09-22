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
    result = _format(
        "alpaca", {"__label_mapping": LABEL_MAPPING, "__system_prompt": PROMPT}
    )

    assert result["final_format"] == "alpaca", result["warnings"]
    rows = result["dataset"]
    assert list(rows["output"]) == ["pos", "neg"]
    assert all(PROMPT in instruction for instruction in rows["instruction"])
    assert list(rows["instruction"]) == [
        f"{PROMPT}\n\nLoved it",
        f"{PROMPT}\n\nHated it",
    ]
    assert list(rows["input"]) == ["", ""]


def test_alpaca_mapping_keeps_system_prompt_without_label_mapping():
    result = _format("alpaca", {"__system_prompt": PROMPT})

    assert result["final_format"] == "alpaca", result["warnings"]
    rows = result["dataset"]
    assert all(PROMPT in instruction for instruction in rows["instruction"])
    assert list(rows["output"]) == ["1", "0"]


def test_alpaca_mapping_keeps_an_integer_zero_label():
    result = _format("alpaca", {})

    assert list(result["dataset"]["output"]) == ["1", "0"]


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


def test_alpaca_mapping_renders_list_cells_as_text_across_batches():
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b", "c", "d"], "labels": [[0], [3], [1, 2], [4]]}),
        format_type = "alpaca",
        batch_size = 2,
        custom_format_mapping = {"text": "instruction", "labels": "output"},
    )

    assert result["final_format"] == "alpaca", result["warnings"]
    assert list(result["dataset"]["output"]) == ["0", "3", "1, 2", "4"]


def test_chatml_mapping_with_advisor_keys_is_unchanged():
    result = _format(
        "chatml", {"__label_mapping": LABEL_MAPPING, "__system_prompt": PROMPT}
    )

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
