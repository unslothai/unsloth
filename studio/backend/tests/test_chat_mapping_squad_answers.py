# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset, format_dataset
from utils.datasets.cells import cell_text


def _squad_rows():
    return Dataset.from_dict(
        {
            "context": ["Architecturally, the school has a Catholic character."],
            "question": ["To whom did the Virgin Mary allegedly appear in 1858?"],
            "answers": [{"text": ["Saint Bernadette Soubirous"], "answer_start": [515]}],
        }
    )


_MAPPING = {"context": "system", "question": "user", "answers": "assistant"}


@pytest.mark.parametrize("mapping", [None, _MAPPING], ids = ["auto_mapping", "user_mapping"])
def test_chat_mapping_trains_the_squad_answer_not_the_dict(mapping):
    result = format_dataset(_squad_rows(), custom_format_mapping = mapping)

    reply = result["dataset"][0]["conversations"][-1]
    assert reply == {"role": "assistant", "content": "Saint Bernadette Soubirous"}


class _TurnTokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"

    def apply_chat_template(self, conversation, **_kwargs):
        return "".join(f"<{turn['role']}>{turn['content']}" for turn in conversation)


def test_template_mapping_trains_the_squad_answer_not_the_dict():
    info = {
        "dataset": _squad_rows(),
        "final_format": "unknown",
        "chat_column": None,
        "is_standardized": False,
        "warnings": [],
        "custom_format_mapping": _MAPPING,
    }
    result = apply_chat_template_to_dataset(info, _TurnTokenizer(), custom_format_mapping = _MAPPING)

    assert result["success"], result["errors"]
    assert result["dataset"]["text"][0].endswith("<assistant>Saint Bernadette Soubirous")


def test_cell_text_reads_the_text_of_an_answer_dict():
    assert cell_text({"text": ["first", "second"], "answer_start": [1, 9]}) == "first"
    assert cell_text({"text": [], "answer_start": []}) == ""
    assert cell_text({"label": 1}) == "{'label': 1}"
    assert cell_text({"label": ["A", "B"], "text": ["x", "y"]}) == (
        "{'label': ['A', 'B'], 'text': ['x', 'y']}"
    )
