# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import datetime
import json

import pytest
from datasets import Dataset

from utils.datasets import format_dataset

_REPLY = {
    "response": "Lower the batch size; l'OOM vient de là.",
    "followups": [],
    "cites": None,
    "resolved": True,
}
_MAPPINGS = pytest.mark.parametrize(
    "mapping",
    [
        None,
        {"User": "user", "Assistant": "assistant"},
        {"User": "user", "Assistant": "assistant", "__system_prompt": "Be brief."},
    ],
    ids = ["auto_mapping", "user_mapping", "template_mapping"],
)


def _reply(cell, mapping):
    row = Dataset.from_dict({"User": ["Training runs out of memory."], "Assistant": [cell]})
    reply = format_dataset(row, custom_format_mapping = mapping)["dataset"][0]["conversations"][-1]
    assert reply["role"] == "assistant"
    return reply["content"]


@_MAPPINGS
def test_structured_cell_trains_as_json(mapping):
    content = _reply(_REPLY, mapping)
    assert json.loads(content) == _REPLY
    assert "l'OOM vient de là" in content


@_MAPPINGS
def test_non_finite_numbers_in_a_structured_cell_train_as_null(mapping):
    def reject(constant):
        raise ValueError(constant)

    content = _reply({"score": float("nan"), "top": [float("inf"), 1.5]}, mapping)
    assert json.loads(content, parse_constant = reject) == {"score": None, "top": [None, 1.5]}


@_MAPPINGS
def test_list_cell_with_a_date_trains_as_json(mapping):
    cell = [None, {"at": datetime.date(2024, 1, 2)}]
    assert _reply(cell, mapping) == '[null, {"at": "2024-01-02"}]'
