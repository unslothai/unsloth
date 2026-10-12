# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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


def _recipe_rows():
    return Dataset.from_dict({"User": ["Training runs out of memory."], "Assistant": [_REPLY]})


@pytest.mark.parametrize(
    "mapping",
    [None, {"User": "user", "Assistant": "assistant"}],
    ids = ["auto_mapping", "user_mapping"],
)
def test_structured_cell_trains_as_json(mapping):
    result = format_dataset(_recipe_rows(), custom_format_mapping = mapping)

    reply = result["dataset"][0]["conversations"][-1]
    assert reply["role"] == "assistant"
    assert json.loads(reply["content"]) == _REPLY
    assert "l'OOM vient de là" in reply["content"]
