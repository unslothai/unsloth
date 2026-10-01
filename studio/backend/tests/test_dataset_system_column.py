# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import format_and_template_dataset


class _TurnTokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"

    def apply_chat_template(self, conversation, **_kwargs):
        return "".join(f"<{turn['role']}>{turn['content']}" for turn in conversation)


def _sharegpt(turns):
    return "conversations", [{"from": role, "value": content} for role, content in turns]


def _chatml(turns):
    return "messages", [{"role": role, "content": content} for role, content in turns]


def _format(rows):
    result = format_and_template_dataset(
        Dataset.from_list(rows), model_name = "stub-model", tokenizer = _TurnTokenizer()
    )
    assert result["success"], result["errors"]
    return list(result["dataset"]["text"])


@pytest.mark.parametrize("shape", [_sharegpt, _chatml], ids = ["sharegpt", "chatml"])
def test_system_column_is_trained_as_the_system_turn(shape):
    column, convo = shape([("user", "q"), ("assistant", "a")])

    texts = _format([{"system": "Here is a list of functions", column: convo}])

    assert texts == ["<system>Here is a list of functions<user>q<assistant>a"]


@pytest.mark.parametrize("shape", [_sharegpt, _chatml], ids = ["sharegpt", "chatml"])
def test_system_column_does_not_override_a_leading_system_turn_or_add_a_blank_one(shape):
    column, with_system = shape([("system", "inline"), ("user", "q"), ("assistant", "a")])
    _, without_system = shape([("user", "q"), ("assistant", "a")])

    texts = _format(
        [
            {"system": "column", column: with_system},
            {"system": "  ", column: without_system},
        ]
    )

    assert texts == ["<system>inline<user>q<assistant>a", "<user>q<assistant>a"]
