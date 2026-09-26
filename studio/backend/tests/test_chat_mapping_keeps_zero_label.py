# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets import apply_chat_template_to_dataset, format_dataset


def test_chat_mapping_keeps_a_zero_label_and_blanks_a_missing_one():
    # A label of 0 is an answer, not an empty cell; a missing label must not train "nan".
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b", "c"], "label": [1, 0, None]}),
        format_type = "auto",
        batch_size = 3,
        custom_format_mapping = {"text": "user", "label": "assistant"},
    )

    replies = [convo[-1] for convo in result["dataset"]["conversations"]]
    assert replies == [
        {"role": "assistant", "content": "1"},
        {"role": "assistant", "content": "0"},
        {"role": "assistant", "content": ""},
    ]


def test_chat_mapping_blanks_a_nan_cell():
    result = format_dataset(
        Dataset.from_dict({"text": ["a", "b"], "score": [0.0, float("nan")]}),
        format_type = "auto",
        batch_size = 2,
        custom_format_mapping = {"text": "user", "score": "assistant"},
    )

    replies = [convo[-1]["content"] for convo in result["dataset"]["conversations"]]
    assert replies == ["0.0", ""]


class _TurnTokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"

    def apply_chat_template(self, conversation, **_kwargs):
        return "".join(f"<{turn['role']}>{turn['content']}" for turn in conversation)


_PERSONA = ["be brief", "  keep the spaces  ", "unicode café 日本"]
_QUESTION = ["What is 2+2?", "multi\nline", "x"]
_ORDINARY_ROWS = {
    "strings": {"persona": _PERSONA, "question": _QUESTION, "response": ["4", "ok", "yes"]},
    "ints": {"persona": _PERSONA, "question": _QUESTION, "response": [4, -7, 42]},
    "floats": {"persona": _PERSONA, "question": _QUESTION, "response": [1.5, -3.0, 1e-05]},
}
_ROLES = {"persona": "system", "question": "user", "response": "assistant"}


def _render(path, columns, mapping):
    dataset = Dataset.from_dict(columns)
    if path == "format_dataset_user_mapping":
        info = format_dataset(dataset, custom_format_mapping = mapping)
    elif path == "format_dataset_auto_mapping":
        info = format_dataset(dataset)
    else:
        info = {
            "dataset": dataset,
            "final_format": "unknown",
            "chat_column": None,
            "is_standardized": False,
            "warnings": [],
        }
        if path == "template_user_mapping":
            info["custom_format_mapping"] = mapping
    result = apply_chat_template_to_dataset(info, _TurnTokenizer(), custom_format_mapping = mapping)
    assert result["success"], result["errors"]
    return result["dataset"]["text"]


_PATHS = [
    "format_dataset_user_mapping",
    "format_dataset_auto_mapping",
    "template_user_mapping",
    "template_auto_mapping",
]


@pytest.mark.parametrize("path", _PATHS)
@pytest.mark.parametrize("kind", list(_ORDINARY_ROWS))
def test_chat_mapping_renders_ordinary_cells_as_before(path, kind):
    columns = _ORDINARY_ROWS[kind]
    assert _render(path, columns, _ROLES) == [
        f"<system>{p}<user>{q}<assistant>{r}" for p, q, r in zip(*columns.values())
    ]


@pytest.mark.parametrize("path", [p for p in _PATHS if p != "format_dataset_auto_mapping"])
def test_chat_mapping_renders_several_columns_per_role_as_before(path):
    columns = {
        "sys": ["s0", "s1"],
        "q": ["q0", "q1"],
        "ctx": ["c0", "c1"],
        "a": ["a0", "a1"],
        "n": [3, 5],
    }
    mapping = {"sys": "system", "q": "user", "ctx": "user", "a": "assistant", "n": "assistant"}
    assert _render(path, columns, mapping) == [
        "<system>s0<user>q0<user>c0<assistant>a0<assistant>3",
        "<system>s1<user>q1<user>c1<assistant>a1<assistant>5",
    ]
