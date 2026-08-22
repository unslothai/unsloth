# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An Alpaca dataset must be rendered with the caller's prompt template."""

import sys
from pathlib import Path

import pytest
from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets.chat_templates import (  # noqa: E402
    DEFAULT_ALPACA_TEMPLATE,
    apply_chat_template_to_dataset,
)


class _Tokenizer:
    """Carries a chat template so the alpaca-template fallback is not taken."""

    chat_template = "{{ messages }}"
    eos_token = "</s>"


def _dataset_info():
    dataset = Dataset.from_dict(
        {
            "instruction": ["What is 2+2?"],
            "input": [""],
            "output": ["4"],
        }
    )
    return {
        "dataset": dataset,
        "final_format": "alpaca",
        "chat_column": None,
        "is_standardized": True,
    }


def test_custom_prompt_template_is_applied():
    result = apply_chat_template_to_dataset(
        _dataset_info(),
        _Tokenizer(),
        custom_prompt_template = "### Q: {instruction}\n### A: {output}",
    )

    assert result["success"] is True
    assert result["errors"] == []
    assert result["dataset"][0]["text"] == "### Q: What is 2+2?\n### A: 4"


def test_default_template_is_used_when_none_is_given():
    result = apply_chat_template_to_dataset(_dataset_info(), _Tokenizer())

    assert result["success"] is True
    text = result["dataset"][0]["text"]
    assert text.startswith("Below is an instruction that describes a task")
    assert "### Instruction:\nWhat is 2+2?" in text
    assert "### Response:\n4" in text


@pytest.mark.parametrize(
    "template, why",
    [
        ("{instruction} -> {answer}", "KeyError: a field that is not instruction/input/output"),
        ("{instruction[9]}", "IndexError: an index into a field"),
        ("{instruction", "ValueError: an unbalanced brace"),
        ("stray }", "ValueError: a stray closing brace"),
        ("{instruction!z}", "ValueError: an unknown conversion"),
        ("{output:d}", "ValueError: a format spec the value cannot take"),
        ("{instruction.foo}", "AttributeError: an attribute lookup on a field"),
        (DEFAULT_ALPACA_TEMPLATE, "IndexError: the module's own positional template"),
    ],
)
def test_a_template_that_cannot_render_fails_the_call(template, why):
    """One caller-supplied constant that does not render cannot render for any row,
    so it is rejected once, up front, rather than blanking every row.

    Before, only KeyError and IndexError were caught, and they were caught per row:
    the map completed, every row came back empty and the call still reported
    success, so a single typo silently produced a dataset of blank training text.
    The four ValueError shapes and the AttributeError one took the opposite path.
    Now all eight land on the same contract."""
    result = apply_chat_template_to_dataset(
        _dataset_info(),
        _Tokenizer(),
        custom_prompt_template = template,
    )

    assert result["success"] is False, why
    assert "text" not in result["dataset"].column_names
    assert len(result["errors"]) == 1, result["errors"]


def test_an_empty_template_is_rejected_rather_than_meaning_the_default():
    # "" used to be falsy, so it silently rendered DEFAULT_ALPACA_TEMPLATE, and
    # "   " made three spaces the entire training text of every row.
    for template in ("", "   "):
        result = apply_chat_template_to_dataset(
            _dataset_info(),
            _Tokenizer(),
            custom_prompt_template = template,
        )
        assert result["success"] is False, repr(template)
        assert "empty" in result["errors"][0]


def test_one_bad_template_reports_one_error_not_one_per_row():
    rows = 50
    dataset = Dataset.from_dict(
        {
            "instruction": [f"q{i}" for i in range(rows)],
            "input": [""] * rows,
            "output": [f"a{i}" for i in range(rows)],
        }
    )
    info = dict(_dataset_info(), dataset = dataset)

    result = apply_chat_template_to_dataset(info, _Tokenizer(), custom_prompt_template = "{answer}")

    assert result["success"] is False
    assert len(result["errors"]) == 1


def test_a_custom_template_is_not_applied_to_a_chatml_dataset_silently():
    dataset = Dataset.from_dict({"conversations": [[{"role": "user", "content": "hi"}]]})
    info = {
        "dataset": dataset,
        "final_format": "chatml_conversations",
        "chat_column": "conversations",
        "is_standardized": True,
    }

    result = apply_chat_template_to_dataset(
        info, _Tokenizer(), custom_prompt_template = "### Q: {instruction}"
    )

    assert any("only applied to the alpaca format" in w for w in result["warnings"]), result[
        "warnings"
    ]
