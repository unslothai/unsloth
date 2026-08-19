# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An Alpaca dataset must be rendered with the caller's prompt template."""

import sys
from pathlib import Path

from datasets import Dataset

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from utils.datasets.chat_templates import apply_chat_template_to_dataset  # noqa: E402


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


def test_custom_prompt_template_with_an_unknown_field_does_not_break_the_map():
    result = apply_chat_template_to_dataset(
        _dataset_info(),
        _Tokenizer(),
        custom_prompt_template = "{instruction} -> {answer}",
    )

    # The per-row handler catches the unknown field, so the map still completes
    # and that row comes back blank. Had the error escaped instead, the outer
    # handler would have returned success False with no "text" column at all.
    #
    # The recorded message in result["errors"] is deliberately not asserted on:
    # `errors` is filled through a closure, and `dataset.map` runs the formatter
    # in a worker process whenever num_proc is set, so those appends never reach
    # this process. That is pre-existing behaviour of the errors list, not
    # something this change introduces.
    assert result["success"] is True
    assert result["dataset"][0]["text"] == ""
