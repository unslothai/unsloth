# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for the deprecated custom prompt template parameter."""

import pytest
from datasets import Dataset

from utils.datasets.chat_templates import apply_chat_template_to_dataset
from utils.datasets.dataset_utils import format_and_template_dataset


class _Tokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"


def _dataset_info(dataset):
    return {
        "dataset": dataset,
        "detected_format": "alpaca",
        "final_format": "alpaca",
        "chat_column": None,
        "is_standardized": True,
        "warnings": [],
    }


def _alpaca_dataset():
    return Dataset.from_dict(
        {
            "instruction": ["What is 2+2?"],
            "input": [""],
            "output": ["4"],
        }
    )


def test_custom_prompt_template_is_rejected_without_changing_the_dataset():
    dataset = _alpaca_dataset()

    with pytest.deprecated_call(match = "cannot persist a matching template"):
        result = apply_chat_template_to_dataset(
            _dataset_info(dataset),
            _Tokenizer(),
            custom_prompt_template = "### Q: {instruction}\n### A: {output}",
        )

    assert result["success"] is False
    assert "deprecated and unsupported" in result["errors"][0]
    assert result["dataset"] is dataset
    assert result["dataset"].column_names == ["instruction", "input", "output"]
    assert len(result["dataset"]) == 1


def test_default_template_is_unchanged_when_parameter_is_omitted():
    result = apply_chat_template_to_dataset(_dataset_info(_alpaca_dataset()), _Tokenizer())

    assert result["success"] is True
    assert result["errors"] == []
    assert result["dataset"][0]["text"] == (
        "Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nWhat is 2+2?\n\n### Input:\n\n\n### Response:\n4"
    )


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"format_type": "raw"},
        {"is_vlm": True},
    ],
)
def test_main_entry_point_rejects_custom_templates_before_every_format_branch(options):
    dataset = _alpaca_dataset()

    with pytest.deprecated_call(match = "cannot persist a matching template"):
        result = format_and_template_dataset(
            dataset,
            model_name = "Qwen2ForCausalLM",
            tokenizer = _Tokenizer(),
            custom_prompt_template = "{instruction}",
            **options,
        )

    assert result["success"] is False
    assert "deprecated and unsupported" in result["errors"][0]
    assert result["dataset"] is dataset
    assert result["detected_format"] == "unknown"
    assert result["final_format"] == "unknown"
    assert result["requires_manual_mapping"] is False
