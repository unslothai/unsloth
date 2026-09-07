# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression coverage for the deprecated custom prompt template parameter."""

import inspect
import warnings

import pytest
from datasets import Dataset, Features, IterableDataset, Value

from utils.datasets import apply_chat_template_to_dataset, format_and_template_dataset


class _Tokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"

    def apply_chat_template(self, conversation, **_kwargs):
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)


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


@pytest.mark.parametrize(
    "template",
    [
        "{instruction}\n{input}\n{output}",
        "{instruction} -> {answer}",
        "{}\n{}\n{}",
        "{0}\n{1}\n{2}",
        "{instruction",
        "stray }",
        "{{instruction}}",
        "{instruction} / {instruction} => {output}",
        "",
        "   ",
    ],
)
def test_custom_prompt_template_is_rejected_without_changing_the_dataset(template):
    dataset = _alpaca_dataset()

    with pytest.deprecated_call(match = "cannot persist a matching template"):
        result = apply_chat_template_to_dataset(
            _dataset_info(dataset),
            _Tokenizer(),
            custom_prompt_template = template,
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


@pytest.mark.parametrize("include_input", [True, False])
def test_default_alpaca_path_preserves_rows_columns_and_text_values(include_input):
    columns = {
        "instruction": ["first {instruction}\n第二行", "second"],
        "output": ["answer {output}", "done"],
        "source_id": [10, 11],
    }
    if include_input:
        columns["input"] = ["context", ""]
    dataset = Dataset.from_dict(columns)

    result = apply_chat_template_to_dataset(
        _dataset_info(dataset),
        _Tokenizer(),
        batch_size = 2,
        num_proc = 1,
    )

    assert result["success"] is True
    assert len(result["dataset"]) == len(dataset)
    assert result["dataset"].column_names == dataset.column_names + ["text"]
    assert list(result["dataset"]["instruction"]) == list(dataset["instruction"])
    assert list(result["dataset"]["output"]) == list(dataset["output"])
    assert "first {instruction}\n第二行" in result["dataset"][0]["text"]
    assert "answer {output}" in result["dataset"][0]["text"]
    expected_input = "context" if include_input else ""
    assert f"### Input:\n{expected_input}\n\n### Response:" in result["dataset"][0]["text"]


def test_main_entry_point_keeps_default_alpaca_output():
    dataset = _alpaca_dataset()

    result = format_and_template_dataset(
        dataset,
        model_name = "Qwen2ForCausalLM",
        tokenizer = _Tokenizer(),
        batch_size = 1,
        num_proc = 1,
    )

    assert result["success"] is True
    assert result["detected_format"] == "alpaca"
    assert result["final_format"] == "alpaca"
    assert len(result["dataset"]) == 1
    assert result["dataset"].column_names == dataset.column_names + ["text"]
    assert result["dataset"][0]["text"].endswith("### Response:\n4")


def test_main_entry_point_keeps_raw_text_path_unchanged():
    dataset = Dataset.from_dict({"body": ["raw one", "raw two"], "id": [1, 2]})

    result = format_and_template_dataset(
        dataset,
        model_name = "Qwen2ForCausalLM",
        tokenizer = _Tokenizer(),
        format_type = "raw",
    )

    assert result["success"] is True
    assert result["final_format"] == "raw_text"
    assert list(result["dataset"]["text"]) == ["raw one", "raw two"]
    assert len(result["dataset"]) == 2


def test_direct_chatml_path_is_unchanged_without_custom_template():
    dataset = Dataset.from_dict(
        {
            "conversations": [
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            ],
            "id": [7],
        }
    )
    dataset_info = {
        "dataset": dataset,
        "detected_format": "chatml",
        "final_format": "chatml_conversations",
        "chat_column": "conversations",
        "is_standardized": True,
        "warnings": [],
    }

    result = apply_chat_template_to_dataset(dataset_info, _Tokenizer(), num_proc = 1)

    assert result["success"] is True
    assert result["dataset"][0]["text"] == "user: hello\nassistant: hi"
    assert result["dataset"].column_names == ["conversations", "id", "text"]
    assert len(result["dataset"]) == 1


def test_public_exports_keep_deprecated_positional_parameter_slots():
    apply_parameters = list(inspect.signature(apply_chat_template_to_dataset).parameters)
    main_parameters = list(inspect.signature(format_and_template_dataset).parameters)

    assert apply_parameters[2:5] == ["model_name", "custom_prompt_template", "add_eos_token"]
    assert main_parameters[8:11] == ["dataset_name", "custom_prompt_template", "add_eos_token"]

    dataset = _alpaca_dataset()
    with pytest.deprecated_call(match = "cannot persist a matching template"):
        result = apply_chat_template_to_dataset(
            _dataset_info(dataset), _Tokenizer(), None, "{instruction}"
        )
    assert result["success"] is False
    assert result["dataset"] is dataset


@pytest.mark.parametrize("entry_point", ["apply", "format"])
def test_deprecation_warning_points_to_external_callsite(entry_point):
    dataset = _alpaca_dataset()

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        if entry_point == "apply":
            result = apply_chat_template_to_dataset(
                _dataset_info(dataset),
                _Tokenizer(),
                custom_prompt_template = "{instruction}",
            )
        else:
            result = format_and_template_dataset(
                dataset,
                model_name = "Qwen2ForCausalLM",
                tokenizer = _Tokenizer(),
                custom_prompt_template = "{instruction}",
            )

    assert result["success"] is False
    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert caught[0].filename == __file__


def test_streaming_dataset_is_not_consumed_when_custom_template_is_rejected():
    visited = []

    def generate_rows():
        visited.append(True)
        yield {"instruction": "question", "input": "", "output": "answer"}

    features = Features(
        {
            "instruction": Value("string"),
            "input": Value("string"),
            "output": Value("string"),
        }
    )
    dataset = IterableDataset.from_generator(generate_rows, features = features)

    with pytest.deprecated_call(match = "cannot persist a matching template"):
        result = apply_chat_template_to_dataset(
            _dataset_info(dataset),
            _Tokenizer(),
            custom_prompt_template = "{instruction}",
        )

    assert result["success"] is False
    assert result["dataset"] is dataset
    assert result["dataset"].features == features
    assert result["dataset"].column_names == ["instruction", "input", "output"]
    assert visited == []


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
