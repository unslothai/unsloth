# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset, load_dataset

from utils.datasets import format_and_template_dataset


class _Tokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"


def _format(dataset):
    result = format_and_template_dataset(
        dataset,
        model_name = "Qwen2ForCausalLM",
        tokenizer = _Tokenizer(),
        batch_size = 2,
        num_proc = 1,
    )
    assert result["success"] is True
    assert result["final_format"] == "alpaca"
    return list(result["dataset"]["text"])


def test_missing_alpaca_values_are_formatted_as_empty_text():
    dataset = Dataset.from_dict(
        {
            "instruction": ["Say hi", None],
            "input": [None, "context"],
            "output": ["hi", None],
        }
    )

    texts = _format(dataset)

    assert texts[0].endswith("### Instruction:\nSay hi\n\n### Input:\n\n\n### Response:\nhi")
    assert texts[1].endswith("### Instruction:\n\n\n### Input:\ncontext\n\n### Response:\n")
    assert all("None" not in text for text in texts)


def test_blank_csv_cells_are_not_trained_as_none(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text('instruction,input,output\nSay hi,,hi\nSay bye,"",\n')
    dataset = load_dataset("csv", data_files = str(csv_path), split = "train")

    texts = _format(dataset)

    assert texts[0].endswith("### Input:\n\n\n### Response:\nhi")
    assert texts[1].endswith("### Input:\n\n\n### Response:\n")
    assert all("None" not in text for text in texts)
