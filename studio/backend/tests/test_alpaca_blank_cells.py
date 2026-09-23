# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from datasets import Dataset, load_dataset

from utils.datasets import convert_alpaca_to_chatml, format_and_template_dataset
from utils.datasets.cells import cell_text


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

    assert texts[0].endswith("### Instruction:\nSay hi\n\n### Input:\n\n\n### Response:\nhi</s>")
    assert texts[1].endswith("### Instruction:\n\n\n### Input:\ncontext\n\n### Response:\n</s>")
    assert all("None" not in text for text in texts)


def test_blank_csv_cells_are_not_trained_as_none(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text('instruction,input,output\nSay hi,,hi\nSay bye,"",\n')
    dataset = load_dataset("csv", data_files = str(csv_path), split = "train")

    texts = _format(dataset)

    assert texts[0].endswith("### Input:\n\n\n### Response:\nhi</s>")
    assert texts[1].endswith("### Input:\n\n\n### Response:\n</s>")
    assert all("None" not in text for text in texts)


def test_alpaca_rows_end_with_exactly_one_eos():
    dataset = Dataset.from_dict(
        {"instruction": ["Add 2+2", "Echo"], "input": ["", ""], "output": ["4", "done</s>"]}
    )

    texts = _format(dataset)

    assert texts[0].endswith("### Response:\n4</s>")
    assert texts[1].endswith("### Response:\ndone</s>")
    assert not texts[1].endswith("</s></s>")


def test_blank_csv_cells_are_not_converted_to_none_for_chatml(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("instruction,input,output\nSay bye,,\n,context,answer\n")
    dataset = load_dataset("csv", data_files = str(csv_path), split = "train")

    conversations = convert_alpaca_to_chatml(dataset, batch_size = 2, num_proc = 1)["conversations"]

    assert conversations[0] == [
        {"role": "user", "content": "Say bye"},
        {"role": "assistant", "content": ""},
    ]
    assert conversations[1] == [
        {"role": "user", "content": "context"},
        {"role": "assistant", "content": "answer"},
    ]


def test_a_blank_cell_in_a_numeric_column_still_converts_to_chatml(tmp_path):
    # The blank cell types the column float, so 1 and 3 arrive as 1.0 and 3.0.
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("instruction,input,output\nA,1,x\nB,,y\nC,3,\n")
    dataset = load_dataset("csv", data_files = str(csv_path), split = "train")

    conversations = convert_alpaca_to_chatml(dataset, batch_size = 3, num_proc = 1)["conversations"]

    assert conversations[0] == [
        {"role": "user", "content": "A\n\n1.0"},
        {"role": "assistant", "content": "x"},
    ]
    assert conversations[1] == [
        {"role": "user", "content": "B"},
        {"role": "assistant", "content": "y"},
    ]
    assert conversations[2] == [
        {"role": "user", "content": "C\n\n3.0"},
        {"role": "assistant", "content": ""},
    ]
    assert all(
        isinstance(message["content"], str)
        for conversation in conversations
        for message in conversation
    )


def test_every_cell_reaches_a_chat_template_as_text():
    # A chat template renders content directly: a non-string is a crash or a repr.
    dataset = Dataset.from_dict({"instruction": ["a"], "input": [True], "output": [7]})

    conversations = convert_alpaca_to_chatml(dataset, batch_size = 1, num_proc = 1)["conversations"]

    assert conversations[0] == [
        {"role": "user", "content": "a\n\nTrue"},
        {"role": "assistant", "content": "7"},
    ]


def test_cell_text_maps_every_empty_spelling_to_empty_text():
    assert cell_text(None) == ""
    assert cell_text(float("nan")) == ""
    assert cell_text("") == ""

    # Present values survive, including the ones `or ""` would have dropped.
    assert cell_text("None") == "None"
    assert cell_text(0) == "0"
    assert cell_text(False) == "False"
    assert cell_text("0") == "0"
    assert cell_text(" ") == " "
