# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from datasets import Dataset

from utils.datasets.raw_text import prepare_raw_text_dataset

CODE = "def f():\n    return 1\n" * 20
ROWS = {
    "repo_name": ["pansapiens/mytardis", "a/b"],
    "path": ["tardis/models.py", "x.py"],
    "content": [CODE, CODE + "print(f())\n"],
}


@pytest.mark.parametrize("streaming", [False, True])
def test_raw_text_trains_the_body_column_not_the_first_string_column(streaming):
    dataset = Dataset.from_dict(ROWS)
    if streaming:
        dataset = dataset.to_iterable_dataset()

    result = prepare_raw_text_dataset(dataset, mode_label = "CPT", split_name = "train")

    assert [row["text"] for row in result.dataset] == ROWS["content"]
    assert any("auto-selecting 'content'" in notice.message for notice in result.notices)


def test_raw_text_keeps_the_first_column_when_no_column_is_longer():
    result = prepare_raw_text_dataset(
        Dataset.from_dict({"content": ["101", "102"], "source": ["corpus", "corpus"]})
    )
    assert list(result.dataset["text"]) == ["101", "102"]


def test_raw_text_uses_the_requested_column_without_the_choice_warning():
    result = prepare_raw_text_dataset(
        Dataset.from_dict({"title": ["A much longer title"], "body": ["Short."]}),
        text_column = "body",
    )
    assert list(result.dataset["text"]) == ["Short."]
    assert result.source_column == "body"
    assert not any("auto-selecting" in notice.message for notice in result.notices)


def test_raw_text_scores_unspaced_scripts_by_length_not_by_spaces():
    doc = "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。"
    result = prepare_raw_text_dataset(
        Dataset.from_dict({"content": [doc, doc], "source": ["news article", "news article"]})
    )
    assert list(result.dataset["text"]) == [doc, doc]


def test_raw_format_eval_split_reuses_the_train_column():
    from utils.datasets.dataset_utils import format_and_template_dataset

    train = format_and_template_dataset(
        Dataset.from_dict({"title": ["A"], "body": ["The body has the most words."]}),
        model_name = "test",
        tokenizer = None,
        format_type = "raw",
    )
    eval_ = format_and_template_dataset(
        Dataset.from_dict({"title": ["A much longer title"], "body": ["Short."]}),
        model_name = "test",
        tokenizer = None,
        format_type = "raw",
        split_name = "eval",
        raw_text_column = train["raw_text_column"],
    )
    assert train["raw_text_column"] == "body"
    assert any("auto-selecting 'body'" in w for w in train["run_warnings"])
    assert not any("auto-selecting" in w for w in eval_["run_warnings"])
    assert list(eval_["dataset"]["text"]) == ["Short."]


@pytest.mark.parametrize("column", ["body", "content", "passage"])
def test_raw_text_discovers_strings_in_an_untyped_stream(column):
    from datasets import IterableDataset

    texts = [None, "", "First document.", "Second document."]
    dataset = IterableDataset.from_generator(lambda: ({column: text} for text in texts))
    assert dataset.features is None

    result = prepare_raw_text_dataset(dataset, append_eos = True, eos_token = "<eos>")

    assert result.source_column == column
    assert [row["text"] for row in result.dataset] == [
        "First document.<eos>",
        "Second document.<eos>",
    ]


def test_raw_text_untyped_stream_keeps_column_choice_and_eval_override():
    from datasets import IterableDataset

    rows = [{"id": 1, "title": "A", "body": "The body has the most words."}]
    dataset = IterableDataset.from_generator(lambda: iter(rows))
    result = prepare_raw_text_dataset(dataset)
    assert result.source_column == "body"
    assert list(result.dataset) == [{"id": 1, "title": "A", "text": rows[0]["body"]}]

    result = prepare_raw_text_dataset(dataset, split_name = "eval", text_column = "title")
    assert result.source_column == "title"
    assert [row["text"] for row in result.dataset] == ["A"]


def test_raw_text_untyped_column_probe_does_not_scan_the_stream():
    from datasets import IterableDataset

    visited = []

    def rows():
        for index in range(1000):
            visited.append(index)
            yield {"body": f"Document {index}."}

    result = prepare_raw_text_dataset(IterableDataset.from_generator(rows))
    assert max(visited) < 100
    assert next(iter(result.dataset))["text"] == "Document 0."


def test_raw_text_does_not_treat_untyped_numeric_columns_as_text():
    from datasets import IterableDataset
    dataset = IterableDataset.from_generator(lambda: iter([{"id": 123}]))
    with pytest.raises(ValueError, match = "string 'text' column"):
        prepare_raw_text_dataset(dataset)


def test_raw_text_streamed_json_without_a_text_column(tmp_path):
    from datasets import load_dataset

    path = tmp_path / "corpus.jsonl"
    path.write_text('{"body": "First document."}\n{"body": "Second document."}\n', encoding = "utf-8")
    dataset = load_dataset(
        "json",
        data_files = str(path),
        split = "train",
        streaming = True,
        cache_dir = str(tmp_path / "cache"),
    )
    result = prepare_raw_text_dataset(dataset)
    assert [row["text"] for row in result.dataset] == ["First document.", "Second document."]
