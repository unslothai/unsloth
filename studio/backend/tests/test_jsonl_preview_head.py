# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""JSON Lines previews skip pyarrow's JSON reader, which segfaulted the server (#15681)."""

from __future__ import annotations

import json

import pytest

from hub.services.datasets import local


def _write_jsonl(
    path,
    rows,
    tail = "",
):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + tail, encoding = "utf-8")


@pytest.fixture
def loader_calls(monkeypatch):
    import datasets

    calls = []
    real = datasets.load_dataset

    def spy(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("path"))
        return real(*args, **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", spy)
    return calls


def test_jsonl_preview_skips_the_arrow_json_reader(tmp_path, loader_calls):
    rows = [
        {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "search", "parameters": {"type": "object"}},
                }
            ],
            "messages": [
                {"role": "user", "content": "q" * 50_000},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c0",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "c0", "content": "r" * 50_000},
            ],
        }
        for _ in range(8)
    ]
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, rows)
    dataset, total = local._stream_file_preview_slice(path, 3)
    assert loader_calls == []
    assert total is None and len(dataset) == 3
    assert dataset.column_names == ["tools", "messages"]
    assert dataset[0]["messages"][2]["content"] == "r" * 50_000


def test_columns_are_the_union_over_rows(tmp_path, loader_calls):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"text": "a"}, {"text": "b", "label": 1}])
    dataset, _ = local._stream_file_preview_slice(path, 10)
    assert dataset.column_names == ["text", "label"]
    assert dataset["label"] == [None, 1]


def test_reading_stops_at_the_preview_size(tmp_path, loader_calls):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"text": str(i)} for i in range(3)], tail = "{not json\n")
    dataset, _ = local._stream_file_preview_slice(path, 3)
    assert dataset["text"] == ["0", "1", "2"] and loader_calls == []


def test_blank_lines_are_skipped_and_an_empty_file_has_no_preview(tmp_path, loader_calls):
    path = tmp_path / "train.jsonl"
    path.write_text('\n{"text": "a"}\n\n{"text": "b"}\n', encoding = "utf-8")
    assert local._stream_file_preview_slice(path, 10)[0]["text"] == ["a", "b"]
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding = "utf-8")
    assert local._stream_file_preview_slice(empty, 10) is None
    assert loader_calls == []


@pytest.mark.parametrize(
    "text",
    [
        json.dumps([{"text": "a"}, {"text": "b"}]),
        json.dumps([{"text": "a"}, {"text": "b"}], indent = 2),
    ],
)
def test_a_file_that_is_not_json_lines_keeps_the_loader(tmp_path, loader_calls, text):
    path = tmp_path / "train.json"
    path.write_text(text, encoding = "utf-8")
    dataset, _ = local._stream_file_preview_slice(path, 10)
    assert loader_calls == ["json"]
    assert dataset["text"] == ["a", "b"]
