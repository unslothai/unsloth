# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""JSON Lines previews skip the datasets loader, whose threaded pyarrow JSON reader segfaulted the server (#15681)."""

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


def test_jsonl_preview_skips_the_threaded_loader(tmp_path, loader_calls):
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


def test_mixed_nested_types_raise_without_the_threaded_loader(tmp_path, loader_calls):
    def row(search):
        call = {"function": {"name": "s", "arguments": {"json_body": {"searchContent": search}}}}
        return {"messages": [{"role": "assistant", "content": "", "tool_calls": [call]}]}

    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [row("text"), row({"q": "x"})] * 50)
    # Token preview reads 100 rows: an Arrow column cannot be both string and struct.
    with pytest.raises((TypeError, ValueError)):
        local._stream_file_preview_slice(path, 100)
    assert loader_calls == []


@pytest.mark.parametrize(
    "lines, expected",
    [
        (['{"id": 9223372036854775808}', '{"id": 1}'], [9223372036854775808.0, 1.0]),
        (['{"id": 9223372036854775807}', '{"id": 9223372036854775808}'], [2.0**63, 2.0**63]),
        (['{"v": 9007199254740993}', '{"v": 1.5}'], [9007199254740992.0, 1.5]),
    ],
)
def test_large_numbers_preview_like_the_loader(tmp_path, loader_calls, lines, expected):
    path = tmp_path / "train.jsonl"
    path.write_text("".join(line + "\n" for line in lines), encoding = "utf-8")
    dataset, _ = local._stream_file_preview_slice(path, 10)
    assert dataset[dataset.column_names[0]] == expected and loader_calls == []


def test_a_float_after_the_preview_rows_types_the_column_like_the_loader(tmp_path, loader_calls):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"output": 7}] * 20 + [{"output": 1.5}])
    dataset, _ = local._stream_file_preview_slice(path, 10)
    assert dataset["output"] == [7.0] * 10 and isinstance(dataset["output"][0], float)


def test_columns_are_the_union_over_rows(tmp_path, loader_calls):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"text": "a"}, {"text": "b", "label": 1}])
    dataset, _ = local._stream_file_preview_slice(path, 10)
    assert dataset.column_names == ["text", "label"]
    assert dataset["label"] == [None, 1]


def test_a_column_first_seen_after_the_preview_rows_is_kept(tmp_path, loader_calls):
    path = tmp_path / "train.jsonl"
    rows = [{"question": "Q", "answer": "A"}] * 10 + [
        {"question": "Q", "answer": "A", "system": "S"}
    ]
    _write_jsonl(path, rows)
    dataset, _ = local._stream_file_preview_slice(path, 10)
    assert dataset.column_names == ["question", "answer", "system"] and len(dataset) == 10


def test_reading_stops_at_the_preview_size(tmp_path, loader_calls):
    path = tmp_path / "train.jsonl"
    _write_jsonl(path, [{"text": str(i)} for i in range(3)], tail = "{not json\n")
    dataset, _ = local._stream_file_preview_slice(path, 3)
    assert dataset["text"] == ["0", "1", "2"] and loader_calls == []
    undecodable = tmp_path / "tail.jsonl"
    _write_jsonl(undecodable, [{"text": str(i)} for i in range(3)])
    undecodable.write_bytes(undecodable.read_bytes() + b"\xff\n")
    assert local._stream_file_preview_slice(undecodable, 3)[0]["text"] == ["0", "1", "2"]


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
