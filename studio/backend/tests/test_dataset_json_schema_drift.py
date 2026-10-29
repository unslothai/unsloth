# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from datasets.exceptions import DatasetGenerationError

from utils.datasets.cache_safe import load_dataset_cache_safe
from utils.datasets.json_rows import json_rows_features

# A tiny block makes datasets type the file from its first rows, as it does a large upload.
_SMALL_BLOCKS = {"chunksize": 256}


def _chat(index, **assistant_keys):
    return {
        "messages": [
            {"role": "user", "content": f"question {index}"},
            {"role": "assistant", "content": f"answer {index}", **assistant_keys},
        ]
    }


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding = "utf-8")
    return str(path)


def _load(files, tmp_path, **kwargs):
    return load_dataset_cache_safe(
        "json",
        data_files = files,
        split = "train",
        cache_dir = str(tmp_path / "cache"),
        **kwargs,
    )


def test_optional_message_key_after_first_block_loads(tmp_path):
    rows = [_chat(i) for i in range(40)] + [_chat(40, weight = 0)]
    path = _write_jsonl(tmp_path / "chat.jsonl", rows)
    with pytest.raises(DatasetGenerationError):
        from datasets import load_dataset
        load_dataset(
            "json", data_files = path, split = "train", cache_dir = str(tmp_path / "raw"), **_SMALL_BLOCKS
        )

    dataset = _load(path, tmp_path, **_SMALL_BLOCKS)

    assert len(dataset) == 41
    assert dataset[0]["messages"][1] == {"role": "assistant", "content": "answer 0", "weight": None}
    assert dataset[40]["messages"][1]["weight"] == 0


def test_tool_calls_only_in_a_later_file_load(tmp_path):
    tool_call = {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    first = _write_jsonl(tmp_path / "a.jsonl", [_chat(0), _chat(1)])
    second = _write_jsonl(
        tmp_path / "b.jsonl", [{**_chat(2, tool_calls = [tool_call]), "source": "x"}]
    )

    dataset = _load([first, second], tmp_path)

    assert len(dataset) == 3
    assert dataset[0]["messages"][1]["tool_calls"] is None
    assert dataset[0]["source"] is None
    assert dataset[2]["messages"][1]["tool_calls"] == [tool_call]
    assert dataset[2]["source"] == "x"


def test_json_array_file_is_unified(tmp_path):
    path = tmp_path / "chat.json"
    path.write_text(json.dumps([{"a": {"x": 1}}, {"a": {"x": 2, "y": "z"}}]), encoding = "utf-8")

    features = json_rows_features([str(path)])

    assert set(features["a"]) == {"x", "y"}


def test_conflicting_types_still_raise(tmp_path):
    rows = [{"score": i} for i in range(40)] + [{"score": "high"}]
    path = _write_jsonl(tmp_path / "bad.jsonl", rows)

    assert json_rows_features([path]) is None
    with pytest.raises(DatasetGenerationError):
        _load(path, tmp_path, **_SMALL_BLOCKS)


def test_explicit_features_are_not_overridden(tmp_path, monkeypatch):
    from datasets import Features, Value

    import utils.datasets.json_rows as json_rows

    monkeypatch.setattr(json_rows, "json_rows_features", lambda files: pytest.fail("retried"))
    rows = [_chat(i) for i in range(40)] + [_chat(40, weight = 0)]
    path = _write_jsonl(tmp_path / "chat.jsonl", rows)

    with pytest.raises(Exception):
        _load(
            path,
            tmp_path,
            features = Features({"messages": [{"role": Value("string")}]}),
            **_SMALL_BLOCKS,
        )
