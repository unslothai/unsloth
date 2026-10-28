# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A failed native JSON reader cannot take the preview caller down with it."""

import json
import os
import sys
import time
from pathlib import Path

import pytest
from datasets import ClassLabel, Features, Value

from hub.services.datasets.local import _stream_file_preview_slice
from utils.datasets import json_preview


def _write_rows(
    tmp_path,
    rows,
    *,
    array = False,
):
    path = tmp_path / ("rows.json" if array else "rows.jsonl")
    text = json.dumps(rows) if array else "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(text, encoding = "utf-8")
    return path


@pytest.mark.parametrize("array", [False, True], ids = ["jsonl", "json-array"])
def test_normal_chatml_rows_and_tools_are_preserved(tmp_path, array):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "lookup", "arguments": "{}"}}],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "lookup"}}],
        }
    ] * 3
    path = _write_rows(tmp_path, rows, array = array)
    dataset, count = _stream_file_preview_slice(path, 2)

    assert count is None
    assert len(dataset) == 2
    assert dataset[0]["tools"] == rows[0]["tools"]
    assert dataset[0]["messages"][1] == rows[0]["messages"][1]
    assert dataset[0]["messages"][0]["content"] == "question"
    assert dataset[0]["messages"][0]["tool_calls"] is None


def test_feature_metadata_survives_the_worker_boundary(tmp_path):
    features = Features({"text": Value("string"), "label": ClassLabel(names = ["no", "yes"])})
    path = _write_rows(tmp_path, [{"text": "a", "label": 1}, {"text": "b", "label": 0}])

    dataset, _ = _stream_file_preview_slice(path, 10, features = features)

    assert dataset.features == features
    assert dataset.to_list() == [{"text": "a", "label": 1}, {"text": "b", "label": 0}]


def test_large_preview_does_not_block_worker_exit(tmp_path):
    rows = [{"text": "x" * 250_000, "index": i} for i in range(4)]
    path = _write_rows(tmp_path, rows)
    dataset, _ = _stream_file_preview_slice(path, 4)
    assert dataset.to_list() == rows


def test_empty_preview_keeps_existing_empty_result(tmp_path):
    path = _write_rows(tmp_path, [{"text": "not requested"}])
    assert _stream_file_preview_slice(path, 0) is None


def test_conflicting_field_types_are_rejected_without_normalization(tmp_path):
    path = _write_rows(tmp_path, [{"arg": "value"}, {"arg": {"key": "value"}}])
    with pytest.raises(ValueError, match = "changed from"):
        _stream_file_preview_slice(path, 10)


def test_malformed_json_is_a_controlled_error(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("{not valid json\n")
    with pytest.raises(ValueError) as exc:
        _stream_file_preview_slice(path, 10)
    assert str(exc.value)


def _exit_reader(**kwargs):
    os._exit(17)


def _stalled_reader(**kwargs):
    time.sleep(120)


def _read_without_tensor_backends(**kwargs):
    from utils.datasets.json_preview_worker import read_json_preview

    assert os.environ["USE_TORCH"] == "0"
    assert os.environ["USE_TF"] == "0"
    read_json_preview(**kwargs)
    assert "torch" not in sys.modules
    assert "tensorflow" not in sys.modules


def _replace_worker(monkeypatch, name):
    from core.training import account_jobs
    original = account_jobs.account_process_spec
    monkeypatch.setattr(
        account_jobs,
        "account_process_spec",
        lambda module, target, env, kwargs: original(__name__, name, env, kwargs),
    )


def _record_lifetime(monkeypatch):
    from utils import process_lifetime

    adopted, forgotten = [], []
    adopt, forget = process_lifetime.adopt_pid, process_lifetime.forget_pid

    def record_adopt(pid):
        adopted.append(pid)
        adopt(pid)

    def record_forget(pid):
        forgotten.append(pid)
        forget(pid)

    monkeypatch.setattr(process_lifetime, "adopt_pid", record_adopt)
    monkeypatch.setattr(process_lifetime, "forget_pid", record_forget)
    return adopted, forgotten


def test_plain_json_skips_tensor_backends_only_in_the_child(tmp_path, monkeypatch):
    path = _write_rows(tmp_path, [{"text": "ok"}])
    monkeypatch.setenv("USE_TORCH", "1")
    monkeypatch.setenv("USE_TF", "1")
    _replace_worker(monkeypatch, "_read_without_tensor_backends")

    dataset, _ = _stream_file_preview_slice(path, 10)

    assert dataset.to_list() == [{"text": "ok"}]
    assert os.environ["USE_TORCH"] == "1"
    assert os.environ["USE_TF"] == "1"


def test_worker_death_is_contained_and_next_preview_succeeds(tmp_path, monkeypatch):
    from utils.paths import tmp_root

    path = _write_rows(tmp_path, [{"text": "still alive"}])
    with monkeypatch.context() as patch:
        _replace_worker(patch, "_exit_reader")
        adopted, forgotten = _record_lifetime(patch)
        with pytest.raises(ValueError, match = "stopped unexpectedly"):
            _stream_file_preview_slice(path, 10)
        assert len(adopted) == 1
        assert forgotten == adopted
        assert not list(tmp_root().glob("unsloth-json-preview-*"))

    dataset, _ = _stream_file_preview_slice(path, 10)
    assert dataset.to_list() == [{"text": "still alive"}]


def test_timeout_reaps_the_worker_and_removes_temporary_files(tmp_path, monkeypatch):
    from utils.paths import tmp_root

    path = _write_rows(tmp_path, [{"text": "ok"}])
    _replace_worker(monkeypatch, "_stalled_reader")
    monkeypatch.setattr(json_preview, "_PREVIEW_TIMEOUT_S", 0.1)
    adopted, forgotten = _record_lifetime(monkeypatch)
    started = time.monotonic()

    with pytest.raises(ValueError, match = "timed out"):
        _stream_file_preview_slice(path, 10)

    assert time.monotonic() - started < 10
    assert len(adopted) == 1
    assert forgotten == adopted
    assert not list(tmp_root().glob("unsloth-json-preview-*"))
