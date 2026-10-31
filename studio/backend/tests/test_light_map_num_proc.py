# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cheap formatting maps over an in-memory dataset run in this process; Arrow-backed datasets and
explicit worker counts keep the pool."""

from __future__ import annotations

import hashlib
import json
import sys

import pytest
from datasets import Dataset, load_from_disk

import utils.hardware
import utils.hardware.hardware as hw
from utils.datasets import (
    apply_chat_template_to_dataset,
    convert_alpaca_to_chatml,
    convert_chatml_to_alpaca,
    format_and_template_dataset,
    standardize_chat_format,
)

POOL = 4


class _Tokenizer:
    chat_template = "{{ messages }}"
    eos_token = "</s>"

    def apply_chat_template(self, conversation, **_kwargs):
        return "\n".join(f"{turn['role']}: {turn['content']}" for turn in conversation)


def _alpaca(n):
    return Dataset.from_dict(
        {
            "instruction": [f"do {i}" for i in range(n)],
            "input": ["" if i % 3 else f"ctx {i}" for i in range(n)],
            "output": [f"done {i}" for i in range(n)],
        }
    )


def _sharegpt(n):
    return Dataset.from_dict(
        {
            "conversations": [
                [{"from": "human", "value": f"q {i}"}, {"from": "gpt", "value": f"a {i}"}]
                for i in range(n)
            ]
        }
    )


def _chatml(n):
    return Dataset.from_dict(
        {
            "messages": [
                [{"role": "user", "content": f"q {i}"}, {"role": "assistant", "content": f"a {i}"}]
                for i in range(n)
            ]
        }
    )


def _on_disk(dataset, tmp_path):
    dataset.save_to_disk(str(tmp_path / "ds"))
    loaded = load_from_disk(str(tmp_path / "ds"))
    assert loaded.cache_files
    return loaded


@pytest.fixture
def map_calls(monkeypatch):
    """Record num_proc of every Dataset.map, with the policy wanting a pool of POOL workers."""
    want = lambda desired = None, **_k: POOL if desired is None else desired
    monkeypatch.setattr(hw, "dataset_map_num_proc", want)
    monkeypatch.setattr(utils.hardware, "dataset_map_num_proc", want)
    monkeypatch.setattr(hw, "_num_proc_override_is_set", lambda: False)
    calls = []
    real_map = Dataset.map

    def spy(
        self,
        function = None,
        *args,
        **kwargs,
    ):
        calls.append((getattr(function, "__name__", "?"), kwargs.get("num_proc")))
        kwargs.pop("num_proc", None)  # the record is the point; run them in process
        return real_map(self, function, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", spy)
    return calls


def _run_all(make):
    """The four cheap maps of the formatting path, on datasets made by make(dataset)."""
    standardize_chat_format(make(_sharegpt(6)))
    convert_alpaca_to_chatml(make(_alpaca(6)))
    convert_chatml_to_alpaca(make(_chatml(6)))
    info = {
        "dataset": make(_alpaca(6)),
        "detected_format": "alpaca",
        "final_format": "alpaca",
        "chat_column": None,
        "is_standardized": True,
        "warnings": [],
    }
    result = apply_chat_template_to_dataset(info, _Tokenizer())
    assert result["success"] is True


def test_cheap_maps_over_in_memory_rows_build_no_pool(map_calls):
    _run_all(lambda ds: ds)

    names = [name for name, _ in map_calls]
    assert names == ["_standardize_dataset", "_convert", "_convert", "_format_alpaca"]
    assert [num_proc for _, num_proc in map_calls] == [None] * 4


def test_arrow_backed_datasets_keep_the_pool(map_calls, tmp_path):
    made = iter(range(100))
    _run_all(lambda ds: _on_disk(ds, tmp_path / str(next(made))))

    assert [num_proc for _, num_proc in map_calls] == [POOL] * 4


def test_an_explicit_count_is_honoured(map_calls):
    convert_alpaca_to_chatml(_alpaca(6), num_proc = 3)
    standardize_chat_format(_sharegpt(6), num_proc = 2)

    assert [num_proc for _, num_proc in map_calls] == [3, 2]


def test_the_escape_hatch_keeps_the_pool(map_calls, monkeypatch):
    monkeypatch.setattr(hw, "_num_proc_override_is_set", lambda: True)

    convert_alpaca_to_chatml(_alpaca(6))

    assert [num_proc for _, num_proc in map_calls] == [POOL]


def test_the_template_map_keeps_the_pool(map_calls):
    """Rendering a chat template per row is the costly map; only the cheap ones moved in process."""
    info = {
        "dataset": _chatml(6),
        "detected_format": "chatml_messages",
        "final_format": "chatml_messages",
        "chat_column": "messages",
        "is_standardized": True,
        "warnings": [],
    }
    result = apply_chat_template_to_dataset(info, _Tokenizer())

    assert result["success"] is True
    assert ("_format_chatml", POOL) in map_calls


def _digest(dataset):
    return hashlib.sha256(json.dumps(dataset.to_list(), sort_keys = True).encode()).hexdigest()


@pytest.mark.skipif(sys.platform == "win32", reason = "needs a fork pool to compare against")
@pytest.mark.parametrize("make", [_alpaca, _sharegpt], ids = ["alpaca", "sharegpt"])
def test_in_process_rows_and_fingerprint_match_the_pool(make, monkeypatch):
    """Same rows, order, columns and fingerprint as the pooled map it replaces."""
    multiprocess = pytest.importorskip("multiprocess")
    if "fork" not in multiprocess.get_all_start_methods():
        pytest.skip("needs fork to build a real worker pool")
    monkeypatch.setattr(hw, "_num_proc_override_is_set", lambda: False)
    monkeypatch.setattr(hw, "dataset_map_num_proc", lambda desired = None, **_k: desired or 2)
    monkeypatch.setattr(
        utils.hardware, "dataset_map_num_proc", lambda desired = None, **_k: desired or 2
    )

    def run(num_proc):
        source = make(2500)
        source._fingerprint = "r12fixedinput00"
        info = format_and_template_dataset(
            source, model_name = "test-model", tokenizer = _Tokenizer(), num_proc = num_proc
        )
        assert info["success"] is True, info.get("errors")
        return info["dataset"]

    pooled, in_process = run(2), run(None)
    assert not in_process.cache_files and not pooled.cache_files
    assert in_process.column_names == pooled.column_names
    assert _digest(in_process) == _digest(pooled)
    assert in_process._fingerprint == pooled._fingerprint
