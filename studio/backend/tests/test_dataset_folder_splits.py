# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local dataset folder trains on what ``datasets.load_dataset`` calls its train split, not every file."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.paths import dataset_files_in_dir  # noqa: E402


def _rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding = "utf-8")


def _names(directory: Path) -> list[str]:
    return [p.name for p in dataset_files_in_dir(directory)]


ROWS = [
    {"instruction": "hi", "input": "", "output": "yo"},
    {"instruction": "a", "input": "", "output": "b"},
]


def test_a_validation_file_is_not_merged_into_training(tmp_path):
    _rows(tmp_path / "train.jsonl", ROWS)
    _rows(tmp_path / "val.jsonl", [{"instruction": "held", "input": "", "output": "out"}])
    assert _names(tmp_path) == ["train.jsonl"]


def test_a_sidecar_json_beside_the_train_split_does_not_break_the_load(tmp_path):
    datasets = pytest.importorskip("datasets")
    _rows(tmp_path / "train.jsonl", ROWS)
    (tmp_path / "filter_report.json").write_text(
        json.dumps({"reasons": [], "kept": 2}), encoding = "utf-8"
    )
    (tmp_path / "dataset_manifest.json").write_text(json.dumps({"run_id": "1"}), encoding = "utf-8")

    files = [str(p) for p in dataset_files_in_dir(tmp_path)]
    assert [Path(f).name for f in files] == ["train.jsonl"]
    loaded = datasets.load_dataset("json", data_files = files, split = "train")
    assert loaded.num_rows == 2 and loaded.column_names == ["instruction", "input", "output"]


def test_split_named_parquet_files_keep_only_train(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    for name in (
        "train-00000-of-00001.parquet",
        "validation-00000-of-00001.parquet",
        "test.parquet",
    ):
        pq.write_table(pa.table({"text": ["x"]}), tmp_path / name)
    assert _names(tmp_path) == ["train-00000-of-00001.parquet"]


def test_a_folder_without_split_names_still_loads_every_file(tmp_path):
    _rows(tmp_path / "part1.jsonl", ROWS)
    _rows(tmp_path / "part2.jsonl", ROWS)
    assert _names(tmp_path) == ["part1.jsonl", "part2.jsonl"]


def test_a_folder_with_no_train_split_still_loads_every_file(tmp_path):
    # Only a validation-named file: picking none of them would leave nothing to train on.
    _rows(tmp_path / "val.jsonl", ROWS)
    _rows(tmp_path / "notes.jsonl", ROWS)
    assert _names(tmp_path) == ["notes.jsonl", "val.jsonl"]


def test_an_eval_folder_takes_its_validation_split_then_test(tmp_path):
    from utils.paths import EVAL_SPLITS

    _rows(tmp_path / "train.jsonl", ROWS)
    _rows(tmp_path / "test.jsonl", ROWS)
    assert [p.name for p in dataset_files_in_dir(tmp_path, EVAL_SPLITS)] == ["test.jsonl"]
    _rows(tmp_path / "val.jsonl", ROWS)
    assert [p.name for p in dataset_files_in_dir(tmp_path, EVAL_SPLITS)] == ["val.jsonl"]


def test_an_eval_folder_without_eval_names_still_loads_every_file(tmp_path):
    from utils.paths import EVAL_SPLITS

    _rows(tmp_path / "train.jsonl", ROWS)
    _rows(tmp_path / "part2.jsonl", ROWS)
    assert [p.name for p in dataset_files_in_dir(tmp_path, EVAL_SPLITS)] == [
        "part2.jsonl",
        "train.jsonl",
    ]


def test_the_folder_preview_reads_the_train_split_not_a_sidecar(tmp_path):
    pytest.importorskip("datasets")
    pytest.importorskip("fastapi")
    from hub.services.datasets.local import _load_local_preview_slice

    _rows(tmp_path / "train.jsonl", ROWS)
    (tmp_path / "filter_report.json").write_text(
        json.dumps({"reasons": [], "kept": 2}), encoding = "utf-8"
    )
    preview, _ = _load_local_preview_slice(
        dataset_path = tmp_path, train_split = "train", preview_size = 10
    )
    assert preview.column_names == ["instruction", "input", "output"]
