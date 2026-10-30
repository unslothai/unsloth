# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import io
import json
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

datasets = pytest.importorskip("datasets")
pa = pytest.importorskip("pyarrow")
from datasets import ClassLabel, Dataset, Features, IterableDataset, Value, load_dataset

from utils.datasets.iterable import take_rows

_ALPACA = [
    {"instruction": f"i{i}", "input": "" if i % 3 else f"x{i}", "output": "o" * (i % 7)}
    for i in range(2500)
]
_ALPACA_FEATURES = Features(
    {"instruction": Value("string"), "input": Value("string"), "output": Value("string")}
)


def _jsonl(tmp_path, name, rows):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding = "utf-8")
    return str(path)


def _hetero():
    for i in range(1500):
        yield {"a": i} if i % 2 else {"a": i, "extra": str(i)}


def _streams(tmp_path):
    alpaca = _jsonl(tmp_path, "alpaca.jsonl", _ALPACA)
    first = _jsonl(tmp_path, "d1.jsonl", [{"a": i, "b": None} for i in range(700)])
    second = _jsonl(tmp_path, "d2.jsonl", [{"a": i, "b": f"s{i}"} for i in range(700)])
    nested = tmp_path / "nested.parquet"
    Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"a{i}"},
                ],
                "meta": {"k": i, "t": [1, 2] if i % 2 else []},
                "label": i % 3,
            }
            for i in range(1300)
        ]
    ).cast_column("label", ClassLabel(names = ["x", "y", "z"])).to_parquet(str(nested))
    return {
        "json_untyped": lambda: load_dataset(
            "json", data_files = alpaca, split = "train", streaming = True
        ),
        "json_typed": lambda: load_dataset(
            "json", data_files = alpaca, split = "train", streaming = True, features = _ALPACA_FEATURES
        ),
        "json_two_files_typed": lambda: load_dataset(
            "json",
            data_files = [first, second],
            split = "train",
            streaming = True,
            features = Features({"a": Value("int64"), "b": Value("string")}),
        ),
        "parquet_nested_classlabel": lambda: load_dataset(
            "parquet", data_files = str(nested), split = "train", streaming = True
        ),
        "generator_untyped_mixed_keys": lambda: IterableDataset.from_generator(_hetero),
        "generator_typed": lambda: IterableDataset.from_generator(
            _hetero, features = Features({"a": Value("int32"), "extra": Value("string")})
        ),
        "sharded_iterable": lambda: Dataset.from_list(_ALPACA).to_iterable_dataset(num_shards = 4),
        "formatted_numpy": lambda: load_dataset(
            "json", data_files = alpaca, split = "train", streaming = True, features = _ALPACA_FEATURES
        ).with_format("numpy"),
    }


@pytest.mark.parametrize("count", [0, 1, 999, 1000, 1001, 2345, 10**6])
def test_take_rows_matches_row_iteration(tmp_path, count):
    for name, make in _streams(tmp_path).items():
        expected = list(make().take(count))
        got = take_rows(make(), count)
        if name == "formatted_numpy":
            assert repr(got) == repr(expected), name
            continue
        assert got == expected, name
        assert [list(row) for row in got] == [list(row) for row in expected], name
        if expected:
            base, head = Dataset.from_list(expected), Dataset.from_list(got)
            assert head.features == base.features, name
            assert head._fingerprint == base._fingerprint, name


def test_take_rows_matches_images(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    from datasets import Image

    rows = []
    for i in range(40):
        buffer = io.BytesIO()
        pil.new("RGB", (4 + i % 5, 3), (i, 0, 0)).save(buffer, format = "PNG")
        rows.append({"image": {"bytes": buffer.getvalue(), "path": None}, "text": f"t{i}"})
    path = tmp_path / "images.parquet"
    Dataset.from_list(rows).cast_column("image", Image()).to_parquet(str(path))

    def make():
        return load_dataset("parquet", data_files = str(path), split = "train", streaming = True)

    expected = list(make().take(25))
    got = take_rows(make(), 25)
    assert [row["text"] for row in got] == [row["text"] for row in expected]
    assert [row["image"].tobytes() for row in got] == [row["image"].tobytes() for row in expected]
    assert [row["image"].size for row in got] == [row["image"].size for row in expected]


def test_take_rows_builds_the_schema_per_batch_not_per_row(tmp_path, monkeypatch):
    path = _jsonl(tmp_path, "alpaca.jsonl", _ALPACA)
    calls = []
    original = Features.arrow_schema

    def counting(self):
        calls.append(1)
        return original.fget(self)

    monkeypatch.setattr(Features, "arrow_schema", property(counting))

    def make():
        return load_dataset(
            "json", data_files = path, split = "train", streaming = True, features = _ALPACA_FEATURES
        )

    expected = list(make().take(2345))
    row_by_row = len(calls)
    calls.clear()
    assert take_rows(make(), 2345) == expected
    # datasets 3.4 builds the schema once per stream on both paths; later releases build it per row.
    assert len(calls) <= min(10, row_by_row)


def test_take_rows_matches_when_files_drift_from_declared_features(tmp_path):
    # README features with per-file inferred schemas that differ: datasets 4.3 cannot join the two
    # files into one Arrow batch (ArrowInvalid), so take_rows must fall back to the row-by-row read.
    (tmp_path / "data").mkdir()
    (tmp_path / "README.md").write_text(
        "---\ndataset_info:\n  features:\n  - name: a\n    dtype: int64\n  - name: b\n"
        "    dtype: string\n  - name: c\n    dtype: float64\nconfigs:\n- config_name: default\n"
        "  data_files:\n  - split: train\n    path: data/*.jsonl\n---\n",
        encoding = "utf-8",
    )
    _jsonl(tmp_path / "data", "1.jsonl", [{"a": i, "b": None, "c": i} for i in range(600)])
    _jsonl(tmp_path / "data", "2.jsonl", [{"b": f"s{i}", "a": i, "c": i + 0.5} for i in range(600)])

    def make():
        return load_dataset(str(tmp_path), split = "train", streaming = True)

    assert make().features
    for count in (599, 1100, 5000):
        expected = list(make().take(count))
        got = take_rows(make(), count)
        assert got == expected
        assert [list(row) for row in got] == [list(row) for row in expected]


def test_take_rows_reads_row_by_row_when_batches_disagree():
    rows = [{"a": 1}, {"a": 2}, {"a": 3}]

    class Taken:
        def __init__(self, count):
            self.count = count

        def iter(self, batch_size):
            yield {"a": [1]}
            raise pa.ArrowInvalid("Schema at index 1 was different")

        def __iter__(self):
            return iter(rows[: self.count])

    class Stream:
        features = {"a": object()}

        def take(self, count):
            return Taken(count)

    assert take_rows(Stream(), 2) == [{"a": 1}, {"a": 2}]


def test_streamed_slices_read_through_take_rows():
    backend = Path(_BACKEND_DIR)
    for relative in ("core/training/trainer.py", "hub/utils/dataset_cache.py"):
        source = (backend / relative).read_text(encoding = "utf-8")
        assert "list(stream.take(" not in source, relative
        assert "take_rows(stream, " in source, relative
