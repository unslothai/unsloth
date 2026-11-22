# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A local Parquet preview reads only its leading rows, and shows what loading the files whole shows."""

from __future__ import annotations

import datetime as dt
import decimal
import io
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402

import hub.services.datasets.local as local  # noqa: E402
import utils.datasets.cache_safe as cache_safe  # noqa: E402


@pytest.fixture(autouse = True)
def _own_datasets_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_cache"))
    import datasets.config
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf_datasets_cache")


def _jpeg(shade: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, 0, 255 - shade)).save(buffer, format = "JPEG")
    return buffer.getvalue()


def _typed_table(rows: int) -> pa.Table:
    return pa.table(
        {
            "instruction": [f"q{i}" if i % 4 else None for i in range(rows)],
            "messages": [
                [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": None}]
                for i in range(rows)
            ],
            "ts": pa.array(
                [dt.datetime(2024, 1, 1, 0, 0, i % 60) for i in range(rows)],
                pa.timestamp("us", tz = "UTC"),
            ),
            "price": pa.array(
                [decimal.Decimal(f"{i}.25") for i in range(rows)], pa.decimal128(9, 2)
            ),
            "tag": pa.array([f"t{i % 3}" for i in range(rows)]).dictionary_encode(),
            "long": pa.array([f"L{i}" for i in range(rows)], pa.large_string()),
            "nested": [{"a": i, "b": [f"x{i}"]} for i in range(rows)],
        }
    )


def _image_file(path: Path, rows: int) -> None:
    from datasets import Features, Image, Value

    features = Features({"image": Image(), "text": Value("string")})
    table = pa.table(
        {
            "image": [{"bytes": _jpeg(i * 9 % 256), "path": f"{i}.jpg"} for i in range(rows)],
            "text": [f"caption {i}" for i in range(rows)],
        },
        schema = features.arrow_schema,
    )
    pq.write_table(table, path, row_group_size = 4)


_LEADING_ROWS = getattr(local, "_parquet_preview_slice", None)


def _preview(
    path: Path,
    monkeypatch,
    *,
    whole: bool,
    split: str = "train",
):
    leading_rows = (lambda files, size: None) if whole else _LEADING_ROWS
    monkeypatch.setattr(local, "_parquet_preview_slice", leading_rows, raising = False)
    try:
        dataset, total = local._load_local_preview_slice(
            dataset_path = path, train_split = split, preview_size = 10
        )
    except Exception as error:  # noqa: BLE001 - the error itself is compared
        return ("error", type(error).__name__, str(error))
    table = dataset.with_format("arrow")[:]
    return total, repr(dataset.features), table.schema, table.to_pylist()


def _fixtures(root: Path) -> list[Path]:
    pq.write_table(_typed_table(37), root / "typed.parquet", row_group_size = 6)
    _image_file(root / "images.parquet", 13)
    pq.write_table(pa.table({"text": ["a", "b", "c"]}), root / "short.parquet")
    shards = root / "shards"
    shards.mkdir()
    for index in range(5):
        pq.write_table(_typed_table(index % 3 + 5), shards / f"part-{index:05d}.parquet")
    recipe = root / "recipe" / "parquet-files"
    recipe.mkdir(parents = True)
    for index in range(3):
        pq.write_table(
            pa.table({"prompt": [f"p{index}{i}" for i in range(4)]}),
            recipe / f"batch_{index}.parquet",
        )
    return [
        root / "typed.parquet",
        root / "images.parquet",
        root / "short.parquet",
        shards,
        root / "recipe",
    ]


def test_parquet_preview_matches_the_whole_load(tmp_path, monkeypatch):
    for path in _fixtures(tmp_path):
        whole = _preview(path, monkeypatch, whole = True)
        leading = _preview(path, monkeypatch, whole = False)
        assert whole[0] != "error", whole
        assert leading == whole, path.name


def test_parquet_preview_does_not_load_the_files_whole(tmp_path, monkeypatch):
    calls = []
    real = cache_safe.load_dataset_cache_safe

    def counting(*args, **kwargs):
        calls.append(kwargs.get("data_files"))
        return real(*args, **kwargs)

    monkeypatch.setattr(cache_safe, "load_dataset_cache_safe", counting)
    for path in _fixtures(tmp_path):
        dataset, total = local._load_local_preview_slice(
            dataset_path = path, train_split = "train", preview_size = 10
        )
        assert len(dataset) == min(10, total)
    assert calls == []
    assert not (tmp_path / "hf_datasets_cache").exists() or not any(
        (tmp_path / "hf_datasets_cache").rglob("*.arrow")
    )


def test_parquet_preview_still_fails_on_data_damaged_past_the_preview_rows(tmp_path, monkeypatch):
    damaged = tmp_path / "damaged.parquet"
    pq.write_table(_typed_table(60), damaged, row_group_size = 10)
    column = pq.read_metadata(damaged).row_group(5).column(0)
    with open(damaged, "r+b") as handle:
        handle.seek(column.data_page_offset)
        handle.write(b"\0" * 64)
    assert _LEADING_ROWS([damaged], 10) is None
    whole = _preview(damaged, monkeypatch, whole = True)
    leading = _preview(damaged, monkeypatch, whole = False)
    assert whole[0] == "error", whole
    assert leading == whole


def test_parquet_preview_leaves_uncertain_inputs_to_the_whole_load(tmp_path, monkeypatch):
    mixed = tmp_path / "mixed"
    mixed.mkdir()
    pq.write_table(pa.table({"text": ["x", "y"]}), mixed / "a.parquet")
    pq.write_table(pa.table({"text": [1, 2]}), mixed / "b.parquet")
    renamed = tmp_path / "renamed"
    renamed.mkdir()
    pq.write_table(pa.table({"text": ["x"]}), renamed / "a.parquet")
    pq.write_table(pa.table({"other": ["y"]}), renamed / "b.parquet")
    empty = tmp_path / "empty.parquet"
    pq.write_table(pa.table({"text": pa.array([], pa.string())}), empty)
    broken = tmp_path / "broken.parquet"
    broken.write_bytes(b"not parquet" * 8)
    typed = tmp_path / "typed.parquet"
    pq.write_table(_typed_table(12), typed)

    cases = [
        (mixed, "train"),
        (renamed, "train"),
        (empty, "train"),
        (broken, "train"),
        (typed, "test"),
    ]
    for path, split in cases:
        if split == "train":
            files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
            assert _LEADING_ROWS(files, 10) is None, path.name
        whole = _preview(path, monkeypatch, whole = True, split = split)
        leading = _preview(path, monkeypatch, whole = False, split = split)
        assert leading == whole, (path.name, split)
