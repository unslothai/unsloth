# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The batch merge streams shards into the merged file instead of loading the dataset as one dataframe.
Its output must stay the file the dataframe path wrote: same rows, order, Arrow schema and pandas metadata."""

import datetime
import shutil
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

import pandas as pd
import pyarrow
import pyarrow.parquet as pq


def _dataframe_read(path):
    return pd.read_parquet(path, dtype_backend = "pyarrow")


@pytest.fixture(name = "worker")
def fixture_worker(monkeypatch):
    try:
        import data_designer.config.utils.io_helpers  # noqa: F401
    except ImportError:
        # Same read as data-designer's read_parquet_dataset for a folder whose shards agree.
        module = types.ModuleType("data_designer.config.utils.io_helpers")
        module.read_parquet_dataset = _dataframe_read
        for name in ("data_designer", "data_designer.config", "data_designer.config.utils"):
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        monkeypatch.setitem(sys.modules, module.__name__, module)
    from core.data_recipe.jobs import worker
    return worker


def _row(n: int, shard: int, drift: bool) -> dict:
    row = {
        "id": n,
        "text": f"row {n} " * (1 + n % 4),
        "score": n / 3,
        "flag": n % 2 == 0,
        "when": datetime.datetime(2026, 1, 1) + datetime.timedelta(minutes = n),
        "tags": [f"t{n % 3}", f"u{n % 5}"][: n % 3],
        "obj": {"a": f"a{n}", "b": n % 7},
        "maybe": None if n % 3 == 0 else f"m{n}",
    }
    if drift:
        row["mixed"] = {"k": n} if shard % 2 == 0 else f'{{"k": {n}}}'
        row["count"] = n if shard % 3 else float(n)
        row["all_null"] = None if shard % 2 == 0 else f"v{n}"
        if shard % 4 == 1:
            del row["maybe"]
    return row


def _write_dataset(base: Path, shards: int, rows: int, drift: bool) -> Path:
    parquet_dir = base / "parquet-files"
    parquet_dir.mkdir(parents = True)
    per = rows // shards
    for shard in range(shards):
        frame = pd.DataFrame([_row(shard * per + i, shard, drift) for i in range(per)])
        frame.to_parquet(parquet_dir / f"batch_{shard:05d}.parquet", index = False)
    return parquet_dir


def _reference(parquet_dir: Path, scratch: Path) -> Path:
    """What the merge wrote before streaming: the whole folder as a pyarrow-backed dataframe, written back."""
    copy = scratch / "reference-input"
    shutil.copytree(parquet_dir, copy)
    out = scratch / "reference.parquet"
    _dataframe_read(copy).to_parquet(out, index = False)
    return out


def _assert_same_file(actual: Path, expected: Path) -> None:
    actual_schema, expected_schema = pq.read_schema(actual), pq.read_schema(expected)
    assert actual_schema.equals(expected_schema, check_metadata = True)
    assert str(actual_schema) == str(expected_schema)
    actual_table, expected_table = pq.read_table(actual), pq.read_table(expected)
    assert actual_table.equals(expected_table)
    assert actual_table.to_pylist() == expected_table.to_pylist()
    assert pq.ParquetFile(actual).metadata.num_row_groups == (
        pq.ParquetFile(expected).metadata.num_row_groups
    )
    assert _dataframe_read(actual).equals(_dataframe_read(expected))


@pytest.mark.parametrize("shards, rows", [(2, 10), (10, 200), (100, 1000)])
@pytest.mark.parametrize("drift", [False, True], ids = ["uniform", "drifting"])
def test_streamed_merge_writes_the_dataframe_merge_file(
    tmp_path: Path, worker, monkeypatch, shards, rows, drift
) -> None:
    base = tmp_path / "dataset"
    parquet_dir = _write_dataset(base, shards, rows, drift)
    worker._unify_batch_schemas(base)
    expected = _reference(parquet_dir, tmp_path)

    def no_dataframe(_path):
        raise AssertionError("the merge loaded the dataset as a dataframe")

    monkeypatch.setattr(
        sys.modules["data_designer.config.utils.io_helpers"], "read_parquet_dataset", no_dataframe
    )
    worker._merge_batches_to_single_parquet(base)

    assert sorted(p.name for p in parquet_dir.iterdir()) == ["batch_00000.parquet"]
    assert not list(base.glob("*.merging"))
    _assert_same_file(parquet_dir / "batch_00000.parquet", expected)


def test_streamed_merge_buffers_a_bounded_amount(tmp_path: Path, worker, monkeypatch) -> None:
    base = tmp_path / "dataset"
    parquet_dir = _write_dataset(base, 10, 100, drift = False)
    expected = pq.read_table(_reference(parquet_dir, tmp_path))

    written: list[int] = []
    original = pq.ParquetWriter.write_table

    def counting(
        self,
        table,
        row_group_size = None,
    ):
        written.append(table.num_rows)
        return original(self, table, row_group_size = row_group_size)

    monkeypatch.setattr(pq.ParquetWriter, "write_table", counting)
    # Past the byte budget a shard is written straight away instead of waiting for a full row group.
    monkeypatch.setattr(worker, "_MERGE_BUFFER_BYTES", 1)
    worker._merge_batches_to_single_parquet(base)

    # The zero-row writes are the schema probe.
    assert [rows for rows in written if rows] == [10] * 10
    merged = pq.read_table(parquet_dir / "batch_00000.parquet")
    assert merged.equals(expected)


def test_streamed_merge_cuts_row_groups_like_write_table(
    tmp_path: Path, worker, monkeypatch
) -> None:
    base = tmp_path / "dataset"
    parquet_dir = _write_dataset(base, 10, 50, drift = False)
    expected = pq.read_table(_reference(parquet_dir, tmp_path))
    monkeypatch.setattr(worker, "_MERGE_ROW_GROUP_ROWS", 7)
    worker._merge_batches_to_single_parquet(base)

    metadata = pq.ParquetFile(parquet_dir / "batch_00000.parquet").metadata
    assert [metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)] == [7] * 7 + [1]
    assert pq.read_table(parquet_dir / "batch_00000.parquet").equals(expected)


def test_shards_whose_types_differ_take_the_dataframe_merge(tmp_path: Path, worker) -> None:
    # Unify could not settle the types (it logs and keeps them): the dataframe path still decides the merge.
    base = tmp_path / "dataset"
    parquet_dir = base / "parquet-files"
    parquet_dir.mkdir(parents = True)
    pd.DataFrame({"n": [1, 2]}).to_parquet(parquet_dir / "batch_00000.parquet", index = False)
    pd.DataFrame({"n": [1.5]}).to_parquet(parquet_dir / "batch_00001.parquet", index = False)
    calls = []
    io_helpers = sys.modules["data_designer.config.utils.io_helpers"]
    real = io_helpers.read_parquet_dataset

    def recording(path):
        calls.append(path)
        return pd.concat(
            [_dataframe_read(p) for p in sorted(Path(path).glob("*.parquet"))], ignore_index = True
        )

    io_helpers.read_parquet_dataset = recording
    try:
        worker._merge_batches_to_single_parquet(base)
    finally:
        io_helpers.read_parquet_dataset = real

    assert calls == [parquet_dir]
    assert pq.read_table(parquet_dir / "batch_00000.parquet").column("n").to_pylist() == [1, 2, 1.5]


def test_a_stray_file_the_folder_read_would_pick_up_takes_the_dataframe_merge(
    tmp_path: Path, worker
) -> None:
    base = tmp_path / "dataset"
    parquet_dir = _write_dataset(base, 2, 4, drift = False)
    stray = parquet_dir / "batch_00002.parquet.unifying"
    pq.write_table(pq.read_table(parquet_dir / "batch_00001.parquet"), stray)
    expected = pq.read_table(_reference(parquet_dir, tmp_path))
    worker._merge_batches_to_single_parquet(base)

    assert pq.read_table(parquet_dir / "batch_00000.parquet").equals(expected)
    assert expected.num_rows == 6


@pytest.mark.parametrize("name", ["_partial.parquet", ".partial.parquet"])
def test_a_shard_name_the_folder_read_skips_stays_out_of_the_merge(
    tmp_path: Path, worker, name
) -> None:
    # The glob lists "_" and "." names, the folder read skips them: the merge must not take their rows.
    base = tmp_path / "dataset"
    parquet_dir = _write_dataset(base, 2, 4, drift = False)
    pq.write_table(pq.read_table(parquet_dir / "batch_00001.parquet"), parquet_dir / name)
    expected = pq.read_table(_reference(parquet_dir, tmp_path))
    worker._merge_batches_to_single_parquet(base)

    assert expected.num_rows == 4
    assert pq.read_table(parquet_dir / "batch_00000.parquet").equals(expected)


def test_an_unreadable_shard_fails_the_merge_as_the_folder_read_does(
    tmp_path: Path, worker
) -> None:
    base = tmp_path / "dataset"
    parquet_dir = _write_dataset(base, 2, 4, drift = False)
    (parquet_dir / "batch_00001.parquet").write_bytes(b"PAR1 not a parquet file")
    with pytest.raises(pyarrow.ArrowInvalid, match = "batch_00001.parquet"):
        worker._merge_batches_to_single_parquet(base)
    assert sorted(p.name for p in parquet_dir.iterdir()) == [
        "batch_00000.parquet",
        "batch_00001.parquet",
    ]
    assert not list(base.glob("*.merging"))
