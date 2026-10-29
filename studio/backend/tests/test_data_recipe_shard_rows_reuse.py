# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Recipe pages reuse shard row counts only while no shard changed, and read the same rows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.data_recipe.jobs import manager as manager_module  # noqa: E402
from core.data_recipe.jobs.manager import JobManager  # noqa: E402

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")


@pytest.fixture
def metadata_queries(monkeypatch):
    calls = []
    real_connect = duckdb.connect

    class _Conn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            if "parquet_file_metadata" in sql:
                calls.append(sql)
            return self._conn.execute(sql, *args, **kwargs)

        def close(self):
            return self._conn.close()

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: _Conn(real_connect(*a, **k)))
    cache = getattr(manager_module, "_SHARD_ROWS_CACHE", None)
    if cache is not None:
        monkeypatch.setattr(manager_module, "_SHARD_ROWS_CACHE", {})
    return calls


def _settle(monkeypatch):
    # Test files are seconds old; treat every stat as settled.
    monkeypatch.setattr(manager_module, "_SHARD_ROWS_SETTLE_NS", 0, raising = False)


def _write(parquet_dir: Path, index: int, values) -> Path:
    path = parquet_dir / f"batch_file_{index:05d}.parquet"
    pd.DataFrame({"a": list(values), "text": [f"row {v}" for v in values]}).to_parquet(
        path, index = False
    )
    return path


def _page(parquet_dir: Path, offset: int, limit: int):
    return JobManager._load_dataset_page_with_duckdb(
        parquet_dir = parquet_dir, limit = limit, offset = offset
    )


def _fresh_page(parquet_dir: Path, offset: int, limit: int):
    saved = dict(getattr(manager_module, "_SHARD_ROWS_CACHE", {}))
    getattr(manager_module, "_SHARD_ROWS_CACHE", {}).clear()
    try:
        return _page(parquet_dir, offset, limit)
    finally:
        getattr(manager_module, "_SHARD_ROWS_CACHE", {}).clear()
        getattr(manager_module, "_SHARD_ROWS_CACHE", {}).update(saved)


def test_repeated_page_reads_shard_footers_once(tmp_path, metadata_queries, monkeypatch):
    _settle(monkeypatch)
    for i in range(4):
        _write(tmp_path, i, range(i * 5, i * 5 + 5))
    first = _page(tmp_path, 0, 7)
    for _ in range(3):
        assert _page(tmp_path, 0, 7) == first
    assert _page(tmp_path, 3, 9) == _fresh_page(tmp_path, 3, 9)
    assert len(metadata_queries) == 2  # one shared, one for the uncached comparison


def test_fresh_shards_are_not_reused(tmp_path, metadata_queries):
    _write(tmp_path, 0, range(3))
    _page(tmp_path, 0, 5)
    _page(tmp_path, 0, 5)
    assert len(metadata_queries) == 2


@pytest.mark.parametrize("offset,limit", [(0, 5), (4, 6), (9, 4), (12, 20), (30, 5)])
def test_pages_follow_every_shard_change(tmp_path, metadata_queries, monkeypatch, offset, limit):
    _settle(monkeypatch)
    for i in range(3):
        _write(tmp_path, i, range(i * 5, i * 5 + 5))
    assert _page(tmp_path, offset, limit) == _fresh_page(tmp_path, offset, limit)

    _write(tmp_path, 3, range(15, 22))
    assert _page(tmp_path, offset, limit) == _fresh_page(tmp_path, offset, limit)
    assert _page(tmp_path, offset, limit)["total"] == 22

    path = tmp_path / "batch_file_00001.parquet"
    size = path.stat().st_size
    _write(tmp_path, 1, [9, 8, 7, 6, 5])
    assert path.stat().st_size == size
    assert _page(tmp_path, offset, limit) == _fresh_page(tmp_path, offset, limit)

    stat = path.stat()
    _write(tmp_path, 1, range(200, 209))
    os.utime(path, ns = (stat.st_atime_ns, stat.st_mtime_ns))
    assert _page(tmp_path, offset, limit) == _fresh_page(tmp_path, offset, limit)
    assert _page(tmp_path, offset, limit)["total"] == 26

    (tmp_path / "batch_file_00000.parquet").unlink()
    assert _page(tmp_path, offset, limit) == _fresh_page(tmp_path, offset, limit)
    pd.DataFrame({"a": [-1], "text": ["dot"]}).to_parquet(tmp_path / "._x.parquet", index = False)
    assert _page(tmp_path, offset, limit) == _fresh_page(tmp_path, offset, limit)


def test_emptied_dir_is_not_served_from_cache(tmp_path, metadata_queries, monkeypatch):
    _settle(monkeypatch)
    path = _write(tmp_path, 0, range(3))
    assert _page(tmp_path, 0, 5)["total"] == 3
    path.unlink()
    assert _page(tmp_path, 0, 5) is None
    assert _fresh_page(tmp_path, 0, 5) is None


def test_added_shard_is_seen_when_the_dir_stamps_do_not_move(
    tmp_path, metadata_queries, monkeypatch
):
    # FAT leaves a directory's mtime alone when an entry is added.
    _settle(monkeypatch)
    real_stat_key = manager_module._stat_key

    def frozen_dir_stat_key(path, settled_before_ns):
        if os.path.isdir(path):
            return ("dir",)
        return real_stat_key(path, settled_before_ns)

    monkeypatch.setattr(manager_module, "_stat_key", frozen_dir_stat_key)
    _write(tmp_path, 0, range(5))
    assert _page(tmp_path, 0, 20)["total"] == 5
    _write(tmp_path, 1, range(5, 12))
    assert _page(tmp_path, 0, 20) == _fresh_page(tmp_path, 0, 20)
    assert _page(tmp_path, 0, 20)["total"] == 12


def test_shard_rewritten_during_the_footer_read_is_not_reused(tmp_path, monkeypatch):
    _settle(monkeypatch)
    monkeypatch.setattr(manager_module, "_SHARD_ROWS_CACHE", {})
    _write(tmp_path, 0, range(5))
    real_connect = duckdb.connect

    class _Conn:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args, **kwargs):
            result = self._conn.execute(sql, *args, **kwargs)
            if "parquet_file_metadata" in sql and not rewrites:
                rows = result.fetchall()
                # The writer replaces the shard after its footer was read, before the stat.
                rewrites.append(_write(tmp_path, 0, range(9)))
                return _Rows(rows)
            return result

        def close(self):
            return self._conn.close()

    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    rewrites = []
    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: _Conn(real_connect(*a, **k)))
    assert _page(tmp_path, 0, 20)["total"] == 5
    assert rewrites
    assert _page(tmp_path, 0, 20)["total"] == 9
