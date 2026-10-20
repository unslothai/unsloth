# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rows of an active full run are readable once a batch lands in parquet-files."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.data_recipe.jobs.constants import EVENT_JOB_STARTED  # noqa: E402
from core.data_recipe.jobs.manager import JobManager  # noqa: E402
from core.data_recipe.jobs.types import Job  # noqa: E402

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
pytest.importorskip("duckdb")


def _manager(status: str) -> JobManager:
    m = JobManager.__new__(JobManager)
    m._lock = threading.Lock()
    m._job = Job(job_id = "job-live")
    m._job.status = status
    m._emit = lambda event: None
    return m


def test_started_event_publishes_planned_artifact_path(tmp_path):
    m = _manager("pending")
    planned = str(tmp_path / "recipe_x")
    m._handle_event(
        m._job,
        {"type": EVENT_JOB_STARTED, "artifact_path": planned, "execution_type": "full"},
    )
    assert m._job.status == "active"
    assert m._job.artifact_path == planned
    assert m._job.execution_type == "full"


def test_active_run_reads_landed_batches(tmp_path):
    m = _manager("active")
    base = tmp_path / "recipe_x"
    m._job.artifact_path = str(base)
    assert m.get_dataset("job-live", limit = 10) is None

    parquet_dir = base / "parquet-files"
    parquet_dir.mkdir(parents = True)
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(parquet_dir / "batch_00000.parquet", index = False)
    page = m.get_dataset("job-live", limit = 10)
    assert page is not None and "error" not in page
    assert page["total"] == 3
    assert [row["a"] for row in page["dataset"]] == [1, 2, 3]


@pytest.mark.parametrize("status", ["completed", "error", "cancelled"])
def test_finished_run_without_parquet_is_an_error(tmp_path, status):
    m = _manager(status)
    m._job.artifact_path = str(tmp_path / "recipe_x")
    assert "error" in m.get_dataset("job-live", limit = 10)


def test_active_run_never_uses_the_fallback_reader(tmp_path, monkeypatch):
    m = _manager("active")
    parquet_dir = tmp_path / "recipe_x" / "parquet-files"
    parquet_dir.mkdir(parents = True)
    m._job.artifact_path = str(parquet_dir.parent)
    monkeypatch.setattr(
        JobManager, "_load_dataset_page_with_duckdb", staticmethod(lambda **_: None)
    )

    def _fallback(**_):
        raise AssertionError("fallback reader used during an active run")

    monkeypatch.setattr(
        JobManager, "_load_dataset_page_with_data_designer", staticmethod(_fallback)
    )
    assert m.get_dataset("job-live", limit = 10) is None


@pytest.mark.parametrize("offset,limit", [(0, 5), (3, 6), (9, 4), (12, 20), (20, 5)])
def test_duckdb_page_matches_shards_read_in_order(tmp_path, offset, limit):
    sizes = [4, 5, 3, 7]
    start = 0
    for i, n in enumerate(sizes):
        pd.DataFrame({"a": list(range(start, start + n))}).to_parquet(
            tmp_path / f"batch_{i:05d}.parquet", index = False
        )
        start += n
    page = JobManager._load_dataset_page_with_duckdb(
        parquet_dir = tmp_path, limit = limit, offset = offset
    )
    assert page["total"] == sum(sizes)
    assert [row["a"] for row in page["dataset"]] == list(range(sum(sizes)))[offset : offset + limit]
