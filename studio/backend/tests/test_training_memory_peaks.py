# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-process memory peaks: measured in the training worker, forwarded on progress / complete
events, kept on the status API and persisted on the run record."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import core.training.training as tr
from core.training.training import TrainingBackend
from models.training import TrainingProgress, TrainingRunSummary, TrainingStatus
from storage import studio_db
from utils import process_memory_peaks as pmp

GIB = 1 << 30


class _FakeCuda:
    def __init__(
        self,
        allocated,
        reserved,
        available = True,
    ):
        self._allocated = allocated
        self._reserved = reserved
        self._available = available

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._allocated)

    def max_memory_allocated(self, i):
        return self._allocated[i]

    def max_memory_reserved(self, i):
        return self._reserved[i]


def _torch(cuda):
    return SimpleNamespace(cuda = cuda)


def _fake_resource(monkeypatch, maxrss):
    fake = types.ModuleType("resource")
    fake.RUSAGE_SELF = 0
    fake.getrusage = lambda who: SimpleNamespace(ru_maxrss = maxrss)
    monkeypatch.setitem(sys.modules, "resource", fake)


# ---------------------------------------------------------------- helper


def test_linux_ru_maxrss_is_kib(monkeypatch):
    monkeypatch.setattr(pmp.sys, "platform", "linux")
    _fake_resource(monkeypatch, 2048)
    assert pmp.host_peaks() == (2048 * 1024, None)


def test_macos_ru_maxrss_is_bytes(monkeypatch):
    monkeypatch.setattr(pmp.sys, "platform", "darwin")
    _fake_resource(monkeypatch, 3 * GIB)
    assert pmp.host_peaks() == (3 * GIB, None)


def test_windows_reads_peak_working_set_and_commit_from_psutil(monkeypatch):
    monkeypatch.setattr(pmp.sys, "platform", "win32")
    info = SimpleNamespace(peak_wset = 5 * GIB, peak_pagefile = 7 * GIB)
    fake_psutil = types.ModuleType("psutil")
    fake_psutil.Process = lambda: SimpleNamespace(memory_info = lambda: info)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    assert pmp.host_peaks() == (5 * GIB, 7 * GIB)


def test_windows_without_psutil_falls_back_to_get_process_memory_info(monkeypatch):
    import ctypes

    monkeypatch.setattr(pmp.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", None)  # import psutil -> ImportError

    def get_info(handle, counters_ref, cb):
        counters = counters_ref._obj
        assert cb == ctypes.sizeof(counters)
        counters.PeakWorkingSetSize = 4 * GIB
        counters.PeakPagefileUsage = 9 * GIB
        return 1

    kernel32 = SimpleNamespace(GetCurrentProcess = lambda: 0, K32GetProcessMemoryInfo = get_info)
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda name, use_last_error = False: kernel32, raising = False
    )
    assert pmp.host_peaks() == (4 * GIB, 9 * GIB)


def test_windows_with_nothing_available_gives_none(monkeypatch):
    import ctypes

    monkeypatch.setattr(pmp.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.delattr(ctypes, "WinDLL", raising = False)
    assert pmp.host_peaks() == (None, None)


def test_gpu_peaks_sum_over_devices():
    cuda = _FakeCuda([1 * GIB, 2 * GIB], [3 * GIB, 4 * GIB])
    peaks = pmp.process_memory_peaks(_torch(cuda))
    assert peaks["gpu_peak_allocated_bytes"] == 3 * GIB
    assert peaks["gpu_peak_reserved_bytes"] == 7 * GIB
    assert peaks["gpu_device_count"] == 2


def test_cuda_unavailable_gives_none_gpu_fields(monkeypatch):
    monkeypatch.setattr(pmp.sys, "platform", "linux")
    _fake_resource(monkeypatch, 1024)
    peaks = pmp.process_memory_peaks(_torch(_FakeCuda([1], [1], available = False)))
    assert peaks["gpu_peak_allocated_bytes"] is None
    assert peaks["gpu_peak_reserved_bytes"] is None
    assert peaks["gpu_device_count"] is None
    assert peaks["host_peak_rss_bytes"] == 1024 * 1024


def test_no_torch_loaded_never_imports_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    allocated, reserved, count = pmp._gpu_peaks()
    assert (allocated, reserved, count) == (None, None, None)


def test_nothing_measurable_returns_none(monkeypatch):
    monkeypatch.setattr(pmp, "host_peaks", lambda: (None, None))
    assert pmp.process_memory_peaks(_torch(_FakeCuda([], [], available = False))) is None


def test_merge_keeps_the_max_and_ignores_junk():
    a = {"gpu_peak_allocated_bytes": 10, "host_peak_rss_bytes": 50}
    merged = pmp.merge_memory_peaks(None, a)
    merged = pmp.merge_memory_peaks(
        merged,
        {"gpu_peak_allocated_bytes": 5, "host_peak_rss_bytes": 60, "gpu_peak_reserved_bytes": "x"},
    )
    assert merged["gpu_peak_allocated_bytes"] == 10
    assert merged["host_peak_rss_bytes"] == 60
    assert merged["gpu_peak_reserved_bytes"] is None
    assert pmp.merge_memory_peaks(merged, None) is merged


# ---------------------------------------------------------------- forwarding + record


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.setattr(tr, "account_is_retired", lambda: False)
    studio_db.create_run(
        id = "run-1",
        model_name = "m",
        dataset_name = "d",
        config_json = "{}",
        started_at = "2026-01-01T00:00:00+00:00",
        total_steps = 10,
    )
    b = TrainingBackend()
    b.current_job_id = "run-1"
    b._db_run_created = True
    yield b
    studio_db.close_wal_keeper()


_PEAKS = {
    "gpu_peak_allocated_bytes": 6 * GIB,
    "gpu_peak_reserved_bytes": 8 * GIB,
    "gpu_device_count": 1,
    "host_peak_rss_bytes": 12 * GIB,
    "host_peak_commit_bytes": None,
}


def test_progress_and_complete_events_carry_peaks_into_status_and_record(backend):
    backend._handle_event(
        {"type": "progress", "step": 1, "total_steps": 10, "loss": 1.0, "memory_peaks": _PEAKS}
    )
    assert backend._progress.memory_peaks["gpu_peak_reserved_bytes"] == 8 * GIB

    # A later, larger host peak arrives with the terminal event; GPU peak stays the max.
    backend._handle_event(
        {
            "type": "complete",
            "output_dir": None,
            "status_message": "Training completed",
            "memory_peaks": {
                **_PEAKS,
                "gpu_peak_allocated_bytes": 1,
                "host_peak_rss_bytes": 13 * GIB,
            },
        }
    )
    peaks = backend._progress.memory_peaks
    assert peaks["gpu_peak_allocated_bytes"] == 6 * GIB
    assert peaks["host_peak_rss_bytes"] == 13 * GIB

    status = TrainingStatus(
        job_id = "run-1",
        phase = "training",
        is_training_running = True,
        message = "",
        memory_peaks = peaks,
    )
    dumped = status.model_dump()
    assert dumped["memory_peaks"]["gpu_peak_allocated_bytes"] == 6 * GIB
    assert dumped["memory_peaks"]["host_peak_commit_bytes"] is None
    sse = TrainingProgress(
        job_id = "run-1", step = 1, total_steps = 10, progress_percent = 10.0, memory_peaks = peaks
    )
    assert sse.memory_peaks.host_peak_rss_bytes == 13 * GIB

    backend._finalize_run_in_db(status = "completed")
    row = studio_db.get_run("run-1")
    assert row["gpu_peak_allocated_bytes"] == 6 * GIB
    assert row["gpu_peak_reserved_bytes"] == 8 * GIB
    assert row["host_peak_rss_bytes"] == 13 * GIB
    assert row["host_peak_commit_bytes"] is None
    listed = studio_db.list_runs()["runs"][0]
    summary = TrainingRunSummary(**{k: v for k, v in listed.items() if k != "config_json"})
    assert summary.host_peak_rss_bytes == 13 * GIB


def test_status_json_omits_peaks_until_reported():
    status = TrainingStatus(job_id = "", phase = "idle", is_training_running = False, message = "")
    assert "memory_peaks" not in status.model_dump()


def test_old_database_gains_nullable_columns_and_old_rows_load(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    # Build a pre-change training_runs table holding one old row.
    conn = studio_db.get_connection()
    try:
        conn.execute("DROP TABLE IF EXISTS training_metrics")
        conn.execute("DROP TABLE training_runs")
        conn.execute(
            """
            CREATE TABLE training_runs (
                id TEXT NOT NULL PRIMARY KEY, status TEXT NOT NULL DEFAULT 'running',
                model_name TEXT NOT NULL, dataset_name TEXT NOT NULL, config_json TEXT NOT NULL,
                started_at TEXT NOT NULL, ended_at TEXT, total_steps INTEGER, final_step INTEGER,
                final_loss REAL, output_dir TEXT, error_message TEXT, duration_seconds REAL,
                loss_sparkline TEXT, display_name TEXT,
                resume_blocked INTEGER NOT NULL DEFAULT 0, resumed_from_run_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO training_runs (id, status, model_name, dataset_name, config_json, started_at) "
            "VALUES ('old', 'completed', 'm', 'd', '{}', '2025-01-01T00:00:00+00:00')"
        )
        conn.commit()
    finally:
        conn.close()
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    try:
        row = studio_db.get_run("old")
        assert row is not None
        for column in pmp.MEMORY_PEAK_FIELDS:
            assert row[column] is None
        listed = studio_db.list_runs()["runs"]
        summary = TrainingRunSummary(**{k: v for k, v in listed[0].items() if k != "config_json"})
        assert summary.gpu_peak_allocated_bytes is None
    finally:
        studio_db.close_wal_keeper()


def test_worker_sends_peaks_on_progress_and_complete():
    """The worker is the process that trains, so it must be the one that samples."""
    import inspect

    import core.training.worker as worker

    src = inspect.getsource(worker)
    assert src.count('"memory_peaks": _process_memory_peaks()') >= 2
    assert "memory_peaks = _process_memory_peaks()" in src  # MLX progress


class _ListQueue:
    def __init__(self):
        self.items = []
        self.closed = False

    def put(
        self,
        item,
        block = True,
        timeout = None,
    ):
        self.items.append(item)

    def close(self):
        self.closed = True


def test_stamping_queue_samples_every_terminal_event(monkeypatch):
    sample = {**_PEAKS, "host_peak_rss_bytes": 20 * GIB}
    monkeypatch.setattr(pmp, "process_memory_peaks", lambda: sample)
    inner = _ListQueue()
    q = pmp.PeakStampingQueue(inner)
    q.put({"type": "error", "error": "CUDA out of memory"})
    q.put({"type": "complete", "output_dir": None})  # the MLX completion path sends no sample
    q.put({"type": "complete", "memory_peaks": _PEAKS})  # an existing sample is kept
    q.put({"type": "progress", "step": 1})  # progress samples are taken by the callback
    q.put({"type": "status", "message": "x"})
    q.put("not a dict")
    assert [e.get("memory_peaks") if isinstance(e, dict) else e for e in inner.items] == [
        sample,
        sample,
        _PEAKS,
        None,
        None,
        "not a dict",
    ]
    q.close()  # other queue methods pass through
    assert inner.closed


def test_stamping_queue_never_blocks_an_event_on_a_failed_sample(monkeypatch):
    def boom():
        raise RuntimeError("no reading")

    monkeypatch.setattr(pmp, "process_memory_peaks", boom)
    inner = _ListQueue()
    pmp.PeakStampingQueue(inner).put({"type": "error", "error": "x"})
    assert inner.items == [{"type": "error", "error": "x"}]


def test_error_event_peak_reaches_the_errored_run_record(backend):
    backend._handle_event(
        {"type": "progress", "step": 1, "total_steps": 10, "memory_peaks": _PEAKS}
    )
    backend._handle_event(
        {
            "type": "error",
            "error": "CUDA out of memory",
            "memory_peaks": {**_PEAKS, "gpu_peak_reserved_bytes": 79 * GIB},
        }
    )
    assert backend._progress.memory_peaks["gpu_peak_reserved_bytes"] == 79 * GIB
    backend._finalize_run_in_db(status = "error")
    assert studio_db.get_run("run-1")["gpu_peak_reserved_bytes"] == 79 * GIB


def test_worker_wraps_its_event_queue():
    import inspect

    import core.training.worker as worker

    src = inspect.getsource(worker.run_training_process)
    assert "event_queue = _PeakStampingQueue(event_queue)" in src
