# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion training records the same per-process memory peaks as the LLM worker: sampled in the
trainer child, forwarded on progress / complete / error events, kept on the status and persisted on
the run record with the LLM field names (bytes)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

import core.training.diffusion_training_service as dts
from core.training.diffusion_training_service import DiffusionTrainingService
from models.training import (
    DiffusionTrainingRunDetail,
    DiffusionTrainingRunSummary,
    DiffusionTrainingStatusResponse,
)
from utils import process_memory_peaks as pmp

GIB = 1 << 30
FIELDS = (
    "gpu_peak_allocated_bytes",
    "gpu_peak_reserved_bytes",
    "host_peak_rss_bytes",
    "host_peak_commit_bytes",
)


@pytest.fixture(autouse = True)
def _fresh_running_maxima(monkeypatch):
    monkeypatch.setattr(pmp, "_banked_cuda", {}, raising = False)
    monkeypatch.setattr(pmp, "_mps_peak", [None, None], raising = False)
    monkeypatch.setattr(pmp, "_tree_rss_peak", [None], raising = False)


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "diffusion"
    d.mkdir(parents = True)
    monkeypatch.setattr(dts, "_runs_dir", lambda: d)
    return d


def _peaks(alloc, reserved, rss):
    return {
        "gpu_peak_allocated_bytes": alloc,
        "gpu_peak_reserved_bytes": reserved,
        "gpu_device_count": 1 if alloc is not None else None,
        "host_peak_rss_bytes": rss,
        "host_peak_commit_bytes": None,
    }


def _service(job_id = "c" * 32):
    service = DiffusionTrainingService()
    service._state.update(job_id = job_id, status = "running", active = True)
    service._config = {"output_dir": None, "base_model": "stabilityai/sdxl-turbo"}
    return service


def _progress(step, peaks):
    return {"type": "progress", "step": step, "total_steps": 4, "loss": 0.1, "memory_peaks": peaks}


# ---------------------------------------------------------------- service: status + run record


def test_progress_peaks_reach_status_and_never_go_down(runs_dir):
    service = _service()
    service._apply_event(_progress(1, _peaks(2 * GIB, 3 * GIB, 5 * GIB)))
    service._apply_event(_progress(2, _peaks(1 * GIB, 4 * GIB, 4 * GIB)))
    peaks = service.status()["memory_peaks"]
    assert peaks["gpu_peak_allocated_bytes"] == 2 * GIB
    assert peaks["gpu_peak_reserved_bytes"] == 4 * GIB
    assert peaks["host_peak_rss_bytes"] == 5 * GIB
    assert peaks["host_peak_commit_bytes"] is None


def test_status_response_carries_the_llm_shape():
    snap = dts._idle_state()
    assert "memory_peaks" in snap and snap["memory_peaks"] is None
    for key in [k for k in snap if k.startswith("metric_")]:
        snap.pop(key)
    assert DiffusionTrainingStatusResponse(**snap).memory_peaks is None
    snap["memory_peaks"] = _peaks(GIB, 2 * GIB, 3 * GIB)
    body = DiffusionTrainingStatusResponse(**snap).model_dump()
    assert body["memory_peaks"]["gpu_peak_reserved_bytes"] == 2 * GIB
    assert body["memory_peaks"]["gpu_device_count"] == 1


def test_complete_peaks_are_persisted_on_the_run_record(runs_dir):
    service = _service()
    service._apply_event(_progress(1, _peaks(2 * GIB, 3 * GIB, 5 * GIB)))
    service._apply_event(
        {"type": "complete", "stopped": False, "memory_peaks": _peaks(GIB, GIB, 6 * GIB)}
    )
    service._persist_run_record()
    rec = json.loads((runs_dir / f"{'c' * 32}.json").read_text())
    assert rec["gpu_peak_allocated_bytes"] == 2 * GIB
    assert rec["gpu_peak_reserved_bytes"] == 3 * GIB
    assert rec["host_peak_rss_bytes"] == 6 * GIB
    assert rec["host_peak_commit_bytes"] is None
    summaries = dts.list_diffusion_runs()
    assert DiffusionTrainingRunSummary(**summaries[0]).host_peak_rss_bytes == 6 * GIB
    detail = DiffusionTrainingRunDetail(**dts.get_diffusion_run("c" * 32))
    assert detail.gpu_peak_allocated_bytes == 2 * GIB


def test_error_event_peaks_reach_the_errored_record(runs_dir):
    service = _service()
    service._apply_event(
        {
            "type": "error",
            "message": "CUDA out of memory",
            "memory_peaks": _peaks(7 * GIB, 8 * GIB, GIB),
        }
    )
    service._persist_run_record()
    rec = json.loads((runs_dir / f"{'c' * 32}.json").read_text())
    assert rec["status"] == "error"
    assert rec["gpu_peak_reserved_bytes"] == 8 * GIB


def test_junk_peaks_are_dropped_and_cpu_runs_keep_none(runs_dir):
    service = _service()
    service._apply_event(
        _progress(1, {"gpu_peak_allocated_bytes": "lots", "host_peak_rss_bytes": -1})
    )
    service._apply_event(
        {"type": "complete", "stopped": False, "memory_peaks": _peaks(None, None, GIB)}
    )
    service._persist_run_record()
    rec = json.loads((runs_dir / f"{'c' * 32}.json").read_text())
    assert rec["gpu_peak_allocated_bytes"] is None
    assert rec["gpu_peak_reserved_bytes"] is None
    assert rec["host_peak_rss_bytes"] == GIB


def test_old_records_without_peaks_still_load(runs_dir):
    (runs_dir / f"{'d' * 32}.json").write_text(
        json.dumps({"job_id": "d" * 32, "status": "completed"})
    )
    (summary,) = dts.list_diffusion_runs()
    parsed = DiffusionTrainingRunSummary(**summary)
    assert all(getattr(parsed, f) is None for f in FIELDS)


def test_a_new_job_starts_without_the_previous_jobs_peaks():
    assert dts._idle_state()["memory_peaks"] is None


# ---------------------------------------------------------------- trainer child: event stamping


class _ListQueue:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)


def _run_child(monkeypatch, trainer):
    import core.training.diffusion_lora_trainer as dlt

    monkeypatch.setattr(dlt, "get_trainer", lambda family: trainer)
    queue = _ListQueue()
    dlt.run_diffusion_training_process(
        event_queue = queue,
        stop_queue = SimpleNamespace(empty = lambda: True),
        config = {
            "base_model": "stabilityai/sdxl-turbo",
            "data_dir": "/nonexistent",
            "output_dir": "/tmp/x",
        },
    )
    return queue.events


def test_trainer_progress_and_complete_events_carry_peaks_on_cpu(monkeypatch):
    def trainer(cfg, on_event, should_stop):
        on_event({"type": "progress", "step": 1, "total_steps": 1, "loss": 0.5})
        on_event({"type": "warning", "message": "note"})
        on_event({"type": "complete", "stopped": False})

    events = _run_child(monkeypatch, trainer)
    by_type = {e["type"]: e for e in events}
    assert "memory_peaks" not in by_type["warning"]
    for kind in ("progress", "complete"):
        peaks = by_type[kind]["memory_peaks"]
        assert set(FIELDS) <= set(peaks)
        if sys.platform != "win32":
            assert peaks["host_peak_rss_bytes"] > 16 * (1 << 20)
            assert peaks["host_peak_commit_bytes"] is None
        torch = sys.modules.get("torch")
        if torch is None or not torch.cuda.is_available():
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is None or not mps.is_available():
                assert peaks["gpu_peak_allocated_bytes"] is None
                assert peaks["gpu_peak_reserved_bytes"] is None


def test_trainer_crash_error_event_carries_peaks(monkeypatch):
    def trainer(cfg, on_event, should_stop):
        raise RuntimeError("boom")

    (event,) = [e for e in _run_child(monkeypatch, trainer) if e["type"] == "error"]
    assert event["message"] == "boom"
    assert event["memory_peaks"]["host_peak_rss_bytes"] is not None or sys.platform == "win32"


def test_stamping_never_imports_torch(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising = False)
    pmp.process_memory_peaks(include_children = True)
    assert "torch" not in sys.modules


# ---------------------------------------------------------------- helper: MPS, banking, tree RSS


class _FakeMps:
    def __init__(self, samples):
        self._samples = list(samples)
        self._cur = None

    def next(self):
        self._cur = self._samples.pop(0)

    def current_allocated_memory(self):
        return self._cur[0]

    def driver_allocated_memory(self):
        return self._cur[1]


def _mps_torch(mps, available = True):
    return SimpleNamespace(
        cuda = SimpleNamespace(is_available = lambda: False),
        backends = SimpleNamespace(mps = SimpleNamespace(is_available = lambda: available)),
        mps = mps,
    )


def test_mps_reports_a_running_max_of_allocated_and_driver_memory():
    mps = _FakeMps([(2 * GIB, 3 * GIB), (1 * GIB, 5 * GIB)])
    torch = _mps_torch(mps)
    mps.next()
    pmp.process_memory_peaks(torch)
    mps.next()
    peaks = pmp.process_memory_peaks(torch)
    assert peaks["gpu_peak_allocated_bytes"] == 2 * GIB
    assert peaks["gpu_peak_reserved_bytes"] == 5 * GIB
    assert peaks["gpu_device_count"] == 1


def test_mps_unavailable_or_broken_degrades_to_none():
    assert (
        pmp.process_memory_peaks(_mps_torch(_FakeMps([]), available = False))[
            "gpu_peak_allocated_bytes"
        ]
        is None
    )

    class Broken:
        def current_allocated_memory(self):
            raise RuntimeError("no mps")

    assert pmp.process_memory_peaks(_mps_torch(Broken()))["gpu_peak_reserved_bytes"] is None


class _ResettableCuda:
    def __init__(self):
        self.alloc, self.reserved = 6 * GIB, 7 * GIB

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def max_memory_allocated(self, i):
        return self.alloc

    def max_memory_reserved(self, i):
        return self.reserved

    def reset_peak_memory_stats(self):
        self.alloc, self.reserved = GIB, GIB


def test_trainer_peak_reset_banks_the_process_peak_first(monkeypatch):
    """The trainer resets the CUDA counters to exclude load and previews from its own figure;
    the process peak must still cover them, as on the LLM side where nothing resets them."""
    from core.training import diffusion_train_common as common

    cuda = _ResettableCuda()
    torch = SimpleNamespace(cuda = cuda)
    monkeypatch.setitem(sys.modules, "torch", torch)
    common.reset_training_peak_memory("cuda")
    assert cuda.alloc == GIB  # the trainer's own counter really was reset
    peaks = pmp.process_memory_peaks(torch)
    assert peaks["gpu_peak_allocated_bytes"] == 6 * GIB
    assert peaks["gpu_peak_reserved_bytes"] == 7 * GIB


def test_reset_on_cpu_is_a_no_op(monkeypatch):
    from core.training import diffusion_train_common as common
    common.reset_training_peak_memory("cpu")
    common.reset_training_peak_memory("mps")


def test_tree_rss_includes_live_children(monkeypatch):
    psutil = pytest.importorskip("psutil")
    # Compare the sampled tree against this process's current RSS: the worker's lifetime ru_maxrss
    # can already exceed current RSS plus the child, which would hide the child entirely.
    monkeypatch.setattr(pmp, "_tree_rss_peak", [None])
    own_now = int(psutil.Process().memory_info().rss)
    size = 300 * (1 << 20)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"import sys,time; b = b'x' * {size}; print('ok', flush=True); time.sleep(60)",
        ],
        stdout = subprocess.PIPE,
    )
    try:
        assert child.stdout.readline().strip() == b"ok"
        tree = pmp._tree_rss()
        reported = pmp.process_memory_peaks(include_children = True)["host_peak_rss_bytes"]
    finally:
        child.kill()
        child.wait()
    assert tree >= own_now + size * 0.9
    assert reported >= tree  # the reported peak covers the sampled tree
    # Running max: the child is gone, the recorded tree peak is not.
    time.sleep(0.1)
    assert pmp._tree_rss() >= tree
    assert pmp.process_memory_peaks(include_children = True)["host_peak_rss_bytes"] >= tree
    # Without children the value is exactly the LLM worker's own-process peak.
    assert pmp.process_memory_peaks()["host_peak_rss_bytes"] == pmp.host_peaks()[0]


def test_tree_rss_without_psutil_keeps_own_peak(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    peaks = pmp.process_memory_peaks(include_children = True)
    if sys.platform != "win32":
        assert peaks["host_peak_rss_bytes"] == pmp.host_peaks()[0]
