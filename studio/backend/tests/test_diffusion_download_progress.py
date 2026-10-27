# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from core.training.diffusion_download_progress import TrainingDownloadProgress, download_snapshot


def bar(
    done = 4,
    total = 10,
    desc = "weights.safetensors",
    unit = "B",
    disable = False,
):
    return SimpleNamespace(n = done, total = total, desc = desc, unit = unit, disable = disable)


def test_download_snapshot_counts_only_active_bytes():
    snapshot = download_snapshot([bar(), bar(1, 100, unit = "it"), bar(disable = True)])
    assert snapshot == {
        "bytes_done": 4,
        "bytes_total": 10,
        "label": "weights.safetensors",
        "scope": "active_files",
    }
    assert download_snapshot([bar(0)])["bytes_done"] == 0


def test_parallel_file_downloads_share_one_current_batch():
    snapshot = download_snapshot([bar(), bar(6, 20, desc = "encoder.safetensors")])
    assert snapshot["bytes_done"] == 10
    assert snapshot["bytes_total"] == 30
    assert snapshot["label"] == "2 model files"


def test_xet_transfer_and_reconstruction_are_not_counted_twice():
    snapshot = download_snapshot(
        [
            bar(10, 10, "Downloading bytes"),
            bar(4, 10, "Reconstructing (incomplete total...)"),
        ]
    )
    assert snapshot["bytes_done"] == 4
    assert snapshot["bytes_total"] == 10


@pytest.mark.parametrize("done, total", [(-1, 10), (float("nan"), 10), (1, float("inf")), (1, 0)])
def test_invalid_or_unknown_byte_counts_stay_indeterminate(done, total):
    assert download_snapshot([bar(done, total)]) is None


def test_download_progress_clears_when_file_bars_close():
    events = []
    current = [bar()]
    monitor = TrainingDownloadProgress(events.append, sample = lambda: download_snapshot(current))
    monitor.poll()
    assert events == []
    monitor.report({"type": "model_load_started"})
    monitor.poll()
    monitor.poll()
    assert len(events) == 2
    current.clear()
    monitor.poll()
    assert events[-1] == {"type": "model_download_progress", "download_progress": None}


@pytest.mark.parametrize(
    "phase", ["preparing", "model_load_completed", "progress", "complete", "error"]
)
def test_byte_progress_cannot_replace_later_training_phases(phase):
    events = []
    monitor = None

    def sample():
        monitor.report({"type": phase})
        return download_snapshot([bar()])

    monitor = TrainingDownloadProgress(events.append, sample = sample)
    monitor.report({"type": "model_load_started"})
    monitor.poll()
    assert [event["type"] for event in events] == ["model_load_started", phase]


def test_monitor_thread_is_stopped_on_a_trainer_error():
    monitor = TrainingDownloadProgress(lambda _ev: None, interval = 3600)
    with pytest.raises(ValueError):
        with monitor as report:
            report({"type": "model_load_started"})
            raise ValueError("trainer failed")
    assert not monitor._thread.is_alive()
