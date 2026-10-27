# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import queue
import threading

import pytest

from core.training.diffusion_training_service import DiffusionTrainingService
from hub.schemas.downloads import CancelDownloadRequest
from hub.services import download_lifecycle, training_downloads
from hub.services.models import downloads
from hub.utils import download_registry


class BlockingDownload:
    """A native transfer which never consumes the trainer's cooperative stop queue."""

    def __init__(self):
        self.done = threading.Event()
        self.thread = threading.Thread(target = self.done.wait, daemon = True)
        self.thread.start()
        self.terminations = 0

    def is_alive(self):
        return self.thread.is_alive()

    def terminate(self):
        self.terminations += 1
        self.done.set()

    def join(self, timeout):
        self.thread.join(timeout)


@pytest.fixture
def rig(monkeypatch):
    registry = download_registry.DownloadRegistry()
    monkeypatch.setattr(download_registry, "get_models_registry", lambda: registry)
    monkeypatch.setattr(downloads, "_registry", registry)
    service = DiffusionTrainingService()
    service._proc = BlockingDownload()
    service._stop_queue = queue.Queue()
    service._state.update(
        job_id = "a" * 32,
        active = True,
        status = "running",
        base_model = "unsloth/FLUX.1-dev",
        in_model_load = True,
    )
    service._download_key = training_downloads.claim(service, "a" * 32, "unsloth/FLUX.1-dev")
    yield service, registry, service._download_key
    service._proc.terminate()
    service._proc.join(1)


def test_metadata_stage_is_discoverable_and_cancellable_through_main_manager(rig):
    service, registry, key = rig
    rows = download_lifecycle.active_download_refs(registry, None, with_variant = True)
    assert [(r.repo_id, r.owner, r.variant) for r in rows] == [
        ("unsloth/FLUX.1-dev", "training", "@training-" + "a" * 32)
    ]
    reading = asyncio.run(
        downloads.get_gguf_download_progress_response(rows[0].repo_id, rows[0].variant)
    )
    assert reading["expected_bytes"] == 0
    assert reading["progress_scope"] == "active_files"
    result = asyncio.run(
        downloads.cancel_download_model_response(
            CancelDownloadRequest(
                repo_id = rows[0].repo_id,
                gguf_variant = rows[0].variant,
                generation = rows[0].generation,
            )
        )
    )
    assert result["state"] == "cancelled"
    assert not service._proc.is_alive()
    assert service._stop_queue.empty(), "blocked Xet never reads this queue"
    assert service.status()["status"] == "stopped"
    assert registry.get_job(key).state == "cancelled"


def test_current_files_never_claim_the_entire_model_is_installed(rig):
    service, registry, key = rig
    service._apply_event(
        {
            "type": "model_download_progress",
            "download_progress": {
                "bytes_done": 10,
                "bytes_total": 10,
                "label": "Reconstructing (incomplete total...)",
            },
        }
    )
    reading = training_downloads.progress(key)
    assert (reading["downloaded_bytes"], reading["expected_bytes"]) == (10, 10)
    assert reading["complete_on_disk"] is False
    assert reading["completed_bytes"] == 0
    service._apply_event({"type": "preparing", "stage": "cache_latents", "done": 1, "total": 60})
    assert registry.get_job(key).state == "idle"
    assert training_downloads.progress(key)["expected_bytes"] == 0
    assert not download_lifecycle.active_download_refs(registry, None, with_variant = True)


@pytest.mark.parametrize("change", ["job", "phase", "generation", "unbound"])
def test_a_stale_download_cancel_never_stops_training_or_a_replacement(rig, change):
    service, registry, key = rig
    generation = registry.current_generation(key)
    if change == "job":
        service._state["job_id"] = "b" * 32
    elif change == "phase":
        service._state.update(in_model_load = False, step = 1)
    elif change == "generation":
        generation += 1
    else:
        generation = None
    training_downloads.cancel(key, generation)
    assert service._proc.is_alive()
    assert service._proc.terminations == 0
    assert service._stop_queue.empty()


def test_training_step_stop_remains_cooperative_and_preserves_save_choice(rig):
    service, _registry, _key = rig
    service._state.update(in_model_load = False, step = 1)
    assert service.stop(save = True)
    assert service._stop_queue.get_nowait() is True
    assert service._proc.is_alive()
    assert service._discard_requested is False


def test_real_child_phase_wins_over_a_delayed_parent_loading_event(rig):
    from core.training.diffusion_training_service import _DownloadPhaseQueue

    service, _registry, _key = rig
    phase = (threading.Event(), threading.Lock())
    phase[0].set()
    service._download_phase = phase
    pending = queue.Queue()
    events = _DownloadPhaseQueue(pending, phase)
    events.put({"type": "model_load_completed"})
    assert not phase[0].is_set()
    assert service._state["in_model_load"] is True  # The parent has not consumed the event.
    assert service.stop(save = False, job_id = "a" * 32, download_only = True) is False
    assert service.stop(save = True) is True
    assert (
        service._proc.is_alive()
    ), "optimizer/preparation cannot be terminated using stale parent status"
    assert service._stop_queue.get_nowait() is True


def test_buffered_events_cannot_resurrect_a_killed_download(rig):
    service, registry, key = rig
    assert service.stop(save = False)
    service._apply_event({"type": "model_load_started"}, proc = service._proc)
    service._apply_event(
        {
            "type": "model_download_progress",
            "download_progress": {"bytes_done": 1, "bytes_total": 2},
        }
    )
    assert service.status()["status"] == "stopped"
    assert service.status()["download_progress"] is None
    assert registry.get_job(key).state == "cancelled"


def test_later_file_loads_get_a_new_generation_without_completing_the_repo(rig):
    service, registry, key = rig
    generation = registry.current_generation(key)
    service._apply_event({"type": "preparing", "stage": "cache_latents", "done": 60, "total": 60})
    service._apply_event({"type": "model_load_started"})
    assert registry.current_generation(key) > generation
    assert registry.get_job(key).state == "running"
    training_downloads.cancel(key, generation)
    assert service._proc.is_alive()


def test_registry_account_is_checked_before_reading_or_cancelling(rig, monkeypatch):
    _service, registry, key = rig
    from fastapi import HTTPException

    monkeypatch.setattr(download_lifecycle, "download_belongs_to_account", lambda *_args: False)
    for action in [
        lambda: training_downloads.progress(key),
        lambda: training_downloads.cancel(key, registry.current_generation(key)),
    ]:
        with pytest.raises(HTTPException) as error:
            action()
        assert error.value.status_code == 404
