# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from core.training.training import TrainingBackend
from models.training import TrainingStartRequest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent


def _load_training_route(name: str):
    spec = importlib.util.spec_from_file_location(
        name,
        _BACKEND_ROOT / "routes" / "training.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_request_reservation_is_idempotent_and_serialized():
    backend = TrainingBackend()

    outcome, pending = backend.reserve_start_request("request-1", "job-1")
    assert outcome == "reserved"
    assert pending.state == "pending"

    outcome, duplicate = backend.reserve_start_request("request-1", "job-other")
    assert outcome == "existing"
    assert duplicate.job_id == "job-1"

    outcome, conflict = backend.reserve_start_request("request-2", "job-2")
    assert outcome == "conflict"
    assert conflict.start_request_id == "request-2"
    assert conflict.state == "rejected"

    rejected = backend.resolve_start_request(
        "request-1",
        state = "rejected",
        message = "Model unavailable",
        error = "Model unavailable",
        error_code = "hf_model_metadata_unavailable",
    )
    assert rejected is not None
    assert rejected.state == "rejected"
    assert rejected.error_code == "hf_model_metadata_unavailable"
    assert backend.status_start_request() == rejected
    assert backend.acknowledge_start_request("request-1") is True
    assert backend.status_start_request() is None
    assert backend.get_start_request("request-1") == rejected

    outcome, next_request = backend.reserve_start_request("request-3", "job-3")
    assert outcome == "reserved"
    assert next_request.job_id == "job-3"


def test_accepted_start_request_remains_queryable():
    backend = TrainingBackend()
    backend.reserve_start_request("request-1", "job-1")
    backend.resolve_start_request(
        "request-1",
        state = "accepted",
        message = "Training queued",
    )

    record = backend.get_start_request("request-1")
    assert record is not None
    assert record.state == "accepted"
    assert record.job_id == "job-1"


def test_cancelled_route_during_spawn_keeps_the_worker_result_authoritative():
    route = _load_training_route("training_route_cancelled_start_test")
    backend = TrainingBackend()
    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    spawn_finished = threading.Event()

    def start_training(**kwargs):
        spawn_entered.set()
        if not release_spawn.wait(timeout = 5):
            raise TimeoutError("test did not release the training spawn")
        backend.current_job_id = kwargs["job_id"]
        backend.current_start_request_id = kwargs["start_request_id"]
        backend._progress.is_training = True
        backend._progress.status_message = "Initializing training..."
        spawn_finished.set()
        return True

    backend.start_training = start_training
    backend.is_training_active = lambda: backend.current_job_id is not None
    request = TrainingStartRequest(
        model_name = "unsloth/test",
        start_request_id = "cancelled-start-request",
        training_type = "LoRA/QLoRA",
        hf_dataset = "org/dataset",
        format_type = "chatml",
        dataset_streaming = True,
        max_steps = 10,
    )

    async def controlled_to_thread(function, *args, **kwargs):
        if function.__name__ != "_run_backend_start":
            return function(*args, **kwargs)
        loop = asyncio.get_running_loop()
        result = loop.create_future()

        def run():
            try:
                value = function(*args, **kwargs)
            except BaseException as exc:
                loop.call_soon_threadsafe(result.set_exception, exc)
            else:
                loop.call_soon_threadsafe(result.set_result, value)

        threading.Thread(target = run, daemon = True).start()
        return await result

    async def wait_for_event(event: threading.Event) -> bool:
        for _ in range(500):
            if event.is_set():
                return True
            await asyncio.sleep(0.01)
        return False

    async def run_cancellation_race():
        handler = asyncio.create_task(
            route.start_training(request, current_subject = "test-user"),
        )
        assert await wait_for_event(spawn_entered)
        handler.cancel()
        with pytest.raises(asyncio.CancelledError):
            await handler

        pending = backend.get_start_request("cancelled-start-request")
        assert pending is not None
        assert pending.state == "pending"

        release_spawn.set()
        assert await wait_for_event(spawn_finished)
        for _ in range(100):
            record = backend.get_start_request("cancelled-start-request")
            if record is not None and record.state != "pending":
                break
            await asyncio.sleep(0.01)
        current_task = asyncio.current_task()
        background_tasks = [
            task for task in asyncio.all_tasks() if task is not current_task
        ]
        if background_tasks:
            await asyncio.wait_for(
                asyncio.gather(*background_tasks, return_exceptions = True),
                timeout = 5,
            )
        return record

    with (
        patch.object(route, "get_training_backend", return_value = backend),
        patch.object(route, "_remote_untrainable_model_format", return_value = None),
        patch.object(route, "load_model_defaults", return_value = {}),
        patch.object(route.asyncio, "to_thread", new = controlled_to_thread),
    ):
        try:
            record = asyncio.run(run_cancellation_race())
        finally:
            release_spawn.set()

    assert record is not None
    assert record.state == "accepted"
    assert backend.current_job_id == record.job_id
    assert backend.is_training_active() is True
