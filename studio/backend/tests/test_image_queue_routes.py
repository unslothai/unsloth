# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /images/queue routes end to end against the fake diffusion backend: enqueue, list, reorder, cancel, clear."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

import core.inference.diffusion as diffusion_module
import routes.inference as inference_routes

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_routes import client  # noqa: E402,F401  (fixture)

QUEUE = "/api/inference/images/queue"


@pytest.fixture
def gated(client, monkeypatch):
    """A loaded fake model whose generate blocks on a per-prompt gate, inside a live TestClient portal."""
    monkeypatch.setattr(inference_routes, "_image_queue", None)
    backend = diffusion_module.get_diffusion_backend()
    gates: dict[str, threading.Event] = {}
    order: list[str] = []
    real_generate = backend.generate

    def generate(*, prompt = None, **kwargs):
        order.append(prompt)
        stop = threading.Event()
        bind_generation_cancel(stop)
        gate = gates.setdefault(prompt, threading.Event())
        deadline = time.monotonic() + 10
        while not gate.wait(0.02) and time.monotonic() < deadline:
            if stop.is_set():
                raise RuntimeError(cancelled_msg)
        if backend.cancel_next:
            backend.cancel_next = False
            raise RuntimeError(cancelled_msg)
        return real_generate(prompt = prompt, **kwargs)

    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG as cancelled_msg
    from core.inference.generation_ticket import bind_generation_cancel

    backend.cancel_next = False

    def cancel_generate(**kwargs):
        backend.cancel_next = True
        gates[order[-1]].set()
        return True

    monkeypatch.setattr(backend, "generate", generate)
    monkeypatch.setattr(backend, "cancel_generate", cancel_generate)
    with client:
        assert (
            client.post(
                "/api/inference/images/load",
                json = {
                    "model_path": "unsloth/Z-Image-Turbo-GGUF",
                    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
                    "base_repo": "unsloth/Z-Image-base",
                },
            ).status_code
            == 200
        )
        yield client, gates, order, backend


def _gate(gates, prompt):
    return gates.setdefault(prompt, threading.Event())


def _jobs(client):
    return client.get(QUEUE).json()["jobs"]


def _wait(
    client,
    predicate,
    timeout = 5.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        jobs = _jobs(client)
        if predicate(jobs):
            return jobs
        time.sleep(0.02)
    raise AssertionError(f"queue never settled: {_jobs(client)}")


def _by_prompt(jobs):
    return {job["prompt"]: job for job in jobs}


def test_queued_jobs_run_in_order_after_a_reorder_and_a_cancel(gated):
    client, gates, order, _ = gated
    ids = {}
    for prompt in ("one", "two", "three", "four"):
        response = client.post(
            QUEUE, json = {"prompt": prompt, "seed": 5, "width": 512, "height": 768}
        )
        assert response.status_code == 200
        ids[prompt] = response.json()["id"]
    _wait(client, lambda jobs: _by_prompt(jobs)["one"]["status"] == "running")

    listed = _jobs(client)
    assert [job["prompt"] for job in listed] == ["one", "two", "three", "four"]
    assert [job["position"] for job in listed] == [None, 0, 1, 2]
    assert listed[1]["width"] == 512 and listed[1]["height"] == 768 and listed[1]["seed"] == 5
    # Omitted steps follow the model, so the job does not claim the schema default.
    assert listed[1]["steps"] is None

    moved = client.post(f"{QUEUE}/{ids['four']}/move", json = {"index": 0}).json()["jobs"]
    assert [job["prompt"] for job in moved] == ["one", "four", "two", "three"]
    assert client.post(f"{QUEUE}/{ids['two']}/cancel").json()["status"] == "cancelled"

    for prompt in ("one", "four", "three"):
        _gate(gates, prompt).set()
    jobs = _wait(client, lambda jobs: all(job["status"] in ("done", "cancelled") for job in jobs))
    assert order == ["one", "four", "three"]
    states = _by_prompt(jobs)
    assert states["two"]["status"] == "cancelled"
    assert all(len(states[p]["image_ids"]) == 1 for p in ("one", "four", "three"))
    # Finished images are in the gallery like a direct generate's.
    gallery = {
        image["id"] for image in client.get("/api/inference/images/gallery").json()["images"]
    }
    assert {states[p]["image_ids"][0] for p in ("one", "four", "three")} <= gallery


def test_cancelling_the_running_job_stops_it_and_the_next_one_runs(gated):
    client, gates, order, _ = gated
    first = client.post(QUEUE, json = {"prompt": "first"}).json()["id"]
    client.post(QUEUE, json = {"prompt": "second"})
    _wait(client, lambda jobs: _by_prompt(jobs)["first"]["status"] == "running")
    assert client.post(f"{QUEUE}/{first}/cancel").status_code == 200
    _gate(gates, "second").set()
    jobs = _wait(client, lambda jobs: _by_prompt(jobs)["second"]["status"] == "done")
    assert _by_prompt(jobs)["first"]["status"] == "cancelled"
    assert _by_prompt(jobs)["first"]["image_ids"] == []


def test_clear_drops_queued_jobs_but_not_the_running_one(gated):
    client, gates, order, _ = gated
    client.post(QUEUE, json = {"prompt": "running"})
    client.post(QUEUE, json = {"prompt": "waiting"})
    _wait(client, lambda jobs: _by_prompt(jobs)["running"]["status"] == "running")
    jobs = client.delete(QUEUE).json()["jobs"]
    assert [job["prompt"] for job in jobs] == ["running"]
    _gate(gates, "running").set()
    _wait(client, lambda jobs: jobs and jobs[0]["status"] == "done")
    assert order == ["running"]


def test_a_model_swapped_in_after_queueing_fails_the_job(gated):
    client, gates, order, backend = gated
    real_status = backend.status
    client.post(QUEUE, json = {"prompt": "blocker"})
    client.post(QUEUE, json = {"prompt": "stale"})
    _wait(client, lambda jobs: _by_prompt(jobs)["blocker"]["status"] == "running")
    backend.status = lambda: {**real_status(), "repo_id": "unsloth/another-model"}
    _gate(gates, "blocker").set()
    jobs = _wait(client, lambda jobs: _by_prompt(jobs)["stale"]["status"] == "failed")
    assert "model changed" in _by_prompt(jobs)["stale"]["error"]
    assert order == ["blocker"]


def test_enqueue_refuses_with_nothing_loaded_or_a_connection(client, monkeypatch):
    monkeypatch.setattr(inference_routes, "_image_queue", None)
    refused = client.post(QUEUE, json = {"prompt": "x"})
    assert refused.status_code == 409 and "No diffusion model is loaded" in refused.json()["detail"]
    connection = client.post(QUEUE, json = {"prompt": "x", "provider_id": "p1", "model": "m"})
    assert connection.status_code == 400
    assert client.get(QUEUE).json() == {"jobs": []}


def test_unknown_job_is_a_404_and_a_running_job_cannot_move(gated):
    client, gates, order, _ = gated
    running = client.post(QUEUE, json = {"prompt": "a"}).json()["id"]
    _wait(client, lambda jobs: jobs[0]["status"] == "running")
    assert client.post(f"{QUEUE}/nope/cancel").status_code == 404
    assert client.post(f"{QUEUE}/{running}/move", json = {"index": 0}).status_code == 409
    assert client.post(f"{QUEUE}/{running}/move", json = {"index": -1}).status_code == 422
    _gate(gates, "a").set()
    _wait(client, lambda jobs: jobs[0]["status"] == "done")


def test_cancelling_a_queued_job_waiting_for_the_engine_leaves_the_render_ahead_alone(
    client, monkeypatch
):
    """A chat image holds the engine; Stop on the queued job behind it must not stop the chat image."""
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG
    from core.inference.generation_ticket import bind_generation_cancel
    from models.inference import DiffusionGenerateRequest

    monkeypatch.setattr(inference_routes, "_image_queue", None)
    backend = diffusion_module.get_diffusion_backend()
    real_generate = backend.generate
    slot = threading.Lock()
    active: list[threading.Event] = []
    chat_gate = threading.Event()
    ran: list[str] = []

    def generate(*, prompt = None, **kwargs):
        cancel = threading.Event()
        bind_generation_cancel(cancel)
        while True:
            if cancel.is_set():
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            if slot.acquire(timeout = 0.02):
                break
        try:
            # Admission rechecks under the slot, as the engine does.
            if cancel.is_set():
                raise RuntimeError(DIFFUSION_CANCELLED_MSG)
            active.append(cancel)
            ran.append(prompt)
            while prompt == "chat" and not chat_gate.is_set():
                if cancel.is_set():
                    raise RuntimeError(DIFFUSION_CANCELLED_MSG)
                time.sleep(0.02)
            return real_generate(prompt = prompt, **kwargs)
        finally:
            active.pop()
            slot.release()

    def cancel_generate(**kwargs):
        # Like the engine: Stop reaches the generation holding the slot, not a waiter.
        if not active:
            return False
        active[-1].set()
        return True

    monkeypatch.setattr(backend, "generate", generate)
    monkeypatch.setattr(backend, "cancel_generate", cancel_generate)
    with client:
        assert (
            client.post(
                "/api/inference/images/load",
                json = {
                    "model_path": "unsloth/Z-Image-Turbo-GGUF",
                    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
                    "base_repo": "unsloth/Z-Image-base",
                },
            ).status_code
            == 200
        )
        chat = client.portal.start_task_soon(
            inference_routes._generate_diffusion_image_core,
            DiffusionGenerateRequest(prompt = "chat"),
        )
        deadline = time.monotonic() + 5
        while ran != ["chat"] and time.monotonic() < deadline:
            time.sleep(0.02)
        queued = client.post(QUEUE, json = {"prompt": "queued"}).json()["id"]
        _wait(client, lambda jobs: jobs[0]["status"] == "running")
        assert client.post(f"{QUEUE}/{queued}/cancel").status_code == 200
        chat_gate.set()
        assert len(chat.result(timeout = 10).images) == 1
        jobs = _wait(client, lambda jobs: jobs[0]["status"] in ("done", "failed", "cancelled"))
        assert ran == ["chat"]
        assert jobs[0]["status"] == "cancelled"


def test_a_queued_job_is_pinned_to_its_model_and_never_retries_on_a_replacement(
    client, monkeypatch
):
    """A swap between the queue's model check and the render must fail the job, not run it on the new model."""
    from core.inference.diffusion_families import DiffusionModelReplacedError, load_identity
    from fastapi import HTTPException
    from models.inference import DiffusionGenerateRequest

    backend = diffusion_module.get_diffusion_backend()
    real_generate = backend.generate
    seen: list = []
    queued_for = load_identity("unsloth/queued-model", None, None)

    def generate(*, expected_load = None, **kwargs):
        seen.append(expected_load)
        loaded = load_identity("unsloth/swapped-in", None, None)
        if expected_load is not None and expected_load != loaded:
            raise DiffusionModelReplacedError(expected_load, loaded)
        return real_generate(**kwargs)

    monkeypatch.setattr(backend, "generate", generate)
    with client:
        assert (
            client.post(
                "/api/inference/images/load",
                json = {
                    "model_path": "unsloth/Z-Image-Turbo-GGUF",
                    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
                    "base_repo": "unsloth/Z-Image-base",
                },
            ).status_code
            == 200
        )
        with pytest.raises(HTTPException) as refused:
            client.portal.call(
                inference_routes._generate_diffusion_image_core,
                DiffusionGenerateRequest(prompt = "p", steps = 4, guidance = 1.0),
                queued_for,
            )
        assert refused.value.status_code == 409
        assert isinstance(refused.value.__cause__, DiffusionModelReplacedError)
        assert seen == [queued_for]


def test_a_queued_job_waits_out_the_gap_between_two_direct_runs(client, monkeypatch):
    """Generate x2 sends two requests back to back; a job queued meanwhile runs after both."""
    monkeypatch.setattr(inference_routes, "_image_queue", None)
    backend = diffusion_module.get_diffusion_backend()
    real_generate = backend.generate
    order: list[str] = []

    def generate(*, prompt = None, **kwargs):
        order.append(prompt)
        if prompt == "direct-1":
            time.sleep(0.3)
        return real_generate(prompt = prompt, **kwargs)

    monkeypatch.setattr(backend, "generate", generate)
    with client:
        assert (
            client.post(
                "/api/inference/images/load",
                json = {
                    "model_path": "unsloth/Z-Image-Turbo-GGUF",
                    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
                    "base_repo": "unsloth/Z-Image-base",
                },
            ).status_code
            == 200
        )
        first = client.portal.start_task_soon(
            lambda: inference_routes.generate_diffusion_image(
                inference_routes.DiffusionGenerateRequest(prompt = "direct-1"), None
            )
        )
        deadline = time.monotonic() + 5
        while "direct-1" not in order and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.post(QUEUE, json = {"prompt": "queued"}).status_code == 200
        first.result(timeout = 10)
        # The page's next run arrives a few hundred ms later.
        time.sleep(0.3)
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "direct-2"}).status_code
            == 200
        )
        _wait(client, lambda jobs: jobs[0]["status"] == "done")
        assert order == ["direct-1", "direct-2", "queued"]


def test_another_quant_of_the_same_repo_fails_the_waiting_job(gated):
    client, gates, order, backend = gated
    real_status = backend.status
    client.post(QUEUE, json = {"prompt": "blocker"})
    client.post(QUEUE, json = {"prompt": "stale"})
    _wait(client, lambda jobs: _by_prompt(jobs)["blocker"]["status"] == "running")
    backend.status = lambda: {**real_status(), "gguf_filename": "z-image-turbo-Q8_0.gguf"}
    _gate(gates, "blocker").set()
    jobs = _wait(client, lambda jobs: _by_prompt(jobs)["stale"]["status"] == "failed")
    assert "model changed" in _by_prompt(jobs)["stale"]["error"]
    assert order == ["blocker"]


def test_a_multi_run_generate_holds_the_queue_across_a_long_gap(client, monkeypatch):
    """Run 1 says another follows: a job queued meanwhile waits past the settle window for run 2."""
    monkeypatch.setattr(inference_routes, "_image_queue", None)
    monkeypatch.setattr(inference_routes, "_DIRECT_RUN_SETTLE_S", 0.2)
    backend = diffusion_module.get_diffusion_backend()
    real_generate = backend.generate
    order: list[str] = []

    def generate(*, prompt = None, **kwargs):
        order.append(prompt)
        if prompt == "direct-1":
            time.sleep(0.3)
        return real_generate(prompt = prompt, **kwargs)

    monkeypatch.setattr(backend, "generate", generate)
    with client:
        assert (
            client.post(
                "/api/inference/images/load",
                json = {
                    "model_path": "unsloth/Z-Image-Turbo-GGUF",
                    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
                    "base_repo": "unsloth/Z-Image-base",
                },
            ).status_code
            == 200
        )
        first = threading.Thread(
            target = client.post,
            args = ("/api/inference/images/generate",),
            kwargs = {"json": {"prompt": "direct-1"}, "headers": {"X-Unsloth-More-Runs": "1"}},
        )
        first.start()
        deadline = time.monotonic() + 5
        while "direct-1" not in order and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.post(QUEUE, json = {"prompt": "queued"}).status_code == 200
        first.join(10)
        # Far past the settle window, as behind a slow proxy.
        time.sleep(1.0)
        assert order == ["direct-1"]
        assert (
            client.post("/api/inference/images/generate", json = {"prompt": "direct-2"}).status_code
            == 200
        )
        _wait(client, lambda jobs: jobs[0]["status"] == "done")
        assert order == ["direct-1", "direct-2", "queued"]


def test_a_failed_run_marked_more_runs_does_not_hold_the_queue(client, monkeypatch):
    monkeypatch.setattr(inference_routes, "_image_queue", None)
    monkeypatch.setattr(inference_routes, "_DIRECT_RUN_SETTLE_S", 0.2)
    backend = diffusion_module.get_diffusion_backend()
    real_generate = backend.generate
    order: list[str] = []
    started = threading.Event()

    def generate(*, prompt = None, **kwargs):
        order.append(prompt)
        if prompt == "direct-1":
            started.set()
            time.sleep(0.3)
            raise RuntimeError("boom")
        return real_generate(prompt = prompt, **kwargs)

    monkeypatch.setattr(backend, "generate", generate)
    with client:
        assert (
            client.post(
                "/api/inference/images/load",
                json = {
                    "model_path": "unsloth/Z-Image-Turbo-GGUF",
                    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
                    "base_repo": "unsloth/Z-Image-base",
                },
            ).status_code
            == 200
        )
        first = threading.Thread(
            target = client.post,
            args = ("/api/inference/images/generate",),
            kwargs = {"json": {"prompt": "direct-1"}, "headers": {"X-Unsloth-More-Runs": "1"}},
        )
        first.start()
        started.wait(5)
        assert client.post(QUEUE, json = {"prompt": "queued"}).status_code == 200
        first.join(10)
        _wait(client, lambda jobs: jobs[0]["status"] == "done", timeout = 5)
        assert order == ["direct-1", "queued"]


def test_a_checkpoint_pin_refuses_another_quant_at_engine_admission():
    from core.inference.diffusion_families import (
        DiffusionModelReplacedError,
        load_identity,
        pin_checkpoint,
        require_pinned_checkpoint,
    )

    loaded = load_identity("unsloth/Z-Image-Turbo-GGUF", "unsloth/Z-Image-base", "z-image")
    require_pinned_checkpoint(loaded, "z-image-turbo-Q8_0.gguf")  # no pin: any checkpoint
    with pin_checkpoint("z-image-turbo-Q4_K_S.gguf"):
        require_pinned_checkpoint(loaded, "z-image-turbo-Q4_K_S.gguf")
        with pytest.raises(DiffusionModelReplacedError):
            require_pinned_checkpoint(loaded, "z-image-turbo-Q8_0.gguf")
    # Every engine checks the pin beside its expected_load admission.
    for module in ("diffusion", "sd_cpp_backend", "mlx_diffusion_backend"):
        source = (
            Path(__file__).resolve().parents[1] / "core" / "inference" / f"{module}.py"
        ).read_text()
        assert "require_pinned_checkpoint(loaded_id, " in source, module


def test_stop_during_a_run_that_still_returns_does_not_hold_the_queue(client, monkeypatch):
    """A Detailer keeps its render on Stop, so the run returns; the page sends no next run."""
    monkeypatch.setattr(inference_routes, "_image_queue", None)
    monkeypatch.setattr(inference_routes, "_DIRECT_RUN_SETTLE_S", 0.2)
    backend = diffusion_module.get_diffusion_backend()
    real_generate = backend.generate
    order: list[str] = []
    started = threading.Event()
    release = threading.Event()

    def generate(*, prompt = None, **kwargs):
        order.append(prompt)
        if prompt == "direct-1":
            started.set()
            release.wait(5)
        return real_generate(prompt = prompt, **kwargs)

    monkeypatch.setattr(backend, "generate", generate)
    # The render keeps going and returns, as a Detailer pass does after Stop.
    monkeypatch.setattr(backend, "cancel_generate", lambda **kwargs: True)
    with client:
        assert (
            client.post(
                "/api/inference/images/load",
                json = {
                    "model_path": "unsloth/Z-Image-Turbo-GGUF",
                    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
                    "base_repo": "unsloth/Z-Image-base",
                },
            ).status_code
            == 200
        )
        first = threading.Thread(
            target = client.post,
            args = ("/api/inference/images/generate",),
            kwargs = {"json": {"prompt": "direct-1"}, "headers": {"X-Unsloth-More-Runs": "1"}},
        )
        first.start()
        started.wait(5)
        assert client.post(QUEUE, json = {"prompt": "queued"}).status_code == 200
        client.post("/api/inference/images/generate/cancel")
        release.set()
        first.join(10)
        _wait(client, lambda jobs: jobs[0]["status"] == "done", timeout = 5)
        assert order == ["direct-1", "queued"]


def test_a_refused_stop_leaves_another_accounts_sequence_hold(client, monkeypatch):
    monkeypatch.setattr(inference_routes, "_direct_sequence_until", time.monotonic() + 60)
    monkeypatch.setattr(inference_routes, "_direct_stops", 7)
    monkeypatch.setattr(
        inference_routes.account_access, "generation_is_foreign", lambda modality: True
    )
    with client:
        assert client.post("/api/inference/images/generate/cancel").json() == {"cancelled": False}
    assert inference_routes._direct_stops == 7
    assert inference_routes._direct_sequence_until > time.monotonic()
