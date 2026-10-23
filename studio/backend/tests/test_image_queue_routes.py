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
        gates.setdefault(prompt, threading.Event()).wait(10)
        if backend.cancel_next:
            backend.cancel_next = False
            raise RuntimeError(cancelled_msg)
        return real_generate(prompt = prompt, **kwargs)

    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG as cancelled_msg

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
