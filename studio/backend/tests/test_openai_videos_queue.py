# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FIFO queueing for POST /v1/videos."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import core.inference.video_gallery as gallery_module
import routes.video as video_routes
from utils.account_context import AccountContext

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_openai_videos_route import (  # noqa: E402,F401 -- fixtures are found by name
    _create,
    _healthy_diffusers,
    _wait_terminal,
    backend,
    client,
)

_OTHER = AccountContext("acct-other", "other", "user")


def _drain() -> None:
    asyncio.run(video_routes._drain_video_queue_once())


def _render_one(
    client,
    backend,
    prompt = "first",
) -> dict:
    """Start a video and hold it rendering until the gate opens."""
    backend.gate.clear()
    job = _create(client, {"prompt": prompt}).json()
    assert job["status"] in ("queued", "in_progress")
    return job


def test_a_create_while_a_video_renders_is_queued_then_starts_with_its_id_seed_and_time(
    client, backend
):
    first = _render_one(client, backend)
    resp = client.post("/v1/videos", json = {"prompt": "second", "seed": 7, "size": "768x512"})
    assert resp.status_code == 200, resp.json()
    queued = resp.json()
    assert queued["status"] == "queued" and queued["progress"] == 0
    assert queued["id"].startswith("video_") and queued["seed"] == 7
    assert queued["size"] == "768x512" and queued["prompt"] == "second"
    listed = {v["id"]: v["status"] for v in client.get("/v1/videos").json()["data"]}
    assert listed[queued["id"]] == "queued" and first["id"] in listed and len(listed) == 2

    _drain()  # the first video still renders: nothing starts
    assert client.get(f"/v1/videos/{queued['id']}").json()["status"] == "queued"
    assert len(video_routes._VIDEO_QUEUE) == 1

    backend.gate.set()
    _wait_terminal(client, first["id"])
    _drain()
    assert video_routes._VIDEO_QUEUE == []
    done = _wait_terminal(client, queued["id"])
    assert done["status"] == "completed"
    assert done["created_at"] == queued["created_at"] and done["seed"] == 7
    assert backend.last_generate_kwargs["prompt"] == "second"
    assert backend.last_generate_kwargs["seed"] == 7


def test_queued_videos_start_first_in_first_out_and_a_new_create_waits_behind_them(client, backend):
    first = _render_one(client, backend)
    a = client.post("/v1/videos", json = {"prompt": "a"}).json()
    b = client.post("/v1/videos", json = {"prompt": "b"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])

    # New requests must join the queue even when the engine is idle.
    c = client.post("/v1/videos", json = {"prompt": "c"}).json()
    assert c["status"] == "queued"
    assert [entry.id for entry in video_routes._VIDEO_QUEUE] == [a["id"], b["id"], c["id"]]

    for expected in (a, b, c):
        _drain()
        assert _wait_terminal(client, expected["id"])["status"] == "completed"
        assert backend.last_generate_kwargs["prompt"] == expected["prompt"]
    assert video_routes._VIDEO_QUEUE == []


def test_deleting_a_queued_video_removes_it_before_it_ever_starts(client, backend):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "never"}).json()
    deleted = client.delete(f"/v1/videos/{queued['id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": queued["id"], "object": "video.deleted", "deleted": True}
    assert video_routes._VIDEO_QUEUE == []
    assert client.get(f"/v1/videos/{queued['id']}").status_code == 404

    backend.gate.set()
    _wait_terminal(client, first["id"])
    _drain()
    assert backend.last_generate_kwargs["prompt"] == "first"


def test_a_video_deleted_while_its_start_is_in_flight_is_deleted_once_started(
    client, backend, monkeypatch
):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "late"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    real_start = video_routes._start_openai_video

    async def _start_then_delete(*args, **kwargs):
        job = await real_start(*args, **kwargs)
        # Delete before startup returns.
        assert client.delete(f"/v1/videos/{queued['id']}").status_code == 200
        return job

    monkeypatch.setattr(video_routes, "_start_openai_video", _start_then_delete)
    _drain()
    assert video_routes._VIDEO_QUEUE == []
    assert client.get(f"/v1/videos/{queued['id']}").status_code == 404
    assert gallery_module.owned_video_path(queued["id"]) is None


def test_a_queued_video_whose_start_is_refused_fails_with_the_reason(client, backend):
    first = _render_one(client, backend)
    doomed = client.post("/v1/videos", json = {"prompt": "doomed"}).json()
    after = client.post("/v1/videos", json = {"prompt": "after"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])

    backend.loaded = False
    _drain()
    failed = client.get(f"/v1/videos/{doomed['id']}").json()
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == video_routes._NO_VIDEO_MODEL_MSG
    assert failed["created_at"] == doomed["created_at"]
    assert [entry.id for entry in video_routes._VIDEO_QUEUE] == [after["id"]]

    backend.loaded = True
    _drain()
    assert _wait_terminal(client, after["id"])["status"] == "completed"


def test_another_accounts_queued_video_is_neither_listed_nor_deletable(client, backend):
    _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "theirs"}).json()
    video_routes._VIDEO_QUEUE[0].account = _OTHER
    assert client.get(f"/v1/videos/{queued['id']}").status_code == 404
    assert queued["id"] not in [v["id"] for v in client.get("/v1/videos").json()["data"]]
    assert client.delete(f"/v1/videos/{queued['id']}").status_code == 404
    assert [entry.id for entry in video_routes._VIDEO_QUEUE] == [queued["id"]]
    backend.gate.set()


def test_the_queue_refuses_past_its_per_account_cap(client, backend, monkeypatch):
    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_MAX_PER_ACCOUNT", 1)
    _render_one(client, backend)
    assert client.post("/v1/videos", json = {"prompt": "one"}).json()["status"] == "queued"
    full = client.post("/v1/videos", json = {"prompt": "two"})
    assert full.status_code == 429
    assert full.json()["error"]["code"] == "rate_limit_exceeded"
    assert len(video_routes._VIDEO_QUEUE) == 1
    backend.gate.set()


def test_invalid_input_is_refused_at_once_rather_than_queued(client, backend):
    _render_one(client, backend)
    bad = client.post("/v1/videos", json = {"prompt": "x", "size": "abc"})
    assert bad.status_code == 400
    assert bad.json()["error"]["param"] == "size"
    assert video_routes._VIDEO_QUEUE == []
    backend.gate.set()


def test_the_chat_tool_path_still_refuses_while_a_video_renders(client, backend):
    from fastapi import HTTPException

    from core.inference.video_families import VIDEO_GENERATION_BUSY_MSG
    from models.inference import VideoJobCreateRequest

    _render_one(client, backend)

    async def _create_now():
        return await video_routes._create_openai_video(
            VideoJobCreateRequest(prompt = "tool"), None, "test-user", None
        )

    try:
        asyncio.run(_create_now())
    except HTTPException as exc:
        assert exc.status_code == 409
        assert exc.detail["error"]["message"] == VIDEO_GENERATION_BUSY_MSG
    else:
        raise AssertionError("the chat tool's create must not queue")
    finally:
        backend.gate.set()
    assert video_routes._VIDEO_QUEUE == []


def test_a_delete_during_the_drains_busy_check_keeps_the_video_from_starting(
    client, backend, monkeypatch
):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "gone"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    real_busy = video_routes._video_engine_busy

    def _busy_check_with_a_delete():
        # Delete during the busy check.
        assert client.delete(f"/v1/videos/{queued['id']}").status_code == 200
        return real_busy()

    monkeypatch.setattr(video_routes, "_video_engine_busy", _busy_check_with_a_delete)
    _drain()
    assert backend.last_generate_kwargs["prompt"] == "first"
    assert video_routes._VIDEO_QUEUE == []
    assert client.get(f"/v1/videos/{queued['id']}").status_code == 404


def test_a_delete_while_a_refused_start_is_recorded_leaves_no_failed_job(
    client, backend, monkeypatch
):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "refused"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    backend.loaded = False
    real_fail = video_routes._fail_queued_video
    deleted = []

    def _fail_with_a_delete(entry, exc):
        # Delete during failure persistence.
        deleted.append(client.delete(f"/v1/videos/{queued['id']}").status_code)
        real_fail(entry, exc)

    monkeypatch.setattr(video_routes, "_fail_queued_video", _fail_with_a_delete)
    _drain()
    assert deleted == [200]
    assert video_routes._VIDEO_QUEUE == []
    assert client.get(f"/v1/videos/{queued['id']}").status_code == 404
    assert queued["id"] not in [v["id"] for v in client.get("/v1/videos").json()["data"]]


def test_creates_arriving_together_cannot_pass_the_cap_together(client, backend, monkeypatch):
    import time

    import httpx

    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_MAX_PER_ACCOUNT", 1)
    real_model_name = video_routes._queued_model_name

    def _slow_model_name(body):
        time.sleep(0.2)
        return real_model_name(body)

    monkeypatch.setattr(video_routes, "_queued_model_name", _slow_model_name)
    _render_one(client, backend)

    async def run():
        transport = httpx.ASGITransport(app = client.app)
        async with httpx.AsyncClient(transport = transport, base_url = "http://studio") as studio:
            return await asyncio.gather(
                *(studio.post("/v1/videos", json = {"prompt": p}) for p in ("one", "two"))
            )

    try:
        statuses = sorted(response.status_code for response in asyncio.run(run()))
    finally:
        backend.gate.set()
    assert statuses == [200, 429]
    assert len(video_routes._VIDEO_QUEUE) == 1
