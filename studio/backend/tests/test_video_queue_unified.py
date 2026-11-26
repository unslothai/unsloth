# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One video queue for the Video page, /v1/videos and the chat tool: order, fairness, places and
ETAs, isolation, caps, cancel, a model-changed head, and the wake-up when a clip ends."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.video as video_routes
from auth.authentication import get_current_subject
from core.inference import video_queue
from routes.video import openai_router, router as video_router
from utils.account_context import AccountContext, bind_account, reset_account
from utils.api_errors import install_api_error_handlers

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_openai_videos_route import (  # noqa: E402,F401 -- fixtures are found by name
    _healthy_diffusers,
    _wait_terminal,
    backend,
    client,
)

_OTHER = AccountContext("b" * 32, "bob", "user")
PAGE = "/api/inference/video/generate"
QUEUE = "/api/inference/video/queue"
CANCEL = "/api/inference/video/generate/cancel"


def _drain() -> bool:
    return asyncio.run(video_routes._drain_video_queue_once())


def _wait_idle(backend, timeout = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not backend.generate_progress().get("active"):
            return
        time.sleep(0.01)
    raise AssertionError(backend.generate_progress())


@pytest.fixture
def prompts(backend, monkeypatch):
    """Every prompt the engine rendered, in order, with its start and end times."""
    seen: list = []
    real = backend.generate

    def _generate(**kwargs):
        row = {"prompt": kwargs.get("prompt"), "start": time.monotonic()}
        seen.append(row)
        try:
            return real(**kwargs)
        finally:
            row["end"] = time.monotonic()

    monkeypatch.setattr(backend, "generate", _generate)
    return seen


@pytest.fixture
def other_account_allowed(monkeypatch):
    from hub.services.models import account_access
    monkeypatch.setattr(video_routes, "_queued_account_active", lambda account: True)
    monkeypatch.setattr(account_access, "require_media_generation_access", lambda *a, **k: None)


def _client_as(account) -> TestClient:
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(openai_router, prefix = "/v1")
    app.include_router(video_router, prefix = "/api/inference")

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    return TestClient(app)


def _busy(
    client,
    backend,
    prompt = "running",
) -> dict:
    backend.gate.clear()
    job = client.post("/v1/videos", json = {"prompt": prompt}).json()
    assert job["status"] in ("queued", "in_progress") and not video_routes._VIDEO_QUEUE
    return job


def _queue(client, prompt, **extra) -> dict:
    job = client.post("/v1/videos", json = {"prompt": prompt, **extra}).json()
    assert job["status"] == "queued", job
    return job


def _finish_running(client, backend, job) -> None:
    backend.gate.set()
    _wait_terminal(client, job["id"])
    _wait_idle(backend)


# ---- order ------------------------------------------------------------------------------------


def test_fair_order_is_fifo_within_an_account_and_round_robin_across():
    order = video_queue.FairOrder()
    entries: list = []

    def add(account, name):
        order.joined(account, entries, lambda e: e[0])
        entries.append((account, name))

    for name in ("a1", "a2", "a3"):
        add("A", name)
    add("B", "b1")
    add("B", "b2")
    add("C", "c1")
    assert [n for _, n in order.order(entries, lambda e: e[0])] == [
        "a1",
        "b1",
        "c1",
        "a2",
        "b2",
        "a3",
    ]
    # a1 starts: A takes its next turn behind B and C.
    entries.remove(("A", "a1"))
    order.started("A")
    assert [n for _, n in order.order(entries, lambda e: e[0])] == ["b1", "c1", "a2", "b2", "a3"]


def test_one_accounts_flood_never_pushes_another_accounts_waiting_video_back():
    order = video_queue.FairOrder()
    entries: list = []

    def add(account, name):
        order.joined(account, entries, lambda e: e[0])
        entries.append((account, name))

    add("A", "a1")
    add("B", "b1")
    place = [n for _, n in order.order(entries, lambda e: e[0])].index("b1")
    for i in range(2, 20):
        add("A", f"a{i}")
        assert [n for _, n in order.order(entries, lambda e: e[0])].index("b1") == place


def test_queued_videos_start_round_robin_across_accounts(
    client, backend, prompts, other_account_allowed
):
    running = _busy(client, backend)
    for name in ("a1", "a2", "a3", "b1"):
        _queue(client, name)
    video_routes._VIDEO_QUEUE[3].account = _OTHER
    _finish_running(client, backend, running)
    for _ in range(4):
        assert _drain() is True
        _wait_idle(backend)
    assert [row["prompt"] for row in prompts] == ["running", "a1", "b1", "a2", "a3"]
    assert video_routes._VIDEO_QUEUE == []


# ---- places and ETAs --------------------------------------------------------------------------


def test_places_and_etas_count_down_and_never_increase(client, backend, monkeypatch):
    running = _busy(client, backend)
    jobs = [_queue(client, f"q{i}") for i in range(3)]
    assert [job["queue_position"] for job in jobs] == [1, 2, 3]
    assert [job["eta_seconds"] for job in jobs] == [None, None, None]

    # One clip of this shape took 2 s per step: every place now has an ETA.
    [key] = {entry.clip_key for entry in video_routes._VIDEO_QUEUE}
    steps = video_routes._VIDEO_QUEUE[0].steps
    assert key[0] == "ltx-2" and steps
    video_queue.clock._per_step[key] = 2.0
    clip = 2.0 * steps

    def _places() -> dict:
        return {
            j["id"]: (j["queue_position"], j["eta_seconds"])
            for j in (client.get(f"/v1/videos/{job['id']}").json() for job in jobs)
            if j["queue_position"] is not None
        }

    first = _places()
    etas = [eta for _, eta in first.values()]
    assert all(eta is not None for eta in etas)
    assert etas[1] - etas[0] == pytest.approx(clip, abs = 0.2)
    assert etas[2] - etas[1] == pytest.approx(clip, abs = 0.2)
    history = [first]
    time.sleep(0.15)
    history.append(_places())

    # Clips now take twice the average: the shown ETAs still only count down.
    _finish_running(client, backend, running)
    for _ in range(3):
        video_queue.clock._per_step[key] = 4.0
        backend.gate.clear()
        assert _drain() is True
        history.append(_places())
        time.sleep(0.05)
        history.append(_places())
        backend.gate.set()
        _wait_idle(backend)
    for before, after in zip(history, history[1:]):
        for video_id, (position, eta) in after.items():
            assert position <= before[video_id][0]
            assert eta <= before[video_id][1]


def test_the_eta_estimate_uses_the_shape_average_and_scales_another_shape():
    clock = video_queue.ClipClock()
    clock.started(("ltx-2", 768, 512, 121), 10)
    clock._running.started_at -= 20.0
    clock.finished(True, 10)
    assert clock.estimate(("ltx-2", 768, 512, 121), 10) == pytest.approx(20.0, rel = 0.05)
    assert clock.estimate(("ltx-2", 768, 512, 61), 10) == pytest.approx(20.0 * 61 / 121, rel = 0.05)
    assert clock.estimate(("wan", 832, 480, 81), 10) is None
    view = video_queue.queue_view(["x", "y"], remaining = 5.0, estimate = lambda entry: 20.0)
    assert list(view.values()) == [(1, 5.0), (2, 25.0)]


# ---- the Video page shares the queue ----------------------------------------------------------


def test_the_page_route_never_jumps_a_waiting_video(client, backend, prompts):
    running = _busy(client, backend)
    waiting = _queue(client, "waiting")
    # The running clip ends; before the queue starts the next one, the page asks for a clip.
    _finish_running(client, backend, running)
    page = client.post(PAGE, json = {"prompt": "page", "seed": 5})
    assert page.status_code == 200, page.json()
    assert page.json()["status"] == "queued" and page.json()["queue_position"] == 2
    assert [row["prompt"] for row in prompts] == ["running"]
    assert [e.id for e in video_routes._ordered_queue()] == [
        waiting["id"],
        page.json()["video_id"],
    ]
    _drain()
    _wait_idle(backend)
    _drain()
    _wait_idle(backend)
    assert [row["prompt"] for row in prompts] == ["running", "waiting", "page"]


def test_a_busy_engine_queues_the_page_clip_instead_of_a_409(client, backend):
    _busy(client, backend)
    first = client.post(PAGE, json = {"prompt": "one"})
    second = client.post(PAGE, json = {"prompt": "two"})
    assert [r.status_code for r in (first, second)] == [200, 200]
    assert [r.json()["queue_position"] for r in (first, second)] == [1, 2]
    jobs = client.get(QUEUE).json()["jobs"]
    assert [(j["prompt"], j["status"], j["source"], j["queue_position"]) for j in jobs] == [
        ("one", "queued", "page", 1),
        ("two", "queued", "page", 2),
    ]
    backend.gate.set()


def test_page_input_the_resident_model_refuses_is_refused_at_once_not_queued(client, backend):
    _busy(client, backend)
    bad = client.post(PAGE, json = {"prompt": "bad", "num_frames": 2})
    assert bad.status_code == 422, bad.json()
    assert video_routes._VIDEO_QUEUE == []
    backend.gate.set()


def test_a_queued_page_clip_that_cannot_start_reports_why_on_the_queue(client, backend):
    running = _busy(client, backend)
    page = client.post(PAGE, json = {"prompt": "doomed"}).json()
    _finish_running(client, backend, running)
    backend.loaded = False
    _drain()
    [job] = client.get(QUEUE).json()["jobs"]
    assert (job["video_id"], job["status"]) == (page["video_id"], "failed")
    assert "load" in job["error"].lower()


# ---- isolation and caps -----------------------------------------------------------------------


def test_another_account_never_sees_or_cancels_a_queued_page_clip(client, backend):
    _busy(client, backend)
    mine = client.post(PAGE, json = {"prompt": "private"}).json()
    with _client_as(_OTHER) as other:
        assert other.get(QUEUE).json()["jobs"] == []
        assert other.get(f"/v1/videos/{mine['video_id']}").status_code == 404
        assert other.delete(f"/v1/videos/{mine['video_id']}").status_code == 404
        refused = other.post(CANCEL, json = {"video_id": mine["video_id"]})
        assert refused.status_code == 409
    assert [e.id for e in video_routes._VIDEO_QUEUE] == [mine["video_id"]]
    assert [j["video_id"] for j in client.get(QUEUE).json()["jobs"]] == [mine["video_id"]]
    # A Video page clip is not a /v1/videos job, for its owner either.
    assert client.get(f"/v1/videos/{mine['video_id']}").status_code == 404
    backend.gate.set()


def test_the_page_route_keeps_the_per_account_and_total_caps(client, backend, monkeypatch):
    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_MAX_PER_ACCOUNT", 1)
    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_MAX", 2)
    _busy(client, backend)
    assert client.post(PAGE, json = {"prompt": "one"}).json()["status"] == "queued"
    full = client.post(PAGE, json = {"prompt": "two"})
    assert full.status_code == 429 and "for this account" in full.json()["detail"]
    api_full = client.post("/v1/videos", json = {"prompt": "three"})
    assert api_full.status_code == 429
    assert api_full.json()["error"]["code"] == "rate_limit_exceeded"
    video_routes._VIDEO_QUEUE[0].account = _OTHER
    assert client.post(PAGE, json = {"prompt": "four"}).json()["status"] == "queued"
    video_routes._VIDEO_QUEUE[1].account = AccountContext("c" * 32, "carol", "user")
    total = client.post(PAGE, json = {"prompt": "five"})
    assert total.status_code == 429 and "2 videos are already waiting" in total.json()["detail"]
    assert len(video_routes._VIDEO_QUEUE) == 2
    backend.gate.set()


# ---- cancel -----------------------------------------------------------------------------------


def test_cancel_a_queued_page_clip_removes_it_and_says_so(client, backend, prompts):
    running = _busy(client, backend)
    page = client.post(PAGE, json = {"prompt": "never"}).json()
    assert client.post(CANCEL, json = {"video_id": page["video_id"]}).json() == {"cancelled": True}
    assert video_routes._VIDEO_QUEUE == []
    [job] = client.get(QUEUE).json()["jobs"]
    assert (job["video_id"], job["status"]) == (page["video_id"], "cancelled")
    _finish_running(client, backend, running)
    _drain()
    assert [row["prompt"] for row in prompts] == ["running"]


def test_cancel_a_starting_page_clip_waits_for_its_start_then_refuses_cleanly(
    client, backend, monkeypatch
):
    monkeypatch.setattr(video_routes, "_DELETE_SETTLE_TIMEOUT_S", 0.2)
    _busy(client, backend)
    page = client.post(PAGE, json = {"prompt": "slow start"}).json()
    [entry] = video_routes._VIDEO_QUEUE
    entry.starting = True
    still = client.post(CANCEL, json = {"video_id": page["video_id"]})
    assert still.status_code == 409 and "starting" in still.json()["detail"]
    assert client.get(QUEUE).json()["jobs"][0]["status"] == "starting"
    entry.starting = False
    assert client.post(CANCEL, json = {"video_id": page["video_id"]}).json() == {"cancelled": True}
    backend.gate.set()


def test_cancel_a_running_page_clip_by_its_id(client, backend):
    running = _busy(client, backend)
    page = client.post(PAGE, json = {"prompt": "stop me"}).json()
    _finish_running(client, backend, running)
    backend.gate.clear()
    _drain()
    progress = client.get("/api/inference/video/generate-progress").json()
    assert progress["active"] is True and progress["video_id"] == page["video_id"]
    assert client.post(CANCEL, json = {"video_id": page["video_id"]}).json() == {"cancelled": True}
    _wait_idle(backend)
    progress = client.get("/api/inference/video/generate-progress").json()
    assert progress["phase"] == "failed" and progress["video_id"] == page["video_id"]
    backend.gate.set()


# ---- a model-changed head ---------------------------------------------------------------------


def test_a_model_changed_head_is_moved_aside_so_the_next_video_starts(
    client, backend, prompts, monkeypatch
):
    from core.inference.video_families import VIDEO_MODEL_CHANGED_MSG

    running = _busy(client, backend)
    stuck = _queue(client, "stuck")
    _queue(client, "next")
    real_start = video_routes._start_openai_video

    async def _start(body, *args, **kwargs):
        if body.prompt == "stuck":
            raise video_routes._openai_video_error(
                409, VIDEO_MODEL_CHANGED_MSG, code = "model_changed", param = "model"
            )
        return await real_start(body, *args, **kwargs)

    monkeypatch.setattr(video_routes, "_start_openai_video", _start)
    _finish_running(client, backend, running)
    assert _drain() is True
    _wait_idle(backend)
    assert [row["prompt"] for row in prompts] == ["running", "next"]
    [entry] = video_routes._VIDEO_QUEUE
    assert entry.id == stuck["id"] and entry.idle_conflicts == 1
    # Still first in line, and failed once it has conflicted VIDEO_QUEUE_IDLE_CONFLICTS times.
    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_IDLE_CONFLICTS", 2)
    _drain()
    failed = client.get(f"/v1/videos/{stuck['id']}").json()
    assert failed["status"] == "failed" and failed["error"]["code"] == "model_changed"


# ---- no poll gap ------------------------------------------------------------------------------


def test_the_queue_starts_the_next_video_the_moment_the_running_one_ends(
    client, backend, prompts, monkeypatch
):
    # A poll would wait 30 s here: only the engine's wake-up can start the next clip in time.
    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_POLL_S", 30.0)
    running = _busy(client, backend)
    queued = _queue(client, "next")
    loop = asyncio.new_event_loop()
    drainer = threading.Thread(
        target = lambda: loop.run_until_complete(video_routes._drain_video_queue()), daemon = True
    )
    drainer.start()
    try:
        time.sleep(0.3)  # the drainer has found the engine busy and waits
        assert [row["prompt"] for row in prompts] == ["running"]
        backend.gate.set()
        done = _wait_terminal(client, queued["id"], timeout = 5.0)
        assert done["status"] == "completed"
        gap = prompts[1]["start"] - prompts[0]["end"]
        assert gap < 1.0, gap
        drainer.join(5)
        assert not drainer.is_alive()
    finally:
        backend.gate.set()
        if drainer.is_alive():
            loop.call_soon_threadsafe(loop.stop)
            drainer.join(5)
        loop.close()


# ---- restyle ----------------------------------------------------------------------------------


def test_restyle_says_how_many_videos_are_queued(client, backend, monkeypatch):
    from fastapi import HTTPException

    import core.inference.gpu_arbiter as gpu_arbiter
    import routes.video_restyle as restyle_routes

    _busy(client, backend)
    _queue(client, "one")
    _queue(client, "two")
    seen = {}

    def _acquire(
        owner,
        register,
        policy = None,
        admit = None,
    ):
        try:
            admit()
        except HTTPException as exc:
            seen["detail"] = exc.detail
            raise

    import core.inference.video as video_module

    monkeypatch.setattr(gpu_arbiter, "acquire_for_request", _acquire)
    monkeypatch.setattr(video_module, "generation_account_in_flight", lambda: "test-user")
    with pytest.raises(HTTPException):
        restyle_routes._take_the_gpu("auto")
    assert seen["detail"] == (
        "A video is generating. 2 more are queued behind it. Try again when they finish."
    )
    backend.gate.set()


def test_a_request_on_its_way_into_the_queue_is_never_passed_by_a_later_one(
    client, backend, prompts, monkeypatch
):
    running = _busy(client, backend)
    real_display = video_routes._page_queue_display
    later = {}

    def _display_while_the_engine_frees(request):
        # The running clip ends while this request builds its queue entry; a later request
        # arriving now finds an idle engine and an empty queue.
        _finish_running(client, backend, running)
        later["response"] = client.post("/v1/videos", json = {"prompt": "second-in"})
        return real_display(request)

    monkeypatch.setattr(video_routes, "_page_queue_display", _display_while_the_engine_frees)
    first = client.post(PAGE, json = {"prompt": "first-in"}).json()
    assert first["status"] == "queued"
    assert later["response"].json()["status"] == "queued"
    assert [row["prompt"] for row in prompts] == ["running"]
    assert [e.shown_prompt for e in video_routes._ordered_queue()] == ["first-in", "second-in"]
    assert video_routes._admitting == set()
    for _ in range(2):
        _drain()
        _wait_idle(backend)
    assert [row["prompt"] for row in prompts] == ["running", "first-in", "second-in"]


def test_a_finished_page_clip_reads_its_own_outcome_after_a_successor_starts(client, backend):
    running = _busy(client, backend)
    first = client.post(PAGE, json = {"prompt": "first page", "seed": 3}).json()
    _queue(client, "successor")
    _finish_running(client, backend, running)
    backend.gate.clear()
    _drain()
    backend.gate.set()
    _wait_idle(backend)
    # The successor takes the progress record before the page's next poll.
    backend.gate.clear()
    _drain()
    shared = client.get("/api/inference/video/generate-progress").json()
    assert shared["active"] is True and shared["video_id"] != first["video_id"]
    mine = client.get(
        "/api/inference/video/generate-progress", params = {"video_id": first["video_id"]}
    ).json()
    assert mine["phase"] == "completed" and mine["video_id"] == first["video_id"]
    assert mine["video"]["id"] == first["video_id"] and mine["video"]["seed"] == 3
    with _client_as(_OTHER) as other:
        theirs = other.get(
            "/api/inference/video/generate-progress", params = {"video_id": first["video_id"]}
        ).json()
        assert theirs.get("video") is None and theirs.get("phase") != "completed"
    backend.gate.set()


def test_a_queued_api_clip_that_cannot_start_reports_why_on_the_queue(client, backend):
    running = _busy(client, backend)
    queued = _queue(client, "doomed api")
    _finish_running(client, backend, running)
    backend.loaded = False
    _drain()
    [job] = client.get(QUEUE).json()["jobs"]
    assert (job["video_id"], job["status"], job["source"]) == (queued["id"], "failed", "api")


def test_a_queued_page_clip_never_renders_on_a_model_swapped_in_while_it_waited(
    client, backend, prompts
):
    import types

    from core.inference.video_families import detect_video_family

    def _resident(repo):
        backend._state = types.SimpleNamespace(
            family = detect_video_family("Lightricks/LTX-2"),
            h3_task = None,
            engine = "diffusers",
            repo_id = repo,
        )

    _resident("unsloth/LTX-2.3-GGUF")
    running = _busy(client, backend)
    page = client.post(PAGE, json = {"prompt": "judged on the first model"}).json()
    assert page["status"] == "queued"
    _finish_running(client, backend, running)
    _resident("someone/other-ltx-build")
    _drain()
    assert [row["prompt"] for row in prompts] == ["running"]
    [job] = client.get(QUEUE).json()["jobs"]
    assert (job["video_id"], job["status"]) == (page["video_id"], "failed")
    assert "model changed" in job["error"]
    # The same resident it was judged on still renders it.
    running = _busy(client, backend, "again")
    page = client.post(PAGE, json = {"prompt": "same model"}).json()
    _finish_running(client, backend, running)
    _drain()
    _wait_idle(backend)
    assert [row["prompt"] for row in prompts][-1] == "same model"


def test_the_queue_caps_the_bytes_waiting_requests_hold(client, backend, monkeypatch):
    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_MAX_BYTES_PER_ACCOUNT", 3000)
    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_MAX_BYTES", 5000)
    big = "x" * 2000
    _busy(client, backend)
    # An account with nothing waiting is admitted whatever its size.
    assert client.post(PAGE, json = {"prompt": big}).json()["status"] == "queued"
    over = client.post(PAGE, json = {"prompt": big})
    assert over.status_code == 429 and "MB" in over.json()["detail"]
    api_over = client.post("/v1/videos", json = {"prompt": big})
    assert api_over.status_code == 429
    assert api_over.json()["error"]["code"] == "rate_limit_exceeded"
    assert client.post(PAGE, json = {"prompt": "small"}).json()["status"] == "queued"
    # The total: another account is refused once the queue holds the process's share.
    video_routes._VIDEO_QUEUE[0].account = _OTHER
    video_routes._VIDEO_QUEUE[0].payload_bytes = 4000
    total = client.post(PAGE, json = {"prompt": big})
    assert total.status_code == 429 and "The waiting videos" in total.json()["detail"]
    assert len(video_routes._VIDEO_QUEUE) == 2
    backend.gate.set()


def test_a_queued_page_clip_never_renders_with_adapters_baked_in_while_it_waited(
    client, backend, prompts
):
    import types

    from core.inference import video_lora
    from core.inference.video_families import detect_video_family

    def _resident(adapters):
        pipe = types.SimpleNamespace()
        setattr(pipe, video_lora.LORAS_BAKED_ATTR, bool(adapters))
        setattr(pipe, video_lora.BAKED_SPECS_ATTR, tuple(adapters))
        backend._state = types.SimpleNamespace(
            family = detect_video_family("Lightricks/LTX-2"),
            h3_task = None,
            engine = "diffusers",
            repo_id = "unsloth/LTX-2.3-GGUF",
            pipe = pipe,
        )

    _resident([("org/style-lora", 1.0)])
    running = _busy(client, backend)
    page = client.post(PAGE, json = {"prompt": "judged with one adapter"}).json()
    _finish_running(client, backend, running)
    # Same repo, kind, quant, task and family: only the baked adapter weight changed.
    _resident([("org/style-lora", 0.5)])
    _drain()
    assert [row["prompt"] for row in prompts] == ["running"]
    [job] = client.get(QUEUE).json()["jobs"]
    assert (job["video_id"], job["status"]) == (page["video_id"], "failed")
    assert "model changed" in job["error"]


@pytest.mark.parametrize("removal", ["delete", "archive", "clear"])
def test_a_clip_removed_from_the_gallery_has_no_outcome_left_to_read_by_id(
    client, backend, removal
):
    running = _busy(client, backend)
    first = client.post(PAGE, json = {"prompt": "removed later"}).json()
    _queue(client, "successor")
    _finish_running(client, backend, running)
    _drain()
    _wait_idle(backend)
    backend.gate.clear()
    _drain()  # the successor holds the shared record
    progress = "/api/inference/video/generate-progress"
    mine = client.get(progress, params = {"video_id": first["video_id"]}).json()
    assert mine["phase"] == "completed" and mine["video"]["id"] == first["video_id"]
    gallery = f"/api/inference/video/gallery/{first['video_id']}"
    if removal == "delete":
        assert client.delete(gallery).status_code == 200
    elif removal == "archive":
        assert client.patch(gallery, json = {"archived": True}).status_code == 200
    else:
        assert client.delete("/api/inference/video/gallery").status_code == 200
    gone = client.get(progress, params = {"video_id": first["video_id"]}).json()
    assert gone.get("video") is None and gone.get("phase") != "completed"
    backend.gate.set()


def test_a_queued_page_clip_never_renders_on_another_checkpoint_of_the_same_repo(
    client, backend, prompts
):
    import types

    from core.inference.video_families import detect_video_family

    def _resident(filename):
        backend._state = types.SimpleNamespace(
            family = detect_video_family("Lightricks/LTX-2"),
            h3_task = None,
            engine = "diffusers",
            repo_id = "unsloth/LTX-2.5-GGUF",
            kind = "gguf",
            gguf_filename = filename,
        )

    _resident("ltx-2.5-dev-Q4_K_M.gguf")
    running = _busy(client, backend)
    page = client.post(PAGE, json = {"prompt": "judged on the dev checkpoint"}).json()
    _finish_running(client, backend, running)
    # Same repo, kind and quant: only the checkpoint file differs.
    _resident("distilled/ltx-2.5-distilled-Q4_K_M.gguf")
    _drain()
    assert [row["prompt"] for row in prompts] == ["running"]
    [job] = client.get(QUEUE).json()["jobs"]
    assert (job["video_id"], job["status"]) == (page["video_id"], "failed")
    assert "model changed" in job["error"]


def test_another_account_clearing_its_gallery_keeps_this_accounts_outcome(client, backend):
    running = _busy(client, backend)
    first = client.post(PAGE, json = {"prompt": "kept"}).json()
    _queue(client, "successor")
    _finish_running(client, backend, running)
    _drain()
    _wait_idle(backend)
    backend.gate.clear()
    _drain()  # the successor holds the shared record
    progress = "/api/inference/video/generate-progress"
    with _client_as(_OTHER) as other:
        assert other.delete("/api/inference/video/gallery").status_code == 200
    mine = client.get(progress, params = {"video_id": first["video_id"]}).json()
    assert mine["phase"] == "completed" and mine["video"]["id"] == first["video_id"]
    backend.gate.set()
