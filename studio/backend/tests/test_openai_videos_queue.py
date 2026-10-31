# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


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


def _delete_in_background(client, video_id: str) -> tuple:
    import threading
    import time

    result = {}
    thread = threading.Thread(
        target = lambda: result.update(response = client.delete(f"/v1/videos/{video_id}"))
    )
    thread.start()
    time.sleep(0.2)
    return thread, result


def _render_one(
    client,
    backend,
    prompt = "first",
) -> dict:
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

    _drain()
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


def test_cancelling_a_queued_video_by_id_keeps_it_from_starting(client, backend):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "never"}).json()
    cancelled = client.post("/api/inference/video/generate/cancel", json = {"video_id": queued["id"]})
    assert cancelled.status_code == 200 and cancelled.json() == {"cancelled": True}
    assert video_routes._VIDEO_QUEUE == []

    backend.gate.set()
    _wait_terminal(client, first["id"])
    _drain()
    assert backend.last_generate_kwargs["prompt"] == "first"


def test_a_stop_by_id_while_a_queued_start_waits_on_admission_stops_its_render(
    client, backend, monkeypatch
):
    import threading

    from core.inference import media_keepwarm

    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "stopped"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    backend.gate.clear()
    real_begin = media_keepwarm.begin_request
    result = {}
    threads = []

    async def _stop_during_admission(*args, **kwargs):
        thread = threading.Thread(
            target = lambda: result.update(
                response = client.post(
                    "/api/inference/video/generate/cancel", json = {"video_id": queued["id"]}
                )
            )
        )
        thread.start()
        threads.append(thread)
        await asyncio.sleep(0.2)
        return await real_begin(*args, **kwargs)

    monkeypatch.setattr(media_keepwarm, "begin_request", _stop_during_admission)
    try:
        _drain()
        threads[0].join(10)
        assert result["response"].status_code == 200, result["response"].json()
        assert result["response"].json() == {"cancelled": True}
        assert _wait_terminal(client, queued["id"])["status"] == "failed"
    finally:
        backend.gate.set()


def test_a_video_deleted_while_its_start_is_in_flight_is_deleted_once_started(
    client, backend, monkeypatch
):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "late"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    real_start = video_routes._start_openai_video
    pending = []

    async def _start_then_delete(*args, **kwargs):
        job = await real_start(*args, **kwargs)
        pending.append(await asyncio.to_thread(_delete_in_background, client, queued["id"]))
        return job

    monkeypatch.setattr(video_routes, "_start_openai_video", _start_then_delete)
    _drain()
    [(thread, result)] = pending
    thread.join(10)
    assert result["response"].status_code == 200, result["response"].json()
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
    pending = []

    def _fail_with_a_delete(entry, exc):
        pending.append(_delete_in_background(client, queued["id"]))
        real_fail(entry, exc)

    monkeypatch.setattr(video_routes, "_fail_queued_video", _fail_with_a_delete)
    _drain()
    [(thread, result)] = pending
    thread.join(10)
    assert result["response"].status_code == 200, result["response"].json()
    assert video_routes._VIDEO_QUEUE == []
    assert client.get(f"/v1/videos/{queued['id']}").status_code == 404
    assert queued["id"] not in [v["id"] for v in client.get("/v1/videos").json()["data"]]


def test_a_delete_during_a_start_that_does_not_settle_is_refused_rather_than_reported_done(
    client, backend, monkeypatch
):
    monkeypatch.setattr(video_routes, "_DELETE_SETTLE_TIMEOUT_S", 0.2)
    _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "slow start"}).json()
    [entry] = video_routes._VIDEO_QUEUE
    entry.starting = True
    still = client.delete(f"/v1/videos/{queued['id']}")
    assert still.status_code == 409
    assert still.json()["error"]["code"] == "video_not_ready"
    assert video_routes._VIDEO_QUEUE == [entry]
    entry.starting = False
    assert client.delete(f"/v1/videos/{queued['id']}").status_code == 200
    assert video_routes._VIDEO_QUEUE == []
    backend.gate.set()


def test_a_queued_video_of_a_disabled_account_never_starts(client, backend):
    first = _render_one(client, backend)
    client.post("/v1/videos", json = {"prompt": "theirs"})
    video_routes._VIDEO_QUEUE[0].account = _OTHER
    backend.gate.set()
    _wait_terminal(client, first["id"])
    _drain()
    assert video_routes._VIDEO_QUEUE == []
    assert backend.last_generate_kwargs["prompt"] == "first"


def _disable_while(monkeypatch, module, name) -> dict:
    live = {"active": True}
    real = getattr(module, name)

    async def _wait_then_disable(*args, **kwargs):
        live["active"] = False
        return await real(*args, **kwargs)

    monkeypatch.setattr(video_routes, "_queued_account_active", lambda account: live["active"])
    monkeypatch.setattr(module, name, _wait_then_disable)
    return live


def _assert_disabled_mid_start_never_renders(client, backend, queued) -> None:
    _drain()
    assert video_routes._VIDEO_QUEUE == []
    assert backend.last_generate_kwargs["prompt"] == "first"
    assert client.get(f"/v1/videos/{queued['id']}").status_code == 404


def test_an_account_disabled_during_the_admission_wait_never_starts(client, backend, monkeypatch):
    from core.inference import media_keepwarm

    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "theirs"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    _disable_while(monkeypatch, media_keepwarm, "begin_request")
    _assert_disabled_mid_start_never_renders(client, backend, queued)


def test_an_account_disabled_during_an_auto_switch_never_starts(client, backend, monkeypatch):
    from core.inference import media_auto_switch

    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "theirs"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    _disable_while(monkeypatch, media_auto_switch, "maybe_auto_switch_media_model")
    _assert_disabled_mid_start_never_renders(client, backend, queued)


def test_the_account_check_reads_the_accounts_current_state(monkeypatch):
    import auth.storage as storage

    accounts = {}
    monkeypatch.setattr(storage, "get_account_by_id", accounts.get)
    assert video_routes._queued_account_active(_OTHER) is False
    accounts[_OTHER.account_id] = _OTHER
    assert video_routes._queued_account_active(_OTHER) is True
    owner = AccountContext("owner", "owner", "owner")
    assert video_routes._queued_account_active(owner) is True


def test_a_managed_account_denied_the_loaded_model_gets_a_403_not_its_name(
    client, backend, monkeypatch
):
    from fastapi import HTTPException

    from hub.services.models import account_access

    _render_one(client, backend)

    def _deny(status, modality = None):
        raise HTTPException(status_code = 403, detail = "Model not found")

    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "require_media_generation_access", _deny)
    denied = client.post("/v1/videos", json = {"prompt": "x"})
    assert denied.status_code == 403
    assert "LTX" not in denied.text
    assert video_routes._VIDEO_QUEUE == []
    backend.gate.set()


def test_a_queued_job_for_another_model_shows_none_of_the_loaded_models_details(
    client, backend, monkeypatch
):
    import utils.openai_auto_switch_settings as auto_switch_settings
    from core.inference import media_auto_switch

    _render_one(client, backend)
    monkeypatch.setattr(auto_switch_settings, "get_media_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(media_auto_switch, "resident_answers_media_request", lambda *a, **k: False)
    queued = client.post("/v1/videos", json = {"prompt": "x", "model": "org/other-video"}).json()
    assert queued["status"] == "queued"
    assert (queued["model"], queued["size"], queued["seconds"]) == (
        "org/other-video",
        "auto",
        "auto",
    )
    backend.gate.set()


def test_a_queued_job_reports_the_size_and_seconds_an_immediate_start_would(client, backend):
    first = _render_one(client, backend)
    immediate_shape = {k: first[k] for k in ("size", "seconds", "model")}
    queued = client.post("/v1/videos", json = {"prompt": "second"}).json()
    assert queued["status"] == "queued"
    assert {k: queued[k] for k in ("size", "seconds", "model")} == immediate_shape
    timed = client.post("/v1/videos", json = {"prompt": "third", "seconds": "4"}).json()
    assert timed["seconds"] == "4.04"
    backend.gate.set()
    _wait_terminal(client, first["id"])
    _drain()
    started = _wait_terminal(client, queued["id"])
    assert (started["size"], started["model"]) == (
        immediate_shape["size"],
        immediate_shape["model"],
    )


def test_creates_arriving_together_cannot_pass_the_cap_together(client, backend, monkeypatch):
    import time

    import httpx

    monkeypatch.setattr(video_routes, "VIDEO_QUEUE_MAX_PER_ACCOUNT", 1)
    real_display = video_routes._queued_job_display

    def _slow_display(*args):
        time.sleep(0.2)
        return real_display(*args)

    monkeypatch.setattr(video_routes, "_queued_job_display", _slow_display)
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


def test_a_queued_start_is_counted_by_the_media_admission_gate(client, backend, monkeypatch):
    from core.inference import media_keepwarm
    from core.inference.gpu_arbiter import VIDEO

    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "counted"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    tracker = media_keepwarm._TRACKERS[VIDEO]
    real_start = video_routes._start_openai_video
    seen = []

    async def _start_and_count(*args, **kwargs):
        seen.append(tracker.outstanding())
        return await real_start(*args, **kwargs)

    monkeypatch.setattr(video_routes, "_start_openai_video", _start_and_count)
    before = tracker.outstanding()
    _drain()
    assert seen == [before + 1]
    assert tracker.outstanding() == before
    assert _wait_terminal(client, queued["id"])["status"] == "completed"


def test_a_queued_video_waits_while_a_model_load_is_in_flight(client, backend):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "after the load"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    backend.loaded = False
    backend.loading = ("unsloth/LTX-2.3-GGUF",)
    _drain()
    assert client.get(f"/v1/videos/{queued['id']}").json()["status"] == "queued"
    assert [entry.id for entry in video_routes._VIDEO_QUEUE] == [queued["id"]]

    backend.loading = ()
    backend.loaded = True
    _drain()
    assert _wait_terminal(client, queued["id"])["status"] == "completed"


def test_a_load_that_starts_during_the_admission_wait_keeps_the_job_queued(
    client, backend, monkeypatch
):
    from core.inference import media_keepwarm

    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "second"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    real_begin = media_keepwarm.begin_request

    async def _switch_starts(*args, **kwargs):
        backend.loaded = False
        backend.loading = ("unsloth/LTX-2.3-GGUF",)
        return await real_begin(*args, **kwargs)

    monkeypatch.setattr(media_keepwarm, "begin_request", _switch_starts)
    _drain()
    assert client.get(f"/v1/videos/{queued['id']}").json()["status"] == "queued"
    assert [entry.id for entry in video_routes._VIDEO_QUEUE] == [queued["id"]]

    monkeypatch.setattr(media_keepwarm, "begin_request", real_begin)
    backend.loading = ()
    backend.loaded = True
    _drain()
    assert _wait_terminal(client, queued["id"])["status"] == "completed"


def test_a_not_loaded_refusal_during_a_load_keeps_the_job_queued(client, backend, monkeypatch):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "second"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    real_start = video_routes._start_openai_video

    async def _load_begins_mid_start(*args, **kwargs):
        backend.loaded = False
        backend.loading = ("unsloth/LTX-2.3-GGUF",)
        return await real_start(*args, **kwargs)

    monkeypatch.setattr(video_routes, "_start_openai_video", _load_begins_mid_start)
    _drain()
    assert client.get(f"/v1/videos/{queued['id']}").json()["status"] == "queued"

    monkeypatch.setattr(video_routes, "_start_openai_video", real_start)
    backend.loading = ()
    backend.loaded = True
    _drain()
    assert _wait_terminal(client, queued["id"])["status"] == "completed"


def test_model_dependent_input_is_refused_at_once_rather_than_queued(client, backend):
    bad = ({"num_frames": 2}, {"size": "769x512"})
    idle = [client.post("/v1/videos", json = {"prompt": "bad", **payload}) for payload in bad]
    assert [r.status_code for r in idle] == [400, 400]
    _render_one(client, backend)
    try:
        busy = [client.post("/v1/videos", json = {"prompt": "bad", **payload}) for payload in bad]
        assert [r.status_code for r in busy] == [400, 400]
        assert [r.json()["error"] for r in busy] == [r.json()["error"] for r in idle]
        assert video_routes._VIDEO_QUEUE == []
    finally:
        backend.gate.set()


def test_a_queued_job_names_the_resident_model_when_auto_switch_is_off(
    client, backend, monkeypatch
):
    import utils.openai_auto_switch_settings as auto_switch_settings

    monkeypatch.setattr(auto_switch_settings, "get_media_auto_switch_enabled", lambda: False)
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "x", "model": "sora-2"}).json()
    assert queued["status"] == "queued"
    assert queued["model"] == first["model"] != "sora-2"
    backend.gate.set()


def test_a_queued_start_holds_an_inference_slot_so_a_switch_cannot_evict_a_running_chat(
    client, backend, monkeypatch
):
    from core.inference import llama_keepwarm, media_switch_backends

    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "behind a chat"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    real_start = video_routes._start_openai_video
    seen = []

    async def _start_and_check(*args, **kwargs):
        seen.append(media_switch_backends.chat_busy())
        return await real_start(*args, **kwargs)

    monkeypatch.setattr(video_routes, "_start_openai_video", _start_and_check)
    before = llama_keepwarm.other_inference_request_count(current_request_counted = False)
    llama_keepwarm._note_pending()
    llama_keepwarm._note_start()
    try:
        _drain()
    finally:
        llama_keepwarm._note_end()
    assert seen == [True]
    assert llama_keepwarm.other_inference_request_count(current_request_counted = False) == before
    assert _wait_terminal(client, queued["id"])["status"] == "completed"


def test_an_oversized_reference_with_an_explicit_size_is_refused_rather_than_queued(
    client, backend
):
    from test_openai_videos_route import _reference_image_bytes

    png = _reference_image_bytes("PNG", (5000, 100))
    wide = {"input_reference": ("frame.png", png, "image/png")}
    _render_one(client, backend)
    try:
        r = _create(client, {"prompt": "wide", "size": "768x512"}, wide)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["param"] == "input_reference"
        assert "4096" in r.json()["error"]["message"]
    finally:
        backend.gate.set()


def test_a_queued_image_to_video_job_reports_the_canvas_of_its_reference(client, backend):
    from test_openai_videos_route import _reference_image_bytes

    png = _reference_image_bytes("PNG", (300, 600))
    portrait = {"input_reference": ("frame.png", png, "image/png")}
    immediate = _create(client, {"prompt": "tall"}, portrait).json()
    assert immediate["size"] == "704x1216"
    _wait_terminal(client, immediate["id"])
    _render_one(client, backend)
    try:
        queued = _create(client, {"prompt": "tall"}, portrait).json()
        assert queued["status"] == "queued"
        assert queued["size"] == immediate["size"]
        assert client.get(f"/v1/videos/{queued['id']}").json()["size"] == immediate["size"]
    finally:
        backend.gate.set()


def test_an_account_disabled_while_its_conditioning_is_prepared_never_starts(
    client, backend, monkeypatch
):
    first = _render_one(client, backend)
    queued = client.post("/v1/videos", json = {"prompt": "theirs"}).json()
    backend.gate.set()
    _wait_terminal(client, first["id"])
    live = {"active": True}
    real_resolve = backend._resolve_keyframes

    def _disable_then_resolve(*args, **kwargs):
        live["active"] = False
        return real_resolve(*args, **kwargs)

    monkeypatch.setattr(video_routes, "_queued_account_active", lambda account: live["active"])
    monkeypatch.setattr(backend, "_resolve_keyframes", _disable_then_resolve)
    _assert_disabled_mid_start_never_renders(client, backend, queued)
    assert backend._generate_job_active is False
    assert backend._active_generate_cancel is None


def test_a_queued_job_for_another_downloaded_model_reports_that_models_size_and_seconds(
    client, backend, monkeypatch
):
    from types import SimpleNamespace

    import core.inference.media_model_index as media_model_index
    import core.inference.video as video_module
    import utils.openai_auto_switch_settings as auto_switch_settings
    from core.inference import media_auto_switch
    from core.inference.video_families import detect_video_family

    wan = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    _render_one(client, backend)
    monkeypatch.setattr(auto_switch_settings, "get_media_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(media_auto_switch, "resident_answers_media_request", lambda *a, **k: False)
    monkeypatch.setattr(
        media_model_index,
        "resolve_local_media_model",
        lambda *a, **k: SimpleNamespace(model_path = wan, gguf_filename = None),
    )
    monkeypatch.setattr(media_model_index, "expected_partition", lambda pick: None)
    monkeypatch.setattr(
        video_module, "_detect_load_family", lambda *a, **k: detect_video_family(wan)
    )
    try:
        queued = client.post("/v1/videos", json = {"prompt": "x", "model": wan}).json()
        assert queued["status"] == "queued"
        assert (queued["size"], queued["seconds"]) == ("832x480", "2.06")
        timed = client.post("/v1/videos", json = {"prompt": "y", "model": wan, "seconds": "4"})
        assert timed.json()["seconds"] == "4.06"
    finally:
        backend.gate.set()
