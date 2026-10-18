# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generate_video chat tool, driven through the real job path on a fake video backend."""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import pytest

import core.inference.gpu_arbiter as gpu_arbiter
import core.inference.media_chat_tools as media_chat_tools
import core.inference.media_keepwarm as media_keepwarm
import core.inference.video as video_module
import core.inference.video_gallery as gallery_module
import routes.video as video_routes
from core.inference import generate_video_tool as tool
from models.inference import VideoJobCreateRequest
from utils.account_context import AccountContext, run_as

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_openai_videos_route import _GatedBackend  # noqa: E402

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def _healthy_diffusers(healthy_diffusers):
    pass


@pytest.fixture
def server_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    media_chat_tools.remember_server_loop(loop)
    yield loop
    media_chat_tools.remember_server_loop(None)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()


@pytest.fixture
def backend(monkeypatch, tmp_path, server_loop):
    backend = _GatedBackend()
    backend.loaded = True
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.VIDEO, lambda: None)
    (tmp_path / "gallery").mkdir()
    monkeypatch.setattr(gallery_module, "gallery_dir", lambda: tmp_path / "gallery")
    monkeypatch.setattr(video_routes, "_jobs", {})
    monkeypatch.setattr(video_routes, "_managed_jobs", {})
    monkeypatch.setattr(media_chat_tools, "media_chat_status", lambda task: backend.status())
    monkeypatch.setattr(tool, "_POLL_S", 0.02)
    yield backend
    backend.gate.set()


@pytest.fixture
def keepwarm(monkeypatch):
    calls: list = []

    async def _begin(owner):
        calls.append(("begin", owner))

    def _end(owner, *, counted = True):
        calls.append(("end", owner, counted))

    monkeypatch.setattr(media_keepwarm, "begin_request", _begin)
    monkeypatch.setattr(media_keepwarm, "end_request", _end)
    return calls


def _run(tmp_path, args, **kwargs):
    return tool.run_generate_video(args, workdir = str(tmp_path / "work"), **kwargs)


def _only_job() -> str:
    jobs = list(video_routes._jobs) or [
        i for jobs in video_routes._managed_jobs.values() for i in jobs
    ]
    assert len(jobs) == 1, jobs
    return jobs[0]


def _wait_phase(
    backend,
    *phases,
    timeout = 5.0,
):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if backend.generate_progress().get("phase") in phases:
            return
        time.sleep(0.01)
    raise AssertionError(backend.generate_progress())


def test_completed_clip_is_copied_into_the_workdir_with_a_files_row(tmp_path, backend, keepwarm):
    out = _run(tmp_path, {"prompt": "  a fox\n runs  ", "seconds": 2, "width": 768, "height": 512})
    text, files = out.split("\n__FILES__:")
    assert "__IMAGES__" not in out
    assert text.startswith("Generated a video (") and text.endswith("Prompt: a fox runs")
    video_id = _only_job()
    name = f"generated-video-{video_id.removeprefix('video_')[:8]}.mp4"
    copied = tmp_path / "work" / name
    assert copied.read_bytes() == gallery_module.owned_video_path(video_id).read_bytes()
    assert json.loads(files) == [{"name": name, "size": copied.stat().st_size}]
    sent = backend.last_generate_kwargs
    assert (sent["prompt"], sent["width"], sent["height"]) == ("a fox\n runs", 768, 512)
    assert keepwarm == [("begin", gpu_arbiter.VIDEO), ("end", gpu_arbiter.VIDEO, True)]


def test_model_none_never_reaches_the_auto_switch_or_a_load(
    tmp_path, backend, keepwarm, monkeypatch
):
    import core.inference.media_auto_switch as mas
    import utils.openai_auto_switch_settings as switch_settings

    requested: list = []
    real_switch = mas.maybe_auto_switch_media_model

    async def _spy(model, **kwargs):
        requested.append(model)
        return await real_switch(model, **kwargs)

    def _setting_read():
        raise AssertionError("the auto-switch went past its model=None early return")

    def _no_acquire(*args, **kwargs):
        raise AssertionError("the GPU arbiter was asked for the video slot")

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _spy)
    monkeypatch.setattr(switch_settings, "get_media_auto_switch_enabled", _setting_read)
    for name in ("acquire", "acquire_async", "acquire_for_load"):
        if hasattr(gpu_arbiter, name):
            monkeypatch.setattr(gpu_arbiter, name, _no_acquire)
    assert "\n__FILES__:" in _run(tmp_path, {"prompt": "x", "model": "other/model"})
    assert requested == [None]
    assert backend.last_load_kwargs == {}


def test_a_running_job_maps_the_busy_409(tmp_path, backend, keepwarm):
    backend.gate.clear()
    assert _run(tmp_path, {"prompt": "first"}, timeout = 0.2) == tool.STILL_RENDERING_MSG
    assert _run(tmp_path, {"prompt": "second"}, timeout = 0.2) == tool.BUSY_MSG
    assert keepwarm[-1] == ("end", gpu_arbiter.VIDEO, False)


def test_timeout_leaves_the_job_running(tmp_path, backend, keepwarm, monkeypatch):
    backend.gate.clear()
    cancels: list = []
    monkeypatch.setattr(backend, "cancel_generate", lambda **kw: cancels.append(kw))
    assert _run(tmp_path, {"prompt": "slow"}, timeout = 0.2) == tool.STILL_RENDERING_MSG
    assert video_routes._lookup_video(_only_job()).status in ("queued", "in_progress")
    assert cancels == []
    assert keepwarm == [("begin", gpu_arbiter.VIDEO), ("end", gpu_arbiter.VIDEO, True)]


def test_the_cancel_event_stops_the_job(tmp_path, backend, keepwarm):
    backend.gate.clear()
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    assert _run(tmp_path, {"prompt": "stop me"}, cancel_event = stop) == tool.STOPPED_MSG
    _wait_phase(backend, "failed")
    job = video_routes._lookup_video(_only_job())
    assert job.status == "failed" and job.error.code == "cancelled"
    assert not (tmp_path / "work").exists()


def test_a_clip_that_completes_inside_the_stop_window_is_still_returned(
    tmp_path, backend, keepwarm, monkeypatch
):
    # Stop lands at the last step: the backend finishes the clip instead of cancelling.
    backend.gate.clear()
    monkeypatch.setattr(backend, "cancel_generate", lambda **kw: backend.gate.set())
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    result = _run(tmp_path, {"prompt": "last step"}, cancel_event = stop)
    assert result.startswith("Generated a video"), result
    assert [p.name for p in (tmp_path / "work").iterdir() if p.suffix == ".mp4"]


def _slow_create(monkeypatch, keepwarm, delay):
    async def _begin(owner):
        keepwarm.append(("begin", owner))
        await asyncio.sleep(delay)

    monkeypatch.setattr(media_keepwarm, "begin_request", _begin)


def test_a_slow_create_is_never_cancelled_so_its_job_stays_tracked(
    tmp_path, backend, keepwarm, monkeypatch
):
    backend.gate.clear()
    _slow_create(monkeypatch, keepwarm, 0.6)
    monkeypatch.setattr(tool, "_CREATE_TIMEOUT_S", 0.1)
    assert _run(tmp_path, {"prompt": "slow create"}, timeout = 30) == tool.STILL_RENDERING_MSG
    deadline = time.monotonic() + 5
    while not (video_routes._jobs or video_routes._managed_jobs) and time.monotonic() < deadline:
        time.sleep(0.02)
    # The create finished after the tool answered, and the job is on record.
    assert video_routes._lookup_video(_only_job()).status in ("queued", "in_progress")


def test_stop_during_a_slow_create_stops_the_job_it_starts(tmp_path, backend, keepwarm, monkeypatch):
    backend.gate.clear()
    _slow_create(monkeypatch, keepwarm, 0.6)
    monkeypatch.setattr(tool, "_STOP_WAIT_S", 0.05)
    stop = threading.Event()
    stop.set()
    assert _run(tmp_path, {"prompt": "stop early"}, cancel_event = stop) == tool.STOPPED_MSG
    deadline = time.monotonic() + 5
    while not (video_routes._jobs or video_routes._managed_jobs) and time.monotonic() < deadline:
        time.sleep(0.02)
    _wait_phase(backend, "failed")
    job = video_routes._lookup_video(_only_job())
    assert job.status == "failed" and job.error.code == "cancelled"


def test_a_failed_job_reports_its_error(tmp_path, backend, keepwarm):
    backend.fail_with = "the sampler blew up"
    assert _run(tmp_path, {"prompt": "x"}).startswith("Error: ")
    assert video_routes._lookup_video(_only_job()).status == "failed"


def test_not_loaded_and_bad_input_are_error_strings(tmp_path, backend, keepwarm, monkeypatch):
    assert _run(tmp_path, {"prompt": " "}) == tool.NO_PROMPT_MSG
    backend.loaded = False
    assert _run(tmp_path, {"prompt": "x"}) == tool.NOT_LOADED_MSG
    monkeypatch.setattr(media_chat_tools, "media_chat_status", lambda task: None)
    assert _run(tmp_path, {"prompt": "x"}) == tool.NOT_LOADED_MSG
    backend.loaded = True
    monkeypatch.setattr(media_chat_tools, "media_chat_status", lambda task: backend.status())
    assert _run(tmp_path, {"prompt": "x", "width": 8, "height": 8}).startswith("Error: ")
    media_chat_tools.remember_server_loop(None)
    assert _run(tmp_path, {"prompt": "x"}) == tool.NO_LOOP_MSG
    assert keepwarm == [("begin", gpu_arbiter.VIDEO), ("end", gpu_arbiter.VIDEO, False)] * 2


def test_another_accounts_generation_maps_to_its_own_error(
    tmp_path, backend, keepwarm, monkeypatch
):
    async def _busy(*args, **kwargs):
        raise gpu_arbiter.GpuBusyForAnotherAccountError(gpu_arbiter.VIDEO, 1)

    monkeypatch.setattr(video_routes, "_create_openai_video", _busy)
    assert _run(tmp_path, {"prompt": "x"}) == tool.OTHER_ACCOUNT_MSG


@pytest.mark.parametrize("has_seed", [False, True])
def test_seed_is_offered_and_forwarded_only_when_the_request_has_the_field(
    tmp_path, backend, keepwarm, monkeypatch, has_seed
):
    import models.inference as models_inference

    class _Seeded(VideoJobCreateRequest):
        seed: Optional[int] = None

    class _Plain(VideoJobCreateRequest):
        pass

    _Plain.model_fields.pop("seed", None)
    monkeypatch.setattr(models_inference, "VideoJobCreateRequest", _Seeded if has_seed else _Plain)
    seen: list = []

    async def _capture(body, reference, subject, hf_token):
        seen.append(body.model_dump(exclude_none = True))
        raise video_routes.HTTPException(status_code = 400, detail = {"error": {"message": "stop"}})

    monkeypatch.setattr(video_routes, "_create_openai_video", _capture)
    schema = tool.generate_video_schema(backend.status())
    assert ("seed" in schema["function"]["parameters"]["properties"]) is has_seed
    assert _run(tmp_path, {"prompt": "x", "seed": 7}) == "Error: stop"
    assert seen == [{"prompt": "x", **({"seed": 7} if has_seed else {})}]


def test_the_schema_names_the_tool_and_appends_the_prompt_hint(backend):
    schema = tool.generate_video_schema({**backend.status(), "prompt_hint": "Say the camera move."})
    fn = schema["function"]
    assert fn["name"] == "generate_video" and fn["parameters"]["required"] == ["prompt"]
    assert fn["description"].startswith("Generate one short video clip with the video model")
    assert fn["description"].endswith(" Say the camera move.")
    assert "Studio" not in fn["description"].replace("Unsloth Studio", "")


def test_a_managed_accounts_job_resolves_under_that_account(
    tmp_path, backend, keepwarm, monkeypatch, server_loop
):
    import contextvars

    from hub.services.models import account_access

    monkeypatch.setattr(account_access, "require_media_generation_access", lambda *a, **k: None)
    # The coroutine must bind the account itself, not lean on the caller's context leaking over.
    schedule = server_loop.call_soon_threadsafe
    monkeypatch.setattr(
        server_loop,
        "call_soon_threadsafe",
        lambda cb, *args, context = None: schedule(cb, *args, context = contextvars.Context()),
    )
    out = run_as(ALICE, _run, tmp_path, {"prompt": "private"})
    assert "\n__FILES__:" in out
    assert video_routes._jobs == {}
    assert list(video_routes._managed_jobs) == [ALICE.account_id]
    video_id = _only_job()
    assert run_as(ALICE, video_routes._lookup_video, video_id).status == "completed"
    assert backend._generate_job_account == ALICE.account_id
