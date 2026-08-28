# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import core.inference.gpu_arbiter as gpu_arbiter
import core.inference.video as video_module
import core.inference.video_gallery as gallery_module
import routes.video as video_routes
from auth.authentication import get_current_subject
from core.inference.video_families import VIDEO_CANCELLED_MSG
from routes.video import (
    _format_seconds,
    _frames_for_seconds,
    _parse_openai_video_seconds,
    _parse_openai_video_size,
)
from routes.video import openai_router, router as video_router
from utils.api_errors import install_api_error_handlers

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_video_routes import _FakeBackend  # noqa: E402


@pytest.mark.parametrize(
    "size, expected",
    [
        (None, None),
        ("", None),
        ("auto", None),
        ("768x512", (768, 512)),
        (" 1216 X 704 ", (1216, 704)),
    ],
)
def test_parse_video_size_ok(size, expected):
    assert _parse_openai_video_size(size) == expected


@pytest.mark.parametrize("size", ["abc", "768", "x512", "16x16", "4096x4096"])
def test_parse_video_size_rejects(size):
    with pytest.raises(ValueError):
        _parse_openai_video_size(size)


@pytest.mark.parametrize(
    "seconds, expected", [(None, None), ("", None), ("4", 4.0), ("2.5", 2.5), ("12", 12.0)]
)
def test_parse_video_seconds_ok(seconds, expected):
    assert _parse_openai_video_seconds(seconds) == expected


@pytest.mark.parametrize("seconds", ["abc", "0", "-1", "nan", "inf", "1000"])
def test_parse_video_seconds_rejects(seconds):
    with pytest.raises(ValueError):
        _parse_openai_video_seconds(seconds)


def test_frames_for_seconds_snaps_to_the_lattice():
    ltx = {"fps": 24, "frame_step": 8, "frame_offset": 1}
    assert _frames_for_seconds(5.0, ltx) == 121
    assert _frames_for_seconds(4.0, ltx) == 97
    assert _frames_for_seconds(0.01, ltx) == 9
    assert _frames_for_seconds(0.2, ltx) == 9
    h3 = {"fps": 24, "frame_step": 17, "frame_offset": 5}
    assert (_frames_for_seconds(5.0, h3) - 5) % 17 == 0


class _GatedBackend(_FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.gate = threading.Event()
        self.gate.set()
        self.fail_with: str | None = None
        self.last_generate_kwargs: dict = {}

    def generate(
        self,
        *,
        prompt,
        seed = None,
        cancel_event = None,
        **kwargs,
    ):
        self.last_generate_kwargs = {"prompt": prompt, "seed": seed, **kwargs}
        if self.fail_with is not None:
            raise ValueError(self.fail_with)
        while not self.gate.wait(timeout = 0.01):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError(VIDEO_CANCELLED_MSG)
        return super().generate(prompt = prompt, seed = seed, cancel_event = cancel_event, **kwargs)


@pytest.fixture(autouse = True)
def _healthy_diffusers(healthy_diffusers):
    pass


@pytest.fixture
def backend(monkeypatch, tmp_path):
    backend = _GatedBackend()
    backend.loaded = True
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setitem(gpu_arbiter._EVICTORS, gpu_arbiter.VIDEO, lambda: None)
    monkeypatch.setattr(gallery_module, "gallery_dir", lambda: tmp_path)
    monkeypatch.setattr(video_routes, "_jobs", {})
    return backend


@pytest.fixture
def client(backend):
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(openai_router, prefix = "/v1")
    app.include_router(openai_router, prefix = "/api/inference")
    app.include_router(video_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _multipart(fields: dict, files: dict | None = None) -> tuple[bytes, str]:
    boundary = "----unsloth-test-boundary"
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        )
    for name, (filename, data, content_type) in (files or {}).items():
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _create(
    client,
    fields: dict,
    files: dict | None = None,
):
    body, content_type = _multipart(fields, files)
    return client.post("/v1/videos", content = body, headers = {"Content-Type": content_type})


def _wait_terminal(
    client,
    video_id: str,
    timeout = 5.0,
) -> dict:
    deadline = time.monotonic() + timeout
    video: dict = {}
    while time.monotonic() < deadline:
        resp = client.get(f"/v1/videos/{video_id}")
        assert resp.status_code == 200, resp.json()
        video = resp.json()
        if video["status"] in ("completed", "failed"):
            return video
        time.sleep(0.01)
    raise AssertionError(f"job never reached a terminal state: {video}")


def _reference_image_bytes(fmt: str, size: tuple[int, int] = (8, 8)) -> bytes:
    """A real, decodable image. The route decodes the reference before any model
    switch, so a magic-byte stub is refused before it can reach begin_generate."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format = fmt)
    return buf.getvalue()


def _reference_data_url(fmt: str, mime: str) -> str:
    import base64
    return f"data:{mime};base64," + base64.b64encode(_reference_image_bytes(fmt)).decode("ascii")


def _save_clip(
    prompt: str,
    created_at: str,
    video_id: str | None = None,
    model: str = "unsloth/LTX-2.3-GGUF",
) -> dict:
    return gallery_module.save(
        b"OLD-MP4",
        {
            "prompt": prompt,
            "negative_prompt": None,
            "width": 768,
            "height": 512,
            "num_frames": 121,
            "fps": 24,
            "duration_s": 5.0,
            "steps": 40,
            "guidance": 4.0,
            "seed": 7,
            "has_audio": True,
            "model": model,
            "created_at": created_at,
        },
        video_id = video_id,
    )


def test_create_like_the_sdk_poll_download_list_delete(client, backend):
    resp = _create(
        client,
        {"prompt": "a red fox in snow", "model": "sora-2", "seconds": "4", "size": "768x512"},
    )
    assert resp.status_code == 200, resp.json()
    job = resp.json()
    assert job["object"] == "video"
    assert job["id"].startswith("video_")
    assert job["status"] in ("queued", "in_progress")
    assert job["progress"] < 100 and job["completed_at"] is None and job["error"] is None
    assert job["prompt"] == "a red fox in snow"
    assert job["size"] == "768x512"
    assert job["seconds"] == "4.04"
    assert job["model"] == "unsloth/LTX-2.3-GGUF"

    done = _wait_terminal(client, job["id"])
    assert done["status"] == "completed" and done["progress"] == 100
    assert done["completed_at"] >= done["created_at"] == job["created_at"]
    assert done["id"] == job["id"]
    assert backend.last_generate_kwargs["width"] == 768
    assert backend.last_generate_kwargs["height"] == 512
    assert backend.last_generate_kwargs["num_frames"] == 97
    # begin_generate resolves keyframes up front and hands the worker _resolved_inputs.
    assert backend.last_generate_kwargs["_resolved_inputs"].first_frame is None

    assert gallery_module.owned_video_path(job["id"]) is not None
    content = client.get(f"/v1/videos/{job['id']}/content")
    assert content.status_code == 200
    assert content.headers["content-type"].startswith("video/mp4")
    assert content.content == b"MP4-FAKE-BYTES"
    assert (
        client.get(f"/v1/videos/{job['id']}/content", params = {"variant": "video"}).status_code
        == 200
    )

    listing = client.get("/v1/videos").json()
    assert listing["object"] == "list"
    assert [v["id"] for v in listing["data"]] == [job["id"]]
    assert listing["first_id"] == listing["last_id"] == job["id"]
    assert listing["has_more"] is False

    deleted = client.delete(f"/v1/videos/{job['id']}")
    assert deleted.status_code == 200
    assert deleted.json() == {"id": job["id"], "object": "video.deleted", "deleted": True}
    assert gallery_module.owned_video_path(job["id"]) is None
    gone = client.get(f"/v1/videos/{job['id']}")
    assert gone.status_code == 404
    assert gone.json()["error"]["code"] == "video_not_found"
    assert client.get("/v1/videos").json()["data"] == []


def test_json_body_and_the_studio_mount_work_too(client, backend):
    resp = client.post("/api/inference/videos", json = {"prompt": "a cat", "seconds": 5})
    assert resp.status_code == 200, resp.json()
    job = resp.json()
    assert job["seconds"] == "5.04" and job["size"] == "768x512"
    done = _wait_terminal(client, job["id"])
    assert done["status"] == "completed"
    assert backend.last_generate_kwargs["num_frames"] == 121
    assert backend.last_generate_kwargs["width"] is None


def test_omitted_seconds_and_size_use_the_family_defaults(client, backend):
    resp = client.post("/v1/videos", json = {"prompt": "a cat"})
    assert resp.status_code == 200, resp.json()
    assert resp.json()["seconds"] == "5.04"
    _wait_terminal(client, resp.json()["id"])
    assert backend.last_generate_kwargs["num_frames"] is None


def test_in_progress_job_reports_progress_and_the_poll_hint(client, backend):
    backend.gate.clear()
    job = _create(client, {"prompt": "slow"}).json()
    resp = client.get(f"/v1/videos/{job['id']}")
    assert resp.status_code == 200
    assert resp.json()["status"] in ("queued", "in_progress")
    assert resp.headers["openai-poll-after-ms"] == "2000"
    content = client.get(f"/v1/videos/{job['id']}/content")
    assert content.status_code == 400
    assert content.json()["error"]["code"] == "video_not_ready"
    busy = _create(client, {"prompt": "another"})
    assert busy.status_code == 409
    assert "error" in busy.json()
    backend.gate.set()
    done = _wait_terminal(client, job["id"])
    assert done["status"] == "completed"
    assert "openai-poll-after-ms" not in client.get(f"/v1/videos/{job['id']}").headers


def test_a_job_outcome_survives_the_next_job(client, backend):
    first = _create(client, {"prompt": "first"}).json()
    assert _wait_terminal(client, first["id"])["status"] == "completed"
    backend.gate.clear()
    second = _create(client, {"prompt": "second"}).json()
    assert client.get(f"/v1/videos/{first['id']}").json()["status"] == "completed"
    assert client.get(f"/v1/videos/{second['id']}").json()["status"] in ("queued", "in_progress")
    ids = [v["id"] for v in client.get("/v1/videos").json()["data"]]
    assert set(ids) == {first["id"], second["id"]}
    backend.gate.set()
    assert _wait_terminal(client, second["id"])["status"] == "completed"


def test_a_failed_job_reports_its_error_and_keeps_it_after_the_next_job(client, backend):
    backend.fail_with = "Prompt rejected by the model."
    job = _create(client, {"prompt": "bad"}).json()
    failed = _wait_terminal(client, job["id"])
    assert failed["status"] == "failed" and failed["progress"] == 0
    assert failed["error"] == {
        "code": "video_generation_failed",
        "message": "Prompt rejected by the model.",
    }
    content = client.get(f"/v1/videos/{job['id']}/content")
    assert content.status_code == 400
    assert content.json()["error"]["code"] == "video_generation_failed"
    assert gallery_module.owned_video_path(job["id"]) is None
    assert [v["status"] for v in client.get("/v1/videos").json()["data"]] == ["failed"]

    # Polling can cross a Studio restart. Replacing the process-local cache must rehydrate
    # the terminal error instead of turning the ID into a 404.
    video_routes._jobs = {}
    reloaded = client.get(f"/v1/videos/{job['id']}")
    assert reloaded.status_code == 200
    assert reloaded.json()["status"] == "failed" and reloaded.json()["error"] == failed["error"]

    backend.fail_with = None
    later = _create(client, {"prompt": "good"}).json()
    assert _wait_terminal(client, later["id"])["status"] == "completed"
    assert client.get(f"/v1/videos/{job['id']}").json()["status"] == "failed"
    assert client.delete(f"/v1/videos/{job['id']}").json()["deleted"] is True
    assert gallery_module.get_job(job["id"]) is None
    assert client.get(f"/v1/videos/{job['id']}").status_code == 404


def test_an_unpolled_failure_keeps_its_error_after_the_next_job(client, backend):
    backend.fail_with = "Prompt rejected before polling."
    failed_job = _create(client, {"prompt": "bad"}).json()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if backend.generate_progress().get("phase") == "failed":
            break
        time.sleep(0.01)
    else:
        raise AssertionError("the first generation never failed")

    backend.fail_with = None
    later = _create(client, {"prompt": "good"})
    assert later.status_code == 200, later.json()
    failed = client.get(f"/v1/videos/{failed_job['id']}").json()
    assert failed["status"] == "failed"
    assert failed["error"]["message"] == "Prompt rejected before polling."


def test_unpolled_terminal_jobs_still_obey_the_memory_cap(client, backend, monkeypatch):
    monkeypatch.setattr(video_routes, "_MAX_REMEMBERED_JOBS", 2)
    ids = []
    for prompt in ("first", "second", "third"):
        job = _create(client, {"prompt": prompt}).json()
        ids.append(job["id"])
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if backend.generate_progress().get("phase") == "completed":
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"{prompt} never completed")
    assert len(video_routes._jobs) == 2
    assert ids[0] not in video_routes._jobs
    assert gallery_module.get_job(ids[0]) is not None


def test_listing_keeps_persisted_failures_beyond_the_memory_cap(client, backend, monkeypatch):
    monkeypatch.setattr(video_routes, "_MAX_REMEMBERED_JOBS", 2)
    ids = []
    for index in range(3):
        job = video_routes._VideoJob(
            id = f"video_failed_{index}",
            created_at = index + 1,
            prompt = f"failed {index}",
            model = "model",
            size = "768x512",
            seconds = "5",
            status = "failed",
            completed_at = index + 1,
            error = {"code": "video_generation_failed", "message": f"failure {index}"},
        )
        gallery_module.save_job(job.id, vars(job))
        ids.append(job.id)

    video_routes._jobs = {}
    listed = client.get("/v1/videos").json()
    assert [video["id"] for video in listed["data"]] == list(reversed(ids))
    assert listed["has_more"] is False


def test_a_worker_outcome_that_arrives_before_the_job_record_is_not_overwritten(client, backend):
    video_id = "video_worker_won_the_race"
    gallery_module.record_job_outcome(
        video_id, completed_at = 123, error = "Failed before the create response."
    )
    video_routes._remember_job(
        video_routes._VideoJob(
            id = video_id,
            created_at = 100,
            prompt = "fast failure",
            model = "model",
            size = "768x512",
            seconds = "5",
        )
    )
    video_routes._jobs = {}
    failed = client.get(f"/v1/videos/{video_id}").json()
    assert failed["status"] == "failed"
    assert failed["completed_at"] == 123
    assert failed["error"]["message"] == "Failed before the create response."


def test_an_overflowing_worker_timestamp_is_ignored():
    job = video_routes._VideoJob(
        id = "video_overflowing_timestamp",
        created_at = 100,
        prompt = "bad persisted timestamp",
        model = "model",
        size = "768x512",
        seconds = "5",
    )
    record = {**vars(job), "_worker_outcome": {"completed_at": float("inf"), "error": None}}

    assert video_routes._job_from_record(record) is None


def test_completed_job_keeps_its_submission_time_after_restart(client, backend, monkeypatch):
    monkeypatch.setattr(video_routes._time, "time", lambda: 100)
    job = _create(client, {"prompt": "slow timestamp"}).json()
    assert _wait_terminal(client, job["id"])["created_at"] == 100
    video_routes._jobs = {}
    restarted = client.get(f"/v1/videos/{job['id']}").json()
    assert restarted["status"] == "completed"
    assert restarted["created_at"] == job["created_at"] == 100


def test_studio_delete_and_clear_remove_unpolled_openai_jobs(client, backend):
    first = _create(client, {"prompt": "delete in Studio"}).json()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if backend.generate_progress().get("phase") == "completed":
            break
        time.sleep(0.01)
    assert client.delete(f"/api/inference/video/gallery/{first['id']}").status_code == 200
    assert gallery_module.get_job(first["id"]) is None
    assert client.get(f"/v1/videos/{first['id']}").status_code == 404

    second = _create(client, {"prompt": "clear in Studio"}).json()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if backend.generate_progress().get("phase") == "completed":
            break
        time.sleep(0.01)
    cleared = client.delete("/api/inference/video/gallery")
    assert cleared.status_code == 200 and cleared.json()["removed"] == 1
    assert gallery_module.get_job(second["id"]) is None
    assert client.get(f"/v1/videos/{second['id']}").status_code == 404


def test_deleting_a_running_job_cancels_it(client, backend):
    backend.gate.clear()
    job = _create(client, {"prompt": "slow"}).json()
    assert client.get(f"/v1/videos/{job['id']}").json()["status"] in ("queued", "in_progress")
    resp = client.delete(f"/v1/videos/{job['id']}")
    assert resp.status_code == 200 and resp.json()["deleted"] is True
    assert client.get(f"/v1/videos/{job['id']}").status_code == 404
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        progress = client.get("/api/inference/video/generate-progress").json()
        if progress.get("phase") in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert progress["phase"] == "failed" and progress["error"] == VIDEO_CANCELLED_MSG
    assert gallery_module.owned_video_path(job["id"]) is None
    backend.gate.set()


def test_cancelled_content_keeps_the_cancelled_error_code(client, backend):
    backend.gate.clear()
    job = _create(client, {"prompt": "cancel through Studio"}).json()
    cancelled = client.post("/api/inference/video/generate/cancel")
    assert cancelled.status_code == 200 and cancelled.json()["cancelled"] is True
    failed = _wait_terminal(client, job["id"])
    assert failed["error"]["code"] == "cancelled"
    content = client.get(f"/v1/videos/{job['id']}/content")
    assert content.status_code == 400
    assert content.json()["error"]["code"] == "cancelled"
    backend.gate.set()


def test_deleting_a_stale_job_does_not_cancel_its_successor(client, backend, monkeypatch):
    successor_cancel = threading.Event()
    with backend._lock:
        backend._gen_video_id = "video_successor"
        backend._active_generate_cancel = successor_cancel
    monkeypatch.setattr(
        video_routes,
        "_lookup_video",
        lambda video_id: type("Video", (), {"status": "queued"})()
        if video_id == "video_target"
        else None,
    )

    resp = client.delete("/v1/videos/video_target")
    assert resp.status_code == 200, resp.json()
    assert successor_cancel.is_set() is False


def test_clips_made_before_this_process_list_and_retrieve_as_completed(client, backend):
    older = _save_clip("older", "2026-01-01T00:00:00Z")
    newer = _save_clip("newer", "2026-01-02T00:00:00Z")
    resp = client.get(f"/v1/videos/{older['id']}")
    assert resp.status_code == 200
    video = resp.json()
    assert video["status"] == "completed" and video["progress"] == 100
    assert video["created_at"] == video["completed_at"] == 1767225600
    assert video["size"] == "768x512" and video["seconds"] == "5" and video["prompt"] == "older"
    assert video["model"] == "unsloth/LTX-2.3-GGUF"

    page = client.get("/v1/videos", params = {"limit": 1}).json()
    assert [v["id"] for v in page["data"]] == [newer["id"]]
    assert page["has_more"] is True and page["last_id"] == newer["id"]
    rest = client.get("/v1/videos", params = {"limit": 1, "after": page["last_id"]}).json()
    assert [v["id"] for v in rest["data"]] == [older["id"]]
    assert rest["has_more"] is False
    ascending = client.get("/v1/videos", params = {"order": "asc"}).json()
    assert [v["id"] for v in ascending["data"]] == [older["id"], newer["id"]]
    unknown = client.get("/v1/videos", params = {"after": "nope"})
    assert unknown.status_code == 400 and unknown.json()["error"]["param"] == "after"

    assert client.get(f"/v1/videos/{older['id']}/content").content == b"OLD-MP4"


def test_archived_clips_remain_in_the_openai_listing(client, backend):
    clip = _save_clip("archived", "2026-01-01T00:00:00Z")
    archived = client.patch(f"/api/inference/video/gallery/{clip['id']}", json = {"archived": True})
    assert archived.status_code == 200, archived.json()
    listed = client.get("/v1/videos").json()["data"]
    assert [video["id"] for video in listed] == [clip["id"]]
    assert client.get(f"/v1/videos/{clip['id']}").status_code == 200


def test_completed_delete_does_not_confirm_when_the_clip_remains(client, backend, monkeypatch):
    clip = _save_clip("kept", "2026-01-01T00:00:00Z")
    monkeypatch.setattr(gallery_module, "delete", lambda _video_id: False)

    resp = client.delete(f"/v1/videos/{clip['id']}")
    assert resp.status_code == 500
    assert gallery_module.get_record(clip["id"]) is not None
    assert client.get(f"/v1/videos/{clip['id']}").status_code == 200


def test_missing_prompt_is_a_400_naming_the_param(client, backend):
    resp = client.post("/v1/videos", json = {})
    assert resp.status_code == 400
    assert resp.json()["error"]["param"] == "prompt"
    resp = _create(client, {"model": "sora-2"})
    assert resp.status_code == 400 and resp.json()["error"]["param"] == "prompt"
    resp = client.post(
        "/v1/videos", content = b"not json", headers = {"Content-Type": "application/json"}
    )
    assert resp.status_code == 400 and "error" in resp.json()


@pytest.mark.parametrize(
    "fields, param",
    [
        ({"prompt": "x", "size": "abc"}, "size"),
        ({"prompt": "x", "size": "640x480"}, "size"),
        ({"prompt": "x", "seconds": "abc"}, "seconds"),
        ({"prompt": "x", "seconds": "0"}, "seconds"),
    ],
)
def test_unservable_shapes_are_400s_naming_the_param(client, backend, fields, param):
    resp = _create(client, fields)
    assert resp.status_code == 400, resp.json()
    assert resp.json()["error"]["param"] == param
    assert client.get("/v1/videos").json()["data"] == []


def test_not_loaded_is_a_503_in_the_openai_envelope(client, backend):
    backend.loaded = False
    resp = _create(client, {"prompt": "x"})
    assert resp.status_code == 503
    assert "No video model loaded" in resp.json()["error"]["message"]


def test_unknown_ids_are_404s(client, backend):
    for path in ("/v1/videos/video_nope", "/v1/videos/video_nope/content"):
        resp = client.get(path)
        assert resp.status_code == 404, path
        assert resp.json()["error"]["code"] == "video_not_found"
    assert client.delete("/v1/videos/video_nope").status_code == 404
    assert client.get("/v1/videos/../etc/passwd/content").status_code in (404, 400)


def test_thumbnail_variant_returns_webp(client, backend, monkeypatch):
    clip = _save_clip("x", "2026-01-01T00:00:00Z")
    thumbnail = b"RIFF\x04\x00\x00\x00WEBP"
    monkeypatch.setattr(gallery_module, "thumbnail", lambda video_id: thumbnail)
    resp = client.get(f"/v1/videos/{clip['id']}/content", params = {"variant": "thumbnail"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/webp")
    assert resp.headers["content-disposition"] == f'attachment; filename="{clip["id"]}.webp"'
    assert resp.content == thumbnail


def test_unsupported_content_variant_names_the_param(client, backend):
    clip = _save_clip("x", "2026-01-01T00:00:00Z")
    resp = client.get(f"/v1/videos/{clip['id']}/content", params = {"variant": "spritesheet"})
    assert resp.status_code == 400 and resp.json()["error"]["param"] == "variant"


def test_input_reference_upload_reaches_the_backend_unclassified(client, backend, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(backend, "begin_generate", lambda **kwargs: captured.update(kwargs))
    png = _reference_image_bytes("PNG")
    resp = _create(
        client, {"prompt": "animate this"}, {"input_reference": ("frame.png", png, "image/png")}
    )
    assert resp.status_code == 200, resp.json()
    assert captured["input_reference"].startswith("data:image/png;base64,")
    assert captured["video_id"] == resp.json()["id"]
    jpeg_url = _reference_data_url("JPEG", "image/jpeg")
    resp = _create(client, {"prompt": "x", "input_reference[image_url]": jpeg_url})
    assert resp.status_code == 200 and captured["input_reference"] == jpeg_url
    png_url = _reference_data_url("PNG", "image/png")
    resp = client.post(
        "/v1/videos",
        json = {"prompt": "x", "input_reference": {"image_url": png_url}},
    )
    assert resp.status_code == 200 and captured["input_reference"] == png_url


def test_input_reference_refusals(client, backend):
    png = b"\x89PNG\r\n\x1a\nfake"
    resp = _create(client, {"prompt": "x"}, {"input_reference": ("frame.png", png, "image/png")})
    assert resp.status_code == 400, resp.json()
    assert resp.json()["error"]["param"] == "input_reference"
    resp = client.post(
        "/v1/videos",
        json = {"prompt": "x", "input_reference": {"image_url": "https://example.com/a.png"}},
    )
    assert resp.status_code == 400 and "not fetched" in resp.json()["error"]["message"]
    resp = client.post(
        "/v1/videos", json = {"prompt": "x", "input_reference": {"file_id": "file_123"}}
    )
    assert resp.status_code == 400 and resp.json()["error"]["param"] == "input_reference"
    resp = _create(client, {"prompt": "x", "input_reference[file_id]": "file_123"})
    assert resp.status_code == 400 and resp.json()["error"]["param"] == "input_reference"
    resp = _create(client, {"prompt": "x"}, {"input_reference": ("a.txt", b"hello", "text/plain")})
    assert resp.status_code == 400 and "must be an image" in resp.json()["error"]["message"]
    assert client.get("/v1/videos").json()["data"] == []


def test_octet_stream_uploads_are_sniffed(client, backend, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(backend, "begin_generate", lambda **kwargs: captured.update(kwargs))
    jpeg = _reference_image_bytes("JPEG")
    resp = _create(
        client,
        {"prompt": "x"},
        {"input_reference": ("frame.bin", jpeg, "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.json()
    assert captured["input_reference"].startswith("data:image/jpeg;base64,")
    webp = _reference_image_bytes("WEBP")
    resp = _create(
        client, {"prompt": "x"}, {"input_reference": ("frame", webp, "application/octet-stream")}
    )
    assert resp.status_code == 200 and captured["input_reference"].startswith(
        "data:image/webp;base64,"
    )
    resp = _create(
        client,
        {"prompt": "x"},
        {"input_reference": ("blob.bin", b"not an image", "application/octet-stream")},
    )
    assert resp.status_code == 400 and resp.json()["error"]["param"] == "input_reference"


def test_model_never_leaks_a_host_path(client, backend, monkeypatch):
    monkeypatch.setattr(
        backend,
        "status",
        lambda: {**_FakeBackend.status(backend), "repo_id": "/srv/models/ltx-2.3-Q4_K_M.gguf"},
    )
    job = _create(client, {"prompt": "x"}).json()
    assert job["model"] == "ltx-2.3-Q4_K_M"
    clip = _save_clip("old", "2026-01-01T00:00:00Z", model = "/srv/models/ltx-2.3-Q4_K_M.gguf")
    assert client.get(f"/v1/videos/{clip['id']}").json()["model"] == "ltx-2.3-Q4_K_M"
    assert "/srv/" not in client.get("/v1/videos").text


def test_create_hands_the_model_to_auto_switch(client, backend, monkeypatch):
    import core.inference.media_auto_switch as mas

    calls: list = []

    async def _record(model, **kwargs):
        calls.append(
            (
                model,
                kwargs["owner"],
                kwargs["openai_errors"],
                kwargs["hf_token"],
                kwargs["before_switch"] is not None,
            )
        )

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _record)
    resp = client.post(
        "/v1/videos",
        json = {"prompt": "x", "model": "unsloth/LTX-2.3-GGUF"},
        headers = {"X-Unsloth-HF-Token": "hf_abc"},
    )
    assert resp.status_code == 200, resp.json()
    assert calls == [("unsloth/LTX-2.3-GGUF", gpu_arbiter.VIDEO, True, "hf_abc", True)]


@pytest.mark.parametrize(
    "created_at, expected",
    [
        ("2026-01-01T00:00:00Z", 1767225600),
        ("2026-01-01T00:00:00+00:00", 1767225600),
        ("2026-01-01T01:00:00+01:00", 1767225600),
        ("2026-01-01T00:00:00", 1767225600),
        (1767225600.7, 1767225600),
        ("garbage", 0),
        (None, 0),
    ],
)
def test_record_epoch_accepts_old_and_new_sidecar_timestamps(created_at, expected):
    assert video_routes._record_epoch({"created_at": created_at}) == expected


def test_an_unservable_duration_is_refused_before_the_model_switch(client, backend, monkeypatch):
    """The preflight judged size but passed None for the frame count.

    A duration no family can serve was therefore only refused inside begin_generate --
    after the resident pipeline had been evicted and the target model fully loaded.
    """
    import types

    from core.inference import media_auto_switch

    switched = {"completed": False}

    async def _fake_switch(
        model,
        *,
        before_switch = None,
        **_kwargs,
    ):
        pick = types.SimpleNamespace(model_path = "Lightricks/LTX-2", gguf_filename = None)
        if before_switch is not None:
            before_switch(pick)
        switched["completed"] = True

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _fake_switch)

    # 120s is inside the route's own bound but ~2880 frames, over the 1024 global ceiling.
    resp = _create(client, {"prompt": "x", "model": "Lightricks/LTX-2", "seconds": "120"})
    assert resp.status_code == 400, resp.json()
    assert resp.json()["error"]["param"] == "seconds"
    assert switched["completed"] is False, "the model was switched before the refusal"


def test_a_reference_only_checkpoint_is_refused_before_the_model_switch(
    client, backend, monkeypatch
):
    import types

    from core.inference import media_auto_switch

    switched = {"completed": False}

    async def _fake_switch(
        model,
        *,
        before_switch = None,
        **_kwargs,
    ):
        pick = types.SimpleNamespace(
            model_path = "MiniMaxAI/MiniMax-H3",
            gguf_filename = "minimax_h3_ref2va-Q4_K_M.gguf",
            model_kind = "gguf",
        )
        if before_switch is not None:
            before_switch(pick)
        switched["completed"] = True

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _fake_switch)
    resp = _create(client, {"prompt": "text only", "model": "MiniMax-H3-Ref2VA"})
    assert resp.status_code == 400, resp.json()
    assert resp.json()["error"]["param"] == "input_reference"
    assert "Ref2VA partition" in resp.json()["error"]["message"]
    assert switched["completed"] is False, "the model was switched before the refusal"


@pytest.mark.parametrize("reference_size", [(8, 8), (5000, 1000)])
def test_ref2va_routes_the_input_image_as_a_reference(client, backend, monkeypatch, reference_size):
    import types

    from core.inference import media_auto_switch
    from core.inference.video_minimax_h3 import H3_TASK_REFERENCES

    switched = {"completed": False}
    captured = {}

    async def _fake_switch(
        model,
        *,
        before_switch = None,
        **_kwargs,
    ):
        pick = types.SimpleNamespace(
            model_path = "MiniMaxAI/MiniMax-H3",
            gguf_filename = "minimax_h3_ref2va-Q4_K_M.gguf",
            model_kind = "gguf",
        )
        if before_switch is not None:
            before_switch(pick)
        switched["completed"] = True

    def _status():
        return {
            **_FakeBackend.status(backend),
            "repo_id": "MiniMaxAI/MiniMax-H3",
            "h3_task": H3_TASK_REFERENCES,
        }

    def _begin(**kwargs):
        captured.update(kwargs)
        return {"width": 768, "height": 512}

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _fake_switch)
    monkeypatch.setattr(backend, "status", _status)
    monkeypatch.setattr(backend, "begin_generate", _begin)

    resp = _create(
        client,
        {"prompt": "follow this subject", "model": "MiniMax-H3-Ref2VA"},
        {
            "input_reference": (
                "subject.png",
                _reference_image_bytes("PNG", reference_size),
                "image/png",
            )
        },
    )
    assert resp.status_code == 200, resp.json()
    assert switched["completed"] is True
    assert captured["input_reference"].startswith("data:image/png;base64,")
    assert "first_frame" not in captured
    assert "reference_images" not in captured


def test_reference_conditioning_follows_the_state_reserved_by_begin_generate(
    client, backend, monkeypatch
):
    import types

    from core.inference.video_families import detect_video_family
    from core.inference.video_minimax_h3 import H3_TASK_REFERENCES

    real_begin = backend.begin_generate

    def _racing_begin(**kwargs):
        with backend._lock:
            backend._state = types.SimpleNamespace(
                **{
                    **vars(backend._state),
                    "family": detect_video_family("MiniMaxAI/MiniMax-H3", None),
                    "h3_task": H3_TASK_REFERENCES,
                    "repo_id": "MiniMaxAI/MiniMax-H3",
                }
            )
        return real_begin(**kwargs)

    monkeypatch.setattr(backend, "begin_generate", _racing_begin)
    response = _create(
        client,
        {"prompt": "follow the newly reserved partition"},
        {"input_reference": ("subject.png", _reference_image_bytes("PNG"), "image/png")},
    )

    assert response.status_code == 200, response.json()
    deadline = time.monotonic() + 5.0
    while "_resolved_inputs" not in backend.last_generate_kwargs and time.monotonic() < deadline:
        time.sleep(0.01)
    resolved = backend.last_generate_kwargs["_resolved_inputs"]
    assert resolved.first_frame is None
    assert len(resolved.references.images) == 1


def test_the_created_job_reports_the_canvas_the_backend_resolved(client, backend, monkeypatch):
    """With a reference image and no size the canvas follows the source aspect.

    The route used to record the family's first resolution preset regardless, so the
    create response advertised one size while the clip rendered at another, and the
    same job's size changed once it was read back from the gallery.
    """
    real = backend.begin_generate

    def _begin(**kwargs):
        real(**kwargs)
        return {"width": 704, "height": 1216}

    monkeypatch.setattr(backend, "begin_generate", _begin)
    job = _create(client, {"prompt": "a tall portrait"}).json()
    assert job["size"] == "704x1216"


def test_deleting_a_job_that_finishes_mid_cancel_removes_the_clip(client, backend, monkeypatch):
    """Cancellation losing the race is not proof that nothing was written.

    The handler used to cancel and return deleted:true without ever deleting the pair,
    so a clip persisted in that window came straight back through retrieve and list.
    """
    backend.gate.clear()
    job = _create(client, {"prompt": "racy"}).json()
    assert client.get(f"/v1/videos/{job['id']}").json()["status"] in ("queued", "in_progress")

    def _cancel_after_it_finishes(expected_video_id):
        assert expected_video_id == job["id"]
        backend.gate.set()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if gallery_module.owned_video_path(job["id"]) is not None:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("the clip was never persisted")
        return False

    monkeypatch.setattr(backend, "cancel_generate", _cancel_after_it_finishes)

    resp = client.delete(f"/v1/videos/{job['id']}")
    assert resp.status_code == 200, resp.json()
    assert resp.json()["deleted"] is True
    assert gallery_module.owned_video_path(job["id"]) is None
    assert client.get(f"/v1/videos/{job['id']}").status_code == 404
    assert [v["id"] for v in client.get("/v1/videos").json()["data"]] == []


def test_a_stale_poller_cannot_recreate_a_deleted_job(client, backend, monkeypatch):
    video_id = "video_delete_poll_race"
    job = video_routes._VideoJob(
        id = video_id,
        created_at = 1,
        prompt = "racy poll",
        model = "model",
        size = "768x512",
        seconds = "5",
    )
    video_routes._jobs[video_id] = job
    gallery_module.save_job(video_id, vars(job))

    save_started = threading.Event()
    release_save = threading.Event()
    forget_started = threading.Event()
    delete_done = threading.Event()
    real_save = gallery_module.save_job
    real_forget = gallery_module.forget_job

    def _delayed_save(saved_id, record):
        save_started.set()
        assert release_save.wait(timeout = 5.0)
        real_save(saved_id, record)

    def _observed_forget(forgotten_id):
        result = real_forget(forgotten_id)
        forget_started.set()
        return result

    monkeypatch.setattr(gallery_module, "save_job", _delayed_save)
    monkeypatch.setattr(gallery_module, "forget_job", _observed_forget)
    monkeypatch.setattr(
        video_routes,
        "_lookup_video",
        lambda requested_id: video_routes._job_to_openai(job) if requested_id == video_id else None,
    )
    monkeypatch.setattr(video_routes, "_await_generate_settled", lambda _video_id: True)

    poller = threading.Thread(target = video_routes._sync_jobs)
    poller.start()
    assert save_started.wait(timeout = 5.0)

    responses = []

    def _delete():
        responses.append(client.delete(f"/v1/videos/{video_id}"))
        delete_done.set()

    deleter = threading.Thread(target = _delete)
    deleter.start()
    if forget_started.wait(timeout = 0.5):
        assert delete_done.wait(timeout = 5.0)
    release_save.set()
    poller.join(timeout = 5.0)
    deleter.join(timeout = 5.0)

    assert not poller.is_alive() and not deleter.is_alive()
    assert responses[0].status_code == 200, responses[0].json()
    assert gallery_module.get_job(video_id) is None
    assert video_id not in video_routes._jobs


def test_terminal_hydration_cannot_restore_a_deleted_job(client, backend, monkeypatch):
    video_id = "video_delete_hydration_race"
    job = video_routes._VideoJob(
        id = video_id,
        created_at = 1,
        prompt = "racy hydration",
        model = "model",
        size = "768x512",
        seconds = "5",
    )
    video_routes._jobs[video_id] = job
    gallery_module.save_job(
        video_id,
        {
            **vars(job),
            "status": "failed",
            "error": {"code": "video_generation_failed", "message": "failed"},
        },
    )

    terminal_read = threading.Event()
    release_read = threading.Event()
    real_get = gallery_module.get_job

    def _delayed_get(requested_id):
        record = real_get(requested_id)
        terminal_read.set()
        assert release_read.wait(timeout = 5.0)
        return record

    monkeypatch.setattr(gallery_module, "get_job", _delayed_get)
    monkeypatch.setattr(
        video_routes,
        "_lookup_video",
        lambda requested_id: video_routes._job_to_openai(job) if requested_id == video_id else None,
    )
    monkeypatch.setattr(video_routes, "_await_generate_settled", lambda _video_id: True)

    poller = threading.Thread(target = video_routes._sync_jobs)
    poller.start()
    assert terminal_read.wait(timeout = 5.0)
    deleted = client.delete(f"/v1/videos/{video_id}")
    assert deleted.status_code == 200, deleted.json()
    release_read.set()
    poller.join(timeout = 5.0)

    assert not poller.is_alive()
    assert real_get(video_id) is None
    assert video_id not in video_routes._jobs


def test_concurrent_pollers_cannot_move_progress_backward(backend, monkeypatch):
    video_id = "video_monotonic_progress"
    job = video_routes._VideoJob(
        id = video_id,
        created_at = 1,
        prompt = "concurrent polls",
        model = "model",
        size = "768x512",
        seconds = "5",
        status = "in_progress",
        progress = 10,
    )
    video_routes._jobs[video_id] = job
    gallery_module.save_job(video_id, vars(job))

    older_waiting = threading.Event()
    release_older = threading.Event()
    real_get = gallery_module.get_job

    def _delayed_get(requested_id):
        if threading.current_thread().name == "older-video-poll":
            older_waiting.set()
            assert release_older.wait(timeout = 5.0)
        return real_get(requested_id)

    def _progress():
        fraction = 0.2 if threading.current_thread().name == "older-video-poll" else 0.3
        return {
            "video_id": video_id,
            "phase": "denoise",
            "active": True,
            "fraction": fraction,
        }

    monkeypatch.setattr(gallery_module, "get_job", _delayed_get)
    monkeypatch.setattr(backend, "generate_progress", _progress)
    older = threading.Thread(target = video_routes._sync_jobs, name = "older-video-poll")
    older.start()
    assert older_waiting.wait(timeout = 5.0)
    video_routes._sync_jobs()
    release_older.set()
    older.join(timeout = 5.0)

    assert not older.is_alive()
    assert job.status == "in_progress"
    assert job.progress == 30
    assert gallery_module.get_job(video_id)["progress"] == 30


def test_an_undecodable_reference_is_refused_before_the_model_switch(client, backend, monkeypatch):
    """A well-formed content type is not a readable image.

    _resolve_keyframes only decodes inside begin_generate, which runs after the auto
    switch, so bad bytes used to evict the resident pipeline and load the target first.
    """
    import types

    from core.inference import media_auto_switch

    switched = {"completed": False}

    async def _fake_switch(
        model,
        *,
        before_switch = None,
        **_kwargs,
    ):
        pick = types.SimpleNamespace(model_path = "Lightricks/LTX-2", gguf_filename = None)
        if before_switch is not None:
            before_switch(pick)
        switched["completed"] = True

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _fake_switch)

    resp = _create(
        client,
        {"prompt": "x", "model": "Lightricks/LTX-2"},
        {"input_reference": ("frame.png", b"\x89PNG\r\n\x1a\nnot-an-image", "image/png")},
    )
    assert resp.status_code == 400, resp.json()
    assert resp.json()["error"]["param"] == "input_reference"
    # The message matters: without the eager decode this path still 400s, but only via
    # the keyframe-conditioning check, leaving real decode failures to run post-switch.
    assert "not a readable image" in resp.json()["error"]["message"], resp.json()
    assert switched["completed"] is False, "the model was switched before the refusal"


def test_a_delete_that_cannot_observe_the_settle_does_not_confirm(client, backend, monkeypatch):
    """Returning deleted:true on a timed-out wait let the worker commit afterwards.

    The clip then reappeared through retrieve/list, so the caller was told a deletion
    happened that had not. Report 409 instead and let them retry.
    """
    monkeypatch.setattr(video_routes, "_DELETE_SETTLE_TIMEOUT_S", 0.05)
    backend.gate.clear()
    try:
        job = _create(client, {"prompt": "still writing"}).json()
        assert client.get(f"/v1/videos/{job['id']}").json()["status"] in (
            "queued",
            "in_progress",
        )
        monkeypatch.setattr(backend, "cancel_generate", lambda expected_video_id: False)
        resp = client.delete(f"/v1/videos/{job['id']}")
        assert resp.status_code == 409, resp.json()
        assert resp.json()["error"]["code"] == "video_not_ready"
    finally:
        backend.gate.set()


def test_the_create_call_opens_an_api_monitor_row(client, backend, monkeypatch):
    """The OpenAI image and audio routes open a row; videos never did.

    Without it the newly supported API is invisible to the API Monitor and its usage
    receipts, so per-subject history silently undercounts video generation.
    """
    from core.inference.api_monitor import api_monitor

    started: list[dict] = []
    relabeled: list[str] = []
    real_start = api_monitor.start
    real_relabel = api_monitor.relabel

    def _spy(**kwargs):
        started.append(kwargs)
        return real_start(**kwargs)

    def _relabel(entry_id, model):
        relabeled.append(model)
        return real_relabel(entry_id, model)

    monkeypatch.setattr(api_monitor, "start", _spy)
    monkeypatch.setattr(api_monitor, "relabel", _relabel)
    resp = _create(client, {"prompt": "a monitored clip"})
    assert resp.status_code == 200, resp.json()
    rows = [r for r in started if r.get("endpoint") == "/v1/videos"]
    assert rows, started
    assert rows[0]["prompt"] == "a monitored clip"
    assert rows[0]["method"] == "POST"
    assert relabeled == [resp.json()["model"]]


def test_the_job_describes_the_run_the_backend_reserved(client, backend, monkeypatch):
    """A load committing between status() and the reservation swaps the family.

    The job used to be described from the earlier snapshot, so it advertised the model
    that had just been replaced and a duration computed against that model's fps.
    """
    real = backend.begin_generate

    def _begin(**kwargs):
        real(**kwargs)
        # What the reservation actually committed to, different from the snapshot.
        return {
            "width": 704,
            "height": 1216,
            "num_frames": 121,
            "fps": 24,
            "model": "unsloth/Some-Other-Video-Model",
        }

    monkeypatch.setattr(backend, "begin_generate", _begin)
    job = _create(client, {"prompt": "swapped underneath", "seconds": "4"}).json()
    assert job["model"] == "unsloth/Some-Other-Video-Model"
    assert job["size"] == "704x1216"
    assert job["seconds"] == _format_seconds(121 / 24)


@pytest.mark.parametrize("swap_before_snapshot", [True, False])
def test_requested_model_cannot_change_before_the_generation_reservation(
    client, backend, monkeypatch, swap_before_snapshot
):
    import types

    from core.inference import media_auto_switch
    from utils import openai_auto_switch_settings

    requested = backend.status()["repo_id"]

    def _swap_model():
        with backend._lock:
            backend._state = types.SimpleNamespace(
                **{
                    **vars(backend._state),
                    "repo_id": "unsloth/Replacement-Video-Model",
                }
            )

    async def _resident_fast_path(*_args, **_kwargs):
        if swap_before_snapshot:
            _swap_model()
        return None

    real_begin = backend.begin_generate

    def _racing_begin(**kwargs):
        if not swap_before_snapshot:
            _swap_model()
        return real_begin(**kwargs)

    monkeypatch.setattr(media_auto_switch, "maybe_auto_switch_media_model", _resident_fast_path)
    monkeypatch.setattr(openai_auto_switch_settings, "get_media_auto_switch_enabled", lambda: True)
    monkeypatch.setattr(backend, "begin_generate", _racing_begin)

    response = _create(client, {"prompt": "keep the requested model", "model": requested})

    assert response.status_code == 409, response.json()
    assert response.json()["error"]["code"] == "model_changed"


def test_requested_duration_is_recomputed_for_the_reserved_family(client, backend, monkeypatch):
    real_status = backend.status

    def _stale_status():
        status = real_status()
        status["defaults"] = {
            **status["defaults"],
            "fps": 16,
            "frame_step": 4,
            "frame_offset": 1,
        }
        return status

    monkeypatch.setattr(backend, "status", _stale_status)
    job = _create(client, {"prompt": "five seconds", "seconds": "5"}).json()
    assert backend.last_generate_kwargs["num_frames"] == 121
    assert job["seconds"] == _format_seconds(121 / 24)
