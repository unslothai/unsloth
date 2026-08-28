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
from routes.video import _frames_for_seconds, _parse_openai_video_seconds, _parse_openai_video_size
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

    backend.fail_with = None
    later = _create(client, {"prompt": "good"}).json()
    assert _wait_terminal(client, later["id"])["status"] == "completed"
    assert client.get(f"/v1/videos/{job['id']}").json()["status"] == "failed"
    assert client.delete(f"/v1/videos/{job['id']}").json()["deleted"] is True
    assert client.get(f"/v1/videos/{job['id']}").status_code == 404


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


def test_only_the_video_variant_exists(client, backend):
    clip = _save_clip("x", "2026-01-01T00:00:00Z")
    resp = client.get(f"/v1/videos/{clip['id']}/content", params = {"variant": "thumbnail"})
    assert resp.status_code == 400 and resp.json()["error"]["param"] == "variant"


def test_input_reference_upload_becomes_the_first_frame(client, backend, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(backend, "begin_generate", lambda **kwargs: captured.update(kwargs))
    png = b"\x89PNG\r\n\x1a\nfake"
    resp = _create(
        client, {"prompt": "animate this"}, {"input_reference": ("frame.png", png, "image/png")}
    )
    assert resp.status_code == 200, resp.json()
    assert captured["first_frame"].startswith("data:image/png;base64,")
    assert captured["video_id"] == resp.json()["id"]
    resp = _create(
        client, {"prompt": "x", "input_reference[image_url]": "data:image/jpeg;base64,AAAA"}
    )
    assert resp.status_code == 200 and captured["first_frame"] == "data:image/jpeg;base64,AAAA"
    resp = client.post(
        "/v1/videos",
        json = {"prompt": "x", "input_reference": {"image_url": "data:image/png;base64,BBBB"}},
    )
    assert resp.status_code == 200 and captured["first_frame"] == "data:image/png;base64,BBBB"


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
    resp = _create(client, {"prompt": "x"}, {"input_reference": ("a.txt", b"hello", "text/plain")})
    assert resp.status_code == 400 and "must be an image" in resp.json()["error"]["message"]
    assert client.get("/v1/videos").json()["data"] == []


def test_octet_stream_uploads_are_sniffed(client, backend, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(backend, "begin_generate", lambda **kwargs: captured.update(kwargs))
    jpeg = b"\xff\xd8\xff\xe0fake"
    resp = _create(
        client,
        {"prompt": "x"},
        {"input_reference": ("frame.bin", jpeg, "application/octet-stream")},
    )
    assert resp.status_code == 200, resp.json()
    assert captured["first_frame"].startswith("data:image/jpeg;base64,")
    webp = b"RIFF\x00\x00\x00\x00WEBPVP8 "
    resp = _create(
        client, {"prompt": "x"}, {"input_reference": ("frame", webp, "application/octet-stream")}
    )
    assert resp.status_code == 200 and captured["first_frame"].startswith("data:image/webp;base64,")
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

    def _cancel_after_it_finishes():
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
