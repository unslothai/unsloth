# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hermetic: the restyle workflow's frame prep, Canny control, request body and runner; sd-server is faked."""

from __future__ import annotations

import base64
import io
import threading

import numpy as np
import pytest
from PIL import Image

from core.inference import video_restyle as vr


@pytest.fixture
def weights(tmp_path, monkeypatch):
    model = tmp_path / "cosmos3.safetensors"
    vae = tmp_path / "wan2.2_vae.safetensors"
    model.write_bytes(b"x")
    vae.write_bytes(b"x")
    monkeypatch.setenv(vr.MODEL_ENV, str(model))
    monkeypatch.setenv(vr.VAE_ENV, str(vae))
    return model, vae


@pytest.mark.parametrize(
    "count, expected", [(5, 5), (8, 5), (9, 9), (120, 117), (121, 121), (400, 121)]
)
def test_snap_frame_count(count, expected):
    assert vr.snap_frame_count(count) == expected


def test_snap_frame_count_refuses_short_clip():
    with pytest.raises(vr.RestyleError):
        vr.snap_frame_count(4)


def test_model_files_names_missing_env(monkeypatch):
    monkeypatch.delenv(vr.MODEL_ENV, raising = False)
    monkeypatch.delenv(vr.VAE_ENV, raising = False)
    with pytest.raises(vr.RestyleError, match = vr.MODEL_ENV):
        vr.model_files()


def test_model_files_from_env(weights):
    files = vr.model_files()
    assert files.diffusion_model == str(weights[0])
    assert files.vae == str(weights[1])


def test_build_restyle_request_trims_to_4n_plus_1():
    req = vr.build_restyle_request(prompt = "  red rally car  ", control_frames = ["f"] * 12, seed = 7)
    assert req["video_frames"] == 9
    assert len(req["control_frames"]) == 9
    assert req["prompt"] == "red rally car"
    assert (req["width"], req["height"], req["fps"], req["seed"]) == (832, 480, 30, 7)
    assert req["output_format"] == "avi"
    # The distilled sampler is fixed inside sd.cpp.
    assert "sample_params" not in req


def test_build_restyle_request_needs_prompt():
    with pytest.raises(vr.RestyleError):
        vr.build_restyle_request(prompt = " ", control_frames = ["f"] * 5, seed = 0)


def test_parse_progress():
    assert vr.parse_progress(
        "[INFO ] cosmos3_transfer.cpp:354 - cosmos3: latent frame 3/31 in 512 ms"
    ) == (3, 31)
    assert vr.parse_progress("cosmos3: control video encoded to 31x60x104 latents") is None


def test_canny_flat_image_has_no_edges():
    assert vr.canny_edges(np.full((40, 60, 3), 128, np.uint8)).max() == 0


def test_canny_step_edge_is_one_thin_line():
    img = np.zeros((40, 60, 3), np.uint8)
    img[:, 30:] = 255
    edges = vr.canny_edges(img)
    cols = np.flatnonzero(edges.any(axis = 0))
    assert edges.dtype == np.uint8 and set(np.unique(edges)) == {0, 255}
    assert len(cols) == 1 and cols[0] in (29, 30)
    assert edges[:, cols[0]].all()


def test_canny_hysteresis_keeps_weak_only_when_connected():
    # Left half: strong step joined to a weak step (kept). Right: an isolated weak step (dropped).
    img = np.zeros((40, 80), np.uint8)
    img[:, 20:] = 40  # |dx| + |dy| = 160 on the line: weak (100..200)
    img[:20, 20:40] = 200  # strong part of the same line
    img[:, 60:] = 80  # isolated weak step at x=60
    edges = vr.canny_edges(np.repeat(img[..., None], 3, -1))
    assert edges[30, 19:21].any()  # weak part, connected
    assert not edges[:, 58:62].any()


def test_fit_frame_covers_target():
    out = vr.fit_frame(Image.new("RGB", (1920, 1080)))
    assert out.size == (vr.WIDTH, vr.HEIGHT)
    out = vr.fit_frame(Image.new("RGB", (300, 600)))
    assert out.size == (vr.WIDTH, vr.HEIGHT)


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format = "PNG")
    return buf.getvalue()


def test_frames_from_images_sorted_and_filtered():
    items = [("b/002.png", _png("blue")), ("b/notes.txt", b"hi"), ("b/001.png", _png("red"))]
    frames = vr.frames_from_images(items)
    assert [f.getpixel((0, 0)) for f in frames] == [(255, 0, 0), (0, 0, 255)]


def _clip(fps, count):
    import av

    buf = io.BytesIO()
    with av.open(buf, mode = "w", format = "mp4") as dst:
        stream = dst.add_stream("libx264", rate = fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(count):
            frame = np.full((48, 64, 3), (i * 10) % 256, np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format = "rgb24")):
                dst.mux(packet)
        for packet in stream.encode():
            dst.mux(packet)
    return buf.getvalue()


@pytest.mark.parametrize("fps, count, expected", [(30, 20, 20), (60, 40, 20), (15, 10, 19)])
def test_frames_from_video_resamples_to_30fps(fps, count, expected):
    frames = vr.frames_from_video(_clip(fps, count))
    assert abs(len(frames) - expected) <= 1


def test_frames_from_video_caps_length():
    assert len(vr.frames_from_video(_clip(30, 140))) == vr.MAX_FRAMES


def test_frames_from_video_rejects_garbage():
    with pytest.raises(vr.RestyleError):
        vr.frames_from_video(b"not a video")


class _FakeServer:
    def __init__(
        self,
        avi,
        *,
        fail = None,
    ):
        self.avi = avi
        self.fail = fail
        self.started = 0
        self.payloads = []

    def is_alive(self):
        return True

    def start(self, files):
        self.started += 1

    def stop(self):
        pass

    def vid_gen(
        self,
        payload,
        *,
        on_step = None,
        cancel_event = None,
    ):
        self.payloads.append(payload)
        if self.fail is not None:
            raise self.fail
        for t in range(1, 3):
            on_step(f"cosmos3: latent frame {t}/2 in 500 ms")
        on_step("cosmos3: 2 latent frames in 1.00s (500.0 ms per latent frame)")
        return self.avi


def _frames(n):
    return [Image.new("RGB", (64, 48), (i * 20 % 256, 0, 0)) for i in range(n)]


def test_runner_end_to_end_with_fake_server(weights, monkeypatch):
    server = _FakeServer(b"AVI")
    runner = vr.RestyleRunner(server_factory = lambda _model: server)
    monkeypatch.setattr(
        "core.inference.video_minimax_h3.transcode_video_to_mp4",
        lambda path, fps: b"MP4:" + path.read_bytes(),
    )
    saved = []
    job = runner.start(prompt = "red car", frames = _frames(7), seed = 3)
    assert runner.progress()["active"] and runner.progress()["total"] == 2
    with pytest.raises(vr.RestyleBusy):
        runner.start(prompt = "again", frames = _frames(5))
    runner.run(job, save = lambda mp4, meta: saved.append((mp4, meta)) or {"id": "v1"})

    state = runner.progress()
    assert state["phase"] == "done" and not state["active"]
    assert (state["step"], state["total"]) == (2, 2)
    assert state["eta_seconds"] is None  # cleared once decoding starts
    assert state["video"] == {"id": "v1"}
    payload = server.payloads[0]
    assert payload["video_frames"] == 5 and len(payload["control_frames"]) == 5
    control = Image.open(io.BytesIO(base64.b64decode(payload["control_frames"][0])))
    assert control.size == (vr.WIDTH, vr.HEIGHT) and control.mode == "RGB"
    assert saved[0][0] == b"MP4:AVI" and saved[0][1]["model"] == vr.MODEL_ID
    # The gallery lists only sidecars its own model accepts.
    from core.inference import video_gallery
    from models.inference import GalleryVideo

    assert all(k in saved[0][1] for k in video_gallery._REQUIRED_META)
    GalleryVideo(id = "v1", url = "/x", **saved[0][1])
    assert state["seed"] == 3
    for kind in ("input", "control", "output"):
        assert runner.media_path(state["id"], kind) is not None
    assert runner.media_path("other", "output") is None
    # The server is stopped when the job ends, so nothing idle holds VRAM between restyles.
    assert runner._server is None and not runner.holds_gpu()
    runner.run(runner.start(prompt = "blue car", frames = _frames(5)))
    assert server.started == 2 and runner._server is None


def test_runner_reports_failure(weights):
    runner = vr.RestyleRunner(
        server_factory = lambda _model: _FakeServer(b"", fail = RuntimeError("boom"))
    )
    runner.run(runner.start(prompt = "x", frames = _frames(5)))
    state = runner.progress()
    assert state["phase"] == "error" and "boom" in state["error"] and not state["active"]


def test_runner_cancel_before_generate(weights):
    server = _FakeServer(b"AVI")
    runner = vr.RestyleRunner(server_factory = lambda _model: server)
    job = runner.start(prompt = "x", frames = _frames(5))
    assert runner.cancel()
    runner.run(job)
    assert runner.progress()["phase"] == "cancelled"
    assert server.payloads == []


def test_runner_start_needs_weights(monkeypatch):
    monkeypatch.delenv(vr.MODEL_ENV, raising = False)
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b""))
    with pytest.raises(vr.RestyleError):
        runner.start(prompt = "x", frames = _frames(5))
    assert not runner.progress()["active"]


def test_route_rejects_mixed_upload(weights, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    client = TestClient(app)
    files = [
        ("video", ("a.mp4", b"x", "video/mp4")),
        ("frames", ("001.png", _png("red"), "image/png")),
    ]
    res = client.post("/api/inference/video/restyle", data = {"prompt": "x"}, files = files)
    assert res.status_code == 400
    res = client.post(
        "/api/inference/video/restyle", data = {"prompt": "x", "seed": "abc"}, files = files[1:]
    )
    assert res.status_code == 400
    assert client.get("/api/inference/video/restyle/nope/output.mp4").status_code == 404


def test_runner_hides_another_accounts_job(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    job = runner.start(prompt = "x", frames = _frames(5), owner = "alice")
    assert runner.progress("bob") == {"active": True, "phase": None, "pending": False}
    assert not runner.cancel("bob")
    assert runner.progress("alice")["id"] == job["id"]
    assert runner.cancel("alice")


def test_cancel_aborts_a_loading_server(weights):
    started = threading.Event()
    released = threading.Event()

    class _SlowServer(_FakeServer):
        def start(self, files):
            started.set()
            assert released.wait(5)
            raise vr_cancelled("startup cancelled")

        def stop(self):
            released.set()

    from core.inference.sd_cpp_engine import SdCppCancelled as vr_cancelled

    runner = vr.RestyleRunner(server_factory = lambda _model: _SlowServer(b""))
    job = runner.start(prompt = "x", frames = _frames(5))
    worker = threading.Thread(target = runner.run, args = (job,))
    worker.start()
    assert started.wait(5)
    assert runner.progress()["phase"] == "loading"
    assert runner.cancel()
    assert released.is_set()  # the cancel stopped the load itself
    worker.join(5)
    assert runner.progress()["phase"] == "cancelled"


def test_cancel_during_export_publishes_nothing(weights, monkeypatch):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))

    def transcode(path, fps):
        assert runner.cancel()
        return b"MP4"

    monkeypatch.setattr("core.inference.video_minimax_h3.transcode_video_to_mp4", transcode)
    saved = []
    job = runner.start(prompt = "x", frames = _frames(5))
    runner.run(job, save = lambda mp4, meta: saved.append(mp4))
    assert runner.progress()["phase"] == "cancelled"
    assert saved == [] and runner.media_path(job["id"], "output") is None


def test_body_middleware_streams_the_restyle_upload():
    import main
    from utils.upload_limits import VIDEO_RESTYLE_UPLOAD_MAX_BYTES, upload_request_limit_bytes

    path = "/api/inference/video/restyle"
    assert path in main._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS
    assert path + "/preview" in main._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS
    for p in (path, path + "/", path + "/preview"):
        assert main._get_upload_passthrough_request_max_bytes(p) == upload_request_limit_bytes(
            VIDEO_RESTYLE_UPLOAD_MAX_BYTES
        )
    # Its JSON siblings keep the normal cap.
    assert path + "/cancel" not in main._BODY_UPLOAD_PASSTHROUGH_EXACT_PATHS


def test_route_refuses_an_oversized_upload_before_reading(weights, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    monkeypatch.setattr(route, "VIDEO_RESTYLE_UPLOAD_MAX_BYTES", 100)
    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    res = TestClient(app).post(
        "/api/inference/video/restyle",
        data = {"prompt": "x"},
        files = [("video", ("a.mp4", b"x" * 200, "video/mp4"))],
    )
    assert res.status_code == 413


def test_cancel_is_refused_once_publishing_starts(weights, monkeypatch):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    monkeypatch.setattr(
        "core.inference.video_minimax_h3.transcode_video_to_mp4", lambda path, fps: b"MP4"
    )
    answers = []

    def save(mp4, meta):
        answers.append(runner.cancel())
        return {"id": "v"}

    runner.run(runner.start(prompt = "x", frames = _frames(5)), save = save)
    assert answers == [False]
    assert runner.progress()["phase"] == "done"


def test_frames_are_fitted_as_they_are_read():
    big = io.BytesIO()
    Image.new("RGB", (1920, 1080), "red").save(big, format = "PNG")
    (frame,) = vr.frames_from_images([("001.png", big.getvalue())])
    assert frame.size == (vr.WIDTH, vr.HEIGHT)
    assert {f.size for f in vr.frames_from_video(_clip(30, 6))} == {(vr.WIDTH, vr.HEIGHT)}


def test_unload_if_idle_refuses_while_a_restyle_holds_the_runner(weights):
    stopped = []

    class _Server(_FakeServer):
        def stop(self):
            stopped.append(True)

    runner = vr.RestyleRunner(server_factory = lambda _model: _Server(b"AVI"))
    runner._ensure_server(vr.model_files())
    runner.start(prompt = "x", frames = _frames(5))
    assert not runner.unload_if_idle()
    assert stopped == [] and runner._server is not None
    runner._update(active = False)
    assert runner.unload_if_idle()
    assert stopped == [True] and runner._server is None


def test_progress_waits_for_start_to_swap_owner_and_state(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    out = []
    runner._lock.acquire()
    reader = threading.Thread(target = lambda: out.append(runner.progress("bob")))
    reader.start()
    reader.join(0.2)
    assert reader.is_alive() and out == []
    runner._lock.release()
    reader.join(5)
    assert out == [{"active": False, "phase": None, "pending": False}]


def test_reservation_refuses_a_second_upload_until_released(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    token = runner.reserve()
    with pytest.raises(vr.RestyleBusy):
        runner.reserve()
    with pytest.raises(vr.RestyleBusy):
        runner.start(prompt = "x", frames = _frames(5))
    runner.release("someone-else")
    with pytest.raises(vr.RestyleBusy):
        runner.reserve()
    runner.release(token)
    other = runner.reserve()
    assert runner.start(prompt = "x", frames = _frames(5), reservation = other)["id"]
    runner.release(other)
    with pytest.raises(vr.RestyleBusy):
        runner.reserve()


def test_route_refuses_before_reading_the_upload(weights, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    monkeypatch.setattr(vr, "runner", runner)
    decoded = []
    monkeypatch.setattr(vr, "frames_from_video", lambda data: decoded.append(data) or _frames(5))
    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    client = TestClient(app)
    token = runner.reserve()
    files = [("video", ("a.mp4", b"x", "video/mp4"))]
    res = client.post("/api/inference/video/restyle", data = {"prompt": "x"}, files = files)
    assert res.status_code == 409 and decoded == []
    runner.release(token)
    res = client.post("/api/inference/video/restyle", data = {"prompt": ""}, files = files)
    assert res.status_code == 400 and len(decoded) == 1
    # The failed request gave its reservation back.
    runner.release(runner.reserve())


def test_media_path_waits_for_start_to_swap_owner_and_state(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    out = []
    runner._lock.acquire()
    reader = threading.Thread(target = lambda: out.append(runner.media_path("old", "output", "bob")))
    reader.start()
    reader.join(0.2)
    assert reader.is_alive() and out == []
    runner._lock.release()
    reader.join(5)
    assert out == [None]


def test_frames_from_video_error_is_plain():
    with pytest.raises(vr.RestyleError) as err:
        vr.frames_from_video(b"not a video")
    assert "MP4" in str(err.value) and "Errno" not in str(err.value)


def test_preview_reports_frames_used_and_edges():
    out = vr.preview(_frames(140), source_seconds = 18.0)
    assert out["frames"] == vr.MAX_FRAMES and out["seconds"] == pytest.approx(121 / 30)
    assert out["source_seconds"] == 18.0
    for key in ("frame_jpeg", "edges_jpeg"):
        image = Image.open(io.BytesIO(base64.b64decode(out[key])))
        assert image.size == (vr.WIDTH, vr.HEIGHT)
    with pytest.raises(vr.RestyleError):
        vr.preview(_frames(3))


def test_availability_names_what_is_missing(weights, monkeypatch, tmp_path):
    monkeypatch.setattr("core.inference.sd_cpp_engine.find_sd_server_binary", lambda: None)
    out = vr.availability()
    assert not out["available"] and "stable-diffusion.cpp" in out["reason"]
    plain = tmp_path / "bin" / "sd-server"
    plain.parent.mkdir()
    plain.write_bytes(b"\x7fELF generic sd-server")
    monkeypatch.setattr("core.inference.sd_cpp_engine.find_sd_server_binary", lambda: str(plain))
    assert not vr.availability()["available"]
    (plain.parent / "libstable-diffusion.so").write_bytes(b"..." + vr._COSMOS3_MARKER + b" %d/%d")
    assert vr.availability() == {"available": True, "reason": None}
    assert not vr.availability({})["available"]


def test_random_seed_is_reported(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    job = runner.start(prompt = "x", frames = _frames(5))
    assert runner.progress()["seed"] == job["seed"]


def test_route_preview_and_status(weights, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    client = TestClient(app)
    res = client.post(
        "/api/inference/video/restyle/preview",
        files = [("video", ("a.mp4", _clip(30, 20), "video/mp4"))],
    )
    assert res.status_code == 200 and res.json()["frames"] == 17
    res = client.post(
        "/api/inference/video/restyle/preview",
        files = [("video", ("a.mp4", b"junk", "video/mp4"))],
    )
    assert res.status_code == 400 and "MP4" in res.json()["detail"]
    monkeypatch.setattr("core.inference.sd_cpp_engine.find_sd_server_binary", lambda: None)
    assert client.get("/api/inference/video/restyle/status").json()["available"] is False


def test_frames_from_video_fits_only_kept_frames(monkeypatch):
    calls = []
    real = vr.fit_frame
    monkeypatch.setattr(vr, "fit_frame", lambda image: calls.append(1) or real(image))
    frames = vr.frames_from_video(_clip(120, 40))
    # 40 frames at 120 fps is a third of a second: 10 frames at 30 fps, each fitted once.
    assert abs(len(frames) - 10) <= 1 and len(calls) == len(frames)


@pytest.mark.parametrize("owner", ["chat", "diffusion", "video"])
def test_gpu_acquire_releases_an_idle_restyle_server(weights, monkeypatch, owner):
    import core.inference.gpu_arbiter as arb

    stopped = []

    class _Server(_FakeServer):
        def stop(self):
            stopped.append(True)

    runner = vr.RestyleRunner(server_factory = lambda _model: _Server(b"AVI"))
    runner._ensure_server(vr.model_files())
    monkeypatch.setattr(vr, "runner", runner)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_release_idle_video_servers", lambda reason: None)
    arb.acquire_for(owner)
    assert stopped == [True] and runner._server is None


def test_gpu_acquire_keeps_a_running_restyle_server(weights, monkeypatch):
    import core.inference.gpu_arbiter as arb

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    server = runner._ensure_server(vr.model_files())
    runner.start(prompt = "x", frames = _frames(5))
    monkeypatch.setattr(vr, "runner", runner)
    monkeypatch.setattr(arb, "_owner", None)
    with pytest.raises(arb.RestyleRunningError):
        arb.acquire_for(arb.CHAT)
    assert runner._server is server


def _route_client(monkeypatch, runner):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from auth.authentication import get_current_subject
    from routes import video_restyle as route

    monkeypatch.setattr(vr, "runner", runner)
    app = FastAPI()
    app.include_router(route.router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    return TestClient(app)


@pytest.mark.parametrize("allowed", [False, True])
def test_route_enforces_the_model_allowlist(weights, monkeypatch, allowed):
    from hub.services.models import account_access

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    client = _route_client(monkeypatch, runner)
    decoded = []
    monkeypatch.setattr(vr, "frames_from_video", lambda data: decoded.append(data) or _frames(5))
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "model_allowlist", lambda: {"some/other-model"})
    monkeypatch.setattr(account_access, "model_allowed", lambda repo_id: allowed)
    files = [("video", ("a.mp4", b"x", "video/mp4"))]
    res = client.post("/api/inference/video/restyle", data = {"prompt": ""}, files = files)
    if allowed:
        assert res.status_code == 400 and len(decoded) == 1
    else:
        assert res.status_code == 403 and decoded == []


def test_route_cancel_runs_off_the_event_loop(weights, monkeypatch):
    import asyncio

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    client = _route_client(monkeypatch, runner)
    loops = []

    def cancel(owner = None):
        try:
            loops.append(asyncio.get_running_loop())
        except RuntimeError:
            loops.append(None)
        return True

    monkeypatch.setattr(runner, "cancel", cancel)
    assert client.post("/api/inference/video/restyle/cancel").json() == {"cancelled": True}
    assert loops == [None]


def test_training_teardown_releases_an_idle_restyle_server(weights, monkeypatch):
    from routes import training

    stopped = []

    class _Server(_FakeServer):
        def stop(self):
            stopped.append(True)

    runner = vr.RestyleRunner(server_factory = lambda _model: _Server(b"AVI"))
    runner._ensure_server(vr.model_files())
    monkeypatch.setattr(vr, "runner", runner)
    training._free_restyle_for_training()
    assert stopped == [True] and runner._server is None


def test_training_teardown_refuses_while_a_restyle_runs(weights, monkeypatch):
    from routes import training
    from routes.training_vram import ManagedEngineStillRunning

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    server = runner._ensure_server(vr.model_files())
    runner.start(prompt = "x", frames = _frames(5))
    monkeypatch.setattr(vr, "runner", runner)
    with pytest.raises(ManagedEngineStillRunning):
        training._free_restyle_for_training()
    assert runner._server is server


def test_diffusion_training_teardown_includes_the_restyle_server(monkeypatch):
    from routes import training

    class _Called(Exception):
        pass

    def called():
        raise _Called

    monkeypatch.setattr(training, "_free_restyle_for_training", called)
    with pytest.raises(_Called):
        training._free_gpu_for_diffusion_training()


def test_fit_frame_never_resamples_past_the_output(monkeypatch):
    real = Image.Image.resize
    sizes = []

    def resize(self, size, *args, **kwargs):
        sizes.append(tuple(size))
        assert size[0] * size[1] <= vr.WIDTH * vr.HEIGHT, size
        return real(self, size, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", resize)
    out = vr.fit_frame(Image.new("RGB", (1, 20000), (9, 9, 9)))
    assert out.size == (vr.WIDTH, vr.HEIGHT) and sizes


def test_retire_cancels_only_the_owners_job(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    runner.start(prompt = "x", frames = _frames(5), owner = "alice")
    assert not runner.retire("bob")
    assert runner.retire("alice") and runner._cancel.is_set()


def test_start_refuses_a_fenced_account(weights):
    from state import active_generations

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    active_generations.fence("alice")
    try:
        with pytest.raises(vr.RestyleError):
            runner.start(prompt = "x", frames = _frames(5), owner = "alice")
    finally:
        active_generations.lift_fence("alice")
    runner.start(prompt = "x", frames = _frames(5), owner = "alice")


def test_route_refuses_a_restyle_beside_training(weights, monkeypatch):
    import core.training as training_pkg

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    client = _route_client(monkeypatch, runner)
    monkeypatch.setattr(vr, "frames_from_video", lambda data: _frames(5))

    class _Training:
        def is_training_active(self):
            return True

    monkeypatch.setattr(training_pkg, "get_training_backend", lambda: _Training())
    files = [("video", ("a.mp4", b"x", "video/mp4"))]
    res = client.post("/api/inference/video/restyle", data = {"prompt": "x"}, files = files)
    assert res.status_code == 409 and "training" in res.json()["detail"]
    assert not runner.progress().get("active")


def test_a_concat_playlist_upload_cannot_read_local_files(tmp_path, monkeypatch):
    (tmp_path / "secret.mp4").write_bytes(_clip(30, 10))
    monkeypatch.chdir(tmp_path)
    playlist = b"ffconcat version 1.0\nfile 'secret.mp4'\n"
    with pytest.raises(vr.RestyleError):
        vr.frames_from_video(playlist)
    assert vr.video_seconds(playlist) is None


def test_route_maps_a_diffusion_training_admission_refusal_to_409(weights, monkeypatch):
    import contextlib

    from core.training.diffusion_training_service import TrainingActiveError
    from routes import inference

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    client = _route_client(monkeypatch, runner)
    monkeypatch.setattr(vr, "frames_from_video", lambda data: _frames(5))

    @contextlib.contextmanager
    def refused():
        raise TrainingActiveError("A diffusion training run holds the GPU.")
        yield

    monkeypatch.setattr(inference, "_diffusion_training_admission", refused)
    files = [("video", ("a.mp4", b"x", "video/mp4"))]
    res = client.post("/api/inference/video/restyle", data = {"prompt": "x"}, files = files)
    assert res.status_code == 409 and not runner.progress().get("active")


def test_frames_from_video_holds_the_last_frame_of_a_slow_source():
    frames = vr.frames_from_video(_clip(1, 4))
    assert len(frames) == 120
    # The 4th second shows the 4th source frame, not the 3rd.
    assert frames[-1].getpixel((0, 0)) != frames[60].getpixel((0, 0))


def test_a_reserved_runner_keeps_its_server(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    server = runner._ensure_server(vr.model_files())
    token = runner.reserve()
    assert not runner.unload_if_idle() and runner._server is server
    runner.release(token)
    assert runner.unload_if_idle() and runner._server is None


class _Video:
    def __init__(
        self,
        loaded,
        loading = (),
    ):
        self.loaded, self.loading, self.unloaded = loaded, loading, 0

    def loading_repo_ids(self):
        return self.loading

    def status(self):
        return {"loaded": self.loaded}

    def unload(self):
        self.unloaded += 1


def _gpu_handoff(
    monkeypatch,
    mode,
    video,
    generating = None,
):
    import core.inference.gpu_arbiter as arb
    import core.inference.video as video_module
    import utils.multi_model_settings as mms

    acquired = []
    monkeypatch.setattr(mms, "get_multi_model_mode", lambda: mode)
    monkeypatch.setattr(video_module, "get_video_backend", lambda: video)
    monkeypatch.setattr(video_module, "generation_account_in_flight", lambda: generating)

    def acquire(
        owner,
        register = None,
        admit = None,
        **kw,
    ):
        admit()
        register()
        acquired.append((owner, kw))

    monkeypatch.setattr(arb, "acquire_for_request", acquire)
    return acquired


def test_start_takes_the_gpu_as_video_and_unloads_the_create_model(weights, monkeypatch):
    from routes import video_restyle as route

    video = _Video(loaded = True)
    acquired = _gpu_handoff(monkeypatch, "off", video)
    route._take_the_gpu()
    assert video.unloaded == 1 and acquired == [("video", {"policy": "off"})]


@pytest.mark.parametrize("mode, generating", [("always", None), ("off", "alice")])
def test_start_refuses_to_drop_a_create_model_it_must_keep(weights, monkeypatch, mode, generating):
    from fastapi import HTTPException

    from routes import video_restyle as route

    video = _Video(loaded = True)
    acquired = _gpu_handoff(monkeypatch, mode, video, generating)
    with pytest.raises(HTTPException) as exc:
        route._take_the_gpu()
    assert exc.value.status_code == 409 and video.unloaded == 0 and acquired == []


def test_start_refuses_while_a_create_model_loads(weights, monkeypatch):
    from fastapi import HTTPException

    from routes import video_restyle as route

    video = _Video(loaded = False, loading = ("org/some-video-model",))
    acquired = _gpu_handoff(monkeypatch, "off", video)
    with pytest.raises(HTTPException) as exc:
        route._take_the_gpu()
    assert exc.value.status_code == 409 and acquired == []


def test_gpu_acquire_is_refused_while_a_restyle_runs(weights, monkeypatch):
    import core.inference.gpu_arbiter as arb

    evicted = []
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    runner.start(prompt = "x", frames = _frames(5))
    monkeypatch.setattr(vr, "runner", runner)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: evicted.append("chat"))
    with pytest.raises(arb.RestyleRunningError):
        arb.acquire_for(arb.DIFFUSION)
    assert evicted == []
    assert arb.RestyleRunningError().as_http_exception().status_code == 409


def test_a_reserved_runner_refuses_other_loads_but_not_its_own_handoff(weights, monkeypatch):
    import core.inference.gpu_arbiter as arb

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    server = runner._ensure_server(vr.model_files())
    runner.reserve()
    monkeypatch.setattr(vr, "runner", runner)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_release_idle_video_servers", lambda reason: None)
    with pytest.raises(arb.RestyleRunningError):
        arb.acquire_for(arb.CHAT)
    token = arb.restyle_handoff.set(True)
    try:
        arb.acquire_for(arb.VIDEO)
    finally:
        arb.restyle_handoff.reset(token)
    assert runner._server is server


def test_an_invalid_request_is_refused_before_the_gpu_handoff(weights, monkeypatch):
    from routes import video_restyle as route

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    client = _route_client(monkeypatch, runner)
    monkeypatch.setattr(vr, "frames_from_video", lambda data: _frames(5))
    took = []
    monkeypatch.setattr(route, "_take_the_gpu", lambda: took.append(True))
    files = [("video", ("a.mp4", b"x", "video/mp4"))]
    res = client.post("/api/inference/video/restyle", data = {"prompt": " "}, files = files)
    assert res.status_code == 400 and took == []


def test_the_create_check_runs_under_the_arbiter_lock(weights, monkeypatch):
    import core.inference.gpu_arbiter as arb
    import core.inference.video as video_module
    import utils.multi_model_settings as mms
    from fastapi import HTTPException

    from routes import video_restyle as route

    held = []

    class _Loading(_Video):
        def loading_repo_ids(self):
            held.append(arb._lock.locked())
            return ("org/some-video-model",)

    evicted = []
    monkeypatch.setattr(mms, "get_multi_model_mode", lambda: "off")
    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Loading(False))
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setitem(arb._EVICTORS, arb.CHAT, lambda: evicted.append("chat"))
    with pytest.raises(HTTPException) as exc:
        route._take_the_gpu()
    assert exc.value.status_code == 409 and held == [True] and evicted == []


def test_retire_drops_the_accounts_reservation(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    runner.reserve("bob")
    assert not runner.retire("alice") and runner.holds_gpu()
    assert not runner.retire("bob") and not runner.holds_gpu()


def test_unload_refuses_another_accounts_server(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    runner.start(prompt = "x", frames = _frames(5), owner = "alice")
    runner._ensure_server(vr.model_files())
    runner._update(active = False)
    with pytest.raises(vr.RestyleBusy):
        runner.unload_if_idle("bob")
    assert runner._server is not None
    assert runner.unload_if_idle("alice") and runner._server is None


def test_a_refused_handoff_keeps_the_create_model(weights, monkeypatch):
    import core.inference.gpu_arbiter as arb
    import core.inference.video as video_module
    import utils.multi_model_settings as mms
    from fastapi import HTTPException

    from routes import video_restyle as route

    video = _Video(loaded = True)
    monkeypatch.setattr(mms, "get_multi_model_mode", lambda: "fit")
    monkeypatch.setattr(video_module, "get_video_backend", lambda: video)
    monkeypatch.setattr(video_module, "generation_account_in_flight", lambda: None)

    def acquire(
        owner,
        register = None,
        admit = None,
        **kw,
    ):
        admit()
        raise HTTPException(status_code = 409, detail = "another account is generating")

    monkeypatch.setattr(arb, "acquire_for_request", acquire)
    with pytest.raises(HTTPException):
        route._take_the_gpu()
    assert video.unloaded == 0


def test_a_resident_server_holds_the_managed_sd_cpp_tree(weights, monkeypatch):
    import core.inference.sd_cpp_backend as backend
    import core.inference.sd_cpp_engine as engine
    import core.inference.sd_cpp_server as server_module

    monkeypatch.setattr(engine, "find_sd_server_binary", lambda: "/managed/sd-server")
    monkeypatch.setattr(engine, "is_managed_binary", lambda binary: True)
    monkeypatch.setattr(server_module, "SdCppServer", lambda binary: _FakeServer(b"AVI"))
    runner = vr.RestyleRunner()
    try:
        runner._ensure_server(vr.model_files())
        assert backend._external_tree_holder_alive()
        runner.unload()
        assert not backend._external_tree_holder_alive()
    finally:
        backend.unregister_tree_holder(runner)


def test_a_stopping_server_still_pins_the_tree(weights):
    import threading

    release = threading.Event()
    stopping = threading.Event()

    class _Slow(_FakeServer):
        def stop(self):
            stopping.set()
            assert release.wait(5)

    runner = vr.RestyleRunner(server_factory = lambda _model: _Slow(b"AVI"))
    runner._ensure_server(vr.model_files())
    worker = threading.Thread(target = runner.unload)
    worker.start()
    assert stopping.wait(5)
    assert runner._server is None and runner.is_alive()
    release.set()
    worker.join(5)
    assert not runner.is_alive()


def test_a_stopping_server_holds_the_gpu_and_unload_waits_for_it(weights):
    import threading

    release = threading.Event()
    stopping = threading.Event()

    class _Slow(_FakeServer):
        def stop(self):
            stopping.set()
            assert release.wait(5)

    runner = vr.RestyleRunner(server_factory = lambda _model: _Slow(b"AVI"))
    runner._ensure_server(vr.model_files())
    first = threading.Thread(target = runner.unload)
    first.start()
    assert stopping.wait(5) and runner.holds_gpu()
    done = []
    second = threading.Thread(target = lambda: done.append(runner.unload_if_idle()))
    second.start()
    second.join(0.3)
    assert done == []
    release.set()
    first.join(5)
    second.join(5)
    assert done == [True] and not runner.holds_gpu()


def test_start_saves_the_chat_kv_around_the_handoff(weights, monkeypatch):
    from routes import video_restyle as route

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    client = _route_client(monkeypatch, runner)
    monkeypatch.setattr(vr, "frames_from_video", lambda data: _frames(5))
    monkeypatch.setattr(runner, "run", lambda job, save = None: None)
    calls = []

    class _Swap:
        def close(self):
            calls.append("close")

    async def save(owner, media, **kw):
        calls.append(("save", owner, media, kw.get("policy")))
        return _Swap()

    monkeypatch.setattr(route, "save_chat_kv_before_swap", save)
    monkeypatch.setattr(route, "get_multi_model_mode", lambda: "off")
    monkeypatch.setattr(route, "_take_the_gpu", lambda mode = None: calls.append(("take", mode)))
    files = [("video", ("a.mp4", b"x", "video/mp4"))]
    res = client.post("/api/inference/video/restyle", data = {"prompt": "x"}, files = files)
    assert res.status_code == 200
    assert calls == [("save", "video", "video", "off"), ("take", "off"), "close"]


def test_a_new_server_waits_for_the_previous_one_to_stop(weights):
    import threading

    release = threading.Event()
    stopping = threading.Event()
    started = []

    class _Slow(_FakeServer):
        def stop(self):
            stopping.set()
            assert release.wait(5)

        def start(self, files):
            started.append(release.is_set())

    runner = vr.RestyleRunner(server_factory = lambda _model: _Slow(b"AVI"))
    runner._ensure_server(vr.model_files())
    started.clear()
    unloading = threading.Thread(target = runner.unload)
    unloading.start()
    assert stopping.wait(5)
    launching = threading.Thread(target = runner._ensure_server, args = (vr.model_files(),))
    launching.start()
    launching.join(0.3)
    assert started == []
    release.set()
    unloading.join(5)
    launching.join(5)
    assert started == [True]


def test_a_dead_resident_server_does_not_pin_the_tree(weights):
    class _Dead(_FakeServer):
        def is_alive(self):
            return False

    runner = vr.RestyleRunner(server_factory = lambda _model: _Dead(b"AVI"))
    runner._ensure_server(vr.model_files())
    assert runner._server is not None and not runner.is_alive()


def test_an_unload_refused_mid_acquire_refuses_the_acquire(weights, monkeypatch):
    import core.inference.gpu_arbiter as arb

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    monkeypatch.setattr(vr, "runner", runner)
    monkeypatch.setattr(runner, "holds_gpu", lambda: False)
    monkeypatch.setattr(runner, "unload_if_idle", lambda owner = None: False)
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(arb, "_release_idle_video_servers", lambda reason: None)
    with pytest.raises(arb.RestyleRunningError):
        arb.acquire_for(arb.CHAT)


def test_the_worker_starts_inside_the_shielded_start(weights, monkeypatch):
    import threading

    from routes import video_restyle as route

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    monkeypatch.setattr(vr, "runner", runner)
    monkeypatch.setattr(route, "_take_the_gpu", lambda mode = None: None)
    ran = threading.Event()
    monkeypatch.setattr(runner, "run", lambda job, save = None: ran.set())
    job = route._admit_and_start(
        prompt = "x", frames = _frames(5), seed = 1, owner = None, reservation = None
    )
    assert job["id"] and ran.wait(5)


def test_the_shielded_start_releases_the_reservation_itself(weights, monkeypatch):
    from fastapi import HTTPException

    from routes import video_restyle as route

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    monkeypatch.setattr(vr, "runner", runner)

    def refuse(mode = None):
        assert runner.holds_gpu()  # still reserved while the handoff runs
        raise HTTPException(status_code = 409, detail = "busy")

    monkeypatch.setattr(route, "_take_the_gpu", refuse)
    token = runner.reserve("alice")
    with pytest.raises(HTTPException):
        route._admit_and_start(
            prompt = "x", frames = _frames(5), seed = 1, owner = "alice", reservation = token
        )
    assert not runner.holds_gpu()


def test_progress_reports_a_pending_start_to_its_account(weights):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    runner.reserve("alice")
    assert runner.progress("alice")["pending"] is True
    assert runner.progress("bob")["pending"] is False


def test_frames_from_video_applies_the_display_rotation(monkeypatch):
    import core.training.diffusion_h3_clips as clips

    sizes = []
    real = vr.fit_frame
    monkeypatch.setattr(clips, "display_rotation_degrees", lambda frame, stream: 90)
    monkeypatch.setattr(vr, "fit_frame", lambda image: sizes.append(image.size) or real(image))
    vr.frames_from_video(_clip(30, 3))
    assert sizes and all(size == (48, 64) for size in sizes)  # coded 64x48, displayed portrait


def test_a_failed_start_leaves_the_previous_owner_and_job(weights, monkeypatch):
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    job = runner.start(prompt = "x", frames = _frames(5), owner = "alice")
    runner._update(active = False, phase = "done")

    def full(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(vr.tempfile, "mkdtemp", full)
    with pytest.raises(OSError):
        runner.start(prompt = "y", frames = _frames(5), owner = "bob")
    assert runner.owned_by("alice") and runner.progress("alice")["id"] == job["id"]
    assert runner.progress("bob").get("id") is None


def test_a_finished_job_has_already_released_its_server(weights, monkeypatch):
    monkeypatch.setattr(
        "core.inference.video_minimax_h3.transcode_video_to_mp4", lambda path, fps: b"MP4"
    )
    seen = []
    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    real = runner._update

    def update(**fields):
        real(**fields)
        if fields.get("active") is False:
            seen.append(runner._server)

    monkeypatch.setattr(runner, "_update", update)
    runner.run(runner.start(prompt = "x", frames = _frames(5)))
    assert runner.progress()["phase"] == "done" and seen == [None]


def test_a_running_restyle_counts_as_inference_activity(weights, monkeypatch):
    import main

    runner = vr.RestyleRunner(server_factory = lambda _model: _FakeServer(b"AVI"))
    monkeypatch.setattr(vr, "runner", runner)
    assert "core.inference.video_restyle" in main._MEDIA_BACKEND_MODULES
    assert not vr.generation_in_flight()
    runner.start(prompt = "x", frames = _frames(5))
    assert vr.generation_in_flight() and main._media_generation_active()


@pytest.mark.parametrize("create_loaded", [False, True])
def test_finishing_releases_the_video_claim_unless_create_holds_it(
    weights, monkeypatch, create_loaded
):
    import core.inference.gpu_arbiter as arb
    import core.inference.video as video_module

    released = []
    monkeypatch.setattr(
        arb, "release_if", lambda owner, idle: idle() and released.append(owner) is None
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Video(loaded = create_loaded))
    runner = vr.RestyleRunner()
    runner._finish(phase = "done")
    assert released == ([] if create_loaded else [arb.VIDEO])


def test_the_video_claim_idle_check_runs_under_the_arbiter_lock(weights, monkeypatch):
    import core.inference.gpu_arbiter as arb
    import core.inference.video as video_module

    held = []

    class _Probe(_Video):
        def status(self):
            held.append(arb._lock.locked())
            return {"loaded": False}

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Probe(False))
    monkeypatch.setattr(arb, "_view", lambda: {arb.VIDEO})
    monkeypatch.setattr(arb, "_drop_resident", lambda owner: None)
    vr.RestyleRunner()._release_video_claim()
    assert held == [True]
