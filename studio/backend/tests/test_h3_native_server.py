# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 native renders through a resident sd-server instead of one sd-cli per render.

Hermetic: the server process and its HTTP client are faked; nothing spawns a binary or opens a
socket. The real-binary check (pixels and audio identical to sd-cli for the same seed, keyframe
and text-only) lives with the benchmark evidence, not here."""

from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path

import pytest

from core.inference import sd_cpp_backend
from core.inference import sd_cpp_server as srv
from core.inference.sd_cpp_args import (
    SdCppModelFiles,
    SdCppVideoGenParams,
    build_sd_cpp_server_command,
    build_vid_gen_request,
    h3_server_eligible,
)
from core.inference.sd_cpp_engine import SdCppCancelled
from core.inference.sd_cpp_server import SdCppServerUnsupported
from core.inference.video import VideoBackend
from core.inference import video_minimax_h3 as h3

# pytest inserts rootdir, not this package, on the path.
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_sd_cpp_server import _FakeClient, _FakePopen, _Resp, _server_with  # noqa: E402

_FILES = SdCppModelFiles(
    diffusion_model = "/m/h3.gguf",
    vae = "/m/vae.safetensors",
    audio_vae = "/m/audio_vae.safetensors",
    llm = "/m/te.gguf",
)


def _params(**overrides) -> SdCppVideoGenParams:
    values = dict(
        prompt = "a dog on a beach",
        width = 960,
        height = 544,
        num_frames = 124,
        fps = 24,
        steps = 20,
        cfg_scale = 1.0,
        seed = 42,
        flow_shift = 12.0,
    )
    values.update(overrides)
    return SdCppVideoGenParams(**values)


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(srv, "adopt_pid", lambda pid: None)
    monkeypatch.setattr(srv, "forget_pid", lambda pid: None)
    monkeypatch.setattr(srv, "child_popen_kwargs", lambda: {})
    monkeypatch.setattr(srv, "windows_hidden_subprocess_kwargs", lambda: {})
    return monkeypatch


def test_vid_gen_request_mirrors_the_cli_argv():
    req = build_vid_gen_request(_params())
    assert req == {
        "prompt": "a dog on a beach",
        "width": 960,
        "height": 544,
        "video_frames": 124,
        "fps": 24,
        "seed": 42,
        "sample_params": {
            "guidance": {"txt_cfg": 1.0},
            "sample_steps": 20,
            "flow_shift": 12.0,
        },
        # AVI needs no WebM build and is what the CUDA prebuilt sd-cli writes.
        "output_format": "avi",
        "output_compression": 90,
    }


def test_vid_gen_request_carries_keyframes_and_rejects_mixing_with_references():
    req = build_vid_gen_request(_params(), images_b64 = {"init_image": "AAA", "end_image": None})
    assert req["init_image"] == "AAA"
    assert "end_image" not in req
    with pytest.raises(ValueError, match = "cannot be combined"):
        build_vid_gen_request(_params(), images_b64 = {"init_image": "AAA"}, ref_images_b64 = ["BBB"])


def test_reference_video_and_audio_stay_on_the_cli():
    assert h3_server_eligible(_params()) is True
    assert h3_server_eligible(_params(ref_images = ("/r.png",))) is True
    for field in ("ref_videos", "ref_video_audios", "ref_audios"):
        params = _params(**{field: ("/r",)})
        assert h3_server_eligible(params) is False
        with pytest.raises(ValueError):
            build_vid_gen_request(params)


def test_server_command_passes_the_audio_vae():
    cmd = build_sd_cpp_server_command("/b/sd-server", _FILES, host = "127.0.0.1", port = 1)
    assert cmd[cmd.index("--audio-vae") + 1] == "/m/audio_vae.safetensors"
    no_audio = build_sd_cpp_server_command(
        "/b/sd-server", SdCppModelFiles(diffusion_model = "/m/z.gguf"), host = "h", port = 1
    )
    assert "--audio-vae" not in no_audio


def test_vid_gen_returns_container_bytes(patched):
    blob = b"RIFF....AVI "
    client = _FakeClient(
        post = lambda url, json: _Resp(202, {"id": "jobV"}),
        get = lambda url: _Resp(
            200,
            {"status": "completed", "result": {"b64_json": base64.b64encode(blob).decode()}},
        ),
    )
    s = _server_with(_FakePopen(), client)
    assert s.vid_gen({"prompt": "x"}, poll_interval = 0.0) == blob
    assert client.post_calls[0][0].endswith("/sdcpp/v1/vid_gen")


@pytest.mark.parametrize(
    "status,text",
    [(404, "Not Found"), (400, '{"error":"loaded model does not support vid_gen"}')],
)
def test_vid_gen_unsupported_is_its_own_error(patched, status, text):
    s = _server_with(_FakePopen(), _FakeClient(post = lambda url, json: _Resp(status, text = text)))
    with pytest.raises(SdCppServerUnsupported):
        s.vid_gen({"prompt": "x"})


def test_vid_gen_other_400_is_a_plain_failure(patched):
    s = _server_with(
        _FakePopen(), _FakeClient(post = lambda url, json: _Resp(400, text = "invalid parameters"))
    )
    with pytest.raises(RuntimeError) as info:
        s.vid_gen({"prompt": "x"})
    assert not isinstance(info.value, SdCppServerUnsupported)


def test_vid_gen_failure_carries_the_log_cause(patched):
    s = _server_with(
        _FakePopen(),
        _FakeClient(
            post = lambda url, json: _Resp(202, {"id": "jobF"}),
            get = lambda url: _Resp(
                200,
                {
                    "status": "failed",
                    "error": {
                        "code": "generation_failed",
                        "message": "generate_video returned no results",
                    },
                },
            ),
        ),
    )
    s._tail.append(
        "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 5463.50 MiB: out of memory"
    )
    with pytest.raises(RuntimeError, match = "out of memory"):
        s.vid_gen({"prompt": "x"}, poll_interval = 0.0)


class _FakeServer:
    def __init__(
        self,
        *,
        data = b"AVI",
        exc = None,
        alive_after = True,
    ):
        self.data = data
        self.exc = exc
        self.alive_after = alive_after
        self.payloads = []
        self._alive = True

    def is_alive(self):
        return self._alive

    def vid_gen(
        self,
        payload,
        *,
        on_step = None,
        cancel_event = None,
    ):
        self.payloads.append(payload)
        if on_step is not None:
            on_step("generate_video 960x544x124")
        if self.exc is not None:
            self._alive = self.alive_after
            raise self.exc
        return self.data


class _FakeSlot:
    def __init__(
        self,
        server = None,
        start_exc = None,
    ):
        self.server = server
        self.start_exc = start_exc
        self.disabled_reason = None
        self.stopped = 0
        self.released = []
        self.got = []
        self.renders = 0
        self.ended = 0

    def get(
        self,
        flags = None,
        env = None,
        *,
        cancel_event = None,
    ):
        self.got.append((flags, env))
        if self.start_exc is not None:
            raise self.start_exc
        return self.server

    def begin_render(self):
        self.renders += 1

    def end_render(self):
        self.ended += 1

    def release(self, reason):
        self.released.append(reason)
        return True

    def alive_signature(self):
        return None

    def stop(self, reason = "stopped"):
        self.stopped += 1

    def disable(self, reason):
        self.disabled_reason = reason
        self.stop()


class _Runtime:
    def __init__(
        self,
        slot,
        offload_flags = ("--offload-to-cpu",),
        env = (),
    ):
        self.server_slot = slot
        self.offload_flags = offload_flags
        self.env = env


def _render(
    slot,
    tmp_path,
    params = None,
    cancel = None,
    **kwargs,
):
    return VideoBackend._h3_native_server_render(
        None,
        _Runtime(slot),
        params or _params(),
        output_path = tmp_path / "out.webm",
        on_log = lambda line: None,
        cancel = cancel or threading.Event(),
        **kwargs,
    )


def test_server_render_writes_the_container(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    server = _FakeServer(data = b"AVI-BYTES")
    out = _render(_FakeSlot(server), tmp_path)
    assert out == tmp_path / "out.webm"
    assert out.read_bytes() == b"AVI-BYTES"
    assert server.payloads[0]["seed"] == 42


def test_keyframes_are_sent_inline(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    first = tmp_path / "first.png"
    first.write_bytes(b"PNG-FIRST")
    server = _FakeServer()
    _render(_FakeSlot(server), tmp_path, _params(init_img = str(first)))
    assert base64.b64decode(server.payloads[0]["init_image"]) == b"PNG-FIRST"


def test_no_slot_or_kill_switch_or_reference_video_uses_the_cli(tmp_path, monkeypatch):
    assert (
        VideoBackend._h3_native_server_render(
            None,
            _Runtime(None),
            _params(),
            output_path = tmp_path / "o",
            on_log = print,
            cancel = threading.Event(),
        )
        is None
    )
    slot = _FakeSlot(_FakeServer())
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_ENV, "0")
    assert _render(slot, tmp_path) is None
    assert slot.stopped == 1
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV)
    server = _FakeServer()
    slot = _FakeSlot(server)
    assert _render(slot, tmp_path, _params(ref_videos = ("/frames",))) is None
    assert server.payloads == []
    # sd-cli reloads the whole model: a resident copy must not sit beside it.
    assert slot.released and slot.got == []


def test_start_failure_falls_back_and_stays_one_shot(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _FakeSlot(start_exc = RuntimeError("sd-server failed to become ready"))
    assert _render(slot, tmp_path) is None
    assert "start failed" in slot.disabled_reason
    slot.start_exc = AssertionError("must not start again")
    assert _render(slot, tmp_path) is None


def test_unsupported_route_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _FakeSlot(_FakeServer(exc = SdCppServerUnsupported("no vid_gen")))
    assert _render(slot, tmp_path) is None
    assert slot.disabled_reason == "no vid_gen"


def test_server_death_mid_render_falls_back(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _FakeSlot(_FakeServer(exc = RuntimeError("connection lost"), alive_after = False))
    assert _render(slot, tmp_path) is None
    assert slot.disabled_reason.startswith("server died")


def test_failed_job_on_a_live_server_raises_and_releases_it(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _FakeSlot(_FakeServer(exc = RuntimeError("out of memory"), alive_after = True))
    with pytest.raises(RuntimeError, match = "out of memory"):
        _render(slot, tmp_path)
    assert slot.stopped == 1
    assert slot.disabled_reason is None


def test_cancel_is_a_cancellation_not_a_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    cancel = threading.Event()
    cancel.set()
    slot = _FakeSlot(_FakeServer(exc = RuntimeError("server stopped"), alive_after = False))
    with pytest.raises(SdCppCancelled):
        _render(slot, tmp_path, cancel = cancel)
    assert slot.disabled_reason is None


def test_sibling_server_binary(tmp_path):
    cli = tmp_path / "sd-cli"
    cli.write_text("")
    assert h3.h3_sibling_server_binary(str(cli)) is None
    (tmp_path / "sd-server").write_text("")
    assert h3.h3_sibling_server_binary(str(cli)) == str(tmp_path / "sd-server")
    assert h3.h3_sibling_server_binary(None) is None


def test_slot_holds_the_managed_tree_while_its_server_lives(monkeypatch):
    started = []

    class _Srv:
        def __init__(self, binary):
            self.binary = binary
            self.alive = False
            self.stopped = False

        def start(
            self,
            files,
            *,
            offload = None,
            env = None,
            extra_args = None,
        ):
            started.append((files, offload, extra_args))
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False
            self.stopped = True

    monkeypatch.setattr(srv, "SdCppServer", _Srv)
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: True)
    monkeypatch.setattr(sd_cpp_backend, "_sd_cpp_backend", None, raising = False)
    slot = h3.H3NativeServerSlot(
        "/managed/sd-server", _FILES, ("--offload-to-cpu", "--diffusion-fa")
    )
    assert sd_cpp_backend._managed_tree_in_use() is False
    server = slot.get()
    assert slot.get() is server
    assert started == [(_FILES, ["--offload-to-cpu", "--diffusion-fa"], ["--rng", "cpu"])]
    assert sd_cpp_backend._managed_tree_in_use() is True
    with sd_cpp_backend._tree_claimed_for_install() as claimed:
        assert claimed is False
    slot.stop()
    assert server.stopped is True
    assert sd_cpp_backend._managed_tree_in_use() is False


def test_slot_respawns_a_dead_server(monkeypatch):
    class _Srv:
        def __init__(self, binary):
            self.alive = False

        def start(
            self,
            files,
            *,
            offload = None,
            env = None,
            extra_args = None,
        ):
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self):
            self.alive = False

    monkeypatch.setattr(srv, "SdCppServer", _Srv)
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: False)
    slot = h3.H3NativeServerSlot("/x/sd-server", _FILES, ())
    first = slot.get()
    first.alive = False
    second = slot.get()
    assert second is not first and second.is_alive()
    slot.stop()


class _LifeSrv:
    """A fake SdCppServer that records every spawn."""

    spawns: list = []

    def __init__(self, binary):
        self.binary = binary
        self.alive = False

    def start(
        self,
        files,
        *,
        offload = None,
        env = None,
        extra_args = None,
    ):
        _LifeSrv.spawns.append({"offload": offload, "env": env, "extra_args": extra_args})
        self.alive = True

    def is_alive(self):
        return self.alive

    def stop(self):
        self.alive = False


@pytest.fixture
def life(monkeypatch):
    _LifeSrv.spawns = []
    monkeypatch.setattr(srv, "SdCppServer", _LifeSrv)
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: False)
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_IDLE_ENV, raising = False)
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slots = []

    def make(**kwargs):
        slot = h3.H3NativeServerSlot("/x/sd-server", _FILES, ("--offload-to-cpu",), **kwargs)
        slots.append(slot)
        return slot

    yield make
    for slot in slots:
        slot.stop()


RESIDENT = ["--diffusion-fa", "--backend", "diffusion=CUDA1,te=CUDA1,vae=CUDA1"]
OFFLOAD = ["--offload-to-cpu", "--diffusion-fa", "--backend", "diffusion=CUDA1,te=CUDA1,vae=CUDA1"]
SAGE_ENV = {h3.H3_QUANT_CUBLAS_ENV: h3.H3_QUANT_CUBLAS_MIN_BATCH}


def test_slot_spawns_with_the_render_flags_and_env(life):
    slot = life()
    server = slot.get(RESIDENT + ["--sage-attn"], SAGE_ENV)
    assert _LifeSrv.spawns == [
        {"offload": RESIDENT + ["--sage-attn"], "env": SAGE_ENV, "extra_args": ["--rng", "cpu"]}
    ]
    assert slot.get(RESIDENT + ["--sage-attn"], dict(SAGE_ENV)) is server
    assert len(_LifeSrv.spawns) == 1
    assert slot.alive_signature()[2] == tuple(RESIDENT + ["--sage-attn"])


@pytest.mark.parametrize(
    "first, second",
    [
        ((RESIDENT, {}), (OFFLOAD, {})),
        ((RESIDENT, {}), (RESIDENT + ["--sage-attn"], SAGE_ENV)),
        ((RESIDENT + ["--sage-attn"], SAGE_ENV), (RESIDENT + ["--sage-attn"], {})),
    ],
)
def test_slot_respawns_when_the_signature_changes(life, first, second):
    slot = life()
    old = slot.get(*first)
    new = slot.get(*second)
    assert new is not old
    assert old.is_alive() is False
    assert [s["offload"] for s in _LifeSrv.spawns] == [first[0], second[0]]
    assert slot.last_release_reason == "signature changed"


def test_slot_signature_covers_the_model_files(life):
    a = life()
    b = h3.H3NativeServerSlot(
        "/x/sd-server",
        SdCppModelFiles(
            diffusion_model = "/m/q8.gguf", vae = _FILES.vae, audio_vae = _FILES.audio_vae, llm = _FILES.llm
        ),
    )
    try:
        assert a._signature_for(tuple(RESIDENT), ()) != b._signature_for(tuple(RESIDENT), ())
    finally:
        b.stop()


def test_idle_timeout_stops_the_server_and_leaves_no_timer(life, monkeypatch):
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_IDLE_ENV, "0.05")
    slot = life()
    slot.begin_render()
    server = slot.get(RESIDENT, {})
    assert slot.end_render() is None
    timer = slot._timer
    assert timer is not None and timer.daemon
    timer.join(2.0)
    assert not timer.is_alive()
    assert server.is_alive() is False and slot.is_alive() is False
    assert slot.last_release_reason == "idle timeout"
    assert slot._timer is None


def test_a_new_render_cancels_the_idle_timer(life, monkeypatch):
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_IDLE_ENV, "0.2")
    slot = life()
    slot.begin_render()
    server = slot.get(RESIDENT, {})
    slot.end_render()
    first = slot._timer
    slot.begin_render()
    assert slot._timer is None
    first.join(1.0)
    assert server.is_alive() is True
    slot._on_idle(slot._timer_token - 1)
    assert server.is_alive() is True
    slot.end_render()
    slot.stop()
    assert slot._timer is None and server.is_alive() is False


def test_stop_cancels_a_pending_timer(life, monkeypatch):
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_IDLE_ENV, "30")
    slot = life()
    slot.begin_render()
    slot.get(RESIDENT, {})
    slot.end_render()
    timer = slot._timer
    slot.stop("unload")
    timer.join(1.0)
    assert not timer.is_alive()


def test_idle_zero_stops_right_after_the_render(life, monkeypatch):
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_IDLE_ENV, "0")
    slot = life()
    slot.begin_render()
    server = slot.get(RESIDENT, {})
    assert slot.end_render() == "idle timeout 0"
    assert server.is_alive() is False


@pytest.mark.parametrize(
    "raw, expected", [("", 180.0), ("45", 45.0), ("-1", 180.0), ("abc", 180.0), ("0", 0.0)]
)
def test_idle_env_parsing(raw, expected):
    assert h3.h3_native_server_idle_s({h3.H3_NATIVE_SERVER_IDLE_ENV: raw}) == expected


def test_memory_pressure_stops_the_server_after_the_render(life):
    reasons = iter(["free VRAM 1 MiB below the 4096 MiB reserve"])
    slot = life(pressure_probe = lambda: next(reasons))
    slot.begin_render()
    server = slot.get(RESIDENT, {})
    assert slot.end_render().startswith("memory pressure: free VRAM")
    assert server.is_alive() is False
    assert slot._timer is None


def test_a_broken_probe_keeps_the_idle_bound(life, monkeypatch):
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_IDLE_ENV, "30")

    def boom():
        raise OSError("nvidia-smi missing")

    slot = life(pressure_probe = boom)
    slot.begin_render()
    server = slot.get(RESIDENT, {})
    assert slot.end_render() is None
    assert server.is_alive() is True and slot._timer is not None


def test_pressure_thresholds():
    gib = 1024**3
    ok = dict(
        vram_free = 40 * gib, vram_total = 80 * gib, host_available = 100 * gib, host_total = 200 * gib
    )
    assert h3.h3_native_server_pressure(**ok) is None
    assert "VRAM" in h3.h3_native_server_pressure(**{**ok, "vram_free": 11 * gib})
    # 16 GiB card: the 4 GiB floor wins over 15%.
    assert "VRAM" in h3.h3_native_server_pressure(
        **{**ok, "vram_free": 3 * gib, "vram_total": 16 * gib}
    )
    assert "host RAM" in h3.h3_native_server_pressure(**{**ok, "host_available": 29 * gib})
    assert (
        h3.h3_native_server_pressure(
            vram_free = None, vram_total = None, host_available = None, host_total = None
        )
        is None
    )


def test_release_stops_an_idle_server_and_defers_a_busy_one(life, monkeypatch):
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_IDLE_ENV, "30")
    idle = life()
    idle.begin_render()
    idle_server = idle.get(RESIDENT, {})
    idle.end_render()
    busy = life()
    busy.begin_render()
    busy_server = busy.get(RESIDENT, {})
    assert h3.release_h3_native_servers("export subprocess starting") >= 2
    assert idle_server.is_alive() is False and idle._timer is None
    assert idle.last_release_reason == "export subprocess starting"
    assert busy_server.is_alive() is True
    assert busy.end_render() == "export subprocess starting"
    assert busy_server.is_alive() is False


def test_gpu_arbiter_releases_idle_video_servers_for_other_owners(monkeypatch):
    from core.inference import gpu_arbiter

    seen = []
    monkeypatch.setattr(h3, "release_h3_native_servers", lambda reason: seen.append(reason) or 0)
    monkeypatch.setattr(gpu_arbiter, "_owner", None)
    monkeypatch.setattr(gpu_arbiter, "raise_if_other_accounts_active", lambda *a, **k: None)
    gpu_arbiter.acquire_for(gpu_arbiter.CHAT, account_id = "a")
    assert seen == ["GPU acquired for chat"]
    gpu_arbiter.acquire_for(gpu_arbiter.VIDEO, account_id = "a", allow_evict = True)
    assert seen == ["GPU acquired for chat"]


def test_render_hands_the_cli_flags_and_env_to_the_slot(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _FakeSlot(_FakeServer())
    _render(slot, tmp_path, flags = RESIDENT + ["--sage-attn"], env = SAGE_ENV)
    assert slot.got == [(RESIDENT + ["--sage-attn"], SAGE_ENV)]
    assert slot.renders == 1 and slot.ended == 1


def test_render_ends_the_slot_render_on_failure_too(tmp_path, monkeypatch):
    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    slot = _FakeSlot(_FakeServer(exc = RuntimeError("out of memory"), alive_after = True))
    with pytest.raises(RuntimeError):
        _render(slot, tmp_path)
    assert slot.ended == 1


def test_kill_switch_stops_a_live_server_and_uses_the_cli(tmp_path, monkeypatch, life):
    slot = life()
    slot.begin_render()
    server = slot.get(RESIDENT, {})
    slot.end_render()
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_ENV, "0")
    assert _render(slot, tmp_path) is None
    assert server.is_alive() is False and slot._timer is None


def _generate_backend(
    monkeypatch,
    tmp_path,
    slot,
    *,
    env = (),
):
    import dataclasses

    from test_h3_native_resident import AUTO_FLAGS, _backend_with_files

    calls: list = []
    backend = _backend_with_files(
        monkeypatch, tmp_path, calls, memory_mode = "auto", flags = AUTO_FLAGS
    )
    backend._state = dataclasses.replace(
        backend._state,
        pipe = dataclasses.replace(backend._state.pipe, server_slot = slot, env = tuple(env)),
    )
    return backend, calls, AUTO_FLAGS


class _StartFailsSlot(_FakeSlot):
    def __init__(self, live = None):
        super().__init__(start_exc = RuntimeError("no server"))
        self.live = live

    def alive_signature(self):
        return self.live


def test_generate_gives_the_server_and_the_fallback_cli_the_same_flags_and_env(
    monkeypatch, tmp_path
):
    import core.inference.video as video

    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    gib = 1024**3
    monkeypatch.setattr(video, "_h3_card_free_bytes", lambda device, ordinal: 90 * gib)
    slot = _StartFailsSlot()
    backend, calls, auto = _generate_backend(
        monkeypatch, tmp_path, slot, env = tuple(SAGE_ENV.items())
    )
    backend.generate(prompt = "a fox", width = 960, height = 544)
    resident = [f for f in auto if f not in ("--offload-to-cpu", "--stream-layers")]
    assert slot.got == [(resident, SAGE_ENV)]
    assert calls[0]["offload"] == resident and calls[0]["env"] == SAGE_ENV


def test_a_live_resident_server_is_not_pushed_back_to_offload_by_its_own_usage(
    monkeypatch, tmp_path
):
    import core.inference.video as video

    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    from test_h3_native_resident import AUTO_FLAGS

    gib = 1024**3
    resident = tuple(f for f in AUTO_FLAGS if f not in ("--offload-to-cpu", "--stream-layers"))
    slot = _StartFailsSlot(live = ("/x/sd-server", (), resident, ()))
    monkeypatch.setattr(video, "_h3_card_free_bytes", lambda *_a: 10 * gib)
    backend, calls, _ = _generate_backend(monkeypatch, tmp_path, slot)
    result = backend.generate(prompt = "a fox", width = 960, height = 544)
    assert slot.got[0][0] == list(resident)
    assert result["offload_policy"] == "none"


def test_a_live_resident_server_does_not_hide_a_clip_too_large_for_the_card(monkeypatch, tmp_path):
    import core.inference.video as video

    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    from test_h3_native_resident import AUTO_FLAGS

    gib = 1024**3
    resident = tuple(f for f in AUTO_FLAGS if f not in ("--offload-to-cpu", "--stream-layers"))
    slot = _StartFailsSlot(live = ("/x/sd-server", (), resident, ()))
    monkeypatch.setattr(video, "_h3_card_free_bytes", lambda *_a: 10 * gib)
    backend, calls, _ = _generate_backend(monkeypatch, tmp_path, slot)
    # 8x pixel volume: ~43 GiB of activations, more than the 10 GiB left.
    result = backend.generate(prompt = "a fox", width = 1920, height = 1088, num_frames = 241)
    assert "--offload-to-cpu" in slot.got[0][0]
    assert result["offload_policy"] == "group"


def test_release_does_not_wait_on_a_server_start(life, monkeypatch):
    monkeypatch.setenv(h3.H3_NATIVE_SERVER_IDLE_ENV, "30")
    slot = life()
    slot.begin_render()
    server = slot.get(RESIDENT, {})
    holder_ready, done = threading.Event(), threading.Event()

    def hold():
        with slot._lock:
            holder_ready.set()
            done.wait(5)

    th = threading.Thread(target = hold)
    th.start()
    holder_ready.wait(2)
    import time as _time

    t0 = _time.monotonic()
    assert slot.release("chat load") is True
    assert _time.monotonic() - t0 < 2.0
    done.set()
    th.join(2)
    assert slot.end_render() == "chat load"
    assert server.is_alive() is False


def test_cancel_aborts_a_server_start_in_progress(monkeypatch):
    """Unload and Cancel wait on the render, which holds the generate lock through the server's model load."""
    import time as _time

    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: False)
    loading = threading.Event()

    class _SlowSrv:
        def __init__(self, binary):
            self.aborted = threading.Event()
            self.stops = 0

        def start(
            self,
            files,
            *,
            offload = None,
            env = None,
            extra_args = None,
        ):
            loading.set()
            if not self.aborted.wait(10):
                raise AssertionError("start was never aborted")
            raise SdCppCancelled("sd-server startup was cancelled.")

        def stop(self):
            self.stops += 1
            self.aborted.set()

        def is_alive(self):
            return False

    monkeypatch.setattr(srv, "SdCppServer", _SlowSrv)
    slot = h3.H3NativeServerSlot("/x/sd-server", _FILES)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    t0 = _time.monotonic()
    with pytest.raises(SdCppCancelled):
        slot.get(RESIDENT, {}, cancel_event = cancel)
    assert loading.is_set() and _time.monotonic() - t0 < 5
    assert slot.alive_signature() is None and slot.disabled_reason is None
    with pytest.raises(SdCppCancelled):
        slot.get(RESIDENT, {}, cancel_event = cancel)


def test_a_live_resident_server_keeps_the_committed_offload_when_the_card_is_unreadable(
    monkeypatch, tmp_path
):
    import core.inference.video as video

    monkeypatch.delenv(h3.H3_NATIVE_SERVER_ENV, raising = False)
    from test_h3_native_resident import AUTO_FLAGS

    resident = tuple(f for f in AUTO_FLAGS if f not in ("--offload-to-cpu", "--stream-layers"))
    slot = _StartFailsSlot(live = ("/x/sd-server", (), resident, ()))
    monkeypatch.setattr(video, "_h3_card_free_bytes", lambda *_a: None)
    backend, calls, _ = _generate_backend(monkeypatch, tmp_path, slot)
    backend.generate(prompt = "a fox", width = 960, height = 544)
    assert slot.got[0][0] == list(AUTO_FLAGS)


def test_host_reserve_is_a_share_of_the_cgroup_limit(monkeypatch):
    import core.inference.diffusion_memory as dm
    import core.inference.video as video

    gib_mib = 1024
    monkeypatch.setattr(video, "_h3_card_memory_bytes", lambda *_a: (None, None))
    monkeypatch.setattr(dm, "_system_memory_mib", lambda: (512 * gib_mib, 400 * gib_mib))
    monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: 20 * gib_mib)
    monkeypatch.setattr(dm, "_cgroup_memory_limit_mib", lambda: 32 * gib_mib)
    assert video._h3_native_server_pressure("cuda", None) is None
    monkeypatch.setattr(dm, "_cgroup_memory_limit_mib", lambda: None)
    assert "host RAM" in video._h3_native_server_pressure("cuda", None)


def test_the_tree_stays_held_until_the_server_has_exited(monkeypatch):
    monkeypatch.setattr("core.inference.sd_cpp_engine.is_managed_binary", lambda b: True)
    seen = []

    class _ExitingSrv:
        def __init__(self, binary):
            self.alive = False

        def start(
            self,
            files,
            *,
            offload = None,
            env = None,
            extra_args = None,
        ):
            self.alive = True

        def is_alive(self):
            return self.alive

        def stop(self):
            seen.append(sd_cpp_backend._external_tree_holder_alive())
            self.alive = False

    monkeypatch.setattr(srv, "SdCppServer", _ExitingSrv)
    slot = h3.H3NativeServerSlot("/x/sd-server", _FILES)
    slot.get(RESIDENT, {})
    slot.release("chat load")
    assert seen == [True]
    assert sd_cpp_backend._external_tree_holder_alive() is False
