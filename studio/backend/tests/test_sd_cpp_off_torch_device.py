# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UNSLOTH_DIFFUSION_SD_CPP_DEVICE; R9700 + RTX 3080 host, all hardware answers stubbed."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from core.inference import diffusion_engine_router as r
from core.inference import sd_cpp_backend
from core.inference.diffusion_families import detect_family
from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

_ENVS = (
    "UNSLOTH_DIFFUSION_ENGINE",
    "UNSLOTH_DIFFUSION_SD_CPP",
    "UNSLOTH_DIFFUSION_SD_CPP_MPS",
    "UNSLOTH_DIFFUSION_SD_CPP_INSTALL",
    "UNSLOTH_DIFFUSION_SD_CPP_DEVICE",
)

_MIXED_HOST = {
    "unknown": False,
    "unanswered": [],
    "devices": [
        {"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 3080"},
        {"vendor": "amd", "index": 0, "name": "AMD Radeon AI PRO R9700"},
    ],
}


@pytest.fixture(autouse = True)
def _pinned_host(monkeypatch):
    for e in _ENVS:
        monkeypatch.delenv(e, raising = False)
    monkeypatch.setattr(r, "_off_torch_warned", set())
    monkeypatch.setattr(
        r,
        "resolve_diffusion_device_target",
        lambda: SimpleNamespace(backend = "rocm", device = "cuda"),
    )
    monkeypatch.setattr(r, "_physical_inventory", lambda: _MIXED_HOST)
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "cuda")
    monkeypatch.setattr(
        r,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None}, unload = lambda: None
        ),
    )
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: None)
    monkeypatch.setattr(r, "_server_binary_runnable", lambda *_a, **_k: True)
    monkeypatch.setattr(r, "SdCppEngine", lambda **_: SimpleNamespace(version = lambda: "sd-cli v0"))
    saved_engine = r._active_engine_name
    saved_reason = r._fallback_reason
    try:
        yield
    finally:
        r._active_engine_name = saved_engine
        r._fallback_reason = saved_reason


def _record_cli_requests(monkeypatch, path = "/opt/sd/sd-cli"):
    asked: list[str] = []

    def _ensure(**kwargs):
        asked.append(kwargs["accelerator"])
        return path

    monkeypatch.setattr(r, "ensure_sd_cpp_binary", _ensure)
    return asked


def test_unset_is_today():
    assert r.off_torch_sd_cpp_device() is None


def test_nvidia_on_a_rocm_torch_names_the_card(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    device = r.off_torch_sd_cpp_device()
    assert device == r.OffTorchDevice(vendor = "nvidia", index = 0, accelerator = "cuda")
    assert device.child_env() == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}


def test_a_cuda_torch_already_drives_nvidia(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia:0")
    assert r.off_torch_sd_cpp_device("cuda") is None


@pytest.mark.parametrize("raw", ["amd", "nvidia:x", "nvidia:-1", "intel:0"])
def test_values_it_cannot_act_on_are_ignored(monkeypatch, raw):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", raw)
    assert r.off_torch_sd_cpp_device() is None


def test_a_card_the_inventory_does_not_list_is_ignored(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia:1")
    assert r.off_torch_sd_cpp_device() is None


@pytest.mark.parametrize(
    "inventory",
    [
        {"unknown": True, "devices": []},
        {"unknown": False, "unanswered": ["nvidia"], "devices": []},
    ],
)
def test_an_unanswered_probe_trusts_the_explicit_setting(monkeypatch, inventory):
    monkeypatch.setattr(r, "_physical_inventory", lambda: inventory)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia:1")
    assert r.off_torch_sd_cpp_device().index == 1


def test_another_vendors_unanswered_probe_does_not_vouch_for_the_card(monkeypatch):
    inventory = {
        "unknown": True,
        "unanswered": ["amd"],
        "devices": [{"vendor": "nvidia", "index": 0, "name": "NVIDIA GeForce RTX 3080"}],
    }
    monkeypatch.setattr(r, "_physical_inventory", lambda: inventory)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia:1")
    assert r.off_torch_sd_cpp_device() is None
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia:0")
    assert r.off_torch_sd_cpp_device().index == 0


def test_only_the_image_path_changes_build(monkeypatch):
    """Video still installs by torch's backend: its loads are placed by torch."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    assert r.image_install_accelerator("rocm") == "cuda"
    assert r._install_accelerator_for("rocm") == "rocm"


def test_mixed_host_with_the_setting_runs_cuda_sd_cpp(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    asked = _record_cli_requests(monkeypatch)
    r.select_and_activate_engine(detect_family("z-image"))
    assert r.active_engine_name() == ENGINE_SD_CPP
    assert asked == ["cuda"]


def test_same_host_without_the_setting_stays_on_diffusers(monkeypatch):
    asked = _record_cli_requests(monkeypatch)
    r.select_and_activate_engine(detect_family("z-image"))
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert "uses diffusers" in (r.active_status()["fallback_reason"] or "")
    assert asked == []


def test_torchs_ordinal_is_not_carried_to_the_other_vendor(monkeypatch):
    """gpu_ordinal 1 is an AMD card here; naming it to a CUDA build would pick a different card."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    _record_cli_requests(monkeypatch)
    seen: list = []
    monkeypatch.setattr(r, "_selected_card", lambda ordinal: seen.append(ordinal))
    r.select_and_activate_engine(detect_family("z-image"), gpu_ordinal = 1)
    assert seen == [None]


def test_no_cuda_build_falls_back_and_says_so(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    _record_cli_requests(monkeypatch, path = None)
    r.select_and_activate_engine(detect_family("z-image"))
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    reason = r.active_status()["fallback_reason"] or ""
    assert "binary unavailable" in reason
    assert "UNSLOTH_DIFFUSION_SD_CPP_DEVICE=nvidia:0 not honoured" in reason


def test_a_family_without_native_assets_is_not_rerouted(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    asked = _record_cli_requests(monkeypatch)
    r.select_and_activate_engine(detect_family("sdxl"))
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert asked == []


def test_explicit_opt_outs_still_win(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    _record_cli_requests(monkeypatch)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP", "0")
    r.select_and_activate_engine(detect_family("z-image"))
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    monkeypatch.delenv("UNSLOTH_DIFFUSION_SD_CPP")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_ENGINE", "diffusers")
    r.select_and_activate_engine(detect_family("z-image"))
    assert r.active_engine_name() == ENGINE_DIFFUSERS


def test_prediction_agrees_with_selection(monkeypatch):
    """The download plan is built from the prediction, so it has to follow the setting too."""
    _record_cli_requests(monkeypatch)
    fam = detect_family("z-image")
    assert r.predict_engine(fam, model_kind = "gguf") == ENGINE_DIFFUSERS
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    assert r.predict_engine(fam, model_kind = "gguf", gpu_ordinal = 1) == ENGINE_SD_CPP


def _state(**overrides):
    fields = dict(
        repo_id = "unsloth/Z-Image-Turbo-GGUF",
        base_repo = "Tongyi-MAI/Z-Image-Turbo",
        family = detect_family("z-image"),
        device = "cuda",
        files = SimpleNamespace(),
    )
    fields.update(overrides)
    return sd_cpp_backend._SdState(**fields)


def test_a_torch_placed_load_spawns_with_the_inherited_env():
    assert _state().spawn_env() is None


def test_an_off_torch_load_pins_its_card_at_spawn():
    env = r.OffTorchDevice(vendor = "nvidia", index = 1, accelerator = "cuda").child_env()
    state = _state(off_torch_device = "nvidia:1", child_env = tuple(sorted(env.items())))
    assert state.spawn_env() == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"}


def test_every_spawn_site_passes_the_pin():
    load = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._run_load)
    assert "env = spawn_env," in load
    assert 'if mode == "server" and off_torch is None' in load
    restart = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._restart_server_on_cpu_backend)
    assert "env = state.spawn_env()," in restart
    oneshot = inspect.getsource(sd_cpp_backend.SdCppDiffusionBackend._generate_oneshot)
    assert "env = state.spawn_env()," in oneshot


def test_the_resident_flag_the_trainer_reads():
    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    backend._state = None
    assert backend.runs_off_torch_device is False
    backend._state = _state()
    assert backend.runs_off_torch_device is False
    backend._state = _state(off_torch_device = "nvidia:0")
    assert backend.runs_off_torch_device is True


def test_the_load_route_skips_the_arbiter_only_once_native_is_active():
    import routes.inference as inference_routes

    source = inspect.getsource(inference_routes)
    start = source.index("off_torch = await asyncio.to_thread(off_torch_sd_cpp_device)")
    window = source[start : start + 25000]
    assert "if off_torch is None:\n            _guard_diffusion_load_against_training()" in window
    settle = window.index("if off_torch is not None:\n            if activated == ENGINE_SD_CPP:")
    assert window.index("activated = active_engine_name()") < settle
    assert "needs_gpu = False" in window[settle : settle + 300]
    assert "_guard_diffusion_load_against_training()" in window[settle : settle + 400]
    assert settle < window.index("if needs_gpu:")


def test_training_keeps_an_off_torch_images_model():
    import routes.training as training_routes
    source = inspect.getsource(training_routes)
    assert 'getattr(diffusion, "runs_off_torch_device", False) is True' in source


def _drive_load_route(
    monkeypatch,
    *,
    predicted,
    training_active,
    video_engine = None,
    video_loading = None,
):
    """Run the real image-load route with every hardware and engine answer stubbed; returns the
    route calls in order, plus what the route raised."""
    import asyncio

    from fastapi import HTTPException

    from core.inference import diffusion, diffusion_compat, diffusion_device
    from core.inference import gpu_arbiter as arb
    from hub.services.models import account_access
    from models.inference import DiffusionLoadRequest
    from routes import inference as route

    calls: list[str] = []
    backend = SimpleNamespace(
        validate_load_request = lambda *_a, **_k: detect_family("z-image"),
        preflight_base_access = lambda *_a, **_k: None,
        assert_precision_available = lambda *_a, **_k: None,
        begin_load = lambda *_a, **_k: calls.append("begin_load") or {},
    )

    async def _no_ordinal(*_a, **_k):
        return None

    def _guard():
        calls.append("training_guard")
        if training_active:
            raise HTTPException(status_code = 409, detail = "training")

    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    monkeypatch.setattr(arb, "_owner", None)
    monkeypatch.setattr(diffusion, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(
        diffusion_device,
        "resolve_diffusion_device_target",
        lambda *_a, **_k: SimpleNamespace(backend = "rocm", device = "cuda"),
    )
    monkeypatch.setattr(r, "predict_engine", lambda *_a, **_k: predicted)
    monkeypatch.setattr(r, "engine_for", lambda *_a: backend)
    monkeypatch.setattr(
        r,
        "select_and_activate_engine",
        lambda *_a, **_k: calls.append("select_and_activate") or backend,
    )
    monkeypatch.setattr(r, "active_engine_name", lambda: predicted)
    monkeypatch.setattr(r, "begin_load_on", lambda _e, start: start())
    monkeypatch.setattr(diffusion_compat, "assert_pick_is_not_speech", lambda *_a, **_k: None)
    monkeypatch.setattr(route, "_guard_diffusion_load_against_training", _guard)
    monkeypatch.setattr(route, "_selected_gpu_ordinal", _no_ordinal)
    monkeypatch.setattr(route, "_assert_native_precision_unset", lambda **_k: None)
    monkeypatch.setattr(route, "_repo_is_in_the_hub_cache", lambda *_a: True)
    monkeypatch.setattr(account_access, "admit_media_load", lambda _k, fn, *_a: fn())
    monkeypatch.setattr(account_access, "note_resident_components", lambda *_a, **_k: None)

    monkeypatch.setattr(
        "core.inference.video.get_video_backend",
        lambda: SimpleNamespace(
            status = lambda: {"engine": video_engine},
            unload = lambda: calls.append("video_unload"),
            _loading = video_loading,
        ),
    )
    monkeypatch.setattr(arb, "release", lambda owner: calls.append(f"release:{owner}"))

    request = DiffusionLoadRequest(model_path = "org/image", gguf_filename = "model.gguf")
    try:
        asyncio.run(route.load_diffusion_model_gated(request, "tester"))
    except HTTPException as exc:
        return calls, exc
    return calls, None


def test_a_load_predicted_for_torch_is_refused_during_training_before_the_engine_switch(
    monkeypatch,
):
    """A refused load must not unload the resident model: activating diffusers would do exactly that."""
    calls, raised = _drive_load_route(monkeypatch, predicted = ENGINE_DIFFUSERS, training_active = True)
    assert raised is not None and raised.status_code == 409
    assert "select_and_activate" not in calls


def test_an_off_torch_native_load_is_admitted_during_training(monkeypatch):
    calls, raised = _drive_load_route(monkeypatch, predicted = ENGINE_SD_CPP, training_active = True)
    assert raised is None, raised
    assert "training_guard" not in calls
    assert calls[-1] == "begin_load"


def test_diffusion_training_keeps_an_off_torch_images_model(monkeypatch):
    import routes.training as training_routes
    from core.inference import gpu_arbiter

    unloaded: list[str] = []
    resident = SimpleNamespace(
        runs_off_torch_device = True,
        is_loaded = True,
        unload = lambda: unloaded.append("images"),
    )
    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: resident)
    monkeypatch.setattr(gpu_arbiter, "release", lambda owner: unloaded.append(f"release:{owner}"))
    monkeypatch.setattr(
        "core.inference.video.get_video_backend",
        lambda: SimpleNamespace(status = lambda: {"loaded": False}, unload = lambda: None),
    )
    monkeypatch.setattr("routes.training_vram.summarize_resident_chat", lambda: {"any": False})
    training_routes._free_gpu_for_diffusion_training()
    assert "images" not in unloaded
    assert f"release:{gpu_arbiter.DIFFUSION}" not in unloaded

    resident.runs_off_torch_device = False
    training_routes._free_gpu_for_diffusion_training()
    assert "images" in unloaded


@pytest.mark.parametrize("installed", ["rocm", "vulkan", "cpu", None])
def test_a_leftover_build_of_another_accelerator_is_not_treated_as_off_torch(
    monkeypatch, installed
):
    """Offline, installs off or no CUDA asset: the ensure returns the tree's build, which reads
    CUDA_VISIBLE_DEVICES as HIP's mask (ROCm) or ignores it, landing on torch's card unguarded."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    _record_cli_requests(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: installed)
    r.select_and_activate_engine(detect_family("z-image"))
    assert r.active_engine_name() == ENGINE_DIFFUSERS
    assert "not honoured" in (r.active_status()["fallback_reason"] or "")


def test_the_cuda_build_is_still_accepted(monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    _record_cli_requests(monkeypatch)
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "cuda")
    r.select_and_activate_engine(detect_family("z-image"))
    assert r.active_engine_name() == ENGINE_SD_CPP


def test_the_load_refuses_to_spawn_another_accelerators_build(monkeypatch):
    device = r.OffTorchDevice(vendor = "nvidia", index = 0, accelerator = "cuda")
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    with pytest.raises(RuntimeError, match = "needs the cuda"):
        sd_cpp_backend._refuse_off_torch_build_mismatch(device, "/opt/sd/sd-server")
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: None)
    with pytest.raises(RuntimeError, match = "unrecorded"):
        sd_cpp_backend._refuse_off_torch_build_mismatch(device, "/opt/sd/sd-server")
    sd_cpp_backend._refuse_off_torch_build_mismatch(None, "/opt/sd/sd-server")


def test_an_in_flight_off_torch_load_counts_as_off_torch():
    backend = sd_cpp_backend.SdCppDiffusionBackend.__new__(sd_cpp_backend.SdCppDiffusionBackend)
    backend._state = None
    backend._loading = sd_cpp_backend._SdLoading(
        repo_id = "org/m", base_repo = "org/m", off_torch_device = "nvidia:0"
    )
    assert backend.runs_off_torch_device is True
    backend._state = _state()
    assert backend.runs_off_torch_device is False
    backend._state = None
    backend._loading.error = "boom"
    assert backend.runs_off_torch_device is False
    backend._loading = sd_cpp_backend._SdLoading(repo_id = "org/m", base_repo = "org/m")
    assert backend.runs_off_torch_device is False


@pytest.mark.parametrize("install_allowed", [False, True])
def test_prediction_filters_a_leftover_build_like_selection(monkeypatch, install_allowed):
    """A wrong prediction skips the route's pre-selection training guard and stages the wrong files."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_INSTALL", "1" if install_allowed else "0")
    _record_cli_requests(monkeypatch)
    monkeypatch.setattr(r, "ensure_sd_server_binary", lambda **_: "/opt/sd/sd-server")
    monkeypatch.setattr(sd_cpp_backend, "_installed_accelerator_of", lambda _b: "rocm")
    predicted = r.predict_engine(detect_family("z-image"), model_kind = "gguf")
    assert predicted == (ENGINE_SD_CPP if install_allowed else ENGINE_DIFFUSERS)
    if not install_allowed:
        r.select_and_activate_engine(detect_family("z-image"))
        assert r.active_engine_name() == predicted


@pytest.mark.parametrize(
    "family, kind, evicted", [("minimax-h3", "gguf", True), ("ltx-2", "pipeline", False)]
)
def test_a_native_video_load_frees_the_off_torch_images_engine(monkeypatch, family, kind, evicted):
    """An off-torch image server holds the managed sd.cpp tree without owning DIFFUSION."""
    import asyncio

    from core.inference import diffusion_compat, diffusion_device, video
    from hub.services.models import account_access
    from models.inference import VideoLoadRequest
    from routes import video as video_route

    unloaded: list[str] = []
    images = SimpleNamespace(runs_off_torch_device = True, unload = lambda: unloaded.append("images"))
    backend = SimpleNamespace(
        validate_load_request = lambda *_a, **_k: SimpleNamespace(name = family, base_repo = None),
        begin_load = lambda *_a, **_k: {},
    )

    async def _no_ordinal(*_a, **_k):
        return None

    monkeypatch.setattr(r, "get_active_diffusion_engine", lambda: images)
    monkeypatch.setattr(video, "get_video_backend", lambda: backend)
    monkeypatch.setattr(video, "assert_video_precision_available", lambda *_a, **_k: None)
    monkeypatch.setattr(video, "resolve_video_model_kind", lambda *_a, **_k: kind, raising = False)
    monkeypatch.setattr(
        video_route, "resolve_video_model_kind", lambda *_a, **_k: kind, raising = False
    )
    monkeypatch.setattr(
        diffusion_device,
        "resolve_diffusion_device_target",
        lambda *_a, **_k: SimpleNamespace(device = "cpu", backend = "cpu"),
    )
    monkeypatch.setattr(diffusion_compat, "assert_pick_is_not_speech", lambda *_a, **_k: None)
    monkeypatch.setattr(video_route, "_guard_video_load_against_training", lambda: None)
    monkeypatch.setattr(video_route, "_selected_gpu_ordinal", _no_ordinal)
    monkeypatch.setattr(account_access, "admit_media_load", lambda _k, fn, *_a: fn())
    monkeypatch.setattr(account_access, "note_resident_components", lambda *_a, **_k: None)

    request = VideoLoadRequest(model_path = "org/video", gguf_filename = "model.gguf")
    asyncio.run(video_route.load_video_model_gated(request, "tester"))
    assert unloaded == (["images"] if evicted else [])


@pytest.mark.parametrize(
    "video_engine, evicted", [("sd_cpp", True), ("diffusers", False), (None, False)]
)
def test_an_off_torch_image_load_frees_a_native_video_model(monkeypatch, video_engine, evicted):
    """The install it may run replaces the build a native H3 model was loaded from."""
    calls, raised = _drive_load_route(
        monkeypatch, predicted = ENGINE_SD_CPP, training_active = False, video_engine = video_engine
    )
    assert raised is None, raised
    assert ("video_unload" in calls) is evicted
    assert calls.index("select_and_activate") > (calls.index("video_unload") if evicted else -1)


def test_a_refused_fallback_keeps_the_resident_engine(monkeypatch):
    """A native prediction whose selection then falls back must be refused before _activate unloads."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    _record_cli_requests(monkeypatch, path = None)
    activated: list = []
    monkeypatch.setattr(r, "_activate", lambda *a, **_k: activated.append(a))

    def _refuse():
        raise RuntimeError("training is running")

    with pytest.raises(RuntimeError, match = "training"):
        r.select_and_activate_engine(detect_family("z-image"), before_fallback = _refuse)
    assert activated == []
    r.select_and_activate_engine(detect_family("z-image"))
    assert activated and activated[-1][0] == ENGINE_DIFFUSERS


@pytest.mark.parametrize(
    "repos, evicted", [(("unsloth/MiniMax-H3-GGUF",), True), (("org/ltx",), False)]
)
def test_an_in_flight_native_video_load_is_cancelled_too(monkeypatch, repos, evicted):
    from core.inference import video_minimax_h3

    h3_repos = (video_minimax_h3.H3_GGUF_REPO,) if evicted else repos
    loading = SimpleNamespace(error = None, asset_repos = h3_repos)
    calls, raised = _drive_load_route(
        monkeypatch, predicted = ENGINE_SD_CPP, training_active = False, video_loading = loading
    )
    assert raised is None, raised
    assert ("video_unload" in calls) is evicted
