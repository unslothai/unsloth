# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UNSLOTH_DIFFUSION_SD_CPP_DEVICE: native image generation on a card torch cannot see.

The host these pin is a ROCm torch beside an NVIDIA card (R9700 + RTX 3080). Every hardware answer
is stubbed: the torch backend, the physical inventory and the binaries, so no test asks the machine
it runs on."""

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


# ── the setting ───────────────────────────────────────────────────────────────


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


def test_only_the_image_path_changes_build(monkeypatch):
    """Video still installs by torch's backend: its loads are placed by torch."""
    monkeypatch.setenv("UNSLOTH_DIFFUSION_SD_CPP_DEVICE", "nvidia")
    assert r.image_install_accelerator("rocm") == "cuda"
    assert r._install_accelerator_for("rocm") == "rocm"


# ── selection: the matched pair ───────────────────────────────────────────────


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


# ── the backend: every spawn opens the named card ─────────────────────────────


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


# ── routes: nothing on torch's card is taken ──────────────────────────────────


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
