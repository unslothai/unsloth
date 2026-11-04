# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""First-paint catalogs use existing state without starting hardware or model runtimes."""

import asyncio
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def catalog(monkeypatch):
    from core.inference import defaults, model_slots
    from routes import inference, models
    from utils.hardware import hardware as hw

    event = threading.Event()
    monkeypatch.setattr(hw, "DETECTION_COMPLETE", event)
    monkeypatch.setattr(hw, "DEVICE", None)
    monkeypatch.setattr(hw, "CHAT_ONLY", True)
    monkeypatch.setattr(hw, "get_device", lambda: pytest.fail("catalog started hardware detection"))
    monkeypatch.setattr(models, "peek_inference_backend", lambda: None)
    monkeypatch.setattr(
        models,
        "get_inference_backend",
        lambda: pytest.fail("catalog created a runtime"),
        raising = False,  # Tripwire if the removed eager factory is called again.
    )
    monkeypatch.setattr(
        inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(is_loaded = False, model_identifier = None),
    )
    monkeypatch.setattr(model_slots, "visible", lambda: [])
    monkeypatch.setattr(models.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(models.account_access, "resident_hidden", lambda *args: False)
    return defaults, models, hw, event


def test_unknown_hardware_keeps_the_standard_catalog_without_a_cpu_verdict(catalog):
    defaults, models, _, _ = catalog
    response = asyncio.run(models.list_models(current_subject = "owner"))
    assert response.default_models == defaults.DEFAULT_MODELS_STANDARD
    assert [model.id for model in response.models] == defaults.DEFAULT_MODELS_STANDARD


@pytest.mark.parametrize("device,chat_only", [("cpu", True), ("cuda", False), ("mlx", False)])
def test_read_only_catalog_uses_a_measured_host_without_reprobing(
    catalog, monkeypatch, device, chat_only
):
    defaults, models, hw, event = catalog
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType(device))
    monkeypatch.setattr(hw, "CHAT_ONLY", chat_only)
    event.set()
    response = asyncio.run(models.list_models(current_subject = "owner"))
    if chat_only:
        expected = defaults.DEFAULT_MODELS_GGUF
    elif device == "mlx":
        expected = defaults.suggestions_for_host(
            defaults.DEFAULT_MODELS_STANDARD, hw.DeviceType.MLX
        )
        assert all(not name.endswith("bnb-4bit") for name in expected)
    else:
        expected = defaults.DEFAULT_MODELS_STANDARD
    assert response.default_models == expected


def test_catalog_updates_after_first_hardware_detection(catalog, monkeypatch):
    defaults, models, hw, event = catalog
    before = asyncio.run(models.list_models(current_subject = "owner"))
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CPU)
    event.set()
    after = asyncio.run(models.list_models(current_subject = "owner"))
    assert before.default_models == defaults.DEFAULT_MODELS_STANDARD
    assert after.default_models == defaults.DEFAULT_MODELS_GGUF


def test_existing_runtime_keeps_its_ranked_catalog_and_loaded_models(catalog, monkeypatch):
    _, models, _, _ = catalog
    backend = SimpleNamespace(
        default_models = ["org/ranked", "org/curated"],
        models = {"org/loaded": {"is_vision": True}},
        active_model_name = "org/loaded",
    )
    monkeypatch.setattr(models, "peek_inference_backend", lambda: backend)
    response = asyncio.run(models.list_models(current_subject = "owner"))
    assert response.default_models == ["org/ranked", "org/curated"]
    assert [model.id for model in response.models] == ["org/ranked", "org/curated", "org/loaded"]
    assert response.models[-1].is_vision


def test_managed_account_catalog_filter_and_hidden_resident_are_preserved(catalog, monkeypatch):
    _, models, _, _ = catalog
    backend = SimpleNamespace(
        default_models = ["org/private", "org/shared"],
        models = {"org/resident": {}},
        active_model_name = "org/resident",
    )
    monkeypatch.setattr(models, "peek_inference_backend", lambda: backend)
    monkeypatch.setattr(models.account_access, "managed_account", lambda: True)
    monkeypatch.setattr(models.account_access, "model_visible", lambda name: name == "org/shared")
    monkeypatch.setattr(models.account_access, "resident_hidden", lambda *args: True)
    response = asyncio.run(models.list_models(current_subject = "managed"))
    assert response.default_models == ["org/shared"]
    assert [model.id for model in response.models] == ["org/shared"]


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_video_status_provisional_catalog_refreshes_after_explicit_detection(monkeypatch, device):
    from core.inference import video
    from utils.hardware import hardware as hw

    event = threading.Event()
    monkeypatch.setattr(hw, "DETECTION_COMPLETE", event)
    monkeypatch.setattr(hw, "DEVICE", None)
    calls, resolves = [], []
    ordinary = SimpleNamespace(name = "ordinary", modular_workflow = False)
    modular = SimpleNamespace(name = "modular", modular_workflow = True)

    def available(*, device):
        calls.append(device)
        return (ordinary,) if device == "mps" else (ordinary, modular)

    def resolve():
        resolves.append(device)
        assert event.is_set(), "idle status initialized the device"
        return SimpleNamespace(device = device)

    monkeypatch.setattr(video, "pipeline_available_video_families", available)
    monkeypatch.setattr(video, "resolve_diffusion_device_target", resolve)
    video._video_family_capabilities.cache_clear()
    try:
        backend = video.VideoBackend()
        assert backend.status()["supported_families"] == ["ordinary", "modular"]
        assert backend.status()["modular_families"] == ["modular"]
        assert resolves == [] and calls == [None]
        event.set()
        assert backend.status()["supported_families"] == ["ordinary", "modular"]
        assert resolves == [], "a torn-down device is not a settled snapshot"
        monkeypatch.setattr(
            hw, "DEVICE", hw.DeviceType.MLX if device == "mps" else hw.DeviceType.CPU
        )
        expected = ["ordinary"] if device == "mps" else ["ordinary", "modular"]
        assert backend.status()["supported_families"] == expected
        assert backend.status()["supported_families"] == expected
        assert resolves == [device] and calls == [None, device]
    finally:
        video._video_family_capabilities.cache_clear()


def test_an_event_without_a_device_is_not_a_measured_catalog(catalog):
    defaults, models, _, event = catalog
    event.set()
    response = asyncio.run(models.list_models(current_subject = "owner"))
    assert response.default_models == defaults.DEFAULT_MODELS_STANDARD


@pytest.mark.parametrize("arch", ["flux", "lumina2"])
def test_local_gguf_listing_defers_the_device_probe_until_hardware_is_known(
    catalog, monkeypatch, arch
):
    from core.inference import diffusion_engine_router, diffusion_families
    from hub.services.models import catalog_classification

    _, _, _, event = catalog
    probed = []
    monkeypatch.setattr(diffusion_families, "detect_family_for_pick", lambda hint: hint)
    monkeypatch.setattr(diffusion_families, "family_gguf_loadable", lambda family: True)
    monkeypatch.setattr(
        diffusion_engine_router,
        "family_buildable_here",
        lambda family, **kwargs: probed.append(family) or False,
    )
    classify = lambda: catalog_classification._arch_to_task(arch, ("model",))
    assert classify() == "text-to-image"
    assert probed == []
    event.set()
    assert classify() == catalog_classification._UNSUPPORTED_DIFFUSION_TASK
    assert probed == ["model"]
