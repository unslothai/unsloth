# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cold startup must stay cold until an authenticated feature needs hardware."""

import asyncio
import threading
import types
from contextlib import nullcontext

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def cold(monkeypatch):
    import main

    monkeypatch.setattr(main, "_startup_ml_warmup_features", frozenset())
    monkeypatch.setattr(main, "_hardware_detection_requested", False)
    monkeypatch.setattr(main._hw_module, "DEVICE", None)
    monkeypatch.setattr(main._hw_module, "DETECTION_COMPLETE", threading.Event())
    monkeypatch.setattr(main, "_superseded_by_mlx_repair", lambda snapshot: False)
    monkeypatch.delenv(main.DISABLE_ENV_VAR, raising = False)
    return main


def test_cold_health_and_liveness_do_not_start_detection(cold, monkeypatch):
    starts = []
    monkeypatch.setattr(cold, "start_background_detection", lambda: starts.append(True))
    app = FastAPI()
    app.add_api_route("/api/health", cold.health_check)
    app.add_api_route("/api/liveness", cold.liveness_check)
    with TestClient(app) as client:
        for _ in range(3):
            for path in ("/api/health", "/api/liveness"):
                reply = client.get(path)
                assert reply.status_code == 200
                assert reply.json()["hardware_detection_deferred"] is True
    assert starts == []


def test_deferred_authed_health_still_names_the_host_platform(cold, monkeypatch):
    async def subject(request):
        return "owner"

    monkeypatch.setattr(cold, "_health_subject", subject)
    app = FastAPI()
    app.add_api_route("/api/health", cold.health_check)
    with TestClient(app) as client:
        reply = client.get("/api/health").json()
    assert reply["hardware_detection_deferred"] is True
    assert reply["device_type"] == {"darwin": "mac", "win32": "windows"}.get(
        cold.sys.platform, "linux"
    )
    assert "apple_silicon" in reply and "file_manager" in reply


def test_anonymous_request_cannot_trigger_hardware_detection(cold, monkeypatch):
    starts = []
    monkeypatch.setattr(cold, "start_background_detection", lambda: starts.append(True))
    app = FastAPI()
    app.add_api_route("/api/health", cold.health_check)
    with TestClient(app) as client:
        reply = client.get("/api/health?detect_hardware=true")
    assert reply.status_code == 401
    assert not cold._hardware_detection_requested
    assert starts == []


def test_authenticated_demand_resolves_and_keeps_the_hardware_verdict(cold, monkeypatch):
    starts = []

    async def subject(request):
        return "owner"

    def detect():
        starts.append(True)
        cold._hw_module.DEVICE = cold._hw_module.DeviceType.CUDA
        cold._hw_module.DETECTION_COMPLETE.set()

    monkeypatch.setattr(cold, "_health_subject", subject)
    monkeypatch.setattr(cold, "start_background_detection", detect)
    monkeypatch.setattr(
        cold,
        "_hardware_snapshot",
        lambda: (False, None, None) if cold._hw_module.DETECTION_COMPLETE.is_set() else None,
    )
    app = FastAPI()
    app.add_api_route("/api/health", cold.health_check)
    with TestClient(app) as client:
        for path in ("/api/health?detect_hardware=true", "/api/health"):
            reply = client.get(path)
            assert reply.status_code == 200
            assert reply.json()["chat_only"] is False
            assert "hardware_detection_deferred" not in reply.json()
            assert "hardware_detecting" not in reply.json()
    assert starts == [True]


def test_opt_in_starts_detection_but_legacy_disable_still_wins(cold, monkeypatch):
    starts = []
    monkeypatch.setattr(cold, "start_background_detection", lambda: starts.append(True))
    monkeypatch.setattr(
        cold,
        "_startup_ml_warmup_features",
        frozenset({"hardware", "text", "datasets", "media", "quantization"}),
    )
    asyncio.run(cold._await_hardware_detection(0))
    assert starts == [True]
    monkeypatch.setenv(cold.DISABLE_ENV_VAR, "1")
    asyncio.run(cold._await_hardware_detection(0))
    assert starts == [True]
    assert cold._hardware_detection_deferred()


@pytest.fixture
def maintenance(cold, monkeypatch):
    from core.training import account_jobs
    from utils import mlx_repair

    calls = []
    monkeypatch.setattr(cold, "join_background_warm", lambda: calls.append("join"))
    monkeypatch.setattr(cold._hw_module, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(
        cold, "_start_linked_folder_auto_sync", lambda generation: calls.append("folders")
    )
    monkeypatch.setattr(account_jobs, "startup_reconciliation_accounts", lambda: [])
    monkeypatch.setattr(mlx_repair, "start_mlx_autorepair_if_needed", lambda: calls.append("mlx"))
    monkeypatch.setattr(cold, "_refresh_dense_quant_capability", lambda: calls.append("dense"))
    monkeypatch.setattr(
        cold, "_refresh_quantised_streaming_capability", lambda: calls.append("streaming")
    )
    monkeypatch.setattr(
        cold,
        "prewarm_diffusers_if_image_models_exist",
        lambda **kwargs: calls.append(("diffusers", kwargs)),
    )
    monkeypatch.setattr(
        cold, "prewarm_quant_probe_if_image_models_exist", lambda: calls.append("quantization")
    )
    monkeypatch.setattr(cold, "background_media_import", lambda: nullcontext(True))
    # Even an unrelated request importing torch must not authorize speculative imports.
    monkeypatch.setitem(cold.sys.modules, "torch", types.ModuleType("torch"))
    return cold, calls


def test_default_keeps_maintenance_without_speculative_ml_imports(maintenance):
    main, calls = maintenance
    main._post_warm_background_work()
    assert calls == ["join", "folders"]


def test_opt_in_keeps_all_warmup_stages(maintenance, monkeypatch):
    main, calls = maintenance
    monkeypatch.setattr(
        main,
        "_startup_ml_warmup_features",
        frozenset({"hardware", "text", "datasets", "media", "quantization"}),
    )
    main._post_warm_background_work()
    assert calls == [
        "join",
        "mlx",
        "dense",
        "streaming",
        "folders",
        ("diffusers", {"warm_quantization": False}),
        "quantization",
    ]


@pytest.mark.parametrize("retired", [False, True])
def test_cold_mlx_repair_waits_for_demand_and_honors_shutdown(maintenance, monkeypatch, retired):
    main, calls = maintenance
    from utils import mlx_repair

    repaired = threading.Event()
    workers = []

    def worker(**kwargs):
        thread = threading.Thread(**kwargs)
        workers.append(thread)
        return thread

    monkeypatch.setattr(main, "threading", types.SimpleNamespace(Thread = worker))
    monkeypatch.setattr(main, "_post_warm_generation", 123)
    monkeypatch.setattr(main._hw_module, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(mlx_repair, "start_mlx_autorepair_if_needed", repaired.set)
    main._post_warm_background_work(123)
    assert calls == ["join", "folders"]
    assert not repaired.is_set()
    if retired:
        main._stop_post_warm_thread()
    main._hw_module.DETECTION_COMPLETE.set()
    for thread in workers:
        thread.join(2)
        assert not thread.is_alive()
    assert repaired.is_set() is (not retired)


@pytest.mark.parametrize("probe,window_open", [(False, True), (True, True), (True, False)])
def test_media_capabilities_are_probed_only_for_the_picker(cold, monkeypatch, probe, window_open):
    calls = []

    class HardwareSnapshotReached(Exception):
        pass

    def snapshot(*args, **kwargs):
        raise HardwareSnapshotReached

    monkeypatch.setattr(cold, "_get_cached_system_gpu_info", snapshot)
    monkeypatch.setattr(cold, "background_media_import", lambda: nullcontext(window_open))
    monkeypatch.setattr(cold, "gate_torch_stack_import", lambda *args: calls.append("gate"))
    monkeypatch.setattr(cold, "_refresh_dense_quant_capability", lambda: calls.append("dense"))
    monkeypatch.setattr(
        cold, "_refresh_quantised_streaming_capability", lambda: calls.append("streaming")
    )
    with pytest.raises(HardwareSnapshotReached):
        cold.get_system_info(current_subject = "owner", probe_media = probe)
    assert calls == (["gate"] if probe else []) + (
        ["dense", "streaming"] if probe and window_open else []
    )


@pytest.mark.parametrize("mask", range(32))
def test_post_warm_respects_every_feature_combination(maintenance, monkeypatch, mask):
    from utils.ml_warmup_settings import effective_ml_warmup_features

    main, calls = maintenance
    names = ("hardware", "text", "datasets", "media", "quantization")
    selected = {feature for bit, feature in enumerate(names) if mask & (1 << bit)}
    effective = effective_ml_warmup_features(selected)
    monkeypatch.setattr(main, "_startup_ml_warmup_features", effective)
    main._post_warm_background_work()
    expected = ["join"]
    if selected & {"hardware", "text", "media", "quantization"}:
        expected.append("mlx")
    if "media" in selected:
        expected.extend(["dense", "streaming"])
    expected.append("folders")
    if "media" in selected:
        expected.append(("diffusers", {"warm_quantization": False}))
    if "quantization" in selected:
        expected.append("quantization")
    assert calls == expected


def test_datasets_only_keeps_automatic_hardware_detection_deferred(cold, monkeypatch):
    monkeypatch.setattr(cold, "_startup_ml_warmup_features", frozenset({"datasets"}))
    monkeypatch.setattr(
        cold,
        "start_background_detection",
        lambda: pytest.fail("datasets-only warm started a hardware probe"),
    )
    assert cold._ml_warmup_enabled()
    assert cold._hardware_detection_deferred()
    assert asyncio.run(cold._await_hardware_detection(0)) is False
