# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unified-memory load refusal (#14267) and the per-load ``allow_oversized_load`` opt-in."""

from __future__ import annotations

import asyncio
import threading
import types

import pytest

from core.inference import diffusion as diffusion_mod
from core.inference import diffusion_memory as dm
from core.inference import video as video_mod
from core.inference.diffusion import DiffusionBackend, _LoadingState
from core.inference.diffusion_device import DiffusionDeviceTarget
from core.inference.diffusion_memory import (
    UNIFIED_OVERSIZE_ENV,
    UNIFIED_OVERSIZE_REFUSAL_CODE,
    DeviceMemory,
    UnifiedMemoryShortfallError,
    plan_diffusion_memory,
    raise_on_unified_memory_shortfall,
    scoped_allow_oversized_load,
    unified_memory_refusal_code,
    unified_memory_shortfall_message,
    unified_oversize_requested,
    unified_oversize_scoped,
)

MIB = 1024 * 1024
GIB_MIB = 1024

# Qwen/Qwen-Image sizes from Hub metadata.
QWEN_IMAGE = [
    ("transformer/diffusion_pytorch_model.safetensors", 40_861_227_138),
    ("text_encoder/model.safetensors", 16_584_475_660),
    ("vae/diffusion_pytorch_model.safetensors", 253_807_696),
]
MAC_64_GIB = 64 * GIB_MIB
MAC_64_WORKING_SET = 48 * GIB_MIB


@pytest.fixture(autouse = True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv(UNIFIED_OVERSIZE_ENV, raising = False)


def _target(device: str = "mps") -> DiffusionDeviceTarget:
    return DiffusionDeviceTarget(
        device = device,
        dtype = "bfloat16",
        backend = device,
        vendor = "apple" if device == "mps" else "nvidia",
        supports_model_cpu_offload = device == "cuda",
        supports_default_torch_compile = False,
        supports_pinned_transfer = device == "cuda",
    )


def _backend(
    monkeypatch,
    snapshot: DeviceMemory,
    *,
    ram_total_mib: int | None = None,
):
    backend = DiffusionBackend()
    target = _target(snapshot.device)
    monkeypatch.setattr(backend, "_target_for_ordinal", lambda *_a, **_k: target)
    monkeypatch.setattr(diffusion_mod, "snapshot_device_memory", lambda _t: snapshot)
    if ram_total_mib is not None:
        monkeypatch.setattr(dm, "_system_memory_mib", lambda: (ram_total_mib, 1024))
    return backend


def _qwen_verdict(backend, files = QWEN_IMAGE):
    fam = types.SimpleNamespace(name = "qwen-image", base_repo = "Qwen/Qwen-Image")
    return backend.declared_footprint_shortfall(
        fam, "Qwen/Qwen-Image", "Qwen/Qwen-Image", kind = "pipeline", declared_files = files
    )


def _plan(
    weights_mib: int,
    *,
    total_mib: int = 16 * GIB_MIB,
    companions_mib = None,
):
    memory = DeviceMemory("mps", "mps", "unified_memory", total_mib, total_mib)
    return plan_diffusion_memory(
        target = _target(),
        device_memory = memory,
        model_dense_mib = weights_mib,
        runtime_headroom_mib = 0,
        companion_dense_mib = companions_mib,
    )


# --- sizing -----------------------------------------------------------------------------------------------------


def test_qwen_image_on_a_64_gib_unified_pool_reproduces_the_report_and_names_the_room_left(
    monkeypatch,
):
    # The issue's numbers: 55,026 MiB weights + 2 GiB overhead vs 64 GiB minus the 20% reserve.
    snapshot = DeviceMemory("mps", "mps", "unified_memory", MAC_64_GIB, MAC_64_GIB)
    message = _qwen_verdict(_backend(monkeypatch, snapshot))
    assert message is not None
    assert "needs about 56 GB of memory for its weights" in message
    assert "only about 51 GB is usable" in message
    assert "currently free" not in message
    assert "Its text encoders and VAE need about 18 GB" in message
    assert "main model of up to about 34 GB" in message


def test_the_pre_download_capacity_view_on_a_mac_does_not_depend_on_what_is_free_now(monkeypatch):
    # 44,200 MiB of weights: fits RAM minus 20% (52,428 MiB) but not 85% of the 48 GiB working set (41,779 MiB).
    files = [
        ("transformer/diffusion_pytorch_model.safetensors", 38_000 * MIB),
        ("text_encoder/model.safetensors", 6_000 * MIB),
        ("vae/diffusion_pytorch_model.safetensors", 200 * MIB),
    ]
    idle = DeviceMemory("mps", "mps", "unified_memory", 60 * GIB_MIB, MAC_64_GIB)
    assert _qwen_verdict(_backend(monkeypatch, idle), files) is None
    # With a model loaded the snapshot returns the working-set view; the capacity verdict must not flip.
    busy = DeviceMemory(
        "mps",
        "mps",
        "unified_memory",
        20 * GIB_MIB,
        MAC_64_WORKING_SET,
        metal_working_set_mib = MAC_64_WORKING_SET,
    )
    assert _qwen_verdict(_backend(monkeypatch, busy, ram_total_mib = MAC_64_GIB), files) is None


def test_a_load_past_ram_capacity_is_still_refused_from_the_working_set_view(monkeypatch):
    busy = DeviceMemory(
        "mps",
        "mps",
        "unified_memory",
        20 * GIB_MIB,
        MAC_64_WORKING_SET,
        metal_working_set_mib = MAC_64_WORKING_SET,
    )
    message = _qwen_verdict(_backend(monkeypatch, busy, ram_total_mib = MAC_64_GIB))
    assert message is not None and "about 51 GB is usable" in message


def test_capacity_view_keeps_the_working_set_when_ram_cannot_be_read(monkeypatch):
    busy = DeviceMemory(
        "mps",
        "mps",
        "unified_memory",
        1024,
        MAC_64_WORKING_SET,
        metal_working_set_mib = MAC_64_WORKING_SET,
    )
    monkeypatch.setattr(dm, "_system_memory_mib", lambda: (None, None))
    assert dm.unified_capacity_memory(busy).free_mib == MAC_64_WORKING_SET
    assert dm.unified_capacity_memory(busy).total_mib == MAC_64_WORKING_SET


def test_companion_split_matches_the_resident_total():
    from core.inference.diffusion_auto_policy import (
        companion_bytes_from_declared,
        resident_bytes_from_declared,
    )

    companions = companion_bytes_from_declared("Qwen/Qwen-Image", QWEN_IMAGE)
    total = resident_bytes_from_declared("Qwen/Qwen-Image", QWEN_IMAGE)
    assert companions == 16_584_475_660 + 253_807_696
    assert total - companions == 40_861_227_138
    assert companion_bytes_from_declared("Qwen/Qwen-Image", QWEN_IMAGE[:1]) is None


def test_a_unet_is_not_counted_as_a_companion():
    from core.inference.diffusion_auto_policy import companion_bytes_from_declared

    companions = (
        ("text_encoder/model.safetensors", 246_144_152),
        ("text_encoder_2/model.safetensors", 1_389_382_176),
        ("vae/diffusion_pytorch_model.safetensors", 334_643_268),
    )
    sdxl = (("unet/diffusion_pytorch_model.safetensors", 5_135_149_760),) + companions
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    assert companion_bytes_from_declared(base, sdxl) == companion_bytes_from_declared(
        base, companions
    )


# --- the override ----------------------------------------------------------------------------------------------


def test_default_still_refuses_with_a_coded_runtime_error():
    plan = _plan(20 * GIB_MIB)
    with pytest.raises(UnifiedMemoryShortfallError) as excinfo:
        raise_on_unified_memory_shortfall(plan, family = "qwen-image")
    assert isinstance(excinfo.value, RuntimeError)
    assert excinfo.value.code == UNIFIED_OVERSIZE_REFUSAL_CODE == "unified-memory-load"
    message = str(excinfo.value)
    assert "Load anyway" in message
    assert "'Allow oversized model loads' in Settings > System > Model memory" in message
    assert "allow_oversized_load" in message
    assert f"{UNIFIED_OVERSIZE_ENV}=1" in message


def test_the_request_flag_lifts_the_refusal_for_its_own_scope_only():
    plan = _plan(20 * GIB_MIB)
    assert unified_memory_shortfall_message(plan) is not None
    with unified_oversize_requested(True):
        assert unified_memory_shortfall_message(plan) is None
        raise_on_unified_memory_shortfall(plan)  # must not raise
    assert unified_memory_shortfall_message(plan) is not None
    with unified_oversize_requested(False):
        assert unified_memory_shortfall_message(plan) is not None


def test_the_override_is_never_silent():
    plan = _plan(20 * GIB_MIB)
    warnings: list[tuple] = []
    logger = types.SimpleNamespace(
        warning = lambda *a: warnings.append(a), error = lambda *a: None, info = lambda *a: None
    )
    with unified_oversize_requested(True):
        raise_on_unified_memory_shortfall(plan, family = "qwen-image", logger = logger)
    assert len(warnings) == 1
    assert warnings[0][1] == "allow_oversized_load"
    with unified_oversize_requested(True):
        raise_on_unified_memory_shortfall(_plan(1024), logger = logger)
    assert len(warnings) == 1


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_the_env_var_still_works(monkeypatch, value):
    plan = _plan(20 * GIB_MIB)
    monkeypatch.setenv(UNIFIED_OVERSIZE_ENV, value)
    assert unified_memory_shortfall_message(plan) is None
    raise_on_unified_memory_shortfall(plan)


def test_the_flag_does_not_leak_into_another_thread():
    plan = _plan(20 * GIB_MIB)
    seen: list = []
    with unified_oversize_requested(True):
        worker = threading.Thread(
            target = lambda: seen.append(unified_memory_shortfall_message(plan))
        )
        worker.start()
        worker.join()
    assert seen and seen[0] is not None


def test_asyncio_to_thread_carries_the_flag_into_the_plan_worker():
    plan = _plan(20 * GIB_MIB)

    async def _plan_route():
        with unified_oversize_requested(True):
            return await asyncio.to_thread(unified_memory_shortfall_message, plan)

    assert asyncio.run(_plan_route()) is None


def test_the_plan_route_wrapper_scopes_the_worker_call():
    plan = _plan(20 * GIB_MIB)

    async def _plan_route(allowed):
        return await asyncio.to_thread(
            unified_oversize_scoped(allowed, unified_memory_shortfall_message), plan
        )

    assert asyncio.run(_plan_route(True)) is None
    assert asyncio.run(_plan_route(False)) is not None


def test_scoped_decorator_reads_the_kwarg_and_nests():
    @scoped_allow_oversized_load
    def probe(**kwargs):
        return dm._unified_oversize_override()

    assert probe() is False
    assert probe(allow_oversized_load = True) is True
    with unified_oversize_requested(True):
        assert probe(allow_oversized_load = False) is True


def test_the_code_survives_a_wrapping_exception():
    try:
        try:
            raise UnifiedMemoryShortfallError("too big")
        except UnifiedMemoryShortfallError as inner:
            raise RuntimeError("load failed") from inner
    except RuntimeError as outer:
        assert unified_memory_refusal_code(outer) == UNIFIED_OVERSIZE_REFUSAL_CODE
    assert unified_memory_refusal_code(RuntimeError("other")) is None
    assert unified_memory_refusal_code(None) is None


# --- API plumbing ----------------------------------------------------------------------------------------------


def test_request_models_default_off_and_accept_the_flag():
    from models.inference import (
        DiffusionDownloadPlanResponse,
        DiffusionLoadProgressResponse,
        DiffusionLoadRequest,
        VideoLoadProgressResponse,
        VideoLoadRequest,
    )

    assert DiffusionLoadRequest(model_path = "Qwen/Qwen-Image").allow_oversized_load is False
    assert DiffusionLoadRequest(
        model_path = "Qwen/Qwen-Image", allow_oversized_load = True
    ).allow_oversized_load
    assert VideoLoadRequest(model_path = "Lightricks/LTX-2").allow_oversized_load is False
    assert VideoLoadRequest(
        model_path = "Lightricks/LTX-2", allow_oversized_load = True
    ).allow_oversized_load
    assert DiffusionLoadProgressResponse(
        phase = "error", error = "x", error_code = "unified-memory-load"
    ).error_code
    assert VideoLoadProgressResponse(phase = "error", error = "x").error_code is None
    assert DiffusionDownloadPlanResponse(incompatible_code = "unified-memory-load").incompatible_code


def test_image_begin_load_hands_the_flag_to_the_worker(monkeypatch):
    backend = DiffusionBackend()
    fam = types.SimpleNamespace(
        name = "qwen-image", base_repo = "Qwen/Qwen-Image", single_file_is_pipeline = False
    )
    monkeypatch.setattr(backend, "validate_load_request", lambda *_a, **_k: fam)
    monkeypatch.setattr(backend, "assert_precision_available", lambda *_a, **_k: None)
    monkeypatch.setattr(backend, "status", lambda: {})
    monkeypatch.setattr(diffusion_mod, "auto_cuda_ordinal_for_load", lambda: None)
    started: list[dict] = []

    def _thread(*, target, kwargs, **_k):
        started.append(kwargs)
        return types.SimpleNamespace(start = lambda: None)

    monkeypatch.setattr(diffusion_mod, "account_thread", _thread)
    backend.begin_load("Qwen/Qwen-Image", allow_oversized_load = True)
    assert started[-1]["allow_oversized_load"] is True
    backend._loading = None
    backend.begin_load("Qwen/Qwen-Image")
    assert started[-1]["allow_oversized_load"] is False


def _refusing_detector(seen: list):
    def _detect(*_a, **_k):
        seen.append(dm._unified_oversize_override())
        raise UnifiedMemoryShortfallError("'qwen-image' needs about 56 GB")

    return _detect


def test_image_worker_reports_the_code_and_scopes_the_flag(monkeypatch):
    backend = DiffusionBackend()
    seen: list = []
    monkeypatch.setattr(diffusion_mod, "detect_family_for_pick", _refusing_detector(seen))
    for allowed in (False, True):
        backend._load_token += 1
        backend._loading = _LoadingState(repo_id = "Qwen/Qwen-Image", base_repo = "Qwen/Qwen-Image")
        backend._run_load(
            repo_id = "Qwen/Qwen-Image", _load_token = backend._load_token, allow_oversized_load = allowed
        )
        progress = backend.load_progress()
        assert progress["phase"] == "error"
        assert progress["error_code"] == UNIFIED_OVERSIZE_REFUSAL_CODE
    assert seen == [False, True]
    assert dm._unified_oversize_override() is False


def test_image_worker_leaves_other_failures_uncoded(monkeypatch):
    backend = DiffusionBackend()

    def _boom(*_a, **_k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(diffusion_mod, "detect_family_for_pick", _boom)
    backend._load_token += 1
    backend._loading = _LoadingState(repo_id = "x", base_repo = "y")
    backend._run_load(repo_id = "x", _load_token = backend._load_token)
    assert backend.load_progress()["error_code"] is None


def test_video_worker_reports_the_code_and_scopes_the_flag(monkeypatch):
    backend = video_mod.VideoBackend()
    seen: list = []
    monkeypatch.setattr(video_mod, "_detect_load_family", _refusing_detector(seen))
    for allowed in (False, True):
        backend._load_token += 1
        backend._loading = video_mod._VideoLoadingState(
            repo_id = "Lightricks/LTX-2", base_repo = "Lightricks/LTX-2"
        )
        backend._run_load(
            repo_id = "Lightricks/LTX-2",
            _load_token = backend._load_token,
            allow_oversized_load = allowed,
        )
        progress = backend.load_progress()
        assert progress["phase"] == "error"
        assert progress["error_code"] == UNIFIED_OVERSIZE_REFUSAL_CODE
    assert seen == [False, True]
