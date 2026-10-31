# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A UNet too large for whole-module offload streams leaf by leaf; a streamed UNet skips the deferred speed profile."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from core.inference import diffusion_memory as dm  # noqa: E402

SDXL_UNET_MIB = 4897
SDXL_VAE_MIB = 159


def _pipe(denoiser = "unet", unet_mib = 4):
    """Diffusers-shaped pipe: a UNet (or transformer) of about ``unet_mib`` MiB of fp32 weights, a VAE, encoders."""
    comps = {
        denoiser: torch.nn.Linear(512, 512 * unet_mib, bias = False),
        "vae": torch.nn.Linear(8, 8),
        "text_encoder": torch.nn.Linear(8, 8),
    }
    calls = []
    pipe = types.SimpleNamespace(components = comps, model_offload_calls = calls, transformer = None)
    for name, module in comps.items():
        setattr(pipe, name, module)
    pipe.enable_model_cpu_offload = lambda device = None: calls.append(device)
    pipe.to = lambda device = None: calls.append(("to", device))
    return pipe


@pytest.fixture
def cuda_free(monkeypatch):
    """Fake one CUDA card with ``state['free']`` MiB free; record _apply_group_offload calls."""
    state = {"free": 0, "group": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device = None: (state["free"] << 20, 24 << 30)
    )

    def fake_group(pipe, device, logger, **kw):
        state["group"].append((device, kw))
        return True

    monkeypatch.setattr(dm, "_apply_group_offload", fake_group)
    monkeypatch.setattr(dm, "keep_cpu_weights_on_offload", lambda pipe, logger = None: 0)
    monkeypatch.delenv(dm.UNET_STREAM_ENV, raising = False)
    return state


def _plan(policy = dm.OFFLOAD_MODEL):
    return dm.MemoryPlan(
        requested_mode = "auto",
        offload_policy = policy,
        vae_tiling = False,
        vae_slicing = False,
        device_memory = None,
        estimates = {},
    )


@pytest.mark.parametrize("budget_mib, free_mib", [(4096, 3684), (6144, 5700)])
def test_sdxl_streams_where_whole_module_offload_cannot_fit(budget_mib, free_mib):
    room = dm.unet_stream_room_mib(free_mib, SDXL_UNET_MIB, SDXL_VAE_MIB)
    assert (
        room is not None
    ), f"{budget_mib} MiB budget: whole-module offload of the 4.8 GiB UNet OOMs"
    reserve = (
        dm._measured_reserve_mib(dm._MEASURED_UNET_STREAMED_PEAK_MIB) + dm._UNET_STREAM_SLACK_MIB
    )
    assert room == max(0, free_mib - SDXL_VAE_MIB - reserve)
    # the reserve covers the measured worst phase of a streamed render with margin
    assert reserve >= dm._MEASURED_UNET_STREAMED_PEAK_MIB * dm._MEASURED_PEAK_MARGIN


@pytest.mark.parametrize("free_mib", [7700, 15000, 22000])
def test_sdxl_keeps_whole_module_offload_when_it_fits(free_mib):
    # 8 GiB budget and up: main's whole-module offload (or the resident plan) stays as it was
    assert dm.unet_stream_room_mib(free_mib, SDXL_UNET_MIB, SDXL_VAE_MIB) is None


def test_more_free_vram_keeps_more_leaves_resident():
    rooms = [
        dm.unet_stream_room_mib(f, SDXL_UNET_MIB, SDXL_VAE_MIB) for f in (3684, 4500, 5000, 5700)
    ]
    assert rooms == sorted(rooms) and rooms[-1] > rooms[0]


def test_fp32_decode_of_an_fp16_vae_keeps_fewer_leaves_resident():
    # T4: diffusers decodes a force_upcast fp16 VAE in fp32, a larger measured peak than a bf16 decode
    for free_mib in (5700, 8000 - 1):
        r16 = dm.unet_stream_room_mib(free_mib, 7000, SDXL_VAE_MIB)
        r32 = dm.unet_stream_room_mib(free_mib, 7000, SDXL_VAE_MIB, fp32_decode = True)
        assert r16 is not None and r32 is not None and r32 < r16
    assert dm._MEASURED_UNET_STREAMED_FP32_DECODE_PEAK_MIB > dm._MEASURED_UNET_STREAMED_PEAK_MIB


@pytest.mark.parametrize(
    "dtype, force_upcast, fp32",
    [(torch.float16, True, True), (torch.float16, False, False), (torch.bfloat16, True, False)],
)
def test_vae_decode_dtype_follows_the_pipeline_upcast(dtype, force_upcast, fp32):
    vae = torch.nn.Linear(4, 4).to(dtype)
    vae.config = types.SimpleNamespace(force_upcast = force_upcast)
    assert dm._vae_decodes_in_fp32(vae) is fp32


def test_streamed_room_is_passed_to_group_offload(cuda_free):
    pipe = _pipe(unet_mib = 4)
    cuda_free["free"] = 8000
    # an 8000 MiB (meta) UNet on 8000 MiB free: whole-module offload cannot fit it, the leftover stays resident
    pipe.unet = pipe.components["unet"] = torch.nn.Linear(
        512, 512 * 8000, bias = False, device = "meta"
    )
    policy, _ = dm.apply_memory_plan(pipe, _plan(), device = "cuda")
    assert policy == dm.OFFLOAD_GROUP
    ((_, kw),) = cuda_free["group"]
    expected = dm.unet_stream_room_mib(8000, 8000, 0)
    assert expected and kw["resident_transformer_mib"] == expected


def test_model_offload_plan_streams_a_unet_that_cannot_fit(cuda_free):
    pipe = _pipe(unet_mib = 4)
    cuda_free["free"] = 2  # MiB: below the UNet plus the measured denoise reserve
    policy, _ = dm.apply_memory_plan(pipe, _plan(), device = "cuda")
    assert policy == dm.OFFLOAD_GROUP
    assert pipe.model_offload_calls == []
    ((device, kw),) = cuda_free["group"]
    assert device == "cuda" and kw["denoiser"] == "unet" and kw["stream_text_encoders"] is True
    assert pipe._unsloth_unet_streamed is True
    assert pipe._unsloth_applied_plan.offload_policy == dm.OFFLOAD_GROUP


def test_model_offload_plan_unchanged_when_the_unet_fits(cuda_free):
    pipe = _pipe(unet_mib = 4)
    cuda_free["free"] = 20_000
    policy, _ = dm.apply_memory_plan(pipe, _plan(), device = "cuda")
    assert policy == dm.OFFLOAD_MODEL
    assert pipe.model_offload_calls == ["cuda"] and cuda_free["group"] == []
    assert not getattr(pipe, "_unsloth_unet_streamed", False)


def test_kill_switch_restores_whole_module_offload(cuda_free, monkeypatch):
    monkeypatch.setenv(dm.UNET_STREAM_ENV, "0")
    pipe = _pipe(unet_mib = 4)
    cuda_free["free"] = 2
    policy, _ = dm.apply_memory_plan(pipe, _plan(), device = "cuda")
    assert policy == dm.OFFLOAD_MODEL
    assert pipe.model_offload_calls == ["cuda"] and cuda_free["group"] == []


def test_transformer_pipelines_and_resident_plans_never_stream(cuda_free):
    cuda_free["free"] = 2
    dit = _pipe(denoiser = "transformer", unet_mib = 4)
    assert dm._stream_unet_instead_of_model_offload(dit, "cuda") is False
    resident = _pipe(unet_mib = 4)
    policy, _ = dm.apply_memory_plan(resident, _plan(dm.OFFLOAD_NONE), device = "cuda")
    assert policy == dm.OFFLOAD_NONE
    assert cuda_free["group"] == [] and resident.model_offload_calls == [("to", "cuda")]


def test_unet_streaming_failure_falls_back_to_whole_module_offload(cuda_free, monkeypatch):
    monkeypatch.setattr(dm, "_apply_group_offload", lambda pipe, device, logger, **kw: False)
    pipe = _pipe(unet_mib = 4)
    cuda_free["free"] = 2
    policy, _ = dm.apply_memory_plan(pipe, _plan(), device = "cuda")
    assert policy == dm.OFFLOAD_MODEL
    assert pipe.model_offload_calls == ["cuda"]
    assert not getattr(pipe, "_unsloth_unet_streamed", False)


def test_cpu_placement_never_streams(cuda_free):
    cuda_free["free"] = 2
    assert dm._stream_unet_instead_of_model_offload(_pipe(unet_mib = 4), "cpu") is False
    assert cuda_free["group"] == []


def test_deferred_speed_profile_skipped_only_for_a_streamed_unet(monkeypatch):
    from core.inference import diffusion as dmod

    ran = []
    monkeypatch.setattr(
        dmod.DiffusionBackend, "_engage_deferred_profile", lambda self, state: ran.append(state)
    )
    monkeypatch.setattr(dmod, "invalidate_cfg_replica", lambda *a, **k: None)
    monkeypatch.setattr(dmod, "drop_displaced_cfg_router", lambda *a, **k: None)
    backend = object.__new__(dmod.DiffusionBackend)
    states = {}
    for streamed in (True, False):
        pipe = types.SimpleNamespace()
        if streamed:
            pipe._unsloth_unet_streamed = True
        states[streamed] = types.SimpleNamespace(
            pipe = pipe,
            speed_deferred = True,
            deferred_static_plan = None,
            resolved = {},
            transformer_cache = None,
        )
        dmod.DiffusionBackend._engage_deferred_speed(backend, states[streamed])
        assert states[streamed].speed_deferred is False
    # the streamed UNet stays eager (its weights must stay under the leaf hooks); every other load still compiles
    assert ran == [states[False]]
