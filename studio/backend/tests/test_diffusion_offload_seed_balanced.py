# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two small-card load failures, Qwen-Image-2.1 sized, CPU only.

1. An offloading plan must seed a pre-quantized (int8 / fp8) denoiser on the host: seeding it onto the GPU whole
   (~7 GiB int8) left the 8 GB streaming hooks no room for their first block.
2. An explicit ``memory_mode=balanced`` is fit-checked against the loaded companions, walking down the streamed
   tiers, instead of loading every companion resident and OOMing at the first generate.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import core.inference.diffusion_memory as dm
from core.inference.diffusion_memory import (
    BALANCED_FIT_CHECK_ENV,
    DeviceMemory,
    OFFLOAD_GROUP,
    OFFLOAD_MODEL,
    OFFLOAD_NONE,
    OFFLOAD_STREAMING,
    PREQUANT_SEED_ON_HOST_ENV,
    plan_diffusion_memory,
    prequant_seed_device,
    refine_balanced_plan_for_components,
    refine_memory_plan_for_components,
)

MIB = 1024 * 1024
BACKEND = Path(__file__).resolve().parents[1]

# Qwen-Image-2.1 auto (int8 DiT + fp8 TE), as planned and as loaded
DENSE = dict(model_dense_mib = 19630, companion_dense_mib = 12182, text_encoder_dense_mib = 10847)
# explicit int8 prices the encoder dense (16.7 GB) but loads fp8 (9.0 GB)
DENSE_EXPLICIT = dict(
    model_dense_mib = 31566, companion_dense_mib = 18024, text_encoder_dense_mib = 16689
)
TE_LOADED_MIB = 8960
VAE_LOADED_MIB = 645
DIT_LOADED_MIB = 7448


class _Target:
    supports_model_cpu_offload = True


def _card(free_mib, total_mib):
    return DeviceMemory("cuda", "cuda", "discrete_vram", free_mib, total_mib)


CARD_24 = _card(24176, 24576)
CARD_16 = _card(15976, 16376)
CARD_12 = _card(11888, 12288)
CARD_8 = _card(7788, 8188)


def _plan(
    card,
    mode = None,
    sizes = DENSE,
):
    return plan_diffusion_memory(
        target = _Target(),
        device_memory = card,
        runtime_headroom_mib = 8192,
        requested_mode = mode,
        **sizes,
    )


def _module(mib):
    m = torch.nn.Module()
    m.w = torch.nn.Parameter(
        torch.empty(mib * MIB, dtype = torch.uint8, device = "meta"), requires_grad = False
    )
    return m


class _Pipe:
    def __init__(self):
        self.transformer = _module(DIT_LOADED_MIB)
        self.text_encoder = _module(TE_LOADED_MIB)
        self.vae = _module(VAE_LOADED_MIB)
        self.components = {
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "vae": self.vae,
            "tokenizer": object(),
            "scheduler": object(),
        }


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(PREQUANT_SEED_ON_HOST_ENV, raising = False)
    monkeypatch.delenv(BALANCED_FIT_CHECK_ENV, raising = False)
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)


@pytest.mark.parametrize("card", [CARD_8, CARD_12, CARD_16])
def test_offloading_auto_plan_seeds_int8_on_the_host(card):
    plan = (
        dm.torchao_streaming_plan(_plan(card))
        if _plan(card).offload_policy == OFFLOAD_MODEL
        else _plan(card)
    )
    assert plan.offload_policy != OFFLOAD_NONE
    assert not dm.plan_keeps_transformer_resident(plan)
    assert prequant_seed_device(plan, "cuda", "int8") == "cpu"
    assert prequant_seed_device(plan, "cuda:1", "fp8") == "cpu"


def test_resident_plans_keep_the_gpu_seed():
    plan24 = _plan(CARD_24)
    assert dm.plan_keeps_transformer_resident(plan24)
    assert prequant_seed_device(plan24, "cuda", "int8") == "cuda"
    full = _plan(_card(47000, 49140))
    assert full.offload_policy == OFFLOAD_NONE
    assert prequant_seed_device(full, "cuda:0", "int8") == "cuda:0"


def test_seed_device_scheme_and_kill_switch(monkeypatch):
    plan = dm.torchao_streaming_plan(_plan(CARD_8))
    assert prequant_seed_device(plan, "cuda", "nvfp4") == "cuda"
    assert prequant_seed_device(plan, "cuda", None) == "cuda"
    monkeypatch.setenv(PREQUANT_SEED_ON_HOST_ENV, "0")
    assert prequant_seed_device(plan, "cuda", "int8") == "cuda"


def test_denoiser_seed_forwards_the_placement(monkeypatch):
    import core.inference.diffusion_denoiser_prequant as dp
    import core.inference.diffusion_prequant as prequant
    import sys
    import types

    # CPU runners lack diffusers; an empty stub module serves the class lookup by name
    if "diffusers" not in sys.modules:
        try:
            import diffusers  # noqa: F401
        except ImportError:
            monkeypatch.setitem(sys.modules, "diffusers", types.ModuleType("diffusers"))
    diffusers = sys.modules["diffusers"]

    seen = {}

    def fake_load(cls, base, source, **kwargs):
        seen.update(kwargs)
        return torch.nn.Linear(2, 2)

    class _Fam:
        name = "qwen-image-2.1"
        transformer_class = "QwenImage21Transformer2DModel"

    monkeypatch.setattr(prequant, "load_prequantized_transformer", fake_load)
    monkeypatch.setattr(dp, "pipeline_seed_supported", lambda fam: True)
    monkeypatch.setattr(dp, "denoiser_prequant_source", lambda *a, **k: object())
    monkeypatch.setattr(diffusers, "QwenImage21Transformer2DModel", object, raising = False)
    out = dp.denoiser_prequant_pipe_kwargs(
        _Fam(),
        "Qwen/Qwen-Image-2.1",
        scheme = "int8",
        dtype = torch.bfloat16,
        device = "cuda",
        placement_device = "cpu",
    )
    assert out
    assert seen["device"] == "cuda"
    assert seen["placement_device"] == "cpu"


def test_loader_materialises_on_the_placement_device():
    src = (BACKEND / "core/inference/diffusion_prequant.py").read_text(encoding = "utf-8")
    assert "transformer = transformer.to(placement_device or device)" in src
    assert "transformer = transformer.to(device)\n" not in src


def test_pipeline_seed_call_passes_the_plan_placement():
    tree = ast.parse((BACKEND / "core/inference/diffusion.py").read_text(encoding = "utf-8"))
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "denoiser_prequant_pipe_kwargs"
    ]
    assert calls, "pipeline seed call not found"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "placement_device" in kw
        value = kw["placement_device"]
        assert (
            isinstance(value, ast.Call)
            and getattr(value.func, "id", None) == "prequant_seed_device"
        )
        assert [getattr(a, "id", None) for a in value.args] == [
            "plan",
            "device",
            "pipeline_seed_scheme",
        ]


def test_balanced_that_fits_is_unchanged():
    # dense TE estimate says companions do not fit, the loaded ones do: keep the plan
    plan = _plan(CARD_24, "balanced", DENSE_EXPLICIT)
    assert plan.offload_policy == OFFLOAD_GROUP and not plan.stream_text_encoders
    out = refine_balanced_plan_for_components(_Pipe(), plan)
    assert out.offload_policy == OFFLOAD_GROUP
    assert not out.stream_text_encoders
    assert out.reasons == plan.reasons


@pytest.mark.parametrize("sizes", [DENSE, DENSE_EXPLICIT])
def test_balanced_16gb_streams_the_text_encoders(sizes):
    plan = _plan(CARD_16, "balanced", sizes)
    out = refine_balanced_plan_for_components(_Pipe(), plan)
    assert out.offload_policy == OFFLOAD_GROUP
    assert out.stream_text_encoders
    assert out.stream_transformer


def test_balanced_12gb_drops_to_whole_module_offload():
    plan = _plan(CARD_12, "balanced", DENSE_EXPLICIT)
    out = refine_memory_plan_for_components(
        _Pipe(), refine_balanced_plan_for_components(_Pipe(), plan)
    )
    assert out.offload_policy == OFFLOAD_MODEL
    assert out.vae_tiling


def test_balanced_8gb_streams_granularly():
    plan = _plan(CARD_8, "balanced", DENSE_EXPLICIT)
    out = refine_memory_plan_for_components(
        _Pipe(), refine_balanced_plan_for_components(_Pipe(), plan)
    )
    assert out.offload_policy == OFFLOAD_STREAMING


@pytest.mark.parametrize("card", [CARD_12, CARD_8])
def test_balanced_torchao_denoiser_streams_instead_of_model_offload(monkeypatch, card):
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    out = refine_balanced_plan_for_components(_Pipe(), _plan(card, "balanced"))
    assert out.offload_policy == OFFLOAD_STREAMING


def test_balanced_kill_switch_and_scope(monkeypatch):
    plan16 = _plan(CARD_16, "balanced")
    auto16 = _plan(CARD_16)
    assert refine_balanced_plan_for_components(_Pipe(), auto16) is auto16
    monkeypatch.setenv(BALANCED_FIT_CHECK_ENV, "0")
    assert refine_balanced_plan_for_components(_Pipe(), plan16) is plan16


def test_balanced_refinement_is_wired_before_component_refinement():
    src = (BACKEND / "core/inference/diffusion.py").read_text(encoding = "utf-8")
    assert (
        "refine_memory_plan_for_components(\n                        pipe, refine_balanced_plan_for_components(pipe, plan)\n"
        in src
    )


def test_measured_placement_honours_the_legacy_cpu_offload_flag():
    src = (BACKEND / "core/inference/diffusion.py").read_text(encoding = "utf-8")
    call = src.index("plan = refine_plan_from_loaded_weights(")
    guard = src.rindex("\n", 0, src.rindex("\n", 0, call))
    assert (
        "if not cpu_offload or normalize_memory_mode(memory_mode) is not None:" in src[guard:call]
    )
