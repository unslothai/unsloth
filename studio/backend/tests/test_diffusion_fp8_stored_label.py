# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Planner identity of an fp8-STORED denoiser: its own learned-peak label, packed verdict, dense labels unchanged."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

torch = pytest.importorskip("torch")

ENVS = (
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION",
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE",
    "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
)
KLEIN_LOADED = {
    "transformer": (3700, "dit"),
    "text_encoder": (7673, "text_encoder"),
    "vae": (160, "other"),
}
KLEIN_FLAT = dict(model_dense_mib = 15258, companion_dense_mib = 7820, text_encoder_dense_mib = 7629)
T4_TOTAL = 15095


class _Dit(torch.nn.Module):
    """A diffusers-like denoiser: ``dtype`` reads the first floating parameter."""

    def __init__(self, dtype, stored: bool):
        super().__init__()
        from core.inference.diffusion_comfy_quant import FP8_STORED_ATTR
        from core.inference.diffusion_native_quant import NATIVE_FP8, native_linear_class

        linear = torch.nn.Linear(64, 64).to(dtype)
        if stored:
            codes = torch.randn(64, 64).to(torch.float8_e4m3fn)
            linear = native_linear_class()(
                linear, NATIVE_FP8, codes = codes, scale = torch.ones(64, 1, dtype = torch.float32)
            )
            setattr(self, FP8_STORED_ATTR, 1)
        self.proj = linear
        self.norm = torch.nn.LayerNorm(64).to(dtype)

    @property
    def dtype(self):
        return next(p.dtype for p in self.parameters() if p.is_floating_point())


def _pipe(dtype = torch.bfloat16, stored = True):
    return types.SimpleNamespace(transformer = _Dit(dtype, stored))


def test_stored_fp8_holds_its_codes_as_buffers_not_parameters():
    dit = _Dit(torch.bfloat16, True)
    assert {p.dtype for p in dit.parameters()} == {torch.bfloat16}
    assert dict(dit.proj.named_buffers())["weight_q"].dtype == torch.uint8


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_stored_fp8_denoiser_has_its_own_label(dtype):
    stored = dm.denoiser_quant_label(_pipe(dtype, True))
    dense = dm.denoiser_quant_label(_pipe(dtype, False))
    name = str(dtype).replace("torch.", "")
    assert dense == f"dense:{name}"  # dense keys unchanged: recorded peaks keep matching
    assert stored.startswith(f"fp8_stored:{name}#")
    assert stored != dense


def test_stored_fp8_label_separates_e4m3_and_e5m2_codes():
    from core.inference.diffusion_native_quant import NATIVE_FP8, native_linear_class

    e4 = _pipe()
    e5 = _pipe()
    codes = torch.randn(64, 64).to(torch.float8_e5m2)
    e5.transformer.proj = native_linear_class()(
        torch.nn.Linear(64, 64).to(torch.bfloat16),
        NATIVE_FP8,
        codes = codes,
        scale = torch.ones(64, 1),
    )
    assert dm.denoiser_quant_label(e4) != dm.denoiser_quant_label(e5)


def test_stored_fp8_learned_peak_key_does_not_collide_with_the_dense_model():
    kw = dict(attention = "sdpa", model = "flux-2-klein-4b.safetensors", vae = "full", te = "bf16")
    stored = dm.learned_peak_label(_pipe(torch.bfloat16, True), **kw)
    dense = dm.learned_peak_label(_pipe(torch.bfloat16, False), **kw)
    assert stored != dense
    assert dense.startswith("dense:bfloat16|")


def test_packed_detector_sees_the_stored_fp8_buffers():
    assert dm._pipe_denoisers_hold_packed(_pipe(torch.float16, True)) is True
    assert dm._pipe_denoisers_hold_packed(_pipe(torch.float16, False)) is False
    # Wan 2.2: the second expert alone stored
    pipe = types.SimpleNamespace(
        transformer = _Dit(torch.bfloat16, False), transformer_2 = _Dit(torch.bfloat16, True)
    )
    assert dm._pipe_denoisers_hold_packed(pipe) is True
    assert dm.denoiser_quant_label(pipe).startswith("fp8_stored:bfloat16#")


def test_unrelated_packed_and_gguf_labels_are_unchanged():
    class GGUFParameter(torch.nn.Parameter):
        pass

    lin = torch.nn.Linear(4, 4)
    lin.weight = GGUFParameter(lin.weight.data, requires_grad = False)
    assert dm.denoiser_quant_label(types.SimpleNamespace(transformer = lin)) == "gguf"
    import core.inference.diffusion_small_host as sh

    dit = torch.nn.Sequential(torch.nn.Linear(2048, 2048)).to(torch.bfloat16)
    sh.quantize_int8_weight_(dit, compute_dtype = torch.float16, work_device = "cpu")
    assert dm.denoiser_quant_label(types.SimpleNamespace(transformer = dit)).startswith(
        "packed:int8#"
    )


# ------------------------------------------------------------------------------------------------ refinement / pricing


@pytest.fixture
def learned_store(tmp_path, monkeypatch):
    import core.inference.diffusion_measured_peaks as mp

    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    monkeypatch.delenv(mp.LEARNED_PEAKS_ENV, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA T4", raising = False)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(KLEIN_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)
    return mp


def _flat_plan(budget_mib: int = 12758):
    reserve = dm._reserve_mib("discrete_vram", T4_TOTAL)
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, T4_TOTAL)
    return dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = memory,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(
            width = None, height = None, family = "flux.2-klein"
        ),
        requested_mode = None,
        **KLEIN_FLAT,
    )


def _refine(pipe, plan):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = "flux.2-klein", speed_mode = "off")


def test_fp16_dense_control_still_takes_the_static_seed(learned_store):
    new = _refine(_pipe(torch.float16, False), _flat_plan())
    assert new.estimates["measured_static_seed"] is True
    assert new.estimates["measured_runtime_headroom_mib"] == 3072


def test_stored_fp8_never_borrows_the_dense_static_seed(learned_store):
    pipe = _pipe(torch.float16, True)
    plan = _flat_plan()
    # nothing learned for this build: the flat plan stands (the dense fp16 table never ran the per-layer widen)
    assert _refine(pipe, plan) is plan
    assert pipe._unsloth_peak_identity[1].startswith("fp8_stored:float16#")


def test_stored_fp8_places_from_its_own_learned_peak_only(learned_store):
    pipe = _pipe(torch.float16, True)
    dense_label = dm.learned_peak_label(_pipe(torch.float16, False))
    stored_label = dm.learned_peak_label(pipe)
    kw = dict(family = "flux.2-klein", speed_mode = "off", width = 1024, height = 1024)
    learned_store.record(**kw, quant = dense_label, peak_mib = 1000)
    plan = _flat_plan()
    assert _refine(pipe, plan) is plan
    learned_store.record(**kw, quant = stored_label, peak_mib = 2000)
    new = _refine(pipe, plan)
    assert new is not plan
    assert new.estimates["measured_static_seed"] is False
    assert new.estimates["measured_learned_peak_mib"] == 2000
    assert new.estimates["measured_runtime_headroom_mib"] == 2304  # 2000 x 1.15 -> 256 MiB step


def _refine_default(pipe, plan):
    return dm.refine_plan_from_loaded_weights(
        pipe, plan, family = "flux.2-klein", speed_mode = "default"
    )


def test_stored_fp8_beside_torchao_layers_never_takes_a_static_seed(learned_store, monkeypatch):
    # a ComfyUI file whose int8 layers run on torchao while its fp8 layers stay stored: the torchao quant table
    # (dense_mib None) never priced the stored layers' per-forward widen
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setitem(dm._MEASURED_IMAGE_PEAK_MIB, "flux.2-klein", 2000)
    plan = _flat_plan()
    dense = _refine_default(_pipe(torch.float16, False), plan)
    assert (
        dense.estimates["measured_static_seed"] is True
    )  # control: a torchao-only load keeps its seed
    pipe = _pipe(torch.float16, True)
    assert _refine_default(pipe, plan) is plan
    kw = dict(family = "flux.2-klein", speed_mode = "default", width = 1024, height = 1024)
    learned_store.record(**kw, quant = dm.learned_peak_label(pipe), peak_mib = 1000)
    new = _refine_default(pipe, plan)
    assert new.estimates["measured_static_seed"] is False
    assert (
        new.estimates["measured_runtime_headroom_mib"] == 1280
    )  # 1000 x 1.15, not the 2000 table seed
