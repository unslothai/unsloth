# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measured activations for DENSE fp16 denoisers on the default / max speed tiers (``diffusion_memory.py``).

Pre-Ampere cards (T4, V100, Turing) load the denoiser dense in fp16, so ``measured_image_runtime_mib`` takes its dense
branch. That branch only knew the eager tier: on the default tier (Studio's default) it returned None and the load
placed from the flat 8 GiB per megapixel estimate (6.8 GiB for a turbo / distilled model), against measured peaks of
about 2.5 GiB. For a family the dense table measured at that DiT size, the default / max tiers now take the larger of the
dense table and the per-phase table's worst phase for the tier. CPU-only; the loaded sizes and the torchao check are stubbed.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm

T4_TOTAL = 15095
T4_BUDGET = 12758
KLEIN_LOADED = {
    "transformer": (7393, "dit"),
    "text_encoder": (7673, "text_encoder"),
    "vae": (160, "other"),
}
KLEIN_FLAT = dict(model_dense_mib = 15258, companion_dense_mib = 7820, text_encoder_dense_mib = 7629)
ZIMAGE_LOADED = {
    "transformer": (11739, "dit"),
    "text_encoder": (7672, "text_encoder"),
    "vae": (160, "other"),
}
ZIMAGE_FLAT = dict(model_dense_mib = 19571, companion_dense_mib = 7832, text_encoder_dense_mib = 7672)
ENVS = (
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION",
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE",
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE_TIERS",
    "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
)


def _memory(budget_mib: int, total_mib: int) -> "dm.DeviceMemory":
    reserve = dm._reserve_mib("discrete_vram", total_mib)
    return dm.DeviceMemory("cuda", "cuda", "discrete_vram", budget_mib + reserve, total_mib)


def _flat_plan(
    sizes: dict,
    family: str,
    budget_mib: int = T4_BUDGET,
    total_mib: int = T4_TOTAL,
):
    return dm.plan_diffusion_memory(
        target = types.SimpleNamespace(supports_model_cpu_offload = True),
        device_memory = _memory(budget_mib, total_mib),
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None, family = family),
        **sizes,
    )


def _pipe(
    monkeypatch,
    loaded: dict,
    dtype_name: str = "float16",
    phase_seeds: bool = False,
    tier_seeds: bool = True,
):
    import torch

    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(loaded))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)
    return types.SimpleNamespace(
        transformer = types.SimpleNamespace(dtype = getattr(torch, dtype_name)),
        _unsloth_phase_seeds = phase_seeds,
        # NVIDIA with a sub-quadratic SDPA kernel (diffusion._measured_target), as on the T4
        _unsloth_dense_tier_seeds = tier_seeds,
    )


@pytest.fixture(autouse = True)
def _no_learned_store(tmp_path, monkeypatch):
    import core.inference.diffusion_measured_peaks as mp
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    mp._cache.update(path = None, mtime = None, data = None)


def _refine(pipe, plan, family: str, speed: str):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = family, speed_mode = speed)


@pytest.mark.parametrize("speed", ["default", "max"])
def test_t4_klein_compiled_tiers_keep_the_dit_resident(monkeypatch, speed):
    pipe = _pipe(monkeypatch, KLEIN_LOADED)
    plan = _flat_plan(KLEIN_FLAT, "flux.2-klein")
    # flat: 8192 MiB of activations, every DiT block streamed
    assert plan.estimates["runtime_headroom_mib"] == 8192
    assert plan.offload_policy == dm.OFFLOAD_GROUP and plan.stream_transformer
    new = _refine(pipe, plan, "flux.2-klein", speed)
    assert new is not plan
    headroom = new.estimates["measured_runtime_headroom_mib"]
    assert headroom == (3072 if speed == "default" else 3328)
    # default: the whole DiT; max (the larger compiled-tier peak): all but the last few blocks
    assert new.resident_transformer_mib == (7393 if speed == "default" else 7222)
    assert 160 + new.resident_transformer_mib + headroom + dm.DEFAULT_BASE_OVERHEAD_MIB <= T4_BUDGET


def test_t4_zimage_keeps_the_flat_plan(monkeypatch):
    """No dense table entry for Z-Image: a partly resident 11.5 GiB DiT ran slower than streaming it whole on a T4."""
    plan = _flat_plan(ZIMAGE_FLAT, "z-image-turbo")
    assert plan.estimates["runtime_headroom_mib"] == 6963
    for phase_seeds in (False, True):
        pipe = _pipe(monkeypatch, ZIMAGE_LOADED, phase_seeds = phase_seeds)
        for speed in ("off", "default", "max"):
            assert _refine(pipe, plan, "z-image", speed) is plan


def test_seed_never_below_a_measured_phase_or_the_eager_seed():
    m = dm.measured_image_runtime_mib
    for family, (eager_peak, max_dit) in dm._MEASURED_DENSE_EAGER_PEAK_MIB.items():
        phases = dm._MEASURED_IMAGE_ACTIVATION_MIB.get(family)
        for speed, column in (("default", 0), ("max", 1)):
            seed = m(family, speed, dense_transformer_mib = max_dit, dense_tier_seeds = True)
            if phases is None:
                assert seed is None, family
                continue
            assert seed >= max(phases[column][:3]) * dm._MEASURED_PEAK_MARGIN, (family, speed)
            assert seed >= m(family, "off", dense_transformer_mib = max_dit), (family, speed)
            # off the measured target (math-only attention, ROCm) the tiers keep the flat estimate
            assert m(family, speed, dense_transformer_mib = max_dit) is None, (family, speed)
            # fp32 compute doubles it, never shrinks it
            assert (
                m(
                    family,
                    speed,
                    dense_transformer_mib = max_dit,
                    compute_bytes = 4,
                    dense_tier_seeds = True,
                )
                >= 2 * seed - 256
            )


def test_larger_dit_bf16_and_kill_switch_keep_the_flat_plan(monkeypatch):
    m = dm.measured_image_runtime_mib
    # klein-9B is above the size the dense table measured
    assert m("flux.2-klein", "default", dense_transformer_mib = 17300) is None
    # an unmeasured family, with or without phase evidence
    assert m("sdxl", "default", dense_transformer_mib = 4900, phase_seeds = True) is None
    plan = _flat_plan(KLEIN_FLAT, "flux.2-klein")
    # bf16 (every Ampere+ card): no static seed, learned peaks only
    assert (
        _refine(_pipe(monkeypatch, KLEIN_LOADED, "bfloat16"), plan, "flux.2-klein", "default")
        is plan
    )
    pipe = _pipe(monkeypatch, KLEIN_LOADED)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE_TIERS", "0")
    assert _refine(pipe, plan, "flux.2-klein", "default") is plan
    # the eager tier is not behind the new switch
    assert _refine(pipe, plan, "flux.2-klein", "off") is not plan


def test_large_card_plan_unchanged(monkeypatch):
    pipe = _pipe(monkeypatch, KLEIN_LOADED)
    resident = _flat_plan(KLEIN_FLAT, "flux.2-klein", 160000, 183359)
    assert resident.offload_policy == dm.OFFLOAD_NONE
    assert _refine(pipe, resident, "flux.2-klein", "default") is resident


def test_guard_charges_bigger_requests_past_the_default_tier_reserve(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    pipe = types.SimpleNamespace(
        _unsloth_measured_reserve = (3072, "flux.2-klein", "default", 7393, 2),
        _unsloth_dense_tier_seeds = True,
    )
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == 0
    # 2048 x 2048 is four megapixels: the seed scales with the pixels, so resident groups stream again
    assert dm.measured_request_extra_mib(pipe, width = 2048, height = 2048) == 12288 - 3072
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024, batch_size = 2) > 0


@pytest.mark.parametrize("speed", ["default", "max"])
def test_unmeasured_target_keeps_the_flat_plan(monkeypatch, speed):
    """Math-only attention or ROCm: the phase table was not measured there, so a dense fp16 load keeps the flat
    reserve on the compiled tiers (the eager dense table is unchanged)."""
    pipe = _pipe(monkeypatch, KLEIN_LOADED, tier_seeds = False)
    plan = _flat_plan(KLEIN_FLAT, "flux.2-klein")
    assert _refine(pipe, plan, "flux.2-klein", speed) is plan
    assert _refine(pipe, plan, "flux.2-klein", "off") is not plan


def test_measured_target_needs_nvidia_and_a_subquadratic_kernel(monkeypatch):
    import core.inference.diffusion as diffusion

    nvidia = types.SimpleNamespace(backend = "cuda", vendor = "nvidia")
    rocm = types.SimpleNamespace(backend = "cuda", vendor = "amd")
    monkeypatch.setattr(diffusion, "sdpa_subquadratic_confirmed", lambda target: True)
    assert diffusion._measured_target(nvidia) is True
    assert diffusion._measured_target(rocm) is False
    assert diffusion._measured_target(types.SimpleNamespace(backend = "mps", vendor = "apple")) is False
    # math-only attention (no fused kernel ran) and a probe that raised are not the measured target
    monkeypatch.setattr(diffusion, "sdpa_subquadratic_confirmed", lambda target: False)
    assert diffusion._measured_target(nvidia) is False

    def boom(target):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(diffusion, "sdpa_subquadratic_confirmed", boom)
    assert diffusion._measured_target(nvidia) is False


def test_unpriced_reference_request_sizes_from_the_flat_estimate(monkeypatch):
    """A FLUX.2 Klein reference render adds the source image's latents to the denoiser's sequence, which the guard does
    not count in condition_pixels and the text-to-image tier seed never measured: it falls back to the flat estimate,
    so the resident DiT groups stream again exactly as the flat plan would have placed them."""
    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    pipe = types.SimpleNamespace(
        _unsloth_measured_reserve = (3072, "flux.2-klein", "default", 7393, 2),
        _unsloth_dense_tier_seeds = True,
    )
    flat = dm.estimate_image_runtime_mib(width = 1024, height = 1024, family = "flux.2-klein")
    assert flat == 8192
    extra = dm.measured_request_extra_mib
    assert extra(pipe, width = 1024, height = 1024, unpriced_references = True) == flat - 3072
    # text-to-image keeps the seed
    assert extra(pipe, width = 1024, height = 1024) == 0
    # the eager dense table is unaffected
    eager = types.SimpleNamespace(
        _unsloth_measured_reserve = (3072, "flux.2-klein", "off", 7393, 2),
        _unsloth_dense_tier_seeds = True,
    )
    assert extra(eager, width = 1024, height = 1024, unpriced_references = True) == 0


def test_generate_passes_unpriced_references_to_the_guard():
    src = (__import__("pathlib").Path(dm.__file__).parent / "diffusion.py").read_text(
        encoding = "utf-8"
    )
    at = src.index("guard_unpriced_refs = bool(")
    window = src[at : at + 400]
    for part in (
        'getattr(fam, "reference", False)',
        "init_pil is not None",
        "ref_resolution is None",
        "not is_flux3_family(fam)",
    ):
        assert part in window, part
    call = src.index("extra_mib = split_aware_request_extra(", at)
    assert "measured_request_extra_mib(" in src[call : call + 500]
    assert "unpriced_references = guard_unpriced_refs," in src[call : call + 800]
