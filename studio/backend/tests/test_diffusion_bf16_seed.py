# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A fresh bf16 load places like a returning one: the bf16 dense table seeds what only a learned peak did before.
Numbers from FLUX.1-schnell bf16 on a Colab L4 (the load's own estimates and its recorded peaks). CPU only."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

import core.inference.diffusion_memory as dm  # noqa: E402

# unsloth/FLUX.1-schnell bf16, L4 load: diffusion.loaded estimates
FLUX_L4_LOADED = {
    "transformer": (22681, "dit"),
    "text_encoder_2": (9319, "text_encoder"),
    "vae": (160, "other"),
}
L4_BUDGET = 20321
ENVS = (
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION",
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE",
    "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_BF16",
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    import core.inference.diffusion_measured_peaks as mp

    for name in ENVS:
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setenv(mp.LEARNED_PEAKS_FILE_ENV, str(tmp_path / "peaks.json"))
    monkeypatch.delenv(mp.LEARNED_PEAKS_ENV, raising = False)
    mp._cache.update(path = None, mtime = None, data = None)
    monkeypatch.setattr(mp, "current_device_name", lambda: "NVIDIA L4", raising = False)
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(FLUX_L4_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: False)
    monkeypatch.setattr(dm, "denoiser_quant_label", lambda pipe: "dense:bfloat16")
    return mp


def _pipe(dtype = torch.bfloat16, measured_target = True):
    return types.SimpleNamespace(
        transformer = types.SimpleNamespace(dtype = dtype),
        _unsloth_dense_tier_seeds = measured_target,
    )


def _l4_plan():
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", 22000, 22563)
    return dm.MemoryPlan(
        requested_mode = dm.MEMORY_MODE_AUTO,
        offload_policy = dm.OFFLOAD_GROUP,
        vae_tiling = False,
        vae_slicing = False,
        device_memory = memory,
        estimates = {
            "safe_device_budget_mib": L4_BUDGET,
            "base_overhead_mib": 2048,
            "text_encoder_dense_mib": 9346,
        },
        stream_text_encoders = False,
        stream_transformer = True,
    )


def _refine(pipe, plan):
    return dm.refine_plan_from_loaded_weights(pipe, plan, family = "flux.1", speed_mode = "off")


def test_a_fresh_bf16_flux_on_an_l4_keeps_part_of_its_denoiser_resident(store):
    fresh = _refine(_pipe(), _l4_plan())
    # 3581 x 1.15 -> 4352: the worst measured bf16 peak
    assert fresh.estimates["measured_runtime_headroom_mib"] == 4352
    assert fresh.estimates["measured_dense_bf16"] is True
    assert fresh.estimates["measured_static_seed"] is True
    assert fresh.stream_text_encoders and not fresh.resident_text_encoder_mib
    # the returning load on the L4 kept 14017 MiB from its 3417 MiB peak (-> 4096); the seed is 256 MiB stricter
    assert fresh.resident_transformer_mib == 14017 - 256


def test_the_fresh_plan_matches_a_returning_load_within_the_seed_margin(store, monkeypatch):
    fresh = _refine(_pipe(), _l4_plan())
    store.record(
        family = "flux.1",
        quant = "dense:bfloat16",
        speed_mode = "off",
        width = 1024,
        height = 1024,
        peak_mib = 3417,
    )
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_BF16", "0")
    returning = _refine(_pipe(), _l4_plan())
    assert returning.resident_transformer_mib == 14017
    monkeypatch.delenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_BF16")
    both = _refine(_pipe(), _l4_plan())
    assert both.resident_transformer_mib == fresh.resident_transformer_mib


def test_the_bf16_kill_switch_restores_the_flat_plan(store, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_BF16", "0")
    plan = _l4_plan()
    assert _refine(_pipe(), plan) is plan


def test_only_the_eager_tier_and_measured_sizes_are_seeded(store):
    m = dm.measured_image_runtime_mib
    assert (
        m("flux.1", "off", dense_transformer_mib = 22681, dense_bf16 = True, dense_tier_seeds = True)
        == 4352
    )
    assert m("flux.1", "off", dense_transformer_mib = 22681, dense_bf16 = True) is None
    assert (
        m("flux.1", "default", dense_transformer_mib = 22681, dense_bf16 = True, dense_tier_seeds = True)
        is None
    )
    assert (
        m("flux.1", "off", dense_transformer_mib = 30000, dense_bf16 = True, dense_tier_seeds = True)
        is None
    )
    assert (
        m("z-image", "off", dense_transformer_mib = 11000, dense_bf16 = True, dense_tier_seeds = True)
        is None
    )
    assert m("flux.1", "off", dense_transformer_mib = 22680) == 2816


def test_math_only_attention_keeps_a_bf16_denoiser_learned_only(store):
    """The seed was measured with a sub-quadratic SDPA kernel; math-only attention / ROCm keep the streamed plan."""
    plan = _l4_plan()
    assert _refine(_pipe(measured_target = False), plan) is plan


def test_packed_and_fp8_stored_bf16_denoisers_stay_learned_only(store, monkeypatch):
    plan = _l4_plan()
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_packed", lambda pipe: True)
    assert _refine(_pipe(), plan) is plan
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_packed", lambda pipe: False)
    monkeypatch.setattr(dm, "_holds_fp8_stored", lambda module: module is not None)
    assert _refine(_pipe(), plan) is plan


def test_the_request_guard_reads_the_bf16_seed_from_the_reserve(store):
    pipe = types.SimpleNamespace(
        _unsloth_dense_tier_seeds = True,
        _unsloth_peak_identity = ("flux.1", "dense:bfloat16", "off"),
        _unsloth_measured_reserve = (
            4352,
            "flux.1",
            "off",
            22681,
            None,
            None,
            None,
            False,
            True,
            False,
            True,
        ),
    )
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == 0
    # 2 MP: 2 x 3581 x 1.15 = 8236 -> 8448
    assert dm.measured_request_extra_mib(pipe, width = 2048, height = 1024) == 8448 - 4352


def test_the_deferred_profile_invalidates_the_bf16_seed(store):
    """Generation 3 engages the compiled tier the eager seed never measured: requests size from the flat estimate."""
    pipe = types.SimpleNamespace(
        _unsloth_dense_tier_seeds = True,
        _unsloth_peak_identity = ("flux.1", "dense:bfloat16", "off"),
        _unsloth_measured_reserve = (
            4352,
            "flux.1",
            "off",
            22681,
            None,
            None,
            None,
            False,
            True,
            False,
            True,
        ),
    )
    assert dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == 0
    pipe._unsloth_peak_static = True
    flat = dm.estimate_image_runtime_mib(width = 1024, height = 1024, batch_size = 1, family = "flux.1")
    assert (
        dm.measured_request_extra_mib(pipe, width = 1024, height = 1024) == max(0, int(flat) - 4352) > 0
    )
