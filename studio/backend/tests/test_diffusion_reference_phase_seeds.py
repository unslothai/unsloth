# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.2 Klein places from its measured phase seed on the first load; a reference request sizes from the flat estimate.

The phase seed was refused to every reference family the request guard cannot price (Klein's source image), so an
automatic int8 Klein on a 12 / 16 GB card had no headroom figure on its first load and streamed every block every
step (klein-9B at 16 GB: 0.18 s/step at a 6 GB peak, vs 0.041 resident). The guard already sizes an unpriced reference
from the flat estimate for the dense tier seeds; the phase seeds now take the same path. CPU only.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

import core.inference.diffusion_memory as dm

ENVS = (
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION",
    "UNSLOTH_DIFFUSION_MEASURED_ACTIVATION_DENSE",
    "UNSLOTH_DIFFUSION_PARTIAL_RESIDENT",
    "UNSLOTH_DIFFUSION_RESIDENT_DIT",
    "UNSLOTH_DIFFUSION_STREAMED_RESIDENCY",
)
# klein-9B auto at a 16 GiB cap (outputs/low_vram_audit): int8 DiT, streamed Qwen3-8B encoder, bf16 VAE, MiB.
KLEIN9_INT8_LOADED = {
    "transformer": (8745, "dit"),
    "text_encoder": (10900, "text_encoder"),
    "vae": (160, "other"),
}


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising = False)


def _klein_pipe(monkeypatch, phase_seeds: bool):
    monkeypatch.setattr(dm, "_loaded_component_mib", lambda pipe: dict(KLEIN9_INT8_LOADED))
    monkeypatch.setattr(dm, "_pipe_denoisers_hold_torchao", lambda pipe: True)
    monkeypatch.setattr(dm, "_holds_fp8_stored", lambda module: False)
    monkeypatch.setattr(dm, "_stream_window_mib", lambda pipe: 1884)
    return types.SimpleNamespace(_unsloth_phase_seeds = phase_seeds)


def _plan_16gb():
    memory = dm.DeviceMemory("cuda", "cuda", "discrete_vram", 15872, 16384)
    return dm.MemoryPlan(
        requested_mode = dm.MEMORY_MODE_AUTO,
        offload_policy = dm.OFFLOAD_GROUP,
        vae_tiling = False,
        vae_slicing = True,
        device_memory = memory,
        estimates = {
            "safe_device_budget_mib": 13824,
            "base_overhead_mib": dm.DEFAULT_BASE_OVERHEAD_MIB,
        },
        stream_text_encoders = True,
        stream_transformer = True,
    )


def test_calibrated_activation_skips_the_reference_rule_only_when_asked(monkeypatch):
    import core.inference.diffusion as d
    from core.inference.diffusion_families import detect_family

    monkeypatch.setattr(d, "sdpa_subquadratic_confirmed", lambda target: True)
    nvidia = types.SimpleNamespace(backend = "cuda", vendor = "nvidia")
    klein = detect_family("black-forest-labs/FLUX.2-klein-4B")
    assert klein.reference and not klein.reference_resolutions
    assert d._calibrated_activation(klein, nvidia) is None
    assert d._calibrated_activation(klein, nvidia, unpriced_references_ok = True) is not None
    rocm = types.SimpleNamespace(backend = "cuda", vendor = "amd")
    assert d._calibrated_activation(klein, rocm, unpriced_references_ok = True) is None


def test_the_load_seeds_the_placement_with_the_reference_rule_skipped():
    src = (Path(dm.__file__).parent / "diffusion.py").read_text(encoding = "utf-8")
    at = src.index("pipe._unsloth_phase_seeds = (")
    assert (
        "_calibrated_activation(fam, target, unpriced_references_ok = True)" in src[at : at + 200]
    )


def test_first_load_klein_int8_on_16gb_keeps_the_transformer_resident(monkeypatch):
    seeded = dm.refine_plan_from_loaded_weights(
        _klein_pipe(monkeypatch, True), _plan_16gb(), family = "flux.2-klein", speed_mode = "default"
    )
    assert seeded.estimates["measured_runtime_headroom_mib"] == 3072
    assert seeded.resident_transformer_mib == 8745
    # without the phase seed (main) there is no headroom figure: the plan streams every block
    plan = _plan_16gb()
    assert (
        dm.refine_plan_from_loaded_weights(
            _klein_pipe(monkeypatch, False), plan, family = "flux.2-klein", speed_mode = "default"
        )
        is plan
    )


def test_unpriced_reference_on_a_phase_seeded_placement_sizes_from_the_flat_estimate():
    pipe = types.SimpleNamespace(
        _unsloth_measured_reserve = (3072, "flux.2-klein", "default", None, None),
        _unsloth_phase_seeds = True,
    )
    flat = dm.estimate_image_runtime_mib(width = 1024, height = 1024, family = "flux.2-klein")
    extra = dm.measured_request_extra_mib
    assert extra(pipe, width = 1024, height = 1024, unpriced_references = True) == flat - 3072 > 0
    assert extra(pipe, width = 1024, height = 1024) == 0
    # a placement without the phase seed (static / learned) is unchanged
    learned = types.SimpleNamespace(
        _unsloth_measured_reserve = (3072, "z-image", "default", None, None),
        _unsloth_phase_seeds = False,
    )
    assert extra(learned, width = 1024, height = 1024, unpriced_references = True) == 0
