# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Klein edits are priced by their sequence tokens on CUDA capability 8.x, from measured peaks.

The request guard priced a 2048x2048 FLUX.3 edit's denoise at the flat estimator's 1024 MiB per output megapixel plus
8192 MiB per weighted condition megapixel (about 900 MiB per 1 MP reference, B200's eager figure). On a 24 GB L4 that
read 9.5 GB (4 images) and 11.3 GB (6 images) against 7.1 / 5.6 GB usable and refused, while the same renders with
allow_oversized ran resident with 2.0-4.5 GiB of denoise activations. The rate below is what L4 and A100-40 measured
per 1k tokens (output canvas plus 4096 per reference); every priced figure must cover its measured peak, and every
request without input images, every other family and every other device class is pinned to the numbers before.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference import diffusion as diff
from core.inference.diffusion_families import _FAMILIES
from core.inference.diffusion_memory import DeviceMemory, calibrated_image_activation

MP = 1024 * 1024
L4_TOTAL = 22_563
_FAM = {f.name: f for f in _FAMILIES}

# Measured, L4 and A100-40 (torch 2.14.1, diffusers 0.42.1), 2048x2048, allow_oversized, phase peak allocated above the
# pre-render baseline, GiB: (model, tier, references, sequence tokens, denoise, vae decode). Both cards read the same.
MEASURED = [
    ("flux.3-klein", "int8 compiled", 0, 16_464, 1.187, 3.227),
    ("flux.3-klein", "int8 compiled", 4, 32_848, 1.989, 3.172),
    ("flux.3-klein", "int8 compiled", 6, 41_040, 2.463, 3.163),
    ("flux.3-klein", "int8 compiled", 10, 57_424, 3.447, 3.175),
    ("flux.3-klein-base", "int8 compiled", 4, 32_848, 2.216, 3.510),
    ("flux.3-klein-base", "int8 compiled", 6, 41_040, 2.792, 3.655),
    ("flux.3-klein-base", "int8 compiled", 10, 57_424, 3.996, 3.994),
    ("flux.3-klein", "bf16 eager", 0, 16_464, 1.658, 3.172),
    ("flux.3-klein", "bf16 eager", 4, 32_848, 3.475, 3.155),
    ("flux.3-klein", "bf16 eager", 6, 41_040, 4.325, 3.144),
    ("flux.3-klein", "bf16 eager", 10, 57_424, 6.057, 3.152),
    ("flux.3-klein-base", "bf16 eager", 4, 32_848, 3.592, 3.379),
    ("flux.3-klein-base", "bf16 eager", 6, 41_040, 4.494, 3.473),
    ("flux.3-klein-base", "bf16 eager", 10, 57_424, 6.334, 3.701),
]


def _denoise(
    refs,
    dense,
    capability = (8, 9),
    width = 2048,
    height = 2048,
    batch = 1,
    rocm = False,
):
    return dm.flux3_conditioned_denoise_mib(
        width = width,
        height = height,
        references = refs,
        batch_size = batch,
        dense = dense,
        capability = capability,
        rocm = rocm,
    )


def _verdict(
    *,
    free,
    family,
    refs,
    denoise,
    width = 2048,
    height = 2048,
    total = L4_TOTAL,
):
    return dm.image_activation_verdict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", free, total),
        width = width,
        height = height,
        family = family,
        condition_pixels = int(refs * MP * _FAM[family.split()[0]].condition_pixel_weight),
        vae_tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
        calibrated_placement = True,
        measured_activation = calibrated_image_activation(family.split()[0], max_speed = False),
        conditioned_denoise_mib = denoise,
    )


@pytest.mark.parametrize("model,tier,refs,tokens,denoise_gib,decode_gib", MEASURED)
def test_priced_denoise_covers_every_measured_peak(
    model, tier, refs, tokens, denoise_gib, decode_gib
):
    dense = tier.startswith("bf16")
    priced = (
        _denoise(max(1, refs), dense)
        if refs
        else _denoise(1, dense) - 4096 * (116 if dense else 76) * 1.25 / 1024
    )
    measured = denoise_gib * 1024
    assert priced >= measured * 1.2, (priced, measured)
    # the whole tiled need (decode term unchanged, fixed overhead) covers the larger phase too
    if refs:
        tiled = max(priced, 2924) + dm.DEFAULT_BASE_OVERHEAD_MIB
        assert tiled >= max(denoise_gib, decode_gib) * 1024 * 1.2


@pytest.mark.parametrize(
    "refs,dense,free,base_action,head_action",
    [
        # free readings of the refused default requests (L4, Studio defaults then bf16)
        (4, False, 9_070, "refuse", "tile"),
        (
            6,
            False,
            7_600,
            "refuse",
            "refuse",
        ),  # 5.55 GB usable after a 4-image oversized run's graphs: still short
        (6, False, 8_000, "refuse", "tile"),
        (10, False, 5_735, "refuse", "refuse"),
        (4, True, 11_230, "refuse", "tile"),
        (6, True, 11_230, "refuse", "tile"),
        (10, True, 11_230, "refuse", "refuse"),  # measured 1.4 GiB left at peak: stays refused
        (4, False, 4_000, "refuse", "refuse"),  # a small card stays refused
    ],
)
def test_l4_verdicts(refs, dense, free, base_action, head_action):
    base = _verdict(free = free, family = "flux.3-klein", refs = refs, denoise = None)
    head = _verdict(free = free, family = "flux.3-klein", refs = refs, denoise = _denoise(refs, dense))
    assert base.action == base_action
    assert head.action == head_action
    assert head.tiled_needed_mib <= base.tiled_needed_mib
    assert head.needed_mib == base.needed_mib  # the untiled figure is untouched


@pytest.mark.parametrize("capability", [(7, 5), (9, 0), (10, 0), (12, 0), None, ()])
def test_unmeasured_device_classes_keep_the_flat_estimate(capability):
    assert _denoise(4, False, capability = capability) is None
    assert _denoise(4, True, capability = capability) is None


def test_rocm_no_references_and_kill_switch_keep_the_flat_estimate(monkeypatch):
    assert _denoise(4, False, rocm = True) is None
    assert _denoise(0, False) is None
    monkeypatch.setenv(dm.FLUX3_TOKEN_PRICED_ENV, "0")
    assert _denoise(4, False) is None


def test_token_count_and_batch():
    # 128x128 canvas tokens + 4 x 4096 reference tokens + 512 prompt tokens
    assert _denoise(4, True) == pytest.approx((16_384 + 4 * 4096 + 512) / 1024 * 116 * 1.25, abs = 1)
    assert _denoise(4, True, batch = 2) == pytest.approx(2 * _denoise(4, True), abs = 2)
    assert _denoise(6, False, width = 1024, height = 1024) < _denoise(6, False)


@pytest.mark.parametrize("family", sorted(_FAM))
@pytest.mark.parametrize("width,height", [(1024, 1024), (2048, 2048), (1536, 1024)])
def test_requests_without_input_images_are_unchanged(family, width, height):
    """Text-to-image (and the hires refine, which drops its condition pixels) ignores the token price entirely."""
    kw = dict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", 16_865 + 2048, L4_TOTAL),
        width = width,
        height = height,
        family = family,
        condition_pixels = 0,
        vae_tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
    )
    base = dm.image_activation_verdict(**kw)
    head = dm.image_activation_verdict(**kw, conditioned_denoise_mib = 1)
    assert base == head


def test_tiled_estimate_only_ever_lowers():
    kw = dict(
        width = 2048,
        height = 2048,
        family = "flux.3-klein",
        condition_pixels = int(4 * MP * 0.11),
        measured = calibrated_image_activation("flux.3-klein", max_speed = False),
    )
    flat = dm.estimate_tiled_image_runtime_mib(**kw)
    assert dm.estimate_tiled_image_runtime_mib(**kw, denoise_mib = 10**9) == flat
    assert dm.estimate_tiled_image_runtime_mib(**kw, denoise_mib = None) == flat
    assert dm.estimate_tiled_image_runtime_mib(**kw, denoise_mib = 3000) < flat


def _state(
    quant,
    optims = ("compiled",),
    kind = "diffusers",
):
    return types.SimpleNamespace(transformer_quant = quant, speed_optims = optims, kind = kind)


class _Target:
    device = "cuda"
    torch_device = "cuda:0"


@pytest.fixture
def ada(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device = None: (8, 9))
    monkeypatch.setattr(torch.version, "hip", None, raising = False)


def test_call_site_prices_flux3_only(ada):
    args = (_Target(), True, 3, 2048, 2048, 1)
    assert diff._flux3_guard_denoise_mib(_FAM["flux.3-klein"], _state("int8"), *args) == _denoise(
        4, False
    )
    assert diff._flux3_guard_denoise_mib(
        _FAM["flux.3-klein-base"], _state(None, ()), *args
    ) == _denoise(4, True)
    # FLUX.2 Klein keeps its own reference pricing; other families keep the flat estimate
    for name in ("flux.2-klein", "qwen-image-edit", "flux.1"):
        if name in _FAM:
            assert diff._flux3_guard_denoise_mib(_FAM[name], _state("int8"), *args) is None


def test_call_site_unmeasured_tiers_keep_the_flat_estimate(ada):
    fam = _FAM["flux.3-klein"]
    args = (_Target(), True, 3, 2048, 2048, 1)
    assert (
        diff._flux3_guard_denoise_mib(fam, _state("int8", ()), *args) is None
    )  # eager quantized: unmeasured
    assert diff._flux3_guard_denoise_mib(fam, _state("nvfp4"), *args) is None
    assert diff._flux3_guard_denoise_mib(fam, _state(None, (), kind = "gguf"), *args) is None
    assert (
        diff._flux3_guard_denoise_mib(fam, _state("int8"), _Target(), False, 0, 2048, 2048, 1)
        is None
    )


def test_long_prompts_are_priced_past_the_512_token_floor():
    """FLUX.3's text window grows with the prompt (max_sequence_length 8192), so a long prompt adds tokens."""
    assert dm.flux3_prompt_tokens(0) == dm.flux3_prompt_tokens(None) == 512
    assert dm.flux3_prompt_tokens(400) == 512
    assert dm.flux3_prompt_tokens(16_000) == 8064
    short = dm.flux3_conditioned_denoise_mib(
        width = 2048, height = 2048, references = 4, dense = True, capability = (8, 9), text_tokens = 512
    )
    assert short == _denoise(4, True)
    long = dm.flux3_conditioned_denoise_mib(
        width = 2048, height = 2048, references = 4, dense = True, capability = (8, 9), text_tokens = 8064
    )
    assert long == pytest.approx((16_384 + 4 * 4096 + 8064) / 1024 * 116 * 1.25, abs = 1)
    # a count under the floor never lowers the estimate
    assert (
        dm.flux3_conditioned_denoise_mib(
            width = 2048, height = 2048, references = 4, dense = True, capability = (8, 9), text_tokens = 10
        )
        == short
    )


def test_call_site_charges_the_longest_prompt(ada):
    fam = _FAM["flux.3-klein-base"]
    args = (_Target(), True, 3, 2048, 2048, 1)
    base = diff._flux3_guard_denoise_mib(fam, _state(None, ()), *args)
    assert diff._flux3_guard_denoise_mib(fam, _state(None, ()), *args, prompt_bytes = 300) == base
    assert diff._flux3_guard_denoise_mib(fam, _state(None, ()), *args, prompt_bytes = 16_000) > base


def test_call_site_rocm_wheel_without_version_hip_keeps_the_flat_estimate(ada, monkeypatch):
    import torch

    monkeypatch.setattr(torch, "__version__", "2.9.1+rocm7.1")
    args = (_Target(), True, 3, 2048, 2048, 1)
    assert diff._flux3_guard_denoise_mib(_FAM["flux.3-klein-base"], _state(None, ()), *args) is None
