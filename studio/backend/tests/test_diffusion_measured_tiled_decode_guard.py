# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generate guard prices a calibrated placement's tiled VAE decode from the family's measured row.

On a T4 (fp16), FLUX.3 Klein loads int8 with the 6 GiB transformer resident and the text encoder streamed, from the
measured activation row. A reference edit at the default 1248x832 then lost the default-resolution exemption (it is
conditioned) and the tiled estimate charged the one 1024 px VAE tile at the flat 8192 MiB per megapixel: 6.5 GiB + 2 GiB
overhead against 5.7 GiB usable, so Studio refused an edit that runs in 8.6 GiB allocated on the 15 GiB card.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference.diffusion_flux3 import REFERENCE_AREA as FLUX3_REFERENCE_AREA
from core.inference.diffusion_memory import DeviceMemory, calibrated_image_activation

# Real T4 reading at the refused edit (Tesla T4, Klein int8 resident, encoder streamed): 7.70 GB free of 14.56 GiB.
T4_TOTAL = 14_910
T4_FREE = 7_885
# One reference, resized to 1 MP and weighted like the FLUX.3 family (condition_pixel_weight 0.11).
ONE_REF = int(FLUX3_REFERENCE_AREA * 0.11)


def _verdict(
    measured,
    *,
    width = 1248,
    height = 832,
    cond = ONE_REF,
    free = T4_FREE,
    family = "flux.3-klein",
):
    return dm.image_activation_verdict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", free, T4_TOTAL),
        width = width,
        height = height,
        family = family,
        condition_pixels = cond,
        vae_tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
        calibrated_placement = True,
        measured_activation = measured,
    )


@pytest.mark.parametrize("family", ["flux.3-klein", "flux.3-klein-base"])
def test_t4_flux3_reference_edit_runs_when_priced_from_the_measured_decode(family):
    flat = _verdict(None, family = family)
    assert flat.action == dm.ACTIVATION_REFUSE  # the T4 refusal this fixes
    act = calibrated_image_activation(family, max_speed = False, compute_bytes = 2)
    fixed = _verdict(act, family = family)
    assert fixed.action != dm.ACTIVATION_REFUSE, fixed.message
    assert fixed.tiled_needed_mib + dm.DEFAULT_BASE_OVERHEAD_MIB <= fixed.budget_mib


def test_a_2048_edit_still_refuses_on_a_t4():
    act = calibrated_image_activation("flux.3-klein", max_speed = False, compute_bytes = 2)
    assert _verdict(act, width = 2048, height = 2048).action == dm.ACTIVATION_REFUSE


def test_float32_compute_prices_the_decode_wider():
    narrow = _verdict(calibrated_image_activation("flux.3-klein", max_speed = False, compute_bytes = 2))
    wide = _verdict(calibrated_image_activation("flux.3-klein", max_speed = False, compute_bytes = 4))
    assert wide.tiled_needed_mib > narrow.tiled_needed_mib


@pytest.mark.parametrize("w,h", [(512, 512), (1024, 1024), (1248, 832), (2048, 2048), (1536, 640)])
@pytest.mark.parametrize("side", [256, 512, 1024])
@pytest.mark.parametrize("family", ["flux.3-klein", "qwen-image-2.1", "sdxl", "z-image"])
def test_the_measured_row_never_prices_higher_than_the_flat_estimate(w, h, side, family):
    for max_speed in (False, True):
        act = calibrated_image_activation(family, max_speed = max_speed)
        kw = dict(
            width = w,
            height = h,
            family = family,
            condition_pixels = ONE_REF,
            tile_side = side,
            vae_sliced = True,
        )
        assert dm.estimate_tiled_image_runtime_mib(
            measured = act, **kw
        ) <= dm.estimate_tiled_image_runtime_mib(**kw)


def test_qwen_tiles_are_priced_against_the_512_px_tile_they_were_measured_with():
    # Qwen-Image-2.1's tiled decode row was measured with 512 px tiles: a 512 px tile is that row, not a quarter of it.
    act = calibrated_image_activation("qwen-image-2.1", max_speed = False)
    priced = dm.estimate_tiled_image_runtime_mib(
        width = 1152,
        height = 1024,
        family = "qwen-image-2.1",
        tile_side = 512,
        vae_sliced = True,
        measured = act,
    )
    assert priced >= act.tiled_decode_mib


def test_an_uncalibrated_placement_ignores_the_measured_row():
    act = calibrated_image_activation("flux.3-klein", max_speed = False, compute_bytes = 2)
    verdict = dm.image_activation_verdict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", T4_FREE, T4_TOTAL),
        width = 1248,
        height = 832,
        family = "flux.3-klein",
        condition_pixels = ONE_REF,
        vae_tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
        calibrated_placement = False,
        measured_activation = act,
    )
    flat = dm.image_activation_verdict(
        device_memory = DeviceMemory("cuda", "cuda", "discrete_vram", T4_FREE, T4_TOTAL),
        width = 1248,
        height = 832,
        family = "flux.3-klein",
        condition_pixels = ONE_REF,
        vae_tile_side = dm.DEFAULT_VAE_TILE_SIDE,
        vae_sliced = True,
        calibrated_placement = False,
    )
    assert verdict.tiled_needed_mib == flat.tiled_needed_mib


def test_placement_measured_activation_reads_the_load_reserve():
    pipe = types.SimpleNamespace(_unsloth_measured_reserve = (2_924, "flux.3-klein", "off", 6_171, 2))
    assert dm.placement_measured_activation(pipe) == calibrated_image_activation(
        "flux.3-klein", max_speed = False, compute_bytes = 2
    )
    pipe_max = types.SimpleNamespace(
        _unsloth_measured_reserve = (2_924, "flux.3-klein", "max", 6_171, 4)
    )
    assert dm.placement_measured_activation(pipe_max) == calibrated_image_activation(
        "flux.3-klein", max_speed = True, compute_bytes = 4
    )
    assert dm.placement_measured_activation(types.SimpleNamespace()) is None
    assert (
        dm.placement_measured_activation(types.SimpleNamespace(_unsloth_measured_reserve = None))
        is None
    )
    unmeasured = types.SimpleNamespace(_unsloth_measured_reserve = (1_000, "no-such-family", "off"))
    assert dm.placement_measured_activation(unmeasured) is None


def test_generate_passes_the_placement_row_to_the_guard():
    """The load's measured reserve first; a calibrated placement without one (FLUX.3 Klein Base on a T4: no
    phase seed) falls back to the row the plan itself read."""
    import inspect

    import core.inference.diffusion as diffusion

    src = " ".join(inspect.getsource(diffusion.DiffusionBackend.generate).split())
    assert "guard_measured = placement_measured_activation(state.pipe)" in src
    assert "_calibrated_activation( fam, guard_target, fp32_scaled = True )" in src
    assert "measured_activation = guard_measured," in src
    for fn in (dm.raise_on_image_activation_shortfall, dm.image_activation_shortfall_message):
        assert "measured_activation" in inspect.signature(fn).parameters


@pytest.mark.parametrize(
    "hint",
    [
        "qwen-image-2.1",
        # The guard's family is the load's variant hint: family name, single file, repo id and base, space-joined.
        "qwen-image-2.1 Qwen/Qwen-Image-2.1",
        "qwen-image-2.1 qwen-image-2.1-Q4_K_M.gguf unsloth/Qwen-Image-2.1-GGUF Qwen/Qwen-Image-2.1",
    ],
)
def test_qwen_tiles_are_priced_from_their_own_512_px_measurement(hint):
    act = calibrated_image_activation("qwen-image-2.1", max_speed = False)
    priced = dm.estimate_tiled_image_runtime_mib(
        width = 1024,
        height = 1536,
        family = hint,
        tile_side = 512,
        vae_sliced = True,
        measured = act,
    )
    flat = dm.estimate_tiled_image_runtime_mib(
        width = 1024, height = 1536, family = hint, tile_side = 512, vae_sliced = True
    )
    # One 512 px tile is the very tile the row measured: never priced under it (nor over the flat estimate).
    assert priced >= min(flat, act.tiled_decode_mib)
