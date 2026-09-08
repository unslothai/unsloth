# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only unit tests for the Transform output-size bound and the refusal it produces.

Reported: Image Transform refused at 2048x2048 no matter how small the Resolution controls were
set, because img2img sized from the upload (clamped to a fixed 2048) and the refusal then advised
changing a control that could not move that number. ``_fit_within`` makes the control bound the
source; ``source_driven`` fixes the remedy sentence where it still cannot.
"""

from __future__ import annotations

import types

import pytest

from core.inference.diffusion import _clamp_max_side, _fit_within
from core.inference.diffusion_memory import DeviceMemory, image_activation_shortfall_message

PIL = pytest.importorskip("PIL.Image")


def _img(w: int, h: int):
    return PIL.new("RGB", (w, h))


def test_oversized_source_is_bounded_by_the_requested_box():
    # The reported case: a big upload with the sliders set small.
    out = _fit_within(_img(4000, 3000), 512, 512)
    assert out.size == (512, 384)  # fits the box, aspect ratio preserved


def test_bound_is_the_box_not_just_the_longest_side():
    # A wide box and a square source: the HEIGHT binds, which a longest-side clamp misses.
    assert _fit_within(_img(1024, 1024), 1024, 256).size == (256, 256)
    # The longest-side clamp leaves it untouched -- the two are not interchangeable.
    assert _clamp_max_side(_img(1024, 1024), 1024).size == (1024, 1024)


def test_small_source_is_never_enlarged():
    # Growing a source is the Upscale workflow; Transform must not silently do it.
    src = _img(384, 256)
    assert _fit_within(src, 2048, 2048) is src
    # Exactly on the box is also a no-op (identity, no resample pass).
    on_box = _img(512, 512)
    assert _fit_within(on_box, 512, 512) is on_box


def test_one_axis_over_still_downscales_both():
    assert _fit_within(_img(2048, 512), 1024, 1024).size == (1024, 256)


def test_degenerate_box_does_not_produce_a_zero_dimension():
    # Only a malformed request gets here, but a 0-px side would raise deep inside the VAE.
    out = _fit_within(_img(1000, 10), 1, 1)
    assert out.size[0] >= 1 and out.size[1] >= 1


def _cuda(free_mib: int, total_mib: int) -> DeviceMemory:
    return DeviceMemory(
        backend = "cuda",
        device = "cuda",
        memory_kind = "discrete_vram",
        free_mib = free_mib,
        total_mib = total_mib,
    )


def _shortfall(**kwargs) -> str:
    # 4096x4096 on a card with ~14 GB free is well past both arms of the guard.
    message = image_activation_shortfall_message(
        device_memory = _cuda(free_mib = 14000, total_mib = 16000),
        width = 4096,
        height = 4096,
        **kwargs,
    )
    assert message is not None
    return message


def test_slider_driven_refusal_keeps_the_resolution_remedy():
    message = _shortfall()
    assert "Generate at a smaller resolution" in message
    assert "Upload a smaller source image" not in message


def test_source_driven_refusal_points_at_the_upload_instead():
    message = _shortfall(source_driven = True)
    assert "Upload a smaller source image" in message
    # The wrong advice must be gone, not merely accompanied.
    assert "Generate at a smaller resolution" not in message
    assert "Resolution setting" in message


def test_batch_note_still_composes_with_either_remedy():
    for source_driven in (False, True):
        message = _shortfall(batch_size = 4, source_driven = source_driven)
        assert "or a smaller batch size" in message
        assert "at a batch of 4" in message
