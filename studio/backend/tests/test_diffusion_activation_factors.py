# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The newer image families resolve to their own measured activation rows, not the flat 8 GiB per megapixel default. CPU only."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion as diffusion
import core.inference.diffusion_memory as dm
from core.inference.diffusion_families import _FAMILIES

# family -> (text encoder, denoise, untiled decode, tiled decode, 2048^2 worst) MiB, B200 measurements
MEASURED = {
    "qwen-image-2.1": (1_849, 2_489, 7_648, 1_697, 9_602),
    "moss-image": (1_940, 528, 3_666, 3_662, 3_750),
    "flux.3-klein": (301, 901, 2_437, 2_437, 3_359),
    "flux.3-klein-base": (301, 901, 2_512, 2_512, 3_359),
}


@pytest.mark.parametrize("family", sorted(MEASURED))
def test_family_resolves_to_its_measured_row(family):
    act = dm.calibrated_image_activation(family, max_speed = True)
    assert act is not None, family
    expected = tuple(int(v * dm._ACTIVATION_MARGIN) for v in MEASURED[family])
    assert (
        act.text_encoder_mib,
        act.denoise_mib,
        act.decode_mib,
        act.tiled_decode_mib,
        act.max_canvas_mib,
    ) == expected
    flat = dm.estimate_image_runtime_mib(width = None, height = None, family = family)
    assert act.headroom(tiled = True) < flat


@pytest.mark.parametrize("family", sorted(MEASURED))
def test_tiers_only_grow_with_speed(family):
    off, fast = dm._MEASURED_IMAGE_ACTIVATION_MIB[family]
    assert all(b >= a for a, b in zip(off, fast))


def test_flux3_base_is_priced_at_least_like_klein():
    base = dm._MEASURED_IMAGE_ACTIVATION_MIB["flux.3-klein-base"]
    klein = dm._MEASURED_IMAGE_ACTIVATION_MIB["flux.3-klein"]
    for tb, tk in zip(base, klein):
        assert all(b >= k for b, k in zip(tb, tk))


@pytest.mark.parametrize("family", sorted(MEASURED))
def test_load_plan_reads_the_row_on_nvidia(monkeypatch, family):
    fam = next((f for f in _FAMILIES if f.name == family), None)
    assert fam is not None, family
    target = types.SimpleNamespace(backend = "cuda", vendor = "nvidia", dtype = None)
    monkeypatch.setattr(diffusion, "sdpa_subquadratic_confirmed", lambda target: True)
    monkeypatch.setattr(diffusion, "_resolve_diffusion_compute_dtype", lambda fam, dtype: None)
    monkeypatch.setattr(diffusion, "_float_load_itemsize", lambda compute: 2)
    act = diffusion._calibrated_activation(fam, target)
    assert act is not None, family
    assert act.decode_mib == int(MEASURED[family][2] * dm._ACTIVATION_MARGIN)
