# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1 Kontext renders at ~1 MP whatever the source size; sizing must follow the render, not the upload."""

import pytest
from PIL import Image

from core.inference.diffusion import _compile_shape_dims
from core.inference.diffusion_families import detect_family


class _Stop(Exception):
    pass


def _diffusers_kontext_size(width, height):
    diffusers = pytest.importorskip("diffusers")
    pipe_cls = getattr(diffusers, "FluxKontextPipeline", None)
    if pipe_cls is None:
        pytest.skip("diffusers has no FluxKontextPipeline")

    class _Fake:
        default_sample_size = 128
        vae_scale_factor = 8

        def check_inputs(self, prompt, prompt_2, height, width, *args, **kwargs):
            raise _Stop(width, height)

    with pytest.raises(_Stop) as stop:
        pipe_cls.__call__(_Fake(), prompt = "p", height = height, width = width)
    return stop.value.args


_CASES = [
    ((96, 64), (1248, 832)),
    ((256, 256), (1024, 1024)),
    ((1000, 500), (1440, 720)),
    ((640, 1136), (768, 1360)),
    ((2048, 1536), (1168, 880)),
]


@pytest.mark.parametrize("size,want", _CASES)
def test_kontext_edit_dims_are_the_pipeline_render_size(size, want):
    kontext = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    assert _compile_shape_dims("edit", Image.new("RGB", size), 512, 512, kontext) == want


@pytest.mark.parametrize("size,want", _CASES)
def test_kontext_cases_match_installed_diffusers(size, want):
    assert _diffusers_kontext_size(*size) == want


def test_kontext_dims_unchanged_at_one_megapixel():
    kontext = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    src = Image.new("RGB", (1024, 1024))
    assert _compile_shape_dims("edit", src, 512, 512, kontext) == (1024, 1024)


def test_other_edit_families_keep_the_source_size():
    qwen_edit = detect_family("Qwen/Qwen-Image-Edit")
    src = Image.new("RGB", (96, 64))
    assert _compile_shape_dims("edit", src, 1024, 1024, qwen_edit) == (96, 64)


def test_kontext_live_preview_tiles_the_rendered_latents():
    torch = pytest.importorskip("torch")
    from core.inference.diffusion_preview import factors_for, latent_grid

    spec = factors_for("flux.1-kontext")
    assert spec is not None
    kontext = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    latents = torch.zeros(1, (832 // spec.down) * (1248 // spec.down), spec.channels)
    w, h = _compile_shape_dims("edit", Image.new("RGB", (768, 512)), 512, 512, kontext)
    assert latent_grid(latents, spec, h, w) is not None
