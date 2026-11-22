# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-Edit multi-image editing on the diffusers engine: the source plus up to two more images reach
QwenImageEditPlusPipeline as one ordered list, a single-image edit is unchanged, and Kontext keeps one image."""

from __future__ import annotations

import base64
import io

import pytest
from PIL import Image

from core.inference import diffusion_conditioning as cond
from core.inference.diffusion import DiffusionBackend
from core.inference.diffusion_families import detect_family, family_for_checkpoint

from .test_diffusion_backend import _FakePipeline, _FakeTransformer, _load_into, fake_runtime  # noqa: F401

COLORS = [(200, 10, 10), (10, 200, 10), (10, 10, 200)]


def _png(size = (64, 48), color = (200, 10, 10)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def qwen_edit(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(
        backend, tmp_path, base_repo = "Qwen/Qwen-Image-Edit-2511", family_override = "qwen-image-edit"
    )
    return backend


def test_qwen_edit_takes_three_images_and_kontext_and_layered_one():
    qwen = detect_family("unsloth/Qwen-Image-Edit-2511-GGUF")
    kontext = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    layered = detect_family("unsloth/Qwen-Image-Layered-GGUF")
    assert qwen.max_condition_images == 3 and cond.edit_takes_extra_images(qwen)
    assert kontext.max_condition_images == 1 and not cond.edit_takes_extra_images(kontext)
    assert layered.max_condition_images == 1 and not cond.edit_takes_extra_images(layered)
    # Reference / unified-edit families are not edit-only: they keep their own paths.
    assert not cond.edit_takes_extra_images(detect_family("unsloth/FLUX.2-klein-4B-GGUF"))


def test_only_the_plus_checkpoints_take_extra_images():
    fam = detect_family("unsloth/Qwen-Image-Edit-2511-GGUF")
    for ids, limit in (
        (("qwen-image-edit-2511-Q4_K_M.gguf", "unsloth/Qwen-Image-Edit-2511-GGUF"), 3),
        (("Qwen/Qwen-Image-Edit-2509",), 3),
        (("qwen_image_edit_2509_fp8_e4m3fn.safetensors",), 3),
        (("Qwen/Qwen-Image-Edit",), 1),
        (("qwen-image-edit-Q4_K_M.gguf", "Qwen/Qwen-Image-Edit-2511"), 1),
        # A local folder naming no checkpoint keeps the family's (2511) limit.
        (("/models/my_edit", None, "Qwen/Qwen-Image-Edit-2511"), 3),
    ):
        scoped = family_for_checkpoint(fam, *ids)
        assert scoped.max_condition_images == limit, ids
        assert cond.edit_takes_extra_images(scoped) is (limit > 1)
    assert family_for_checkpoint(fam, "Qwen/Qwen-Image-Edit-2511") is fam


def test_the_original_edit_checkpoint_keeps_one_image(fake_runtime, tmp_path):
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(
        backend, tmp_path, base_repo = "Qwen/Qwen-Image-Edit", family_override = "qwen-image-edit"
    )
    c = backend.status()["conditioning"]
    assert c["max_condition_images"] == 1 and c["multi_image_edit"] is False
    with pytest.raises(ValueError, match = "Reference images are not supported"):
        backend.generate(prompt = "x", steps = 4, seed = 1, init_image = _png(), reference_images = [_png()])


def test_status_advertises_the_multi_image_edit(qwen_edit):
    c = qwen_edit.status()["conditioning"]
    assert c["max_condition_images"] == 3 and c["multi_image_edit"] is True
    assert c["unified_edit"] is False and c["localized_edit_modes"] == []


def test_extra_images_reach_the_pipeline_in_request_order(qwen_edit):
    pipe = qwen_edit._state.pipe
    qwen_edit.generate(
        prompt = "the person in image 1 holds the cup from image 2 in the room of image 3",
        steps = 4,
        guidance = 4.0,
        seed = 1,
        init_image = _png((64, 48), COLORS[0]),
        reference_images = [_png((32, 32), COLORS[1]), _png((40, 24), COLORS[2])],
    )
    images = pipe.last_kwargs["image"]
    assert isinstance(images, list) and len(images) == 3
    assert [img.getpixel((0, 0)) for img in images] == COLORS
    # The output takes the source's size, not the last image's.
    assert (pipe.last_kwargs.get("width"), pipe.last_kwargs.get("height")) in (
        (64, 48),
        (None, None),
    )


def test_a_single_image_edit_still_passes_the_bare_image(qwen_edit):
    pipe = qwen_edit._state.pipe
    for refs in (None, [], [""]):
        qwen_edit.generate(
            prompt = "make it night",
            steps = 4,
            guidance = 4.0,
            seed = 1,
            init_image = _png(),
            reference_images = refs,
        )
        image = pipe.last_kwargs["image"]
        assert isinstance(image, Image.Image) and image.size == (64, 48)


def test_a_fourth_image_is_refused_not_truncated(qwen_edit):
    with pytest.raises(ValueError, match = "at most 3 input images"):
        qwen_edit.generate(
            prompt = "x", steps = 4, seed = 1, init_image = _png(), reference_images = [_png()] * 3
        )


def test_kontext_refuses_extra_images(fake_runtime, tmp_path):
    import sys

    sys.modules["diffusers"].FluxKontextPipeline = _FakePipeline
    sys.modules["diffusers"].FluxTransformer2DModel = _FakeTransformer
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(
        backend,
        tmp_path,
        base_repo = "black-forest-labs/FLUX.1-Kontext-dev",
        family_override = "flux.1-kontext",
    )
    assert backend.status()["conditioning"]["multi_image_edit"] is False
    with pytest.raises(ValueError, match = "Reference images are not supported"):
        backend.generate(prompt = "x", steps = 4, seed = 1, init_image = _png(), reference_images = [_png()])
