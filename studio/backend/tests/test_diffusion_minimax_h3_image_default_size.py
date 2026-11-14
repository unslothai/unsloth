# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image's default output size: 2048x2048 for text-to-image, Image 1's reference size for an edit (the
pipeline with no size, and the backend's size for an edit request that names none)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def test_an_edit_without_a_size_matches_image_1s_reference_size():
    from core.inference.diffusion_conditioning import match_source_size

    h3 = types.SimpleNamespace(
        name = "minimax-h3-image",
        dimension_multiple = 32,
        max_output_side = 6272,
        max_output_pixels = 20 * 1024 * 1024,
    )
    assert match_source_size(h3, (1475, 981), 1024) == (1536, 1024)
    assert match_source_size(h3, (600, 400), 1024) == (1536, 1024)
    assert match_source_size(h3, (5000, 3000), 1024) == (3424, 2048)
    assert match_source_size(h3, (1203, 1203), 1024) == (1216, 1216)
    other = types.SimpleNamespace(name = "flux.1-kontext", dimension_multiple = 16)
    assert match_source_size(other, (1475, 981), 1024) != (1536, 1024)


@pytest.fixture(scope = "module")
def pipe(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    try:
        from diffusers import DiffusionPipeline  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - a broken diffusers / hub pair
        pytest.skip(f"diffusers pipelines unavailable: {exc}")
    import _minimax_h3_image_tiny as tiny
    from core.inference import diffusion_minimax_h3_image as h3i

    root = tmp_path_factory.mktemp("h3img_default_size")
    ref = tiny.make_tiny_release(root)
    te = tiny.make_text_encoder(ref["tokenizer"])
    return h3i.load_minimax_h3_image_pipeline(str(root), device = "cpu", text_encoder = te)


def test_the_pipeline_defaults_an_edit_to_image_1s_reference_size(pipe):
    import numpy as np
    import torch
    from PIL import Image

    if pipe.processor is None:
        pytest.skip("no Qwen3-VL processor in this transformers")
    pic = Image.fromarray((np.random.RandomState(0).rand(40, 60, 3) * 255).astype("uint8"))
    out = pipe(
        prompt = "<Picture 1>",
        image = [pic],
        generator = torch.Generator().manual_seed(0),
        min_generation_area = 1024,
        output_type = "latent",
    )
    # 60x40 -> short edge 1024 -> 1536x1024 pixels -> a 64x96 latent
    assert tuple(out.images[0].shape[-2:]) == (64, 96)
    assert pipe._unsloth_h3_last_stats["canvas"] == (1536, 1024)
