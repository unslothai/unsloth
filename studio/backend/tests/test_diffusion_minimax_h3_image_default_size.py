# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image's default output size: the reference workflow's 2048x2048 for text-to-image and Image 1 at 4 MP
cut to the 32 grid for an edit (the backend's size for a request that names none); other families keep 1024."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
for p in (_BACKEND, Path(__file__).resolve().parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

H3 = types.SimpleNamespace(
    name = "minimax-h3-image",
    dimension_multiple = 32,
    max_output_side = 6272,
    max_output_pixels = 20 * 1024 * 1024,
)


def test_reference_edit_size_is_image_1_at_4mp_cut_to_32():
    from core.inference.diffusion_minimax_h3_image import reference_edit_size

    assert reference_edit_size(1475, 981) == (2496, 1664)
    assert reference_edit_size(600, 400) == (2496, 1664)
    assert reference_edit_size(427, 640) == (1664, 2496)
    assert reference_edit_size(1203, 1203) == (2048, 2048)
    assert reference_edit_size(2048, 2048) == (2048, 2048)
    assert reference_edit_size(5000, 3000) == (2624, 1568)


def test_an_edit_without_a_size_matches_the_reference_edit_size():
    from core.inference.diffusion_conditioning import match_source_size
    from core.inference.diffusion_minimax_h3_image import reference_edit_size

    for source in ((1475, 981), (600, 400), (427, 640), (1203, 1203), (2048, 2048), (5000, 3000)):
        assert match_source_size(H3, source, 1024) == reference_edit_size(*source)
    # A family bound under the reference size falls back to the area fit inside it.
    small = types.SimpleNamespace(**{**vars(H3), "max_output_side": 2048})
    w, h = match_source_size(small, (1475, 981), 1024)
    assert max(w, h) <= 2048 and w % 32 == 0 and h % 32 == 0
    other = types.SimpleNamespace(name = "flux.1-kontext", dimension_multiple = 16)
    assert match_source_size(other, (1475, 981), 1024) == (1248, 832)


def test_text_to_image_without_a_size_defaults_per_family():
    from core.inference.diffusion_conditioning import omitted_txt2img_size
    assert omitted_txt2img_size("minimax-h3-image") == (2048, 2048)
    for family in ("flux.1", "sdxl", "z-image", "qwen-image", None):
        assert omitted_txt2img_size(family) is None


class _Engine:
    def __init__(self, family):
        self.family = family
        self.calls = []

    def status(self):
        return {"loaded": True, "repo_id": "org/m", "base_repo": None, "family": self.family}

    def generate_progress(self):
        return {"active": False}

    def generate(self, **kwargs):
        from PIL import Image

        self.calls.append(kwargs)
        size = (kwargs.get("width") or 1024, kwargs.get("height") or 1024)
        return {"images": [Image.new("RGB", size, "blue")], "seed": 7, "repo_id": "org/m"}


@pytest.mark.parametrize(
    "family, body, expected",
    [
        ("minimax-h3-image", {}, (2048, 2048)),
        ("minimax-h3-image", {"width": 1024, "height": 1024}, (1024, 1024)),
        ("minimax-h3-image", {"width": 1536, "height": 1024}, (1536, 1024)),
        ("flux.1", {}, (1024, 1024)),
        ("sdxl", {}, (1024, 1024)),
        ("minimax-h3-image", {"workflow": "edit", "init_image": "x"}, (None, None)),
        ("minimax-h3-image", {"init_image": "x"}, (1024, 1024)),
    ],
)
def test_the_generate_route_resolves_an_omitted_size_per_family(
    family, body, expected, monkeypatch, tmp_path
):
    from core.inference import diffusion_engine_router
    from models.inference import DiffusionGenerateRequest
    from routes import inference
    from state import active_generations

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(inference, "_enabled_agent_skills", lambda: [], raising = False)
    engine = _Engine(family)
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: engine)
    active_generations.reset_for_tests()
    request = DiffusionGenerateRequest(prompt = "x", steps = 6, guidance = 1.0, **body)
    try:
        asyncio.run(inference._generate_diffusion_image_core(request))
    except Exception:  # noqa: BLE001 - only the size the engine was handed is under test
        pass
    finally:
        active_generations.reset_for_tests()
    assert engine.calls, "the engine was never called"
    assert (engine.calls[0]["width"], engine.calls[0]["height"]) == expected


class _StatusFailsEngine(_Engine):
    def status(self):
        raise RuntimeError("status read failed")


def test_an_omitted_text_to_image_size_survives_a_failing_status_read(monkeypatch, tmp_path):
    """Steps and guidance named, size omitted: a status read that fails leaves the schema's size, as before the
    family default existed, instead of failing the run."""
    from core.inference import diffusion_engine_router
    from models.inference import DiffusionGenerateRequest
    from routes import inference
    from state import active_generations

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(inference, "_enabled_agent_skills", lambda: [], raising = False)
    engine = _StatusFailsEngine("minimax-h3-image")
    monkeypatch.setattr(diffusion_engine_router, "get_active_diffusion_engine", lambda: engine)
    active_generations.reset_for_tests()
    request = DiffusionGenerateRequest(prompt = "x", steps = 6, guidance = 1.0)
    try:
        asyncio.run(inference._generate_diffusion_image_core(request))
    finally:
        active_generations.reset_for_tests()
    assert engine.calls, "the run failed before the engine was called"
    assert (engine.calls[0]["width"], engine.calls[0]["height"]) == (1024, 1024)


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


def test_the_pipeline_defaults_an_edit_to_the_reference_edit_size(pipe, monkeypatch):
    # The pipeline itself, with no size: Picture 1 at 4 MP cut to the 32 grid (the reference edit workflow), or
    # Image 1's reference size (short edge 1024..2048) with the workflow's Picture 1 rescale turned off.
    import numpy as np
    from PIL import Image

    from core.inference import diffusion_minimax_h3_image as h3i

    seen = {}

    def spy(self, prompt, vae_refs, vision_refs, height, width, *a, **k):
        seen["size"] = (width, height)
        return None

    monkeypatch.setattr(type(pipe), "_render_one", spy)
    pic = Image.fromarray((np.random.RandomState(0).rand(40, 60, 3) * 255).astype("uint8"))
    pipe(prompt = "<Picture 1>", image = [pic])
    assert seen["size"] == h3i.reference_edit_size(60, 40) == (2496, 1664)
    monkeypatch.setenv(h3i.SOURCE_RESIZE_ENV, "0")
    pipe(prompt = "<Picture 1>", image = [pic])
    assert seen["size"] == (1536, 1024)


@pytest.mark.parametrize(
    "source, want",
    [((600, 100), (4992, 832)), ((1536, 1024), (2496, 1664)), ((100, 600), (832, 4992))],
)
def test_the_pipeline_default_ignores_the_vae_short_edge_enlargement(
    pipe, monkeypatch, source, want
):
    # a panorama's VAE copy is enlarged to the 1024 short edge; the output default stays the workflow size
    from PIL import Image

    from core.inference import diffusion_minimax_h3_image as h3i

    seen = {}

    def spy(self, prompt, vae_refs, vision_refs, height, width, *a, **k):
        seen["size"] = (width, height)
        seen["vae"] = h3i.picture_size(vae_refs[0])
        return None

    monkeypatch.delenv(h3i.SOURCE_RESIZE_ENV, raising = False)
    monkeypatch.setattr(type(pipe), "_render_one", spy)
    pipe(prompt = "<Picture 1>", image = Image.new("RGB", source, (40, 80, 120)))
    assert seen["size"] == h3i.reference_edit_size(*source) == want


def test_the_pipeline_default_follows_the_exif_orientation(pipe, monkeypatch):
    from PIL import Image

    from core.inference import diffusion_minimax_h3_image as h3i

    seen = {}

    def spy(self, prompt, vae_refs, vision_refs, height, width, *a, **k):
        seen["size"] = (width, height)
        return None

    monkeypatch.delenv(h3i.SOURCE_RESIZE_ENV, raising = False)
    monkeypatch.setattr(type(pipe), "_render_one", spy)
    pic = Image.new("RGB", (600, 100), (40, 80, 120))
    pic.getexif()[0x0112] = 6
    pipe(prompt = "<Picture 1>", image = pic)
    assert seen["size"] == h3i.reference_edit_size(100, 600)


def test_decode_takes_the_decoder_dtype_after_an_fp16_reference_encode():
    """An fp16 encoder (reference encode) must not round the float32 latent the decoder takes."""
    import torch

    from core.inference import diffusion_minimax_h3_image as H

    vae = torch.nn.Module()
    vae.encoder = torch.nn.Conv2d(3, 4, 1)
    vae.quant_conv = torch.nn.Conv2d(8, 8, 1)
    vae.post_quant_conv = torch.nn.Conv2d(4, 4, 1)
    vae.decoder = torch.nn.Conv2d(4, 3, 1)
    H.set_encoder_dtype(vae, torch.float16)
    assert H._module_dtype(vae) == torch.float16
    assert H._decoder_input_dtype(vae) == torch.float32
