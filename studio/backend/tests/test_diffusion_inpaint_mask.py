# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inpaint "Mask blur" and "Inpaint area: Only masked" (AUTOMATIC1111 semantics), no GPU."""

from __future__ import annotations

import inspect
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

from core.inference import diffusion_mask
from models.inference import DiffusionGenerateRequest, DiffusionStatusResponse, GalleryImage

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_backend import (  # noqa: E402,F401
    _FakeInpaintPipe,
    _FakeInpaintPipeline,
    _loaded_backend,
    _tiny_mask_b64,
    _tiny_png_b64,
    fake_runtime,
)
from test_diffusion_routes import client  # noqa: E402,F401  (fixture)


def _mask():
    from PIL import Image

    img = Image.new("L", (64, 64), 0)
    img.paste(255, (16, 16, 48, 48))
    return img


def test_blur_mask_none_and_zero_return_the_same_object():
    m = _mask()
    assert diffusion_mask.blur_mask(m, None) is m
    assert diffusion_mask.blur_mask(m, 0) is m


def test_blur_mask_matches_diffusers_blur():
    from PIL import ImageFilter

    m = _mask()
    out = diffusion_mask.blur_mask(m, 4)
    assert out.size == m.size and out.mode == "L"
    # diffusers VaeImageProcessor.blur is exactly PIL GaussianBlur(radius).
    assert out.tobytes() == m.filter(ImageFilter.GaussianBlur(4)).tobytes()
    assert out.tobytes() != m.tobytes()


def test_resolve_padding():
    assert diffusion_mask.resolve_padding(None, None) is None
    assert diffusion_mask.resolve_padding(False, 64) is None
    assert diffusion_mask.resolve_padding(True, None) == 32
    assert diffusion_mask.resolve_padding(True, 0) == 0
    assert diffusion_mask.resolve_padding(True, 100) == 100


def test_padding_mask_crop_signature_probe():
    class With:
        def __call__(
            self,
            prompt = None,
            padding_mask_crop = None,
        ):
            pass

    class Without:
        def __call__(
            self,
            prompt = None,
            **kwargs,
        ):
            pass

    assert diffusion_mask.pipeline_accepts_padding_mask_crop(With)
    assert not diffusion_mask.pipeline_accepts_padding_mask_crop(Without)
    assert not diffusion_mask.inpaint_class_accepts_padding_mask_crop(None)
    assert not diffusion_mask.inpaint_class_accepts_padding_mask_crop("NoSuchPipelineClass")


def test_request_fields_default_none_and_bounds():
    req = DiffusionGenerateRequest(prompt = "x")
    assert req.mask_blur is None and req.inpaint_only_masked is None and req.inpaint_padding is None
    ok = DiffusionGenerateRequest(
        prompt = "x", mask_blur = 64, inpaint_only_masked = True, inpaint_padding = 256
    )
    assert (ok.mask_blur, ok.inpaint_only_masked, ok.inpaint_padding) == (64, True, 256)
    DiffusionGenerateRequest(prompt = "x", mask_blur = 0, inpaint_padding = 0)
    for bad in (
        {"mask_blur": -1},
        {"mask_blur": 65},
        {"inpaint_padding": -1},
        {"inpaint_padding": 257},
    ):
        with pytest.raises(ValidationError):
            DiffusionGenerateRequest(prompt = "x", **bad)


def test_status_and_gallery_models_carry_the_fields():
    assert DiffusionStatusResponse.model_fields["supports_inpaint_only_masked"].default is False
    for name in ("mask_blur", "inpaint_only_masked", "inpaint_padding"):
        assert GalleryImage.model_fields[name].default is None


def test_route_forwards_only_set_fields():
    from routes.inference import _inpaint_mask_kwargs

    def diffusers_generate(
        prompt,
        mask_blur = None,
        inpaint_only_masked = None,
        inpaint_padding = None,
    ):
        pass

    def native_generate(prompt, mask_image = None):
        pass

    unset = DiffusionGenerateRequest(prompt = "x")
    assert _inpaint_mask_kwargs(diffusers_generate, unset) == {}
    full = DiffusionGenerateRequest(
        prompt = "x", mask_blur = 4, inpaint_only_masked = True, inpaint_padding = 16
    )
    assert _inpaint_mask_kwargs(diffusers_generate, full) == {
        "mask_blur": 4,
        "inpaint_only_masked": True,
        "inpaint_padding": 16,
    }
    assert _inpaint_mask_kwargs(native_generate, full) == {}


def test_a1111_parameters_text():
    from core.inference.image_gallery import _params_text

    base = {
        "prompt": "p",
        "steps": 4,
        "guidance": 5,
        "seed": 1,
        "width": 64,
        "height": 64,
        "workflow": "inpaint",
        "strength": 0.8,
    }
    plain = _params_text(base)
    assert "Mask blur" not in plain and "Inpaint area" not in plain
    text = _params_text(
        {**base, "mask_blur": 4, "inpaint_only_masked": True, "inpaint_padding": 32}
    )
    assert "Mask blur: 4, Inpaint area: Only masked, Masked area padding: 32" in text


class _CropInpaintPipe(_FakeInpaintPipe):
    def __call__(
        self,
        *,
        prompt = None,
        image = None,
        mask_image = None,
        padding_mask_crop = None,
        **kw,
    ):
        return super().__call__(
            prompt = prompt,
            image = image,
            mask_image = mask_image,
            padding_mask_crop = padding_mask_crop,
            **kw,
        )


class _CropInpaintPipeline(_FakeInpaintPipeline):
    __call__ = _CropInpaintPipe.__call__

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeInpaintPipeline.from_pipe(base_pipe, **kwargs)
        return _CropInpaintPipe()


def _inpaint(backend, **extra):
    return backend.generate(
        prompt = "a red door",
        steps = 4,
        guidance = 0.0,
        seed = 5,
        init_image = _tiny_png_b64(),
        mask_image = _tiny_mask_b64(),
        strength = 0.7,
        **extra,
    )


def _call():
    call = dict(_FakeInpaintPipe.last_kwargs)
    return call, call["mask_image"].tobytes()


def test_unset_none_zero_false_keep_the_plain_inpaint_call(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)
    out = _inpaint(backend)
    plain, plain_mask = _call()
    assert "padding_mask_crop" not in plain
    assert out["mask_blur"] is None and out["inpaint_only_masked"] is None
    assert out["inpaint_padding"] is None
    for extra in (
        {"mask_blur": None, "inpaint_only_masked": None, "inpaint_padding": None},
        {"mask_blur": 0, "inpaint_only_masked": False, "inpaint_padding": 64},
    ):
        out = _inpaint(backend, **extra)
        call, mask = _call()
        assert mask == plain_mask
        assert set(call) == set(plain)
        assert out["mask_blur"] is None and out["inpaint_padding"] is None


def test_mask_blur_blurs_the_mask_handed_to_the_pipeline(fake_runtime, tmp_path):
    from PIL import ImageFilter

    backend = _loaded_backend(tmp_path)
    _inpaint(backend)
    _, plain_mask = _call()
    plain_img = _FakeInpaintPipe.last_kwargs["mask_image"]
    out = _inpaint(backend, mask_blur = 4)
    _, blurred = _call()
    assert blurred != plain_mask
    assert blurred == plain_img.filter(ImageFilter.GaussianBlur(4)).tobytes()
    assert out["mask_blur"] == 4


def test_mask_settings_ignored_outside_inpaint(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)
    out = backend.generate(
        prompt = "x", steps = 4, mask_blur = 8, inpaint_only_masked = True, inpaint_padding = 8
    )
    assert out["mask_blur"] is None and out["inpaint_only_masked"] is None


def test_only_masked_refused_when_the_pipeline_cannot_crop(fake_runtime, tmp_path):
    backend = _loaded_backend(tmp_path)
    assert "padding_mask_crop" not in inspect.signature(_FakeInpaintPipe.__call__).parameters
    assert backend.status()["supports_inpaint_only_masked"] is False
    with pytest.raises(ValueError, match = "Only masked"):
        _inpaint(backend, inpaint_only_masked = True)


def test_only_masked_passes_padding_mask_crop(fake_runtime, tmp_path, monkeypatch):
    import diffusers

    # generate() resolves the family's inpaint class by name; the status probe reads its __call__.
    monkeypatch.setattr(diffusers, "ZImageInpaintPipeline", _CropInpaintPipeline, raising = False)
    backend = _loaded_backend(tmp_path)
    assert backend.status()["supports_inpaint_only_masked"] is True
    out = _inpaint(backend, inpaint_only_masked = True)
    assert _FakeInpaintPipe.last_kwargs["padding_mask_crop"] == 32
    assert (out["inpaint_only_masked"], out["inpaint_padding"]) == (True, 32)
    out = _inpaint(backend, inpaint_only_masked = True, inpaint_padding = 0)
    assert _FakeInpaintPipe.last_kwargs["padding_mask_crop"] == 0
    out = _inpaint(backend, inpaint_only_masked = False, inpaint_padding = 50)
    assert _FakeInpaintPipe.last_kwargs.get("padding_mask_crop") is None
    assert out["inpaint_only_masked"] is None


def test_status_flag_false_when_nothing_loaded():
    from core.inference.diffusion import DiffusionBackend
    assert DiffusionBackend().status().get("supports_inpaint_only_masked", False) is False


@pytest.mark.parametrize(
    "field",
    [
        {"mask_blur": 4},
        {"inpaint_only_masked": True},
        {"inpaint_padding": 48},
        {"detailer": {"targets": "face"}},
    ],
)
def test_external_connection_refuses_inpaint_mask_fields(client, monkeypatch, field):
    import routes.inference as inference_routes

    looked_up = []
    monkeypatch.setattr(
        inference_routes.providers_db, "get_provider", lambda pid: looked_up.append(pid)
    )
    response = client.post(
        "/api/inference/images/generate",
        json = {"prompt": "p", "provider_id": "openai-1", "model": "gpt-image-1", **field},
    )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "An external image connection only supports text to image."
    assert looked_up == []


class _PilInpaintPipe(_FakeInpaintPipe):
    """Returns a solid green picture over the whole canvas, as a pipeline with no overlay does."""

    def __call__(self, **kw):
        from PIL import Image

        super().__call__(**kw)
        n = kw.get("num_images_per_prompt", 1)
        size = kw["image"].size
        return types.SimpleNamespace(images = [Image.new("RGB", size, (0, 200, 0)) for _ in range(n)])


class _PilInpaintPipeline(_FakeInpaintPipeline):
    __call__ = _PilInpaintPipe.__call__

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeInpaintPipeline.from_pipe(base_pipe, **kwargs)
        return _PilInpaintPipe()


class _PilCropInpaintPipe(_PilInpaintPipe):
    def __call__(
        self,
        *,
        padding_mask_crop = None,
        **kw,
    ):
        return super().__call__(padding_mask_crop = padding_mask_crop, **kw)


class _PilCropInpaintPipeline(_FakeInpaintPipeline):
    __call__ = _PilCropInpaintPipe.__call__

    @classmethod
    def from_pipe(cls, base_pipe, **kwargs):
        _FakeInpaintPipeline.from_pipe(base_pipe, **kwargs)
        return _PilCropInpaintPipe()


def test_mask_blur_pastes_the_render_back_through_the_soft_mask(
    fake_runtime, tmp_path, monkeypatch
):
    import diffusers

    # Real inpaint pipelines binarize mask_image, so the soft edge only reaches the picture through the overlay.
    monkeypatch.setattr(diffusers, "ZImageInpaintPipeline", _PilInpaintPipeline, raising = False)
    backend = _loaded_backend(tmp_path)
    out = _inpaint(backend, mask_blur = 4)
    img = out["images"][0]
    source, render = (120, 30, 30), (0, 200, 0)
    # Mask square is x, y in [16, 48): far outside keeps the source, the centre is the render.
    assert img.getpixel((2, 2)) == source
    assert img.getpixel((32, 32)) == render
    edge = img.getpixel((15, 32))
    assert edge not in (source, render)
    assert render[1] > edge[1] > source[1] and source[0] > edge[0] > render[0]
    assert out["mask_blur"] == 4
    # Unblurred inpaint is unchanged: the pipeline's own output comes back as is.
    plain = _inpaint(backend)["images"][0]
    assert plain.getpixel((2, 2)) == render


def test_only_masked_with_blur_is_not_overlaid_twice(fake_runtime, tmp_path, monkeypatch):
    import diffusers

    # padding_mask_crop already pastes back through the blurred mask inside the pipeline.
    monkeypatch.setattr(diffusers, "ZImageInpaintPipeline", _PilCropInpaintPipeline, raising = False)
    backend = _loaded_backend(tmp_path)
    out = _inpaint(backend, mask_blur = 4, inpaint_only_masked = True)
    assert _FakeInpaintPipe.last_kwargs["padding_mask_crop"] == 32
    assert out["images"][0].getpixel((2, 2)) == (0, 200, 0)


def test_only_masked_with_an_empty_mask_runs_as_a_plain_inpaint(
    fake_runtime, tmp_path, monkeypatch
):
    import base64
    import io

    import diffusers
    from PIL import Image

    # diffusers' crop region needs a nonzero mask pixel; an all-black mask would fail before denoising.
    monkeypatch.setattr(diffusers, "ZImageInpaintPipeline", _CropInpaintPipeline, raising = False)
    backend = _loaded_backend(tmp_path)
    buf = io.BytesIO()
    Image.new("L", (64, 64), 0).save(buf, format = "PNG")
    out = backend.generate(
        prompt = "a red door",
        steps = 4,
        guidance = 0.0,
        seed = 5,
        init_image = _tiny_png_b64(),
        mask_image = base64.b64encode(buf.getvalue()).decode(),
        strength = 0.7,
        inpaint_only_masked = True,
        mask_blur = 4,
    )
    assert _FakeInpaintPipe.last_kwargs.get("padding_mask_crop") is None
    assert out["inpaint_only_masked"] is None and out["inpaint_padding"] is None
    # A painted mask still crops.
    _inpaint(backend, inpaint_only_masked = True)
    assert _FakeInpaintPipe.last_kwargs["padding_mask_crop"] == 32
