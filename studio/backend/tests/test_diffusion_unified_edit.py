# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 unified editing: the reference / edit dispatch, image order and alpha, output
geometry, count limits and the localized-edit layers, against the real generate() with a pipeline
double whose signature matches QwenImage21Pipeline's."""

from __future__ import annotations

import base64
import io
import types

import pytest
from PIL import Image

from core.inference import diffusion_conditioning as cond
from core.inference.diffusion import DiffusionBackend, _compile_shape_dims
from core.inference.diffusion_families import detect_family, default_generation_params
from core.inference.diffusion_memory import estimate_image_runtime_mib

from .test_diffusion_backend import (  # noqa: F401
    _FakeImage,
    _FakePipe,
    _FakeTransformer,
    _load_into,
    fake_runtime,
)


class _Fake21Pipe(_FakePipe):
    """QwenImage21Pipeline's call signature: ``image`` list, explicit size, ``output_resolution``."""

    def __call__(
        self,
        *,
        prompt=None,
        image=None,
        negative_prompt=None,
        true_cfg_scale=1.0,
        height=None,
        width=None,
        num_inference_steps=40,
        num_images_per_prompt=1,
        generator=None,
        callback_on_step_end=None,
        output_resolution=1024,
        use_kv_cache=True,
    ):
        self.last_kwargs = {
            "prompt": prompt,
            "image": image,
            "height": height,
            "width": width,
            "true_cfg_scale": true_cfg_scale,
            "output_resolution": output_resolution,
            "num_images_per_prompt": num_images_per_prompt,
        }
        return types.SimpleNamespace(images=[_FakeImage() for _ in range(num_images_per_prompt)])


class _Fake21Pipeline:
    @classmethod
    def from_pretrained(cls, base, **kwargs):
        return _Fake21Pipe()


def _images(call) -> list:
    """The ``image`` argument as a list: one image goes to the pipeline bare, as upstream accepts."""
    image = call["image"]
    return image if isinstance(image, list) else [image]


def _png(
    size=(64, 64),
    color=(120, 30, 30, 255),
    mode="RGBA",
    exif=None,
) -> str:
    buf = io.BytesIO()
    img = Image.new(mode, size, color if mode != "RGB" else color[:3])
    kwargs = {"exif": exif} if exif is not None else {}
    img.save(buf, format="PNG", **kwargs)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def backend21(fake_runtime, tmp_path):
    import diffusers

    diffusers.QwenImage21Pipeline = _Fake21Pipeline
    diffusers.QwenImage21Transformer2DModel = _FakeTransformer
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(backend, tmp_path, family_override="qwen-image-2.1")
    return backend


def _flux2(fake_runtime, tmp_path):
    import diffusers

    from .test_diffusion_backend import _FakeInpaintPipeline, _FakePipeline

    diffusers.Flux2KleinPipeline = _FakePipeline
    diffusers.Flux2KleinInpaintPipeline = _FakeInpaintPipeline
    diffusers.Flux2Transformer2DModel = _FakeTransformer
    (tmp_path / "model.gguf").write_bytes(b"x")
    backend = DiffusionBackend()
    _load_into(backend, tmp_path, family_override="flux.2-klein")
    return backend


def test_status_advertises_create_reference_and_edit_with_its_limits(backend21):
    status = backend21.status()
    assert status["workflows"] == ["txt2img", "reference", "edit"]
    c = status["conditioning"]
    assert c["max_condition_images"] == 10 and c["alpha"] is True
    assert c["dimension_multiple"] == 32
    assert (c["max_output_side"], c["max_output_pixels"]) == (2752, 2400 * 1792)
    assert c["reference_resolutions"] == [512, 1024, 2048]
    assert c["unified_edit"] is True and c["localized_edit_modes"] == ["annotate", "paint", "mask"]


def test_edit_only_and_reference_families_keep_their_contracts():
    kontext = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    assert kontext.edit is True and kontext.unified_edit is False
    flux2 = detect_family("black-forest-labs/FLUX.2-klein-4B")
    assert flux2.reference is True and flux2.unified_edit is False
    assert flux2.max_condition_images == 4 and flux2.condition_image_mode == "RGB"
    from core.inference.diffusion import _family_workflows

    assert _family_workflows(kontext) == ["edit"]
    assert "edit" not in _family_workflows(flux2)


def test_sampling_defaults_resolve_for_every_21_artifact_name():
    for name in (
        "Qwen/Qwen-Image-2.1",
        "unsloth/Qwen-Image-2.1-GGUF",
        "unsloth/Qwen-Image-2.1-FP8",
    ):
        assert default_generation_params(name) == (40, 1.0), name
    # The generic key still owns the other Qwen-Image checkpoints.
    assert default_generation_params("Qwen/Qwen-Image-2512") == (20, 4.0)


def test_explicit_edit_hands_ordered_rgba_images_and_explicit_geometry(backend21):
    colors = [(10 * i, 0, 0, 40 + i) for i in range(1, 5)]
    out = backend21.generate(
        prompt="put them together",
        steps=4,
        seed=1,
        width=1024,
        height=768,
        workflow="edit",
        init_image=_png(color=colors[0]),
        reference_images=[_png(color=c) for c in colors[1:]],
        reference_resolution=512,
    )
    call = backend21._state.pipe.last_kwargs
    assert [im.mode for im in call["image"]] == ["RGBA"] * 4
    # Order kept exactly, alpha included.
    assert [im.getpixel((0, 0)) for im in call["image"]] == colors
    assert (call["width"], call["height"]) == (1024, 768)
    assert call["output_resolution"] == 512
    assert out["workflow"] == "edit" and out["reference_resolution"] == 512


def test_omitted_size_matches_image_1_not_the_last_reference(backend21):
    backend21.generate(
        prompt="edit",
        steps=4,
        seed=1,
        width=None,
        height=None,
        workflow="edit",
        init_image=_png(size=(300, 200)),
        reference_images=[_png(size=(100, 400))],
    )
    call = backend21._state.pipe.last_kwargs
    # 3:2 at the 1024 area on the 32 px grid, as upstream's calculate_dimensions rounds it.
    assert (call["width"], call["height"]) == (1248, 832)
    assert cond.match_source_size(detect_family("Qwen/Qwen-Image-2.1"), (300, 200), 1024) == (
        1248,
        832,
    )


def test_unified_edit_registers_the_requested_shape_not_the_source(backend21):
    fam = backend21._state.family
    src = Image.new("RGBA", (96, 64))
    assert _compile_shape_dims("edit", src, 1024, 768, fam) == (1024, 768)
    # Edit-only families still size from the source.
    kontext = detect_family("black-forest-labs/FLUX.1-Kontext-dev")
    assert _compile_shape_dims("edit", src, 1024, 768, kontext) == (96, 64)


@pytest.mark.parametrize("extras,ok", [(0, True), (3, True), (9, True), (10, False)])
def test_ten_images_in_total_and_overflow_is_refused(backend21, extras, ok):
    kwargs = dict(
        prompt="p",
        steps=2,
        workflow="edit",
        init_image=_png(),
        reference_images=[_png() for _ in range(extras)],
        width=512,
        height=512,
    )
    if ok:
        backend21.generate(**kwargs)
        assert len(_images(backend21._state.pipe.last_kwargs)) == 1 + extras
    else:
        with pytest.raises(ValueError, match="at most 10 input images"):
            backend21.generate(**kwargs)


def test_flux2_keeps_four_images_and_refuses_the_fifth(fake_runtime, tmp_path):
    backend = _flux2(fake_runtime, tmp_path)
    backend.generate(
        prompt="p",
        steps=2,
        init_image=_png(mode="RGB"),
        reference_images=[_png(mode="RGB") for _ in range(3)],
    )
    assert all(im.mode == "RGB" for im in backend._state.pipe.last_kwargs["image"])
    with pytest.raises(ValueError, match="at most 4 input images"):
        backend.generate(
            prompt="p",
            steps=2,
            init_image=_png(mode="RGB"),
            reference_images=[_png(mode="RGB") for _ in range(4)],
        )
    with pytest.raises(ValueError, match="Instruction editing is not supported"):
        backend.generate(prompt="p", steps=2, workflow="edit", init_image=_png())
    with pytest.raises(ValueError, match="reference_resolution is not supported"):
        backend.generate(prompt="p", steps=2, init_image=_png(), reference_resolution=1024)


def test_output_bounds_follow_the_family(backend21, fake_runtime, tmp_path):
    base = dict(prompt="p", steps=2)
    with pytest.raises(ValueError, match="multiples of 32"):
        backend21.generate(width=1040, height=1024, **base)
    backend21.generate(width=2752, height=1536, **base)
    with pytest.raises(ValueError, match="at most 4,300,800 pixels"):
        backend21.generate(width=2752, height=2752, **base)
    with pytest.raises(ValueError, match="2752px per side"):
        backend21.generate(width=2784, height=512, **base)
    # Another family keeps 2048 even though the transport now carries 2752.
    other = DiffusionBackend()
    _load_into(other, tmp_path)
    with pytest.raises(ValueError, match="2048px per side"):
        other.generate(width=2752, height=1536, **base)


@pytest.mark.parametrize(
    "extra,match",
    [
        ({"strength": 0.5}, "strength is not supported"),
        ({"mask_image": _png(mode="L", color=(255,))}, "mask_image is not supported"),
        ({"upscale": 2.0}, "upscale is not supported"),
        (
            {"controlnet": ("cn", _png(), "canny", 0.7, 0.0, 1.0)},
            "controlnet is not supported",
        ),
    ],
)
def test_explicit_workflows_refuse_what_they_would_drop(backend21, extra, match):
    with pytest.raises(ValueError, match=match):
        backend21.generate(prompt="p", steps=2, workflow="edit", init_image=_png(), **extra)


def test_missing_source_and_misplaced_settings_are_refused(backend21):
    with pytest.raises(ValueError, match="requires a source image"):
        backend21.generate(prompt="p", steps=2, workflow="edit")
    with pytest.raises(ValueError, match="applies only to the reference and edit"):
        backend21.generate(prompt="p", steps=2, reference_resolution=1024)
    with pytest.raises(ValueError, match="must be one of 512, 1024, 2048"):
        backend21.generate(
            prompt="p", steps=2, workflow="edit", init_image=_png(), reference_resolution=768
        )
    with pytest.raises(ValueError, match="localized_edit needs the edit workflow"):
        backend21.generate(
            prompt="p",
            steps=2,
            workflow="reference",
            init_image=_png(),
            localized_edit=cond.LocalizedEdit("paint", _png(mode="L", color=(255,))),
        )


def test_omitted_workflow_keeps_the_reference_path(backend21):
    out = backend21.generate(prompt="p", steps=2, width=512, height=512, init_image=_png())
    assert out["workflow"] == "reference"
    call = backend21._state.pipe.last_kwargs
    assert (call["width"], call["height"]) == (512, 512)
    assert call["output_resolution"] == 1024


def test_annotate_composites_marks_and_keeps_alpha_elsewhere(backend21):
    overlay = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    overlay.putpixel((5, 5), (255, 0, 0, 255))
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    backend21.generate(
        prompt="p",
        steps=2,
        width=512,
        height=512,
        workflow="edit",
        init_image=_png(color=(0, 0, 200, 0)),
        localized_edit=cond.LocalizedEdit("annotate", base64.b64encode(buf.getvalue()).decode()),
    )
    marked = _images(backend21._state.pipe.last_kwargs)
    assert len(marked) == 1 and marked[0].mode == "RGBA"
    assert marked[0].getpixel((5, 5)) == (255, 0, 0, 255)
    assert marked[0].getpixel((30, 30)) == (0, 0, 200, 0)


def test_paint_whitens_the_region_at_the_source_geometry(backend21):
    mask = Image.new("L", (32, 32), 0)
    for x in range(16):
        for y in range(16):
            mask.putpixel((x, y), 255)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    backend21.generate(
        prompt="p",
        steps=2,
        width=512,
        height=512,
        workflow="edit",
        init_image=_png(color=(10, 20, 30, 255)),
        localized_edit=cond.LocalizedEdit("paint", base64.b64encode(buf.getvalue()).decode()),
    )
    painted = _images(backend21._state.pipe.last_kwargs)[0]
    assert painted.size == (64, 64)  # the half-size layer was scaled to the source
    assert painted.getpixel((4, 4)) == (255, 255, 255, 255)
    assert painted.getpixel((60, 60)) == (10, 20, 30, 255)


def test_mask_is_image_2_binary_white_region_and_counts_toward_the_limit(backend21):
    mask = Image.new("L", (64, 64), 0)
    mask.putpixel((1, 1), 200)
    buf = io.BytesIO()
    mask.save(buf, format="PNG")
    layer = base64.b64encode(buf.getvalue()).decode()
    backend21.generate(
        prompt="p",
        steps=2,
        width=512,
        height=512,
        workflow="edit",
        init_image=_png(color=(9, 9, 9, 255)),
        reference_images=[_png(color=(1, 2, 3, 255))],
        localized_edit=cond.LocalizedEdit("mask", layer),
    )
    images = backend21._state.pipe.last_kwargs["image"]
    assert [im.getpixel((0, 0))[:3] for im in images] == [(9, 9, 9), (0, 0, 0), (1, 2, 3)]
    assert images[1].getpixel((1, 1))[:3] == (255, 255, 255)
    with pytest.raises(ValueError, match="counting the mask"):
        backend21.generate(
            prompt="p",
            steps=2,
            width=512,
            height=512,
            workflow="edit",
            init_image=_png(),
            reference_images=[_png() for _ in range(9)],
            localized_edit=cond.LocalizedEdit("mask", layer),
        )
    empty = io.BytesIO()
    Image.new("L", (64, 64), 0).save(empty, format="PNG")
    with pytest.raises(ValueError, match="mask is empty"):
        backend21.generate(
            prompt="p",
            steps=2,
            width=512,
            height=512,
            workflow="edit",
            init_image=_png(),
            localized_edit=cond.LocalizedEdit("mask", base64.b64encode(empty.getvalue()).decode()),
        )


def test_layer_for_another_image_is_refused():
    fam = detect_family("Qwen/Qwen-Image-2.1")
    with pytest.raises(ValueError, match="same image"):
        cond.decode_condition_images(
            fam,
            _png(size=(64, 64)),
            None,
            cond.LocalizedEdit("paint", _png(size=(64, 32), mode="L", color=(255,))),
        )


def test_transparent_palette_png_and_exif_rotation_decode_for_the_family():
    fam = detect_family("Qwen/Qwen-Image-2.1")
    pal = Image.new("P", (40, 20))
    pal.putpalette([255, 0, 0] + [0, 0, 0] * 255)
    pal.info["transparency"] = 0
    buf = io.BytesIO()
    pal.save(buf, format="PNG", transparency=0)
    (img,) = cond.decode_condition_images(fam, base64.b64encode(buf.getvalue()).decode(), None)
    assert img.mode == "RGBA" and img.getpixel((0, 0))[3] == 0

    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    jpg = io.BytesIO()
    Image.new("RGB", (40, 20)).save(jpg, format="JPEG", exif=exif.tobytes())
    (rot,) = cond.decode_condition_images(fam, base64.b64encode(jpg.getvalue()).decode(), None)
    assert rot.size == (20, 40)


def test_aggregate_source_pixels_are_bounded_before_decoding(monkeypatch):
    fam = detect_family("Qwen/Qwen-Image-2.1")
    monkeypatch.setattr(cond, "MAX_CONDITION_SOURCE_PIXELS", 64 * 64 * 2)
    with pytest.raises(ValueError, match="too large together"):
        cond.decode_condition_images(fam, _png(), [_png(), _png()])


def test_condition_images_count_toward_working_memory():
    base = estimate_image_runtime_mib(width=1024, height=1024)
    weight = detect_family("Qwen/Qwen-Image-2.1").condition_pixel_weight
    four = estimate_image_runtime_mib(
        width=1024, height=1024, condition_pixels=int(4 * 1024 * 1024 * weight)
    )
    # Measured: about 2.5 GiB per 1024-square reference on top of the output's own.
    assert 4 * 2400 <= four - base <= 4 * 2900


def test_offloaded_vision_tower_builds_position_embeddings_on_the_compute_device():
    """Under leaf-level group offload Qwen3-VL read ``pos_embed.weight.device`` while the weight was
    still on the CPU, so every image-conditioned call on a streamed encoder died with a device
    mismatch. The pin moves the result to ``grid_thw``'s device, per instance."""
    from core.inference.diffusion_memory import _pin_vision_embedding_device

    class _Out:
        def __init__(self, device):
            self.device = device

        def to(self, device):
            return _Out(device)

    class _Vision:
        def fast_pos_embed_interpolate(self, grid_thw):
            return _Out("cpu")

        def modules(self):
            return iter([self])

    vision = _Vision()
    assert _pin_vision_embedding_device(vision) == 1
    assert (
        vision.fast_pos_embed_interpolate(types.SimpleNamespace(device="cuda:0")).device == "cuda:0"
    )
    # Idempotent, and a module without the method or without modules() is left alone.
    assert _pin_vision_embedding_device(vision) == 0
    assert _pin_vision_embedding_device(object()) == 0
    assert _Vision().fast_pos_embed_interpolate(None).device == "cpu"


@pytest.mark.parametrize(
    "source,resolution",
    [((4096, 256), 512), ((256, 4096), 512), ((20000, 10), 1024), ((4096, 256), 2048)],
)
def test_an_elongated_source_matches_to_a_size_the_request_accepts(source, resolution):
    """4096x256 at the 512 tier rounded to 2048x128, which the API refuses (both sides >= 256).
    The short side now stays at the minimum and the long side is capped at the family's bound."""
    for name in ("Qwen/Qwen-Image-2.1", "Tongyi-MAI/Z-Image-Turbo"):
        fam = detect_family(name)
        w, h = cond.match_source_size(fam, source, resolution)
        cond.check_output_size(fam, w, h)
    assert cond.match_source_size(detect_family("Qwen/Qwen-Image-2.1"), (4096, 256), 512) == (
        2752,
        256,
    )


def test_an_output_side_below_the_minimum_is_refused():
    with pytest.raises(ValueError, match="at least 256px"):
        cond.check_output_size(detect_family("Qwen/Qwen-Image-2.1"), 2048, 128)
