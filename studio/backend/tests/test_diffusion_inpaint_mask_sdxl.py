# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inpaint "Mask blur" / "Only masked" through Studio's real diffusers backend with tiny SDXL."""

from __future__ import annotations

import base64
import functools
import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")
np = pytest.importorskip("numpy")

from PIL import Image, ImageFilter  # noqa: E402

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

TINY_SDXL = "hf-internal-testing/tiny-stable-diffusion-xl-pipe"
SIZE = 128


def _b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _source():
    arr = np.zeros((SIZE, SIZE, 3), dtype = np.uint8)
    arr[..., 0] = np.linspace(0, 255, SIZE, dtype = np.uint8)[None, :]
    arr[..., 1] = np.linspace(255, 0, SIZE, dtype = np.uint8)[:, None]
    arr[..., 2] = 90
    return Image.fromarray(arr, "RGB")


def _mask():
    m = Image.new("L", (SIZE, SIZE), 0)
    m.paste(255, (40, 48, 72, 80))
    return m


@pytest.fixture(scope = "module")
def backend():
    from conftest import allow_outbound
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(TINY_SDXL, local_files_only = True)
    except Exception:  # noqa: BLE001
        try:
            with allow_outbound():
                path = snapshot_download(TINY_SDXL)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"tiny SDXL pipeline not reachable: {exc}")
    from core.inference.diffusion import DiffusionBackend

    b = DiffusionBackend()
    # Eager: Speed=Auto compiles from the 3rd image, whose kernels round differently from eager.
    b.load_pipeline(path, family_override = "sdxl", speed_mode = "off")
    yield b
    b.unload()


@pytest.fixture
def spy(monkeypatch):
    """Record every StableDiffusionXLInpaintPipeline call's mask + padding_mask_crop; wraps keeps the signature."""
    calls = []
    cls = diffusers.StableDiffusionXLInpaintPipeline
    original = cls.__call__

    @functools.wraps(original)
    def recording(self, *args, **kwargs):
        calls.append(
            {
                "mask": np.array(kwargs.get("mask_image")).copy(),
                "padding_mask_crop": kwargs.get("padding_mask_crop", "absent"),
            }
        )
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, "__call__", recording)
    return calls


def _render(backend, **extra):
    out = backend.generate(
        prompt = "a red door",
        steps = 3,
        guidance = 5.0,
        seed = 7,
        init_image = _b64(_source()),
        mask_image = _b64(_mask()),
        strength = 0.9,
        **extra,
    )
    assert out["workflow"] == "inpaint"
    return out, np.asarray(out["images"][0].convert("RGB"), dtype = np.int16)


def test_status_reports_only_masked(backend):
    st = backend.status()
    assert "inpaint" in st["workflows"]
    assert st["supports_inpaint_only_masked"] is True


def test_unset_none_zero_false_are_bit_identical(backend, spy):
    out, ref = _render(backend)
    assert out["mask_blur"] is None and out["inpaint_only_masked"] is None
    assert spy[-1]["padding_mask_crop"] == "absent"
    ref_mask = spy[-1]["mask"]
    # A/A: the same request twice is bit-identical, so any difference below is the setting.
    _, again = _render(backend)
    assert np.array_equal(again, ref)
    for extra in (
        {"mask_blur": None, "inpaint_only_masked": None, "inpaint_padding": None},
        {"mask_blur": 0, "inpaint_only_masked": False, "inpaint_padding": 64},
    ):
        _, img = _render(backend, **extra)
        assert np.array_equal(img, ref), extra
        assert np.array_equal(spy[-1]["mask"], ref_mask)
        assert spy[-1]["padding_mask_crop"] == "absent"


def test_mask_blur_changes_mask_and_output(backend, spy):
    _, ref = _render(backend)
    ref_mask = spy[-1]["mask"]
    out, img = _render(backend, mask_blur = 8)
    blurred = spy[-1]["mask"]
    assert out["mask_blur"] == 8
    assert not np.array_equal(blurred, ref_mask)
    assert len(np.unique(ref_mask)) == 2 and len(np.unique(blurred)) > 2
    assert img.shape == ref.shape
    assert int(np.abs(img - ref).max()) > 0


def test_only_masked_crops_and_keeps_source_size(backend, spy):
    _, ref = _render(backend)
    out, img = _render(backend, inpaint_only_masked = True)
    assert spy[-1]["padding_mask_crop"] == 32
    assert (out["inpaint_only_masked"], out["inpaint_padding"]) == (True, 32)
    assert img.shape == (SIZE, SIZE, 3)
    assert int(np.abs(img - ref).max()) > 0
    src = np.asarray(_source(), dtype = np.int16)
    assert int(np.abs(img[:8, :8] - src[:8, :8]).max()) <= 2
    _, img16 = _render(backend, inpaint_only_masked = True, inpaint_padding = 16)
    assert spy[-1]["padding_mask_crop"] == 16
    assert not np.array_equal(img16, img)


def test_only_masked_refused_when_pipeline_lacks_padding_mask_crop(backend, monkeypatch):
    cls = diffusers.StableDiffusionXLInpaintPipeline
    original = cls.__call__

    def no_crop(
        self,
        prompt = None,
        image = None,
        mask_image = None,
        **kwargs,
    ):
        return original(self, prompt = prompt, image = image, mask_image = mask_image, **kwargs)

    monkeypatch.setattr(cls, "__call__", no_crop)
    assert backend.status()["supports_inpaint_only_masked"] is False
    with pytest.raises(ValueError, match = "Only masked"):
        _render(backend, inpaint_only_masked = True)


def test_mask_blur_feathers_the_render_into_the_source(backend, monkeypatch):
    # The pipeline binarizes mask_image, so the blur reaches the picture only through the overlay back onto the source.
    cls = diffusers.StableDiffusionXLInpaintPipeline
    original = cls.__call__
    raw = []

    @functools.wraps(original)
    def recording(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        raw.append(np.asarray(result.images[0].convert("RGB"), dtype = np.float64))
        return result

    monkeypatch.setattr(cls, "__call__", recording)
    _, ref = _render(backend)
    _, img = _render(backend, mask_blur = 8)
    src = np.asarray(_source(), dtype = np.float64)
    # Far from the mask the source survives; the plain render re-encodes it through the VAE.
    assert int(np.abs(img[:8, :8] - src[:8, :8]).max()) <= 2
    assert int(np.abs(ref[:8, :8] - src[:8, :8]).max()) > 2
    # Everywhere, A1111's apply_overlay: the render pasted over the source through the blurred mask.
    m = (np.asarray(_mask().filter(ImageFilter.GaussianBlur(8)), dtype = np.float64) / 255.0)[
        ..., None
    ]
    expected = m * raw[-1] + (1 - m) * src
    assert float(np.abs(img - expected).max()) <= 1.5
    # Just outside the painted square the mask is partial, so the pixel is a real mix of the two.
    y, x = 64, 36
    assert 0.05 < m[y, x, 0] < 0.5
    assert float(np.abs(raw[-1][y, x] - src[y, x]).max()) > 10
