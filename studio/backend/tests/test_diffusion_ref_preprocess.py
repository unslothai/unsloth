"""CPU tests for core.inference.diffusion_ref_preprocess (GPU reference preprocessing for FLUX.2 edits)."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from core.inference import diffusion_ref_preprocess as rp  # noqa: E402

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

flux2_ip = pytest.importorskip("diffusers.pipelines.flux2.image_processor")


def _img(
    w,
    h,
    seed = 0,
    mode = "RGB",
):
    rng = np.random.default_rng(seed)
    im = Image.fromarray(rng.integers(0, 256, size = (h, w, 3), dtype = np.uint8), "RGB")
    return im.convert(mode) if mode != "RGB" else im


class _Pipe:
    def __init__(self, device = None):
        self.image_processor = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
        self._execution_device = device


def _stock_chain(proc, img):
    w, h = img.size
    if w * h > 1024 * 1024:
        img = proc._resize_to_target_area(img, 1024 * 1024)
        w, h = img.size
    return proc.preprocess(img, height = h // 16 * 16, width = w // 16 * 16, resize_mode = "crop")


def test_env_switch(monkeypatch):
    monkeypatch.setenv(rp.ENV, "0")
    assert not rp.enabled()
    monkeypatch.setenv(rp.ENV, "1")
    assert rp.enabled()
    monkeypatch.delenv(rp.ENV)
    assert rp.enabled()


def test_not_installed_without_cuda(monkeypatch):
    monkeypatch.setattr(rp, "_cuda_ok", lambda: False)
    pipe = _Pipe()
    assert rp.install(pipe) is False
    assert not getattr(pipe.image_processor, rp._MARK, False)


def test_only_flux2_processor(monkeypatch):
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)

    class P:
        image_processor = object()

    assert rp.install(P()) is False


def test_install_off_by_env(monkeypatch):
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)
    monkeypatch.setenv(rp.ENV, "0")
    assert rp.install(_Pipe()) is False


def test_deferred_resize_matches_pillow():
    img = _img(1500, 900, 1)
    w, h = rp._target_size(img, 1024 * 1024)
    proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    stock = proc._resize_to_target_area(img, 1024 * 1024)
    d = rp.DeferredResize(img, w, h)
    assert d.size == stock.size and (d.width, d.height) == stock.size
    assert d._resized is None  # size alone does not resize
    assert d.mode == "RGB" and d._resized is not None  # any other attribute materialises
    assert np.array_equal(np.asarray(d.materialize()), np.asarray(stock))


def test_installed_without_device_is_stock(monkeypatch):
    """Installed, but no CUDA execution device: both patched calls give the stock result exactly."""
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)
    pipe = _Pipe(device = None)
    stock_proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    assert rp.install(pipe) is True
    assert rp.install(pipe) is False  # idempotent
    for w, h in ((1500, 900), (640, 480)):
        img = _img(w, h, w)
        got = _stock_chain(pipe.image_processor, img)
        ref = _stock_chain(stock_proc, img)
        assert torch.equal(got, ref) and got.stride() == ref.stride()


@pytest.mark.parametrize("size", [(640, 480), (1023, 1023), (1000, 999), (513, 1025)])
@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_crop_branch_equal_on_cpu(size, mode):
    """Images at or under 1 MP take no resize: the crop + table path equals the stock tensor bit for bit."""
    proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    img = _img(*size, seed = size[0], mode = mode)
    w, h = img.size
    ref = proc.preprocess(img, height = h // 16 * 16, width = w // 16 * 16, resize_mode = "crop")
    got = rp._gpu_preprocess(proc, img, h // 16 * 16, w // 16 * 16, "cpu")
    assert got.dtype == ref.dtype and got.shape == ref.shape
    assert torch.equal(got, ref) and got.stride() == ref.stride()


def test_table_is_diffusers_normalize():
    proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    codes = np.arange(256, dtype = np.uint8)[None, :, None].repeat(3, axis = 2)
    ref = proc.normalize(proc.numpy_to_pt(proc.pil_to_numpy([Image.fromarray(codes, "RGB")])))
    assert torch.equal(rp._table("cpu"), ref[0, 0, 0])


def test_config_gate():
    proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    assert rp._config_ok(proc)
    proc2 = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16, do_normalize = False)
    assert not rp._config_ok(proc2)


def test_kernel_crop_dims_not_specialized():
    src = (BACKEND / "core" / "inference" / "diffusion_lanczos_triton.py").read_text()
    assert "crop_width: tl.constexpr" not in src and "crop_height: tl.constexpr" not in src
    assert "SPDX-License-Identifier: Apache-2.0" in src


def test_installed_at_load():
    src = (BACKEND / "core" / "inference" / "diffusion.py").read_text()
    assert "from .diffusion_ref_preprocess import install as install_ref_preprocess" in src
