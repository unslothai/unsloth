# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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


def test_not_installed_without_cuda(monkeypatch):
    monkeypatch.setattr(rp, "_cuda_ok", lambda: False)
    pipe = _Pipe()
    assert rp.install(pipe) is False
    assert not getattr(pipe.image_processor, rp._MARK, False)


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
    assert d._resized is None
    assert d.mode == "RGB" and d._resized is not None
    assert np.array_equal(np.asarray(d.materialize()), np.asarray(stock))


def test_installed_without_device_is_stock(monkeypatch):
    """Installed, but no CUDA execution device: both patched calls give the stock result exactly."""
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)
    pipe = _Pipe(device = None)
    stock_proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    assert rp.install(pipe) is True
    assert rp.install(pipe) is False
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


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P", "CMYK", "L"])
def test_non_rgb_resize_stays_stock(monkeypatch, mode):
    """Pillow resizes RGBA / LA premultiplied, P with NEAREST and CMYK in its own space before preprocess converts: only
    RGB sources may defer to the RGB Lanczos kernel; the rest must get the stock resized image."""
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)
    pipe = _Pipe(device = torch.device("cuda"))
    stock_proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    assert rp.install(pipe) is True
    rgb = _img(1500, 900, 7)
    assert isinstance(
        pipe.image_processor._resize_to_target_area(rgb, 1024 * 1024), rp.DeferredResize
    )
    src = _img(1500, 900, 7, mode = "RGB")
    if mode in ("RGBA", "LA"):
        alpha = Image.fromarray(
            np.random.default_rng(8).integers(0, 256, (900, 1500), dtype = np.uint8), "L"
        )
        src = src.convert(mode)
        src.putalpha(alpha)
    elif mode == "P":
        src = src.quantize(64)
    else:
        src = src.convert(mode)
    got = pipe.image_processor._resize_to_target_area(src, 1024 * 1024)
    ref = stock_proc._resize_to_target_area(src, 1024 * 1024)
    assert not isinstance(got, rp.DeferredResize)
    assert got.mode == ref.mode and np.array_equal(np.asarray(got), np.asarray(ref))


def test_kernel_launches_on_operand_device(monkeypatch):
    """Triton launches on the current device: the resize must run under torch.cuda.device(image.device)."""
    pytest.importorskip("triton")
    from core.inference import diffusion_lanczos_triton as K

    entered = []

    class _Dev:
        def __init__(self, d):
            self.d = d

        def __enter__(self):
            entered.append(self.d)

        def __exit__(self, *a):
            entered.append("exit")

    class _Img:
        is_cuda = True
        dtype = torch.uint8
        ndim = 3
        shape = (40, 30, 3)
        device = "cuda:1"

    from core.inference import rocm_bf16

    monkeypatch.setattr(rocm_bf16, "is_rocm_torch", lambda _t: False)
    monkeypatch.setattr(K.torch.cuda, "device", _Dev)
    monkeypatch.setattr(K, "_launch", lambda *a, **k: (list(entered), "out"))
    seen, out = K.lanczos_resize(_Img(), 20, 15)
    assert out == "out" and seen == ["cuda:1"] and entered == ["cuda:1", "exit"]


def _emulate(arr, w, h, vertical_first):
    """The kernel's integer math in numpy (Pillow 8 bpc rounding), in the given pass order."""
    from core.inference import diffusion_lanczos_triton as K

    def one(a, axis, n):
        c, b, _ = K._coefficients(a.shape[axis], n, torch.device("cpu"))
        c, b = c.numpy().astype(np.int64), b.numpy()
        a = np.moveaxis(a, axis, 0).astype(np.int64)
        out = np.empty((n, *a.shape[1:]), np.int64)
        for i in range(n):
            f, k = b[i]
            out[i] = (np.tensordot(c[i, :k], a[f : f + k], axes = 1) + (1 << 21)) >> 22
        return np.moveaxis(np.clip(out, 0, 255).astype(np.uint8), 0, axis)

    if vertical_first:
        return one(one(arr, 0, h), 1, w)
    return one(one(arr, 1, w), 0, h)


@pytest.mark.parametrize(
    "src, dst",
    [((60, 200), (30, 70)), ((200, 60), (70, 30)), ((40, 130), (37, 64)), ((97, 13), (11, 31))],
)
def test_pass_order_matches_installed_pillow(src, dst):
    """Pillow >= 13 runs the vertical pass first when the height shrinks by more than twice the width; the GPU path must
    pick the same order or tall references (phone screenshots, > 2:1) stop being byte-identical."""
    pytest.importorskip("triton")
    (sw, sh), (w, h) = src, dst
    arr = np.random.default_rng(sw * sh).integers(0, 256, (sh, sw, 3), dtype = np.uint8)
    ref = np.asarray(Image.fromarray(arr, "RGB").resize((w, h), Image.Resampling.LANCZOS))
    got = _emulate(arr, w, h, rp._pillow_vertical_first(sw, sh, w, h))
    assert np.array_equal(got, ref)


def _flux2_request(pipe, img):
    w, h = img.size
    proc = pipe.image_processor
    img = proc._resize_to_target_area(img, 512 * 512)
    w, h = img.size
    return proc.preprocess(img, height = h // 16 * 16, width = w // 16 * 16, resize_mode = "crop")


def test_warm_probes_at_load_not_on_the_first_request(monkeypatch):
    """warm() after placement runs the device probe; the first edit request then pays no probe / Triton compile."""
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)
    monkeypatch.setattr(rp, "_PROBED", {})
    calls = []
    monkeypatch.setattr(rp, "_probe", lambda proc, a, p, device: calls.append(str(device)) or True)
    sentinel = torch.zeros(1, 3, 16, 16)
    monkeypatch.setattr(rp, "_gpu_preprocess", lambda *a, **k: sentinel)
    pipe = _Pipe(device = None)
    assert rp.install(pipe) is True
    assert rp.warm(pipe) is None and calls == []  # not placed yet: nothing to probe
    pipe._execution_device = torch.device("cuda:0")
    assert rp.warm(pipe) is True
    assert calls == ["cuda:0"] and rp._PROBED == {"cuda:0": True}

    def _no_probe(*a, **k):
        raise AssertionError("probe ran on the request path")

    monkeypatch.setattr(rp, "_probe", _no_probe)
    assert _flux2_request(pipe, _img(1500, 900, 3)) is sentinel
    assert rp.warm(pipe) is True  # a second warm is a dict hit


def test_request_on_an_unwarmed_device_still_probes(monkeypatch):
    monkeypatch.setattr(rp, "_cuda_ok", lambda: True)
    monkeypatch.setattr(rp, "_PROBED", {})
    calls = []
    monkeypatch.setattr(rp, "_probe", lambda proc, a, p, device: calls.append(str(device)) or False)
    pipe = _Pipe(device = torch.device("cuda:1"))
    stock = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    assert rp.install(pipe) is True
    img = _img(1500, 900, 4)
    got = _flux2_request(pipe, img)
    assert calls == ["cuda:1"] and rp._PROBED == {"cuda:1": False}
    ref = _flux2_request(type("P", (), {"image_processor": stock})(), img)
    assert torch.equal(got, ref)


def test_warm_is_a_no_op_without_the_install(monkeypatch):
    monkeypatch.setattr(rp, "_PROBED", {})
    monkeypatch.setattr(
        rp, "_probe", lambda *a, **k: (_ for _ in ()).throw(AssertionError("probed"))
    )
    assert rp.warm(_Pipe(device = torch.device("cuda:0"))) is None
    assert rp.warm(object()) is None


def test_load_warms_after_placement():
    src = (BACKEND / "core" / "inference" / "diffusion.py").read_text(encoding = "utf-8")
    install_at = src.index("install_ref_preprocess(pipe, logger = logger)")
    placed_at = src.index("cuda_graph.arm_block_graphs(", install_at)
    warm_at = src.index("warm_ref_preprocess(pipe, logger = logger)")
    assert install_at < placed_at < warm_at


def test_lanczos_refuses_sizes_past_int32_offsets():
    """The kernels use 32-bit flat offsets per image: any buffer of 2**31 elements or more must be refused."""
    pytest.importorskip("triton")
    from core.inference import diffusion_lanczos_triton as K

    assert K.fits_int32(4000, 6000, 1024, 1536)
    side = 26755  # 3 * 26755**2 = 2147490075 >= 2**31
    assert not K.fits_int32(side, side, 1024, 1024)
    assert not K.fits_int32(1024, 1024, side, side)  # an upscale's output counts too
    assert not K.fits_int32(side, 100, 100, side)  # either pass order's intermediate
    assert K.fits_int32(26754, 26754, 1024, 1024)  # 3 * 26754**2 = 2147329548 < 2**31

    class _Huge:
        is_cuda = True
        dtype = torch.uint8
        ndim = 3
        shape = (side, side, 3)
        device = "cuda:0"

    from core.inference import rocm_bf16

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rocm_bf16, "is_rocm_torch", lambda _t: False)
        mp.setattr(K, "_launch", lambda *a, **k: pytest.fail("launched a wrapping kernel"))
        with pytest.raises(ValueError, match = "32-bit"):
            K.lanczos_resize(_Huge(), 1024, 1024)


def test_huge_reference_is_refused_before_the_upload():
    """A > 715 MP reference takes the CPU path: _gpu_preprocess raises before it copies the pixels to the GPU."""
    pytest.importorskip("triton")
    side = 26755

    class _Src:
        mode = "RGB"
        size = (side, side)

        def __array__(self, *a, **k):
            raise AssertionError("uploaded a reference the kernels cannot index")

    proc = flux2_ip.Flux2ImageProcessor(vae_scale_factor = 16)
    img = rp.DeferredResize(_Src(), 1024, 1024)
    with pytest.raises(ValueError, match = "32-bit"):
        rp._gpu_preprocess(proc, img, 1024, 1024, "cpu")
