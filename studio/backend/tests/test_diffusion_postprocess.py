# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``diffusion_postprocess.py``: the decoded image goes to uint8 on the device with output bit-identical to
diffusers' stock ``VaeImageProcessor.postprocess(output_type="pil")``, every case it does not cover takes the stock
path, the kill switch holds, and the generate path installs it on the pipe it renders with."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from core.inference import diffusion_postprocess as dp

torch = pytest.importorskip("torch")
ip_mod = pytest.importorskip("diffusers.image_processor")
VaeImageProcessor = ip_mod.VaeImageProcessor

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA device")


class _Pipe:
    def __init__(self, processor):
        self.image_processor = processor


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(dp.DEVICE_POSTPROCESS_ENV, raising = False)


def _decoded(
    dtype,
    batch = 1,
    size = 64,
    channels_last = False,
    device = "cpu",
    channels = 3,
):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(batch, channels, size, size, generator = g) * 0.7
    # Exact rounding ties after the denormalize: (v * 0.5 + 0.5) * 255 == k + 0.5.
    ties = (torch.arange(0, 255, dtype = torch.float32) + 0.5) / 255 * 2 - 1
    x.view(-1)[: ties.numel()] = ties
    x.view(-1)[300:310] = torch.tensor([-3.0, 3.0, -1.0, 1.0, 0.0, 1e-8, -1e-8, 0.999, -0.999, 2.0])
    x = x.to(dtype = dtype, device = device)
    return x.contiguous(memory_format = torch.channels_last) if channels_last else x


def _stock(
    processor,
    image,
    do_denormalize = None,
):
    return VaeImageProcessor.postprocess(
        processor, image, output_type = "pil", do_denormalize = do_denormalize
    )


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("channels", [3, 4])
def test_uint8_conversion_matches_numpy_bit_for_bit(dtype, channels_last, channels):
    proc = VaeImageProcessor()
    image = proc.denormalize(
        _decoded(getattr(torch, dtype), batch = 2, channels_last = channels_last, channels = channels)
    )
    ours = dp.uint8_hwc(image).numpy()
    stock = (VaeImageProcessor.pt_to_numpy(image) * 255).round().astype("uint8")
    assert ours.dtype == np.uint8 and ours.shape == stock.shape
    assert np.array_equal(ours, stock)


@needs_cuda
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("channels,mode", [(3, "RGB"), (4, "RGBA")])
def test_device_postprocess_is_bit_identical_to_stock(dtype, channels_last, channels, mode):
    """4 channels is Qwen-Image-2.1's VAE output; stock returns RGBA for it."""
    proc = VaeImageProcessor(vae_scale_factor = 16)
    image = _decoded(
        getattr(torch, dtype),
        batch = 2,
        size = 128,
        channels_last = channels_last,
        device = "cuda",
        channels = channels,
    )
    stock = _stock(proc, image)
    assert dp.install(_Pipe(proc))
    ours = proc.postprocess(image, output_type = "pil")
    assert len(ours) == len(stock) == 2
    for a, b in zip(ours, stock):
        assert a.mode == b.mode == mode and a.size == b.size
        assert np.array_equal(np.asarray(a), np.asarray(b))


@needs_cuda
@pytest.mark.parametrize("channels", [3, 4])
def test_device_path_actually_runs(monkeypatch, channels):
    proc = VaeImageProcessor()
    calls = []
    real = dp.uint8_hwc
    monkeypatch.setattr(
        dp, "uint8_hwc", lambda image: calls.append(image.device.type) or real(image)
    )
    dp.install(_Pipe(proc))
    proc.postprocess(_decoded(torch.bfloat16, device = "cuda", channels = channels), output_type = "pil")
    assert calls == ["cuda"]


@pytest.mark.parametrize(
    "case",
    ["cpu_tensor", "no_normalize", "partial_denormalize", "grayscale", "integer"],
)
def test_uncovered_inputs_return_none_for_the_stock_path(case):
    proc = VaeImageProcessor(do_normalize = case != "no_normalize")
    device = "cpu" if case == "cpu_tensor" else "meta"
    image = torch.empty(2, 3, 8, 8, device = device)
    do_denormalize = None
    if case == "partial_denormalize":
        do_denormalize = [True, False]
    elif case == "grayscale":
        image = torch.empty(2, 1, 8, 8, device = device)
    elif case == "integer":
        image = torch.empty(2, 3, 8, 8, device = device, dtype = torch.uint8)
    assert dp.to_pil_on_device(proc, image, do_denormalize) is None


@needs_cuda
def test_nan_input_returns_none_for_the_stock_path():
    image = torch.full((1, 3, 8, 8), float("nan"), device = "cuda")
    assert dp.to_pil_on_device(VaeImageProcessor(), image) is None


def test_stock_path_still_serves_other_output_types_and_cpu():
    proc = VaeImageProcessor()
    dp.install(_Pipe(proc))
    image = _decoded(torch.float32)
    assert isinstance(proc.postprocess(image, output_type = "np"), np.ndarray)
    assert isinstance(proc.postprocess(image, output_type = "pt"), torch.Tensor)
    ours = proc.postprocess(image, output_type = "pil")
    assert np.array_equal(np.asarray(ours[0]), np.asarray(_stock(proc, image)[0]))


def test_install_is_idempotent_and_uninstall_restores():
    proc = VaeImageProcessor()
    pipe = _Pipe(proc)
    assert dp.install(pipe) is True
    patched = proc.postprocess
    assert dp.install(pipe) is False
    assert proc.postprocess is patched
    dp.uninstall(pipe)
    assert "postprocess" not in proc.__dict__
    assert proc.postprocess.__func__ is VaeImageProcessor.postprocess


def test_kill_switch(monkeypatch):
    monkeypatch.setenv(dp.DEVICE_POSTPROCESS_ENV, "0")
    proc = VaeImageProcessor()
    assert dp.install(_Pipe(proc)) is False
    assert "postprocess" not in proc.__dict__


def test_processor_that_overrides_postprocess_is_left_alone():
    class Custom(VaeImageProcessor):
        def postprocess(
            self,
            image,
            output_type = "pil",
            do_denormalize = None,
        ):
            return "custom"

    proc = Custom()
    assert dp.install(_Pipe(proc)) is False
    assert dp.install(_Pipe(object())) is False
    assert dp.install(object()) is False


def test_generate_installs_it_on_the_render_pipe():
    """The call sits in generate's chunk loop, ahead of the pipe call, so every workflow pipe (from_pipe builds its own
    image processor) is covered."""
    src = Path(__file__).resolve().parents[1] / "core" / "inference" / "diffusion.py"
    tree = ast.parse(src.read_text(encoding = "utf-8"))
    gen = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "generate"
        and any(a.arg == "prompt" for a in n.args.kwonlyargs)
    )
    calls = [
        n
        for n in ast.walk(gen)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "install_device_postprocess"
    ]
    assert len(calls) == 1
    assert isinstance(calls[0].args[0], ast.Name) and calls[0].args[0].id == "pipe"
    protect = next(
        n
        for n in ast.walk(gen)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "protect_generation"
    )
    assert calls[0].lineno < protect.lineno
