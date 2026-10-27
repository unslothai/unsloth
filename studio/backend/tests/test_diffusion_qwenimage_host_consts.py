# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image denoiser forward without host-to-device copies (diffusion_qwenimage_host_consts). The CUDA test
captures a tiny Qwen-Image-Edit-2511-shaped transformer (``zero_cond_t``) into a CUDA graph, compiled and eager."""

from __future__ import annotations

import pytest
import torch

import core.inference.diffusion_qwenimage_host_consts as hc  # noqa: E402

tq = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage")


def _tiny(device, zero_cond_t = True):
    torch.manual_seed(0)
    return (
        tq.QwenImageTransformer2DModel(
            patch_size = 2,
            in_channels = 16,
            out_channels = 4,
            num_layers = 2,
            attention_head_dim = 16,
            num_attention_heads = 2,
            joint_attention_dim = 32,
            axes_dims_rope = (4, 6, 6),
            zero_cond_t = zero_cond_t,
        )
        .to(device, torch.bfloat16)
        .eval()
    )


def _inputs(device):
    g = torch.Generator("cpu").manual_seed(1)
    return dict(
        hidden_states = torch.randn(1, 128, 16, generator = g).to(device, torch.bfloat16),
        encoder_hidden_states = torch.randn(1, 7, 32, generator = g).to(device, torch.bfloat16),
        encoder_hidden_states_mask = torch.ones(1, 7, dtype = torch.long, device = device),
        timestep = torch.full((1,), 0.5, device = device),
        img_shapes = [[(1, 8, 8), (1, 8, 8)]],
        return_dict = False,
    )


@pytest.fixture(autouse = True)
def _clean(monkeypatch):
    monkeypatch.delenv(hc.QWEN_HOST_CONSTS_ENV, raising = False)
    hc.uninstall()
    yield
    hc.uninstall()


def test_install_swaps_the_module_torch_and_uninstall_restores_it():
    m = _tiny("cpu")
    assert tq.torch is torch
    assert hc.install(m)
    assert tq.torch is not torch and tq.torch.cat is torch.cat and tq.torch.float32 is torch.float32
    assert hc.install(m)  # idempotent
    hc.uninstall(m)
    assert tq.torch is torch
    assert "qwenimage_host_consts" not in m.__dict__.get("_unsloth_first_call_hooks", {})


def test_kill_switch_and_other_classes(monkeypatch):
    assert not hc.install(torch.nn.Linear(2, 2))
    monkeypatch.setenv(hc.QWEN_HOST_CONSTS_ENV, "0")
    assert not hc.install(_tiny("cpu"))
    assert tq.torch is torch


def test_cached_constant_is_reused_and_equal(monkeypatch):
    monkeypatch.setattr(hc, "_MEMO_DEVICE_TYPES", ("cpu",))
    hc.install(_tiny("cpu"))
    rows = [[0] * 64 + [1] * 64]
    a = tq.torch.tensor(rows, device = "cpu", dtype = torch.int)
    b = tq.torch.tensor([[0] * 64 + [1] * 64], device = "cpu", dtype = torch.int)
    assert a is b and torch.equal(a, torch.tensor(rows, dtype = torch.int))
    assert (
        tq.torch.tensor(rows, device = "cpu", dtype = torch.long) is not a
    )  # dtype is part of the key
    assert tq.torch.as_tensor(7, device = "cpu") is tq.torch.as_tensor(7, device = "cpu")
    # tensors, extra kwargs and no device go to torch unchanged (new objects)
    x = torch.ones(3)
    assert tq.torch.as_tensor(x) is x
    assert tq.torch.tensor(rows, dtype = torch.int) is not tq.torch.tensor(rows, dtype = torch.int)


def test_forward_is_bit_identical_on_cpu(monkeypatch):
    m = _tiny("cpu")
    with torch.no_grad():
        ref = m(**_inputs("cpu"))[0]
    monkeypatch.setattr(hc, "_MEMO_DEVICE_TYPES", ("cpu",))
    assert hc.install(m)
    with torch.no_grad():
        out = m(**_inputs("cpu"))[0]
        again = m(**_inputs("cpu"))[0]
    assert torch.equal(out, ref) and torch.equal(again, ref)


def _capture(m, kw):
    with torch.no_grad():
        ref = m(**kw)[0].clone()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                m(**kw)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(**kw)[0]
        g.replay()
        torch.cuda.synchronize()
    return ref, out


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
@pytest.mark.parametrize("compiled", [False, True])
def test_edit_denoiser_step_captures_as_a_cuda_graph(compiled):
    """Fails without the patch: ``Cannot copy between CPU and CUDA tensors during CUDA graph capture``."""
    m = _tiny("cuda")
    if compiled:
        torch._dynamo.reset()
        m.forward = torch.compile(m.forward)
    assert hc.install(m)
    ref, out = _capture(m, _inputs("cuda"))
    assert torch.equal(out, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
def test_stock_edit_denoiser_step_does_not_capture():
    """The failure this patch removes, on the same tiny model (guards the test above against a vacuous pass)."""
    m = _tiny("cuda")
    with pytest.raises(RuntimeError, match = "CUDA graph capture"):
        _capture(m, _inputs("cuda"))
