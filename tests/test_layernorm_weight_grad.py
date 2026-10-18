# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import pytest
from real_accelerator import (
    has_real_cuda,
)
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "CUDA Triton kernels required"),
]

TOL = {torch.float32: 1e-4, torch.float16: 2e-3, torch.bfloat16: 1.6e-2}


def _layer(
    cols,
    dtype,
    bias = True,
    trainable = True,
):
    layer = torch.nn.LayerNorm(cols, bias = bias, device = "cuda", dtype = dtype)
    with torch.no_grad():
        layer.weight.uniform_(0.5, 1.5)
        if bias:
            layer.bias.uniform_(-0.5, 0.5)
    layer.requires_grad_(trainable)
    return layer


def _fast(layer, inputs, grad):
    from unsloth.kernels.layernorm import fast_layernorm

    x = inputs.clone().requires_grad_()
    y = fast_layernorm(layer, x)
    y.backward(grad.clone())
    return y.detach(), x.grad, layer.weight.grad, None if layer.bias is None else layer.bias.grad


def _exact(layer, inputs, grad):
    x = inputs.double().requires_grad_()
    w = layer.weight.detach().double().requires_grad_()
    b = None if layer.bias is None else layer.bias.detach().double().requires_grad_()
    y = torch.nn.functional.layer_norm(x, (inputs.shape[-1],), w, b, layer.eps)
    y.backward(grad.double())
    return y, x.grad, w.grad, None if b is None else b.grad


def _close(actual, exact, dtype):
    err = (actual.double() - exact).norm() / exact.norm().clamp_min(1e-12)
    assert err < TOL[dtype], err.item()


# 3 rows is below the program count, 9000 above; 1000 is not a power of two.
@pytest.mark.parametrize("bias", [True, False], ids = ["bias", "no_bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", [128, 384, 1000, 4096])
@pytest.mark.parametrize("rows", [3, 9000])
def test_layernorm_trainable_weight_and_bias_get_grads(rows, cols, dtype, bias):
    torch.manual_seed(3407)
    layer = _layer(cols, dtype, bias)
    inputs = torch.randn(rows, cols, device = "cuda", dtype = dtype)
    grad = torch.randn(rows, cols, device = "cuda", dtype = dtype)
    y, dx, dw, db = _fast(layer, inputs, grad)
    ey, edx, edw, edb = _exact(layer, inputs, grad)
    assert dw is not None and dw.dtype == dtype and dw.shape == (cols,)
    _close(y, ey, dtype)
    _close(dx, edx, dtype)
    _close(dw, edw, dtype)
    if bias:
        assert db is not None and db.dtype == dtype and db.shape == (cols,)
        _close(db, edb, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cols", [128, 1000, 4096])
def test_layernorm_trainable_matches_frozen_y_and_dx(cols, dtype):
    torch.manual_seed(0)
    inputs = torch.randn(2050, cols, device = "cuda", dtype = dtype)
    grad = torch.randn(2050, cols, device = "cuda", dtype = dtype)
    layer = _layer(cols, dtype)
    y, dx, _, _ = _fast(layer, inputs, grad)
    layer.zero_grad(set_to_none = True)
    layer.requires_grad_(False)
    fy, fdx, fdw, fdb = _fast(layer, inputs, grad)
    assert fdw is None and fdb is None
    assert torch.equal(y, fy)
    # Persistent loop contracts some products differently: <= 1 ulp.
    ulp = 2**-7 if dtype == torch.bfloat16 else 2**-10
    torch.testing.assert_close(dx.float(), fdx.float(), rtol = ulp, atol = 1e-6)


def test_layernorm_only_bias_trainable():
    torch.manual_seed(1)
    layer = _layer(512, torch.bfloat16)
    layer.weight.requires_grad_(False)
    inputs = torch.randn(700, 512, device = "cuda", dtype = torch.bfloat16)
    grad = torch.randn(700, 512, device = "cuda", dtype = torch.bfloat16)
    _, _, dw, db = _fast(layer, inputs, grad)
    assert dw is None
    _close(db, grad.double().sum(0), torch.bfloat16)


def test_layernorm_only_weight_trainable():
    torch.manual_seed(5)
    layer = _layer(512, torch.bfloat16)
    layer.bias.requires_grad_(False)
    inputs = torch.randn(700, 512, device = "cuda", dtype = torch.bfloat16)
    grad = torch.randn(700, 512, device = "cuda", dtype = torch.bfloat16)
    _, _, dw, db = _fast(layer, inputs, grad)
    assert db is None
    _close(dw, _exact(layer, inputs, grad)[2], torch.bfloat16)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_layernorm_compiled_matches_eager_bytes(dtype):
    from unsloth.kernels.layernorm import fast_layernorm

    # Inductor raises SkipFrame for bf16 graphs without native bf16 (T4, sm_75), so fullgraph fails.
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported(including_emulation = False):
        pytest.skip("Inductor does not compile bf16 without native bf16 support")
    torch.manual_seed(6)
    inputs = torch.randn(3000, 1000, device = "cuda", dtype = dtype)
    grad = torch.randn(3000, 1000, device = "cuda", dtype = dtype)
    layer = _layer(1000, dtype)
    eager = _fast(layer, inputs, grad)
    layer.zero_grad(set_to_none = True)
    compiled = torch.compile(lambda x: fast_layernorm(layer, x), fullgraph = True)
    x = inputs.clone().requires_grad_()
    y = compiled(x)
    y.backward(grad.clone())
    for a, b in zip(eager, (y.detach(), x.grad, layer.weight.grad, layer.bias.grad)):
        assert torch.equal(a, b)


def test_layernorm_dw_is_deterministic():
    torch.manual_seed(2)
    inputs = torch.randn(33000, 1024, device = "cuda", dtype = torch.bfloat16)
    grad = torch.randn(33000, 1024, device = "cuda", dtype = torch.bfloat16)
    runs = []
    for _ in range(3):
        layer = _layer(1024, torch.bfloat16)
        torch.manual_seed(2)
        runs.append(_fast(layer, inputs, grad)[2:])
    for dw, db in runs[1:]:
        assert torch.equal(dw, runs[0][0]) and torch.equal(db, runs[0][1])


def test_layernorm_empty_input_gives_zero_grads():
    layer = _layer(256, torch.float16)
    inputs = torch.randn(0, 256, device = "cuda", dtype = torch.float16)
    y, dx, dw, db = _fast(layer, inputs, torch.randn(0, 256, device = "cuda", dtype = torch.float16))
    assert y.shape == (0, 256) and dx.shape == (0, 256)
    assert dw is not None and not dw.any() and db is not None and not db.any()


def test_layernorm_no_bias_forward_matches_torch():
    torch.manual_seed(4)
    layer = _layer(768, torch.float32, bias = False, trainable = False)
    from unsloth.kernels.layernorm import fast_layernorm

    inputs = torch.randn(4, 9, 768, device = "cuda")
    expected = torch.nn.functional.layer_norm(inputs, (768,), layer.weight, None, layer.eps)
    torch.testing.assert_close(fast_layernorm(layer, inputs), expected, rtol = 1e-5, atol = 1e-5)
