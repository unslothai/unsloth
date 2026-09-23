# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""compressed-tensors FP8 Linears run on Unsloth's FP8 kernels, stay FP8, and pass the input gradient.

Also covers `weight_dequant` on a transposed per-row FP8 view: fast_lora's backward passes `W.t()`, and
for a square weight the shape alone cannot say which axis the scale belongs to.
"""

import types

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs CUDA")
ct_quant = pytest.importorskip("compressed_tensors.quantization")


@pytest.fixture(autouse = True)
def _zoo_with_float_only_cast(monkeypatch):
    # Routing waits for an unsloth_zoo whose compiled LoRA forward never casts inputs to FP8.
    from unsloth.models import loader_utils
    monkeypatch.setattr(loader_utils, "_zoo_peft_forward_keeps_fp8_inputs", lambda: True)


def _quantize(
    W,
    strategy,
    block = None,
):
    o, i = W.shape
    if strategy == "tensor":
        s = W.abs().amax().view(1) / 448
        full = s.expand(o, i)
    elif strategy == "channel":
        s = W.abs().amax(1, keepdim = True) / 448
        full = s.expand(o, i)
    else:
        bo, bi = block
        s = torch.empty(-(-o // bo), -(-i // bi), device = W.device)
        for a in range(s.shape[0]):
            for b in range(s.shape[1]):
                s[a, b] = W[a * bo : (a + 1) * bo, b * bi : (b + 1) * bi].abs().amax() / 448
        full = s.repeat_interleave(bo, 0)[:o].repeat_interleave(bi, 1)[:, :i]
    Wq = (W / full).to(torch.float8_e4m3fn)
    return Wq, s.to(torch.bfloat16), Wq.float() * full.to(torch.bfloat16).float()


def _ct_model(
    o,
    i,
    strategy,
    bias = False,
    block = None,
    weight_type = "float",
):
    torch.manual_seed(0)
    lin = torch.nn.Linear(i, o, bias = bias, device = "cuda", dtype = torch.bfloat16)
    Wq, s, ref = _quantize(torch.randn(o, i, device = "cuda") * 0.02, strategy, block)
    lin.weight = torch.nn.Parameter(Wq, requires_grad = False)
    lin.weight_scale = torch.nn.Parameter(s, requires_grad = False)
    kwargs = dict(num_bits = 8, type = weight_type, strategy = strategy, symmetric = True, dynamic = False)
    if block is not None:
        kwargs["block_structure"] = list(block)
    lin.quantization_scheme = ct_quant.QuantizationScheme(
        targets = ["Linear"],
        weights = ct_quant.QuantizationArgs(**kwargs),
        input_activations = ct_quant.QuantizationArgs(
            num_bits = 8, type = "float", strategy = "token", dynamic = True, symmetric = True
        ),
    )

    def ct_forward(self, x):
        raise AssertionError("compressed-tensors forward must not run on a routed module")

    lin.forward = types.MethodType(ct_forward, lin)
    model = torch.nn.Module()
    model.lin = lin
    model.config = types.SimpleNamespace(quantization_config = {"quant_method": "compressed-tensors"})
    return model, ref


@pytest.mark.parametrize("shape", [(256, 256), (384, 256), (256, 384)])
def test_weight_dequant_transposed_row_scale(shape):
    from unsloth.kernels.fp8 import weight_dequant

    Wq, s, ref = _quantize(torch.randn(*shape, device = "cuda") * 0.02, "channel")
    torch.testing.assert_close(weight_dequant(Wq, s, torch.float32), ref)
    # The square case picked the scale axis by shape and scaled the wrong one.
    torch.testing.assert_close(weight_dequant(Wq.t(), s, torch.float32), ref.t())


@pytest.mark.parametrize(
    "strategy,shape,block",
    [
        ("tensor", (384, 256), None),
        ("channel", (256, 256), None),
        ("channel", (384, 256), None),
        ("block", (256, 256), (128, 128)),
        ("block", (320, 256), (128, 128)),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_fp8_modules_route_to_unsloth_kernels(strategy, shape, block, bias):
    from unsloth.models.loader_utils import _route_compressed_tensors_fp8_to_unsloth

    model, ref = _ct_model(*shape, strategy, bias = bias, block = block)
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 1
    lin = model.lin
    assert lin.weight.dtype == torch.float8_e4m3fn
    X = torch.randn(2, 5, shape[1], device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    y = lin(X)
    y_ref = X.float() @ ref.t() + (lin.bias.float() if bias else 0)
    (dX,) = torch.autograd.grad(y.float().sum(), X)
    dX_ref = torch.ones(2, 5, shape[0], device = "cuda") @ ref
    # Block / per-tensor kernels quantize activations to FP8 (a few percent); the gradient is exact.
    assert float((y.float() - y_ref).norm() / y_ref.norm()) < 0.05
    assert float((dX.float() - dX_ref).norm() / dX_ref.norm()) < 0.01


def test_non_fp8_or_unsupported_modules_are_left_alone(monkeypatch):
    from unsloth.models.loader_utils import _route_compressed_tensors_fp8_to_unsloth

    model, _ = _ct_model(256, 256, "channel", weight_type = "int")
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 0
    model, _ = _ct_model(256, 256, "channel")
    model.lin.quantization_scheme.output_activations = (
        model.lin.quantization_scheme.input_activations
    )
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 0
    model, _ = _ct_model(256, 256, "channel")
    model.lin.weight = torch.nn.Parameter(model.lin.weight.to(torch.bfloat16), requires_grad = False)
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 0
    model, _ = _ct_model(256, 256, "channel")
    model.lin.weight_scale = torch.nn.Parameter(
        torch.ones(3, 1, device = "cuda"), requires_grad = False
    )
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 0
    model, _ = _ct_model(256, 256, "channel")
    monkeypatch.setenv("UNSLOTH_COMPRESSED_TENSORS_FP8_KERNELS", "0")
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 0


def test_decompress_hook_is_removed():
    from unsloth.models.loader_utils import _route_compressed_tensors_fp8_to_unsloth

    model, _ = _ct_model(256, 256, "channel")
    model.ct_decompress_hook = model.register_forward_pre_hook(lambda module, args: None)
    assert len(model._forward_pre_hooks) == 1
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 1
    # Left in place, the first forward would decompress every routed weight back to 16 bit.
    assert not hasattr(model, "ct_decompress_hook")
    assert len(model._forward_pre_hooks) == 0


def test_older_zoo_keeps_the_compressed_tensors_path(monkeypatch):
    from unsloth.models import loader_utils

    monkeypatch.setattr(loader_utils, "_zoo_peft_forward_keeps_fp8_inputs", lambda: False)
    model, _ = _ct_model(256, 256, "channel")
    assert loader_utils._route_compressed_tensors_fp8_to_unsloth(model) == 0
    assert not hasattr(model.lin, "_unsloth_compressed_tensors_fp8")


@pytest.mark.parametrize("strategy, block", [("channel", None), ("block", (128, 128))])
def test_single_token_fast_path_adds_the_bias_once(strategy, block):
    from unsloth.kernels.utils import fast_linear_forward
    from unsloth.models.loader_utils import _route_compressed_tensors_fp8_to_unsloth

    model, ref = _ct_model(256, 256, strategy, bias = True, block = block)
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 1
    lin = model.lin
    with torch.no_grad():
        lin.bias.fill_(1.0)
        X = torch.randn(1, 1, 256, device = "cuda", dtype = torch.bfloat16)
        y = fast_linear_forward(lin, X)
        y_ref = X.float() @ ref.t() + 1.0
    assert float((y.float() - y_ref).norm() / y_ref.norm()) < 0.05


def test_a_model_mixing_fp8_and_other_schemes_is_not_partly_routed():
    from unsloth.models.loader_utils import _route_compressed_tensors_fp8_to_unsloth

    model, _ = _ct_model(256, 256, "channel")
    other, _ = _ct_model(256, 256, "channel", weight_type = "int")
    model.other = other.lin
    # Routing half the model would leave the rest to a whole-model decompression that also rewrites the
    # routed FP8 weights, so nothing is routed.
    assert _route_compressed_tensors_fp8_to_unsloth(model) == 0
    assert not hasattr(model.lin, "_unsloth_compressed_tensors_fp8")


@pytest.mark.parametrize("shape", [(2048, 2048), (256, 2048), (11008, 2048), (2048, 11008), (100, 300), (33, 7)])
@pytest.mark.parametrize("rows", [(1,), (4,), (2, 1)])
def test_decode_gemv_matches_the_dequantized_matmul(shape, rows):
    from unsloth.kernels.fp8 import can_use_fp8_rowwise_gemv, fp8_rowwise_gemv

    if _fp8_kernel_unsupported_here():
        pytest.skip("triton cannot read float8_e4m3fn on this GPU")
    model, ref = _ct_model(*shape, "channel")
    W, s = model.lin.weight, model.lin.weight_scale
    X = torch.randn(*rows, shape[1], device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        assert can_use_fp8_rowwise_gemv(X, W, s)
        y = fp8_rowwise_gemv(X, W, s)
        y_again = fp8_rowwise_gemv(X, W, s)
    y_ref = X.float() @ ref.t()
    assert y.shape == (*rows, shape[0]) and y.dtype == X.dtype
    assert float((y.float() - y_ref).norm() / y_ref.norm()) < 0.01
    assert torch.equal(y, y_again)  # fixed-order split-K reduction


def test_decode_gemv_refuses_what_it_cannot_compute():
    from unsloth.kernels.fp8 import can_use_fp8_rowwise_gemv

    model, _ = _ct_model(2048, 16384, "channel")
    W, s = model.lin.weight, model.lin.weight_scale
    X = torch.randn(1, 16384, device = "cuda", dtype = torch.bfloat16)
    with torch.no_grad():
        # A 128x128 block grid for this weight has 16 * 128 == 2048 == N elements.
        assert not can_use_fp8_rowwise_gemv(X, W, torch.ones(16, 128, device = "cuda"))
        assert not can_use_fp8_rowwise_gemv(torch.randn(5, 16384, device = "cuda", dtype = torch.bfloat16), W, s)
        assert not can_use_fp8_rowwise_gemv(X.float(), W, s)
        assert not can_use_fp8_rowwise_gemv(X[:, :2048], W.t(), s)
    assert not can_use_fp8_rowwise_gemv(X.requires_grad_(True), W, s)


def _fp8_kernel_unsupported_here():
    from unsloth.kernels.fp8 import _fp8_kernel_unsupported

    return _fp8_kernel_unsupported(torch.empty(1, device = "cuda", dtype = torch.float8_e4m3fn))
