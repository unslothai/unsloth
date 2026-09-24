"""Fused LoRA gradients must retain the stored FP8 weight's scale axes.

Load the numeric functions without importing Unsloth's GPU/model integration.
The small rowwise shapes take FBGEMM's real dequant fallback, so these tests
also run on CPU and GPUs without native FP8 matmul support.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope = "module")
def kernels():
    root = Path(__file__).resolve().parents[1] / "unsloth" / "kernels"
    namespace = {"torch": torch, "torch_matmul": torch.matmul, "Float8Tensor": type(None)}

    def load(
        file,
        name,
        owner = None,
        alias = None,
    ):
        path = root / file
        tree = ast.parse(path.read_text(encoding = "utf-8"))
        if owner:
            tree = next(
                node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner
            )
        node = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        ][-1]
        node.decorator_list = []
        node.name = alias or name
        exec(compile(ast.Module(body = [node], type_ignores = []), str(path), "exec"), namespace)
        return namespace[node.name]

    load("fp8.py", "weight_dequant")
    load("utils.py", "fast_dequantize")
    row_forward = load("fp8.py", "forward", "FbgemmFp8Linear_matmul", "row_forward")
    namespace["fp8_linear"] = lambda X, W, scale: row_forward(SimpleNamespace(), X, W, scale)
    load("utils.py", "matmul_lora")
    return {
        name: load("fast_lora.py", "backward", name, name)
        for name in ("LoRA_W", "LoRA_QKV", "LoRA_MLP")
    }


@pytest.fixture(params = ["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def projection(out_features, in_features, kind, device):
    values = (
        torch.arange(out_features * in_features, device = device).reshape(out_features, in_features)
        % 7
        - 3
    ).to(torch.bfloat16)
    if kind == "bf16":
        weight, scale, reference = values, None, values
    else:
        weight = values.to(torch.float8_e4m3fn)
        scale = (torch.arange(out_features, device = device, dtype = torch.float32) * 3 + 2).reshape(
            -1, 1
        )
        if kind == "uniform":
            scale.fill_(2)
        elif kind == "tensor":
            scale = scale.new_tensor(2)
        reference = weight.to(torch.bfloat16) * scale.to(torch.bfloat16)
    A = (
        (torch.arange(in_features, device = device) % 3 - 1).to(torch.bfloat16).reshape(1, -1) / 4
    ).requires_grad_()
    B = (
        (torch.arange(out_features, device = device) % 3 - 1).to(torch.bfloat16).reshape(-1, 1) / 2
    ).requires_grad_()
    return (weight, scale, A, B, 0.5), reference


def linear(X, parameters, reference):
    _, _, A, B, scale = parameters
    return X @ reference.T + ((X @ A.T) @ B.T) * scale


def inputs(features, device):
    return (
        torch.arange(2 * features, device = device, dtype = torch.bfloat16).reshape(1, 2, features) / 4
    ).requires_grad_()


def check(actual, expected):
    # Allow BF16 accumulation-order differences between the fused and autograd
    # adapter terms; the scale-axis regression is far larger than one ULP.
    torch.testing.assert_close(actual, expected, rtol = 1e-2, atol = 0.125)


def test_square_row_scales_exact_reproducer(kernels, device):
    W = torch.tensor([[1, 2], [3, 4]], device = device, dtype = torch.float8_e4m3fn)
    scale = torch.tensor([[2.0], [5.0]], device = device)
    X = torch.zeros((1, 1, 2), device = device, dtype = torch.bfloat16)
    A = X.new_zeros((1, 2))
    B = X.new_zeros((2, 1))
    ctx = SimpleNamespace(custom_saved_tensors = (W, scale, 1), saved_tensors = (A, B, X))
    result = kernels["LoRA_W"](ctx, X.new_ones((1, 1, 2)))[0]
    # The old transpose-before-dequantize path returned [8, 30].
    torch.testing.assert_close(result, X.new_tensor([[[17, 24]]]), rtol = 0, atol = 0)


@pytest.mark.parametrize("shape", [(2, 2), (3, 2), (2, 3)])
@pytest.mark.parametrize("kind", ["row", "uniform", "tensor", "bf16"])
def test_lora_w_gradients(kernels, device, shape, kind):
    n, k = shape
    parameters, reference = projection(n, k, kind, device)
    W, quant, A, B, scale = parameters
    X = inputs(k, device)
    out = linear(X, parameters, reference)
    grad = torch.ones_like(out)
    out.backward(grad)
    ctx = SimpleNamespace(
        custom_saved_tensors = (W, quant, scale), saved_tensors = (A, B, X.detach().clone())
    )
    with torch.no_grad():
        actual = kernels["LoRA_W"](ctx, grad)
    for value, expected in zip((actual[0], actual[3], actual[4]), (X.grad, A.grad, B.grad)):
        check(value, expected)


@pytest.mark.parametrize("sizes", [(4, 4, 4), (4, 2, 2)])
@pytest.mark.parametrize("kind", ["row", "tensor", "bf16"])
@pytest.mark.parametrize("inplace", [False, True])
def test_lora_qkv_gradients(kernels, device, sizes, kind, inplace):
    X = inputs(4, device)
    projections = [projection(n, 4, kind, device) for n in sizes]
    outputs = [linear(X, p, W) for p, W in projections]
    grads = [torch.full_like(out, i + 1) for i, out in enumerate(outputs)]
    torch.autograd.backward(outputs, grads)
    ctx = SimpleNamespace(
        custom_saved_tensors = tuple(
            value for (W, quant, A, B, scale), _ in projections for value in (W, quant, scale)
        ),
        saved_tensors = (
            X.detach().clone(),
            *(value for (_, _, A, B, _), _ in projections for value in (A, B)),
        ),
        inplace = inplace,
    )
    with torch.no_grad():
        actual = kernels["LoRA_QKV"](ctx, *grads)
    check(actual[0], X.grad)
    for offset, ((_, _, A, B, _), _) in zip((3, 8, 13), projections):
        check(actual[offset], A.grad)
        check(actual[offset + 1], B.grad)


@pytest.mark.parametrize("hidden", [2, 3])
@pytest.mark.parametrize("kind", ["row", "bf16"])
@pytest.mark.parametrize("inplace", [False, True])
def test_lora_mlp_gradients(kernels, device, hidden, kind, inplace):
    X = inputs(2, device)
    gate, gateW = projection(hidden, 2, kind, device)
    up, upW = projection(hidden, 2, kind, device)
    down, downW = projection(2, hidden, kind, device)
    e, g = linear(X, gate, gateW), linear(X, up, upW)
    # A product gate isolates the projection gradients from Triton activation
    # kernels. LoRA_MLP accepts the activation derivative as a callable.
    out = linear(e * g, down, downW)
    grad = torch.ones_like(out)
    out.backward(grad)
    backward_gate = lambda D, e, g: (e * g, D * e, D * g)
    parameters = (gate, up, down)
    ctx = SimpleNamespace(
        custom_saved_tensors = (
            *(value for W, quant, A, B, scale in parameters for value in (W, quant, scale)),
            backward_gate,
        ),
        saved_tensors = (
            *(value for W, quant, A, B, scale in parameters for value in (A, B)),
            X.detach().clone(),
            e.detach(),
            g.detach(),
        ),
        inplace = inplace,
    )
    with torch.no_grad():
        actual = kernels["LoRA_MLP"](ctx, grad)
    check(actual[0], X.grad)
    for offset, (_, _, A, B, _) in zip((3, 8, 13), parameters):
        check(actual[offset], A.grad)
        check(actual[offset + 1], B.grad)
