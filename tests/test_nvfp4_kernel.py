# SPDX-License-Identifier: AGPL-3.0-only
# Packed NVFP4 (compressed-tensors nvfp4-pack-quantized) must dequantize exactly as compressed-tensors does and train as W4A16.
import os

import pytest

torch = pytest.importorskip("torch")
ct_helpers = pytest.importorskip("compressed_tensors.compressors.nvfp4.helpers")
ct_forward = pytest.importorskip("compressed_tensors.quantization.lifecycle.forward")

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason = "needs CUDA")


@pytest.fixture(scope = "module")
def N():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import nvfp4
    return nvfp4


def _random_nvfp4(
    rows,
    cols,
    device,
    global_scale = 7.3,
    seed = 0,
):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    packed = torch.randint(0, 256, (rows, cols // 2), dtype = torch.uint8, generator = g)
    scale = (torch.rand(rows, cols // 16, generator = g) * 3 + 0.01).to(torch.float8_e4m3fn)
    return packed.to(device), scale.to(device), torch.tensor([global_scale], device = device)


def _reference(packed, scale, global_scale, dtype):
    from compressed_tensors.quantization import QuantizationArgs

    rows, half = packed.shape
    x_q = ct_helpers.unpack_fp4_from_uint8(packed, rows, half * 2, dtype = dtype)
    # The weight args NVFP4PackedCompressor.decompress passes; compressed-tensors >= 0.19 no longer infers them.
    args = QuantizationArgs(num_bits = 4, type = "float", strategy = "tensor_group", group_size = 16)
    return ct_forward.dequantize(
        x_q = x_q, scale = scale.to(dtype), global_scale = global_scale, dtype = dtype, args = args
    )


def _devices():
    return ["cpu", "cuda"] if CUDA else ["cpu"]


@pytest.mark.parametrize("device", _devices())
def test_every_code_and_sign(N, device):
    # Byte b holds code b & 15 in the even column and b >> 4 in the odd one.
    packed = torch.arange(256, dtype = torch.uint8).reshape(16, 16).to(device)
    scale = torch.ones(16, 2, device = device).to(torch.float8_e4m3fn)
    out = N.nvfp4_dequantize(packed, scale, torch.ones(1, device = device), torch.float32)
    lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    codes = [(b & 15, b >> 4) for b in range(256)]
    expect = torch.tensor(
        [[(-1 if c & 8 else 1) * lut[c & 7] for c in pair] for pair in codes]
    ).reshape(16, 32)
    assert torch.equal(out.cpu(), expect)
    assert torch.signbit(out.cpu()[0, 16])  # code 8 is -0.0, as in compressed-tensors


@needs_cuda
def test_every_e4m3_group_scale(N):
    # Before sm_89 Triton has no fp8e4nv, so the kernel decodes the scale bytes itself: all 256 codes, incl.
    # subnormals, -0.0 and NaN, must equal torch's own fp8 conversion.
    codes = torch.arange(256, dtype = torch.uint8, device = "cuda").reshape(16, 16)
    scale = codes.view(torch.float8_e4m3fn)
    packed = torch.full((16, 128), 0x22, dtype = torch.uint8, device = "cuda")  # every value 1.0
    out = N.nvfp4_dequantize(packed, scale, torch.ones(1, device = "cuda"), torch.float32)
    expect = scale.float().repeat_interleave(16, dim = 1)
    assert torch.equal(out.isnan(), expect.isnan())
    assert torch.equal(out.nan_to_num(), expect.nan_to_num())
    finite = ~expect.isnan()
    assert torch.equal(
        out.signbit()[finite], expect.signbit()[finite]
    )  # -0.0 (0x80) keeps its sign


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(64, 256), (48, 400), (257, 96)])
def test_matches_compressed_tensors_bit_exact(N, device, dtype, shape):
    packed, scale, gs = _random_nvfp4(*shape, device, global_scale = 0.37)
    out = N.nvfp4_dequantize(packed, scale, gs, dtype)
    assert out.dtype == dtype and out.shape == shape
    assert torch.equal(out, _reference(packed, scale, gs, dtype))


@needs_cuda
def test_triton_equals_torch_reference(N):
    packed, scale, gs = _random_nvfp4(320, 1024, "cuda")
    tri = N.nvfp4_dequantize(packed, scale, gs)
    ref = N._nvfp4_dequantize_torch(packed, scale, gs, torch.bfloat16)
    assert torch.equal(tri, ref)


@pytest.mark.parametrize("device", _devices())
def test_transposed_view_is_the_transposed_weight(N, device):
    packed, scale, gs = _random_nvfp4(96, 256, device)
    dense = N.nvfp4_dequantize(packed, scale, gs)
    assert torch.equal(N.nvfp4_dequantize(packed.t(), scale, gs), dense.t())


def test_too_few_scales_is_an_error(N):
    packed, scale, gs = _random_nvfp4(16, 64, "cpu")
    with pytest.raises(ValueError):
        N.nvfp4_dequantize(packed, scale[:, :2], gs)


def _dense_reference_grads(W, X, dY, bias):
    X = X.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True) if bias is not None else None
    out = X @ W.t()
    if b is not None:
        out = out + b
    out.backward(dY)
    return out.detach(), X.grad, None if b is None else b.grad


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("with_bias", [False, True])
def test_autograd_matches_the_dense_weight(N, device, with_bias):
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    packed, scale, gs = _random_nvfp4(192, 128, device)
    W = N.nvfp4_dequantize(packed, scale, gs, dtype)
    X = torch.randn(2, 5, 128, device = device, dtype = dtype, requires_grad = True)
    dY = torch.randn(2, 5, 192, device = device, dtype = dtype)
    bias = torch.randn(192, device = device, dtype = dtype, requires_grad = True) if with_bias else None
    out = N.nvfp4_linear(X, packed, scale, gs, bias)
    out.backward(dY)
    ref_out, ref_dX, ref_db = _dense_reference_grads(W, X, dY, bias)
    assert out.dtype == dtype
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(X.grad, ref_dX)
    assert X.grad.abs().sum() > 0
    if with_bias:
        torch.testing.assert_close(bias.grad, ref_db)


def test_float32_bias_keeps_the_activation_dtype(N):
    packed, scale, gs = _random_nvfp4(32, 64, "cpu")
    X = torch.randn(3, 64, dtype = torch.bfloat16)
    out = N.nvfp4_linear(X, packed, scale, gs, torch.randn(32, dtype = torch.float32))
    assert out.dtype == torch.bfloat16


def test_backward_saves_only_the_packed_weight(N):
    device = "cuda" if CUDA else "cpu"
    packed, scale, gs = _random_nvfp4(256, 512, device)
    X = torch.randn(4, 512, device = device, requires_grad = True)
    saved = []

    def pack_hook(t):
        saved.append((tuple(t.shape), t.dtype))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda t: t):
        N.nvfp4_linear(X, packed, scale, gs)
    dense = [s for s in saved if s[0] == (256, 512) or s[0] == (512, 256)]
    assert dense == [], saved


@needs_cuda
def test_fullgraph_compile_has_no_breaks_and_matches_eager(N):
    import torch._dynamo

    packed, scale, gs = _random_nvfp4(384, 256, "cuda")
    step = lambda X: N.nvfp4_linear(X, packed, scale, gs).float().square().mean()
    X = torch.randn(64, 256, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
    step(X).backward()
    eager = X.grad.clone()
    X.grad = None
    torch._dynamo.reset()
    assert torch._dynamo.explain(step)(X).graph_break_count == 0
    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph = True)
    for rows in (16, 48, 64):  # several shapes: dynamic-shape recompiles must stay correct
        Xr = torch.randn(rows, 256, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        compiled(Xr).backward()
    compiled(X).backward()
    assert X.grad.abs().sum() > 0
    torch.testing.assert_close(X.grad, eager)
