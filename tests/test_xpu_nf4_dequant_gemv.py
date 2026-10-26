# SPDX-License-Identifier: AGPL-3.0-only
"""XPU fast_dequantize / fast_gemv on CPU: lifted from kernels/utils.py (defined only when DEVICE_TYPE == "xpu"),
bitsandbytes ctypes kernels replaced by torch twins."""

import ast
import ctypes
import inspect
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from unsloth.kernels import utils as ku

NF4 = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
)
CODE2 = torch.linspace(-1, 1, 256)
BLOCKSIZE, BLOCKSIZE2 = 64, 256


def _dequant_4bit(packed, absmax, code, blocksize, n):
    p = packed.reshape(-1)
    idx = torch.stack([p >> 4, p & 15], 1).reshape(-1)[:n].long()
    return code[idx] * absmax.float().repeat_interleave(blocksize)[:n]


def _quant_state(shape, dtype, nested):
    w = torch.randn(shape) * 0.02
    flat = w.reshape(-1, BLOCKSIZE)
    absmax = flat.abs().amax(1).clamp_min(1e-8)
    idx = (
        (flat / absmax[:, None]).unsqueeze(-1).sub(NF4).abs().argmin(-1).reshape(-1).to(torch.uint8)
    )
    packed = ((idx[0::2] << 4) | idx[1::2]).reshape(-1, 1)
    qs = SimpleNamespace(
        shape = torch.Size(shape),
        dtype = dtype,
        blocksize = BLOCKSIZE,
        code = NF4,
        quant_type = "nf4",
        offset = None,
        state2 = None,
        absmax = absmax,
    )
    if nested:
        offset = absmax.mean()
        n = absmax.numel()
        centred = torch.nn.functional.pad(absmax - offset, (0, -n % BLOCKSIZE2)).reshape(
            -1, BLOCKSIZE2
        )
        absmax2 = centred.abs().amax(1).clamp_min(1e-12)
        q8 = (
            (centred / absmax2[:, None])
            .unsqueeze(-1)
            .sub(CODE2)
            .abs()
            .argmin(-1)
            .reshape(-1)[:n]
            .to(torch.uint8)
        )
        qs.absmax, qs.offset = q8, offset
        qs.state2 = SimpleNamespace(absmax = absmax2, code = CODE2, blocksize = BLOCKSIZE2)
        absmax = CODE2[q8.long()] * absmax2.repeat_interleave(BLOCKSIZE2)[:n] + offset
    ref = _dequant_4bit(packed, absmax, NF4, BLOCKSIZE, w.numel()).reshape(shape).to(dtype)
    return packed, qs, ref


@pytest.fixture
def xpu():
    calls = []
    live = ["stream-0"]

    def absmax_kernel(code2, absmax, absmax2, out, blocksize2, n, stream):
        calls.append(("absmax", stream))
        out.copy_(code2[absmax.long()] * absmax2.repeat_interleave(blocksize2.value)[: n.value])

    def nf4_kernel(_, W, absmax, out, blocksize, n, stream):
        calls.append(("nf4", stream))
        out.view(-1).copy_(_dequant_4bit(W, absmax, NF4, blocksize.value, n.value))

    def gemv_kernel(m, n, k, X, W, absmax, code, out, lda, ldb, ldc, blocksize, stream):
        assert m.value == 1, "the XPU gemv takes m = 1, n = rows"
        calls.append(("gemv", stream, absmax.dtype))
        Wd = _dequant_4bit(W, absmax, code, blocksize.value, n.value * k.value).reshape(
            n.value, k.value
        )
        out.view(-1).copy_((X.reshape(1, -1).float() @ Wd.t()).reshape(-1))

    def bnb_dequantize_4bit(
        W,
        quant_state,
        out = None,
    ):
        calls.append(("bitsandbytes", None))
        return (
            _dequant_4bit(
                W,
                quant_state.absmax,
                quant_state.code,
                quant_state.blocksize,
                quant_state.shape.numel(),
            )
            .reshape(quant_state.shape)
            .to(quant_state.dtype)
        )

    ns = dict(vars(ku))
    ns.update(
        get_ptr = lambda t: t,
        torch_gpu_device = lambda device: __import__("contextlib").nullcontext(),
        _get_tensor_stream = lambda t: live[0],
        XPU_STREAMS = {None: "import-time"},
        WEIGHT_BUFFERS = {None: None},
        ABSMAX_BUFFERS = {None: None},
        cdequantize_blockwise_fp32 = absmax_kernel,
        cdequantize_blockwise_fp16_nf4 = nf4_kernel,
        cdequantize_blockwise_bf16_nf4 = nf4_kernel,
        cdequantize_blockwise_fp32_nf4 = nf4_kernel,
        cgemm_4bit_inference_naive_fp16 = gemv_kernel,
        cgemm_4bit_inference_naive_bf16 = gemv_kernel,
        bnb_functional = SimpleNamespace(dequantize_4bit = bnb_dequantize_4bit),
        DEVICE_TYPE = "xpu",
        ctypes_c_int = ctypes.c_int,
        ctypes_c_int32 = ctypes.c_int32,
    )
    tree = ast.parse(inspect.getsource(ku))
    found = set()
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "DEVICE_TYPE == 'xpu' and HAS_XPU_STREAM"
        ):
            defs = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            exec(compile(ast.Module(body = defs, type_ignores = []), ku.__file__, "exec"), ns)
            found.update(n.name for n in defs)
    assert {"fast_dequantize", "fast_gemv"} <= found
    return SimpleNamespace(
        dequantize = ns["fast_dequantize"], gemv = ns["fast_gemv"], calls = calls, live = live
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "plain"])
@pytest.mark.parametrize("global_buffer", [False, True])
def test_dequantize_matches_reference(xpu, dtype, nested, global_buffer):
    torch.manual_seed(0)
    W, qs, ref = _quant_state((128, 256), dtype, nested)
    out = xpu.dequantize(W, qs, use_global_buffer = global_buffer)
    torch.testing.assert_close(out, ref, rtol = 0, atol = 0)
    kinds = [c[0] for c in xpu.calls]
    assert kinds == (["absmax", "nf4"] if nested else ["bitsandbytes"])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "plain"])
def test_gemv_matches_reference(xpu, dtype, nested):
    torch.manual_seed(0)
    W, qs, ref = _quant_state((192, 256), dtype, nested)
    X = torch.randn(1, 1, 256, dtype = dtype)
    out = xpu.gemv(X, W, qs)
    torch.testing.assert_close(out.float(), (X.float() @ ref.float().t()), rtol = 2e-2, atol = 2e-3)
    assert [c[0] for c in xpu.calls] == (["absmax", "gemv"] if nested else ["gemv"])
    assert xpu.calls[-1][2] == torch.float32


@pytest.mark.parametrize("nested", [True, False], ids = ["nested", "plain"])
def test_gemv_rows_not_starting_a_block_fall_back_to_dequant(xpu, nested):
    torch.manual_seed(0)
    W, qs, ref = _quant_state((8, 96), torch.bfloat16, nested)  # 96 % 64 != 0: rows straddle blocks
    X = torch.randn(1, 1, 96, dtype = torch.bfloat16)
    out = xpu.gemv(X, W, qs)
    torch.testing.assert_close(out.float(), (X @ ref.t()).float(), rtol = 2e-2, atol = 2e-3)
    assert "gemv" not in [c[0] for c in xpu.calls]


def test_kernels_read_the_live_stream(xpu):
    torch.manual_seed(0)
    W, qs, _ = _quant_state((128, 256), torch.bfloat16, True)
    X = torch.randn(1, 1, 256, dtype = torch.bfloat16)
    xpu.live[0] = "side-stream"
    xpu.dequantize(W, qs)
    xpu.gemv(X, W, qs)
    assert {c[1] for c in xpu.calls} == {"side-stream"}
