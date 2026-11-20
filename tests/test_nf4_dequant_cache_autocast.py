# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Compiled NF4 matmul_lora (the traced unsloth::nf4_cached_matmul route) must keep autocast's semantics of the
torch.matmul it replaces: under autocast both operands are cast to the autocast dtype, whatever X's dtype. The
eager path (fast_dequantize + torch.matmul) is the reference. Needs CUDA, bitsandbytes and the Triton NF4 path."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level = True)
bnb_functional = pytest.importorskip("bitsandbytes.functional")

from unsloth.kernels import dequant_cache as dc  # noqa: E402
from unsloth.kernels import utils as kernel_utils  # noqa: E402


def _compiled_and_eager(X, packed, quant_state, autocast_dtype, **compile_options):
    def forward(x):
        return kernel_utils.matmul_lora(x, packed, quant_state, None, None, None, fused = False)

    torch._dynamo.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype = autocast_dtype):
        eager = forward(X)
        compiled = torch.compile(forward, **compile_options)(X)
    return eager, compiled


@pytest.fixture(autouse = True)
def _route(monkeypatch):
    print("dequant_cache imported from", dc.__file__)
    if not (dc._HAS_OPS and getattr(kernel_utils, "_USE_NF4_KERNELS", False)):
        pytest.skip("traced NF4 op route unavailable")
    monkeypatch.setattr(dc, "_MODE", "auto")
    seen = []
    original = dc.nf4_matmul

    def counted(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(dc, "nf4_matmul", counted)
    yield seen
    dc.clear_dequant_cache()


@pytest.mark.parametrize(
    "compile_options", [{}, {"fullgraph": True}, {"fullgraph": True, "dynamic": True}]
)
def test_fp16_input_under_bf16_autocast_does_not_overflow(_route, compile_options):
    # 128 * 64 * 64 = 524288: finite in bf16, inf in fp16.
    dense = torch.full((128, 128), 64.0, device = "cuda", dtype = torch.float16)
    packed, quant_state = bnb_functional.quantize_4bit(
        dense, quant_type = "nf4", compress_statistics = False
    )
    X = torch.full((4, 128), 64.0, device = "cuda", dtype = torch.float16)
    eager, compiled = _compiled_and_eager(X, packed, quant_state, torch.bfloat16, **compile_options)
    assert _route, "the traced NF4 op route did not run"
    assert eager.dtype == torch.bfloat16 and torch.isfinite(eager).all()
    assert compiled.dtype == eager.dtype, (compiled.dtype, eager.dtype)
    assert torch.equal(compiled, eager), (compiled[0, 0].item(), eager[0, 0].item())


@pytest.mark.parametrize(
    "x_dtype, weight_dtype, autocast_dtype",
    [
        (torch.bfloat16, torch.bfloat16, torch.float16),
        (torch.float32, torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16, torch.float16),  # unchanged: already the autocast dtype
    ],
)
def test_compiled_matches_eager_autocast_dtype(_route, x_dtype, weight_dtype, autocast_dtype):
    torch.manual_seed(0)
    dense = torch.randn(256, 128, device = "cuda", dtype = weight_dtype)
    packed, quant_state = bnb_functional.quantize_4bit(
        dense, quant_type = "nf4", compress_statistics = True
    )
    X = torch.randn(2, 5, 128, device = "cuda", dtype = x_dtype)
    eager, compiled = _compiled_and_eager(X, packed, quant_state, autocast_dtype, fullgraph = True)
    assert _route, "the traced NF4 op route did not run"
    assert eager.dtype == autocast_dtype
    assert compiled.dtype == eager.dtype, (compiled.dtype, eager.dtype)
    torch.testing.assert_close(compiled, eager, rtol = 0, atol = 0)


def test_without_autocast_output_keeps_input_dtype(_route):
    torch.manual_seed(0)
    dense = torch.randn(256, 128, device = "cuda", dtype = torch.bfloat16)
    packed, quant_state = bnb_functional.quantize_4bit(dense, quant_type = "nf4")
    X = torch.randn(3, 128, device = "cuda", dtype = torch.bfloat16)

    def forward(x):
        return kernel_utils.matmul_lora(x, packed, quant_state, None, None, None, fused = False)

    torch._dynamo.reset()
    with torch.no_grad():
        eager = forward(X)
        compiled = torch.compile(forward, fullgraph = True)(X)
    assert _route and compiled.dtype == torch.bfloat16
    torch.testing.assert_close(compiled, eager, rtol = 0, atol = 0)
