# SPDX-License-Identifier: AGPL-3.0-only
# Block-FP8 linears sit inside compiled modules (e.g. Gemma 3 attention / MLP), so they must trace without graph breaks.
import os

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason = "block-FP8 Triton kernels need CUDA sm89+",
)


def test_block_fp8_linear_compiles_fullgraph_and_matches_eager():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    import torch._dynamo
    from unsloth.kernels import fp8 as F

    def step(X, W, s):
        return F.fp8_linear(X, W, s).float().square().mean()

    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph = True)
    # Several weight shapes (the second triggers automatic dynamic shapes); K = 200 takes the dequant fallback.
    for N, K in [(512, 256), (256, 256), (256, 768), (768, 256), (512, 200)]:
        torch.manual_seed(0)
        X = torch.randn(256, K, device = "cuda", dtype = torch.bfloat16, requires_grad = True)
        W = (torch.randn(N, K, device = "cuda") * 0.02).to(torch.float8_e4m3fn)
        s = torch.rand(-(-N // 128), -(-K // 128), device = "cuda") + 0.5
        s.block_size = [128, 128]

        compiled(X, W, s).backward()
        grad_compiled, X.grad = X.grad, None
        step(X, W, s).backward()
        assert X.grad.norm() > 0
        torch.testing.assert_close(grad_compiled, X.grad, rtol = 0, atol = 0)

    explain = torch._dynamo.explain(step)(X, W, s)
    assert explain.graph_break_count == 0, explain.break_reasons
