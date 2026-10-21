# SPDX-License-Identifier: AGPL-3.0-only
# Block-FP8 linears sit inside compiled modules, so they must trace without graph breaks.
import os

import pytest

torch = pytest.importorskip("torch")

cuda_available = torch.cuda.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
dev = "cuda" if cuda_available else "xpu" if xpu_available else "cpu"

pytestmark = pytest.mark.skipif(
    not ((cuda_available and torch.cuda.get_device_capability() >= (8, 9)) or xpu_available),
    reason = "block-FP8 Triton kernels need CUDA sm89+ or XPU",
)


def test_block_fp8_linear_compiles_fullgraph_and_matches_eager():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    import torch._dynamo
    from unsloth.kernels import fp8 as F

    def step(X, W, s):
        return F.fp8_linear(X, W, s).float().square().mean()

    torch._dynamo.reset()
    compiled = torch.compile(step, fullgraph = True)
    # The second shape triggers automatic dynamic shapes; K = 200 takes the dequant fallback.
    for N, K in [(512, 256), (256, 256), (256, 768), (768, 256), (512, 200)]:
        torch.manual_seed(0)
        X = torch.randn(256, K, device = dev, dtype = torch.bfloat16, requires_grad = True)
        W = (torch.randn(N, K, device = dev) * 0.02).to(torch.float8_e4m3fn)
        s = torch.rand(-(-N // 128), -(-K // 128), device = dev) + 0.5
        s.block_size = [128, 128]

        compiled(X, W, s).backward()
        grad_compiled, X.grad = X.grad, None
        step(X, W, s).backward()
        assert X.grad.norm() > 0
        torch.testing.assert_close(grad_compiled, X.grad, rtol = 0, atol = 0)

    explain = torch._dynamo.explain(step)(X, W, s)
    assert explain.graph_break_count == 0, explain.break_reasons
