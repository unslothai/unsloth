# SPDX-License-Identifier: AGPL-3.0-only
import os

import pytest
from real_accelerator import has_real_cuda

torch = pytest.importorskip("torch")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA"),
    pytest.mark.skipif(torch.version.hip is not None, reason = "the fused path is CUDA only"),
]


@pytest.fixture
def N(monkeypatch):
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import nvfp4

    # Default row gate of a GPU with a measured limit (L4, sm_89: 256 rows).
    monkeypatch.setattr(nvfp4, "_fused_env", lambda: ("auto", None))
    monkeypatch.setattr(nvfp4, "_device_capability", lambda index: (8, 9))
    return nvfp4


def _random_nvfp4(rows, cols):
    g = torch.Generator(device = "cpu").manual_seed(0)
    packed = torch.randint(0, 256, (rows, cols // 2), dtype = torch.uint8, generator = g)
    scale = (torch.rand(rows, -(-cols // 16), generator = g) * 3 + 0.01).to(torch.float8_e4m3fn)
    return packed.cuda(), scale.cuda(), torch.tensor([7.3], device = "cuda")


@pytest.mark.parametrize("k", [4112, 80, 4096 + 32])
def test_dx_gate_checks_the_reduction_width_too(N, k):
    # dX reduces over N and writes K columns: an unaligned K is the padded-tile case the gate keeps dense.
    packed, scale, gs = _random_nvfp4(4096, k)
    G = torch.randn(16, 4096, device = "cuda", dtype = torch.float16)
    X = torch.randn(16, k, device = "cuda", dtype = torch.float16)
    assert not N._fused_ok(X, packed, scale, gs, k)
    assert not N._fused_ok(G, packed, scale, gs, packed.shape[0])


def test_aligned_shapes_still_fused_both_ways(N):
    packed, scale, gs = _random_nvfp4(4096, 4096)
    G = torch.randn(16, 4096, device = "cuda", dtype = torch.float16)
    assert N._fused_ok(G, packed, scale, gs, 4096)
    assert N._fused_ok(G, packed, scale, gs, packed.shape[0])
    X3 = torch.randn(2, 8, 4096, device = "cuda", dtype = torch.float16)
    assert N._fused_ok(X3, packed, scale, gs, 4096)
    # A strided 3D batch that still views as 2D needs no copy.
    assert N._fused_ok(torch.randn(2, 8, 8192, device = "cuda", dtype = torch.float16)[..., :4096], packed, scale, gs, 4096)

