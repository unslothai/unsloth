# SPDX-License-Identifier: AGPL-3.0-only
# Block-FP8 kernels past 2**31 elements (8 GRPO rows x 16K tokens x QwQ-32B's 27648-wide MLP) raised
# "illegal memory access" from int32 offsets (#3921).
import os

import pytest

torch = pytest.importorskip("torch")

cuda_available = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (cuda_available and torch.cuda.get_device_capability() >= (8, 9)),
    reason = "FP8 GEMMs need CUDA sm89+",
)


@pytest.fixture(scope = "module")
def F():
    os.environ.setdefault("UNSLOTH_IS_PRESENT", "1")
    from unsloth.kernels import fp8
    return fp8


def _need_free_gb(gb):
    if torch.cuda.mem_get_info()[0] < gb * 1024**3:
        pytest.skip(f"needs {gb} GB free GPU memory")


def test_act_quant_past_int32(F):
    _need_free_gb(16)
    M, K = (2**31) // 4096 + 64, 4096
    X = torch.randn(M, K, device = "cuda", dtype = torch.bfloat16)
    q, s = F.act_quant(X, 128)
    tail = slice(M - 64, M)
    q_ref, s_ref = F.act_quant(X[tail].contiguous(), 128)
    torch.cuda.synchronize()
    assert torch.equal(s[tail], s_ref)
    assert torch.equal(q[tail].view(torch.uint8), q_ref.view(torch.uint8))


def test_block_matmul_output_past_int32(F):
    _need_free_gb(16)
    N, K = 32768, 256
    M = (2**31) // N + 256
    w = (torch.randn(N, K, device = "cuda") * 0.05).to(torch.float8_e4m3fn)
    w_s = torch.rand(N // 128, K // 128, device = "cuda") * 0.01 + 0.001
    X = torch.randn(M, K, device = "cuda", dtype = torch.bfloat16)
    q, s = F.act_quant(X, 128)
    out = F.w8a8_block_fp8_matmul_triton(q, w, s, w_s, [128, 128], output_dtype = torch.bfloat16)
    rows = slice(M - 512, M)  # straddles element 2**31
    ref = F.w8a8_block_fp8_matmul_triton(
        q[rows].contiguous(), w, s[rows].contiguous(), w_s, [128, 128], output_dtype = torch.bfloat16
    )
    torch.cuda.synchronize()
    assert torch.equal(out[rows], ref)


def test_torchao_route_hands_large_launches_to_int64_kernel(F, monkeypatch):
    calls = []
    monkeypatch.setattr(F, "w8a8_block_fp8_matmul_triton", lambda *a, **k: calls.append("triton"))
    monkeypatch.setattr(
        F, "torchao_blockwise_gemm", lambda *a, **k: calls.append("torchao") or a[0]
    )
    # meta tensors: only shapes matter for routing
    meta = lambda *shape: torch.empty(*shape, device = "meta", dtype = torch.float8_e4m3fn)
    w_s = torch.empty(216, 40, device = "meta")
    F.torchao_block_matmul(
        meta(4096, 5120), meta(27648, 5120), torch.empty(4096, 40, device = "meta"), w_s, [128, 128]
    )
    F.torchao_block_matmul(
        meta(81920, 5120), meta(27648, 5120), torch.empty(81920, 40, device = "meta"), w_s, [128, 128]
    )
    assert calls == ["torchao", "triton"]
