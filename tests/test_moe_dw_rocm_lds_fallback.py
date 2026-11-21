# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Per-expert dW tile over the card's shared memory with no smaller tile known (ROCm: no shared memory model
here, so _card_smem is None). The heuristic 64x64x256 4-stage tile needs ~80 KiB of LDS on gfx1151 (64 KiB):
it must fall back to the legacy kernel (which fit it before the per-expert kernel) instead of raising
OutOfResources, give the legacy result bit for bit, and not retry the per-expert launch on the next call.
Simulated on NVIDIA: _card_smem -> None and a per-expert launch that raises OutOfResources."""

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

HEURISTIC = (64, 64, 256, 4)


@pytest.fixture
def gg(monkeypatch):
    stubs = {
        "unsloth": ROOT / "unsloth",
        "unsloth.kernels": ROOT / "unsloth" / "kernels",
        "unsloth.kernels.moe": ROOT / "unsloth" / "kernels" / "moe",
    }
    for name, path in stubs.items():
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)
    sys.modules["unsloth"].DEVICE_TYPE = "cuda"
    device_type = types.ModuleType("unsloth.device_type")
    device_type.cuda_visible_capabilities = lambda: ()
    monkeypatch.setitem(sys.modules, device_type.__name__, device_type)
    utils = types.ModuleType("unsloth.kernels.utils")
    utils.torch_gpu_device = lambda device: torch.cuda.device(device)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            monkeypatch.delitem(sys.modules, name)
    try:
        module = importlib.import_module("unsloth.kernels.moe.grouped_gemm.interface")
    except Exception as e:  # pragma: no cover - depends on the installed Triton
        pytest.skip(f"grouped_gemm interface is unimportable here: {type(e).__name__}: {e}")
    getattr(module, "_PER_EXPERT_DW_FITS", {}).clear()
    yield module
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


def _ref(X, dY, m_sizes, E, N, K):
    X, dY = X.double(), dY.double()
    out = torch.zeros(E, N, K, dtype = torch.float64, device = X.device)
    start = 0
    for e, m in enumerate(m_sizes.tolist()):
        out[e] = dY[start : start + m].T @ X[start : start + m]
        start += m
    return out


class _OverLds:
    """Stands in for the per-expert kernel on a card whose LDS it overflows."""

    def __init__(self, error):
        self.calls = 0
        self.error = error

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls += 1
            raise self.error(81920, 65536, "shared memory")

        return launch


def _inputs(dtype):
    E, N, K, T, topk = 16, 256, 512, 64, 4
    g = torch.Generator(device = "cuda").manual_seed(7)
    X = torch.randn(T, K, device = "cuda", dtype = dtype, generator = g)
    dY = torch.randn(T * topk, N, device = "cuda", dtype = dtype, generator = g)
    m_sizes = torch.full((E,), T * topk // E, device = "cuda", dtype = torch.int32)
    m_sizes[0] += m_sizes[1]
    m_sizes[1] = 0  # an empty expert: the legacy kernel skips it, so dW must start at zero
    gather = torch.randperm(T * topk, device = "cuda", generator = g).to(torch.int32)
    return X, dY, m_sizes, gather, topk, E, N, K


@pytest.mark.parametrize("accumulate", [False, True], ids = ["fresh", "accumulate"])
def test_lds_overflow_falls_back_to_legacy(monkeypatch, gg, accumulate):
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    X, dY, m_sizes, gather, topk, E, N, K = _inputs(dtype)
    M, BN, BK, stages = HEURISTIC
    kw = dict(
        BLOCK_SIZE_M = M, BLOCK_SIZE_N = BN, BLOCK_SIZE_K = BK, num_warps = 8, num_stages = stages,
        permute_x = True,
    )  # fmt: skip
    monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", "legacy")
    legacy = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw)
    monkeypatch.delenv("UNSLOTH_MOE_DW_KERNEL")
    torch.cuda.synchronize()

    over = _OverLds(gg.OutOfResources)
    monkeypatch.setattr(gg, "_grouped_gemm_dW_per_expert_kernel", over)
    monkeypatch.setattr(gg, "_card_smem", lambda index: None)  # as on ROCm
    gg._PER_EXPERT_DW_FITS.clear()
    gg._DW_CARD_FITS.clear()
    for _ in range(2):
        if accumulate:
            got = torch.full((E, N, K), 0.5, device = "cuda", dtype = torch.float32)
            gg._grouped_gemm_dW_accumulate_impl(X, dY, m_sizes, gather, got, topk, **kw)
            torch.cuda.synchronize()
            torch.testing.assert_close(got, legacy.float() + 0.5, rtol = 1e-2, atol = 1e-2)
        else:
            got = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw)
            torch.cuda.synchronize()
            assert torch.equal(got, legacy)
            assert bool((got[1] == 0).all())
    assert over.calls == 1, over.calls  # the second call goes straight to the legacy kernel
    rel = (legacy.double() - _ref(X[gather.long() // topk], dY, m_sizes, E, N, K)).norm()
    assert rel / _ref(X[gather.long() // topk], dY, m_sizes, E, N, K).norm() < 5e-3
