# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An explicit dW tile the legacy kernel launched must still launch once the per-expert kernel is the default.

The per-expert kernel double buffers both operand tiles, so 128x256x256 at 4 stages compiles to
131,584 B on sm_80 / sm_86 / sm_89 / sm_120 (legacy 69,632 B) and 262,656 B on sm_90 (legacy
131,072 B): over the 101,376 B / 232,448 B limits (Triton metadata.shared, CPU cross-compiled,
Triton 3.8). Fitted configs (heuristic, default, refit disk entries) never exceed the limit;
a caller-supplied tile, or one fitted against another card, can, and raised OutOfResources."""

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

GPUS = {
    "A100": ((8, 0), 166912),
    "A10": ((8, 6), 101376),
    "L4": ((8, 9), 101376),
    "H100": ((9, 0), 232448),
    "B200": ((10, 0), 232448),
    "RTX5090": ((12, 0), 101376),
}
EXPLICIT = (128, 256, 256, 4)  # fits the legacy kernel on every card above
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


def _card(monkeypatch, gg, capability, smem):
    props = types.SimpleNamespace(
        shared_memory_per_block_optin = smem, major = capability[0], minor = capability[1]
    )
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a, **k: props)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    getattr(gg, "_PER_EXPERT_DW_FITS", {}).clear()


@pytest.mark.parametrize("gpu", list(GPUS))
def test_explicit_tile_routes_by_operand_card(monkeypatch, gg, gpu):
    capability, smem = GPUS[gpu]
    _card(monkeypatch, gg, capability, smem)
    device = torch.device("cuda", 0)
    M, N, K, stages = EXPLICIT
    # Over the limit everywhere except A100's 166,912 B (131,584 B there).
    assert gg._per_expert_dW_fits(device, 2, M, N, K, stages) == (gpu == "A100")
    # float32 X runs the per-expert kernel single stage: 262,144 B of operands fits no card.
    assert not gg._per_expert_dW_fits(device, 4, M, N, K, stages)
    M, N, K, stages = HEURISTIC
    assert gg._per_expert_dW_fits(device, 2, M, N, K, stages)
    assert gg._per_expert_dW_fits(device, 4, M, N, K, stages)


def test_unknown_device_keeps_the_per_expert_kernel(monkeypatch, gg):
    def boom(*a, **k):
        raise RuntimeError("no CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert gg._per_expert_dW_fits(torch.device("cuda", 0), 2, *EXPLICIT)


def test_explicit_tile_launches_on_this_gpu(monkeypatch, gg):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if getattr(torch.version, "hip", None):
        pytest.skip("ROCm reports shared memory differently")
    smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    if smem > 232448:
        pytest.skip(f"{smem} B shared memory fits the per-expert explicit tile")
    E, N, K, T, topk = 128, 1536, 2048, 64, 8
    g = torch.Generator(device = "cuda").manual_seed(4)
    X = torch.randn(T, K, device = "cuda", dtype = torch.bfloat16, generator = g)
    dY = torch.randn(T * topk, N, device = "cuda", dtype = torch.bfloat16, generator = g)
    m_sizes = torch.full((E,), T * topk // E, device = "cuda", dtype = torch.int32)
    m_sizes[0] += m_sizes[1]
    m_sizes[1] = 0
    gather = torch.randperm(T * topk, device = "cuda", generator = g).to(torch.int32)
    M, BN, BK, stages = EXPLICIT
    kw = dict(
        BLOCK_SIZE_M = M,
        BLOCK_SIZE_N = BN,
        BLOCK_SIZE_K = BK,
        num_warps = 8,
        num_stages = stages,
        permute_x = True,
    )
    monkeypatch.setenv("UNSLOTH_MOE_DW_KERNEL", "legacy")
    legacy = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw)
    monkeypatch.delenv("UNSLOTH_MOE_DW_KERNEL")
    default = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw)
    torch.cuda.synchronize()
    assert torch.equal(default, legacy)
    assert bool((default[1] == 0).all())
    # The accumulate path routes the same way: dW += dW.
    acc = default.float()
    gg._grouped_gemm_dW_accumulate_impl(X, dY, m_sizes, gather, acc, topk, **kw)
    torch.cuda.synchronize()
    torch.testing.assert_close(acc, 2 * default.float(), rtol = 1e-2, atol = 1e-2)
