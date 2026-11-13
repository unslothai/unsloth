# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An explicit dW tile the legacy kernel launched must still launch with the per-expert kernel as default, and
a tile too big for this card (T4: 64 KiB) launches with a fitted copy instead of raising OutOfResources.

128x256x256 at 4 stages needs 131,584 B (sm_80/86/89/120) / 262,656 B (sm_90) per-expert vs 69,632 /
131,072 B legacy (Triton 3.8 metadata.shared), over the 101,376 / 232,448 B limits."""

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
    assert gg._per_expert_dW_fits(device, 2, M, N, K, stages) == (gpu == "A100")
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
    acc = default.float()
    gg._grouped_gemm_dW_accumulate_impl(X, dY, m_sizes, gather, acc, topk, **kw)
    torch.cuda.synchronize()
    torch.testing.assert_close(acc, 2 * default.float(), rtol = 1e-2, atol = 1e-2)


# Triton 3.6 and 3.8 metadata.shared (identical) of the per-expert dW kernel, bf16, unpermuted operands: its one
# M loop is software pipelined, so X is buffered num_stages - 1 times (sm_80-class) / both tiles per stage (sm_90).
PIPELINED_SHARED = {
    (8, 9): {
        (64, 64, 256, 2): 40960,
        (64, 64, 256, 3): 73728,
        (64, 64, 256, 4): 106496,
        (32, 64, 256, 4): 53248,
        (128, 128, 64, 3): 65536,
        (128, 128, 64, 4): 81920,
    },
    (9, 0): {
        (64, 64, 256, 2): 81920,
        (64, 64, 256, 4): 163840,
        (32, 64, 256, 3): 61440,
        (128, 128, 64, 4): 196608,
    },
}


@pytest.mark.parametrize("capability", list(PIPELINED_SHARED))
def test_pipelined_estimate_covers_compiled_shared_memory(gg, capability):
    from unsloth.kernels.moe.autotune_cache import _estimate_smem
    for (M, N, K, stages), shared in PIPELINED_SHARED[capability].items():
        tile = types.SimpleNamespace(
            BLOCK_SIZE_M = M, BLOCK_SIZE_N = N, BLOCK_SIZE_K = K, num_stages = stages
        )
        need = _estimate_smem("dW", tile, 2, 2, capability, True)
        assert shared <= need <= shared + 2048, (capability, M, N, K, stages, need, shared)


@pytest.mark.parametrize("gpu", ["A10", "L4", "RTX5090"])
def test_unpermuted_heuristic_tile_drops_a_stage(monkeypatch, gg, gpu):
    # 64x64x256 x 4 stages unpermuted needs 106,496 B on a 101,376 B card: run 3 stages, same tile.
    capability, smem = GPUS[gpu]
    _card(monkeypatch, gg, capability, smem)
    device = torch.device("cuda", 0)
    assert gg._per_expert_dW_stages(device, 2, *HEURISTIC, pipelined = True) == 3
    assert gg._per_expert_dW_stages(device, 2, *HEURISTIC) == 4  # gathered loads do not pipeline


@pytest.mark.parametrize("gpu", ["A100", "H100", "B200"])
def test_unpermuted_heuristic_tile_unchanged_with_enough_smem(monkeypatch, gg, gpu):
    capability, smem = GPUS[gpu]
    _card(monkeypatch, gg, capability, smem)
    device = torch.device("cuda", 0)
    assert gg._per_expert_dW_stages(device, 2, *HEURISTIC, pipelined = True) == 4


def _ref(X, dY, m_sizes, E, N, K):
    X, dY = X.double(), dY.double()
    out = torch.zeros(E, N, K, dtype = torch.float64, device = X.device)
    start = 0
    for e, m in enumerate(m_sizes.tolist()):
        out[e] = dY[start : start + m].T @ X[start : start + m]
        start += m
    return out


def test_unpermuted_heuristic_tile_launches_on_this_gpu(gg):
    # L4 / A10 on main: OutOfResources(106496, 101376) for the Qwen3-30B gate_up dW (UNSLOTH_MOE_DISABLE_AUTOTUNE=1).
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    E, N, K, T, topk = 128, 1536, 2048, 256, 8
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    g = torch.Generator(device = "cuda").manual_seed(5)
    X = torch.randn(T * topk, K, device = "cuda", dtype = dtype, generator = g)
    dY = torch.randn(T * topk, N, device = "cuda", dtype = dtype, generator = g)
    m_sizes = torch.full((E,), T * topk // E, device = "cuda", dtype = torch.int32)
    gather = torch.arange(T * topk, device = "cuda", dtype = torch.int32)
    M, BN, BK, stages = HEURISTIC
    kw = dict(BLOCK_SIZE_M = M, BLOCK_SIZE_N = BN, BLOCK_SIZE_K = BK, num_warps = 8, num_stages = stages)
    out = gg._grouped_gemm_dW_impl(X, dY, m_sizes, gather, topk, **kw)
    torch.cuda.synchronize()
    ref = _ref(X, dY, m_sizes, E, N, K)
    assert ((out.double() - ref).norm() / ref.norm()) < 5e-3
