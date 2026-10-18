# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The heuristic (UNSLOTH_MOE_DISABLE_AUTOTUNE=1) and default (autotune failed) MoE
kernel configs used fixed tiles. The grouped GEMM kernels static_assert that BLOCK_SIZE_K
divides K (forward, dX) and BLOCK_SIZE_N divides N (dX), so e.g. intermediate size 1408
(DeepSeek-V2-Lite, Qwen1.5-MoE-A2.7B) crashed in the heuristic backward (BLOCK_SIZE_K 256).
The tiles now shrink per GEMM to a power of two dividing N / K, and shapes that already
divided keep exactly the old configs."""

import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("pandas")

MOE_DIR = ROOT / "unsloth" / "kernels" / "moe"

# Fixed tiles before the fix: (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, num_warps, num_stages)
OLD_HEURISTIC = ((64, 128, 128, 8, 3), (64, 64, 256, 8, 4), (64, 64, 256, 8, 4))
OLD_DEFAULT = ((128, 128, 64, 8, 3),) * 3

# (hidden_size, moe_intermediate_size)
SHAPES = [(2048, 1408), (2048, 768), (4096, 1536), (2880, 2880), (4096, 14336), (1024, 352)]
# Shapes whose fallback configs worked before the fix: must stay byte-identical.
# Qwen3-30B-A3B, Qwen3-235B-A22B, Mixtral-8x7B, OLMoE-1B-7B.
UNCHANGED_SHAPES = [(2048, 768), (4096, 1536), (4096, 14336), (2048, 1024)]


@pytest.fixture
def ac(monkeypatch):
    # Import autotune_cache without running unsloth/__init__.py (needs a GPU).
    for name, path in (
        ("unsloth", ROOT / "unsloth"),
        ("unsloth.kernels", ROOT / "unsloth" / "kernels"),
        ("unsloth.kernels.moe", MOE_DIR),
    ):
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, pkg)
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            monkeypatch.delitem(sys.modules, name)
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    yield module
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


def _gemms(hidden, inter):
    # Exactly what unsloth_zoo forward_triton_grouped_gemm passes per GEMM, and the (N, K)
    # each GEMM really has. GEMM1: gate_up, K = H, N = 2I. GEMM2: down, K = I, N = H.
    return [
        (dict(hidden_dim = hidden, intermediate_dim = inter), (2 * inter, hidden)),
        (
            dict(
                hidden_dim = inter,
                intermediate_dim = hidden // 2 if hidden % 2 == 0 else hidden,
            ),
            (hidden, inter),
        ),
    ]


def _tiles(configs):
    return tuple(
        (c.BLOCK_SIZE_M, c.BLOCK_SIZE_N, c.BLOCK_SIZE_K, c.num_warps, c.num_stages) for c in configs
    )


def _divides(old, n, k):
    return all(n % t[1] == 0 and k % t[2] == 0 for t in old)


@pytest.mark.parametrize("hidden,inter", SHAPES + [(2048, 1024)])
@pytest.mark.parametrize("which", ["heuristic", "default"])
def test_fallback_tiles_divide_each_gemm(ac, hidden, inter, which):
    fn = ac._get_heuristic_configs if which == "heuristic" else ac._get_default_configs
    old = OLD_HEURISTIC if which == "heuristic" else OLD_DEFAULT
    for kwargs, (n, k) in _gemms(hidden, inter):
        tiles = _tiles(fn(n_dim = 2 * kwargs["intermediate_dim"], k_dim = kwargs["hidden_dim"]))
        for tile, old_tile in zip(tiles, old):
            m, bn, bk, warps, stages = tile
            assert n % bn == 0 and k % bk == 0, (which, hidden, inter, n, k, tile)
            assert bn >= 16 and bk >= 16
            # Only N / K shrink; never grow (shared memory never exceeds the old config).
            assert bn <= old_tile[1] and bk <= old_tile[2]
            assert (m, warps, stages) == (old_tile[0], old_tile[3], old_tile[4])
        if _divides(old, n, k):
            assert tiles == old, (which, hidden, inter, n, k)


@pytest.mark.parametrize("hidden,inter", UNCHANGED_SHAPES)
@pytest.mark.parametrize("which", ["heuristic", "default"])
def test_working_shapes_keep_exact_configs(ac, hidden, inter, which):
    fn = ac._get_heuristic_configs if which == "heuristic" else ac._get_default_configs
    for kwargs, _ in _gemms(hidden, inter):
        new = fn(n_dim = 2 * kwargs["intermediate_dim"], k_dim = kwargs["hidden_dim"])
        assert [c.__dict__ for c in new] == [c.__dict__ for c in fn()]


def test_no_dims_keeps_old_configs(ac):
    assert _tiles(ac._get_heuristic_configs()) == OLD_HEURISTIC
    assert _tiles(ac._get_default_configs()) == OLD_DEFAULT


def test_1408_backward_k_shrinks(ac):
    # GEMM2 of DeepSeek-V2-Lite / Qwen1.5-MoE: K = 1408 = 128 * 11.
    _, dx, dw = ac._get_heuristic_configs(n_dim = 2048, k_dim = 1408)
    assert (dx.BLOCK_SIZE_K, dw.BLOCK_SIZE_K) == (128, 128)
    assert (dx.BLOCK_SIZE_N, dw.BLOCK_SIZE_N) == (64, 64)


def test_fit_block_size_edges(ac):
    fit = ac._fit_block_size
    assert fit(256, 1408) == 128
    assert fit(128, 2880) == 64
    assert fit(256, 352) == 32
    assert fit(64, None) == 64
    # No power of two >= 16 divides: keep today's tile (kernel asserts as before).
    assert fit(128, 2 * 2881) == 128
    assert fit(64, 24) == 64


@pytest.mark.parametrize("hidden,inter", SHAPES)
def test_get_or_autotune_wires_dims(ac, monkeypatch, hidden, inter):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (7, 5))
    monkeypatch.setattr(ac, "load_cached_config", lambda key: None)
    monkeypatch.setattr(ac, "save_cached_config", lambda *a, **k: None)

    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(ac, "_run_moe_autotuning", boom)
    for env in ("1", "0"):
        monkeypatch.setenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", env)
        ac.clear_cache()
        for kwargs, (n, k) in _gemms(hidden, inter):
            configs = ac.get_or_autotune_moe_kernels(
                num_experts = 8, top_k = 2, dtype = torch.bfloat16, **kwargs
            )
            for c in configs:
                assert n % c.BLOCK_SIZE_N == 0 and k % c.BLOCK_SIZE_K == 0, (env, n, k, c)


# Shared memory step-down of the heuristic dX (measured 122,880 B at 64x64x256 x 4 stages on
# sm_80+; 101,376 B limit on L4 / RTX PRO 6000). The device is mocked.
def _device(ac, monkeypatch, capability, smem):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(ac, "_device_smem_per_block", lambda: smem)


@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (12, 0)])
def test_heuristic_dx_steps_down_on_small_smem(ac, monkeypatch, capability):
    _device(ac, monkeypatch, capability, 101376)
    fwd, dx, dw = ac._get_heuristic_configs(n_dim = 1536, k_dim = 2048)
    assert _tiles((fwd, dx, dw)) == (OLD_HEURISTIC[0], (64, 64, 256, 8, 3), OLD_HEURISTIC[2])


@pytest.mark.parametrize(
    "capability,smem",
    [((7, 5), 65536), ((8, 0), 166912), ((9, 0), 232448), ((10, 0), 232448), ((8, 7), 166912)],
)
@pytest.mark.parametrize("hidden,inter", UNCHANGED_SHAPES)
def test_heuristic_unchanged_with_enough_smem(ac, monkeypatch, capability, smem, hidden, inter):
    for kwargs, _ in _gemms(hidden, inter):
        dims = dict(n_dim = 2 * kwargs["intermediate_dim"], k_dim = kwargs["hidden_dim"])
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        old = [c.__dict__ for c in ac._get_heuristic_configs(**dims)]
        _device(ac, monkeypatch, capability, smem)
        assert [c.__dict__ for c in ac._get_heuristic_configs(**dims)] == old


def test_heuristic_shrunk_dx_is_not_stepped(ac, monkeypatch):
    # 1408 GEMM2: dX already 64x64x128 x 4 stages after the dims fit (73,728 B), not in the table.
    _device(ac, monkeypatch, (8, 9), 101376)
    _, dx, _ = ac._get_heuristic_configs(n_dim = 2048, k_dim = 1408)
    assert (dx.BLOCK_SIZE_K, dx.num_stages) == (128, 4)


@pytest.mark.parametrize("smem", [None, 65536])
def test_heuristic_unknown_or_tiny_smem_keeps_config(ac, monkeypatch, smem):
    # Unknown limit, or one the step-down does not fit either (81,920 B): keep today's config.
    _device(ac, monkeypatch, (8, 9), smem)
    assert _tiles(ac._get_heuristic_configs(n_dim = 1536, k_dim = 2048)) == OLD_HEURISTIC


def test_default_configs_not_stepped(ac, monkeypatch):
    _device(ac, monkeypatch, (8, 9), 101376)
    assert _tiles(ac._get_default_configs(n_dim = 1536, k_dim = 2048)) == OLD_DEFAULT


def test_get_or_autotune_heuristic_steps_down(ac, monkeypatch):
    _device(ac, monkeypatch, (8, 9), 101376)
    monkeypatch.setenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", "1")
    _, dx, _ = ac.get_or_autotune_moe_kernels(
        num_experts = 128, hidden_dim = 2048, intermediate_dim = 768, top_k = 8, dtype = torch.bfloat16
    )
    assert dx.num_stages == 3
