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
# Heuristic dW when K leaves BLOCK_SIZE_K 64 and 128 divides N (gpt-oss gate_up: N = 5760, K = 2880).
WIDE_DW = (64, 128, 64, 8, 3)

# (hidden_size, moe_intermediate_size)
SHAPES = [(2048, 1408), (2048, 768), (4096, 1536), (2880, 2880), (4096, 14336), (1024, 352)]
# Qwen3-30B-A3B, Qwen3-235B-A22B, Mixtral-8x7B, OLMoE-1B-7B: their configs must stay byte-identical.
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
    # As unsloth_zoo forward_triton_grouped_gemm calls it. GEMM1: K = H, N = 2I. GEMM2: K = I, N = H.
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


@pytest.fixture
def no_smem_fit(ac, monkeypatch):
    # The dims fit alone: on a card with less shared memory (L4, A10: 101,376 B) the smem fit also steps the
    # heuristic dX down to 3 stages, which the smem tests below cover.
    monkeypatch.setattr(ac, "_device_smem_per_block", lambda *a, **k: None)


@pytest.mark.parametrize("hidden,inter", SHAPES + [(2048, 1024)])
@pytest.mark.parametrize("which", ["heuristic", "default"])
def test_fallback_tiles_divide_each_gemm(ac, no_smem_fit, monkeypatch, hidden, inter, which):
    monkeypatch.setattr(ac, "_wide_dw_device_ok", lambda device = None: True)
    fn = ac._get_heuristic_configs if which == "heuristic" else ac._get_default_configs
    old = OLD_HEURISTIC if which == "heuristic" else OLD_DEFAULT
    for kwargs, (n, k) in _gemms(hidden, inter):
        tiles = _tiles(fn(n_dim = 2 * kwargs["intermediate_dim"], k_dim = kwargs["hidden_dim"]))
        for kind, tile, old_tile in zip(("fwd", "dX", "dW"), tiles, old):
            m, bn, bk, warps, stages = tile
            assert n % bn == 0 and k % bk == 0, (which, hidden, inter, n, k, tile)
            if (
                which == "heuristic"
                and kind == "dW"
                and bk == 64
                and n % 128 == 0
                and n * k >= 768 * 1024
            ):
                assert tile == WIDE_DW, (
                    hidden,
                    inter,
                    n,
                    k,
                    tile,
                )  # K left BLOCK_SIZE_K 64: wider N tile
                continue
            assert bn >= 16 and bk >= 16
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


def test_no_dims_keeps_old_configs(ac, no_smem_fit):
    assert _tiles(ac._get_heuristic_configs()) == OLD_HEURISTIC
    assert _tiles(ac._get_default_configs()) == OLD_DEFAULT


def test_1408_backward_k_shrinks(ac):
    # GEMM2 of DeepSeek-V2-Lite / Qwen1.5-MoE: K = 1408 = 128 * 11.
    _, dx, dw = ac._get_heuristic_configs(n_dim = 2048, k_dim = 1408)
    assert (dx.BLOCK_SIZE_K, dw.BLOCK_SIZE_K) == (128, 128)
    assert (dx.BLOCK_SIZE_N, dw.BLOCK_SIZE_N) == (64, 64)


def test_heuristic_dw_widens_n_when_k_leaves_64(ac, monkeypatch):
    monkeypatch.setattr(ac, "_wide_dw_device_ok", lambda device = None: True)
    # gpt-oss-20b: GEMM1 N = 5760, K = 2880 -> dW 64 x 128 x 64 at 3 stages; fwd / dX tiles unchanged.
    fwd, dx, dw = ac._get_heuristic_configs(n_dim = 5760, k_dim = 2880)
    assert _tiles((fwd, dx)) == ((64, 128, 64, 8, 3), (64, 64, 64, 8, 4))
    assert _tiles((dw,)) == (WIDE_DW,)
    # GEMM2 N = 2880: 128 does not divide it, so dW keeps the fitted 64 x 64 x 64 at 4 stages.
    assert _tiles(ac._get_heuristic_configs(n_dim = 2880, k_dim = 2880)[2:]) == ((64, 64, 64, 8, 4),)
    # K = 1408 keeps BLOCK_SIZE_K 128 and K = 352 drops to 32: neither is widened.
    assert _tiles(ac._get_heuristic_configs(n_dim = 2048, k_dim = 1408)[2:]) == ((64, 64, 128, 8, 4),)
    assert _tiles(ac._get_heuristic_configs(n_dim = 2048, k_dim = 352)[2:]) == ((64, 64, 32, 8, 4),)
    # A small N x K output keeps the fitted tile: half the programs leave the card idle.
    assert _tiles(ac._get_heuristic_configs(n_dim = 1024, k_dim = 576)[2:]) == ((64, 64, 64, 8, 4),)
    assert _tiles(ac._get_heuristic_configs(n_dim = 1024, k_dim = 192)[2:]) == ((64, 64, 64, 8, 4),)
    assert _tiles(ac._get_heuristic_configs(n_dim = 1024, k_dim = 960)[2:]) == (WIDE_DW,)
    # the fallback (autotune failed) configs are not touched
    assert _tiles(ac._get_default_configs(n_dim = 5760, k_dim = 2880)[2:]) == ((128, 128, 64, 8, 3),)


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


def test_heuristic_unknown_smem_keeps_config(ac, monkeypatch):
    _device(ac, monkeypatch, (8, 9), None)
    assert _tiles(ac._get_heuristic_configs(n_dim = 1536, k_dim = 2048)) == OLD_HEURISTIC


def test_heuristic_tiny_smem_steps_down_until_it_fits(ac, monkeypatch):
    _device(ac, monkeypatch, (8, 9), 65536)
    configs = ac._get_heuristic_configs(n_dim = 1536, k_dim = 2048)
    for kind, c in zip(("fwd", "dX", "dW"), configs):
        assert ac._estimate_smem(kind, c, 2, 2, (8, 9)) <= 65536


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


def test_smem_limit_follows_the_current_device(ac, monkeypatch):
    # Mixed GPUs in one process: the limit must come from the current device, not the first one seen.
    import types

    limits = {0: 166912, 1: 101376}
    current = {"index": 0}
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current["index"])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: types.SimpleNamespace(shared_memory_per_block_optin = limits[index]),
    )
    assert ac._device_smem_per_block() == 166912
    current["index"] = 1
    assert ac._device_smem_per_block() == 101376


@pytest.mark.parametrize(
    "available,hip,capability,ok",
    [
        (True, None, (7, 5), False),
        (True, None, (8, 0), True),
        (True, None, (10, 0), True),
        (True, "7.1", (11, 5), False),
        (False, None, (8, 0), False),
    ],
)
def test_wide_dw_tile_only_on_sm80_plus_nvidia(ac, monkeypatch, available, hip, capability, ok):
    # T4 (sm75): the 128-wide dW tile was 7-9x slower than 64 x 64 (fp16, torch 2.14); ROCm unmeasured.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    assert ac._wide_dw_device_ok() is ok


def test_t4_keeps_the_fitted_dw_tile(ac, no_smem_fit, monkeypatch):
    monkeypatch.setattr(ac, "_wide_dw_device_ok", lambda device = None: False)
    assert _tiles(ac._get_heuristic_configs(n_dim = 5760, k_dim = 2880)[2:]) == ((64, 64, 64, 8, 4),)
