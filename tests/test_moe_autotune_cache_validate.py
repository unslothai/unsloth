# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Disk cache hits in the MoE autotune cache are returned without re-tuning. Entries written
before the cache carried a schema / Triton identity (when the loader still discarded them) or
before a kernel change (the per-expert dW kernel needs about 2x the shared memory of the legacy
dW kernel for the same tile) must not be launched unvalidated: unversioned entries are misses,
and every loaded entry is refitted to the tensor dims and the device shared memory."""

import hashlib
import importlib
import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

MOE_DIR = ROOT / "unsloth" / "kernels" / "moe"
SM86_SMEM = 101376


@pytest.fixture
def ac(monkeypatch, tmp_path):
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
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 6))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None)
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    monkeypatch.setattr(module, "_device_smem_per_block", lambda: SM86_SMEM)
    module.clear_cache()
    yield module
    module.clear_cache()
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


DIMS = dict(num_experts = 128, hidden_dim = 2048, intermediate_dim = 768, top_k = 8)


def _configs(fwd, dx, dw):
    from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
        KernelConfigBackward_dW,
        KernelConfigBackward_dX,
        KernelConfigForward,
    )
    def kw(t):
        return dict(
            zip(("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "num_warps", "num_stages"), t)
        )

    return (
        KernelConfigForward(**kw(fwd)),
        KernelConfigBackward_dX(**kw(dx)),
        KernelConfigBackward_dW(**kw(dw)),
    )


SMALL = (64, 64, 64, 4, 2)  # fits sm86 for every kind and divides every dim used here


def _tile(c):
    return (c.BLOCK_SIZE_M, c.BLOCK_SIZE_N, c.BLOCK_SIZE_K, c.num_warps, c.num_stages)


class _Autotune:
    def __init__(self, ac):
        self.calls = 0
        self.ac = ac

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return _configs(SMALL, SMALL, SMALL)


def _legacy_key(
    dims,
    dtype = torch.bfloat16,
    capability = (8, 6),
    seq_len = 8192,
):
    key_data = dict(
        num_experts = dims["num_experts"],
        hidden_dim = dims["hidden_dim"],
        intermediate_dim = dims["intermediate_dim"],
        top_k = dims["top_k"],
        dtype = str(dtype),
        device_capability = capability,
        seq_len = seq_len,
    )
    return hashlib.md5(json.dumps(key_data, sort_keys = True).encode()).hexdigest()


def _legacy_payload(configs):
    # Exactly what the pre-schema save_cached_config wrote.
    return {
        "timestamp": 0.0,
        "device_capability": [8, 6],
        "config_fwd": configs[0].__dict__,
        "config_bwd_dx": configs[1].__dict__,
        "config_bwd_dw": configs[2].__dict__,
        "metadata": {},
    }


def test_legacy_unversioned_cache_file_is_ignored(ac, monkeypatch):
    legacy = _configs(SMALL, SMALL, (128, 256, 256, 8, 4))
    payload = _legacy_payload(legacy)
    cache_dir = Path(ac._get_cache_file_path("x")).parent
    legacy_file = cache_dir / f"{_legacy_key(DIMS)}.json"
    current_key = ac._get_cache_key(dtype = torch.bfloat16, device_capability = (8, 6), **DIMS)
    for path in {legacy_file, Path(ac._get_cache_file_path(current_key))}:
        path.write_text(json.dumps(payload), encoding = "utf-8")

    tuner = _Autotune(ac)
    monkeypatch.setattr(ac, "_run_moe_autotuning", tuner)
    configs = ac.get_or_autotune_moe_kernels(dtype = torch.bfloat16, **DIMS)
    assert tuner.calls == 1, "an unversioned disk entry was reused without re-tuning"
    assert [_tile(c) for c in configs] == [SMALL, SMALL, SMALL]
    assert legacy_file.exists(), "a legacy file under another key must be left alone"


def test_oversized_dw_tile_from_disk_is_fitted(ac, monkeypatch):
    # Legacy-dW-valid tile: 132,096 B on the per-expert dW kernel vs 101,376 B on sm86.
    key = ac._get_cache_key(dtype = torch.bfloat16, device_capability = (8, 6), **DIMS)
    ac.save_cached_config(key, *_configs(SMALL, SMALL, (128, 256, 256, 8, 4)))
    tuner = _Autotune(ac)
    monkeypatch.setattr(ac, "_run_moe_autotuning", tuner)
    configs = ac.get_or_autotune_moe_kernels(dtype = torch.bfloat16, **DIMS)
    assert tuner.calls == 0
    assert _tile(configs[2]) == (64, 256, 256, 8, 3)
    for kind, config in zip(("fwd", "dX", "dW"), configs):
        assert ac._estimate_smem(kind, config, 2, 2, (8, 6)) <= SM86_SMEM


def test_block_k_larger_than_dim_from_disk_is_fitted(ac, monkeypatch):
    # N = 2 * 64 = 128, K = 64: a saved BLOCK_SIZE_K = 128 fails tl.static_assert(K % BLOCK_SIZE_K == 0).
    dims = dict(num_experts = 8, hidden_dim = 64, intermediate_dim = 64, top_k = 2)
    key = ac._get_cache_key(dtype = torch.bfloat16, device_capability = (8, 6), **dims)
    tile = (64, 128, 128, 4, 2)
    ac.save_cached_config(key, *_configs(tile, tile, (32, 128, 128, 4, 2)))
    tuner = _Autotune(ac)
    monkeypatch.setattr(ac, "_run_moe_autotuning", tuner)
    configs = ac.get_or_autotune_moe_kernels(dtype = torch.bfloat16, **dims)
    assert tuner.calls == 0
    for config in configs:
        assert 128 % config.BLOCK_SIZE_N == 0
        assert 64 % config.BLOCK_SIZE_K == 0


def test_valid_current_entry_hits_without_autotune(ac, monkeypatch):
    saved = _configs((64, 128, 64, 4, 3), (64, 64, 128, 4, 3), (64, 64, 64, 8, 2))
    key = ac._get_cache_key(dtype = torch.bfloat16, device_capability = (8, 6), **DIMS)
    ac.save_cached_config(key, *saved)
    tuner = _Autotune(ac)
    monkeypatch.setattr(ac, "_run_moe_autotuning", tuner)
    first = ac.get_or_autotune_moe_kernels(dtype = torch.bfloat16, **DIMS)
    assert tuner.calls == 0
    assert [c.__dict__ for c in first] == [c.__dict__ for c in saved]
    assert os.path.exists(ac._get_cache_file_path(key))
    monkeypatch.setattr(ac, "load_cached_config", lambda *a, **k: pytest.fail("disk re-read"))
    assert ac.get_or_autotune_moe_kernels(dtype = torch.bfloat16, **DIMS) is first
    assert tuner.calls == 0
