# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The on-disk MoE autotune cache stores device_capability through JSON, which turns the
(major, minor) tuple into a list. load_cached_config compared that list to the live tuple,
so every lookup looked like a device change: the entry was deleted and autotuning re-ran on
every process. A saved config must now be reused, and a real capability change must still
invalidate it."""

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
    # The cache lives under ~/.cache/unsloth/moe_autotune: point ~ at a scratch dir.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_MOE_DISABLE_AUTOTUNE", raising = False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))
    module = importlib.import_module("unsloth.kernels.moe.autotune_cache")
    # Ample SMEM so the load-time refit keeps tuned tiles as saved.
    monkeypatch.setattr(module, "_device_smem_per_block", lambda: 1 << 20)
    module.clear_cache()
    yield module
    module.clear_cache()
    for name in list(sys.modules):
        if name.startswith("unsloth.kernels.moe."):
            sys.modules.pop(name, None)


DIMS = dict(num_experts = 128, hidden_dim = 2048, intermediate_dim = 768, top_k = 8)


def _tuned_configs():
    from unsloth.kernels.moe.grouped_gemm.kernels.tuning import (
        KernelConfigBackward_dW,
        KernelConfigBackward_dX,
        KernelConfigForward,
    )

    # Distinct from both the heuristic and the default tiles.
    return (
        KernelConfigForward(
            BLOCK_SIZE_M = 64, BLOCK_SIZE_N = 256, BLOCK_SIZE_K = 128, num_warps = 4, num_stages = 3
        ),
        KernelConfigBackward_dX(
            BLOCK_SIZE_M = 32, BLOCK_SIZE_N = 128, BLOCK_SIZE_K = 64, num_warps = 4, num_stages = 4
        ),
        KernelConfigBackward_dW(
            BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 64, BLOCK_SIZE_K = 32, num_warps = 8, num_stages = 5
        ),
    )


def _key(ac, capability = (9, 0)):
    return ac._get_cache_key(dtype = torch.bfloat16, device_capability = capability, **DIMS)


def _no_autotune(*args, **kwargs):
    raise AssertionError("autotuning re-ran although a cached config was saved")


def test_saved_config_is_reused_across_processes(ac, monkeypatch):
    tuned = _tuned_configs()
    key = _key(ac)
    ac.save_cached_config(key, *tuned)
    cache_file = ac._get_cache_file_path(key)
    with open(cache_file, encoding = "utf-8") as f:
        assert json.load(f)["device_capability"] == [9, 0]  # JSON has no tuples

    # A new process: empty in-memory cache, autotuning must not run.
    ac.clear_cache()
    monkeypatch.setattr(ac, "_run_moe_autotuning", _no_autotune)
    configs = ac.get_or_autotune_moe_kernels(dtype = torch.bfloat16, **DIMS)
    assert [c.__dict__ for c in configs] == [c.__dict__ for c in tuned]
    assert [type(c) for c in configs] == [type(c) for c in tuned]
    assert os.path.exists(cache_file), "a valid cache entry was deleted"


def test_load_cached_config_hit(ac):
    key = _key(ac)
    ac.save_cached_config(key, *_tuned_configs())
    loaded = ac.load_cached_config(key)
    assert loaded is not None
    assert loaded["config_fwd"]["BLOCK_SIZE_N"] == 256
    assert os.path.exists(ac._get_cache_file_path(key))


def test_capability_change_still_invalidates(ac, monkeypatch):
    key = _key(ac)
    ac.save_cached_config(key, *_tuned_configs())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    assert ac.load_cached_config(key) is None
    assert not os.path.exists(ac._get_cache_file_path(key))


@pytest.mark.parametrize("stored", [None, "9.0", 90])
def test_malformed_capability_invalidates(ac, stored):
    key = _key(ac)
    ac.save_cached_config(key, *_tuned_configs())
    cache_file = ac._get_cache_file_path(key)
    with open(cache_file, encoding = "utf-8") as f:
        data = json.load(f)
    data["device_capability"] = stored
    with open(cache_file, "w", encoding = "utf-8") as f:
        json.dump(data, f)
    assert ac.load_cached_config(key) is None
    assert not os.path.exists(cache_file)
