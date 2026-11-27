# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""On Apple Silicon with MLX, torch.cuda.get_device_properties returns _UnslothDeviceStats. Inductor's cache key
(CacheBase.get_system) reads gcnArchName whenever torch.version.cuda is None, so the stats object must answer it,
or every CPU torch.compile with the inductor backend fails with AttributeError."""

import pytest
import torch

import unsloth


def test_device_stats_answers_the_attributes_inductor_reads():
    stats = unsloth._UnslothDeviceStats("Apple M2 Max", 64 << 30)
    assert stats.name == "Apple M2 Max"
    assert stats.gcnArchName == "Apple M2 Max"
    assert (stats.major, stats.minor) == (0, 0)


def test_inductor_cache_key_accepts_the_mlx_device_stats(monkeypatch):
    codecache = pytest.importorskip("torch._inductor.codecache")
    get_system = getattr(getattr(codecache, "CacheBase", None), "get_system", None)
    get_system = getattr(get_system, "__wrapped__", None)
    if get_system is None:
        pytest.skip("this torch has no CacheBase.get_system")
    stats = unsloth._UnslothDeviceStats("Apple M2 Max", 64 << 30)
    # What _patch_mlx_torch_cuda_compat_api installs on an MLX host.
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device = None: stats)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    system = get_system()
    assert system["device"]["name"] == "Apple M2 Max"
