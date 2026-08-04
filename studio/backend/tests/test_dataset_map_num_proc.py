# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``dataset_map_num_proc()``.

``None``, not ``1``, is the disable sentinel: on ``datasets`` >= 4.0 (Studio pins
4.3.0) ``map()`` takes the pool branch for any ``num_proc >= 1``, so ``1`` still
builds a ``Pool(1)``. ``datasets`` 3.x runs ``1`` in-process, so the version
split is asserted per installed release rather than hard-coded.

There is deliberately no CUDA-initialized guard here; see the note in
``dataset_map_num_proc``'s docstring. The XPU guard is pre-existing behaviour and
is covered below so it cannot regress.

The probes are monkeypatched so this runs on any host, GPU or not.
"""

from __future__ import annotations

import sys
import types

import pytest

import utils.hardware.hardware as hw


def _patch_device(
    monkeypatch,
    device,
    *,
    visible_gpus: int = 1,
):
    monkeypatch.setattr(hw, "get_device", lambda: device)
    monkeypatch.setattr(hw, "get_visible_gpu_count", lambda: visible_gpus)


def _patch_runtime(monkeypatch, name, *, is_initialized):
    """Install a fake ``torch.<name>`` whose is_initialized() we control.

    ``is_initialized`` may be a bool or a callable that raises, to model a probe
    that fails rather than answering.
    """
    import torch

    if callable(is_initialized):
        probe = is_initialized
    else:
        probe = lambda: is_initialized  # noqa: E731

    monkeypatch.setattr(torch, name, types.SimpleNamespace(is_initialized = probe), raising = False)


# ---------- CUDA: initialization must NOT disable workers ----------


def test_dataset_map_num_proc_parallelizes_on_initialized_cuda(monkeypatch):
    """An initialized CUDA context must not disable dataset workers.

    Forking after torch.cuda._lazy_init() is a real hazard in general, but the
    map child only runs the tokenizer and never touches CUDA. 300 forced-fork
    map() runs on an initialized context produced no failures, and since
    detect_hardware() always initializes CUDA, a guard here would serialize
    tokenization for every CUDA training run. Pinned so it is not added back
    without new evidence.
    """
    _patch_device(monkeypatch, hw.DeviceType.CUDA)
    _patch_runtime(monkeypatch, "cuda", is_initialized = True)
    assert hw.dataset_map_num_proc(4) == 4


def test_dataset_map_num_proc_cuda_respects_multi_gpu_cap(monkeypatch):
    # CUDA still routes through safe_num_proc, which caps to 4 on multi-GPU.
    _patch_device(monkeypatch, hw.DeviceType.CUDA)
    _patch_runtime(monkeypatch, "cuda", is_initialized = True)
    monkeypatch.setattr(hw, "get_visible_gpu_count", lambda: 2)
    assert hw.dataset_map_num_proc(16) == 4


# ---------- XPU (regression guard: the pre-existing behaviour must not move) ----------


def test_dataset_map_num_proc_none_after_xpu_init(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = True)
    assert hw.dataset_map_num_proc(4) is None


def test_dataset_map_num_proc_parallel_before_xpu_init(monkeypatch):
    _patch_device(monkeypatch, hw.DeviceType.XPU)
    _patch_runtime(monkeypatch, "xpu", is_initialized = False)
    assert hw.dataset_map_num_proc(4) == 4


# ---------- platform ----------


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_dataset_map_num_proc_none_on_spawn_platforms(monkeypatch, platform):
    # Must be None and never 1: datasets >= 4.0 builds a Pool(1) for num_proc=1.
    monkeypatch.setattr(sys, "platform", platform)
    assert hw.dataset_map_num_proc(4) is None


def test_dataset_map_num_proc_cpu_host_parallelizes(monkeypatch):
    # No accelerator to corrupt, so preprocessing may use workers.
    _patch_device(monkeypatch, hw.DeviceType.CPU)
    assert hw.dataset_map_num_proc(4) == 4


# ---------- the value actually reaches datasets as "no pool" ----------


def test_none_builds_no_pool_but_a_count_does(monkeypatch):
    """The disable sentinel must actually reach ``datasets`` as "no pool".

    This is the property the whole module rests on, so assert it against the
    installed ``datasets`` rather than trusting the docstring.
    """
    datasets = pytest.importorskip("datasets")
    import multiprocess

    pools_built = []
    real_pool = multiprocess.Pool

    def _spy_pool(*args, **kwargs):
        pools_built.append((args, kwargs))
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(multiprocess, "Pool", _spy_pool)
    monkeypatch.setattr(datasets.arrow_dataset, "Pool", _spy_pool, raising = False)

    dataset = datasets.Dataset.from_dict({"text": [f"row {i}" for i in range(8)]})
    _count = lambda batch: {"n": [len(t) for t in batch["text"]]}  # noqa: E731

    mapped = dataset.map(_count, batched = True, num_proc = None)
    assert len(mapped) == 8
    assert pools_built == [], f"Dataset.map built a worker pool: {pools_built}"

    # Control: the spy is live, so the assertion above means something.
    dataset.map(_count, batched = True, num_proc = 2)
    assert len(pools_built) == 1, "Pool spy never fired; the no-pool check is vacuous"


def test_num_proc_one_is_not_a_disable_sentinel():
    """Pin the reason ``dataset_map_num_proc`` returns ``None`` and never ``1``.

    ``datasets`` changed this: 3.x runs ``num_proc=1`` in-process, 4.x (Studio
    pins 4.3.0) takes the pool branch for any ``num_proc >= 1`` and builds a
    ``Pool(1)``. Only ``None`` is in-process on both, so this asserts per
    installed version rather than picking one and hard-coding it.
    """
    datasets = pytest.importorskip("datasets")
    import multiprocess  # noqa: F401
    from packaging.version import Version

    pools_built = []
    real_pool = datasets.arrow_dataset.Pool

    def _spy_pool(*args, **kwargs):
        pools_built.append((args, kwargs))
        return real_pool(*args, **kwargs)

    dataset = datasets.Dataset.from_dict({"text": [f"row {i}" for i in range(8)]})
    _count = lambda batch: {"n": [len(t) for t in batch["text"]]}  # noqa: E731

    original = datasets.arrow_dataset.Pool
    datasets.arrow_dataset.Pool = _spy_pool
    try:
        dataset.map(_count, batched = True, num_proc = None)
        assert pools_built == [], "num_proc=None must always run in-process"

        dataset.map(_count, batched = True, num_proc = 1)
        built_for_one = len(pools_built)
    finally:
        datasets.arrow_dataset.Pool = original

    if Version(datasets.__version__) >= Version("4.0.0"):
        assert built_for_one == 1, (
            f"datasets {datasets.__version__} was expected to build a Pool(1); "
            "if that changed, dataset_map_num_proc's docstring needs updating"
        )
    else:
        assert built_for_one == 0
