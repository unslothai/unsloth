# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 CPU offload (UNSLOTH_FSDP2_CPU_OFFLOAD) steps the optimizer on the CPU shards. torchrun
(nproc > 1) and accelerate launch export OMP_NUM_THREADS=1, so that step ran on one core per rank; the
step now gets a share of the rank's cores and the pool is restored afterwards. A user's thread count stays."""

import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402


class _Recording(torch.optim.SGD):
    seen = []

    def step(self, closure = None):
        _Recording.seen.append(torch.get_num_threads())
        return super().step(closure)


def _args(use_cpu = False):
    return types.SimpleNamespace(optim = "adamw_torch_fused", optim_args = None, use_cpu = use_cpu)


@pytest.fixture
def native(monkeypatch):
    module = _load_native()
    before = torch.get_num_threads()
    for name in (
        "UNSLOTH_CPU_THREADS",
        "MKL_NUM_THREADS",
        "ACCELERATE_MIXED_PRECISION",
        "ACCELERATE_USE_CPU",
    ):
        monkeypatch.delenv(name, raising = False)
    # torchrun --nproc-per-node 2 on a host whose rank sees 8 physical cores.
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "test")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    runtime = types.ModuleType("unsloth_zoo.cpu.runtime")
    runtime.cpu_compute_cores = lambda: 8
    runtime._affinity_split_per_rank = lambda ranks: False
    for name in ("unsloth_zoo", "unsloth_zoo.cpu"):
        pkg = types.ModuleType(name)
        pkg.__path__ = []
        monkeypatch.setitem(sys.modules, name, pkg)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.cpu.runtime", runtime)
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    torch.set_num_threads(1)
    _Recording.seen = []
    yield module
    for handle in getattr(module, "_OFFLOAD_STEP_HOOKS", None) or ():
        handle.remove()
    if hasattr(module, "_OFFLOAD_STEP_HOOKS"):
        module._OFFLOAD_STEP_HOOKS = None
    torch.set_num_threads(before)


def _step():
    p = torch.nn.Parameter(torch.ones(4))
    p.grad = torch.ones(4)
    opt = _Recording([p], lr = 0.1)
    opt.step()
    opt.step()
    return opt


def test_offloaded_step_undoes_launcher_single_thread(native):
    native._swap_bnb_optimizer(_args(), cpu_shards = True)
    _step()
    # 8 cores / 2 local ranks; the pool is back to the launcher's 1 after every step.
    assert _Recording.seen == [4, 4]
    assert torch.get_num_threads() == 1


def test_user_thread_counts_are_kept(native, monkeypatch):
    native._swap_bnb_optimizer(_args(), cpu_shards = True)
    torch.set_num_threads(3)  # the user's own torch.set_num_threads
    _step()
    assert _Recording.seen == [3, 3]
    # A user's OMP_NUM_THREADS=1 (no launcher marker) stays 1.
    monkeypatch.delenv("TORCHELASTIC_RUN_ID")
    torch.set_num_threads(1)
    _Recording.seen = []
    _step()
    assert _Recording.seen == [1, 1]
    # UNSLOTH_CPU_THREADS sets the count.
    monkeypatch.setenv("UNSLOTH_CPU_THREADS", "2")
    _Recording.seen = []
    _step()
    assert _Recording.seen == [2, 2]
    assert torch.get_num_threads() == 1


def test_gpu_shards_and_cpu_trainer_are_left_alone(native):
    native._swap_bnb_optimizer(_args(), cpu_shards = False)
    native._swap_bnb_optimizer(
        _args(use_cpu = True), cpu_shards = True
    )  # CPU Trainer: zoo sizes its pool
    _step()
    assert _Recording.seen == [1, 1]
