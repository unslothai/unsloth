# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""native_fsdp2_unsharded restores each unit's exact reshard-after-forward setting: int values
(the setter takes only bool) and torch 2.6 / 2.7 (no setter). One-rank gloo group on the CPU."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402


@pytest.fixture
def sharded(tmp_path, monkeypatch):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    native = _load_native()
    dist.init_process_group(
        "gloo", init_method = f"file://{tmp_path / 'store'}", rank = 0, world_size = 1
    )
    try:
        mesh = init_device_mesh("cpu", (1,))
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
        for block in model:
            fully_shard(block, mesh = mesh, reshard_after_forward = True)
            block._unsloth_reshard_after_forward = True
        fully_shard(model, mesh = mesh)
        model._unsloth_reshard_after_forward = False
        setattr(model, native._NATIVE_FLAG, True)
        model(torch.randn(1, 4))  # lazy init
        yield native, model
    finally:
        dist.destroy_process_group()


def _infos(model):
    return [m._get_fsdp_state()._fsdp_param_group.post_forward_mesh_info for m in model]


def _gathered(module):
    from torch.distributed.tensor import DTensor
    return not isinstance(module.weight, DTensor)


def _generate_like(native, model):
    before = _infos(model)
    with native.native_fsdp2_unsharded(model):
        model(torch.randn(1, 4))
        assert _gathered(model[0]) and _gathered(model[1])
    assert not _gathered(model[0]) and not _gathered(model[1])
    after = _infos(model)
    assert all(a is b for a, b in zip(after, before)), (before, after)
    assert all(info is not None for info in before)


def test_restores_the_exact_setting(sharded):
    native, model = sharded
    _generate_like(native, model)


def test_int_reshard_after_forward_is_restored(sharded):
    native, model = sharded
    for block in model:
        block._unsloth_reshard_after_forward = (
            2  # what shard_model records for FSDP2Config(... = 2)
        )
    _generate_like(native, model)


def test_torch_without_set_reshard_after_forward(sharded, monkeypatch):
    from torch.distributed.fsdp import FSDPModule

    native, model = sharded
    if hasattr(FSDPModule, "set_reshard_after_forward"):
        monkeypatch.delattr(FSDPModule, "set_reshard_after_forward")  # torch 2.6 / 2.7
    _generate_like(native, model)
