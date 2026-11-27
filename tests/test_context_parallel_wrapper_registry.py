# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp


def _torch_cp_registry():
    try:
        from torch.distributed.tensor.experimental._context_parallel import _attention
    except ImportError:
        from torch.distributed.tensor.experimental import _attention
    return getattr(_attention, "_replaced_functions", None)


@pytest.fixture
def gloo_cp_manager(monkeypatch, tmp_path):
    import torch.distributed as dist

    if _torch_cp_registry() is None or not cp._torch_supports_context_parallel():
        pytest.skip("this torch has no CP wrapper registry")
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group(
        "gloo", init_method = f"file://{tmp_path / 'rendezvous'}", rank = 0, world_size = 1
    )
    monkeypatch.setattr(cp, "DEVICE_TYPE_TORCH", "cpu")
    try:
        yield cp.ContextParallelManager(1)
    finally:
        dist.destroy_process_group()


def _cp_step(manager):
    with manager.apply({"input_ids": torch.ones(1, 8, dtype = torch.long)}):
        pass


def test_ring_steps_do_not_grow_the_cp_wrapper_registry(gloo_cp_manager):
    import torch.nn.functional as F

    registry = _torch_cp_registry()
    original = F.scaled_dot_product_attention
    before = dict(registry)
    seen = []
    with gloo_cp_manager.apply({"input_ids": torch.ones(1, 8, dtype = torch.long)}):
        seen.append(F.scaled_dot_product_attention in registry)  # the real torch wrapper engaged
    for _ in range(50):
        _cp_step(gloo_cp_manager)
    assert seen == [True]
    assert registry == before
    assert F.scaled_dot_product_attention is original


def test_cp_wrappers_created_outside_unsloth_are_kept(gloo_cp_manager):
    import torch.nn.functional as F

    registry = _torch_cp_registry()
    original = F.scaled_dot_product_attention
    before = set(registry)
    foreign = lambda *a, **k: None
    registry[foreign] = ("scaled_dot_product_attention", original)
    try:
        buffer = torch.ones(1, 8)
        with cp.context_parallel(gloo_cp_manager.mesh, buffers = [buffer], buffer_seq_dims = [1]):
            outer = F.scaled_dot_product_attention
            assert outer in registry
            _cp_step(gloo_cp_manager)  # nested inside a user's own CP context
            assert F.scaled_dot_product_attention is outer and outer in registry
            with gloo_cp_manager.apply({"input_ids": torch.ones(1, 8, dtype = torch.long)}):
                _cp_step(gloo_cp_manager)  # re-entrant Unsloth step
            assert F.scaled_dot_product_attention is outer and outer in registry
        assert foreign in registry and outer in registry
    finally:
        for wrapper in set(registry) - before:
            registry.pop(wrapper, None)
        F.scaled_dot_product_attention = original
