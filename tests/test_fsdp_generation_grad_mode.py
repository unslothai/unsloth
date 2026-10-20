# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""unsloth#3551: GRPO generation and logprob passes must not run FSDP under inference mode."""

from __future__ import annotations

import pytest
import torch

from unsloth.models._utils import _get_inference_mode_context_manager

_FSDP_ENV = ("ACCELERATE_USE_FSDP", "FSDP_VERSION")


@pytest.fixture
def no_launcher(monkeypatch):
    for name in _FSDP_ENV:
        monkeypatch.delenv(name, raising = False)
    state = pytest.importorskip("accelerate.state")
    monkeypatch.setattr(state.AcceleratorState, "_shared_state", {})
    return monkeypatch


def _mode_inside(model):
    with _get_inference_mode_context_manager(model):
        return torch.is_inference_mode_enabled(), torch.is_grad_enabled()


def test_single_process_keeps_inference_mode(no_launcher):
    assert _mode_inside(torch.nn.Linear(2, 2)) == (True, False)


@pytest.mark.parametrize(
    "name, value, inference",
    [
        ("ACCELERATE_USE_FSDP", "true", False),
        ("ACCELERATE_USE_FSDP", "false", True),
        ("FSDP_VERSION", "2", False),
        ("FSDP_VERSION", "0", True),
    ],
)
def test_the_accelerate_launcher_env_decides(no_launcher, name, value, inference):
    no_launcher.setenv(name, value)
    assert _mode_inside(torch.nn.Linear(2, 2)) == (inference, False)


def test_trainer_args_fsdp_without_the_launcher(no_launcher):
    """`TrainingArguments(fsdp=...)` reaches only the Accelerator, never the env."""
    from accelerate.state import AcceleratorState
    from accelerate.utils import DistributedType

    no_launcher.setattr(
        AcceleratorState, "_shared_state", {"distributed_type": DistributedType.FSDP}
    )
    assert _mode_inside(torch.nn.Linear(2, 2)) == (False, False)


def test_torchao_still_gets_no_grad(no_launcher):
    model = torch.nn.Linear(2, 2)
    model.torchao_config = type("Cfg", (), {"qat_scheme": None})()
    assert _mode_inside(model) == (False, False)


def _fsdp2_generate_then_train(tmp_path, context):
    import torch.distributed as dist

    fully_shard = pytest.importorskip("torch.distributed.fsdp").fully_shard
    dist.init_process_group("gloo", init_method = f"file://{tmp_path}/pg", rank = 0, world_size = 1)
    try:
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
        fully_shard(model[0], reshard_after_forward = False)
        fully_shard(model, reshard_after_forward = False)
        x = torch.randn(3, 8)
        with context(model):
            model(x)
        model(x).sum().backward()
        return model[0].weight.grad
    finally:
        dist.destroy_process_group()


def test_fsdp2_trains_after_a_generation_pass(no_launcher, tmp_path):
    no_launcher.setenv("ACCELERATE_USE_FSDP", "true")
    grad = _fsdp2_generate_then_train(tmp_path, _get_inference_mode_context_manager)
    assert grad is not None


def test_fsdp2_under_inference_mode_is_the_reported_failure(no_launcher, tmp_path):
    """Negative control: what the helper returned before unsloth#3551."""
    with pytest.raises(RuntimeError, match = "[Ii]nference tensor"):
        _fsdp2_generate_then_train(tmp_path, lambda model: torch.inference_mode())
