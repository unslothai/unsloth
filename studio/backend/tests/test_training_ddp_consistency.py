# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise trainer helpers without importing the GPU-heavy ML stack."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from core.training.dataset_bounds import world_size_from_env
from core.training.training import TrainingBackend
from utils.hardware import hardware


def _trainer_method(name):
    path = Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    trainer = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnslothTrainer"
    )
    method = next(
        node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"math": math, "world_size_from_env": world_size_from_env, "logger": MagicMock()}
    exec(compile(ast.Module([method], []), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize(
    "world_size, samples, expected", [(1, 100, 39), (4, 100, 12), (4, 101, 12)]
)
def test_epoch_steps_account_for_distributed_sampler(monkeypatch, world_size, samples, expected):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    assert _trainer_method("_calculate_total_steps")(None, samples, 2, 4, 3, 0) == expected


def test_explicit_max_steps_is_not_divided(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert _trainer_method("_calculate_total_steps")(None, 100, 2, 4, 3, 17) == 17


@pytest.mark.parametrize("is_world_process_zero", [True, False])
@pytest.mark.parametrize("should_stop, save_on_stop", [(False, True), (True, True), (True, False)])
def test_finalization_only_writing_rank_mutates_auxiliary_files(
    is_world_process_zero, should_stop, save_on_stop
):
    trainer = MagicMock()
    trainer.is_world_process_zero.return_value = is_world_process_zero
    # Saving auxiliary files is independent of save strategy/should_save.
    trainer.args = SimpleNamespace(should_save = False, save_strategy = "no")
    owner = SimpleNamespace(
        trainer = trainer,
        tokenizer = MagicMock(),
        _patch_adapter_config = MagicMock(),
        _update_progress = MagicMock(),
        should_stop = should_stop,
        save_on_stop = save_on_stop,
    )
    _trainer_method("_finalize_training")(owner, "/tmp/ddp-output")
    saving = not should_stop or save_on_stop
    assert trainer.save_model.call_count == int(saving)
    assert owner.tokenizer.save_pretrained.call_count == int(saving and is_world_process_zero)
    assert owner._patch_adapter_config.call_count == int(saving and is_world_process_zero)
    assert trainer._save_checkpoint.call_count == int(should_stop and save_on_stop)


@pytest.mark.parametrize(
    "device, rocm",
    [
        (hardware.DeviceType.CPU, False),
        (hardware.DeviceType.XPU, False),
        (hardware.DeviceType.CUDA, True),
    ],
)
def test_unsupported_ddp_rejected_before_vram_hook(monkeypatch, device, rocm):
    monkeypatch.setattr(hardware, "DEVICE", device)
    monkeypatch.setattr(hardware, "IS_ROCM", rocm)
    hook = MagicMock()
    with pytest.raises(ValueError, match = "NVIDIA CUDA"):
        TrainingBackend().start_training(
            job_id = "ddp-unsupported",
            before_spawn = hook,
            model_name = "unsloth/test",
            training_type = "LoRA/QLoRA",
            parallelism_mode = "ddp",
            gpu_ids = [0, 1],
        )
    hook.assert_not_called()
