# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

from __future__ import annotations

import os

import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl


class _Accelerator:
    def __init__(self, device = "cpu"):
        from accelerate.utils import DistributedType

        self.device = torch.device(device)
        self.distributed_type = DistributedType.NO
        self.calls = []

    def prepare_model(
        self,
        model,
        device_placement = None,
        evaluation_mode = False,
    ):
        self.calls.append((device_placement, evaluation_mode))
        return model


def _frozen(module):
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def test_gkd_init_routes_teacher_through_placement():
    import ast
    import importlib.util

    # Read TRL's file: importing unsloth replaces GKDTrainer.__init__ with a wrapper.
    for name in ("trl.trainer.gkd_trainer", "trl.experimental.gkd.gkd_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            spec = None
        if spec is not None and spec.origin:
            break
    text = open(spec.origin).read()
    cls = next(
        n
        for n in ast.walk(ast.parse(text))
        if isinstance(n, ast.ClassDef) and n.name == "GKDTrainer"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    source = ast.get_source_segment(text, init)
    assert "self.accelerator.prepare_model(teacher_model, evaluation_mode=True)" in source
    edited = rl.gkd_trainer_prepare_teacher("__init__", source)
    assert "_unsloth_prepare_frozen_teacher(self.accelerator, teacher_model)" in edited
    assert "prepare_model(teacher_model" not in edited
    assert rl.gkd_trainer_prepare_teacher("compute_loss", source) == source
    assert rl.gkd_trainer_prepare_teacher in rl.RL_FUNCTIONS["gkd_trainer"]
    pre = "\n".join(rl.RL_PRE_ITEMS["gkd_trainer"])
    assert "def _unsloth_prepare_frozen_teacher(" in pre


def test_one_card_and_trainable_teachers_keep_accelerate_call():
    for teacher in (_frozen(torch.nn.Linear(4, 4)), torch.nn.Linear(4, 4), torch.nn.Module()):
        accelerator = _Accelerator()
        assert rl._unsloth_prepare_frozen_teacher(accelerator, teacher) is teacher
        assert accelerator.calls == [(None, True)]


def test_kill_switch_keeps_accelerate_call(monkeypatch):
    monkeypatch.setenv("UNSLOTH_KD_TEACHER_PLACEMENT", "0")
    accelerator = _Accelerator("cuda:0")
    teacher = _frozen(torch.nn.Linear(4, 4))
    assert rl._unsloth_prepare_frozen_teacher(accelerator, teacher) is teacher
    assert accelerator.calls == [(None, True)]
