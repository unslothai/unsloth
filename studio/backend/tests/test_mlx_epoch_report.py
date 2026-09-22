# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import importlib.util
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset

_BACKEND = Path(__file__).resolve().parents[1]


def _load_worker_module():
    spec = importlib.util.spec_from_file_location(
        "mlx_epoch_report_worker_under_test",
        _BACKEND / "core" / "training" / "worker.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_worker = _load_worker_module()


class _StubMLXTrainer:
    def __init__(self):
        self._step_callbacks = []
        self.state = SimpleNamespace(epoch = 0.0)

    def add_step_callback(self, fn):
        self._step_callbacks.append(fn)

    def train(self, step, total_steps, state_epoch):
        self.state.epoch = state_epoch
        for cb in self._step_callbacks:
            cb(step, total_steps, 1.25, 2e-4, 900.0, 3.5, 12.0, 4096, 0.7)


def _step_callback_block():
    tree = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training)))
    body = tree.body[0].body
    starts = [i for i, n in enumerate(body) if ast.unparse(n) == "start_step = 0"]
    ends = [i for i, n in enumerate(body) if ast.unparse(n) == "trainer.add_step_callback(_on_step)"]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0]
    return body[starts[0] : ends[0] + 1]


def _run_step_callback(rows, step, total_steps, state_epoch, is_vlm = False, **config):
    block = compile(ast.Module(body = _step_callback_block(), type_ignores = []), "<on_step>", "exec")
    events = []
    trainer = _StubMLXTrainer()
    namespace = dict(vars(_worker))
    namespace.update(
        {
            "config": config,
            "dataset": Dataset.from_dict({"text": ["row"] * rows}),
            "trainer": trainer,
            "resume_from_checkpoint": None,
            "batch_size": config["batch_size"],
            "grad_accum": config["gradient_accumulation_steps"],
            "num_epochs": config["num_epochs"],
            "mlx_world_size": 1,
            "is_vlm": is_vlm,
            "wandb_run": None,
            "tb_writer": None,
            "_send": lambda event_type, **kw: events.append((event_type, kw)),
        }
    )
    exec(block, namespace)
    trainer.train(step, total_steps, state_epoch)
    return [kw for kind, kw in events if kind == "progress"]


_DEFAULT_MAX_STEPS_CONFIG = dict(
    batch_size = 4,
    gradient_accumulation_steps = 8,
    num_epochs = 3,
    max_steps = 60,
)


@pytest.mark.parametrize(
    "rows,step,total_steps,state_epoch",
    [(7680, 60, 60, 0.25), (100, 60, 60, 15.0), (100, 12, 12, 3.0)],
    ids = ["bounded-corpus", "hundred-row-dataset", "epoch-mode-ragged-epoch"],
)
def test_mlx_step_callback_reports_the_trainer_epoch(rows, step, total_steps, state_epoch):
    progress = _run_step_callback(rows, step, total_steps, state_epoch, **_DEFAULT_MAX_STEPS_CONFIG)

    assert len(progress) == 1
    assert progress[0]["step"] == step and progress[0]["total_steps"] == total_steps
    assert progress[0]["epoch"] == state_epoch


def test_mlx_vlm_step_callback_counts_streamed_passes():
    progress = _run_step_callback(
        11,
        6,
        6,
        1.0,
        is_vlm = True,
        batch_size = 2,
        gradient_accumulation_steps = 2,
        num_epochs = 3,
        max_steps = 6,
    )

    assert progress[0]["epoch"] == 2.0


def test_mlx_step_callback_keeps_the_zero_total_guard():
    progress = _run_step_callback(7680, 0, 0, 0.0, **_DEFAULT_MAX_STEPS_CONFIG)

    assert progress[0]["epoch"] == 0


def test_epoch_after_steps_counts_micro_batches_per_pass(monkeypatch):
    from core.training.dataset_bounds import (
        WORLD_SIZE_ENV_FILES,
        WORLD_SIZE_ENV_VARS,
        epoch_after_steps,
    )

    for name in WORLD_SIZE_ENV_VARS + WORLD_SIZE_ENV_FILES:
        monkeypatch.delenv(name, raising = False)

    assert epoch_after_steps(60, 4, 8, 7680) == 0.25
    assert epoch_after_steps(20, 4, 8, 7680) == 0.08
    assert epoch_after_steps(6, 2, 2, 11) == 2.0
    assert epoch_after_steps(6, 2, 2, 10) == 2.4
    assert epoch_after_steps(60, 4, 8, 7680, world_size = 4) == 1.0
    assert epoch_after_steps(60, 4, 8, 7680, world_size = "not a number") == 0.25
    assert epoch_after_steps(0, 4, 8, 7680) == 0.0
    assert epoch_after_steps(60, 4, 8, 0) == 0.0
    assert epoch_after_steps(60, 4, 8, None) == 0.0
