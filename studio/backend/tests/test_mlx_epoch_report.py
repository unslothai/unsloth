# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import ast
import inspect
import textwrap
from types import SimpleNamespace

import core.training.worker as _worker
from datasets import Dataset

_TREE = ast.parse(textwrap.dedent(inspect.getsource(_worker._run_mlx_training)))


class _StubMLXTrainer:
    def __init__(self):
        self._step_callbacks = []
        self.state = SimpleNamespace(epoch = None)

    def add_step_callback(self, fn):
        self._step_callbacks.append(fn)

    def train(self, step, state_epoch):
        self.state.epoch = state_epoch
        for cb in self._step_callbacks:
            cb(step, step, 1.25, 2e-4, 900.0, 3.5, 12.0, 4096, 0.7)


def _step_callback_block():
    body = _TREE.body[0].body
    bound = [
        n
        for n in body
        if ast.unparse(n) == "mlx_kept_row_fraction = [1.0]"
        or (isinstance(n, ast.FunctionDef) and n.name in ("_on_bound", "_slice"))
    ]
    starts = [i for i, n in enumerate(body) if ast.unparse(n) == "start_step = 0"]
    ends = [
        i for i, n in enumerate(body) if ast.unparse(n) == "trainer.add_step_callback(_on_step)"
    ]
    assert len(bound) == 3 and len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0]
    return bound + body[starts[0] : ends[0] + 1]


def _run_step_callback(
    step,
    state_epoch,
    rows = 0,
    max_train_rows = None,
):
    block = compile(ast.Module(body = _step_callback_block(), type_ignores = []), "<on_step>", "exec")
    events = []
    trainer = _StubMLXTrainer()
    namespace = dict(vars(_worker))
    namespace.update(
        {
            "trainer": trainer,
            "resume_from_checkpoint": None,
            "slice_start": None,
            "slice_end": None,
            "mlx_split_names_rows": False,
            "mlx_max_train_rows": max_train_rows,
            "mlx_max_train_rows_seed": 3407,
            "wandb_run": None,
            "tb_writer": None,
            "_send": lambda event_type, **kw: events.append((event_type, kw)),
        }
    )
    exec(block, namespace)
    namespace["_slice"](Dataset.from_dict({"text": ["row"] * rows}))
    trainer.train(step, state_epoch)
    return [kw for kind, kw in events if kind == "progress"]


def test_mlx_step_callback_reports_the_trainer_epoch():
    progress = _run_step_callback(60, 15.3333)

    assert len(progress) == 1
    assert progress[0]["step"] == 60 and progress[0]["total_steps"] == 60
    assert progress[0]["epoch"] == 15.33


def test_mlx_step_callback_reports_zero_before_the_trainer_has_an_epoch():
    assert _run_step_callback(1, None)[0]["epoch"] == 0


def test_mlx_step_callback_counts_a_bounded_run_over_the_whole_dataset():
    progress = _run_step_callback(60, 2.0, rows = 1000, max_train_rows = 250)

    assert progress[0]["epoch"] == 0.5


def test_mlx_training_never_streams():
    keyword_values = [
        n.value for n in ast.walk(_TREE) if isinstance(n, ast.keyword) and n.arg == "streaming"
    ]
    subscript_values = [
        n.value
        for n in ast.walk(_TREE)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Subscript) and ast.unparse(t.slice) == "'streaming'"
            for t in n.targets
        )
    ]
    for value in keyword_values + subscript_values:
        assert isinstance(value, ast.Constant) and value.value is False, ast.unparse(value)
