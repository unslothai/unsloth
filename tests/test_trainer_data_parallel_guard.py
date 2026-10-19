# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""An Unsloth LoRA model under a plain Trainer on two GPUs trains on one (Kaggle T4x2 Whisper,
EmbeddingGemma). Extracted with ast so no GPU is needed."""

import ast
import os

import pytest
import torch
from transformers.training_args import ParallelMode

SOURCE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models", "_utils.py"
)


def _guard():
    src = open(SOURCE, encoding = "utf-8").read()
    node = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_keep_unsloth_models_off_data_parallel"
    )
    ns = {}
    exec(ast.get_source_segment(src, node), ns)
    return ns["_keep_unsloth_models_off_data_parallel"]


class _Args:
    def __init__(
        self,
        n_gpu,
        mode = ParallelMode.NOT_DISTRIBUTED,
        per_device = 2,
    ):
        self._n_gpu = n_gpu
        self.parallel_mode = mode
        self.per_device_train_batch_size = per_device

    @property
    def n_gpu(self):
        return self._n_gpu

    @property
    def train_batch_size(self):
        return self.per_device_train_batch_size * max(1, self.n_gpu)


def _wrapped(marked = True):
    inner = torch.nn.Linear(2, 2)
    if marked:
        inner._unsloth_disable_data_parallel = True
    return torch.nn.Sequential(inner)  # the marker sits inside, as in a SentenceTransformer


def test_a_marked_model_trains_on_one_gpu():
    args = _Args(2)
    assert _guard()(_wrapped(), args) is True
    assert args.n_gpu == 1 and args.train_batch_size == 2


@pytest.mark.parametrize(
    "model, args",
    [
        (_wrapped(marked = False), _Args(2)),  # not an Unsloth LoRA model: DataParallel as before
        (_wrapped(), _Args(1)),  # one GPU already
        (_wrapped(), _Args(2, ParallelMode.DISTRIBUTED)),  # DDP owns placement
        (_wrapped(), None),
        (None, _Args(2)),
    ],
)
def test_everything_else_is_left_alone(model, args):
    before = None if args is None else args.n_gpu
    assert _guard()(model, args) is False
    if args is not None:
        assert args.n_gpu == before


def _init_patch():
    import functools

    src = open(SOURCE, encoding = "utf-8").read()
    ns = {"functools": functools}
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef) and n.name in (
            "_keep_unsloth_models_off_data_parallel",
            "_patch_trainer_init_data_parallel",
        ):
            exec(ast.get_source_segment(src, n), ns)
    return ns["_patch_trainer_init_data_parallel"]


class _FakeTrainer:
    def __init__(
        self,
        model = None,
        args = None,
    ):
        self.model, self.args = model, args
        self._train_batch_size = args.train_batch_size


@pytest.mark.parametrize("positional", [False, True])
def test_the_marker_alone_guards_trainer_init(positional):
    # Fast encoders never call patch_gradient_accumulation_fix; marking must be enough.
    Trainer = type("Trainer", (_FakeTrainer,), {})
    _init_patch()(Trainer)
    _init_patch()(Trainer)
    args = _Args(2)
    trainer = Trainer(_wrapped(), args) if positional else Trainer(model = _wrapped(), args = args)
    assert args.n_gpu == 1 and trainer._train_batch_size == 2
    unmarked = _Args(2)
    assert Trainer(model = _wrapped(marked = False), args = unmarked)._train_batch_size == 4


def test_marking_installs_the_init_guard():
    src = open(SOURCE, encoding = "utf-8").read()
    fn = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "_patch_transformers_trainer_data_parallel"
    )
    assert "_patch_trainer_init_data_parallel(Trainer)" in ast.get_source_segment(src, fn)
