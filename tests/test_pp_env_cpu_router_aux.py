# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU backend pipeline step: the MoE router aux loss is weighted the way the trainer's own compute_loss
weights it in one process (stock TRL SFTTrainer: its coefficient x aux / accumulation steps), not added at
full weight on every accumulation step (#17054)."""

import functools
import types

import pytest
import torch

from unsloth.models import pp_env


class _Args:
    device = torch.device("cpu")
    gradient_accumulation_steps = 4
    loss_type = "nll"


class _Accelerator:
    scaler = None

    def unwrap_model(self, model):
        return model


def _trl_1_15_compute_loss(self, *a, **k):
    # The attributes TRL >= 1.15 reads to add coef x aux / accumulation steps itself.
    raise AssertionError((self.router_aux_loss_coef, self.current_gradient_accumulation_steps))


def _trl_1_14_compute_loss(self, *a, **k):
    # TRL 1.14 sets router_aux_loss_coef but leaves the term to the model forward.
    raise AssertionError(self.aux_loss_enabled)


def _trainer(
    stage,
    zoo_wrapped = True,
    aux_loss_enabled = True,
    coef = 0.5,
    current_steps = 4,
    trl = _trl_1_15_compute_loss,
):
    @functools.wraps(trl)
    def compute_loss(self, *a, **k):
        raise AssertionError("the pipeline step never calls compute_loss")

    if zoo_wrapped:
        compute_loss._unsloth_cpu = True  # unsloth_zoo.cpu.trainer._patch_sft_compute_loss
    cls = type(
        "CpuSFTTrainer", (), {"compute_loss": compute_loss, "_prepare_inputs": lambda self, x: x}
    )
    trainer = cls()
    trainer.args = _Args()
    trainer.accelerator = _Accelerator()
    trainer.optimizer = None
    trainer.aux_loss_enabled = aux_loss_enabled
    trainer.router_aux_loss_coef = coef
    trainer.current_gradient_accumulation_steps = current_steps
    trainer.__dict__["_unsloth_pipeline_parallel"] = stage
    return trainer


class _Stage:
    def __init__(self, moe = True):
        self.ctx = types.SimpleNamespace(moe = object() if moe else None, trainer_aux = None)
        self.seen = []

    def step(self, input_ids, labels, **kwargs):
        self.seen.append(self.ctx.trainer_aux)
        return torch.tensor(1.0)

    def eval_loss(self, input_ids, labels, **kwargs):
        self.seen.append(self.ctx.trainer_aux)
        return torch.tensor(1.0)


@pytest.fixture
def plain_helpers(monkeypatch):
    monkeypatch.setattr(pp_env, "_note_trained", lambda stage: None)
    monkeypatch.setattr(
        pp_env, "_batch", lambda stage, inputs: (inputs["input_ids"], inputs["labels"], {})
    )
    monkeypatch.setattr(pp_env, "_global_items", lambda trainer, stage, n: n)
    monkeypatch.setattr(pp_env, "_first_stage_batch", lambda stage, i, l, e, n: (i, l, e, n))
    monkeypatch.setattr(pp_env, "resolve_microbatches", lambda stage, rows, train = True: (1, 0))
    monkeypatch.setattr(pp_env, "_refuse_split_router_aux", lambda stage, m: None)
    monkeypatch.setattr(pp_env, "_pad_rows", lambda i, l, e, pad: (i, l, e))
    monkeypatch.setattr(pp_env, "_eval_items", lambda trainer, inputs, labels: 6)


def _batch():
    return {
        "input_ids": torch.ones(2, 4, dtype = torch.long),
        "labels": torch.ones(2, 4, dtype = torch.long),
    }


def test_stage_takes_trl_coefficient_over_accumulation_steps(plain_helpers):
    stage = _Stage()
    model = torch.nn.Linear(2, 2)
    pp_env._pp_training_step(_trainer(stage), model, _batch(), num_items_in_batch = 6)
    assert stage.seen == [(0.5, 4.0)]
    assert stage.ctx.trainer_aux is None


def test_the_last_accumulation_window_divides_by_its_own_length(plain_helpers):
    # A shorter final window: TRL divides by current_gradient_accumulation_steps, not the configured GA.
    stage = _Stage()
    trainer = _trainer(stage, current_steps = 3)
    pp_env._pp_training_step(trainer, torch.nn.Linear(2, 2), _batch(), num_items_in_batch = 6)
    assert stage.seen == [(0.5, 3.0)]


def test_without_a_window_token_count_the_term_is_added_once(plain_helpers):
    # TRL adds the aux term undivided then; the step's loss_scale = 1 / GA averages the steps.
    stage = _Stage()
    pp_env._pp_training_step(
        _trainer(stage), torch.nn.Linear(2, 2), _batch(), num_items_in_batch = None
    )
    assert stage.seen == [(0.5, 1.0)]


def test_cleared_when_the_stage_raises(plain_helpers):
    stage = _Stage()

    def boom(*a, **k):
        raise RuntimeError("stage failed")

    stage.step = boom
    with pytest.raises(RuntimeError, match = "stage failed"):
        pp_env._pp_training_step(
            _trainer(stage), torch.nn.Linear(2, 2), _batch(), num_items_in_batch = 6
        )
    assert stage.ctx.trainer_aux is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(zoo_wrapped = False),
        dict(aux_loss_enabled = False),
        dict(coef = None),
        dict(trl = _trl_1_14_compute_loss),
    ],
    ids = ["gpu_trainer", "aux_off", "no_coef", "trl_1_14"],
)
def test_the_model_forward_keeps_the_term_elsewhere(plain_helpers, kwargs):
    stage = _Stage()
    pp_env._pp_training_step(
        _trainer(stage, **kwargs), torch.nn.Linear(2, 2), _batch(), num_items_in_batch = 6
    )
    assert stage.seen == [None]


def test_dense_models_carry_nothing(plain_helpers):
    stage = _Stage(moe = False)
    pp_env._pp_training_step(_trainer(stage), torch.nn.Linear(2, 2), _batch(), num_items_in_batch = 6)
    assert stage.seen == [None]


def test_chunked_ce_step_leaves_the_term_to_the_forward(plain_helpers):
    stage = _Stage()
    inner = torch.nn.Linear(2, 2)
    inner._unsloth_cpu_chunked_ce = True
    pp_env._pp_training_step(_trainer(stage), inner, _batch(), num_items_in_batch = 6)
    assert stage.seen == [None]


def test_eval_adds_the_trl_term_once(plain_helpers):
    # TRL's prediction step passes no num_items_in_batch, and zoo falls through to TRL under no_grad even
    # where the chunked CE would arm for training.
    stage = _Stage()
    inner = torch.nn.Linear(2, 2)
    inner._unsloth_cpu_chunked_ce = True
    pp_env._pp_prediction_step(_trainer(stage), inner, _batch(), prediction_loss_only = True)
    assert stage.seen == [(0.5, 1.0)]
    assert stage.ctx.trainer_aux is None
