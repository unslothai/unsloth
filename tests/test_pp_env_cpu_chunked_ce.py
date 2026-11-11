# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""CPU pipeline step must arm zoo's chunked CE itself: the schedule never calls compute_loss."""

import pytest
import torch

from unsloth.models import pp_env


class _Args:
    device = torch.device("cpu")
    gradient_accumulation_steps = 1
    loss_type = "nll"


class _Accelerator:
    scaler = None

    def unwrap_model(self, model):
        return model


def _trainer(
    model,
    stage,
    zoo_wrapped = True,
    loss_type = "nll",
):
    def compute_loss(self, *a, **k):
        raise AssertionError("the pipeline step never calls compute_loss")

    if zoo_wrapped:
        compute_loss._unsloth_cpu = (
            True  # what unsloth_zoo.cpu.trainer._patch_sft_compute_loss marks
        )
    cls = type(
        "CpuSFTTrainer", (), {"compute_loss": compute_loss, "_prepare_inputs": lambda self, x: x}
    )
    trainer = cls()
    trainer.args = _Args()
    trainer.args.loss_type = loss_type
    trainer.accelerator = _Accelerator()
    trainer.optimizer = None
    trainer.__dict__["_unsloth_pipeline_parallel"] = stage
    return trainer


class _Stage:
    def __init__(
        self,
        inner,
        fail = False,
    ):
        self.inner, self.fail, self.armed = inner, fail, []

    def step(self, input_ids, labels, **kwargs):
        self.armed.append(bool(getattr(self.inner, "_unsloth_cpu_chunked_ce_active", False)))
        if self.fail:
            raise RuntimeError("stage failed")
        return torch.tensor(1.0)


def _models():
    inner = torch.nn.Linear(
        2, 2
    )  # the causal LM whose forward zoo's patch_chunked_causal_lm_loss replaced
    inner._unsloth_cpu_chunked_ce = True
    peft = torch.nn.Module()
    peft.base_model = torch.nn.Module()
    peft.base_model.model = inner
    return peft, inner


@pytest.fixture
def plain_helpers(monkeypatch):
    monkeypatch.setattr(pp_env, "_note_trained", lambda stage: None)
    monkeypatch.setattr(
        pp_env, "_batch", lambda stage, inputs: (inputs["input_ids"], inputs["labels"], {})
    )
    monkeypatch.setattr(pp_env, "_global_items", lambda trainer, stage, n: n)
    monkeypatch.setattr(pp_env, "_first_stage_batch", lambda stage, i, l, e, n: (i, l, e, n))
    monkeypatch.setattr(pp_env, "resolve_microbatches", lambda stage, rows: (2, 0))
    monkeypatch.setattr(pp_env, "_refuse_split_router_aux", lambda stage, m: None)
    monkeypatch.setattr(pp_env, "_pad_rows", lambda i, l, e, pad: (i, l, e))


def _step(trainer, model):
    batch = {
        "input_ids": torch.ones(2, 4, dtype = torch.long),
        "labels": torch.ones(2, 4, dtype = torch.long),
    }
    return pp_env._pp_training_step(trainer, model, batch, num_items_in_batch = 6)


def test_pp_step_arms_cpu_chunked_ce_like_compute_loss(plain_helpers):
    peft, inner = _models()
    stage = _Stage(inner)
    _step(_trainer(peft, stage), peft)
    assert stage.armed == [True]
    assert inner._unsloth_cpu_chunked_ce_active is False


def test_pp_step_disarms_when_the_stage_raises(plain_helpers):
    peft, inner = _models()
    stage = _Stage(inner, fail = True)
    with pytest.raises(RuntimeError, match = "stage failed"):
        _step(_trainer(peft, stage), peft)
    assert stage.armed == [True] and inner._unsloth_cpu_chunked_ce_active is False


@pytest.mark.parametrize(
    "zoo_wrapped,loss_type", [(False, "nll"), (True, "dft")], ids = ["no_zoo_wrapper", "dft"]
)
def test_pp_step_leaves_chunked_ce_off_where_compute_loss_would(
    plain_helpers, zoo_wrapped, loss_type
):
    # Unwrapped compute_loss or non-nll loss keeps the model's own forward, as zoo does.
    peft, inner = _models()
    stage = _Stage(inner)
    _step(_trainer(peft, stage, zoo_wrapped = zoo_wrapped, loss_type = loss_type), peft)
    assert stage.armed == [False]
    assert not vars(inner).get("_unsloth_cpu_chunked_ce_active", False)
