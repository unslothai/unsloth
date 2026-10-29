# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Frozen BatchNorm / InstanceNorm keep their running stats through LoRA training."""

import copy
import inspect
import pickle

import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
from torch import nn  # noqa: E402

peft = pytest.importorskip("peft")
U = pytest.importorskip("unsloth.models._utils")
vision = pytest.importorskip("unsloth.models.vision")
FastBaseModel = vision.FastBaseModel
TRAIN = U._unsloth_train_if_needed


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 4, 1)
        self.norm = nn.BatchNorm1d(4)
        self.inorm = nn.InstanceNorm1d(4, affine = True, track_running_stats = True)
        self.proj = nn.Linear(4, 4)

    def forward(self, x):
        h = self.inorm(self.norm(self.conv(x)))
        return self.proj(h.transpose(1, 2))


def _lora(modules_to_save = None):
    torch.manual_seed(0)
    cfg = peft.LoraConfig(
        r = 2, lora_alpha = 4, target_modules = ["proj"], modules_to_save = modules_to_save
    )
    return peft.get_peft_model(_Tiny(), cfg)


def _stats(model):
    return {n: b.detach().clone() for n, b in model.named_buffers() if "running" in n}


def _moved(before, after):
    return sorted(n for n in before if not torch.equal(before[n], after[n]))


def _train_step(model):
    x = torch.randn(3, 4, 8) * 5 + 2
    model(x).float().pow(2).mean().backward()


def _norms(model):
    return [
        m
        for m in model.modules()
        if isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm))
    ]


def test_trainer_step_keeps_frozen_norm_stats(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FREEZE_NORM_RUNNING_STATS", raising = False)
    model = _lora()
    assert not any(p.requires_grad for m in _norms(model) for p in m.parameters())
    before = _stats(model)
    TRAIN(model)
    _train_step(model)
    assert model.training
    assert all(not m.training for m in _norms(model))
    assert _moved(before, _stats(model)) == []
    assert any(p.grad is not None for n, p in model.named_parameters() if "lora_" in n)
    FastBaseModel.for_inference(model)
    assert all(not m.training for m in model.modules())
    FastBaseModel.for_training(model)
    model.train()
    _train_step(model)
    assert all(not m.training for m in _norms(model))
    assert _moved(before, _stats(model)) == []


def test_trainable_norm_still_updates(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FREEZE_NORM_RUNNING_STATS", raising = False)
    torch.manual_seed(0)
    model = _Tiny()
    before = _stats(model)
    TRAIN(model)
    _train_step(model)
    assert all(m.training for m in _norms(model))
    assert set(_moved(before, _stats(model))) == set(before)


def test_unfreezing_later_restores_train_mode(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FREEZE_NORM_RUNNING_STATS", raising = False)
    model = _lora()
    TRAIN(model)
    norm = model.base_model.model.norm
    assert not norm.training
    for p in norm.parameters():
        p.requires_grad_(True)
    model.train()
    assert norm.training


def test_modules_to_save_norm_trains(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FREEZE_NORM_RUNNING_STATS", raising = False)
    model = _lora(modules_to_save = ["norm"])
    TRAIN(model)
    wrapper = model.base_model.model.norm
    trained = wrapper.modules_to_save["default"]
    before = trained.running_mean.clone()
    _train_step(model)
    assert trained.training
    assert not torch.equal(before, trained.running_mean)


def test_opt_out_env_restores_old_behaviour(monkeypatch):
    monkeypatch.setenv("UNSLOTH_FREEZE_NORM_RUNNING_STATS", "0")
    model = _lora()
    before = _stats(model)
    TRAIN(model)
    _train_step(model)
    assert all(m.training for m in _norms(model))
    assert set(_moved(before, _stats(model))) == set(before)


def test_eval_state_dict_copy_and_pickle_unaffected(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FREEZE_NORM_RUNNING_STATS", raising = False)
    model = _lora()
    keys = set(model.state_dict())
    TRAIN(model)
    assert set(model.state_dict()) == keys
    model.eval()
    assert all(not m.training for m in model.modules())
    clone = copy.deepcopy(model)
    clone.train()
    assert all(not m.training for m in _norms(clone))
    assert clone.base_model.model.norm.train.args[0] is clone.base_model.model.norm
    restored = pickle.loads(pickle.dumps(clone))
    restored.train()
    assert all(not m.training for m in _norms(restored))


def test_post_patch_model_installs_the_guard():
    src = inspect.getsource(FastBaseModel.post_patch_model)
    assert "_unsloth_freeze_norm_running_stats(model)" in src
