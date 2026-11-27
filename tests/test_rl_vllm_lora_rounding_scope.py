# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""RLOO, Online DPO and AsyncGRPO score vLLM rollouts too, so they train in vLLM's base-then-LoRA order, as GRPO does
(tests/test_grpo_vllm_lora_rounding.py): only while train / evaluate run, and only when vLLM samples."""

from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch
import unsloth  # noqa: F401

from unsloth.kernels import utils as kernel_utils
from unsloth.models import rl
from unsloth.models import _utils as models_utils


@pytest.fixture
def fused_state(monkeypatch):
    monkeypatch.setattr(kernel_utils, "_LORA_FUSED_PASSES_ENV", None)
    monkeypatch.setattr(kernel_utils, "LORA_FUSED_PASSES", True)
    return kernel_utils


def _trl_class(module_names, class_name):
    for name in module_names:
        module = sys.modules.get(name)
        if module is None:
            try:
                module = importlib.import_module(name)
            except Exception:
                continue
        cls = getattr(module, class_name, None)
        if cls is not None:
            return cls
    return None


_TRAINERS = {
    "rloo": (("trl.trainer.rloo_trainer",), "RLOOTrainer"),
    # TRL 0.29+ keeps Online DPO only under trl.experimental.
    "online_dpo": (
        ("trl.trainer.online_dpo_trainer", "trl.experimental.online_dpo.online_dpo_trainer"),
        "OnlineDPOTrainer",
    ),
}


@pytest.mark.parametrize("name", sorted(_TRAINERS))
def test_patched_trainer_wraps_train_and_evaluate(name):
    cls = _trl_class(*_TRAINERS[name])
    if cls is None:
        pytest.skip(f"this TRL has no {_TRAINERS[name][1]}")
    assert cls.__name__.startswith("Unsloth"), f"{cls} was not patched"
    for method in ("train", "evaluate"):
        assert getattr(getattr(cls, method), "_unsloth_vllm_lora_rounding", False), (name, method)


def test_patched_async_grpo_wraps_train():
    module = rl._import_trl_async_grpo_module()
    if module is None or not hasattr(module, "AsyncGRPOTrainer"):
        pytest.skip("this TRL has no AsyncGRPOTrainer")
    assert getattr(module.AsyncGRPOTrainer.train, "_unsloth_vllm_lora_rounding", False)


class _Recorder:
    def __init__(self, fail = False):
        self.fail, self.seen = fail, []

    def train(self, *args, **kwargs):
        self.seen.append(("train", kernel_utils.LORA_FUSED_PASSES))
        self.evaluate()
        if self.fail:
            raise ValueError("boom")
        return "trained"

    def evaluate(self, *args, **kwargs):
        self.seen.append(("evaluate", kernel_utils.LORA_FUSED_PASSES))
        return {}


def _flag_on_instance(trainer, use_vllm):
    trainer.use_vllm = use_vllm


def _flag_on_args(trainer, use_vllm):
    # TRL 0.22's Online DPO reads args.use_vllm and never sets self.use_vllm.
    trainer.args = types.SimpleNamespace(use_vllm = use_vllm)


@pytest.mark.parametrize("where", [_flag_on_instance, _flag_on_args], ids = ["self", "args"])
@pytest.mark.parametrize("use_vllm", [True, False])
def test_use_vllm_scopes_the_order(fused_state, where, use_vllm):
    cls = type("T", (_Recorder,), {})
    rl._wrap_grpo_vllm_lora_rounding(cls)
    trainer = cls()
    where(trainer, use_vllm)
    assert trainer.train() == "trained"
    assert trainer.seen == [("train", not use_vllm), ("evaluate", not use_vllm)]
    assert fused_state.LORA_FUSED_PASSES is True


def test_always_ignores_a_missing_flag(fused_state):
    cls = type("T", (_Recorder,), {})
    rl._wrap_grpo_vllm_lora_rounding(cls, always = True)
    trainer = cls()
    assert trainer.train() == "trained"
    assert trainer.seen == [("train", False), ("evaluate", False)]
    assert fused_state.LORA_FUSED_PASSES is True


class _Model(torch.nn.Module):
    def __init__(
        self,
        training,
        gc = "unsloth",
    ):
        super().__init__()
        self.train(training)
        self._unsloth_gradient_checkpointing = gc
        self.calls = []

    def for_training(self, use_gradient_checkpointing = True):
        self.calls.append(("for_training", use_gradient_checkpointing))
        self.train(True)

    def for_inference(self):
        self.calls.append(("for_inference",))
        self.train(False)


class _Server:
    aborted = 0

    def abort_all(self):
        _Server.aborted += 1


@pytest.fixture
def fake_async(monkeypatch):
    module = types.ModuleType("fake_async_grpo_trainer")
    seen = []

    class AsyncGRPOTrainer:
        fail = False

        def __init__(self, model, *args, **kwargs):
            self.model = model

        def train(self, *args, **kwargs):
            seen.append(("train", kernel_utils.LORA_FUSED_PASSES, self.model.training))
            self.evaluate()
            if self.fail:
                raise ValueError("boom")
            return "trained"

        def evaluate(self, *args, **kwargs):
            seen.append(("evaluate", kernel_utils.LORA_FUSED_PASSES))
            return {}

    module.create_model_from_path = lambda model_id, *a, **k: model_id
    module.add_fused_lm_head = lambda model, **k: None
    module.AsyncGRPOTrainer = AsyncGRPOTrainer
    monkeypatch.setattr(rl, "_import_trl_async_grpo_module", lambda: module)
    resets = []
    monkeypatch.setattr(models_utils, "_unsloth_reset_stray_compile_cache", resets.append)
    rl.patch_trl_async_grpo()
    return AsyncGRPOTrainer, seen, resets


def _bare(
    cls,
    model,
    server = None,
):
    # The patched __init__ starts a vLLM server; these checks need only what train reads.
    trainer = object.__new__(cls)
    trainer.model = model
    trainer.args = types.SimpleNamespace(gradient_checkpointing = True)
    if server is not None:
        trainer._unsloth_vllm_server = server
    return trainer


@pytest.mark.parametrize("training", [True, False])
def test_async_grpo_train_rounds_as_vllm_and_enters_training_mode(
    fused_state, fake_async, training
):
    cls, seen, resets = fake_async
    assert getattr(cls.train, "_unsloth_vllm_lora_rounding", False)
    model = _Model(training)
    trainer = _bare(cls, model, server = _Server())
    aborted = _Server.aborted
    assert trainer.train() == "trained"
    # vLLM order inside train (and the evaluate it runs), in training mode with the model's own GC setting.
    assert seen == [("train", False, True), ("evaluate", False)]
    assert model.calls[0] == ("for_training", "unsloth")
    assert resets == [trainer]
    assert _Server.aborted == aborted + 1
    # The previous mode and the fused order come back afterwards.
    assert model.training is training
    assert model.calls[1:] == ([] if training else [("for_inference",)])
    assert fused_state.LORA_FUSED_PASSES is True
    seen.clear()
    trainer.evaluate()
    assert seen == [("evaluate", False)]
    assert fused_state.LORA_FUSED_PASSES is True


@pytest.mark.parametrize("training", [True, False])
def test_async_grpo_train_restores_after_failure(fused_state, fake_async, training):
    cls, seen, _ = fake_async
    cls.fail = True
    model = _Model(training)
    trainer = _bare(cls, model, server = _Server())
    aborted = _Server.aborted
    with pytest.raises(ValueError):
        trainer.train()
    assert _Server.aborted == aborted + 1
    assert model.training is training
    assert model.calls[1:] == ([] if training else [("for_inference",)])
    assert fused_state.LORA_FUSED_PASSES is True


def test_predict_is_scoped_too(fused_state):
    class T:
        use_vllm = True

        def predict(self, *args, **kwargs):
            return kernel_utils.LORA_FUSED_PASSES

    rl._wrap_grpo_vllm_lora_rounding(T)
    assert T().predict() is False
    assert fused_state.LORA_FUSED_PASSES is True


def test_async_grpo_model_without_for_training(fused_state, fake_async):
    cls, seen, _ = fake_async
    model = torch.nn.Linear(2, 2)
    trainer = _bare(cls, model)
    assert trainer.train() == "trained"
    assert seen[0] == ("train", False, True)
    assert fused_state.LORA_FUSED_PASSES is True
