# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Evaluation under auto padding-free (#3470).

1. compute_metrics / preprocess_logits_for_metrics must block padding-free, as UNSLOTH_RETURN_LOGITS=1
   does: the generated trainer only sets that flag after the padding-free decision, so the metrics
   function used to receive one packed row per batch instead of one row per example.
2. A packed eval batch runs without a KV cache: transformers skips its packed-sequence mask once a
   cache exists, and outside train() for_inference turns use_cache back on, so evaluate() let every
   packed example attend to the ones before it (Gemma 3 270M eval_loss 3.82 instead of 2.37).
"""

from __future__ import annotations

import ast
import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

RL_SOURCE = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl.py"


def _prediction_step():
    tree = ast.parse(RL_SOURCE.read_text(encoding = "utf-8"))
    patch_rl = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "PatchRL")
    step = next(
        n
        for n in ast.walk(patch_rl)
        if isinstance(n, ast.FunctionDef) and n.name == "unsloth_prediction_step"
    )
    namespace = {"os": os, "torch": torch, "nested_detach": lambda x: x}
    exec(compile(ast.Module(body = [step], type_ignores = []), str(RL_SOURCE), "exec"), namespace)
    return namespace["unsloth_prediction_step"]


class _Trainer:
    label_names = ["labels"]
    can_return_loss = False
    args = SimpleNamespace(device = "cpu", past_index = -1)

    def __init__(self):
        self.model = SimpleNamespace()
        self.seen = []

    def _prepare_inputs(self, inputs):
        return inputs

    def compute_loss_context_manager(self):
        return contextlib.nullcontext()

    def _get_num_items_in_batch(self, batches, device):
        return None

    def compute_loss(self, model, inputs, **kwargs):
        self.seen.append(dict(inputs))
        return torch.tensor(1.0), (None, torch.zeros(1, 2))


@pytest.mark.parametrize("prediction_loss_only", [True, False])
def test_a_packed_eval_batch_runs_without_a_cache(prediction_loss_only):
    trainer = _Trainer()
    inputs = {"labels": torch.zeros(1, 4), "packed_seq_lengths": torch.tensor([2, 2])}
    _prediction_step()(trainer, trainer.model, inputs, prediction_loss_only, None)
    assert trainer.seen[0]["use_cache"] is False
    assert "use_cache" not in inputs, "the collated batch was mutated"


def test_a_padded_eval_batch_is_left_alone():
    trainer = _Trainer()
    _prediction_step()(trainer, trainer.model, {"labels": torch.zeros(1, 4)}, False, None)
    assert "use_cache" not in trainer.seen[0]


def test_an_explicit_use_cache_wins():
    trainer = _Trainer()
    inputs = {
        "labels": torch.zeros(1, 4),
        "packed_seq_lengths": torch.tensor([4]),
        "use_cache": True,
    }
    _prediction_step()(trainer, trainer.model, inputs, False, None)
    assert trainer.seen[0]["use_cache"] is True


@pytest.fixture
def _patched(monkeypatch):
    import unsloth

    if getattr(unsloth, "DEVICE_TYPE", None) == "mlx":
        pytest.skip("unsloth.trainer is the MLX shim, where padding-free does not apply")
    import unsloth.trainer as trainer_module

    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    injected = []
    monkeypatch.setattr(
        trainer_module, "enable_padding_free_metadata", lambda m, t: injected.append(m)
    )
    monkeypatch.setattr(trainer_module, "enable_sample_packing", lambda m, t: None)

    class _StubSFTTrainer:
        # The generated UnslothSFTTrainer sets UNSLOTH_RETURN_LOGITS here, after the wrapper decided.
        def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            eval_dataset = None,
            processing_class = None,
            compute_loss_func = None,
            compute_metrics = None,
            preprocess_logits_for_metrics = None,
            **kwargs,
        ):
            self.model = model
            self.args = args
            if compute_metrics is not None or preprocess_logits_for_metrics is not None:
                os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    return module, injected


class _Forward(torch.nn.Module):
    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return None


@pytest.mark.parametrize(
    "passed",
    [
        {"compute_metrics": lambda p: {}},
        {"preprocess_logits_for_metrics": lambda logits, labels: logits},
        "positional",
    ],
)
def test_metrics_block_padding_free(_patched, passed):
    module, injected = _patched
    config = SimpleNamespace(packing = False, padding_free = None, max_length = 512)
    if passed == "positional":
        module.SFTTrainer(_Forward(), config, None, None, None, None, None, lambda p: {})
    else:
        module.SFTTrainer(model = _Forward(), args = config, **passed)
    assert config.padding_free is False
    assert injected == []


def test_loss_only_eval_keeps_padding_free(_patched):
    module, injected = _patched
    config = SimpleNamespace(packing = False, padding_free = None, max_length = 512)
    module.SFTTrainer(model = _Forward(), args = config)
    assert config.padding_free is True
    assert len(injected) == 1
