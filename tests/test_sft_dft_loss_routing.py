# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""SFT loss_type="dft" must not silently train as NLL under the Unsloth compute_loss; ast-loaded, CPU-only."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"


def _trainer_class():
    if importlib.util.find_spec("trl") is None:
        pytest.skip("trl not installed")
    import trl.trainer.utils as trl_utils

    if not hasattr(trl_utils, "add_fused_lm_head"):
        pytest.skip("TRL < 1.15 computes dft through compute_loss_func")
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sft_trainer_compute_loss"
    ]
    namespace = {"inspect": inspect, "RL_FUNCTIONS": None}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), namespace)
    source = namespace["sft_trainer_compute_loss"](
        "compute_loss", "def compute_loss(self):\n    pass\n"
    )

    class Base:
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs = False,
            num_items_in_batch = None,
        ):
            return ("super", num_items_in_batch)

    # Compiled into a class body, as in the generated trainer, so the zero-argument super() resolves.
    generated = {"torch": torch, "Base": Base}
    exec("class Generated(Base):\n" + textwrap.indent(textwrap.dedent(source), "    "), generated)

    class Trainer(generated["Generated"]):
        def __init__(
            self,
            loss_type,
            accepts_loss_kwargs = True,
        ):
            self.args = SimpleNamespace(loss_type = loss_type, average_tokens_across_devices = False)
            self.compute_loss_func = None
            self.label_smoother = None
            self.model_accepts_loss_kwargs = accepts_loss_kwargs
            self.aux_loss_enabled = False

    return Trainer


class _FusedModel:
    def __init__(self):
        self.calls = []
        self.log_probs = torch.tensor([[-0.5, -2.0, 0.0], [-1.0, 0.0, 0.0]], requires_grad = True)
        self.mask = torch.tensor([[True, True, False], [True, False, False]])

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not kwargs.get("fused_lm_head"):
            assert "labels" not in kwargs
            return SimpleNamespace(logits = torch.zeros(2, 3, 5))
        return SimpleNamespace(log_probs = self.log_probs, label_mask = self.mask, aux_loss = None)


def _dft(lp, denom):
    return (-(lp * lp.exp())).sum().item() / denom


def test_dft_runs_on_the_fused_head_not_the_nll_loss():
    Trainer = _trainer_class()
    model = _FusedModel()
    loss = Trainer("dft").compute_loss(model, {"input_ids": None}, num_items_in_batch = 3)
    assert len(model.calls) == 1, "dft was routed to the NLL forward, so it trains as plain NLL"
    assert model.calls[0].get("fused_lm_head") is True
    assert model.calls[0]["use_cache"] is False
    assert loss.item() == pytest.approx(_dft(model.log_probs.detach(), 3))
    loss.backward()
    # The DFT weight exp(log_p) is detached: d/dlp of -lp * sg(exp(lp)) is -exp(lp).
    expected = -model.log_probs.detach().exp() / 3
    assert torch.allclose(model.log_probs.grad, expected)


def test_dft_eval_outputs_carry_logits():
    Trainer = _trainer_class()
    model = _FusedModel()
    loss, outputs = Trainer("dft").compute_loss(
        model, {"input_ids": None, "labels": None}, return_outputs = True, num_items_in_batch = 3
    )
    assert loss.item() == pytest.approx(_dft(model.log_probs.detach(), 3))
    assert outputs.logits.shape == (2, 3, 5)
    assert [bool(c.get("fused_lm_head")) for c in model.calls] == [True, False]


def test_dft_refuses_a_qat_lm_head():
    Trainer = _trainer_class()
    model = _FusedModel()
    head = torch.nn.Linear(4, 5, bias = False)
    head.weight_fake_quantizer = torch.nn.Identity()
    model.get_output_embeddings = lambda: head
    with pytest.raises(NotImplementedError, match = "QAT"):
        Trainer("dft").compute_loss(model, {"input_ids": None}, num_items_in_batch = 3)
    assert model.calls == []
    model.get_output_embeddings = lambda: torch.nn.Linear(4, 5, bias = False)
    Trainer("dft").compute_loss(model, {"input_ids": None}, num_items_in_batch = 3)
    assert len(model.calls) == 1


def test_dft_device_average_ignores_tensor_parallel_ranks():
    Trainer = _trainer_class()
    trainer = Trainer("dft")
    trainer.args.average_tokens_across_devices = True
    trainer.accelerator = SimpleNamespace(num_processes = 4)
    trainer.get_tp_size = lambda: 2
    loss = trainer.compute_loss(_FusedModel(), {"input_ids": None}, num_items_in_batch = 3)
    assert loss.item() == pytest.approx(2 * _dft(_FusedModel().log_probs.detach(), 3))


def test_dft_without_loss_kwargs_is_a_per_micro_batch_mean():
    Trainer = _trainer_class()
    model = _FusedModel()
    loss = Trainer("dft", accepts_loss_kwargs = False).compute_loss(model, {}, num_items_in_batch = 100)
    assert loss.item() == pytest.approx(_dft(model.log_probs.detach(), 3))


def test_nll_keeps_the_unsloth_path():
    Trainer = _trainer_class()
    model = _FusedModel()
    assert Trainer("nll").compute_loss(model, {}, num_items_in_batch = 7) == ("super", 7)
    assert model.calls == []


def test_dft_with_label_smoothing_still_trains_dft():
    Trainer = _trainer_class()
    trainer = Trainer("dft")
    trainer.label_smoother = lambda *args, **kwargs: pytest.fail("dft was trained as smoothed NLL")
    model = _FusedModel()
    loss = trainer.compute_loss(model, {"input_ids": None}, num_items_in_batch = 3)
    assert [bool(c.get("fused_lm_head")) for c in model.calls] == [True]
    assert loss.item() == pytest.approx(_dft(model.log_probs.detach(), 3))


def test_nll_with_label_smoothing_keeps_the_unsloth_path():
    Trainer = _trainer_class()
    trainer = Trainer("nll")
    trainer.label_smoother = object()
    model = _FusedModel()
    assert trainer.compute_loss(model, {}, num_items_in_batch = 7) == ("super", 7)
    assert model.calls == []
