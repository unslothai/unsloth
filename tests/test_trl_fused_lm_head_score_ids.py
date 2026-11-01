# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""_UnslothFusedLinearHead against TRL >= 1.17's kernel slot, which takes score_ids after support_ids and returns a
sixth output (score_log_probs). CPU only: the adapter is lifted from rl.py via ast, the zoo kernel is a fake."""

from __future__ import annotations

import ast
import inspect
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

RL_SOURCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "unsloth", "models", "rl.py"
)
OUTPUTS = ("log_probs", "entropy")


def _adapter_class():
    src = open(RL_SOURCE_PATH, encoding = "utf-8").read()
    ns = {"torch": torch, "inspect": inspect, "sys": sys, "os": os}
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == "_UnslothFusedLinearHead":
            exec(compile(ast.Module([node], []), RL_SOURCE_PATH, "exec"), ns)
    return ns["_UnslothFusedLinearHead"]


def _kernel(with_score_ids):
    calls = []

    if with_score_ids:

        def forward(
            ctx,
            hidden,
            weight,
            bias,
            targets,
            temperature,
            chunk_size,
            final_logit_softcapping = None,
            logit_scale = 1.0,
            outputs = (),
            support_ids = None,
            score_ids = None,
        ):
            pass
    else:

        def forward(
            ctx,
            hidden,
            weight,
            bias,
            targets,
            temperature,
            chunk_size,
            final_logit_softcapping = None,
            logit_scale = 1.0,
            outputs = (),
            support_ids = None,
        ):
            pass

    n_out = 6 if with_score_ids else 5

    def apply(*args):
        calls.append(args)
        return ("trl",) * n_out

    kernel = types.SimpleNamespace(forward = staticmethod(forward), apply = apply)
    kernel.__module__ = "fake_trl_utils"
    return kernel, calls


@pytest.fixture
def fake_zoo(monkeypatch):
    calls = []

    def fused_linear_logprobs(hidden, weight, targets, **kwargs):
        calls.append(kwargs)
        log_probs = torch.zeros(hidden.shape[0])
        return (log_probs, torch.ones(hidden.shape[0])) if kwargs["return_entropy"] else log_probs

    fl = types.ModuleType("unsloth_zoo.fused_losses.fused_linear")
    fl.fused_linear_logprobs = fused_linear_logprobs
    fl.fused_linear_rows = lambda *a: 1
    losses = types.ModuleType("unsloth_zoo.fused_losses")
    losses.fused_linear = fl
    zoo = types.ModuleType("unsloth_zoo")
    zoo.fused_losses = losses
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.fused_losses", losses)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.fused_losses.fused_linear", fl)
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    return calls


def _args(chunk_size = 1 << 20):
    hidden = torch.randn(4, 8)
    weight = torch.randn(16, 8)
    targets = torch.randint(0, 16, (4,))
    return [hidden, weight, None, targets, 1.0, chunk_size, None, 1.0, OUTPUTS]


def test_score_ids_slot_is_accepted_and_served_by_the_zoo_kernel(fake_zoo):
    # TRL 1.17 always passes both id tensors, None included, and unpacks six outputs.
    kernel, trl_calls = _kernel(with_score_ids = True)
    head = _adapter_class()(kernel)
    out = head.apply(*_args(), None, None)
    assert len(out) == 6 and out[-1] is None
    assert out[0].shape == (4,) and out[1].shape == (4,)
    assert len(fake_zoo) == 1 and not trl_calls


@pytest.mark.parametrize("which", ["support", "score", "both"])
def test_id_tensors_go_to_trl_with_both_slots(fake_zoo, which):
    kernel, trl_calls = _kernel(with_score_ids = True)
    head = _adapter_class()(kernel)
    support = torch.zeros(4, 2, dtype = torch.long) if which != "score" else None
    score = torch.ones(4, 2, dtype = torch.long) if which != "support" else None
    assert head.apply(*_args(), support, score) == ("trl",) * 6
    assert not fake_zoo
    (call,) = trl_calls
    assert len(call) == 11 and call[9] is support and call[10] is score


def test_kill_switch_passes_both_slots_through(fake_zoo, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    kernel, trl_calls = _kernel(with_score_ids = True)
    head = _adapter_class()(kernel)
    assert head.apply(*_args(), None, None) == ("trl",) * 6
    assert len(trl_calls[0]) == 9


def test_support_ids_only_kernel_keeps_five_outputs(fake_zoo):
    kernel, trl_calls = _kernel(with_score_ids = False)
    head = _adapter_class()(kernel)
    out = head.apply(*_args(), None)
    assert len(out) == 5 and len(fake_zoo) == 1
    support = torch.zeros(4, 2, dtype = torch.long)
    assert head.apply(*_args(), support) == ("trl",) * 5
    assert len(trl_calls[0]) == 10 and trl_calls[0][9] is support


def test_real_trl_fused_head_runs_through_the_adapter(monkeypatch):
    trl_utils = pytest.importorskip("trl.trainer.utils")
    kernel = getattr(trl_utils, "_ChunkedLogProbFunction", None)
    add_fused_lm_head = getattr(trl_utils, "add_fused_lm_head", None)
    if kernel is None or add_fused_lm_head is None:
        pytest.skip("this TRL has no fused LM head")
    # An installed unsloth may already have wrapped it.
    while "function" in getattr(kernel, "__dict__", {}):
        kernel = kernel.function
    if "score_ids" not in inspect.signature(kernel.forward).parameters:
        pytest.skip("this TRL's kernel takes no score_ids")
    from transformers import LlamaConfig, LlamaForCausalLM

    # No zoo kernel: the adapter serves the call from TRL's own kernel.
    monkeypatch.setitem(sys.modules, "unsloth_zoo.fused_losses.fused_linear", None)
    monkeypatch.setattr(trl_utils, "_ChunkedLogProbFunction", _adapter_class()(kernel))
    torch.manual_seed(0)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size = 32,
            hidden_size = 16,
            intermediate_size = 32,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            num_key_value_heads = 2,
        )
    )
    ids = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    with torch.no_grad():
        logits = model(input_ids = ids).logits[:, :-1].float()
    want = logits.log_softmax(-1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    add_fused_lm_head(model, outputs = ("log_probs",))
    out = model(input_ids = ids, labels = ids, fused_lm_head = True)
    torch.testing.assert_close(out.log_probs, want, rtol = 1e-5, atol = 1e-5)
