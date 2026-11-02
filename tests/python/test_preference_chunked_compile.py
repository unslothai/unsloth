# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""ORPO / CPO hidden-state scoring traces as one graph (no data-dependent branch, no boolean slice assignment) and
matches TRL's math in fp64, eager and compiled. CPU only: the helper is lifted from source via ast and the zoo
kernels are replaced by dense references, so no package import, model or GPU is needed."""

from __future__ import annotations

import ast
import os
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

SOURCE = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "rl_replacements.py"
VOCAB, HIDDEN, SEQ, PAIRS = 23, 8, 7, 2


def _transform(logits, multiply, divide, softcap):
    if multiply:
        logits = logits * multiply
    if divide:
        logits = logits / divide
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return logits


def _logits(
    hidden,
    weight,
    bias,
    multiply = 0.0,
    divide = 0.0,
    softcap = 0.0,
):
    return _transform(F.linear(hidden, weight, bias), multiply, divide, softcap)


def _chunked(
    hidden,
    weight,
    index,
    return_mean_logits = False,
    lm_head_bias = None,
    logit_scale_multiply = 0.0,
    logit_scale_divide = 0.0,
    logit_softcapping = 0.0,
    **_,
):
    logits = _logits(
        hidden, weight, lm_head_bias, logit_scale_multiply, logit_scale_divide, logit_softcapping
    )
    logps = logits.log_softmax(-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)
    return (logps, logits.mean(-1)) if return_mean_logits else logps


def _masked(
    hidden,
    weight,
    labels,
    mask = None,
    **kwargs,
):
    return torch.where(mask, _chunked(hidden, weight, labels, **kwargs), 0.0)


def _mean_logits(hidden, weight, bias, multiply, divide, softcap):
    return _logits(hidden, weight, bias, multiply, divide, softcap).mean(-1)


TRANSFORMS = (0.7, 0.0, 3.0)


def _load(masked_kernel):
    tree = ast.parse(SOURCE.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_unsloth_pref_chunked_forward"
    )
    ns = {
        "torch": torch,
        "os": os,
        "unsloth_chunked_logprobs": _chunked,
        "_unsloth_pref_masked_logprobs": _masked if masked_kernel else None,
        "_unsloth_linear_mean_logits": _mean_logits if masked_kernel else None,
        "_unsloth_pref_logit_transforms": lambda model: TRANSFORMS,
        "_unsloth_pref_returns_hidden_states": lambda model, tensor, weight: True,
    }
    exec(compile(ast.Module(body = [node], type_ignores = []), str(SOURCE), "exec"), ns)
    return ns["_unsloth_pref_chunked_forward"]


class _Trainer:
    def __init__(
        self,
        cpo_alpha = 1.0,
        loss_type = "sigmoid",
    ):
        self.cpo_alpha = cpo_alpha
        self.loss_type = loss_type
        self.accelerator = types.SimpleNamespace(
            unwrap_model = lambda m: m, device = torch.device("cpu")
        )


def _batch(split):
    g = torch.Generator().manual_seed(0)
    rows = 2 * PAIRS
    input_ids = torch.randint(1, VOCAB, (rows, SEQ), generator = g)
    attention_mask = torch.ones(rows, SEQ, dtype = torch.long)
    attention_mask[0, -2:] = 0  # right padding on a chosen row
    attention_mask[PAIRS + 1, -1:] = 0  # and on a rejected row
    labels = torch.where(attention_mask == 1, input_ids, -100)
    labels[:, :3] = -100  # prompt
    if split:
        # A completion token scored against a label other than its input id: ORPO needs a second NLL pass.
        labels[0, 4] = (input_ids[0, 4] + 1) % VOCAB
    states = torch.randn(rows, SEQ, HIDDEN, generator = g, dtype = torch.float64, requires_grad = True)
    head = torch.nn.Linear(HIDDEN, VOCAB, bias = True).double()
    with torch.no_grad():
        head.weight.copy_(torch.randn(VOCAB, HIDDEN, generator = g, dtype = torch.float64))
        head.bias.copy_(torch.randn(VOCAB, generator = g, dtype = torch.float64))
    batch = {
        "concatenated_input_ids": input_ids,
        "concatenated_labels": labels,
        "concatenated_attention_mask": attention_mask,
    }
    return head, batch, states


def _oracle(kind, trainer, head, batch, states):
    """TRL's ORPO / CPO concatenated_forward math on dense logits."""
    logits = _logits(states, head.weight, head.bias, *TRANSFORMS)
    labels = batch["concatenated_labels"]
    shifted, scored = labels[:, 1:], logits[:, :-1]
    mask = shifted != -100
    per_token = scored.log_softmax(-1).gather(-1, shifted.clamp(min = 0).unsqueeze(-1)).squeeze(-1)
    logps = (per_token * mask).sum(-1)
    if kind == "orpo" or trainer.loss_type in ("ipo", "simpo"):
        logps = logps / mask.sum(-1)
    if kind == "orpo":
        nll_labels = torch.where(
            batch["concatenated_attention_mask"] == 1, batch["concatenated_input_ids"], -100
        )
        nll = F.cross_entropy(scored[:PAIRS].flatten(0, 1), nll_labels[:PAIRS, 1:].flatten())
        means = scored.mean(-1)
    else:
        nll = F.cross_entropy(scored[:PAIRS].flatten(0, 1), shifted[:PAIRS].flatten())
        if trainer.cpo_alpha == 0:
            nll = torch.tensor(0.0)  # TRL: torch.tensor(0.0).to(self.accelerator.device)
        means = logits.mean(-1)
    return logps[:PAIRS], logps[PAIRS:], means[:PAIRS], means[PAIRS:], nll


def _scalar(outputs):
    weights = (1.0, -0.5, 0.25, 0.125, 2.0)
    return sum(w * o.sum() for w, o in zip(weights, outputs))


def _grads(fn, head, states):
    head.zero_grad(set_to_none = True)
    states.grad = None
    outputs = fn()
    _scalar(outputs).backward()
    return [o.detach() for o in outputs], [
        states.grad.clone(),
        head.weight.grad.clone(),
        head.bias.grad.clone(),
    ]


CASES = [
    ("orpo", {}, False),
    ("orpo", {}, True),
    ("cpo", {"cpo_alpha": 1.0, "loss_type": "sigmoid"}, False),
    ("cpo", {"cpo_alpha": 1.0, "loss_type": "simpo"}, True),
    ("cpo", {"cpo_alpha": 0.0, "loss_type": "sigmoid"}, False),
]


@pytest.mark.parametrize("masked_kernel", [True, False], ids = ["masked", "dense"])
@pytest.mark.parametrize("kind,kwargs,split", CASES)
def test_traces_without_graph_breaks_and_matches_trl(kind, kwargs, split, masked_kernel):
    forward = _load(masked_kernel)
    trainer = _Trainer(**kwargs)
    head, batch, states = _batch(split)

    def body(states, input_ids, labels, attention_mask):
        concatenated = {
            "concatenated_input_ids": input_ids,
            "concatenated_labels": labels,
            "concatenated_attention_mask": attention_mask,
        }
        outputs, _ = forward(trainer, head, head, concatenated, PAIRS, states, kind)
        return outputs

    args = (
        states,
        batch["concatenated_input_ids"],
        batch["concatenated_labels"],
        batch["concatenated_attention_mask"],
    )

    torch._dynamo.reset()
    explained = torch._dynamo.explain(body)(*args)
    assert explained.graph_break_count == 0, explained.break_reasons
    assert explained.graph_count == 1

    want, want_grads = _grads(lambda: _oracle(kind, trainer, head, batch, states), head, states)
    got, got_grads = _grads(lambda: body(*args), head, states)
    for a, b in zip(got + got_grads, want + want_grads):
        torch.testing.assert_close(a, b, rtol = 1e-10, atol = 1e-12)

    for dynamic in (False, True):
        torch._dynamo.reset()
        compiled = torch.compile(body, fullgraph = True, backend = "aot_eager", dynamic = dynamic)
        out, grads = _grads(lambda: compiled(*args), head, states)
        for a, b in zip(out + grads, got + got_grads):
            torch.testing.assert_close(a, b, rtol = 1e-10, atol = 1e-12)
    torch._dynamo.reset()
