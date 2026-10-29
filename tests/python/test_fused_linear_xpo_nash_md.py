# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 0.22 - 1.10 XPO / Nash-MD ``_compute_logprobs`` scored from hidden states by unsloth_zoo's fused linear op:
TRL's own method against the rewritten one on a float64 tiny Qwen3 (every returned log-prob tensor and the policy
gradients), plus the kill switch."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import os
import re
import textwrap

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
_TRAINERS = {"xpo": "XPOTrainer", "nash_md": "NashMDTrainer"}


def _module(kind):
    pytest.importorskip("trl")
    import trl.import_utils as iu

    iu._mergekit_available = iu._llm_blender_available = (
        False  # broken dist-info fails the trainer import (judges.py)
    )
    for name in (f"trl.experimental.{kind}.{kind}_trainer", f"trl.trainer.{kind}_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        if f"class {_TRAINERS[kind]}" in text and "def _compute_logprobs(" in text:
            return importlib.import_module(name), text
    pytest.skip(reason = f"TRL >= 1.14 has no {_TRAINERS[kind]}")


def _source(kind, text):
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == _TRAINERS[kind]:
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_compute_logprobs":
                    return ast.get_source_segment(text, fn, padded = True)
    raise AssertionError("no _compute_logprobs")


def _bind(kind, rewrite):
    module, text = _module(kind)
    source = _source(kind, text)
    ns = dict(vars(module))
    ns.update(os = os, re = re, torch = torch, inspect = inspect, textwrap = textwrap)
    for item in rl.RL_PRE_ITEMS[f"{kind}_trainer"]:
        exec(compile(item, "<pre item>", "exec"), ns)
    new = source
    if rewrite:
        for edit in rl.RL_FUNCTIONS[f"{kind}_trainer"]:
            new = edit("_compute_logprobs", new)
        assert "_unsloth_odpo_hidden_logps" in new
    exec(compile(textwrap.dedent(new), f"<{kind} _compute_logprobs>", "exec"), ns)
    return ns["_compute_logprobs"]


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


def _honour_hidden_states(model):
    original = model.forward

    def forward(*args, **kwargs):
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") != "1":
            return original(*args, **kwargs)
        kwargs["output_hidden_states"] = True
        out = original(*args, **kwargs)
        out.logits = out.hidden_states[-1]
        return out

    model.forward = forward
    return model


_BASE = {}


def _models():
    from transformers import AutoModelForCausalLM
    if "m" not in _BASE:
        torch.manual_seed(0)
        ref = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype = torch.float64)
        policy = copy.deepcopy(ref)
        with torch.no_grad():
            for p in policy.parameters():
                p.add_(0.05 * torch.randn_like(p))
        ref.requires_grad_(False)
        _BASE["m"] = (policy, ref)
    return tuple(_honour_hidden_states(copy.deepcopy(m)) for m in _BASE["m"])


def _data(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, 1000, (3, 14), generator = g)
    mask = torch.ones_like(ids)
    mask[0, 11:] = 0  # completion padding
    mask[2, 8:] = 0
    ids = ids.masked_fill(mask == 0, 0)
    return {"input_ids": ids, "attention_mask": mask, "raw": ["p"] * 3}


def _run(kind, fn):
    policy, ref = _models()
    trainer = type("T", (), {"_compute_logprobs": fn})()
    trainer.accelerator = _Accelerator()
    trainer.ref_model = ref
    args = (_data(1), _data(2)) if kind == "xpo" else (_data(1),)
    calls = fused_linear.CALLS["logprobs"]
    outs = fn(trainer, policy, *args, 6)
    used = fused_linear.CALLS["logprobs"] - calls
    total = sum((o.double() * (i + 1)).sum() for i, o in enumerate(outs) if o.requires_grad)
    total.backward()
    grads = {n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None}
    return [o.detach().double() for o in outs], grads, used, [o.requires_grad for o in outs]


@pytest.mark.parametrize("kind", ["xpo", "nash_md"])
def test_matches_trl_fp64(kind, monkeypatch):
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    monkeypatch.delenv("UNSLOTH_PREFERENCE_CHUNKED", raising = False)
    want = _run(kind, _bind(kind, False))
    got = _run(kind, _bind(kind, True))
    assert want[2] == 0 and got[2] == (4 if kind == "xpo" else 2)
    assert got[3] == want[3]  # policy passes keep a graph, reference passes do not
    for a, b in zip(want[0], got[0]):
        assert (a - b).abs().max().item() <= 1e-5
    assert set(got[1]) == set(want[1])
    for n, g in want[1].items():
        assert ((got[1][n] - g).abs().max() / g.abs().max().clamp_min(1e-30)).item() <= 1e-4, n


@pytest.mark.parametrize("kind", ["xpo", "nash_md"])
def test_kill_switch(kind, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    want = _run(kind, _bind(kind, False))
    got = _run(kind, _bind(kind, True))
    assert got[2] == 0
    for a, b in zip(want[0], got[0]):
        torch.testing.assert_close(a, b, atol = 0, rtol = 0)


@pytest.mark.parametrize("kind", ["xpo", "nash_md"])
def test_rewrite_is_anchored_and_idempotent(kind):
    _, text = _module(kind)
    source = _source(kind, text)
    new = rl.xpo_trainer_chunked_logprobs("_compute_logprobs", source)
    assert new != source
    assert rl.xpo_trainer_chunked_logprobs("_compute_logprobs", new) == new
    broken = source.replace("return token_logprobs", "return token_logprobs, logits")
    assert rl.xpo_trainer_chunked_logprobs("_compute_logprobs", broken) == broken
