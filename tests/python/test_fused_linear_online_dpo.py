# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL Online DPO ``_forward`` scored from hidden states by unsloth_zoo's fused linear op: TRL's own ``_forward``
(and Unsloth's right-align rewrite of it) against the full RL_FUNCTIONS["online_dpo_trainer"] chain on a float64
tiny Qwen3 (per-position log-probs, padding positions included, and their gradients), the kill switches, and the
paths that must keep TRL's math (vision inputs, a forward that ignores the flag)."""

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


def _odpo_module():
    """TRL's module defining ``OnlineDPOTrainer._forward`` (<= 0.25: trl.trainer, 0.26+: trl.experimental)."""
    pytest.importorskip("trl")
    import trl.import_utils as iu

    iu._mergekit_available = iu._llm_blender_available = (
        False  # broken dist-info fails the trainer import (judges.py)
    )
    for name in (
        "trl.experimental.online_dpo.online_dpo_trainer",
        "trl.trainer.online_dpo_trainer",
    ):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        if "class OnlineDPOTrainer" in text and "def _forward(" in text:
            return importlib.import_module(name), text
    pytest.skip(reason = "TRL has no OnlineDPOTrainer._forward")


def _source(text):
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == "OnlineDPOTrainer":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "_forward":
                    return ast.get_source_segment(text, fn, padded = True)
    raise AssertionError("no _forward")


def _bind(module, source, edits):
    ns = dict(vars(module))
    ns.update(os = os, re = re, torch = torch, inspect = inspect, textwrap = textwrap)
    for item in rl.RL_PRE_ITEMS["online_dpo_trainer"]:
        exec(compile(item, "<online dpo pre item>", "exec"), ns)
    new = source
    for edit in edits:
        new = edit("_forward", new)
    exec(compile(textwrap.dedent(new), "<online dpo _forward>", "exec"), ns)
    return new, ns["_forward"]


def _chain(full):
    edits = list(rl.RL_FUNCTIONS["online_dpo_trainer"])
    if not full:
        edits = [e for e in edits if e is not rl.online_dpo_trainer_chunked_logprobs]
    return edits


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model


def _trainer(forward, ref_model = None):
    cls = type("T", (), {"_forward": forward})
    t = cls()
    t.max_length = 64
    t.accelerator = _Accelerator()
    t.ref_model = ref_model
    return t


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


def _model(dtype = torch.float64, honour = True):
    from transformers import AutoModelForCausalLM

    if "m" not in _BASE:
        torch.manual_seed(0)
        _BASE["m"] = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype = torch.float64)
    model = copy.deepcopy(_BASE["m"]).to(dtype)
    return _honour_hidden_states(model) if honour else model


def _inputs(left_pad = True, vocab = 1000):
    g = torch.Generator().manual_seed(1)
    prompt_ids = torch.randint(5, vocab, (4, 7), generator = g)
    prompt_mask = torch.ones_like(prompt_ids)
    if left_pad:
        prompt_mask[1, :3] = 0
        prompt_mask[3, :1] = 0
        prompt_ids = prompt_ids.masked_fill(prompt_mask == 0, 0)
    completion_ids = torch.randint(5, vocab, (4, 9), generator = g)
    completion_mask = torch.ones_like(completion_ids)
    completion_mask[0, 6:] = 0
    completion_mask[2, 4:] = 0
    completion_ids = completion_ids.masked_fill(completion_mask == 0, 0)
    return prompt_ids, prompt_mask, completion_ids, completion_mask


def _score(
    forward,
    model,
    inputs,
    grad = True,
):
    trainer = _trainer(forward)
    model.zero_grad(set_to_none = True)
    calls = fused_linear.CALLS["logprobs"]
    with torch.set_grad_enabled(grad):
        logps = forward(trainer, model, *inputs)
    used = fused_linear.CALLS["logprobs"] - calls
    grads = {}
    if grad:
        # A functional of every position, padding included (TRL's objective/kl and objective/entropy read them).
        w = torch.linspace(-1.0, 1.0, logps.numel(), dtype = torch.float64).view_as(logps)
        (logps.double() * w).sum().backward()
        grads = {
            n: p.grad.double().clone() for n, p in model.named_parameters() if p.grad is not None
        }
    return logps.detach().double(), grads, used, trainer


def _max_err(want, got):
    lp = (got[0] - want[0]).abs().max().item()
    gr = max(
        ((got[1][n] - g).abs().max() / g.abs().max().clamp_min(1e-30)).item()
        for n, g in want[1].items()
    )
    return lp, gr


@pytest.mark.parametrize("left_pad", [True, False])
def test_online_dpo_matches_unsloth_base_fp64(left_pad, monkeypatch):
    module, text = _odpo_module()
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    monkeypatch.delenv("UNSLOTH_PREFERENCE_CHUNKED", raising = False)
    source = _source(text)
    _, base = _bind(module, source, _chain(False))
    new, head = _bind(module, source, _chain(True))
    assert "_unsloth_odpo_hidden_logps" in new
    inputs = _inputs(left_pad)
    want = _score(base, _model(), inputs)
    got = _score(head, _model(), inputs)
    assert want[2] == 0 and got[2] == 1
    assert set(got[1]) == set(want[1])
    lp, gr = _max_err(want, got)
    bf16 = _score(base, _model(torch.bfloat16), inputs)
    blp, bgr = _max_err(want, bf16)
    fused_bf16 = _score(head, _model(torch.bfloat16), inputs)
    flp, fgr = _max_err(want, fused_bf16)
    print(
        f"\nOnline DPO left_pad={left_pad}: fused fp64 logp {lp:.2e} grad {gr:.2e} | "
        f"stock bf16 logp {blp:.2e} grad {bgr:.2e} | fused bf16 logp {flp:.2e} grad {fgr:.2e}"
    )
    assert lp <= 1e-5 and gr <= 1e-4
    assert lp <= blp and gr <= bgr
    assert flp <= blp


def test_online_dpo_matches_pure_trl_without_left_padding(monkeypatch):
    """No left padding: Unsloth's right-align is the identity, so the full chain must equal TRL's own text."""
    module, text = _odpo_module()
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    source = _source(text)
    _, trl_forward = _bind(module, source, [])
    _, head = _bind(module, source, _chain(True))
    inputs = _inputs(left_pad = False)
    want = _score(trl_forward, _model(), inputs)
    got = _score(head, _model(), inputs)
    assert got[2] == 1
    lp, gr = _max_err(want, got)
    assert lp <= 1e-5 and gr <= 1e-4, (lp, gr)


def test_online_dpo_reference_pass_keeps_no_graph(monkeypatch):
    module, text = _odpo_module()
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    _, base = _bind(module, _source(text), _chain(False))
    _, head = _bind(module, _source(text), _chain(True))
    inputs = _inputs()
    want = _score(base, _model(), inputs, grad = False)
    got = _score(head, _model(), inputs, grad = False)
    assert got[2] == 1
    assert (got[0] - want[0]).abs().max().item() <= 1e-5


@pytest.mark.parametrize("switch", ["UNSLOTH_FUSED_LINEAR", "UNSLOTH_PREFERENCE_CHUNKED"])
def test_online_dpo_kill_switch_keeps_unsloth_base(switch, monkeypatch):
    module, text = _odpo_module()
    monkeypatch.setenv(switch, "0")
    _, base = _bind(module, _source(text), _chain(False))
    _, head = _bind(module, _source(text), _chain(True))
    inputs = _inputs()
    want = _score(base, _model(), inputs)
    got = _score(head, _model(), inputs)
    assert got[2] == 0
    assert getattr(got[3], "_unsloth_dpo_chunked_calls", 0) == 0
    torch.testing.assert_close(got[0], want[0], atol = 0, rtol = 0)


def test_online_dpo_forward_ignoring_the_flag_keeps_trl_math(monkeypatch):
    module, text = _odpo_module()
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    _, base = _bind(module, _source(text), _chain(False))
    _, head = _bind(module, _source(text), _chain(True))
    inputs = _inputs()
    want = _score(base, _model(honour = False), inputs)
    got = _score(head, _model(honour = False), inputs)
    assert got[2] == 0
    assert "forward returned logits, not hidden states" in got[3]._unsloth_dpo_chunked_fallbacks
    torch.testing.assert_close(got[0], want[0], atol = 0, rtol = 0)


def test_online_dpo_head_declines_vision_inputs(monkeypatch):
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    trainer = _trainer(None)
    model = _model()
    batch = {"attention_mask": torch.ones(1, 2), "pixel_values": torch.zeros(1)}
    assert rl._unsloth_odpo_chunked_head(trainer, model, batch, "_forward") is None
    assert (
        rl._unsloth_odpo_chunked_head(
            trainer, model, {"attention_mask": batch["attention_mask"]}, "_forward"
        )
        is not None
    )


def test_online_dpo_rewrite_is_anchored_and_idempotent():
    _, text = _odpo_module()
    source = _source(text)
    for chain in ([rl.online_dpo_trainer__forward], []):
        start = source
        for edit in chain:
            start = edit("_forward", start)
        new = rl.online_dpo_trainer_chunked_logprobs("_forward", start)
        assert new != start
        compile(textwrap.dedent(new), "x", "exec")
        assert rl.online_dpo_trainer_chunked_logprobs("_forward", new) == new
    # Another reader of the logits (a temperature divide TRL might add) leaves the source untouched.
    broken = source.replace(
        "logprobs = torch.take_along_dim(logits.log_softmax(dim=-1)",
        "logits = logits / 0.7\n        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1)",
    )
    assert broken != source
    assert rl.online_dpo_trainer_chunked_logprobs("_forward", broken) == broken


# TRL 0.22's ``_forward`` tail (no vision kwargs, ``start_idx:-1``), as the chain sees it.
_TRL_022_FORWARD = """
    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)

        # There is 1 offset, because the model predict the next token
        prompt_len = prompt_ids.size(1)
        start_idx = prompt_len - 1 if prompt_len > 0 else 0
        logits = output.logits[:, start_idx:-1]

        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logprobs
"""


def test_online_dpo_trl_022_layout(monkeypatch):
    module, _ = _odpo_module()
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    new, head = _bind(module, _TRL_022_FORWARD, _chain(True))
    assert "_unsloth_odpo_hidden_logps" in new and "_unsloth_left_pad" in new
    _, base = _bind(module, _TRL_022_FORWARD, _chain(False))
    inputs = _inputs()
    want = _score(base, _model(), inputs)
    got = _score(head, _model(), inputs)
    assert got[2] == 1
    lp, gr = _max_err(want, got)
    assert lp <= 1e-5 and gr <= 1e-4, (lp, gr)
