# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 0.22 - 1.10 BCO ``forward`` (policy + reference pass) and ``compute_reference_log_probs`` (precompute pass)
scored from hidden states by unsloth_zoo's fused linear op: TRL's own BCOTrainer against one whose two methods went
through Unsloth's RL_FUNCTIONS chain, on a float64 tiny Qwen3 (loss, gradients, every logged metric, precomputed
reference log-probs), plus the kill switches."""

from __future__ import annotations

import ast
import copy
import importlib
import importlib.util
import inspect
import os
import re
import sys
import textwrap

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
_METHODS = ("forward", "compute_reference_log_probs")
_STOCK = {}


def _bco_module():
    """TRL's real BCO module (0.22 - 0.25: trl.trainer, 0.26 - 1.10: trl.experimental), re-executed from its file:
    importing unsloth replaces the trainer class in TRL's own module with Unsloth's generated one."""
    pytest.importorskip("trl")
    import trl.import_utils as iu

    iu._mergekit_available = False
    for name in ("trl.experimental.bco.bco_trainer", "trl.trainer.bco_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        if "class BCOTrainer" in text and "def compute_reference_log_probs" in text:
            if name not in _STOCK:
                importlib.import_module(name)
                fresh_spec = importlib.util.spec_from_file_location(name + "_stock", spec.origin)
                fresh = importlib.util.module_from_spec(fresh_spec)
                fresh.__package__ = name.rsplit(".", 1)[0]
                sys.modules[fresh_spec.name] = fresh
                fresh_spec.loader.exec_module(fresh)
                _STOCK[name] = (fresh, text)
            return _STOCK[name]
    pytest.skip(reason = "TRL >= 1.14 has no BCOTrainer")


def _sources(text):
    out = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == "BCOTrainer":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in _METHODS:
                    out[fn.name] = ast.get_source_segment(text, fn, padded = True)
    assert set(out) == set(_METHODS)
    return out


def _patched_class(module, text):
    """BCOTrainer with ``forward`` / ``compute_reference_log_probs`` as unsloth/models/rl.py generates them: every
    RL_FUNCTIONS["bco_trainer"] edit, in a namespace holding RL_PRE_ITEMS["bco_trainer"]."""
    ns = dict(vars(module))
    ns.update(os = os, re = re, torch = torch, inspect = inspect, textwrap = textwrap)
    for item in rl.RL_PRE_ITEMS["bco_trainer"]:
        exec(compile(item, "<bco pre item>", "exec"), ns)
    body = {}
    for name, source in _sources(text).items():
        new = source
        for edit in rl.RL_FUNCTIONS["bco_trainer"]:
            new = edit(name, new)
        assert new != source, f"{name} was not rewritten"
        exec(compile(textwrap.dedent(new), f"<bco {name}>", "exec"), ns)
        body[name] = ns[name]
    return type("PatchedBCOTrainer", (module.BCOTrainer,), body)


def _honour_hidden_states(model):
    """What Unsloth's forward does under UNSLOTH_RETURN_HIDDEN_STATES=1: the final hidden states as ``.logits``."""
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


def _dataset():
    from datasets import Dataset

    g = torch.Generator().manual_seed(0)
    words = "the of and to in is that for it as with was on be by this are from at or an".split()

    def text(n):
        return " ".join(words[i] for i in torch.randint(len(words), (n,), generator = g).tolist())

    return Dataset.from_list(
        [
            {"prompt": text(4 + i), "completion": text(6 + 3 * i), "label": i % 2 == 0}
            for i in range(4)
        ]
    )


_MODELS = {}


def _models(dtype):
    from transformers import AutoModelForCausalLM
    if dtype not in _MODELS:
        torch.manual_seed(0)
        ref = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype = torch.float64)
        policy = copy.deepcopy(ref)
        with (
            torch.no_grad()
        ):  # a policy that differs from the reference, so the rewards are not all zero
            for p in policy.parameters():
                p.add_(0.05 * torch.randn_like(p))
        _MODELS[dtype] = (policy.to(dtype), ref.to(dtype))
    return _MODELS[dtype]


def _run(
    cls,
    module,
    dtype = torch.float64,
    precompute = False,
    honour = True,
):
    from transformers import AutoTokenizer

    policy, ref = (copy.deepcopy(m) for m in _models(dtype))
    for m in (policy, ref):
        if not hasattr(m, "warnings_issued"):
            m.warnings_issued = {}
        if honour:
            _honour_hidden_states(m)
    tok = AutoTokenizer.from_pretrained(TINY)
    cfg = module.BCOConfig(
        output_dir = "/tmp/_unsloth_bco_test",
        per_device_train_batch_size = 4,
        report_to = "none",
        use_cpu = True,
        max_length = 64,
        precompute_ref_log_probs = precompute,
        dataset_num_proc = None,
        bf16 = False,
        fp16 = False,
    )
    cfg.dataset_num_proc = None
    trainer = cls(
        model = policy, ref_model = ref, args = cfg, train_dataset = _dataset(), processing_class = tok
    )
    calls = fused_linear.CALLS["logprobs"]
    batch = next(iter(trainer.get_train_dataloader()))
    precompute_used = fused_linear.CALLS["logprobs"] - calls
    # TRL stores the precomputed log-probs as float32; score one batch directly for the unrounded values.
    direct = None
    if precompute:
        padded = {k: v for k, v in batch.items() if k != "reference_logps"}
        direct = trainer.compute_reference_log_probs(padded).double()
    calls = fused_linear.CALLS["logprobs"]
    loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch)
    loss.backward()
    used = fused_linear.CALLS["logprobs"] - calls
    grads = {
        n: p.grad.double().clone()
        for n, p in trainer.model.named_parameters()
        if p.grad is not None
    }
    ref_logps = batch.get("reference_logps")
    return dict(
        loss = loss.detach().double(),
        metrics = metrics,
        grads = grads,
        ref_logps = None if ref_logps is None else ref_logps.double(),
        direct = direct,
        used = used,
        precompute_used = precompute_used,
        trainer = trainer,
    )


def _err(want, got):
    out = {"loss": (got["loss"] - want["loss"]).abs().item()}
    out["grad"] = max(
        ((got["grads"][n] - g).abs().max() / g.abs().max().clamp_min(1e-30)).item()
        for n, g in want["grads"].items()
    )
    out["metrics"] = max(
        abs(got["metrics"][k] - v) / max(abs(v), 1e-6) for k, v in want["metrics"].items()
    )
    if want["ref_logps"] is not None:
        out["ref_logps"] = (got["ref_logps"] - want["ref_logps"]).abs().max().item()
        out["direct_ref_logps"] = (got["direct"] - want["direct"]).abs().max().item()
    return out


@pytest.mark.parametrize("precompute", [False, True])
def test_bco_matches_trl_fp64(precompute, monkeypatch):
    module, text = _bco_module()
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    monkeypatch.delenv("UNSLOTH_PREFERENCE_CHUNKED", raising = False)
    want = _run(module.BCOTrainer, module, precompute = precompute)
    got = _run(_patched_class(module, text), module, precompute = precompute)
    # forward: policy + reference (the reference pass is skipped when precomputed); precompute: one per batch.
    assert got["used"] >= (1 if precompute else 2), got["used"]
    assert (got["precompute_used"] >= 1) == precompute, got["precompute_used"]
    assert want["used"] == want["precompute_used"] == 0
    assert getattr(got["trainer"], "_unsloth_dpo_chunked_calls", 0) >= 2
    assert set(got["metrics"]) == set(want["metrics"])
    assert set(got["grads"]) == set(want["grads"])
    err = _err(want, got)
    bf16 = _err(want, _run(module.BCOTrainer, module, dtype = torch.bfloat16, precompute = precompute))
    print(f"\nBCO precompute={precompute} fused fp64 err {err} | stock bf16 err {bf16}")
    assert err["loss"] <= 1e-5 and err["grad"] <= 1e-4 and err["metrics"] <= 1e-4, err
    if precompute:
        assert err["ref_logps"] <= 1e-4 and err["direct_ref_logps"] <= 1e-4, err
    for key in err:
        assert err[key] <= max(bf16[key], 1e-5), (key, err, bf16)


@pytest.mark.parametrize("switch", ["UNSLOTH_FUSED_LINEAR", "UNSLOTH_PREFERENCE_CHUNKED"])
def test_bco_kill_switch_keeps_trl_path(switch, monkeypatch):
    module, text = _bco_module()
    monkeypatch.setenv(switch, "0")
    want = _run(module.BCOTrainer, module, precompute = True)
    got = _run(_patched_class(module, text), module, precompute = True)
    assert got["used"] == got["precompute_used"] == 0
    assert getattr(got["trainer"], "_unsloth_dpo_chunked_calls", 0) == 0
    torch.testing.assert_close(got["loss"], want["loss"], atol = 0, rtol = 0)
    torch.testing.assert_close(got["ref_logps"], want["ref_logps"], atol = 0, rtol = 0)
    torch.testing.assert_close(got["direct"], want["direct"], atol = 0, rtol = 0)


def test_bco_forward_ignoring_the_flag_keeps_trl_math(monkeypatch):
    module, text = _bco_module()
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR", raising = False)
    want = _run(module.BCOTrainer, module)
    got = _run(_patched_class(module, text), module, honour = False)
    assert got["used"] == 0
    torch.testing.assert_close(got["loss"], want["loss"], atol = 0, rtol = 0)


def test_bco_rewrite_is_anchored_and_idempotent():
    _, text = _bco_module()
    for name, source in _sources(text).items():
        new = rl.bco_trainer_chunked_logprobs(name, source)
        assert new != source, name
        compile(textwrap.dedent(new), name, "exec")
        assert rl.bco_trainer_chunked_logprobs(name, new) == new, name
        # TRL's statement stays as the fallback branch.
        assert "self.get_batch_logps(" in new, name
    # A forward that reads the logits elsewhere (here: a mean) is left alone.
    forward = _sources(text)["forward"]
    broken = forward.replace(
        "chosen_logits = completion_logits[chosen_idx, ...]",
        "chosen_logits = completion_logits.mean(-1)",
    )
    assert broken != forward
    assert rl.bco_trainer_chunked_logprobs("forward", broken) == broken
    # A precompute pass whose logits get another reader is left alone.
    ref = _sources(text)["compute_reference_log_probs"]
    broken = ref.replace(
        "        return completion_logps",
        "        return completion_logps, completion_logits.mean()",
    )
    assert broken != ref
    assert rl.bco_trainer_chunked_logprobs("compute_reference_log_probs", broken) == broken
