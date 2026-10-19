# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 1.6 - 1.14 KTO ``_compute_loss`` + ``_compute_kl_logps`` and TRL 0.29 - 1.14 DPO precomputed reference
log-probs scored from hidden states by unsloth_zoo's fused linear op: TRL's own methods against the rewritten
ones on the same tiny model (loss, gradients, every logged metric), plus sabotaged rewrites the comparison must
catch."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import inspect
import os
import textwrap
import types
from collections import defaultdict

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl_replacements as rl

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")
dpo_test = pytest.importorskip("tests.python.test_fused_linear_dpo_compute_loss")
_TinyLM, _Accelerator = dpo_test._TinyLM, dpo_test._Accelerator
V = dpo_test.V

_METHODS = ("_compute_loss", "_compute_kl_logps")
_MASK = re.compile(
    r'(batch\["(?:KL_)?completion_mask"\]\[:, 1:\]|shift_completion_mask|inputs\["completion_mask"\]\[\.\.\., 1:\]),?$'
)


def _break_mask(source, anchor):
    """Score every row in the fused call that follows ``anchor`` instead of TRL's completion mask."""
    start = source.index(anchor)
    end = source.index("\n", start)
    line = source[start:end]
    mask = _MASK.search(line).group(1)
    return source[:start] + line.replace(mask, f"torch.ones_like({mask})") + source[end:]


def _kto_module():
    """TRL's KTO trainer module that defines ``_compute_kl_logps`` (1.6 - 1.9: experimental, 1.10+: trainer)."""
    pytest.importorskip("trl")
    for name in ("trl.trainer.kto_trainer", "trl.experimental.kto.kto_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        if "def _compute_kl_logps" in text and "def _compute_loss(" in text:
            return name, text
    pytest.skip(reason = "TRL < 1.6 KTO scores through get_batch_logps (rewritten elsewhere)")


def _kto_sources():
    name, text = _kto_module()
    if "selective_log_softmax" not in text or "add_fused_lm_head" in text:
        pytest.skip(reason = "TRL >= 1.15 fuses the head itself")
    out = {}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == "KTOTrainer":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name in _METHODS:
                    out[fn.name] = ast.get_source_segment(text, fn, padded = True)
    assert set(out) == set(_METHODS)
    return name, out


def _rewritten(sources):
    new = {k: rl.kto_trainer_compute_loss_chunked(k, v) for k, v in sources.items()}
    for k in _METHODS:
        assert new[k] != sources[k] and "_unsloth_fused_pass_logps" in new[k], k
    return new


def _bind(module_name, sources):
    module = importlib.import_module(module_name)
    ns = dict(vars(module))
    ns.update({name: getattr(rl, name) for name in dir(rl) if name.startswith("_unsloth_")})
    for name, src in sources.items():
        exec(compile(textwrap.dedent(src), f"<kto {name}>", "exec"), ns)
    return {name: ns[name] for name in sources}


def _batch(old_keys, label_list):
    gen = torch.Generator().manual_seed(5)
    ids = torch.randint(0, V, (4, 11), generator = gen)
    kl_ids = torch.randint(0, V, (4, 12), generator = gen)
    mask = torch.zeros_like(ids)
    kl_mask = torch.zeros_like(kl_ids)
    for row, (a, b) in enumerate(((4, 11), (3, 9), (5, 10), (2, 11))):
        mask[row, a:b] = 1
    for row, (a, b) in enumerate(((3, 12), (5, 10), (4, 12), (6, 11))):
        kl_mask[row, a:b] = 1
    # KL completions the policy (not the reference) likes, so the clamped KL term is not 0.
    policy = _TinyLM(0)
    with torch.no_grad():
        for t in range(kl_ids.shape[1] - 1):
            kl_ids[:, t + 1] = policy(kl_ids[:, t : t + 1]).logits[:, 0].argmax(-1)
    attn = torch.ones_like(ids)
    attn[1, 9:] = 0
    kl_attn = torch.ones_like(kl_ids)
    kl_attn[1, 10:] = 0
    labels = [True, False, True, False]
    prefix = "completion_" if old_keys else ""
    return {
        f"{prefix}input_ids": ids,
        f"{prefix}attention_mask": attn,
        "completion_mask": mask,
        f"KL_{prefix}input_ids": kl_ids,
        f"KL_{prefix}attention_mask": kl_attn,
        "KL_completion_mask": kl_mask,
        "label": labels if label_list else torch.tensor(labels),
    }


def _run(
    module_name,
    sources,
    loss_type = "kto",
):
    methods = _bind(module_name, sources)
    model, ref = _TinyLM(0), _TinyLM(1)
    ref.requires_grad_(False)
    trainer = types.SimpleNamespace(
        model = model,
        ref_model = ref,
        accelerator = _Accelerator(),
        beta = 0.1,
        loss_type = loss_type,
        desirable_weight = 1.0,
        undesirable_weight = 1.3,
        calculate_KL = loss_type == "kto",
        precompute_ref_logps = False,
        aux_loss_enabled = False,
        use_liger_kernel = False,
        is_fsdp_enabled = False,
        is_encoder_decoder = False,
        args = types.SimpleNamespace(gradient_checkpointing_kwargs = None),
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens = 0,
        _tp_size = 1,
    )
    trainer._compute_kl_logps = types.MethodType(methods["_compute_kl_logps"], trainer)
    text = sources["_compute_loss"]
    batch = _batch('"KL_completion_input_ids"' in text, 'torch.tensor(batch["label"])' in text)
    calls = fused_linear.CALLS["logprobs"]
    loss = methods["_compute_loss"](trainer, model, batch, False)
    used = fused_linear.CALLS["logprobs"] - calls
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    metrics = {k: v[-1] for k, v in trainer._metrics["train"].items()}
    return loss.detach(), grads, metrics, trainer, used


def _compare(want, got):
    want_loss, want_grads, want_metrics = want[:3]
    got_loss, got_grads, got_metrics = got[:3]
    torch.testing.assert_close(got_loss, want_loss, atol = 1e-5, rtol = 1e-5)
    for name, grad in want_grads.items():
        torch.testing.assert_close(got_grads[name], grad, atol = 1e-5, rtol = 1e-4, msg = name)
    assert set(got_metrics) == set(want_metrics)
    for key, value in want_metrics.items():
        assert got_metrics[key] == pytest.approx(value, rel = 1e-5, abs = 1e-5), key


def test_kto_rewrite_is_anchored():
    _, sources = _kto_sources()
    new = _rewritten(sources)
    for name in _METHODS:
        compile(textwrap.dedent(new[name]), name, "exec")
        assert rl.kto_trainer_compute_loss_chunked(name, new[name]) == new[name]
        assert rl.kto_trainer_compute_loss_chunked("compute_loss", sources[name]) == sources[name]
    # Another reader of the policy logits the rewrite does not know leaves TRL's body alone.
    extra = sources["_compute_loss"].replace(
        "        return (loss, outputs)",
        "        _ = outputs.logits.sum()\n        return (loss, outputs)",
    )
    assert extra != sources["_compute_loss"]
    assert rl.kto_trainer_compute_loss_chunked("_compute_loss", extra) == extra
    extra = sources["_compute_kl_logps"].replace(
        "        return KL_logps", "        _ = shift_KL_logits.sum()\n        return KL_logps"
    )
    assert extra != sources["_compute_kl_logps"]
    assert rl.kto_trainer_compute_loss_chunked("_compute_kl_logps", extra) == extra


@pytest.mark.parametrize("loss_type", ["kto", "apo_zero_unpaired"])
def test_kto_rewrite_matches_trl(loss_type):
    module_name, sources = _kto_sources()
    want = _run(module_name, sources, loss_type)
    got = _run(module_name, _rewritten(sources), loss_type)
    trainer = got[3]
    kl_passes = 2 if loss_type == "kto" else 0
    assert trainer._unsloth_dpo_chunked_calls == 2 + kl_passes
    assert got[4] >= 2 + kl_passes and want[4] == 0
    assert not getattr(trainer, "_unsloth_dpo_chunked_fallbacks", {})
    _compare(want, got)
    metrics = got[2]
    assert {"logps/chosen", "logps/rejected", "logits/chosen", "logits/rejected", "kl"} <= set(
        metrics
    )
    if loss_type == "kto":
        assert metrics["kl"] > 0  # the KL pass reaches the loss
    if (
        "entropy_from_logits" in sources["_compute_loss"]
        or "_and_entropy" in sources["_compute_loss"]
    ):
        assert "entropy" in metrics


@pytest.mark.parametrize(
    "sabotage",
    ["kl_mask", "policy_mask", "ref_mask", "entropy", "logits_metric"],
)
def test_kto_comparison_catches_a_broken_rewrite(sabotage, monkeypatch):
    module_name, sources = _kto_sources()
    new = _rewritten(sources)
    if sabotage in ("kl_mask", "policy_mask", "ref_mask"):
        var = {
            "kl_mask": "_unsloth_kto_kl, shift_KL_logits",
            "policy_mask": "_unsloth_kto_chunked, shift_logits",
            "ref_mask": "_unsloth_kto_ref, ref_shift_logits",
        }
        method = "_compute_kl_logps" if sabotage == "kl_mask" else "_compute_loss"
        new[method] = _break_mask(new[method], f"self, {var[sabotage]}, ")
    else:
        real = rl._unsloth_fused_pass_logps

        def broken(*args, **kwargs):
            out = real(*args, **kwargs)
            if not kwargs.get("stats"):
                return out
            logps, entropy, mean_logits = out
            if sabotage == "entropy":
                return logps, entropy * 1.001, mean_logits
            return logps, entropy, mean_logits * 1.001

        monkeypatch.setattr(rl, "_unsloth_fused_pass_logps", broken)
        stats = "stats = True" in new["_compute_loss"]
        if sabotage == "entropy" and not stats:
            pytest.skip(reason = "this TRL logs no entropy")
        # TRL 1.6 - 1.8 log the logits sums over every position instead of the masked row means.
        if sabotage == "logits_metric" and "_unsloth_fused_pass_vocab_sums" in new["_compute_loss"]:
            real_sums = rl._unsloth_fused_pass_vocab_sums
            monkeypatch.setattr(
                rl, "_unsloth_fused_pass_vocab_sums", lambda *a: real_sums(*a) * 1.001
            )
    want = _run(module_name, sources)
    got = _run(module_name, new)
    with pytest.raises(AssertionError):
        _compare(want, got)


@pytest.mark.parametrize("reason", ["kill_switch", "use_liger_kernel"])
def test_kto_rewrite_keeps_trl_path(reason, monkeypatch):
    module_name, sources = _kto_sources()
    new = _rewritten(sources)
    if reason == "kill_switch":
        monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
        want = _run(module_name, sources)
        got = _run(module_name, new)
    else:
        # The liger branches need liger; only check the head decision keeps TRL's path.
        trainer = types.SimpleNamespace(use_liger_kernel = True)
        assert rl._unsloth_fused_pass_head(trainer, None, {}, False, "_compute_loss") is None
        assert trainer._unsloth_dpo_chunked_fallbacks
        return
    assert got[4] == 0 and got[3]._unsloth_dpo_chunked_fallbacks
    assert torch.equal(got[0], want[0]) and got[2] == want[2]


def _dpo_ref_source():
    pytest.importorskip("trl")
    spec = importlib.util.find_spec("trl.trainer.dpo_trainer")
    text = open(spec.origin, encoding = "utf-8").read()
    if "add_fused_lm_head" in text or "def _compute_loss(" not in text:
        pytest.skip(reason = "TRL >= 1.15 fuses the head; TRL < 0.29 scores in concatenated_forward")
    name = (
        "_compute_ref_log_probs"
        if "def _compute_ref_log_probs" in text
        else "compute_ref_log_probs"
    )
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == "DPOTrainer":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == name:
                    return name, ast.get_source_segment(text, fn, padded = True)
    pytest.fail("no compute_ref_log_probs")


def _dpo_ref_run(name, source):
    import trl.trainer.dpo_trainer as dpo

    ns = dict(vars(dpo))
    ns.update({k: getattr(rl, k) for k in dir(rl) if k.startswith("_unsloth_")})
    exec(compile(textwrap.dedent(source), "<dpo ref>", "exec"), ns)
    trainer = dpo_test._trainer(["sigmoid"])
    trainer.ref_model.requires_grad_(
        True
    )  # a trainable head must still get no graph from the reference pass
    args = (trainer.ref_model,) if "model" in inspect.signature(ns[name]).parameters else ()
    calls = fused_linear.CALLS["logprobs"]
    chosen, rejected = ns[name](trainer, *args, dpo_test._inputs())
    return torch.cat([chosen, rejected]), fused_linear.CALLS["logprobs"] - calls, trainer


def test_dpo_precomputed_reference_matches_trl(monkeypatch):
    name, source = _dpo_ref_source()
    new = rl.dpo_trainer_ref_log_probs_chunked(name, source)
    assert new != source and "_unsloth_fused_pass_logps" in new
    assert rl.dpo_trainer_ref_log_probs_chunked(name, new) == new
    want, want_calls, _ = _dpo_ref_run(name, source)
    got, got_calls, trainer = _dpo_ref_run(name, new)
    assert want_calls == 0 and got_calls == 1 and trainer._unsloth_dpo_chunked_calls == 1
    assert not got.requires_grad
    torch.testing.assert_close(got, want, atol = 1e-5, rtol = 1e-5)
    # Broken: score every row, not only the completion.
    bad, _, _ = _dpo_ref_run(name, _break_mask(new, "self, _unsloth_dpo_ref, ref_shift_logits, "))
    with pytest.raises(AssertionError):
        torch.testing.assert_close(bad, want, atol = 1e-5, rtol = 1e-5)
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    off, off_calls, _ = _dpo_ref_run(name, new)
    assert off_calls == 0 and torch.equal(off, want)
