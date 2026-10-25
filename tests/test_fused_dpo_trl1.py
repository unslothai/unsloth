# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FUSED_PREFERENCE_LOSS=1 on TRL >= 1.0: ``unsloth_fused_dpo_compute_loss_v1`` must
give TRL's ``DPOTrainer._compute_loss`` loss, gradients and logged metrics (float64, CPU)."""

from __future__ import annotations

import inspect
import os
import re
import types
from collections import defaultdict

import pytest
import torch

pytest.importorskip("transformers")
trl = pytest.importorskip("trl")


def _pristine_dpo_trainer():
    """TRL's own DPOTrainer, loaded fresh from its file: tests/conftest.py imports unsloth, which
    replaces trl.DPOTrainer with the compiled UnslothDPOTrainer, and that is not the oracle here."""
    import importlib.util
    import sys

    import trl.trainer.dpo_trainer as installed

    name = "trl.trainer._pristine_dpo_trainer"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, installed.__file__)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].DPOTrainer


DPOTrainer = _pristine_dpo_trainer()

if not hasattr(DPOTrainer, "_compute_loss") or "completion_mask" not in inspect.getsource(
    DPOTrainer._compute_loss
):
    pytest.skip("TRL < 1.0 DPOTrainer has no _compute_loss", allow_module_level = True)

from tests.test_fused_preference_loss import (  # noqa: E402
    PAD,
    VOCAB,
    HiddenStatesLlama,
    _Accelerator,
    _model,
    dpo_module,  # noqa: F401  (fixture)
)


def _rl_replacements_namespace():
    import pathlib
    import textwrap

    text = (
        pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
    ).read_text(encoding = "utf-8")
    start = text.index("def _insert_after_docstring(")
    end = text.index('RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_fused_preference_forward)')
    namespace = {"os": os, "re": re, "textwrap": textwrap}
    exec(text[start:end], namespace)
    return namespace


_RR = _rl_replacements_namespace()
_TRL_SOURCE = inspect.getsource(DPOTrainer._compute_loss)
# What the installed TRL's _compute_loss does; None = the hook is not installed on this TRL.
TRL_SEMANTICS = _RR["_dpo_v1_semantics"](_TRL_SOURCE)
needs_hook = pytest.mark.skipif(TRL_SEMANTICS is None, reason = "the hook does not apply to this TRL")


def _batch(
    seed,
    n = 3,
    width = 12,
):
    gen = torch.Generator().manual_seed(seed)
    rows = 2 * n
    ids = torch.full((rows, width), PAD, dtype = torch.long)
    attention = torch.zeros((rows, width), dtype = torch.long)
    completion = torch.zeros((rows, width), dtype = torch.long)
    for i in range(rows):
        prompt = int(torch.randint(2, 5, (1,), generator = gen))
        answer = int(torch.randint(1, width - prompt + 1, (1,), generator = gen))
        ids[i, : prompt + answer] = torch.randint(1, VOCAB, (prompt + answer,), generator = gen)
        attention[i, : prompt + answer] = 1
        completion[i, prompt : prompt + answer] = 1
    return {"input_ids": ids, "attention_mask": attention, "completion_mask": completion}


def _trainer(model, ref_model, **overrides):
    trainer = types.SimpleNamespace()
    trainer.__dict__.update(
        accelerator = _Accelerator(),
        model = model,
        ref_model = ref_model,
        precompute_ref_logps = False,
        aux_loss_enabled = False,
        use_liger_kernel = False,
        is_encoder_decoder = False,
        is_vision_model = False,
        ld_alpha = None,
        f_divergence_type = "reverse_kl",
        f_alpha_divergence_coef = 1.0,
        loss_types = ["sigmoid"],
        loss_weights = [1.0],
        beta = 0.1,
        label_smoothing = 0.0,
        use_weighting = False,
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens = 0,
    )
    if "_tp_size" in _TRL_SOURCE:
        # Only releases that read it set it (1.13+); older ones would raise on it.
        trainer._tp_size = 1
    trainer.args = types.SimpleNamespace(
        gradient_checkpointing_kwargs = None, discopop_tau = 0.05, padding_free = False
    )
    for key, value in overrides.items():
        setattr(trainer, key, value)
    if "loss_weights" not in overrides:
        trainer.loss_weights = [1.0] * len(trainer.loss_types)
    return trainer


CASES = [
    dict(),
    dict(loss_types = ["ipo"]),
    dict(loss_types = ["hinge"]),
    dict(loss_types = ["robust"], label_smoothing = 0.1),
    dict(loss_types = ["exo_pair"], label_smoothing = 0.1),
    dict(loss_types = ["nca_pair"]),
    dict(loss_types = ["bco_pair"]),
    dict(loss_types = ["sppo_hard"]),
    dict(loss_types = ["aot"]),
    dict(loss_types = ["aot_unpaired"]),
    dict(loss_types = ["apo_zero"]),
    dict(loss_types = ["apo_down"]),
    dict(loss_types = ["discopop"]),
    dict(loss_types = ["sigmoid_norm"]),
    dict(loss_types = ["sigmoid", "sft"], loss_weights = [0.7, 0.3]),
    dict(ld_alpha = 0.5),
    dict(use_weighting = True),
    dict(f_divergence_type = "forward_kl"),
    dict(f_divergence_type = "js_divergence"),
    dict(beta = 2.0),
    dict(softcap = 3.0),
    dict(bias = True),
]


def _run(trainer, model, batch, fused, module):
    model.zero_grad(set_to_none = True)
    model.train()
    if fused:
        HiddenStatesLlama.hidden_calls = 0
        loss = module.unsloth_fused_dpo_compute_loss_v1(
            trainer, model, dict(batch), False, trl_semantics = TRL_SEMANTICS
        )
        assert loss is not None, "fused path declined a supported case"
        assert HiddenStatesLlama.hidden_calls >= 1
    else:
        loss = DPOTrainer._compute_loss(trainer, model, dict(batch), False)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    metrics = {k: list(v) for k, v in trainer._metrics["train"].items()}
    return loss.detach(), metrics, grads


def _same(
    base,
    head,
    tol = 1e-9,
):
    (loss_b, metrics_b, grads_b), (loss_h, metrics_h, grads_h) = base, head
    assert torch.isfinite(loss_b)
    torch.testing.assert_close(loss_h, loss_b, rtol = tol, atol = tol)
    assert set(metrics_b) == set(metrics_h)
    for key in metrics_b:
        assert metrics_h[key] == pytest.approx(metrics_b[key], rel = 1e-7, abs = 1e-7), key
    assert set(grads_b) == set(grads_h)
    assert any(g.abs().sum() > 0 for g in grads_b.values())
    for name in grads_b:
        torch.testing.assert_close(grads_h[name], grads_b[name], rtol = tol, atol = tol, msg = name)


def _declined_here(case):
    # Pre-1.10 TRL: the fused path keeps forward_kl / js on TRL's own function.
    return (
        TRL_SEMANTICS["legacy_fdiv"] and case.get("f_divergence_type", "reverse_kl") != "reverse_kl"
    )


def _loss_type_known(case):
    known = set(re.findall(r'loss_type == "([^"]+)"', _TRL_SOURCE))
    return set(case.get("loss_types", ["sigmoid"])) <= known


@needs_hook
@pytest.mark.parametrize(
    "case", CASES, ids = lambda c: "-".join(f"{k}={v}" for k, v in c.items()) or "default"
)
@pytest.mark.parametrize("reference", ["ref_model", "precomputed"])
def test_compute_loss_matches_trl(dpo_module, monkeypatch, case, reference):
    case = dict(case)
    if not _loss_type_known(case):
        pytest.skip("loss type not in this TRL")
    softcap = case.pop("softcap", 0.0)
    bias = case.pop("bias", False)
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, softcap))
    batch = _batch(seed = 5)
    results = []
    for fused in (False, True):
        model = _model(0, softcap, bias)
        ref_model = _model(1, softcap, bias)
        run_batch = dict(batch)
        trainer = _trainer(model, ref_model, **case)
        if reference == "precomputed":
            with torch.no_grad():
                logits = ref_model(
                    input_ids = batch["input_ids"], attention_mask = batch["attention_mask"]
                ).logits
                logps = torch.log_softmax(logits[:, :-1], -1).gather(
                    -1, batch["input_ids"][:, 1:, None]
                )[..., 0]
                logps = (logps * batch["completion_mask"][:, 1:]).sum(1)
            run_batch["ref_chosen_logps"], run_batch["ref_rejected_logps"] = logps.chunk(2)
            trainer.precompute_ref_logps = True
            trainer.ref_model = None
        if fused and _declined_here(case):
            assert (
                dpo_module.unsloth_fused_dpo_compute_loss_v1(
                    trainer, model, dict(run_batch), False, trl_semantics = TRL_SEMANTICS
                )
                is None
            )
            return
        results.append(_run(trainer, model, run_batch, fused, dpo_module))
    _same(*results)


@needs_hook
def test_compute_loss_peft_reference(dpo_module, monkeypatch):
    peft = pytest.importorskip("peft")
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    batch = _batch(seed = 7)
    results = []
    for fused in (False, True):
        base = _model(0)
        model = peft.get_peft_model(
            base,
            peft.LoraConfig(
                r = 4, lora_alpha = 4, target_modules = ["q_proj", "v_proj"], init_lora_weights = False
            ),
        ).double()
        results.append(_run(_trainer(model, None), model, batch, fused, dpo_module))
    _same(*results)


@pytest.mark.parametrize(
    "overrides, inputs_extra, return_outputs",
    [
        (dict(use_liger_kernel = True), {}, False),
        (dict(aux_loss_enabled = True), {}, False),
        (dict(is_vision_model = True), {}, False),
        (dict(f_divergence_type = "alpha_divergence"), {}, False),
        (dict(loss_types = ["some_future_loss"]), {}, False),
        (dict(), {"pixel_values": torch.zeros(1)}, False),
        (dict(), {}, True),
    ],
)
def test_compute_loss_declines(dpo_module, monkeypatch, overrides, inputs_extra, return_outputs):
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    model = _model(0)
    trainer = _trainer(model, _model(1), **overrides)
    batch = dict(_batch(seed = 1), **inputs_extra)
    semantics = dict(
        TRL_SEMANTICS or {"legacy_fdiv": False, "entropy_mean": False, "tp_size": True}
    )
    assert (
        dpo_module.unsloth_fused_dpo_compute_loss_v1(
            trainer, model, batch, return_outputs, trl_semantics = semantics
        )
        is None
    )


def test_compute_loss_declines_without_semantics(dpo_module, monkeypatch):
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    model = _model(0)
    assert (
        dpo_module.unsloth_fused_dpo_compute_loss_v1(
            _trainer(model, _model(1)), model, _batch(1), False
        )
        is None
    )


def test_legacy_fdiv_keeps_only_reverse_kl(dpo_module, monkeypatch):
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    legacy = {"legacy_fdiv": True, "entropy_mean": False, "tp_size": False}
    for f_div in ("forward_kl", "js_divergence"):
        model = _model(0)
        trainer = _trainer(model, _model(1), f_divergence_type = f_div)
        assert (
            dpo_module.unsloth_fused_dpo_compute_loss_v1(
                trainer, model, _batch(1), False, trl_semantics = legacy
            )
            is None
        )


def test_compute_loss_flag_off(dpo_module, monkeypatch):
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_LOSS", "0")
    model = _model(0)
    semantics = {"legacy_fdiv": False, "entropy_mean": False, "tp_size": True}
    assert (
        dpo_module.unsloth_fused_dpo_compute_loss_v1(
            _trainer(model, _model(1)), model, _batch(1), False, trl_semantics = semantics
        )
        is None
    )


def test_rl_replacement_hooks_installed_trl_compute_loss(monkeypatch):
    hook = _RR["dpo_trainer_fused_preference_compute_loss"]
    source = _TRL_SOURCE
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_LOSS", "0")
    assert hook("_compute_loss", source) == source
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_LOSS", "1")
    patched = hook("_compute_loss", source)
    if TRL_SEMANTICS is None:
        # 0.29.x (truncates inside) and 1.15+ (TRL's own fused lm head) keep TRL's function.
        assert patched == source
        assert "self._truncate_inputs(" in source or "fused_lm_head=True" in source
        return
    assert (
        "unsloth_fused_dpo_compute_loss_v1(self, model, inputs, return_outputs, trl_semantics = "
        in patched
    )
    assert repr(TRL_SEMANTICS) in patched
    compile(__import__("textwrap").dedent(patched), "<patched>", "exec")
    # The flags follow the source: a release that reads _tp_size gets it, one that does not never does.
    assert TRL_SEMANTICS["tp_size"] == ("// self._tp_size" in source)
    # A TRL that logs a metric the replacement does not reproduce keeps its own function.
    extra = source.replace('self._metrics[mode]["entropy"]', 'self._metrics[mode]["entropy_new"]')
    assert hook("_compute_loss", extra) == extra
    assert hook("compute_loss", source) == source
    # Shapes the replacement does not reproduce keep TRL's function.
    for marker in ("self._truncate_inputs(", "fused_lm_head=True"):
        mutated = source.replace("mode = ", f"{marker}\n        mode = ", 1)
        assert hook("_compute_loss", mutated) == mutated
    # A release with neither f-divergence spelling keeps TRL's function.
    no_fdiv = source.replace(
        "1 - torch.exp(-chosen_logratios)", "torch.expm1(chosen_logratios)"
    ).replace("-torch.exp(-chosen_logratios)", "torch.expm1(chosen_logratios)")
    assert hook("_compute_loss", no_fdiv) == no_fdiv


def test_semantics_for_recorded_release_spellings():
    """The three behaviour switches, from the lines that differ between TRL 1.0 and 1.14."""
    sem = _RR["_dpo_v1_semantics"]
    base = _TRL_SOURCE if TRL_SEMANTICS is not None else None
    if base is None:
        pytest.skip("installed TRL source is not a hooked release")
    modern = (
        base.replace("-torch.exp(-chosen_logratios)", "1 - torch.exp(-chosen_logratios)")
        if TRL_SEMANTICS["legacy_fdiv"]
        else base
    )
    assert sem(modern)["legacy_fdiv"] is False
    legacy = modern.replace(
        "chosen_scores = 1 - torch.exp(-chosen_logratios)",
        "chosen_scores = -torch.exp(-chosen_logratios)",
    )
    assert sem(legacy)["legacy_fdiv"] is True
    assert (
        sem(
            modern + "\n        entropy = per_token_entropy[shift_completion_mask.bool()].mean()\n"
        )["entropy_mean"]
        is True
    )
    assert sem(modern.replace("// self._tp_size", ""))["tp_size"] is False
