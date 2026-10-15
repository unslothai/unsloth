# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FUSED_PREFERENCE_LOSS=1 must reproduce TRL's DPO / ORPO / CPO exactly.

The fused path scores completions from the final hidden states through the output head in
row chunks (unsloth_zoo.fused_losses.preference_loss) instead of building batch x seq x vocab
logits. Each case runs TRL's own get_batch_loss_metrics twice on a tiny float64 Llama, once
through TRL's concatenated_forward and once through Unsloth's, and compares the loss, every
logged metric and the parameter gradients. TRL >= 0.29 DPO has no concatenated_forward, so
the DPO cases need TRL 0.22-0.28; ORPO / CPO keep theirs through the 1.x experimental copies.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import pathlib
import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")

from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _zoo_preference_loss():
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("unsloth_zoo not installed")
    path = pathlib.Path(list(spec.submodule_search_locations)[0]) / "fused_losses" / "preference_loss.py"
    if not path.exists():
        pytest.skip("unsloth_zoo without fused_losses/preference_loss.py")
    return _load("_zoo_preference_loss_under_test", path)


@pytest.fixture
def dpo_module(monkeypatch):
    zoo = _zoo_preference_loss()
    module = _load("_unsloth_dpo_under_test", ROOT / "unsloth" / "models" / "dpo.py")
    monkeypatch.setattr(
        module, "_zoo_kernels", lambda: (zoo.fused_linear_token_logps, zoo.linear_mean_logits)
    )
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_LOSS", "1")
    return module


VOCAB, HIDDEN = 61, 16


class HiddenStatesLlama(LlamaForCausalLM):
    """Honours UNSLOTH_RETURN_HIDDEN_STATES the way Unsloth's forwards do."""

    softcap = 0.0
    calls = 0
    hidden_calls = 0

    def forward(self, input_ids = None, attention_mask = None, position_ids = None,
                logits_to_keep = 0, use_cache = None, **kwargs):
        type(self).calls += 1
        hidden = self.model(
            input_ids = input_ids, attention_mask = attention_mask, position_ids = position_ids,
            use_cache = False,
        ).last_hidden_state
        if logits_to_keep:
            hidden = hidden[:, -logits_to_keep:]
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            type(self).hidden_calls += 1
            return CausalLMOutputWithPast(logits = hidden)
        logits = self.lm_head(hidden)
        if self.softcap:
            logits = self.softcap * torch.tanh(logits / self.softcap)
        return CausalLMOutputWithPast(logits = logits)


def _model(seed, softcap = 0.0, bias = False, vocab = VOCAB, hidden = HIDDEN):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size = vocab, hidden_size = hidden, intermediate_size = 32, num_hidden_layers = 2,
        num_attention_heads = 2, num_key_value_heads = 2, max_position_embeddings = 64,
        tie_word_embeddings = False,
    )
    model = HiddenStatesLlama(config).double()
    if bias:
        model.lm_head.bias = torch.nn.Parameter(torch.randn(vocab, dtype = torch.float64))
    with torch.no_grad():
        model.lm_head.weight.mul_(8.0)
    model.softcap = softcap
    return model


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def gather_for_metrics(self, x):
        return x

    def gather(self, x):
        return x


def _rows(gen, n, lo, hi, pad, left):
    lengths = torch.randint(lo, hi + 1, (n,), generator = gen)
    width = int(lengths.max())
    ids = torch.full((n, width), pad, dtype = torch.long)
    mask = torch.zeros((n, width), dtype = torch.long)
    for i, length in enumerate(lengths.tolist()):
        tokens = torch.randint(1, VOCAB, (length,), generator = gen)
        if left:
            ids[i, width - length:] = tokens
            mask[i, width - length:] = 1
        else:
            ids[i, :length] = tokens
            mask[i, :length] = 1
    return ids, mask


PAD = 0


def _dpo_batch(seed, n = 3):
    gen = torch.Generator().manual_seed(seed)
    prompt_ids, prompt_mask = _rows(gen, n, 2, 6, PAD, left = True)
    chosen_ids, chosen_mask = _rows(gen, n, 1, 7, PAD, left = False)
    rejected_ids, rejected_mask = _rows(gen, n, 1, 5, PAD, left = False)
    return {
        "prompt_input_ids": prompt_ids, "prompt_attention_mask": prompt_mask,
        "chosen_input_ids": chosen_ids, "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": rejected_ids, "rejected_attention_mask": rejected_mask,
    }


def _dpo_trainer(module, model, ref_model, fused, **overrides):
    from trl import DPOTrainer

    if not hasattr(DPOTrainer, "concatenated_forward"):
        pytest.skip("TRL >= 0.29 DPOTrainer has no concatenated_forward")

    class Stub:
        concatenated_inputs = staticmethod(DPOTrainer.concatenated_inputs)
        dpo_loss = DPOTrainer.dpo_loss
        get_batch_loss_metrics = DPOTrainer.get_batch_loss_metrics
        compute_ref_log_probs = DPOTrainer.compute_ref_log_probs
        null_ref_context = DPOTrainer.null_ref_context

        def concatenated_forward(self, model, batch, is_ref_model = False):
            if fused:
                out = module.unsloth_fused_dpo_concatenated_forward(self, model, batch, is_ref_model)
                assert out is not None, "fused path declined a supported case"
                return out
            return DPOTrainer.concatenated_forward(self, model, batch, is_ref_model = is_ref_model)

    trainer = Stub()
    loss_type = overrides.pop("loss_type", "sigmoid")
    if loss_type == "aot_pair" and "aot_unpaired" in __import__("inspect").getsource(DPOTrainer.dpo_loss):
        loss_type = "aot_unpaired"
    trainer.__dict__.update(
        accelerator = _Accelerator(), model = model, ref_model = ref_model,
        is_encoder_decoder = False, aux_loss_enabled = False, is_peft_model = False,
        ref_adapter_name = None, _peft_has_been_casted_to_bf16 = False,
        padding_value = PAD, pad_token_id = PAD, label_pad_token_id = -100,
        max_length = None, truncation_mode = "keep_end", use_logits_to_keep = False,
        padding_free = False, loss_type = loss_type if isinstance(loss_type, list) else [loss_type],
        loss_weights = None, beta = 0.1, label_smoothing = 0.0, reference_free = False,
        f_divergence_type = "reverse_kl", f_divergence_params = {}, use_weighting = False,
    )
    args = dict(use_liger_loss = False, use_liger_kernel = False, rpo_alpha = None, ld_alpha = None, discopop_tau = 0.05)
    for key in list(overrides):
        if key in args:
            args[key] = overrides.pop(key)
    trainer.args = types.SimpleNamespace(**args)
    trainer.__dict__.update(overrides)
    return trainer


def _run(trainer, model, batch):
    model.zero_grad(set_to_none = True)
    loss, metrics = trainer.get_batch_loss_metrics(model, dict(batch))
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    return loss.detach(), metrics, grads


def _assert_same(base, head, tol = 1e-9):
    (loss_b, metrics_b, grads_b), (loss_h, metrics_h, grads_h) = base, head
    assert torch.isfinite(loss_b)
    torch.testing.assert_close(loss_h, loss_b, rtol = tol, atol = tol)
    assert set(metrics_b) == set(metrics_h)
    for key in metrics_b:
        assert metrics_h[key] == pytest.approx(metrics_b[key], rel = tol, abs = tol), key
    assert set(grads_b) == set(grads_h)
    assert any(g.abs().sum() > 0 for g in grads_b.values())
    for name in grads_b:
        torch.testing.assert_close(grads_h[name], grads_b[name], rtol = tol, atol = tol, msg = name)


DPO_CASES = [
    dict(),
    dict(loss_type = "ipo"),
    dict(loss_type = "hinge"),
    dict(loss_type = "sigmoid", label_smoothing = 0.1),
    dict(loss_type = "robust", label_smoothing = 0.1),
    dict(loss_type = "exo_pair", label_smoothing = 0.1),
    dict(loss_type = "nca_pair"),
    dict(loss_type = "sppo_hard"),
    dict(loss_type = "apo_zero"),
    dict(loss_type = "apo_down"),
    dict(loss_type = "aot"),
    dict(loss_type = "aot_pair"),  # renamed aot_unpaired in TRL 0.28
    dict(loss_type = "discopop"),
    dict(loss_type = ["sigmoid", "sft"], loss_weights = [0.7, 0.3]),
    dict(rpo_alpha = 0.5),
    dict(ld_alpha = 0.5),
    dict(use_weighting = True),
    dict(beta = 2.0),
    dict(reference_free = True),
    dict(max_length = 7, truncation_mode = "keep_end"),
    dict(max_length = 7, truncation_mode = "keep_start"),
    dict(use_logits_to_keep = True),
    dict(padding_free = True),
    dict(softcap = 3.0),
    dict(bias = True),
]


@pytest.mark.parametrize("case", DPO_CASES, ids = lambda c: "-".join(f"{k}={v}" for k, v in c.items()) or "default")
@pytest.mark.parametrize("reference", ["ref_model", "precomputed"])
def test_dpo_matches_trl(dpo_module, monkeypatch, case, reference):
    case = dict(case)
    softcap = case.pop("softcap", 0.0)
    bias = case.pop("bias", False)
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, softcap))
    batch = _dpo_batch(seed = 3)
    results = []
    for fused in (False, True):
        model = _model(0, softcap, bias)
        ref_model = _model(1, softcap, bias)
        trainer = _dpo_trainer(dpo_module, model, ref_model, fused, **dict(case))
        run_batch = dict(batch)
        if reference == "precomputed":
            # TRL precompute_ref_log_probs stores these per row; reuse the reference pass of this arm.
            with torch.no_grad():
                ref_chosen, ref_rejected = trainer.compute_ref_log_probs(run_batch)
            run_batch["ref_chosen_logps"], run_batch["ref_rejected_logps"] = ref_chosen, ref_rejected
        HiddenStatesLlama.hidden_calls = 0
        results.append(_run(trainer, model, run_batch))
        if fused:
            assert HiddenStatesLlama.hidden_calls >= 1, "fused path never ran"
        else:
            assert HiddenStatesLlama.hidden_calls == 0
    _assert_same(*results)


def test_dpo_policy_equals_reference_gives_log2(dpo_module, monkeypatch):
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    model = _model(0)
    trainer = _dpo_trainer(dpo_module, model, _model(0), fused = True)
    loss, metrics = trainer.get_batch_loss_metrics(model, _dpo_batch(seed = 5))
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0, dtype = torch.float64)).item(), abs = 1e-12)
    assert metrics["rewards/margins"] == pytest.approx(0.0, abs = 1e-12)


def test_dpo_declines_vision_and_flag_off(dpo_module, monkeypatch):
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    model = _model(0)
    trainer = _dpo_trainer(dpo_module, model, _model(1), fused = True)
    batch = _dpo_batch(seed = 3)
    batch["pixel_values"] = torch.zeros(3, 1)
    assert dpo_module.unsloth_fused_dpo_concatenated_forward(trainer, model, batch) is None
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_LOSS", "0")
    assert dpo_module.unsloth_fused_dpo_concatenated_forward(trainer, model, _dpo_batch(seed = 3)) is None


def test_dpo_falls_back_to_logits_when_forward_ignores_flag(dpo_module, monkeypatch):
    """A forward that ignores UNSLOTH_RETURN_HIDDEN_STATES still gets TRL's numbers."""
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    monkeypatch.setattr(HiddenStatesLlama, "forward", lambda self, *a, **k: LlamaForCausalLM.forward(self, *a, **{x: y for x, y in k.items() if x != "use_cache"}))
    batch = _dpo_batch(seed = 3)
    results = []
    for fused in (False, True):
        model = _model(0)
        trainer = _dpo_trainer(dpo_module, model, _model(1), fused)
        results.append(_run(trainer, model, batch))
    _assert_same(*results)


def _orpo_batch(seed, n = 3):
    gen = torch.Generator().manual_seed(seed)
    batch = {}
    prompt_lengths = torch.randint(1, 5, (n,), generator = gen).tolist()
    prompts = [torch.randint(1, VOCAB, (p,), generator = gen) for p in prompt_lengths]
    for side, hi in (("chosen", 7), ("rejected", 5)):
        answers = [torch.randint(1, VOCAB, (int(torch.randint(1, hi + 1, (1,), generator = gen)),), generator = gen) for _ in range(n)]
        width = max(len(p) + len(a) for p, a in zip(prompts, answers))
        ids = torch.full((n, width), PAD, dtype = torch.long)
        mask = torch.zeros((n, width), dtype = torch.long)
        labels = torch.full((n, width), -100, dtype = torch.long)
        for i, (p, a) in enumerate(zip(prompts, answers)):
            row = torch.cat([p, a])
            ids[i, :len(row)] = row
            mask[i, :len(row)] = 1
            labels[i, len(p):len(row)] = a
        batch[f"{side}_input_ids"], batch[f"{side}_attention_mask"], batch[f"{side}_labels"] = ids, mask, labels
    return batch


def _orpo_trainer(module, model, method, fused, **overrides):
    if method == "orpo":
        try:
            from trl import ORPOTrainer as Trainer
        except ImportError:
            from trl.experimental.orpo import ORPOTrainer as Trainer
    else:
        try:
            from trl import CPOTrainer as Trainer
        except ImportError:
            from trl.experimental.cpo import CPOTrainer as Trainer

    class Stub:
        concatenated_inputs = staticmethod(Trainer.concatenated_inputs)
        get_batch_logps = staticmethod(Trainer.get_batch_logps)
        get_batch_loss_metrics = Trainer.get_batch_loss_metrics

        def concatenated_forward(self, model, batch):
            if fused:
                out = module.unsloth_fused_orpo_concatenated_forward(self, model, batch, method = method)
                if out is not None:
                    return out
                assert fused == "auto", "fused path declined a supported case"
            return Trainer.concatenated_forward(self, model, batch)

    if method == "orpo":
        Stub.odds_ratio_loss = Trainer.odds_ratio_loss
    else:
        Stub.cpo_loss = Trainer.cpo_loss
    trainer = Stub()
    trainer.__dict__.update(
        accelerator = _Accelerator(), is_encoder_decoder = False, aux_loss_enabled = False,
        padding_value = PAD, pad_token_id = PAD, label_pad_token_id = -100, beta = 0.1,
        loss_type = "sigmoid", cpo_alpha = 1.0, simpo_gamma = 0.5, label_smoothing = 0.0, alpha = 0.0,
    )
    trainer.__dict__.update(overrides)
    return trainer


ORPO_CASES = [
    ("orpo", dict()),
    ("orpo", dict(beta = 1.0)),
    ("orpo", dict(softcap = 3.0)),
    ("cpo", dict()),
    ("cpo", dict(loss_type = "hinge")),
    ("cpo", dict(loss_type = "ipo")),
    ("cpo", dict(loss_type = "simpo", cpo_alpha = 0.0)),
    ("cpo", dict(loss_type = "simpo", cpo_alpha = 0.5, label_smoothing = 0.1)),
    ("cpo", dict(alpha = 0.3)),
    ("cpo", dict(bias = True)),
]


@pytest.mark.parametrize("method, case", ORPO_CASES, ids = lambda c: str(c))
def test_orpo_cpo_match_trl(dpo_module, monkeypatch, method, case):
    case = dict(case)
    softcap = case.pop("softcap", 0.0)
    bias = case.pop("bias", False)
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, softcap))
    batch = _orpo_batch(seed = 4)
    results = []
    for fused in (False, True):
        model = _model(0, softcap, bias)
        trainer = _orpo_trainer(dpo_module, model, method, fused, **case)
        if method == "cpo" and case.get("alpha", 0.0) and not hasattr(type(trainer), "cpo_loss"):
            pytest.skip("CPO alpha needs TRL with AlphaPO")
        HiddenStatesLlama.hidden_calls = 0
        results.append(_run(trainer, model, batch))
        assert (HiddenStatesLlama.hidden_calls >= 1) == fused
    _assert_same(*results)


def test_orpo_under_ddp_find_unused_falls_back(dpo_module, monkeypatch, tmp_path):
    """DDP(find_unused_parameters = True) would mark the trainable head ready twice."""
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if dist.is_initialized():
        pytest.skip("a process group is already initialised")
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))

    class Unwrapping(_Accelerator):
        def unwrap_model(self, model):
            return getattr(model, "module", model)

    batch = _orpo_batch(seed = 4)
    reference = _run(_orpo_trainer(dpo_module, _model(0), "orpo", False), _model(0), batch)
    dist.init_process_group("gloo", init_method = f"file://{tmp_path / 'store'}", rank = 0, world_size = 1)
    try:
        for find_unused in (True, False):
            model = _model(0)
            ddp = DistributedDataParallel(model, find_unused_parameters = find_unused)
            trainer = _orpo_trainer(dpo_module, model, "orpo", "auto", accelerator = Unwrapping())
            HiddenStatesLlama.hidden_calls = 0
            model.zero_grad(set_to_none = True)
            loss, metrics, _ = _run(trainer, ddp, batch)
            grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
            assert (HiddenStatesLlama.hidden_calls >= 1) == (not find_unused)
            _assert_same(reference, (loss, metrics, grads))
    finally:
        dist.destroy_process_group()


def _real_hidden_states_helpers():
    text = (ROOT / "unsloth" / "models" / "rl_replacements.py").read_text()
    start = text.index("def _unsloth_grpo_returns_hidden_states(")
    end = text.index("_GRPO_HIDDEN_STATES_WIDTH_DISPATCH = ")
    package = types.ModuleType("_fake_unsloth_models")
    package.__path__ = []
    helpers = types.ModuleType("_fake_unsloth_models.rl_replacements")
    exec(text[start:end], helpers.__dict__)
    return package, helpers


@pytest.mark.parametrize("honours_flag", [False, True])
def test_square_head_needs_explicit_hidden_states_signal(dpo_module, monkeypatch, honours_flag):
    """vocab_size == hidden_size: logits look like hidden states, so only a marked forward is fused."""
    package, helpers = _real_hidden_states_helpers()
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, helpers.__name__, helpers)
    monkeypatch.setattr(dpo_module, "__package__", package.__name__)
    monkeypatch.setattr(dpo_module, "_logit_transforms", lambda unwrapped: (0.0, 0.0, 0.0))
    if honours_flag:
        monkeypatch.setattr(HiddenStatesLlama, "__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__", True, raising = False)
    else:
        monkeypatch.setattr(HiddenStatesLlama, "forward", lambda self, *a, **k: LlamaForCausalLM.forward(self, *a, **{x: y for x, y in k.items() if x != "use_cache"}))
    batch = _orpo_batch(seed = 4)
    results = []
    for fused in (False, "auto"):
        model = _model(0, vocab = 64, hidden = 64)
        trainer = _orpo_trainer(dpo_module, model, "orpo", fused)
        HiddenStatesLlama.hidden_calls = 0
        results.append(_run(trainer, model, batch))
        assert (HiddenStatesLlama.hidden_calls >= 1) == (honours_flag and bool(fused))
    _assert_same(*results)


def test_rl_replacement_injects_dispatch_only_when_enabled(monkeypatch):
    """The generated trainer is untouched unless the flag is set at patch time."""
    source = (
        "    def concatenated_forward(\n"
        "        self, model, batch, is_ref_model: bool = False\n"
        "    ) -> dict:\n"
        '        """Doc."""\n'
        "        num_examples = batch['prompt_input_ids'].shape[0]\n"
        "        return num_examples\n"
    )
    rl_path = ROOT / "unsloth" / "models" / "rl_replacements.py"
    text = rl_path.read_text()
    start = text.index("def _insert_after_docstring(")
    end = text.index('RL_FUNCTIONS["dpo_trainer"].append(dpo_trainer_fused_preference_forward)')
    namespace = {"os": os, "re": __import__("re"), "textwrap": __import__("textwrap")}
    exec(text[start:end], namespace)
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_LOSS", "0")
    assert namespace["dpo_trainer_fused_preference_forward"]("concatenated_forward", source) == source
    monkeypatch.setenv("UNSLOTH_FUSED_PREFERENCE_LOSS", "1")
    patched = namespace["dpo_trainer_fused_preference_forward"]("concatenated_forward", source)
    assert patched != source
    compile(__import__("textwrap").dedent(patched), "<patched>", "exec")
    lines = patched.split("\n")
    doc = lines.index('        """Doc."""')
    assert lines[doc + 1].strip().startswith("from unsloth.models.dpo import unsloth_fused_dpo_concatenated_forward")
    assert namespace["dpo_trainer_fused_preference_forward"]("concatenated_inputs", source) == source
