# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL 0.29 - 1.14 DPO ``_compute_loss`` scored from hidden states by unsloth_zoo's fused linear op: the rewritten
TRL body against TRL's own body on the same tiny model (loss, gradients and every logged metric), plus the legacy
(TRL 1.1 - 1.14) ``_ChunkedLogProbFunction`` signature for the fused-head adapter."""

from __future__ import annotations

import ast
import importlib.util
import os
import textwrap
import types
from collections import defaultdict

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl as unsloth_rl
import unsloth.models.rl_replacements as rl

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")

H, V = 16, 37


def _trl_compute_loss_source():
    """TRL's own ``DPOTrainer._compute_loss`` read from its file (importing unsloth rebinds DPOTrainer)."""
    pytest.importorskip("trl")
    spec = importlib.util.find_spec("trl.trainer.dpo_trainer")
    text = open(spec.origin, encoding = "utf-8").read()
    if "add_fused_lm_head" in text:
        pytest.skip(reason = "TRL >= 1.15 fuses the head itself")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_loss":
            return textwrap.dedent(ast.get_source_segment(text, node, padded = True))
    pytest.skip(reason = "TRL < 0.29 scores DPO in concatenated_forward")


class _TinyLM(torch.nn.Module):
    """Honours ``UNSLOTH_RETURN_HIDDEN_STATES`` like an Unsloth-compiled forward: ``.logits`` holds the hidden states."""

    is_gradient_checkpointing = False

    def __init__(self, seed):
        super().__init__()
        from transformers import LlamaConfig

        gen = torch.Generator().manual_seed(seed)
        self.config = LlamaConfig(
            vocab_size = V, hidden_size = H, num_hidden_layers = 1, num_attention_heads = 2
        )
        self.embed = torch.nn.Embedding(V, H)
        self.mix = torch.nn.Linear(H, H)
        self.lm_head = torch.nn.Linear(H, V, bias = False)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator = gen) * 0.7)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        use_cache = None,
        **kwargs,
    ):
        hidden = torch.tanh(self.mix(self.embed(input_ids)))
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
            return types.SimpleNamespace(logits = hidden)
        return types.SimpleNamespace(logits = self.lm_head(hidden))


class _Accelerator:
    device = torch.device("cpu")

    def unwrap_model(self, model):
        return model

    def gather_for_metrics(self, x):
        return x.reshape(-1) if torch.is_tensor(x) else x

    gather = gather_for_metrics


def _trainer(loss_types, use_weighting = False):
    model, ref = _TinyLM(0), _TinyLM(1)
    ref.requires_grad_(False)
    trainer = types.SimpleNamespace(
        model = model,
        ref_model = ref,
        accelerator = _Accelerator(),
        beta = 0.1,
        ld_alpha = None,
        loss_types = loss_types,
        loss_weights = [1.0] * len(loss_types),
        precompute_ref_logps = False,
        f_divergence_type = "reverse_kl",
        f_alpha_divergence_coef = 1.0,
        label_smoothing = 0.0,
        use_weighting = use_weighting,
        aux_loss_enabled = False,
        use_liger_kernel = False,
        is_fsdp_enabled = False,
        args = types.SimpleNamespace(gradient_checkpointing_kwargs = None, discopop_tau = 0.05),
        _metrics = {"train": defaultdict(list), "eval": defaultdict(list)},
        _total_train_tokens = 0,
        _tp_size = 1,
        padding_free = False,
        is_encoder_decoder = False,
    )
    trainer._truncate_inputs = lambda *xs: xs
    return trainer


def _inputs():
    gen = torch.Generator().manual_seed(3)
    input_ids = torch.randint(0, V, (4, 13), generator = gen)  # [chosen x2, rejected x2]
    completion_mask = torch.zeros_like(input_ids)
    for row, (start, end) in enumerate(((5, 13), (4, 10), (6, 12), (3, 13))):
        completion_mask[row, start:end] = 1
    # Every other next token is the policy's argmax (the model is per-position), so token accuracy is not 0.
    policy = _TinyLM(0)
    with torch.no_grad():
        for t in range(0, input_ids.shape[1] - 1, 2):
            input_ids[:, t + 1] = policy(input_ids[:, t : t + 1]).logits[:, 0].argmax(-1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 10:] = 0
    attention_mask[2, 12:] = 0
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
    }


def _bind(source):
    import trl.trainer.dpo_trainer as dpo

    ns = dict(vars(dpo))
    ns.update({name: getattr(rl, name) for name in dir(rl) if name.startswith("_unsloth_")})
    exec(compile(source, "<dpo_compute_loss>", "exec"), ns)
    return ns["_compute_loss"]


def _run(
    source,
    loss_types,
    use_weighting = False,
):
    trainer = _trainer(loss_types, use_weighting)
    loss = _bind(source)(trainer, trainer.model, _inputs(), False)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters()}
    metrics = {k: v[-1] for k, v in trainer._metrics["train"].items()}
    return loss.detach(), grads, metrics, trainer


def test_compute_loss_is_rewritten_once():
    source = _trl_compute_loss_source()
    new = rl.dpo_trainer_compute_loss_chunked("_compute_loss", source)
    assert new != source and "_unsloth_dpo_loss_stats" in new
    compile(new, "<dpo>", "exec")
    assert rl.dpo_trainer_compute_loss_chunked("_compute_loss", new) == new
    assert rl.dpo_trainer_compute_loss_chunked("compute_loss", source) == source
    # Another full-logits reader the rewrite does not know leaves TRL's body alone.
    extra = source.replace(
        "    return (loss, outputs)",
        "    _ = shift_logits.sum()\n    return (loss, outputs)",
    )
    assert extra != source and rl.dpo_trainer_compute_loss_chunked("_compute_loss", extra) == extra


@pytest.mark.parametrize("loss_types", [["sigmoid"], ["sigmoid", "sft"], ["ipo"]])
def test_rewritten_compute_loss_matches_trl(loss_types):
    source = _trl_compute_loss_source()
    new = rl.dpo_trainer_compute_loss_chunked("_compute_loss", source)
    want_loss, want_grads, want_metrics, _ = _run(source, loss_types)
    calls = fused_linear.CALLS["logprobs"]
    got_loss, got_grads, got_metrics, trainer = _run(new, loss_types)
    assert fused_linear.CALLS["logprobs"] == calls + 2
    assert trainer._unsloth_dpo_chunked_calls == 2
    assert not getattr(trainer, "_unsloth_dpo_chunked_fallbacks", {})
    torch.testing.assert_close(got_loss, want_loss, atol = 1e-5, rtol = 1e-5)
    for name, grad in want_grads.items():
        torch.testing.assert_close(got_grads[name], grad, atol = 1e-5, rtol = 1e-4, msg = name)
    assert set(got_metrics) == set(want_metrics)
    for key, value in want_metrics.items():
        assert got_metrics[key] == pytest.approx(value, rel = 1e-5, abs = 1e-5), key
    assert {
        "entropy",
        "logits/chosen",
        "logits/rejected",
        "mean_token_accuracy",
    } <= set(got_metrics)
    assert 0.0 < got_metrics["mean_token_accuracy"] < 1.0


@pytest.mark.parametrize("reason", ["use_weighting", "kill_switch"])
def test_rewritten_compute_loss_keeps_trl_path(reason, monkeypatch):
    source = _trl_compute_loss_source()
    new = rl.dpo_trainer_compute_loss_chunked("_compute_loss", source)
    weighting = reason == "use_weighting"
    if reason == "kill_switch":
        monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    want_loss, _, want_metrics, _ = _run(source, ["sigmoid"], use_weighting = weighting)
    calls = fused_linear.CALLS["logprobs"]
    got_loss, _, got_metrics, trainer = _run(new, ["sigmoid"], use_weighting = weighting)
    assert fused_linear.CALLS["logprobs"] == calls
    assert trainer._unsloth_dpo_chunked_fallbacks
    assert torch.equal(got_loss, want_loss) and got_metrics == want_metrics


def _run_with(source, setup):
    trainer = _trainer(["sigmoid"])
    setup(trainer)
    loss = _bind(source)(trainer, trainer.model, _inputs(), False)
    loss.backward()
    return loss.detach(), trainer


def _shared_head(trainer):
    trainer.ref_model = trainer.model


def _trainable_ref(trainer):
    trainer.ref_model.requires_grad_(True)


@pytest.mark.parametrize(
    "setup", [_shared_head, _trainable_ref], ids = ["shared_head", "trainable_ref"]
)
def test_reference_pass_gets_no_gradient(setup):
    """TRL scores the reference under no_grad: neither a trainable reference head nor the policy's own head (adapter
    disabled reference) may get a gradient from it."""
    source = _trl_compute_loss_source()
    new = rl.dpo_trainer_compute_loss_chunked("_compute_loss", source)
    want_loss, want = _run_with(source, setup)
    got_loss, got = _run_with(new, setup)
    assert got._unsloth_dpo_chunked_calls == 2
    torch.testing.assert_close(got_loss, want_loss, atol = 1e-5, rtol = 1e-5)
    for (name, w), g in zip(want.model.named_parameters(), got.model.parameters()):
        torch.testing.assert_close(g.grad, w.grad, atol = 1e-5, rtol = 1e-4, msg = name)
    if want.ref_model is not want.model:
        assert all(p.grad is None for p in want.ref_model.parameters())
        assert all(p.grad is None for p in got.ref_model.parameters())


def test_hooked_head_keeps_trl_path():
    """The fused op reads the head's weight directly: a forward hook on it would be skipped."""
    source = _trl_compute_loss_source()
    new = rl.dpo_trainer_compute_loss_chunked("_compute_loss", source)

    def setup(trainer):
        for model in (trainer.model, trainer.ref_model):
            model.lm_head.register_forward_hook(lambda module, args, output: output * 0.5)

    want_loss, _ = _run_with(source, setup)
    calls = fused_linear.CALLS["logprobs"]
    got_loss, trainer = _run_with(new, setup)
    assert fused_linear.CALLS["logprobs"] == calls
    assert "output head has hooks or a patched forward" in trainer._unsloth_dpo_chunked_fallbacks
    torch.testing.assert_close(got_loss, want_loss)


class _LegacyKernel:
    """TRL 1.1 - 1.14's ``_ChunkedLogProbFunction``: no ``outputs`` argument, returns ``(log_probs, entropy)``."""

    @staticmethod
    def forward(
        ctx,
        last_hidden,
        weight,
        bias,
        targets,
        temperature,
        chunk_size,
        final_logit_softcapping = None,
        logit_scale = 1.0,
    ):
        raise AssertionError("not reached")

    @staticmethod
    def apply(*args):
        assert len(args) == 8
        return "trl_logps", "trl_entropy"


def test_adapter_keeps_the_legacy_kernel_contract():
    gen = torch.Generator().manual_seed(7)
    hidden = torch.randn(9, H, generator = gen, requires_grad = True)
    weight = torch.randn(V, H, generator = gen)
    targets = torch.randint(0, V, (9,), generator = gen)
    adapter = unsloth_rl._UnslothFusedLinearHead(_LegacyKernel)
    assert adapter._legacy
    calls = fused_linear.CALLS["logprobs"]
    logps, entropy = adapter.apply(hidden, weight, None, targets, 0.8, 4096, None, 1.0)
    assert fused_linear.CALLS["logprobs"] == calls + 1
    log_p = (hidden.detach().double() @ weight.double().t() / 0.8).log_softmax(-1)
    torch.testing.assert_close(
        logps.detach().double(),
        log_p.gather(-1, targets[:, None]).squeeze(-1),
        atol = 1e-5,
        rtol = 1e-5,
    )
    torch.testing.assert_close(
        entropy.detach().double(), -(log_p.exp() * log_p).sum(-1), atol = 1e-5, rtol = 1e-5
    )
    (logps.sum() + entropy.sum()).backward()
    assert hidden.grad is not None and hidden.grad.isfinite().all()
    # A call it cannot reproduce still reaches TRL with TRL's 8 arguments.
    assert adapter.apply(hidden, weight, None, targets, 1.0, 4096, None, 0.0) == (
        "trl_logps",
        "trl_entropy",
    )
