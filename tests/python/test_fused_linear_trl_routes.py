# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""TRL >= 1.15's fused lm_head served by unsloth_zoo's fused linear kernel, and KTO's no-grad KL pass scored from
hidden states."""

from __future__ import annotations

import inspect
import os
import sys
import textwrap
import types

import pytest
import torch

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth.models.rl as unsloth_rl
import unsloth.models.rl_replacements as rl

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")

N, H, V = 11, 8, 29


class _Kernel:
    """Stands in for TRL's ChunkedLogProbFunction and records every call it serves."""

    _calls = 0

    @classmethod
    def apply(cls, *args):
        cls._calls += 1
        return ("trl",) * 5


def _reference(hidden, weight, bias, targets, temperature, softcap, logit_scale):
    """TRL's outputs in float64: scale, softcap, then temperature; mean_logits after the temperature."""
    z = hidden.double() @ weight.double().t() + bias.double()
    z = z * logit_scale
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    z = z / temperature
    log_p = z.log_softmax(-1)
    logps = log_p.gather(-1, targets[:, None]).squeeze(-1)
    entropy = -(log_p.exp() * log_p).sum(-1)
    lssp = (2 * log_p).logsumexp(-1)
    mean = z.mean(-1)
    top1 = z.argmax(-1) == targets
    return logps, entropy, lssp, mean, top1


def _inputs(seed = 0):
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(N, H, generator = gen, requires_grad = True)
    weight = torch.randn(V, H, generator = gen)
    bias = torch.randn(V, generator = gen)
    targets = torch.randint(0, V, (N,), generator = gen)
    return hidden, weight, bias, targets


def test_adapter_matches_trl_semantics():
    hidden, weight, bias, targets = _inputs()
    adapter = unsloth_rl._UnslothFusedLinearHead(_Kernel)
    before = _Kernel._calls
    calls = fused_linear.CALLS["logprobs"]
    got = adapter.apply(
        hidden,
        weight,
        bias,
        targets,
        0.7,
        4096,
        5.0,
        1.5,
        ("log_probs", "entropy", "log_sum_sq_probs", "mean_logits", "is_top1"),
    )
    assert _Kernel._calls == before and fused_linear.CALLS["logprobs"] == calls + 1
    want = _reference(hidden.detach(), weight, bias, targets, 0.7, 5.0, 1.5)
    for g, w in zip(got[:4], want[:4]):
        torch.testing.assert_close(g.detach().double(), w, atol = 1e-5, rtol = 1e-5)
    assert torch.equal(got[4], want[4])
    (got[0].sum() + got[1].sum()).backward()
    assert hidden.grad is not None and hidden.grad.isfinite().all()


def test_adapter_returns_none_for_unrequested_outputs():
    hidden, weight, bias, targets = _inputs(1)
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, None, targets, 1.0, 4096, None, 1.0, ()
    )
    assert len(got) == 5 and got[0] is not None and all(x is None for x in got[1:])


@pytest.mark.parametrize("case", ["zero_scale", "fp32_hidden_half_head", "unknown_output"])
def test_adapter_hands_unreproduced_calls_to_trl(case):
    hidden, weight, bias, targets = _inputs(2)
    scale, outputs = 1.0, ("log_probs",)
    if case == "zero_scale":
        scale = 0.0
    elif case == "fp32_hidden_half_head":
        weight = weight.bfloat16()
    else:
        outputs = ("log_probs", "something_new")
    before = _Kernel._calls
    got = unsloth_rl._UnslothFusedLinearHead(_Kernel).apply(
        hidden, weight, bias, targets, 1.0, 4096, None, scale, outputs
    )
    assert _Kernel._calls == before + 1 and got[0] == "trl"


def test_patch_is_idempotent(monkeypatch):
    utils = types.ModuleType("trl.trainer.utils")
    utils._ChunkedLogProbFunction = _Kernel
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    trainer = sys.modules.get("trl.trainer")
    if trainer is not None:
        monkeypatch.setattr(trainer, "utils", utils, raising = False)
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    first = utils._ChunkedLogProbFunction
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    assert (
        isinstance(first, unsloth_rl._UnslothFusedLinearHead)
        and utils._ChunkedLogProbFunction is first
    )
    assert first.function is _Kernel


def test_patch_respects_the_kill_switch(monkeypatch):
    utils = types.ModuleType("trl.trainer.utils")
    utils._ChunkedLogProbFunction = _Kernel
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    trainer = sys.modules.get("trl.trainer")
    if trainer is not None:
        monkeypatch.setattr(trainer, "utils", utils, raising = False)
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    assert utils._ChunkedLogProbFunction is _Kernel


class _NoBiasKernel(_Kernel):
    """TRL 1.1 - 1.13's ``_ChunkedLogProbFunction``: no ``bias`` (and no ``outputs``) argument."""

    @staticmethod
    def forward(
        ctx,
        last_hidden,
        weight,
        targets,
        temperature,
        chunk_size,
        final_logit_softcapping = None,
        logit_scale = 1.0,
    ):
        raise AssertionError("not reached")


def test_patch_leaves_kernels_without_bias(monkeypatch):
    """The adapter's positional layout has ``bias`` third: wrapping an older kernel would shift every argument."""
    utils = types.ModuleType("trl.trainer.utils")
    utils._ChunkedLogProbFunction = _NoBiasKernel
    monkeypatch.setitem(sys.modules, "trl.trainer.utils", utils)
    trainer = sys.modules.get("trl.trainer")
    if trainer is not None:
        monkeypatch.setattr(trainer, "utils", utils, raising = False)
    unsloth_rl.patch_trl_fused_lm_head_kernel()
    assert utils._ChunkedLogProbFunction is _NoBiasKernel


class _HeadLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import LlamaConfig

        self.config = LlamaConfig(
            vocab_size = V, hidden_size = H, num_hidden_layers = 1, num_attention_heads = 2
        )
        self.lm_head = torch.nn.Linear(H, V, bias = False)

    def get_output_embeddings(self):
        return self.lm_head


def _selector_trainer():
    return types.SimpleNamespace(accelerator = types.SimpleNamespace(unwrap_model = lambda m: m))


def _selected(selector, trainer, model):
    if selector == "orpo_cpo":
        return rl._unsloth_pref_chunked_head(trainer, model) is not None
    return rl._unsloth_dpo_chunked_head(trainer, model, {}) is not None


@pytest.mark.parametrize("selector", ["orpo_cpo", "dpo_kto"])
def test_preference_selectors_respect_the_kill_switch(selector, monkeypatch):
    model, trainer = _HeadLM(), _selector_trainer()
    assert _selected(selector, trainer, model)
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    assert not _selected(selector, trainer, model)


@pytest.mark.parametrize("selector", ["orpo_cpo", "dpo_kto"])
@pytest.mark.parametrize("hook", ["forward_hook", "forward_pre_hook", "patched_forward"])
def test_preference_selectors_reject_hooked_heads(selector, hook):
    """The chunked / fused kernels read the head's weight directly, skipping whatever its forward would add."""
    model, trainer = _HeadLM(), _selector_trainer()
    assert _selected(selector, trainer, model)
    head = model.lm_head
    if hook == "forward_hook":
        head.register_forward_hook(lambda module, args, output: output * 0.5)
    elif hook == "forward_pre_hook":
        head.register_forward_pre_hook(lambda module, args: (args[0] * 2,))
    else:
        head.forward = lambda x: torch.nn.functional.linear(x, head.weight) * 0.5
    assert not _selected(selector, trainer, model)


def _kto_kl_source():
    """TRL's own _compute_kl_logps, read from its file: importing unsloth rebinds KTOTrainer to the generated class."""
    import ast
    import importlib.util

    pytest.importorskip("trl")
    for name in ("trl.trainer.kto_trainer", "trl.experimental.kto.kto_trainer"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception:
            spec = None
        if spec is None or spec.origin is None:
            continue
        text = open(spec.origin, encoding = "utf-8").read()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name == "_compute_kl_logps":
                return textwrap.dedent(ast.get_source_segment(text, node, padded = True))
    pytest.skip(reason = "this TRL computes KTO's KL term elsewhere")


def test_kto_kl_pass_is_rewritten_once():
    source = _kto_kl_source()
    from packaging.version import Version
    import trl

    if Version(trl.__version__) >= Version("1.6"):
        pytest.skip(reason = "the KL rewrite targets TRL < 1.6, whose KL pass calls get_batch_logps")
    new = rl.kto_trainer_chunked_kl_logps("_compute_kl_logps", source)
    assert new != source and "_unsloth_kto_hidden_logps" in new
    compile(new, "<kto_kl>", "exec")
    assert rl.kto_trainer_chunked_kl_logps("_compute_kl_logps", new) == new
    assert rl.kto_trainer_chunked_kl_logps("forward", source) == source
    assert rl.kto_trainer_chunked_kl_logps in rl.RL_FUNCTIONS["kto_trainer"]


def test_kto_scores_completion_tokens_only():
    """The KTO helper sends only completion positions to the head and keeps the per-position vocab sums exact."""
    gen = torch.Generator().manual_seed(5)
    hidden = torch.randn(2, 9, H, generator = gen)
    head = torch.nn.Linear(H, V, bias = False)
    labels = torch.randint(0, V, (2, 9), generator = gen)
    labels[:, :4] = -100
    before = fused_linear.CALLS["logprobs"]
    logps, sums = rl._unsloth_kto_hidden_logps(
        types.SimpleNamespace(), (None, head, (0.0, 0.0, 0.0)), hidden, labels, -100
    )
    assert fused_linear.CALLS["logprobs"] == before + 1
    z = hidden.double() @ head.weight.double().t()
    log_p = z[:, :-1].log_softmax(-1)
    shifted = labels[:, 1:]
    mask = shifted != -100
    want = (log_p.gather(-1, shifted.clamp(min = 0)[..., None]).squeeze(-1) * mask).sum(-1)
    torch.testing.assert_close(logps.detach().double(), want, atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(sums.squeeze(-1).double(), z.sum(-1), atol = 1e-3, rtol = 1e-5)
