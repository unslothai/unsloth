# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""PPOTrainer's rollout, reference and training log-probs (and entropy metric) scored from hidden states by the zoo
fused linear kernel; the value head keeps reading the critic backbone."""

from __future__ import annotations

import ast
import copy
import importlib
import os
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401,E402
import unsloth.models.rl_replacements as rl  # noqa: E402
from unsloth.models import _fused_rl_logprobs as frl  # noqa: E402

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")


def _ppo_module():
    for name in ("trl.trainer.ppo_trainer", "trl.experimental.ppo.ppo_trainer"):
        try:
            module = importlib.import_module(name)
        except Exception:
            continue
        if "class PPOTrainer" in open(module.__file__, encoding = "utf-8").read():
            return module
    pytest.skip("this TRL has no PPOTrainer")


def _trl_train_source(module):
    text = open(module.__file__, encoding = "utf-8").read()
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "PPOTrainer"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "train")
    return "\n".join(text.split("\n")[fn.lineno - 1 : fn.end_lineno]) + "\n"


H, V, CL = 16, 40, 4


class _Policy(torch.nn.Module):
    def __init__(
        self,
        dtype = torch.float64,
        honours_flag = True,
    ):
        super().__init__()
        from transformers import Qwen3Config

        self.config = Qwen3Config(hidden_size = H, vocab_size = V, num_hidden_layers = 1)
        gen = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(V, H, dtype = dtype)
        self.embed.weight.data = torch.randn(V, H, generator = gen, dtype = dtype)
        self.mix = torch.nn.Linear(H, H, dtype = dtype)
        self.mix.weight.data = torch.randn(H, H, generator = gen, dtype = dtype) / H**0.5
        self.mix.bias.data = torch.zeros(H, dtype = dtype)
        self.lm_head = torch.nn.Linear(H, V, bias = False, dtype = dtype)
        self.lm_head.weight.data = torch.randn(V, H, generator = gen, dtype = dtype)
        self.honours_flag = honours_flag
        self.forwards = []

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask = None,
        **kwargs,
    ):
        h = torch.tanh(self.mix(self.embed(input_ids)))
        hidden = self.honours_flag and os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
        self.forwards.append("hidden" if hidden else "logits")
        return SimpleNamespace(logits = h if hidden else self.lm_head(h), hidden_states = (h,))

    def generate(
        self,
        input_ids,
        attention_mask,
        generation_config = None,
        return_dict_in_generate = False,
        output_scores = False,
        **kwargs,
    ):
        new = (input_ids[:, -1:] * 7 + torch.arange(1, 6)) % V
        out = SimpleNamespace(sequences = torch.cat([input_ids, new], 1))
        if output_scores:
            out.scores = [
                torch.randn(input_ids.shape[0], V, dtype = torch.float64) for _ in range(new.shape[1])
            ]
        return out


class _Value(torch.nn.Module):
    base_model_prefix = "backbone"

    def __init__(self):
        super().__init__()
        self.backbone = _Policy()
        self.score = torch.nn.Linear(H, 1, dtype = torch.float64)


def _wrapper(module, policy):
    value = _Value()

    class _Backbone(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, **kwargs):
            out = self.inner(**kwargs)
            return SimpleNamespace(hidden_states = out.hidden_states)

    value.backbone = _Backbone(value.backbone)
    policy.is_gradient_checkpointing = False
    return module.PolicyAndValueWrapper(policy, value)


def _route(model):
    trainer = SimpleNamespace(
        model = model,
        accelerator = SimpleNamespace(unwrap_model = lambda m, **k: m),
        is_fsdp_enabled = False,
        is_deepspeed_enabled = False,
    )
    return frl.ppo_route(trainer), trainer


def _query_responses(B = 3, L = 10):
    gen = torch.Generator().manual_seed(2)
    qr = torch.randint(1, V, (B, L), generator = gen)
    qr[0, 0] = 0  # left pad
    return qr


def test_ppo_train_rewrite_applies_to_installed_trl():
    module = _ppo_module()
    source = _trl_train_source(module)
    new = rl.ppo_trainer_fused_logprobs("train", source)
    assert new != source, "PPO train rewrite did not apply to the installed TRL"
    ast.parse("class A:\n" + new)
    for needle in (
        "_unsloth_ppo_route = _unsloth_ppo_route_of(self)",
        "batch_generation if _unsloth_ppo_route is None else _unsloth_ppo_batch_generation",
        "logprob = _unsloth_ppo_policy_logprobs(",
        "ref_output = _unsloth_ppo_forward(",
        "output, vpred_temp = _unsloth_ppo_forward(",
        "new_logprobs, _unsloth_entropy = _unsloth_scored",
        "entropy = _unsloth_entropy",
    ):
        assert needle in new, needle
    for line in (
        "new_logprobs = selective_log_softmax(logits, mb_responses)",
        "ref_logprob = selective_log_softmax(ref_logits, response)",
        "logprob = selective_log_softmax(logits, response)",
    ):
        assert line in new
    assert rl.ppo_trainer_fused_logprobs("train", new) == new
    assert rl.ppo_trainer_fused_logprobs("generate_completions", source) == source
    drifted = source.replace(
        "new_logprobs = selective_log_softmax(logits, mb_responses)", "new_logprobs = f(logits)"
    )
    assert rl.ppo_trainer_fused_logprobs("train", drifted) == drifted
    assert rl.ppo_trainer_fused_logprobs in rl.RL_FUNCTIONS["ppo_trainer"]


def _stock_train_logps(module, model, qr, T):
    output, vpred_temp = module.forward(model, qr, 0)
    logits = output.logits[:, CL - 1 : -1]
    logits /= T
    logps = module.selective_log_softmax(logits, qr[:, CL:])
    prob_dist = torch.nn.functional.softmax(logits, dim = -1)
    entropy = torch.logsumexp(logits, dim = -1) - torch.sum(prob_dist * logits, dim = -1)
    return logps, entropy.detach(), vpred_temp


def _fused_train_logps(module, route, model, qr, T):
    output, vpred_temp = frl.ppo_forward(route, module.forward, model, qr, 0)
    out = frl.ppo_score(route, model, output.logits[:, CL - 1 : -1], qr[:, CL:], T, entropy = True)
    assert out is not None
    return out[0], out[1], vpred_temp


def test_ppo_training_pass_matches_trl_fp64():
    module = _ppo_module()
    T = 0.7 + 1e-7
    model = _wrapper(module, _Policy())
    route, trainer = _route(model)
    assert route is not None
    qr = _query_responses()
    calls = fused_linear.CALLS["logprobs"]
    results = []
    for fn in (
        lambda: _fused_train_logps(module, route, model, qr, T),
        lambda: _stock_train_logps(module, model, qr, T),
    ):
        model.zero_grad()
        logps, ent, vpred = fn()
        adv = torch.linspace(-1, 1, logps.shape[1], dtype = torch.float64)
        ratio = torch.exp(logps - logps.detach())
        loss = (-adv * ratio).mean() + 0.1 * vpred.square().mean()
        loss.backward()
        grads = [
            p.grad.detach().clone()
            for p in (
                model.policy.mix.weight,
                model.policy.embed.weight,
                model.value_model.score.weight,
            )
        ]
        results.append((logps.detach(), ent, vpred.detach(), loss.detach(), grads))
    assert fused_linear.CALLS["logprobs"] > calls and trainer._unsloth_ppo_fused_calls == 1
    (a, b) = results
    for x, y in zip(a[:4], b[:4]):
        torch.testing.assert_close(x.double(), y.double(), atol = 1e-6, rtol = 1e-6)
    for x, y in zip(a[4], b[4]):
        torch.testing.assert_close(x.double(), y.double(), atol = 1e-6, rtol = 1e-5)
    assert not a[1].requires_grad


def test_ppo_rollout_and_reference_logprobs_match_trl_fp64():
    module = _ppo_module()
    T = 0.9 + 1e-7
    policy = _Policy()
    model = _wrapper(module, policy)
    route, _ = _route(model)
    qr = _query_responses()
    with torch.no_grad():
        got = frl.ppo_policy_logprobs(
            route, module.forward, module.selective_log_softmax, model, qr, qr[:, CL:], CL, 0, T
        )
        logits = module.forward(policy, qr, 0).logits[:, CL - 1 : -1] / T
        want = module.selective_log_softmax(logits, qr[:, CL:])
    torch.testing.assert_close(got.double(), want.double(), atol = 1e-6, rtol = 1e-6)
    # A reference model whose forward ignores the flag: TRL's lines run on its real logits.
    ref = _Policy(honours_flag = False)
    out = frl.ppo_forward(route, module.forward, ref, qr, 0)
    assert frl.ppo_score(route, ref, out.logits[:, CL - 1 : -1], qr[:, CL:], T) is None
    got = frl.ppo_policy_logprobs(
        route, module.forward, module.selective_log_softmax, ref, qr, qr[:, CL:], CL, 0, T
    )
    want = module.selective_log_softmax(
        module.forward(ref, qr, 0).logits[:, CL - 1 : -1] / T, qr[:, CL:]
    )
    torch.testing.assert_close(got, want)


def test_ppo_bf16_training_logps_error_not_worse_than_stock():
    module = _ppo_module()
    T = 0.7 + 1e-7
    exact = _wrapper(module, _Policy())
    qr = _query_responses()
    ref, _, _ = _stock_train_logps(module, exact, qr, T)
    low = copy.deepcopy(exact).to(torch.bfloat16)
    route, _ = _route(low)
    fused, _, _ = _fused_train_logps(module, route, low, qr, T)
    stock, _, _ = _stock_train_logps(module, low, qr, T)
    # Mean, not max: one bf16-rounded hidden row dominates a max either way.
    err_fused = (fused.double() - ref.double()).abs().mean()
    err_stock = (stock.double() - ref.double()).abs().mean()
    assert err_fused <= err_stock + 1e-6, (err_fused, err_stock)


def test_ppo_generation_without_scores_matches_trl():
    module = _ppo_module()
    batch_generation = getattr(module, "batch_generation", None)
    if batch_generation is None:
        from trl.trainer.utils import batch_generation
    policy = _Policy()
    queries = _query_responses(B = 5, L = 4)
    want, logitss = batch_generation(policy, queries, 2, 0, None)
    got, none = frl.ppo_batch_generation(policy, queries, 2, 0, None)
    assert none is None and logitss.shape[-1] == V
    torch.testing.assert_close(got, want)


def test_ppo_kill_switch_keeps_trl(monkeypatch):
    module = _ppo_module()
    model = _wrapper(module, _Policy())
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    route, trainer = _route(model)
    assert route is None
    assert trainer._unsloth_ppo_fused_fallbacks == {"UNSLOTH_FUSED_LINEAR=0": 1}
    calls = fused_linear.CALLS["logprobs"]
    model.policy.forwards.clear()
    output, _ = frl.ppo_forward(route, module.forward, model, _query_responses(), 0)
    assert output.logits.shape[-1] == V and model.policy.forwards == ["logits"]
    assert frl.ppo_score(route, model, output.logits, _query_responses(), 1.0) is None
    assert fused_linear.CALLS["logprobs"] == calls


def test_ppo_reference_with_hooked_head_keeps_trl():
    module = _ppo_module()
    T = 0.9 + 1e-7
    route, trainer = _route(_wrapper(module, _Policy()))
    assert route is not None
    ref = _Policy()
    ref.lm_head.register_forward_hook(lambda *a: None)
    qr = _query_responses()
    with torch.no_grad():
        out = frl.ppo_forward(route, module.forward, ref, qr, 0)
        assert ref.forwards == ["logits"] and out.logits.shape[-1] == V
        assert frl.ppo_score(route, ref, out.logits[:, CL - 1 : -1], qr[:, CL:], T) is None
        got = frl.ppo_policy_logprobs(
            route, module.forward, module.selective_log_softmax, ref, qr, qr[:, CL:], CL, 0, T
        )
        want = module.selective_log_softmax(
            module.forward(ref, qr, 0).logits[:, CL - 1 : -1] / T, qr[:, CL:]
        )
    torch.testing.assert_close(got, want)
    assert "output head has hooks or a patched forward" in trainer._unsloth_ppo_fused_fallbacks


def test_ppo_hooked_head_keeps_trl():
    module = _ppo_module()
    model = _wrapper(module, _Policy())
    model.policy.lm_head.register_forward_hook(lambda *a: None)
    route, trainer = _route(model)
    assert route is None
    assert "output head has hooks or a patched forward" in trainer._unsloth_ppo_fused_fallbacks
