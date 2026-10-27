# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""RLOOTrainer (TRL < 1.15) log-probs and entropies scored from hidden states by the zoo fused linear kernel.

TRL's own ``_get_per_token_logps_and_entropies`` is run on the same fake model as the reference (float64), so the
test pins the rewrite to whatever the installed TRL computes."""

from __future__ import annotations

import copy
import os
import sys
import textwrap
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401,E402
import unsloth.models.rl_replacements as rl  # noqa: E402

fused_linear = pytest.importorskip("unsloth_zoo.fused_losses.fused_linear")
rloo = pytest.importorskip("trl.trainer.rloo_trainer")

H, V = 16, 40


class _Model(torch.nn.Module):
    """Unsloth-style CausalLM: ``.logits`` is the last hidden state under UNSLOTH_RETURN_HIDDEN_STATES=1."""

    def __init__(
        self,
        dtype = torch.float64,
        softcap = None,
    ):
        super().__init__()
        from transformers import Qwen3Config

        self.config = Qwen3Config(hidden_size = H, vocab_size = V, num_hidden_layers = 1)
        if softcap:
            self.config.final_logit_softcapping = softcap
        gen = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(V, H).to(dtype)
        self.embed.weight.data = torch.randn(V, H, generator = gen, dtype = dtype)
        self.mix = torch.nn.Linear(H, H, dtype = dtype)
        self.mix.weight.data = torch.randn(H, H, generator = gen, dtype = dtype) / H**0.5
        self.mix.bias.data = torch.zeros(H, dtype = dtype)
        self.lm_head = torch.nn.Linear(H, V, bias = False, dtype = dtype)
        self.lm_head.weight.data = torch.randn(V, H, generator = gen, dtype = dtype)
        self.softcap = softcap
        self.forwards = []

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask = None,
        logits_to_keep = 0,
        use_cache = None,
        **kwargs,
    ):
        h = torch.tanh(self.mix(self.embed(input_ids)))
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        hidden = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
        self.forwards.append("hidden" if hidden else "logits")
        if hidden:
            return SimpleNamespace(logits = h)
        z = self.lm_head(h)
        if self.softcap:
            z = self.softcap * torch.tanh(z / self.softcap)
        return SimpleNamespace(logits = z)


class _Accel:
    def unwrap_model(self, model, **kwargs):
        return model


def _trl_source(name):
    # TRL's own text: importing unsloth swaps trl.trainer.rloo_trainer.RLOOTrainer for the generated class.
    import ast

    text = open(rloo.__file__, encoding = "utf-8").read()
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "RLOOTrainer"
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return "\n".join(text.split("\n")[fn.lineno - 1 : fn.end_lineno]) + "\n"


def _methods():
    stock = _trl_source("_get_per_token_logps_and_entropies")
    if "fused_lm_head=True" in stock:
        pytest.skip("TRL >= 1.15 scores through its own fused lm_head (served by the zoo adapter)")
    new = rl.rloo_trainer_fused_logprobs("_get_per_token_logps_and_entropies", stock)
    assert new != stock, "RLOO rewrite did not apply to the installed TRL"
    namespace = dict(vars(rloo))
    exec(compile("class _T:\n" + new, "<rloo>", "exec"), namespace)
    exec(compile("class _S:\n" + stock, "<rloo_stock>", "exec"), namespace)
    return namespace["_T"], namespace["_S"]


def _trainer(cls, temperature = 0.7):
    t = cls.__new__(cls)
    t.temperature = temperature
    t.model_kwarg_keys = ("input_ids", "attention_mask", "logits_to_keep")
    t.accelerator = _Accel()
    return t


def _batch(
    B = 3,
    L = 9,
    k = 5,
):
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, V, (B, L), generator = gen)
    mask = torch.ones(B, L, dtype = torch.long)
    mask[0, :2] = 0
    mask[1, -2:] = 0  # padded completion
    return ids, mask, k


def _call(t, model, entropy):
    ids, mask, k = _batch()
    out = t._get_per_token_logps_and_entropies(
        model, ids, mask, k, batch_size = 2, compute_entropy = entropy
    )
    return out[0], out[1], out


@pytest.mark.parametrize("softcap", [None, 3.0])
def test_rloo_fused_matches_trl_fp64(softcap):
    T, S = _methods()
    model = _Model(softcap = softcap)
    calls = fused_linear.CALLS["logprobs"]
    logps, ent, out = _call(_trainer(T), model, True)
    assert fused_linear.CALLS["logprobs"] > calls
    assert "hidden" in model.forwards and "logits" not in model.forwards
    ref_logps, ref_ent, ref_out = _call(_trainer(S), model, True)
    assert len(out) == len(ref_out)
    assert logps.dtype == torch.float32 or logps.dtype == ref_logps.dtype
    torch.testing.assert_close(logps.double(), ref_logps.double(), atol = 1e-5, rtol = 1e-5)
    torch.testing.assert_close(ent.double(), ref_ent.double(), atol = 1e-5, rtol = 1e-5)
    assert not ent.requires_grad


def test_rloo_fused_gradients_match_trl_fp64():
    T, S = _methods()
    model = _Model()
    ids, mask, k = _batch()
    grads = []
    for cls in (T, S):
        model.zero_grad()
        out = _trainer(cls)._get_per_token_logps_and_entropies(
            model, ids, mask, k, compute_entropy = True
        )
        cm = mask[:, -k:].double()
        loss = -((out[0].double() * torch.linspace(-1, 1, k)).mul(cm).sum() / cm.sum())
        loss.backward()
        grads.append(
            (
                loss.detach(),
                [p.grad.detach().clone() for p in (model.mix.weight, model.embed.weight)],
            )
        )
    torch.testing.assert_close(grads[0][0], grads[1][0], atol = 1e-6, rtol = 1e-6)
    for a, b in zip(grads[0][1], grads[1][1]):
        torch.testing.assert_close(a.double(), b.double(), atol = 1e-6, rtol = 1e-5)


def test_rloo_fused_bf16_error_not_worse_than_stock():
    T, S = _methods()
    exact = _Model(torch.float64)
    ref, _, _ = _call(_trainer(S), exact, False)
    low = copy.deepcopy(exact).to(torch.bfloat16)
    fused, _, _ = _call(_trainer(T), low, False)
    stock, _, _ = _call(_trainer(S), low, False)
    keep = _batch()[1][:, -_batch()[2] :].bool()
    # Mean, not max: one bf16-rounded hidden row dominates a max either way.
    err_fused = (fused.double() - ref)[keep].abs().mean()
    err_stock = (stock.double() - ref)[keep].abs().mean()
    assert err_fused <= err_stock + 1e-6, (err_fused, err_stock)


def test_rloo_kill_switch_and_fallbacks(monkeypatch):
    T, S = _methods()
    model = _Model()
    monkeypatch.setenv("UNSLOTH_FUSED_LINEAR", "0")
    calls = fused_linear.CALLS["logprobs"]
    t = _trainer(T)
    logps, ent, _ = _call(t, model, True)
    assert fused_linear.CALLS["logprobs"] == calls
    assert model.forwards == ["logits", "logits"]
    assert "UNSLOTH_FUSED_LINEAR=0" in t._unsloth_rloo_fused_fallbacks
    monkeypatch.delenv("UNSLOTH_FUSED_LINEAR")
    # Vision inputs keep TRL's body.
    if "pixel_values" in _trl_source("_get_per_token_logps_and_entropies").split(")", 1)[0]:
        t = _trainer(T)
        ids, mask, k = _batch()
        t._get_per_token_logps_and_entropies(model, ids, mask, k, pixel_values = torch.zeros(3, 1))
        assert t._unsloth_rloo_fused_fallbacks == {"vision inputs": 1}


def test_rloo_forward_ignoring_the_flag_hands_back_to_trl():
    T, S = _methods()

    class _Plain(_Model):
        def forward(self, *a, **kw):
            old = os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
            try:
                return super().forward(*a, **kw)
            finally:
                if old is not None:
                    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = old

    model = _Plain()
    t = _trainer(T)
    logps, _, _ = _call(t, model, False)
    ref, _, _ = _call(_trainer(S), model, False)
    torch.testing.assert_close(logps, ref)
    assert t._unsloth_rloo_fused_off is True


def test_rloo_rewrite_idempotent_and_scoped():
    stock = _trl_source("_get_per_token_logps_and_entropies")
    if "fused_lm_head=True" in stock:
        pytest.skip("TRL >= 1.15")
    new = rl.rloo_trainer_fused_logprobs("_get_per_token_logps_and_entropies", stock)
    assert rl.rloo_trainer_fused_logprobs("_get_per_token_logps_and_entropies", new) == new
    assert rl.rloo_trainer_fused_logprobs("_compute_loss", stock) == stock
    assert rl.rloo_trainer_fused_logprobs in rl.RL_FUNCTIONS["rloo_trainer"]


def test_rloo_fused_scorer_compiles_without_graph_breaks():
    from unsloth.models._fused_rl_logprobs import score_hidden_states

    model = _Model(torch.float32)
    head_info = (model.lm_head, (0.0, 0.0, 0.0), None)
    ids, mask, k = _batch()
    hidden = torch.randn(3, k, H)

    def fn(h):
        return score_hidden_states(model, h, ids[:, -k:], 0.7, entropy = True, head_info = head_info)

    torch._dynamo.reset()
    explain = torch._dynamo.explain(fn)(hidden)
    assert explain.graph_break_count == 0, explain.break_reasons
