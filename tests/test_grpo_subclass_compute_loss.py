# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""GRPOTrainer subclasses that override ``_compute_loss`` (TRL's GMPOTrainer, user objectives) train their own
objective under Unsloth instead of the fused GRPO loss.

Unsloth's generated GRPO ``compute_loss`` is executed from its real source in a stand-in class; TRL's GMPO
``_compute_loss`` is executed from TRL's own source (read with ``ast``, since importing unsloth may rebind TRL's
classes). The float64 oracle is the GMPO objective written out independently of TRL (arXiv 2507.20673, eq. 7, as
TRL implements it: token clipping in log space, sign-aware, geometric mean over the completion, mean over rows)."""

from __future__ import annotations

import ast
import importlib
import os
import textwrap
from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401,E402
import unsloth.models.rl_replacements as rl  # noqa: E402

H, V = 16, 40
EPS_LOW, EPS_HIGH = 0.4, 0.4


class _Model(torch.nn.Module):
    """Unsloth-style CausalLM: ``.logits`` is the last hidden state under UNSLOTH_RETURN_HIDDEN_STATES=1 when
    ``honours_flag`` is set; otherwise it always returns real logits (a forward without hidden-state support)."""

    def __init__(
        self,
        honours_flag = True,
        dtype = torch.float64,
    ):
        super().__init__()
        from transformers import Qwen3Config

        self.config = Qwen3Config(hidden_size = H, vocab_size = V, num_hidden_layers = 1)
        gen = torch.Generator().manual_seed(0)
        self.embed = torch.nn.Embedding(V, H).to(dtype)
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
        logits_to_keep = 0,
        use_cache = None,
        **kwargs,
    ):
        h = torch.tanh(self.mix(self.embed(input_ids)))
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        hidden = self.honours_flag and os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
        self.forwards.append("hidden" if hidden else "logits")
        return SimpleNamespace(logits = h if hidden else self.lm_head(h))


class _Accel:
    num_processes = 1
    scaler = None

    def unwrap_model(self, model, **kwargs):
        return model

    def reduce(
        self,
        tensor,
        reduction = "sum",
    ):
        return tensor

    def gather(self, tensor):
        return tensor


def _class_method_source(path, class_name, method):
    text = open(path, encoding = "utf-8").read()
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    return "\n".join(text.split("\n")[fn.lineno - 1 : fn.end_lineno]) + "\n"


def _gmpo_compute_loss():
    try:
        module = importlib.import_module("trl.experimental.gmpo.gmpo_trainer")
    except Exception:
        pytest.skip("this TRL has no GMPOTrainer (added in TRL 1.15)")
    source = _class_method_source(module.__file__, "GMPOTrainer", "_compute_loss")
    namespace = dict(vars(module))
    exec(compile("class _GMPO:\n" + source, "<trl gmpo>", "exec"), namespace)
    return namespace["_GMPO"]._compute_loss


def _unsloth_compute_loss():
    """Unsloth's generated GRPO compute_loss, built from the real rewrite and run in rl_replacements' namespace."""
    stub = "def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):\n    pass\n"
    source = textwrap.dedent(rl.grpo_trainer_compute_loss("compute_loss", stub))
    assert source != stub
    namespace = dict(vars(rl))
    exec(compile(source, "<unsloth grpo compute_loss>", "exec"), namespace)
    return namespace["compute_loss"]


def _reference_logps(
    self,
    model,
    input_ids,
    attention_mask,
    logits_to_keep,
    batch_size = None,
    compute_entropy = False,
    **kwargs,
):
    """TRL's float64 scorer contract on plain logits: log_softmax(logits / T)[target], entropy under no_grad."""
    logits = model(
        input_ids = input_ids, attention_mask = attention_mask, logits_to_keep = logits_to_keep + 1
    ).logits
    logits = logits[:, :-1, :][:, -logits_to_keep:, :] / self.temperature
    targets = input_ids[:, -logits_to_keep:]
    logps = torch.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1)
    entropies = None
    if compute_entropy:
        with torch.no_grad():
            p = torch.softmax(logits, -1)
            entropies = -(p * torch.log_softmax(logits, -1)).sum(-1)
    return logps, entropies, None


def _trainer(
    model,
    beta = 0.0,
    temperature = 0.7,
    ga = 2,
):
    t = SimpleNamespace()
    t.model = model
    t.temperature = temperature
    t.model_kwarg_keys = ("input_ids", "attention_mask", "logits_to_keep")
    t.accelerator = _Accel()
    t.args = SimpleNamespace(bf16 = False, fp16 = False, mixed_precision = "no")
    t.top_entropy_quantile = 1.0
    t.epsilon_low, t.epsilon_high = EPS_LOW, EPS_HIGH
    t.beta = beta
    t.current_gradient_accumulation_steps = ga
    t._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    return t


def _bind(cls, model, **kwargs):
    """A real object of ``cls`` carrying the stand-in trainer state."""
    obj = cls.__new__(cls)
    obj.__dict__.update(vars(_trainer(model, **kwargs)))
    return obj


def _inputs(
    beta = 0.0,
    B = 4,
    P = 4,
    C = 6,
    seed = 1,
):
    gen = torch.Generator().manual_seed(seed)
    prompt_ids = torch.randint(0, V, (B, P), generator = gen)
    prompt_mask = torch.ones(B, P, dtype = torch.long)
    prompt_mask[0, :2] = 0  # left-padded prompt
    completion_ids = torch.randint(0, V, (B, C), generator = gen)
    completion_mask = torch.ones(B, C, dtype = torch.long)
    completion_mask[1, -3:] = 0  # right-padded completion
    completion_mask[2, 1:] = 0  # one-token completion
    advantages = torch.tensor([1.3, -0.7, 0.4, -1.1], dtype = torch.float64)[:B]
    # Old policy away from the current one: ratios on both sides of the log-space clip (+-0.4).
    old = torch.randn(B, C, generator = gen, dtype = torch.float64) * 0.5
    inputs = {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": advantages,
        "old_per_token_logps_offset": old,
    }
    return inputs


def _with_old(
    inputs,
    model,
    temperature = 0.7,
    beta = 0.0,
):
    """old = current + offset (no grad), ref = current + 0.3 * offset: exact ratios known to the oracle."""
    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], 1)
    mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
    k = inputs["completion_ids"].size(1)
    with torch.no_grad():
        cur, _, _ = _reference_logps(SimpleNamespace(temperature = temperature), model, ids, mask, k)
    out = {k_: v for k_, v in inputs.items() if k_ != "old_per_token_logps_offset"}
    out["old_per_token_logps"] = cur + inputs["old_per_token_logps_offset"]
    if beta:
        out["ref_per_token_logps"] = cur + 0.3 * inputs["old_per_token_logps_offset"]
    return out, cur


def _oracle(cur_logps, inputs, beta, ga):
    """GMPO in float64 from the paper, independent of TRL's code."""
    mask = inputs["completion_mask"].double()
    adv = inputs["advantages"].double()
    log_r = cur_logps - inputs["old_per_token_logps"]
    rows = []
    for i in range(log_r.size(0)):
        terms = []
        for t in range(log_r.size(1)):
            if mask[i, t] == 0:
                continue
            x = log_r[i, t]
            clipped = x.clamp(-EPS_LOW, EPS_HIGH)
            # sgn(A) * min(sgn(A) * x, sgn(A) * clip(x)) for A > 0; max for A <= 0 (TRL's convention at A = 0).
            terms.append(torch.minimum(x, clipped) if adv[i] > 0 else torch.maximum(x, clipped))
        geo = (
            torch.exp(torch.stack(terms).mean())
            if terms
            else torch.tensor(1.0, dtype = torch.float64)
        )
        loss_i = -geo * adv[i]
        if beta:
            d = inputs["ref_per_token_logps"][i] - cur_logps[i]
            kl = torch.exp(d) - d - 1
            loss_i = loss_i + beta * (kl * mask[i]).sum() / mask[i].sum().clamp(min = 1.0)
        rows.append(loss_i)
    return torch.stack(rows).mean() / ga


def _user_compute_loss(self, model, inputs):
    """A user objective in the shape TRL documents for subclasses (any TRL with _compute_loss)."""
    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], 1)
    mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
    out = self._get_per_token_logps_and_entropies(
        model, ids, mask, inputs["completion_ids"].size(1), compute_entropy = True
    )
    cm = inputs["completion_mask"].to(out[0].dtype)
    return -(out[0] * inputs["advantages"][:, None].to(out[0].dtype) * cm).sum() / cm.sum()


def _classes(objective = None):
    gmpo = _gmpo_compute_loss() if objective is None else objective
    fused = _unsloth_compute_loss()

    def _trl_grpo_compute_loss(self, model, inputs):
        raise AssertionError(
            "the generated class's own _compute_loss must never run on the fused path"
        )

    def _fused_scorer(self, *args, **kwargs):
        raise AssertionError("the fused GRPO loss ran: the subclass _compute_loss was bypassed")

    Fused = type(
        "UnslothGRPOTrainerStandIn",
        (),
        {
            "compute_loss": fused,
            "_compute_loss": _trl_grpo_compute_loss,
            "_get_per_token_logps_and_entropies": _fused_scorer,
        },
    )
    Sub = type("GMPOTrainerStandIn", (Fused,), {"_compute_loss": gmpo})
    return Fused, Sub, gmpo


def test_subclass_override_is_dispatched_plain_grpo_is_not():
    """Fails before the fix: the fused compute_loss never reached a subclass _compute_loss."""
    from unsloth.models._grpo_subclass_loss import grpo_subclass_compute_loss

    Fused, Sub, gmpo = _classes(_user_compute_loss)
    model = _Model()
    assert grpo_subclass_compute_loss(_bind(Fused, model)) is None
    assert grpo_subclass_compute_loss(_bind(Sub, model)) is gmpo

    # A subclass that overrides compute_loss and calls super() still reaches its _compute_loss override.
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs = False,
        num_items_in_batch = None,
    ):
        return super(Wrapper, self).compute_loss(model, inputs, return_outputs, num_items_in_batch)

    Wrapper = type("Wrapper", (Sub,), {"compute_loss": compute_loss})
    assert grpo_subclass_compute_loss(_bind(Wrapper, model)) is gmpo
    # Overriding other methods only keeps the fused loss.
    Other = type("Other", (Fused,), {"log": lambda self, *a: None})
    assert grpo_subclass_compute_loss(_bind(Other, model)) is None


@pytest.mark.parametrize("beta", [0.0, 0.04])
@pytest.mark.parametrize("honours_flag", [True, False], ids = ["fused_route", "logits_route"])
def test_gmpo_loss_matches_trl_and_fp64_oracle(beta, honours_flag):
    Fused, Sub, gmpo = _classes()
    model = _Model(honours_flag = honours_flag)
    raw = _inputs()
    inputs, _ = _with_old(raw, model, beta = beta)

    # TRL's GMPO objective with an exact float64 scorer.
    trl_trainer = _bind(Sub, model, beta = beta)
    trl_trainer._get_per_token_logps_and_entropies = lambda *a, **k: _reference_logps(
        trl_trainer, *a, **k
    )
    model.zero_grad()
    trl_loss = gmpo(trl_trainer, model, dict(inputs))
    trl_loss.backward()
    trl_grads = [p.grad.detach().clone() for p in (model.mix.weight, model.embed.weight)]

    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], 1)
    mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
    cur, _, _ = _reference_logps(trl_trainer, model, ids, mask, inputs["completion_ids"].size(1))
    oracle = _oracle(cur.detach(), inputs, beta, ga = 2)
    assert abs(trl_loss.item() - oracle.item()) < 1e-12

    # Unsloth: the generated fused compute_loss on the GMPO subclass.
    trainer = _bind(Sub, model, beta = beta)
    model.zero_grad()
    model.forwards.clear()
    loss = Sub.compute_loss(trainer, model, dict(inputs))
    loss.backward()
    grads = [p.grad.detach().clone() for p in (model.mix.weight, model.embed.weight)]

    route = "fused" if honours_flag else "logits"
    assert trainer._unsloth_grpo_subclass_loss_calls == 1
    assert trainer._unsloth_grpo_subclass_scorer == {route: 1}
    assert model.forwards == (["hidden"] if honours_flag else ["logits"])
    assert "_get_per_token_logps_and_entropies" not in trainer.__dict__
    assert abs(loss.item() - trl_loss.item()) < 1e-6, (loss.item(), trl_loss.item())
    assert abs(loss.item() - oracle.item()) < 1e-6
    for a, b in zip(grads, trl_grads):
        torch.testing.assert_close(a.double(), b.double(), atol = 1e-6, rtol = 1e-5)
    # GMPO's metrics were logged as TRL logs them.
    assert len(trainer._metrics["train"]["entropy"]) == 1
    assert "clip_ratio/region_mean" in trainer._metrics["train"]


@pytest.mark.parametrize("honours_flag", [True, False], ids = ["fused_route", "logits_route"])
def test_user_subclass_objective_trains_under_unsloth(honours_flag):
    """Any TRL: a user _compute_loss gets its value and gradient, not the fused GRPO loss."""
    _, Sub, objective = _classes(_user_compute_loss)
    model = _Model(honours_flag = honours_flag)
    inputs, _ = _with_old(_inputs(), model)
    reference = _bind(Sub, model)
    reference._get_per_token_logps_and_entropies = lambda *a, **k: _reference_logps(
        reference, *a, **k
    )
    model.zero_grad()
    want = objective(reference, model, dict(inputs))
    want.backward()
    want_grads = [p.grad.detach().clone() for p in (model.mix.weight, model.embed.weight)]
    trainer = _bind(Sub, model)
    model.zero_grad()
    got = Sub.compute_loss(trainer, model, dict(inputs))
    got.backward()
    assert trainer._unsloth_grpo_subclass_scorer == {("fused" if honours_flag else "logits"): 1}
    assert abs(got.item() - want.item()) < 1e-6
    for p, w in zip((model.mix.weight, model.embed.weight), want_grads):
        torch.testing.assert_close(p.grad.double(), w.double(), atol = 1e-6, rtol = 1e-5)


def test_gmpo_differs_from_the_grpo_objective_the_fused_loss_computes():
    """What the run silently trained before: GRPO's token-mean clipped ratio, not GMPO's geometric mean."""
    _, _, gmpo = _classes()
    model = _Model()
    inputs, cur = _with_old(_inputs(), model)
    gmpo_value = _oracle(cur, inputs, beta = 0.0, ga = 2).item()
    mask = inputs["completion_mask"].double()
    ratio = torch.exp(cur - inputs["old_per_token_logps"])
    adv = inputs["advantages"][:, None]
    per_token = -torch.minimum(ratio * adv, ratio.clamp(1 - 0.2, 1 + 0.2) * adv)
    grpo_value = (((per_token * mask).sum(-1) / mask.sum(-1).clamp(min = 1.0)).mean() / 2).item()
    assert abs(gmpo_value - grpo_value) > 1e-2, (gmpo_value, grpo_value)


def test_subclass_scorer_matches_trl_stock_scorer_and_arity():
    """The gradient-carrying scorer against TRL's own _get_per_token_logps_and_entropies where TRL's reads plain
    logits (TRL < 1.15; later TRL scores through its fused lm_head, pinned above by the float64 reference)."""
    from unsloth.models._grpo_subclass_loss import (
        _returns_aux_loss,
        grpo_subclass_logps_and_entropies,
    )

    grpo = importlib.import_module("trl.trainer.grpo_trainer")
    stock_source = _class_method_source(
        grpo.__file__, "GRPOTrainer", "_get_per_token_logps_and_entropies"
    )
    ids = torch.randint(0, V, (3, 9), generator = torch.Generator().manual_seed(2))
    mask = torch.ones(3, 9, dtype = torch.long)
    mask[0, :2] = 0
    for honours_flag in (True, False):
        model = _Model(honours_flag = honours_flag)
        trainer = _trainer(model)
        out = grpo_subclass_logps_and_entropies(
            trainer, model, ids, mask, 5, batch_size = 2, compute_entropy = True
        )
        assert len(out) == (3 if _returns_aux_loss() else 2)
        assert out[0].requires_grad and not out[1].requires_grad
        if "fused_lm_head=True" in stock_source:
            ref = _reference_logps(trainer, model, ids, mask, 5, compute_entropy = True)
        else:
            namespace = dict(vars(grpo))
            exec(compile("class _S:\n" + stock_source, "<trl stock>", "exec"), namespace)
            ref = namespace["_S"]._get_per_token_logps_and_entropies(
                trainer, model, ids, mask, 5, batch_size = 2, compute_entropy = True
            )
            assert len(ref) == len(out)
        torch.testing.assert_close(out[0].double(), ref[0].double(), atol = 1e-5, rtol = 1e-5)
        torch.testing.assert_close(out[1].double(), ref[1].double(), atol = 1e-5, rtol = 1e-5)


def test_subclass_scorer_refuses_inputs_it_cannot_slice():
    from unsloth.models._grpo_subclass_loss import grpo_subclass_logps_and_entropies

    model = _Model()
    ids = torch.randint(0, V, (2, 6))
    mask = torch.ones(2, 6, dtype = torch.long)
    with pytest.raises(NotImplementedError, match = "pixel_values"):
        grpo_subclass_logps_and_entropies(
            _trainer(model), model, ids, mask, 3, pixel_values = torch.zeros(1)
        )
    # None-valued vision keys (GMPO passes them all) are fine.
    grpo_subclass_logps_and_entropies(
        _trainer(model), model, ids, mask, 3, pixel_values = None, num_images = None
    )


def test_old_logps_shortcuts_stand_down_for_subclass_losses(monkeypatch):
    """Unsloth's skip-the-old-policy-forward markers are read only by the fused loss; a subclass loss reads TRL's
    old_per_token_logps, so TRL must score pi_old for it."""
    Fused, Sub, _ = _classes(_user_compute_loss)
    monkeypatch.setenv("UNSLOTH_GRPO_ROLLOUT_LOGPROBS_AS_OLD", "1")
    model = _Model()
    model.train(False)

    def _state(cls):
        t = _bind(cls, model)
        t.use_vllm = True
        t.vllm_importance_sampling_correction = True
        t.num_iterations = 2
        t.args = SimpleNamespace(
            gradient_accumulation_steps = 4,
            steps_per_generation = 1,
            loss_type = "dapo",
            bf16 = False,
            fp16 = False,
            mixed_precision = "no",
        )
        return t

    cmask = torch.ones(2, 3, dtype = torch.long)
    logps = torch.zeros(2, 3)
    sub = _state(Sub)
    assert rl._unsloth_grpo_old_logps_from_new(sub) is False
    assert rl._unsloth_grpo_old_logps_store(sub) is False
    with pytest.warns(UserWarning, match = "overrides _compute_loss"):
        assert rl._unsloth_grpo_rollout_logps_as_old(sub, logps, cmask) is False
    # The plain fused trainer keeps whatever the shortcuts decide (no subclass refusal reason).
    plain = _state(Fused)
    import warnings

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        rl._unsloth_grpo_rollout_logps_as_old(plain, logps, cmask)
    assert not any("overrides _compute_loss" in str(w.message) for w in caught)
