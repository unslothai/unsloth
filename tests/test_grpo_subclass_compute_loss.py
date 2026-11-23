# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""GRPOTrainer subclasses that override ``_compute_loss`` (TRL's GMPOTrainer, user objectives) train their own
objective under Unsloth instead of the fused GRPO loss.

Unsloth's generated GRPO ``compute_loss`` and scorers are executed from their real rewrites in a stand-in class; TRL's
GMPO ``_compute_loss`` is executed from TRL's own source (read with ``ast``, since importing unsloth may rebind TRL's
classes). The float64 oracle is the GMPO objective written out independently of TRL (arXiv 2507.20673, eq. 7, as
TRL implements it: token clipping in log space, sign-aware, geometric mean over the completion, mean over rows)."""

from __future__ import annotations

import ast
import functools
import importlib
import os
import textwrap
import warnings
from collections import defaultdict
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
import unsloth  # noqa: F401,E402
import unsloth.models.rl_replacements as rl  # noqa: E402
from unsloth.models._grpo_subclass_loss import (  # noqa: E402
    grpo_subclass_compute_loss,
    grpo_subclass_logps_and_entropies,
)

H, V = 16, 40
EPS_LOW, EPS_HIGH = 0.4, 0.4


def _trl_at_least(minimum):
    from importlib.metadata import version

    from packaging.version import Version
    return Version(version("trl")) >= Version(minimum)


# TRL >= 1.15 scores through its fused lm_head: completion pads score 0, the entropy keeps its gradient.
MODERN = _trl_at_least("1.15.0")


def _returns_aux_loss():
    return _trl_at_least("1.7.0")


class _Model(torch.nn.Module):
    """Unsloth-style CausalLM: ``.logits`` is the last hidden state under UNSLOTH_RETURN_HIDDEN_STATES=1 when
    ``honours_flag`` is set; otherwise it always returns real logits (a forward without hidden-state support).
    ``embed_tokens`` marks it as a backbone, so the scorer's left-padding mask scope reaches it."""

    def __init__(
        self,
        honours_flag = True,
        dtype = torch.float64,
    ):
        super().__init__()
        from transformers import Qwen3Config

        self.config = Qwen3Config(hidden_size = H, vocab_size = V, num_hidden_layers = 1)
        gen = torch.Generator().manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(V, H).to(dtype)
        self.embed_tokens.weight.data = torch.randn(V, H, generator = gen, dtype = dtype)
        self.mix = torch.nn.Linear(H, H, dtype = dtype)
        self.mix.weight.data = torch.randn(H, H, generator = gen, dtype = dtype) / H**0.5
        self.mix.bias.data = torch.zeros(H, dtype = dtype)
        self.lm_head = torch.nn.Linear(H, V, bias = False, dtype = dtype)
        self.lm_head.weight.data = torch.randn(V, H, generator = gen, dtype = dtype)
        self.honours_flag = honours_flag
        self.forwards = []
        self.keep_padding_mask = []

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
        from transformers.modeling_outputs import CausalLMOutputWithPast

        h = torch.tanh(self.mix(self.embed_tokens(input_ids)))
        if logits_to_keep:
            h = h[:, -logits_to_keep:]
        hidden = self.honours_flag and os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
        self.forwards.append("hidden" if hidden else "logits")
        self.keep_padding_mask.append(getattr(self, "_unsloth_keep_padding_mask", False))
        return CausalLMOutputWithPast(logits = h if hidden else self.lm_head(h))


class _Accel:
    num_processes = 1
    scaler = None

    def unwrap_model(self, model, **kwargs):
        return getattr(model, "module", model)

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
    fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method), None)
    if fn is None:
        return None
    # Without decorators (TRL's profiling wrapper needs a real trainer).
    return "\n".join(text.split("\n")[fn.lineno - 1 : fn.end_lineno]) + "\n"


def _grpo_module():
    return importlib.import_module("trl.trainer.grpo_trainer")


def _gmpo_compute_loss():
    try:
        module = importlib.import_module("trl.experimental.gmpo.gmpo_trainer")
    except ModuleNotFoundError as error:
        # Only a TRL without GMPO skips: a broken transitive import must fail the test.
        if error.name not in {
            "trl.experimental",
            "trl.experimental.gmpo",
            "trl.experimental.gmpo.gmpo_trainer",
        }:
            raise
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


def _unsloth_scorer():
    """Unsloth's generated _get_per_token_logps_and_entropies (TRL >= 0.20), or None on TRL without it."""
    grpo = _grpo_module()
    stock = _class_method_source(grpo.__file__, "GRPOTrainer", "_get_per_token_logps_and_entropies")
    if stock is None:
        return None
    source = textwrap.dedent(
        rl.grpo_trainer__get_per_token_logps_and_entropies(
            "_get_per_token_logps_and_entropies", stock
        )
    )
    import unsloth_zoo.rl_replacements as zoo_rl

    # The generated module inlines zoo's helpers (left packing, autotuning) next to rl_replacements' names.
    namespace = dict(vars(grpo))
    for module in (zoo_rl, rl):
        namespace.update({k: v for k, v in vars(module).items() if not k.startswith("__")})
    exec(compile(source, "<unsloth grpo scorer>", "exec"), namespace)
    return namespace["_get_per_token_logps_and_entropies"]


def _unsloth_legacy_scorer():
    """Unsloth's generated legacy _get_per_token_logps (what TRL 0.18 / 0.19 call)."""
    stub = "def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None):\n    pass\n"
    source = textwrap.dedent(rl.grpo_trainer__get_per_token_logps("_get_per_token_logps", stub))
    namespace = dict(vars(rl))
    exec(compile(source, "<unsloth legacy scorer>", "exec"), namespace)
    return namespace["_get_per_token_logps"]


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
    t.args = SimpleNamespace(
        bf16 = False,
        fp16 = False,
        mixed_precision = "no",
        temperature = temperature,
        unsloth_grpo_mini_batch = None,
        unsloth_logit_chunk_multiplier = None,
    )
    t.processing_class = SimpleNamespace(pad_token_id = 0)
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
    B = 4,
    P = 4,
    C = 6,
    seed = 1,
):
    gen = torch.Generator().manual_seed(seed)
    # Ids from 1: 0 is the pad id, which Unsloth's own scorer packs on.
    prompt_ids = torch.randint(1, V, (B, P), generator = gen)
    prompt_mask = torch.ones(B, P, dtype = torch.long)
    prompt_mask[0, :2] = 0  # left-padded prompt: mixed prompt lengths
    prompt_ids = prompt_ids.masked_fill(prompt_mask == 0, 0)
    completion_ids = torch.randint(1, V, (B, C), generator = gen)
    completion_mask = torch.ones(B, C, dtype = torch.long)
    completion_mask[1, -3:] = 0  # right-padded completion
    completion_mask[2, 1:] = 0  # one-token completion
    completion_ids = completion_ids.masked_fill(completion_mask == 0, 0)
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


def _ids_and_mask(inputs):
    ids = torch.cat([inputs["prompt_ids"], inputs["completion_ids"]], 1)
    mask = torch.cat([inputs["prompt_mask"], inputs["completion_mask"]], 1)
    return ids, mask, inputs["completion_ids"].size(1)


def _with_old(
    inputs,
    model,
    temperature = 0.7,
    beta = 0.0,
):
    """old = current + offset (no grad), ref = current + 0.3 * offset: exact ratios known to the oracle."""
    ids, mask, k = _ids_and_mask(inputs)
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
    ids, mask, k = _ids_and_mask(inputs)
    out = self._get_per_token_logps_and_entropies(model, ids, mask, k, compute_entropy = True)
    cm = inputs["completion_mask"].to(out[0].dtype)
    return -(out[0] * inputs["advantages"][:, None].to(out[0].dtype) * cm).sum() / cm.sum()


def _legacy_user_compute_loss(self, model, inputs):
    """The same objective on TRL 0.18 / 0.19, whose scorer is _get_per_token_logps."""
    ids, mask, k = _ids_and_mask(inputs)
    logps = self._get_per_token_logps(model, ids, mask, k)
    cm = inputs["completion_mask"].to(logps.dtype)
    return -(logps * inputs["advantages"][:, None].to(logps.dtype) * cm).sum() / cm.sum()


def _classes(objective = None, scorer_overrides = None):
    """(Fused stand-in, subclass, objective): the stand-in carries Unsloth's generated compute_loss and scorers."""
    gmpo = _gmpo_compute_loss() if objective is None else objective
    fused = _unsloth_compute_loss()

    def _trl_grpo_compute_loss(self, model, inputs):
        raise AssertionError(
            "the generated class's own _compute_loss must never run on the fused path"
        )

    body = {"compute_loss": fused, "_compute_loss": _trl_grpo_compute_loss}
    scorer = _unsloth_scorer()
    if scorer is not None:
        body["_get_per_token_logps_and_entropies"] = scorer
    if objective is _legacy_user_compute_loss:
        # Only TRL 0.18 / 0.19 carry it; the fused compute_loss prefers it when present.
        body["_get_per_token_logps"] = _unsloth_legacy_scorer()
    Fused = type("UnslothGRPOTrainerStandIn", (), body)
    Sub = type("GMPOTrainerStandIn", (Fused,), {"_compute_loss": gmpo, **(scorer_overrides or {})})
    return Fused, Sub, gmpo


def _needs_modern_scorer():
    if _unsloth_scorer() is None:
        pytest.skip("this TRL has no _get_per_token_logps_and_entropies")


def test_subclass_override_is_dispatched_plain_grpo_is_not():
    """Fails before the fix: the fused compute_loss never reached a subclass _compute_loss."""
    _needs_modern_scorer()
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
    _needs_modern_scorer()
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
    trl_grads = [p.grad.detach().clone() for p in (model.mix.weight, model.embed_tokens.weight)]

    ids, mask, k = _ids_and_mask(inputs)
    cur, _, _ = _reference_logps(trl_trainer, model, ids, mask, k)
    oracle = _oracle(cur.detach(), inputs, beta, ga = 2)
    assert abs(trl_loss.item() - oracle.item()) < 1e-12

    # Unsloth: the generated fused compute_loss on the GMPO subclass.
    trainer = _bind(Sub, model, beta = beta)
    model.zero_grad()
    model.forwards.clear()
    loss = Sub.compute_loss(trainer, model, dict(inputs))
    loss.backward()
    grads = [p.grad.detach().clone() for p in (model.mix.weight, model.embed_tokens.weight)]

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
    _needs_modern_scorer()
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
    want_grads = [p.grad.detach().clone() for p in (model.mix.weight, model.embed_tokens.weight)]
    trainer = _bind(Sub, model)
    model.zero_grad()
    got = Sub.compute_loss(trainer, model, dict(inputs))
    got.backward()
    assert trainer._unsloth_grpo_subclass_scorer == {("fused" if honours_flag else "logits"): 1}
    assert abs(got.item() - want.item()) < 1e-6
    for p, w in zip((model.mix.weight, model.embed_tokens.weight), want_grads):
        torch.testing.assert_close(p.grad.double(), w.double(), atol = 1e-6, rtol = 1e-5)


def test_gmpo_differs_from_the_grpo_objective_the_fused_loss_computes():
    """What the run silently trained before: GRPO's token-mean clipped ratio, not GMPO's geometric mean."""
    _gmpo_compute_loss()
    model = _Model()
    inputs, cur = _with_old(_inputs(), model)
    gmpo_value = _oracle(cur, inputs, beta = 0.0, ga = 2).item()
    mask = inputs["completion_mask"].double()
    ratio = torch.exp(cur - inputs["old_per_token_logps"])
    adv = inputs["advantages"][:, None]
    per_token = -torch.minimum(ratio * adv, ratio.clamp(1 - 0.2, 1 + 0.2) * adv)
    grpo_value = (((per_token * mask).sum(-1) / mask.sum(-1).clamp(min = 1.0)).mean() / 2).item()
    assert abs(gmpo_value - grpo_value) > 1e-2, (gmpo_value, grpo_value)


@pytest.mark.parametrize(
    "score_key,beta", [("old_per_token_logps", 0.0), ("ref_per_token_logps", 0.04)]
)
@pytest.mark.parametrize("honours_flag", [True, False], ids = ["fused_route", "logits_route"])
def test_rollout_scores_feed_the_subclass_objective(score_key, beta, honours_flag):
    """Finding 3 / 16: with mixed prompt lengths the generated rollout scorer returned Unsloth's left-packed window
    (completion + max_left_pad wide), so GMPO's first subtraction raised. The old / reference scores must have the
    completion's width and, at the rollout's weights, equal the gradient pass exactly (ratio 1, KL 0)."""
    _needs_modern_scorer()
    try:
        objective = _gmpo_compute_loss()
    except pytest.skip.Exception:
        objective = _user_compute_loss
    _, Sub, _ = _classes(objective)
    model = _Model(honours_flag = honours_flag)
    raw = _inputs()
    inputs = {k: v for k, v in raw.items() if k != "old_per_token_logps_offset"}
    trainer = _bind(Sub, model, beta = beta)
    ids, mask, width = _ids_and_mask(inputs)
    with torch.no_grad():
        scored = trainer._get_per_token_logps_and_entropies(model, ids, mask, width, batch_size = 2)[
            0
        ]
    assert scored.shape == inputs["completion_ids"].shape
    inputs[score_key] = scored
    model.zero_grad()
    loss = Sub.compute_loss(trainer, model, inputs)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.mix.weight.grad).all()
    with torch.no_grad():
        current = trainer._get_per_token_logps_and_entropies(model, ids, mask, width)[0]
    cm = inputs["completion_mask"].bool()
    # The scored pass ran in batches of 2 and this one in a single batch: float64 matmuls of different shapes may
    # differ in the last bit on some CPUs, so exact equality is not the contract. A misaligned window is off by O(1).
    torch.testing.assert_close(current[cm], scored[cm], rtol = 0, atol = 1e-12)
    if objective is not _user_compute_loss:
        # Ratio exactly 1 and KL exactly 0: GMPO's loss is -mean(A) / ga.
        want = -inputs["advantages"].mean() / 2
        assert abs(loss.item() - want.item()) < 1e-12


def test_ddp_trainable_head_scores_from_logits(tmp_path):
    """Finding 1: under DDP(find_unused_parameters=True) a trainable head used outside the forward is marked ready
    twice. The scorer must take the logits route for every pass; a frozen head keeps the fused route."""
    _needs_modern_scorer()
    import torch.distributed as dist

    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    created = not dist.is_initialized()
    if created:
        dist.init_process_group(
            "gloo", init_method = f"file://{tmp_path / 'pg'}", rank = 0, world_size = 1
        )
    try:
        _, Sub, _ = _classes(_user_compute_loss)
        inputs, _ = _with_old(_inputs(), _Model())
        for frozen in (False, True):
            model = _Model(honours_flag = True)
            model.lm_head.weight.requires_grad_(not frozen)
            ddp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters = True)
            trainer = _bind(Sub, model)
            trainer.model_wrapped = ddp
            loss = Sub.compute_loss(trainer, ddp, dict(inputs))
            loss.backward()
            assert torch.isfinite(model.mix.weight.grad).all()
            assert trainer._unsloth_grpo_subclass_scorer == {("fused" if frozen else "logits"): 1}
    finally:
        if created:
            dist.destroy_process_group()


@pytest.mark.parametrize("prior", [None, "0", "1"])
def test_scorer_restores_the_callers_forward_mode(monkeypatch, prior):
    """Finding 2: a trailing write left UNSLOTH_RETURN_HIDDEN_STATES at "0" whatever the caller had."""
    if prior is None:
        monkeypatch.delenv("UNSLOTH_RETURN_HIDDEN_STATES", raising = False)
    else:
        monkeypatch.setenv("UNSLOTH_RETURN_HIDDEN_STATES", prior)
    model = _Model()
    ids, mask, k = _ids_and_mask(_inputs())
    grpo_subclass_logps_and_entropies(_trainer(model), model, ids, mask, k)
    assert model.forwards == ["hidden"]
    assert os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES") == prior


def test_vllm_importance_sampling_correction_stays_live_for_subclass_objectives():
    """Finding 4: Unsloth disables TRL's vLLM importance-sampling block (its own loss corrects), which left a
    subclass objective without importance_sampling_ratio (stock GRPO's _compute_loss raised KeyError)."""
    grpo = _grpo_module()
    source = _class_method_source(grpo.__file__, "GRPOTrainer", "_generate_and_score_completions")
    if source is None or "vllm_importance_sampling_correction" not in source:
        pytest.skip("this TRL has no vLLM importance-sampling correction")
    rewritten = textwrap.dedent(
        rl.grpo_trainer__generate_and_score_completions("_generate_and_score_completions", source)
    )
    conditions = [
        node.test
        for node in ast.walk(ast.parse(rewritten))
        if isinstance(node, ast.If)
        and ast.unparse(node.test).endswith(
            "self.use_vllm and self.vllm_importance_sampling_correction"
        )
    ]
    assert len(conditions) >= 2  # the ratio, then its metrics / export
    Fused, Sub, _ = _classes(_user_compute_loss) if _unsloth_scorer() else (None, None, None)
    if Fused is None:
        pytest.skip("this TRL has no _get_per_token_logps_and_entropies")
    names = {"_unsloth_grpo_subclass_owner": grpo_subclass_compute_loss}
    for cls, want in ((Sub, True), (Fused, False)):
        trainer = _bind(cls, _Model())
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        for test in conditions:
            got = eval(
                compile(ast.Expression(test), "<condition>", "eval"), names, {"self": trainer}
            )
            assert bool(got) is want, (cls.__name__, ast.unparse(test))


def test_subclass_scoring_methods_and_descriptors_are_honoured():
    """Findings 5 / 24: the dispatch bound Unsloth's scorer on the instance, shadowing a subclass's own scorer (and
    the context parallel instance wrapper), and called the objective unbound, which broke descriptors."""
    _needs_modern_scorer()
    calls = []

    def _get_per_token_logps_and_entropies(self, *args, **kwargs):
        calls.append(torch.is_grad_enabled())
        return super(Sub, self)._get_per_token_logps_and_entropies(*args, **kwargs)

    _, Sub, _ = _classes(
        _user_compute_loss,
        {"_get_per_token_logps_and_entropies": _get_per_token_logps_and_entropies},
    )
    model = _Model()
    inputs, _ = _with_old(_inputs(), model)
    trainer = _bind(Sub, model)
    Sub.compute_loss(trainer, model, dict(inputs)).backward()
    assert calls == [True]
    assert trainer._unsloth_grpo_subclass_scorer == {"fused": 1}

    def _scaled(self, model, inputs, scale):
        return scale * _user_compute_loss(self, model, inputs)

    _, Partial, _ = _classes(functools.partialmethod(_scaled, scale = 2.0))
    trainer = _bind(Partial, model)
    got = Partial.compute_loss(trainer, model, dict(inputs))
    want = _user_compute_loss(_bind(Sub, model), model, dict(inputs))
    assert abs(got.item() - 2.0 * want.item()) < 1e-9


@pytest.mark.parametrize("honours_flag", [True, False], ids = ["fused_capable", "logits_only"])
def test_sampling_candidates_are_scored_like_stock_trl(honours_flag):
    """Finding 6: TRL >= 1.18 passes vLLM's sampling candidates (sampling_mask) to every scoring pass and
    normalises the log-probs over them. They were refused with NotImplementedError."""
    model = _Model(honours_flag = honours_flag)
    inputs = _inputs()
    ids, mask, k = _ids_and_mask(inputs)
    gen = torch.Generator().manual_seed(5)
    B, K = ids.size(0), 5
    support = torch.randint(0, V, (B, k, K), generator = gen)
    support[..., 0] = inputs["completion_ids"]  # the sampled token is a candidate
    support[..., 3:] = -1  # padded candidate lists
    support[0, 0, 1:] = -1  # a single candidate
    support[inputs["completion_mask"] == 0] = -1
    trainer = _trainer(model)
    logps, entropies = grpo_subclass_logps_and_entropies(
        trainer, model, ids, mask, k, batch_size = 3, compute_entropy = True, sampling_mask = support
    )[:2]
    assert trainer._unsloth_grpo_subclass_scorer == {"candidates": 2}

    logits = model(input_ids = ids, attention_mask = mask, logits_to_keep = k + 1).logits
    logits = logits[:, :-1][:, -k:] / trainer.temperature
    cand = logits.gather(-1, support.clamp(min = 0)).masked_fill(support < 0, -float("inf"))
    want = logits.gather(-1, inputs["completion_ids"][..., None]).squeeze(-1) - torch.logsumexp(
        cand, -1
    )
    cm = inputs["completion_mask"].bool()
    torch.testing.assert_close(logps[cm], want[cm], atol = 1e-10, rtol = 0)
    assert (logps[cm] <= 0).all() and (logps[0, 0] == 0)
    # The entropy stays that of the whole vocabulary, as in TRL's kernel.
    p = torch.softmax(logits, -1)
    torch.testing.assert_close(entropies[cm], -(p * torch.log_softmax(logits, -1)).sum(-1)[cm])
    (logps[cm].sum()).backward()
    assert torch.isfinite(model.mix.weight.grad).all()


def test_kept_set_reconstruction_is_refused_for_subclass_objectives(monkeypatch):
    """Finding 7: with no recorded candidates, Unsloth's kept-set scoring rebuilds top_k / top_p / min_p sets from
    its own scorer, which a subclass objective's current pass never uses. The gate must refuse it."""
    _needs_modern_scorer()
    Fused, Sub, _ = _classes(_user_compute_loss)
    model = _Model()
    assert rl._unsloth_grpo_support_mismatch(_bind(Fused, model)) == []
    reasons = rl._unsloth_grpo_support_mismatch(_bind(Sub, model))
    assert any("overrides _compute_loss" in r for r in reasons), reasons

    import unsloth_zoo.rl_replacements as zoo_rl

    monkeypatch.setattr(
        zoo_rl, "grpo_truncated_support_kwargs", lambda trainer: {"top_k": 3}, raising = False
    )
    trainer = _bind(Sub, model)
    ids, mask, k = _ids_and_mask(_inputs())
    with pytest.warns(UserWarning, match = "overrides _compute_loss"):
        trainer._get_per_token_logps_and_entropies(model, ids, mask, k)
    assert trainer._unsloth_truncated_support_refused is True


def test_legacy_scorer_serves_subclass_objectives():
    """Finding 8: TRL 0.18 / 0.19 score through _get_per_token_logps, which Unsloth turns into `return None` (the
    fused loss scores itself). A subclass objective got None. Plain GRPO keeps None, including TRL's positional
    batch_size call."""
    legacy = _unsloth_legacy_scorer()
    Fused, Sub, objective = _classes(_legacy_user_compute_loss)
    model = _Model()
    inputs, _ = _with_old(_inputs(), model)
    ids, mask, k = _ids_and_mask(inputs)
    plain = _bind(Fused, model)
    assert legacy(plain, model, ids, mask, k) is None
    assert legacy(plain, model, ids, mask, k, 2) is None
    assert legacy(plain, model, ids, mask, k, compute_efficient = True) is None

    trainer = _bind(Sub, model)
    with torch.no_grad():
        got = legacy(trainer, model, ids, mask, k, 2)
        want = grpo_subclass_logps_and_entropies(_trainer(model), model, ids, mask, k)[0]
    assert got.shape == inputs["completion_ids"].shape
    torch.testing.assert_close(got, want, atol = 0, rtol = 0)

    reference = _bind(Sub, model)
    reference._get_per_token_logps = lambda *a, **kw: _reference_logps(reference, *a, **kw)[0]
    want_loss = objective(reference, model, dict(inputs))
    got_loss = Sub.compute_loss(trainer, model, dict(inputs))
    # The fused kernel scores in float32.
    assert abs(got_loss.item() - want_loss.item()) < 1e-6


def test_sample_weighting_stands_down_for_subclass_objectives(monkeypatch):
    """Finding 11: the fused-loss sample weights were computed (and exclusions logged) for a subclass objective
    that never applies them."""
    _needs_modern_scorer()
    Fused, Sub, _ = _classes(_user_compute_loss)
    monkeypatch.setattr(rl, "_unsloth_grpo_sample_weight_mode", lambda: (True, False))
    rewards = torch.tensor([[1.0], [1.0], [0.0], [2.0]])
    cmask = torch.ones(4, 3)

    def state(cls):
        t = _bind(cls, _Model())
        t.args.loss_type = "dapo"
        t.num_generations = 2
        t.reward_weights = torch.ones(1)
        return t

    plain = rl._unsloth_grpo_sample_weight(state(Fused), "train", rewards, cmask, None, slice(0, 4))
    assert plain is not None
    with pytest.warns(UserWarning, match = "supplies its own _compute_loss"):
        assert (
            rl._unsloth_grpo_sample_weight(state(Sub), "train", rewards, cmask, None, slice(0, 4))
            is None
        )


@pytest.mark.parametrize("honours_flag", [True, False], ids = ["fused_route", "logits_route"])
def test_scorer_matches_trl_contract(honours_flag):
    """Arity, float64 values, completion-pad zeros and entropy gradient as the installed TRL defines them
    (findings 9, 10, 13): TRL >= 1.15 zeroes completion-pad scores and keeps the entropy differentiable; older
    TRL computes the entropy under no_grad."""
    inputs = _inputs()
    ids, mask, k = _ids_and_mask(inputs)
    model = _Model(honours_flag = honours_flag)
    trainer = _trainer(model)
    out = grpo_subclass_logps_and_entropies(
        trainer, model, ids, mask, k, batch_size = 3, compute_entropy = True
    )
    assert len(out) == (3 if _returns_aux_loss() else 2)
    logps, entropies = out[:2]
    assert logps.requires_grad
    assert entropies.requires_grad is MODERN

    logits = model(input_ids = ids, attention_mask = mask, logits_to_keep = k + 1).logits
    logits = logits[:, :-1][:, -k:] / trainer.temperature
    want = torch.log_softmax(logits, -1).gather(-1, inputs["completion_ids"][..., None]).squeeze(-1)
    p = torch.softmax(logits, -1)
    want_entropy = -(p * torch.log_softmax(logits, -1)).sum(-1)
    cm = inputs["completion_mask"].bool()
    torch.testing.assert_close(logps[cm].double(), want[cm], atol = 1e-6, rtol = 1e-6)
    torch.testing.assert_close(entropies[cm].double(), want_entropy[cm], atol = 1e-6, rtol = 1e-6)
    if MODERN:
        assert torch.count_nonzero(logps[~cm]) == 0
        assert torch.count_nonzero(entropies[~cm]) == 0
        (got,) = torch.autograd.grad(entropies[cm].sum(), model.mix.weight, retain_graph = True)
        (ref,) = torch.autograd.grad(want_entropy[cm].sum(), model.mix.weight)
        assert ref.abs().max() > 0
        torch.testing.assert_close(got.double(), ref, atol = 1e-5, rtol = 1e-4)
    # Under no_grad (rollout scoring) nothing keeps a graph.
    with torch.no_grad():
        logps, entropies = grpo_subclass_logps_and_entropies(
            trainer, model, ids, mask, k, compute_entropy = True
        )[:2]
    assert not logps.requires_grad and not entropies.requires_grad


def test_trl_stock_scorer_parity_where_it_reads_logits():
    """Against TRL's own _get_per_token_logps_and_entropies where TRL's reads plain logits (TRL < 1.15; later TRL
    scores through its fused lm_head, whose contract the test above pins)."""
    grpo = _grpo_module()
    stock_source = _class_method_source(
        grpo.__file__, "GRPOTrainer", "_get_per_token_logps_and_entropies"
    )
    if stock_source is None or "fused_lm_head=True" in stock_source:
        pytest.skip("TRL's scorer here does not read plain logits")
    ids, mask, _ = _ids_and_mask(_inputs())
    namespace = dict(vars(grpo))
    exec(compile("class _S:\n" + stock_source, "<trl stock>", "exec"), namespace)
    for honours_flag in (True, False):
        model = _Model(honours_flag = honours_flag)
        trainer = _trainer(model)
        out = grpo_subclass_logps_and_entropies(
            trainer, model, ids, mask, 5, batch_size = 2, compute_entropy = True
        )
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "0"
        try:
            ref = namespace["_S"]._get_per_token_logps_and_entropies(
                trainer, model, ids, mask, 5, batch_size = 2, compute_entropy = True
            )
        finally:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        assert len(ref) == len(out)
        torch.testing.assert_close(out[0].double(), ref[0].double(), atol = 1e-5, rtol = 1e-5)
        torch.testing.assert_close(out[1].double(), ref[1].double(), atol = 1e-5, rtol = 1e-5)


def test_head_projection_and_forward_run_in_scope(monkeypatch):
    """Findings 14 / 15: the fused head projection ran outside the forward's autocast, and the forward lacked the
    left-padding mask scope PPO and distillation use (GPT-OSS drops the mask otherwise)."""
    import unsloth.models._fused_rl_logprobs as fused

    seen = []
    original = fused.score_hidden_states

    def spy(*args, **kwargs):
        seen.append(torch.is_autocast_enabled("cpu"))
        return original(*args, **kwargs)

    monkeypatch.setattr(fused, "score_hidden_states", spy)
    model = _Model()
    trainer = _trainer(model)
    trainer.args.bf16, trainer.args.mixed_precision = True, "bf16"
    ids, mask, k = _ids_and_mask(_inputs())
    grpo_subclass_logps_and_entropies(trainer, model, ids, mask, k)
    assert seen == [True]
    assert model.keep_padding_mask == [True]
    assert "_unsloth_keep_padding_mask" not in model.__dict__


def test_scorer_edge_inputs():
    """Findings 22 / 23: an unset trainer temperature reads args.temperature; an empty completion window returns
    empty scores instead of scoring the prompt; inputs the scorer cannot slice are refused."""
    model = _Model()
    ids, mask, k = _ids_and_mask(_inputs())
    trainer = _trainer(model, temperature = 0.7)
    want = grpo_subclass_logps_and_entropies(trainer, model, ids, mask, k)[0]
    unset = _trainer(model, temperature = 0.7)
    unset.temperature = None
    torch.testing.assert_close(
        grpo_subclass_logps_and_entropies(unset, model, ids, mask, k)[0], want
    )

    out = grpo_subclass_logps_and_entropies(trainer, model, ids, mask, 0, compute_entropy = True)
    assert out[0].shape == (ids.size(0), 0) and out[1].shape == (ids.size(0), 0)

    with pytest.raises(NotImplementedError, match = "pixel_values"):
        grpo_subclass_logps_and_entropies(trainer, model, ids, mask, 3, pixel_values = torch.zeros(1))
    # None-valued vision keys (GMPO passes them all) are fine.
    grpo_subclass_logps_and_entropies(
        trainer, model, ids, mask, 3, pixel_values = None, num_images = None
    )


def test_gmpo_warns_that_it_ignores_the_vllm_correction():
    """Finding 20: upstream GMPO does not apply vLLM's importance-sampling ratio; say so once."""
    _needs_modern_scorer()
    _, Sub, _ = _classes()
    model = _Model()
    inputs, _ = _with_old(_inputs(), model)
    trainer = _bind(Sub, model)
    trainer.use_vllm = True
    trainer.vllm_importance_sampling_correction = True
    with pytest.warns(UserWarning, match = "does not apply the vLLM importance-sampling"):
        Sub.compute_loss(trainer, model, dict(inputs))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Sub.compute_loss(trainer, model, dict(inputs))


def test_old_logps_shortcuts_stand_down_for_subclass_losses(monkeypatch):
    """Unsloth's skip-the-old-policy-forward markers are read only by the fused loss; a subclass loss reads TRL's
    old_per_token_logps, so TRL must score pi_old for it."""
    _needs_modern_scorer()
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

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        rl._unsloth_grpo_rollout_logps_as_old(plain, logps, cmask)
    assert not any("overrides _compute_loss" in str(w.message) for w in caught)
