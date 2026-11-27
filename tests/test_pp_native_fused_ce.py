# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The DeepSeek-V4.1 last pipeline stage scores lm_head + CE through the chunked fused CE: no [mb, seq, vocab]
logits, the same loss and gradients as lm_head then loss_function, and the logits route when the fused one cannot
apply (kill switch, live adapter or bias on the head)."""

import importlib.machinery
import importlib.util
import os
import sys
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if not os.path.exists(os.path.join(REPO, "unsloth", "models", "deepseek_v41", "modeling.py")):
    pytest.skip("this tree has no native DeepSeek-V4.1", allow_module_level = True)


def _packages():
    """unsloth / unsloth.models as plain namespace packages (importing unsloth itself needs a GPU)."""
    for name, path in (("unsloth", ("unsloth",)), ("unsloth.models", ("unsloth", "models"))):
        package = types.ModuleType(name)
        package.__path__ = [os.path.join(REPO, *path)]
        package.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package = True)
        package.__spec__.submodule_search_locations = package.__path__
        sys.modules.setdefault(name, package)


def _load(name):
    full = f"unsloth.models.{name}"
    if full not in sys.modules:
        _packages()
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(REPO, "unsloth", "models", f"{name}.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        spec.loader.exec_module(module)
    return sys.modules[full]


# unsloth_zoo on its CPU backend while this module imports it. Only for the import: left set, every later
# test in the worker and every interpreter it spawns would take the CPU route too.
_FORCE_CPU = os.environ.get("UNSLOTH_FORCE_CPU")
if not torch.cuda.is_available():
    os.environ.setdefault("UNSLOTH_FORCE_CPU", "1")
try:
    pytest.importorskip("unsloth_zoo.fused_losses.cross_entropy_loss")
    ppn = _load("pp_native")
    from unsloth.models.deepseek_v41 import modeling as dsv41
    from unsloth.models.deepseek_v41.configuration import DeepseekV41Config
finally:
    if _FORCE_CPU is None:
        os.environ.pop("UNSLOTH_FORCE_CPU", None)
    else:
        os.environ["UNSLOTH_FORCE_CPU"] = _FORCE_CPU

VOCAB = 512
SEQ = 24


def test_modules_come_from_this_tree():
    root = os.path.join(REPO, "unsloth", "models")
    assert os.path.dirname(ppn.__file__) == root
    assert os.path.dirname(os.path.dirname(dsv41.__file__)) == root


def build(seed = 0):
    text = dict(
        vocab_size = VOCAB,
        hidden_size = 64,
        moe_intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        head_dim = 32,
        qk_rope_head_dim = 8,
        q_lora_rank = 32,
        o_groups = 4,
        o_lora_rank = 16,
        n_routed_experts = 4,
        num_experts_per_tok = 2,
        sliding_window = 8,
        compress_ratios = [0, 0, 0],
        engram_layer_ids = [1],
        engram_vocab_size = 101,
        engram_max_ngram_size = 3,
        engram_n_heads = 2,
        engram_head_dim = 32,
        engram_compressed_vocab_size = 97,
        num_nextn_predict_layers = 1,
        max_position_embeddings = 1024,
    )
    primes, offsets = dsv41.engram_layout(
        tuple(text["engram_layer_ids"]),
        text["engram_max_ngram_size"],
        text["engram_n_heads"],
        text["engram_vocab_size"],
    )
    text["engram_num_embeddings"] = [o[-1] + p[-1][-1] for o, p in zip(offsets, primes)]
    config = DeepseekV41Config(
        text_config = text, vision_config = {"model_type": "deepseek_v41_vision"}
    )
    torch.manual_seed(seed)
    model = dsv41.DeepseekV41ForCausalLM(config).float()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 or name.endswith(("q_weight", "k_weight")):
                p.add_(torch.randn_like(p) * 0.1)
    model.bind_token_map([i % 97 for i in range(VOCAB)])
    # A model built from a config has no checkpoint to bind its engram tables to: give them rows.
    for rows, layer_id in zip(text["engram_num_embeddings"], text["engram_layer_ids"]):
        model.model.set_engram_table(layer_id, torch.randn(rows, text["engram_head_dim"]) * 0.5)
    for layer in model.model.layers:
        if layer.engram is not None:
            layer.engram.clamp_value = 1e-2
    model.config._attn_implementation = "eager"
    return model


def batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(1, VOCAB - 1, (2, SEQ), generator = g)
    labels = ids.clone()
    labels[0, :5] = -100
    labels[1, -3:] = -100
    return ids, labels


def _stage(model, n_items):
    """One stage owning every layer (first and last), as _model_stage_forward reads it."""
    ctx = types.SimpleNamespace(
        stage_forward = (model.model, model, None),
        is_first = True,
        is_last = True,
        start = 0,
        end = len(model.model.layers),
        num_items_in_batch = n_items,
    )
    return types.SimpleNamespace(_pp_ctx = ctx, _wire = lambda t: t)


def _grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def _stage_loss(model, ids, labels, n_items):
    model.zero_grad(set_to_none = True)
    loss = ppn._PPStageModule._model_stage_forward(_stage(model, n_items), ids, {"labels": labels})
    loss.backward()
    return loss.detach(), _grads(model)


def _reference(model, ids, labels, n_items):
    """The model's own forward: lm_head then loss_function."""
    model.zero_grad(set_to_none = True)
    extra = {} if n_items is None else {"num_items_in_batch": n_items}
    loss = model(input_ids = ids, labels = labels, **extra).loss
    loss.backward()
    return loss.detach(), _grads(model)


def _assert_parity(got, want):
    (loss, grads), (ref_loss, ref_grads) = got, want
    assert abs(float(loss) - float(ref_loss)) <= 1e-6, (float(loss), float(ref_loss))
    assert set(grads) == set(ref_grads)
    for name, ref in ref_grads.items():
        diff = float((grads[name] - ref).abs().max())
        assert diff <= 1e-6 + 1e-5 * float(ref.abs().max()), (name, diff)


class _VocabLogits(TorchDispatchMode):
    """Records every op output shaped [*, *, vocab]: the full logits the fused route never builds."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab, self.seen = vocab, []

    def __torch_dispatch__(
        self,
        func,
        types,
        args = (),
        kwargs = None,
    ):
        out = func(*args, **(kwargs or {}))
        for t in out if isinstance(out, (tuple, list)) else (out,):
            if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.shape[-1] == self.vocab:
                self.seen.append((str(func), tuple(t.shape)))
        return out


def _stats():
    stats = getattr(ppn, "PP_FUSED_CE_STATS", None)
    assert stats is not None, "pp_native has no fused last-stage CE"
    return dict(stats)


@pytest.mark.parametrize("n_items", [None, 37])
def test_last_stage_fused_ce_matches_the_logits_route(n_items):
    model = build()
    ids, labels = batch()
    before = _stats()
    got = _stage_loss(model, ids, labels, n_items)
    after = _stats()
    assert after["fused"] - before["fused"] == 1 and after["logits"] == before["logits"]
    _assert_parity(got, _reference(model, ids, labels, n_items))
    assert "lm_head.weight" in got[1]


def test_last_stage_builds_no_vocab_logits():
    """Flip: the merge base runs lm_head(h) and builds [mb, seq, vocab] logits (and their fp32 copy / grad)."""
    model = build()
    ids, labels = batch()
    calls = []
    hook = model.lm_head.register_forward_hook(lambda *a: calls.append(1))
    try:
        with _VocabLogits(VOCAB) as mode:
            loss = ppn._PPStageModule._model_stage_forward(
                _stage(model, None), ids, {"labels": labels}
            )
            loss.backward()
    finally:
        hook.remove()
    assert calls == [] and mode.seen == [], (len(calls), mode.seen[:4])


def test_kill_switch_and_unfusable_heads_take_the_logits_route(monkeypatch):
    ids, labels = batch()
    model = build()
    want = _reference(model, ids, labels, 37)
    monkeypatch.setenv("UNSLOTH_PP_FUSED_CE", "0")
    before = _stats()
    _assert_parity(_stage_loss(model, ids, labels, 37), want)
    assert _stats()["logits"] - before["logits"] == 1 and _stats()["fused"] == before["fused"]
    monkeypatch.delenv("UNSLOTH_PP_FUSED_CE")

    # an unsloth_zoo without the fused CE
    monkeypatch.setitem(sys.modules, "unsloth_zoo.fused_losses.cross_entropy_loss", None)
    before = _stats()
    _assert_parity(_stage_loss(model, ids, labels, 37), want)
    assert _stats()["logits"] - before["logits"] == 1 and _stats()["fused"] == before["fused"]
    monkeypatch.delitem(sys.modules, "unsloth_zoo.fused_losses.cross_entropy_loss")

    # a head bias leaves lm_head.weight alone short of the logits
    biased = build()
    biased.lm_head.bias = torch.nn.Parameter(torch.randn(VOCAB) * 0.1)
    want = _reference(biased, ids, labels, None)
    before = _stats()
    _assert_parity(_stage_loss(biased, ids, labels, None), want)
    assert _stats()["logits"] - before["logits"] == 1 and _stats()["fused"] == before["fused"]


def test_lora_head_takes_the_logits_route_and_a_plain_peft_head_fuses():
    from peft import LoraConfig, get_peft_model
    ids, labels = batch()
    for targets, route in ((["lm_head", "q_b_proj"], "logits"), (["q_b_proj"], "fused")):
        torch.manual_seed(3)
        peft = get_peft_model(
            build(), LoraConfig(r = 4, target_modules = targets, init_lora_weights = False)
        )
        model = peft.get_base_model()
        want = _reference(model, ids, labels, 37)
        before = _stats()
        _assert_parity(_stage_loss(model, ids, labels, 37), want)
        after = _stats()
        assert after[route] - before[route] == 1, (targets, before, after)
        if route == "logits":  # the head adapter trains on this route
            assert any("lm_head.lora_B" in n for n in want[1])


class _Scaler:
    """GradScaler stand-in: the scale the pipeline's _loss_fn multiplies the returned loss by."""

    def __init__(self, scale):
        self.scale_value = scale

    def get_scale(self):
        return self.scale_value


def test_fused_ce_saves_gradients_at_the_pipeline_loss_scale(monkeypatch):
    """float16 + GradScaler: the fused CE computes its gradients in the forward, so they must be saved at the
    step's loss scale (else they underflow before the scaled backward reaches them). The returned loss stays
    unscaled and the scaled backward gives the scaled gradients, not the scale squared."""
    from unsloth_zoo.fused_losses import cross_entropy_loss as zoo_ce

    seen = []
    original = zoo_ce.unsloth_fused_ce_loss

    def recording(*args, **kwargs):
        seen.append(kwargs.get("scaling"))
        return original(*args, **kwargs)

    monkeypatch.setattr(zoo_ce, "unsloth_fused_ce_loss", recording)
    model = build()
    ids, labels = batch()
    scale = 2.0**12
    stage = _stage(model, 37)
    stage._pp_ctx.loss_scaler = _Scaler(scale)
    model.zero_grad(set_to_none = True)
    loss = ppn._PPStageModule._model_stage_forward(stage, ids, {"labels": labels})
    (loss * scale).backward()  # PipelineParallel._loss_fn: scaler.scale(loss)
    assert seen == [scale]
    unscaled = {n: g / scale for n, g in _grads(model).items()}
    _assert_parity((loss.detach(), unscaled), _reference(model, ids, labels, 37))
