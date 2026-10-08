# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Under LoRA the accepts_loss_kwargs walk must reach the loss head, not the backbone."""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_accepts_loss_kwargs_remote_mean_loss import _load, _models  # noqa: E402


_HEADS = '''
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


def unsloth_fused_ce_loss(**kwargs):
    return kwargs


def unsloth_fused_lm_head_loss(*args, **kwargs):
    return kwargs


class Backbone(nn.Module):
    accepts_loss_kwargs = False

    def forward(self, input_ids=None, **kwargs):
        return input_ids


class HFStyle(nn.Module):
    base_model_prefix = "model"

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix)


class NativeForConditionalGeneration(HFStyle):
    """Qwen2.5-VL / Gemma 3 shape: backbone declares False, head loss divides by the count."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        return self.loss_function(logits, labels, vocab_size=1, **kwargs)


class StaleFlagForConditionalGeneration(NativeForConditionalGeneration):
    """Gemma 4 on transformers 5.17: a stale False on the head itself."""
    accepts_loss_kwargs = False


class LegacyMeanForConditionalGeneration(HFStyle):
    """Gemma 4 on transformers 5.5: attention filtered CrossEntropyLoss mean, kwargs unused."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :]
                shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask != 0].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, 1), shift_labels.view(-1))
        return loss


def Compiled_forward(self, input_ids=None, labels=None, **kwargs):
    hidden_states = self.model(input_ids)
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None)
    if labels is None:
        loss = None
    elif n_items is not None:
        loss = unsloth_fused_ce_loss(hidden_states=hidden_states, labels=labels, n_items=n_items)
    else:
        loss = self.loss_function(hidden_states, labels, vocab_size=1, **kwargs)
    return loss


class CompiledForConditionalGeneration(HFStyle):
    """The compile cache's standalone class: a thin forward into the rewritten function."""
    accepts_loss_kwargs = False

    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return Compiled_forward(self, input_ids=input_ids, labels=labels, **kwargs)


def _temporary_patch_wrap(forward):
    # zoo temporary_patches/gemma4_moe.py: a functools.wraps wrapper over the compiled forward,
    # so the class forward's own co_filename is the patch module, not the compile cache.
    import functools

    @functools.wraps(forward)
    def wrapped(self, *args, **kwargs):
        return forward(self, *args, **kwargs)
    return wrapped


class WrappedCompiledForConditionalGeneration(CompiledForConditionalGeneration):
    forward = _temporary_patch_wrap(CompiledForConditionalGeneration.forward)


class AstLegacyForCausalLM(HFStyle):
    """zoo AST legacy route: fused, but the count is never passed, so the loss is a mean."""
    accepts_loss_kwargs = True

    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.lm_head = nn.Linear(1, 1)

    def forward(self, input_ids=None, labels=None, **kwargs):
        hidden_states = self.model(input_ids)
        if labels is not None:
            loss = unsloth_fused_lm_head_loss(hidden_states, self.lm_head, labels)
        else:
            loss = None
        return loss


class MixedAuxForCausalLM(HFStyle):
    """Main loss divides by the count, an auxiliary head loss is a plain mean (CSM depth decoder)."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        loss = self.loss_function(logits, labels, vocab_size=1, **kwargs)
        aux = CrossEntropyLoss()(logits.view(-1, 1), labels.view(-1))
        return loss + aux


class ReturnLogitsFallbackForConditionalGeneration(HFStyle):
    """zoo regex rewrite of a stock `loss_function(...)` without **kwargs (Qwen3-VL <= 5.16.1):
    the fused branch gets the count, the UNSLOTH_RETURN_LOGITS=1 branch still returns a mean."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        n_items = kwargs.get("num_items_in_batch", None)
        logits = self.model(input_ids)
        if os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "0":
            return unsloth_fused_ce_loss(hidden_states = logits, labels = labels, n_items = n_items)
        return self.loss_function(logits, labels, vocab_size = 1)


class CopiedKwargsForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **lm_kwargs):
        loss_kwargs = dict(lm_kwargs)
        logits = self.model(input_ids)
        return self.loss_function(logits, labels, vocab_size=1, **loss_kwargs)


class PoppedCountForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        return self.loss_function(self.model(input_ids), labels, vocab_size=1, **kwargs)


class FilteredCountForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k != "num_items_in_batch"}
        return self.loss_function(self.model(input_ids), labels, vocab_size=1, **kwargs)


class SwallowedCountForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        return self.loss_function(self.model(input_ids), labels, vocab_size=1, **kwargs)


class NestedLossForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        def compute(**kwargs):
            return self.loss_function(self.model(input_ids), labels, vocab_size=1, **kwargs)
        return compute()


class WrongKeyForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        n_items = kwargs.get("packed_seq_lengths")
        if "num_items_in_batch" in kwargs:
            pass
        return unsloth_fused_ce_loss(hidden_states = self.model(input_ids), labels = labels, n_items = n_items)


class OuterRewardModule(nn.Module):
    """A user module holding an HF head but computing its own mean loss (ShieldGemma2-style shape)."""
    def __init__(self):
        super().__init__()
        self.model = StaleFlagForConditionalGeneration()

    def forward(self, input_ids=None, labels=None):
        logits = self.model.model(input_ids)
        return CrossEntropyLoss()(logits.view(-1, 1), labels.view(-1))


class GuardedFallbackForConditionalGeneration(HFStyle):
    """zoo regex rewrite after the count is threaded: fused branch, RETURN_LOGITS branch with the
    count, non-causal branch passing it only when there is one."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        n_items = None
        if n_items is None and 'kwargs' in locals():
            n_items = kwargs.get("num_items_in_batch", None)
            if n_items is None: n_items = kwargs.get("n_items", None)
        logits = self.model(input_ids)
        if os.environ.get("UNSLOTH_RETURN_LOGITS", "0") == "0":
            return unsloth_fused_ce_loss(hidden_states = logits, labels = labels, n_items = n_items)
        elif self.loss_function.__name__.endswith("ForCausalLMLoss"):
            return self.loss_function(logits, labels, vocab_size = 1, num_items_in_batch = n_items)
        return self.loss_function(logits, labels, vocab_size = 1, **({} if n_items is None else {'num_items_in_batch': n_items}))


class UncountedStockLossForCausalLM(HFStyle):
    """transformers 5.16 XGLM: a stock loss_function called without the count reduces by the mean."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.loss_function(self.model(input_ids), labels, vocab_size = 1)


def _user_loss(logits, labels, vocab_size = None, **kwargs):
    # Takes the keyword, still averages.
    return CrossEntropyLoss()(logits.view(-1, 1), labels.view(-1))


class CustomLossForCausalLM(NativeForConditionalGeneration):
    def __init__(self):
        super().__init__()
        self.loss_function = _user_loss


class DelegatedMethodMeanForCausalLM(HFStyle):
    """ProphetNet: labels handed to a method that averages them."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def _compute_loss(self, logits, labels, ignore_index = -100):
        return torch.nn.functional.nll_loss(logits, labels.view(-1), reduction = "mean")

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
        return loss


class _PredLayer(nn.Module):
    def forward(self, x, y = None):
        return torch.nn.functional.cross_entropy(x, y.view(-1), reduction = "mean")


class DelegatedChildMeanLMHeadModel(HFStyle):
    """XLM: labels handed positionally to a child module whose forward averages them."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.pred_layer = _PredLayer()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.pred_layer(self.model(input_ids), labels)


class _TextDecoder(nn.Module):
    def forward(self, input_ids = None, labels = None, reduction = "mean", **kwargs):
        loss_fct = CrossEntropyLoss(reduction = reduction, label_smoothing = 0.1)
        return loss_fct(input_ids, labels)


class DelegatedReductionForConditionalGeneration(HFStyle):
    """BLIP captioning: labels plus an explicit reduction="mean" handed to the text decoder."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.text_decoder = _TextDecoder()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.text_decoder(input_ids = input_ids, labels = labels, reduction = "mean")


class _CountingDecoder(nn.Module):
    def forward(self, input_ids = None, labels = None, **kwargs):
        return unsloth_fused_ce_loss(hidden_states = input_ids, labels = labels, n_items = kwargs.get("num_items_in_batch", None))


class DelegatedCountForConditionalGeneration(HFStyle):
    """Labels and **kwargs (the count) handed to a child that divides by it."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.decoder = _CountingDecoder()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.decoder(input_ids = input_ids, labels = labels, **kwargs)


class DelegatedNoCountForConditionalGeneration(DelegatedCountForConditionalGeneration):
    """Same child, but the count never reaches it, so it falls back to a mean."""
    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.decoder(input_ids = input_ids, labels = labels)


class CompositeMeanEncoderDecoderModel(HFStyle):
    """EncoderDecoderModel: the count moved to the decoder's kwargs, the loss its own CE mean."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs_decoder = {}
        if "num_items_in_batch" in kwargs:
            kwargs_decoder["num_items_in_batch"] = kwargs.pop("num_items_in_batch", None)
        logits = self.model(input_ids, **kwargs_decoder)
        loss_fct = CrossEntropyLoss()
        return loss_fct(logits.reshape(-1, 1), labels.view(-1))


def Thin_forward(self, input_ids = None, labels = None, **kwargs):
    logits = self.model(input_ids)
    return CrossEntropyLoss()(logits.view(-1, 1), labels.view(-1))


class ThinCompiledForConditionalGeneration(HFStyle):
    """Qwen2-Audio in the compile cache: a thin class forward over a mean CE implementation."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids = None, labels = None, **kwargs):
        return Thin_forward(self, input_ids = input_ids, labels = labels, **kwargs)


class ClearedCarrierForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        extra = dict(kwargs)
        extra.clear()
        return self.loss_function(self.model(input_ids), labels, vocab_size=1, **extra)


class UpdatedCarrierForCausalLM(HFStyle):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        kwargs.update(dict(scale = 1))
        return self.loss_function(self.model(input_ids), labels, vocab_size=1, **kwargs)


class ForwardingWrapper(nn.Module):
    """A user wrapper under a custom attribute name, handing its **kwargs to the head."""
    def __init__(self, head):
        super().__init__()
        self.inner = head

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


class DroppingWrapper(nn.Module):
    """A user wrapper that calls the head without the count."""
    def __init__(self, head):
        super().__init__()
        self.inner = head

    def forward(self, input_ids = None, labels = None, **kwargs):
        return self.inner(input_ids = input_ids, labels = labels)


def mask_attention_mask_out(labels = None, attention_mask = None):
    return labels


class WrappedLabelsReductionForConditionalGeneration(DelegatedReductionForConditionalGeneration):
    """BLIP in the compile cache: labels handed on wrapped by the compiler's attention-mask helper."""
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.text_decoder(
            input_ids = input_ids,
            labels = mask_attention_mask_out(labels = labels, attention_mask = attention_mask),
            reduction = "mean",
        )


def unsloth_loss_count_kwargs(loss_function, n_items):
    return {} if n_items is None else {"num_items_in_batch": n_items}


class CountKwargsForCausalLM(HFStyle):
    """zoo hook rewrite: the count reaches loss_function through unsloth_loss_count_kwargs."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        return self.loss_function(
            logits, labels, vocab_size=1,
            **unsloth_loss_count_kwargs(self.loss_function, kwargs.get("num_items_in_batch", None)),
        )


class FlaggedForwardingWrapper(nn.Module):
    """A user wrapper with its own stale flag, forwarding **kwargs to an `inner` head."""
    accepts_loss_kwargs = False

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


class NoneCountWrapper(nn.Module):
    """Names the count but hands its child None, so the count never reaches the head."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, input_ids=None, labels=None):
        return self.inner(input_ids=input_ids, labels=labels, num_items_in_batch=None)


class KwOnlyMeanDelegateForCausalLM(HFStyle):
    """Hands labels to a helper whose keyword-only reduction defaults to a mean."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def _loss(self, logits, labels, *, reduction="mean"):
        return torch.nn.functional.cross_entropy(logits, labels, reduction=reduction)

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        return self._loss(logits, labels)


class MarkedForConditionalGeneration(NativeForConditionalGeneration):
    _unsloth_counts_unshifted_labels = True


class FallbackCountReadForCausalLM(HFStyle):
    """zoo count route: n_items falls back to the alias when num_items_in_batch is present but None."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        return unsloth_count_aware_cross_entropy(logits, labels, n_items=(kwargs.get('num_items_in_batch', None) if kwargs.get('num_items_in_batch', None) is not None else kwargs.get('n_items', None)), shift=False)


class LoraLike(nn.Module):
    """peft BaseTuner: the wrapped model under `model`, unknown attributes forwarded to it."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)



class _MeanDecoder(nn.Module):
    def forward(self, input_ids = None, labels = None, **kwargs):
        return CrossEntropyLoss()(input_ids.view(-1, 1), labels.view(-1))


class CountedWithDelegatedMeanForCausalLM(HFStyle):
    """Main loss divides by the count, an auxiliary mean is computed by a child receiving labels."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.aux_decoder = _MeanDecoder()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        loss = self.loss_function(logits, labels, vocab_size=1, **kwargs)
        return loss + self.aux_decoder(input_ids = logits, labels = labels)


class MixedDelegatesForConditionalGeneration(HFStyle):
    """One child divides by the count, another averages: not provably either."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.decoder = _CountingDecoder()
        self.aux_decoder = _MeanDecoder()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.decoder(input_ids = input_ids, labels = labels, **kwargs) + self.aux_decoder(input_ids = input_ids, labels = labels)


class DelegatedPositionalSumForConditionalGeneration(HFStyle):
    """reduction="sum" passed positionally must override the child's "mean" default."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.text_decoder = _TextDecoder()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.text_decoder(input_ids, labels, "sum")


def Dropping_forward(self, input_ids=None, labels=None, **kwargs):
    return unsloth_fused_ce_loss(hidden_states=input_ids, labels=labels, n_items=kwargs.get("num_items_in_batch", None))


class DroppingThinForConditionalGeneration(HFStyle):
    """A thin class forward that never hands its **kwargs to the implementation."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return Dropping_forward(self, input_ids=input_ids, labels=labels)


class OwnsModuleChildForCausalLM(nn.Module):
    """A loss head whose backbone is `transformer` and which also owns a child named `module`."""
    def __init__(self):
        super().__init__()
        self.transformer = Backbone()
        self.module = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        return self.loss_function(self.transformer(input_ids), labels, vocab_size=1, **kwargs)

class PeftModelForCausalLM(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_model = LoraLike(model)

    def get_base_model(self):
        return self.base_model.model

    def forward(self, *args, **kwargs):
        return self.get_base_model()(*args, **kwargs)
'''


@pytest.fixture
def ns():
    return _load()


@pytest.fixture
def mods(tmp_path):
    return _models(tmp_path, _HEADS, "loss_heads_for_ga_test")


def _trainer_sees(model):
    head = model.get_base_model() if hasattr(model, "get_base_model") else model
    if hasattr(head, "accepts_loss_kwargs"):
        return head.accepts_loss_kwargs
    import inspect

    return any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in inspect.signature(head.forward).parameters.values()
    )


def test_old_getattr_walk_lands_on_the_backbone(mods):
    peft = mods.PeftModelForCausalLM(mods.NativeForConditionalGeneration())
    assert type(peft.base_model.base_model).__name__ == "Backbone"


def test_walk_reaches_the_head_under_peft(ns, mods):
    head = mods.NativeForConditionalGeneration()
    peft = mods.PeftModelForCausalLM(head)
    chain = [type(m).__name__ for m in ns["_loss_kwargs_chain"](peft)]
    assert chain == [
        "PeftModelForCausalLM",
        "LoraLike",
        "NativeForConditionalGeneration",
        "Backbone",
    ]
    assert ns["_loss_head"](peft) is head


@pytest.mark.parametrize("wrap", ["data_parallel", "torch_compile"])
def test_walk_reaches_the_head_through_training_wrappers(ns, mods, wrap):
    head = mods.NativeForConditionalGeneration()
    peft = mods.PeftModelForCausalLM(head)
    outer = torch.nn.DataParallel(peft) if wrap == "data_parallel" else torch.compile(peft)
    assert any(m is head for m in ns["_loss_kwargs_chain"](outer))
    assert ns["_loss_head"](outer) is head


@pytest.mark.parametrize("attr", ["module", "_orig_mod", "_fsdp_wrapped_module"])
def test_a_plain_module_with_a_wrapper_named_child_is_not_a_training_wrapper(ns, mods, attr):
    outer = nn.Module()
    outer.add_module(attr, mods.NativeForConditionalGeneration())
    assert ns["_loss_head"](outer) is None


@pytest.mark.parametrize(
    "cls",
    [
        "NativeForConditionalGeneration",
        "StaleFlagForConditionalGeneration",
        "CompiledForConditionalGeneration",
        "WrappedCompiledForConditionalGeneration",
    ],
)
def test_a_consuming_head_gets_the_count_despite_a_false_flag(ns, mods, cls):
    head = getattr(mods, cls)()
    peft = mods.PeftModelForCausalLM(head)
    result = ns["apply_accepts_loss_kwargs_fix"](peft)
    assert result.startswith("True"), result
    assert _trainer_sees(peft) is True
    assert _trainer_sees(head) is True


def test_full_finetuning_head_gets_the_count(ns, mods):
    head = mods.StaleFlagForConditionalGeneration()
    ns["apply_accepts_loss_kwargs_fix"](head)
    assert head.accepts_loss_kwargs is True


def test_a_legacy_mean_head_falls_back(ns, mods):
    head = mods.LegacyMeanForConditionalGeneration()
    assert ns["_forward_consumes_num_items_in_batch"](head) is False
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert _trainer_sees(peft) is False


def test_fused_without_the_count_is_a_mean_even_if_declared(ns, mods):
    head = mods.AstLegacyForCausalLM()
    assert ns["_forward_consumes_num_items_in_batch"](head) is False
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert _trainer_sees(peft) is False


def test_mixed_main_and_aux_mean_is_undecided(ns, mods):
    assert ns["_forward_consumes_num_items_in_batch"](mods.MixedAuxForCausalLM()) is None


def test_a_mean_fallback_branch_is_undecided(ns, mods, monkeypatch):
    head = mods.ReturnLogitsFallbackForConditionalGeneration()
    monkeypatch.setitem(ns, "_zoo_counts_fallback_branches", lambda: True)
    assert ns["_forward_consumes_num_items_in_batch"](head) is None
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert _trainer_sees(peft) is False


def test_an_older_zoo_keeps_the_fused_branch_answer(ns, mods, monkeypatch):
    # An older unsloth_zoo never gives these fallbacks the count; only the fused branch trains.
    head = mods.ReturnLogitsFallbackForConditionalGeneration()
    monkeypatch.setitem(ns, "_zoo_counts_fallback_branches", lambda: False)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    assert ns["_forward_consumes_num_items_in_batch"](head) is True
    monkeypatch.setenv("UNSLOTH_RETURN_LOGITS", "1")
    assert ns["_forward_consumes_num_items_in_batch"](head) is None
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS")
    monkeypatch.setitem(ns, "_zoo_counts_fallback_branches", lambda: True)
    assert ns["_forward_consumes_num_items_in_batch"](head) is None


def test_copied_kwargs_dict_counts(ns, mods):
    assert ns["_forward_consumes_num_items_in_batch"](mods.CopiedKwargsForCausalLM()) is True


def test_a_second_call_redecides_its_own_shadow(ns, mods):
    head = mods.StaleFlagForConditionalGeneration()
    ns["apply_accepts_loss_kwargs_fix"](head)
    assert head.accepts_loss_kwargs is True
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert _trainer_sees(peft) is True
    legacy = mods.AstLegacyForCausalLM.forward.__get__(head)
    head.forward = legacy
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert _trainer_sees(peft) is False


def test_a_user_assignment_on_the_head_still_wins(ns, mods):
    head = mods.LegacyMeanForConditionalGeneration()
    head.accepts_loss_kwargs = True
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert head.accepts_loss_kwargs is True
    assert peft.accepts_loss_kwargs is True


def test_remote_mean_loss_stays_false_under_the_peft_walk(ns, tmp_path, mods):
    remote = _models(tmp_path)
    head = remote.NemotronHForCausalLM()
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert head.accepts_loss_kwargs is False
    assert _trainer_sees(peft) is False


def _bare(cls):
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    return obj


@pytest.mark.parametrize(
    "module, name",
    [
        ("transformers.models.gemma4.modeling_gemma4", "Gemma4ForConditionalGeneration"),
        (
            "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl",
            "Qwen2_5_VLForConditionalGeneration",
        ),
        ("transformers.models.gemma3.modeling_gemma3", "Gemma3ForConditionalGeneration"),
        ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
    ],
)
def test_installed_transformers_heads_consume(ns, module, name):
    import importlib

    try:
        cls = getattr(importlib.import_module(module), name)
    except Exception as exc:
        pytest.skip(reason = f"{name} not in the installed transformers: {exc}")
    import inspect

    source = inspect.getsource(cls.forward)
    result = ns["_forward_consumes_num_items_in_batch"](_bare(cls))
    import ast
    import textwrap

    counted = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "loss_function"
        and any(kw.arg in (None, "num_items_in_batch") for kw in node.keywords)
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
    )
    if not counted:
        assert result is not True
        return
    assert result is True


@pytest.mark.parametrize(
    "cls",
    [
        "PoppedCountForCausalLM",
        "FilteredCountForCausalLM",
        "SwallowedCountForCausalLM",
        "NestedLossForCausalLM",
        "WrongKeyForCausalLM",
    ],
)
def test_a_count_dropped_before_the_loss_is_not_consuming(ns, mods, cls):
    assert ns["_forward_consumes_num_items_in_batch"](getattr(mods, cls)()) is not True


def test_an_outer_module_with_its_own_loss_is_not_its_inner_head(ns, mods):
    outer = mods.OuterRewardModule()
    assert ns["_loss_head"](outer) is None
    peft = mods.PeftModelForCausalLM(outer)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert _trainer_sees(peft) is not True


def test_a_fully_threaded_fallback_is_consuming(ns, mods):
    head = mods.GuardedFallbackForConditionalGeneration()
    assert ns["_forward_consumes_num_items_in_batch"](head) is True


def test_the_count_aware_cross_entropy_helper_is_a_counted_loss(ns, tmp_path):
    from test_accepts_loss_kwargs_remote_mean_loss import _models as _mk

    src = (
        _HEADS
        + """

def unsloth_count_aware_cross_entropy(logits, labels, n_items = None, **kwargs):
    return logits


class AlignedCountedForCausalLM(HFStyle):
    _unsloth_counts_unshifted_labels = True

    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        n_items = kwargs.get("num_items_in_batch", None)
        logits = self.model(input_ids)
        return unsloth_count_aware_cross_entropy(logits, labels, n_items = n_items, shift = False)


class AlignedUncountedForCausalLM(AlignedCountedForCausalLM):
    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.model(input_ids)
        return unsloth_count_aware_cross_entropy(logits, labels, shift = False)
"""
    )
    m = _mk(tmp_path, src, "count_aware_heads")
    assert ns["_forward_consumes_num_items_in_batch"](m.AlignedCountedForCausalLM()) is True
    assert ns["_forward_consumes_num_items_in_batch"](m.AlignedUncountedForCausalLM()) is False


@pytest.mark.parametrize(
    "cls",
    [
        "UncountedStockLossForCausalLM",
        "DelegatedMethodMeanForCausalLM",
        "DelegatedChildMeanLMHeadModel",
        "DelegatedReductionForConditionalGeneration",
        "DelegatedNoCountForConditionalGeneration",
        "CompositeMeanEncoderDecoderModel",
        "ThinCompiledForConditionalGeneration",
        "WrappedLabelsReductionForConditionalGeneration",
    ],
)
def test_a_mean_loss_reached_through_labels_is_false(ns, mods, cls):
    head = getattr(mods, cls)()
    assert ns["_forward_consumes_num_items_in_batch"](head) is False
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert _trainer_sees(peft) is False


def test_labels_and_the_count_handed_to_a_counting_child_consume(ns, mods):
    head = mods.DelegatedCountForConditionalGeneration()
    assert ns["_forward_consumes_num_items_in_batch"](head) is True


def test_a_user_loss_function_is_undecided(ns, mods):
    assert ns["_forward_consumes_num_items_in_batch"](mods.CustomLossForCausalLM()) is None


@pytest.mark.parametrize("cls", ["ClearedCarrierForCausalLM", "UpdatedCarrierForCausalLM"])
def test_a_mutated_carrier_is_not_consuming(ns, mods, cls):
    assert ns["_forward_consumes_num_items_in_batch"](getattr(mods, cls)()) is not True


def test_a_forwarding_wrapper_head_is_shadowed_too(ns, mods):
    head = mods.StaleFlagForConditionalGeneration()
    wrapper = mods.ForwardingWrapper(head)
    peft = mods.PeftModelForCausalLM(wrapper)
    assert ns["_loss_head"](peft) is head
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert wrapper.accepts_loss_kwargs is True
    assert head.accepts_loss_kwargs is True
    assert ns["_instance_accepts_loss_kwargs"](peft) is None


def test_a_wrapper_dropping_the_count_is_not_transparent(ns, mods):
    head = mods.NativeForConditionalGeneration()
    assert ns["_loss_head"](mods.DroppingWrapper(head)) is None


@pytest.mark.parametrize(
    "module, name",
    [
        ("transformers.models.xlm.modeling_xlm", "XLMWithLMHeadModel"),
        ("transformers.models.encoder_decoder.modeling_encoder_decoder", "EncoderDecoderModel"),
        ("transformers.models.speecht5.modeling_speecht5", "SpeechT5ForSpeechToText"),
        (
            "transformers.models.prophetnet.modeling_prophetnet",
            "ProphetNetForConditionalGeneration",
        ),
    ],
)
def test_installed_transformers_delegated_or_composite_heads(ns, module, name):
    import importlib

    try:
        cls = getattr(importlib.import_module(module), name)
    except Exception as exc:
        pytest.skip(reason = f"{name} not in the installed transformers: {exc}")
    head = _bare(cls)
    assert ns["_loss_head"](head) is head
    assert ns["_forward_consumes_num_items_in_batch"](head) is not True


def test_installed_transformers_classification_heads_are_still_not_loss_heads(ns):
    from transformers.models.llama.modeling_llama import LlamaForSequenceClassification
    assert ns["_loss_head"](_bare(LlamaForSequenceClassification)) is None


def test_the_count_through_unsloth_loss_count_kwargs_is_consuming(ns, mods):
    assert ns["_forward_consumes_num_items_in_batch"](mods.CountKwargsForCausalLM()) is True


def test_a_wrapper_with_its_own_stale_flag_gets_the_decision(ns, mods):
    outer = mods.FlaggedForwardingWrapper(mods.NativeForConditionalGeneration())
    ns["apply_accepts_loss_kwargs_fix"](outer)
    assert outer.accepts_loss_kwargs is True
    assert outer.inner.accepts_loss_kwargs is True


def test_a_wrapper_passing_a_none_count_is_not_transparent(ns, mods):
    outer = mods.NoneCountWrapper(mods.NativeForConditionalGeneration())
    assert ns["_loss_head"](outer) is None


def test_a_keyword_only_mean_default_in_a_delegate_is_false(ns, mods):
    head = mods.KwOnlyMeanDelegateForCausalLM()
    assert ns["_forward_consumes_num_items_in_batch"](head) is False


def test_the_count_convention_is_recorded_for_the_batch_counter(ns, mods):
    shifted = mods.PeftModelForCausalLM(mods.NativeForConditionalGeneration())
    ns["apply_accepts_loss_kwargs_fix"](shifted)
    assert ns["_num_items_labels"](shifted) == "shifted"
    assert shifted.get_base_model().__dict__["_unsloth_num_items_labels"] == "shifted"

    unshifted = mods.PeftModelForCausalLM(mods.MarkedForConditionalGeneration())
    ns["apply_accepts_loss_kwargs_fix"](unshifted)
    assert ns["_num_items_labels"](unshifted) == "unshifted"

    mean = mods.PeftModelForCausalLM(mods.AstLegacyForCausalLM())
    ns["apply_accepts_loss_kwargs_fix"](mean)
    assert ns["_num_items_labels"](mean) is None


def test_a_recorded_convention_is_dropped_when_the_forward_stops_counting(ns, mods):
    head = mods.NativeForConditionalGeneration()
    peft = mods.PeftModelForCausalLM(head)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert ns["_num_items_labels"](peft) == "shifted"
    head.forward = mods.AstLegacyForCausalLM.forward.__get__(head)
    head.lm_head = nn.Linear(1, 1)
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert ns["_num_items_labels"](peft) is None


def test_the_dispatcher_reads_the_recorded_convention_through_a_training_wrapper(ns, mods):
    pytest.importorskip("unsloth_zoo.loss_utils").__dict__.get(
        "counts_unshifted_labels"
    ) or pytest.skip(reason = "unsloth_zoo without the unshifted-label count")
    peft = mods.PeftModelForCausalLM(mods.MarkedForConditionalGeneration())
    ns["apply_accepts_loss_kwargs_fix"](peft)
    assert ns["_head_counts_unshifted_labels"](torch.nn.DataParallel(peft)) is True
    plain = mods.PeftModelForCausalLM(mods.NativeForConditionalGeneration())
    ns["apply_accepts_loss_kwargs_fix"](plain)
    assert ns["_head_counts_unshifted_labels"](torch.nn.DataParallel(plain)) is False


def test_a_conditional_count_read_is_consuming(ns, mods):
    assert ns["_forward_consumes_num_items_in_batch"](mods.FallbackCountReadForCausalLM()) is True


def test_a_counted_loss_beside_a_delegated_mean_is_undecided(ns, mods):
    assert (
        ns["_forward_consumes_num_items_in_batch"](mods.CountedWithDelegatedMeanForCausalLM())
        is None
    )


def test_mixed_delegated_verdicts_are_undecided(ns, mods):
    assert (
        ns["_forward_consumes_num_items_in_batch"](mods.MixedDelegatesForConditionalGeneration())
        is None
    )


def test_a_positional_reduction_overrides_the_delegate_default(ns, mods):
    assert (
        ns["_forward_consumes_num_items_in_batch"](
            mods.DelegatedPositionalSumForConditionalGeneration()
        )
        is None
    )


def test_a_thin_forward_that_drops_kwargs_is_not_unwrapped(ns, mods):
    assert (
        ns["_forward_consumes_num_items_in_batch"](mods.DroppingThinForConditionalGeneration())
        is False
    )


def test_a_child_named_module_is_not_walked_as_a_wrapper(ns, mods):
    head = mods.OwnsModuleChildForCausalLM()
    assert ns["_loss_kwargs_child"](head) is not head.module


def test_a_compiled_wrapper_keeps_the_value_and_its_marker_together(ns, mods):
    head = mods.StaleFlagForConditionalGeneration()
    compiled = torch.compile(head, backend = "eager")
    ns["apply_accepts_loss_kwargs_fix"](compiled)
    inner = compiled._orig_mod.__dict__
    assert inner["accepts_loss_kwargs"] is True and ns["_is_guess"](inner)
    assert compiled.__dict__["accepts_loss_kwargs"] is True and ns["_is_guess"](compiled.__dict__)
