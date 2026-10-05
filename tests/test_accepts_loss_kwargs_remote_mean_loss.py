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
"""**kwargs forward returning its own CrossEntropyLoss mean must not count as consuming
num_items_in_batch (NemotronH, GA=4: logged loss 4.22 vs eval 0.84). Helpers loaded by AST, CPU only."""

import ast
import inspect
import os
import pathlib
import re
import sys
import textwrap
import types

import pytest
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

_UTILS = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
_NAMES = {
    "_unsloth_compile_cache_leaves",
    "_forward_is_unsloth_compiled",
    "_find_concrete_accepts_loss_kwargs",
    "_shadow_accepts_loss_kwargs",
    "_forward_ignores_num_items_in_batch",
    "_instance_accepts_loss_kwargs",
    "_ce_calls_all_mean",
    "_CE_PARAMS",
    "_is_const",
    "_DEFAULT_CE_NAMESPACE",
    "_resolve_ce_callee",
    "_dotted_name",
    "_scan_ce_calls",
    "_GUESSED_LOSS_KWARGS",
    "_loss_kwargs_chain",
    "_clear_guessed_accepts_loss_kwargs",
    "_is_guess",
    "_TORCH_CE",
    "apply_accepts_loss_kwargs_fix",
}


def _load():
    src = _UTILS.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NAMES:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _NAMES for t in node.targets
        ):
            keep.append(node)
    ns = {"ast": ast, "re": re, "inspect": inspect, "os": os, "textwrap": textwrap, "torch": torch}
    exec(compile(ast.Module(body = keep, type_ignores = []), str(_UTILS), "exec"), ns)
    return ns


# On disk so inspect.getsource works, like a real remote module.
_MODELS = '''
import torch
from torch import nn
from torch.nn import CrossEntropyLoss


class Backbone(nn.Module):
    def forward(self, input_ids, **kwargs):
        return input_ids


class NemotronHForCausalLM(nn.Module):
    """Remote code: **kwargs only for generate, loss is a plain mean."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):  # for now we need this for generation
        logits = torch.zeros(1)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 1), labels.view(-1))
        return loss


class NativeForCausalLM(nn.Module):
    """transformers-native: kwargs reach self.loss_function, which divides by num_items_in_batch."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1)
        return self.loss_function(logits, labels, 1, **kwargs)


class HandRolledForCausalLM(nn.Module):
    """Own cross_entropy that does honour num_items_in_batch."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, num_items_in_batch=None, **kwargs):
        logits = torch.zeros(1)
        loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
        return loss / num_items_in_batch


class SumLossForCausalLM(nn.Module):
    """Remote code whose objective is a summed loss: dividing by GA would change it."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1)
        loss_fct = CrossEntropyLoss(reduction="sum")
        return loss_fct(logits.view(-1, 1), labels.view(-1))


class ExplicitMeanForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1)
        return torch.nn.functional.cross_entropy(logits, labels, reduction="mean")


class InstanceDeclaredForCausalLM(nn.Module):
    """Remote code that declares loss-kwargs support on the instance in __init__."""
    def __init__(self):
        super().__init__()
        self.model = Backbone()
        self.accepts_loss_kwargs = True

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1)
        loss_fct = CrossEntropyLoss()
        return loss_fct(logits.view(-1, 1), labels.view(-1))


class LossModuleForCausalLM(nn.Module):
    """Builds its CrossEntropyLoss once in __init__ and calls it from forward."""
    def __init__(self, reduction="mean"):
        super().__init__()
        self.model = Backbone()
        self.loss_fct = CrossEntropyLoss(reduction=reduction)

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1)
        return self.loss_fct(logits.view(-1, 1), labels.view(-1))


class QWenLMHeadModel(nn.Module):
    """Remote causal LM named *LMHeadModel rather than *CausalLM."""
    def __init__(self):
        super().__init__()
        self.transformer = Backbone()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1)
        loss_fct = CrossEntropyLoss()
        return loss_fct(logits.view(-1, 1), labels.view(-1))


class NoKwargsForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None):
        loss_fct = CrossEntropyLoss()
        return loss_fct(input_ids, labels)
'''


_PRETRAINED_MODELS = """
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel


class RemoteMeanLossConfig(PretrainedConfig):
    model_type = "remote_mean_loss_for_ga_test"


class RemoteMeanLossForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = RemoteMeanLossConfig

    def __init__(self, config):
        super().__init__(config)
        self.proj = nn.Linear(4, 4)
        self.post_init()

    def _init_weights(self, module):
        pass

    def forward(self, input_ids=None, labels=None, **kwargs):  # for now we need this for generation
        logits = self.proj(torch.zeros(1, 4))
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels.view(-1))
        return loss
"""


def _models(
    tmp_path,
    source = _MODELS,
    name = "remote_models_for_ga_test",
):
    p = tmp_path / f"{name}.py"
    p.write_text(source, encoding = "utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib
        mod = importlib.import_module(name)
        return importlib.reload(mod)
    finally:
        sys.path.remove(str(tmp_path))


class _PeftLike(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.base_model = types.SimpleNamespace(model = inner)
        self.base_model.base_model = None

    def forward(self, *args, **kwargs):
        return None


def test_remote_mean_loss_forward_is_not_a_loss_kwargs_consumer(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.NemotronHForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert getattr(model, "accepts_loss_kwargs", None) is False


def test_remote_mean_loss_under_a_peft_wrapper(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    inner = mods.NemotronHForCausalLM()
    wrapped = _PeftLike(inner)
    ns["apply_accepts_loss_kwargs_fix"](wrapped)
    assert getattr(inner, "accepts_loss_kwargs", None) is False
    assert getattr(wrapped, "accepts_loss_kwargs", None) is False


def test_consumers_keep_the_hf_default(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    for cls in (
        mods.NativeForCausalLM,
        mods.HandRolledForCausalLM,
        mods.NoKwargsForCausalLM,
        mods.SumLossForCausalLM,
    ):
        model = cls()
        ns["apply_accepts_loss_kwargs_fix"](model)
        assert not hasattr(model, "accepts_loss_kwargs"), cls.__name__


def test_explicit_class_attribute_still_wins(tmp_path):
    ns = _load()
    mods = _models(tmp_path)

    class Declared(mods.NemotronHForCausalLM):
        accepts_loss_kwargs = True

    model = Declared()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert model.accepts_loss_kwargs is True


def test_remote_mean_loss_under_a_real_peft_model(tmp_path):
    peft = pytest.importorskip("peft")
    ns = _load()
    mods = _models(tmp_path, _PRETRAINED_MODELS, "remote_pretrained_for_ga_test")
    inner = mods.RemoteMeanLossForCausalLM(mods.RemoteMeanLossConfig())
    wrapped = peft.get_peft_model(
        inner, peft.LoraConfig(r = 2, target_modules = ["proj"], task_type = "CAUSAL_LM")
    )
    assert "CausalLM" in type(wrapped).__name__
    ns["apply_accepts_loss_kwargs_fix"](wrapped)
    assert getattr(wrapped.get_base_model(), "accepts_loss_kwargs", None) is False
    assert getattr(wrapped, "accepts_loss_kwargs", None) is False


def test_explicit_mean_reduction_is_still_a_mean_loss(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.ExplicitMeanForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert getattr(model, "accepts_loss_kwargs", None) is False


def test_instance_declaration_is_kept(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.InstanceDeclaredForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert model.accepts_loss_kwargs is True
    wrapped = _PeftLike(model)
    ns["apply_accepts_loss_kwargs_fix"](wrapped)
    assert model.accepts_loss_kwargs is True


def test_repeated_calls_keep_the_first_answer(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.NemotronHForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    ns["apply_accepts_loss_kwargs_fix"](_PeftLike(model))
    assert model.accepts_loss_kwargs is False


@pytest.mark.parametrize(
    "call",
    [
        'CrossEntropyLoss(None, None, -100, None, "sum")(logits, labels)',
        'torch.nn.functional.cross_entropy(logits, labels, None, None, -100, None, "none")',
        "CrossEntropyLoss(size_average=False)(logits, labels)",
        "CrossEntropyLoss(reduce=False)(logits, labels)",
        "CrossEntropyLoss(*loss_args)(logits, labels)",
        "CrossEntropyLoss(None, False)(logits, labels)",
        "CrossEntropyLoss(None, None, -100, False)(logits, labels)",
        "torch.nn.functional.cross_entropy(logits, labels, None, False)",
        "torch.nn.functional.cross_entropy(logits, labels, None, None, -100, False)",
    ],
)
def test_non_mean_reductions_in_any_spelling_keep_the_hf_default(call):
    ns = _load()
    source = f"def forward(self, logits, labels, **kwargs):\n    return {call}\n"
    assert ns["_ce_calls_all_mean"](source) is False


@pytest.mark.parametrize(
    "call",
    [
        "CrossEntropyLoss()(logits, labels)",
        "CrossEntropyLoss(None, None, -100)(logits, labels)",
        'CrossEntropyLoss(ignore_index=-100, reduction="mean")(logits, labels)',
        "torch.nn.functional.cross_entropy(logits, labels, None, None, -100)",
        "CrossEntropyLoss(size_average=None, reduce=True)(logits, labels)",
    ],
)
def test_mean_reductions_in_any_spelling_count_as_a_mean(call):
    ns = _load()
    source = f"def forward(self, logits, labels, **kwargs):\n    return {call}\n"
    assert ns["_ce_calls_all_mean"](source) is True


def test_a_mention_without_a_call_is_not_a_mean_loss():
    ns = _load()
    source = (
        "def forward(self, logits, labels, **kwargs):\n"
        '    """Unlike CrossEntropyLoss() this returns a sum."""\n'
        "    return (logits - labels).abs().sum()\n"
    )
    assert ns["_ce_calls_all_mean"](source) is False


def test_a_model_helper_named_cross_entropy_is_not_the_torch_op():
    ns = _load()
    source = (
        "def forward(self, logits, labels, **kwargs):\n"
        "    return self.loss_helpers.cross_entropy(logits, labels)\n"
    )
    assert ns["_ce_calls_all_mean"](source) is False
    for torch_spelling in (
        "torch.nn.functional.cross_entropy(logits, labels)",
        "F.cross_entropy(logits, labels)",
        "nn.CrossEntropyLoss()(logits, labels)",
        "CrossEntropyLoss()(logits, labels)",
    ):
        source = f"def forward(self, logits, labels, **kwargs):\n    return {torch_spelling}\n"
        assert ns["_ce_calls_all_mean"](source) is True, torch_spelling


def test_a_wrapper_added_after_load_gets_the_load_time_answer(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    inner = mods.NemotronHForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](inner)

    class UserWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

    wrapped = UserWrapper(inner)
    ns["apply_accepts_loss_kwargs_fix"](wrapped)
    assert wrapped.accepts_loss_kwargs is False
    assert inner.accepts_loss_kwargs is False


def test_the_instance_forward_is_the_one_inspected(tmp_path):
    import functools

    ns = _load()
    mods = _models(tmp_path)

    model = mods.NemotronHForCausalLM()

    def consuming_forward(
        input_ids = None,
        labels = None,
        num_items_in_batch = None,
        **kwargs,
    ):
        return None

    model.forward = consuming_forward
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert not hasattr(model, "accepts_loss_kwargs")

    # An accelerate-style hook (partial + update_wrapper) still resolves to the original mean-loss forward.
    hooked = mods.NemotronHForCausalLM()
    old_forward = hooked.forward

    def new_forward(module, *args, **kwargs):
        return old_forward(*args, **kwargs)

    hooked.forward = functools.update_wrapper(functools.partial(new_forward, hooked), old_forward)
    ns["apply_accepts_loss_kwargs_fix"](hooked)
    assert hooked.accepts_loss_kwargs is False


def test_a_loss_module_built_in_init_counts_by_its_reduction(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.LossModuleForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert model.accepts_loss_kwargs is False
    summed = mods.LossModuleForCausalLM(reduction = "sum")
    ns["apply_accepts_loss_kwargs_fix"](summed)
    assert not hasattr(summed, "accepts_loss_kwargs")


def test_an_lm_head_model_is_a_causal_lm(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.QWenLMHeadModel()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert model.accepts_loss_kwargs is False


def test_a_forward_replaced_after_load_is_checked_again(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.NemotronHForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert model.accepts_loss_kwargs is False

    def consuming_forward(
        input_ids = None,
        labels = None,
        num_items_in_batch = None,
        **kwargs,
    ):
        return None

    model.forward = consuming_forward
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert not hasattr(model, "accepts_loss_kwargs")


def test_an_assignment_after_the_guess_wins(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.NemotronHForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert model.accepts_loss_kwargs is False
    model.accepts_loss_kwargs = True
    ns["apply_accepts_loss_kwargs_fix"](_PeftLike(model))
    assert model.accepts_loss_kwargs is True


_OWN_HELPER_MODEL = """
import torch
from torch import nn


def cross_entropy(logits, labels):
    return torch.nn.functional.cross_entropy(logits, labels, reduction="sum")


class OwnHelperForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1, 2)
        return cross_entropy(logits, labels)
"""


def test_a_bare_name_bound_to_a_model_helper_is_not_the_torch_op(tmp_path):
    ns = _load()
    mods = _models(tmp_path, _OWN_HELPER_MODEL, "own_helper_for_ga_test")
    model = mods.OwnHelperForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert not hasattr(model, "accepts_loss_kwargs")


_LOCAL_HELPER_MODEL = """
import torch
from torch import nn


class LocalHelperForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()

    def summed_loss(self, logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, reduction="sum")

    def forward(self, input_ids=None, labels=None, **kwargs):
        cross_entropy = self.summed_loss
        logits = torch.zeros(1, 2)
        return cross_entropy(logits, labels)
"""


def test_a_bare_name_bound_inside_forward_is_not_the_torch_op(tmp_path):
    ns = _load()
    mods = _models(tmp_path, _LOCAL_HELPER_MODEL, "local_helper_for_ga_test")
    model = mods.LocalHelperForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert not hasattr(model, "accepts_loss_kwargs")


_OWN_F_MODEL = """
import torch
from torch import nn


class _Helpers:
    @staticmethod
    def cross_entropy(logits, labels):
        return torch.nn.functional.cross_entropy(logits, labels, reduction="sum")


F = _Helpers()


class OwnFForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = torch.zeros(1, 2)
        return F.cross_entropy(logits, labels)
"""


def test_a_qualifier_bound_to_a_model_helper_is_not_the_torch_op(tmp_path):
    ns = _load()
    mods = _models(tmp_path, _OWN_F_MODEL, "own_f_for_ga_test")
    model = mods.OwnFForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert not hasattr(model, "accepts_loss_kwargs")


def test_a_guess_on_the_peft_causal_head_is_cleared_when_forward_changes(tmp_path):
    peft = pytest.importorskip("peft")
    ns = _load()
    # base_model_prefix = "model", as in real checkpoints, so .base_model walks past the causal head.
    source = _PRETRAINED_MODELS.replace(
        "    config_class = RemoteMeanLossConfig\n",
        '    config_class = RemoteMeanLossConfig\n    base_model_prefix = "model"\n',
    ).replace(
        "        self.proj = nn.Linear(4, 4)\n",
        "        self.proj = nn.Linear(4, 4)\n        self.model = nn.Linear(4, 4)\n",
    )
    mods = _models(tmp_path, source, "remote_prefixed_for_ga_test")
    inner = mods.RemoteMeanLossForCausalLM(mods.RemoteMeanLossConfig())
    assert inner.base_model is not inner
    ns["apply_accepts_loss_kwargs_fix"](inner)
    assert inner.accepts_loss_kwargs is False
    wrapped = peft.get_peft_model(
        inner, peft.LoraConfig(r = 2, target_modules = ["proj"], task_type = "CAUSAL_LM")
    )

    def consuming_forward(
        input_ids = None,
        labels = None,
        num_items_in_batch = None,
        **kwargs,
    ):
        return None

    wrapped.get_base_model().forward = consuming_forward
    ns["apply_accepts_loss_kwargs_fix"](wrapped)
    assert "accepts_loss_kwargs" not in wrapped.get_base_model().__dict__
