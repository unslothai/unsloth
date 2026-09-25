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
"""A forward that takes **kwargs but returns its own CrossEntropyLoss mean must not
count as consuming num_items_in_batch.

HF Trainer reads any **kwargs on forward as "the model normalises by num_items_in_batch"
and then skips the 1/GA scaling in training_step. Remote code such as NemotronH
(Nemotron-Labs-Teacher) keeps **kwargs only for generate, so with GA=4 the logged loss
was 4x the eval loss (4.22 vs 0.84 on the real checkpoint) and the gradients 4x too.

The helpers are pulled out of unsloth/models/_utils.py by AST, so this runs on CPU
without importing unsloth.
"""

import ast
import inspect
import os
import pathlib
import re
import sys
import textwrap
import types

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
    "_OWN_CE_LOSS",
    "apply_accepts_loss_kwargs_fix",
}


def _load():
    src = _UTILS.read_text()
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _NAMES:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in _NAMES for t in node.targets
        ):
            keep.append(node)
    ns = {"re": re, "inspect": inspect, "os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(_UTILS), "exec"), ns)
    return ns


# The model files have to exist on disk for inspect.getsource, like a real remote module.
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


class NoKwargsForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Backbone()

    def forward(self, input_ids=None, labels=None):
        loss_fct = CrossEntropyLoss()
        return loss_fct(input_ids, labels)
'''


def _models(tmp_path):
    p = tmp_path / "remote_models_for_ga_test.py"
    p.write_text(_MODELS)
    sys.path.insert(0, str(tmp_path))
    try:
        import importlib
        mod = importlib.import_module("remote_models_for_ga_test")
        return importlib.reload(mod)
    finally:
        sys.path.remove(str(tmp_path))


class _PeftLike(nn.Module):
    """PeftModelForCausalLM -> LoraModel -> base model, as the Trainer sees it."""
    def __init__(self, inner):
        super().__init__()
        self.base_model = types.SimpleNamespace(model=inner)
        # LoraModel exposes the wrapped model as .model as well
        self.base_model.base_model = None

    def forward(self, *args, **kwargs):
        return None


def test_remote_mean_loss_forward_is_not_a_loss_kwargs_consumer(tmp_path):
    ns = _load()
    mods = _models(tmp_path)
    model = mods.NemotronHForCausalLM()
    ns["apply_accepts_loss_kwargs_fix"](model)
    # HF Trainer reads hasattr(unwrapped_model, "accepts_loss_kwargs") first.
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
    for cls in (mods.NativeForCausalLM, mods.HandRolledForCausalLM, mods.NoKwargsForCausalLM):
        model = cls()
        ns["apply_accepts_loss_kwargs_fix"](model)
        # Untouched: HF falls back to its own signature inspection, as before.
        assert not hasattr(model, "accepts_loss_kwargs"), cls.__name__


def test_explicit_class_attribute_still_wins(tmp_path):
    ns = _load()
    mods = _models(tmp_path)

    class Declared(mods.NemotronHForCausalLM):
        accepts_loss_kwargs = True

    model = Declared()
    ns["apply_accepts_loss_kwargs_fix"](model)
    assert model.accepts_loss_kwargs is True
