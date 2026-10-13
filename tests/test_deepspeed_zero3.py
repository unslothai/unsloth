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
"""DeepSpeed ZeRO-3 (#225): the load must not pass a device_map transformers refuses, and the
Trainer must make ZeRO-3 gather the weights Unsloth's fast paths read from a parent's forward."""

import sys
import types

import pytest
import torch

transformers = pytest.importorskip("transformers")

from unsloth.models import _utils
from unsloth.models import loader_utils


@pytest.fixture
def zero3(monkeypatch):
    def set_enabled(enabled):
        monkeypatch.setattr(
            transformers.integrations, "is_deepspeed_zero3_enabled", lambda: enabled
        )

    return set_enabled


def test_zero3_drops_the_placement_unsloth_picked(zero3):
    zero3(True)
    for device_map in (loader_utils.DEFAULT_DEVICE_MAP, "sequential", "auto", None, {"": "cuda:1"}):
        assert loader_utils.deepspeed_zero3_device_map(device_map) is None, device_map


def test_zero3_keeps_an_explicit_multi_device_map(zero3):
    zero3(True)
    device_map = {"model.layers.0": 0, "lm_head": 1}
    assert loader_utils.deepspeed_zero3_device_map(device_map) is device_map


def test_without_zero3_the_device_map_is_untouched(zero3):
    zero3(False)
    for device_map in (loader_utils.DEFAULT_DEVICE_MAP, "sequential", {"": "cuda:1"}, None):
        assert loader_utils.deepspeed_zero3_device_map(device_map) is device_map


def _tiny_llama():
    config = transformers.LlamaConfig(
        vocab_size = 32,
        hidden_size = 16,
        intermediate_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        tie_word_embeddings = False,
    )
    return transformers.LlamaForCausalLM(config)


@pytest.fixture
def fake_deepspeed(monkeypatch):
    calls = {"leaf": [], "external": []}

    def set_z3_leaf_modules(model, classes):
        calls["leaf"].append(list(classes))
        for module in model.modules():
            if type(module) in classes:
                module._z3_leaf = True

    def register_external_parameter(module, param):
        calls["external"].append((module, param))

    deepspeed = types.ModuleType("deepspeed")
    deepspeed.utils = types.ModuleType("deepspeed.utils")
    deepspeed.utils.set_z3_leaf_modules = set_z3_leaf_modules
    deepspeed.zero = types.ModuleType("deepspeed.zero")
    deepspeed.zero.register_external_parameter = register_external_parameter
    for name, module in (
        ("deepspeed", deepspeed),
        ("deepspeed.utils", deepspeed.utils),
        ("deepspeed.zero", deepspeed.zero),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def _trainer(model, deepspeed = True):
    return types.SimpleNamespace(model = model, is_deepspeed_enabled = deepspeed)


def test_zero3_gathers_decoder_layers_whole_and_head_weights_with_their_parent(
    zero3, fake_deepspeed
):
    zero3(True)
    model = _tiny_llama()

    n = _utils._prepare_model_for_deepspeed_zero3(_trainer(model))

    layer_class = type(model.model.layers[0])
    assert fake_deepspeed["leaf"] == [[layer_class]]
    assert all(getattr(layer, "_z3_leaf", False) for layer in model.model.layers)
    owners = {id(param): module for module, param in fake_deepspeed["external"]}
    # The fused loss reads lm_head.weight, the fast forward norm.weight, outside their own forward.
    assert owners[id(model.lm_head.weight)] is model
    assert owners[id(model.model.norm.weight)] is model.model
    assert owners[id(model.model.embed_tokens.weight)] is model.model
    # Nothing inside a decoder layer: the leaf gathers those.
    layer_params = {id(p) for p in model.model.layers.parameters()}
    assert not layer_params & set(owners)
    assert n == len(fake_deepspeed["external"]) == 3

    # Idempotent: a second Trainer on the same model registers nothing more.
    assert _utils._prepare_model_for_deepspeed_zero3(_trainer(model)) == 0
    assert len(fake_deepspeed["external"]) == 3


@pytest.mark.parametrize("deepspeed, enabled", [(False, True), (True, False)])
def test_no_zero3_no_change(zero3, fake_deepspeed, deepspeed, enabled):
    zero3(enabled)
    model = _tiny_llama()
    assert _utils._prepare_model_for_deepspeed_zero3(_trainer(model, deepspeed)) == 0
    assert fake_deepspeed == {"leaf": [], "external": []}
    assert not any(getattr(m, "_z3_leaf", False) for m in model.modules())


def test_one_layer_model_still_gets_a_leaf(zero3, fake_deepspeed):
    zero3(True)
    model = _tiny_llama()
    del model.model.layers[1]
    _utils._prepare_model_for_deepspeed_zero3(_trainer(model))
    assert fake_deepspeed["leaf"] == [[type(model.model.layers[0])]]
    owners = {id(param) for _, param in fake_deepspeed["external"]}
    assert not {id(p) for p in model.model.layers.parameters()} & owners


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(4, 4)


class _MoeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.router = torch.nn.Linear(4, 2)
        self.experts = torch.nn.ModuleList([_Expert(), _Expert()])


class _MoeModel(torch.nn.Module):
    def __init__(self, n_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList([_MoeBlock() for _ in range(n_layers)])
        self.head = torch.nn.Linear(4, 4)


@pytest.mark.parametrize("n_layers", [1, 2])
def test_moe_experts_stay_inside_their_layer_leaf(zero3, fake_deepspeed, n_layers):
    """Per-expert leaves would all-gather by routing, which differs across ranks."""
    zero3(True)
    model = _MoeModel(n_layers)
    _utils._prepare_model_for_deepspeed_zero3(_trainer(model))
    assert fake_deepspeed["leaf"] == [[_MoeBlock]]
