# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""QAT prepare / convert / export must leave block-diagonal grouped Linears in full precision."""

import copy
import importlib.util
import warnings

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

from unsloth.models import _utils as qat  # noqa: E402
from unsloth.models.grouped_linear_lora import is_grouped_linear  # noqa: E402

G, D, R = 2, 128, 64  # groups, inputs per group, outputs per group


class _GroupedLinear(torch.nn.Linear):
    """Upstream layout: x [..., groups, in_features] -> [..., groups, out_features // groups]."""

    def __init__(self, in_features, out_features, n_groups):
        super().__init__(in_features, out_features, bias = False)
        self.n_groups = n_groups

    def forward(self, x):
        w = self.weight.view(self.n_groups, self.out_features // self.n_groups, -1)
        return torch.einsum("...gd,grd->...gr", x, w)


class _Model(torch.nn.Module):
    def __init__(self, grouped_in = D):
        super().__init__()
        from transformers import PretrainedConfig

        self.config = PretrainedConfig()
        torch.manual_seed(0)
        self.embed_tokens = torch.nn.Embedding(16, 128)
        self.o_a_proj = _GroupedLinear(grouped_in, G * R, n_groups = G)
        self.o_b_proj = torch.nn.Linear(G * R, 128, bias = False)
        self.lm_head = torch.nn.Linear(128, 16, bias = False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, x):
        return self.lm_head(self.o_b_proj(self.o_a_proj(x).flatten(-2)))


def _x(grouped_in = D):
    return torch.randn(3, 5, G, grouped_in, generator = torch.Generator().manual_seed(1))


def _fake_quantized(module):
    return "FakeQuantized" in type(module).__name__ or hasattr(module, "weight_fake_quantizer")


def _prepare(model, scheme):
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = qat._prepare_model_for_qat(model, scheme)
    return model, [str(w.message) for w in caught]


def _train_step(model, x):
    model(x).float().pow(2).mean().backward()


@pytest.mark.parametrize("scheme", ["int4", "int8", "int8-int4", "cactus"])
def test_every_scheme_leaves_the_grouped_linear_alone_and_persists_it(scheme):
    model, x = _Model(), _x()
    expected = model.o_a_proj(x).detach()
    model, messages = _prepare(model, scheme)
    assert type(model.o_a_proj) is _GroupedLinear and not _fake_quantized(model.o_a_proj)
    assert torch.equal(model.o_a_proj(x), expected)
    assert _fake_quantized(model.o_b_proj)
    _train_step(model, x)
    # The stored config (read again by convert and export) carries the exclusion, structurally.
    for _, filter_fn in model._torchao_config.base_config_and_filter_fns:
        assert not filter_fn(model.o_a_proj, "o_a_proj")
        assert not qat._unquantized_filter(filter_fn)(model.o_a_proj, "o_a_proj")
        assert not copy.deepcopy(filter_fn)(copy.deepcopy(model.o_a_proj), "o_a_proj")
    assert any("grouped" in m and "o_a_proj" in m for m in messages)
    # Scans never report the grouped layer for another reason.
    assert not any("not divisible" in m and "o_a_proj" in m for m in messages)


def test_the_warning_names_only_layers_the_exclusion_changes():
    # in_features 100 is not divisible by 128: int4 would not select it anyway, so it is not named.
    model, _ = _prepare(_Model(grouped_in = 100), "int4")
    _, messages = model, _
    assert not any("grouped" in m and "o_a_proj" in m for m in messages)


@pytest.mark.parametrize("module_by_module", [False, True])
def test_convert_and_reload_metadata_keep_the_grouped_linear(module_by_module, monkeypatch):
    model, x = _Model(), _x()
    expected = model.o_a_proj(x).detach()
    model, _ = _prepare(model, "int8")
    _train_step(model, x)
    reload_config = qat._merged_torchao_config_for_peft_qat(model)
    assert "o_a_proj" in reload_config.modules_to_not_convert
    if module_by_module:
        monkeypatch.setattr(qat, "_TORCHAO_CONVERT_DEVICE", ["cpu"])
    for exported in (model, copy.deepcopy(model)):
        qat._convert_torchao_model(exported)
        assert type(exported.o_a_proj.weight) is torch.nn.Parameter
        assert torch.equal(exported.o_a_proj(x), expected)
        assert exported(x).shape == (3, 5, 16)
        assert "o_a_proj" not in exported.config.quantization_config.quant_type.module_fqn_to_config


def test_builders_never_select_a_grouped_linear():
    model = _Model()
    for filter_fn in (
        qat._group_size_linear_filter(model, 32, "int4"),
        qat._block_linear_filter(model, 32, "mxfp8", skip_output_head = True),
    ):
        assert not filter_fn(model.o_a_proj, "o_a_proj")
        assert filter_fn(model.o_b_proj, "o_b_proj")


def test_grouped_lora_on_the_real_grouped_layout():
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from unsloth.models.grouped_linear_lora import register_grouped_linear_lora

    model, x = _Model(), _x()
    config = LoraConfig(r = 8, target_modules = ["o_a_proj", "o_b_proj"], lora_dropout = 0.0)
    assert register_grouped_linear_lora(config, model)
    model = get_peft_model(model, config)
    lora = model.base_model.model.o_a_proj
    assert getattr(lora, "_unsloth_grouped_lora", False)
    model, messages = _prepare(model, "int4")
    lora = model.base_model.model.o_a_proj
    assert is_grouped_linear(lora.base_layer) and not _fake_quantized(lora.base_layer)
    assert lora(x).shape == (3, 5, G, R)
    _train_step(model, x)
    assert lora.lora_A["default"].weight.grad is not None
    assert any("grouped" in m and "o_a_proj" in m for m in messages)
    reload_config = qat._merged_torchao_config_for_peft_qat(model)
    assert "o_a_proj" in reload_config.modules_to_not_convert
