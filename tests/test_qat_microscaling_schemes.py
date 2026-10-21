# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Saving torchao NVFP4 / MX QAT linears (forward fake quantized, no quantizer children), CPU only."""

from __future__ import annotations

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

import unsloth.models._utils as qat_utils
import unsloth.save as save_mod
from unsloth.models._utils import TorchAOConfig, _prepare_model_for_qat


def _tiny_qwen3():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 1,
        head_dim = 32,
        tie_word_embeddings = False,
    )
    return Qwen3ForCausalLM(config).to(torch.bfloat16)


def _mx_linear():
    from torchao.prototype.qat import MXFakeQuantizedLinear

    return MXFakeQuantizedLinear



def _mx_torchao_config():
    from torchao.prototype.mx_formats import MXDynamicActivationMXWeightConfig
    from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

    return TorchAOConfig(
        qat_scheme = "custom-mx",
        base_config_and_filter_fns = [
            (
                MXDynamicActivationMXWeightConfig(kernel_preference = KernelPreference.EMULATED),
                lambda m, fqn: isinstance(m, torch.nn.Linear) and "lm_head" not in fqn,
            )
        ],
    )


def test_strip_restores_plain_linears():
    model = _prepare_model_for_qat(_tiny_qwen3(), _mx_torchao_config())
    q_proj = model.model.layers[0].self_attn.q_proj
    weight = q_proj.weight
    assert isinstance(q_proj, _mx_linear())
    qat_utils._strip_torchao_fake_quantization(model)
    assert not any(isinstance(m, _mx_linear()) for m in model.modules())
    q_proj = model.model.layers[0].self_attn.q_proj
    assert type(q_proj) is torch.nn.Linear
    assert torch.equal(q_proj.weight, weight)


def test_peft_save_merges_plain_linears(monkeypatch, tmp_path):
    from peft import LoraConfig, get_peft_model

    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_torchao_with_given_config", lambda **kw: seen.update(kw)
    )
    model = get_peft_model(
        _tiny_qwen3(), LoraConfig(r = 16, target_modules = ["q_proj", "down_proj"])
    )
    model = _prepare_model_for_qat(model, _mx_torchao_config())
    assert any(isinstance(m, _mx_linear()) for m in model.modules())
    save_mod._unsloth_save_torchao_with_attached_config(
        model = model, save_directory = str(tmp_path), tokenizer = object()
    )
    # The 16-bit merge sees plain modules, as for the quantizer-child schemes.
    assert not any(isinstance(m, _mx_linear()) for m in model.modules())
    assert seen["torchao_config"] is not None
