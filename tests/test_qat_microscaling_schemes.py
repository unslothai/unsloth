# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""qat_scheme = "nvfp4" shorthand and saving torchao NVFP4 / MX QAT linears, CPU only."""

from __future__ import annotations

import importlib.util
import warnings

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


def _nvfp4_linear():
    from torchao.prototype.qat import NVFP4FakeQuantizedLinear
    return NVFP4FakeQuantizedLinear


def test_nvfp4_scheme_refuses_without_fp4_gemm():
    model = _tiny_qwen3()
    with pytest.raises(RuntimeError, match = "FP4 tensor cores"):
        _prepare_model_for_qat(model, "nvfp4")
    assert not any(isinstance(m, _nvfp4_linear()) for m in model.modules())


def test_nvfp4_scheme_layers_where_fp4_gemm_runs(monkeypatch):
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

    monkeypatch.setattr(qat_utils, "_qat_forward_runs", lambda *a: True)
    model = _tiny_qwen3()
    model.model.layers[0].odd = torch.nn.Linear(128, 40, bias = False)
    with warnings.catch_warnings(record = True):
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "nvfp4")
    ((base_config, _),) = model._torchao_config.base_config_and_filter_fns
    assert isinstance(base_config, NVFP4DynamicActivationNVFP4WeightConfig)
    assert (
        base_config.use_triton_kernel is False
    )  # the Triton quantizer needs the optional MSLK package
    layer = model.model.layers[0]
    for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.mlp.down_proj):
        assert isinstance(proj, _nvfp4_linear())
    assert type(model.lm_head) is torch.nn.Linear  # 4-bit head costs too much accuracy
    assert type(layer.odd) is torch.nn.Linear


def test_nvfp4_plain_module_without_hf_heads(monkeypatch):
    monkeypatch.setattr(qat_utils, "_qat_forward_runs", lambda *a: True)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias = False)).to(torch.bfloat16)
    model = _prepare_model_for_qat(model, "nvfp4")
    assert isinstance(model[0], _nvfp4_linear())


def test_unknown_scheme_still_rejected():
    with pytest.raises(ValueError, match = "Unexpected QAT scheme"):
        _prepare_model_for_qat(_tiny_qwen3(), "mxfp8")


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
    model = get_peft_model(_tiny_qwen3(), LoraConfig(r = 16, target_modules = ["q_proj", "down_proj"]))
    model = _prepare_model_for_qat(model, _mx_torchao_config())
    assert any(isinstance(m, _mx_linear()) for m in model.modules())
    save_mod._unsloth_save_torchao_with_attached_config(
        model = model, save_directory = str(tmp_path), tokenizer = object()
    )
    assert not any(isinstance(m, _mx_linear()) for m in model.modules())
    assert seen["torchao_config"] is not None


def _probe_ok(
    monkeypatch,
    bias_grad = True,
    calls = None,
    bad_widths = (),
):
    def forward_runs(
        base_config,
        device,
        dtype = torch.bfloat16,
        out_features = 128,
    ):
        if calls is not None and out_features == 128:
            calls.append((str(device), dtype))
        return out_features not in bad_widths

    monkeypatch.setattr(qat_utils, "_qat_forward_runs", forward_runs)
    monkeypatch.setattr(qat_utils, "_qat_bias_grad_supported", lambda *a: bias_grad, raising = False)


def test_nvfp4_refuses_float16_layers(monkeypatch):
    # torchao's non-Triton NVFP4 quantizer asserts bfloat16 / float32 at the first forward.
    _probe_ok(monkeypatch)
    model = _tiny_qwen3().to(torch.float16)
    with pytest.raises(RuntimeError, match = "bfloat16 or float32"):
        _prepare_model_for_qat(model, "nvfp4")
    assert not any(isinstance(m, _nvfp4_linear()) for m in model.modules())


def test_nvfp4_skips_peft_modules_to_save_head(monkeypatch):
    from peft import LoraConfig, get_peft_model

    _probe_ok(monkeypatch)
    model = get_peft_model(
        _tiny_qwen3(),
        LoraConfig(r = 16, target_modules = ["q_proj"], modules_to_save = ["lm_head"]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _prepare_model_for_qat(model, "nvfp4")
    head = model.base_model.model.lm_head
    assert not any(isinstance(m, _nvfp4_linear()) for m in head.modules())
    assert isinstance(model.base_model.model.model.layers[0].mlp.down_proj, _nvfp4_linear())


def test_nvfp4_probes_every_layer_placement(monkeypatch):
    calls = []
    _probe_ok(monkeypatch, calls = calls)
    model = _tiny_qwen3()
    model.model.layers[0].mlp.down_proj.to(torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _prepare_model_for_qat(model, "nvfp4")
    assert sorted(calls, key = str) == sorted(
        [("cpu", torch.bfloat16), ("cpu", torch.float32)], key = str
    )


def test_nvfp4_keeps_failing_narrow_widths_in_full_precision(monkeypatch):
    # tiny Qwen3 k_proj / v_proj have out_features 32 (one KV head of 32).
    _probe_ok(monkeypatch, bad_widths = (32,))
    model = _tiny_qwen3()
    attn = model.model.layers[0].self_attn
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "nvfp4")
    assert not isinstance(attn.k_proj, _nvfp4_linear())
    assert not isinstance(attn.v_proj, _nvfp4_linear())
    assert isinstance(attn.q_proj, _nvfp4_linear())
    assert any("out_features [32]" in str(w.message) for w in caught)


def test_nvfp4_converts_narrow_widths_that_run(monkeypatch):
    _probe_ok(monkeypatch)
    model = _tiny_qwen3()
    model = _prepare_model_for_qat(model, "nvfp4")
    assert isinstance(model.model.layers[0].self_attn.k_proj, _nvfp4_linear())


def test_nvfp4_probes_opted_in_adapter_placements(monkeypatch):
    calls = []
    _probe_ok(monkeypatch, calls = calls)
    monkeypatch.setenv("UNSLOTH_QAT_FAKE_QUANTIZE_LORA", "1")
    model = _tiny_qwen3()
    layer = model.model.layers[0].mlp
    layer.lora_A = torch.nn.Linear(256, 16, bias = False).to(torch.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _prepare_model_for_qat(model, "nvfp4")
    assert ("cpu", torch.float32) in calls


@pytest.mark.parametrize("bias_grad", [False, True])
def test_nvfp4_biased_layers_need_bias_grad(monkeypatch, bias_grad):
    _probe_ok(monkeypatch, bias_grad = bias_grad)
    model = _tiny_qwen3()
    layer = model.model.layers[0]
    layer.self_attn.q_proj = torch.nn.Linear(128, 128, bias = True).to(torch.bfloat16)
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        model = _prepare_model_for_qat(model, "nvfp4")
    assert isinstance(layer.self_attn.q_proj, _nvfp4_linear()) is bias_grad
    assert isinstance(layer.mlp.down_proj, _nvfp4_linear())
    assert any("no bias" in str(w.message) for w in caught) is not bias_grad


def test_nvfp4_accepts_rank4_inputs(monkeypatch):
    linear_cls = _nvfp4_linear()

    def fake_2d_forward(self, x):
        assert x.dim() == 2, x.shape  # torchao's addmm dispatch takes 2D operands only
        return x @ self.weight.t()

    _probe_ok(monkeypatch)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128, bias = False)).to(torch.bfloat16)
    model = _prepare_model_for_qat(model, "nvfp4")
    monkeypatch.setattr(linear_cls, "forward", fake_2d_forward)
    for shape in [(6, 64), (2, 3, 64), (2, 3, 5, 64)]:
        out = model(torch.randn(*shape, dtype = torch.bfloat16))
        assert out.shape == (*shape[:-1], 128)
