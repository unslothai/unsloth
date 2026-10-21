# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Opt-in torchao float8 training for full finetuning: layer selection, refusals, emulated numerics (CPU)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

from unsloth.models import float8_training as f8t


def _tiny_qwen3(dtype = torch.float32):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 256,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 32,
        tie_word_embeddings = False,
    )
    return Qwen3ForCausalLM(config).to(dtype)


def _float8_linear():
    from torchao.float8.float8_linear import Float8Linear

    return Float8Linear


def _emulated_config(recipe = "rowwise"):
    from torchao.float8 import Float8LinearConfig

    config = Float8LinearConfig.from_recipe_name(recipe)
    object.__setattr__(config, "emulate", True)
    return config


def test_resolve_values():
    assert f8t.resolve_float8_training(None) is None
    assert f8t.resolve_float8_training(False) is None
    assert f8t.resolve_float8_training(True) == "rowwise"
    assert f8t.resolve_float8_training("TensorWise") == "tensorwise"
    with pytest.raises(ValueError):
        f8t.resolve_float8_training("mxfp8")
    with pytest.raises(TypeError):
        f8t.resolve_float8_training(3)


def test_off_is_a_no_op():
    model = _tiny_qwen3()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert f8t.convert_model_to_float8_training(model, None) == []
    assert f8t.convert_model_to_float8_training(model, False) == []
    assert not any(isinstance(m, _float8_linear()) for m in model.modules())
    assert not hasattr(model, "_unsloth_float8_linears")
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_filter_skips_head_router_odd_dims_and_frozen():
    model = _tiny_qwen3()
    model.model.layers[0].mlp.gate = torch.nn.Linear(128, 16, bias = False)  # MoE router name
    model.model.layers[0].odd = torch.nn.Linear(128, 40, bias = False)  # 40 % 16 != 0
    model.model.layers[1].self_attn.o_proj.weight.requires_grad_(False)
    converted = f8t.convert_model_to_float8_training(
        model, _emulated_config(), size_filter = False, check_hardware = False
    )
    assert "lm_head" not in converted
    assert "model.layers.0.mlp.gate" not in converted
    assert "model.layers.0.odd" not in converted
    assert "model.layers.1.self_attn.o_proj" not in converted
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert f"model.layers.0.self_attn.{name}" in converted
    for name in ("gate_proj", "up_proj", "down_proj"):
        assert f"model.layers.0.mlp.{name}" in converted
        assert f"model.layers.1.mlp.{name}" in converted
    assert len(converted) == 13
    assert model._unsloth_float8_linears == converted
    assert type(model.lm_head) is torch.nn.Linear


def test_size_filter_matches_torchao_rowwise_thresholds():
    keep = f8t.float8_training_filter(torch.nn.Module(), "rowwise")
    big = torch.nn.Linear(4096, 14336, bias = False, device = "meta")
    # meta weights are refused; check the size rule directly
    assert f8t._size_ok("rowwise", 14336, 4096)
    assert not f8t._size_ok("rowwise", 2048, 4096)  # N <= 2048
    assert not f8t._size_ok("rowwise", 6144, 1024)  # K <= 1024
    assert not f8t._size_ok("rowwise", 4096, 2048)  # N <= 4096 and K <= 2048
    assert f8t._size_ok("tensorwise", 2048, 1024)
    assert not f8t._size_ok("tensorwise", 1024, 4096)
    assert keep(big, "model.layers.0.mlp.up_proj") is False


def test_tiny_model_size_filter_converts_nothing_and_warns():
    model = _tiny_qwen3()
    with pytest.warns(UserWarning, match = "no eligible Linear"):
        converted = f8t.convert_model_to_float8_training(
            model, _emulated_config(), check_hardware = False
        )
    assert converted == []


def test_state_dict_keys_unchanged():
    model = _tiny_qwen3()
    keys = list(model.state_dict().keys())
    f8t.convert_model_to_float8_training(
        model, _emulated_config(), size_filter = False, check_hardware = False
    )
    assert list(model.state_dict().keys()) == keys


@pytest.mark.parametrize("recipe", ["rowwise", "tensorwise"])
def test_emulated_loss_and_grad_parity_vs_fp32(recipe):
    ids = torch.randint(0, 256, (2, 32), generator = torch.Generator().manual_seed(1))
    ref = _tiny_qwen3()
    out = ref(input_ids = ids, labels = ids)
    out.loss.backward()
    ref_grads = {n: p.grad.clone() for n, p in ref.named_parameters()}

    model = _tiny_qwen3()
    converted = f8t.convert_model_to_float8_training(
        model, _emulated_config(recipe), size_filter = False, check_hardware = False
    )
    calls = []
    for name, m in model.named_modules():
        if isinstance(m, _float8_linear()):
            m.register_forward_hook(lambda *a, _n = name: calls.append(_n))
    out8 = model(input_ids = ids, labels = ids)
    out8.loss.backward()
    assert sorted(set(calls)) == sorted(converted)  # every converted layer ran in fp8
    assert abs(out8.loss.item() - out.loss.item()) / out.loss.item() < 2e-2
    for n, p in model.named_parameters():
        cos = torch.nn.functional.cosine_similarity(
            p.grad.flatten().double(), ref_grads[n].flatten().double(), dim = 0
        )
        assert cos > 0.98, (n, cos.item())


def test_refuses_without_cuda():
    model = _tiny_qwen3()
    with pytest.raises(RuntimeError, match = "compute capability 8.9"):
        f8t.convert_model_to_float8_training(model, True, size_filter = False)
    assert not any(isinstance(m, _float8_linear()) for m in model.modules())


def test_refuses_pre_sm89(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA A10G")
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match = "A10G is 8.6"):
        f8t._check_hardware(torch.device("cuda", 0), "rowwise")


def test_refuses_when_scaled_mm_probe_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA L4")
    monkeypatch.setattr(torch.version, "hip", None)

    def boom(*a, **k):
        raise RuntimeError("rowwise scaling needs sm90")

    monkeypatch.setattr(f8t, "_probe_float8_gemm", boom)
    with pytest.raises(RuntimeError, match = "tensorwise"):
        f8t._check_hardware(torch.device("cuda", 0), "rowwise")


def test_loader_rejects_float8_training_for_lora():
    from unsloth import FastLanguageModel, FastModel

    for loader in (FastLanguageModel, FastModel):
        with pytest.raises(ValueError, match = "full_finetuning = True"):
            loader.from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM", float8_training = True
            )
