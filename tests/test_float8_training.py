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
    assert f8t.resolve_float8_training(True) == "tensorwise"
    assert f8t.resolve_float8_training("TensorWise") == "tensorwise"
    with pytest.raises(ValueError, match = "slower than bfloat16"):
        f8t.resolve_float8_training("rowwise")
    with pytest.raises(ValueError):
        f8t.resolve_float8_training("mxfp8")
    config = _emulated_config("rowwise")
    assert f8t.resolve_float8_training(config) is config
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
        f8t._check_hardware(torch.device("cuda", 0), "tensorwise")
    with pytest.raises(RuntimeError, match = "A10G is 8.6"):
        f8t._check_hardware(torch.device("cuda", 0), _emulated_config("tensorwise"))


def test_recipe_scaling_of_configs():
    assert f8t._recipe_scaling("tensorwise") == "tensorwise"
    assert f8t._recipe_scaling(_emulated_config("tensorwise")) == "tensorwise"
    assert f8t._recipe_scaling(_emulated_config("rowwise")) == "rowwise"


@pytest.mark.parametrize(
    "cap, name, hip, allowed",
    [
        ((12, 0), "NVIDIA RTX PRO 6000 Blackwell", None, True),  # measured 15% faster
        ((12, 1), "NVIDIA RTX 5090 class", None, True),
        ((10, 0), "NVIDIA B200", None, False),  # not timed
        ((9, 0), "NVIDIA H100", None, False),  # not timed
        ((9, 4), "AMD Instinct MI300X", "6.4", False),  # ROCm not timed
    ],
)
def test_named_recipe_only_where_measured_faster(monkeypatch, cap, name, hip, allowed):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)
    monkeypatch.setattr(torch.version, "hip", hip)
    probed = []
    monkeypatch.setattr(
        f8t, "_probe_float8_gemm", lambda d, r: probed.append(f8t._recipe_scaling(r))
    )
    if allowed:
        f8t._check_hardware(torch.device("cuda", 0), "tensorwise")
        assert probed == ["tensorwise"]
    else:
        with pytest.raises(RuntimeError, match = "not been measured faster") as e:
            f8t._check_hardware(torch.device("cuda", 0), "tensorwise")
        assert "Float8LinearConfig.from_recipe_name('tensorwise')" in str(e.value)
        assert probed == []
    # An explicit Float8LinearConfig is the opt in for any fp8 GPU; only the GEMM probe gates it.
    probed.clear()
    f8t._check_hardware(torch.device("cuda", 0), _emulated_config("rowwise"))
    assert probed == ["rowwise"]


def test_refuses_tensorwise_on_sm89(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA L4")
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(f8t, "_probe_float8_gemm", lambda *a: None)
    with pytest.raises(RuntimeError, match = "NaN gradients"):
        f8t._check_hardware(torch.device("cuda", 0), "tensorwise")
    f8t._check_hardware(torch.device("cuda", 0), _emulated_config("rowwise"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    f8t._check_hardware(torch.device("cuda", 0), "tensorwise")


def test_refuses_when_scaled_mm_probe_fails(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "NVIDIA RTX PRO 6000 Blackwell")
    monkeypatch.setattr(torch.version, "hip", None)

    def boom(*a, **k):
        raise RuntimeError("scaled_mm not built for sm120")

    monkeypatch.setattr(f8t, "_probe_float8_gemm", boom)
    with pytest.raises(RuntimeError, match = "tensorwise scaling\\) is not supported"):
        f8t._check_hardware(torch.device("cuda", 0), "tensorwise")
    with pytest.raises(RuntimeError, match = "rowwise scaling\\) is not supported"):
        f8t._check_hardware(torch.device("cuda", 0), _emulated_config("rowwise"))


def test_loader_rejects_float8_training_for_lora():
    from unsloth import FastLanguageModel, FastModel
    for loader in (FastLanguageModel, FastModel):
        with pytest.raises(ValueError, match = "full_finetuning = True"):
            loader.from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM", float8_training = True
            )


@pytest.mark.parametrize("qat_scheme", ["int8-int4", "fp8-fp8", "phone-deployment"])
def test_loader_rejects_float8_training_with_qat(monkeypatch, qat_scheme):
    # QAT replaces nn.Linear with FakeQuantizedLinear before float8 conversion, which then matches nothing.
    from unsloth import FastLanguageModel, FastModel

    converted = []
    monkeypatch.setattr(
        "unsloth.models.loader.convert_model_to_float8_training",
        lambda *a, **k: converted.append(1) or [],
    )
    for loader in (FastLanguageModel, FastModel):
        with pytest.raises(ValueError, match = "cannot be combined"):
            loader.from_pretrained(
                "trl-internal-testing/tiny-Qwen3ForCausalLM",
                full_finetuning = True,
                load_in_4bit = False,
                float8_training = True,
                qat_scheme = qat_scheme,
            )
    assert converted == []


def _cuda_like(monkeypatch, cap, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: cap)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: name)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(f8t, "_probe_float8_gemm", lambda *a: None)


def test_explicit_tensorwise_config_refused_on_sm89(monkeypatch):
    # The NaN refusal is about the scaling, so an explicit config gets it too; rowwise configs stay allowed.
    _cuda_like(monkeypatch, (8, 9), "NVIDIA L4")
    with pytest.raises(RuntimeError, match = "NaN gradients"):
        f8t._check_hardware(torch.device("cuda", 0), _emulated_config("tensorwise"))
    f8t._check_hardware(torch.device("cuda", 0), _emulated_config("rowwise"))


@pytest.mark.parametrize("recipe", ["rowwise", "tensorwise"])
def test_converted_layers_pad_the_inner_dim(recipe):
    # Variable-length batches flatten to token counts not divisible by 16; the fp8 backward GEMM needs them padded.
    model = _tiny_qwen3()
    config = _emulated_config(recipe)
    assert not config.pad_inner_dim
    f8t.convert_model_to_float8_training(model, config, size_filter = False)
    layers = [m for m in model.modules() if isinstance(m, _float8_linear())]
    assert layers and all(m.config.pad_inner_dim for m in layers)
    assert not config.pad_inner_dim  # the caller's config is not mutated


def test_fsdp_float8_all_gather_refused_before_conversion():
    # It wraps weights in a tensor subclass that splits shared parameters (doubled AdamW updates) and that
    # safetensors cannot save, so it is refused before any layer is swapped.
    from dataclasses import replace

    model = _tiny_qwen3()
    config = replace(_emulated_config("tensorwise"), enable_fsdp_float8_all_gather = True)
    with pytest.raises(ValueError, match = "enable_fsdp_float8_all_gather"):
        f8t.convert_model_to_float8_training(model, config, size_filter = False)
    assert not any(isinstance(m, _float8_linear()) for m in model.modules())


def test_shared_weights_stay_shared():
    model = _tiny_qwen3()
    model.model.layers[1].mlp.up_proj.weight = model.model.layers[0].mlp.up_proj.weight
    before = len(list(model.parameters()))
    f8t.convert_model_to_float8_training(model, _emulated_config("tensorwise"), size_filter = False)
    assert len(list(model.parameters())) == before
    assert model.model.layers[1].mlp.up_proj.weight is model.model.layers[0].mlp.up_proj.weight


def _probe_calls(monkeypatch, recipe):
    seen = []

    def fake_scaled_mm(a, b, scale_a, scale_b, **kwargs):
        seen.append((a.dtype, b.dtype, tuple(scale_a.shape), tuple(scale_b.shape)))

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    f8t._probe_float8_gemm(torch.device("cpu"), recipe)
    return seen


def _probe_dtypes(monkeypatch, recipe):
    seen = []

    def fake_scaled_mm(a, b, **kwargs):
        seen.append((a.dtype, b.dtype))

    monkeypatch.setattr(torch, "_scaled_mm", fake_scaled_mm)
    f8t._probe_float8_gemm(torch.device("cpu"), recipe)
    return seen


def _fnuz_config():
    from dataclasses import replace

    # What torchao's from_recipe_name builds on MI300, where its e4m3 / e5m2 dtypes are the fnuz variants.
    base = _emulated_config("tensorwise")
    casts = {}
    for name in ("input", "input_for_grad_weight", "weight", "weight_for_grad_input"):
        casts[f"cast_config_{name}"] = replace(
            getattr(base, f"cast_config_{name}"), target_dtype = torch.float8_e4m3fnuz
        )
    for name in ("grad_output", "grad_output_for_grad_weight"):
        casts[f"cast_config_{name}"] = replace(
            getattr(base, f"cast_config_{name}"), target_dtype = torch.float8_e5m2fnuz
        )
    return replace(base, **casts)


def test_probe_uses_the_config_fp8_dtypes(monkeypatch):
    # MI300 (gfx942): torchao configs carry the fnuz types; probing e4m3fn there would refuse a supported layer.
    assert _probe_dtypes(monkeypatch, _fnuz_config()) == [
        (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz),
        (torch.float8_e5m2fnuz, torch.float8_e4m3fnuz),
    ]


def test_rocm_explicit_fnuz_config_reaches_the_probe_with_its_dtypes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 4))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "AMD Instinct MI300X")
    monkeypatch.setattr(torch.version, "hip", "6.4")
    seen = []
    monkeypatch.setattr(
        f8t, "_probe_float8_gemm", lambda d, r: seen.extend(f8t._probe_dtype_pairs(r))
    )
    f8t._check_hardware(torch.device("cuda", 0), _fnuz_config())
    assert (torch.float8_e4m3fnuz, torch.float8_e4m3fnuz) in seen
    assert torch.float8_e4m3fn not in {d for pair in seen for d in pair}


def test_probe_named_recipes_cover_forward_and_backward(monkeypatch):
    assert _probe_dtypes(monkeypatch, "tensorwise") == [
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e4m3fn),
    ]
    assert _probe_dtypes(monkeypatch, "rowwise") == [(torch.float8_e4m3fn, torch.float8_e4m3fn)]


def test_probe_scales_follow_each_cast_config(monkeypatch):
    # torchao's rowwise_with_gw_hp: rowwise forward; grad_input = rowwise grad_output x tensorwise
    # weight_for_grad_input, which torchao runs rowwise by repeating the tensorwise scale; grad_weight in high
    # precision (casts disabled, not probed).
    from torchao.float8 import Float8LinearConfig

    config = Float8LinearConfig.from_recipe_name("rowwise_with_gw_hp")
    e4m3 = torch.float8_e4m3fn
    assert _probe_calls(monkeypatch, config) == [
        (e4m3, e4m3, (16, 1), (1, 16)),
        (e4m3, e4m3, (16, 1), (1, 16)),
    ]
    assert f8t._probe_gemms(config) == [((e4m3, True), (e4m3, True)), ((e4m3, True), (e4m3, False))]
    assert _probe_calls(monkeypatch, "tensorwise") == [
        (e4m3, e4m3, (), ()),
        (torch.float8_e5m2, e4m3, (), ()),
    ]
    assert _probe_calls(monkeypatch, "rowwise") == [(e4m3, e4m3, (16, 1), (1, 16))]


def _counting_align_hook():
    from accelerate.hooks import AlignDevicesHook
    class CountingAlignDevicesHook(AlignDevicesHook):
        calls = []

        def pre_forward(self, module, *args, **kwargs):
            CountingAlignDevicesHook.calls.append(id(module))
            return super().pre_forward(module, *args, **kwargs)

    return CountingAlignDevicesHook


def test_accelerate_dispatch_hooks_survive_conversion():
    pytest.importorskip("accelerate")
    from accelerate.hooks import add_hook_to_module

    hook_cls = _counting_align_hook()
    model = _tiny_qwen3()
    hooks = {}
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            hooks[name] = hook_cls(execution_device = torch.device("cpu"), io_same_device = True)
            add_hook_to_module(m, hooks[name])
    weights = {name: m.weight for name, m in model.named_modules() if name in hooks}

    converted = f8t.convert_model_to_float8_training(
        model, _emulated_config("tensorwise"), size_filter = False, check_hardware = False
    )
    assert converted
    modules = dict(model.named_modules())
    after = {name: m for name, m in modules.items() if getattr(m, "_hf_hook", None) is not None}
    assert set(after) == set(hooks)
    for name in converted:
        assert isinstance(modules[name], _float8_linear())
        assert modules[name]._hf_hook is hooks[name]
        assert modules[name].weight is weights[name]
        assert "init_hook" not in vars(hooks[name])

    hook_cls.calls.clear()
    ids = torch.randint(0, 256, (2, 32), generator = torch.Generator().manual_seed(1))
    out = model(input_ids = ids, labels = ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    assert {id(modules[name]) for name in converted} <= set(hook_cls.calls)


def test_conversion_adds_no_hooks_to_an_undispatched_model():
    model = _tiny_qwen3()
    converted = f8t.convert_model_to_float8_training(
        model, _emulated_config("tensorwise"), size_filter = False, check_hardware = False
    )
    assert converted
    assert not any(hasattr(m, "_hf_hook") for m in model.modules())
