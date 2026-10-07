# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""PEFT's v4 -> v5 MoE config conversion must not take LoRA away from same-named nn.Linear layers.

DeepSeek-V3 style models keep dense `gate_proj` / `up_proj` / `down_proj` Linears (shared experts and the
first_k_dense_replace layers) next to the fused expert parameters. PEFT rewrote every such target into an
expert parameter target, so the dense layers silently got no LoRA and a v4 adapter's weights for them
were dropped on load.
"""

import copy
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")
twc = pytest.importorskip("peft.utils.transformers_weight_conversion")

from torch import nn

if not hasattr(twc, "convert_peft_config_for_transformers") or not hasattr(
    transformers, "DeepseekV3Config"
):
    pytest.skip(reason = "peft without the transformers v5 MoE conversion", allow_module_level = True)
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip(
        reason = "transformers v4 keeps unfused experts, nothing is converted",
        allow_module_level = True,
    )

from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer, ParamWrapper

IMPORT_FIXES = Path(__file__).resolve().parents[1] / "unsloth" / "import_fixes.py"
_PATCHED_ATTRS = (
    "convert_peft_config_for_transformers",
    "_convert_peft_config_moe",
    "build_peft_weight_mapping",
    "_unsloth_moe_target_conversion_patch",
    "_unsloth_weight_converter_compat_patch",
)
_MISSING = object()


@pytest.fixture(autouse = True)
def patched_peft():
    saved = {name: getattr(twc, name, _MISSING) for name in _PATCHED_ATTRS}
    pattern_map = twc._MODEL_TO_CONVERSION_PATTERN
    saved_patterns = dict(pattern_map)
    spec = importlib.util.spec_from_file_location("_unsloth_import_fixes_moe_linear", IMPORT_FIXES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.patch_peft_weight_converter_compatibility()
    yield
    for name, value in saved.items():
        if value is _MISSING:
            if hasattr(twc, name):
                delattr(twc, name)
        else:
            setattr(twc, name, value)
    pattern_map.clear()
    pattern_map.update(saved_patterns)


def _deepseek_v3():
    torch.manual_seed(0)
    config = transformers.DeepseekV3Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 48,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 4,
        q_lora_rank = 16,
        kv_lora_rank = 8,
        qk_nope_head_dim = 4,
        qk_rope_head_dim = 4,
        v_head_dim = 8,
        n_routed_experts = 4,
        n_shared_experts = 1,
        n_group = 1,
        topk_group = 1,
        num_experts_per_tok = 2,
        first_k_dense_replace = 1,
        max_position_embeddings = 64,
        use_cache = False,
        attn_implementation = "eager",
        # Under a CUDA spoof Unsloth would route experts to a GPU kernel.
        experts_implementation = "eager",
    )
    return transformers.DeepseekV3ForCausalLM(config).eval()


def _qwen2_moe():
    torch.manual_seed(0)
    config = transformers.Qwen2MoeConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 48,
        moe_intermediate_size = 16,
        shared_expert_intermediate_size = 24,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        num_experts = 4,
        num_experts_per_tok = 2,
        max_position_embeddings = 64,
        attn_implementation = "eager",
        experts_implementation = "eager",
    )
    return transformers.Qwen2MoeForCausalLM(config).eval()


def _dense_linears(model):
    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.rpartition(".")[-1] in ("gate_proj", "up_proj", "down_proj")
    }


def _lora_linears(model):
    return {
        name.replace("base_model.model.", "", 1)
        for name, module in model.named_modules()
        if isinstance(module, LoraLayer) and not isinstance(module, ParamWrapper)
    }


@pytest.mark.parametrize(
    "target_modules", [["o_proj", "gate_proj", "up_proj", "down_proj"], "all-linear"]
)
def test_dense_and_shared_expert_linears_keep_lora(target_modules):
    model = _deepseek_v3()
    dense = _dense_linears(model)
    assert len(dense) == 6  # layer 0 MLP + layer 1 shared experts
    peft_model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = target_modules))

    lora_linears = _lora_linears(peft_model)
    assert dense <= lora_linears
    assert "model.layers.1.mlp.gate" not in lora_linears
    if isinstance(target_modules, str) and not hasattr(twc, "_resolve_string_target_modules"):
        return  # peft 0.19 cannot resolve a string against the model
    experts = peft_model.base_model.model.model.layers[1].mlp.experts
    assert isinstance(experts, ParamWrapper)


def test_regex_keeps_its_scope():
    model = _deepseek_v3()
    # No literal dot: a dotted string skips the conversion altogether.
    regex = r"model\Wlayers\W1\Wmlp\Wshared_experts\W(gate_proj|up_proj|down_proj)"
    peft_model = get_peft_model(model, LoraConfig(r = 4, lora_alpha = 8, target_modules = regex))
    lora_linears = _lora_linears(peft_model)
    assert {
        f"model.layers.1.mlp.shared_experts.{leaf}"
        for leaf in ("gate_proj", "up_proj", "down_proj")
    } <= lora_linears
    assert not any(name.startswith("model.layers.0.mlp.") for name in lora_linears)


def test_non_linear_namesake_does_not_veto_the_linears():
    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    model = _deepseek_v3()
    mlp0 = model.model.layers[0].mlp
    mlp0.down_proj = Wrapped(mlp0.down_proj)
    peft_model = get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["gate_proj", "up_proj", "down_proj"])
    )
    lora_linears = _lora_linears(peft_model)
    assert "model.layers.1.mlp.shared_experts.down_proj" in lora_linears
    assert "model.layers.0.mlp.down_proj.inner" not in lora_linears
    assert "model.layers.0.mlp.gate_proj" in lora_linears


def test_quantized_linear_that_is_not_nn_linear_is_restored():
    class QuantLinear(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.in_features, self.out_features = inner.in_features, inner.out_features
            self.qweight = nn.Parameter(inner.weight.detach().clone(), requires_grad = False)

    model = _deepseek_v3()
    shared = model.model.layers[1].mlp.shared_experts
    for leaf in ("gate_proj", "up_proj", "down_proj"):
        setattr(shared, leaf, QuantLinear(getattr(shared, leaf)))
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["gate_proj", "up_proj", "down_proj"])
    twc.convert_peft_config_for_transformers(config, model, None)
    assert {"gate_proj", "up_proj", "down_proj"} <= set(config.target_modules)
    assert "gate" not in set(config.target_modules)


def test_eetq_style_linear_without_feature_attributes_is_restored():
    class EetqLinear(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.weight = nn.Parameter(inner.weight.detach().t().clone(), requires_grad = False)

    model = _deepseek_v3()
    mlp0 = model.model.layers[0].mlp
    mlp0.down_proj = EetqLinear(mlp0.down_proj)
    config = LoraConfig(r = 4, lora_alpha = 8, target_modules = ["gate_proj", "up_proj", "down_proj"])
    twc.convert_peft_config_for_transformers(config, model, None)
    assert "down_proj" in set(config.target_modules)


def test_qwen2_moe_converts_like_transformers_5_5():
    # transformers <= 5.5 mapped qwen2_moe onto itself, so PEFT's fused-pair check applies.
    with pytest.raises(ValueError, match = "without also targeting up_proj"):
        get_peft_model(_qwen2_moe(), LoraConfig(r = 4, lora_alpha = 8, target_modules = ["gate_proj"]))


def test_explicit_target_parameters_are_left_alone():
    model = _deepseek_v3()
    config = LoraConfig(
        r = 4,
        lora_alpha = 8,
        target_modules = ["o_proj", "gate_proj", "up_proj", "down_proj"],
        target_parameters = ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    )
    peft_model = get_peft_model(model, config)
    converted = peft_model.peft_config["default"]
    assert set(converted.target_modules) == {"o_proj", "gate_proj", "up_proj", "down_proj"}
    assert set(converted.target_parameters) == {"mlp.experts.gate_up_proj", "mlp.experts.down_proj"}
    assert not converted.rank_pattern


def test_model_without_dense_namesakes_converts_as_before():
    torch.manual_seed(0)
    config = transformers.Qwen3MoeConfig(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 48,
        moe_intermediate_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        num_experts = 4,
        num_experts_per_tok = 2,
        max_position_embeddings = 64,
    )
    model = transformers.Qwen3MoeForCausalLM(config)
    lora = LoraConfig(
        r = 4, lora_alpha = 8, target_modules = ["q_proj", "gate_proj", "up_proj", "down_proj"]
    )
    twc.convert_peft_config_for_transformers(lora, model, None)
    assert set(lora.target_modules) == {"q_proj"}
    assert set(lora.target_parameters) == {"gate_up_proj", "down_proj"}
    assert lora.rank_pattern == {r".*\.gate_up_proj": 8}


@pytest.mark.parametrize("make_model", [_deepseek_v3, _qwen2_moe], ids = ["deepseek_v3", "qwen2_moe"])
def test_v4_adapter_reloads_dense_and_expert_weights(tmp_path, make_model):
    """A transformers v4 adapter (per-expert Linear keys) matches the model merged by hand.

    qwen2_moe also checks the base family is mapped onto itself: without it its experts stay unconverted.
    """
    from safetensors.torch import save_file

    model = make_model()
    reference = copy.deepcopy(model)
    ref_modules = dict(reference.named_modules())
    r, alpha = 4, 8
    scale = alpha / r
    targets = ["o_proj", "gate_proj", "up_proj", "down_proj"]
    generator = torch.Generator().manual_seed(0)

    def rand(*shape):
        return torch.randn(*shape, generator = generator) * 0.05

    state_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rpartition(".")[-1] in targets:
            A, B = rand(r, module.in_features), rand(module.out_features, r)
            state_dict[f"base_model.model.{name}.lora_A.weight"] = A
            state_dict[f"base_model.model.{name}.lora_B.weight"] = B
            ref_modules[name].weight.data += scale * B @ A
        gate_up = getattr(module, "gate_up_proj", None)
        if isinstance(gate_up, torch.Tensor) and gate_up.ndim == 3:
            n_experts, two_i, hidden = gate_up.shape
            inter = two_i // 2
            for e in range(n_experts):
                for leaf, (out_f, in_f) in (
                    ("gate_proj", (inter, hidden)),
                    ("up_proj", (inter, hidden)),
                    ("down_proj", (hidden, inter)),
                ):
                    A, B = rand(r, in_f), rand(out_f, r)
                    state_dict[f"base_model.model.{name}.{e}.{leaf}.lora_A.weight"] = A
                    state_dict[f"base_model.model.{name}.{e}.{leaf}.lora_B.weight"] = B
                    delta = scale * B @ A
                    ref = ref_modules[name]
                    if leaf == "gate_proj":
                        ref.gate_up_proj.data[e, :inter] += delta
                    elif leaf == "up_proj":
                        ref.gate_up_proj.data[e, inter:] += delta
                    else:
                        ref.down_proj.data[e] += delta
    save_file(state_dict, str(tmp_path / "adapter_model.safetensors"))
    LoraConfig(
        r = r, lora_alpha = alpha, target_modules = targets, task_type = "CAUSAL_LM"
    ).save_pretrained(tmp_path)

    x = torch.randint(0, 64, (1, 16), generator = torch.Generator().manual_seed(1))
    with torch.no_grad():
        base_logits = model(input_ids = x).logits
        expected = reference(input_ids = x).logits
        # Suites that spoof CUDA on a CPU runner would send the load to a missing GPU.
        loaded = PeftModel.from_pretrained(model, str(tmp_path), torch_device = "cpu").eval()
        got = loaded(input_ids = x).logits

    assert (expected - base_logits).abs().max() > 1e-2
    assert torch.allclose(got, expected, atol = 1e-5, rtol = 1e-4), (got - expected).abs().max()
