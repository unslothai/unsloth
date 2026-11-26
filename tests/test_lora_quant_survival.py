# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Quantized merged export: the LoRA survival estimate warns when the update rounds away. CPU, tiny model."""

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")
peft = pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

_PATH = os.path.join(
    os.path.dirname(__file__), os.pardir, "unsloth", "utils", "lora_quant_survival.py"
)
_SAVE = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "save.py")


def _load():
    # By file path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location("_u_lora_quant_survival", _PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_lora(delta_scale):
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        vocab_size = 128,
        max_position_embeddings = 64,
    )
    model = transformers.LlamaForCausalLM(config).to(torch.bfloat16)
    lora = peft.LoraConfig(r = 8, lora_alpha = 8, target_modules = ["q_proj", "v_proj", "down_proj"])
    model = peft.get_peft_model(model, lora)
    for module in model.modules():
        if hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {}):
            W = module.base_layer.weight.float()
            r = module.lora_A["default"].weight.shape[0]
            target = W.abs().mean().item() * delta_scale
            with torch.no_grad():
                module.lora_A["default"].weight.normal_()
                module.lora_B["default"].weight.normal_().mul_(target / r**0.5)
    return model


@pytest.mark.parametrize("scheme", ["FP8_DYNAMIC", "NVFP4", "MXFP4", "FP8_BLOCK", "torchao_fp8"])
def test_small_update_warns_large_update_does_not(scheme):
    M = _load()
    small, large = _tiny_lora(0.005), _tiny_lora(2.0)
    before = {k: v.clone() for k, v in small.state_dict().items()}
    logs, warns = [], []
    low = M.report_lora_quant_survival(small, scheme, log = logs.append, warn = warns.append)
    assert low is not None and low < M.SURVIVAL_WARN_BELOW, low
    assert len(warns) == 1 and "survives" in warns[0] and len(logs) == 1
    after = small.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    logs, warns = [], []
    high = M.report_lora_quant_survival(large, scheme, log = logs.append, warn = warns.append)
    assert high is not None and high > 0.8, high
    assert warns == [] and len(logs) == 1


def test_unmodelled_scheme_and_plain_model_report_nothing():
    M = _load()
    logs = []
    assert M.report_lora_quant_survival(_tiny_lora(1.0), "W4A16", log = logs.append) is None
    plain = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            hidden_size = 32,
            intermediate_size = 64,
            num_hidden_layers = 1,
            num_attention_heads = 2,
            vocab_size = 64,
        )
    )
    assert M.report_lora_quant_survival(plain, "NVFP4", log = logs.append) is None
    assert logs == []


def test_reference_fp8_grid_matches_torchao():
    ao = pytest.importorskip("torchao.quantization")
    M = _load()
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 64, bias = False).to(torch.bfloat16)
    w = lin.weight.detach().float().clone()
    try:
        ao.quantize_(lin, ao.Float8WeightOnlyConfig())
        scale = lin.weight.scale.float()
        deq = lin.weight.dequantize().float()
    except Exception as error:
        pytest.skip(f"torchao float8 weight-only unavailable on this host: {error!r}")
    assert scale.shape == (64, 1)
    torch.testing.assert_close((M._fp8(w / scale) * scale).bfloat16().float(), deq, rtol = 0, atol = 0)
    # The torchao_fp8 reference reproduces the export bit for bit, scale rounding included.
    ref = M._quantize(w, M._SCHEME_FORMATS["torchao_fp8"], w.abs().amax())
    torch.testing.assert_close(ref.bfloat16().float(), deq, rtol = 0, atol = 0)


def test_both_quantized_exports_report_before_merging():
    src = open(_SAVE, encoding = "utf-8").read()
    for marker in (
        'print(f"Unsloth: Merging to 16bit before {scheme} quantization...")',
        'print(f"Unsloth: Merging to 16bit before torchao {kind} quantization...")',
    ):
        head = src[: src.index(marker)]
        assert "_report_lora_quant_survival(model, " in head[-400:], marker


def test_compressed_export_keeps_its_decorator():
    import ast

    tree = ast.parse(open(_SAVE, encoding = "utf-8").read())
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert fns["_unsloth_save_compressed_tensors"].decorator_list
    assert not fns["_report_lora_quant_survival"].decorator_list


def _low_rank_lora(shrink):
    """One LoRA layer whose update is exactly -shrink * W (W has the LoRA's rank)."""
    torch.manual_seed(0)
    U, V = torch.randn(64, 4), torch.randn(4, 128)
    net = torch.nn.Sequential(torch.nn.Linear(128, 64, bias = False))
    with torch.no_grad():
        net[0].weight.copy_(U @ V * 0.05)
    net = peft.get_peft_model(
        net.to(torch.bfloat16), peft.LoraConfig(r = 4, lora_alpha = 4, target_modules = ["0"])
    )
    layer = next(
        m for m in net.modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})
    )
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(V)
        layer.lora_B["default"].weight.copy_(-shrink * 0.05 * U)
    return net


@pytest.mark.parametrize("scheme", ["NVFP4", "FP8"])
def test_per_tensor_scale_follows_a_shrinking_merge(scheme):
    # Scales follow the merged tensor; on the unmerged grid this read 0.48 / 0.22 and warned.
    M = _load()
    ((_, cosine, _),) = M.lora_quant_survival(_low_rank_lora(0.02), scheme)
    assert cosine > 0.6, cosine


def test_layers_the_export_leaves_16bit_are_not_sampled():
    M = _load()
    model = _tiny_lora(0.005)
    seen = []
    ratios = M.lora_quant_survival(
        model, "NVFP4", skip = lambda name, module: seen.append(name) or "v_proj" in name
    )
    assert seen and all(not n.startswith("base_model.") for n in seen)
    assert ratios and all("v_proj" not in n for n, _, _ in ratios)
    assert M.lora_quant_survival(model, "NVFP4", skip = lambda name, module: True) is None


def test_residual_initialised_adapters_are_not_sampled():
    # PiSSA / OLoRA: B @ A includes the initial factors, so it is not the trained update.
    M = _load()
    model = _tiny_lora(2.0)
    layers = [
        m for m in model.modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})
    ]
    for module in layers[1:]:
        module._unsloth_initial_lora = {"default": (None, None, 1.0)}
    ratios = M.lora_quant_survival(model, "NVFP4")
    assert ratios is not None and len(ratios) == 1


def test_float16_merge_is_measured_after_the_bfloat16_reload():
    # 1.0615234375 + 0.001953125 crosses an fp8 boundary in float16, but both reload as bf16 1.0625.
    M = _load()
    net = torch.nn.Sequential(torch.nn.Linear(1, 1, bias = False))
    with torch.no_grad():
        net[0].weight.fill_(1.0615234375)
    net = peft.get_peft_model(
        net.to(torch.float16), peft.LoraConfig(r = 1, lora_alpha = 1, target_modules = ["0"])
    )
    layer = next(
        m for m in net.modules() if hasattr(m, "lora_A") and "default" in getattr(m, "lora_A", {})
    )
    with torch.no_grad():
        layer.lora_A["default"].weight.fill_(1.0)
        layer.lora_B["default"].weight.fill_(0.001953125)
    ((_, cosine, norm_ratio),) = M.lora_quant_survival(net, "FP8")
    assert cosine == 0.0 and norm_ratio == 0.0


def test_fp4_and_mx_scales_match_compressed_tensors():
    M = _load()
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 0.7421875, 0.26, 7.0, -0.75, -2.5])
    want = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 0.5, 0.5, 6.0, -1.0, -2.0])
    torch.testing.assert_close(M._fp4(x), want, rtol = 0, atol = 0)
    # Block max 7 rounds up to 8 before the exponent is taken: MXFP4 scale 2, not 1.
    amax = torch.tensor([7.0, 6.0, 4.0, 1.0])
    torch.testing.assert_close(M._pow2_scale(amax, 2), torch.tensor([2.0, 1.0, 1.0, 0.25]))
    ct = pytest.importorskip("compressed_tensors.quantization.utils.mxfp_utils")
    amax = torch.rand(4096) * 100
    for bits, emax in ((4, 2), (8, 8)):
        ref = torch.exp2(ct.generate_mx_scales(amax, num_bits = bits) - 127)
        torch.testing.assert_close(M._pow2_scale(amax, emax), ref, rtol = 0, atol = 0)
