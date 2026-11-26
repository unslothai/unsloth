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
    # dW entries about delta_scale times the base weight's typical magnitude.
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
    # Read-only: the model the export then merges is untouched.
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
    # Same per-row e4m3 grid; torchao rounds its scale to the weight dtype, so feed it the same scale.
    assert scale.shape == (64, 1)
    torch.testing.assert_close((M._fp8(w / scale) * scale).bfloat16().float(), deq, rtol = 0, atol = 0)
    ref = M._quantize(w, "fp8_channel", w.abs().amax())
    assert ((ref - w).norm() / w.norm() - (deq - w).norm() / w.norm()).abs() < 2e-3


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
