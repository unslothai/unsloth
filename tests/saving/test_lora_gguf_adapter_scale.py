# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
"""llama.cpp adds `alpha / r * B @ A` to the original base, so the adapter the GGUF converter reads must mean that."""

import copy
import json

import pytest
import torch

import unsloth.save as save_mod
from unsloth.models._utils import snapshot_residual_lora_init

peft = pytest.importorskip("peft")
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

R = 4
TARGETS = ("v_proj", "o_proj")

# Stands in for llama.cpp's converter: keeps what it was handed next to the output.
_CONVERTER = """
import shutil, sys
out = sys.argv[sys.argv.index("--outfile") + 1]
shutil.copytree(sys.argv[1], out + ".input", ignore = shutil.ignore_patterns("*.gguf*"))
open(out, "wb").write(b"GGUF")
"""


def _trained_model(tmp_path, **config_kwargs):
    torch.manual_seed(0)
    base_dir = str(tmp_path / "base")
    config = LlamaConfig(
        vocab_size = 128,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 4,
        num_key_value_heads = 4,
    )
    base = LlamaForCausalLM(config)
    base.save_pretrained(base_dir)
    original = copy.deepcopy(base.state_dict())
    base.config._name_or_path = base_dir
    lora = LoraConfig(r = R, lora_alpha = 8, target_modules = list(TARGETS), **config_kwargs)
    lora.base_model_name_or_path = base_dir
    model = get_peft_model(base, lora)
    snapshot_residual_lora_init(model, lora.init_lora_weights)
    model.peft_config["default"].base_model_name_or_path = base_dir
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.add_(0.05 * torch.randn_like(param))
    return model, original


def _export(model, tmp_path, monkeypatch):
    llama_dir = tmp_path / "llama.cpp"
    llama_dir.mkdir()
    (llama_dir / "convert_lora_to_gguf.py").write_text(_CONVERTER, encoding = "utf-8")
    monkeypatch.setattr(save_mod, "LLAMA_CPP_DEFAULT_DIR", str(llama_dir))
    monkeypatch.setattr(save_mod, "install_llama_cpp", lambda **_kwargs: None)
    out = tmp_path / "export"
    gguf = save_mod._unsloth_save_lora_gguf(model, None, str(out), outtype = "f32")
    return out, gguf + ".input"


def _config(folder):
    with open(f"{folder}/adapter_config.json", encoding = "utf-8") as f:
        return json.load(f)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {},
        {"use_rslora": True},
        {"init_lora_weights": "pissa"},
        {"init_lora_weights": "olora"},
        {"init_lora_weights": "pissa", "use_rslora": True},
    ],
    ids = ["default", "rslora", "pissa", "olora", "pissa-rslora"],
)
def test_converter_input_applies_to_the_original_base(tmp_path, monkeypatch, config_kwargs):
    model, original = _trained_model(tmp_path, **config_kwargs)
    trained = copy.deepcopy(model).merge_and_unload().state_dict()
    save_before = model.save_pretrained

    out, converter_input = _export(model, tmp_path, monkeypatch)

    config = _config(converter_input)
    weights = load_file(f"{converter_input}/adapter_model.safetensors")
    assert not config["use_rslora"]
    for target in TARGETS:
        key = f"model.layers.0.self_attn.{target}"
        A = weights[f"base_model.model.{key}.lora_A.weight"]
        B = weights[f"base_model.model.{key}.lora_B.weight"]
        applied = original[f"{key}.weight"] + config["lora_alpha"] / A.shape[0] * (B @ A)
        torch.testing.assert_close(applied, trained[f"{key}.weight"], atol = 1e-5, rtol = 1e-4)
    # The adapter saved for PEFT and the live model stay as trained.
    native = _config(out)
    assert (native["r"], native["lora_alpha"]) == (R, 8)
    assert native["use_rslora"] == config_kwargs.get("use_rslora", False)
    assert model.save_pretrained == save_before


def test_rslora_with_alpha_pattern_is_refused(tmp_path, monkeypatch):
    model, _ = _trained_model(tmp_path, use_rslora = True, alpha_pattern = {"v_proj": 2})
    with pytest.raises(NotImplementedError, match = "single alpha"):
        _export(model, tmp_path, monkeypatch)
