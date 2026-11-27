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
final = out[: -len(".tmp")] if out.endswith(".tmp") else out  # the export writes a partial file, then renames it
shutil.copytree(sys.argv[1], final + ".input", ignore = shutil.ignore_patterns("*.gguf*"))
open(out, "wb").write(b"GGUF")
"""


def _trained_model(
    tmp_path,
    snapshot = True,
    set_scale = None,
    **config_kwargs,
):
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
    if snapshot:
        snapshot_residual_lora_init(model, lora.init_lora_weights if snapshot is True else snapshot)
    if snapshot in ("loftq", "lora_ga"):
        # The base these inits leave behind: W - s0 * B0 @ A0 (LoftQ fits B0 @ A0 unscaled, so its s0 is 1).
        with torch.no_grad():
            for module in model.modules():
                for A0, B0, scaling0 in getattr(module, "_unsloth_initial_lora", {}).values():
                    module.get_base_layer().weight.sub_(scaling0 * (B0 @ A0))
    model.peft_config["default"].base_model_name_or_path = base_dir
    if set_scale is not None:
        # Rescaled after init: the live scale no longer matches the configured one or the initial one.
        for module in model.modules():
            if hasattr(module, "set_scale") and "default" in getattr(module, "scaling", {}):
                module.set_scale("default", set_scale)
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
        {"alpha_pattern": {"o_proj": 3}},
        {"use_rslora": True, "rank_pattern": {"v_proj": 2}, "alpha_pattern": {"o_proj": 3}},
        {"init_lora_weights": "pissa", "alpha_pattern": {"o_proj": 3}},
        {"init_lora_weights": False, "snapshot": "loftq"},
        {"init_lora_weights": "pissa", "set_scale": 2.0},
        {"set_scale": 0.5},
        {"use_rslora": True, "set_scale": 0.5},
    ],
    ids = [
        "default",
        "rslora",
        "pissa",
        "olora",
        "pissa-rslora",
        "alpha_pattern",
        "rslora-patterns",
        "pissa-alpha_pattern",
        "loftq-unit-scale",
        "pissa-set_scale",
        "set_scale",
        "rslora-set_scale",
    ],
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
    assert native.get("alpha_pattern", {}) == config_kwargs.get("alpha_pattern", {})
    assert native["use_rslora"] == config_kwargs.get("use_rslora", False)
    assert model.save_pretrained == save_before


def test_plain_lora_is_handed_over_as_saved(tmp_path, monkeypatch):
    model, _ = _trained_model(tmp_path)
    out, converter_input = _export(model, tmp_path, monkeypatch)
    assert _config(converter_input) == _config(out)
    torch.testing.assert_close(
        load_file(f"{converter_input}/adapter_model.safetensors"),
        load_file(f"{out}/adapter_model.safetensors"),
    )


@pytest.mark.parametrize(
    "config_kwargs",
    [
        # PEFT cannot write rsLoRA with patterns relative to the original base.
        {"init_lora_weights": "pissa", "use_rslora": True, "rank_pattern": {"v_proj": 2}},
        # No initial factors recorded: a PiSSA adapter loaded without Unsloth.
        {"init_lora_weights": "pissa", "snapshot": False},
    ],
    ids = ["pissa-rslora-patterns", "pissa-unrecorded"],
)
def test_residual_adapter_without_original_base_form_is_refused(
    tmp_path, monkeypatch, config_kwargs
):
    model, _ = _trained_model(tmp_path, **config_kwargs)
    with pytest.raises(NotImplementedError, match = "rewritten base"):
        _export(model, tmp_path, monkeypatch)


def test_inkling2_adapter_folds_the_scale_itself(tmp_path, monkeypatch):
    # inkling2_lora_gguf_adapter scales lora_B from the live model, so it must get the unscaled adapter.
    import shutil

    model, _ = _trained_model(tmp_path, use_rslora = True)
    monkeypatch.setattr(save_mod, "is_inkling2_config", lambda config: True)
    copy_dir = str(tmp_path / "inkling2")
    monkeypatch.setattr(
        save_mod,
        "inkling2_lora_gguf_adapter",
        lambda model, folder, **kwargs: shutil.copytree(folder, copy_dir),
    )
    out, converter_input = _export(model, tmp_path, monkeypatch)
    assert _config(converter_input)["use_rslora"]
    torch.testing.assert_close(
        load_file(f"{converter_input}/adapter_model.safetensors"),
        load_file(f"{out}/adapter_model.safetensors"),
    )


@pytest.mark.parametrize("init", ["pissa", "olora"])
def test_reloaded_residual_adapter_exports_on_the_original_base(tmp_path, monkeypatch, init):
    # The loader's reload wrapper records the initial factors, so a reloaded adapter exports like a fresh one.
    from peft import PeftConfig, PeftModel
    from unsloth.models._utils import snapshot_residual_lora_load

    model, original = _trained_model(tmp_path, init_lora_weights = init)
    trained = copy.deepcopy(model).merge_and_unload().state_dict()
    adapter = str(tmp_path / "adapter")
    model.save_pretrained(adapter)
    with snapshot_residual_lora_load(PeftConfig.from_pretrained(adapter)):
        reloaded = PeftModel.from_pretrained(
            LlamaForCausalLM.from_pretrained(str(tmp_path / "base")), adapter
        )
    _, converter_input = _export(reloaded, tmp_path, monkeypatch)
    config = _config(converter_input)
    weights = load_file(f"{converter_input}/adapter_model.safetensors")
    for target in TARGETS:
        key = f"model.layers.0.self_attn.{target}"
        A = weights[f"base_model.model.{key}.lora_A.weight"]
        B = weights[f"base_model.model.{key}.lora_B.weight"]
        applied = original[f"{key}.weight"] + config["lora_alpha"] / A.shape[0] * (B @ A)
        torch.testing.assert_close(applied, trained[f"{key}.weight"], atol = 1e-5, rtol = 1e-4)


def test_lora_ga_converted_save_keeps_its_scale(tmp_path, monkeypatch):
    # LoRA-GA saves relative to the original base on its own; a later set_scale is already in those factors.
    model, original = _trained_model(
        tmp_path, init_lora_weights = False, snapshot = "lora_ga", set_scale = 2.0
    )
    trained = copy.deepcopy(model).merge_and_unload().state_dict()
    _, converter_input = _export(model, tmp_path, monkeypatch)
    config = _config(converter_input)
    weights = load_file(f"{converter_input}/adapter_model.safetensors")
    for target in TARGETS:
        key = f"model.layers.0.self_attn.{target}"
        A = weights[f"base_model.model.{key}.lora_A.weight"]
        B = weights[f"base_model.model.{key}.lora_B.weight"]
        applied = original[f"{key}.weight"] + config["lora_alpha"] / A.shape[0] * (B @ A)
        torch.testing.assert_close(applied, trained[f"{key}.weight"], atol = 1e-5, rtol = 1e-4)


@pytest.mark.parametrize("init", ["pissa", "olora"])
def test_inkling2_keeps_the_configured_scale_on_converted_rows(tmp_path, monkeypatch, init):
    # Inkling 2's own fold reads the live scale; rows converted to the original base carry it already.
    model, original = _trained_model(tmp_path, init_lora_weights = init, set_scale = 2.0)
    trained = copy.deepcopy(model).merge_and_unload().state_dict()
    monkeypatch.setattr(save_mod, "is_inkling2_config", lambda config: True)
    _, converter_input = _export(model, tmp_path, monkeypatch)
    config = _config(converter_input)
    weights = load_file(f"{converter_input}/adapter_model.safetensors")
    for target in TARGETS:
        key = f"model.layers.0.self_attn.{target}"
        A = weights[f"base_model.model.{key}.lora_A.weight"]
        B = weights[f"base_model.model.{key}.lora_B.weight"]
        applied = original[f"{key}.weight"] + config["lora_alpha"] / A.shape[0] * (B @ A)
        torch.testing.assert_close(applied, trained[f"{key}.weight"], atol = 1e-5, rtol = 1e-4)
