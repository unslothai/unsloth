# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save_pretrained_torchao on QAT models (unslothai/unsloth#4980), CPU only."""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("torchao") is None, reason = "require torchao"
)

import unsloth.save as save_mod
from unsloth.models._utils import (
    _convert_torchao_model,
    _merged_torchao_config_for_peft_qat,
    _prepare_model_for_qat,
)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(256, 256, bias = False)
        self.down_proj = torch.nn.Linear(256, 256, bias = False)
        self.small = torch.nn.Linear(64, 64, bias = False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 256)
        self.layers = torch.nn.ModuleList([_Block()])
        self.config = type("cfg", (), {"model_type": "llama"})()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return None


def _int8_qat_config():
    from torchao.quantization import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis

    from unsloth.models._utils import TorchAOConfig

    return TorchAOConfig(
        qat_scheme = "int8",
        base_config_and_filter_fns = [
            (
                IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0)),
                lambda m, _: isinstance(m, torch.nn.Linear) and m.in_features >= 128,
            )
        ],
    )


def _is_quantized(weight):
    from torchao.utils import TorchAOBaseTensor
    return isinstance(weight, TorchAOBaseTensor)


@pytest.mark.parametrize("manual_convert", [False, True])
def test_convert_quantizes_after_manual_convert(manual_convert):
    # The QAT notebook runs quantize_(model, QATConfig(step = "convert")) before saving.
    from torchao.quantization import quantize_
    from torchao.quantization.qat import QATConfig

    model = _prepare_model_for_qat(_Model(), _int8_qat_config())
    if manual_convert:
        quantize_(model, QATConfig(step = "convert"))
        assert not _is_quantized(model.layers[0].q_proj.weight)
    _convert_torchao_model(model)
    block = model.layers[0]
    assert _is_quantized(block.q_proj.weight)
    assert _is_quantized(block.down_proj.weight)
    assert not _is_quantized(block.small.weight)


def _peft_qat_model(qat_config, **lora_kwargs):
    from peft import LoraConfig, get_peft_model
    peft_model = get_peft_model(
        _Model(),
        LoraConfig(r = 8, target_modules = ["q_proj", "down_proj", "small"], **lora_kwargs),
    )
    return _prepare_model_for_qat(peft_model, qat_config)


def test_peft_qat_config_uses_merged_module_names():
    qat_config = _int8_qat_config()
    config = _merged_torchao_config_for_peft_qat(_peft_qat_model(qat_config))
    assert config.quant_type is qat_config.base_config_and_filter_fns[0][0]
    # Everything the QAT filter left alone, named as in the merged checkpoint.
    assert sorted(config.modules_to_not_convert) == ["embed_tokens", "layers.0.small"]


@pytest.mark.parametrize("modules_to_save", [None, ["embed_tokens"]])
def test_peft_qat_multi_config_uses_merged_module_names(modules_to_save):
    from torchao.quantization import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis

    qat_config = _int8_qat_config()
    embedding_config = IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0))
    qat_config.base_config_and_filter_fns.append(
        (embedding_config, lambda m, _: isinstance(m, torch.nn.Embedding))
    )
    peft_model = _peft_qat_model(qat_config, modules_to_save = modules_to_save)
    config = _merged_torchao_config_for_peft_qat(peft_model)
    names = {
        k.removesuffix(".weight"): v for k, v in config.quant_type.module_fqn_to_config.items()
    }
    assert set(names) == {"layers.0.q_proj", "layers.0.down_proj", "embed_tokens"}
    assert names["embed_tokens"] is embedding_config
    assert config.include_input_output_embeddings


def test_peft_qat_default_config_keys_every_linear_weight():
    # int8-int4 shape: an Embedding entry plus a default for Linears.
    from torchao.quantization import IntxWeightOnlyConfig
    from torchao.quantization.granularity import PerAxis

    from unsloth.models._utils import TorchAOConfig

    embedding_config = IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0))
    linear_config = IntxWeightOnlyConfig(weight_dtype = torch.int8, granularity = PerAxis(0))
    qat_config = TorchAOConfig(
        qat_scheme = "int8-int4",
        base_config_and_filter_fns = [
            (embedding_config, lambda m, _: isinstance(m, torch.nn.Embedding)),
            (linear_config, None),
        ],
    )
    names = _merged_torchao_config_for_peft_qat(
        _peft_qat_model(qat_config)
    ).quant_type.module_fqn_to_config
    if "_default" in names:  # transformers 4.x / torchao < 0.15: module names only
        assert names == {"embed_tokens": embedding_config, "_default": linear_config}
    else:
        assert names == {
            "embed_tokens.weight": embedding_config,
            "layers.0.q_proj.weight": linear_config,
            "layers.0.down_proj.weight": linear_config,
            "layers.0.small.weight": linear_config,
        }


def test_qat_model_accepts_torchao_config(monkeypatch, tmp_path):
    # The QAT notebook passes torchao_config = Int4WeightOnlyConfig() to a QAT model.
    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_torchao_with_given_config", lambda **kw: seen.update(kw)
    )
    monkeypatch.setattr(
        save_mod,
        "_strip_torchao_fake_quantization",
        lambda model: seen.update(stripped = model),
    )
    model = _Model()
    model._torchao_config = _int8_qat_config()
    config = object()
    save_mod.unsloth_save_pretrained_torchao(
        model, str(tmp_path), tokenizer = object(), torchao_config = config
    )
    assert seen["torchao_config"] is config
    assert seen["stripped"] is model


def test_peft_qat_save_reloads_with_merged_names(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(
        save_mod, "_unsloth_save_torchao_with_given_config", lambda **kw: seen.update(kw)
    )
    qat_config = _int8_qat_config()
    peft_model = _peft_qat_model(qat_config)
    save_mod._unsloth_save_torchao_with_attached_config(
        model = peft_model, save_directory = str(tmp_path), tokenizer = object()
    )
    config = seen["torchao_config"]
    assert config.quant_type is qat_config.base_config_and_filter_fns[0][0]
    assert sorted(config.modules_to_not_convert) == ["embed_tokens", "layers.0.small"]
    # The merge reads 16-bit base layers: nothing fake-quantized or already quantized is left.
    for module in peft_model.modules():
        assert not hasattr(module, "weight_fake_quantizer")
        assert not _is_quantized(getattr(module, "weight", None))


class _TiedModel(_Model):
    def __init__(self):
        super().__init__()
        self.config = type("cfg", (), {"model_type": "llama", "tie_word_embeddings": True})()
        self.lm_head = torch.nn.Linear(256, 32, bias = False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_output_embeddings(self):
        return self.lm_head


def test_int8_int4_lora_rank_below_group_size():
    # lora_B at rank 16 has in_features 16, which PerGroup(32) rejects.
    from peft import LoraConfig, get_peft_model

    from unsloth.models._utils import _peft_untied_lm_head

    peft_model = get_peft_model(
        _TiedModel(), LoraConfig(r = 16, target_modules = ["q_proj", "down_proj"])
    )
    _prepare_model_for_qat(peft_model, "int8-int4")
    block = peft_model.base_model.model.layers[0]
    assert hasattr(block.q_proj.base_layer, "weight_fake_quantizer")
    # Only the frozen base weight is fake-quantized (torchtune QATLoRALinear); adapters merge before quantization.
    assert not hasattr(block.q_proj.lora_A["default"], "weight_fake_quantizer")
    assert not hasattr(block.q_proj.lora_B["default"], "weight_fake_quantizer")
    lm_head = peft_model.base_model.model.lm_head
    assert _peft_untied_lm_head(peft_model) == ("lm_head.weight", lm_head.weight)
    assert lm_head.weight.data_ptr() != peft_model.base_model.model.embed_tokens.weight.data_ptr()
    assert not lm_head.weight.requires_grad


@pytest.mark.parametrize("layout", ["single", "consolidated", "sharded"])
def test_untie_merged_checkpoint(tmp_path, layout):
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {
        "thinker.model.embed_tokens.weight": torch.randn(32, 8),
        "thinker.model.norm.weight": torch.ones(8),
    }
    if layout == "sharded":
        # A local checkpoint can already use the name the head would get.
        save_file(tensors, tmp_path / "model-lm_head.safetensors")
        weight_map = {k: "model-lm_head.safetensors" for k in tensors}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding = "utf-8"
        )
    else:
        name = "model.safetensors" if layout == "single" else "consolidated.safetensors"
        save_file(tensors, tmp_path / name)
        # A reused directory can still hold an adapter save.
        save_file({"x": torch.zeros(1)}, tmp_path / "adapter_model.safetensors")
    config = {
        "thinker_config": {"text_config": {"tie_word_embeddings": True}},
        "talker_config": {"tie_word_embeddings": True},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    # The model's own head, which can differ from the checkpoint's (trained) embedding.
    lm_head = torch.randn(32, 8)

    save_mod._untie_merged_checkpoint(str(tmp_path), "thinker.lm_head.weight", lm_head)

    config = json.loads((tmp_path / "config.json").read_text(encoding = "utf-8"))
    assert config["thinker_config"]["text_config"]["tie_word_embeddings"] is False
    assert config["talker_config"]["tie_word_embeddings"] is True
    if layout == "sharded":
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding = "utf-8"))
        shard = tmp_path / index["weight_map"]["thinker.lm_head.weight"]
        with safe_open(str(tmp_path / "model-lm_head.safetensors"), "pt") as h:
            assert set(h.keys()) == set(tensors)
    else:
        shard = tmp_path / name
    with safe_open(str(shard), "pt") as h:
        assert torch.equal(h.get_tensor("thinker.lm_head.weight"), lm_head)


@pytest.mark.parametrize("peft", [True, False], ids = ["peft", "full"])
@pytest.mark.parametrize("qat_scheme", ["int8-int4", "int8"])
def test_phone_deployment_save_writes_executorch_config(monkeypatch, tmp_path, qat_scheme, peft):
    target = tmp_path / "out"
    # A "-torchao" sibling left by an earlier LoRA export must not attract a full model's config.
    os.makedirs(str(target) + "-torchao")

    class _PeftStub(_Model):
        pass

    def fake_save(model, save_directory, **kw):
        os.makedirs(save_directory + "-torchao" if peft else save_directory, exist_ok = True)

    monkeypatch.setattr(save_mod, "_unsloth_save_torchao_with_attached_config", fake_save)
    monkeypatch.setattr(save_mod, "PeftModel", _PeftStub)
    monkeypatch.setattr(save_mod, "PeftModelForCausalLM", _PeftStub)
    model = _PeftStub() if peft else _Model()
    model._torchao_config = _int8_qat_config()
    model._torchao_config.qat_scheme = qat_scheme
    model._torchao_config._unsloth_builtin_recipe = True
    save_mod.unsloth_save_pretrained_torchao(model, str(target), tokenizer = object())
    path = tmp_path / ("out-torchao" if peft else "out") / "executorch_export_config.yaml"
    stale = tmp_path / ("out" if peft else "out-torchao") / "executorch_export_config.yaml"
    assert not stale.exists()
    if qat_scheme != "int8-int4":
        assert not path.exists()
        return
    import yaml

    config = yaml.safe_load(path.read_text())
    assert config["quantization"] == {
        "qmode": "8da4w",
        "group_size": 32,
        "use_hqq": False,
        "embedding_quantize": "8,0",
    }
    assert config["backend"]["xnnpack"]["enabled"] is True


@pytest.mark.parametrize("case", ["custom_recipe", "gemma3", "hub"])
def test_phone_deployment_config_only_where_it_applies(monkeypatch, tmp_path, capsys, case):
    target = tmp_path / "out"

    def fake_save(model, save_directory, **kw):
        os.makedirs(save_directory, exist_ok = True)

    monkeypatch.setattr(save_mod, "_unsloth_save_torchao_with_attached_config", fake_save)
    model = _Model()
    model._torchao_config = _int8_qat_config()
    model._torchao_config.qat_scheme = "int8-int4"
    if case != "custom_recipe":
        # A recipe Unsloth built from the scheme name; a custom TorchAOConfig keeps only the label.
        model._torchao_config._unsloth_builtin_recipe = True
    if case == "gemma3":
        model.config.model_type = "gemma3_text"
    kwargs = dict(push_to_hub = True, token = "x") if case == "hub" else {}
    save_mod.unsloth_save_pretrained_torchao(model, str(target), tokenizer = object(), **kwargs)
    assert not (target / "executorch_export_config.yaml").exists()
    printed = capsys.readouterr().out
    if case == "custom_recipe":
        assert "ExecuTorch" not in printed
    elif case == "gemma3":
        assert "optimum-executorch" in printed and "group size 32" in printed
    else:
        assert "group_size=32" in printed


class _TokenizerStub:
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok = True)


def test_full_finetune_qat_save_reloads_quantized(tmp_path):
    # Full finetuning converts in place; transformers 5 refused the torchao subclasses without an hf_quantizer.
    from transformers import AutoModelForCausalLM, LlamaConfig

    config = LlamaConfig(
        vocab_size = 64,
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        tie_word_embeddings = False,
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    model = _prepare_model_for_qat(model, _int8_qat_config())
    save_mod._unsloth_save_torchao_with_attached_config(model, str(tmp_path), _TokenizerStub())
    assert getattr(model, "hf_quantizer", None) is None

    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype = torch.bfloat16)
    assert reloaded.config.quantization_config is not None
    ours = dict(model.named_modules())
    n_quantized = 0
    for name, module in reloaded.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        before, after = ours[name].weight, module.weight
        assert type(after) is type(before), name
        if _is_quantized(before):
            n_quantized += 1
            assert torch.equal(after.dequantize(), before.dequantize()), name
        else:
            assert torch.equal(after, before), name
    assert n_quantized == 8
