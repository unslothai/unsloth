# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""save_pretrained_torchao on QAT models (unslothai/unsloth#4980), CPU only."""

from __future__ import annotations

import importlib.util

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

    from unsloth.models._utils import _peft_untied_embedding_keys

    peft_model = get_peft_model(
        _TiedModel(), LoraConfig(r = 16, target_modules = ["q_proj", "down_proj"])
    )
    _prepare_model_for_qat(peft_model, "int8-int4")
    block = peft_model.base_model.model.layers[0]
    assert hasattr(block.q_proj.base_layer, "weight_fake_quantizer")
    assert hasattr(block.q_proj.lora_A["default"], "weight_fake_quantizer")
    assert not hasattr(block.q_proj.lora_B["default"], "weight_fake_quantizer")
    assert _peft_untied_embedding_keys(peft_model) == ("embed_tokens.weight", "lm_head.weight")
    lm_head = peft_model.base_model.model.lm_head
    assert lm_head.weight.data_ptr() != peft_model.base_model.model.embed_tokens.weight.data_ptr()
    assert not lm_head.weight.requires_grad


@pytest.mark.parametrize("sharded", [False, True])
def test_untie_merged_checkpoint(tmp_path, sharded):
    import json

    from safetensors import safe_open
    from safetensors.torch import save_file

    embed = torch.randn(32, 8)
    tensors = {"model.embed_tokens.weight": embed, "model.norm.weight": torch.ones(8)}
    if sharded:
        save_file(tensors, tmp_path / "model-00001-of-00001.safetensors")
        weight_map = {k: "model-00001-of-00001.safetensors" for k in tensors}
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map}), encoding = "utf-8"
        )
    else:
        save_file(tensors, tmp_path / "model.safetensors")
    config = {
        "tie_word_embeddings": True,
        "thinker_config": {"text_config": {"tie_word_embeddings": True}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    save_mod._untie_merged_checkpoint(str(tmp_path), "model.embed_tokens.weight", "lm_head.weight")

    config = json.loads((tmp_path / "config.json").read_text(encoding = "utf-8"))
    assert config["tie_word_embeddings"] is False
    assert config["thinker_config"]["text_config"]["tie_word_embeddings"] is False
    if sharded:
        index = json.loads((tmp_path / "model.safetensors.index.json").read_text(encoding = "utf-8"))
        shard = tmp_path / index["weight_map"]["lm_head.weight"]
    else:
        shard = tmp_path / "model.safetensors"
    with safe_open(str(shard), "pt") as h:
        assert torch.equal(h.get_tensor("lm_head.weight"), embed)
