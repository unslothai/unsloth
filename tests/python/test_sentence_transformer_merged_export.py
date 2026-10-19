# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Real encoder export regressions without importing Unsloth's GPU patch stack."""

import ast
import copy
import json
import inspect
import os
from pathlib import Path
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("sentence_transformers")
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer, models
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizer,
    BitsAndBytesConfig,
    MPNetConfig,
    MPNetModel,
)

SOURCE = Path(__file__).resolve().parents[2] / "unsloth/models/sentence_transformer.py"


@pytest.fixture(autouse = True)
def offline_export(monkeypatch):
    import huggingface_hub.constants

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(threads)


def exports():
    tree = ast.parse(SOURCE.read_text())
    ns = dict(copy = copy, torch = torch, types = types, os = os)
    cls = type(
        "FastSentenceTransformer", (), {"_add_unsloth_branding": staticmethod(lambda *a: None)}
    )
    ns["FastSentenceTransformer"] = cls
    for name in (
        "_normalize_save_method",
        "_copy_sentence_encoder_for_merge",
        "_save_base_config_for_processor_resume",
        "_patch_transformer_module_save_config",
        "_save_pretrained_merged",
    ):
        fn = min(
            (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name),
            key = lambda n: n.lineno,
        )
        fn.decorator_list = []
        exec(compile(ast.Module(body = [fn], type_ignores = []), str(SOURCE), "exec"), ns)
        if name.startswith("_save_base") or name.startswith("_patch_"):
            setattr(cls, name, staticmethod(ns[name]))
    return ns


def make_model(
    tmp_path,
    architecture,
    quantized = False,
):
    if quantized:
        pytest.importorskip("bitsandbytes")
        if not torch.cuda.is_available():
            pytest.skip("real 4bit export test requires CUDA")
    torch.manual_seed(42)
    base = tmp_path / "base"
    config_cls, model_cls = (
        (BertConfig, BertModel) if architecture == "bert" else (MPNetConfig, MPNetModel)
    )
    config = config_cls(
        vocab_size = 16,
        hidden_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        intermediate_size = 64,
        max_position_embeddings = 64,
        hidden_dropout_prob = 0.0,
        attention_probs_dropout_prob = 0.0,
    )
    model_cls(config).save_pretrained(base)
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "good", "bye"]
    if "vocab" in inspect.signature(BertTokenizer).parameters:
        tokenizer = BertTokenizer(vocab = {x: i for i, x in enumerate(tokens)})
    else:
        vocab_file = base / "vocab.txt"
        vocab_file.write_text("\n".join(tokens) + "\n")
        tokenizer = BertTokenizer(vocab_file = str(vocab_file))
    tokenizer.save_pretrained(base)
    kwargs = {"dtype": torch.float16} if quantized else {}
    if quantized:
        kwargs.update(
            quantization_config = BitsAndBytesConfig(
                load_in_4bit = True,
                bnb_4bit_quant_type = "nf4",
                bnb_4bit_compute_dtype = torch.float16,
                bnb_4bit_use_double_quant = True,
            ),
            device_map = {"": 0},
        )
    model_arg = (
        "model_kwargs"
        if "model_kwargs" in inspect.signature(models.Transformer).parameters
        else "model_args"
    )
    transformer = models.Transformer(str(base), **{model_arg: kwargs})
    model = SentenceTransformer(
        modules = [transformer, models.Pooling(32)], device = "cuda" if quantized else "cpu"
    )
    inner = transformer.auto_model
    if quantized:
        inner = prepare_model_for_kbit_training(inner, use_gradient_checkpointing = False)
    inner = get_peft_model(
        inner,
        LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = ["query", "value"] if architecture == "bert" else ["q", "v"],
        ),
    )
    setattr(
        transformer,
        "model"
        if isinstance(getattr(type(transformer), "auto_model", None), property)
        else "auto_model",
        inner,
    )
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std = 0.01)
    ns = exports()
    ns["_patch_transformer_module_save_config"](transformer, transformer.auto_model.config)
    model.save_pretrained_merged = types.MethodType(ns["_save_pretrained_merged"], model)
    model._dtype = torch.float16
    return model


def train_step(model, optimizer):
    model.train()
    features = model.tokenize(["hello world", "good bye"])
    features = {
        k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in features.items()
    }
    output = model(features)["sentence_embedding"]
    loss = output[:, 0].sum()
    loss.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for n, p in model.named_parameters()
        if "lora_" in n
    )
    optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize(
    "architecture,quantized", [("bert", False), ("mpnet", False), ("bert", True)]
)
def test_export_keeps_training_and_writes_dense_model(tmp_path, architecture, quantized):
    model = make_model(tmp_path, architecture, quantized)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 0.001)
    train_step(model, optimizer)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    identities = {n: id(p) for n, p in model.named_parameters()}
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    model.eval()
    expected = model.encode(["hello world", "good bye"], convert_to_tensor = True)
    expected_merged = {}
    from peft.tuners.lora.layer import LoraLayer
    from peft.utils.integrations import dequantize_module_weight

    for name, layer in model[0].auto_model.get_base_model().named_modules():
        if isinstance(layer, LoraLayer):
            weight = dequantize_module_weight(layer.get_base_layer()).float()
            for adapter in layer.active_adapters:
                weight = weight + layer.get_delta_weight(adapter).float()
            expected_merged[name + ".weight"] = weight.half().cpu()
    dest = tmp_path / "merged"
    model.save_pretrained_merged(str(dest))
    assert identities == {n: id(p) for n, p in model.named_parameters()}
    assert trainable == {n for n, p in model.named_parameters() if p.requires_grad}
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol = 0, atol = 0)
    assert model.training is False
    assert not list(dest.rglob("adapter_config.json"))
    config = json.loads((dest / "config.json").read_text())
    assert "quantization_config" not in config
    weights = load_file(str(dest / "model.safetensors"))
    assert all(w.dtype == torch.float16 for w in weights.values() if w.is_floating_point())
    assert not any("lora_" in name or "quant_state" in name or "absmax" in name for name in weights)
    assert not any(w.dtype == torch.uint8 for w in weights.values())
    for name, weight in expected_merged.items():
        torch.testing.assert_close(weights[name], weight, rtol = 0.002, atol = 0.0001)
    loaded = SentenceTransformer(str(dest), device = "cpu")
    actual = loaded.encode(["hello world", "good bye"], convert_to_tensor = True)
    torch.testing.assert_close(actual.float(), expected.cpu().float(), rtol = 0.02, atol = 0.003)
    model[0].auto_model.save_pretrained(tmp_path / "adapter")
    assert len(load_file(str(tmp_path / "adapter" / "adapter_model.safetensors"))) == len(
        [n for n in before if "lora_" in n]
    )
    train_step(model, optimizer)


def test_failed_export_keeps_live_model(tmp_path, monkeypatch):
    model = make_model(tmp_path, "bert")
    before = {n: id(p) for n, p in model.named_parameters()}

    def fail(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(BertModel, "save_pretrained", fail)
    with pytest.raises(OSError, match = "simulated write failure"):
        model.save_pretrained_merged(str(tmp_path / "merged"))
    assert before == {n: id(p) for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 0.001)
    train_step(model, optimizer)


def test_compiled_encoder_is_exported_without_unloading_original(tmp_path):
    model = make_model(tmp_path, "bert")
    transformer = model[0]
    field = (
        "model"
        if isinstance(getattr(type(transformer), "auto_model", None), property)
        else "auto_model"
    )
    original = transformer.auto_model
    setattr(transformer, field, torch.compile(original, backend = "eager"))
    model._dtype = torch.bfloat16
    dest = tmp_path / "compiled-merged"
    model.save_pretrained_merged(str(dest))
    assert transformer.auto_model._orig_mod is original
    assert any("lora_" in n for n, p in original.named_parameters())
    assert all(
        v.dtype == torch.bfloat16
        for v in load_file(str(dest / "model.safetensors")).values()
        if v.is_floating_point()
    )
    SentenceTransformer(str(dest), device = "cpu")


def test_qlora_nondefault_active_adapter_and_modules_to_save(tmp_path):
    model = make_model(tmp_path, "bert", True)
    inner = model[0].auto_model
    inner.add_adapter(
        "second",
        LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = ["query", "value"],
            modules_to_save = ["embeddings.LayerNorm"],
        ),
    )
    inner.set_adapter("second")
    for n, p in inner.named_parameters():
        if ".second." in n and "lora_B" in n:
            torch.nn.init.normal_(p, std = 0.03)
    model.eval()
    expected = model.encode(["hello world"], convert_to_tensor = True)
    model.save_pretrained_merged(str(tmp_path / "merged-second"))
    assert inner.active_adapters == ["second"]
    loaded = SentenceTransformer(str(tmp_path / "merged-second"), device = "cpu")
    actual = loaded.encode(["hello world"], convert_to_tensor = True)
    torch.testing.assert_close(actual.float(), expected.cpu().float(), rtol = 0.02, atol = 0.003)


def test_quantized_model_without_adapters_exports_dense(tmp_path):
    model = make_model(tmp_path, "bert", True)
    transformer = model[0]
    field = (
        "model"
        if isinstance(getattr(type(transformer), "auto_model", None), property)
        else "auto_model"
    )
    setattr(transformer, field, transformer.auto_model.unload())
    model.save_pretrained_merged(str(tmp_path / "merged-base"))
    assert transformer.auto_model.is_loaded_in_4bit
    config = json.loads((tmp_path / "merged-base" / "config.json").read_text())
    assert "quantization_config" not in config
    weights = load_file(str(tmp_path / "merged-base" / "model.safetensors"))
    assert all(w.dtype == torch.float16 for w in weights.values() if w.is_floating_point())
    assert not any("quant_state" in n for n in weights)
