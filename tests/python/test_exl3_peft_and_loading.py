# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""EXL3 adapter validation, source resolution, dtype and conversion regressions."""

import argparse
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

import unsloth.exllama as X
from unsloth import save as S
from unsloth.exllama import loader as EL, patcher as P, quantize as Q
from unsloth.exllama.quant_linear import ExllamaV3Linear
from unsloth.models import loader as ML, vision as V


class Boundary(Exception):
    pass


class _Inner:
    def __init__(self, weight):
        self.out_features, self.in_features = weight.shape
        self.trellis = torch.zeros(1, dtype = torch.int16)
        self.weight = weight.detach().t().contiguous()

    def get_weight_tensor(self):
        return self.weight


class _Toy(torch.nn.Module):
    def __init__(self, exl3 = True):
        super().__init__()
        self.config = {"model_type": "custom"}
        self.name_or_path = "local-base"
        self.proj = torch.nn.Linear(4, 6, bias = False)
        if exl3:
            self.proj = ExllamaV3Linear(4, 6, _Inner(self.proj.weight), compute_dtype = torch.float32)

    def forward(self, x):
        return self.proj(x)


def _config():
    return LlamaConfig(
        hidden_size = 128,
        intermediate_size = 256,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        vocab_size = 256,
        architectures = ["LlamaForCausalLM"],
    )


def _llama(exl3 = True):
    model = LlamaForCausalLM(_config())
    model.max_seq_length = 16
    if exl3:
        attention = model.model.layers[0].self_attn
        attention.q_proj = ExllamaV3Linear(
            128, 128, _Inner(attention.q_proj.weight), compute_dtype = torch.float32
        )
        model._unsloth_exl3_backend = True
    return model


@pytest.mark.parametrize("api", [ML.FastModel, ML.FastLanguageModel])
def test_public_dora_rejected_before_adapter_injection(monkeypatch, api):
    monkeypatch.setenv("UNSLOTH_USE_NEW_MODEL", "1")
    model = _llama()
    original = model.model.layers[0].self_attn.q_proj
    with pytest.raises(ValueError, match = "DoRA is not supported on EXL3"):
        api.get_peft_model(
            model, r = 2, target_modules = ["q_proj"], use_dora = True, use_gradient_checkpointing = False
        )
    assert model.model.layers[0].self_attn.q_proj is original
    assert not any("lora_A" in name for name, _ in model.named_modules())


def test_direct_peft_add_adapter_rejects_dora_but_lora_stays_finite():
    model = get_peft_model(_Toy(), LoraConfig(r = 2, target_modules = ["proj"]))
    x = torch.randn(3, 4)
    before = model(x).detach()
    with pytest.raises(ValueError, match = "DoRA is not supported on EXL3"):
        model.add_adapter("dora", LoraConfig(r = 2, target_modules = ["proj"], use_dora = True))
    assert "dora" not in model.base_model.model.proj.lora_A
    torch.testing.assert_close(model(x), before)
    assert torch.isfinite(model(x)).all()


@pytest.mark.parametrize("exl3", [False, True])
def test_saved_dora_adapter_loading_is_guarded(tmp_path, exl3):
    dense = get_peft_model(_Toy(False), LoraConfig(r = 2, target_modules = ["proj"], use_dora = True))
    dense.save_pretrained(tmp_path)
    base = _Toy(exl3)
    original = base.proj
    if exl3:
        with pytest.raises(ValueError, match = "DoRA is not supported on EXL3"):
            PeftModel.from_pretrained(base, tmp_path, local_files_only = True)
        assert base.proj is original
    else:
        loaded = PeftModel.from_pretrained(base, tmp_path, local_files_only = True)
        assert torch.isfinite(loaded(torch.randn(2, 4))).all()


def test_public_dense_dora_still_initializes_finite(monkeypatch):
    model = _llama(False)
    original = V._get_peft_model
    outputs = []

    def capture(base, config, *args, **kwargs):
        adapted = original(base, config, *args, **kwargs)
        layer = adapted.base_model.model.model.layers[0].self_attn.q_proj
        outputs.append(layer(torch.ones(2, 128, dtype = layer.lora_A["default"].weight.dtype)))
        raise Boundary()

    monkeypatch.setattr(V, "_get_peft_model", capture)
    with pytest.raises(Boundary):
        ML.FastModel.get_peft_model(
            model, r = 2, target_modules = ["q_proj"], use_dora = True, use_gradient_checkpointing = False
        )
    assert torch.isfinite(outputs[0]).all()


@pytest.mark.parametrize("method", ["merge_adapter", "merge_and_unload"])
def test_peft_inplace_merge_rejected_before_any_layer_changes(method):
    model = get_peft_model(_Toy(), LoraConfig(r = 2, target_modules = ["proj"]))
    x = torch.randn(2, 4)
    with torch.no_grad():
        model.base_model.model.proj.lora_B["default"].weight.fill_(0.1)
    before = model(x).detach()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    with pytest.raises(ValueError, match = "EXL3 does not support in-place quantized merges"):
        getattr(model, method)()
    assert not model.base_model.model.proj.merged
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, parameters[name])
    torch.testing.assert_close(model(x), before)


def test_dense_peft_merge_still_works():
    model = get_peft_model(_Toy(False), LoraConfig(r = 2, target_modules = ["proj"]))
    with torch.no_grad():
        model.base_model.model.proj.lora_B["default"].weight.fill_(0.1)
    x = torch.randn(2, 4)
    expected = model(x).detach()
    merged = model.merge_and_unload()
    torch.testing.assert_close(merged(x), expected)


@pytest.mark.parametrize(
    "method", ["merged_4bit", "merged_4bit_forced", "forced_merged_4bit", " MERGED_4BIT_FORCED "]
)
def test_public_quantized_export_rejected_before_writes(tmp_path, method):
    model = get_peft_model(_llama(), LoraConfig(r = 2, target_modules = ["q_proj"]))
    S.patch_saving_functions(model)
    tokenizer = Mock()
    original = model.base_model.model.model.layers[0].self_attn.q_proj
    output = tmp_path / "export"
    with pytest.raises(ValueError, match = "merged_16bit"):
        model.save_pretrained_merged(str(output), tokenizer, method)
    assert model.base_model.model.model.layers[0].self_attn.q_proj is original
    assert not original.merged
    assert not output.exists()
    tokenizer.save_pretrained.assert_not_called()


def test_public_16bit_export_reaches_existing_writer(monkeypatch, tmp_path):
    model = get_peft_model(_llama(), LoraConfig(r = 2, target_modules = ["q_proj"]))
    S.patch_saving_functions(model)
    monkeypatch.setattr(S, "_preflight_merge_disk", Mock(side_effect = Boundary("16bit writer")))
    with pytest.raises(Boundary, match = "16bit writer"):
        model.save_pretrained_merged(str(tmp_path), Mock(), save_method = "merged_16bit")


class _LoadedToy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name_or_path = "checkpoint"
        self.proj = torch.nn.Module()
        self.proj.inner = _Inner(torch.ones(128, 128, dtype = torch.float16))
        self.proj.in_features = self.proj.out_features = 128
        self.embed = torch.nn.Embedding(256, 128)

    def add_model_tags(self, *args, **kwargs):
        pass

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return None


@pytest.fixture
def loader_boundaries(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = _config()
    config.to_json_file(checkpoint / "config.json")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(EL, "is_exllama_available", lambda: True)
    monkeypatch.setattr(EL, "_cuda_available", lambda: True)
    for module in (X, EL, Q):
        monkeypatch.setattr(module, "require_exllama", lambda: None)
    monkeypatch.setattr(P, "patch_transformers_exl3", lambda: True)
    monkeypatch.setattr(EL, "patch_transformers_exl3", lambda: True)
    monkeypatch.setattr(P, "exllama_supported_architectures", lambda: {"LlamaForCausalLM"})
    monkeypatch.setattr(ML, "hf_login", lambda token = None: token)
    monkeypatch.setattr(ML, "is_bfloat16_supported", lambda: True)
    for name in (
        "patch_compiled_autograd",
        "patch_compiling_bitsandbytes",
        "restore_transformers_family",
        "patch_loss_functions",
    ):
        monkeypatch.setattr(ML, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ML, "unsloth_compile_transformers", lambda **kwargs: (kwargs["model_types"], False)
    )
    monkeypatch.setattr(ML, "apply_unsloth_gradient_checkpointing", lambda *args, **kwargs: False)
    monkeypatch.delenv("UNSLOTH_QUANT_BACKEND", raising = False)
    construct = Mock(side_effect = lambda **kwargs: (_LoadedToy(config), types.SimpleNamespace()))
    monkeypatch.setattr(ML.FastBaseModel, "from_pretrained", construct)
    return checkpoint, construct


@pytest.mark.parametrize("entry", ["load_in_exl3", "quantization_config"])
@pytest.mark.parametrize(
    "preferred,explicit,bf16,expected",
    [
        (torch.float32, None, True, torch.float32),
        ("float16", None, True, torch.float16),
        (torch.float32, torch.bfloat16, True, torch.bfloat16),
        ("bfloat16", None, False, torch.float16),
    ],
)
def test_config_dtype_controls_construction_and_real_finalization(
    loader_boundaries, monkeypatch, entry, preferred, explicit, bf16, expected
):
    checkpoint, construct = loader_boundaries
    monkeypatch.setattr(ML, "is_bfloat16_supported", lambda: bf16)

    def prepare(name, **kwargs):
        config = EL.resolve_exl3_config(
            kwargs["load_in_exl3"],
            kwargs["quantization_config"],
            compute_dtype = kwargs["compute_dtype"],
        )
        return EL.Exl3LoadPlan(str(checkpoint), config, False, name)

    monkeypatch.setattr(X, "prepare_exl3_checkpoint", prepare)
    model, _ = ML.FastLanguageModel.from_pretrained(
        str(checkpoint),
        **{entry: EL.Exl3Config(compute_dtype = preferred)},
        dtype = explicit,
        offload_embedding = False,
        use_gradient_checkpointing = False,
    )
    assert construct.call_args.kwargs["dtype"] is expected
    assert model.embed.weight.dtype is expected
    assert model.proj.quant_state.dtype is expected


@pytest.mark.parametrize("explicit", [False, True])
def test_bad_tiles_use_native_fallback_only_when_implicit(loader_boundaries, monkeypatch, explicit):
    checkpoint, construct = loader_boundaries
    config = json.loads((checkpoint / "config.json").read_text())
    config.update(hidden_size = 576, intermediate_size = 1536)
    (checkpoint / "config.json").write_text(json.dumps(config))
    converter = Mock(
        side_effect = AssertionError("incompatible dimensions must not reach conversion")
    )
    monkeypatch.setattr(EL, "quantize_to_exl3", converter)
    if explicit:
        with pytest.raises(ValueError, match = "hidden_size=576"):
            ML.FastLanguageModel.from_pretrained(str(checkpoint), load_in_exl3 = True)
        construct.assert_not_called()
    else:
        ML.FastLanguageModel.from_pretrained(
            str(checkpoint), offload_embedding = False, use_gradient_checkpointing = False
        )
        assert construct.call_count == 1
    converter.assert_not_called()


def test_implicit_conversion_errors_are_not_silently_downgraded(loader_boundaries, monkeypatch):
    checkpoint, _ = loader_boundaries
    monkeypatch.setattr(EL, "quantize_to_exl3", Mock(side_effect = RuntimeError("converter failed")))
    with pytest.raises(RuntimeError, match = "converter failed"):
        ML.FastLanguageModel.from_pretrained(str(checkpoint))


@pytest.mark.parametrize("base_provider", ["huggingface", "modelscope"])
def test_modelscope_adapter_uses_complete_local_snapshot_for_all_reads(
    loader_boundaries, monkeypatch, tmp_path, base_provider
):
    checkpoint, construct = loader_boundaries
    adapter = tmp_path / "adapter"
    dense = get_peft_model(
        _llama(False), LoraConfig(r = 2, target_modules = ["q_proj"], task_type = "CAUSAL_LM")
    )
    dense.save_pretrained(adapter)
    (adapter / EL.EXL3_ADAPTER_STAMP).write_text(
        json.dumps(
            {
                "source_model": "base/source",
                "source_provider": base_provider,
                "exl3_config": {"bits": 3},
            }
        )
    )
    provider = types.ModuleType("modelscope")
    provider.snapshot_download = Mock(return_value = str(adapter))
    monkeypatch.setitem(sys.modules, "modelscope", provider)
    monkeypatch.setattr(ML, "USE_MODELSCOPE", True)
    prepare = Mock(
        side_effect = lambda name, **kwargs: EL.Exl3LoadPlan(
            str(checkpoint), EL.Exl3Config(bits = 3), False, name, source_provider = base_provider
        )
    )
    monkeypatch.setattr(X, "prepare_exl3_checkpoint", prepare)
    read_config = Mock(wraps = ML.PeftConfig.from_pretrained)
    monkeypatch.setattr(ML.PeftConfig, "from_pretrained", read_config)
    tokenizer_source = Mock(wraps = ML._resolve_checkpoint_tokenizer_name)
    monkeypatch.setattr(ML, "_resolve_checkpoint_tokenizer_name", tokenizer_source)
    load_weights = Mock(side_effect = Boundary("adapter weights"))
    monkeypatch.setattr(ML.PeftModel, "from_pretrained", load_weights)
    with pytest.raises(Boundary, match = "adapter weights"):
        ML.FastLanguageModel.from_pretrained(
            "ms/adapter",
            local_files_only = True,
            offload_embedding = False,
            use_gradient_checkpointing = False,
        )
    assert any(call.args[0] == str(adapter) for call in read_config.call_args_list)
    assert all(call.args[0] != "ms/adapter" for call in read_config.call_args_list)
    assert tokenizer_source.call_args.args[0] == str(adapter)
    assert load_weights.call_args.args[1] == str(adapter)
    assert prepare.call_args.args[0] == "base/source"
    assert prepare.call_args.kwargs["use_modelscope"] is (base_provider == "modelscope")
    assert any(
        "allow_patterns" not in call.kwargs for call in provider.snapshot_download.call_args_list
    )


def test_overwrite_removes_stale_work_even_without_output(tmp_path, monkeypatch):
    source, work, output = tmp_path / "source-b", tmp_path / "work", tmp_path / "new-output"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"hidden_size": 128, "intermediate_size": 256}))
    (work / "ckpt").mkdir(parents = True)
    (work / "ckpt" / "job.json").write_text(json.dumps({"next_module_idx": 5}))
    (work / "qtensors").mkdir()
    old_tensor = work / "qtensors" / "old-model.safetensors"
    old_tensor.write_bytes(b"old tensor")
    cm = types.ModuleType("exllamav3.conversion.convert_model")
    cm.parser = argparse.ArgumentParser()
    captured = []

    def prepare(args):
        captured.append(args)
        assert not old_tensor.exists()
        assert not (work / "ckpt" / "job.json").exists()
        assert args.resume is False
        raise Boundary("fresh conversion")

    cm.prepare = prepare
    cm.main = Mock()
    package, conversion = types.ModuleType("exllamav3"), types.ModuleType("exllamav3.conversion")
    package.conversion, conversion.convert_model = conversion, cm
    for name, module in {
        "exllamav3": package,
        "exllamav3.conversion": conversion,
        "exllamav3.conversion.convert_model": cm,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(Q, "require_exllama", lambda: None)
    monkeypatch.setattr(P, "patch_transformers_exl3", lambda: True)
    monkeypatch.setattr(Q, "_maybe_enable_uncalibrated", lambda cfg: None)
    with pytest.raises(Boundary, match = "fresh conversion"):
        Q.quantize_to_exl3(
            str(source),
            Q.Exl3Config(calibrate = False),
            out_dir = str(output),
            work_dir = str(work),
            overwrite = True,
        )
    assert captured[0].in_dir == str(source)
    cm.main.assert_not_called()
