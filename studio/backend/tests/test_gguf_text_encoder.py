# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import logging
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.t5.configuration_t5 import T5Config


_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

if "structlog" not in sys.modules:
    sys.modules["structlog"] = types.SimpleNamespace(
        get_logger = lambda *a, **k: logging.getLogger("structlog.stub"),
    )
if "loggers" not in sys.modules:
    sys.modules["loggers"] = types.SimpleNamespace(
        get_logger = lambda *a, **k: logging.getLogger("loggers.stub"),
    )


class _TinyLanguageModel(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.config = config
        self.rotary_emb = None

    def forward(self, *args, **kwargs):
        return {"args": args, "kwargs": kwargs}


def _fake_flux2_config() -> PretrainedConfig:
    cfg = PretrainedConfig()
    text_cfg = PretrainedConfig()
    text_cfg.num_attention_heads = 2
    text_cfg.num_key_value_heads = 1
    cfg.text_config = text_cfg
    return cfg


def test_strip_gguf_quant_suffix_matches_comfyui_gguf_names():
    import core.inference.gguf_text_encoder as g

    assert (
        g.strip_gguf_quant_suffix("Qwen2.5-VL-7B-Instruct-Q4_K_M")
        == "Qwen2.5-VL-7B-Instruct"
    )
    assert (
        g.strip_gguf_quant_suffix("Mistral-Small-UD-Q4_K_XL")
        == "Mistral-Small"
    )
    assert g.strip_gguf_quant_suffix("mmproj-BF16") == "mmproj-BF16"


def test_resolve_text_encoder_mmproj_prefers_matching_sibling(tmp_path):
    import core.inference.gguf_text_encoder as g

    text = tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    matching = tmp_path / "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf"
    generic = tmp_path / "mmproj-BF16.gguf"
    text.touch()
    matching.touch()
    generic.touch()

    assert g.resolve_text_encoder_mmproj_gguf(text) == matching


def test_resolve_text_encoder_mmproj_uses_single_generic_sibling(tmp_path):
    import core.inference.gguf_text_encoder as g

    text = tmp_path / "custom-qwen-Q4_K_M.gguf"
    generic = tmp_path / "mmproj-BF16.gguf"
    text.touch()
    generic.touch()

    assert g.resolve_text_encoder_mmproj_gguf(text) == generic


def test_inspect_text_encoder_gguf_reports_qwen2vl_mmproj(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    text = tmp_path / "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    mmproj = tmp_path / "Qwen2.5-VL-7B-Instruct-mmproj-BF16.gguf"
    text.touch()
    mmproj.touch()
    reader = object()

    monkeypatch.setattr(g, "_open_gguf_reader", lambda path: reader)
    monkeypatch.setattr(
        g,
        "_read_gguf_scalar_field",
        lambda r, name, field_type: {
            "general.architecture": "qwen2vl",
            "general.type": "model",
        }.get(name),
    )

    info = g.inspect_text_encoder_gguf(text)

    assert info.path == text
    assert info.architecture == "qwen2vl"
    assert info.model_type == "model"
    assert info.supported_by_lazy_loader is True
    assert info.requires_mmproj is True
    assert info.mmproj_path == mmproj


def test_gguf_tensor_logical_shape_prefers_comfy_orig_shape_metadata():
    import numpy as np

    import core.inference.gguf_text_encoder as g

    class _FakeField:
        def __init__(self, values):
            self.parts = [np.array([value], dtype = np.int32) for value in values]
            self.data = list(range(len(values)))

    class _FakeReader:
        def get_field(self, name):
            if name == "comfy.gguf.orig_shape.proj.weight":
                return _FakeField([2, 3, 4])
            return None

    tensor = SimpleNamespace(name = "proj.weight", shape = (12, 2))

    assert g._gguf_tensor_logical_shape(tensor, reader = _FakeReader()) == (2, 3, 4)
    assert g._gguf_tensor_logical_shape(tensor) == (2, 12)


def test_materialize_qwen2vl_rotary_buffers_replaces_meta_modules(monkeypatch):
    import core.inference.gguf_text_encoder as g

    class _MetaRotary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.empty(3, device = "meta"))

    class _TextRotary(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(config.max_position_embeddings))
            self.register_buffer("original_inv_freq", torch.ones(config.max_position_embeddings))

    class _VisionRotary(nn.Module):
        def __init__(self, dim) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(dim))

    root = nn.Module()
    root.model = nn.Module()
    root.model.language_model = nn.Module()
    root.model.visual = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.Module()
    layer.self_attn.rotary_emb = _MetaRotary()
    root.model.language_model.layers = nn.ModuleList([layer])
    root.model.language_model.rotary_emb = _MetaRotary()
    root.model.visual.rotary_pos_emb = _MetaRotary()
    config = SimpleNamespace(
        text_config = SimpleNamespace(max_position_embeddings = 4),
        vision_config = SimpleNamespace(hidden_size = 16, num_heads = 2),
    )
    monkeypatch.setattr(
        g,
        "_qwen2_5_vl_rotary_classes",
        lambda: (_TextRotary, _VisionRotary),
    )

    g._materialize_qwen2vl_rotary_buffers(root, config)

    assert root.model.language_model.rotary_emb.inv_freq.device == torch.device("cpu")
    assert (
        root.model.language_model.layers[0].self_attn.rotary_emb.inv_freq.device
        == torch.device("cpu")
    )
    assert root.model.visual.rotary_pos_emb.inv_freq.device == torch.device("cpu")
    assert not any(buffer.is_meta for _name, buffer in root.named_buffers())


def test_materialize_qwen3_rotary_buffers_replaces_meta_modules(monkeypatch):
    import core.inference.gguf_text_encoder as g

    class _MetaRotary(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.empty(3, device = "meta"))

    class _Qwen3Rotary(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(config.max_position_embeddings))
            self.register_buffer("original_inv_freq", torch.ones(config.max_position_embeddings))

    root = nn.Module()
    root.model = nn.Module()
    root.model.rotary_emb = _MetaRotary()
    config = SimpleNamespace(max_position_embeddings = 4)
    monkeypatch.setattr(g, "_qwen3_rotary_class", lambda: _Qwen3Rotary)

    g._materialize_qwen3_rotary_buffers(root, config)

    assert root.model.rotary_emb.inv_freq.device == torch.device("cpu")
    assert not any(buffer.is_meta for _name, buffer in root.named_buffers())


def test_read_gguf_scalar_field_decodes_uint8_string_arrays():
    np = pytest.importorskip("numpy")
    import core.inference.gguf_text_encoder as g

    value = np.frombuffer(b"qwen2vl", dtype = np.uint8)
    field = SimpleNamespace(parts = [value], data = [0])
    reader = SimpleNamespace(get_field = lambda name: field)

    assert g._read_gguf_scalar_field(reader, "general.architecture", str) == "qwen2vl"


def test_map_qwen2vl_text_gguf_name_matches_transformers_names():
    import core.inference.gguf_text_encoder as g

    assert (
        g.map_qwen2vl_text_gguf_name("token_embd.weight").hf_name
        == "model.language_model.embed_tokens.weight"
    )
    assert (
        g.map_qwen2vl_text_gguf_name("output_norm.weight").hf_name
        == "model.language_model.norm.weight"
    )
    assert g.map_qwen2vl_text_gguf_name("output.weight").hf_name == "lm_head.weight"
    assert (
        g.map_qwen2vl_text_gguf_name("blk.0.attn_q.weight").hf_name
        == "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert (
        g.map_qwen2vl_text_gguf_name("blk.0.attn_k.bias").hf_name
        == "model.language_model.layers.0.self_attn.k_proj.bias"
    )
    assert (
        g.map_qwen2vl_text_gguf_name("blk.27.ffn_down.weight").hf_name
        == "model.language_model.layers.27.mlp.down_proj.weight"
    )
    assert g.map_qwen2vl_text_gguf_name("v.blk.0.attn_q.weight") is None


def test_map_qwen3_text_gguf_name_matches_transformers_names():
    import core.inference.gguf_text_encoder as g

    assert (
        g.map_qwen3_text_gguf_name("token_embd.weight").hf_name
        == "model.embed_tokens.weight"
    )
    assert g.map_qwen3_text_gguf_name("output_norm.weight").hf_name == "model.norm.weight"
    assert g.map_qwen3_text_gguf_name("output.weight") is None
    assert (
        g.map_qwen3_text_gguf_name("blk.4.attn_q.weight").hf_name
        == "model.layers.4.self_attn.q_proj.weight"
    )
    assert (
        g.map_qwen3_text_gguf_name("blk.4.ffn_gate.weight").hf_name
        == "model.layers.4.mlp.gate_proj.weight"
    )
    assert (
        g.map_qwen3_text_gguf_name("blk.4.attn_q_norm.weight").hf_name
        == "model.layers.4.self_attn.q_norm.weight"
    )
    assert (
        g.map_qwen3_text_gguf_name("blk.4.attn_k_norm.weight").hf_name
        == "model.layers.4.self_attn.k_norm.weight"
    )
    assert (
        g.map_qwen3_text_gguf_name("blk.4.attn_v_norm.weight").hf_name
        == "model.layers.4.self_attn.v_norm.weight"
    )


def test_map_llama_text_gguf_name_marks_qk_reverse_permute():
    import core.inference.gguf_text_encoder as g

    q = g.map_llama_text_gguf_name(
        "blk.1.attn_q.weight",
        num_attention_heads = 8,
        num_key_value_heads = 2,
    )
    k = g.map_llama_text_gguf_name(
        "blk.1.attn_k.weight",
        num_attention_heads = 8,
        num_key_value_heads = 2,
    )

    assert q.hf_name == "model.layers.1.self_attn.q_proj.weight"
    assert q.reverse_permute_heads == 8
    assert k.hf_name == "model.layers.1.self_attn.k_proj.weight"
    assert k.reverse_permute_heads == 2
    assert g.map_llama_text_gguf_name("output.weight") is None


def test_map_qwen3vl_text_gguf_name_matches_text_model_names():
    import core.inference.gguf_text_encoder as g

    assert (
        g.map_qwen3vl_text_gguf_name("token_embd.weight").hf_name
        == "embed_tokens.weight"
    )
    assert g.map_qwen3vl_text_gguf_name("output_norm.weight").hf_name == "norm.weight"
    assert (
        g.map_qwen3vl_text_gguf_name("blk.2.attn_q_norm.weight").hf_name
        == "layers.2.self_attn.q_norm.weight"
    )
    assert (
        g.map_qwen3vl_text_gguf_name("blk.2.attn_v_norm.weight").hf_name
        == "layers.2.self_attn.v_norm.weight"
    )
    assert (
        g.map_qwen3vl_text_gguf_name("blk.2.ffn_down.weight").hf_name
        == "layers.2.mlp.down_proj.weight"
    )


def test_map_gemma3_text_gguf_name_matches_gemma3_norm_names():
    import core.inference.gguf_text_encoder as g

    assert (
        g.map_gemma3_text_gguf_name("token_embd.weight").hf_name
        == "model.embed_tokens.weight"
    )
    assert (
        g.map_gemma3_text_gguf_name("blk.5.ffn_norm.weight").hf_name
        == "model.layers.5.pre_feedforward_layernorm.weight"
    )
    assert (
        g.map_gemma3_text_gguf_name("blk.5.post_ffw_norm.weight").hf_name
        == "model.layers.5.post_feedforward_layernorm.weight"
    )
    assert (
        g.map_gemma3_text_gguf_name("blk.5.post_attention_norm.weight").hf_name
        == "model.layers.5.post_attention_layernorm.weight"
    )
    assert g.map_gemma3_text_gguf_name("output_norm.weight").value_offset == -1.0
    assert g.map_gemma3_text_gguf_name("blk.5.attn_norm.weight").value_offset == -1.0
    assert g.map_gemma3_text_gguf_name("blk.5.attn_q_norm.weight").value_offset == -1.0
    assert g.map_gemma3_text_gguf_name("blk.5.attn_k_norm.weight").value_offset == -1.0
    assert g.map_gemma3_text_gguf_name("blk.5.attn_v_norm.weight").value_offset == 0.0
    assert g.map_gemma3_text_gguf_name("output.weight") is None


def test_map_t5_text_gguf_name_matches_transformers_names():
    import core.inference.gguf_text_encoder as g

    assert g.map_t5_text_gguf_name("token_embd.weight").hf_name == "shared.weight"
    assert (
        g.map_t5_text_gguf_name("enc.output_norm.weight").hf_name
        == "encoder.final_layer_norm.weight"
    )
    assert (
        g.map_t5_text_gguf_name("enc.blk.0.attn_q.weight").hf_name
        == "encoder.block.0.layer.0.SelfAttention.q.weight"
    )
    assert (
        g.map_t5_text_gguf_name("enc.blk.0.attn_rel_b.weight").hf_name
        == "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"
    )
    assert (
        g.map_t5_text_gguf_name("enc.blk.3.ffn_gate.weight").hf_name
        == "encoder.block.3.layer.1.DenseReluDense.wi_0.weight"
    )
    assert (
        g.map_t5_text_gguf_name("enc.blk.3.ffn_up.weight").hf_name
        == "encoder.block.3.layer.1.DenseReluDense.wi_1.weight"
    )
    assert (
        g.map_t5_text_gguf_name("enc.blk.3.ffn_down.weight").hf_name
        == "encoder.block.3.layer.1.DenseReluDense.wo.weight"
    )
    assert g.map_t5_text_gguf_name("blk.0.attn_q.weight") is None


def test_map_qwen2vl_mmproj_gguf_name_marks_grouped_visual_targets():
    import core.inference.gguf_text_encoder as g

    patch = g.map_qwen2vl_mmproj_gguf_name("v.patch_embd.weight")
    patch_1 = g.map_qwen2vl_mmproj_gguf_name("v.patch_embd.weight.1")
    assert patch.hf_name == "model.visual.patch_embed.proj.weight"
    assert patch.stack_index == 0
    assert patch_1.hf_name == "model.visual.patch_embed.proj.weight"
    assert patch_1.stack_index == 1

    q = g.map_qwen2vl_mmproj_gguf_name("v.blk.3.attn_q.weight")
    k_bias = g.map_qwen2vl_mmproj_gguf_name("v.blk.3.attn_k.bias")
    assert q.hf_name == "model.visual.blocks.3.attn.qkv.weight"
    assert q.qkv_part == "q"
    assert k_bias.hf_name == "model.visual.blocks.3.attn.qkv.bias"
    assert k_bias.qkv_part == "k"

    assert (
        g.map_qwen2vl_mmproj_gguf_name("v.blk.3.attn_out.weight").hf_name
        == "model.visual.blocks.3.attn.proj.weight"
    )
    assert (
        g.map_qwen2vl_mmproj_gguf_name("v.blk.3.ffn_gate.bias").hf_name
        == "model.visual.blocks.3.mlp.gate_proj.bias"
    )
    assert (
        g.map_qwen2vl_mmproj_gguf_name("v.post_ln.weight").hf_name
        == "model.visual.merger.ln_q.weight"
    )
    assert (
        g.map_qwen2vl_mmproj_gguf_name("mm.2.weight").hf_name
        == "model.visual.merger.mlp.2.weight"
    )


def test_lazy_flux2_text_encoder_uses_hub_text_encoder_subfolder(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    calls: list[tuple[object, dict]] = []
    replace_calls: list[dict] = []

    def fake_from_pretrained(path_or_repo, **kwargs):
        calls.append((path_or_repo, kwargs))
        return _fake_flux2_config()

    def fake_replace(language_model, *, gguf_path, compute_dtype, resident_device = None):
        replace_calls.append(
            {
                "gguf_path": gguf_path,
                "compute_dtype": compute_dtype,
                "resident_device": resident_device,
            }
        )
        return "reader"

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(g, "MistralModel", _TinyLanguageModel)
    monkeypatch.setattr(g, "MistralRotaryEmbedding", lambda cfg: ("rotary", cfg))
    monkeypatch.setattr(
        g,
        "_replace_mistral_modules_with_lazy_gguf",
        fake_replace,
    )

    encoder = g.LazyFlux2MistralTextEncoder.from_gguf(
        tmp_path / "text.gguf",
        base_repo_or_path = "black-forest-labs/FLUX.2-dev",
        compute_dtype = torch.bfloat16,
        resident_device = "cpu",
        token = "hf_test",
    )

    assert calls == [
        (
            "black-forest-labs/FLUX.2-dev",
            {"subfolder": "text_encoder", "token": "hf_test"},
        )
    ]
    assert isinstance(encoder, Mistral3ForConditionalGeneration)
    assert encoder.dtype is torch.bfloat16
    assert encoder.device == torch.device("cpu")
    assert encoder._gguf_reader == "reader"
    assert encoder.language_model.rotary_emb[0] == "rotary"
    assert replace_calls == [
        {
            "gguf_path": tmp_path / "text.gguf",
            "compute_dtype": torch.bfloat16,
            "resident_device": "cpu",
        }
    ]


def test_lazy_flux2_text_encoder_uses_local_text_encoder_dir(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    base = tmp_path / "FLUX.2-dev"
    (base / "text_encoder").mkdir(parents = True)
    calls: list[tuple[object, dict]] = []

    def fake_from_pretrained(path_or_repo, **kwargs):
        calls.append((path_or_repo, kwargs))
        return _fake_flux2_config()

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(g, "MistralModel", _TinyLanguageModel)
    monkeypatch.setattr(g, "MistralRotaryEmbedding", lambda cfg: ("rotary", cfg))
    monkeypatch.setattr(
        g,
        "_replace_mistral_modules_with_lazy_gguf",
        lambda language_model, *, gguf_path, compute_dtype, resident_device = None: "reader",
    )

    encoder = g.LazyFlux2MistralTextEncoder.from_gguf(
        tmp_path / "text.gguf",
        base_repo_or_path = base,
        compute_dtype = torch.float16,
        token = "hf_local",
    )

    assert calls == [(base / "text_encoder", {"token": "hf_local"})]
    assert isinstance(encoder, Mistral3ForConditionalGeneration)
    assert encoder.dtype is torch.float16


def test_lazy_flux2_text_encoder_device_ignores_resident_quant_buffers():
    import core.inference.gguf_text_encoder as g

    language_model = nn.Module()
    language_model.register_buffer("qweight", torch.zeros(1, dtype = torch.uint8))
    language_model.register_buffer("position_ids", torch.empty(1, device = "meta"))

    encoder = g.LazyFlux2MistralTextEncoder(
        _fake_flux2_config(),
        language_model,
        compute_dtype = torch.bfloat16,
    )

    assert encoder.device == torch.device("meta")


def test_replace_mistral_modules_keeps_quantized_linear_bias_lazy(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeTensor:
        def __init__(self, name, tensor_type, data):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = tuple(reversed(tuple(data.shape)))

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

        @staticmethod
        def GGUFReader(path):
            return reader

    class _SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = nn.Linear(2, 2, bias = True)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _SelfAttention()

    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                num_attention_heads = 2,
                num_key_value_heads = 1,
            )
            self.layers = nn.ModuleList([_Layer()])

    reader = SimpleNamespace(
        tensors = [
            _FakeTensor(
                "blk.0.attn_v.weight",
                "QW",
                torch.ones(2, 2, dtype = torch.uint8),
            ),
            _FakeTensor(
                "blk.0.attn_v.bias",
                "QB",
                torch.ones(2, dtype = torch.uint8),
            ),
        ]
    )
    language_model = _LanguageModel()
    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        if quant_type == "QW":
            return torch.eye(2, dtype = dtype)
        assert quant_type == "QB"
        assert logical_shape == (2,)
        return torch.tensor([5.0, 7.0], dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)

    got_reader = g._replace_mistral_modules_with_lazy_gguf(
        language_model,
        gguf_path = tmp_path / "mistral.gguf",
        compute_dtype = torch.float32,
        resident_device = "cpu",
    )

    v_proj = language_model.layers[0].self_attn.v_proj
    assert got_reader is reader
    assert isinstance(v_proj, g.LazyGGUFLinear)
    assert v_proj.qweight.device == torch.device("cpu")
    assert v_proj.qbias.device == torch.device("cpu")
    assert v_proj.bias is None

    out = v_proj(torch.tensor([[1.0, 2.0]], dtype = torch.float32))

    torch.testing.assert_close(out, torch.tensor([[6.0, 9.0]]))


def test_dequantize_bf16_gguf_bytes_without_diffusers_import(monkeypatch):
    import builtins

    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

        @staticmethod
        def quant_shape_from_byte_shape(shape, quant_type):
            assert quant_type == _FakeQuantTypes.BF16
            return (*shape[:-1], shape[-1] // 2)

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("diffusers.quantizers"):
            raise AssertionError("BF16 GGUF decode should not import diffusers quantizers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    raw = torch.tensor([0x80, 0x3F, 0x80, 0xBF], dtype = torch.uint8)

    decoded = g._dequantize_gguf_bytes(
        raw,
        _FakeQuantTypes.BF16,
        dtype = torch.bfloat16,
        logical_shape = (2,),
    )

    assert decoded.dtype is torch.bfloat16
    torch.testing.assert_close(
        decoded.float(),
        torch.tensor([1.0, -1.0], dtype = torch.float32),
    )

    decoded_matrix = g._dequantize_gguf_bytes(
        raw.reshape(1, 4),
        _FakeQuantTypes.BF16,
    )

    assert decoded_matrix.shape == (1, 2)
    torch.testing.assert_close(
        decoded_matrix,
        torch.tensor([[1.0, -1.0]], dtype = torch.float32),
    )


def test_replace_mapped_text_modules_materializes_dense_standalone_bias(
    monkeypatch,
    tmp_path,
):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeTensor:
        def __init__(self, name, tensor_type, data):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = tuple(reversed(tuple(data.shape)))

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

    with torch.device("meta"):
        root = nn.Module()
        root.proj = nn.Linear(2, 2, bias = True)

    reader = SimpleNamespace(
        tensors = [
            _FakeTensor(
                "proj.weight",
                _FakeQuantTypes.F32,
                torch.eye(2, dtype = torch.float32),
            ),
            _FakeTensor(
                "proj.bias",
                _FakeQuantTypes.F32,
                torch.tensor([3.0, 4.0], dtype = torch.float32),
            ),
        ],
        get_field = lambda name: None,
    )

    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)
    monkeypatch.setattr(g, "_open_gguf_reader", lambda path: reader)

    stats_reader, stats = g.replace_mapped_text_modules_with_lazy_gguf(
        root,
        tmp_path / "dense.gguf",
        map_name = lambda name: g._GGUFNameTarget(name),
        compute_dtype = torch.bfloat16,
        resident_device = "cpu",
    )

    assert stats_reader is reader
    assert stats.loaded == 2
    assert stats.materialized == 2
    assert stats.lazy_linear == 0
    assert root.proj.weight.device == torch.device("cpu")
    assert root.proj.bias.device == torch.device("cpu")
    assert root.proj.weight.dtype is torch.bfloat16
    assert root.proj.bias.dtype is torch.bfloat16
    torch.testing.assert_close(root.proj.bias.float(), torch.tensor([3.0, 4.0]))


def test_lazy_qwen2vl_text_encoder_uses_text_and_mmproj_replacements(
    monkeypatch,
    tmp_path,
):
    import core.inference.gguf_text_encoder as g

    calls: list[tuple[object, dict]] = []
    text_calls: list[dict] = []
    mmproj_calls: list[dict] = []

    class _TinyQwen(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.frozen = False

        def requires_grad_(self, requires_grad = True):
            self.frozen = not requires_grad
            return self

    def fake_from_pretrained(path_or_repo, **kwargs):
        calls.append((path_or_repo, kwargs))
        return _fake_flux2_config()

    def fake_text_replace(root, gguf_path, *, map_name, compute_dtype, resident_device = None):
        text_calls.append(
            {
                "root": root,
                "gguf_path": gguf_path,
                "map_name": map_name,
                "compute_dtype": compute_dtype,
                "resident_device": resident_device,
            }
        )
        return "text-reader", g.LazyGGUFReplacementStats(
            loaded = 1,
            lazy_linear = 1,
            lazy_embedding = 0,
            materialized = 0,
        )

    def fake_mmproj_replace(root, mmproj_gguf_path, *, compute_dtype):
        mmproj_calls.append(
            {
                "root": root,
                "mmproj_gguf_path": mmproj_gguf_path,
                "compute_dtype": compute_dtype,
            }
        )
        return "mmproj-reader", g.Qwen2VLMmprojReplacementStats(
            loaded = 1,
            materialized = 1,
            qkv_groups = 0,
            stacked_patch_embeddings = 0,
        )

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(g, "_qwen2_5_vl_model_class", lambda: _TinyQwen)
    monkeypatch.setattr(g, "_materialize_qwen2vl_rotary_buffers", lambda root, config: None)
    monkeypatch.setattr(g, "replace_mapped_text_modules_with_lazy_gguf", fake_text_replace)
    monkeypatch.setattr(g, "replace_qwen2vl_mmproj_modules_with_gguf", fake_mmproj_replace)

    encoder = g.LazyQwen2VLTextEncoder.from_gguf(
        tmp_path / "text.gguf",
        base_repo_or_path = "Qwen/Qwen-Image",
        mmproj_gguf_path = tmp_path / "mmproj.gguf",
        compute_dtype = torch.bfloat16,
        resident_device = "cpu",
        token = "hf_qwen",
    )

    assert isinstance(encoder, _TinyQwen)
    assert encoder.frozen is True
    assert calls == [
        (
            "Qwen/Qwen-Image",
            {"subfolder": "text_encoder", "token": "hf_qwen"},
        )
    ]
    assert text_calls == [
        {
            "root": encoder,
            "gguf_path": tmp_path / "text.gguf",
            "map_name": g.map_qwen2vl_text_gguf_name,
            "compute_dtype": torch.bfloat16,
            "resident_device": "cpu",
        }
    ]
    assert mmproj_calls == [
        {
            "root": encoder,
            "mmproj_gguf_path": tmp_path / "mmproj.gguf",
            "compute_dtype": torch.bfloat16,
        }
    ]
    assert encoder._gguf_reader == "text-reader"
    assert encoder._mmproj_gguf_reader == "mmproj-reader"
    assert encoder._gguf_mmproj_path == tmp_path / "mmproj.gguf"


def test_lazy_qwen3_text_encoder_uses_text_replacements(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    calls: list[tuple[object, dict]] = []
    text_calls: list[dict] = []
    config = Qwen3Config(
        vocab_size = 16,
        hidden_size = 8,
        intermediate_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        max_position_embeddings = 16,
        layer_types = ["full_attention"],
    )

    def fake_from_pretrained(path_or_repo, **kwargs):
        calls.append((path_or_repo, kwargs))
        return config

    def fake_text_replace(root, gguf_path, *, map_name, compute_dtype, resident_device = None):
        text_calls.append(
            {
                "root": root,
                "gguf_path": gguf_path,
                "map_name": map_name,
                "compute_dtype": compute_dtype,
                "resident_device": resident_device,
            }
        )
        return "qwen3-reader", g.LazyGGUFReplacementStats(
            loaded = 1,
            lazy_linear = 1,
            lazy_embedding = 0,
            materialized = 0,
        )

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(g, "replace_mapped_text_modules_with_lazy_gguf", fake_text_replace)

    encoder = g.LazyQwen3TextEncoder.from_gguf(
        tmp_path / "qwen3.gguf",
        base_repo_or_path = "Tongyi-MAI/Z-Image-Turbo",
        compute_dtype = torch.bfloat16,
        resident_device = "cpu",
        token = "hf_z",
    )

    assert isinstance(encoder, g.LazyQwen3TextEncoder)
    assert not any(param.requires_grad for param in encoder.parameters())
    assert not any(buffer.is_meta for _name, buffer in encoder.named_buffers())
    assert calls == [
        (
            "Tongyi-MAI/Z-Image-Turbo",
            {"subfolder": "text_encoder", "token": "hf_z"},
        )
    ]
    assert text_calls == [
        {
            "root": encoder,
            "gguf_path": tmp_path / "qwen3.gguf",
            "map_name": g.map_qwen3_text_gguf_name,
            "compute_dtype": torch.bfloat16,
            "resident_device": "cpu",
        }
    ]
    assert encoder._gguf_reader == "qwen3-reader"


def test_lazy_llama_text_encoder_uses_text_replacements(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    calls: list[tuple[object, dict]] = []
    text_calls: list[dict] = []
    config = LlamaConfig(
        vocab_size = 16,
        hidden_size = 8,
        intermediate_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        max_position_embeddings = 16,
    )

    def fake_from_pretrained(path_or_repo, **kwargs):
        calls.append((path_or_repo, kwargs))
        return config

    def fake_text_replace(root, gguf_path, *, map_name, compute_dtype, resident_device = None):
        mapped = map_name("blk.0.attn_q.weight")
        text_calls.append(
            {
                "root": root,
                "gguf_path": gguf_path,
                "hf_name": mapped.hf_name,
                "reverse_permute_heads": mapped.reverse_permute_heads,
                "compute_dtype": compute_dtype,
                "resident_device": resident_device,
            }
        )
        return "llama-reader", g.LazyGGUFReplacementStats(
            loaded = 1,
            lazy_linear = 1,
            lazy_embedding = 0,
            materialized = 0,
        )

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(g, "replace_mapped_text_modules_with_lazy_gguf", fake_text_replace)

    encoder = g.LazyLlamaTextEncoder.from_gguf(
        tmp_path / "llama.gguf",
        base_repo_or_path = "owner/llama-text",
        compute_dtype = torch.bfloat16,
        resident_device = "cpu",
        token = "hf_llama",
    )

    assert isinstance(encoder, g.LazyLlamaTextEncoder)
    assert not any(param.requires_grad for param in encoder.parameters())
    assert not any(buffer.is_meta for _name, buffer in encoder.named_buffers())
    assert calls == [
        (
            "owner/llama-text",
            {"subfolder": "text_encoder", "token": "hf_llama"},
        )
    ]
    assert text_calls == [
        {
            "root": encoder,
            "gguf_path": tmp_path / "llama.gguf",
            "hf_name": "model.layers.0.self_attn.q_proj.weight",
            "reverse_permute_heads": 2,
            "compute_dtype": torch.bfloat16,
            "resident_device": "cpu",
        }
    ]
    assert encoder._gguf_reader == "llama-reader"


def test_lazy_qwen3vl_text_encoder_uses_text_replacements(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    calls: list[tuple[object, dict]] = []
    text_calls: list[dict] = []
    config = Qwen3VLTextConfig(
        vocab_size = 16,
        hidden_size = 8,
        intermediate_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 4,
        max_position_embeddings = 16,
        rope_scaling = {"rope_type": "default", "mrope_section": [1, 1, 2]},
    )

    def fake_from_pretrained(path_or_repo, **kwargs):
        calls.append((path_or_repo, kwargs))
        return config

    def fake_text_replace(root, gguf_path, *, map_name, compute_dtype, resident_device = None):
        text_calls.append(
            {
                "root": root,
                "gguf_path": gguf_path,
                "map_name": map_name,
                "compute_dtype": compute_dtype,
                "resident_device": resident_device,
            }
        )
        return "qwen3vl-reader", g.LazyGGUFReplacementStats(
            loaded = 1,
            lazy_linear = 1,
            lazy_embedding = 0,
            materialized = 0,
        )

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(g, "replace_mapped_text_modules_with_lazy_gguf", fake_text_replace)

    encoder = g.LazyQwen3VLTextEncoder.from_gguf(
        tmp_path / "qwen3vl.gguf",
        base_repo_or_path = "owner/qwen3vl-text",
        compute_dtype = torch.float16,
        resident_device = "cpu",
    )

    assert isinstance(encoder, g.LazyQwen3VLTextEncoder)
    assert not any(param.requires_grad for param in encoder.parameters())
    assert not any(buffer.is_meta for _name, buffer in encoder.named_buffers())
    assert text_calls == [
        {
            "root": encoder,
            "gguf_path": tmp_path / "qwen3vl.gguf",
            "map_name": g.map_qwen3vl_text_gguf_name,
            "compute_dtype": torch.float16,
            "resident_device": "cpu",
        }
    ]
    assert encoder._gguf_reader == "qwen3vl-reader"


def test_lazy_qwen3vl_text_encoder_accepts_multimodal_config(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    text_config = Qwen3VLTextConfig(
        vocab_size = 8,
        hidden_size = 4,
        intermediate_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
    )
    multimodal_config = SimpleNamespace(text_config = text_config)
    seen: dict[str, object] = {}

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", lambda *args, **kwargs: multimodal_config)
    monkeypatch.setattr(g, "_materialize_qwen3vl_rotary_buffers", lambda model, config: seen.setdefault("config", config))

    def fake_text_replace(root, gguf_path, *, map_name, compute_dtype, resident_device = None):
        return "qwen3vl-reader", g.LazyGGUFReplacementStats(
            loaded = 0,
            lazy_linear = 0,
            lazy_embedding = 0,
            materialized = 0,
        )

    monkeypatch.setattr(g, "replace_mapped_text_modules_with_lazy_gguf", fake_text_replace)

    encoder = g.LazyQwen3VLTextEncoder.from_gguf(
        tmp_path / "qwen3vl.gguf",
        base_repo_or_path = "owner/qwen3vl",
        subfolder = "",
        compute_dtype = torch.float32,
    )

    assert isinstance(encoder, g.LazyQwen3VLTextEncoder)
    assert seen["config"] is text_config
    assert encoder.config is text_config


def test_lazy_gemma3_text_encoder_uses_text_replacements(monkeypatch, tmp_path):
    import core.inference.gguf_text_encoder as g

    text_calls: list[dict] = []
    config = Gemma3TextConfig(
        vocab_size = 16,
        hidden_size = 8,
        intermediate_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 4,
        max_position_embeddings = 16,
    )

    def fake_text_replace(root, gguf_path, *, map_name, compute_dtype, resident_device = None):
        text_calls.append(
            {
                "root": root,
                "gguf_path": gguf_path,
                "map_name": map_name,
                "compute_dtype": compute_dtype,
                "resident_device": resident_device,
            }
        )
        return "gemma3-reader", g.LazyGGUFReplacementStats(
            loaded = 1,
            lazy_linear = 1,
            lazy_embedding = 0,
            materialized = 0,
        )

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", lambda *a, **k: config)
    monkeypatch.setattr(g, "replace_mapped_text_modules_with_lazy_gguf", fake_text_replace)

    encoder = g.LazyGemma3TextEncoder.from_gguf(
        tmp_path / "gemma3.gguf",
        base_repo_or_path = "owner/gemma3-text",
        compute_dtype = torch.bfloat16,
        resident_device = "cpu",
    )

    assert isinstance(encoder, g.LazyGemma3TextEncoder)
    assert not any(param.requires_grad for param in encoder.parameters())
    assert not any(buffer.is_meta for _name, buffer in encoder.named_buffers())
    assert text_calls == [
        {
            "root": encoder,
            "gguf_path": tmp_path / "gemma3.gguf",
            "map_name": g.map_gemma3_text_gguf_name,
            "compute_dtype": torch.bfloat16,
            "resident_device": "cpu",
        }
    ]
    assert encoder._gguf_reader == "gemma3-reader"


def test_lazy_t5_text_encoder_uses_text_replacements_and_component_subfolder(
    monkeypatch,
    tmp_path,
):
    import core.inference.gguf_text_encoder as g

    calls: list[tuple[object, dict]] = []
    text_calls: list[dict] = []
    config = T5Config(
        vocab_size = 16,
        d_model = 8,
        d_ff = 16,
        num_layers = 1,
        num_heads = 2,
        d_kv = 4,
        feed_forward_proj = "gated-gelu",
        is_encoder_decoder = False,
        use_cache = False,
    )

    def fake_from_pretrained(path_or_repo, **kwargs):
        calls.append((path_or_repo, kwargs))
        return config

    def fake_text_replace(root, gguf_path, *, map_name, compute_dtype, resident_device = None):
        root.shared = nn.Embedding(config.vocab_size, config.d_model)
        text_calls.append(
            {
                "root": root,
                "gguf_path": gguf_path,
                "map_name": map_name,
                "compute_dtype": compute_dtype,
                "resident_device": resident_device,
            }
        )
        return "t5-reader", g.LazyGGUFReplacementStats(
            loaded = 1,
            lazy_linear = 0,
            lazy_embedding = 1,
            materialized = 0,
        )

    monkeypatch.setattr(g.AutoConfig, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(g, "replace_mapped_text_modules_with_lazy_gguf", fake_text_replace)

    encoder = g.LazyT5TextEncoder.from_gguf(
        tmp_path / "t5.gguf",
        base_repo_or_path = "black-forest-labs/FLUX.1-dev",
        subfolder = "text_encoder_2",
        compute_dtype = torch.bfloat16,
        resident_device = "cpu",
        token = "hf_t5",
    )

    assert isinstance(encoder, g.LazyT5TextEncoder)
    assert encoder.encoder.embed_tokens is encoder.shared
    assert not any(param.requires_grad for param in encoder.parameters())
    assert calls == [
        (
            "black-forest-labs/FLUX.1-dev",
            {"subfolder": "text_encoder_2", "token": "hf_t5"},
        )
    ]
    assert text_calls == [
        {
            "root": encoder,
            "gguf_path": tmp_path / "t5.gguf",
            "map_name": g.map_t5_text_gguf_name,
            "compute_dtype": torch.bfloat16,
            "resident_device": "cpu",
        }
    ]
    assert encoder._gguf_reader == "t5-reader"


def test_lazy_gguf_linear_dequantizes_during_forward(monkeypatch):
    import core.inference.gguf_text_encoder as g

    calls = 0

    def fake_dequant(qweight, quant_type, *, dtype = None):
        nonlocal calls
        calls += 1
        assert qweight.dtype is torch.uint8
        assert quant_type == "Q4"
        assert dtype is None
        return torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype = torch.float32)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    layer = g.LazyGGUFLinear(
        torch.ones(2, 2, dtype = torch.uint8),
        "Q4",
        in_features = 2,
        out_features = 2,
        compute_dtype = torch.float32,
    )

    out = layer(torch.tensor([[10.0, 20.0]], dtype = torch.float32))

    assert calls == 1
    torch.testing.assert_close(out, torch.tensor([[50.0, 110.0]]))


def test_lazy_gguf_linear_weight_is_metadata_only_for_dtype_inspection():
    import core.inference.gguf_text_encoder as g

    layer = g.LazyGGUFLinear(
        torch.ones(2, 2, dtype = torch.uint8),
        "Q4",
        in_features = 2,
        out_features = 2,
        compute_dtype = torch.bfloat16,
    )

    assert layer.weight.dtype is torch.bfloat16
    assert layer.weight.numel() == 0
    assert layer.qweight.dtype is torch.uint8


def test_lazy_gguf_linear_applies_materialized_bias(monkeypatch):
    import core.inference.gguf_text_encoder as g

    monkeypatch.setattr(
        g,
        "_dequantize_gguf_bytes",
        lambda qweight, quant_type, *, dtype = None: torch.eye(2, dtype = dtype),
    )
    layer = g.LazyGGUFLinear(
        torch.ones(2, 2, dtype = torch.uint8),
        "Q4",
        in_features = 2,
        out_features = 2,
        compute_dtype = torch.float32,
        bias = torch.tensor([5.0, 7.0]),
    )

    out = layer(torch.tensor([[1.0, 2.0]], dtype = torch.float32))

    torch.testing.assert_close(out, torch.tensor([[6.0, 9.0]]))


def test_lazy_gguf_linear_dequantizes_quantized_bias(monkeypatch):
    import core.inference.gguf_text_encoder as g

    calls: list[str] = []

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        calls.append(quant_type)
        if quant_type == "QW":
            return torch.eye(2, dtype = dtype)
        assert quant_type == "QB"
        assert logical_shape == (2,)
        return torch.tensor([5.0, 7.0], dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    layer = g.LazyGGUFLinear(
        torch.ones(2, 2, dtype = torch.uint8),
        "QW",
        in_features = 2,
        out_features = 2,
        compute_dtype = torch.float32,
        qbias = torch.ones(2, dtype = torch.uint8),
        bias_quant_type = "QB",
        bias_logical_shape = (2,),
    )

    out = layer(torch.tensor([[1.0, 2.0]], dtype = torch.float32))

    assert calls == ["QW", "QB"]
    assert layer.bias is None
    torch.testing.assert_close(out, torch.tensor([[6.0, 9.0]]))


def test_replace_mapped_text_modules_with_lazy_gguf_handles_qwen2vl_biases(
    monkeypatch,
):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

    class _FakeTensor:
        def __init__(self, name, tensor_type, data):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = tuple(reversed(tuple(data.shape)))

    class _SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 2, bias = True)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _SelfAttention()

    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(4, 2)
            self.layers = nn.ModuleList([_Layer()])
            self.norm = nn.LayerNorm(2)

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = _LanguageModel()
            self.lm_head = nn.Linear(2, 4, bias = False)

    root = _Root()
    reader = SimpleNamespace(
        tensors = [
            _FakeTensor(
                "token_embd.weight",
                "Q4",
                torch.ones(4, 2, dtype = torch.uint8),
            ),
            _FakeTensor(
                "blk.0.attn_q.weight",
                "Q4",
                torch.ones(2, 2, dtype = torch.uint8),
            ),
            _FakeTensor(
                "blk.0.attn_q.bias",
                "F32",
                torch.tensor([5.0, 7.0], dtype = torch.float32),
            ),
            _FakeTensor(
                "output_norm.weight",
                "F32",
                torch.tensor([3.0, 4.0], dtype = torch.float32),
            ),
        ]
    )
    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)
    monkeypatch.setattr(g, "_open_gguf_reader", lambda path: reader)
    monkeypatch.setattr(
        g,
        "_dequantize_gguf_bytes",
        lambda qweight, quant_type, *, dtype = None: torch.eye(2, dtype = dtype),
    )

    got_reader, stats = g.replace_mapped_text_modules_with_lazy_gguf(
        root,
        "text.gguf",
        map_name = g.map_qwen2vl_text_gguf_name,
        compute_dtype = torch.float32,
        resident_device = "cpu",
    )

    q_proj = root.model.language_model.layers[0].self_attn.q_proj
    assert got_reader is reader
    assert stats.loaded == 4
    assert stats.lazy_linear == 1
    assert stats.lazy_embedding == 1
    assert stats.materialized == 2
    assert isinstance(q_proj, g.LazyGGUFLinear)
    assert isinstance(root.model.language_model.embed_tokens, g.LazyGGUFEmbedding)
    torch.testing.assert_close(q_proj.bias, torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(
        root.model.language_model.norm.weight,
        torch.tensor([3.0, 4.0]),
    )
    assert q_proj.qweight.device == torch.device("cpu")

    out = q_proj(torch.tensor([[1.0, 2.0]], dtype = torch.float32))

    torch.testing.assert_close(out, torch.tensor([[6.0, 9.0]]))


def test_replace_mapped_text_modules_keeps_quantized_linear_bias_lazy(monkeypatch):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

    class _FakeTensor:
        def __init__(self, name, tensor_type, data):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = tuple(reversed(tuple(data.shape)))

    class _SelfAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 2, bias = True)

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _SelfAttention()

    class _LanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Layer()])

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = _LanguageModel()

    root = _Root()
    reader = SimpleNamespace(
        tensors = [
            _FakeTensor(
                "blk.0.attn_q.weight",
                "QW",
                torch.ones(2, 2, dtype = torch.uint8),
            ),
            _FakeTensor(
                "blk.0.attn_q.bias",
                "QB",
                torch.ones(2, dtype = torch.uint8),
            ),
        ]
    )
    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)
    monkeypatch.setattr(g, "_open_gguf_reader", lambda path: reader)

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        if quant_type == "QW":
            return torch.eye(2, dtype = dtype)
        assert quant_type == "QB"
        assert logical_shape == (2,)
        return torch.tensor([5.0, 7.0], dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)

    got_reader, stats = g.replace_mapped_text_modules_with_lazy_gguf(
        root,
        "text.gguf",
        map_name = g.map_qwen2vl_text_gguf_name,
        compute_dtype = torch.float32,
        resident_device = "cpu",
    )

    q_proj = root.model.language_model.layers[0].self_attn.q_proj
    assert got_reader is reader
    assert stats.loaded == 2
    assert stats.lazy_linear == 1
    assert stats.lazy_embedding == 0
    assert stats.materialized == 0
    assert isinstance(q_proj, g.LazyGGUFLinear)
    assert q_proj.qweight.device == torch.device("cpu")
    assert q_proj.qbias.device == torch.device("cpu")
    assert q_proj.bias is None

    out = q_proj(torch.tensor([[1.0, 2.0]], dtype = torch.float32))

    torch.testing.assert_close(out, torch.tensor([[6.0, 9.0]]))


def test_replace_mapped_text_modules_applies_gemma3_norm_correction(monkeypatch):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

    class _FakeTensor:
        def __init__(self, name, tensor_type, data):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = tuple(reversed(tuple(data.shape)))

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(2)

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.norm = nn.LayerNorm(2)
            self.model.layers = nn.ModuleList([_Layer()])

    root = _Root()
    reader = SimpleNamespace(
        tensors = [
            _FakeTensor(
                "output_norm.weight",
                "F32",
                torch.tensor([2.0, 3.0], dtype = torch.float32),
            ),
            _FakeTensor(
                "blk.0.attn_norm.weight",
                "Q4",
                torch.tensor([1, 2], dtype = torch.uint8),
            ),
        ]
    )
    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)
    monkeypatch.setattr(g, "_open_gguf_reader", lambda path: reader)
    monkeypatch.setattr(
        g,
        "_dequantize_gguf_bytes",
        lambda qweight, quant_type, *, dtype = None, logical_shape = None: torch.tensor(
            [4.0, 5.0],
            dtype = dtype,
        ),
    )

    got_reader, stats = g.replace_mapped_text_modules_with_lazy_gguf(
        root,
        "gemma3.gguf",
        map_name = g.map_gemma3_text_gguf_name,
        compute_dtype = torch.float32,
        resident_device = "cpu",
    )

    assert got_reader is reader
    assert stats.loaded == 2
    assert stats.lazy_linear == 0
    assert stats.lazy_embedding == 0
    assert stats.materialized == 2
    torch.testing.assert_close(root.model.norm.weight, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(
        root.model.layers[0].input_layernorm.weight,
        torch.tensor([3.0, 4.0]),
    )


def test_replace_qwen2vl_mmproj_modules_groups_qkv_and_stacks_patch_embed(
    monkeypatch,
):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

    class _FakeTensor:
        def __init__(self, name, data, tensor_type = "F32"):
            self.name = name
            self.tensor_type = tensor_type
            self.data = data
            self.shape = tuple(reversed(tuple(data.shape)))

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(2, 6, bias = True)
            self.proj = nn.Linear(2, 2, bias = True)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = _Attention()

    class _PatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv3d(3, 2, kernel_size = (2, 1, 1), bias = False)

    class _Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = _PatchEmbed()
            self.blocks = nn.ModuleList([_Block()])

    class _Root(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = _Visual()

    q_w = torch.full((2, 2), 1.0)
    k_w = torch.full((2, 2), 2.0)
    v_w = torch.full((2, 2), 3.0)
    q_b = torch.tensor([1.0, 2.0])
    k_b = torch.tensor([3.0, 4.0])
    v_b = torch.tensor([5.0, 6.0])
    patch_0 = torch.full((2, 3, 1, 1), 7.0)
    patch_1 = torch.full((2, 3, 1, 1), 8.0)
    reader = SimpleNamespace(
        tensors = [
            _FakeTensor("v.blk.0.attn_q.weight", q_w),
            _FakeTensor("v.blk.0.attn_k.weight", k_w),
            _FakeTensor("v.blk.0.attn_v.weight", v_w),
            _FakeTensor("v.blk.0.attn_q.bias", q_b),
            _FakeTensor("v.blk.0.attn_k.bias", k_b),
            _FakeTensor("v.blk.0.attn_v.bias", v_b),
            _FakeTensor("v.patch_embd.weight", patch_0),
            _FakeTensor("v.patch_embd.weight.1", patch_1),
        ]
    )
    root = _Root()
    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)
    monkeypatch.setattr(g, "_open_gguf_reader", lambda path: reader)

    got_reader, stats = g.replace_qwen2vl_mmproj_modules_with_gguf(
        root,
        "mmproj.gguf",
        compute_dtype = torch.float32,
    )

    assert got_reader is reader
    assert stats.loaded == 8
    assert stats.materialized == 8
    assert stats.qkv_groups == 2
    assert stats.stacked_patch_embeddings == 1
    torch.testing.assert_close(
        root.model.visual.blocks[0].attn.qkv.weight,
        torch.cat([q_w, k_w, v_w], dim = 0),
    )
    torch.testing.assert_close(
        root.model.visual.blocks[0].attn.qkv.bias,
        torch.cat([q_b, k_b, v_b], dim = 0),
    )
    torch.testing.assert_close(
        root.model.visual.patch_embed.proj.weight,
        torch.stack([patch_0, patch_1], dim = 2),
    )


def test_materialize_bfloat16_gguf_tensor_respects_compute_dtype(monkeypatch):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

    tensor = SimpleNamespace(
        name = "v.blk.0.attn_q.weight",
        tensor_type = "BF16",
        data = torch.ones(2, 2, dtype = torch.uint8),
        shape = (2, 2),
    )
    dequant_calls: list[dict] = []

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        dequant_calls.append(
            {
                "quant_type": quant_type,
                "dtype": dtype,
                "logical_shape": logical_shape,
            }
        )
        # Diffusers' BF16 dequant helper can return fp32 despite the
        # requested dtype. Studio must enforce compute_dtype before
        # installing mmproj / visual weights into the model.
        return torch.ones(2, 2, dtype = torch.float32)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)

    value = g._materialize_gguf_tensor(
        tensor,
        compute_dtype = torch.bfloat16,
        gguf = _FakeGGUF,
    )

    assert value.dtype is torch.bfloat16
    assert dequant_calls == [
        {
            "quant_type": "BF16",
            "dtype": torch.bfloat16,
            "logical_shape": (2, 2),
        }
    ]


def test_replace_qwen2vl_mmproj_modules_requires_complete_qkv_group(monkeypatch):
    import core.inference.gguf_text_encoder as g

    class _FakeQuantTypes:
        F32 = "F32"
        F16 = "F16"
        BF16 = "BF16"

    class _FakeGGUF:
        GGMLQuantizationType = _FakeQuantTypes

    class _FakeTensor:
        def __init__(self, name, data):
            self.name = name
            self.tensor_type = "F32"
            self.data = data
            self.shape = tuple(reversed(tuple(data.shape)))

    root = nn.Module()
    root.model = nn.Module()
    root.model.visual = nn.Module()
    root.model.visual.blocks = nn.ModuleList([nn.Module()])
    root.model.visual.blocks[0].attn = nn.Module()
    root.model.visual.blocks[0].attn.qkv = nn.Linear(2, 6, bias = False)
    reader = SimpleNamespace(
        tensors = [
            _FakeTensor("v.blk.0.attn_q.weight", torch.ones(2, 2)),
            _FakeTensor("v.blk.0.attn_k.weight", torch.ones(2, 2)),
        ]
    )
    monkeypatch.setattr(g, "_require_gguf", lambda: _FakeGGUF)
    monkeypatch.setattr(g, "_open_gguf_reader", lambda path: reader)

    with pytest.raises(RuntimeError, match = "missing \\['v'\\]"):
        g.replace_qwen2vl_mmproj_modules_with_gguf(
            root,
            "mmproj.gguf",
            compute_dtype = torch.float32,
        )


def test_lazy_gguf_linear_can_keep_quantized_weight_cpu_resident():
    import core.inference.gguf_text_encoder as g

    layer = g.LazyGGUFLinear(
        torch.ones(2, 2, dtype = torch.uint8),
        "Q4",
        in_features = 2,
        out_features = 2,
        compute_dtype = torch.float32,
        resident_device = "cpu",
    )

    layer._apply(lambda tensor: tensor.to("meta"))

    assert layer.qweight.device == torch.device("cpu")
    assert layer.qweight.dtype is torch.uint8


def test_lazy_gguf_linear_can_pin_cpu_resident_quantized_weight(monkeypatch):
    import core.inference.gguf_text_encoder as g

    pinned: list[torch.Tensor] = []

    def fake_pin(tensor, *, pin_memory = None):
        pinned.append(tensor)
        return tensor

    monkeypatch.setattr(g, "_pin_cpu_tensor_for_transfer", fake_pin)
    layer = g.LazyGGUFLinear(
        torch.ones(2, 2, dtype = torch.uint8),
        "Q4",
        in_features = 2,
        out_features = 2,
        compute_dtype = torch.float32,
        resident_device = "cpu",
    )

    assert pinned
    assert layer.qweight.device == torch.device("cpu")
    assert layer.qweight.dtype is torch.uint8


def test_lazy_gguf_linear_without_resident_device_follows_module_apply():
    import core.inference.gguf_text_encoder as g

    layer = g.LazyGGUFLinear(
        torch.ones(2, 2, dtype = torch.uint8),
        "Q4",
        in_features = 2,
        out_features = 2,
        compute_dtype = torch.float32,
    )

    layer._apply(lambda tensor: tensor.to("meta"))

    assert layer.qweight.device.type == "meta"


def test_lazy_gguf_embedding_dequantizes_only_requested_rows(monkeypatch):
    import core.inference.gguf_text_encoder as g

    seen_rows: list[torch.Tensor] = []

    def fake_dequant(qweight, quant_type, *, dtype = None):
        seen_rows.append(qweight.clone())
        assert quant_type == "Q4"
        return qweight.to(dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    layer = g.LazyGGUFEmbedding(
        torch.tensor(
            [
                [10, 11],
                [20, 21],
                [30, 31],
                [40, 41],
            ],
            dtype = torch.uint8,
        ),
        "Q4",
        num_embeddings = 4,
        embedding_dim = 2,
        compute_dtype = torch.float32,
    )

    out = layer(torch.tensor([[3, 1, 3]]))

    assert len(seen_rows) == 1
    torch.testing.assert_close(
        seen_rows[0],
        torch.tensor([[20, 21], [40, 41]], dtype = torch.uint8),
    )
    torch.testing.assert_close(
        out,
        torch.tensor([[[40.0, 41.0], [20.0, 21.0], [40.0, 41.0]]]),
    )


def test_lazy_gguf_embedding_preserves_max_norm_semantics(monkeypatch):
    import core.inference.gguf_text_encoder as g

    def fake_dequant(qweight, quant_type, *, dtype = None):
        assert quant_type == "Q4"
        return qweight.to(dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    layer = g.LazyGGUFEmbedding(
        torch.tensor(
            [
                [3, 4],
                [6, 8],
                [0, 2],
            ],
            dtype = torch.uint8,
        ),
        "Q4",
        num_embeddings = 3,
        embedding_dim = 2,
        compute_dtype = torch.float32,
        max_norm = 1.0,
        norm_type = 2.0,
    )

    out = layer(torch.tensor([[1, 0]]))

    torch.testing.assert_close(
        out,
        torch.tensor([[[0.6, 0.8], [0.6, 0.8]]]),
        rtol = 1e-5,
        atol = 1e-6,
    )


def test_lazy_gguf_embedding_can_keep_quantized_weight_cpu_resident():
    import core.inference.gguf_text_encoder as g

    layer = g.LazyGGUFEmbedding(
        torch.ones(4, 2, dtype = torch.uint8),
        "Q4",
        num_embeddings = 4,
        embedding_dim = 2,
        compute_dtype = torch.float32,
        resident_device = "cpu",
    )

    layer._apply(lambda tensor: tensor.to("meta"))

    assert layer.qweight.device == torch.device("cpu")
    assert layer.qweight.dtype is torch.uint8


def test_lazy_gguf_conv2d_dequantizes_in_forward(monkeypatch):
    import core.inference.gguf_text_encoder as g

    seen: list[tuple[torch.Tensor, str, torch.dtype | None, tuple[int, ...] | None]] = []

    def fake_dequant(qweight, quant_type, *, dtype = None, logical_shape = None):
        seen.append((qweight.clone(), quant_type, dtype, logical_shape))
        return torch.ones(logical_shape, dtype = dtype)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    layer = g.LazyGGUFConv2d(
        torch.ones(1, 2, dtype = torch.uint8),
        "Q4",
        in_channels = 1,
        out_channels = 1,
        kernel_size = (1, 1),
        stride = (1, 1),
        padding = (0, 0),
        dilation = (1, 1),
        groups = 1,
        compute_dtype = torch.float32,
        logical_shape = (1, 1, 1, 1),
    )

    out = layer(torch.full((1, 1, 2, 2), 3.0))

    assert out.shape == (1, 1, 2, 2)
    torch.testing.assert_close(out, torch.full((1, 1, 2, 2), 3.0))
    assert seen[0][1:] == ("Q4", torch.float32, (1, 1, 1, 1))


def test_lazy_gguf_conv2d_dequantizes_quantized_bias_in_forward(monkeypatch):
    import core.inference.gguf_text_encoder as g

    seen: list[tuple[str, torch.Tensor, str, tuple[int, ...] | None]] = []

    def fake_dequant(qbuffer, quant_type, *, dtype = None, logical_shape = None):
        label = "bias" if tuple(qbuffer.shape) == (1,) else "weight"
        seen.append((label, qbuffer.clone(), quant_type, logical_shape))
        return qbuffer.to(dtype = dtype).reshape(logical_shape)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    layer = g.LazyGGUFConv2d(
        torch.ones(1, 1, 1, 1, dtype = torch.uint8),
        "Q4",
        in_channels = 1,
        out_channels = 1,
        kernel_size = (1, 1),
        stride = (1, 1),
        padding = (0, 0),
        dilation = (1, 1),
        groups = 1,
        compute_dtype = torch.float32,
        logical_shape = (1, 1, 1, 1),
        qbias = torch.tensor([2], dtype = torch.uint8),
        bias_quant_type = "Q4",
        bias_logical_shape = (1,),
    )

    out = layer(torch.full((1, 1, 1, 1), 3.0))

    torch.testing.assert_close(out, torch.tensor([[[[5.0]]]]))
    assert [item[0] for item in seen] == ["weight", "bias"]
    assert seen[1][2:] == ("Q4", (1,))


def test_lazy_gguf_conv2d_can_keep_quantized_weight_cpu_resident():
    import core.inference.gguf_text_encoder as g

    layer = g.LazyGGUFConv2d(
        torch.ones(1, 2, dtype = torch.uint8),
        "Q4",
        in_channels = 1,
        out_channels = 1,
        kernel_size = (1, 1),
        stride = (1, 1),
        padding = (0, 0),
        dilation = (1, 1),
        groups = 1,
        compute_dtype = torch.float32,
        resident_device = "cpu",
        logical_shape = (1, 1, 1, 1),
    )

    layer._apply(lambda tensor: tensor.to("meta"))

    assert layer.qweight.device == torch.device("cpu")
    assert layer.qweight.dtype is torch.uint8


def test_lazy_gguf_layer_norm_dequantizes_in_forward(monkeypatch):
    import core.inference.gguf_text_encoder as g

    seen: list[tuple[torch.Tensor, str, torch.dtype | None, tuple[int, ...] | None]] = []

    def fake_dequant(qbuffer, quant_type, *, dtype = None, logical_shape = None):
        seen.append((qbuffer.clone(), quant_type, dtype, logical_shape))
        return qbuffer.to(dtype = dtype).reshape(logical_shape)

    monkeypatch.setattr(g, "_dequantize_gguf_bytes", fake_dequant)
    layer = g.LazyGGUFLayerNorm(
        normalized_shape = (2,),
        eps = 0.0,
        compute_dtype = torch.float32,
        qweight = torch.tensor([2, 2], dtype = torch.uint8),
        weight_quant_type = "Q4",
        weight_logical_shape = (2,),
        bias = torch.tensor([1.0, -1.0]),
    )

    out = layer(torch.tensor([[1.0, 3.0]]))

    torch.testing.assert_close(out, torch.tensor([[-1.0, 1.0]]))
    assert seen[0][1:] == ("Q4", torch.float32, (2,))


def test_lazy_gguf_group_norm_can_keep_quantized_params_cpu_resident():
    import core.inference.gguf_text_encoder as g

    layer = g.LazyGGUFGroupNorm(
        num_groups = 1,
        num_channels = 2,
        eps = 1e-5,
        compute_dtype = torch.float32,
        resident_device = "cpu",
        qweight = torch.tensor([1, 1], dtype = torch.uint8),
        weight_quant_type = "Q4",
        weight_logical_shape = (2,),
        qbias = torch.tensor([0, 0], dtype = torch.uint8),
        bias_quant_type = "Q4",
        bias_logical_shape = (2,),
    )

    layer._apply(lambda tensor: tensor.to("meta"))

    assert layer.qweight.device == torch.device("cpu")
    assert layer.qbias.device == torch.device("cpu")
    assert layer.qweight.dtype is torch.uint8
    assert layer.qbias.dtype is torch.uint8


def test_patch_gguf_text_encoder_updates_lazy_quant_buffer_residency():
    import core.inference.gguf_text_encoder as g

    root = nn.Sequential(
        g.LazyGGUFLinear(
            torch.ones(2, 2, dtype = torch.uint8),
            "Q4",
            in_features = 2,
            out_features = 2,
            compute_dtype = torch.float32,
        ),
        g.LazyGGUFEmbedding(
            torch.ones(4, 2, dtype = torch.uint8),
            "Q4",
            num_embeddings = 4,
            embedding_dim = 2,
            compute_dtype = torch.float32,
        ),
        g.LazyGGUFLayerNorm(
            normalized_shape = (2,),
            eps = 1e-5,
            compute_dtype = torch.float32,
            qbias = torch.ones(2, dtype = torch.uint8),
            bias_quant_type = "Q4",
            bias_logical_shape = (2,),
        ),
    )

    assert g.patch_gguf_text_encoder_for_resident_device(root, "cpu") == 3
    root._apply(lambda tensor: tensor.to("meta"))

    assert root[0].qweight.device == torch.device("cpu")
    assert root[1].qweight.device == torch.device("cpu")
    assert root[2].qbias.device == torch.device("cpu")


def test_patch_gguf_text_encoder_keeps_upstream_gguf_parameter_resident():
    import core.inference.gguf_text_encoder as g

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            return self

    class _FakeGGUFLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = _FakeGGUFParameter(
                torch.ones(2, 2, dtype = torch.float32),
                quant_type = "Q4_K_M",
            )
            self.forward_devices: list[torch.device] = []

        def forward_native(self, inputs):
            self.forward_devices.append(self.weight.device)
            return inputs

        def forward(self, inputs):
            return self.forward_native(inputs)

    root = nn.Sequential(_FakeGGUFLinear())
    layer = root[0]

    assert g.patch_gguf_text_encoder_for_resident_device(root, "cpu") == 1
    assert layer.weight.device == torch.device("cpu")
    assert layer.weight.quant_type == "Q4_K_M"

    root.to(dtype = torch.float64)

    assert layer.weight.device == torch.device("cpu")
    assert layer.weight.dtype == torch.float32
    assert layer.weight.quant_type == "Q4_K_M"
    out = root(torch.ones(1, 2, dtype = torch.float32))
    assert out.shape == (1, 2)
    assert layer.forward_devices == [torch.device("cpu")]


def test_patch_gguf_text_encoder_refreshes_cpu_param_for_pinned_transfer(monkeypatch):
    import core.inference.gguf_text_encoder as g

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            return self

    class _FakeGGUFLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = _FakeGGUFParameter(
                torch.ones(2, 2, dtype = torch.float32),
                quant_type = "Q4_K_M",
            )

        def forward_native(self, inputs):
            return inputs

        def forward(self, inputs):
            return self.forward_native(inputs)

    pinned: list[torch.Tensor] = []

    def fake_pin(tensor, *, pin_memory = None):
        pinned.append(tensor)
        return tensor

    monkeypatch.setattr(
        g,
        "_pin_cpu_resident_gguf_tensors_enabled",
        lambda pin_memory = None: True,
    )
    monkeypatch.setattr(g, "_pin_cpu_tensor_for_transfer", fake_pin)
    root = nn.Sequential(_FakeGGUFLinear())

    assert g.patch_gguf_text_encoder_for_resident_device(root, "cpu") == 1

    assert pinned
    assert root[0].weight.device == torch.device("cpu")
    assert root[0].weight.quant_type == "Q4_K_M"


def test_patch_gguf_text_encoder_keeps_upstream_gguf_bias_resident():
    import core.inference.gguf_text_encoder as g

    class _FakeGGUFParameter(torch.nn.Parameter):
        def __new__(cls, data = None, requires_grad = False, quant_type = None):
            data = torch.empty(0) if data is None else data
            self = torch.Tensor._make_subclass(cls, data, requires_grad)
            self.quant_type = quant_type
            return self

    class _FakeGGUFLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = _FakeGGUFParameter(
                torch.ones(2, 2, dtype = torch.float32),
                quant_type = "QW",
            )
            self.bias = _FakeGGUFParameter(
                torch.ones(2, dtype = torch.float32),
                quant_type = "QB",
            )
            self.forward_devices: list[tuple[torch.device, torch.device]] = []

        def forward_native(self, inputs):
            self.forward_devices.append((self.weight.device, self.bias.device))
            return inputs

        def forward(self, inputs):
            return self.forward_native(inputs)

    root = nn.Sequential(_FakeGGUFLinear())
    layer = root[0]

    assert g.patch_gguf_text_encoder_for_resident_device(root, "cpu") == 1
    root.to(dtype = torch.float64)

    assert layer.weight.device == torch.device("cpu")
    assert layer.bias.device == torch.device("cpu")
    assert layer.weight.dtype == torch.float32
    assert layer.bias.dtype == torch.float32
    assert layer.weight.quant_type == "QW"
    assert layer.bias.quant_type == "QB"

    out = root(torch.ones(1, 2, dtype = torch.float32))

    assert out.shape == (1, 2)
    assert layer.forward_devices == [(torch.device("cpu"), torch.device("cpu"))]


def test_lazy_gguf_requires_gguf_dependency(monkeypatch):
    import builtins
    import core.inference.gguf_text_encoder as g

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "gguf":
            raise ModuleNotFoundError("No module named 'gguf'", name = "gguf")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match = "requires the gguf package"):
        g._require_gguf()
