# SPDX-License-Identifier: AGPL-3.0-only
"""A 16-bit load (`load_in_16bit=True`) of a compressed-tensors `pack-quantized` INT4 checkpoint (Kimi-K2.7-Code)
kept compressed-tensors' packed Linear (weight_packed, no `.weight`), so `get_peft_model` failed with
AttributeError: 'Linear' object has no attribute 'weight'. It now decompresses to 16-bit like the FP8 path."""

import json
import os

import pytest
import torch

ct = pytest.importorskip("compressed_tensors")


def _has_cuda():
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def _tiny_int4_checkpoint(path):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationConfig,
        QuantizationScheme,
        apply_quantization_config,
    )
    from compressed_tensors.compressors import ModelCompressor
    from tokenizers import Tokenizer, models, pre_tokenizers

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size = 256,
        hidden_size = 64,
        intermediate_size = 128,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        max_position_embeddings = 128,
    )
    model = LlamaForCausalLM(cfg).to(torch.bfloat16)
    qc = QuantizationConfig(
        config_groups = {
            "group_0": QuantizationScheme(
                targets = ["Linear"],
                weights = QuantizationArgs(
                    num_bits = 4, type = "int", symmetric = True, strategy = "group", group_size = 32
                ),
            )
        },
        ignore = ["lm_head"],
        format = "pack-quantized",
    )
    apply_quantization_config(model, qc)
    for _, m in model.named_modules():
        if hasattr(m, "weight_scale"):
            w = m.weight.data.float().view(m.weight.shape[0], -1, 32)
            m.weight_scale.data = (w.abs().amax(-1) / 7).clamp_min(1e-6).to(m.weight_scale.dtype)
    comp = ModelCompressor.from_pretrained_model(model, quantization_format = "pack-quantized")
    comp.compress_model(model)
    model.save_pretrained(path)
    comp.update_config(path)
    vocab = {f"t{i}": i for i in range(256)}
    tok = Tokenizer(models.WordLevel(vocab = vocab, unk_token = "t0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object = tok, unk_token = "t0", pad_token = "t1", eos_token = "t2"
    ).save_pretrained(path)
    assert (
        json.load(open(os.path.join(path, "config.json")))["quantization_config"]["format"]
        == "pack-quantized"
    )


def test_packed_int_format_is_detected():
    from unsloth.models import vision

    assert vision._compressed_tensors_packed_int({"format": "pack-quantized"})
    assert not vision._compressed_tensors_packed_int({"format": "mxfp4-pack-quantized"})
    assert not vision._compressed_tensors_packed_int({"format": "float-quantized"})
    cfg = vision._compressed_tensors_decompress_config()
    assert getattr(cfg, "dequantize", False) or getattr(cfg, "run_compressed", True) is False


@pytest.mark.skipif(not _has_cuda(), reason = "FastModel loads need an accelerator")
def test_int4_checkpoint_16bit_load_trains_lora(tmp_path):
    _tiny_int4_checkpoint(str(tmp_path))
    from unsloth import FastModel

    model, tok = FastModel.from_pretrained(
        str(tmp_path),
        max_seq_length = 64,
        load_in_4bit = False,
        load_in_16bit = True,
        dtype = torch.bfloat16,
    )
    q = model.model.layers[0].self_attn.q_proj
    assert getattr(q, "weight", None) is not None and q.weight.dtype == torch.bfloat16
    model = FastModel.get_peft_model(
        model,
        r = 4,
        lora_alpha = 4,
        target_modules = ["q_proj", "down_proj"],
        use_gradient_checkpointing = False,
    )
    ids = torch.randint(3, 256, (1, 16), device = model.get_input_embeddings().weight.device)
    model.train()
    model(input_ids = ids, labels = ids).loss.backward()
    grads = [p.grad for n, p in model.named_parameters() if "lora_B" in n]
    assert grads and all(g is not None and g.abs().sum() > 0 for g in grads)
