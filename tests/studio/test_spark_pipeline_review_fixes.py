# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Layer-split stages that must compute what the full model computes, and shard loading."""

from __future__ import annotations

import asyncio
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
_POSIX_ONLY = pytest.mark.skipif(sys.platform == "win32", reason = "runs the peer's bash script")


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline():
    return _load("spark_pipeline_for_review_fixes", "studio/spark_pipeline.py")


@pytest.fixture
def cluster():
    return _load("spark_cluster_for_review_fixes", "studio/spark_cluster.py")


@pytest.fixture
def lb():
    return _load("spark_lb_for_review_fixes", "studio/spark_lb.py")


_TINY = dict(
    vocab_size = 64,
    hidden_size = 32,
    intermediate_size = 64,
    num_hidden_layers = 2,
    num_attention_heads = 4,
    num_key_value_heads = 2,
    max_position_embeddings = 64,
    use_cache = False,
)


def _stage_delta(pipeline, torch, model) -> float:
    model.eval()
    owner, layers = pipeline.find_layers(model)
    stage = pipeline.stage_module_cls()(
        model,
        owner,
        list(range(len(layers))),
        is_first = True,
        is_last = True,
        grad_checkpoint = False,
    ).eval()
    torch.manual_seed(1)
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        return (stage(ids) - model(ids).logits).abs().max().item()


def test_granite_stage_applies_the_embedding_multiplier(pipeline) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "GraniteForCausalLM"):
        pytest.skip("no Granite in this transformers")
    config = transformers.GraniteConfig(
        **_TINY,
        embedding_multiplier = 12.0,
        residual_multiplier = 0.22,
        logits_scaling = 16.0,
        attention_multiplier = 0.0078125,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    assert _stage_delta(pipeline, torch, transformers.GraniteForCausalLM(config)) == 0.0


def test_qwen3_5_stage_feeds_rotary_three_position_axes(pipeline) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Qwen3_5ForCausalLM"):
        pytest.skip("no Qwen3.5 in this transformers")
    config = transformers.Qwen3_5TextConfig(
        **_TINY,
        head_dim = 8,
        linear_key_head_dim = 8,
        linear_value_head_dim = 8,
        linear_num_key_heads = 2,
        linear_num_value_heads = 4,
        # Full attention only: the rotary is what failed, and the test harness's CUDA spoof would
        # route a linear-attention layer to a GPU kernel.
        layer_types = ["full_attention", "full_attention"],
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    assert _stage_delta(pipeline, torch, transformers.Qwen3_5ForCausalLM(config)) == 0.0


@pytest.mark.parametrize("arch", ["Llama", "Gemma"])
def test_ordinary_rope_and_gemma_scaling_are_unchanged(pipeline, arch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = getattr(transformers, f"{arch}Config")(**_TINY, head_dim = 8)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = getattr(transformers, f"{arch}ForCausalLM")(config)
    assert _stage_delta(pipeline, torch, model) == 0.0


def _llama_checkpoint(
    transformers,
    out: Path,
    quant = None,
) -> str:
    config = transformers.LlamaConfig(**_TINY)
    transformers.LlamaForCausalLM(config).save_pretrained(str(out), safe_serialization = True)
    if quant is not None:
        cfg = json.loads((out / "config.json").read_text())
        cfg["quantization_config"] = quant
        (out / "config.json").write_text(json.dumps(cfg))
    return str(out)


def test_shard_load_refuses_a_quantized_checkpoint(pipeline, tmp_path) -> None:
    """The stage is built as plain modules from the config, so an FP8 weight was cast to bf16 with its
    scale dropped: every layer silently wrong, and the meta check passed."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import load_file, save_file

    saved = Path(_llama_checkpoint(transformers, tmp_path / "fp8"))
    shard = next(saved.glob("*.safetensors"))
    tensors = load_file(str(shard))
    key = "model.layers.0.self_attn.q_proj.weight"
    tensors[key] = tensors[key].to(torch.float8_e4m3fn)
    tensors[key.replace("weight", "weight_scale_inv")] = torch.ones(1, 1)
    save_file(tensors, str(shard), metadata = {"format": "pt"})
    with pytest.raises(RuntimeError, match = "quantized"):
        pipeline.build_stage_model(
            str(saved),
            rank = 0,
            world = 2,
            device = "cpu",
            shard_load = True,
            dtype = torch.bfloat16,
            log = lambda *a, **k: None,
        )

    declared = Path(
        _llama_checkpoint(
            transformers,
            tmp_path / "declared",
            quant = {"quant_method": "fp8", "weight_block_size": [128, 128]},
        )
    )
    with pytest.raises(RuntimeError, match = "fp8"):
        pipeline.build_stage_model(
            str(declared),
            rank = 0,
            world = 2,
            device = "cpu",
            shard_load = True,
            dtype = torch.bfloat16,
            log = lambda *a, **k: None,
        )


def test_shard_load_keeps_integer_buffers_integer(pipeline, tmp_path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(4, 4))
            self.register_buffer("ids", torch.arange(4, dtype = torch.int64))

    save_file(
        {"w": torch.randn(4, 4), "ids": torch.arange(4, dtype = torch.int64) + 3},
        str(tmp_path / "model.safetensors"),
    )
    model = Toy()
    pipeline._materialise(model, str(tmp_path), None, "cpu", torch.bfloat16, lambda *a: None)
    assert model.w.dtype == torch.bfloat16
    assert model.ids.dtype == torch.int64 and model.ids.tolist() == [3, 4, 5, 6]


def _tiny_gpt2(pipeline, **overrides):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.GPT2Config(
        n_layer = 1, n_embd = 16, n_head = 2, n_positions = 32, vocab_size = 64, **overrides
    )
    model = transformers.GPT2LMHeadModel(config)
    return pipeline.apply_lora(model, 4), config


def test_ddp_flags_unused_parameters_for_optional_cross_attention(pipeline):
    """Regression: LoRA on GPT-2's cross-attention never runs on a text batch, so DDP must not
    expect a gradient for it; only MoE turned find_unused_parameters on before."""
    model, config = _tiny_gpt2(pipeline, add_cross_attention = True)
    assert any("crossattention" in n and "lora_" in n for n, _ in model.named_parameters())
    assert pipeline.ddp_unused_parameters_reason(model, config)


def test_ddp_flags_unused_parameters_for_layer_drop_and_mtp_modules(pipeline):
    from types import SimpleNamespace

    import torch

    dense = torch.nn.Module()
    dense.q_proj = torch.nn.Linear(2, 2)
    assert pipeline.ddp_unused_parameters_reason(dense, SimpleNamespace(layerdrop = 0.1))
    assert pipeline.ddp_unused_parameters_reason(dense, SimpleNamespace(layerdrop = 0.0)) is None
    with_mtp = torch.nn.Module()
    with_mtp.mtp = torch.nn.Linear(2, 2)
    assert pipeline.ddp_unused_parameters_reason(with_mtp, SimpleNamespace())
    with_mtp.mtp.requires_grad_(False)
    assert pipeline.ddp_unused_parameters_reason(with_mtp, SimpleNamespace()) is None


def test_ddp_keeps_the_fast_path_for_a_dense_text_model(pipeline):
    model, config = _tiny_gpt2(pipeline)
    assert pipeline.ddp_unused_parameters_reason(model, config) is None


def test_ddp_unused_parameter_check_reuses_unsloth_defaults(pipeline):
    imported = "unsloth" in sys.modules
    defaults = pipeline._unsloth_ddp_defaults()
    assert defaults is not None and callable(defaults._config_may_skip)
    # Loaded by path: importing the unsloth package would patch transformers in the trainer.
    assert ("unsloth" in sys.modules) == imported
