# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Moonshine's LM head is `proj_out`; untied on moonshine-streaming, so 4-bit loads quantized it and the
fused CE loss failed reading it as a plain weight (`mat1 and mat2 shapes cannot be multiplied`)."""

import ast
from pathlib import Path

import pytest

VISION_PATH = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "vision.py"


def _architecture_skip_modules():
    tree = ast.parse(VISION_PATH.read_text(encoding = "utf-8"))
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_architecture_skip_modules"
    )
    ns = {}
    exec(compile(ast.Module([node], []), str(VISION_PATH), "exec"), ns)
    return ns["_architecture_skip_modules"]


def test_moonshine_keeps_its_lm_head_unquantized():
    skip = _architecture_skip_modules()
    for model_types in (["moonshine_streaming", "moonshine_streaming_encoder"], ["moonshine"]):
        assert "proj_out" in skip(model_types), model_types
    for model_types in (["whisper"], ["llama"], ["gemma3n"], [], None):
        assert "proj_out" not in skip(model_types), model_types


def test_only_the_head_matches_the_transformers_skip_matcher():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    utils = pytest.importorskip("transformers.quantizers.quantizers_utils")
    should_convert = getattr(utils, "should_convert_module", None)
    config_class = getattr(transformers, "MoonshineStreamingConfig", None)
    if should_convert is None or config_class is None:
        pytest.skip("needs transformers 5 with Moonshine Streaming")
    config = config_class(
        encoder_config = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
        },
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 2,
        vocab_size = 128,
    )
    with torch.device("meta"):
        model = transformers.AutoModelForSpeechSeq2Seq.from_config(config)
    skip = _architecture_skip_modules()(["moonshine_streaming"])
    kept = [
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and not should_convert(n, skip)
    ]
    assert kept == ["proj_out"]
