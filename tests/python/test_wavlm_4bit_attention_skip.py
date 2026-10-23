# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""WavLM's attention passes q/k/v/out_proj.weight to F.multi_head_attention_forward, so 4-bit loads
failed on the first step (`expecting query weights shape of (64, 64), but got torch.Size([2048, 1])`)."""

import ast
from pathlib import Path

import pytest

VISION_PATH = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "vision.py"
ATTENTION = {"attention.q_proj", "attention.k_proj", "attention.v_proj", "attention.out_proj"}


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


def test_wavlm_keeps_its_attention_projections_unquantized():
    skip = _architecture_skip_modules()
    assert ATTENTION <= set(skip(["wavlm"]))
    for model_types in (["wav2vec2"], ["hubert"], ["llama"], ["whisper"], [], None):
        assert not ATTENTION & set(skip(model_types)), model_types


def test_only_attention_projections_match_the_transformers_skip_matcher():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    utils = pytest.importorskip("transformers.quantizers.quantizers_utils")
    should_convert = getattr(utils, "should_convert_module", None)
    if should_convert is None:
        pytest.skip("transformers < 5 matches skip keys per path component")
    config = transformers.WavLMConfig(
        hidden_size = 32,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 64,
        conv_dim = (8, 8),
        conv_stride = (5, 2),
        conv_kernel = (10, 3),
        num_conv_pos_embeddings = 4,
        vocab_size = 16,
    )
    with torch.device("meta"):
        model = transformers.WavLMForCTC(config)
    skip = _architecture_skip_modules()(["wavlm"])
    kept = {
        ".".join(n.split(".")[-2:])
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear) and not should_convert(n, skip)
    }
    # Unsloth's alias-expanding matcher also keeps the attention's tiny gate Linear(head_dim, 8) in 16-bit.
    assert ATTENTION <= kept <= ATTENTION | {"attention.gru_rel_pos_linear"}


def test_a_bart_style_decoder_is_still_quantized():
    utils = pytest.importorskip("transformers.quantizers.quantizers_utils")
    should_convert = getattr(utils, "should_convert_module", None)
    if should_convert is None:
        pytest.skip("transformers < 5 matches skip keys per path component")
    skip = _architecture_skip_modules()(["speech-encoder-decoder", "wavlm", "bart"])
    for name in (
        "decoder.model.decoder.layers.0.self_attn.q_proj",
        "decoder.model.decoder.layers.0.encoder_attn.out_proj",
    ):
        assert should_convert(name, skip), name
    assert not should_convert("encoder.encoder.layers.0.attention.q_proj", skip)


def _transformers4_skips(name, keys):
    """transformers 4.52-4.57 `_replace_with_bnb_linear`: bare leaf name, an ancestor key, or the full path."""
    return name.rsplit(".", 1)[-1] in keys or any(k + "." in name or k == name for k in keys)


def test_transformers_4_matcher_keeps_wavlm_attention_and_a_bart_decoder_quantized():
    skip = _architecture_skip_modules()(["speech-encoder-decoder", "wavlm", "bart"])
    for proj in ("q_proj", "k_proj", "v_proj", "out_proj"):
        assert _transformers4_skips(f"wavlm.encoder.layers.0.attention.{proj}", skip), proj
    for name in (
        "wavlm.encoder.layers.0.feed_forward.intermediate_dense",
        "decoder.model.decoder.layers.0.self_attn.q_proj",
        "decoder.model.decoder.layers.0.encoder_attn.out_proj",
        "decoder.model.decoder.layers.0.fc1",
    ):
        assert not _transformers4_skips(name, skip), name
