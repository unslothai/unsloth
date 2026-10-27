# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Qwen3-ASR checkpoints in the qwen-asr package layout (Qwen/Qwen3-ASR-0.6B, -1.7B) load their real weights."""

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"
LOADER_PATH = REPO_ROOT / "unsloth" / "models" / "loader.py"


def _helpers():
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    wanted = {"_QWEN3_ASR_PACKAGE_KEY_MAPPING", "_convert_qwen3_asr_package_config"}
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in wanted)
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
    ]
    ns = {"re": re}
    exec(compile(ast.Module(nodes, []), str(UTILS_PATH), "exec"), ns)
    return ns["_convert_qwen3_asr_package_config"], ns["_QWEN3_ASR_PACKAGE_KEY_MAPPING"]


def _tiny_native():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    Config = getattr(transformers, "Qwen3ASRConfig", None)
    Model = getattr(transformers, "Qwen3ASRForConditionalGeneration", None)
    if Config is None or Model is None:
        pytest.skip("transformers has no Qwen3-ASR")
    audio = dict(
        d_model = 32,
        encoder_layers = 1,
        encoder_attention_heads = 2,
        encoder_ffn_dim = 64,
        output_dim = 32,
        downsample_hidden_size = 16,
        num_mel_bins = 128,
    )
    text = dict(
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        num_key_value_heads = 1,
        head_dim = 16,
        vocab_size = 256,
    )
    config = Config(
        audio_config = audio,
        text_config = text,
        audio_token_id = 5,
        pad_token_id = 0,
        eos_token_id = 1,
        tie_word_embeddings = True,
    )
    torch.manual_seed(0)
    return Model(config).eval(), audio, text


def _package_layout(model, audio, text, path):
    """Re-save a native model the way the qwen-asr package does."""
    import json
    from safetensors.torch import save_file

    inverse = [
        (r"^model\.multi_modal_projector\.linear_1\.", "thinker.audio_tower.proj1."),
        (r"^model\.multi_modal_projector\.linear_2\.", "thinker.audio_tower.proj2."),
        (r"^model\.audio_tower\.", "thinker.audio_tower."),
        (r"^model\.language_model\.", "thinker.model."),
        (r"^lm_head\.", "thinker.lm_head."),
    ]
    out = {}
    for key, value in model.state_dict().items():
        for pattern, repl in inverse:
            if re.match(pattern, key):
                key = re.sub(pattern, repl, key, count = 1)
                break
        out[key] = value.contiguous().clone()
    save_file(out, str(path / "model.safetensors"), metadata = {"format": "pt"})
    thinker = {
        "model_type": "qwen3_asr",
        "audio_config": audio,
        "text_config": text,
        "audio_token_id": 5,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "tie_word_embeddings": True,
    }
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ASRForConditionalGeneration"],
                "model_type": "qwen3_asr",
                "thinker_config": thinker,
            }
        )
    )


def test_package_layout_loads_the_checkpoint_weights(tmp_path):
    import torch
    from transformers import AutoConfig, Qwen3ASRForConditionalGeneration

    convert, mapping = _helpers()
    native, audio, text = _tiny_native()
    _package_layout(native, audio, text, tmp_path)

    plain = AutoConfig.from_pretrained(tmp_path)
    assert (
        plain.text_config.hidden_size != text["hidden_size"]
    )  # the defect: sizes fall back to defaults

    config, key_mapping = convert(plain)
    assert key_mapping == mapping
    assert config.text_config.hidden_size == text["hidden_size"]
    assert config.audio_config.output_dim == audio["output_dim"]

    loaded = Qwen3ASRForConditionalGeneration.from_pretrained(
        tmp_path, config = config, key_mapping = key_mapping
    )
    want, got = native.state_dict(), loaded.state_dict()
    assert set(want) == set(got)
    assert all(torch.equal(want[k], got[k]) for k in want)


def test_native_and_other_configs_pass_through():
    convert, _ = _helpers()
    native, _, _ = _tiny_native()
    assert convert(native.config) == (native.config, None)
    converted, mapping = convert(native.config)
    assert convert(converted)[1] is None  # idempotent

    class Other:
        model_type = "qwen2_audio"
        thinker_config = {"audio_config": {}, "text_config": {}}

    other = Other()
    assert convert(other) == (other, None)

    class Aligner:  # Qwen/Qwen3-ForcedAligner-0.6B: same layout, classifier head
        model_type = "qwen3_asr"
        thinker_config = {
            "model_type": "qwen3_forced_aligner",
            "audio_config": {},
            "text_config": {},
        }

    aligner = Aligner()
    assert convert(aligner) == (aligner, None)


def test_fast_model_applies_and_drops_the_mapping():
    source = LOADER_PATH.read_text(encoding = "utf-8")
    assert "_convert_qwen3_asr_package_config(model_config)" in source
    assert "if user_config is None and not fast_inference:" in source
    assert "_drop_text_only_key_mapping(model, _qwen3_asr_key_mapping)" in source
