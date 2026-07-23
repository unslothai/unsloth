# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for tokenizer-based audio_type detection, covering Gemma 3n
(<audio_soft_token>) and Gemma 4 (<|audio|>) audio-input tokens."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from transformers import AutoConfig

from utils.models.model_config import (
    ModelConfig,
    _AUDIO_TOKEN_PATTERNS,
    _classify_audio_capability,
    is_audio_input_type,
)


def _classify(tokens: list[str]) -> str | None:
    """Mirror _check_token_patterns: first match in dict order wins."""
    for audio_type, check in _AUDIO_TOKEN_PATTERNS.items():
        if check(tokens):
            return audio_type
    return None


def test_gemma3n_audio_soft_token_is_audio_vlm():
    assert _classify(["<bos>", "<audio_soft_token>", "<image_soft_token>"]) == "audio_vlm"


def test_gemma4_pipe_audio_token_is_audio_vlm():
    # Gemma 4 uses <|audio|> (and <|image|>) instead of *_soft_token.
    assert _classify(["<bos>", "<|image|>", "<|audio|>"]) == "audio_vlm"


def test_csm_uppercase_audio_not_classified_as_audio_vlm():
    # csm uses uppercase <|AUDIO|> + <|audio_eos|>; must stay csm, not audio_vlm.
    tokens = ["<|AUDIO|>", "<|audio_eos|>"]
    assert _classify(tokens) == "csm"


def test_audio_vlm_and_whisper_accept_audio_input():
    assert is_audio_input_type("audio_vlm") is True
    assert is_audio_input_type("whisper") is True
    assert is_audio_input_type("snac") is False
    assert is_audio_input_type(None) is False


def test_non_audio_tokens_classify_none():
    assert _classify(["<bos>", "<eos>", "<pad>"]) is None


@pytest.mark.parametrize(
    ("model_type", "audio_type", "expected"),
    [
        *(
            (name, "csm", ("audio_vlm", True, True))
            for name in ("qwen2_audio", "qwen2_5_omni", "qwen3_omni_moe", "granite_speech")
        ),
        ("qwen2_audio_encoder", None, (None, False, False)),
        ("whisper", None, (None, True, False)),
        ("llama", None, (None, False, True)),
    ],
)
def test_structured_model_type_controls_chat_capability(model_type, audio_type, expected):
    with patch(
        "utils.models.model_config.load_model_config",
        return_value = AutoConfig.for_model(model_type),
    ):
        assert _classify_audio_capability("org/model", audio_type) == expected


@pytest.mark.parametrize("model_type", ["qwen3_asr", "whisper", "jukebox", "speech_to_text_2"])
def test_unknown_non_chat_config_uses_raw_model_type(tmp_path, model_type):
    (tmp_path / "config.json").write_text(f'{{"model_type": "{model_type}"}}')
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"gguf")
    with patch("utils.models.model_config.load_model_config", side_effect = ValueError("unknown")):
        config = ModelConfig.from_identifier(str(tmp_path))

    assert config.is_chat_capable is False
    assert config.has_audio_input is (model_type in {"qwen3_asr", "whisper"})


def test_unknown_remote_non_chat_config_uses_active_cache(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model_type": "qwen3_asr"}')
    selected_cache = str(tmp_path / "hub")

    def cached_config(
        *_args,
        cache_dir = None,
        **_kwargs,
    ):
        assert cache_dir == selected_cache
        return str(config_path)

    with (
        patch("utils.models.model_config.load_model_config", side_effect = ValueError("unknown")),
        patch("utils.models.model_config.active_hf_hub_cache", return_value = selected_cache),
        patch("huggingface_hub.hf_hub_download", side_effect = cached_config),
    ):
        assert _classify_audio_capability("org/asr", None) == (None, True, False)
