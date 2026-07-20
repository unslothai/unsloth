# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for tokenizer-based audio_type detection, covering Gemma 3n
(<audio_soft_token>) and Gemma 4 (<|audio|>) audio-input tokens."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from transformers import AutoConfig

from utils.models.model_config import (
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
        *(
            (name, None, (None, False, False))
            for name in (
                "wav2vec2",
                "hubert",
                "speecht5",
                "bark",
                "clap",
                "encodec",
                "dac",
                "mimi",
                "xcodec",
            )
        ),
        ("llama", None, (None, False, True)),
    ],
)
def test_structured_model_type_controls_chat_capability(model_type, audio_type, expected):
    with patch(
        "utils.models.model_config.load_model_config",
        return_value = AutoConfig.for_model(model_type),
    ):
        assert _classify_audio_capability("org/model", audio_type) == expected


@pytest.mark.parametrize("audio_type", ["csm", "whisper", "snac"])
def test_audio_metadata_rejects_non_chat_model_when_config_is_missing(audio_type):
    with patch("utils.models.model_config.load_model_config", side_effect = OSError("missing")):
        assert _classify_audio_capability("org/model", audio_type)[2] is False
