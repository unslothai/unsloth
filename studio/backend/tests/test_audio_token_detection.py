# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for tokenizer-based audio_type detection, covering Gemma 3n
(<audio_soft_token>) and Gemma 4 (<|audio|>) audio-input tokens."""

from __future__ import annotations

from utils.models.model_config import _AUDIO_TOKEN_PATTERNS, is_audio_input_type


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


def test_orpheus_snac_codebook_beats_a_stray_audio_marker():
    """Orpheus ships 28k <custom_token_N> SNAC codes AND a lone <|audio|>.

    audio_vlm used to be tested first and won, so the TTS model came back as an
    audio-INPUT model: is_audio stayed False (mlx_inference excludes audio_vlm)
    and the Audio page refused it with "not a TTS audio model".
    """
    tokens = ["<|audio|>"] + [f"<custom_token_{i}>" for i in range(28683)]
    assert _classify(tokens) == "snac"
    assert is_audio_input_type(_classify(tokens)) is False


def test_a_codec_family_is_not_shadowed_by_a_stray_audio_marker():
    """The same precedence has to hold for every output codec, not just snac."""
    assert _classify(["<|audio|>", "<|bicodec_semantic_0|>"]) == "bicodec"
    assert (
        _classify(
            [
                "<|audio|>",
                "<|audio_start|>",
                "<|audio_end|>",
                "<|text_start|>",
                "<|text_end|>",
            ]
        )
        == "dac"
    )
