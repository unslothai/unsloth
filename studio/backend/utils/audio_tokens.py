# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tokenizer-based audio_type classification.

Directly under ``utils`` so the cache scanner can classify a snapshot without
dragging in ``utils/models/__init__.py`` and the model-config stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

NATIVE_OUTPUT_AUDIO_TYPES = frozenset(
    {
        "higgs_tts2",
        "moss_tts_local",
        "moss_tts_nano",
        "higgs_tts3",
        "minimax_music3",
    }
)

VALID_AUDIO_TYPES = (
    "snac",
    "csm",
    "bicodec",
    "dac",
    *sorted(NATIVE_OUTPUT_AUDIO_TYPES),
    "whisper",
    "audio_vlm",
)

# Emit speech; a chat turn sent to one comes back as audio, never as text.
TTS_AUDIO_TYPES = frozenset({"snac", "csm", "bicodec", "dac"})
OUTPUT_AUDIO_TYPES = TTS_AUDIO_TYPES | NATIVE_OUTPUT_AUDIO_TYPES


def _count_prefix_exceeds(tokens, prefix: str, threshold: int) -> bool:
    """``sum(...) > threshold``, but stopping at the answer: summing counted all 28k of
    Orpheus's codes to settle a question the first 10,001 decide."""
    count = 0
    for token in tokens:
        if token.startswith(prefix):
            count += 1
            if count > threshold:
                return True
    return False


# ORDER MATTERS: first match wins, so codec fingerprints precede the generic audio_vlm
# marker. Orpheus carries 28k <custom_token_N> SNAC codes AND a stray <|audio|>, and
# audio_vlm first typed it as audio-input.
AUDIO_TOKEN_PATTERNS = {
    "csm": lambda tokens: "<|AUDIO|>" in tokens and "<|audio_eos|>" in tokens,
    "whisper": lambda tokens: "<|startoftranscript|>" in tokens,
    "bicodec": lambda tokens: any(t.startswith("<|bicodec_") for t in tokens),
    "dac": lambda tokens: (
        "<|audio_start|>" in tokens
        and "<|audio_end|>" in tokens
        and "<|text_start|>" in tokens
        and "<|text_end|>" in tokens
    ),
    "snac": lambda tokens: _count_prefix_exceeds(tokens, "<custom_token_", 10000),
    # Generic, so last. Gemma 3n <audio_soft_token>; Gemma 4 <|audio|>, not csm's <|AUDIO|>.
    "audio_vlm": lambda tokens: "<audio_soft_token>" in tokens or "<|audio|>" in tokens,
}

# Every substring a pattern needs, so text holding none of them is settled without a
# parse -- json.loads of an ordinary large tokenizer_config was the bulk of a cold /loras
# scan. The patterns are lambdas, so this cannot be derived from them; a codec added
# there without its marker here would silently stop being detected, and
# test_audio_token_detection.py fails when the two drift.
AUDIO_TOKEN_MARKERS = (
    "<|AUDIO|>",  # csm
    "<|startoftranscript|>",  # whisper
    "<|bicodec_",  # bicodec
    "<|audio_start|>",  # dac
    "<custom_token_",  # snac
    "<audio_soft_token>",  # audio_vlm (Gemma 3n)
    "<|audio|>",  # audio_vlm (Gemma 4)
)

AUDIO_TOKENIZER_CONFIG_PATHS = (
    "tokenizer_config.json",
    "LLM/tokenizer_config.json",
)

# A codebook tokenizer runs to a few MB, and the inventory scan reads one per repo.
_MAX_TOKENIZER_CONFIG_BYTES = 32 * 1024 * 1024


def may_hold_audio_tokens(raw: str) -> bool:
    """Whether a tokenizer_config's raw text is worth parsing. A false True costs only
    the parse that would have happened anyway; a false False misclassifies."""
    return any(marker in raw for marker in AUDIO_TOKEN_MARKERS)


def classify_audio_tokens(tok_config: dict) -> Optional[str]:
    """The audio_type a parsed tokenizer_config fingerprints, or None."""
    added = tok_config.get("added_tokens_decoder", {})
    if not added:
        return None
    token_contents = [value.get("content", "") for value in added.values()]
    for audio_type, check_fn in AUDIO_TOKEN_PATTERNS.items():
        if check_fn(token_contents):
            return audio_type
    return None


def is_audio_input_type(audio_type: Optional[str]) -> bool:
    """True if an audio_type accepts audio input: whisper (ASR), audio_vlm (Gemma3n)."""
    return audio_type in ("whisper", "audio_vlm")


def is_tts_audio_type(audio_type: Optional[str]) -> bool:
    """True for a speech-emitting codec. audio_vlm is absent on purpose: Gemma 3n takes
    audio in and answers in text."""
    return audio_type in TTS_AUDIO_TYPES


def is_output_audio_type(audio_type: Optional[str]) -> bool:
    """True for a model that emits audio instead of a text chat response."""
    return audio_type in OUTPUT_AUDIO_TYPES


def detect_local_tts_audio_type(directory) -> Optional[str]:
    """The TTS codec a downloaded model directory fingerprints, or None. Local files
    only. Whisper and audio_vlm answer None: both of those chat."""
    try:
        root = Path(directory)
        if not root.is_dir():
            return None
    except OSError:
        return None
    for tok_path in AUDIO_TOKENIZER_CONFIG_PATHS:
        tok_file = root / tok_path
        try:
            if not tok_file.is_file() or tok_file.stat().st_size > _MAX_TOKENIZER_CONFIG_BYTES:
                continue
            raw = tok_file.read_text(encoding = "utf-8-sig")
            if not may_hold_audio_tokens(raw):
                continue
            audio_type = classify_audio_tokens(json.loads(raw))
        except Exception:
            continue
        if is_tts_audio_type(audio_type):
            return audio_type
    return None
