# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tokenizer-based audio_type classification, with no heavy imports.

Lives directly under ``utils`` so the hub cache scanner can classify a snapshot
without pulling in ``utils/models/__init__.py`` and the model-config stack.
``utils.models.model_config`` re-exports these for its Hub-fetching probe, so
the patterns have exactly one definition.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

VALID_AUDIO_TYPES = ("snac", "csm", "bicodec", "dac", "whisper", "audio_vlm")

# Emit speech; a chat turn sent to one comes back as audio, never as text.
TTS_AUDIO_TYPES = frozenset({"snac", "csm", "bicodec", "dac"})


def _count_prefix_exceeds(tokens, prefix: str, threshold: int) -> bool:
    """Whether more than ``threshold`` tokens start with ``prefix``.

    Equivalent to ``sum(...) > threshold`` but stops at the answer. Summing counted every
    one of Orpheus's 28k codes to decide a question settled by the first 10,001.
    """
    count = 0
    for token in tokens:
        if token.startswith(prefix):
            count += 1
            if count > threshold:
                return True
    return False


# Tokenizer token patterns → audio_type (all 6 types from tokenizer_config.json).
# ORDER MATTERS: first match wins, so the specific codec fingerprints go before the
# generic audio_vlm marker. Orpheus carries 28k <custom_token_N> SNAC codes AND a
# stray <|audio|>; audio_vlm first typed it as audio-input, leaving is_audio False.
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
    # Generic, so last. Gemma 3n <audio_soft_token>; Gemma 4 <|audio|> (not csm's
    # <|AUDIO|>). Neither carries a codebook, so nothing above shadows them.
    "audio_vlm": lambda tokens: "<audio_soft_token>" in tokens or "<|audio|>" in tokens,
}

# Every substring a pattern above needs. A tokenizer_config whose text contains NONE of
# these cannot match any pattern, whatever it holds, so the answer is settled without
# parsing it. That matters because an ordinary text checkpoint carries a large
# tokenizer_config and json.loads of it was the bulk of a cold /loras scan.
# MUST cover every pattern: AUDIO_TOKEN_PATTERNS is lambdas, so this cannot be derived
# from them, and a codec added there without its marker here would silently stop being
# detected. test_audio_token_detection.py fails if the two drift.
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

# A codebook tokenizer runs to a few MB; anything past this is not one of ours, and the
# inventory scan reads these on the event loop.
_MAX_TOKENIZER_CONFIG_BYTES = 32 * 1024 * 1024


def may_hold_audio_tokens(raw: str) -> bool:
    """Whether a tokenizer_config's raw text is worth parsing.

    Conservative: a false True only costs the parse that would have happened anyway.
    A false False would misclassify, which is why the markers are pinned by a test.
    """
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
    """True for a speech-emitting codec. audio_vlm is deliberately absent: Gemma 3n
    takes audio in and answers in text, so it is an ordinary chat model."""
    return audio_type in TTS_AUDIO_TYPES


def detect_local_tts_audio_type(directory) -> Optional[str]:
    """The TTS codec a downloaded model directory fingerprints, or None.

    Local files only, so it never reaches the Hub. Whisper and audio_vlm answer None:
    this exists to keep speech-emitting models out of chat, and both of those chat.
    """
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
