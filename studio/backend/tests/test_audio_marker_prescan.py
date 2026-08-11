# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settling a tokenizer_config without parsing it.

An audio pattern can only match if its marker text appears in the file at all, so the raw
text is scanned first and an ordinary text checkpoint never reaches json.loads. Ordinary
checkpoints carry big tokenizers: unsloth/gemma-3-270m-it is 1.16 MB with 6415 added
tokens, and parsing it was most of the cost of answering "is this an audio model".
"""

from __future__ import annotations

import json

from utils.models.model_config import (
    _AUDIO_TOKEN_MARKERS,
    _AUDIO_TOKEN_PATTERNS,
    _may_hold_audio_tokens,
)


def _classify(tokens: list[str]) -> str | None:
    for audio_type, check in _AUDIO_TOKEN_PATTERNS.items():
        if check(tokens):
            return audio_type
    return None


def test_every_pattern_has_a_marker():
    """The markers cannot be derived from the patterns, which are lambdas, so a codec
    added there without a marker here would silently stop being detected."""
    # Fails when a codec is added, which is the point: add its marker too.
    assert set(_AUDIO_TOKEN_PATTERNS) == {
        "csm", "whisper", "bicodec", "dac", "snac", "audio_vlm",
    }

    samples = {
        "csm": ["<|AUDIO|>", "<|audio_eos|>"],
        "whisper": ["<|startoftranscript|>"],
        "bicodec": ["<|bicodec_semantic_0|>"],
        "dac": ["<|audio_start|>", "<|audio_end|>", "<|text_start|>", "<|text_end|>"],
        "snac": [f"<custom_token_{i}>" for i in range(10001)],
        "audio_vlm": ["<audio_soft_token>"],
    }
    for audio_type, tokens in samples.items():
        # Whatever each pattern matches, the prescan must let it through to the parse.
        assert _classify(tokens) == audio_type, audio_type
        assert _may_hold_audio_tokens(json.dumps(tokens)), audio_type
    assert _may_hold_audio_tokens(json.dumps(["<|image|>", "<|audio|>"]))
    assert len(set(_AUDIO_TOKEN_MARKERS)) == len(_AUDIO_TOKEN_MARKERS)


def test_an_ordinary_text_tokenizer_is_settled_without_a_parse(monkeypatch, tmp_path):
    from utils.models import model_config

    config = {
        "added_tokens_decoder": {
            str(i): {"content": f"<|extra_token_{i}|>", "special": True}
            for i in range(5000)
        }
    }
    checkpoint = tmp_path / "run"
    checkpoint.mkdir()
    (checkpoint / "tokenizer_config.json").write_text(json.dumps(config))

    parsed = []
    real_loads = model_config.json.loads
    monkeypatch.setattr(
        model_config.json, "loads",
        lambda raw, *a, **kw: (parsed.append(len(raw)), real_loads(raw, *a, **kw))[1],
    )

    result, definitive = model_config._detect_audio_from_tokenizer(
        str(checkpoint), local_files_only = True
    )
    assert result is None
    # Read successfully, so "not audio" is a definitive answer, not an unknown.
    assert definitive is True
    assert parsed == [], parsed


def test_an_audio_tokenizer_is_still_detected(tmp_path):
    from utils.models import model_config

    checkpoint = tmp_path / "orpheus"
    checkpoint.mkdir()
    (checkpoint / "tokenizer_config.json").write_text(json.dumps({
        "added_tokens_decoder": {
            str(i): {"content": f"<custom_token_{i}>"} for i in range(10500)
        }
    }))
    assert model_config._detect_audio_from_tokenizer(
        str(checkpoint), local_files_only = True
    ) == ("snac", True)


def test_a_half_written_tokenizer_stays_unknown(tmp_path):
    """A training run part-way through writing its tokenizer must not become a definitive
    "not audio" and be cached for the life of the process. It stays unknown, exactly as it
    did when json.loads raised on the truncated text."""
    from utils.models import model_config

    checkpoint = tmp_path / "mid_write"
    checkpoint.mkdir()
    whole = json.dumps({"added_tokens_decoder": {"0": {"content": "<|plain|>"}}})
    (checkpoint / "tokenizer_config.json").write_text(whole[: len(whole) // 2])

    assert model_config._detect_audio_from_tokenizer(
        str(checkpoint), local_files_only = True
    ) == (None, False)

    (checkpoint / "tokenizer_config.json").write_text(whole)
    assert model_config._detect_audio_from_tokenizer(
        str(checkpoint), local_files_only = True
    ) == (None, True)
