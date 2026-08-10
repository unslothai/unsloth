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

    audio_vlm was tested first and won, so a TTS model came back as audio-INPUT:
    is_audio stayed False and the Audio page refused it.
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


class _Resp:
    def __init__(
        self,
        status_code: int,
        payload = None,
    ):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no body")
        return self._payload


def _detect_checked(
    monkeypatch,
    responses,
    model = "acme/tts-model",
):
    """Drive detect_audio_type_checked with a faked Hub, no local cache."""
    from utils.models import model_config as mc

    monkeypatch.setattr(mc, "_audio_detection_cache", {})
    monkeypatch.setattr(mc, "get_cache_path", lambda *a, **k: None)
    monkeypatch.setattr(mc, "_env_offline", lambda: False)

    import requests

    monkeypatch.setattr(requests, "get", lambda url, **kw: responses.pop(0))
    return mc.detect_audio_type_checked(model)


def test_a_gated_repo_is_not_reported_as_definitively_non_audio(monkeypatch):
    # 401 on every tokenizer_config path: nothing was read, so None means unknown.
    audio_type, definitive = _detect_checked(monkeypatch, [_Resp(401), _Resp(401)])
    assert audio_type is None
    assert definitive is False


def test_a_readable_repo_without_audio_tokens_is_definitive(monkeypatch):
    # 200 with a plain tokenizer, then a 404 for the LLM/ variant: a real negative.
    plain = {"added_tokens_decoder": {"0": {"content": "<bos>"}}}
    audio_type, definitive = _detect_checked(monkeypatch, [_Resp(200, plain), _Resp(404)])
    assert audio_type is None
    assert definitive is True


def test_a_detected_codec_is_definitive(monkeypatch):
    snac = {
        "added_tokens_decoder": {str(i): {"content": f"<custom_token_{i}>"} for i in range(10_001)}
    }
    audio_type, definitive = _detect_checked(monkeypatch, [_Resp(200, snac)])
    assert audio_type == "snac"
    assert definitive is True
