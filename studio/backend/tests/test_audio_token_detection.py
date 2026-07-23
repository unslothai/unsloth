# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for tokenizer-based audio_type detection, covering Gemma 3n
(<audio_soft_token>) and Gemma 4 (<|audio|>) audio-input tokens."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from core.inference import llama_cpp
from utils.models import model_config
from utils.models import gguf_metadata
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


def test_structured_audio_chat_family_overrides_codec_markers(monkeypatch):
    monkeypatch.setattr(
        model_config,
        "_NON_CHAT_AUDIO_MODEL_TYPES",
        model_config._NON_CHAT_AUDIO_MODEL_TYPES | {"qwen2_audio"},
    )
    monkeypatch.setattr(
        model_config, "_raw_config_model_type", lambda *args, **kwargs: "qwen2_audio"
    )
    assert _classify_audio_capability("model", "csm") == ("audio_vlm", True, True)


def test_qwen3_asr_identity_wins_over_chat_like_tokens(monkeypatch):
    monkeypatch.setattr(model_config, "_raw_config_model_type", lambda *args, **kwargs: "qwen3_asr")
    assert _classify_audio_capability("model", "audio_vlm") == (None, True, False)


def test_whisper_identity_preserves_loader_audio_type(monkeypatch):
    monkeypatch.setattr(model_config, "_raw_config_model_type", lambda *args, **kwargs: "whisper")
    assert _classify_audio_capability("model", "whisper") == ("whisper", True, False)


def test_cached_audio_projector_is_forwarded_and_clears_false_vision():
    source = inspect.getsource(model_config.ModelConfig.from_identifier)
    assert "gguf_mmproj_file = cached_mmproj" in source
    assert "projector_has_vision is not True" in source
    assert "has_vision = False" in source
    assert "gguf_is_vision = False" in source


def test_audio_projector_loads_retry_and_share_download_exclusion():
    from core.inference import llama_cpp

    backend_source = inspect.getsource(llama_cpp.LlamaCppBackend.load_model)
    guard_source = inspect.getsource(llama_cpp._with_gguf_load_marker)
    route_source = (Path(__file__).resolve().parent.parent / "routes" / "inference.py").read_text()

    assert "has_audio_input = has_audio_input" in backend_source
    assert "has_audio_input and not self._has_audio_input" in inspect.getsource(
        llama_cpp.LlamaCppBackend._already_in_target_state
    )
    assert 'kwargs.get("is_vision") or kwargs.get("has_audio_input")' in guard_source
    assert "(config.is_vision or config.has_audio_input)" in route_source
    assert "config and config.has_audio_input" in route_source
    assert "await asyncio.to_thread(" in route_source
    assert "_resolve_load_model_config" in route_source


class _AudioProbeClient:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def post(self, url, json):
        if url.endswith("/tokenize"):
            tokens = [1] if json["content"] in {"<|AUDIO|>", "<|audio_eos|>"} else []
            return SimpleNamespace(status_code = 200, json = lambda: {"tokens": tokens})
        return SimpleNamespace(status_code = 200, json = lambda: {"content": ""})


def test_runtime_audio_probe_preserves_chat_capability(monkeypatch):
    backend = llama_cpp.LlamaCppBackend()
    backend._gguf_path = "/fake/audio-chat.gguf"
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "is_loaded", property(lambda _self: True))
    monkeypatch.setattr(llama_cpp.httpx, "Client", lambda **_kwargs: _AudioProbeClient())
    monkeypatch.setattr(gguf_metadata, "detect_gguf_audio_type", lambda _path: "audio_vlm")

    assert backend._detect_audio_type_strict() == "audio_vlm"


def test_runtime_audio_probe_keeps_identity_confirmed_csm(monkeypatch):
    backend = llama_cpp.LlamaCppBackend()
    backend._gguf_path = "/fake/csm.gguf"
    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "is_loaded", property(lambda _self: True))
    monkeypatch.setattr(llama_cpp.httpx, "Client", lambda **_kwargs: _AudioProbeClient())
    monkeypatch.setattr(gguf_metadata, "detect_gguf_audio_type", lambda _path: "csm")

    assert backend._detect_audio_type_strict() == "csm"
