# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A TTS model must never be chat-loadable.

The Audio page loads speech models into the single slot chat reads, and
``openai_chat_completions`` answers a turn on one by SYNTHESIZING the prompt
rather than refusing it. Auto-load picks the smallest downloaded model and TTS
models are small, so one became the default chat model on a fresh install.

Architecture cannot answer this -- Orpheus and OuteTTS are ``LlamaForCausalLM``,
Spark is ``Qwen2ForCausalLM`` -- so the codec vocabulary in
``tokenizer_config.json`` is the signal, with the curated ids covering the GGUF
companions that ship no tokenizer at all.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

from utils.audio_tokens import (
    AUDIO_TOKEN_PATTERNS,
    TTS_AUDIO_TYPES,
    detect_local_tts_audio_type,
    is_tts_audio_type,
)
from utils.hidden_models import is_curated_stt_repo_id, is_curated_tts_repo_id


def _model_dir(tmp_path: Path, name: str, architectures, tokens) -> Path:
    path = tmp_path / name
    path.mkdir(parents = True, exist_ok = True)
    (path / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": architectures}),
        encoding = "utf-8",
    )
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    str(index): {"content": token} for index, token in enumerate(tokens)
                }
            }
        ),
        encoding = "utf-8",
    )
    return path


def _snac_tokens() -> list[str]:
    # Orpheus ships a stray <|audio|> beside its codebook; the codec must still win.
    return ["<|audio|>"] + [f"<custom_token_{index}>" for index in range(10_002)]


def test_orpheus_shaped_directory_is_detected_as_tts(tmp_path):
    path = _model_dir(tmp_path, "orpheus", ["LlamaForCausalLM"], _snac_tokens())
    assert detect_local_tts_audio_type(path) == "snac"


def test_every_tts_codec_is_detected(tmp_path):
    cases = {
        "csm": ["<|AUDIO|>", "<|audio_eos|>"],
        "bicodec": ["<|bicodec_semantic_0|>"],
        "dac": ["<|audio_start|>", "<|audio_end|>", "<|text_start|>", "<|text_end|>"],
        "snac": _snac_tokens(),
    }
    # Pinned against the source of truth so a codec added there without a case here fails.
    assert set(cases) == set(TTS_AUDIO_TYPES)
    for audio_type, tokens in cases.items():
        path = _model_dir(tmp_path, audio_type, ["LlamaForCausalLM"], tokens)
        assert detect_local_tts_audio_type(path) == audio_type


def test_a_speech_model_is_not_chattable_despite_a_causal_lm_head(tmp_path):
    """The whole point: the suffix rule below it answers True for this directory."""
    from hub.services.models.common import _local_transformers_can_chat

    path = _model_dir(tmp_path, "orpheus", ["LlamaForCausalLM"], _snac_tokens())
    assert _local_transformers_can_chat(path) is False


def test_an_ordinary_chat_model_stays_chattable(tmp_path):
    from hub.services.models.common import _local_transformers_can_chat
    path = _model_dir(tmp_path, "llama", ["LlamaForCausalLM"], ["<bos>", "<eos>"])
    assert _local_transformers_can_chat(path) is True


def test_an_audio_input_chat_model_stays_chattable(tmp_path):
    """Gemma 3n takes audio IN and answers in text, so the probe must not claim it."""
    from hub.services.models.common import _local_transformers_can_chat

    path = _model_dir(
        tmp_path, "gemma3n", ["Gemma3nForConditionalGeneration"], ["<audio_soft_token>"]
    )
    assert detect_local_tts_audio_type(path) is None
    assert _local_transformers_can_chat(path) is True


def test_whisper_is_not_claimed_by_the_tts_probe(tmp_path):
    """STT has its own path (stt_only / is_curated_stt_repo_id); the two must not overlap."""
    path = _model_dir(
        tmp_path, "whisper", ["WhisperForConditionalGeneration"], ["<|startoftranscript|>"]
    )
    assert detect_local_tts_audio_type(path) is None


def test_a_directory_without_a_tokenizer_is_not_tts(tmp_path):
    path = tmp_path / "bare"
    path.mkdir()
    (path / "config.json").write_text('{"architectures":["LlamaForCausalLM"]}', encoding = "utf-8")
    assert detect_local_tts_audio_type(path) is None


def test_unreadable_targets_answer_none(tmp_path):
    assert detect_local_tts_audio_type(tmp_path / "missing") is None
    assert detect_local_tts_audio_type(tmp_path / "missing" / "config.json") is None


def test_is_tts_audio_type_excludes_the_input_only_types():
    for audio_type in TTS_AUDIO_TYPES:
        assert is_tts_audio_type(audio_type)
    assert not is_tts_audio_type("whisper")
    assert not is_tts_audio_type("audio_vlm")
    assert not is_tts_audio_type(None)


def test_the_tts_set_is_a_subset_of_the_classifier():
    # A type here that the patterns cannot produce would never fire.
    assert TTS_AUDIO_TYPES <= set(AUDIO_TOKEN_PATTERNS)


def test_curated_tts_repo_ids_cover_the_gguf_companion():
    """A GGUF repo carries no tokenizer_config, so only the ids can answer."""
    assert is_curated_tts_repo_id("unsloth/orpheus-3b-0.1-ft-GGUF")
    assert is_curated_tts_repo_id("UNSLOTH/Orpheus-3B-0.1-FT-GGUF")
    assert is_curated_tts_repo_id("unsloth/csm-1b")
    assert is_curated_tts_repo_id("unsloth/Spark-TTS-0.5B")
    assert is_curated_tts_repo_id("unsloth/Llama-OuteTTS-1.0-1B")
    assert not is_curated_tts_repo_id("unsloth/gemma-4-E2B-it")
    assert not is_curated_tts_repo_id(None)
    # The two curated sets describe different halves of the Audio page.
    assert not is_curated_tts_repo_id("unsloth/whisper-large-v3")
    assert not is_curated_stt_repo_id("unsloth/orpheus-3b-0.1-ft")


def test_a_curated_tts_repo_row_is_not_chat_loadable(tmp_path):
    """can_chat is what auto-load filters on, and a GGUF row's capabilities come from
    the file format alone."""
    from hub.services.models.cache_inventory import _cache_inventory_fields

    fields = _cache_inventory_fields(
        "unsloth/orpheus-3b-0.1-ft-GGUF",
        "gguf",
        snapshot_path = tmp_path,
    )
    assert fields["capabilities"]["can_chat"] is False


def test_an_ordinary_gguf_repo_row_still_chats(tmp_path):
    from hub.services.models.cache_inventory import _cache_inventory_fields
    fields = _cache_inventory_fields(
        "unsloth/gemma-4-E2B-it-GGUF",
        "gguf",
        snapshot_path = tmp_path,
    )
    assert fields["capabilities"]["can_chat"] is True


def test_a_lora_over_a_speech_base_is_not_chattable(tmp_path):
    """Studio trains Orpheus LoRAs, and an adapter resolves its base to decide this, so
    without the probe every voice fine-tune became chat-loadable too."""
    from hub.services.models.common import _local_path_can_chat

    base = _model_dir(tmp_path, "orpheus-base", ["LlamaForCausalLM"], _snac_tokens())
    adapter = tmp_path / "my-voice-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(base)}), encoding = "utf-8"
    )

    assert _local_path_can_chat(adapter) is False


def test_the_tts_only_flag_clears_can_chat(tmp_path):
    """The probe's answer for an uncurated safetensors copy reaches the row."""
    from hub.services.models.cache_inventory import _cache_inventory_fields

    fields = _cache_inventory_fields(
        "someone/my-finetuned-voice",
        "gguf",
        snapshot_path = tmp_path,
        tts_only = True,
    )
    assert fields["capabilities"]["can_chat"] is False
