# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What counts as proof of a second modality when the auto-switch judges a downloaded checkpoint: the
transformers 5 spellings (unsloth#10951, a conversion that kept its placeholder token ids after losing its vision
tower), and that a nested ``text_config`` is NOT proof, since transformers 5 nests one in text-only configs too."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import local_model_resolver as resolver
from utils.hardware import hardware as hw


@pytest.fixture(autouse=True)
def _host_serves_non_gguf(monkeypatch):
    """Pin the host-capability gates: these tests are about the config rules, not this machine."""
    monkeypatch.setattr(resolver, "_host_has_a_non_gguf_backend", lambda: True)
    # the device, not the helper, so a test setting DEVICE for itself still wins.
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA, raising=False)


@pytest.fixture(autouse=True)
def _clean_resolver_index():
    """Drop the scan cache around every test: the index is keyed by id and has a TTL."""
    resolver.invalidate_index()
    yield
    resolver.invalidate_index()


def _checkpoint(root, name: str, config: dict):
    """An on-disk non-GGUF checkpoint carrying *config*: what the index scan needs to see."""
    path = root / name
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"\0" * 32)
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return SimpleNamespace(id=str(path), path=str(path))


def _classifies(info, config: dict):
    return resolver._is_generative_chat_config(Path(info.path), config)


# ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit from #10951: a language-only MLX conversion, vision_config gone.
REPORTED_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {"hidden_size": 2048, "model_type": "qwen3_5_moe_text"},
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}
# caslca/Qwen3.8-27B-mlx-uniform-4bit, the working control from the same report: it kept vision_config.
CONTROL_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {"hidden_size": 5120, "model_type": "qwen3_5_text"},
    "vision_config": {"hidden_size": 1152, "model_type": "qwen3_5_vision"},
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}


def test_the_reported_language_only_conversion_is_servable(tmp_path):
    """#10951: installed, unloaded, and 404 model_not_found from the API until it is accepted."""
    info = _checkpoint(tmp_path, "Ornith-1.5-35B-A3B-MLX-4bit", REPORTED_CONFIG)

    assert _classifies(info, REPORTED_CONFIG) is True
    assert resolver.local_servable_model(info) == (False, ())


def test_the_control_from_the_same_report_stays_servable(tmp_path):
    info = _checkpoint(tmp_path, "Qwen3.8-27B-mlx-uniform-4bit", CONTROL_CONFIG)

    assert _classifies(info, CONTROL_CONFIG) is True
    assert resolver.local_servable_model(info) == (False, ())


@pytest.mark.parametrize(
    "marker",
    [
        "image_token_id",
        "video_token_id",
        "video_token_index",
        "vision_start_token_id",
        "vision_end_token_id",
    ],
)
def test_one_visual_token_id_is_enough_on_its_own(tmp_path, marker):
    """A conversion keeps whichever ids its template still emits, so no id may be mandatory."""
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        marker: 151655,
    }
    info = _checkpoint(tmp_path, f"conversion-{marker}", config)

    assert resolver.local_servable_model(info) == (False, ())


@pytest.mark.parametrize(
    ("architecture", "model_type"),
    [
        ("T5ForConditionalGeneration", "t5"),
        ("BartForConditionalGeneration", "bart"),
    ],
)
def test_a_text_seq2seq_is_still_refused(tmp_path, architecture, model_type):
    config = {"architectures": [architecture], "model_type": model_type}
    info = _checkpoint(tmp_path, model_type, config)

    assert _classifies(info, config) is False
    assert resolver.local_servable_model(info) is None


@pytest.mark.parametrize(
    ("name", "config"),
    [
        # transformers 5.17 ClvpConfig: text_config and no modality key, on a ForConditionalGeneration name.
        (
            "clvp",
            {
                "architectures": ["ClvpModelForConditionalGeneration"],
                "model_type": "clvp",
                "text_config": {"hidden_size": 768, "model_type": "clvp_text_model"},
            },
        ),
        (
            "t5-with-text-config",
            {
                "architectures": ["T5ForConditionalGeneration"],
                "model_type": "t5",
                "text_config": {"hidden_size": 768, "model_type": "t5"},
            },
        ),
    ],
)
def test_a_nested_text_config_alone_is_not_proof_of_a_modality(tmp_path, name, config):
    info = _checkpoint(tmp_path, name, config)

    assert _classifies(info, config) is False
    assert resolver.local_servable_model(info) is None


@pytest.mark.parametrize(
    "partner",
    ["vision_config", "audio_config", "image_token_id", "video_token_id", "image_token_index"],
)
def test_a_text_config_counts_beside_a_modality_sibling(tmp_path, partner):
    """The other half of the rule: text_config is not refused, it is simply not the evidence."""
    value = {"hidden_size": 1} if partner.endswith("_config") else 151655
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {"hidden_size": 2048},
        partner: value,
    }
    info = _checkpoint(tmp_path, f"paired-{partner}", config)

    assert resolver.local_servable_model(info) == (False, ())


def test_text_config_is_not_recorded_as_a_modality_key():
    assert "text_config" not in resolver._VISUAL_TOKEN_ID_KEYS
    assert not resolver._config_declares_multimodality({"text_config": {"hidden_size": 1}})
    # "text" is not a modality word either, or the word match would readmit it by the back door.
    assert "text" not in resolver._MODALITY_KEY_WORDS


def _safe_dir_name(text: str) -> str:
    """A fixture directory name Windows will accept: `{"id": 1}` spells out as `bad-marker-dict-{'id': 1}`,
    which windows-latest rejected with NotADirectoryError [WinError 267] while Linux built it happily."""
    import re

    return re.sub(r"[^A-Za-z0-9._-]", "_", text)[:60]


@pytest.mark.parametrize(
    "value",
    [None, False, True, -1, "151655", "", [], (), {}, {"id": 1}, [1, None], [1, "2"]],
    ids=[
        "null",
        "false",
        "true",
        "negative",
        "string",
        "empty-string",
        "empty-list",
        "empty-tuple",
        "empty-dict",
        "dict",
        "list-with-null",
        "list-with-string",
    ],
)
def test_a_visual_marker_that_is_not_a_token_id_does_not_admit_anything(tmp_path, value):
    """The key is there, the value is not a vocabulary index. `true` is separate: bool is an int subclass."""
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "image_token_id": value,
    }
    info = _checkpoint(
        tmp_path, _safe_dir_name(f"bad-marker-{type(value).__name__}-{value!r}"), config
    )

    assert "image_token_id" in config, "the case is only interesting while the KEY is present"
    assert _classifies(info, config) is False
    assert resolver.local_servable_model(info) is None


@pytest.mark.parametrize(
    ("architecture", "model_type"),
    [("T5ForConditionalGeneration", "t5"), ("BartForConditionalGeneration", "bart")],
)
@pytest.mark.parametrize(
    "marker",
    [
        "image_token_id",
        "video_token_id",
        "video_token_index",
        "vision_start_token_id",
        "vision_end_token_id",
    ],
)
def test_a_seq2seq_carrying_a_null_visual_marker_is_still_refused(
    tmp_path, architecture, model_type, marker
):
    """The counterexample a presence-only predicate passes: T5 and BART carrying a new key, set to null."""
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "text_config": {"hidden_size": 768, "model_type": model_type},
        marker: None,
    }
    info = _checkpoint(tmp_path, f"{model_type}-{marker}-null", config)

    assert _classifies(info, config) is False
    assert resolver.local_servable_model(info) is None


@pytest.mark.parametrize("value", [0, 1, 151655, [151655], [151655, 151656], (151655,)])
def test_a_real_visual_marker_still_admits_the_reported_conversion(tmp_path, value):
    """Zero is a legal vocabulary index, and transformers 5 writes a LIST for a multi-slot model."""
    config = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "image_token_id": value,
    }
    info = _checkpoint(tmp_path, _safe_dir_name(f"good-marker-{value!r}"), config)

    assert _classifies(info, config) is True
    assert resolver.local_servable_model(info) == (False, ())


def test_a_visual_marker_does_not_get_an_audio_family_past_its_own_gate(tmp_path, monkeypatch):
    """A visual marker must not carry an audio family past the MLX refusal its own branch applies."""
    config = {
        "architectures": ["CsmForConditionalGeneration"],
        "model_type": "csm",
        "audio_token_id": 128002,
        "image_token_id": 151655,
    }
    info = _checkpoint(tmp_path, "csm-with-a-visual-marker", config)

    monkeypatch.setattr(resolver, "_host_serves_mlx", lambda: True)
    assert (
        _classifies(info, config) is False
    ), "a visual marker carried an audio family past the MLX refusal"
    assert resolver.local_servable_model(info) is None

    resolver.invalidate_index()
    monkeypatch.setattr(resolver, "_host_serves_mlx", lambda: False)
    assert _classifies(info, config) is True


@pytest.mark.parametrize(
    ("architecture", "model_type"),
    [
        ("HiggsAudioV2ForConditionalGeneration", "higgs_audio_v2"),
        ("VibeVoiceAsrForConditionalGeneration", "vibevoice_asr"),
    ],
)
def test_an_audio_token_id_does_not_admit_a_family_off_the_audio_allowlist(
    tmp_path, architecture, model_type
):
    """These carry audio_token_id and no audio_config in transformers 5.17: a marker-based audio rule would
    advertise a TTS and an ASR model as chat targets, around _SUPPORTED_CONDITIONAL_AUDIO_MODEL_TYPES."""
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "text_config": {"hidden_size": 1},
        "audio_token_id": 128003,
    }
    info = _checkpoint(tmp_path, model_type, config)

    assert resolver.local_servable_model(info) is None


def test_csm_keeps_its_host_dependent_audio_verdict(tmp_path, monkeypatch):
    """csm is on the audio allowlist, so its verdict stays the host's: MLX rejects TTS outright."""
    config = {
        "architectures": ["CsmForConditionalGeneration"],
        "model_type": "csm",
        "audio_token_id": 128002,
    }
    info = _checkpoint(tmp_path, "csm-1b", config)

    monkeypatch.setattr(resolver, "_host_serves_mlx", lambda: False)
    assert resolver.local_servable_model(info) == (False, ())

    resolver.invalidate_index()
    monkeypatch.setattr(resolver, "_host_serves_mlx", lambda: True)
    assert _classifies(info, config) is False
