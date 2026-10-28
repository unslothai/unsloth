# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native audio security targets list every companion repo that actually loads."""

import json

import pytest

from core.inference.native_audio import (
    HIGGS_TTS3_CODEC_REPO,
    MOSS_LOCAL_CODEC_REPO,
    native_audio_security_targets,
)


def _write(path, name, value):
    path.mkdir(parents = True, exist_ok = True)
    (path / name).write_text(json.dumps(value), encoding = "utf-8")


def test_moss_local_audio_tokenizer_config_codec_is_a_target(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write(checkpoint, "config.json", {"model_type": "moss_tts_local"})
    _write(
        checkpoint,
        "audio_tokenizer_config.json",
        {"audio_tokenizer_class": "AutoModel", "audio_tokenizer_name_or_path": "acme/other-codec"},
    )
    targets = native_audio_security_targets(str(checkpoint), "moss_tts_local")
    assert "acme/other-codec" in targets
    assert targets[:2] == [str(checkpoint), MOSS_LOCAL_CODEC_REPO]


def test_moss_local_audio_tokenizer_config_same_codec_is_not_duplicated(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    _write(checkpoint, "processor_config.json", {"audio_tokenizer_name_or_path": "acme/codec"})
    _write(
        checkpoint, "audio_tokenizer_config.json", {"audio_tokenizer_name_or_path": "acme/codec"}
    )
    assert native_audio_security_targets(str(checkpoint), "moss_tts_local") == [
        str(checkpoint),
        "acme/codec",
    ]


@pytest.mark.parametrize(
    ("audio_type", "metadata_file", "codec_key"),
    (
        ("moss_tts_local", "processor_config.json", "audio_tokenizer_name_or_path"),
        ("moss_tts_local", "audio_tokenizer_config.json", "audio_tokenizer_name_or_path"),
        ("higgs_tts2", "audio_tokenizer_config.json", "audio_tokenizer_name_or_path"),
        ("higgs_tts3", "config.json", "audio_tokenizer_id"),
    ),
)
def test_adapter_only_codec_adds_its_base(tmp_path, audio_type, metadata_file, codec_key):
    checkpoint = tmp_path / "checkpoint"
    codec = tmp_path / "codec"
    _write(checkpoint, metadata_file, {codec_key: str(codec)})
    _write(
        codec,
        "adapter_config.json",
        {"peft_type": "LORA", "base_model_name_or_path": "acme/hidden-base"},
    )
    targets = native_audio_security_targets(str(checkpoint), audio_type)
    assert str(codec) in targets
    assert "acme/hidden-base" in targets


def test_local_codec_with_config_is_not_swapped_for_its_base(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    codec = tmp_path / "codec"
    _write(checkpoint, "config.json", {"audio_tokenizer_id": str(codec)})
    _write(codec, "config.json", {"model_type": "higgs_audio_v2_tokenizer"})
    _write(codec, "adapter_config.json", {"base_model_name_or_path": "acme/unused-base"})
    assert native_audio_security_targets(str(checkpoint), "higgs_tts3") == [
        str(checkpoint),
        str(codec),
    ]


def test_stock_layout_targets_unchanged_and_curated_codec_not_probed(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    _write(checkpoint, "config.json", {"model_type": "higgs_multimodal_qwen3"})
    import core.inference.native_audio as native_audio

    real_read = native_audio._read_audio_metadata
    read = []

    def tracking_read(model_name, filename, *args, **kwargs):
        read.append((model_name, filename))
        return real_read(model_name, filename, *args, **kwargs)

    monkeypatch.setattr(native_audio, "_read_audio_metadata", tracking_read)
    assert native_audio_security_targets(str(checkpoint), "higgs_tts3") == [
        str(checkpoint),
        HIGGS_TTS3_CODEC_REPO,
    ]
    assert all(name != HIGGS_TTS3_CODEC_REPO for name, _ in read)


def test_oversized_codec_adapter_config_fails_closed(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    codec = tmp_path / "codec"
    _write(checkpoint, "config.json", {"audio_tokenizer_id": str(codec)})
    _write(
        codec,
        "adapter_config.json",
        {"base_model_name_or_path": "acme/hidden-base", "padding": "x" * 1_000_000},
    )
    with pytest.raises(ValueError, match = "security inspection limit"):
        native_audio_security_targets(str(checkpoint), "higgs_tts3")
