# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What counts as proof of a second modality when the auto-switch judges a downloaded checkpoint.

``_is_generative_chat_config`` accepts an architecture ending in ``ForConditionalGeneration`` only
when the config proves it carries a modality a chat request can be served with, because T5 and BART
wear the same suffix and this path has no seq2seq branch. Two things are held here:

* the transformers 5 spellings of that proof, so a conversion that dropped its vision tower but kept
  its placeholder token ids is still served (unsloth#10951), and
* that a nested ``text_config`` is NOT that proof, because transformers 5 nests one in text-only
  configs too. Accepting it alone would admit ``ClvpModelForConditionalGeneration`` and any seq2seq
  that grows a ``text_config``, which is exactly what the gate exists to refuse.

The config bodies for the reported checkpoint and its control are the real ones, reduced to the keys
this gate reads.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from core.inference import local_model_resolver as resolver
from utils.hardware import hardware as hw


@pytest.fixture(autouse = True)
def _host_serves_non_gguf(monkeypatch):
    """Pin the host-capability gates, as test_openai_auto_switch.py does for the same classifier.

    These tests are about the config rules. Left unpinned they would pass or fail by whether this
    machine has torch or MLX, and the MLX gate has its own test below.
    """
    monkeypatch.setattr(resolver, "_host_has_a_non_gguf_backend", lambda: True)
    # the device itself, not the helper, so a test setting DEVICE for itself still wins.
    monkeypatch.setattr(hw, "DEVICE", hw.DeviceType.CUDA, raising = False)


@pytest.fixture(autouse = True)
def _clean_resolver_index():
    """Drop the scan cache around every test: the index is keyed by id and has a TTL."""
    resolver.invalidate_index()
    yield
    resolver.invalidate_index()


def _checkpoint(root, name: str, config: dict):
    """An on-disk non-GGUF checkpoint carrying *config*: what the index scan needs to see."""
    path = root / name
    path.mkdir(parents = True)
    (path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (path / "model.safetensors").write_bytes(b"\0" * 32)
    (path / "tokenizer.json").write_text("{}", encoding = "utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding = "utf-8")
    return SimpleNamespace(id = str(path), path = str(path))


# ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit, the checkpoint reported in #10951: a language-only MLX
# conversion of a VLM. architectures still says ForConditionalGeneration, the vision tower and its
# vision_config are gone, and the placeholder token ids are all that is left to judge it by.
REPORTED_CONFIG = {
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "model_type": "qwen3_5_moe",
    "text_config": {"hidden_size": 2048, "model_type": "qwen3_5_moe_text"},
    "image_token_id": 151655,
    "video_token_id": 151656,
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
}
# caslca/Qwen3.8-27B-mlx-uniform-4bit, the control from the same report: the same suffix on the same
# host, working, and differing only in keeping a top-level vision_config.
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

    assert resolver._is_generative_chat_config(REPORTED_CONFIG) is True
    assert resolver.local_servable_model(info) == (False, ())


def test_the_control_from_the_same_report_stays_servable(tmp_path):
    info = _checkpoint(tmp_path, "Qwen3.8-27B-mlx-uniform-4bit", CONTROL_CONFIG)

    assert resolver._is_generative_chat_config(CONTROL_CONFIG) is True
    assert resolver.local_servable_model(info) == (False, ())


@pytest.mark.parametrize(
    "marker",
    ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"],
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
    """The pair this gate exists to exclude. Neither carries any modality key."""
    config = {"architectures": [architecture], "model_type": model_type}
    info = _checkpoint(tmp_path, model_type, config)

    assert resolver._is_generative_chat_config(config) is False
    assert resolver.local_servable_model(info) is None


@pytest.mark.parametrize(
    ("name", "config"),
    [
        # transformers 5.17 ClvpConfig: text_config and no modality key at all, on
        # ClvpModelForConditionalGeneration. A voice model with no chat serving path here.
        (
            "clvp",
            {
                "architectures": ["ClvpModelForConditionalGeneration"],
                "model_type": "clvp",
                "text_config": {"hidden_size": 768, "model_type": "clvp_text_model"},
            },
        ),
        # And the case that makes the rule matter for the gate's own exclusions: a text seq2seq
        # that grows a text_config is still a text seq2seq.
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

    assert resolver._is_generative_chat_config(config) is False
    assert resolver.local_servable_model(info) is None


@pytest.mark.parametrize(
    "partner",
    ["vision_config", "audio_config", "image_token_id", "video_token_id", "image_token_index"],
)
def test_a_text_config_counts_beside_a_modality_sibling(tmp_path, partner):
    """The other half of the rule: text_config is not refused, it is simply not the evidence.

    Paired with a vision or audio sibling, or with a per-modality token id, the same config is
    served. Parametrised so the pairing is checked rather than described in a comment.
    """
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
    """Stated at the constant, so re-adding it has to argue with this test."""
    assert "text_config" not in resolver._MULTIMODAL_CONFIG_KEYS
    # audio_token_id is excluded for its own reason: the audio families are admitted by name, and a
    # generic marker would bypass that allowlist.
    assert "audio_token_id" not in resolver._MULTIMODAL_CONFIG_KEYS


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
    """These carry audio_token_id and no audio_config in transformers 5.17.

    ``_SUPPORTED_CONDITIONAL_AUDIO_MODEL_TYPES`` is the audio decision, and it is a list of the
    model types that have a serving path here. A marker-based audio rule would advertise a TTS and
    an ASR model as chat targets and would also bypass the MLX refusal below.
    """
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "text_config": {"hidden_size": 1},
        "audio_token_id": 128003,
    }
    info = _checkpoint(tmp_path, model_type, config)

    assert resolver.local_servable_model(info) is None


def test_csm_keeps_its_host_dependent_audio_verdict(tmp_path, monkeypatch):
    """csm is on the audio allowlist, and it carries audio_token_id and no audio_config.

    Its verdict must stay the host's: served where a Transformers worker runs, refused on an MLX
    host whose worker rejects TTS outright. A generic audio marker would have served it on both.
    """
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
    assert resolver._is_generative_chat_config(config) is False
