# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Training LFM3-Audio-2B (audio in, no vision tower) loads through AutoModelForMultimodalLM, like Chat;
Gemma 3n / 4 (audio + vision) keep the default FastModel load."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest

_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod


_stub_if_missing(
    "unsloth", ("FastLanguageModel", "FastVisionModel", "FastModel", "is_bfloat16_supported")
)
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

import core.training.trainer as trainer_mod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)

import transformers  # noqa: E402

transformers = sys.modules["transformers"]


class _Lfm3AudioForConditionalGeneration:
    pass


class _Lfm3AudioProcessor:
    feature_extractor = object()
    tokenizer = object()


@pytest.fixture
def load(monkeypatch):
    calls = {}

    def from_pretrained(**kwargs):
        calls.setdefault("from_pretrained", []).append(kwargs)
        return _Lfm3AudioForConditionalGeneration(), calls.get("returns", _Lfm3AudioProcessor())

    fake_unsloth = types.ModuleType("unsloth")
    fake_unsloth.FastModel = types.SimpleNamespace(from_pretrained = from_pretrained)
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

    noop = lambda *a, **k: None
    for name, value in {
        "detect_audio_type_checked": lambda *a, **k: ("audio_vlm", True),
        "clear_gpu_cache": noop,
        "get_device_map": lambda gpu_ids: "auto",
        "get_visible_gpu_count": lambda: 1,
        "raise_if_offloaded": noop,
        "restore_hf_cache_repo_identity": lambda *a, **k: None,
        "is_bfloat16_supported": lambda: True,
        "should_use_mlx_training_backend": lambda: False,
    }.items():
        monkeypatch.setattr(trainer_mod, name, value)
    import utils.cache_cleanup as cache_cleanup

    monkeypatch.setattr(cache_cleanup, "clear_unsloth_compiled_cache", noop)

    def run(
        vision,
        returns = None,
        vision_known = True,
        multimodal = None,
    ):
        multimodal = not vision if multimodal is None else multimodal
        monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **k: vision)
        monkeypatch.setattr(
            trainer_mod,
            "audio_vlm_multimodal_lm_checked",
            lambda *a, **k: (multimodal, vision_known),
        )
        if returns is not None:
            calls["returns"] = returns
        trainer = trainer_mod.UnslothTrainer()
        trainer._detect_diffusion_lm = noop
        trainer._update_progress = lambda **kw: calls.setdefault("errors", []).append(
            kw.get("error")
        )
        ok = trainer.load_model(
            "LiquidAI/LFM3-Audio-2B",
            is_dataset_audio = True,
            load_in_4bit = False,
            trust_remote_code = True,
            local_files_only = True,
        )
        assert ok, [e for e in calls.get("errors", []) if e]
        assert trainer.is_audio_vlm
        return trainer, calls["from_pretrained"][-1]

    return run


def test_audio_without_vision_loads_the_multimodal_class(load):
    if getattr(transformers, "AutoModelForMultimodalLM", None) is None:
        pytest.skip("transformers without AutoModelForMultimodalLM")
    trainer, kwargs = load(vision = False)
    assert kwargs.get("auto_model") is transformers.AutoModelForMultimodalLM
    assert isinstance(trainer.tokenizer, _Lfm3AudioProcessor)


def test_audio_with_vision_keeps_the_default_load(load):
    _, kwargs = load(vision = True)  # Gemma 3n / Gemma 4
    assert "auto_model" not in kwargs


def test_an_inconclusive_vision_probe_keeps_the_default_load(load):
    # A transient probe failure reads as no vision; Gemma 3n / 4 must not lose their default load to it.
    _, kwargs = load(vision = False, vision_known = False)
    assert "auto_model" not in kwargs


def test_audio_without_a_multimodal_class_keeps_the_default_load(load):
    _, kwargs = load(vision = False, multimodal = False)  # AudioFlamingo3
    assert "auto_model" not in kwargs


@pytest.mark.parametrize("with_audio", [True, False])
def test_s3_manifest_selects_audio_input_for_the_downloaded_rows(monkeypatch, with_audio):
    import io
    import json
    import wave

    from core.training import s3_dataset
    from .test_s3_audio_dataset import _FakeS3Client

    repo = "LiquidAI/LFM3-Audio-2B"
    row = {"text": "A spoken line."}
    objects = {}
    if with_audio:
        row["audio"] = "speech.wav"
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16_000)
            wav.writeframes(bytes(800))
        objects["training/speech.wav"] = buffer.getvalue()
    objects["training/train.jsonl"] = json.dumps(row) + "\n"
    client = _FakeS3Client(objects, objects)
    monkeypatch.setattr(s3_dataset, "boto3_available", lambda: True)
    monkeypatch.setattr(s3_dataset, "_build_s3_client", lambda config: client)
    monkeypatch.setattr(
        trainer_mod, "detect_audio_type_checked", lambda *a, **kw: ("audio_vlm", True)
    )
    monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **kw: False)
    monkeypatch.setattr(trainer_mod, "should_use_mlx_training_backend", lambda: False)
    monkeypatch.setattr(trainer_mod, "_load_adapter_aware_tokenizer", lambda *a, **kw: object())

    trainer = trainer_mod.UnslothTrainer()
    trainer._detect_diffusion_lm = lambda *a: None
    trainer.pre_detect_and_load_tokenizer(repo, max_seq_length = 256, is_dataset_audio = False)
    assert not trainer.is_audio_vlm

    def format_text(dataset, **kwargs):
        assert not with_audio
        return {"success": True, "dataset": dataset}

    monkeypatch.setattr(trainer_mod, "format_and_template_dataset", format_text)
    formatted, evaluation = trainer.load_and_format_dataset(
        None,
        s3_config = {"bucket": "stub-bucket", "prefix": "training/", "use_iam_role": True},
    )
    assert evaluation is None
    assert trainer._is_dataset_audio is with_audio
    assert trainer.is_audio_vlm is with_audio
    if with_audio:
        parts = [part for message in formatted[0]["messages"] for part in message["content"]]
        assert any(part.get("text") == row["text"] for part in parts)
        [audio] = [part for part in parts if part["type"] == "audio"]
        assert len(audio["audio"]) == 400
    else:
        assert formatted["dataset"][0]["text"] == row["text"]
