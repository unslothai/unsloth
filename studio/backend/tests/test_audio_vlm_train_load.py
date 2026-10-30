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


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "FastModel", "is_bfloat16_supported"))
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

    def run(vision, returns = None):
        monkeypatch.setattr(trainer_mod, "is_vision_model", lambda *a, **k: vision)
        if returns is not None:
            calls["returns"] = returns
        trainer = trainer_mod.UnslothTrainer()
        trainer._detect_diffusion_lm = noop
        trainer._update_progress = lambda **kw: calls.setdefault("errors", []).append(kw.get("error"))
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

