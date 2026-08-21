# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_and_format_dataset threads an explicit hf_token into every remote call.

The GPU worker resolves the user's token from config and hands it to
load_and_format_dataset, but the method built its load_dataset and
get_dataset_split_names kwargs with only path/name/revision. A gated or
private dataset therefore fell back to the ambient HF_TOKEN env var
(get_dataset_split_names resolves None through huggingface_hub.get_token()),
so on a host whose env token differs from the request token the metadata
probes and the eventual load could run under the wrong identity.

These tests pin that the token is forwarded when provided — on the streaming
and eager paths, and through the auto-detected eval-split probe — and stays
absent when it is not (so the environment fallback keeps working for callers
that never had a token)."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    """Register a stub module for a dep the backend pytest job does not install.

    Same helper and reason as test_audio_type_inconclusive.py: core.training.trainer
    imports unsloth and trl at module scope while the studio-backend pytest job installs
    studio.txt plus torch and transformers and stops there. A real install is left alone.
    __spec__ = None keeps the trainer's own _ensure_real_packages namespace-shadow guard a
    no-op on the stub.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - unusable here either way, so stub it
        pass
    _STUBBED.append(name)
    mod = types.ModuleType(name)
    mod.__spec__ = None
    for attr in attrs:
        setattr(mod, attr, MagicMock())
    sys.modules[name] = mod
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, mod)


_stub_if_missing("unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported"))
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

import core.training.trainer as trainer_mod  # noqa: E402

# Drop the stubs now that the trainer holds its own references, for the same reason as
# test_audio_type_inconclusive.py: the whole suite would otherwise run against them.
for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


_load_and_format_dataset = trainer_mod.UnslothTrainer.load_and_format_dataset
_auto_detect_eval = trainer_mod.UnslothTrainer._auto_detect_eval_split_from_hf


class _Dataset:
    """The minimum a loaded dataset must look like on the paths under test."""

    column_names = ["text"]

    def __len__(self):
        # >= MIN_EVAL_ROWS (16) so an auto-detected candidate split is accepted.
        return 16


def _trainer():
    """A stand-in carrying only what load_and_format_dataset reads."""
    trainer = SimpleNamespace(
        should_stop = False,
        _audio_type = None,
        is_audio_vlm = False,
        is_vlm = False,
        model_name = "org/model",
        tokenizer = None,
        _update_progress = lambda **kwargs: None,
        _resolve_eval_split_from_dataset = lambda dataset: None,
    )
    trainer._auto_detect_eval_split_from_hf = _auto_detect_eval.__get__(trainer)
    trainer.load_and_format_dataset = _load_and_format_dataset.__get__(trainer)
    return trainer


def _patch_dataset_loading(monkeypatch, load_calls, probe_calls):
    def fake_load_dataset(**kwargs):
        load_calls.append(kwargs)
        return _Dataset()

    def fake_split_names(**kwargs):
        probe_calls.append(kwargs)
        return ["train", "validation"]

    monkeypatch.setattr(trainer_mod, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        trainer_mod,
        "format_and_template_dataset",
        lambda dataset, **kwargs: {"dataset": dataset, "detected_format": "test", "success": True},
    )
    monkeypatch.setattr(sys.modules["datasets"], "get_dataset_split_names", fake_split_names)


def test_streaming_path_forwards_explicit_token(monkeypatch):
    load_calls: list[dict] = []
    probe_calls: list[dict] = []
    _patch_dataset_loading(monkeypatch, load_calls, probe_calls)

    result = _trainer().load_and_format_dataset(
        "org/gated",
        subset = "en",
        train_split = "train",
        eval_split = "validation",
        dataset_streaming = True,
        eval_steps = 1,
        dataset_revision = "dataset-commit",
        hf_token = "hf_0123456789abcdef",
    )

    assert result is not None
    # Streaming eval probes the splits before loading them.
    assert probe_calls == [
        {
            "path": "org/gated",
            "config_name": "en",
            "revision": "dataset-commit",
            "token": "hf_0123456789abcdef",
        }
    ]
    # Train load first, eval load second.
    assert [call.get("split") for call in load_calls] == ["train", "validation"]
    for call in load_calls:
        assert call["path"] == "org/gated"
        assert call["name"] == "en"
        assert call["revision"] == "dataset-commit"
        assert call["token"] == "hf_0123456789abcdef"
        assert call["streaming"] is True


def test_eager_path_forwards_explicit_token(monkeypatch):
    load_calls: list[dict] = []
    probe_calls: list[dict] = []
    _patch_dataset_loading(monkeypatch, load_calls, probe_calls)

    result = _trainer().load_and_format_dataset(
        "org/gated",
        dataset_streaming = False,
        hf_token = "hf_0123456789abcdef",
    )

    assert result is not None
    assert len(load_calls) == 1
    assert load_calls[0]["token"] == "hf_0123456789abcdef"
    assert "streaming" not in load_calls[0]
    assert probe_calls == []


def test_auto_detect_eval_forwards_explicit_token(monkeypatch):
    load_calls: list[dict] = []
    probe_calls: list[dict] = []
    _patch_dataset_loading(monkeypatch, load_calls, probe_calls)

    result = _trainer().load_and_format_dataset(
        "org/gated",
        subset = "en",
        dataset_streaming = False,
        eval_steps = 1,
        dataset_revision = "dataset-commit",
        hf_token = "hf_0123456789abcdef",
    )

    assert result is not None
    # Auto-detect (no explicit eval_split) probes the splits, then loads a candidate.
    assert probe_calls == [
        {
            "path": "org/gated",
            "config_name": "en",
            "revision": "dataset-commit",
            "token": "hf_0123456789abcdef",
        }
    ]
    # Train load first, then the auto-detected "validation" candidate.
    assert [call.get("split") for call in load_calls] == ["train", "validation"]
    for call in load_calls:
        assert call["path"] == "org/gated"
        assert call["name"] == "en"
        assert call["revision"] == "dataset-commit"
        assert call["token"] == "hf_0123456789abcdef"
        assert "streaming" not in call


def test_token_stays_absent_when_not_provided(monkeypatch):
    load_calls: list[dict] = []
    probe_calls: list[dict] = []
    _patch_dataset_loading(monkeypatch, load_calls, probe_calls)

    result = _trainer().load_and_format_dataset(
        "org/dataset",
        dataset_streaming = True,
    )

    assert result is not None
    assert len(load_calls) == 1
    assert "token" not in load_calls[0]
    assert probe_calls == []
