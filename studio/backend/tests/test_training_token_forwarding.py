# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_and_format_dataset must forward an explicit hf_token to every remote call.

Without it, load_dataset and get_dataset_split_names resolve None through
huggingface_hub.get_token() and read a gated dataset under the ambient HF_TOKEN
instead of the request identity. The no-token case is pinned too: that env fallback
is the only credential a caller without a token has."""

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
    """Stub a dep the backend pytest job does not install, as test_audio_type_inconclusive.py does.

    core.training.trainer imports unsloth and trl at module scope; that job installs studio.txt
    plus torch and transformers and stops. __spec__ = None keeps the trainer's own
    _ensure_real_packages namespace-shadow guard a no-op on the stub.
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

# Trainer holds its own references now; leaving the stubs would run the whole suite on them.
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
    assert probe_calls == [
        {
            "path": "org/gated",
            "config_name": "en",
            "revision": "dataset-commit",
            "token": "hf_0123456789abcdef",
        }
    ]
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
