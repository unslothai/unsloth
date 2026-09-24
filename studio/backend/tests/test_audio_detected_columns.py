# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The trainer trains on the audio and text columns the dataset check accepted."""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import importlib  # noqa: E402
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import pytest  # noqa: E402
from datasets import Dataset  # noqa: E402

from hub.utils.dataset_format import check_dataset_format  # noqa: E402


_STUBBED: list[str] = []


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unusable imports
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

from core.training import trainer as tmod  # noqa: E402

for _name in reversed(_STUBBED):
    sys.modules.pop(_name, None)


_AUDIO = {"array": [0.0] * 160, "sampling_rate": 16000}


@pytest.fixture
def audio_trainer(monkeypatch):
    monkeypatch.setattr(tmod, "should_use_mlx_training_backend", lambda *a, **k: False)
    return tmod.UnslothTrainer()


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"audio": _AUDIO, "normalized_text": "hi"}, ("audio", "normalized_text", None)),
        ({"audio": _AUDIO, "prompt": "hi"}, ("audio", "prompt", None)),
        ({"audio": _AUDIO, "label": "hi"}, ("audio", "label", None)),
        ({"wav": _AUDIO, "text": "hi"}, ("wav", "text", None)),
        ({"clip": _AUDIO, "text": "hi", "speaker": "a"}, ("clip", "text", "speaker")),
    ],
)
def test_resolves_the_columns_the_check_accepted(audio_trainer, row, expected):
    dataset = Dataset.from_list([row])
    check = check_dataset_format(dataset, is_vlm = False)
    assert check["requires_manual_mapping"] is False

    resolved = audio_trainer._resolve_audio_columns(dataset, None)

    assert (resolved["audio_col"], resolved["text_col"], resolved["speaker_col"]) == expected
    assert resolved["audio_col"] == check["detected_audio_column"]


def test_exact_names_and_explicit_mapping_are_unchanged(audio_trainer):
    dataset = Dataset.from_list(
        [{"audio": _AUDIO, "text": "hi", "prompt": "p", "speaker": "a", "source": "0"}]
    )

    assert audio_trainer._resolve_audio_columns(dataset, None) == {
        "audio_col": "audio",
        "text_col": "text",
        "speaker_col": "source",
    }
    assert audio_trainer._resolve_audio_columns(dataset, {"audio": "audio", "prompt": "text"}) == {
        "audio_col": "audio",
        "text_col": "prompt",
        "speaker_col": None,
    }


def test_text_only_dataset_still_resolves_nothing(audio_trainer):
    dataset = Dataset.from_list([{"instruction": "a", "output": "b"}])

    assert audio_trainer._resolve_audio_columns(dataset, None) == {
        "audio_col": None,
        "text_col": None,
        "speaker_col": None,
    }
