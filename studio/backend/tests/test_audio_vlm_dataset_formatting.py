# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage for per-example audio VLM instructions."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock


def _stub_if_missing(name, attrs):
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
        return
    except Exception:  # noqa: BLE001 - stub unavailable optional training dependencies
        pass
    module = types.ModuleType(name)
    module.__spec__ = None
    for attr in attrs:
        setattr(module, attr, MagicMock())
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, module)


_stub_if_missing(
    "unsloth", ("FastLanguageModel", "FastVisionModel", "is_bfloat16_supported")
)
_stub_if_missing("unsloth.chat_templates", ("get_chat_template",))
_stub_if_missing("trl", ("SFTTrainer", "SFTConfig"))

from core.training.trainer import UnslothTrainer  # noqa: E402


class _AudioDataset:
    def __init__(self, rows):
        self.rows = rows
        self.column_names = list(rows[0])
        self.messages = None

    def cast_column(self, column, feature):
        return self

    def map(self, fn, **kwargs):
        batch = {
            column: [row[column] for row in self.rows] for column in self.column_names
        }
        self.messages = fn(batch)["messages"]
        return self

    def __len__(self):
        return len(self.rows)


def _format(rows, mapping = None):
    dataset = _AudioDataset(rows)
    trainer = SimpleNamespace(_update_progress = lambda **kwargs: None)
    trainer._resolve_audio_columns = UnslothTrainer._resolve_audio_columns.__get__(trainer)
    trainer._format_audio_vlm_dataset = UnslothTrainer._format_audio_vlm_dataset.__get__(trainer)
    return trainer._format_audio_vlm_dataset(dataset, mapping)


def _row(**extra):
    return {
        "audio": {"array": [0.0, 0.1]},
        "text": "The transcript.",
        **extra,
    }


def test_audio_vlm_retains_mapped_instruction():
    dataset = _format(
        [_row(task = "Identify the speaker's emotion.")],
        {"audio": "audio", "task": "user", "text": "text"},
    )

    assert (
        dataset.messages[0][1]["content"][1]["text"]
        == "Identify the speaker's emotion."
    )


def test_audio_vlm_uses_transcription_fallback_for_blank_instruction():
    dataset = _format(
        [_row(task = "   ")],
        {"audio": "audio", "task": "instruction", "text": "text"},
    )

    assert (
        dataset.messages[0][1]["content"][1]["text"]
        == "Please transcribe this audio."
    )


def test_whisper_audio_text_dataset_keeps_transcription_fallback():
    dataset = _format([_row()])

    assert (
        dataset.messages[0][1]["content"][1]["text"]
        == "Please transcribe this audio."
    )
