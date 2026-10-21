# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""'Assistant completions only' for audio VLMs (Gemma 3n / Gemma 4 audio).

The audio VLM collator builds its labels per batch, so the shared dataset-level masking never reaches it. These
tests exec the collator block of core/training/trainer.py (importing the trainer drags in the whole torch/unsloth
stack) against a small fake processor.
"""

from __future__ import annotations

import logging
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from utils.datasets.completion_masking import apply_completion_masking

PAD, AUDIO, USER, MODEL, END = 0, 5, 1, 2, 3
VOCAB = {"<pad>": PAD, "<user>": USER, "<model>": MODEL, "<end>": END, "<audio>": AUDIO}


def _collator_block():
    lines = (
        (Path(__file__).resolve().parents[1] / "core" / "training" / "trainer.py")
        .read_text(encoding = "utf-8")
        .splitlines()
    )
    start = next(
        i
        for i, line in enumerate(lines)
        if line.strip() == 'audio_col_name = getattr(self, "_audio_vlm_audio_col", "audio")'
    )
    end = next(
        i
        for i in range(start, len(lines))
        if lines[i].strip() == "data_collator = audio_vlm_collate_fn"
    )
    return textwrap.dedent("\n".join(lines[start:end]))


class _Tokenizer:
    pad_token_id = PAD
    audio_token_id = AUDIO

    def __init__(self, preset_markers):
        if preset_markers:
            self._unsloth_input_part = "<user>"
            self._unsloth_output_part = "<model>"

    def id(self, word):
        return VOCAB.setdefault(word, len(VOCAB) + 10)


class _Processor:
    def __init__(self, preset_markers = True):
        self.tokenizer = _Tokenizer(preset_markers)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        parts = []
        for message in messages:
            marker = "<model>" if message["role"] == "assistant" else "<user>"
            parts.append(f"{marker} {message['content']} <end>")
        return " ".join(parts)

    def __call__(self, text, audio, return_tensors, padding):
        rows = [[self.tokenizer.id(word) for word in t.split()] for t in text]
        width = max(len(row) for row in rows)
        return {"input_ids": torch.tensor([[PAD] * (width - len(row)) + row for row in rows])}


def _fake_train_on_responses_only(
    trainer,
    tokenizer = None,
    return_function = False,
    **kwargs,
):
    """Keeps old labels from each <model> marker to its <end>, as unsloth_zoo does."""
    assert trainer is None and return_function

    def mask(batch):
        out = []
        for ids, old in zip(batch["input_ids"].tolist(), batch["labels"].tolist()):
            new, keep = [-100] * len(ids), False
            for i, token in enumerate(ids):
                if keep:
                    new[i] = old[i]
                keep = (keep and token != END) or token == MODEL
            out.append(new)
        return {"labels": torch.tensor(out)}

    return mask


@pytest.fixture
def fake_unsloth(monkeypatch):
    chat_templates = types.ModuleType("unsloth.chat_templates")
    chat_templates.train_on_responses_only = _fake_train_on_responses_only
    monkeypatch.setitem(sys.modules, "unsloth.chat_templates", chat_templates)
    monkeypatch.setitem(sys.modules, "unsloth", types.ModuleType("unsloth"))


def _collate(
    processor,
    model_name = "unmapped/audio-model",
    **training_args,
):
    namespace = {
        "processor": processor,
        "training_args": training_args,
        "self": types.SimpleNamespace(_audio_vlm_audio_col = "speech", model_name = model_name),
        "logger": logging.getLogger("test_audio_vlm_completion_masking"),
        "types": types,
        "apply_completion_masking": apply_completion_masking,
        "audio_response_masker": None,
    }
    exec(_collator_block(), namespace)
    examples = [
        {
            "messages": [
                {"role": "system", "content": "transcribe accurately"},
                {"role": "user", "content": "<audio> <audio> please transcribe"},
                {"role": "assistant", "content": answer},
            ],
            "speech": {"array": [0.0]},
        }
        for answer in ("hello world", "a longer answer here")
    ]
    return namespace["audio_vlm_collate_fn"](examples)


def _targets(batch):
    return [
        [token for token, label in zip(ids, labels) if label != -100]
        for ids, labels in zip(batch["input_ids"].tolist(), batch["labels"].tolist())
    ]


def _words(*words):
    return [VOCAB[w] for w in words]


def test_completion_only_trains_on_the_assistant_turn_only(fake_unsloth):
    batch = _collate(_Processor(), train_on_completions = True)
    assert _targets(batch) == [
        _words("hello", "world", "<end>"),
        _words("a", "longer", "answer", "here", "<end>"),
    ]


@pytest.mark.parametrize(
    "training_args",
    [{}, {"train_on_completions": False}, {"train_on_completions": True, "is_cpt": True}],
)
def test_disabled_option_and_cpt_keep_full_sequence_labels(fake_unsloth, training_args):
    batch = _collate(_Processor(), **training_args)
    for ids, labels in zip(batch["input_ids"].tolist(), batch["labels"].tolist()):
        assert labels == [-100 if token in (PAD, AUDIO) else token for token in ids]


def test_undetectable_markers_warn_and_train_full_sequences(fake_unsloth, monkeypatch, caplog):
    dataset_utils = types.ModuleType("unsloth_zoo.dataset_utils")

    def fail(processor):
        raise ValueError("no chat template")

    dataset_utils.get_chat_template_parts = fail
    monkeypatch.setitem(sys.modules, "unsloth_zoo", types.ModuleType("unsloth_zoo"))
    monkeypatch.setitem(sys.modules, "unsloth_zoo.dataset_utils", dataset_utils)
    with caplog.at_level(logging.WARNING):
        batch = _collate(_Processor(preset_markers = False), train_on_completions = True)
    assert "could not be applied" in caplog.text
    for ids, labels in zip(batch["input_ids"].tolist(), batch["labels"].tolist()):
        assert labels == [-100 if token in (PAD, AUDIO) else token for token in ids]
