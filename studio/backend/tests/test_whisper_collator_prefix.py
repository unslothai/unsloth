# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Whisper labels must drop <|startoftranscript|>, which the model re-adds when shifting them right."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from utils.datasets.data_collators import DataCollatorSpeechSeq2SeqWithPadding

EOT, SOT, EN, TRANSCRIBE, NO_TS = 50257, 50258, 50259, 50360, 50364


class _Batch(dict):
    __getattr__ = dict.__getitem__


class _Pad:
    def pad(
        self,
        rows,
        return_tensors = "pt",
    ):
        key = next(iter(rows[0]))
        width = max(len(r[key]) for r in rows)
        ids = torch.tensor([list(r[key]) + [EOT] * (width - len(r[key])) for r in rows])
        mask = torch.tensor([[1] * len(r[key]) + [0] * (width - len(r[key])) for r in rows])
        return _Batch({key: ids, "attention_mask": mask})


class _Tokenizer(_Pad):
    bos_token_id = EOT

    def convert_tokens_to_ids(self, token):
        return {"<|startoftranscript|>": SOT}[token]


class _Processor:
    feature_extractor = _Pad()
    tokenizer = _Tokenizer()


def _features():
    return [
        {"input_features": [0.0], "labels": [SOT, EN, TRANSCRIBE, NO_TS, 11, 12, EOT]},
        {"input_features": [0.0], "labels": [SOT, EN, TRANSCRIBE, NO_TS, 13, EOT]},
    ]


def test_strips_decoder_start_from_model_config():
    labels = DataCollatorSpeechSeq2SeqWithPadding(_Processor(), decoder_start_token_id = SOT)(
        _features()
    )["labels"]
    assert labels[:, 0].tolist() == [EN, EN]
    assert labels[1].tolist() == [EN, TRANSCRIBE, NO_TS, 13, EOT, -100]


def test_strips_startoftranscript_without_a_model_config_value():
    labels = DataCollatorSpeechSeq2SeqWithPadding(_Processor())(_features())["labels"]
    assert labels[:, 0].tolist() == [EN, EN]
