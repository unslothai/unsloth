# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The chat store replaces a model's row from /api/models/list, so a GGUF row must carry audio input."""

import asyncio
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.inference as inf  # noqa: E402
import routes.models as models_route  # noqa: E402


class _FakeLlama:
    is_vision = True
    is_loaded = True
    model_identifier = "unsloth/gemma-4-E2B-it-GGUF"
    _audio_type = "audio_vlm"

    def __init__(self, has_audio_input):
        self._has_audio_input = has_audio_input


class _FakeUnsloth:
    default_models: list = []
    models: dict = {}


def _gguf_row(monkeypatch, llama):
    monkeypatch.setattr(models_route, "peek_inference_backend", lambda: _FakeUnsloth())
    monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: llama)
    (row,) = asyncio.run(models_route.list_models(current_subject = "unsloth")).models
    return row


def test_gguf_with_an_audio_mmproj_lists_audio_input(monkeypatch):
    row = _gguf_row(monkeypatch, _FakeLlama(True))
    assert row.is_gguf is True
    assert row.has_audio_input is True


def test_gguf_without_audio_lists_no_audio_input(monkeypatch):
    assert _gguf_row(monkeypatch, _FakeLlama(False)).has_audio_input is False


def test_a_backend_without_the_attribute_lists_no_audio_input(monkeypatch):
    class _Old(_FakeLlama):
        pass

    llama = _Old(False)
    del llama._has_audio_input
    assert _gguf_row(monkeypatch, llama).has_audio_input is False
