# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unset generation limit (``None``) resolves to the context the prompt leaves free.

unsloth_fast_generate raises once ``input_length + max_new_tokens`` passes the window, so no
flat default fits every prompt. An explicit limit is never reduced.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.runtime_context import (
    UNSET_GENERATION_BUDGET,
    generation_budget_within_context,
)

_WINDOW = 2048
_PROMPT_LEN = 37


def _model(window = _WINDOW):
    return SimpleNamespace(config = SimpleNamespace(max_position_embeddings = window))


def test_an_unset_budget_becomes_the_free_context():
    assert generation_budget_within_context(_model(), _PROMPT_LEN, None) == _WINDOW - _PROMPT_LEN


def test_an_explicit_budget_is_never_reduced():
    # A request that does not fit must raise, not come back shortened.
    assert generation_budget_within_context(_model(), _PROMPT_LEN, 512) == 512
    assert generation_budget_within_context(_model(), _PROMPT_LEN, _WINDOW) == _WINDOW
    wide = SimpleNamespace(
        max_seq_length = 1024, config = SimpleNamespace(max_position_embeddings = 32768)
    )
    assert generation_budget_within_context(wide, _PROMPT_LEN, 4096) == 4096


def test_a_prompt_with_no_room_left_gets_the_default_not_a_token():
    # A floor of 1 clears a native 32768 guard on a 1024 load and returns one token.
    assert generation_budget_within_context(_model(window = 32), 64, None) == UNSET_GENERATION_BUDGET
    narrow = SimpleNamespace(
        max_seq_length = 1024, config = SimpleNamespace(max_position_embeddings = 32768)
    )
    assert generation_budget_within_context(narrow, 1100, None) == UNSET_GENERATION_BUDGET
    assert generation_budget_within_context(_model(window = 32), 31, None) == 1


def test_the_selected_window_wins_over_a_wider_checkpoint():
    # from_pretrained keeps config.max_position_embeddings at max(requested, native), so the
    # config alone would serve a --max-seq-length 1024 load the whole 32768.
    model = SimpleNamespace(
        max_seq_length = 1024,
        config = SimpleNamespace(max_position_embeddings = 32768),
    )
    assert generation_budget_within_context(model, _PROMPT_LEN, None) == 1024 - _PROMPT_LEN


def test_a_model_declaring_no_window_falls_back():
    assert (
        generation_budget_within_context(SimpleNamespace(), _PROMPT_LEN, None)
        == UNSET_GENERATION_BUDGET
    )
    for unusable in (None, "n/a", True):
        assert (
            generation_budget_within_context(_model(window = unusable), 1, None)
            == UNSET_GENERATION_BUDGET
        )
    assert generation_budget_within_context(SimpleNamespace(), _PROMPT_LEN, 256) == 256


def test_a_zero_budget_is_a_value_not_an_absence():
    assert generation_budget_within_context(_model(), _PROMPT_LEN, 0) == 0


# The call-site tests below import the backend, which pulls unsloth/unsloth_zoo: absent on CPU CI.


class _FakeTensor:
    def __init__(self, length):
        self.shape = (1, length)

    def to(self, _device):
        return self

    def __getitem__(self, _idx):
        return 2


class _FakeEncoding(dict):
    def to(self, _device):
        return self


class _FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 2
    all_special_tokens: list = []

    def __call__(
        self,
        _prompt,
        return_tensors = None,
        add_special_tokens = True,
    ):
        return _FakeEncoding(input_ids = _FakeTensor(_PROMPT_LEN))


class _FakeModel:
    """Carries the one guard unsloth_fast_generate applies before generating."""

    def __init__(self, window):
        self.max_seq_length = window
        self.config = SimpleNamespace(max_position_embeddings = window)
        self.device = "cpu"
        self.calls = []

    def generate(self, **kwargs):
        ids, budget = kwargs.get("input_ids"), kwargs.get("max_new_tokens")
        if ids is not None and budget is not None:
            if ids.shape[-1] + budget > self.config.max_position_embeddings:
                raise ValueError(
                    f"Unsloth: input length {ids.shape[-1]} + max_new_tokens {budget} "
                    "exceeds the maximum sequence length of "
                    f"{self.config.max_position_embeddings}!"
                )
        self.calls.append(kwargs)
        return _FakeTensor(ids.shape[-1] + 1)


class _EmptyStreamer:
    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration

    def end(self):
        pass


def _streaming_backend(monkeypatch, window = _WINDOW):
    try:
        from core.inference.inference import InferenceBackend
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - env-dependent
        # Skips because the CPU job installs no unsloth; the helper tests above cover
        # the same bound, so the budget itself is never left unasserted.
        pytest.skip(f"full inference backend unavailable ({type(exc).__name__}: {exc})")

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "unsloth/tinyllama-chat-bnb-4bit"
    backend.last_generation_stats = None
    backend._generation_lock = threading.Lock()
    model = _FakeModel(window)
    backend.models = {
        backend.active_model_name: {
            "model": model,
            "tokenizer": _FakeTokenizer(),
            "is_vision": False,
            "chat_turn_end_eos_ids": [2],
        }
    }
    monkeypatch.setattr(
        backend, "_make_text_streamer", lambda *a, **k: _EmptyStreamer(), raising = False
    )
    return backend, model


def _run(backend, max_new_tokens):
    return list(
        backend.generate_stream(
            "PROMPT",
            temperature = 0.0,
            max_new_tokens = max_new_tokens,
            repetition_penalty = 1.0,
        )
    )


def test_generate_stream_resolves_an_unset_budget(monkeypatch):
    backend, model = _streaming_backend(monkeypatch)

    _run(backend, None)

    assert model.calls, "generate was never reached"
    assert model.calls[0]["max_new_tokens"] == _WINDOW - _PROMPT_LEN


def test_generate_stream_passes_a_fitting_budget_through(monkeypatch):
    backend, model = _streaming_backend(monkeypatch)

    _run(backend, 512)

    assert model.calls[0]["max_new_tokens"] == 512


def _vision_backend(monkeypatch, window = _WINDOW):
    try:
        from core.inference.inference import InferenceBackend
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"full inference backend unavailable ({type(exc).__name__}: {exc})")

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "vision-model"
    backend.last_generation_stats = None
    backend._generation_lock = threading.Lock()
    model = _FakeModel(window)
    tokenizer = _FakeTokenizer()
    backend.models = {
        backend.active_model_name: {
            "model": model,
            "tokenizer": tokenizer,
            "processor": tokenizer,
            "is_vision": True,
            "chat_turn_end_eos_ids": [2],
        }
    }
    monkeypatch.setattr(
        backend, "_make_text_streamer", lambda *a, **k: _EmptyStreamer(), raising = False
    )
    monkeypatch.setattr(backend, "format_chat_prompt", lambda *a, **k: "PROMPT", raising = False)
    return backend, model


def test_the_vision_path_resolves_an_unset_budget(monkeypatch):
    """Without this, a VLM turn reaches generate() with max_new_tokens=None, which
    transformers reads as its own tiny max_length default rather than no limit."""
    backend, model = _vision_backend(monkeypatch)

    list(
        backend._generate_vision_response(
            [{"role": "user", "content": "hi"}],
            "",
            None,
            0.0,
            1.0,
            0,
            0.0,
            None,
            1.0,
        )
    )

    assert model.calls, "generate was never reached"
    assert model.calls[0]["max_new_tokens"] == _WINDOW - _PROMPT_LEN
