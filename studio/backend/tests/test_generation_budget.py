# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A generation budget sized to the whole context window cannot also fit a prompt.

unsloth_fast_generate (unsloth/models/llama.py) raises when
``input_length + max_new_tokens > config.max_position_embeddings``. An unset client
limit resolves to the full window -- 2048, what ``load_model`` falls back to for
``--max-seq-length 0`` -- so on a model whose window is 2048 (tinyllama-chat, which
Unsloth's own mapper ships) every nonempty prompt raised instead of generating.
"""

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.runtime_context import generation_budget_within_context

_WINDOW = 2048
_PROMPT_LEN = 37


def _model(window = _WINDOW):
    return SimpleNamespace(config = SimpleNamespace(max_position_embeddings = window))


def test_a_whole_window_budget_is_fitted_to_the_free_context():
    assert generation_budget_within_context(_model(), _PROMPT_LEN, _WINDOW) == _WINDOW - _PROMPT_LEN


def test_a_budget_that_already_fits_is_left_alone():
    assert generation_budget_within_context(_model(), _PROMPT_LEN, 512) == 512


def test_a_prompt_that_fills_the_window_keeps_the_real_overflow():
    # Nothing is left to generate, so the budget must not be shrunk into a value
    # that hides a prompt which genuinely does not fit.
    assert generation_budget_within_context(_model(window = 32), 64, 256) == 256


def test_the_selected_window_wins_over_a_wider_checkpoint():
    # from_pretrained keeps config.max_position_embeddings at max(requested, native)
    # and attaches the requested limit, so reading the config alone would serve a
    # --max-seq-length 1024 load 2048 new tokens.
    model = SimpleNamespace(
        max_seq_length = 1024,
        config = SimpleNamespace(max_position_embeddings = 32768),
    )
    assert generation_budget_within_context(model, _PROMPT_LEN, _WINDOW) == 1024 - _PROMPT_LEN


def test_a_model_declaring_no_window_is_passed_through():
    assert generation_budget_within_context(SimpleNamespace(), _PROMPT_LEN, _WINDOW) == _WINDOW
    assert generation_budget_within_context(_model(window = None), 1, 256) == 256
    assert generation_budget_within_context(_model(window = "n/a"), 1, 256) == 256
    assert generation_budget_within_context(_model(window = True), 1, 256) == 256


def test_an_unset_budget_stays_unset():
    assert generation_budget_within_context(_model(), _PROMPT_LEN, None) is None
    assert generation_budget_within_context(_model(), _PROMPT_LEN, 0) == 0


# ── The call site ──────────────────────────────────────────────────────────────
# Importing the backend pulls unsloth/unsloth_zoo, which the CPU CI job does not
# install; these run wherever the full stack is present.


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

    def __call__(self, _prompt, return_tensors = None, add_special_tokens = True):
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


def test_generate_stream_fits_a_whole_window_budget(monkeypatch):
    backend, model = _streaming_backend(monkeypatch)

    _run(backend, _WINDOW)

    assert model.calls, "generate was never reached"
    assert model.calls[0]["max_new_tokens"] == _WINDOW - _PROMPT_LEN


def test_generate_stream_passes_a_fitting_budget_through(monkeypatch):
    backend, model = _streaming_backend(monkeypatch)

    _run(backend, 512)

    assert model.calls[0]["max_new_tokens"] == 512
