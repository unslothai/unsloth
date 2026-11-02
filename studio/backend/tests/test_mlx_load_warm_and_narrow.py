# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load serves through the batch before it reports loaded."""

from __future__ import annotations

import inspect

import pytest

from core.inference import mlx_inference
from core.inference.mlx_inference import MLXInferenceBackend


class _Session:
    def __init__(self, backend, fail):
        self.backend, self.fail, self.rows_in_flight = backend, fail, 0

    def admit(self, request, _handle):
        self.backend.temperatures.append(request["temperature"])
        self.rows_in_flight = 1

    def step(self):
        self.rows_in_flight = 0
        yield "warm", self.fail

    def close(self):
        self.backend.closed += 1


def _warmed(fail = None, room = None):
    backend = MLXInferenceBackend.__new__(MLXInferenceBackend)
    vars(backend).update(temperatures = [], closed = 0, cleared = 0, _prompt_cache_room = room)

    def clear():
        backend.cleared += 1
        backend._prompt_cache_room = None

    backend.resident_unavailable_reason = lambda _request: None
    backend.open_resident_batch = lambda *, width: _Session(backend, fail)
    backend._clear_prompt_cache = clear
    backend._warm_resident_batch()
    return backend


def test_warm_serves_greedy_twice_then_sampled_and_clears_the_cache():
    backend = _warmed(room = 123)
    assert backend.temperatures == [0.0, 0.0, 0.7] and backend.closed == 3
    assert backend.cleared == 1 and backend._prompt_cache_room == 123


def test_the_load_warms_the_batch_with_the_rng_preserved():
    source = inspect.getsource(MLXInferenceBackend.load_model)
    assert "with mlx_rng_preserved():\n                self._warm_resident_batch()" in source


def test_a_failed_warm_leaves_the_load_standing_and_the_cache_clear():
    backend = _warmed(fail = ValueError("row"))
    assert backend.temperatures == [0.0] and backend.closed == 1 and backend.cleared == 1


def test_a_dead_metal_queue_still_fails_the_load(monkeypatch):
    monkeypatch.setattr(mlx_inference, "is_metal_queue_dead", lambda _error: True)
    with pytest.raises(ValueError):
        _warmed(fail = ValueError("queue"))
