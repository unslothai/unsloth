# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_kill_orphaned_servers", lambda self: 0)
    monkeypatch.setattr(llama_cpp_module.atexit, "register", lambda *_args, **_kwargs: None)
    return LlamaCppBackend()


def test_effective_parallel_slots_initial_value_is_one(backend):
    assert backend.effective_parallel_slots == 1


def test_effective_parallel_slots_commit_uses_final_positive_parallel(backend):
    backend._commit_effective_parallel_slots(3)

    assert backend.effective_parallel_slots == 3


@pytest.mark.parametrize("value", [None, 0, -2, "not-an-int"])
def test_effective_parallel_slots_commit_invalid_value_falls_back_to_one(backend, value):
    backend._commit_effective_parallel_slots(value)

    assert backend.effective_parallel_slots == 1


def test_effective_parallel_slots_reset_returns_to_one(backend):
    backend._commit_effective_parallel_slots(4)

    backend._reset_effective_parallel_slots()

    assert backend.effective_parallel_slots == 1


def test_effective_parallel_slots_unload_resets_to_one(backend):
    backend._commit_effective_parallel_slots(4)

    backend.unload_model()

    assert backend.effective_parallel_slots == 1
